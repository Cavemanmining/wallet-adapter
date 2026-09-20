"""Tests for the delivery gate, :mod:`jarvis_alerts.validate`.

Design: jarvis_alerts/contracts.py, the three things delivery needs (a
durable leased outbox, an injected transport, at-least-once with recorded
attempts) and the retry policy at its foot.  The gate drives the real
outbox, worker and FakeTransport through a seeded scenario and asserts the
end state; these tests hold the gate itself to four promises:

* a healthy run reports zero problems, and the scenario really did contain
  every hazard the gate claims to exercise (crashes, late devices, dedupe
  collisions, transient, permanent and gone devices, an outage that
  outlasts the retry budget and then clears, and one that outlives the
  alert itself);
* each injected defect in ``validate.DEFECTS`` -- ``skip_reclaim`` and
  ``lose_on_crash`` through to ``no_revive`` -- is reported, with the kind
  that names it and a repro that reproduces it: the gate is shown to fail
  before it passes;
* results are deterministic for a fixed seed;
* ``n_profiles=50, n_alerts=20`` finishes in under 30 seconds.

No test sleeps or reads the wall clock except the timing test, which
measures the gate from outside.  Nothing here touches a subscription blob;
the last test checks the report does not either.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Runnable as `pytest tests/test_alerts_validate.py` or
# `python3 tests/test_alerts_validate.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_alerts.validate import DEFECTS, GateReport, Problem, main, run_gate
from lucifer_gen.seed import format_seed

SEED = 0x5EED_A1E7_0000_0001
SMALL = dict(n_profiles=12, n_alerts=10, seed=SEED)

#: The kinds each defect must surface, whatever the seed.  The gate lists
#: every finding, so a defect usually trips more than these.
DEFECT_KINDS = {
    "skip_reclaim": {"row_left_leased", "not_delivered"},
    "double_lease": {"duplicate_ok_send", "ok_call_count_mismatch", "lease_not_exclusive"},
    "lose_on_crash": {"delivered_without_send", "ok_call_count_mismatch"},
    "no_lease": {"lease_not_exclusive"},
    "no_backoff": {"backoff_not_applied"},
    "lease_gone": {"leased_for_gone_device"},
    "attempts_lost": {"attempt_record_mismatch", "attempt_total_mismatch"},
    # A row whose outage is over is left DEAD: the observer sees it in the
    # round the revival was due, the ledger sees the exhausting row never
    # delivered, and the run never runs out of outstanding rows.
    "no_revive": {"revive_skipped", "exhaustion_mismatch", "not_converged"},
}


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def healthy() -> GateReport:
    return run_gate(**SMALL)


@pytest.fixture(scope="module", params=DEFECTS)
def defective(request) -> GateReport:
    return run_gate(**SMALL, inject_defect=request.param)


# --------------------------------------------------------------------------
# The healthy run
# --------------------------------------------------------------------------


def test_healthy_run_reports_zero_problems(healthy: GateReport) -> None:
    assert healthy.ok
    assert healthy.problems == []
    assert healthy.first_failure is None
    assert healthy.kinds() == []
    assert healthy.converged
    c = healthy.counts
    assert c["problems"] == 0
    assert c["converged"] == 1
    assert c["rows_leased"] == 0
    assert c["rows_pending_live"] == 0
    assert c["worker_errors"] == 0


def test_healthy_run_delivered_exactly_what_the_transport_accepted(healthy: GateReport) -> None:
    c = healthy.counts
    assert c["rows_delivered"] > 0
    assert c["rows_delivered"] == c["transport_ok"]
    # Every row is accounted for by exactly one final state.
    assert c["rows"] == c["rows_delivered"] + c["rows_dead"] + c["rows_pending_parked"] + c["rows_pending_live"] + c["rows_leased"]
    # Every expected pair produced a row, and no row is unexpected.
    assert c["rows"] == c["pairs_expected"]
    # Retries happened: attempts exceed rows, and every transport call was recorded.
    assert c["attempts"] > c["rows"]
    assert c["attempts"] >= c["transport_calls"]
    # Dead letters come only from the permanent, gone and exhausting devices.
    assert c["rows_dead"] <= c["transport_permanent"] + c["transport_gone"] + c["devices_went_gone"] * 2 + c["rows_exhausted"]
    # Every mark the worker made is one recorded attempt.
    assert c["attempts"] == c["marks"]


def test_healthy_scenario_exercises_every_hazard(healthy: GateReport) -> None:
    """The gate is only worth something if the scenario contains what it
    says: crashes with rows in hand, late devices that get backfilled,
    dedupe collisions, transient failures near 15 percent, a permanently
    rejecting device and a device that went gone partway."""
    c = healthy.counts
    assert c["profiles"] == SMALL["n_profiles"]
    # The waves, the two dedupe probes per keyed profile, and one prelude
    # alert per profile with a stale device (published before the clock
    # jumps ALERT_MAX_AGE_S).
    assert c["publishes"] == SMALL["n_profiles"] * SMALL["n_alerts"] + c["probes"] + c["publishes_prelude"]
    assert c["publishes_prelude"] == c["devices_stale"] >= 1
    assert c["profiles"] <= c["devices"] <= 3 * c["profiles"]
    assert c["devices_late"] >= 1
    assert c["rows_backfilled"] >= 1
    assert c["devices_permanent"] >= 1
    assert c["devices_gone_scripted"] >= 1
    assert c["devices_went_gone"] >= 1
    assert c["worker_pruned"] == c["devices_went_gone"]
    assert c["alerts_deduped"] >= 1
    assert c["alerts_stored"] + c["alerts_deduped"] == c["publishes"]
    assert c["crashes"] >= 1
    assert c["rows_abandoned"] >= 1
    assert c["transport_transient"] >= 1
    assert 0.05 < c["transport_transient"] / c["transport_calls"] < 0.30
    assert c["transport_permanent"] >= 1
    assert c["transport_gone"] >= 1
    assert c["rows_dead"] >= 1


def test_healthy_scenario_exercises_the_policy_end_the_dedupe_probes_and_repair(healthy: GateReport) -> None:
    """The retry budget is spent on purpose, by an exhausting device whose
    outage then clears and by a stale device whose alert ages out first;
    both sides of the dedupe window are probed at every size; a repeat
    repairs what the earlier alert did not reach; a late device is
    backfilled while a device of its profile is gone."""
    c = healthy.counts
    assert c["devices_exhaust"] >= 1 and c["devices_stale"] >= 1
    # Both ends of the policy: the exhausting device's row dies EXHAUSTED and
    # is brought back by the outbox itself once the cooldown has passed, and
    # the stale device's row dies EXHAUSTED past ALERT_MAX_AGE_S and stays
    # dead.  The ledger counts one of each per such device.
    assert c["rows_exhausted"] == c["devices_exhaust"] + c["devices_stale"]
    assert c["rows_exhausted_revived"] == c["devices_exhaust"]
    assert c["rows_exhausted_stale"] == c["devices_stale"]
    assert c["rows_revived"] == c["worker_revived"] == c["devices_exhaust"]
    assert c["probes"] >= 2 * c["profiles"] // 2                         # recent and stale, for every keyed profile
    assert c["rows_repaired"] + c["rows_requeued"] >= 1
    assert c["marks"] == c["attempts"] and c["releases"] == 0 and c["worker_expired"] == 0
    # The slow worker held the gone devices' rows while they went gone: every
    # such row came back onto a gone device and is parked, never handed out.
    assert c["stalls"] == 2 and c["devices_stalled"] == c["devices_gone_scripted"] >= 1
    assert c["rows_stalled"] >= c["devices_stalled"]
    assert c["rows_pending_parked"] >= c["devices_stalled"]
    assert c["alerts_deduped"] >= c["profiles"] // 2                     # the recent probe collapses


def test_show_defects_exit_code_fails_when_a_defect_slips_through(monkeypatch, capsys) -> None:
    from jarvis_alerts import validate as module

    real = module.run_gate

    def blind(n_profiles, n_alerts, seed, crash_every=7, inject_defect=None, **kw):
        return real(n_profiles, n_alerts, seed, crash_every, None, **kw)    # every defect run comes back green

    monkeypatch.setattr(module, "run_gate", blind)
    assert main(["--profiles", "3", "--alerts", "3", "--show-defects"]) == 1
    out = capsys.readouterr().out
    assert "result: OK" in out and out.count("NOT CAUGHT") == len(DEFECTS)


def test_gate_is_deterministic_across_hash_seeds(tmp_path) -> None:
    """Sets of strings would iterate differently under another
    PYTHONHASHSEED; the report must not depend on it."""
    import os
    import subprocess
    code = ("import json; from jarvis_alerts.validate import run_gate; "
            "print(json.dumps(run_gate(9, 6, 0xABC).to_dict(), sort_keys=True))")
    outputs = []
    for hash_seed in ("0", "1", "12345"):
        proc = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True,
                              env={**os.environ, "PYTHONHASHSEED": hash_seed}, timeout=120)
        assert proc.returncode == 0, proc.stderr
        outputs.append(proc.stdout)
    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0])["ok"] is True


def test_healthy_summary_and_dict(healthy: GateReport) -> None:
    text = healthy.summary()
    assert "result: OK" in text
    assert format_seed(SEED) in text
    assert "FAIL" not in text
    d = healthy.to_dict()
    json.dumps(d)  # JSON-friendly
    assert d["ok"] is True and d["problems"] == [] and d["seed"] == format_seed(SEED)
    assert d["counts"] == healthy.counts


# --------------------------------------------------------------------------
# The gate fails before it passes: injected defects
# --------------------------------------------------------------------------


def test_each_injected_defect_is_reported(defective: GateReport) -> None:
    assert not defective.ok
    assert defective.problems
    first = defective.first_failure
    assert isinstance(first, Problem)
    assert first is defective.problems[0]
    assert defective.counts["problems"] == len(defective.problems)
    assert DEFECT_KINDS[defective.inject_defect] <= set(defective.kinds()), defective.kinds()


def test_defect_first_failure_carries_enough_to_reproduce(defective: GateReport) -> None:
    first = defective.first_failure
    assert first.repro == defective.repro
    assert defective.inject_defect in first.repro
    assert format_seed(SEED) in first.repro
    assert f"n_profiles={SMALL['n_profiles']}" in first.repro
    # A row-level finding names the row and its owner by ids only.
    row_level = [p for p in defective.problems if p.row_id is not None]
    assert row_level, "no row-level finding"
    p = row_level[0]
    assert p.profile_id.startswith("p") and p.device_id.startswith("d") and p.alert_id.startswith(p.profile_id)
    assert p.detail
    assert p.kind in str(p) and f"row={p.row_id}" in str(p)


def test_repro_string_reproduces_the_failure(defective: GateReport) -> None:
    """The repro on a Problem is a call that gives the same problems back."""
    again = eval(defective.first_failure.repro, {"run_gate": run_gate})
    assert again.problems == defective.problems
    assert again.counts == defective.counts


def test_defect_summary_says_fail(defective: GateReport) -> None:
    text = defective.summary(max_problems=2)
    assert "result: FAIL" in text
    assert "repro: run_gate(" in text
    assert defective.first_failure.kind in text
    if len(defective.problems) > 2:
        assert f"{len(defective.problems) - 2} more" in text


def test_skip_reclaim_leaves_rows_leased_and_alerts_undelivered() -> None:
    report = run_gate(**SMALL, inject_defect="skip_reclaim")
    assert report.first_failure.kind == "row_left_leased"
    assert report.counts["rows_leased"] >= 1
    assert not report.converged
    assert "not_converged" in report.kinds()


def test_double_lease_sends_rows_twice() -> None:
    report = run_gate(**SMALL, inject_defect="double_lease")
    assert report.counts["transport_ok"] > report.counts["rows_delivered"]
    dup = [p for p in report.problems if p.kind == "duplicate_ok_send"]
    assert dup and "2 times" in dup[0].detail


def test_lose_on_crash_acknowledges_without_sending() -> None:
    report = run_gate(**SMALL, inject_defect="lose_on_crash")
    assert report.first_failure.kind == "delivered_without_send"
    assert report.counts["rows_delivered"] > report.counts["transport_ok"]


def test_no_revive_leaves_the_row_of_a_finished_outage_dead(healthy: GateReport) -> None:
    """The defect this package's cooldown exists to prevent: an outage that
    outlasted the retry budget is over, and the row stays DEAD anyway."""
    report = run_gate(**SMALL, inject_defect="no_revive")
    assert report.first_failure.kind == "revive_skipped"
    assert not report.converged and "not_converged" in report.kinds()
    assert report.counts["rows_revived"] == 0 and report.counts["worker_revived"] == 0
    assert report.counts["rows_exhausted_revived"] == 0            # none came back ...
    assert healthy.counts["rows_exhausted_revived"] >= 1           # ... and one should have
    # The deaths themselves are untouched: only the revival is suppressed,
    # so the stale device's row still dies exactly as it must.
    assert report.counts["rows_exhausted_stale"] == report.counts["devices_stale"] >= 1


def test_defects_are_caught_for_other_seeds() -> None:
    for seed in (1, 0xDEADBEEF, 99):
        for defect in DEFECTS:
            report = run_gate(12, 12, seed, 7, defect)
            assert not report.ok, (seed, defect)
            assert DEFECT_KINDS[defect] <= set(report.kinds()), (seed, defect, report.kinds())


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_results_are_deterministic_for_a_fixed_seed() -> None:
    a = run_gate(20, 8, 0xABC)
    b = run_gate(20, 8, 0xABC)
    assert a.ok and b.ok
    assert a.counts == b.counts
    assert a.to_dict() == b.to_dict()


def test_problems_are_deterministic_for_a_fixed_seed() -> None:
    a = run_gate(10, 8, 0xABC, inject_defect="lose_on_crash")
    b = run_gate(10, 8, 0xABC, inject_defect="lose_on_crash")
    assert a.problems == b.problems
    assert a.counts == b.counts


def test_different_seeds_give_different_scenarios() -> None:
    a = run_gate(20, 8, 1).counts
    b = run_gate(20, 8, 2).counts
    keys = ("devices", "devices_late", "alerts_deduped", "transport_calls", "transport_transient")
    assert tuple(a[k] for k in keys) != tuple(b[k] for k in keys)


def test_healthy_across_seeds_and_crash_cadences() -> None:
    for seed in (1, 0xFFFF_FFFF_FFFF_FFFF, 12345):
        for crash_every in (0, 2, 3, 7, 13):
            report = run_gate(9, 6, seed, crash_every)
            assert report.ok and report.converged, (seed, crash_every, report.kinds())
            if crash_every == 0:
                assert report.counts["crashes"] == 0 and report.counts["rows_abandoned"] == 0
            else:
                assert report.counts["crashes"] >= 1


# --------------------------------------------------------------------------
# Time budget
# --------------------------------------------------------------------------


def test_gate_finishes_50_profiles_20_alerts_under_30s() -> None:
    started = time.perf_counter()
    report = run_gate(n_profiles=50, n_alerts=20, seed=SEED)
    elapsed = time.perf_counter() - started
    assert elapsed < 30.0, f"{elapsed:.1f}s"
    assert report.ok, report.summary()
    assert report.converged
    c = report.counts
    assert c["profiles"] == 50 and c["publishes"] == 1000 + c["probes"] + c["publishes_prelude"]
    assert c["rows_delivered"] >= 1000
    assert c["devices_permanent"] == 5
    assert c["devices_gone_scripted"] == 7
    assert c["devices_exhaust"] == 4 and c["devices_stale"] == 4
    assert c["rows_revived"] == 4


# --------------------------------------------------------------------------
# Arguments, privacy and the CLI
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_profiles=0, n_alerts=1, seed=1),
        dict(n_profiles=1, n_alerts=0, seed=1),
        dict(n_profiles=1, n_alerts=1, seed=1, crash_every=1),
        dict(n_profiles=1, n_alerts=1, seed=1, crash_every=-3),
        dict(n_profiles=1, n_alerts=1, seed=1, inject_defect="nonsense"),
        dict(n_profiles=1, n_alerts=1, seed="abc"),
    ],
)
def test_bad_arguments_are_refused(kwargs) -> None:
    with pytest.raises(ValueError):
        run_gate(**kwargs)


def test_smallest_scenarios_are_healthy() -> None:
    for n_profiles, n_alerts in ((1, 1), (1, 2), (2, 1), (3, 3)):
        report = run_gate(n_profiles, n_alerts, SEED)
        assert report.ok and report.converged, (n_profiles, n_alerts, report.kinds())


def test_file_backed_outbox_is_healthy(tmp_path) -> None:
    report = run_gate(8, 5, SEED, db_path=str(tmp_path / "gate.sqlite"))
    assert report.ok and report.converged
    assert (tmp_path / "gate.sqlite").exists()


def test_report_never_carries_a_subscription_blob(healthy: GateReport, defective: GateReport) -> None:
    """The gate registers blobs of the form {"endpoint": "fake://p/d"};
    nothing of that shape may reach a problem, a count or the summary."""
    for report in (healthy, defective):
        for text in (report.summary(max_problems=1000), json.dumps(report.to_dict()), repr(report.problems)):
            assert "fake://" not in text
            assert "endpoint" not in text


def test_main_exit_codes_and_output(capsys) -> None:
    assert main(["--profiles", "6", "--alerts", "5", "--seed", hex(SEED)]) == 0
    out = capsys.readouterr().out
    assert "result: OK" in out

    assert main(["--profiles", "6", "--alerts", "5", "--seed", str(SEED), "--defect", "skip_reclaim"]) == 1
    out = capsys.readouterr().out
    assert "result: FAIL" in out and "row_left_leased" in out

    assert main(["--profiles", "6", "--alerts", "5", "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ok"] is True and parsed["counts"]["profiles"] == 6

    assert main(["--profiles", "8", "--alerts", "6", "--show-defects"]) == 0
    out = capsys.readouterr().out
    assert out.count(": caught;") == len(DEFECTS)
    assert "NOT CAUGHT" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
