"""Tests for the framework gate, jarvis_bots.validate.

Design: jarvis_bots/contracts.py and jarvis_bots/web/README.md.  The gate
drives real :class:`jarvis_bots.supervisor.Supervisor` rounds against a
fleet of synthetic bots that misbehave on purpose, and then argues with
the result.  These tests hold the gate itself to the only standard that
makes a gate worth having:

* a healthy run reports **zero** problems, and really did contain every
  hazard it claims to test -- a quarantine, a probe out of one, a pause
  and a resume, a flapping attention key, bursts, slow ticks, and all
  four shapes of malformed return;
* **every** injected defect is caught, with the kind that names it: the
  gate is shown to fail before it is trusted, and it is also shown to
  catch breakage it does not ship a defect for, so the checks are not
  merely tuned to the shipped ones;
* the same seed gives the same report twice and a different seed does
  not, because the only randomness is ``lucifer_gen.seed`` and the clock
  is an injected cell;
* 500 rounds finish well inside 20 seconds.

Nothing here sleeps, opens a socket or reads a wall clock, except the one
timing test, which measures the gate from outside.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Runnable as `pytest tests/test_bots_validate.py` or
# `python3 tests/test_bots_validate.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_bots import validate
from jarvis_bots.contracts import QUARANTINE_AFTER_FAILURES, BotState
from jarvis_bots.supervisor import Supervisor
from jarvis_bots.validate import (
    DEFECTS,
    MIN_ROUNDS,
    GateError,
    GateReport,
    run_gate,
)

SEED = 0xB075_6A7E_0000_0001
#: Three unrelated seeds, so a green gate is not a property of one script.
SEEDS = (SEED, 0x5EED_A1E7_0000_0002, 0x0123_4567_89AB_CDEF)
ROUNDS = 200

#: Which problem kind each defect must produce.  A defect that trips some
#: *other* check is not good enough: the gate has to name what went wrong.
EXPECTED_KIND: Dict[str, str] = {
    "no_isolation": "isolation_broken",
    "no_quarantine": "quarantine_wrong",
    "narrowing_backoff": "backoff_narrowed",
    "tick_paused": "paused_ticked",
    "attention_leak": "attention_unbounded",
    "realert_every_round": "realerted",
    "state_drift": "badge_wrong",
    # the bot-side close path (RESOLVED_FLAG), which the gate could not see
    # at all until a synthetic bot closed its own key
    "never_resolves": "attention_wrong",
    "loose_resolution": "attention_wrong",
    "cross_bot_close": "attention_wrong",
    "late_resolution": "attention_wrong",
    "resolution_poisons_alerts": "alert_memory_leak",
    "state_not_restored": "state_not_restored",
}


# --------------------------------------------------------------------------
# A healthy run
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", SEEDS)
def test_healthy_run_reports_no_problems(seed: int) -> None:
    """The framework as shipped holds every rule in contracts.py."""
    report = run_gate(ROUNDS, seed)
    assert report.ok, report.summary()
    assert report.problems == []
    assert report.first_failure is None
    assert report.suppressed == 0


@pytest.mark.parametrize("rounds", [MIN_ROUNDS, ROUNDS, 313])
def test_healthy_run_holds_at_several_lengths(rounds: int) -> None:
    """The checks are about the rules, not about a run of one length."""
    report = run_gate(rounds, SEED)
    assert report.ok, report.summary()
    assert report.counts["rounds"] == rounds


@pytest.mark.parametrize("seed", SEEDS)
def test_healthy_run_is_not_vacuous(seed: int) -> None:
    """A green gate that never met a hazard proves nothing.

    The gate has its own :meth:`_Gate._check_scenario` guard; this is the
    same demand from outside, on the numbers an operator reads.
    """
    counts = run_gate(ROUNDS, seed).counts
    assert counts["quarantines"] >= 1, "nothing was ever quarantined"
    assert counts["probes"] >= 1, "no probe ran out of a quarantine"
    assert counts["raises"] >= 10, "the failing bots barely failed"
    assert counts["rejections"] >= 8, "no malformed returns were rejected"
    assert counts["pauses"] == 1 and counts["resumes"] == 1
    assert counts["paused_rounds"] >= 10, "the pause window was too short to matter"
    assert counts["attention_cleared"] >= 2, "no question was ever answered"
    assert counts["bursts"] >= 5, "no bot ever burst"
    assert counts["slow_rounds"] >= 1, "no slow tick was reported"
    assert counts["status_breaks"] >= 1, "no card's status() ever raised"
    assert counts["alerts"] >= 8, "hardly anything reached the phone"
    assert counts["events"] >= 200
    assert counts["round_crashes"] == 0, "a healthy round must never raise"


def test_the_badge_is_bounded_and_the_alerts_are_not_per_round() -> None:
    """contracts.py: the badge "counts distinct open keys", and one open
    question is one push, not one per round."""
    counts = run_gate(ROUNDS, SEED).counts
    assert counts["max_attention"] <= counts["attention_keys_declared"]
    assert counts["max_attention"] >= 3, "the badge never got interesting"
    # One push per episode of attention, plus the keyless ACTION notices.
    # Either way: far fewer pushes than rounds-with-an-open-question.
    assert counts["alerts"] < counts["rounds"]
    assert counts["alerts"] >= counts["episodes"]


def test_every_count_is_an_int() -> None:
    """``counts`` is JSON-friendly; no float sneaks in from the clock."""
    for value in run_gate(MIN_ROUNDS, SEED).counts.values():
        assert isinstance(value, int) and not isinstance(value, bool)


# --------------------------------------------------------------------------
# Shown to fail before it is trusted
# --------------------------------------------------------------------------


def test_the_defects_are_the_documented_ones() -> None:
    """The list the gate ships is the list it is asked for."""
    assert set(DEFECTS) == {
        "no_isolation",
        "no_quarantine",
        "narrowing_backoff",
        "tick_paused",
        "attention_leak",
        "realert_every_round",
        "state_drift",
        "never_resolves",
        "loose_resolution",
        "cross_bot_close",
        "late_resolution",
        "resolution_poisons_alerts",
        "state_not_restored",
    }
    assert set(EXPECTED_KIND) == set(DEFECTS)
    assert len(set(DEFECTS)) == len(DEFECTS)


@pytest.mark.parametrize("defect", DEFECTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_every_defect_is_caught(defect: str, seed: int) -> None:
    report = run_gate(ROUNDS, seed, defect)
    assert not report.ok, f"{defect} went unreported: {report.summary()}"
    assert report.inject_defect == defect


@pytest.mark.parametrize("defect", DEFECTS)
@pytest.mark.parametrize("seed", SEEDS)
def test_every_defect_is_named(defect: str, seed: int) -> None:
    """Catching it is not enough; the report has to say what broke."""
    report = run_gate(ROUNDS, seed, defect)
    assert EXPECTED_KIND[defect] in report.kinds(), report.summary()


def test_no_isolation_is_caught_by_the_control_bot() -> None:
    """The point of the control: a failing neighbour must not cost the
    healthy bot a round.  contracts.py: "A bot never blocks another"."""
    report = run_gate(ROUNDS, SEED, "no_isolation")
    kinds = report.kinds()
    assert "isolation_broken" in kinds
    assert "round_crashed" in kinds


def test_no_quarantine_is_caught_at_the_threshold() -> None:
    """``QUARANTINE_AFTER_FAILURES`` is the number the gate holds to."""
    healthy = run_gate(ROUNDS, SEED)
    assert healthy.ok
    broken = run_gate(ROUNDS, SEED, "no_quarantine")
    problem = next(p for p in broken.problems if p.kind == "quarantine_wrong")
    assert str(QUARANTINE_AFTER_FAILURES) in problem.detail


def test_narrowing_backoff_is_caught_as_narrower_than_base() -> None:
    """contracts.py's ``backoff_interval``: "Never narrower than base"."""
    report = run_gate(ROUNDS, SEED, "narrowing_backoff")
    problem = next(p for p in report.problems if p.kind == "backoff_narrowed")
    assert "narrower than the base interval" in problem.detail


def test_tick_paused_is_caught_three_ways() -> None:
    """"Paused means paused" is three promises: no tick, no attention, no
    push.  Breaking the switch has to break all three visibly."""
    kinds = set(run_gate(ROUNDS, SEED, "tick_paused").kinds())
    assert {"paused_ticked", "paused_alert", "paused_attention"} <= kinds


def test_attention_leak_is_caught_as_unbounded() -> None:
    report = run_gate(ROUNDS, SEED, "attention_leak")
    problem = next(p for p in report.problems if p.kind == "attention_unbounded")
    assert "without bound" in problem.detail


def test_state_drift_is_caught_against_the_launcher_readme() -> None:
    """web/README.md is a contract, not a suggestion."""
    kinds = set(run_gate(ROUNDS, SEED, "state_drift").kinds())
    assert "badge_wrong" in kinds
    assert "launcher_invalid" in kinds


# -- and breakage the gate does not ship a defect for ----------------------


def _probe(monkeypatch: Any, name: str, supervisor_class: type) -> GateReport:
    """Run the gate against a one-off broken supervisor.

    Proof that the checks are about the contract rather than tuned to the
    shipped defects: none of the classes below is one of them.
    """
    monkeypatch.setitem(validate._DEFECT_CLASSES, name, supervisor_class)
    monkeypatch.setattr(validate, "DEFECTS", DEFECTS + (name,))
    return run_gate(ROUNDS, SEED, name)


def test_a_supervisor_that_never_reports_a_slow_tick_is_caught(monkeypatch) -> None:
    class NoSlow(Supervisor):
        def run_round(self, now=None):
            report = super().run_round(now)
            report.slow = ()
            return report

    assert "slow_wrong" in _probe(monkeypatch, "no_slow", NoSlow).kinds()


def test_a_card_missing_a_documented_key_is_caught(monkeypatch) -> None:
    class DropLastEvent(Supervisor):
        def launcher_state(self, now=None, *, include_detail=False):
            state = super().launcher_state(now, include_detail=include_detail)
            for card in state["bots"]:
                card.pop("last_event")
            return state

    kinds = _probe(monkeypatch, "drop_last_event", DropLastEvent).kinds()
    assert "launcher_invalid" in kinds


def test_pause_that_hides_attention_instead_of_clearing_it_is_caught(monkeypatch) -> None:
    """The filter in ``attention_items`` would make this invisible; the
    persisted state is where it shows."""

    class HidePausedAttention(Supervisor):
        def clear_bot_attention(self, bot_id):
            return 0

    kinds = _probe(monkeypatch, "hide_paused", HidePausedAttention).kinds()
    assert "paused_attention" in kinds


def test_alert_memory_that_outlives_its_question_is_caught(monkeypatch) -> None:
    class KeepAlertMemory(Supervisor):
        def clear_attention(self, bot_id, key):
            pair = (bot_id, key)
            kept = self._alerted.get(pair)
            out = super().clear_attention(bot_id, key)
            if kept is not None:
                self._alerted[pair] = kept
            return out

    kinds = _probe(monkeypatch, "keep_alert_memory", KeepAlertMemory).kinds()
    assert "alert_memory_leak" in kinds


def test_a_pause_that_is_not_persisted_is_caught(monkeypatch) -> None:
    """contracts.py: "a pause that forgets itself at the next restart is
    not a pause"."""

    class ForgetPause(Supervisor):
        def save_state(self):
            state = super().save_state()
            for entry in state["bots"].values():
                entry["paused"] = False
            return state

    assert "paused_wrong" in _probe(monkeypatch, "forget_pause", ForgetPause).kinds()


def test_a_foreign_stamped_event_that_is_accepted_is_caught(monkeypatch) -> None:
    """An event carrying another bot's id must never become that bot's
    attention, card entry or push."""
    import jarvis_bots.supervisor as supervisor_module
    from jarvis_bots.contracts import Event

    class TrustEverything(Supervisor):
        def run_round(self, now=None):
            real = supervisor_module._checked_events
            supervisor_module._checked_events = lambda bot_id, events: (
                tuple(e for e in events if isinstance(e, Event))
                if isinstance(events, (list, tuple))
                else ()
            )
            try:
                return super().run_round(now)
            finally:
                supervisor_module._checked_events = real

    kinds = _probe(monkeypatch, "trust_everything", TrustEverything).kinds()
    assert "malformed_kept" in kinds


def test_a_successful_tick_that_rearms_early_is_caught(monkeypatch) -> None:
    class RearmEarly(Supervisor):
        def run_round(self, now=None):
            report = super().run_round(now)
            for bot_id in self._registry.ids():
                health = self._health_of(bot_id)
                if health.consecutive_failures == 0 and health.next_due_at:
                    health.next_due_at -= 60.0
            return report

    assert "schedule_wrong" in _probe(monkeypatch, "rearm_early", RearmEarly).kinds()


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_report_twice() -> None:
    """The clock is an injected cell and the only randomness is
    ``lucifer_gen.seed``, so a run is a pure function of its arguments."""
    first = run_gate(300, SEED)
    second = run_gate(300, SEED)
    assert first.to_dict() == second.to_dict()
    assert first.counts == second.counts
    assert [str(p) for p in first.problems] == [str(p) for p in second.problems]


@pytest.mark.parametrize("defect", DEFECTS)
def test_a_defect_run_is_deterministic_too(defect: str) -> None:
    assert run_gate(ROUNDS, SEED, defect).to_dict() == run_gate(
        ROUNDS, SEED, defect
    ).to_dict()


def test_a_different_seed_gives_a_different_run() -> None:
    """Otherwise the seed is decoration and one script is all there is."""
    a = run_gate(ROUNDS, SEEDS[0]).counts
    b = run_gate(ROUNDS, SEEDS[1]).counts
    assert a != b


def test_the_fleet_is_rebuilt_from_scratch_each_run() -> None:
    """No state leaks between runs through a module-level default."""
    run_gate(ROUNDS, SEED, "attention_leak")
    assert run_gate(ROUNDS, SEED).ok


# --------------------------------------------------------------------------
# Speed
# --------------------------------------------------------------------------


def test_five_hundred_rounds_finish_in_under_twenty_seconds() -> None:
    """The one test that reads a wall clock, and it does so from outside
    the gate: nothing inside it may call ``time.time()``."""
    started = time.monotonic()
    report = run_gate(500, SEED)
    elapsed = time.monotonic() - started
    assert report.ok, report.summary()
    assert report.counts["rounds"] == 500
    assert elapsed < 20.0, f"500 rounds took {elapsed:.1f}s"


# --------------------------------------------------------------------------
# The report, and the seams around it
# --------------------------------------------------------------------------


def test_the_report_is_json() -> None:
    report = run_gate(ROUNDS, SEED, "realert_every_round")
    text = json.dumps(report.to_dict(), sort_keys=True)
    back = json.loads(text)
    assert back["ok"] is False
    assert back["inject_defect"] == "realert_every_round"
    assert back["n_rounds"] == ROUNDS
    assert back["seed"].startswith("0x")
    assert any(p["kind"] == "realerted" for p in back["problems"])


def test_the_report_carries_a_repro_line() -> None:
    report = run_gate(ROUNDS, SEED, "no_quarantine")
    assert "run_gate(" in report.repro and "no_quarantine" in report.repro
    assert str(ROUNDS) in report.repro


def test_the_cli_seam_reads_ok_and_lines() -> None:
    """``jarvis_bots.cli gate`` reads ``ok`` and ``lines`` off whatever the
    gate returns, and supplies the defect under the name ``defect``."""
    report = run_gate(MIN_ROUNDS, SEED, defect="state_drift")
    assert report.inject_defect == "state_drift"
    assert report.ok is False
    assert isinstance(report.lines, tuple) and report.lines
    assert all(isinstance(line, str) for line in report.lines)
    assert run_gate(MIN_ROUNDS, SEED).ok is True


def test_the_two_defect_argument_names_may_not_disagree() -> None:
    with pytest.raises(GateError):
        run_gate(MIN_ROUNDS, SEED, "no_isolation", defect="state_drift")
    # ...but agreeing is fine.
    assert run_gate(MIN_ROUNDS, SEED, "no_isolation", defect="no_isolation").ok is False


def test_problems_cap_per_kind_but_are_counted() -> None:
    """A broken rule can fail on every one of 500 rounds; the report stays
    readable and says how much it left out."""
    report = run_gate(400, SEED, "attention_leak")
    per_kind: Dict[str, int] = {}
    for problem in report.problems:
        per_kind[problem.kind] = per_kind.get(problem.kind, 0) + 1
    assert per_kind and max(per_kind.values()) <= validate.MAX_PER_KIND
    assert report.suppressed > 0


def test_a_problem_prints_where_it_happened() -> None:
    text = str(run_gate(ROUNDS, SEED, "tick_paused").problems[0])
    assert "bot=" in text and "round=" in text


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rounds", [0, 1, 10, MIN_ROUNDS - 1, -5])
def test_a_run_too_short_to_prove_anything_is_refused(rounds: int) -> None:
    """A gate that can be asked for a vacuous pass will be."""
    with pytest.raises(GateError):
        run_gate(rounds, SEED)


def test_an_unknown_defect_is_refused() -> None:
    with pytest.raises(GateError) as caught:
        run_gate(ROUNDS, SEED, "no_such_defect")
    assert "no_such_defect" in str(caught.value)
    for defect in DEFECTS:
        assert defect in str(caught.value)


def test_n_rounds_must_be_an_int() -> None:
    for bad in (None, 200.0, "200", True):
        with pytest.raises(GateError):
            run_gate(bad, SEED)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# The fleet really contains what the gate claims
# --------------------------------------------------------------------------


def test_the_fleet_covers_every_required_behaviour() -> None:
    """Nine bots, one per hazard, with the control registered last so a
    broken isolation boundary shows up as a missed round."""
    gate = validate._Gate(ROUNDS, SEED, None)
    ids = [bot.info.id for bot in gate.bots]
    assert ids == [
        "always_fails",
        "flaky",
        "malformed",
        "flapper",
        "burster",
        "pauser",
        "resolver",
        "slowpoke",
        "healthy",
    ]
    assert ids[-1] == "healthy", "the control must tick after every hazard"
    assert gate.registry.ids() == ids
    assert gate.max_keys == 5

    # Two different bots hold the *same* key string, so "whose question is
    # this?" is a thing the gate can get wrong. A supervisor that closes a
    # key under whatever bot asks looked identical to a correct one while
    # every bot used a key nobody else did.
    shared = [b.info.id for b in gate.bots if validate.SHARED_KEY in b.keys]
    assert sorted(shared) == ["flapper", "resolver"]

    # And they close it by the two different mechanisms: the app calling
    # clear_attention, and the bot emitting RESOLVED_FLAG. Before the
    # second one was in the fleet, _close_attention was called zero times
    # in a 240-round run and every way of breaking it left the gate green.
    assert isinstance(gate.by_id["resolver"], validate._ResolverBot)


def test_the_intermittent_bot_can_never_be_mistaken_for_a_broken_one() -> None:
    """Its failure pattern has no run of three, wrap included, so it can
    never reach ``QUARANTINE_AFTER_FAILURES`` however the rounds fall."""
    for seed in SEEDS + (0xDEAD_BEEF, 0x1, 0xFFFF_FFFF_FFFF_FFFF):
        pattern = validate._Gate(ROUNDS, seed, None).by_id["flaky"]._pattern
        assert any(pattern) and not all(pattern)
        doubled = pattern * 2
        runs = [len(r) for r in re.findall(r"1+", "".join("1" if p else "0" for p in doubled))]
        assert max(runs) < QUARANTINE_AFTER_FAILURES, pattern


def test_the_slow_bot_is_slow_without_sleeping() -> None:
    """It moves the injected clock, which is what the supervisor times a
    tick with.  contracts.py: nothing calls ``time.time()``."""
    assert validate.SLOW_BY_S > validate.SLOW_TICK_S
    assert validate.SLOW_BY_S < validate.ROUND_S, "the clock would have to go backwards"
    gate = validate._Gate(MIN_ROUNDS, SEED, None)
    before = gate.clock()
    gate.by_id["slowpoke"].tick(before)
    assert gate.clock() - before > validate.SLOW_TICK_S


def test_the_malformed_bot_ships_all_four_bad_shapes() -> None:
    shapes = set(validate._MalformedBot.SHAPES)
    assert shapes == {
        "clean",
        "not_a_sequence",
        "non_event_member",
        "bare_event",
        "foreign_id",
    }
    # Alternating with a clean tick, so it keeps producing them instead of
    # disappearing into quarantine.
    assert validate._MalformedBot.SHAPES[1::2] == ("clean",) * 4
    counts = run_gate(ROUNDS, SEED).counts
    assert counts["rejections"] >= 8


def test_the_gate_itself_holds_the_contract_vocabulary() -> None:
    """The card states the launcher validator accepts are BotState's own."""
    assert validate.CARD_STATES == frozenset(s.ui for s in BotState)
    assert validate.BADGE_STATES == frozenset({"ok", "warn", "error"})
    assert set(validate.CARD_KEYS) == {
        "id", "name", "blurb", "kind", "state", "attention", "href",
        "can_pause", "stats", "last_event",
    }


# --------------------------------------------------------------------------
# The boundary, and the injection rules
# --------------------------------------------------------------------------


def _source() -> str:
    return (ROOT / "jarvis_bots" / "validate.py").read_text(encoding="utf-8")


def _code() -> str:
    """The module with its comments and string literals removed.

    The prose in this module says "no sockets" and "nothing calls
    time.time()", so a scan of the raw text would flag the very sentences
    that promise the opposite.  What matters is what the code does, so the
    scans below read the tokens that are not comments or strings.
    """
    import io
    import tokenize

    kept = []
    for token in tokenize.generate_tokens(io.StringIO(_source()).readline):
        if token.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        kept.append(token.string)
    return " ".join(kept)


def test_no_wall_clock_in_the_gate() -> None:
    """contracts.py: "Time is injected everywhere. Nothing here calls
    time.time()." The clock is an injected cell, and the slow bot is slow
    because it advances that cell."""
    code = _code()
    for word in ("time", "sleep", "monotonic", "perf_counter", "datetime"):
        assert word not in code.split(), f"the gate's code names {word!r}"
    assert not re.search(r"^import time$", _source(), re.MULTILINE)


def test_randomness_only_from_lucifer_gen() -> None:
    source = _source()
    assert not re.search(r"^import random$", source, re.MULTILINE)
    assert not re.search(r"^from random import", source, re.MULTILINE)
    assert "random" not in _code().split()
    assert "from lucifer_gen.seed import" in source
    assert "SeedFields.parse" in source


def test_no_network_anywhere() -> None:
    code = _code().lower()
    for word in ("socket", "urllib", "requests", "urlopen", "httplib"):
        assert word not in code.split(), f"the gate's code names {word!r}"


def test_nothing_here_transacts() -> None:
    """SCOPE: this framework schedules bots and surfaces what they find.
    The widest thing a bot does is return an event with a link."""
    code = _code().lower()
    for word in (
        "add_to_cart", "addtocart", "checkout", "payment", "credit_card",
        "purchase", "captcha", "proxy_rotate", "bypass",
    ):
        assert word not in code, f"the gate's code names {word!r}"


def test_no_money_to_get_wrong() -> None:
    """There is no money in this lane at all, which is the only way to be
    sure none of it is a float."""
    code = _code().lower()
    for word in ("cents", "price", "budget", "currency"):
        assert word not in code, f"the gate's code names {word!r}"
    assert not re.search(r"/\s*100", _source())


def test_the_gate_does_not_edit_what_it_tests() -> None:
    """The defects are subclasses.  contracts.py, the supervisor, the
    registry and the launcher files are imported and never touched."""
    source = _source()
    assert "class _NoIsolation(Supervisor)" in source
    for forbidden in ("monkeypatch", "setattr(Supervisor", "contracts.QUARANTINE"):
        assert forbidden not in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
