"""Tests for :mod:`jarvis_bots.bots.health_bot`.

Run from the repository root::

    python3 -m pytest tests/test_health_bot.py -q

Nothing here reads a wall clock or opens a socket, with one deliberate
exception: the two tests for :func:`~jarvis_bots.bots.health_bot.urllib_probe`
stand up a ``http.server`` on an ephemeral loopback port, because that
function *is* the network and testing it against a mock would only test the
mock.  No external host is contacted.

What is covered, and why
------------------------
Every rule at its boundary, because every one of these rules is a promise
about *when* the owner is interrupted:

* a critical check is silent on one failure and an ACTION on exactly two;
* a non-critical one is a NOTICE until exactly three;
* recovery closes the key the framework's way (below ACTION, same key,
  ``resolved=True``), so the badge empties;
* a 200 carrying a 300 byte HTML body from an 'asset' check is the hosting
  fallback and is an ACTION -- the failure mode that hides a bad deploy;
* version drift after a deploy names both versions, and a matching version
  says nothing;
* frontend up with its paired backend down is *one* alert, the orphaned
  one, not a pile of independent ones;
* a probe that raises on every check still completes the round;
* the snapshot carries the counters a restart would otherwise re-learn
  wrongly.
"""

from __future__ import annotations

import http.server
import json
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis_bots.bots.health_bot import (  # noqa: E402
    BOT_ID,
    CRITICAL_FAILS_FOR_ACTION,
    DRIFT_TICKS_FOR_ACTION,
    INFO,
    LATENCY_TICKS_FOR_NOTICE,
    MAX_BODY_BYTES,
    NONCRITICAL_FAILS_FOR_ACTION,
    ORPHANED_KEY,
    VERSION_DRIFT_KEY,
    Check,
    CheckResult,
    HealthBot,
    HealthBotError,
    asset_key,
    build,
    build_request,
    down_key,
    latency_key,
    parse_version,
    urllib_probe,
)
from jarvis_bots.contracts import BotState, Severity  # noqa: E402
from jarvis_bots.registry import BotRegistry  # noqa: E402
from jarvis_bots.supervisor import RESOLVED_FLAG, Supervisor  # noqa: E402


# ---------------------------------------------------------------------------
# Injected clock and probe
# ---------------------------------------------------------------------------


class FakeClock:
    """An injected clock (contracts.py: "Time is injected everywhere")."""

    def __init__(self, start: float = 1_000_000.0, step: float = 180.0) -> None:
        self.t = float(start)
        self.step = float(step)

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: Optional[float] = None) -> float:
        self.t += self.step if seconds is None else float(seconds)
        return self.t


class StubProbe:
    """A scripted probe.  ``answers[name]`` is the result for that check, and
    ``raises`` makes every call blow up."""

    def __init__(self, answers: Optional[Dict[str, CheckResult]] = None) -> None:
        self.answers: Dict[str, CheckResult] = dict(answers or {})
        self.raises = False
        self.calls: List[str] = []

    def __call__(self, check: Check) -> CheckResult:
        self.calls.append(check.name)
        if self.raises:
            raise RuntimeError("probe exploded")
        answer = self.answers.get(check.name)
        if answer is None:
            return ok(check.name)
        return answer

    def set(self, name: str, result: CheckResult) -> None:
        self.answers[name] = result


def ok(name: str, *, body: str = "ok", status: int = 200, ms: float = 12.0,
       body_bytes: Optional[int] = None) -> CheckResult:
    return CheckResult(
        name=name,
        ok=True,
        status=status,
        elapsed_ms=ms,
        body_excerpt=body,
        body_bytes=len(body.encode()) if body_bytes is None else body_bytes,
    )


def down(name: str, *, error: str = "ConnectionRefusedError", status: Optional[int] = None,
         ms: float = 5.0) -> CheckResult:
    return CheckResult(name=name, ok=False, status=status, elapsed_ms=ms, error=error)


def actions(events: Sequence) -> List:
    return [e for e in events if e.severity is Severity.ACTION]


def notices(events: Sequence) -> List:
    return [e for e in events if e.severity is Severity.NOTICE]


def keyed(events: Sequence, key: str) -> List:
    return [e for e in events if e.attention_key == key]


# ---------------------------------------------------------------------------
# Fixtures: the app this bot was written for
# ---------------------------------------------------------------------------

BACKEND = Check(name="backend", url="http://127.0.0.1:8080/healthz", kind="backend",
                critical=True, expect_contains="ok")
FRONTEND = Check(name="frontend", url="https://app.example/", kind="frontend",
                 critical=True)
BUNDLE = Check(name="bundle", url="https://app.example/assets/main.js", kind="asset",
               expect_contains="//# sourceMappingURL")
VERSION = Check(name="version", url="https://app.example/version.json", kind="version")
DOCS = Check(name="docs", url="https://app.example/docs", kind="frontend", critical=False)


def make_bot(checks=(BACKEND,), probe=None, clock=None, **kwargs) -> tuple:
    clock = clock or FakeClock()
    probe = probe or StubProbe()
    return HealthBot(list(checks), probe, clock, **kwargs), probe, clock


def run(bot: HealthBot, clock: FakeClock, rounds: int = 1) -> List:
    """Tick ``rounds`` times, advancing the injected clock between them."""
    events: List = []
    for _ in range(rounds):
        events.extend(bot.tick(clock.t))
        clock.advance()
    return events


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_matches_the_launcher_contract():
    assert INFO.id == BOT_ID == "health"
    assert INFO.name == "App health"
    assert INFO.kind == "radar"
    assert INFO.interval_s == 180.0
    assert INFO.href == "/bots/health"


def test_the_bot_opens_no_sockets_without_a_probe():
    with pytest.raises(HealthBotError):
        HealthBot([BACKEND], "not callable", FakeClock())
    with pytest.raises(HealthBotError):
        HealthBot([], StubProbe(), FakeClock())


# ---------------------------------------------------------------------------
# A critical check: one blip is not an alarm, two is
# ---------------------------------------------------------------------------


def test_one_failure_of_a_critical_check_is_silent():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    events = run(bot, clock, 1)
    assert events == []
    assert bot.open_attention_keys() == []


def test_two_consecutive_failures_raise_exactly_one_action():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    events = run(bot, clock, 2)
    raised = actions(events)
    assert len(raised) == 1
    (event,) = raised
    assert event.attention_key == down_key("backend")
    assert event.wants_attention
    assert "failing" in event.text
    # "say in the text how long it has been failing"
    assert "180s" in event.text or "min" in event.text
    assert bot.open_attention_keys() == [down_key("backend")]


def test_a_standing_failure_does_not_re_raise_every_tick():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    events = run(bot, clock, 6)
    assert len(actions(events)) == 1


def test_exactly_two_is_the_boundary_for_a_critical_check():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    first = bot.tick(clock.t)
    clock.advance()
    assert actions(first) == []
    second = bot.tick(clock.t)
    assert len(actions(second)) == 1
    assert CRITICAL_FAILS_FOR_ACTION == 2


def test_a_failure_that_recovers_in_between_starts_the_count_again():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    run(bot, clock, 1)
    probe.set("backend", ok("backend"))
    run(bot, clock, 1)
    probe.set("backend", down("backend"))
    events = run(bot, clock, 1)
    assert actions(events) == []


def test_recovery_resolves_the_key_the_frameworks_way():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    run(bot, clock, 2)
    assert bot.open_attention_keys() == [down_key("backend")]

    probe.set("backend", ok("backend"))
    events = run(bot, clock, 1)
    (resolution,) = keyed(events, down_key("backend"))
    # Exactly the convention poke_bot and the scaffold template use.
    assert resolution.severity < Severity.ACTION
    assert resolution.data[RESOLVED_FLAG] is True
    assert not resolution.wants_attention
    assert bot.open_attention_keys() == []


def test_a_recovered_fault_that_returns_alerts_again():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    run(bot, clock, 2)
    probe.set("backend", ok("backend"))
    run(bot, clock, 1)
    probe.set("backend", down("backend"))
    events = run(bot, clock, 2)
    assert len(actions(events)) == 1


# ---------------------------------------------------------------------------
# A non-critical check: three
# ---------------------------------------------------------------------------


def test_a_non_critical_check_is_a_notice_before_it_is_an_action():
    bot, probe, clock = make_bot(checks=(DOCS,))
    probe.set("docs", down("docs"))

    first = bot.tick(clock.t)
    clock.advance()
    assert [e.severity for e in first] == [Severity.NOTICE]
    assert not first[0].wants_attention

    second = bot.tick(clock.t)
    clock.advance()
    assert [e.severity for e in second] == [Severity.NOTICE]

    third = bot.tick(clock.t)
    assert len(actions(third)) == 1
    assert third[0].attention_key == down_key("docs")
    assert NONCRITICAL_FAILS_FOR_ACTION == 3


def test_exactly_three_is_the_boundary_for_a_non_critical_check():
    bot, probe, clock = make_bot(checks=(DOCS,))
    probe.set("docs", down("docs"))
    two_rounds = run(bot, clock, 2)
    assert actions(two_rounds) == []
    assert bot.open_attention_keys() == []
    third = run(bot, clock, 1)
    assert len(actions(third)) == 1
    assert bot.open_attention_keys() == [down_key("docs")]


def test_a_non_critical_recovery_resolves_too():
    bot, probe, clock = make_bot(checks=(DOCS,))
    probe.set("docs", down("docs"))
    run(bot, clock, 3)
    probe.set("docs", ok("docs"))
    events = run(bot, clock, 1)
    (resolution,) = keyed(events, down_key("docs"))
    assert resolution.data[RESOLVED_FLAG] is True
    assert bot.open_attention_keys() == []


# ---------------------------------------------------------------------------
# Version drift: the deploy that shipped nothing
# ---------------------------------------------------------------------------


def _version_bot(live: str, expected: Optional[str] = None):
    bot, probe, clock = make_bot(checks=(VERSION,), expected_version=expected)
    probe.set("version", ok("version", body=json.dumps({"version": live})))
    return bot, probe, clock


def test_version_drift_after_a_deploy_alerts_and_names_both_versions():
    bot, probe, clock = _version_bot(live="1.4.0")
    bot.deployed("1.5.0", at=clock.t)

    first = bot.tick(clock.t)
    clock.advance()
    assert actions(first) == []  # one tick of drift is not yet an alarm

    second = bot.tick(clock.t)
    (event,) = actions(second)
    assert event.attention_key == VERSION_DRIFT_KEY
    assert "1.4.0" in event.text and "1.5.0" in event.text
    assert event.data["live_version"] == "1.4.0"
    assert event.data["expected_version"] == "1.5.0"
    assert DRIFT_TICKS_FOR_ACTION == 2


def test_matching_versions_stay_silent():
    bot, probe, clock = _version_bot(live="1.5.0")
    bot.deployed("1.5.0", at=clock.t)
    events = run(bot, clock, 5)
    assert events == []
    assert bot.live_version == "1.5.0"
    assert bot.open_attention_keys() == []


def test_drift_without_a_declared_deploy_is_not_drift():
    """Nothing was deployed, so there is nothing the live build is failing
    to be.  The rule arms on deployed(), not on the bot starting up."""
    bot, probe, clock = _version_bot(live="1.4.0")
    assert run(bot, clock, 4) == []


def test_a_fixed_deploy_resolves_the_drift_key():
    bot, probe, clock = _version_bot(live="1.4.0")
    bot.deployed("1.5.0", at=clock.t)
    run(bot, clock, 2)
    assert bot.open_attention_keys() == [VERSION_DRIFT_KEY]

    probe.set("version", ok("version", body=json.dumps({"version": "1.5.0"})))
    events = run(bot, clock, 1)
    (resolution,) = keyed(events, VERSION_DRIFT_KEY)
    assert resolution.data[RESOLVED_FLAG] is True
    assert resolution.severity < Severity.ACTION
    assert bot.open_attention_keys() == []


def test_a_new_deploy_restarts_the_drift_count():
    bot, probe, clock = _version_bot(live="1.4.0")
    bot.deployed("1.5.0", at=clock.t)
    run(bot, clock, 1)
    bot.deployed("1.6.0", at=clock.t)  # the fix was re-deployed
    events = run(bot, clock, 1)
    assert actions(events) == []


def test_a_version_endpoint_that_is_down_is_not_drift():
    bot, probe, clock = _version_bot(live="1.4.0")
    bot.deployed("1.5.0", at=clock.t)
    probe.set("version", down("version"))
    events = run(bot, clock, 4)
    assert keyed(events, VERSION_DRIFT_KEY) == []


@pytest.mark.parametrize(
    "body,expected",
    [
        ('{"version": "2.0.1"}', "2.0.1"),
        ('{"build": "abc123"}', "abc123"),
        ("  3.1.4  \n", "3.1.4"),
        ("<html><body>nope</body></html>", None),
        ("", None),
    ],
)
def test_parse_version_reads_what_a_deploy_script_writes(body, expected):
    assert parse_version(body) == expected


# ---------------------------------------------------------------------------
# Asset presence: the hosting fallback
# ---------------------------------------------------------------------------


def test_hosting_fallback_index_html_for_a_missing_js_asset_alerts():
    """A 200 with a 300 byte HTML body is Firebase serving index.html for an
    asset the build dropped.  Status-code monitoring calls this healthy."""
    bot, probe, clock = make_bot(checks=(BUNDLE,))
    fallback = "<!doctype html><html><head><title>App</title></head><body>" + "x" * 240
    assert 200 < len(fallback.encode()) < 512
    probe.set("bundle", ok("bundle", body=fallback, body_bytes=len(fallback.encode())))

    events = bot.tick(clock.t)
    (event,) = actions(events)
    assert event.attention_key == asset_key("bundle")
    assert "not the asset" in event.text
    assert event.data["body_bytes"] == len(fallback.encode())
    assert event.data["floor_bytes"] == 512


def test_an_asset_with_proper_content_is_silent():
    bot, probe, clock = make_bot(checks=(BUNDLE,))
    bundle = "console.log(1);" * 200 + "//# sourceMappingURL=main.js.map"
    probe.set(
        "bundle",
        # A real bundle: the marker is in the first 200 characters the probe
        # kept, and the body is far over the floor.
        CheckResult(
            name="bundle",
            ok=True,
            status=200,
            elapsed_ms=30.0,
            body_excerpt="//# sourceMappingURL=main.js.map",
            body_bytes=len(bundle.encode()),
        ),
    )
    assert bot.tick(clock.t) == ()
    assert bot.open_attention_keys() == []


def test_an_asset_over_the_floor_without_its_marker_still_alerts():
    bot, probe, clock = make_bot(checks=(BUNDLE,))
    probe.set(
        "bundle",
        CheckResult(name="bundle", ok=True, status=200, body_excerpt="<!doctype html>",
                    body_bytes=4096),
    )
    (event,) = actions(bot.tick(clock.t))
    assert event.attention_key == asset_key("bundle")
    assert "sourceMappingURL" in event.text


def test_the_asset_floor_is_configurable():
    bot, probe, clock = make_bot(checks=(BUNDLE,), asset_floor_bytes=64)
    small = "//# sourceMappingURL=main.js.map"
    probe.set("bundle", ok("bundle", body=small, body_bytes=100))
    assert bot.tick(clock.t) == ()


def test_an_asset_404_is_an_ordinary_failure_not_an_asset_fault():
    """A 404 is the down rule's business; the asset rule exists only for the
    200 that lies."""
    bot, probe, clock = make_bot(checks=(BUNDLE,))
    probe.set("bundle", CheckResult(name="bundle", ok=False, status=404))
    events = run(bot, clock, 3)
    assert keyed(events, asset_key("bundle")) == []
    assert [e.attention_key for e in actions(events)] == [down_key("bundle")]


def test_a_recovered_asset_resolves_its_key():
    bot, probe, clock = make_bot(checks=(BUNDLE,))
    probe.set("bundle", ok("bundle", body="<!doctype html>", body_bytes=300))
    run(bot, clock, 1)
    assert bot.open_attention_keys() == [asset_key("bundle")]
    probe.set("bundle", ok("bundle", body="//# sourceMappingURL=x", body_bytes=9000))
    events = run(bot, clock, 1)
    (resolution,) = keyed(events, asset_key("bundle"))
    assert resolution.data[RESOLVED_FLAG] is True
    assert bot.open_attention_keys() == []


# ---------------------------------------------------------------------------
# The reachability pair
# ---------------------------------------------------------------------------


def test_frontend_up_backend_down_is_one_orphaned_alert_not_two_independent_ones():
    bot, probe, clock = make_bot(
        checks=(FRONTEND, BACKEND), pairs={"frontend": "backend"}
    )
    probe.set("frontend", ok("frontend"))
    probe.set("backend", down("backend", error="ConnectionRefusedError"))

    events = run(bot, clock, 3)
    raised = actions(events)
    assert [e.attention_key for e in raised] == [ORPHANED_KEY]
    (event,) = raised
    assert "page is up" in event.text
    assert "cannot reach its data" in event.text
    assert event.data["frontends"] == ["frontend"]
    assert event.data["backends"] == ["backend"]
    # The generic "backend is down" ACTION is not also standing: one fault,
    # one item in the badge.
    assert bot.open_attention_keys() == [ORPHANED_KEY]


def test_both_halves_down_is_not_an_orphaned_frontend():
    bot, probe, clock = make_bot(
        checks=(FRONTEND, BACKEND), pairs={"frontend": "backend"}
    )
    probe.set("frontend", down("frontend"))
    probe.set("backend", down("backend"))
    events = run(bot, clock, 2)
    keys = sorted(e.attention_key for e in actions(events))
    assert keys == [down_key("backend"), down_key("frontend")]
    assert ORPHANED_KEY not in bot.open_attention_keys()


def test_the_orphaned_key_resolves_when_the_backend_comes_back():
    bot, probe, clock = make_bot(
        checks=(FRONTEND, BACKEND), pairs={"frontend": "backend"}
    )
    probe.set("backend", down("backend"))
    run(bot, clock, 2)
    assert bot.open_attention_keys() == [ORPHANED_KEY]
    probe.set("backend", ok("backend"))
    events = run(bot, clock, 1)
    (resolution,) = keyed(events, ORPHANED_KEY)
    assert resolution.data[RESOLVED_FLAG] is True
    assert bot.open_attention_keys() == []


def test_pairing_is_configuration_and_is_checked():
    with pytest.raises(HealthBotError):
        HealthBot([FRONTEND, BACKEND], StubProbe(), FakeClock(),
                  pairs={"frontend": "nope"})
    with pytest.raises(HealthBotError):
        HealthBot([FRONTEND, BACKEND], StubProbe(), FakeClock(),
                  pairs={"frontend": "frontend"})


def test_an_unpaired_backend_still_gets_its_own_alert():
    """Without a declared pair there is no inference: the backend failing is
    the backend failing."""
    bot, probe, clock = make_bot(checks=(FRONTEND, BACKEND))
    probe.set("backend", down("backend"))
    events = run(bot, clock, 2)
    assert [e.attention_key for e in actions(events)] == [down_key("backend")]


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


def test_latency_needs_three_consecutive_ticks():
    bot, probe, clock = make_bot(checks=(BACKEND,), latency_ms=100.0)
    probe.set("backend", ok("backend", ms=450.0))

    assert bot.tick(clock.t) == ()
    clock.advance()
    assert bot.tick(clock.t) == ()
    clock.advance()
    third = bot.tick(clock.t)
    assert [e.severity for e in third] == [Severity.NOTICE]
    assert third[0].attention_key == latency_key("backend")
    assert not third[0].wants_attention
    assert LATENCY_TICKS_FOR_NOTICE == 3


def test_a_fast_tick_resets_the_latency_streak():
    bot, probe, clock = make_bot(checks=(BACKEND,), latency_ms=100.0)
    probe.set("backend", ok("backend", ms=450.0))
    run(bot, clock, 2)
    probe.set("backend", ok("backend", ms=10.0))
    run(bot, clock, 1)
    probe.set("backend", ok("backend", ms=450.0))
    assert run(bot, clock, 2) == []


def test_persistent_slowness_says_it_once():
    bot, probe, clock = make_bot(checks=(BACKEND,), latency_ms=100.0)
    probe.set("backend", ok("backend", ms=450.0))
    events = run(bot, clock, 10)
    assert len(notices(events)) == 1


# ---------------------------------------------------------------------------
# A probe that raises
# ---------------------------------------------------------------------------


def test_a_probe_raising_on_every_check_still_completes_the_round():
    checks = (BACKEND, FRONTEND, BUNDLE, VERSION, DOCS)
    bot, probe, clock = make_bot(checks=checks)
    probe.raises = True

    first = bot.tick(clock.t)  # never propagates
    clock.advance()
    assert probe.calls == [c.name for c in checks]
    second = bot.tick(clock.t)

    # Every check is recorded as failing: the two criticals reach ACTION on
    # the second round, and each non-critical has its NOTICE.
    assert sorted(e.attention_key for e in actions(second)) == [
        down_key("backend"),
        down_key("frontend"),
    ]
    reported = {e.data.get("check") for e in list(first) + list(second)}
    assert {c.name for c in checks} <= reported
    for event in list(first) + list(second):
        if event.data.get("error"):
            assert "probe exploded" in event.data["error"]


def test_a_probe_returning_the_wrong_type_is_a_failure_not_a_crash():
    bot, probe, clock = make_bot()
    probe.set("backend", ok("backend"))
    bot._probe = lambda check: "nope"  # noqa: SLF001 - the point of the test
    events = run(bot, clock, 2)
    (event,) = actions(events)
    assert "not a CheckResult" in event.data["error"]


# ---------------------------------------------------------------------------
# The card
# ---------------------------------------------------------------------------


def test_status_reports_checks_version_and_the_slowest_check():
    bot, probe, clock = make_bot(checks=(BACKEND, BUNDLE, VERSION))
    probe.set("backend", ok("backend", ms=15.0))
    probe.set("bundle", ok("bundle", body="//# sourceMappingURL=x", body_bytes=9000, ms=240.0))
    probe.set("version", ok("version", body='{"version": "1.5.0"}', ms=30.0))
    run(bot, clock, 1)

    status = bot.status()
    assert status.state is BotState.RUNNING
    stats = {s.label: s.value for s in status.stats}
    assert stats["Checks"] == "3 of 3 passing"
    assert stats["Live version"] == "1.5.0"
    assert stats["Slowest"].startswith("bundle")
    assert "240" in stats["Slowest"]


def test_status_counts_a_failing_check_as_not_passing():
    bot, probe, clock = make_bot(checks=(BACKEND, DOCS))
    probe.set("docs", down("docs"))
    run(bot, clock, 1)
    stats = {s.label: s.value for s in bot.status().stats}
    assert stats["Checks"] == "1 of 2 passing"
    assert stats["Live version"] == "unknown"


# ---------------------------------------------------------------------------
# Snapshot and restore
# ---------------------------------------------------------------------------


def test_snapshot_restore_preserves_counters_version_and_open_keys():
    checks = (BACKEND, DOCS, VERSION)
    bot, probe, clock = make_bot(checks=checks)
    probe.set("backend", down("backend"))
    probe.set("docs", down("docs"))
    probe.set("version", ok("version", body='{"version": "1.4.0"}'))
    bot.deployed("1.5.0", at=clock.t)
    run(bot, clock, 2)

    taken = bot.snapshot()
    assert json.loads(json.dumps(taken)) == taken  # JSON-able, as promised

    fresh, fresh_probe, fresh_clock = make_bot(checks=checks, clock=FakeClock(clock.t))
    fresh.restore(taken)

    assert fresh.expected_version == "1.5.0"
    assert fresh.live_version == "1.4.0"
    assert fresh.open_attention_keys() == bot.open_attention_keys()
    assert fresh.snapshot()["checks"] == taken["checks"]
    assert fresh.snapshot()["drift_ticks"] == taken["drift_ticks"]


def test_a_restored_bot_does_not_re_alert_what_is_already_in_the_badge():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    run(bot, clock, 2)

    fresh, fresh_probe, fresh_clock = make_bot(clock=FakeClock(clock.t))
    fresh.restore(bot.snapshot())
    fresh_probe.set("backend", down("backend"))
    assert actions(run(fresh, fresh_clock, 3)) == []
    assert fresh.open_attention_keys() == [down_key("backend")]


def test_a_restored_bot_resolves_a_key_it_did_not_open_itself():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    run(bot, clock, 2)

    fresh, fresh_probe, fresh_clock = make_bot(clock=FakeClock(clock.t))
    fresh.restore(bot.snapshot())
    fresh_probe.set("backend", ok("backend"))
    (resolution,) = keyed(run(fresh, fresh_clock, 1), down_key("backend"))
    assert resolution.data[RESOLVED_FLAG] is True


def test_a_snapshot_from_a_newer_build_is_refused_not_half_read():
    bot, probe, clock = make_bot()
    with pytest.raises(HealthBotError):
        bot.restore({"version": 99, "checks": {}})


def test_pausing_forgets_the_open_keys_so_resume_is_honest():
    bot, probe, clock = make_bot()
    probe.set("backend", down("backend"))
    run(bot, clock, 2)
    bot.on_pause()
    assert bot.open_attention_keys() == []
    assert len(actions(run(bot, clock, 1))) == 1  # re-raised on resume


# ---------------------------------------------------------------------------
# Under a real supervisor
# ---------------------------------------------------------------------------


def test_the_bot_registers_and_ticks_under_a_real_supervisor():
    clock = FakeClock(start=1_000_000.0, step=200.0)
    probe = StubProbe()
    probe.set("backend", down("backend"))
    bot = HealthBot([FRONTEND, BACKEND], probe, clock, pairs={"frontend": "backend"})

    registry = BotRegistry([bot])
    supervisor = Supervisor(registry, clock)

    supervisor.run_round(clock.t)
    clock.advance()
    report = supervisor.run_round(clock.t)

    assert report.failed == 0
    assert supervisor.attention_count(BOT_ID) == 1
    (item,) = supervisor.attention_items(BOT_ID)
    assert item.key == ORPHANED_KEY

    probe.set("backend", ok("backend"))
    clock.advance()
    supervisor.run_round(clock.t)
    assert supervisor.attention_count(BOT_ID) == 0  # the badge empties itself


# ---------------------------------------------------------------------------
# urllib_probe, against a local server on an ephemeral port
# ---------------------------------------------------------------------------


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # noqa: D102 - keep the test output clean
        return

    def do_GET(self):  # noqa: N802
        if self.path == "/big":
            body = b"A" * (MAX_BODY_BYTES * 2)
            self._send(200, body, "text/plain")
        elif self.path == "/healthz":
            self._send(200, b'{"status":"ok"}', "application/json")
        elif self.path == "/moved":
            self.send_response(302)
            self.send_header("Location", "/healthz")
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/boom":
            self._send(500, b"nope", "text/plain")
        else:
            self._send(404, b"missing", "text/plain")

    def _send(self, status, body, ctype):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def local_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_build_request_is_a_no_cache_get():
    request = build_request(BUNDLE)
    assert request.get_method() == "GET"
    assert request.full_url == BUNDLE.url
    # A cached 200 from before the deploy is the one answer this bot must
    # never accept.
    assert request.get_header("Cache-control") == "no-cache"
    assert request.get_header("User-agent")


def test_build_request_refuses_a_non_http_url():
    with pytest.raises(HealthBotError):
        build_request(Check(name="f", url="file:///etc/passwd", kind="asset"))


def test_urllib_probe_against_a_local_server(local_server):
    check = Check(name="backend", url=f"{local_server}/healthz", kind="backend",
                  expect_contains='"status":"ok"', timeout_s=5.0)
    result = urllib_probe(check)
    assert result.ok
    assert result.status == 200
    assert result.name == "backend"
    assert '"status":"ok"' in result.body_excerpt
    assert result.body_bytes == len(b'{"status":"ok"}')
    assert result.error == ""
    assert result.elapsed_ms >= 0.0


def test_urllib_probe_bounds_what_it_reads(local_server):
    check = Check(name="big", url=f"{local_server}/big", kind="asset", timeout_s=5.0)
    result = urllib_probe(check, max_body_bytes=1024, excerpt_chars=50)
    assert result.ok
    assert result.body_bytes == 1024  # not the 128 kB the server offered
    assert len(result.body_excerpt) == 50


def test_urllib_probe_reports_a_redirect_instead_of_following_it(local_server):
    check = Check(name="moved", url=f"{local_server}/moved", kind="frontend",
                  timeout_s=5.0)
    result = urllib_probe(check)
    assert not result.ok
    assert result.status == 302
    assert "/healthz" in result.error
    assert "not followed" in result.error


def test_urllib_probe_reports_a_bad_status(local_server):
    check = Check(name="boom", url=f"{local_server}/boom", kind="backend", timeout_s=5.0)
    result = urllib_probe(check)
    assert not result.ok
    assert result.status == 500


def test_urllib_probe_never_raises_on_a_dead_port():
    # Port 1 on loopback: nothing external is contacted.
    check = Check(name="dead", url="http://127.0.0.1:1/", kind="backend", timeout_s=1.0)
    result = urllib_probe(check)
    assert not result.ok
    assert result.status is None
    assert result.error


def test_urllib_probe_is_what_build_injects(local_server):
    check = Check(name="backend", url=f"{local_server}/healthz", kind="backend",
                  critical=True, timeout_s=5.0)
    bot = build([check], FakeClock())
    assert bot._probe is urllib_probe  # noqa: SLF001 - the wiring is the point
    assert bot.tick(1_000_000.0) == ()
