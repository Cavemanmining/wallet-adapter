"""Tests for the service watch bot.

The claim this suite exists to defend is one sentence from the incident
that produced the bot: a ComfyUI unit crash-looped about 15,000 times
against a GPU that had vanished, and every tool that looked at it said
``active (running)``.  So the first test below is the regression test for
that incident by name --
:func:`test_crash_loop_while_active_is_one_action_the_comfyui_incident` --
and the rest are the ways the rule could be wrong in the other direction:
firing twice, never resolving, firing on a deploy, firing on restarts that
are merely old.

What is checked, and which line of which contract it comes from:

* ``contracts.py``, "the badge counts distinct open keys, so one restock
  nagging across ten ticks is one item of attention, not ten" -- one loop
  is one ACTION no matter how many ticks it survives, and a loop that is
  also momentarily ``failed`` is still one key, not two;
* ``contracts.py``, "Events are the only output" -- every failure path,
  including a probe that raises, comes back as an event and never as an
  exception;
* ``supervisor.RESOLVED_FLAG`` -- a service coming back healthy closes its
  own request, checked against a real :class:`Supervisor` and its badge
  rather than against the bot's own bookkeeping;
* ``contracts.py``, "State is the bot's, persistence is ours" -- the
  rolling window survives a simulated process restart through JSON, which
  matters because restarting the watcher is exactly when a loop is most
  likely to be in progress.

Nothing here sleeps, opens a socket, runs ``systemctl`` or reads a real
clock: the probe is a stub, the clock is a fake that only moves when a test
moves it, and the ``systemctl`` parser is exercised against captured
output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_bots.contracts import BotState, Event, Severity
from jarvis_bots.registry import BotRegistry, check_bot
from jarvis_bots.supervisor import RESOLVED_FLAG, Supervisor
from jarvis_bots.bots.service_bot import (
    BOT_ID,
    DEFAULT_RESTART_THRESHOLD,
    DEFAULT_WINDOW_S,
    INFO,
    PROBE_ATTENTION_KEY,
    PROBE_FAILURES_BEFORE_ACTION,
    SHOW_PROPERTIES,
    SNAPSHOT_VERSION,
    ServiceBot,
    ServiceBotError,
    ServiceSample,
    absent_key,
    build,
    crashloop_key,
    failed_key,
    memory_key,
    oom_key,
    parse_systemctl_show,
    stopped_key,
    systemctl_command,
    systemctl_probe,
)

SVC = "comfyui.service"


# --------------------------------------------------------------------------
# fakes: a clock that only moves when a test moves it, a probe that returns
# whatever the test last put in it
# --------------------------------------------------------------------------


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


class StubProbe:
    """The injected probe.  Holds what the next call returns, counts calls,
    and can be told to raise -- the three things every test here needs."""

    def __init__(self, samples: Sequence[ServiceSample] = ()) -> None:
        self.samples: List[ServiceSample] = list(samples)
        self.error: Optional[BaseException] = None
        self.calls = 0

    def __call__(self) -> List[ServiceSample]:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return list(self.samples)


def sample(
    name: str = SVC,
    active_state: str = "active",
    *,
    sub_state: str = "running",
    n_restarts: int = 0,
    active_enter_timestamp: Optional[float] = None,
    main_pid: Optional[int] = 4242,
    memory_bytes: Optional[int] = 512 * 1024 * 1024,
    exit_code: Optional[int] = 0,
    result: str = "success",
) -> ServiceSample:
    return ServiceSample(
        name=name,
        active_state=active_state,
        sub_state=sub_state,
        n_restarts=n_restarts,
        active_enter_timestamp=active_enter_timestamp,
        main_pid=main_pid,
        memory_bytes=memory_bytes,
        exit_code=exit_code,
        result=result,
    )


def make_bot(
    clock: FakeClock,
    probe: StubProbe,
    services: Sequence[str] = (SVC,),
    **kwargs: Any,
) -> ServiceBot:
    return ServiceBot(list(services), clock=clock, probe=probe, **kwargs)


def tick_at(
    bot: ServiceBot,
    clock: FakeClock,
    probe: StubProbe,
    at: float,
    samples: Sequence[ServiceSample],
) -> List[Event]:
    """One round: move the clock, set what the probe sees, tick."""
    clock.t = float(at)
    probe.samples = list(samples)
    return list(bot.tick(clock.t))


def actions(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.severity >= Severity.ACTION]


def keys(events: Sequence[Event]) -> List[str]:
    return [e.attention_key for e in events if e.attention_key]


def resolutions(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.data.get(RESOLVED_FLAG) is True]


# --------------------------------------------------------------------------
# the incident
# --------------------------------------------------------------------------


def test_crash_loop_while_active_is_one_action_the_comfyui_incident() -> None:
    """The regression test for the real incident.

    A unit that restarts four times in ten minutes while ``ActiveState``
    reads ``active`` every single time it is looked at.  That is what the
    ComfyUI unit looked like for days: systemd restarting it against a GPU
    that was not there, and every check reporting it up.  Exactly one
    ACTION, under the crash-loop key, naming the service, the count, the
    window and the last exit result.
    """
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    base = clock.t
    # The first look is only a baseline: NRestarts is cumulative, so the
    # window is built out of the differences after it.
    assert tick_at(bot, clock, probe, base, [sample(n_restarts=0)]) == []

    seen: List[Event] = []
    for i, restarts in enumerate((1, 2, 3, 4), start=1):
        at = base + i * 120.0
        seen.extend(
            tick_at(
                bot,
                clock,
                probe,
                at,
                [
                    sample(
                        # Up, running, and looking perfectly healthy -- for
                        # the four seconds since the last restart.
                        active_state="active",
                        sub_state="running",
                        n_restarts=restarts,
                        active_enter_timestamp=at - 4.0,
                        exit_code=1,
                        result="exit-code",
                    )
                ],
            )
        )

    raised = actions(seen)
    assert len(raised) == 1, [e.text for e in raised]
    event = raised[0]
    assert event.attention_key == crashloop_key(SVC) == f"svc:crashloop:{SVC}"
    assert event.severity is Severity.ACTION
    assert event.wants_attention
    assert event.href == INFO.href
    assert event.bot_id == BOT_ID

    text = event.text
    assert SVC in text
    assert "4 times" in text
    assert "10 min" in text  # the window, in words
    assert "exit-code" in text
    assert "active" in text  # the contradiction is the finding
    assert "failing fast" in text
    assert event.data["restarts"] == 4
    assert event.data["window_s"] == DEFAULT_WINDOW_S
    assert event.data["active_state"] == "active"


def test_repeat_ticks_do_not_duplicate_the_crash_loop() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    at = _drive_into_crash_loop(bot, clock, probe)

    for i in range(1, 6):
        later = at + i * 30.0
        events = tick_at(
            bot,
            clock,
            probe,
            later,
            [sample(n_restarts=4 + i, active_enter_timestamp=later - 3.0,
                    result="exit-code", exit_code=1)],
        )
        assert events == [], [e.text for e in events]
    assert bot.open_attention_keys() == [crashloop_key(SVC)]


def test_recovery_resolves_the_crash_loop() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    at = _drive_into_crash_loop(bot, clock, probe)

    # An hour later, still the same NRestarts: the restarts have aged out
    # of the window and the unit has been up the whole time.
    recovered = at + 3600.0
    events = tick_at(
        bot,
        clock,
        probe,
        recovered,
        [sample(n_restarts=4, active_enter_timestamp=at, result="success")],
    )
    assert len(events) == 1
    resolution = events[0]
    assert resolution.attention_key == crashloop_key(SVC)
    assert resolution.severity is Severity.NOTICE
    assert resolution.data[RESOLVED_FLAG] is True
    # Below ACTION, so the supervisor's close path and its open path can
    # never both fire for one event.
    assert not resolution.wants_attention
    assert bot.open_attention_keys() == []

    # And it does not resolve twice.
    assert tick_at(
        bot, clock, probe, recovered + 120.0,
        [sample(n_restarts=4, active_enter_timestamp=at)],
    ) == []


def test_restarts_spread_thinly_over_a_long_period_do_not_alert() -> None:
    """Four restarts, but one every twenty minutes: not a loop.

    This is the test that proves the window is a window.  Without the
    ageing-out the same four restarts would look identical to the
    incident.
    """
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    at = clock.t
    seen = tick_at(bot, clock, probe, at, [sample(n_restarts=0)])
    for i, restarts in enumerate((1, 2, 3, 4), start=1):
        at = clock.t + 1200.0
        seen.extend(
            tick_at(
                bot, clock, probe, at,
                [sample(n_restarts=restarts, active_enter_timestamp=at - 600.0)],
            )
        )
    assert seen == [], [e.text for e in seen]
    assert bot.open_attention_keys() == []


# --------------------------------------------------------------------------
# boundaries
# --------------------------------------------------------------------------


def test_exactly_at_the_restart_threshold_is_silent() -> None:
    """Three restarts in the window, with a threshold of three.

    The rule is "more than", and it has to be somewhere: three restarts in
    ten minutes is a bad afternoon with a deploy in it, and a badge that
    fires on it is a badge the owner learns to ignore before the fourth
    one matters.
    """
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    at = clock.t
    seen = tick_at(bot, clock, probe, at, [sample(n_restarts=0)])
    for i, restarts in enumerate((1, 2, 3), start=1):
        seen.extend(
            tick_at(bot, clock, probe, at + i * 60.0, [sample(n_restarts=restarts)])
        )
    assert seen == []
    assert bot.open_attention_keys() == []

    # One more, still inside the window, and it is a loop.
    events = tick_at(bot, clock, probe, at + 240.0, [sample(n_restarts=4)])
    assert keys(events) == [crashloop_key(SVC)]


def test_a_restart_exactly_on_the_window_edge_still_counts() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe, window_s=600.0, restart_threshold=3)

    base = clock.t
    tick_at(bot, clock, probe, base, [sample(n_restarts=0)])
    tick_at(bot, clock, probe, base + 1.0, [sample(n_restarts=1)])
    tick_at(bot, clock, probe, base + 2.0, [sample(n_restarts=2)])
    assert tick_at(bot, clock, probe, base + 3.0, [sample(n_restarts=3)]) == []

    # now - window_s == base + 1.0 exactly: the oldest restart is on the
    # edge, and the edge is inclusive, so this is four in the window.
    events = tick_at(bot, clock, probe, base + 601.0, [sample(n_restarts=4)])
    assert keys(events) == [crashloop_key(SVC)]
    assert events[0].data["restarts"] == 4


def test_a_restart_one_second_past_the_window_edge_does_not_count() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe, window_s=600.0, restart_threshold=3)

    base = clock.t
    tick_at(bot, clock, probe, base, [sample(n_restarts=0)])
    tick_at(bot, clock, probe, base + 1.0, [sample(n_restarts=1)])
    tick_at(bot, clock, probe, base + 2.0, [sample(n_restarts=2)])
    tick_at(bot, clock, probe, base + 3.0, [sample(n_restarts=3)])

    # One second later than the test above: the first restart has fallen
    # out, so this is three in the window and three is not a loop.
    events = tick_at(bot, clock, probe, base + 602.0, [sample(n_restarts=4)])
    assert events == [], [e.text for e in events]
    assert bot.open_attention_keys() == []


def test_a_single_deploy_restart_is_silent() -> None:
    """One restart, four seconds of uptime, nothing said.

    "Uptime under 60s counts toward the crash-loop window but is not on
    its own an alert": a deploy restarts things, and a watcher that
    shouted about it would be muted within a week.
    """
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    at = clock.t
    tick_at(bot, clock, probe, at, [sample(n_restarts=7, active_enter_timestamp=at - 86400)])
    deployed = at + 120.0
    events = tick_at(
        bot, clock, probe, deployed,
        [sample(n_restarts=8, active_enter_timestamp=deployed - 4.0)],
    )
    assert events == [], [e.text for e in events]
    assert bot.open_attention_keys() == []
    assert bot.status().state is BotState.RUNNING


# --------------------------------------------------------------------------
# the other rules
# --------------------------------------------------------------------------


def test_a_failed_service_alerts() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    at = clock.t
    tick_at(bot, clock, probe, at, [sample(n_restarts=0)])
    events = tick_at(
        bot, clock, probe, at + 120.0,
        [sample(active_state="failed", sub_state="failed", n_restarts=0,
                exit_code=1, result="exit-code")],
    )
    assert keys(events) == [failed_key(SVC)]
    assert events[0].severity is Severity.ACTION
    assert "failed" in events[0].text and SVC in events[0].text

    # and it resolves when the unit is up again
    back = tick_at(bot, clock, probe, at + 240.0, [sample(n_restarts=0)])
    assert resolutions(back) and back[0].attention_key == failed_key(SVC)
    assert bot.open_attention_keys() == []


def test_a_stopped_service_alerts_distinctly_from_a_crashed_one() -> None:
    """Stopped and crash-looping are different questions, so different keys.

    A unit that was up and is now down with the restart counter untouched
    was *stopped*: nothing is bringing it back, and the fix is to start it.
    A unit that keeps dying is a crash loop, and starting it again fixes
    nothing.
    """
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    at = clock.t
    tick_at(bot, clock, probe, at, [sample(n_restarts=2)])
    events = tick_at(
        bot, clock, probe, at + 120.0,
        [sample(active_state="inactive", sub_state="dead", n_restarts=2,
                main_pid=None, memory_bytes=None, result="success")],
    )
    assert keys(events) == [stopped_key(SVC)]
    assert events[0].severity is Severity.ACTION
    assert stopped_key(SVC) != crashloop_key(SVC)
    assert "stopped" in events[0].text

    # It stays one open request while it stays down, and closes when the
    # unit is started again.
    assert tick_at(
        bot, clock, probe, at + 240.0,
        [sample(active_state="inactive", sub_state="dead", n_restarts=2)],
    ) == []
    back = tick_at(bot, clock, probe, at + 360.0, [sample(n_restarts=2)])
    assert [e.attention_key for e in resolutions(back)] == [stopped_key(SVC)]


def test_a_crash_loop_is_not_also_reported_as_stopped_or_failed() -> None:
    """One fault, one key.

    A looping unit dips through ``activating``, ``failed`` and
    ``inactive`` on its way round.  Reporting each dip under its own key
    would turn one crash loop into three items in the badge.
    """
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    at = _drive_into_crash_loop(bot, clock, probe, states=("activating", "active"))

    seen: List[Event] = []
    for i, state in enumerate(("failed", "activating", "inactive", "active"), start=1):
        later = at + i * 20.0
        seen.extend(
            tick_at(
                bot, clock, probe, later,
                [sample(active_state=state, sub_state=state, n_restarts=4 + i,
                        active_enter_timestamp=later - 2.0, result="exit-code",
                        exit_code=1)],
            )
        )
    assert seen == [], [e.text for e in seen]
    assert bot.open_attention_keys() == [crashloop_key(SVC)]


def test_an_absent_service_alerts() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    events = tick_at(bot, clock, probe, clock.t, [])
    assert keys(events) == [absent_key(SVC)]
    assert events[0].severity is Severity.ACTION
    assert SVC in events[0].text

    # Silent while it stays missing, resolved when it turns up.
    assert tick_at(bot, clock, probe, clock.t + 120.0, []) == []
    back = tick_at(bot, clock, probe, clock.t + 240.0, [sample()])
    assert [e.attention_key for e in resolutions(back)] == [absent_key(SVC)]


def test_an_absent_service_does_not_close_its_other_requests() -> None:
    """"I cannot see it" is not evidence that it recovered."""
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    at = _drive_into_crash_loop(bot, clock, probe)

    events = tick_at(bot, clock, probe, at + 60.0, [])
    assert keys(events) == [absent_key(SVC)]
    assert bot.open_attention_keys() == [absent_key(SVC), crashloop_key(SVC)]


def test_oom_kill_is_called_out_by_name() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    at = clock.t
    tick_at(bot, clock, probe, at, [sample(n_restarts=0)])
    events = tick_at(
        bot, clock, probe, at + 120.0,
        # Note the state: systemd has already restarted it, so it reads
        # active. The OOM kill is still the thing that happened.
        [sample(active_state="active", n_restarts=1, result="oom-kill",
                exit_code=None, active_enter_timestamp=at + 118.0)],
    )
    assert keys(events) == [oom_key(SVC)]
    event = events[0]
    assert event.severity is Severity.ACTION
    assert "out-of-memory" in event.text
    assert "oom-kill" in event.text
    assert SVC in event.text

    # It stays open until the unit has been seen running clean.
    assert tick_at(
        bot, clock, probe, at + 240.0,
        [sample(active_state="active", n_restarts=1, result="oom-kill")],
    ) == []
    back = tick_at(bot, clock, probe, at + 360.0, [sample(n_restarts=1)])
    assert [e.attention_key for e in resolutions(back)] == [oom_key(SVC)]


def test_memory_growth_over_the_ceiling_is_a_notice_not_an_action() -> None:
    clock, probe = FakeClock(), StubProbe()
    ceiling = 1024 * 1024 * 1024
    bot = make_bot(clock, probe, memory_ceiling_bytes=ceiling)

    at = clock.t
    assert tick_at(bot, clock, probe, at, [sample(memory_bytes=ceiling)]) == []
    events = tick_at(
        bot, clock, probe, at + 120.0, [sample(memory_bytes=ceiling * 2)]
    )
    assert keys(events) == [memory_key(SVC)]
    assert events[0].severity is Severity.NOTICE
    assert not events[0].wants_attention
    assert "2.0 GiB" in events[0].text

    back = tick_at(bot, clock, probe, at + 240.0, [sample(memory_bytes=ceiling // 2)])
    assert [e.attention_key for e in resolutions(back)] == [memory_key(SVC)]


def test_no_memory_ceiling_means_no_memory_notices() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    at = clock.t
    tick_at(bot, clock, probe, at, [sample(memory_bytes=64 * 1024 ** 3)])
    assert tick_at(
        bot, clock, probe, at + 120.0, [sample(memory_bytes=128 * 1024 ** 3)]
    ) == []


# --------------------------------------------------------------------------
# a probe that raises
# --------------------------------------------------------------------------


def test_a_raising_probe_is_an_error_event_and_escalates_after_three() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    probe.error = RuntimeError("systemctl: /usr/bin/systemctl: no such file")

    first = list(bot.tick(clock.t))
    assert len(first) == 1
    assert first[0].severity is Severity.ERROR
    # An ERROR with no key: it is a line in the feed, not a standing
    # request for a decision.
    assert first[0].attention_key is None
    # The type name only -- an exception's message may quote a path or a
    # command line, and an event is rendered on a page.
    assert "RuntimeError" in first[0].text
    assert "/usr/bin/systemctl" not in first[0].text

    clock.t += 120.0
    second = list(bot.tick(clock.t))
    assert [e.severity for e in second] == [Severity.ERROR]
    assert second[0].attention_key is None

    clock.t += 120.0
    third = list(bot.tick(clock.t))
    assert len(third) == 1
    assert third[0].severity is Severity.ACTION
    assert third[0].attention_key == PROBE_ATTENTION_KEY
    assert third[0].wants_attention
    assert third[0].data["failures"] == PROBE_FAILURES_BEFORE_ACTION

    # Already open: no second copy of the same complaint.
    clock.t += 120.0
    assert list(bot.tick(clock.t)) == []

    # And a probe that reads again closes it.
    probe.error = None
    clock.t += 120.0
    probe.samples = [sample()]
    recovered = list(bot.tick(clock.t))
    assert [e.attention_key for e in resolutions(recovered)] == [PROBE_ATTENTION_KEY]
    assert bot.open_attention_keys() == []


def test_a_probe_returning_junk_is_an_event_not_an_exception() -> None:
    clock = FakeClock()

    def bad_probe() -> Any:
        return ["not-a-sample"]

    bot = ServiceBot([SVC], clock=clock, probe=bad_probe)
    events = list(bot.tick(clock.t))
    assert [e.severity for e in events] == [Severity.ERROR]
    assert "TypeError" in events[0].text
    # Crucially: the watched service was not reported absent on the way.
    assert bot.open_attention_keys() == []


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_snapshot_restore_preserves_the_rolling_window_across_a_restart() -> None:
    """The window survives the process, because restarting the watcher is
    exactly when a loop is likely to be in progress."""
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    base = clock.t
    tick_at(bot, clock, probe, base, [sample(n_restarts=0)])
    for i, restarts in enumerate((1, 2, 3), start=1):
        assert tick_at(
            bot, clock, probe, base + i * 60.0, [sample(n_restarts=restarts)]
        ) == []

    # Through JSON, as the supervisor's store would carry it.
    state = json.loads(json.dumps(bot.snapshot()))
    assert state["version"] == SNAPSHOT_VERSION

    clock2, probe2 = FakeClock(base + 240.0), StubProbe()
    restarted = make_bot(clock2, probe2)
    restarted.restore(state)

    events = tick_at(
        restarted, clock2, probe2, base + 240.0,
        [sample(n_restarts=4, active_enter_timestamp=base + 239.0,
                result="exit-code", exit_code=1)],
    )
    assert keys(events) == [crashloop_key(SVC)]
    assert events[0].data["restarts"] == 4

    # The control: a bot that had *not* restored would see its first
    # sample as a baseline and count nothing.
    clock3, probe3 = FakeClock(base + 240.0), StubProbe()
    forgetful = make_bot(clock3, probe3)
    assert tick_at(
        forgetful, clock3, probe3, base + 240.0, [sample(n_restarts=4)]
    ) == []


def test_snapshot_carries_open_keys_and_last_states() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    at = _drive_into_crash_loop(bot, clock, probe)

    state = json.loads(json.dumps(bot.snapshot()))
    assert [row["key"] for row in state["open"]] == [crashloop_key(SVC)]
    assert state["last_seen"][SVC]["n_restarts"] == 4

    clock2, probe2 = FakeClock(at), StubProbe()
    restored = make_bot(clock2, probe2)
    restored.restore(state)
    assert restored.open_attention_keys() == [crashloop_key(SVC)]

    # A restart does not re-raise what is already in the badge...
    assert tick_at(
        restored, clock2, probe2, at + 60.0,
        [sample(n_restarts=4, active_enter_timestamp=at, result="exit-code")],
    ) == []
    # ...and can still close it, which is the whole reason the keys are
    # persisted at all.
    later = at + 3600.0
    closing = tick_at(
        restored, clock2, probe2, later,
        [sample(n_restarts=4, active_enter_timestamp=at)],
    )
    assert [e.attention_key for e in resolutions(closing)] == [crashloop_key(SVC)]


def test_restore_is_tolerant_of_rubbish_and_refuses_the_future() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)

    for junk in ({}, None, [], {"restarts": "nonsense", "open": "nope"}):
        bot.restore(junk)  # type: ignore[arg-type]
    assert bot.open_attention_keys() == []

    # State for a service nobody watches any more is dropped: an open key
    # that is never evaluated again can never be resolved.
    bot.restore(
        {
            "version": SNAPSHOT_VERSION,
            "restarts": {"gone.service": [[1.0, 9.0]]},
            "last_seen": {"gone.service": {"active_state": "active"}},
            "open": [{"key": "svc:crashloop:gone.service", "since": 1.0}],
        }
    )
    assert bot.open_attention_keys() == []

    with pytest.raises(ServiceBotError):
        bot.restore({"version": SNAPSHOT_VERSION + 1})


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------


def test_status_counts_healthy_services_and_names_the_worst_offender() -> None:
    clock, probe = FakeClock(), StubProbe()
    names = [SVC, "ollama.service", "nginx.service"]
    bot = make_bot(clock, probe, services=names)

    base = clock.t
    healthy_two = [
        sample("ollama.service", active_enter_timestamp=base - 7200.0),
        sample("nginx.service", active_enter_timestamp=base - 300.0),
    ]
    tick_at(bot, clock, probe, base, [sample(n_restarts=0)] + healthy_two)
    for i, restarts in enumerate((1, 2, 3, 4), start=1):
        at = base + i * 60.0
        tick_at(
            bot, clock, probe, at,
            [sample(n_restarts=restarts, active_enter_timestamp=at - 3.0)] + healthy_two,
        )

    status = bot.status()
    assert status.state is BotState.RUNNING
    values = {stat.label: stat.value for stat in status.stats}
    assert values["Healthy"] == "2 of 3"
    assert SVC in values["Most restarts"] and "x4" in values["Most restarts"]
    assert "hour" in values["Longest uptime"]
    assert "ollama.service" in values["Longest uptime"]
    assert "waiting on you" in status.detail


def test_status_is_honest_before_the_first_tick() -> None:
    clock, probe = FakeClock(), StubProbe()
    bot = make_bot(clock, probe)
    status = bot.status()
    assert status.state is BotState.RUNNING
    assert "not read yet" in status.detail
    assert {stat.label for stat in status.stats} == {
        "Healthy",
        "Most restarts",
        "Longest uptime",
    }


def test_wiring_mistakes_fail_at_construction() -> None:
    clock = FakeClock()
    with pytest.raises(ServiceBotError):
        ServiceBot([], clock=clock, probe=StubProbe())
    with pytest.raises(ServiceBotError):
        ServiceBot([SVC], clock=clock, probe="systemctl")  # type: ignore[arg-type]
    with pytest.raises(ServiceBotError):
        ServiceBot([SVC], clock=clock, probe=StubProbe(), window_s=0)
    with pytest.raises(ServiceBotError):
        ServiceBot([SVC], clock=clock, probe=StubProbe(), restart_threshold=0)
    with pytest.raises(ValueError):
        ServiceBot([SVC], clock="not a clock", probe=StubProbe())  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the systemctl probe: argv, parser, and a unit that is not there
# --------------------------------------------------------------------------


HEALTHY_SHOW = """\
Id=ollama.service
LoadState=loaded
ActiveState=active
SubState=running
NRestarts=0
ActiveEnterTimestamp=@1758330000
MainPID=1234
MemoryCurrent=734003200
ExecMainStatus=0
Result=success
"""

CRASHLOOPING_SHOW = """\
Id=comfyui.service
LoadState=loaded
ActiveState=active
SubState=running
NRestarts=15021
ActiveEnterTimestamp=@1758339997
MainPID=90210
MemoryCurrent=1073741824
ExecMainStatus=1
Result=exit-code
"""

# What systemd says about a unit it has never heard of: the properties are
# all there, they are just empty, and LoadState gives it away.
MISSING_SHOW = """\
Id=nope.service
LoadState=not-found
ActiveState=inactive
SubState=dead
NRestarts=0
ActiveEnterTimestamp=
MainPID=0
MemoryCurrent=[not set]
ExecMainStatus=
Result=success
"""

DEAD_SHOW = """\
Id=backup.service
LoadState=loaded
ActiveState=inactive
SubState=dead
NRestarts=0
ActiveEnterTimestamp=
MainPID=0
MemoryCurrent=[not set]
ExecMainStatus=0
Result=success
"""

OOM_SHOW = """\
Id=hungry.service
LoadState=loaded
ActiveState=failed
SubState=failed
NRestarts=3
ActiveEnterTimestamp=@1758330500
MainPID=0
MemoryCurrent=18446744073709551615
ExecMainStatus=9
Result=oom-kill
"""

# An older systemd that does not know --timestamp=unix prints a date.
PRETTY_TIMESTAMP_SHOW = """\
Id=old.service
LoadState=loaded
ActiveState=active
SubState=running
NRestarts=2
ActiveEnterTimestamp=Sat 2026-09-19 12:00:00 UTC
MainPID=77
MemoryCurrent=1024
ExecMainStatus=0
Result=success
"""


def test_systemctl_command_asks_for_the_properties_the_parser_reads() -> None:
    argv = systemctl_command("comfyui.service")
    assert argv[:3] == ["systemctl", "show", "comfyui.service"]
    assert "--timestamp=unix" in argv
    prop = [a for a in argv if a.startswith("--property=")]
    assert len(prop) == 1
    asked = prop[0].split("=", 1)[1].split(",")
    assert asked == list(SHOW_PROPERTIES)
    for needed in ("ActiveState", "NRestarts", "Result", "LoadState"):
        assert needed in asked
    # Built the same way twice: no set iteration in the argv.
    assert systemctl_command("comfyui.service") == argv
    with pytest.raises(ValueError):
        systemctl_command("")


def test_parse_systemctl_show_reads_a_healthy_unit() -> None:
    parsed = parse_systemctl_show("ollama.service", HEALTHY_SHOW)
    assert parsed == ServiceSample(
        name="ollama.service",
        active_state="active",
        sub_state="running",
        n_restarts=0,
        active_enter_timestamp=1758330000.0,
        main_pid=1234,
        memory_bytes=734003200,
        exit_code=0,
        result="success",
    )
    assert parsed.is_up
    assert parsed.uptime_s(1758330060.0) == 60.0


def test_parse_systemctl_show_reads_the_incident() -> None:
    parsed = parse_systemctl_show("comfyui.service", CRASHLOOPING_SHOW)
    assert parsed is not None
    # active and running, with fifteen thousand restarts behind it.
    assert parsed.active_state == "active"
    assert parsed.is_up
    assert parsed.n_restarts == 15021
    assert parsed.result == "exit-code"
    assert parsed.exit_code == 1
    assert parsed.uptime_s(1758340000.0) == 3.0


def test_parse_systemctl_show_returns_none_for_a_unit_that_does_not_exist() -> None:
    """A unit systemd has never heard of is *absent*, not inactive.

    Reading it as inactive is the bug that matters here: the bot would
    report a typo'd unit name as a stopped service and the owner would go
    looking for a service that was never there.
    """
    assert parse_systemctl_show("nope.service", MISSING_SHOW) is None
    assert parse_systemctl_show("nothing.service", "") is None
    assert parse_systemctl_show("noise.service", "garbage without an equals") is None

    # ...whereas a real unit that is simply not running parses fine.
    dead = parse_systemctl_show("backup.service", DEAD_SHOW)
    assert dead is not None
    assert dead.active_state == "inactive"
    assert dead.main_pid is None  # MainPID=0 is "no process", not pid zero
    assert dead.memory_bytes is None  # [not set] is not zero bytes
    assert dead.active_enter_timestamp is None


def test_parse_systemctl_show_reads_an_oom_kill_and_the_unset_sentinel() -> None:
    parsed = parse_systemctl_show("hungry.service", OOM_SHOW)
    assert parsed is not None
    assert parsed.result == "oom-kill"
    assert parsed.exit_code == 9
    assert parsed.active_state == "failed"
    # 18446744073709551615 is systemd's "no accounting", not 16 exabytes.
    assert parsed.memory_bytes is None


def test_parse_systemctl_show_refuses_to_guess_at_a_localised_date() -> None:
    parsed = parse_systemctl_show("old.service", PRETTY_TIMESTAMP_SHOW)
    assert parsed is not None
    assert parsed.active_enter_timestamp is None
    assert parsed.uptime_s(1758340000.0) is None
    # Everything else still reads, so an old systemd is degraded, not broken.
    assert parsed.n_restarts == 2


def test_parse_systemctl_show_skips_unknown_properties() -> None:
    text = HEALTHY_SHOW + "SomeNewSystemdProperty=whatever\n\nTrailing junk\n"
    parsed = parse_systemctl_show("ollama.service", text)
    assert parsed is not None and parsed.active_state == "active"


def test_systemctl_probe_drops_the_units_that_are_not_there() -> None:
    """The probe end to end, with the one call that shells out replaced."""
    captured: List[List[str]] = []
    outputs = {
        "ollama.service": HEALTHY_SHOW,
        "comfyui.service": CRASHLOOPING_SHOW,
        "nope.service": MISSING_SHOW,
    }

    def fake_run(argv: Sequence[str], timeout_s: float) -> str:
        captured.append(list(argv))
        assert timeout_s > 0
        return outputs[argv[2]]

    names = ["ollama.service", "nope.service", "comfyui.service"]
    samples = systemctl_probe(names, run=fake_run)

    assert [s.name for s in samples] == ["ollama.service", "comfyui.service"]
    # One invocation per unit, so an absent unit is an unambiguous answer.
    assert [argv[2] for argv in captured] == names
    assert all(argv[0] == "systemctl" and argv[1] == "show" for argv in captured)


def test_the_bot_never_shells_out_itself() -> None:
    """``subprocess`` appears once in this module, in the runner the probe
    uses.  The bot is handed a callable and cannot reach a process without
    one -- which is what makes every test above offline."""
    source = (ROOT / "jarvis_bots" / "bots" / "service_bot.py").read_text()
    assert source.count("subprocess.run") == 1
    class_body = source.split("class ServiceBot(BaseBot):", 1)[1].split(
        "\n# ---", 1
    )[0]
    for forbidden in ("subprocess", "os.system", "time.time("):
        assert forbidden not in class_body


# --------------------------------------------------------------------------
# under a real supervisor
# --------------------------------------------------------------------------


def test_registers_and_ticks_under_a_real_supervisor() -> None:
    clock, probe = FakeClock(), StubProbe([sample(n_restarts=0)])
    bot = build([SVC], clock=clock, probe=probe)
    assert check_bot(bot) is INFO

    registry = BotRegistry([bot])
    supervisor = Supervisor(registry, clock)
    assert registry.ids() == [BOT_ID]

    base = clock.t
    report = supervisor.run_round(base)
    assert report.ticked == 1 and report.failed == 0
    assert supervisor.attention_count() == 0

    for i, restarts in enumerate((1, 2, 3, 4), start=1):
        clock.t = base + i * INFO.interval_s
        probe.samples = [
            sample(n_restarts=restarts, active_enter_timestamp=clock.t - 2.0,
                   result="exit-code", exit_code=1)
        ]
        report = supervisor.run_round(clock.t)
        assert report.failed == 0

    assert supervisor.attention_count() == 1
    item = supervisor.attention_items(BOT_ID)[0]
    assert item.key == crashloop_key(SVC)
    assert item.href == INFO.href
    assert supervisor.badge_status()["attention"] == 1
    assert supervisor.bots_wanting_attention() == [BOT_ID]

    card = supervisor.launcher_state()["bots"][0]
    assert card["id"] == BOT_ID
    assert card["kind"] == "radar"
    assert card["state"] == BotState.RUNNING.ui
    assert card["attention"] == 1

    # Recovery empties the badge with no app-side wiring: the resolution
    # event carries the key, sits below ACTION and sets resolved=True,
    # which is what Supervisor.RESOLVED_FLAG reads.
    clock.t = base + 7200.0
    probe.samples = [sample(n_restarts=4, active_enter_timestamp=base)]
    supervisor.run_round(clock.t)
    assert supervisor.attention_count() == 0
    assert supervisor.badge_status()["attention"] == 0


def test_state_survives_a_supervisor_save_and_load() -> None:
    clock, probe = FakeClock(), StubProbe([sample(n_restarts=0)])
    bot = build([SVC], clock=clock, probe=probe)
    supervisor = Supervisor(BotRegistry([bot]), clock)

    base = clock.t
    supervisor.run_round(base)
    for i, restarts in enumerate((1, 2, 3), start=1):
        clock.t = base + i * INFO.interval_s
        probe.samples = [sample(n_restarts=restarts)]
        supervisor.run_round(clock.t)
    assert supervisor.attention_count() == 0

    state = json.loads(json.dumps(supervisor.save_state()))

    clock2, probe2 = FakeClock(clock.t), StubProbe()
    bot2 = build([SVC], clock=clock2, probe=probe2)
    supervisor2 = Supervisor(BotRegistry([bot2]), clock2)
    supervisor2.load_state(state)

    clock2.t = clock.t + INFO.interval_s
    probe2.samples = [
        sample(n_restarts=4, active_enter_timestamp=clock2.t - 1.0,
               result="exit-code", exit_code=1)
    ]
    supervisor2.run_round(clock2.t)
    # The fourth restart lands on a window the new process remembers.
    assert supervisor2.attention_count() == 1
    assert supervisor2.attention_items(BOT_ID)[0].key == crashloop_key(SVC)


# --------------------------------------------------------------------------
# shared drivers
# --------------------------------------------------------------------------


def _drive_into_crash_loop(
    bot: ServiceBot,
    clock: FakeClock,
    probe: StubProbe,
    states: Sequence[str] = ("active",),
) -> float:
    """Four restarts in four minutes; returns the time of the last tick."""
    base = clock.t
    tick_at(bot, clock, probe, base, [sample(n_restarts=0)])
    at = base
    for i, restarts in enumerate((1, 2, 3, 4), start=1):
        at = base + i * 60.0
        state = states[(i - 1) % len(states)]
        events = tick_at(
            bot, clock, probe, at,
            [sample(active_state=state, sub_state=state, n_restarts=restarts,
                    active_enter_timestamp=at - 3.0, result="exit-code",
                    exit_code=1)],
        )
    assert keys(events) == [crashloop_key(SVC)], [e.text for e in events]
    return at
