"""Tests for the GPU watch.

The bot exists because a card vanished after a reboot and nobody noticed
for days while a service crash-looped against it, so the tests are
weighted the way the bot is: the missing-card path is tested for the
thing that actually goes wrong with watchers -- raising once, not once a
tick, and clearing itself when the card comes back -- and every other
check is tested at its exact boundary, because a threshold that is one
degree out is a threshold that either never fires or never stops.

Two of them are about *silence*, which is the harder half:

* the owner's CMP 170HX is a Gen 2 x4 card with Gen 3 fused off.  It is
  ticked here twenty times and must never say a word about its link,
  because a bot that cries about correct hardware every five minutes is
  a bot that gets muted, and a muted bot cannot tell you a card is gone.
* a probe that raises is an event, never an exception: four raises in a
  row would quarantine this bot for half an hour, and a quarantined GPU
  watch is exactly the blindness it was written to end.

Nothing here opens a socket, runs a process or reads a real clock: the
probe is a stub a test drives, the clock only moves when a test moves it,
and :func:`nvidia_smi_probe` is exercised against captured text through
its injected runner.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_bots.bots.gpu_bot import (
    BOT_ID,
    DEFAULT_TEMP_ACTION_C,
    DEFAULT_TEMP_NOTICE_C,
    INFO,
    MEMORY_PRESSURE_TICKS,
    NVIDIA_SMI_QUERY_FIELDS,
    PROBE_FAILURES_FOR_ACTION,
    PROBE_KEY,
    SNAPSHOT_VERSION,
    THROTTLE_TICKS,
    GpuBot,
    GpuBotError,
    GpuProbeError,
    GpuSample,
    build,
    decode_throttle_reasons,
    ecc_key,
    fan_key,
    missing_key,
    nvidia_smi_command,
    nvidia_smi_probe,
    parse_nvidia_smi_csv,
    temperature_key,
)
from jarvis_bots.contracts import BotState, Event, Severity
from jarvis_bots.registry import BotRegistry, check_bot
from jarvis_bots.supervisor import RESOLVED_FLAG, Supervisor

T0 = 1_700_000_000.0

# The owner's actual fleet, by uuid, so a test reads like the machine.
P40_A = "GPU-11111111-1111-1111-1111-111111111111"
P40_B = "GPU-22222222-2222-2222-2222-222222222222"
CMP = "GPU-33333333-3333-3333-3333-333333333333"
RTX = "GPU-44444444-4444-4444-4444-444444444444"


# --------------------------------------------------------------------------
# stubs
# --------------------------------------------------------------------------


class Clock:
    """contracts.py: "Time is injected everywhere.  Nothing here calls
    time.time()." """

    def __init__(self, at: float = T0) -> None:
        self.at = float(at)

    def __call__(self) -> float:
        return self.at

    def advance(self, seconds: float = 300.0) -> float:
        self.at += float(seconds)
        return self.at


class Probe:
    """A stub ``nvidia-smi``.  ``samples`` is what the next call returns;
    setting ``fault`` makes it raise instead, which is what a real probe
    does when it cannot look."""

    def __init__(self, samples: Sequence[GpuSample] = ()) -> None:
        self.samples: List[GpuSample] = list(samples)
        self.fault: Optional[BaseException] = None
        self.calls = 0

    def __call__(self) -> List[GpuSample]:
        self.calls += 1
        if self.fault is not None:
            raise self.fault
        return list(self.samples)


def p40(uuid: str = P40_A, index: int = 0, **over: Any) -> GpuSample:
    """A Tesla P40: passively cooled, so it reports no fan at all."""
    fields: Dict[str, Any] = dict(
        index=index,
        uuid=uuid,
        name="Tesla P40",
        memory_total_mb=24576.0,
        memory_used_mb=1024.0,
        temperature_c=55.0,
        fan_percent=None,
        power_w=52.0,
        power_limit_w=250.0,
        pcie_gen_current=3,
        pcie_gen_max=3,
        pcie_width_current=16,
        pcie_width_max=16,
        ecc_errors=0,
        utilization_pct=0.0,
        throttle_reasons=(),
    )
    fields.update(over)
    return GpuSample(**fields)


def cmp170(**over: Any) -> GpuSample:
    """The CMP 170HX: Gen 2 x4 for real, with Gen 3 fused off.  This is
    correct hardware and the bot must never complain about it."""
    fields: Dict[str, Any] = dict(
        index=2,
        uuid=CMP,
        name="NVIDIA CMP 170HX",
        memory_total_mb=8192.0,
        memory_used_mb=512.0,
        temperature_c=61.0,
        fan_percent=45.0,
        power_w=180.0,
        power_limit_w=250.0,
        pcie_gen_current=2,
        pcie_gen_max=3,
        pcie_width_current=4,
        pcie_width_max=16,
        ecc_errors=None,
        utilization_pct=99.0,
        throttle_reasons=(),
    )
    fields.update(over)
    return GpuSample(**fields)


def rtx(**over: Any) -> GpuSample:
    fields: Dict[str, Any] = dict(
        index=3,
        uuid=RTX,
        name="NVIDIA GeForce RTX 5070",
        memory_total_mb=12288.0,
        memory_used_mb=2048.0,
        temperature_c=48.0,
        fan_percent=30.0,
        power_w=90.0,
        power_limit_w=250.0,
        pcie_gen_current=5,
        pcie_gen_max=5,
        pcie_width_current=16,
        pcie_width_max=16,
        ecc_errors=None,
        utilization_pct=12.0,
        throttle_reasons=(),
    )
    fields.update(over)
    return GpuSample(**fields)


def fleet() -> List[GpuSample]:
    """All four cards, all healthy."""
    return [p40(P40_A, 0), p40(P40_B, 1), cmp170(), rtx()]


def make(samples: Sequence[GpuSample] = (), **kwargs: Any) -> Tuple[GpuBot, Probe, Clock]:
    clock = Clock()
    probe = Probe(samples)
    bot = build(clock=clock, probe=probe, **kwargs)
    return bot, probe, clock


def run(bot: GpuBot, clock: Clock, advance: float = 300.0) -> List[Event]:
    """One tick, on the round's own ``now``."""
    events = list(bot.tick(clock.at))
    clock.advance(advance)
    return events


def actions(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.severity is Severity.ACTION]


def notices(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.severity is Severity.NOTICE]


def keys(events: Sequence[Event]) -> List[str]:
    return [e.attention_key for e in events if e.attention_key]


# --------------------------------------------------------------------------
# identity
# --------------------------------------------------------------------------


def test_identity_is_what_the_launcher_expects() -> None:
    bot, _, _ = make(fleet())
    assert check_bot(bot) is INFO
    assert (INFO.id, INFO.name, INFO.kind) == ("gpu", "GPU watch", "radar")
    assert INFO.interval_s == 300.0
    assert INFO.href == "/bots/gpu"
    assert BOT_ID == "gpu"


def test_a_bot_without_a_probe_fails_at_construction_not_in_a_round() -> None:
    with pytest.raises(GpuBotError):
        build(clock=Clock(), probe="nvidia-smi")  # type: ignore[arg-type]
    with pytest.raises(GpuBotError):
        build(clock=Clock(), probe=Probe(), temp_action_c=70.0, temp_notice_c=80.0)


# --------------------------------------------------------------------------
# the fleet
# --------------------------------------------------------------------------


def test_first_tick_records_the_baseline_and_raises_nothing() -> None:
    bot, _, clock = make(fleet())
    assert run(bot, clock) == []
    assert set(bot.expected_fleet()) == {P40_A, P40_B, CMP, RTX}
    assert bot.expected_fleet()[P40_B] == {"name": "Tesla P40", "index": 1}
    assert bot.open_keys() == []
    assert bot.status().state is BotState.RUNNING


def test_a_card_that_disappears_is_one_action_and_stays_one() -> None:
    bot, probe, clock = make(fleet())
    run(bot, clock)

    # The incident: the second P40 is not in nvidia-smi's answer any more.
    probe.samples = [p40(P40_A, 0), cmp170(), rtx()]
    first = run(bot, clock)
    raised = actions(first)
    assert len(raised) == 1
    assert raised[0].attention_key == missing_key(P40_B)
    assert raised[0].wants_attention
    # Named by what it was, because it is not here to name itself.
    assert "Tesla P40 #1" in raised[0].text
    assert raised[0].data["uuid"] == P40_B
    assert raised[0].href == INFO.href

    # Still gone, three ticks later: one standing request, not four.
    for _ in range(3):
        assert actions(run(bot, clock)) == []
    assert bot.open_keys() == [missing_key(P40_B)]


def test_the_card_coming_back_resolves_the_request() -> None:
    bot, probe, clock = make(fleet())
    run(bot, clock)
    probe.samples = [p40(P40_A, 0), cmp170(), rtx()]
    run(bot, clock)

    probe.samples = fleet()
    back = run(bot, clock)
    assert len(back) == 1
    event = back[0]
    # The framework's own close signal: same key, below ACTION so it
    # cannot re-open, resolved=True in the data.
    assert event.attention_key == missing_key(P40_B)
    assert event.severity is Severity.NOTICE
    assert event.data[RESOLVED_FLAG] is True
    assert not event.wants_attention
    assert bot.open_keys() == []
    # And it is not re-resolved for ever after.
    assert run(bot, clock) == []


def test_a_new_card_is_a_notice_and_joins_the_fleet() -> None:
    bot, probe, clock = make([p40(P40_A, 0), p40(P40_B, 1), cmp170()])
    run(bot, clock)

    probe.samples = fleet()
    events = run(bot, clock)
    assert actions(events) == []
    assert len(notices(events)) == 1
    assert "RTX 5070" in notices(events)[0].text
    assert notices(events)[0].attention_key is None
    assert RTX in bot.expected_fleet()

    # Joined means joined: it is quiet next tick, and missed if it leaves.
    assert run(bot, clock) == []
    probe.samples = [p40(P40_A, 0), p40(P40_B, 1), cmp170()]
    assert keys(actions(run(bot, clock))) == [missing_key(RTX)]


# --------------------------------------------------------------------------
# heat
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "temperature, severity",
    [
        (79.9, None),
        (DEFAULT_TEMP_NOTICE_C, Severity.NOTICE),   # exactly 80: a notice
        (84.9, Severity.NOTICE),
        (DEFAULT_TEMP_ACTION_C, Severity.ACTION),   # exactly 85: an action
        (94.0, Severity.ACTION),
    ],
)
def test_temperature_thresholds_at_their_exact_boundaries(
    temperature: float, severity: Optional[Severity]
) -> None:
    bot, probe, clock = make([p40()])
    run(bot, clock)  # baseline at 55C

    probe.samples = [p40(temperature_c=temperature)]
    events = run(bot, clock)
    if severity is None:
        assert events == []
        return
    assert len(events) == 1
    assert events[0].severity is severity
    if severity is Severity.ACTION:
        assert events[0].attention_key == temperature_key(P40_A)
    else:
        assert events[0].attention_key is None


def test_a_hot_card_asks_once_and_the_request_closes_when_it_cools() -> None:
    bot, probe, clock = make([p40()])
    run(bot, clock)

    probe.samples = [p40(temperature_c=91.0)]
    assert len(actions(run(bot, clock))) == 1
    probe.samples = [p40(temperature_c=93.0)]
    assert run(bot, clock) == []  # hotter, but the same open question
    assert bot.open_keys() == [temperature_key(P40_A)]

    probe.samples = [p40(temperature_c=57.0)]
    cooled = run(bot, clock)
    assert len(cooled) == 1
    assert cooled[0].data[RESOLVED_FLAG] is True
    assert cooled[0].attention_key == temperature_key(P40_A)
    assert bot.open_keys() == []


def test_the_cards_own_slowdown_temperature_wins_when_the_probe_supplies_it() -> None:
    """A card that slows itself at 92C is not in trouble at 85."""
    bot, probe, clock = make([p40(temperature_slowdown_c=92.0)])
    run(bot, clock)

    probe.samples = [p40(temperature_c=86.0, temperature_slowdown_c=92.0)]
    assert actions(run(bot, clock)) == []  # under its own limit: a notice band

    probe.samples = [p40(temperature_c=92.0, temperature_slowdown_c=92.0)]
    raised = actions(run(bot, clock))
    assert len(raised) == 1
    assert raised[0].data["limit_c"] == 92.0


def test_a_card_that_reports_no_temperature_is_not_a_cold_card() -> None:
    bot, probe, clock = make([p40()])
    run(bot, clock)
    probe.samples = [p40(temperature_c=None)]
    assert run(bot, clock) == []


# --------------------------------------------------------------------------
# the fan
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fan, util, temp, stuck",
    [
        (0.0, 80.0, 75.0, True),     # the fault: working, hot, fan stopped
        (0.0, 30.0, 75.0, False),    # exactly 30% load: not "above 30"
        (0.0, 30.1, 75.0, True),
        (0.0, 80.0, 60.0, False),    # exactly 60C: not "above 60"
        (0.0, 80.0, 60.1, True),
        (1.0, 80.0, 75.0, False),    # a fan that is turning is not stuck
        (None, 99.0, 82.0, False),   # a passive P40 has no fan to report
    ],
)
def test_stuck_fan_only_under_the_stated_conditions(
    fan: Optional[float], util: float, temp: float, stuck: bool
) -> None:
    bot, probe, clock = make([p40(fan_percent=fan, utilization_pct=0.0)])
    run(bot, clock)

    probe.samples = [p40(fan_percent=fan, utilization_pct=util, temperature_c=temp)]
    raised = [e for e in actions(run(bot, clock)) if e.attention_key == fan_key(P40_A)]
    assert bool(raised) is stuck


def test_a_stuck_fan_asks_once_and_closes_when_it_spins() -> None:
    bot, probe, clock = make([p40()])
    run(bot, clock)
    hot = dict(fan_percent=0.0, utilization_pct=99.0, temperature_c=70.0)
    probe.samples = [p40(**hot)]
    assert len(actions(run(bot, clock))) == 1
    assert run(bot, clock) == []
    probe.samples = [p40(fan_percent=60.0, utilization_pct=99.0, temperature_c=70.0)]
    closed = run(bot, clock)
    assert closed[0].attention_key == fan_key(P40_A)
    assert closed[0].data[RESOLVED_FLAG] is True


# --------------------------------------------------------------------------
# ECC
# --------------------------------------------------------------------------


def test_ecc_errors_increasing_is_an_action_and_steady_is_silent() -> None:
    bot, probe, clock = make([p40(ecc_errors=2)])
    run(bot, clock)

    probe.samples = [p40(ecc_errors=2)]
    assert run(bot, clock) == []  # two errors it already knew about

    probe.samples = [p40(ecc_errors=5)]
    raised = actions(run(bot, clock))
    assert len(raised) == 1
    assert raised[0].attention_key == ecc_key(P40_A)
    assert raised[0].data == {"uuid": P40_A, "ecc_errors": 5, "ecc_gained": 3}

    probe.samples = [p40(ecc_errors=5)]
    assert run(bot, clock) == []  # steady again: nothing new to say


def test_a_card_with_no_ecc_counters_never_raises_one() -> None:
    bot, probe, clock = make([rtx()])  # a consumer card: ecc_errors is None
    for _ in range(5):
        run(bot, clock)
    assert bot.open_keys() == []


# --------------------------------------------------------------------------
# memory
# --------------------------------------------------------------------------


def test_memory_pressure_needs_three_consecutive_ticks_and_is_only_a_notice() -> None:
    full = p40(memory_used_mb=24000.0)  # 97.7% of 24576
    bot, probe, clock = make([p40()])
    run(bot, clock)

    probe.samples = [full]
    assert run(bot, clock) == []       # tick 1 over
    assert run(bot, clock) == []       # tick 2 over
    third = run(bot, clock)            # tick 3 over: one notice, no badge
    assert len(third) == 1
    assert third[0].severity is Severity.NOTICE
    assert third[0].attention_key is None
    assert MEMORY_PRESSURE_TICKS == 3

    assert run(bot, clock) == []       # and not again on the fourth


def test_memory_falling_back_resets_the_count() -> None:
    bot, probe, clock = make([p40()])
    run(bot, clock)
    probe.samples = [p40(memory_used_mb=24000.0)]
    run(bot, clock)
    run(bot, clock)
    probe.samples = [p40(memory_used_mb=1024.0)]
    run(bot, clock)
    probe.samples = [p40(memory_used_mb=24000.0)]
    assert run(bot, clock) == []
    assert run(bot, clock) == []
    assert len(run(bot, clock)) == 1


def test_exactly_95_percent_is_not_above_95_percent() -> None:
    bot, probe, clock = make([p40()])
    run(bot, clock)
    probe.samples = [p40(memory_used_mb=24576.0 * 0.95)]
    for _ in range(4):
        assert run(bot, clock) == []


# --------------------------------------------------------------------------
# PCIe: the check that must stay silent
# --------------------------------------------------------------------------


def test_the_cmp_170hx_never_alerts_about_its_link() -> None:
    """Gen 2 x4 with a Gen 3 x16 maximum is what this card *is*.

    The whole fleet runs for twenty ticks with the 170HX reporting the
    link it will report for the rest of its life.  A single word about it
    here would be a word every five minutes for ever on the real machine.
    """
    bot, probe, clock = make(fleet())
    said: List[Event] = []
    for _ in range(20):
        said.extend(run(bot, clock))
    assert said == []
    assert [e for e in said if e.data.get("uuid") == CMP] == []

    # Even after a restart from disk, because the best link is persisted.
    fresh = build(clock=clock, probe=probe)
    fresh.restore(json.loads(json.dumps(bot.snapshot())))
    assert run(fresh, clock) == []


def test_a_genuine_degradation_from_a_better_link_does_alert() -> None:
    bot, probe, clock = make([rtx()])  # Gen 5 x16
    run(bot, clock)

    probe.samples = [rtx(pcie_gen_current=1, pcie_width_current=4)]
    events = run(bot, clock)
    assert len(events) == 1
    assert events[0].severity is Severity.NOTICE
    assert "Gen 1 x4" in events[0].text and "Gen 5 x16" in events[0].text
    assert events[0].data["best_gen"] == 5

    # A bad day does not lower the bar: recovering and dropping again is
    # measured against the best link, not against yesterday's.
    probe.samples = [rtx()]
    run(bot, clock)
    probe.samples = [rtx(pcie_gen_current=4, pcie_width_current=16)]
    assert len(run(bot, clock)) == 1


def test_a_width_drop_at_the_same_generation_is_a_degradation() -> None:
    bot, probe, clock = make([p40()])  # Gen 3 x16
    run(bot, clock)
    probe.samples = [p40(pcie_width_current=8)]
    assert len(run(bot, clock)) == 1


# --------------------------------------------------------------------------
# throttling
# --------------------------------------------------------------------------


def test_a_thermal_or_power_throttle_needs_two_consecutive_ticks() -> None:
    bot, probe, clock = make([cmp170()])
    run(bot, clock)

    probe.samples = [cmp170(throttle_reasons=("sw_power_cap",))]
    assert run(bot, clock) == []
    second = run(bot, clock)
    assert len(second) == 1
    assert second[0].severity is Severity.NOTICE
    assert "sw_power_cap" in second[0].text
    assert THROTTLE_TICKS == 2
    assert run(bot, clock) == []  # said once, not every tick after


def test_an_idle_or_ambiguous_throttle_reason_is_not_worth_a_line() -> None:
    bot, probe, clock = make([cmp170()])
    run(bot, clock)
    probe.samples = [cmp170(throttle_reasons=("gpu_idle", "hw_slowdown"))]
    for _ in range(4):
        assert run(bot, clock) == []


# --------------------------------------------------------------------------
# the probe itself
# --------------------------------------------------------------------------


def test_a_raising_probe_is_an_event_and_three_in_a_row_escalate() -> None:
    bot, probe, clock = make(fleet())
    run(bot, clock)

    probe.fault = OSError("nvidia-smi: couldn't communicate with the driver")
    first = run(bot, clock)          # not an exception: the round survives
    assert len(first) == 1
    assert first[0].severity is Severity.ERROR
    assert first[0].attention_key is None
    # Only the exception's type name is quoted, never its message.
    assert "OSError" in first[0].text
    assert "couldn't communicate" not in first[0].text

    second = run(bot, clock)
    assert second[0].severity is Severity.ERROR
    assert second[0].attention_key is None

    third = run(bot, clock)
    assert len(third) == 1
    assert third[0].severity is Severity.ACTION
    assert third[0].attention_key == PROBE_KEY
    assert third[0].wants_attention
    assert PROBE_FAILURES_FOR_ACTION == 3

    assert run(bot, clock) == []  # the question is open; it is not re-asked

    probe.fault = None
    back = run(bot, clock)
    assert back[0].attention_key == PROBE_KEY
    assert back[0].data[RESOLVED_FLAG] is True
    assert bot.open_keys() == []


def test_a_failed_probe_never_empties_the_fleet() -> None:
    """The failure that would recreate the original incident in reverse:
    reading "I could not look" as "there are no GPUs" would report all
    four cards missing at once and then forget them."""
    bot, probe, clock = make(fleet())
    run(bot, clock)
    probe.fault = RuntimeError("driver reloading")
    run(bot, clock)
    probe.fault = None
    assert run(bot, clock) == []
    assert set(bot.expected_fleet()) == {P40_A, P40_B, CMP, RTX}


def test_a_probe_that_returns_nonsense_is_a_probe_failure_not_an_empty_fleet() -> None:
    bot, probe, clock = make(fleet())
    run(bot, clock)
    probe.samples = ["Tesla P40"]  # type: ignore[list-item]
    events = run(bot, clock)
    assert events[0].severity is Severity.ERROR
    assert set(bot.expected_fleet()) == {P40_A, P40_B, CMP, RTX}


def test_a_bug_in_the_bot_is_an_event_too_not_a_quarantine() -> None:
    """Belt and braces: the probe is not the only thing that can throw,
    and the one bot that must never stop running is the one that reports
    a card has gone."""

    class Broken(GpuBot):
        def _temperature(self, sample: GpuSample, now: float) -> List[Event]:
            raise ZeroDivisionError("a threshold that divided by nothing")

    clock = Clock()
    bot = Broken(clock=clock, probe=Probe(fleet()))
    events = bot.tick(clock.at)
    assert len(events) == 1
    assert events[0].severity is Severity.ERROR
    assert "ZeroDivisionError" in events[0].text
    assert "divided by nothing" not in events[0].text
    assert bot.status().state is BotState.RUNNING


# --------------------------------------------------------------------------
# the card the launcher renders
# --------------------------------------------------------------------------


def test_status_is_idle_before_the_first_look_and_running_after() -> None:
    bot, _, clock = make(fleet())
    assert bot.status().state is BotState.IDLE
    run(bot, clock)
    status = bot.status()
    assert status.state is BotState.RUNNING
    labels = {stat.label: stat.value for stat in status.stats}
    assert labels["Cards"] == "4 of 4 present"
    assert labels["Hottest"] == "61C NVIDIA CMP 170HX #2"
    assert labels["Memory in use"].endswith(f"of {(24576*2+8192+12288)/1024:.1f} GiB")
    assert all(isinstance(stat.value, str) for stat in status.stats)


def test_a_missing_card_keeps_the_bot_running_and_says_so_in_the_stats() -> None:
    """The bot is fine; the hardware is not.  Quarantining the messenger
    in the UI would hide the message."""
    bot, probe, clock = make(fleet())
    run(bot, clock)
    probe.samples = [p40(P40_A, 0), cmp170(), rtx()]
    run(bot, clock)

    status = bot.status()
    assert status.state is BotState.RUNNING
    labels = {stat.label: stat.value for stat in status.stats}
    assert labels["Cards"] == "3 of 4 present -- 1 MISSING card"
    assert labels["Missing"] == "Tesla P40 #1"
    assert "not on the bus" in status.detail


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_snapshot_restore_round_trips_through_json() -> None:
    bot, probe, clock = make(fleet())
    run(bot, clock)
    # Give it something in every field that matters: an open request, a
    # best link, ECC counts, and two counters part way to firing.
    probe.samples = [
        p40(P40_A, 0, temperature_c=90.0, ecc_errors=7, memory_used_mb=24000.0),
        cmp170(throttle_reasons=("hw_thermal_slowdown",)),
        rtx(),
    ]
    run(bot, clock)
    before = bot.snapshot()
    assert before["version"] == SNAPSHOT_VERSION

    restored = build(clock=clock, probe=probe)
    restored.restore(json.loads(json.dumps(before)))
    assert restored.snapshot() == before

    # That tick opened three requests -- the P40 that vanished, the heat
    # on the other one, and its new ECC errors -- and all three come back.
    open_now = sorted([ecc_key(P40_A), missing_key(P40_B), temperature_key(P40_A)])
    assert restored.open_keys() == bot.open_keys() == open_now
    assert set(restored.expected_fleet()) == {P40_A, P40_B, CMP, RTX}

    # And the restored bot behaves like the one it came from: the open
    # heat request closes on the next cool reading, the ECC count is still
    # 7 so 7 again is silent, the missing card is not re-asked about, and
    # the throttle counter was one tick from speaking.
    probe.samples = [
        p40(P40_A, 0, temperature_c=50.0, ecc_errors=7, memory_used_mb=24000.0),
        cmp170(throttle_reasons=("hw_thermal_slowdown",)),
        rtx(),
    ]
    events = run(restored, clock)
    texts = " | ".join(e.text for e in events)
    assert "cooled" in texts
    assert "throttling" in texts
    assert actions(events) == []
    assert restored.open_keys() == sorted([ecc_key(P40_A), missing_key(P40_B)])


def test_a_restart_still_knows_the_fleet_it_never_saw() -> None:
    """The point of persisting the baseline: a bot that boots while a card
    is already gone still reports it gone."""
    bot, probe, clock = make(fleet())
    run(bot, clock)
    state = json.loads(json.dumps(bot.snapshot()))

    after_reboot = Probe([p40(P40_A, 0), cmp170(), rtx()])
    fresh = build(clock=clock, probe=after_reboot)
    fresh.restore(state)
    raised = actions(run(fresh, clock))
    assert keys(raised) == [missing_key(P40_B)]
    assert "Tesla P40 #1" in raised[0].text


def test_restore_is_tolerant_of_rubbish_and_refuses_the_future() -> None:
    bot, _, _ = make(fleet())
    bot.restore({})
    bot.restore({"baseline": "not a mapping", "ecc": 7, "open": []})
    assert bot.expected_fleet() == {}
    with pytest.raises(GpuBotError):
        bot.restore({"version": SNAPSHOT_VERSION + 1, "baseline": {}})


def test_pausing_forgets_the_open_requests_so_resume_re_raises() -> None:
    bot, probe, clock = make(fleet())
    run(bot, clock)
    probe.samples = [p40(P40_A, 0, temperature_c=95.0), cmp170(), rtx()]
    assert len(actions(run(bot, clock))) == 2  # the gone P40 and the hot one

    bot.on_pause()
    assert bot.open_keys() == []
    bot.on_resume()
    assert len(actions(run(bot, clock))) == 2


# --------------------------------------------------------------------------
# nvidia_smi_probe: the parser, against captured text
# --------------------------------------------------------------------------

#: Real-shaped output for the owner's machine.  The P40s report no fan
#: (passively cooled) and the consumer cards report no ECC counters, so
#: three of the four rows carry [N/A] fields -- which is the normal case,
#: not an edge case.
CAPTURED = """\
0, GPU-11111111-1111-1111-1111-111111111111, Tesla P40, 24576, 1024, 52, [N/A], 51.23, 250.00, 3, 3, 16, 16, 0, 0, 0x0000000000000000
1, GPU-22222222-2222-2222-2222-222222222222, Tesla P40, 24576, 23997, 84, [N/A], 243.57, 250.00, 3, 3, 16, 16, 4, 100, 0x0000000000000024
2, GPU-33333333-3333-3333-3333-333333333333, NVIDIA CMP 170HX, 8192, 512, 61, 45, 180.11, 250.00, 2, 3, 4, 16, [N/A], 99, 0x0000000000000000
3, GPU-44444444-4444-4444-4444-444444444444, NVIDIA GeForce RTX 5070, 12288, [N/A], 48, 30, [N/A], [N/A], 5, 5, 16, 16, [N/A], 12, [N/A]
"""


def test_the_command_says_noheader_nounits_and_asks_for_every_field() -> None:
    argv = nvidia_smi_command()
    assert argv[0] == "nvidia-smi"
    assert argv[1] == "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY_FIELDS)
    assert argv[2] == "--format=csv,noheader,nounits"
    assert nvidia_smi_command("/usr/bin/nvidia-smi")[0] == "/usr/bin/nvidia-smi"


def test_the_parser_reads_captured_output_including_na_fields() -> None:
    samples = parse_nvidia_smi_csv(CAPTURED)
    assert [s.uuid for s in samples] == [P40_A, P40_B, CMP, RTX]

    first = samples[0]
    assert first.index == 0
    assert first.name == "Tesla P40"
    assert first.memory_total_mb == 24576.0
    assert first.temperature_c == 52.0
    assert first.fan_percent is None          # [N/A]: no fan, not a stopped fan
    assert first.power_w == 51.23
    assert first.pcie_gen_current == 3
    assert first.ecc_errors == 0              # zero is a reading, None is not
    assert first.throttle_reasons == ()
    assert first.temperature_slowdown_c is None

    hot = samples[1]
    assert hot.temperature_c == 84.0
    assert hot.memory_fraction == pytest.approx(23997 / 24576)
    assert hot.ecc_errors == 4
    assert hot.throttle_reasons == ("sw_power_cap", "sw_thermal_slowdown")
    assert hot.throttling_hot_or_capped() == ("sw_power_cap", "sw_thermal_slowdown")

    cmp_row = samples[2]
    assert (cmp_row.pcie_gen_current, cmp_row.pcie_gen_max) == (2, 3)
    assert (cmp_row.pcie_width_current, cmp_row.pcie_width_max) == (4, 16)
    assert cmp_row.ecc_errors is None
    assert cmp_row.label == "NVIDIA CMP 170HX #2"

    consumer = samples[3]
    assert consumer.memory_used_mb is None
    assert consumer.power_w is None and consumer.power_limit_w is None
    assert consumer.memory_fraction is None
    assert consumer.throttle_reasons == ()


def test_the_parser_skips_blanks_and_a_header_nobody_suppressed() -> None:
    header = ", ".join(NVIDIA_SMI_QUERY_FIELDS)
    text = f"{header}\n\n{CAPTURED}\n   \n"
    assert len(parse_nvidia_smi_csv(text)) == 4
    assert parse_nvidia_smi_csv("") == []


def test_the_parser_refuses_a_row_it_cannot_trust() -> None:
    # A short row means the query and the parser have drifted; reading it
    # as three cards when four were asked for would report one missing.
    with pytest.raises(GpuProbeError):
        parse_nvidia_smi_csv("0, GPU-1, Tesla P40, 24576\n")
    with pytest.raises(GpuProbeError):
        parse_nvidia_smi_csv(CAPTURED.replace(P40_A, "[N/A]"))
    with pytest.raises(GpuProbeError):
        parse_nvidia_smi_csv(CAPTURED.replace(", 24576, 1024,", ", lots, 1024,"))
    with pytest.raises(GpuProbeError):
        parse_nvidia_smi_csv(None)  # type: ignore[arg-type]


def test_throttle_masks_decode_to_names() -> None:
    assert decode_throttle_reasons("0x0000000000000000") == ()
    assert decode_throttle_reasons("[N/A]") == ()
    assert decode_throttle_reasons("0x0000000000000004") == ("sw_power_cap",)
    assert decode_throttle_reasons("0x0000000000000041") == (
        "gpu_idle",
        "hw_thermal_slowdown",
    )
    # A driver that spells them out instead of masking them still parses.
    assert decode_throttle_reasons("SW Power Cap") == ("sw_power_cap",)


def test_nvidia_smi_probe_runs_the_command_it_built() -> None:
    seen: Dict[str, Any] = {}

    def runner(argv: Sequence[str], timeout_s: float) -> str:
        seen["argv"] = tuple(argv)
        seen["timeout"] = timeout_s
        return CAPTURED

    samples = nvidia_smi_probe(runner=runner, timeout_s=7.0)
    assert seen["argv"] == nvidia_smi_command()
    assert seen["timeout"] == 7.0
    assert [s.index for s in samples] == [0, 1, 2, 3]


def test_the_bot_never_calls_the_real_probe() -> None:
    """Injected, not imported: the module builds the command and the bot
    does not know it exists."""
    source = (ROOT / "jarvis_bots" / "bots" / "gpu_bot.py").read_text(encoding="utf-8")
    body = source.split("class GpuBot(BaseBot):", 1)[1].split("\ndef build(", 1)[0]
    for forbidden in ("nvidia_smi_probe", "subprocess", "nvidia_smi_command", "time.time"):
        assert forbidden not in body


# --------------------------------------------------------------------------
# under a real Supervisor
# --------------------------------------------------------------------------


def test_it_registers_and_ticks_under_a_real_supervisor() -> None:
    clock = Clock()
    probe = Probe(fleet())
    bot = build(clock=clock, probe=probe)
    supervisor = Supervisor(BotRegistry([bot]), clock)

    report = supervisor.run_round(clock.at)
    assert report.ticked == 1 and report.failed == 0
    assert supervisor.badge_status() == {"attention": 0, "state": "ok"}

    # The card goes away: one badge item, and the bot stays running.
    probe.samples = [p40(P40_A, 0), cmp170(), rtx()]
    clock.advance(INFO.interval_s)
    report = supervisor.run_round(clock.at)
    assert report.ticked == 1 and report.failed == 0
    assert supervisor.attention_count(BOT_ID) == 1
    assert supervisor.attention_items(BOT_ID)[0].key == missing_key(P40_B)
    assert supervisor.bot_state(BOT_ID) is BotState.RUNNING
    assert supervisor.health(BOT_ID).consecutive_failures == 0

    # Still gone for an hour: still one item, not twelve.
    for _ in range(12):
        clock.advance(INFO.interval_s)
        supervisor.run_round(clock.at)
    assert supervisor.attention_count(BOT_ID) == 1

    # It comes back and the badge empties itself, with nothing wired.
    probe.samples = fleet()
    clock.advance(INFO.interval_s)
    supervisor.run_round(clock.at)
    assert supervisor.attention_count(BOT_ID) == 0
    assert supervisor.badge_status()["attention"] == 0


def test_a_failing_probe_does_not_get_the_bot_quarantined() -> None:
    clock = Clock()
    probe = Probe(fleet())
    bot = build(clock=clock, probe=probe)
    supervisor = Supervisor(BotRegistry([bot]), clock)
    supervisor.run_round(clock.at)

    probe.fault = OSError("no driver")
    for _ in range(8):
        clock.advance(INFO.interval_s)
        report = supervisor.run_round(clock.at)
        assert report.failed == 0
    assert supervisor.health(BOT_ID).consecutive_failures == 0
    assert supervisor.bot_state(BOT_ID) is BotState.RUNNING
    assert supervisor.attention_count(BOT_ID) == 1  # the probe fault itself


def test_state_survives_a_supervisor_restart() -> None:
    clock = Clock()
    probe = Probe(fleet())
    store: Dict[str, Any] = {}
    supervisor = Supervisor(BotRegistry([build(clock=clock, probe=probe)]), clock, store=store)
    supervisor.run_round(clock.at)
    supervisor.save_state()

    # A new process, and the card did not come back from the reboot.
    after = Probe([p40(P40_A, 0), cmp170(), rtx()])
    revived = Supervisor(
        BotRegistry([build(clock=clock, probe=after)]),
        clock,
        store=json.loads(json.dumps(store)),
    )
    revived.load_state()
    clock.advance(INFO.interval_s)
    revived.run_round(clock.at)
    assert [item.key for item in revived.attention_items(BOT_ID)] == [missing_key(P40_B)]
