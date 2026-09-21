"""The disk watch, against a stub probe and a fixed clock.

What is worth testing here is not "does it notice 90 percent" -- that is an
``if`` -- but the claims the bot is *for*:

* a disk filling at a steady rate is projected to the right date, and the
  alert arrives when the projection crosses the threshold, not when the
  disk is already full;
* "it is full" and "it will be full" are two questions with two keys, open
  at the same time, closed independently;
* the bot refuses to guess from two readings and says so;
* a big delete -- which this machine does after every render -- resets the
  trend instead of producing a negative rate or a date in the past;
* the failures people miss (inodes, read-only) get their own alerts;
* a probe that cannot look escalates rather than going quiet.

Nothing here sleeps, opens a socket, shells out or reads the wall clock:
the clock is a counter a test moves and the probe is a stub that returns
what a test set.  The one exception is :func:`statvfs_probe`, which is
pointed at a real temporary directory, because a probe nobody ever ran
against a real filesystem is a probe that does not work.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jarvis_bots.bots.disk_bot import (  # noqa: E402
    ESCALATE_AFTER_FAILURES,
    INFO,
    MAX_SAMPLES_PER_MOUNT,
    MIN_SAMPLES,
    PROBE_ATTENTION_KEY,
    DiskBot,
    DiskBotError,
    DuPlan,
    MountSample,
    build,
    du_probe,
    filling_key,
    full_key,
    inodes_key,
    linear_trend,
    parse_du_output,
    readonly_key,
    statvfs_probe,
)
from jarvis_bots.contracts import BotState, Event, Severity  # noqa: E402
from jarvis_bots.registry import BotRegistry  # noqa: E402
from jarvis_bots.supervisor import RESOLVED_FLAG, Supervisor  # noqa: E402

GB = 1024 ** 3
HOUR = 3600.0
DAY = 86400.0
T0 = 1_700_000_000.0
MP = "/data"


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


class Clock:
    """The injected clock: it moves when a test moves it, and never else."""

    def __init__(self, at: float = T0) -> None:
        self.at = float(at)

    def __call__(self) -> float:
        return self.at

    def advance(self, seconds: float) -> float:
        self.at += float(seconds)
        return self.at


class StubProbe:
    """The injected probe.  Returns what the test set, or raises what it set."""

    def __init__(self, *samples: MountSample) -> None:
        self.samples: List[MountSample] = list(samples)
        self.raises: Optional[BaseException] = None
        self.calls = 0

    def set(self, *samples: MountSample) -> None:
        self.samples = list(samples)

    def __call__(self) -> Sequence[MountSample]:
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return list(self.samples)


def mount(
    mountpoint: str = MP,
    *,
    total: int = 100 * GB,
    free: int = 50 * GB,
    used: Optional[int] = None,
    inodes_total: int = 1_000_000,
    inodes_free: int = 900_000,
    read_only: bool = False,
    device: str = "/dev/sda1",
    filesystem: str = "ext4",
) -> MountSample:
    return MountSample(
        mountpoint=mountpoint,
        device=device,
        filesystem=filesystem,
        total_bytes=total,
        free_bytes=free,
        used_bytes=total - free if used is None else used,
        inodes_total=inodes_total,
        inodes_free=inodes_free,
        read_only=read_only,
    )


def make_bot(probe: StubProbe, clock: Clock, **kwargs) -> DiskBot:
    kwargs.setdefault("mountpoints", [MP])
    return DiskBot(clock=clock, probe=probe, **kwargs)


def tick(
    bot: DiskBot,
    clock: Clock,
    probe: StubProbe,
    *samples: MountSample,
    advance: float = HOUR,
) -> List[Event]:
    """Move the clock, hand the probe its answer, and tick once."""
    clock.advance(advance)
    if samples:
        probe.set(*samples)
    return list(bot.tick(clock.at))


def keyed(events: Sequence[Event], key: str) -> List[Event]:
    return [event for event in events if event.attention_key == key]


def actions(events: Sequence[Event]) -> List[Event]:
    return [event for event in events if event.severity is Severity.ACTION]


# ---------------------------------------------------------------------------
# the projection: the headline rule
# ---------------------------------------------------------------------------


def test_steady_fill_projects_the_right_date_and_alerts_at_the_threshold():
    """A disk losing 1 GiB an hour is projected to the hour, and the ACTION
    arrives the tick the projection crosses three days -- not before, and
    not once at 90 percent."""
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    total = 500 * GB
    free = 200 * GB
    fired: List[Event] = []
    free_when_fired = None

    while free > 60 * GB and not fired:
        free -= GB  # one gibibyte an hour == 24 GiB/day
        events = tick(bot, clock, probe, mount(total=total, free=free))
        hits = keyed(events, filling_key(MP))
        if hits:
            fired = hits
            free_when_fired = free

    assert fired, "a disk filling at 24 GB/day was never projected"
    event = fired[0]
    assert event.severity is Severity.ACTION
    assert event.href == INFO.href

    # It fired the first tick the projection reached three days, which at
    # 24 GiB/day is 72 GiB of headroom -- and no earlier.
    assert free_when_fired == 72 * GB

    # The rate, within a hair of the truth.
    assert event.data["rate_bytes_per_day"] == pytest.approx(24 * GB, rel=1e-6)

    # The date: three days after the reading that raised it, to the minute.
    assert event.data["full_at"] == pytest.approx(clock.at + 3 * DAY, abs=60.0)
    assert event.data["days_to_full"] == pytest.approx(3.0, abs=1e-3)
    assert "3.0 days" in event.text
    assert "UTC" in event.text  # the projected date, spelled out

    # Still filling, still one badge item: a standing question is asked once.
    for _ in range(5):
        free -= GB
        more = tick(bot, clock, probe, mount(total=total, free=free))
        assert not actions(more), "the same question was asked twice"
    assert bot.open_attention_keys() == [filling_key(MP)]


def test_ninety_two_percent_but_stable_alerts_on_the_absolute_rule_only():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    sample = mount(total=100 * GB, free=8 * GB, used=92 * GB)
    seen: List[Event] = []
    for _ in range(6):
        seen.extend(tick(bot, clock, probe, sample))

    assert bot.open_attention_keys() == [full_key(MP)]
    assert len(keyed(seen, full_key(MP))) == 1
    assert not keyed(seen, filling_key(MP))
    assert "92.0%" in keyed(seen, full_key(MP))[0].text
    # And it says why it is not projecting, rather than saying nothing.
    assert "steady" in bot.reasons()[MP]


def test_full_now_and_filling_are_two_open_keys_at_once():
    """"It is full" and "it will be full" must not be merged: the second is
    the one that is still actionable."""
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    total = 1000 * GB
    free = 88 * GB
    raised: List[Event] = []
    for _ in range(5):
        free -= 8 * GB  # under the 10 GB drop threshold, so no NOTICE
        raised.extend(tick(bot, clock, probe, mount(total=total, free=free)))

    keys = bot.open_attention_keys()
    assert full_key(MP) in keys and filling_key(MP) in keys
    assert full_key(MP) != filling_key(MP)
    assert len(set(keys)) == len(keys)

    opened = [event.attention_key for event in actions(raised)]
    assert sorted(opened) == sorted([full_key(MP), filling_key(MP)])


# ---------------------------------------------------------------------------
# refusing to guess
# ---------------------------------------------------------------------------


def test_fewer_than_four_samples_projects_nothing_and_says_so():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    free = 40 * GB
    for _ in range(MIN_SAMPLES - 1):
        free -= 5 * GB
        events = tick(bot, clock, probe, mount(total=1000 * GB, free=free))
        assert not keyed(events, filling_key(MP))

    projection = bot.projections()[MP]
    assert projection.days_to_full is None
    assert projection.samples == MIN_SAMPLES - 1
    assert "not enough history" in projection.reason
    assert f"{MIN_SAMPLES - 1} of the {MIN_SAMPLES}" in projection.reason


def test_under_an_hour_of_history_projects_nothing_and_says_so():
    """Four readings a quarter of an hour apart during a render would
    "prove" the disk fills before lunch."""
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    free = 40 * GB
    for _ in range(4):
        free -= 5 * GB
        events = tick(
            bot, clock, probe, mount(total=1000 * GB, free=free), advance=900.0
        )
        assert not keyed(events, filling_key(MP))

    projection = bot.projections()[MP]
    assert projection.samples >= MIN_SAMPLES
    assert projection.span_s < HOUR
    assert projection.days_to_full is None
    assert "not enough history" in projection.reason
    assert "hour is the minimum" in projection.reason
    assert "45 minutes" in projection.reason


def test_a_large_delete_resets_the_trend_instead_of_projecting_nonsense():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    total = 500 * GB
    free = 120 * GB
    for _ in range(10):
        free -= 8 * GB
        tick(bot, clock, probe, mount(total=total, free=free))

    projecting = bot.projections()[MP]
    assert projecting.days_to_full is not None  # it was trending down
    assert filling_key(MP) in bot.open_attention_keys()

    # The render cleanup: 200 GB freed between one look and the next.
    free += 200 * GB
    events = tick(bot, clock, probe, mount(total=total, free=free))

    projection = bot.projections()[MP]
    assert projection.days_to_full is None, "projected a date across a delete"
    assert projection.fill_bytes_per_day <= 0.0 or projection.days_to_full is None
    assert "trend reset" in projection.reason
    assert len(bot.history(MP)) == 1, "history before the delete was kept"

    # And the standing question is closed, not left hanging.
    closed = keyed(events, filling_key(MP))
    assert len(closed) == 1
    assert closed[0].severity is Severity.NOTICE
    assert closed[0].data[RESOLVED_FLAG] is True
    assert filling_key(MP) not in bot.open_attention_keys()

    # Nothing absurd came out of it: no negative rate, no date in the past.
    for event in events:
        assert event.data.get("days_to_full") in (None,)
        assert (event.data.get("rate_bytes_per_day") or 0.0) <= 0.0


def test_a_slow_rise_in_free_space_never_projects():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    free = 50 * GB
    for _ in range(8):
        free += 512 * 1024 * 1024  # half a gibibyte an hour, cleaning up
        events = tick(bot, clock, probe, mount(total=500 * GB, free=free))
        assert not keyed(events, filling_key(MP))
    assert bot.projections()[MP].days_to_full is None
    assert "steady or growing" in bot.projections()[MP].reason


# ---------------------------------------------------------------------------
# the other failures
# ---------------------------------------------------------------------------


def test_inode_exhaustion_alerts_separately_from_bytes():
    """The one people miss: df shows plenty of space and every write fails."""
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    events = tick(
        bot,
        clock,
        probe,
        mount(
            total=1000 * GB,
            free=900 * GB,  # 10% used: nowhere near the byte threshold
            inodes_total=5_000_000,
            inodes_free=100_000,  # 98% of the inodes are gone
        ),
    )

    assert bot.open_attention_keys() == [inodes_key(MP)]
    assert not keyed(events, full_key(MP))
    alert = keyed(events, inodes_key(MP))[0]
    assert alert.severity is Severity.ACTION
    assert "inodes" in alert.text
    assert "no space left on device" in alert.text
    assert alert.data["percent_inodes_used"] == pytest.approx(98.0)


def test_a_filesystem_without_an_inode_table_never_alerts_on_inodes():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)
    events = tick(
        bot, clock, probe, mount(inodes_total=0, inodes_free=0, free=95 * GB)
    )
    assert not keyed(events, inodes_key(MP))


def test_read_only_alerts_and_recovery_resolves():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    raised = tick(bot, clock, probe, mount(read_only=True))
    alert = keyed(raised, readonly_key(MP))[0]
    assert alert.severity is Severity.ACTION
    assert "read-only" in alert.text
    assert "/dev/sda1" in alert.text
    assert bot.open_attention_keys() == [readonly_key(MP)]

    healed = tick(bot, clock, probe, mount(read_only=False))
    closed = keyed(healed, readonly_key(MP))[0]
    assert closed.severity is Severity.NOTICE
    assert closed.severity < Severity.ACTION  # so it cannot re-open the key
    assert closed.wants_attention is False
    assert closed.data[RESOLVED_FLAG] is True
    assert bot.open_attention_keys() == []


def test_recovery_from_a_full_disk_resolves_the_absolute_key():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    tick(bot, clock, probe, mount(total=100 * GB, free=5 * GB, used=95 * GB))
    assert bot.open_attention_keys() == [full_key(MP)]

    freed = tick(bot, clock, probe, mount(total=100 * GB, free=60 * GB, used=40 * GB))
    closed = keyed(freed, full_key(MP))
    assert len(closed) == 1
    assert closed[0].data[RESOLVED_FLAG] is True
    assert "back under" in closed[0].text
    assert bot.open_attention_keys() == []


def test_a_sudden_fifteen_gigabyte_drop_is_a_notice():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    tick(bot, clock, probe, mount(total=500 * GB, free=100 * GB))
    events = tick(bot, clock, probe, mount(total=500 * GB, free=85 * GB))

    drops = [event for event in events if event.data.get("rule") == f"disk:drop:{MP}"]
    assert len(drops) == 1
    notice = drops[0]
    assert notice.severity is Severity.NOTICE
    assert notice.attention_key is None, "a write is news, not a standing question"
    assert notice.data["lost_bytes"] == 15 * GB
    assert "15.0 GB" in notice.text

    # A drop under the threshold says nothing at all.
    quiet = tick(bot, clock, probe, mount(total=500 * GB, free=80 * GB))
    assert not [e for e in quiet if e.data.get("rule") == f"disk:drop:{MP}"]


def test_the_drop_threshold_is_configurable():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock, drop_bytes=2 * GB)
    tick(bot, clock, probe, mount(total=500 * GB, free=100 * GB))
    events = tick(bot, clock, probe, mount(total=500 * GB, free=97 * GB))
    assert [e for e in events if e.data.get("rule") == f"disk:drop:{MP}"]


# ---------------------------------------------------------------------------
# boundaries
# ---------------------------------------------------------------------------


def test_exactly_ninety_percent_alerts_and_a_hair_under_does_not():
    clock, probe = Clock(), StubProbe()
    probe_a, clock_a = StubProbe(), Clock()
    at_ninety = make_bot(probe_a, clock_a)
    events = tick(
        at_ninety,
        clock_a,
        probe_a,
        mount(total=100 * GB, free=10 * GB, used=90 * GB),
    )
    assert keyed(events, full_key(MP)), "exactly 90% must alert"

    bot = make_bot(probe, clock)
    just_under = tick(
        bot,
        clock,
        probe,
        mount(total=100 * GB, free=10 * GB + 1, used=90 * GB - 1),
    )
    assert not keyed(just_under, full_key(MP)), "89.99...% must not"


def test_the_percent_threshold_is_configurable():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock, percent_threshold=50.0)
    events = tick(bot, clock, probe, mount(total=100 * GB, free=49 * GB, used=51 * GB))
    assert keyed(events, full_key(MP))


def test_exactly_three_projected_days_alerts():
    """The boundary is inclusive, and must not depend on the last bit of a
    float division: 72 GiB of headroom lost at exactly 24 GiB a day is
    exactly three days."""
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock, projected_days_threshold=3.0)

    total = 500 * GB
    free = 78 * GB
    events: List[Event] = []
    for _ in range(5):  # 77, 76, 75, 74, 73 -- all over three days
        free -= GB
        events = tick(bot, clock, probe, mount(total=total, free=free))
        assert not keyed(events, filling_key(MP))
    assert bot.projections()[MP].days_to_full == pytest.approx(73 / 24, abs=1e-3)

    free -= GB  # 72 GiB: exactly three days
    events = tick(bot, clock, probe, mount(total=total, free=free))
    alert = keyed(events, filling_key(MP))
    assert alert, "a disk exactly three days from full must alert"
    assert alert[0].data["days_to_full"] == pytest.approx(3.0, abs=1e-6)


def test_the_days_threshold_is_configurable():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock, projected_days_threshold=30.0)
    total = 5000 * GB
    free = 300 * GB
    events: List[Event] = []
    for _ in range(4):
        free -= GB
        events = tick(bot, clock, probe, mount(total=total, free=free))
    # 296 GiB at 24 GiB/day is twelve days: inside a thirty day threshold.
    assert keyed(events, filling_key(MP))


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------


def test_snapshot_history_is_bounded_by_count_and_round_trips():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock, history_window_s=3650 * DAY)

    free = 4000 * GB
    for _ in range(MAX_SAMPLES_PER_MOUNT + 120):
        free -= GB
        tick(bot, clock, probe, mount(total=8000 * GB, free=free))

    snapshot = bot.snapshot()
    assert len(snapshot["history"][MP]) == MAX_SAMPLES_PER_MOUNT
    assert json.loads(json.dumps(snapshot)) == snapshot  # JSON-able, as promised

    restored = make_bot(StubProbe(), Clock(), history_window_s=3650 * DAY)
    restored.restore(json.loads(json.dumps(snapshot)))
    assert restored.snapshot() == snapshot
    assert restored.history(MP) == bot.history(MP)


def test_snapshot_history_is_bounded_by_age():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)  # the default three day window

    free = 4000 * GB
    for _ in range(200):  # 200 hours, well past three days
        free -= GB
        tick(bot, clock, probe, mount(total=8000 * GB, free=free))

    history = bot.history(MP)
    assert len(history) <= MAX_SAMPLES_PER_MOUNT
    assert history[-1][0] - history[0][0] <= bot.history_window_s
    assert len(history) < 200


def test_a_restored_bot_keeps_projecting_without_relearning():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)

    free = 300 * GB
    for _ in range(6):
        free -= 4 * GB
        tick(bot, clock, probe, mount(total=500 * GB, free=free))
    before = bot.projections()[MP]
    assert before.days_to_full is not None

    fresh_clock = Clock(clock.at)
    fresh = make_bot(probe, fresh_clock)
    fresh.restore(bot.snapshot())
    free -= 4 * GB
    tick(fresh, fresh_clock, probe, mount(total=500 * GB, free=free))

    after = fresh.projections()[MP]
    assert after.days_to_full is not None
    assert after.fill_bytes_per_day == pytest.approx(before.fill_bytes_per_day, rel=0.2)


def test_restore_drops_history_for_mountpoints_no_longer_watched():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock, mountpoints=[MP, "/scratch"])
    tick(bot, clock, probe, mount(MP), mount("/scratch", free=10 * GB))
    tick(bot, clock, probe, mount(MP), mount("/scratch", free=9 * GB))
    snapshot = bot.snapshot()
    assert set(snapshot["history"]) == {MP, "/scratch"}

    narrowed = make_bot(StubProbe(), Clock(), mountpoints=[MP])
    narrowed.restore(snapshot)
    assert set(narrowed.snapshot()["history"]) == {MP}


def test_restore_survives_rubbish_and_refuses_the_future():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)
    bot.restore({})
    bot.restore({"history": "not a mapping", "open": 7, "ticks": "soon"})
    assert bot.history(MP) == []
    with pytest.raises(DiskBotError):
        bot.restore({"version": 99})


def test_pause_forgets_attention_so_resume_can_re_raise():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)
    full = mount(total=100 * GB, free=2 * GB, used=98 * GB)
    tick(bot, clock, probe, full)
    assert bot.open_attention_keys() == [full_key(MP)]

    bot.on_pause()
    assert bot.open_attention_keys() == []
    assert bot.history(MP), "the readings are measurement, not attention"

    bot.on_resume()
    events = tick(bot, clock, probe, full)
    assert keyed(events, full_key(MP)), "a resumed bot must say it again"


# ---------------------------------------------------------------------------
# the probe's own health
# ---------------------------------------------------------------------------


def test_a_raising_probe_is_an_error_event_and_escalates_after_three():
    clock, probe = Clock(), StubProbe(mount())
    bot = make_bot(probe, clock)

    tick(bot, clock, probe)  # one good look first
    probe.raises = OSError("[Errno 5] Input/output error: '/data'")

    first = tick(bot, clock, probe)
    assert len(first) == 1
    assert first[0].severity is Severity.ERROR
    assert first[0].attention_key is None
    assert "OSError" in first[0].text

    second = tick(bot, clock, probe)
    assert second[0].severity is Severity.ERROR
    assert second[0].attention_key is None

    third = tick(bot, clock, probe)
    escalation = keyed(third, PROBE_ATTENTION_KEY)
    assert escalation, f"{ESCALATE_AFTER_FAILURES} failures in a row must escalate"
    assert escalation[0].severity is Severity.ACTION
    assert escalation[0].data["failures"] == ESCALATE_AFTER_FAILURES
    assert bot.open_attention_keys() == [PROBE_ATTENTION_KEY]

    # Already escalated: quiet, because the badge is already carrying it.
    assert tick(bot, clock, probe) == []

    probe.raises = None
    healed = tick(bot, clock, probe, mount())
    closed = keyed(healed, PROBE_ATTENTION_KEY)
    assert closed and closed[0].data[RESOLVED_FLAG] is True
    assert bot.open_attention_keys() == []


def test_a_probe_returning_rubbish_is_reported_not_raised():
    clock = Clock()

    def probe():
        return ["/data is 90% full"]

    bot = DiskBot(clock=clock, probe=probe, mountpoints=[MP])
    clock.advance(HOUR)
    events = list(bot.tick(clock.at))
    assert len(events) == 1
    assert events[0].severity is Severity.ERROR
    assert "DiskBotError" in events[0].text


def test_a_tick_never_raises_whatever_the_probe_does():
    clock = Clock()

    def probe():
        raise KeyError("nope")

    bot = DiskBot(clock=clock, probe=probe, mountpoints=[MP])
    for _ in range(6):
        clock.advance(HOUR)
        bot.tick(clock.at)  # must not raise


# ---------------------------------------------------------------------------
# the card
# ---------------------------------------------------------------------------


def test_status_is_idle_before_the_first_look_and_running_after():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock)
    assert bot.status().state is BotState.IDLE

    tick(bot, clock, probe, mount(total=100 * GB, free=40 * GB))
    status = bot.status()
    assert status.state is BotState.RUNNING
    values = {stat.label: stat.value for stat in status.stats}
    assert MP in values["Tightest"]
    assert "40.0 GB free" in values["Tightest"]
    assert "60.0%" in values["Tightest"]
    assert values["Fills in"] == "stable"


def test_status_names_the_tightest_mountpoint_and_the_worst_projection():
    clock, probe = Clock(), StubProbe()
    bot = make_bot(probe, clock, mountpoints=[MP, "/scratch"])

    roomy, tight = 400 * GB, 90 * GB
    for _ in range(5):
        roomy -= GB  # 24 GiB/day, 16 days out
        tight -= 8 * GB  # 192 GiB/day, under a day
        tick(
            bot,
            clock,
            probe,
            mount(MP, total=1000 * GB, free=roomy),
            mount("/scratch", total=1000 * GB, free=tight),
        )

    values = {stat.label: stat.value for stat in bot.status().stats}
    assert values["Tightest"].startswith("/scratch")
    assert values["Watching"] == "2 filesystems"
    assert "/scratch" in values["Fills in"] and "stable" not in values["Fills in"]
    assert bot.tightest().mountpoint == "/scratch"
    assert bot.worst_projection().mountpoint == "/scratch"


def test_status_says_when_the_probe_is_failing():
    clock, probe = Clock(), StubProbe(mount())
    bot = make_bot(probe, clock)
    tick(bot, clock, probe)
    probe.raises = PermissionError("denied")
    tick(bot, clock, probe)
    assert "PermissionError" in bot.status().detail


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_construction_refuses_nonsense():
    clock = Clock()
    with pytest.raises(DiskBotError):
        DiskBot(clock=clock, probe="not callable")
    with pytest.raises(DiskBotError):
        DiskBot(clock=clock, probe=StubProbe(), percent_threshold=0.0)
    with pytest.raises(DiskBotError):
        DiskBot(clock=clock, probe=StubProbe(), percent_threshold=101.0)
    with pytest.raises(DiskBotError):
        DiskBot(clock=clock, probe=StubProbe(), projected_days_threshold=0.0)
    with pytest.raises(DiskBotError):
        DiskBot(clock=clock, probe=StubProbe(), drop_bytes=0)
    with pytest.raises(DiskBotError):
        DiskBot(clock=clock, probe=StubProbe(), mountpoints=[])
    with pytest.raises(ValueError):
        DiskBot(clock="not a clock", probe=StubProbe())
    with pytest.raises(DiskBotError):
        build(clock=clock)  # no probe and no mountpoints


def test_build_with_no_probe_watches_the_named_mountpoints_for_real(tmp_path):
    """The default wiring: no probe, so build() reads the named mountpoints
    through statvfs_probe.  Ticked against a real directory, end to end."""
    clock = Clock()
    bot = build(clock=clock, mountpoints=[str(tmp_path)])
    clock.advance(HOUR)
    events = bot.tick(clock.at)

    assert isinstance(events, tuple)
    assert bot.status().state is BotState.RUNNING
    sample = bot.tightest()
    assert sample is not None and sample.mountpoint == str(tmp_path)
    assert sample.total_bytes > 0
    assert bot.projections()[str(tmp_path)].days_to_full is None  # one reading


def test_identity_matches_the_launcher_contract():
    assert INFO.id == "disk"
    assert INFO.name == "Disk watch"
    assert INFO.kind == "grid"
    assert INFO.interval_s == 900.0
    assert INFO.href == "/bots/disk"


def test_a_sample_refuses_impossible_numbers():
    with pytest.raises(ValueError):
        MountSample(mountpoint="")
    with pytest.raises(ValueError):
        MountSample(mountpoint=MP, free_bytes=-1)
    with pytest.raises(TypeError):
        MountSample(mountpoint=MP, total_bytes=1.5)


# ---------------------------------------------------------------------------
# the trend, on its own
# ---------------------------------------------------------------------------


def test_linear_trend_recovers_a_known_slope():
    points = [(T0 + i * HOUR, 100 * GB - i * GB) for i in range(10)]
    slope = linear_trend(points)
    assert slope == pytest.approx(-GB / HOUR, rel=1e-9)


def test_linear_trend_has_no_answer_for_one_point_or_one_instant():
    assert linear_trend([]) is None
    assert linear_trend([(T0, 5)]) is None
    assert linear_trend([(T0, 5), (T0, 9)]) is None


# ---------------------------------------------------------------------------
# the two real probes
# ---------------------------------------------------------------------------


def test_statvfs_probe_reads_a_real_directory(tmp_path):
    (tmp_path / "frame.exr").write_bytes(b"0" * 4096)

    samples = statvfs_probe([str(tmp_path)])

    assert len(samples) == 1
    sample = samples[0]
    assert isinstance(sample, MountSample)
    assert sample.mountpoint == str(tmp_path)
    assert sample.total_bytes > 0
    assert sample.free_bytes > 0
    assert sample.used_bytes >= 0
    assert sample.free_bytes <= sample.total_bytes
    assert sample.used_bytes <= sample.total_bytes
    assert 0.0 <= sample.percent_used <= 100.0
    assert sample.inodes_total >= 0
    assert sample.inodes_free >= 0
    assert 0.0 <= sample.percent_inodes_used <= 100.0
    assert sample.read_only is False  # a temp dir we just wrote to


def test_statvfs_probe_raises_rather_than_reporting_a_plausible_zero(tmp_path):
    with pytest.raises(OSError):
        statvfs_probe([str(tmp_path / "nothing-here")])


def test_statvfs_probe_labels_the_device_from_the_mount_table(tmp_path):
    """The device and filesystem come from /proc/self/mounts, which is
    parsed with its octal escapes; a machine without it simply gets no
    labels."""
    mounts = tmp_path / "mounts"
    root = tmp_path / "disk cache"
    root.mkdir()
    mounts.write_text(
        f"/dev/nvme0n1p2 {str(root).replace(' ', chr(92) + '040')} xfs rw,relatime 0 0\n"
    )

    sample = statvfs_probe([str(root)], mounts_path=str(mounts))[0]
    assert sample.device == "/dev/nvme0n1p2"
    assert sample.filesystem == "xfs"

    nowhere = statvfs_probe([str(root)], mounts_path=str(tmp_path / "absent"))[0]
    assert nowhere.device == "" and nowhere.filesystem == ""


def test_du_probe_builds_the_command_line():
    plan = du_probe("/data/renders", 2)
    assert isinstance(plan, DuPlan)
    assert plan.argv == ("du", "-k", "-x", "-d", "2", "--", "/data/renders")
    assert plan.path == "/data/renders"
    assert plan.depth == 2
    # -k, because du's unit is otherwise whatever BLOCKSIZE says it is; -x,
    # so it does not walk into other filesystems; -- so a path starting
    # with a dash is a path.
    assert du_probe("-weird").argv[-1] == "-weird"
    assert du_probe("/x").argv[4] == "1"  # the default depth

    with pytest.raises(ValueError):
        du_probe("/x", -1)
    with pytest.raises(TypeError):
        du_probe("/x", 1.5)
    with pytest.raises(ValueError):
        du_probe("")


CAPTURED_DU = (
    "4\t/data/renders/scene-01/cache\n"
    "1048576\t/data/renders/scene-01\n"
    "du: cannot read directory '/data/renders/private': Permission denied\n"
    "2097152\t/data/renders/scene 02\n"
    "3145732\t/data/renders\n"
)


def test_du_probe_parses_captured_output():
    pairs = du_probe("/data/renders", 2).parse(CAPTURED_DU)

    assert pairs == [
        ("/data/renders/scene-01/cache", 4 * 1024),
        ("/data/renders/scene-01", 1048576 * 1024),
        ("/data/renders/scene 02", 2097152 * 1024),
        ("/data/renders", 3145732 * 1024),
    ]
    # The unreadable directory is skipped, not fatal: ninety directories
    # read is better than none.
    assert all("cannot read" not in path for path, _ in pairs)
    # du's own order is kept -- children before the parent.
    assert pairs[-1][0] == "/data/renders"


def test_parse_du_output_edge_cases():
    assert parse_du_output("") == []
    assert parse_du_output("\n  \n") == []
    assert parse_du_output("8\t/a\r\n") == [("/a", 8192)]
    # Only the first tab separates, so a path containing one survives.
    assert parse_du_output("8\t/a\tb") == [("/a\tb", 8192)]
    # Space separated (some du builds, and copy-pasted output).
    assert parse_du_output("8 /a") == [("/a", 8192)]
    assert parse_du_output("total\n") == []
    with pytest.raises(TypeError):
        parse_du_output(b"8\t/a")


# ---------------------------------------------------------------------------
# under a real supervisor
# ---------------------------------------------------------------------------


def test_registers_and_ticks_under_a_real_supervisor():
    """The claim contracts.py opens with: a second bot is a class and a
    page, not a refactor.  Registered, scheduled, alerted, badged and
    persisted by the framework, with nothing disk-shaped in it."""
    clock, probe = Clock(), StubProbe(mount(total=500 * GB, free=200 * GB))
    bot = build(clock=clock, probe=probe, mountpoints=[MP])
    pushed: List[Event] = []

    registry = BotRegistry([bot])
    supervisor = Supervisor(registry, clock, alerts=pushed.append)

    total, free = 500 * GB, 200 * GB
    for _ in range(140):
        clock.advance(INFO.interval_s)
        free -= 2 * GB  # 192 GiB/day at the 900s interval
        probe.set(mount(total=total, free=max(free, GB)))
        report = supervisor.run_round(clock.at)
        assert report.failed == 0
        if free <= GB:
            break

    badge = supervisor.badge_status()
    assert badge["state"] == "ok"
    assert badge["attention"] >= 1
    keys = {item.key for item in supervisor.attention_items("disk")}
    assert filling_key(MP) in keys
    assert full_key(MP) in keys
    assert keys == set(bot.open_attention_keys())
    assert pushed, "an ACTION event reached the injected alert service"

    card = supervisor.launcher_state(clock.at)["bots"][0]
    assert card["id"] == "disk"
    assert card["name"] == "Disk watch"
    assert card["kind"] == "grid"
    assert card["state"] == "running"
    assert card["href"] == "/bots/disk"
    assert card["attention"] == badge["attention"]
    assert [stat["label"] for stat in card["stats"]] == [
        "Tightest",
        "Watching",
        "Fills in",
    ]

    # Recovery: the owner freed a lot of space, and the badge empties.
    before = len(supervisor.attention_items("disk"))
    assert before >= 2
    clock.advance(INFO.interval_s)
    probe.set(mount(total=total, free=450 * GB))
    supervisor.run_round(clock.at)
    assert supervisor.attention_items("disk") == []
    assert supervisor.badge_status()["attention"] == 0
    assert bot.open_attention_keys() == []

    # And the state the framework persists comes back.
    state = supervisor.save_state()
    assert json.loads(json.dumps(state)) == state
    twin_clock = Clock(clock.at)
    twin = build(clock=twin_clock, probe=StubProbe(), mountpoints=[MP])
    twin_supervisor = Supervisor(BotRegistry([twin]), twin_clock)
    twin_supervisor.load_state(json.loads(json.dumps(state)))
    assert twin.history(MP) == bot.history(MP)


def test_a_broken_probe_does_not_quarantine_the_bot():
    """Three quarters of an hour of failures escalate to the badge; the bot
    keeps its schedule, because a quarantined disk watch is exactly no disk
    watch."""
    clock, probe = Clock(), StubProbe(mount())
    bot = build(clock=clock, probe=probe, mountpoints=[MP])
    supervisor = Supervisor(BotRegistry([bot]), clock)

    probe.raises = OSError("gone")
    for _ in range(8):
        clock.advance(INFO.interval_s)
        report = supervisor.run_round(clock.at)
        assert report.failed == 0
        assert report.quarantined == 0
    assert supervisor.quarantined_ids() == []
    assert [item.key for item in supervisor.attention_items("disk")] == [
        PROBE_ATTENTION_KEY
    ]
