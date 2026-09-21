"""Tests for drop windows: jarvis_poke.snipe and PollScheduler.retime_source.

Design: jarvis_poke/snipe.py -- "A 'snipe' is not a faster scraper", with
three rails stated there and checked here: the 30s floor, the bounded
window, and a pause outranking a window.

Most of this file is written as things the snipe layer must *refuse* or
must *not* do, because the failure mode of a feature called "sniping" is
that it quietly becomes impolite, and the failure mode of a feature that
promises speed is that it is fast at deciding and silent at delivering.
So the preflight tests are as heavy as the scheduling ones.

Nothing here sleeps, nothing reaches a network, and every clock is
injected -- the same contract the rest of the package keeps.
"""

from __future__ import annotations

import datetime as _dt
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_poke.alerts_bridge import BridgeError
from jarvis_poke.contracts import (
    Action,
    FetchPolicy,
    FetchResult,
    Product,
    ProductKind,
    Rule,
    Verdict,
)
from jarvis_poke.snipe import (
    ALERT_PATH_MAX_AGE_S,
    DEFAULT_TTL_S,
    MAX_OPEN_PER_DAY_S,
    MAX_WINDOW_S,
    SNIPE_DATA_KEYS,
    SNIPE_KIND,
    ArmCheck,
    DropWindow,
    EventKind,
    Phase,
    SnipeAlertBridge,
    SnipeController,
    SnipeError,
    SnipePlan,
    daily_windows,
)
from jarvis_poke.sources import PollScheduler, SchedulerError

from tests.test_poke_sources import (  # noqa: E402  - reuse the real fixtures
    T0,
    SimClock,
    make_catalog,
    make_policies,
    sku_of,
)


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class FakeScheduler:
    """Just enough scheduler to watch the controller's decisions."""

    def __init__(self, policies: Optional[Dict[str, FetchPolicy]] = None) -> None:
        self._p = dict(policies or {"alpha": FetchPolicy("alpha", min_interval_s=300.0)})
        self.retimes: List[tuple] = []
        self.paused: Dict[str, Dict[str, Any]] = {}

    def policies(self) -> Dict[str, FetchPolicy]:
        return dict(self._p)

    def policy(self, source: str) -> FetchPolicy:
        try:
            return self._p[source]
        except KeyError:
            raise SchedulerError(f"no FetchPolicy for source {source!r}") from None

    def set_policy(self, policy: FetchPolicy) -> None:
        self._p[policy.source] = policy

    def retime_source(self, source: str, now: float) -> int:
        self.retimes.append((source, now, self._p[source].min_interval_s))
        return 1

    def pause_state(self, now: float) -> Dict[str, Dict[str, Any]]:
        return {
            source: self.paused.get(source, {"paused_until": 0.0, "reason": ""})
            for source in self._p
        }


def window(**over: Any) -> DropWindow:
    kwargs: Dict[str, Any] = dict(
        name="restock",
        source="alpha",
        opens_at=T0 + 3600.0,
        closes_at=T0 + 3600.0 + 1800.0,
        interval_s=30.0,
    )
    kwargs.update(over)
    return DropWindow(**kwargs)


def controller(sched: Optional[FakeScheduler] = None, plan: Optional[SnipePlan] = None,
               **kwargs: Any) -> SnipeController:
    sched = sched or FakeScheduler()
    plan = plan if plan is not None else SnipePlan.of(window())
    kwargs.setdefault("alert_probe", lambda profile: {"devices": 2, "confirmed_at": T0 - 3600.0})
    kwargs.setdefault("rules_probe", lambda source: [Rule("alpha-etb", max_price=6000)])
    kwargs.setdefault("budget_probe", lambda: {"remaining_cents": 20000})
    return SnipeController(sched, plan, **kwargs)


PRODUCT = Product("alpha-etb", "Alpha Elite Trainer Box", "AA01",
                  ProductKind.ELITE_TRAINER_BOX, 4999, "2024-01-01")


def buy(at: float, **over: Any) -> Verdict:
    kwargs: Dict[str, Any] = dict(
        product_id="alpha-etb", action=Action.BUY, at=at, source="alpha",
        sku="A-1", price=4499, landed=4499, market=5200, quantity=1,
        url="https://alphamart.example.com/p/alpha-etb",
        reasons=("in stock", "under the cap"),
    )
    kwargs.update(over)
    return Verdict(**kwargs)


class FakeService:
    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    def publish(self, profile_id, kind, title, body, data=None, priority=None,
                dedupe_key=None):
        self.sent.append({"profile_id": profile_id, "kind": kind, "title": title,
                          "body": body, "data": data, "priority": priority,
                          "dedupe_key": dedupe_key})
        return f"a{len(self.sent)}"


# ==========================================================================
# rail 1: the 30s floor
# ==========================================================================


@pytest.mark.parametrize("interval", [0.0, 1.0, 29.999, -30.0])
def test_window_refuses_an_interval_under_the_floor(interval):
    with pytest.raises(ValueError):
        window(interval_s=interval)


def test_window_floor_is_the_fetchpolicy_floor_not_a_second_copy():
    # The window builds a FetchPolicy to validate, so the two can never
    # drift apart. If FetchPolicy accepts it, so does the window.
    assert window(interval_s=30.0).interval_s == 30.0
    with pytest.raises(ValueError, match="30s"):
        window(interval_s=29.0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_window_refuses_non_finite_numbers(bad):
    with pytest.raises(SnipeError):
        window(interval_s=bad)
    with pytest.raises(SnipeError):
        window(opens_at=bad)
    with pytest.raises(SnipeError):
        window(ttl_s=bad)


def test_window_refuses_zero_or_negative_ttl():
    with pytest.raises(SnipeError, match="ttl_s"):
        window(ttl_s=0.0)


def test_window_refuses_closing_before_it_opens():
    with pytest.raises(SnipeError, match="closes at or before"):
        window(closes_at=T0)


# ==========================================================================
# rail 2: a window is bounded
# ==========================================================================


def test_window_longer_than_the_cap_is_refused():
    with pytest.raises(SnipeError, match="cap is"):
        window(closes_at=T0 + 3600.0 + MAX_WINDOW_S + 1.0)


def test_a_days_worth_of_windows_over_the_cap_is_refused():
    hour = 3600.0
    windows = [
        window(name=f"w{i}", opens_at=T0 + i * 2 * hour, closes_at=T0 + i * 2 * hour + hour)
        for i in range(8)                      # 8 hours of tight polling
    ]
    with pytest.raises(SnipeError, match="cap is"):
        SnipePlan.of(windows)


def test_overlapping_windows_are_counted_once_against_the_daily_cap():
    # Eight wholly overlapping one-hour windows are one hour of polling,
    # not eight. Their durations sum to well past the six-hour cap, so a
    # plan that summed instead of taking the union would refuse this --
    # which is the bug this test exists to catch. Eight, not five,
    # precisely so the sum clears the cap and the test can fail.
    hour = 3600.0
    count = int(MAX_OPEN_PER_DAY_S // hour) + 2
    windows = [
        window(name=f"w{i}", opens_at=T0, closes_at=T0 + hour, interval_s=30.0 + i)
        for i in range(count)
    ]
    plan = SnipePlan.of(windows)
    assert len(plan.windows) == count


def test_a_window_spanning_midnight_is_charged_to_both_days():
    # 23:30 to 00:30 UTC: half an hour on each day, and neither day is
    # over the cap, so it stands.
    midnight = _dt.datetime(2026, 3, 2, tzinfo=_dt.timezone.utc).timestamp()
    plan = SnipePlan.of(window(opens_at=midnight - 1800.0, closes_at=midnight + 1800.0))
    assert plan.windows[0].duration_s == 3600.0
    # ...and a day already near the cap plus the overspill tips it over.
    day_before = [
        window(name=f"d{i}", opens_at=midnight - (i + 2) * 3600.0,
               closes_at=midnight - (i + 1) * 3600.0)
        for i in range(int(MAX_OPEN_PER_DAY_S // 3600))
    ]
    with pytest.raises(SnipeError, match="cap is"):
        SnipePlan.of(day_before + [window(opens_at=midnight - 1800.0,
                                          closes_at=midnight + 1800.0)])


def test_two_windows_with_the_same_identity_are_refused():
    with pytest.raises(SnipeError, match="share an identity"):
        SnipePlan.of(window(), window())


# ==========================================================================
# rail 3: a pause outranks a window (against the real scheduler)
# ==========================================================================


def _real_scheduler():
    clock = SimClock()
    catalog = make_catalog()
    sched = PollScheduler(catalog, make_policies(), clock)
    return sched, clock, catalog


def test_a_window_cannot_shorten_a_host_requested_pause():
    sched, clock, catalog = _real_scheduler()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    sched.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), T0)
    # The host asked for an hour.
    sched.pause_source("alpha", T0 + 3600.0, "retry-after 3600s")
    before = sched.pause_state(T0)["alpha"]["paused_until"]

    plan = SnipePlan.of(window(source="alpha", opens_at=T0 + 60.0,
                               closes_at=T0 + 60.0 + 1800.0))
    ctl = SnipeController(sched, plan)
    ctl.sync(T0 + 61.0)

    after = sched.pause_state(T0 + 61.0)["alpha"]["paused_until"]
    assert after == before, "retime moved a pause; a window must never do that"
    allowed, reason = sched.can_poll(sku, T0 + 61.0)
    assert not allowed and "pause" in reason.lower()
    # And the window really is open and really did tighten the rate --
    # otherwise this test would pass for the wrong reason.
    assert sched.policy("alpha").min_interval_s == 30.0


def test_retime_pulls_the_gate_in_when_a_window_opens():
    sched, clock, catalog = _real_scheduler()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    sched.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), T0)
    assert not sched.can_poll(sku, T0 + 60.0)[0]        # 300s policy: not due

    sched.set_policy(replace(sched.policy("alpha"), min_interval_s=30.0))
    assert not sched.can_poll(sku, T0 + 60.0)[0], (
        "set_policy alone must not move a gate already on the books -- if it "
        "did, retime_source would be pointless and this test is stale"
    )
    moved = sched.retime_source("alpha", T0 + 60.0)
    assert moved >= 1
    assert sched.can_poll(sku, T0 + 60.0)[0]


def test_retime_pushes_the_gate_out_when_a_window_closes():
    sched, clock, catalog = _real_scheduler()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    sched.set_policy(replace(sched.policy("alpha"), min_interval_s=30.0))
    sched.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), T0)
    assert sched.can_poll(sku, T0 + 40.0)[0]

    sched.set_policy(replace(sched.policy("alpha"), min_interval_s=600.0))
    sched.retime_source("alpha", T0 + 40.0)
    assert not sched.can_poll(sku, T0 + 40.0)[0]
    assert sched.can_poll(sku, T0 + 601.0)[0]


def test_retime_never_produces_a_gate_inside_the_floor():
    sched, clock, catalog = _real_scheduler()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    sched.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), T0)
    sched.set_policy(replace(sched.policy("alpha"), min_interval_s=30.0))
    sched.retime_source("alpha", T0 + 1.0)
    assert sched.effective_due_at(sku) >= T0 + 30.0
    assert not sched.can_poll(sku, T0 + 29.0)[0]


def test_retime_leaves_a_listing_never_polled_alone():
    sched, clock, catalog = _real_scheduler()
    sku = sku_of(catalog, "alpha", "beta-box")
    assert sched.can_poll(sku, T0)[0]
    sched.set_policy(replace(sched.policy("alpha"), min_interval_s=600.0))
    sched.retime_source("alpha", T0)
    assert sched.can_poll(sku, T0)[0], (
        "a listing we have never fetched has an open gate; retiming it to "
        "0 + interval would invent a delay out of nothing"
    )


def test_retime_refuses_an_unknown_source_and_a_nan_now():
    sched, _clock, _catalog = _real_scheduler()
    with pytest.raises(SchedulerError):
        sched.retime_source("nosuch", T0)
    with pytest.raises(SchedulerError):
        sched.retime_source("alpha", float("nan"))


def test_the_whole_thing_end_to_end_against_the_real_scheduler():
    sched, clock, catalog = _real_scheduler()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    open_at = T0 + 1800.0
    plan = SnipePlan.of(window(source="alpha", opens_at=open_at,
                               closes_at=open_at + 900.0, interval_s=30.0))
    ctl = SnipeController(sched, plan, alert_probe=lambda p: {"devices": 1,
                                                              "confirmed_at": T0})
    polls = 0
    moment = T0
    # 75 minutes at one tick a minute: before, during and after.
    while moment <= open_at + 1800.0:
        ctl.sync(moment)
        if sched.can_poll(sku, moment)[0]:
            sched.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), moment)
            polls += 1
        moment += 60.0
    assert sched.policy("alpha").min_interval_s == 300.0, "window did not close"
    # During the 900s window at a 60s tick we get roughly one poll a tick;
    # outside it, one per five. The point is that it is visibly more.
    assert polls >= 15, polls


# ==========================================================================
# the controller: opening, closing, precedence
# ==========================================================================


def test_sync_opens_and_closes_and_restores_the_exact_base_policy():
    sched = FakeScheduler({"alpha": FetchPolicy("alpha", min_interval_s=300.0,
                                                user_agent="Custom/9",
                                                max_errors_before_pause=2)})
    base = sched.policy("alpha")
    ctl = controller(sched)
    assert ctl.sync(T0) == []
    events = ctl.sync(T0 + 3601.0)
    assert [e.kind for e in events] == [EventKind.OPENED]
    live = sched.policy("alpha")
    assert live.min_interval_s == 30.0
    # A window changes the rate and nothing else.
    assert live.user_agent == base.user_agent
    assert live.max_errors_before_pause == base.max_errors_before_pause
    events = ctl.sync(T0 + 3600.0 + 1801.0)
    assert [e.kind for e in events] == [EventKind.CLOSED]
    assert sched.policy("alpha") == base


def test_sync_is_idempotent_inside_a_window():
    sched = FakeScheduler()
    ctl = controller(sched)
    ctl.sync(T0 + 3601.0)
    assert ctl.sync(T0 + 3602.0) == []
    assert ctl.sync(T0 + 3700.0) == []
    assert len(sched.retimes) == 1, "a policy re-installed every tick is a write storm"


def test_the_tightest_overlapping_window_wins_and_ties_break_on_name():
    wide = window(name="release-day", opens_at=T0, closes_at=T0 + 3600.0, interval_s=60.0)
    tight = window(name="the-minute", opens_at=T0 + 1800.0, closes_at=T0 + 1860.0,
                   interval_s=30.0)
    plan = SnipePlan.of(wide, tight)
    assert plan.open_at("alpha", T0 + 10.0).name == "release-day"
    assert plan.open_at("alpha", T0 + 1810.0).name == "the-minute"
    assert plan.open_at("alpha", T0 + 1900.0).name == "release-day"
    assert plan.open_at("alpha", T0 + 7200.0) is None

    a = window(name="aaa", opens_at=T0, closes_at=T0 + 600.0, interval_s=45.0)
    b = window(name="bbb", opens_at=T0 + 1.0, closes_at=T0 + 600.0, interval_s=45.0)
    assert SnipePlan.of(a, b).open_at("alpha", T0 + 100.0).name == "aaa"


def test_moving_between_two_overlapping_windows_re_tightens():
    wide = window(name="release-day", opens_at=T0, closes_at=T0 + 3600.0, interval_s=60.0)
    tight = window(name="the-minute", opens_at=T0 + 1800.0, closes_at=T0 + 1860.0,
                   interval_s=30.0)
    sched = FakeScheduler()
    ctl = controller(sched, SnipePlan.of(wide, tight))
    ctl.sync(T0 + 10.0)
    assert sched.policy("alpha").min_interval_s == 60.0
    ctl.sync(T0 + 1810.0)
    assert sched.policy("alpha").min_interval_s == 30.0
    ctl.sync(T0 + 1870.0)
    assert sched.policy("alpha").min_interval_s == 60.0, "fell all the way back"
    ctl.sync(T0 + 3700.0)
    assert sched.policy("alpha").min_interval_s == 300.0


def test_a_policy_installed_elsewhere_is_adopted_not_clobbered():
    # The app re-checks robots.txt mid-window and installs a new policy.
    # Closing the window must not undo that.
    sched = FakeScheduler()
    ctl = controller(sched)
    ctl.sync(T0 + 3601.0)
    assert sched.policy("alpha").min_interval_s == 30.0
    sched.set_policy(FetchPolicy("alpha", min_interval_s=900.0, user_agent="NewUA/2"))
    events = ctl.sync(T0 + 3602.0)
    assert [e.kind for e in events] == [EventKind.REBASED]
    assert sched.policy("alpha").min_interval_s == 30.0, "the window still rules the rate"
    assert sched.policy("alpha").user_agent == "NewUA/2"
    ctl.sync(T0 + 3600.0 + 1801.0)
    restored = sched.policy("alpha")
    assert restored.min_interval_s == 900.0, "closing restored a stale base"
    assert restored.user_agent == "NewUA/2"


def test_a_window_for_an_unknown_source_is_refused_at_construction():
    with pytest.raises(SchedulerError):
        SnipeController(FakeScheduler(), SnipePlan.of(window(source="nosuch")))


def test_arm_records_are_forgotten_once_a_window_is_over():
    sched = FakeScheduler()
    ctl = controller(sched)
    ctl.sync(T0 + 3100.0)               # arming
    assert ctl._armed
    ctl.sync(T0 + 3600.0 + 1801.0)      # past the close
    assert not ctl._armed, "a month of windows would grow this list for ever"


def test_restore_re_asserts_rather_than_trusting_the_file():
    sched = FakeScheduler()
    ctl = controller(sched)
    ctl.sync(T0 + 3601.0)
    snap = ctl.snapshot()
    # A restart: a fresh controller over a scheduler still holding the
    # tightened policy, and a clock now past the close.
    sched2 = FakeScheduler()
    sched2.set_policy(FetchPolicy("alpha", min_interval_s=30.0))
    ctl2 = SnipeController(sched2, SnipePlan.of(window()))
    events = ctl2.restore(snap, T0 + 3600.0 + 1801.0)
    assert [e.kind for e in events] == [EventKind.CLOSED]
    assert sched2.policy("alpha").min_interval_s == 300.0, (
        "a restart left the host being polled every 30s with nobody watching"
    )


def test_restore_refuses_a_newer_state_version():
    with pytest.raises(SnipeError, match="newer"):
        controller().restore({"version": 99}, T0)


def test_next_change_at_lets_a_supervisor_sleep():
    plan = SnipePlan.of(window())
    w = plan.windows[0]
    assert plan.next_change_at(T0) == w.arms_at
    assert plan.next_change_at(w.arms_at) == w.opens_at
    assert plan.next_change_at(w.opens_at) == w.closes_at
    assert plan.next_change_at(w.closes_at) is None


def test_phases():
    w = window()
    assert w.phase(T0) is Phase.IDLE
    assert w.phase(w.arms_at) is Phase.ARMING
    assert w.phase(w.opens_at) is Phase.OPEN
    assert w.phase(w.closes_at) is Phase.CLOSED


# ==========================================================================
# the preflight: the part that decides whether the phone rings
# ==========================================================================


def _report(now: Optional[float] = None, **kwargs: Any):
    ctl = controller(**kwargs)
    w = window()
    return ctl.preflight(w, now if now is not None else w.arms_at)


def _check(report, name: str) -> ArmCheck:
    found = [c for c in report.checks if c.name == name]
    assert found, f"no {name} check in {[c.name for c in report.checks]}"
    return found[0]


def test_a_healthy_preflight_is_ready_and_says_so():
    report = _report()
    assert report.ready
    assert not report.unknowns
    assert "armed, every check green" in report.summary()


def test_no_push_subscription_blocks_the_window():
    report = _report(alert_probe=lambda p: {"devices": 0, "confirmed_at": None})
    assert not report.ready
    assert "never delivered" in _check(report, "alert_path").detail
    assert "NOT READY" in report.summary()


def test_a_push_subscription_nobody_has_confirmed_in_weeks_blocks():
    stale = T0 - ALERT_PATH_MAX_AGE_S - 1.0
    report = _report(alert_probe=lambda p: {"devices": 1, "confirmed_at": stale})
    assert not report.ready
    assert "send a test" in _check(report, "alert_path").detail


def test_a_subscription_with_no_confirmation_on_record_is_unknown_not_fine():
    report = _report(alert_probe=lambda p: {"devices": 1, "confirmed_at": None})
    check = _check(report, "alert_path")
    assert check.unknown and report.ready
    assert "unverified" in report.summary()


def test_an_unwired_alert_probe_reports_unknown_and_says_to_wire_it():
    report = _report(alert_probe=None)
    check = _check(report, "alert_path")
    assert check.unknown, "an unwired probe must never read as green"
    assert "most worth wiring" in check.detail


def test_a_probe_that_raises_blocks_and_leaks_nothing():
    def boom(profile):
        raise RuntimeError("endpoint https://push.example/secret-token-abc123")

    report = _report(alert_probe=boom)
    detail = _check(report, "alert_path").detail
    assert not report.ready
    assert "secret-token-abc123" not in detail and "RuntimeError" in detail


def test_robots_disallow_blocks_the_window():
    sched = FakeScheduler({"alpha": FetchPolicy("alpha", min_interval_s=300.0,
                                                robots_allows=False)})
    ctl = controller(sched)
    report = ctl.preflight(window(), window().arms_at)
    assert not report.ready
    assert "robots.txt" in _check(report, "policy").detail


def test_a_pause_lasting_past_the_open_blocks_the_window():
    sched = FakeScheduler()
    w = window()
    sched.paused["alpha"] = {"paused_until": w.opens_at + 60.0, "reason": "retry-after"}
    report = controller(sched).preflight(w, w.arms_at)
    assert not report.ready
    assert "outranks" in _check(report, "pause").detail


def test_a_pause_that_expires_before_the_open_does_not_block():
    sched = FakeScheduler()
    w = window()
    sched.paused["alpha"] = {"paused_until": w.opens_at - 60.0, "reason": "retry-after"}
    assert controller(sched).preflight(w, w.arms_at).ready


def test_no_enabled_rule_blocks_the_window():
    report = _report(rules_probe=lambda s: [Rule("alpha-etb", max_price=6000, enabled=False)])
    assert not report.ready
    assert "never raise a thing" in _check(report, "rules").detail


def test_a_rule_scoped_to_another_source_does_not_count():
    report = _report(
        rules_probe=lambda s: [Rule("alpha-etb", max_price=6000,
                                    allowed_sources=("bravo",))]
    )
    assert not report.ready


def test_an_empty_budget_blocks_the_window():
    report = _report(budget_probe=lambda: {"remaining_cents": 0})
    assert not report.ready
    assert "nothing left" in _check(report, "budget").detail


def test_a_budget_below_the_cheapest_cap_blocks_the_window():
    report = _report(budget_probe=lambda: {"remaining_cents": 1000})
    assert not report.ready
    assert "affordable" in _check(report, "budget").detail


def test_sync_runs_the_preflight_once_at_arm_time_and_reports_it():
    sched = FakeScheduler()
    ctl = controller(sched, alert_probe=lambda p: {"devices": 0, "confirmed_at": None})
    assert ctl.sync(T0) == []
    events = ctl.sync(window().arms_at + 1.0)
    assert [e.kind for e in events] == [EventKind.NOT_READY]
    assert events[0].wants_attention
    assert events[0].report is not None and not events[0].report.ready
    assert ctl.sync(window().arms_at + 2.0) == [], "preflight ran twice"


def test_a_failed_preflight_still_opens_the_window():
    # A broken alert path is a reason to shout, not a reason to stop
    # watching: the owner may fix it while the window is open.
    sched = FakeScheduler()
    ctl = controller(sched, alert_probe=lambda p: {"devices": 0, "confirmed_at": None})
    ctl.sync(window().arms_at + 1.0)
    ctl.sync(T0 + 3601.0)
    assert sched.policy("alpha").min_interval_s == 30.0


# ==========================================================================
# daily_windows
# ==========================================================================


def test_daily_windows_are_utc_and_deterministic():
    made = daily_windows("weekly-restock", "alpha", first_day="2026-04-06",
                         at_utc="11:00", duration_s=1800.0, days=3)
    assert [w.name for w in made] == [
        "weekly-restock-2026-04-06", "weekly-restock-2026-04-07",
        "weekly-restock-2026-04-08",
    ]
    first = _dt.datetime.fromtimestamp(made[0].opens_at, tz=_dt.timezone.utc)
    assert (first.hour, first.minute) == (11, 0)
    assert made[1].opens_at - made[0].opens_at == 86400.0
    again = daily_windows("weekly-restock", "alpha", first_day="2026-04-06",
                          at_utc="11:00", duration_s=1800.0, days=3)
    assert made == again


def test_daily_windows_over_a_week_still_respects_the_daily_cap():
    made = daily_windows("r", "alpha", first_day="2026-04-06", at_utc="11:00",
                         duration_s=1800.0, days=7)
    assert len(SnipePlan.of(made).windows) == 7
    too_much = daily_windows("r", "alpha", first_day="2026-04-06", at_utc="11:00",
                             duration_s=MAX_WINDOW_S, days=7)
    # 2h a day is under the 6h cap, so this stands; 4 such series would not.
    assert len(SnipePlan.of(too_much).windows) == 7


@pytest.mark.parametrize("bad", [
    {"first_day": "06-04-2026"}, {"at_utc": "25:00"}, {"at_utc": "eleven"},
    {"days": 0}, {"days": 900},
])
def test_daily_windows_refuses_nonsense(bad):
    kwargs = dict(first_day="2026-04-06", at_utc="11:00", duration_s=1800.0, days=2)
    kwargs.update(bad)
    with pytest.raises(SnipeError):
        daily_windows("r", "alpha", **kwargs)


# ==========================================================================
# the alert
# ==========================================================================


def _bridge(now: List[float], **kwargs: Any):
    ctl = controller(**kwargs)
    return SnipeAlertBridge(FakeService(), lambda: now[0], ctl), ctl


def test_a_snipe_alert_carries_exactly_the_snipe_keys():
    now = [T0 + 3601.0]
    bridge, _ = _bridge(now)
    bridge.publish_verdict(buy(at=now[0] - 4.0), PRODUCT)
    sent = bridge.service.sent[0]
    assert sent["kind"] == SNIPE_KIND
    assert sorted(sent["data"]) == sorted(SNIPE_DATA_KEYS)
    assert sent["data"]["window"] == "restock"
    assert sent["data"]["latency_ms"] == 4000
    assert sent["data"]["seen_at"] == int(now[0] - 4.0)
    assert sent["data"]["expires_at"] > now[0]
    assert sent["title"].startswith("DROP:")
    assert "Seen 4s ago" in sent["body"]


def test_outside_a_window_it_is_an_ordinary_buy_alert_with_the_default_ttl():
    now = [T0]
    bridge, _ = _bridge(now)
    bridge.publish_verdict(buy(at=T0), PRODUCT)
    data = bridge.service.sent[0]["data"]
    assert data["window"] is None
    assert data["expires_at"] == int(T0 + DEFAULT_TTL_S)
    assert bridge.service.sent[0]["title"].startswith("Buy ")


def test_a_verdict_stamped_in_the_future_is_zero_latency_not_negative():
    now = [T0 + 3601.0]
    bridge, _ = _bridge(now)
    bridge.publish_verdict(buy(at=now[0] + 30.0), PRODUCT)
    assert bridge.service.sent[0]["data"]["latency_ms"] == 0


def test_the_snipe_bridge_still_refuses_everything_the_buy_bridge_refuses():
    now = [T0 + 3601.0]
    bridge, _ = _bridge(now)
    assert bridge.publish_verdict(buy(at=T0, action=Action.WATCH), PRODUCT) is None
    with pytest.raises(BridgeError):
        bridge.publish_verdict(buy(at=T0, price=None, landed=None), PRODUCT)
    with pytest.raises(BridgeError):
        bridge.publish_verdict(buy(at=T0, product_id="other"), PRODUCT)


def test_the_snipe_bridge_still_dedupes_a_flapping_listing():
    now = [T0 + 3601.0]
    bridge, _ = _bridge(now)
    first = bridge.publish_verdict(buy(at=now[0]), PRODUCT)
    now[0] += 20.0
    again = bridge.publish_verdict(buy(at=now[0]), PRODUCT)
    assert first == again and len(bridge.service.sent) == 1
    assert bridge.suppressed == 1


def test_latency_is_tracked_and_bounded():
    now = [T0 + 3601.0]
    bridge, _ = _bridge(now)
    for i in range(500):
        now[0] += 400.0            # past the dedupe window each time
        bridge.publish_verdict(buy(at=now[0] - (i % 7), landed=4000 + i, price=4000 + i),
                               PRODUCT)
    assert len(bridge.latencies) == 200, "unbounded latency history"
    assert bridge.median_latency_s is not None and bridge.median_latency_s >= 0.0


def test_ttl_for_uses_the_window_when_there_is_one():
    ctl = controller(plan=SnipePlan.of(window(ttl_s=120.0)))
    assert ctl.ttl_for("alpha", T0) == DEFAULT_TTL_S
    assert ctl.ttl_for("alpha", T0 + 3601.0) == 120.0


def test_expires_at_never_outlives_the_window_by_more_than_the_ttl():
    w = window(ttl_s=600.0)
    assert w.expires_at(w.opens_at) == w.opens_at + 600.0
    assert w.expires_at(w.closes_at + 10_000.0) == w.closes_at + 600.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


# ==========================================================================
# the bot: does the window actually make it watch harder?
# ==========================================================================
#
# The two tests above this point prove the scheduler will *allow* a faster
# poll inside a window. That is only half of it: a bot that asks once every
# five minutes gets one poll every five minutes however willing the
# scheduler is. These prove the other half.

from tests.test_poke_bot import (  # noqa: E402
    SHOP_A,
    SHOP_B,
    Rig,
    rule,
)
from jarvis_bots.contracts import Severity as BotSeverity  # noqa: E402

BOT_T0 = 1_700_000_000.0


def _plan_for(open_at: float, **over: Any) -> SnipePlan:
    kwargs: Dict[str, Any] = dict(
        name="restock", source=SHOP_A, opens_at=open_at,
        closes_at=open_at + 900.0, interval_s=30.0,
    )
    kwargs.update(over)
    return SnipePlan.of(DropWindow(**kwargs))


def _rig(open_at: float, **probes: Any) -> Rig:
    plan = _plan_for(open_at)
    probes.setdefault("alert_probe", lambda p: {"devices": 1, "confirmed_at": BOT_T0})
    return Rig(
        rules=[rule("sv08-surging-sparks-etb", 6000)],
        snipe=lambda sched: SnipeController(sched, plan, **probes),
    )


def test_the_bot_tightens_its_own_tick_rate_for_a_window():
    # Arm time, not open time: the supervisor reads bot.info *before* the
    # tick, so a tighten at open time lands a whole round late.
    rig = _rig(BOT_T0 + 3600.0)
    assert rig.bot.info.interval_s == 300.0
    rig.tick(advance=100.0)
    assert rig.bot.info.interval_s == 300.0
    rig.clock.advance(2950.0)                 # inside pre_arm_s
    rig.bot.tick(rig.clock.at)
    assert rig.bot.info.interval_s == 30.0, (
        "the scheduler would allow a 30s poll but the bot only asks every "
        "300s, so the window buys nothing"
    )
    rig.clock.advance(4000.0)                 # past the close
    rig.bot.tick(rig.clock.at)
    assert rig.bot.info.interval_s == 300.0


def test_a_window_really_does_produce_more_polls_through_the_whole_stack():
    opens_at, closes_at = BOT_T0 + 1800.0, BOT_T0 + 2700.0
    rig = _rig(opens_at)
    polls = []
    moment = rig.clock.at
    # Tick at whatever rate the bot currently asks for, the way the
    # supervisor would, for two hours across the 15-minute window.
    end = moment + 7200.0
    while moment < end:
        before = rig.bot._polls
        rig.bot.tick(moment)
        polls.append((moment, rig.bot._polls - before))
        moment += rig.bot.info.interval_s
        rig.clock.at = moment
    inside = sum(n for at, n in polls if opens_at <= at < closes_at)
    after = sum(n for at, n in polls if at >= closes_at)
    # Rates, not totals: the window is 15 minutes and the quiet tail is
    # 75, so raw counts would flatter the slow period.
    inside_rate = inside / (closes_at - opens_at)
    after_rate = after / (end - closes_at)
    assert inside >= 10, inside
    assert inside_rate > 5 * after_rate, (inside_rate, after_rate)
    # ...and the quiet period is still being watched, just politely.
    assert after >= 1


def test_a_not_ready_preflight_reaches_the_badge_and_clears_when_fixed():
    devices = [0]
    rig = _rig(BOT_T0 + 3600.0,
               alert_probe=lambda p: {"devices": devices[0], "confirmed_at": BOT_T0})
    rig.clock.advance(3050.0)
    events = list(rig.bot.tick(rig.clock.at))
    bad = [e for e in events if e.severity is BotSeverity.ACTION]
    assert len(bad) == 1
    assert bad[0].attention_key == "snipe:not-ready:restock"
    assert "NOT READY" in bad[0].text or "not ready" in bad[0].text.lower()
    # The owner re-subscribes the phone; a later window arms clean.
    devices[0] = 2
    plan2 = _plan_for(BOT_T0 + 10800.0, name="restock2")
    rig.bot._snipe.plan = plan2
    rig.bot._snipe._armed = []
    rig.clock.advance(7200.0)
    events = list(rig.bot.tick(rig.clock.at))
    assert not [e for e in events if e.severity is BotSeverity.ACTION]


def test_a_broken_controller_never_stops_the_round():
    class Exploding:
        plan = None

        def sync(self, now):
            raise RuntimeError("https://push.example/secret-endpoint")

        def active_window(self, source, now):
            return None

    rig = Rig(rules=[rule("sv08-surging-sparks-etb", 6000)], snipe=Exploding())
    events = list(rig.tick())
    errors = [e for e in events if e.severity is BotSeverity.ERROR]
    assert len(errors) == 1
    assert "secret-endpoint" not in errors[0].text
    assert "RuntimeError" in errors[0].text
    # The point of catching it here rather than letting PokeBot.tick's own
    # handler take it: the round carries on. A drop-window controller that
    # throws must not cost the bot the polling it exists to do.
    assert rig.bot._polls >= 1, (
        "the tick aborted; the bot stopped watching because a scheduling "
        "helper threw"
    )
    assert rig.bot.status().state.name in {"RUNNING", "IDLE"}


def test_a_bot_with_no_snipe_controller_behaves_exactly_as_before():
    plain = Rig(rules=[rule("sv08-surging-sparks-etb", 6000)])
    assert plain.bot._snipe is None
    assert plain.bot.info.interval_s == 300.0
    plain.tick()
    assert plain.bot.info.interval_s == 300.0


def test_the_bot_refuses_a_mis_wired_snipe_controller():
    with pytest.raises(Exception, match="sync"):
        Rig(rules=[rule("sv08-surging-sparks-etb", 6000)], snipe=object())
