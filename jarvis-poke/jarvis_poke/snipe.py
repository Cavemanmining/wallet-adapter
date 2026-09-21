"""Drop windows: watch harder when it matters, and prove the phone will ring.

A "snipe" is not a faster scraper.  The tool already polls on a schedule
and already alerts on a BUY; what it lacks is the two things that decide
whether the owner actually gets a limited product:

1. **Being early.**  A restock at 11:00 that we look at every five minutes
   is, on average, two and a half minutes of somebody else's head start.
   Inside a known drop window the interval tightens to the floor and the
   average wait becomes fifteen seconds.
2. **The notification working.**  The commonest way a snipe is missed is
   not a slow poll.  It is an expired push subscription, a source paused
   by a ``Retry-After`` nobody noticed, a rule that was disabled last
   month, or a budget with nothing left in it -- each of which is silent
   until the moment it matters.  So a window *arms* before it opens:
   :meth:`SnipeController.preflight` runs the checks ahead of time, while
   there is still time to fix them.

What this module does not do
----------------------------
It does not check out.  ``jarvis_poke.contracts``: "It does not check
out."  The terminal output is still a verdict, an alert and a deep link
the owner taps.  This module makes that link arrive sooner and makes it
far likelier to arrive at all; it does not press the button.

Nor does it get around anyone's rate limit.  Three rails, all in code:

* **The 30s floor.**  A window's interval is a
  :class:`~jarvis_poke.contracts.FetchPolicy` interval and gets that
  class's own validation, which refuses anything under 30 seconds.  The
  tightest a window can be is the politest thing the package would ever
  have done anyway; a window only decides *when* to be that attentive.
* **A window is bounded.**  :data:`MAX_WINDOW_S` caps one window and
  :data:`MAX_OPEN_PER_DAY_S` caps the union of them per source per day.
  "Snipe all day" is just fast polling with a nicer name, and
  :class:`SnipePlan` refuses it at construction.
* **A pause outranks a window.**  Tightening is done by moving the
  interval gate, never the pause: if a host sent ``Retry-After: 86400``
  we wait the day, window or no window.  ``PollScheduler.retime_source``
  keeps the two apart, and ``test_snipe.py`` proves it by opening a
  window over a paused host and watching nothing get polled.

Determinism, as everywhere in this package: no clocks of its own, no
network, no randomness.  ``now`` is passed in, the scheduler and the alert
service are injected, and the probes that answer "is the phone reachable"
are the app's to supply.
"""

from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import (
    Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple,
)

from jarvis_alerts.contracts import Priority

from jarvis_poke.alerts_bridge import DATA_KEYS as BUY_DATA_KEYS
from jarvis_poke.alerts_bridge import AlertBridge, BridgeError, landed_cents
from jarvis_poke.contracts import FetchPolicy, Product, Rule, Verdict, fmt_cents

__all__ = [
    "ALERT_PATH_MAX_AGE_S",
    "DEFAULT_PRE_ARM_S",
    "DEFAULT_TTL_S",
    "MAX_OPEN_PER_DAY_S",
    "MAX_WINDOW_S",
    "MIN_WINDOW_INTERVAL_S",
    "SNIPE_DATA_KEYS",
    "SNIPE_KIND",
    "ArmCheck",
    "ArmReport",
    "DropWindow",
    "EventKind",
    "Phase",
    "SnipeAlertBridge",
    "SnipeController",
    "SnipeError",
    "SnipeEvent",
    "SnipePlan",
    "daily_windows",
]

#: The tightest a window may poll.  Not a preference: ``FetchPolicy``
#: raises below this and every window interval is validated through it.
MIN_WINDOW_INTERVAL_S = 30.0

#: One window may not run longer than two hours.  A drop is an event.
MAX_WINDOW_S = 2 * 60 * 60.0

#: Nor may a source's windows cover more than six hours of any UTC day,
#: measured as the union of their open intervals (overlaps counted once).
MAX_OPEN_PER_DAY_S = 6 * 60 * 60.0

#: How far ahead of a window the preflight runs, by default: long enough
#: to re-subscribe a phone or top up a budget before the drop.
DEFAULT_PRE_ARM_S = 10 * 60.0

#: How long a snipe alert is worth showing.  Past this the listing is
#: almost certainly gone and a buzz is just noise; the payload carries
#: ``expires_at`` so the service worker can decide.
DEFAULT_TTL_S = 15 * 60.0

#: A push subscription confirmed longer ago than this is treated as
#: unproven.  Web Push endpoints rotate; an untested one is a silent
#: failure waiting for the worst possible moment.
ALERT_PATH_MAX_AGE_S = 7 * 24 * 60 * 60.0

#: The alert ``kind`` a window-time buy carries, distinct from the normal
#: ``poke_buy`` so the client can treat it as time-critical.
SNIPE_KIND = "poke_snipe"

#: The buy payload plus the four keys a snipe adds.  Same flat-scalar
#: rule; :func:`jarvis_poke.alerts_bridge._check_payload` enforces both.
SNIPE_DATA_KEYS = BUY_DATA_KEYS + ("window", "expires_at", "seen_at", "latency_ms")

Clock = Callable[[], float]


class SnipeError(ValueError):
    """A window, plan or controller that cannot be trusted to behave."""


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SnipeError(f"{name} must be a number of seconds; got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise SnipeError(
            f"{name} must be finite: a NaN compares False against every bound, "
            f"which turns a rail into a no-op"
        )
    return number


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SnipeError(f"{name} must be a non-empty string; got {value!r}")
    return value


# --------------------------------------------------------------------------
# a window
# --------------------------------------------------------------------------


class Phase(Enum):
    """Where a window is relative to ``now``."""

    IDLE = "idle"        # too early even to check
    ARMING = "arming"    # preflight time: fix things now
    OPEN = "open"        # polling tight
    CLOSED = "closed"    # over


@dataclass(frozen=True)
class DropWindow:
    """A stretch of time worth watching one source harder.

    ``interval_s`` is the poll interval *while open*.  It is validated by
    building a :class:`~jarvis_poke.contracts.FetchPolicy`, so the 30s
    floor is the same floor the rest of the package obeys rather than a
    second copy of it that could drift.

    ``pre_arm_s`` is how long before ``opens_at`` the preflight runs;
    ``ttl_s`` is how long an alert raised in this window stays worth
    showing.
    """

    name: str
    source: str
    opens_at: float
    closes_at: float
    interval_s: float = MIN_WINDOW_INTERVAL_S
    pre_arm_s: float = DEFAULT_PRE_ARM_S
    ttl_s: float = DEFAULT_TTL_S
    note: str = ""

    def __post_init__(self) -> None:
        _text(self.name, "window name")
        _text(self.source, "window source")
        opens = _finite(self.opens_at, "opens_at")
        closes = _finite(self.closes_at, "closes_at")
        if closes <= opens:
            raise SnipeError(
                f"window {self.name!r} closes at or before it opens "
                f"({closes} <= {opens})"
            )
        if closes - opens > MAX_WINDOW_S:
            raise SnipeError(
                f"window {self.name!r} runs {(closes - opens) / 60:.0f} minutes; the "
                f"cap is {MAX_WINDOW_S / 60:.0f}. A window that long is not a drop, "
                f"it is a faster baseline -- raise min_interval_s instead"
            )
        interval = _finite(self.interval_s, "interval_s")
        # Delegate the floor to the one class that owns it.  A bad interval
        # fails here, at construction, rather than at 11:00 on drop day.
        FetchPolicy(source=self.source, min_interval_s=interval)
        pre_arm = _finite(self.pre_arm_s, "pre_arm_s")
        if pre_arm < 0.0:
            raise SnipeError("pre_arm_s must not be negative")
        if _finite(self.ttl_s, "ttl_s") <= 0.0:
            raise SnipeError("ttl_s must be positive: an alert nobody may show is a loss")

    # -- derived ----------------------------------------------------------

    @property
    def arms_at(self) -> float:
        return self.opens_at - self.pre_arm_s

    @property
    def duration_s(self) -> float:
        return self.closes_at - self.opens_at

    @property
    def key(self) -> str:
        """Identity across a restart: source, name and the exact open time.
        Tomorrow's 11:00 window is a different window from today's."""
        return f"{self.source}|{self.name}|{self.opens_at:.0f}"

    def phase(self, now: float) -> Phase:
        moment = _finite(now, "now")
        if moment >= self.closes_at:
            return Phase.CLOSED
        if moment >= self.opens_at:
            return Phase.OPEN
        if moment >= self.arms_at:
            return Phase.ARMING
        return Phase.IDLE

    def is_open(self, now: float) -> bool:
        return self.phase(now) is Phase.OPEN

    def policy_from(self, base: FetchPolicy) -> FetchPolicy:
        """The base policy with this window's interval.

        Everything else -- the user agent, the robots verdict, the error
        budget -- is the app's and is carried through untouched.  A window
        changes the rate and nothing else.
        """
        if not isinstance(base, FetchPolicy):
            raise SnipeError(f"not a FetchPolicy: {base!r}")
        if base.source != self.source:
            raise SnipeError(
                f"window {self.name!r} is for {self.source!r} but the base policy is "
                f"for {base.source!r}"
            )
        return replace(base, min_interval_s=self.interval_s)

    def expires_at(self, at: float) -> float:
        """When an alert raised at ``at`` stops being worth showing:
        ``ttl_s`` later, but never past the window's own close plus the
        ttl -- a snipe for a drop that ended is not a live snipe."""
        return min(_finite(at, "at") + self.ttl_s, self.closes_at + self.ttl_s)


def daily_windows(
    name: str,
    source: str,
    *,
    first_day: str,
    at_utc: str,
    duration_s: float,
    days: int = 1,
    **kwargs: Any,
) -> Tuple[DropWindow, ...]:
    """Expand "11:00 UTC every day for a week" into absolute windows.

    ``first_day`` is ``YYYY-MM-DD`` and ``at_utc`` is ``HH:MM``, both UTC,
    both explicit: a recurring window stored as "11:00 local" moves an hour
    twice a year and silently misses two drops.  Each expanded window is
    named ``<name>-<date>`` so the key is stable and a report can name the
    day.  The windows are returned, not scheduled; hand them to
    :class:`SnipePlan`, which is where the per-day cap is applied.
    """
    _text(name, "name")
    _text(source, "source")
    if not isinstance(days, int) or isinstance(days, bool) or days < 1:
        raise SnipeError("days must be a positive integer")
    if days > 60:
        raise SnipeError("days is capped at 60; re-plan rather than schedule a quarter")
    try:
        day = _dt.date.fromisoformat(_text(first_day, "first_day"))
        hour, minute = (int(part) for part in _text(at_utc, "at_utc").split(":", 1))
    except ValueError as exc:
        raise SnipeError(
            f"first_day must be YYYY-MM-DD and at_utc must be HH:MM (UTC): {exc}"
        ) from None
    if not (0 <= hour < 24 and 0 <= minute < 60):
        raise SnipeError(f"at_utc out of range: {at_utc!r}")
    length = _finite(duration_s, "duration_s")
    out: List[DropWindow] = []
    for offset in range(days):
        start = _dt.datetime.combine(
            day + _dt.timedelta(days=offset),
            _dt.time(hour, minute),
            tzinfo=_dt.timezone.utc,
        ).timestamp()
        out.append(
            DropWindow(
                name=f"{name}-{(day + _dt.timedelta(days=offset)).isoformat()}",
                source=source,
                opens_at=start,
                closes_at=start + length,
                **kwargs,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------
# a plan
# --------------------------------------------------------------------------


def _union_seconds_per_day(windows: Sequence[DropWindow]) -> Dict[str, float]:
    """Open seconds per UTC day, overlaps counted once.

    Summing durations would let four overlapping "safety net" windows read
    as four hours when they are one, so the cap is applied to the union.  A
    window spanning midnight is split at the boundary and charged to both
    days, because the point of the cap is how hard one host is hit on one
    day.
    """
    spans = sorted((w.opens_at, w.closes_at) for w in windows)
    merged: List[List[float]] = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    per_day: Dict[str, float] = {}
    day_s = 86400.0
    for start, end in merged:
        cursor = start
        while cursor < end:
            day_index = math.floor(cursor / day_s)
            boundary = (day_index + 1) * day_s
            chunk_end = min(end, boundary)
            key = _dt.datetime.fromtimestamp(
                day_index * day_s, tz=_dt.timezone.utc
            ).date().isoformat()
            per_day[key] = per_day.get(key, 0.0) + (chunk_end - cursor)
            cursor = chunk_end
    return per_day


@dataclass(frozen=True)
class SnipePlan:
    """Every window, checked as a whole.

    A single window can be innocent and a set of them abusive, so the
    per-source per-day cap lives here rather than on :class:`DropWindow`.
    """

    windows: Tuple[DropWindow, ...]

    def __post_init__(self) -> None:
        for window in self.windows:
            if not isinstance(window, DropWindow):
                raise SnipeError(f"not a DropWindow: {window!r}")
        keys = [w.key for w in self.windows]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise SnipeError(f"two windows share an identity: {duplicates}")
        for source in self.sources:
            per_day = _union_seconds_per_day(self.for_source(source))
            for day, seconds in sorted(per_day.items()):
                if seconds > MAX_OPEN_PER_DAY_S:
                    raise SnipeError(
                        f"{source!r} would be polled tight for {seconds / 3600:.1f}h on "
                        f"{day}; the cap is {MAX_OPEN_PER_DAY_S / 3600:.0f}h per day. "
                        f"That is a faster baseline, not a drop window"
                    )

    @classmethod
    def of(cls, *windows: Any) -> "SnipePlan":
        """``SnipePlan.of(w1, w2)`` or ``SnipePlan.of(list_of_windows)``."""
        flat: List[DropWindow] = []
        for item in windows:
            if isinstance(item, DropWindow):
                flat.append(item)
            elif isinstance(item, Iterable):
                flat.extend(item)
            else:
                raise SnipeError(f"not a DropWindow or an iterable of them: {item!r}")
        return cls(tuple(flat))

    @property
    def sources(self) -> Tuple[str, ...]:
        seen: List[str] = []
        for window in self.windows:
            if window.source not in seen:
                seen.append(window.source)
        return tuple(seen)

    def for_source(self, source: str) -> Tuple[DropWindow, ...]:
        return tuple(w for w in self.windows if w.source == source)

    def open_at(self, source: str, now: float) -> Optional[DropWindow]:
        """The window in force for a source, or ``None``.

        Overlapping windows are allowed and the tightest wins -- a general
        "release day" window at 60s with a "the minute itself" window at
        30s inside it does the obvious thing.  Ties break on the name, so
        two equally tight windows always resolve the same way and a
        report never flickers between them.
        """
        candidates = [w for w in self.for_source(source) if w.is_open(now)]
        if not candidates:
            return None
        return min(candidates, key=lambda w: (w.interval_s, w.name))

    def arming_at(self, now: float) -> Tuple[DropWindow, ...]:
        return tuple(w for w in self.windows if w.phase(now) is Phase.ARMING)

    def next_change_at(self, now: float) -> Optional[float]:
        """The next moment the plan wants something to happen -- an arm, an
        open or a close.  A supervisor can sleep until then instead of
        waking every second to find nothing to do."""
        moments = []
        for window in self.windows:
            for moment in (window.arms_at, window.opens_at, window.closes_at):
                if moment > now:
                    moments.append(moment)
        return min(moments) if moments else None


# --------------------------------------------------------------------------
# preflight
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArmCheck:
    """One thing that has to be true for a window to be worth opening.

    ``blocking`` separates "this snipe cannot work" from "this is worth
    knowing".  ``ok is False and blocking is False`` with a detail saying
    so is also how an *unwired* probe reports: not knowing is not the same
    as being fine, and pretending otherwise is how a dead push endpoint
    survives to drop day.
    """

    name: str
    ok: bool
    blocking: bool
    detail: str

    @property
    def unknown(self) -> bool:
        return not self.ok and not self.blocking


@dataclass(frozen=True)
class ArmReport:
    """What the preflight found, in time to do something about it."""

    window: DropWindow
    at: float
    checks: Tuple[ArmCheck, ...]

    @property
    def blockers(self) -> Tuple[ArmCheck, ...]:
        return tuple(c for c in self.checks if c.blocking and not c.ok)

    @property
    def unknowns(self) -> Tuple[ArmCheck, ...]:
        return tuple(c for c in self.checks if c.unknown)

    @property
    def ready(self) -> bool:
        return not self.blockers

    def summary(self) -> str:
        minutes = max(0.0, self.window.opens_at - self.at) / 60.0
        head = f"{self.window.name} opens in {minutes:.0f} min"
        if self.blockers:
            return (
                f"{head}: NOT READY -- "
                + "; ".join(f"{c.name}: {c.detail}" for c in self.blockers)
            )
        if self.unknowns:
            return (
                f"{head}: armed, but unverified -- "
                + "; ".join(f"{c.name}: {c.detail}" for c in self.unknowns)
            )
        return f"{head}: armed, every check green"


# --------------------------------------------------------------------------
# events
# --------------------------------------------------------------------------


class EventKind(Enum):
    ARMED = "armed"          # preflight ran and passed
    NOT_READY = "not_ready"  # preflight ran and found a blocker
    OPENED = "opened"        # the tight policy is now in force
    CLOSED = "closed"        # the base policy is back
    REBASED = "rebased"      # somebody else changed the policy; we adopted it


@dataclass(frozen=True)
class SnipeEvent:
    """Something the controller did, for the bot to turn into an alert."""

    at: float
    kind: EventKind
    source: str
    window: str
    detail: str
    report: Optional[ArmReport] = None

    @property
    def wants_attention(self) -> bool:
        """NOT_READY is the one the owner has to see: it is the difference
        between a snipe that works and one that fails silently."""
        return self.kind is EventKind.NOT_READY


# --------------------------------------------------------------------------
# the controller
# --------------------------------------------------------------------------


class SnipeController:
    """Drives one :class:`~jarvis_poke.sources.PollScheduler` from a plan.

    Call :meth:`sync` on every tick.  It opens and closes windows, runs
    preflights at arm time, and returns what it did.  It owns no clock and
    no state that is not in :meth:`snapshot`.

    Probes, all optional and all injected -- this package makes no network
    calls and reads no registry:

    ``alert_probe(profile_id)``  ``{"devices": int, "confirmed_at": float|None}``
                                 -- how many live push subscriptions the
                                 profile has and when one last worked.
    ``rules_probe(source)``      the :class:`~jarvis_poke.contracts.Rule`
                                 objects that could fire for this source.
    ``budget_probe()``           ``{"remaining_cents": int}``.

    An unwired probe is reported as unknown, never as fine.
    """

    def __init__(
        self,
        scheduler: Any,
        plan: SnipePlan,
        *,
        profile_id: str = "owner",
        alert_probe: Optional[Callable[[str], Mapping[str, Any]]] = None,
        rules_probe: Optional[Callable[[str], Sequence[Rule]]] = None,
        budget_probe: Optional[Callable[[], Mapping[str, Any]]] = None,
        alert_path_max_age_s: float = ALERT_PATH_MAX_AGE_S,
    ) -> None:
        # ``policies`` is deliberately not in this list: on the real
        # PollScheduler it is a property, not a method, and requiring it
        # as a callable rejected the very class this controller drives.
        for method in ("policy", "set_policy", "retime_source"):
            if not callable(getattr(scheduler, method, None)):
                raise SnipeError(
                    f"scheduler has no {method}(); SnipeController needs a "
                    f"jarvis_poke.sources.PollScheduler"
                )
        if not isinstance(plan, SnipePlan):
            raise SnipeError(f"not a SnipePlan: {plan!r}")
        self.scheduler = scheduler
        self.plan = plan
        self.profile_id = _text(profile_id, "profile_id")
        self.alert_probe = alert_probe
        self.rules_probe = rules_probe
        self.budget_probe = budget_probe
        self.alert_path_max_age_s = _finite(alert_path_max_age_s, "alert_path_max_age_s")
        #: source -> the policy to go back to when a window closes.
        self._base: Dict[str, FetchPolicy] = {}
        #: source -> the key of the window whose policy is installed.
        self._applied: Dict[str, str] = {}
        #: window keys already preflighted, so a preflight runs once.
        self._armed: List[str] = []
        for source in plan.sources:
            # Fail now, loudly, rather than at the drop: a window for a
            # source the scheduler has never heard of would never fire.
            self._base[source] = self.scheduler.policy(source)

    def __repr__(self) -> str:
        return (
            f"<SnipeController windows={len(self.plan.windows)} "
            f"open={sorted(self._applied)}>"
        )

    # -- the tick ----------------------------------------------------------

    def sync(self, now: float) -> List[SnipeEvent]:
        """Bring the scheduler in line with the plan at ``now``."""
        moment = _finite(now, "now")
        events: List[SnipeEvent] = []
        for window in self.plan.arming_at(moment):
            if window.key in self._armed:
                continue
            self._armed.append(window.key)
            report = self.preflight(window, moment)
            events.append(
                SnipeEvent(
                    at=moment,
                    kind=EventKind.ARMED if report.ready else EventKind.NOT_READY,
                    source=window.source,
                    window=window.name,
                    detail=report.summary(),
                    report=report,
                )
            )
        for source in self.plan.sources:
            events.extend(self._sync_source(source, moment))
        self._forget_old(moment)
        return events

    def _sync_source(self, source: str, now: float) -> List[SnipeEvent]:
        events: List[SnipeEvent] = []
        events.extend(self._adopt_external(source, now))
        window = self.plan.open_at(source, now)
        applied = self._applied.get(source)
        wanted = window.key if window is not None else None
        if wanted == applied:
            return events
        if window is not None:
            policy = window.policy_from(self._base[source])
            self.scheduler.set_policy(policy)
            moved = self.scheduler.retime_source(source, now)
            self._applied[source] = window.key
            events.append(
                SnipeEvent(
                    at=now,
                    kind=EventKind.OPENED,
                    source=source,
                    window=window.name,
                    detail=(
                        f"polling {source} every {window.interval_s:.0f}s until "
                        f"the window closes ({moved} gate(s) moved in)"
                    ),
                )
            )
        else:
            base = self._base[source]
            self.scheduler.set_policy(base)
            moved = self.scheduler.retime_source(source, now)
            name = applied.split("|")[1] if applied else "window"
            events.append(
                SnipeEvent(
                    at=now,
                    kind=EventKind.CLOSED,
                    source=source,
                    window=name,
                    detail=(
                        f"back to every {base.min_interval_s:.0f}s "
                        f"({moved} gate(s) pushed out)"
                    ),
                )
            )
            self._applied.pop(source, None)
        return events

    def _adopt_external(self, source: str, now: float) -> List[SnipeEvent]:
        """Notice a policy somebody else installed, and keep it.

        The app re-installs a policy when a robots.txt re-check changes the
        verdict or the owner edits the rate.  If that lands while a window
        is open, closing the window would otherwise restore the *stale*
        base and quietly undo them.  So: whenever the live policy is
        neither our base nor the tightened one we installed, it is
        somebody else's and it becomes the new base.
        """
        live = self.scheduler.policy(source)
        base = self._base[source]
        if live == base:
            return []
        applied = self._applied.get(source)
        if applied is not None:
            window = self._window_by_key(applied)
            if window is not None and live == window.policy_from(base):
                return []
        self._base[source] = replace(live, min_interval_s=live.min_interval_s)
        if applied is not None:
            # Keep the window's rate in force over the new base.
            window = self._window_by_key(applied)
            if window is not None:
                self.scheduler.set_policy(window.policy_from(self._base[source]))
        return [
            SnipeEvent(
                at=now,
                kind=EventKind.REBASED,
                source=source,
                window=applied.split("|")[1] if applied else "",
                detail=(
                    f"adopted a policy installed elsewhere: every "
                    f"{live.min_interval_s:.0f}s, robots_allows={live.robots_allows}"
                ),
            )
        ]

    def _window_by_key(self, key: str) -> Optional[DropWindow]:
        for window in self.plan.windows:
            if window.key == key:
                return window
        return None

    def _forget_old(self, now: float) -> None:
        """Drop arm records for windows that are over, so a controller
        left running for a month does not grow a list of every window it
        ever armed."""
        live = {
            w.key for w in self.plan.windows if w.phase(now) is not Phase.CLOSED
        }
        self._armed = [key for key in self._armed if key in live]

    # -- preflight ---------------------------------------------------------

    def preflight(self, window: DropWindow, now: float) -> ArmReport:
        """Everything that has to be true, checked while it can be fixed."""
        if not isinstance(window, DropWindow):
            raise SnipeError(f"not a DropWindow: {window!r}")
        moment = _finite(now, "now")
        checks: List[ArmCheck] = [
            self._check_policy(window),
            self._check_pause(window, moment),
            self._check_alert_path(moment),
            self._check_rules(window),
            self._check_budget(window),
        ]
        return ArmReport(window=window, at=moment, checks=tuple(checks))

    def _check_policy(self, window: DropWindow) -> ArmCheck:
        try:
            policy = self.scheduler.policy(window.source)
        except Exception as exc:  # the scheduler's own error type
            return ArmCheck("policy", False, True, f"no policy for {window.source}: {exc}")
        if not policy.robots_allows:
            return ArmCheck(
                "policy", False, True,
                f"robots.txt disallows {window.source}; this window will never poll",
            )
        return ArmCheck(
            "policy", True, True,
            f"{window.source} allowed, tightening to {window.interval_s:.0f}s",
        )

    def _check_pause(self, window: DropWindow, now: float) -> ArmCheck:
        state = self.scheduler.pause_state(now) if callable(
            getattr(self.scheduler, "pause_state", None)
        ) else None
        if not isinstance(state, Mapping):
            return ArmCheck("pause", False, False, "scheduler reported no pause state")
        row = state.get(window.source) or {}
        until = row.get("paused_until") or 0.0
        if not isinstance(until, (int, float)) or until <= window.opens_at:
            return ArmCheck("pause", True, True, "not paused when the window opens")
        reason = str(row.get("reason") or row.get("pause_reason") or "paused")
        return ArmCheck(
            "pause", False, True,
            f"{window.source} is paused past the window ({reason}); a pause outranks "
            f"a window, so nothing will be polled",
        )

    def _check_alert_path(self, now: float) -> ArmCheck:
        if self.alert_probe is None:
            return ArmCheck(
                "alert_path", False, False,
                "no alert-path probe wired -- this is the check most worth wiring: a "
                "dead push subscription is the commonest way a snipe is missed",
            )
        try:
            answer = self.alert_probe(self.profile_id)
        except Exception as exc:
            return ArmCheck("alert_path", False, True, f"probe raised: {type(exc).__name__}")
        if not isinstance(answer, Mapping):
            return ArmCheck("alert_path", False, True, f"probe returned {type(answer).__name__}")
        devices = answer.get("devices")
        if not isinstance(devices, int) or isinstance(devices, bool) or devices < 1:
            return ArmCheck(
                "alert_path", False, True,
                "no live push subscription for this profile; a BUY would be decided "
                "and never delivered",
            )
        confirmed = answer.get("confirmed_at")
        if not isinstance(confirmed, (int, float)) or isinstance(confirmed, bool):
            return ArmCheck(
                "alert_path", False, False,
                f"{devices} device(s), but none has a confirmed delivery on record",
            )
        age = now - float(confirmed)
        if age > self.alert_path_max_age_s:
            return ArmCheck(
                "alert_path", False, True,
                f"{devices} device(s), but the last confirmed delivery was "
                f"{age / 86400:.0f} days ago; push endpoints rotate -- send a test",
            )
        return ArmCheck(
            "alert_path", True, True,
            f"{devices} device(s), last confirmed {age / 3600:.1f}h ago",
        )

    def _check_rules(self, window: DropWindow) -> ArmCheck:
        if self.rules_probe is None:
            return ArmCheck("rules", False, False, "no rules probe wired")
        try:
            rules = list(self.rules_probe(window.source) or ())
        except Exception as exc:
            return ArmCheck("rules", False, True, f"probe raised: {type(exc).__name__}")
        usable = [
            rule for rule in rules
            if isinstance(rule, Rule)
            and rule.enabled
            and (not rule.allowed_sources or window.source in rule.allowed_sources)
        ]
        if not usable:
            return ArmCheck(
                "rules", False, True,
                f"no enabled rule can fire for {window.source}; the window would poll "
                f"hard and never raise a thing",
            )
        cheapest = min(rule.max_price for rule in usable)
        return ArmCheck(
            "rules", True, True,
            f"{len(usable)} rule(s) armed, lowest cap {fmt_cents(cheapest)}",
        )

    def _check_budget(self, window: DropWindow) -> ArmCheck:
        if self.budget_probe is None:
            return ArmCheck("budget", False, False, "no budget probe wired")
        try:
            answer = self.budget_probe()
        except Exception as exc:
            return ArmCheck("budget", False, True, f"probe raised: {type(exc).__name__}")
        if not isinstance(answer, Mapping):
            return ArmCheck("budget", False, True, f"probe returned {type(answer).__name__}")
        remaining = answer.get("remaining_cents")
        if not isinstance(remaining, int) or isinstance(remaining, bool):
            return ArmCheck("budget", False, True, "probe gave no integer remaining_cents")
        need = self._cheapest_cap(window.source)
        if remaining <= 0:
            return ArmCheck(
                "budget", False, True,
                "nothing left in the budget; every verdict in this window would be a SKIP",
            )
        if need is not None and remaining < need:
            return ArmCheck(
                "budget", False, True,
                f"{fmt_cents(remaining)} left but the cheapest armed rule caps at "
                f"{fmt_cents(need)}; nothing in this window is affordable",
            )
        return ArmCheck("budget", True, True, f"{fmt_cents(remaining)} available")

    def _cheapest_cap(self, source: str) -> Optional[int]:
        if self.rules_probe is None:
            return None
        try:
            rules = [
                rule for rule in (self.rules_probe(source) or ())
                if isinstance(rule, Rule) and rule.enabled
            ]
        except Exception:
            return None
        return min((rule.max_price for rule in rules), default=None)

    # -- what the rest of the app asks ------------------------------------

    def active_window(self, source: str, now: float) -> Optional[DropWindow]:
        return self.plan.open_at(source, _finite(now, "now"))

    def ttl_for(self, source: str, now: float) -> float:
        """How long an alert raised now is worth showing.  Inside a window
        that is the window's ``ttl_s``; outside it is the default, because
        an ordinary restock keeps for longer than a drop does."""
        window = self.active_window(source, now)
        return window.ttl_s if window is not None else DEFAULT_TTL_S

    def snapshot(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "applied": dict(self._applied),
            "armed": list(self._armed),
            "base": {
                source: {
                    "min_interval_s": policy.min_interval_s,
                    "robots_allows": policy.robots_allows,
                }
                for source, policy in self._base.items()
            },
        }

    def restore(self, snapshot: Mapping[str, Any], now: float) -> List[SnipeEvent]:
        """Reload and then *re-assert*, rather than assume.

        A restart between a window opening and closing leaves the
        scheduler holding a tightened policy with nobody tracking it.
        Restoring the bookkeeping is not enough, so this ends with a
        :meth:`sync`: whatever the file said, the scheduler is put back
        in line with the plan and the clock.
        """
        version = snapshot.get("version", 1)
        if not isinstance(version, int) or version > 1:
            raise SnipeError(f"snipe state version {version!r} is newer than this build")
        applied = snapshot.get("applied") or {}
        armed = snapshot.get("armed") or []
        self._applied = {
            str(k): str(v) for k, v in applied.items() if str(k) in self._base
        }
        self._armed = [str(k) for k in armed]
        base = snapshot.get("base") or {}
        for source, row in base.items():
            if source not in self._base or not isinstance(row, Mapping):
                continue
            interval = row.get("min_interval_s")
            if isinstance(interval, (int, float)) and not isinstance(interval, bool):
                # Only the rate is restored; the user agent and the robots
                # verdict belong to the app's live policy, not to a file
                # that may predate a robots.txt change.
                try:
                    self._base[source] = replace(
                        self._base[source], min_interval_s=float(interval)
                    )
                except ValueError:
                    pass
        return self.sync(_finite(now, "now"))


# --------------------------------------------------------------------------
# the alert
# --------------------------------------------------------------------------


class SnipeAlertBridge(AlertBridge):
    """An :class:`~jarvis_poke.alerts_bridge.AlertBridge` that knows about
    windows.

    Three differences from the ordinary buy alert, and nothing else:

    * the ``kind`` is :data:`SNIPE_KIND`, so the client can route a
      time-critical one differently from a bargain it can think about;
    * the payload carries ``window``, ``expires_at``, ``seen_at`` and
      ``latency_ms`` -- the last being how long it took us to get from
      seeing the listing to handing the alert over, which is the number
      that says whether the snipe path is actually fast;
    * the title says the window and, past the ttl, says it is late rather
      than pretending otherwise.

    Deduplication, the outbox, the flat-scalar payload rule and the refusal
    to publish anything but a BUY are all inherited unchanged.
    """

    def __init__(self, alert_service: Any, clock: Clock, controller: SnipeController,
                 **kwargs: Any) -> None:
        kwargs.setdefault("kind", SNIPE_KIND)
        super().__init__(alert_service, clock, **kwargs)
        if not isinstance(controller, SnipeController):
            raise BridgeError(f"not a SnipeController: {controller!r}")
        self.controller = controller
        #: seconds from observation to hand-off, newest last.  Bounded.
        self.latencies: List[float] = []

    def _window(self, verdict: Verdict) -> Optional[DropWindow]:
        if not verdict.source:
            return None
        return self.controller.active_window(verdict.source, self.now())

    def _latency_s(self, verdict: Verdict) -> float:
        """Observation to hand-off.  Never negative: a verdict stamped in
        the future is a clock disagreement, not a negative latency, and
        reporting -3s as speed would flatter exactly the bug worth
        seeing."""
        return max(0.0, self.now() - float(verdict.at))

    def alert_title(self, product: Product, verdict: Verdict) -> str:
        window = self._window(verdict)
        price = fmt_cents(landed_cents(verdict))
        quantity = f"{verdict.quantity} x " if verdict.quantity > 1 else ""
        if window is None:
            return f"Buy {quantity}{product.name} at {price}"
        return f"DROP: {quantity}{product.name} at {price}"

    def alert_body(self, verdict: Verdict) -> str:
        text = super().alert_body(verdict)
        window = self._window(verdict)
        latency = self._latency_s(verdict)
        parts = [text]
        if window is not None:
            parts.append(f"Window: {window.name}.")
        parts.append(f"Seen {latency:.0f}s ago.")
        return " ".join(parts)

    def build_data(self, verdict: Verdict) -> Dict[str, Any]:
        from jarvis_poke.alerts_bridge import _check_payload

        data = dict(super().build_data(verdict))
        window = self._window(verdict)
        latency = self._latency_s(verdict)
        self.latencies.append(latency)
        del self.latencies[:-200]
        data["window"] = window.name if window is not None else None
        data["expires_at"] = int(
            window.expires_at(self.now()) if window is not None
            else self.now() + DEFAULT_TTL_S
        )
        data["seen_at"] = int(verdict.at)
        data["latency_ms"] = int(latency * 1000)
        _check_payload(data, SNIPE_DATA_KEYS)
        return data

    @property
    def median_latency_s(self) -> Optional[float]:
        if not self.latencies:
            return None
        ordered = sorted(self.latencies)
        middle = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[middle]
        return (ordered[middle - 1] + ordered[middle]) / 2.0


# --------------------------------------------------------------------------
# smoke run: python3 -m jarvis_poke.snipe
# --------------------------------------------------------------------------


def _demo() -> Dict[str, Any]:
    """A window arming, opening and closing over a fake scheduler, with a
    preflight that finds a dead push subscription.  No network, no clock."""
    base = FetchPolicy(source="examplemart", min_interval_s=300.0)

    class FakeScheduler:
        def __init__(self) -> None:
            self._p = {"examplemart": base}
            self.retimes: List[Tuple[str, float]] = []

        def policies(self) -> Dict[str, FetchPolicy]:
            return dict(self._p)

        def policy(self, source: str) -> FetchPolicy:
            return self._p[source]

        def set_policy(self, policy: FetchPolicy) -> None:
            self._p[policy.source] = policy

        def retime_source(self, source: str, now: float) -> int:
            self.retimes.append((source, now))
            return 1

        def pause_state(self, now: float) -> Dict[str, Any]:
            return {"examplemart": {"paused_until": 0.0, "reason": ""}}

    t0 = 1_700_000_000.0
    scheduler = FakeScheduler()
    window = DropWindow(
        name="surging-sparks-restock",
        source="examplemart",
        opens_at=t0 + 3600.0,
        closes_at=t0 + 3600.0 + 1800.0,
        interval_s=30.0,
        note="placeholder retailer",
    )
    controller = SnipeController(
        scheduler,
        SnipePlan.of(window),
        alert_probe=lambda profile: {"devices": 0, "confirmed_at": None},
        rules_probe=lambda source: [Rule(product_id="sv08-etb", max_price=6000)],
        budget_probe=lambda: {"remaining_cents": 20000},
    )
    log: List[str] = []
    for moment in (t0, window.arms_at + 1, window.opens_at + 1, window.closes_at + 1):
        for event in controller.sync(moment):
            log.append(f"{event.kind.value}: {event.detail}")
        log.append(
            f"  @{moment - t0:+.0f}s interval="
            f"{scheduler.policy('examplemart').min_interval_s:.0f}s"
        )
    return {"log": log, "retimes": len(scheduler.retimes)}


if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    import json
    print(json.dumps(_demo(), indent=2))
