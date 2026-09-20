"""The part that is written once: scheduling, isolation, health, the badge.

Contract: :mod:`jarvis_bots.contracts` -- "Everything else, scheduling,
failure isolation, health, persistence, the badge count, alert delivery, is
the supervisor's job and is written once."  Each of the five rules stated
there is implemented in one named place here:

============================  =======================================
contracts.py rule             where
============================  =======================================
"A bot never blocks another"  :meth:`Supervisor.run_round`'s try/except
                              around a single ``tick`` *and* around
                              reading and applying what it returned
"A sick bot backs off"        :meth:`Supervisor._record_failure`
"Paused means paused"         :meth:`Supervisor.pause` and the two
                              guards in ``run_round`` / ``_maybe_alert``
"Events are the only output"  :meth:`Supervisor._apply_events` -- the
                              only caller of the alert service, and the
                              only thing that opens or closes an item of
                              attention (see :data:`RESOLVED_FLAG`)
"State is the bot's,          :meth:`Supervisor.save_state` /
persistence is ours"          :meth:`Supervisor.load_state` /
                              :meth:`Supervisor._adopt_orphans`, which
                              makes "load then register" as good an
                              order as "register then load"
============================  =======================================

The boundary
------------
This framework schedules bots and surfaces what they find.  Nothing here
buys, pays or transacts, and no bot can make it: the widest thing a tick
can do is return an :class:`~jarvis_bots.contracts.Event` with an ``href``,
which becomes a notification a person taps.  There is no code path from an
event to a purchase, by construction.

Injection
---------
``clock`` is a callable returning unix seconds -- nothing here calls
``time.time()``, including the tick timer, which measures with the same
injected clock so a test can make a tick take an hour without waiting.
``alerts`` is an injected service (``jarvis_alerts.api.AlertService``, or
anything with its ``publish``), exactly as ``jarvis_alerts`` takes an
injected sender; ``store`` is an injected persistence seam.  No module in
this package opens a socket.  No randomness is drawn anywhere in it, so a
fixed sequence of rounds is a pure function of the ticks and the clock.

Scheduling, precisely
---------------------
A bot is ticked in a round when all of these hold, checked in this order:

1. it is not paused;
2. it is not inside a live quarantine (``now >= quarantined_until``);
3. it is due (``health.next_due_at <= now``).

An expired quarantine is *consumed* at step 2: ``quarantined_until`` is
cleared and that one tick is the probe.  If the probe raises, the failure
count is already at or past the threshold, so the bot is quarantined again
by the same code that quarantined it the first time -- one probe per
quarantine, never a retry loop.

Everything a bot's own code can reach -- ``tick``, reading what it
returned, and applying it -- is inside one ``try`` per bot, and that
``try`` catches ``BaseException``.  ``KeyboardInterrupt`` is recorded
against the bot and then re-raised, because Ctrl-C is the operator's and
not the fleet's.

Interval arithmetic uses :func:`jarvis_bots.contracts.backoff_interval`
through :func:`_widened_interval`, which re-applies the floor that function
documents but loses for intervals above ``BACKOFF_CAP_S`` -- see that
helper.  The one place this module departs from plain backoff is the moment
of quarantine, where the next attempt is ``max(now + widened,
quarantined_until)``: QUARANTINE_S is documented as "how long a quarantine
lasts before one probe tick is allowed through", so the quarantine holds
the probe back, and the ``max`` keeps it from ever being *released* sooner
than the back-off that failure earned.  Against ``now + interval_s`` it
was released sooner -- a 600s bot went from a 3600s gap after its third
failure to an 1800s gap after its fourth -- and ``BotInfo.interval_s``
says the supervisor "may widen this when the bot is failing, never narrow
it".
"""

from __future__ import annotations

import collections.abc
import dataclasses
import json
import os
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from jarvis_bots.base import Clock, parse_severity
from jarvis_bots.contracts import (
    MAX_ATTENTION_PER_BOT,
    MAX_EVENTS_PER_TICK,
    QUARANTINE_AFTER_FAILURES,
    QUARANTINE_S,
    SLOW_TICK_S,
    AttentionItem,
    Bot,
    BotState,
    BotStatus,
    Event,
    Health,
    RoundReport,
    Severity,
    backoff_interval,
)
from jarvis_bots.registry import BotRegistry

__all__ = [
    "ALERT_KIND",
    "DEFAULT_PROFILE_ID",
    "EVENT_HISTORY",
    "MAX_FAILURES_PERSISTED",
    "RESOLVED_FLAG",
    "STATE_KEY",
    "STATE_VERSION",
    "JsonFileStore",
    "Supervisor",
    "SupervisorError",
]

#: The alert ``kind`` every bot event carries.  One kind, so the client can
#: route all of them to the same tap handler and read ``data['bot_id']`` to
#: decide where to land; the same choice ``jarvis_poke.alerts_bridge`` makes.
ALERT_KIND = "bot_event"

#: Whose phone, when the app does not say.  ``jarvis_alerts`` is
#: multi-profile; a personal assistant has one owner.
DEFAULT_PROFILE_ID = "owner"

#: The key in ``Event.data`` by which a bot says "that question is closed".
#:
#: contracts.py gives a bot a way to *open* a standing request for a
#: decision -- an event at ACTION or above carrying an ``attention_key`` --
#: and no way to close one, although the badge it drives is a count of what
#: is *open*.  Without a closing signal the only exit is
#: :meth:`Supervisor.clear_attention`, which the app has to call, so every
#: bot's first restock sits in the badge for ever and the owner learns to
#: ignore it.
#:
#: The signal both bots written against this framework already emit is the
#: same one: re-raise the *same key* at a severity below ACTION -- so
#: :attr:`~jarvis_bots.contracts.Event.wants_attention` is false and the
#: request is not re-opened -- with ``resolved=True`` in ``data``.  See
#: ``jarvis_bots/templates/bot.py.tmpl`` ("a supervisor that closes keys on
#: the flag and one that closes them on the severity agree") and
#: :meth:`jarvis_bots.bots.poke_bot.PokeBot._resolve`.  The flag is
#: required, not merely inferred from the severity: an INFO line that
#: happens to mention an open key is a line in the feed, and silently
#: emptying the badge on it would be worse than never emptying it.
RESOLVED_FLAG = "resolved"

#: How many of a bot's recent events the supervisor keeps, per bot, for the
#: detail page's feed.  Bounded on purpose: this is a ring the launcher
#: reads and the state file carries, not a log.  ``jarvis_bots/api.py``
#: serves it and ``templates/page.html.tmpl`` renders up to 20 of them.
EVENT_HISTORY = 20

#: An upper bound on a restored ``consecutive_failures``.  The counter only
#: ever feeds :func:`~jarvis_bots.contracts.backoff_interval`, which is
#: capped long before this, so nothing is lost by refusing to believe a
#: state file that claims a bot has failed 10**9 times in a row -- and an
#: unbounded ``int()`` out of a file is how an arithmetic overflow gets
#: into the failure handler.
MAX_FAILURES_PERSISTED = 10_000

#: Bumped when the shape of :meth:`Supervisor.save_state` changes.
STATE_VERSION = 1

#: The key used when the store is a plain mutable mapping.
STATE_KEY = "jarvis_bots"

#: Severity -> the priority name ``jarvis_alerts.api.parse_priority``
#: accepts.  Names, not the enum, so this module does not have to import
#: ``jarvis_alerts`` to talk to it -- the alert service stays injectable and
#: a fake in a test needs nothing but a ``publish``.
_PRIORITY_FOR_SEVERITY: Dict[Severity, str] = {
    Severity.DEBUG: "low",
    Severity.INFO: "low",
    Severity.NOTICE: "normal",
    Severity.ACTION: "high",
    Severity.ERROR: "high",
}

#: An error string longer than this is truncated before it goes into health:
#: ``last_error`` is rendered on a card and persisted every save.
_MAX_ERROR_CHARS = 500


class SupervisorError(Exception):
    """Misuse of the supervisor itself: an unusable store, an unusable
    alert service, pausing a bot that declares ``can_pause=False``, or a
    state file this version cannot read.

    Errors *from a bot* are never this: they are caught, recorded in that
    bot's :class:`~jarvis_bots.contracts.Health`, and the round continues.
    """


# --------------------------------------------------------------------------
# Persistence seam
# --------------------------------------------------------------------------


class JsonFileStore:
    """The simplest store that survives a restart: one JSON file.

    Supplied because "persistence is ours" and a framework that makes a bot
    cheap to add should not also make the app write this.  Anything with
    ``load()``/``save(state)`` (or ``load_state()``/``save_state(state)``,
    or a plain dict) works just as well -- :class:`Supervisor` only needs
    those two calls.

    The write is atomic (temp file in the same directory, then
    ``os.replace``) so a crash mid-save leaves the previous state rather
    than half of the new one, and the file is created ``0600``, the same way
    ``jarvis_alerts.outbox`` creates its outbox: a snapshot lists what the
    owner is watching.
    """

    def __init__(self, path: str) -> None:
        self.path = str(path)

    def load(self) -> Dict[str, Any]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                state = json.load(handle)
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            raise SupervisorError(f"cannot read bot state from {self.path}: {exc}") from exc
        if not isinstance(state, dict):
            raise SupervisorError(f"{self.path} does not hold a state object")
        return state

    def save(self, state: Dict[str, Any]) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".bots-", suffix=".tmp", delete=False
        )
        try:
            with handle:
                json.dump(state, handle, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(handle.name, 0o600)
            os.replace(handle.name, self.path)
        except BaseException:
            try:
                os.unlink(handle.name)
            except OSError:  # pragma: no cover - the replace already won
                pass
            raise

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<JsonFileStore {self.path!r}>"


def _store_ops(store: Any) -> Tuple[Callable[[], Any], Callable[[Dict[str, Any]], None]]:
    """Adapt whatever the app injected to ``(load, save)``.

    Three shapes are accepted, in this order: ``load``/``save``,
    ``load_state``/``save_state``, and a plain mutable mapping keyed by
    :data:`STATE_KEY`.  Duck typing rather than a required base class,
    because the app's own store already exists and should not have to
    inherit from this package to be usable by it.
    """
    for load_name, save_name in (("load", "save"), ("load_state", "save_state")):
        loader = getattr(store, load_name, None)
        saver = getattr(store, save_name, None)
        if callable(loader) and callable(saver):
            return loader, saver
    if isinstance(store, MutableMapping):
        return (
            lambda: store.get(STATE_KEY),
            lambda state: store.__setitem__(STATE_KEY, state),
        )
    raise SupervisorError(
        f"store must have load()/save(state) (or load_state()/save_state(state)), "
        f"or be a mutable mapping; got {type(store).__name__}"
    )


def _health_from_dict(raw: Any) -> Health:
    """Rebuild a :class:`Health` from persisted fields, ignoring any it does
    not know.  Unknown keys are dropped rather than raising, so a state file
    written by a later version still restores the fields this one has."""
    if not isinstance(raw, dict):
        raise SupervisorError(f"health must be an object; got {type(raw).__name__}")
    known = {field.name for field in dataclasses.fields(Health)}
    kwargs: Dict[str, Any] = {}
    for name, value in raw.items():
        if name not in known:
            continue
        if name in ("consecutive_failures", "total_failures", "total_ticks"):
            # Bounded, and never negative: a bare int() out of a file is an
            # unbounded exponent in backoff_interval, and a negative count
            # would make a failing bot look healthy.
            kwargs[name] = max(0, min(MAX_FAILURES_PERSISTED, int(value)))
        elif name == "last_error":
            kwargs[name] = str(value)[:_MAX_ERROR_CHARS]
        else:
            number = float(value)
            if number != number:  # NaN compares false against every deadline
                number = 0.0
            kwargs[name] = number
    return Health(**kwargs)


def _json_scalar(value: Any) -> bool:
    """True for the values an alert payload may carry.

    ``jarvis_alerts`` keeps the owner's device data opaque and nothing is
    allowed to ride along in a notification; a nested object in ``data`` is
    how a blob, a token or a whole config gets there.  Flat scalars only.
    """
    return value is None or isinstance(value, (str, int, float, bool))


# --------------------------------------------------------------------------
# The supervisor
# --------------------------------------------------------------------------


class Supervisor:
    """Runs the registered bots and owns everything about them that is not
    their own work.

    ``registry``            :class:`~jarvis_bots.registry.BotRegistry`;
                            ticks happen in registration order.
    ``clock``               callable returning unix seconds.
    ``store``               optional persistence seam (see :func:`_store_ops`).
                            Pausing writes through it, so a pause survives a
                            restart -- a pause that forgets itself overnight
                            is worse than no pause button.
    ``alerts``              optional ``jarvis_alerts.api.AlertService`` (or
                            anything with its ``publish``, or a plain
                            callable taking the event).
    ``alert_min_severity``  events at or above this earn a push; the default
                            is ACTION, which contracts.py describes as "the
                            owner should decide something".
    ``profile_id``          whose phone, passed through to the alert service.
    ``alert_kind``          the alert ``kind`` for routing on the client.

    Not thread safe, and deliberately not: a round is a loop, and the app
    calls it from one scheduler.
    """

    def __init__(
        self,
        registry: BotRegistry,
        clock: Clock,
        store: Any = None,
        alerts: Any = None,
        alert_min_severity: Union[Severity, int, str] = Severity.ACTION,
        *,
        profile_id: str = DEFAULT_PROFILE_ID,
        alert_kind: str = ALERT_KIND,
    ) -> None:
        if registry is None:
            raise SupervisorError("a supervisor needs a registry")
        if not callable(clock):
            raise SupervisorError(
                "a supervisor needs an injected clock: a callable returning unix "
                "seconds (contracts.py: nothing here calls time.time())"
            )
        if alerts is not None and not callable(getattr(alerts, "publish", None)):
            if not callable(alerts):
                raise SupervisorError(
                    f"alerts must have publish() like jarvis_alerts.AlertService, "
                    f"or be a callable taking an Event; got {type(alerts).__name__}"
                )
        if not isinstance(profile_id, str) or not profile_id:
            raise SupervisorError("profile_id must be a non-empty string")
        if not isinstance(alert_kind, str) or not alert_kind:
            raise SupervisorError("alert_kind must be a non-empty string")

        self._registry = registry
        self._clock = clock
        self._store = store
        self._store_load, self._store_save = (
            _store_ops(store) if store is not None else (None, None)
        )
        self._alerts = alerts
        self._alert_min = parse_severity(alert_min_severity)
        self._profile_id = profile_id
        self._alert_kind = alert_kind

        #: bot id -> health.  The supervisor's, not the bot's (contracts.py:
        #: "Owned by the supervisor, not the bot").
        self._health: Dict[str, Health] = {}
        #: bot ids the owner has switched off.
        self._paused: Dict[str, bool] = {}
        #: (bot id, attention key) -> item.  A dict keyed this way *is* the
        #: collapsing rule: "one restock nagging across ten ticks is one item
        #: of attention, not ten".
        self._attention: Dict[Tuple[str, str], AttentionItem] = {}
        #: (bot id, attention key) -> when we alerted, so one open request
        #: does not re-alert every round.
        self._alerted: Dict[Tuple[str, str], float] = {}
        #: bot id -> its most recent event, for the launcher card.
        self._last_event: Dict[str, Event] = {}
        #: bot id -> its last :data:`EVENT_HISTORY` events, oldest first.
        #: The detail page's "Recent events" feed reads this; before it
        #: existed the generated page had nothing in the package that could
        #: fill it.  A bounded deque, not a log: it is persisted.
        self._events: Dict[str, "collections.deque[Event]"] = {}
        #: bot id -> the last snapshot() that worked.  A bot whose snapshot
        #: raises is stored with this rather than with {}: the fallback used
        #: to look only at _orphans, which is empty for a registered bot, so
        #: one failed snapshot silently erased everything the bot knew.
        self._last_snapshot: Dict[str, Dict[str, Any]] = {}
        #: State for ids that are not registered right now, kept so a bot
        #: commented out of the config for a day does not lose its snapshot.
        #: Adopted by :meth:`_adopt_orphans` the moment the id turns up in
        #: the registry, so "load the state file, then register the bots"
        #: is as good an order as the other one.
        self._orphans: Dict[str, Any] = {}
        #: Attention rows for ids that were not registered at load time,
        #: kept for the same reason and adopted with the rest.
        self._orphan_attention: Dict[Tuple[str, str], Dict[str, Any]] = {}
        #: Set when an alert or a snapshot failed; neither may break a round.
        self.last_alert_error: str = ""
        self.last_state_error: str = ""
        #: Set when a bot's output was refused for being too large: a tick
        #: over :data:`~jarvis_bots.contracts.MAX_EVENTS_PER_TICK` events or
        #: a bot past :data:`~jarvis_bots.contracts.MAX_ATTENTION_PER_BOT`
        #: open keys.  A cap that trims in silence is a cap that hides a bug.
        self.last_overflow_error: str = ""

    # -- accessors -----------------------------------------------------------

    @property
    def registry(self) -> BotRegistry:
        return self._registry

    @property
    def alert_min_severity(self) -> Severity:
        return self._alert_min

    def now(self) -> float:
        """The injected clock's current value."""
        return float(self._clock())

    def _health_of(self, bot_id: str) -> Health:
        # Every public read of health, pause or quarantine comes through
        # here or through is_paused, so adopting first is what keeps a bot
        # registered after load_state() from *reading* as healthy right up
        # until the first round.  It early-returns when there is nothing
        # kept aside, which is the normal case.
        if self._orphans or self._orphan_attention:
            self._adopt_orphans()
        health = self._health.get(bot_id)
        if health is None:
            health = Health()
            self._health[bot_id] = health
        return health

    def health(self, bot_id: str) -> Health:
        """This bot's health record, created on first use.  Raises
        :class:`~jarvis_bots.contracts.RegistryError` for an unknown id."""
        self._registry.get(bot_id)
        return self._health_of(bot_id)

    def is_paused(self, bot_id: str) -> bool:
        if self._orphans or self._orphan_attention:
            self._adopt_orphans()
        return bool(self._paused.get(bot_id))

    def paused_ids(self) -> List[str]:
        return [b for b in self._registry.ids() if self.is_paused(b)]

    def is_quarantined(self, bot_id: str) -> bool:
        """True while a quarantine is outstanding, including after it has
        expired and before the probe has run.  A bot is out of quarantine
        when a tick of it succeeds, not when a timer elapses -- the badge
        should not go green on the strength of a clock."""
        return self._health_of(bot_id).quarantined_until > 0.0

    def quarantined_ids(self) -> List[str]:
        return [b for b in self._registry.ids() if self.is_quarantined(b)]

    def last_event(self, bot_id: str) -> Optional[Event]:
        """The most recent event this bot returned, or ``None``."""
        if self._orphans or self._orphan_attention:
            self._adopt_orphans()
        return self._last_event.get(bot_id)

    def events(self, bot_id: str, limit: int = EVENT_HISTORY) -> List[Event]:
        """This bot's recent events, newest first, at most ``limit``.

        Bounded at :data:`EVENT_HISTORY` per bot and kept in memory and in
        the state file.  This is the feed the detail page renders; it is
        deliberately a short ring rather than a log, because the supervisor
        is not a logging service and the state file is read whole on every
        save.
        """
        if self._orphans or self._orphan_attention:
            self._adopt_orphans()
        feed = self._events.get(bot_id)
        if not feed:
            return []
        newest = list(reversed(feed))
        if limit is not None and limit >= 0:
            newest = newest[: int(limit)]
        return newest

    def _adopt_orphans(self) -> None:
        """Take back the state of any kept-aside id that is now registered.

        :meth:`load_state` can only restore a bot the registry already
        knows.  Plenty of wirings register *after* loading -- a lazy
        registry, a feature-flagged bot, a ``register_from`` over config
        resolved at startup -- and for those, the restore used to be lost
        twice over: the bot came up healthy and un-quarantined with its
        back-off reset, and the next ``save_state`` overwrote the kept copy
        with the empty one.  Called at the top of every round and before
        every save, so neither order can lose anything.
        """
        if not self._orphans and not self._orphan_attention:
            return
        for bot_id in list(self._orphans):
            bot = self._registry.find(bot_id)
            if bot is None:
                continue
            raw = self._orphans.pop(bot_id)
            if not isinstance(raw, dict):
                continue
            self._restore_one(bot, bot_id, raw)
        for pair in list(self._orphan_attention):
            if pair[0] not in self._registry:
                continue
            raw = self._orphan_attention.pop(pair)
            self._attention[pair] = AttentionItem(
                bot_id=pair[0],
                key=pair[1],
                text=str(raw.get("text", "")),
                href=str(raw.get("href", "")),
                since=float(raw.get("since", 0.0)),
            )
            alerted_at = raw.get("alerted_at")
            if alerted_at is not None:
                self._alerted[pair] = float(alerted_at)

    def bot_detail(
        self, bot_id: str, now: Optional[float] = None, *, limit: int = EVENT_HISTORY
    ) -> Dict[str, Any]:
        """``GET /api/bots/<id>``: the object the generated detail page reads.

        ``jarvis_bots/templates/page.html.tmpl`` documents exactly this
        shape and ``scaffold.py`` says of what it writes that "Nothing is a
        stub"; until this existed nothing in the package could produce it,
        because the supervisor kept one event per bot rather than a feed.
        It is the launcher card, plus ``detail``, plus the bot's open
        requests, plus its recent events::

            {"generated_at": int,
             "bot": {<the launcher card>, "detail": str,
                     "attention_items": [{"key","text","href","since"}],
                     "events": [{"at","severity","text","href"}]}}

        Newest event first.  Raises
        :class:`~jarvis_bots.contracts.RegistryError` for an unknown id, so
        a request for a bot that does not exist is a 404 and not an empty
        page.
        """
        self._registry.get(bot_id)
        at = float(self._clock() if now is None else now)
        state = self.launcher_state(at, include_detail=True)
        card = next((c for c in state["bots"] if c["id"] == bot_id), None)
        if card is None:  # pragma: no cover - get() already raised
            raise SupervisorError(f"no card for {bot_id!r}")
        card["attention_items"] = [
            {
                "key": item.key,
                "text": item.text,
                "href": item.href,
                "since": int(item.since),
            }
            for item in self.attention_items(bot_id)
        ]
        card["events"] = [
            {
                "at": int(event.at),
                "severity": event.severity.name.lower(),
                "text": event.text,
                "href": event.href,
            }
            for event in self.events(bot_id, limit)
        ]
        return {"generated_at": int(at), "bot": card}

    def forget(self, bot_id: str) -> None:
        """Drop every trace of a bot the app has unregistered for good:
        health, pause, attention, alert memory, last event.  Separate from
        :meth:`BotRegistry.unregister` so that rebuilding the registry from
        config is not destructive."""
        self._health.pop(bot_id, None)
        self._paused.pop(bot_id, None)
        self._last_event.pop(bot_id, None)
        self._events.pop(bot_id, None)
        self._last_snapshot.pop(bot_id, None)
        self._orphans.pop(bot_id, None)
        self.clear_bot_attention(bot_id)

    # -- the round -----------------------------------------------------------

    def run_round(self, now: Optional[float] = None) -> RoundReport:
        """Tick every bot that is due, isolated from each other.

        contracts.py: "A bot never blocks another.  Ticks are isolated; an
        exception is caught, recorded against that bot's health, and the
        round continues."  Nothing between the ``try`` and the ``except``
        touches another bot, so there is no shared object a failing tick can
        leave in a bad state.

        Returns a :class:`~jarvis_bots.contracts.RoundReport`.  Its counts
        are *this round's*: ``quarantined`` is how many bots this round put
        into quarantine, not how many are in it (that is
        :meth:`quarantined_ids`).  ``slow`` lists the ids whose tick took
        longer than ``SLOW_TICK_S``; the framework cannot kill a tick, but a
        bot hogging the round should be visible.
        """
        at = float(self._clock() if now is None else now)
        report = RoundReport(at=at)
        slow: List[str] = []
        # State kept aside for an id that was not registered when the file
        # was loaded, for a bot that has since been registered (a lazy
        # registry, a feature flag, a config resolved after startup).  Done
        # here rather than only in load_state so the order of "load" and
        # "register" stops mattering: a quarantined bot that came back
        # healthy, with its back-off reset, is the worst of the two.
        self._adopt_orphans()

        for bot in self._registry.all():
            # Identity is read once, here, and every use below is of this
            # copy.  The loop used to capture bot_id before the tick and
            # re-read bot.info after it, so a bot that rebuilt its own
            # info inside tick() filed its event, its card entry, its push
            # and its dedupe key under *another bot's id*, and set its own
            # next tick to whatever interval it liked.  info is documented
            # as "static identity, declared once".
            info = bot.info
            bot_id = info.id
            health = self._health_of(bot_id)

            # Rule: "Paused means paused."  Not ticked, and nothing below
            # this line can run for it.
            if self.is_paused(bot_id):
                report.skipped += 1
                continue

            probe = False
            if health.quarantined_until > 0.0:
                if at < health.quarantined_until:
                    report.skipped += 1
                    continue
                probe = True

            if health.next_due_at > at:
                report.skipped += 1
                continue

            if probe:
                # Consume the quarantine before the tick, so one probe is one
                # probe even if this process dies inside it.
                health.quarantined_until = 0.0

            started = float(self._clock())
            try:
                # Everything a bot's own code can reach is inside this try,
                # and that includes reading what it returned and applying
                # it.  _apply_events used to sit in the else: clause, where
                # a single hand-built event with a bad severity or an
                # unprintable attention key aborted the round -- without
                # recording a failure, so the bot never backed off and the
                # next round died at the same place, for ever.
                events = _checked_events(bot_id, bot.tick(at))
                if bot.info is not info:
                    raise ValueError(
                        f"{bot_id!r} replaced its own info during tick(); "
                        f"BotInfo is static identity, declared once, and "
                        f"everything the supervisor keeps is filed under it"
                    )
                self._apply_events(bot, events, at, report)
            except BaseException as exc:
                # BaseException, not Exception.  contracts.py promises
                # "Raising is allowed and is handled" without qualification,
                # and SystemExit is what argparse, sys.exit and a handful of
                # HTTP and DNS libraries raise on a fatal config -- caught as
                # Exception, they took the fleet down while leaving health
                # untouched, so the launcher stayed green.  KeyboardInterrupt
                # is recorded and then re-raised: the failure is the bot's,
                # but Ctrl-C is the operator's and must still stop the
                # process.
                elapsed = float(self._clock()) - started
                self._record_failure(bot, health, at, exc, report)
                if isinstance(exc, KeyboardInterrupt):
                    raise
            else:
                elapsed = float(self._clock()) - started
                health.total_ticks += 1
                health.consecutive_failures = 0
                health.last_error = ""
                health.last_tick_at = at
                health.last_ok_at = at
                health.quarantined_until = 0.0
                health.next_due_at = at + _interval_of(info)
                report.ticked += 1

            if elapsed > SLOW_TICK_S:
                slow.append(bot_id)

        report.slow = tuple(slow)
        return report

    def _record_failure(
        self,
        bot: Bot,
        health: Health,
        at: float,
        exc: BaseException,
        report: RoundReport,
    ) -> None:
        """Rule: "A sick bot backs off."  Consecutive failures widen the
        interval through :func:`~jarvis_bots.contracts.backoff_interval`
        and then quarantine the bot, "so a broken bot degrades instead of
        hammering"."""
        base = _interval_of(bot.info)
        health.total_ticks += 1
        health.total_failures += 1
        # Bounded for the same reason the restored value is: the counter is
        # an exponent, and this is the handler a round cannot survive an
        # exception from.
        health.consecutive_failures = min(
            MAX_FAILURES_PERSISTED, health.consecutive_failures + 1
        )
        health.last_tick_at = at
        health.last_error = _error_text(exc)
        widened = _widened_interval(base, health.consecutive_failures)
        health.next_due_at = at + widened
        report.failed += 1
        if health.consecutive_failures >= QUARANTINE_AFTER_FAILURES:
            # The quarantine decides when the probe runs (QUARANTINE_S is
            # "how long a quarantine lasts before one probe tick is allowed
            # through"), never sooner than the back-off this failure just
            # earned.  Against ``at + base`` it was sooner: a bot on a
            # 600s interval was due in 3600s after its third failure and in
            # 1800s after its fourth, so reaching quarantine *narrowed* the
            # interval -- the one thing BotInfo.interval_s says the
            # supervisor may never do.
            health.quarantined_until = at + QUARANTINE_S
            health.next_due_at = max(at + widened, health.quarantined_until)
            report.quarantined += 1

    # -- events, attention, alerts -------------------------------------------

    def _apply_events(
        self, bot: Bot, events: Sequence[Event], at: float, report: RoundReport
    ) -> None:
        """Rule: "Events are the only output."  A bot returns events; this is
        the only place that decides what they mean -- an entry on the card, an
        item of attention, its *closing*, and at or above
        ``alert_min_severity`` a push.

        Three things one event can be, checked in this order:

        * ``wants_attention`` (ACTION or above with a key): open or refresh a
          standing request for a decision;
        * a **resolution** (:data:`RESOLVED_FLAG` set, a key, below ACTION):
          close that request.  See :data:`RESOLVED_FLAG` for why the
          framework needs this at all and why the flag is required rather
          than inferred;
        * anything else: a line in the feed.

        The two are exclusive by construction -- an event at ACTION with a
        key opens, and only a sub-ACTION event can close -- so a bot cannot
        write one that both opens and closes the same question in one round
        and leave the badge depending on the order they are read in.
        """
        if not events:
            return
        report.events += len(events)
        self._last_event[bot.info.id] = events[-1]
        feed = self._events.get(bot.info.id)
        if feed is None:
            feed = self._events[bot.info.id] = collections.deque(maxlen=EVENT_HISTORY)
        feed.extend(events)
        for event in events:
            closing: Optional[Tuple[str, str]] = None
            if event.wants_attention:
                if self._open_attention(event) is None:
                    # Refused for being past MAX_ATTENTION_PER_BOT.  A key
                    # the badge will not carry must not buzz the phone
                    # either: a bot minting keys in a loop would otherwise
                    # be capped on the page and uncapped on the device.
                    continue
            elif _is_resolution(event):
                closing = (event.bot_id, str(event.attention_key))
                self._close_attention(event)
            if self._maybe_alert(bot, event, at):
                report.alerts += 1
            if closing is not None:
                # _close_attention dropped the alert memory for this key --
                # that is how "if the same question is asked again later it
                # is news again and earns a fresh push" works -- and then
                # _maybe_alert, publishing the resolution itself, put the
                # key straight back.  The question was closed; the memory of
                # having pushed about it must not outlive it, or the next
                # time it opens the badge goes to 1 and the phone stays
                # silent for the rest of the process's life.
                self._alerted.pop(closing, None)

    def _close_attention(self, event: Event) -> bool:
        """Close the request an event reports resolved.  True if one was open.

        Closing is :meth:`clear_attention`, which is also what drops the
        alert memory for that key: if the same question is asked again
        later -- the listing is back, the folder fills up again -- it is
        news again and earns a fresh push.  A resolution for a key that is
        not open is not an error: a bot that restarted, or one the owner
        paused and resumed, may report a question closed that the
        supervisor already forgot.
        """
        return self.clear_attention(event.bot_id, str(event.attention_key))

    def _open_attention(self, event: Event) -> Optional[AttentionItem]:
        """Open or update one standing request for a decision, or refuse it.

        ``None`` means refused: this bot already holds
        :data:`~jarvis_bots.contracts.MAX_ATTENTION_PER_BOT` open keys.

        contracts.py: the badge "counts distinct open keys".  A repeat of a
        key updates the text and the link and keeps ``since``, because the
        question has been open since it was first asked -- restamping it
        every round would make "open for three days" read as "open for
        thirty seconds".
        """
        key = (event.bot_id, str(event.attention_key))
        existing = self._attention.get(key)
        if existing is None:
            held = sum(1 for owner, _k in self._attention if owner == event.bot_id)
            if held >= MAX_ATTENTION_PER_BOT:
                # A bot minting a new key every tick would grow the badge
                # for as long as the process runs.  Refused and reported,
                # not trimmed in silence.
                self.last_overflow_error = (
                    f"{event.bot_id}: refused attention key "
                    f"{str(event.attention_key)!r}; already holding "
                    f"{held} open requests (MAX_ATTENTION_PER_BOT)"
                )
                return None
        item = AttentionItem(
            bot_id=event.bot_id,
            key=key[1],
            text=event.text,
            href=event.href,
            since=existing.since if existing is not None else event.at,
        )
        self._attention[key] = item
        return item

    def attention_items(self, bot_id: Optional[str] = None) -> List[AttentionItem]:
        """Open requests for a decision, oldest registration first.  Items of
        paused bots are never included -- they are cleared on pause, and the
        filter keeps that true even for state restored from an older file."""
        if self._orphans or self._orphan_attention:
            self._adopt_orphans()
        return [
            item
            for (owner, _key), item in self._attention.items()
            if (bot_id is None or owner == bot_id) and not self.is_paused(owner)
        ]

    def attention_count(self, bot_id: Optional[str] = None) -> int:
        return len(self.attention_items(bot_id))

    def bots_wanting_attention(self) -> List[str]:
        """The distinct bots with at least one open item.  The badge counts
        keys, not bots (see :meth:`badge_status`); this is here for a caller
        that wants the other number."""
        seen: List[str] = []
        for item in self.attention_items():
            if item.bot_id not in seen:
                seen.append(item.bot_id)
        return seen

    def clear_attention(self, bot_id: str, key: str) -> bool:
        """Close one request.  True if it was open.

        The alert memory for that key goes with it, so if the same question
        is asked again later it is news again and earns a fresh push.
        """
        pair = (bot_id, key)
        self._alerted.pop(pair, None)
        return self._attention.pop(pair, None) is not None

    def clear_bot_attention(self, bot_id: str) -> int:
        """Close every request from one bot; returns how many."""
        keys = [pair for pair in self._attention if pair[0] == bot_id]
        for pair in keys:
            self._attention.pop(pair, None)
            self._alerted.pop(pair, None)
        return len(keys)

    def _maybe_alert(self, bot: Bot, event: Event, at: float) -> bool:
        """Publish through the injected alert service, or don't.

        Not alerted: below ``alert_min_severity``; any event from a paused
        bot ("a badge asking you to act on something you switched off is a
        lie", and so is the push); a repeat of an attention key that is
        already open, which is what stops one restock alerting every round
        for as long as it stays in stock.
        """
        if self._alerts is None or event.severity < self._alert_min:
            return False
        if self.is_paused(bot.info.id):
            return False
        pair: Optional[Tuple[str, str]] = (
            (bot.info.id, str(event.attention_key)) if event.attention_key else None
        )
        if pair is not None and pair in self._alerted:
            return False
        try:
            self._publish(bot, event)
        except Exception as exc:
            # An alert service that is down must not fail the bot that had
            # something to say, nor the round. It is recorded and visible.
            self.last_alert_error = _error_text(exc)
            return False
        if pair is not None and event.wants_attention:
            # Only an event that *opens* a request for a decision writes the
            # memory.  It used to be written for any event carrying a key,
            # which meant a sub-ACTION event -- a resolution, a NOTICE
            # mentioning the same key -- filed the key as "already pushed"
            # under a lowered alert_min_severity, and the real ACTION push
            # that followed was suppressed for the rest of the process's
            # life.  The memory is about the open question, so it is written
            # by the event that opens it and dropped by the one that closes
            # it (clear_attention).
            self._alerted[pair] = float(at)
        return True

    def _publish(self, bot: Bot, event: Event) -> None:
        publish = getattr(self._alerts, "publish", None)
        if callable(publish):
            publish(
                self._profile_id,
                self._alert_kind,
                bot.info.name,
                event.text,
                data=self._alert_data(bot, event),
                priority=_PRIORITY_FOR_SEVERITY[event.severity],
                dedupe_key=(
                    f"bot:{bot.info.id}|{event.attention_key}"
                    if event.attention_key
                    else None
                ),
            )
            return
        self._alerts(event)  # a plain callable, for a simple wiring or a test

    def _alert_data(self, bot: Bot, event: Event) -> Dict[str, Any]:
        """The alert payload: who, how loud, where to land, plus the event's
        own flat scalars.  ``url`` is the deep link the notification opens --
        the bot's own ``href`` when the event did not give one.  Non-scalar
        entries in ``event.data`` are dropped rather than serialised; see
        :func:`_json_scalar`."""
        data: Dict[str, Any] = {
            "bot_id": bot.info.id,
            "severity": event.severity.name.lower(),
            "attention_key": event.attention_key,
            "url": event.href or bot.info.href,
        }
        for name, value in event.data.items():
            if name not in data and _json_scalar(value):
                data[name] = value
        return data

    # -- pausing --------------------------------------------------------------

    def pause(self, bot_id: str) -> None:
        """Switch a bot off until the owner switches it back on.

        contracts.py: "A paused bot is not ticked and reports no attention,
        because a badge asking you to act on something you switched off is a
        lie."  So this clears the bot's open items as well as stopping its
        ticks, and the pause is written through the store: a pause that
        forgets itself at the next restart is not a pause.
        """
        bot = self._registry.get(bot_id)
        if not getattr(bot.info, "can_pause", True):
            raise SupervisorError(
                f"{bot_id!r} declares can_pause=False; the launcher does not "
                f"offer a pause control for it"
            )
        if self.is_paused(bot_id):
            return
        self._paused[bot_id] = True
        self.clear_bot_attention(bot_id)
        _safe_hook(bot, "on_pause", self)
        self._persist()

    def resume(self, bot_id: str) -> None:
        """Switch a bot back on.  It becomes due immediately, unless it is
        still quarantined -- resuming is the owner saying "run this again",
        not an override of the supervisor's own back-off."""
        bot = self._registry.get(bot_id)
        if not self.is_paused(bot_id):
            return
        self._paused[bot_id] = False
        health = self._health_of(bot_id)
        if health.quarantined_until <= 0.0:
            health.next_due_at = float(self._clock())
        _safe_hook(bot, "on_resume", self)
        self._persist()

    def set_paused(self, bot_id: str, paused: bool) -> None:
        """The shape the launcher's ``POST /api/bots/pause`` body has:
        ``{"bot_id": ..., "paused": true}``."""
        if paused:
            self.pause(bot_id)
        else:
            self.resume(bot_id)

    # -- what the page and the badge read --------------------------------------

    def badge_status(self) -> Dict[str, Any]:
        """``GET /api/bots/status``: ``{"attention": int, "state": str}``.

        ``state`` is error when any bot is quarantined, warn when any is
        paused and none quarantined, else ok -- failure outranks pause,
        because one of them is the owner's decision and the other is not.

        ``attention`` is the number of *distinct open keys*, which is
        contracts.py's rule for the badge ("one restock nagging across ten
        ticks is one item of attention, not ten").  Note that
        ``web/README.md`` glosses the same number as "how many bots want a
        decision"; they agree for one open question per bot and differ when
        a bot has two.  The contract wins; :meth:`bots_wanting_attention`
        gives the other reading.
        """
        state = "ok"
        if self.quarantined_ids():
            state = "error"
        elif self.paused_ids():
            state = "warn"
        return {"attention": self.attention_count(), "state": state}

    def bot_state(self, bot_id: str) -> BotState:
        """The state the card shows: the supervisor's view wins over the
        bot's own, because pause and quarantine are the supervisor's facts."""
        if self.is_paused(bot_id):
            return BotState.PAUSED
        if self.is_quarantined(bot_id):
            return BotState.QUARANTINED
        try:
            status = self._registry.get(bot_id).status()
        except Exception:
            return BotState.QUARANTINED
        return status.state if isinstance(status, BotStatus) else BotState.IDLE

    def launcher_state(
        self, now: Optional[float] = None, *, include_detail: bool = False
    ) -> Dict[str, Any]:
        """``GET /api/bots/``: exactly the object ``web/README.md`` documents.

        Keys, and nothing else, because a key the page does not know is
        ignored in silence and a key it wants and does not get renders as
        blank::

            {"generated_at": int,
             "bots": [{"id", "name", "blurb", "kind", "state", "attention",
                       "href", "can_pause", "stats": [{"label", "value"}],
                       "last_event": {"at", "text"} | null}]}

        ``generated_at`` is what the page measures relative times against,
        "so a phone with a wrong clock still reads correctly"; it and
        ``last_event.at`` are whole seconds, as in the README's example.

        A ``status()`` that raises does not take the page down with it: that
        bot's card renders in the error state with no stats.  ``state`` is
        the launcher's own vocabulary (``BotState.ui``): idle, running,
        paused, error.

        ``include_detail`` adds ``detail`` (from
        :class:`~jarvis_bots.contracts.BotStatus`) and the event's
        ``severity`` and ``href``.  It is off by default because those keys
        are not in the documented contract; turn it on for a page that knows
        about them.
        """
        at = float(self._clock() if now is None else now)
        cards: List[Dict[str, Any]] = []
        for bot_id in self._registry.ids():
            bot = self._registry.find(bot_id)
            if bot is None:  # pragma: no cover - ids() came from the registry
                continue
            info = bot.info
            try:
                # The id on the card is the id the bot is *registered*
                # under, which is what health, pause and attention are
                # filed under.  A bot whose info.id has drifted from its
                # registration would otherwise render someone else's card.
                if getattr(info, "id", None) != bot_id:
                    raise ValueError(
                        f"registered as {bot_id!r} but its info now says "
                        f"{getattr(info, 'id', None)!r}; BotInfo is static identity"
                    )
                status = bot.status()
                if not isinstance(status, BotStatus):
                    raise TypeError(f"status() returned {type(status).__name__}")
                stats = [{"label": s.label, "value": s.value} for s in status.stats]
                detail = status.detail
                state = status.state
            except Exception as exc:
                stats, detail, state = [], _error_text(exc), BotState.QUARANTINED
            if self.is_paused(bot_id):
                state = BotState.PAUSED
            elif self.is_quarantined(bot_id):
                state = BotState.QUARANTINED
            # A paused bot keeps its last event on the card: contracts.py's
            # "reports no attention" is about being asked to act, and the
            # state pill already spells out "paused".  What it last found is
            # history, and history is what the owner reads to decide whether
            # to switch it back on.
            event = self._last_event.get(bot_id)
            card: Dict[str, Any] = {
                "id": bot_id,
                "name": getattr(info, "name", bot_id),
                "blurb": getattr(info, "blurb", ""),
                "kind": getattr(info, "kind", "bot"),
                "state": state.ui,
                "attention": self.attention_count(bot_id),
                "href": getattr(info, "href", ""),
                "can_pause": bool(getattr(info, "can_pause", True)),
                "stats": stats,
                "last_event": (
                    None if event is None else {"at": int(event.at), "text": event.text}
                ),
            }
            if include_detail:
                card["detail"] = detail
                if event is not None:
                    card["last_event"]["severity"] = event.severity.name.lower()
                    card["last_event"]["href"] = event.href
            cards.append(card)
        return {"generated_at": int(at), "bots": cards}

    # -- persistence -----------------------------------------------------------

    def save_state(self) -> Dict[str, Any]:
        """Snapshot every bot plus the supervisor's own bookkeeping, write it
        through the store if there is one, and return it.

        contracts.py: "State is the bot's, persistence is ours.  A bot hands
        over a JSON-able snapshot and gets it back on the next start."  What
        is ours and goes with it: health (or a restart resets a failing bot's
        back-off and it hammers again), the pause flags, the open attention
        items, and which of them have already been alerted -- otherwise every
        restart re-pushes every open request.

        A bot whose ``snapshot()`` raises does not stop the save: it is
        recorded in :attr:`last_state_error` and that bot is stored with the
        snapshot it had before, or none.
        """
        self._adopt_orphans()
        bots: Dict[str, Any] = {}
        for bot in self._registry.all():
            bot_id = bot.info.id
            try:
                snapshot = bot.snapshot()
                if not isinstance(snapshot, dict):
                    raise TypeError(f"snapshot() returned {type(snapshot).__name__}")
                self._last_snapshot[bot_id] = snapshot
            except BaseException as exc:
                self.last_state_error = f"{bot_id}: {_error_text(exc)}"
                # The last snapshot that worked, not {}.  The fallback used
                # to look in _orphans, which is empty for a bot that is
                # registered -- which is every bot this loop reaches -- so
                # one transient failure in snapshot() (a locked cache file,
                # a half-built object) silently replaced everything the bot
                # knew with nothing, and the next restart restored the
                # nothing.  Losing an update is recoverable; erasing the
                # watchlist is not.
                snapshot = self._last_snapshot.get(bot_id, {})
                if isinstance(exc, KeyboardInterrupt):
                    raise
            event = self._last_event.get(bot_id)
            bots[bot_id] = {
                "paused": self.is_paused(bot_id),
                "health": dataclasses.asdict(self._health_of(bot_id)),
                "snapshot": snapshot,
                "last_event": None if event is None else _event_to_dict(event),
                "events": [_event_to_dict(e) for e in self._events.get(bot_id, ())],
            }
        for bot_id, kept in self._orphans.items():
            # Not setdefault: _adopt_orphans has already moved anything the
            # registry now knows out of _orphans, so whatever is left here
            # belongs to an id with no live entry to lose to.
            if bot_id not in bots:
                bots[bot_id] = kept
        attention = [dataclasses.asdict(item) for item in self._attention.values()]
        alerted = [[owner, key, at] for (owner, key), at in self._alerted.items()]
        for pair, raw in self._orphan_attention.items():
            attention.append(
                {
                    "bot_id": pair[0],
                    "key": pair[1],
                    "text": str(raw.get("text", "")),
                    "href": str(raw.get("href", "")),
                    "since": float(raw.get("since", 0.0)),
                }
            )
            if raw.get("alerted_at") is not None:
                alerted.append([pair[0], pair[1], float(raw["alerted_at"])])
        state = {
            "version": STATE_VERSION,
            "saved_at": float(self._clock()),
            "bots": bots,
            "attention": attention,
            "alerted": alerted,
        }
        if self._store_save is not None:
            self._store_save(state)
        return state

    def load_state(self, state: Optional[Dict[str, Any]] = None) -> bool:
        """Restore what :meth:`save_state` wrote.  False if there was nothing.

        State for an id that is not registered now is *kept aside*, not
        dropped, and written out again by the next save: a bot taken out of
        the config for an afternoon should not come back having forgotten
        everything it knew.
        """
        if state is None:
            if self._store_load is None:
                raise SupervisorError("load_state() needs a store or an explicit state")
            state = self._store_load()
        if not state:
            return False
        if not isinstance(state, dict):
            raise SupervisorError(f"state must be an object; got {type(state).__name__}")
        version = state.get("version", STATE_VERSION)
        if not isinstance(version, int) or version > STATE_VERSION:
            raise SupervisorError(
                f"state was written by a newer version ({version!r} > "
                f"{STATE_VERSION}); refusing to guess at its shape"
            )
        bots = state.get("bots")
        if bots is None:
            bots = {}
        if not isinstance(bots, dict):
            raise SupervisorError(
                f"state['bots'] must be an object keyed by bot id; got "
                f"{type(bots).__name__}"
            )

        self._orphans = {}
        for bot_id, raw in bots.items():
            if not isinstance(raw, dict):
                raise SupervisorError(f"state for {bot_id!r} must be an object")
            bot = self._registry.find(bot_id)
            if bot is None:
                self._orphans[bot_id] = raw
                continue
            self._restore_one(bot, bot_id, raw)

        self._attention = {}
        self._orphan_attention = {}
        held: List[Tuple[str, str]] = []
        for raw in state.get("attention") or []:
            if not isinstance(raw, dict):
                raise SupervisorError("state['attention'] holds objects")
            owner = str(raw.get("bot_id", ""))
            key = str(raw.get("key", ""))
            if not owner or not key:
                continue
            if self._paused.get(owner):
                # contracts.py: "A paused bot is not ticked and reports no
                # attention."  pause() clears the items, but a file written
                # before that rule -- or one where the pause was restored
                # in the same load -- would hand them straight back, and
                # pause() early-returns for a bot that is already paused.
                continue
            if owner not in self._registry:
                # Kept aside with the rest of that bot's state rather than
                # dropped: the id may be registered a moment from now.
                self._orphan_attention[(owner, key)] = dict(raw)
                held.append((owner, key))
                continue
            self._attention[(owner, key)] = AttentionItem(
                bot_id=owner,
                key=key,
                text=str(raw.get("text", "")),
                href=str(raw.get("href", "")),
                since=float(raw.get("since", 0.0)),
            )
        # Alert memory only for questions that are still open; anything else
        # would grow without bound across restarts.
        self._alerted = {}
        for row in state.get("alerted") or []:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                continue
            pair = (str(row[0]), str(row[1]))
            if pair in self._attention:
                self._alerted[pair] = float(row[2])
            elif pair in self._orphan_attention:
                self._orphan_attention[pair]["alerted_at"] = float(row[2])
        # If the bots were registered before the load this is a no-op; if
        # they are registered after it, the next round or save picks them up.
        self._adopt_orphans()
        return True

    def _restore_one(self, bot: Bot, bot_id: str, raw: Dict[str, Any]) -> None:
        """Put one bot's persisted state back.  Shared by :meth:`load_state`
        and :meth:`_adopt_orphans`, so a bot registered after the load is
        restored by exactly the same code as one registered before it."""
        self._health[bot_id] = _health_from_dict(raw.get("health") or {})
        self._paused[bot_id] = bool(raw.get("paused"))
        snapshot = raw.get("snapshot") or {}
        if not isinstance(snapshot, dict):
            raise SupervisorError(f"snapshot for {bot_id!r} must be an object")
        self._last_snapshot[bot_id] = dict(snapshot)
        try:
            # Hooks are not fired here: on_pause/on_resume mark the moment
            # the owner flips the switch, and a restart is not that moment.
            bot.restore(snapshot)
        except Exception as exc:
            self.last_state_error = f"{bot_id}: {_error_text(exc)}"
        feed: "collections.deque[Event]" = collections.deque(maxlen=EVENT_HISTORY)
        for row in raw.get("events") or []:
            if not isinstance(row, dict):
                continue
            restored_event = _event_from_dict(bot_id, row)
            if restored_event is not None:
                feed.append(restored_event)
        if feed:
            self._events[bot_id] = feed
        event = raw.get("last_event")
        if isinstance(event, dict):
            restored = _event_from_dict(bot_id, event)
            if restored is not None:
                self._last_event[bot_id] = restored
        elif feed:
            self._last_event[bot_id] = feed[-1]

    def _persist(self) -> None:
        if self._store_save is not None:
            self.save_state()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"<Supervisor bots={len(self._registry)} attention="
            f"{self.attention_count()} state={self.badge_status()['state']}>"
        )


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


#: The floor ``BotInfo.__post_init__`` and ``registry.check_bot`` both
#: enforce, restated here because they enforce it once, at wiring time, and
#: this is read every round.
_MIN_INTERVAL_S = 5.0


def _interval_of(info: Any) -> float:
    """``info.interval_s``, floored and finite.

    Registration checks the floor; nothing re-checks it, and the value is
    read on every round.  A bot that lowers its own interval afterwards --
    or one whose interval arrives as a NaN out of a config -- would be
    ticked as fast as the scheduler runs, which is the one thing
    ``BotInfo`` says cannot happen.
    """
    try:
        interval = float(info.interval_s)
    except (TypeError, ValueError, AttributeError):
        return _MIN_INTERVAL_S
    if interval != interval:  # NaN: every "is it due" comparison is false
        return _MIN_INTERVAL_S
    return max(_MIN_INTERVAL_S, interval)


def _widened_interval(base_s: float, consecutive_failures: int) -> float:
    """:func:`~jarvis_bots.contracts.backoff_interval`, floored at ``base_s``.

    ``backoff_interval`` applied the cap after the floor and so returned
    *less* than base for a bot whose interval was above ``BACKOFF_CAP_S``
    -- a two-hour bot that failed once retried twice as often as when it
    was healthy.  That is fixed in contracts.py, where the defect was, and
    this is now exactly ``backoff_interval`` for every input.

    It stays because the floor is a scheduling invariant the supervisor
    owns ("The supervisor may widen this when the bot is failing, never
    narrow it"), and one line here is cheaper than trusting that nobody
    ever reorders those two calls again.
    """
    return max(float(base_s), backoff_interval(float(base_s), consecutive_failures))


def _is_resolution(event: Event) -> bool:
    """True when ``event`` says one open request for a decision is closed.

    All three conditions, because each rules out a different mistake:

    * a non-empty ``attention_key`` -- there is no closing a question that
      was never keyed;
    * ``data[RESOLVED_FLAG] is True`` exactly, not merely truthy: a bot
      carrying ``resolved="no"`` or ``resolved=0`` in its payload should
      not empty the badge, and ``is True`` is the one test that cannot be
      passed by accident;
    * below ACTION, so :attr:`~jarvis_bots.contracts.Event.wants_attention`
      is false.  An event that is asking for a decision is not also
      reporting one made, and this keeps the open path and the close path
      from ever both firing for one event.
    """
    return (
        bool(event.attention_key)
        and event.data.get(RESOLVED_FLAG) is True
        and not event.wants_attention
    )


def _error_text(exc: BaseException) -> str:
    """The one line a failure leaves on the card, safely.

    ``str(exc)`` runs the exception's own ``__str__``, which is a bot's code
    when the exception is a bot's class.  This is called from inside the
    failure handler, so an exception raised *there* would abort the round it
    is recording -- the failure that isolation exists to contain, escaping
    through the isolation.
    """
    try:
        text = f"{type(exc).__name__}: {exc}".strip()
    except BaseException:  # noqa: BLE001 - a __str__ of the bot's own
        text = f"{type(exc).__name__}: (its __str__ raised)"
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 1] + "…"


def _checked_events(bot_id: str, events: Any) -> Tuple[Event, ...]:
    """Validate what a tick returned, inside the tick's own try/except.

    A bot that returns something that is not a sequence of its own events is
    failing, and should be recorded as failing rather than corrupting the
    round: an event stamped with another bot's id would file attention, a
    card entry and a push under that bot.
    """
    if events is None:
        return ()
    if isinstance(events, Event):
        raise TypeError("tick() returns a sequence of events, not a single Event")
    if isinstance(events, (str, bytes)) or not isinstance(events, collections.abc.Iterable):
        raise TypeError(f"tick() must return a sequence of Events; got {type(events).__name__}")
    out: List[Event] = []
    for item in events:
        # Counted, and stopped.  ``Iterable`` includes a generator, and a
        # tick returning an endless one hung the round *here*, in the loop
        # written to keep a bad tick from corrupting it -- nothing raised,
        # so no try/except could help and no other bot ever ticked again.
        if len(out) >= MAX_EVENTS_PER_TICK:
            raise ValueError(
                f"tick() returned more than {MAX_EVENTS_PER_TICK} events; a tick "
                f"is one unit of work, and an unbounded return (a generator, a "
                f"runaway loop) would hold the round open for every other bot"
            )
        if not isinstance(item, Event):
            raise TypeError(f"tick() returned a {type(item).__name__}, not an Event")
        if item.bot_id != bot_id:
            raise ValueError(
                f"event is stamped {item.bot_id!r} but came from {bot_id!r}; use "
                f"BaseBot.event so the id is stamped for you"
            )
        _check_event_fields(item)
        out.append(item)
    return tuple(out)


def _check_event_fields(event: Event) -> None:
    """Re-check the fields ``Event.__post_init__`` checks.

    ``Event`` validates itself now, so in practice this never fires.  It is
    here because it is cheap and because the round is not a place to be
    trusting: ``dataclasses.replace`` on a mutated instance, a subclass that
    overrides ``__post_init__``, or ``object.__new__`` all produce an
    ``isinstance``-passing event with junk in it, and the two fields below
    are read *outside* any try/except the caller could add -- ``severity``
    in ``wants_attention`` and ``attention_key`` in ``str(...)``.
    """
    if not isinstance(event.severity, Severity):
        raise TypeError(
            f"event severity must be a Severity; got "
            f"{type(event.severity).__name__}"
        )
    if event.attention_key is not None and (
        not isinstance(event.attention_key, str) or not event.attention_key.strip()
    ):
        raise ValueError(
            f"attention_key must be a non-empty string or None; got "
            f"{type(event.attention_key).__name__}"
        )
    if not isinstance(event.text, str):
        raise TypeError(f"event text must be a string; got {type(event.text).__name__}")
    if not isinstance(event.data, dict):
        raise TypeError(f"event data must be a dict; got {type(event.data).__name__}")
    at = event.at
    if not isinstance(at, (int, float)) or isinstance(at, bool) or at != at:
        raise ValueError(f"event at must be unix seconds; got {at!r}")


def _safe_hook(bot: Bot, name: str, supervisor: "Supervisor") -> None:
    """Call ``on_pause``/``on_resume`` without letting it break the switch.

    The owner pressed pause; whether the bot's own cleanup worked does not
    change that it is paused.  A failure is recorded, not raised.
    """
    hook = getattr(bot, name, None)
    if hook is None:
        return
    try:
        hook()
    except Exception as exc:
        supervisor.last_state_error = f"{bot.info.id}.{name}: {_error_text(exc)}"


def _event_to_dict(event: Event) -> Dict[str, Any]:
    return {
        "at": float(event.at),
        "severity": int(event.severity),
        "text": event.text,
        "attention_key": event.attention_key,
        "href": event.href,
        "data": {k: v for k, v in event.data.items() if _json_scalar(v)},
    }


def _event_from_dict(bot_id: str, raw: Dict[str, Any]) -> Optional[Event]:
    try:
        return Event(
            bot_id=bot_id,
            at=float(raw.get("at", 0.0)),
            severity=parse_severity(raw.get("severity", int(Severity.INFO))),
            text=str(raw.get("text", "")),
            attention_key=raw.get("attention_key") or None,
            href=str(raw.get("href", "")),
            data=dict(raw.get("data") or {}),
        )
    except (TypeError, ValueError):
        return None
