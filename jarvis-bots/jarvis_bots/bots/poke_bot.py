"""The Pokemon buying assistant, as a bot on the framework.

:mod:`jarvis_bots.contracts` opens by saying the Pokemon assistant "is the
first one; the point of this module is that it is not special.  Adding the
second bot should be a class and a page, not a refactor."  This module is
the test of that claim: the whole of ``jarvis_poke`` -- a catalog, rules, a
price history, a decision engine and a polite poll scheduler -- reaches the
launcher through ``info``, :meth:`PokeBot.tick` and
:meth:`PokeBot.status`, plus the two persistence hooks.  There is no
poke-shaped hole anywhere in :mod:`jarvis_bots`, and nothing in this file
needed one.

What a tick does
----------------
1. Ask the scheduler what is *due* and poll only that.  The scheduler owns
   politeness (``jarvis_poke.contracts``: "Sources declare a minimum poll
   interval and are rate limited per host"), so this bot never decides to
   look at anything; it asks, and looks at what it is offered.
2. Feed every observation into the price history, which is what the market
   reference is computed from.
3. Evaluate the rules for the products that were polled, through the
   injected :class:`~jarvis_poke.engine.DecisionEngine`.
4. Turn each verdict into events.

What a tick never does
----------------------
``jarvis_poke.contracts``, "What this is not": "It does not check out."
:mod:`jarvis_bots.base`, on the same rule: "A bot that concludes the owner
should buy something raises an event carrying a link, and a person acts on
it."  The strongest thing this bot can do is raise an
:class:`~jarvis_bots.contracts.Event` at ``ACTION`` whose ``href`` is the
seller's own page.  Nothing here carts, pays, reserves a checkout slot or
calls the engine's spend-committing methods.

Events, and what the badge does with them
-----------------------------------------
contracts.py: "Events are the only output" and the badge "counts distinct
open keys, so one restock nagging across ten ticks is one item of
attention, not ten".

The key is ``buy|<product id>|<source>|<landed cents>``, which is the shape
:mod:`jarvis_poke.alerts_bridge` already dedupes on and for the same
reasons:

* a listing that flaps -- in stock, out, in again, which is what a restock
  looks like from outside -- comes back at the *same* key, so it is one
  item of attention and, through
  :func:`jarvis_alerts.outbox`'s window, one push;
* a genuine price change is a *different* key, which is the point: "the
  first alert said $54.99 and the owner passed; $44.99 is a different
  decision".  The bot resolves the old key in the same tick, so the badge
  shows the live offer and not both;
* the price in the key is the **landed** price, because that is the number
  the rule's ceiling is measured against.

One product holds at most one open key at a time: a product is one
decision.

Severity, chosen so the phone stays quiet
-----------------------------------------
:meth:`Supervisor._maybe_alert` pushes everything at or above
``alert_min_severity`` (``ACTION`` by default), and ``Severity.ERROR`` is
*above* ``ACTION``.  A failed poll raised at ERROR would therefore buzz a
phone every time a retailer returned a 503, with no attention key to dedupe
on.  So:

``ACTION``   a BUY: the owner should decide something.  Carries the key.
``NOTICE``   the feed.  A failed poll, a source coming back, a WATCH or a
             NO_STOCK that *changed*, and the resolution of an open key.
``ERROR``    a source the scheduler has actually paused, and an unexpected
             internal fault -- each raised once per episode, not per tick.

Failure is an event, not an exception
-------------------------------------
The assignment for this lane, and the reason: "the supervisor should not
have to quarantine a bot over one bad poll".
:meth:`jarvis_poke.sources.PollScheduler.poll_once` already swallows a
fetcher or parser that raises so that "one broken listing must not stop a
pass"; this bot wraps the injected pair so it can still *report* what
happened, and wraps the whole tick so that nothing reaches the supervisor
as an exception.  Only an exception's type name is kept, never its message,
which "may quote a URL or a page" -- the same rule ``jarvis_poke.sources``
holds itself to.

Injection, money and time
-------------------------
Every collaborator is injected, the fetcher and parser included: this
module opens no socket and knows no retailer's page structure.  Money is
integer cents throughout and is formatted exactly once, by
``jarvis_poke.contracts.fmt_cents``, on its way onto a
:class:`~jarvis_bots.contracts.Stat` -- ``BaseBot.stat`` "never does
arithmetic on it".  The clock is injected and ``tick`` uses the ``now`` it
is handed, "so every bot in a round agrees on when the round was".  No
randomness is drawn here at all; the scheduler's jitter comes from
``lucifer_gen.seed``, which is the only permitted source.

Closing an attention key
------------------------
``contracts.py`` gives a bot a way to open a standing request for a
decision and none to close one.  This bot follows the convention
``jarvis_bots/templates/bot.py.tmpl`` sets -- the resolution is a
``NOTICE`` carrying the same ``attention_key`` and ``resolved=True`` in its
data -- and the supervisor now reads exactly that flag (see
``jarvis_bots.supervisor.RESOLVED_FLAG``), so a sold-out listing leaves the
badge on its own with nothing wired after the round.

:func:`close_resolved` was the app-side workaround written while the
framework had no reader.  It is kept because it is harmless and still
correct -- clearing a key the supervisor has already cleared is a no-op --
and because an app driving a supervisor of its own may want the list of
keys a round closed.  New wiring does not need it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from dataclasses import replace as _replace

from jarvis_bots.base import BaseBot, Clock
from jarvis_bots.contracts import BotInfo, BotState, BotStatus, Event, Severity
from jarvis_poke.snipe import EventKind as SnipeEventKind
from jarvis_poke.contracts import (
    Action,
    FetchResult,
    Observation,
    SourceSku,
    Verdict,
    WatchState,
    fmt_cents,
)

__all__ = [
    "INFO",
    "SNAPSHOT_VERSION",
    "DEFAULT_OBSERVATION_TTL_S",
    "PokeBot",
    "PokeBotError",
    "attention_key_for",
    "close_resolved",
]


#: Static identity, "declared once" (contracts.py).  ``kind`` picks the
#: launcher's cart glyph, per ``jarvis_bots/web/README.md``; ``href`` is
#: what its Open button points at; ``interval_s`` is five minutes, which is
#: the shortest that can ever be useful given that
#: :class:`jarvis_poke.contracts.FetchPolicy` will not poll a host faster
#: than 300s anyway.
INFO = BotInfo(
    id="poke",
    name="Pokemon buying assistant",
    blurb=(
        "Watches sealed Pokemon product prices and stock, and tells you when "
        "one clears your rules. It links you to the listing; you buy it."
    ),
    kind="cart",
    interval_s=300.0,
    href="/bots/poke",
)

#: Bumped when :meth:`PokeBot.snapshot` changes shape.  A newer snapshot is
#: refused rather than half-read, because a half-read watch state is a
#: forgotten cooldown and a forgotten cooldown is a re-alert.
SNAPSHOT_VERSION = 1

#: How old an observation may be and still count as evidence about what is
#: on sale *now*.  A judgement call the contracts leave open: the history
#: keeps everything (that is what the market reference is built from), but
#: "this listing is in stock at $47.99" stops being true long before the
#: row stops being useful as a price sample.  An hour is twelve of this
#: bot's ticks and twelve of a source's minimum intervals.
DEFAULT_OBSERVATION_TTL_S = 3600.0

#: Reasons long enough, or URL-shaped enough, to be a page or a link are
#: dropped from an event rather than quoted.  ``jarvis_poke.sources`` keeps
#: "only the exception's *type name* ... never its message, which may quote
#: a URL or a page"; a fetcher's own ``reason`` string deserves the same
#: suspicion, since this bot did not write it.
MAX_REASON_CHARS = 60

#: The framework's tick floor (``BotInfo`` raises below 5s).  A window
#: interval can never be under 30s, so this is belt and braces -- but it is
#: the framework's number, not a second copy of it, because a bot that
#: proposed a 1s interval would be ticked as fast as the scheduler runs.
_MIN_TICK_S = 5.0


class PokeBotError(ValueError):
    """A wiring mistake: a collaborator that cannot do its job.

    Raised from ``__init__`` only.  contracts.py has the supervisor catch
    everything a *tick* raises, but a bot wired to an object with no
    ``evaluate`` is a programming error, and the moment to find out is the
    line that builds it -- the same stance
    :mod:`jarvis_bots.registry` takes.
    """


# --------------------------------------------------------------------------
# keys
# --------------------------------------------------------------------------


def attention_key_for(verdict: Verdict) -> Optional[str]:
    """``buy|<product>|<source>|<landed cents>``, or ``None``.

    ``None`` when the verdict names no listing -- NO_STOCK, or a SKIP taken
    before any listing was chosen.  There is nothing to be asked about, so
    there is no key.

    The landed price is in the key on purpose (see the module docstring and
    :mod:`jarvis_poke.alerts_bridge`, which keys its alerts the same way):
    it makes a price change a new question and a flap at one price the same
    question.  It is an ``int`` of cents, never formatted and never a
    float, so two runs agree on the key byte for byte.
    """
    if not isinstance(verdict, Verdict):
        raise PokeBotError(f"not a Verdict: {verdict!r}")
    if verdict.source is None or verdict.landed is None:
        return None
    return f"buy|{verdict.product_id}|{verdict.source}|{int(verdict.landed)}"


def close_resolved(supervisor: Any, events: Sequence[Event]) -> List[str]:
    """Close the attention keys the events in this round reported resolved.

    No longer required: ``Supervisor._apply_events`` reads the same
    ``resolved=True`` flag this bot sets (``supervisor.RESOLVED_FLAG``), so
    a round closes its own keys.  Against that supervisor this returns an
    empty list, because the keys were closed before it was called.

    It is kept for an app driving a supervisor of its own, or one that
    wants the keys a round closed::

        report = supervisor.run_round(now)
        closed = close_resolved(supervisor, bot.drain_resolved())

    Returns the keys that were open and are now closed.  Events without the
    flag are ignored, so this is safe to hand every event of a round, and
    clearing an already-cleared key is a no-op.
    """
    closed: List[str] = []
    for event in events:
        if not isinstance(event, Event):
            continue
        if not event.data.get("resolved"):
            continue
        key = event.attention_key
        if not key:
            continue
        if supervisor.clear_attention(event.bot_id, key):
            closed.append(key)
    return closed


def _safe_reason(text: Any) -> str:
    """A fetcher's ``reason``, or nothing, on the side of nothing."""
    if not isinstance(text, str):
        return ""
    cleaned = " ".join(text.split())
    if not cleaned or len(cleaned) > MAX_REASON_CHARS or "://" in cleaned:
        return ""
    return cleaned


def _pct(value: Optional[float]) -> str:
    """A discount as the card and the alert spell it."""
    return "" if value is None else f"{value:.1f}% off"


# --------------------------------------------------------------------------
# the bot
# --------------------------------------------------------------------------


class PokeBot(BaseBot):
    """``jarvis_poke`` on the bot framework.

    ``catalog``    a :class:`jarvis_poke.catalog.Catalog`: what may be
                   watched and where it is listed.  Used for product names
                   on the card and in the alert, and nothing else -- the
                   engine holds its own reference.
    ``rules``      a :class:`jarvis_poke.rules.RuleSet`: the owner's
                   ceilings and the budget ledger.
    ``history``    a :class:`jarvis_poke.prices.PriceHistory`: every
                   observation, and the market reference built from them.
    ``engine``     a :class:`jarvis_poke.engine.DecisionEngine`.
    ``scheduler``  a :class:`jarvis_poke.sources.PollScheduler`.  It, not
                   this bot, decides what may be polled.
    ``fetcher``    injected, per ``jarvis_poke.contracts``: "The package
                   makes no network calls itself: the app injects a
                   fetcher, exactly as the alert transports take an
                   injected sender."  So does this bot.
    ``parser``     injected for the same reason; one per source is the
                   app's business.
    ``clock``      required by :class:`~jarvis_bots.base.BaseBot`.  It must
                   be *the same callable* the engine was given: the engine
                   stamps its own verdicts from its own clock, so two
                   clocks means two ideas of when the cooldown started.
                   Checked at construction, loosely, because that is the
                   only moment both can be read side by side.

    ``observation_ttl_s`` is how far back a tick looks for evidence that a
    listing is on sale right now; see
    :data:`DEFAULT_OBSERVATION_TTL_S`.

    Not thread safe, like the supervisor that drives it: a round is a loop.
    """

    info = INFO

    def __init__(
        self,
        catalog: Any,
        rules: Any,
        history: Any,
        engine: Any,
        scheduler: Any,
        fetcher: Any,
        parser: Any,
        clock: Clock,
        *,
        info: Optional[BotInfo] = None,
        observation_ttl_s: float = DEFAULT_OBSERVATION_TTL_S,
        snipe: Any = None,
    ) -> None:
        super().__init__(clock, info)

        self._catalog = _needs(catalog, ("find",), "catalog")
        self._rules = _needs(rules, ("get", "enabled_rules", "remaining"), "rules")
        self._history = _needs(history, ("append", "for_product"), "history")
        self._engine = _needs(engine, ("evaluate", "watch_state"), "engine")
        self._scheduler = _needs(scheduler, ("due", "poll_once"), "scheduler")
        if not callable(fetcher):
            raise PokeBotError(
                "fetcher must be callable(url, headers, policy) -> FetchResult; "
                f"got {type(fetcher).__name__}. This package opens no sockets."
            )
        if not callable(parser):
            raise PokeBotError(
                "parser must be callable(sku, body, at) -> Observation; got "
                f"{type(parser).__name__}. This package parses no retailer's page."
            )
        self._fetcher = fetcher
        self._parser = parser

        ttl = float(observation_ttl_s)
        if ttl <= 0.0:
            raise PokeBotError("observation_ttl_s must be positive")
        self.observation_ttl_s = ttl

        self._check_shared_clock()

        #: product id -> the attention key currently open for it.  One
        #: product is one decision, so one key.
        self._open: Dict[str, str] = {}
        #: attention key -> when it was first raised, for the snapshot.
        self._open_since: Dict[str, float] = {}
        #: product id -> the last verdict's shape, for the card and for
        #: deciding whether a WATCH is news.  Small and JSON-able.
        self._current: Dict[str, Dict[str, Any]] = {}
        #: source id -> whether the scheduler had it paused last tick.
        self._source_paused: Dict[str, bool] = {}
        #: The last internal fault reported at ERROR, so a bot that is
        #: broken says so once rather than every five minutes.
        self._last_fault = ""
        #: Events this tick reported resolved, for :meth:`drain_resolved`.
        self._resolved: List[Event] = []

        #: Optional :class:`jarvis_poke.snipe.SnipeController`.  Without
        #: one this bot behaves exactly as before; with one it watches
        #: harder inside a drop window.  Duck-typed rather than imported
        #: as a type, so a test can pass a double and the framework keeps
        #: its one-way dependency on jarvis_poke.
        if snipe is not None:
            for method in ("sync", "active_window"):
                if not callable(getattr(snipe, method, None)):
                    raise PokeBotError(
                        f"snipe has no {method}(); expected a "
                        f"jarvis_poke.snipe.SnipeController"
                    )
        self._snipe = snipe
        #: The tick interval to go back to when a window closes.  Captured
        #: from ``info`` rather than from :data:`INFO`, so a bot given a
        #: custom interval keeps it.
        self._base_interval_s = float(self.info.interval_s)
        #: The name of the window whose tick rate is installed, or "".
        self._tick_window = ""

        self._ticks = 0
        self._polls = 0
        self._poll_failures = 0
        self._observations = 0
        self._last_tick_at = 0.0

    # -- the work ----------------------------------------------------------

    def tick(self, now: float) -> Sequence[Event]:
        """One polite pass: poll what is due, decide, and say what changed.

        contracts.py: "Do one unit of work.  Must return promptly and must
        not sleep."  Nothing here sleeps or retries; a source that is not
        due is simply not looked at until it is.

        This never raises.  contracts.py allows it and the supervisor
        handles it, but the failures this bot actually meets -- a retailer
        returning a 503, a page whose parser threw -- are ordinary, and
        four of them in a row would quarantine the bot for half an hour
        over nothing.  They come back as events instead.
        """
        self._resolved = []
        try:
            return self._tick(float(now))
        except Exception as exc:  # noqa: BLE001 - deliberate; see the docstring
            return (self._fault_event(exc),)

    def _tick(self, now: float) -> Tuple[Event, ...]:
        events: List[Event] = []
        self._ticks += 1
        self._last_tick_at = now

        events.extend(self._sync_snipe(now))

        due = list(self._scheduler.due(now))
        for sku in due:
            events.extend(self._poll(sku, now))

        for product_id in sorted({sku.product_id for sku in due}):
            events.extend(self._decide(product_id, now))

        events.extend(self._source_health(now))
        return tuple(events)

    # -- drop windows --------------------------------------------------------

    def _sync_snipe(self, now: float) -> List[Event]:
        """Let the snipe controller open and close windows, and follow it.

        Two rates move, and they are not the same rate:

        * the **poll** rate, which the controller sets on the scheduler and
          which the scheduler's own 30s floor governs;
        * this bot's **tick** rate, which decides how often anything is
          even asked.  Tightening the first without the second buys
          nothing: a scheduler willing to poll every 30s that is asked
          once every 300s still polls every 300s.

        So the tick rate follows the window too, and it follows it from
        *arm* time rather than open time.  The supervisor reads
        ``bot.info`` before the tick and computes the next due time from
        that copy, so a tighten during the tick lands one round late --
        which, at a 300s round, is the whole first five minutes of the
        drop.  Arming ten minutes early costs a handful of extra ticks
        against a scheduler that is still refusing to poll, and buys a bot
        that is already awake when the window opens.

        A controller that raises is not allowed to stop the round: the
        polling below is the bot's actual job and works without any of
        this.  The failure comes back as an event.
        """
        if self._snipe is None:
            return []
        events: List[Event] = []
        try:
            for change in self._snipe.sync(now):
                events.append(self._snipe_event(change, now))
        except Exception as exc:  # noqa: BLE001 - see the docstring
            return [
                Event(
                    at=now,
                    bot_id=self.info.id,
                    severity=Severity.ERROR,
                    text=(
                        f"drop-window controller failed ({type(exc).__name__}); "
                        f"watching at the normal rate"
                    ),
                )
            ]
        events.extend(self._follow_tick_rate(now))
        return events

    def _snipe_event(self, change: Any, now: float) -> Event:
        """One controller event as a framework event.

        Only ``NOT_READY`` carries an attention key: it is the one the
        owner has to act on, and it has to survive in the badge until they
        do.  Opening and closing a window is bookkeeping -- worth a feed
        line, not a badge item -- and a badge that fills up with "window
        opened" is a badge nobody reads on the day it matters.
        """
        kind = getattr(change, "kind", None)
        detail = _safe_reason(getattr(change, "detail", ""))
        if kind is SnipeEventKind.NOT_READY:
            return Event(
                at=now,
                bot_id=self.info.id,
                severity=Severity.ACTION,
                text=f"Drop window not ready: {detail}",
                attention_key=f"snipe:not-ready:{change.window}",
                href=self.info.href,
            )
        if kind is SnipeEventKind.ARMED:
            # The same key, resolved: an owner who fixed the push
            # subscription after the first warning gets the badge back.
            return Event(
                at=now,
                bot_id=self.info.id,
                severity=Severity.NOTICE,
                text=f"Drop window armed: {detail}",
                attention_key=f"snipe:not-ready:{change.window}",
                data={"resolved": True},
            )
        severity = Severity.NOTICE if kind is SnipeEventKind.OPENED else Severity.INFO
        return Event(at=now, bot_id=self.info.id, severity=severity, text=detail)

    def _follow_tick_rate(self, now: float) -> List[Event]:
        """Match this bot's own tick interval to the window in force."""
        window = self._next_window(now)
        wanted_name = window.name if window is not None else ""
        if wanted_name == self._tick_window:
            return []
        if window is not None:
            interval = max(_MIN_TICK_S, float(window.interval_s))
        else:
            interval = self._base_interval_s
        try:
            self.info = _replace(self.info, interval_s=interval)
        except ValueError:
            return []
        self._tick_window = wanted_name
        return [
            Event(
                at=now,
                bot_id=self.info.id,
                severity=Severity.INFO,
                text=(
                    f"checking every {interval:.0f}s for {wanted_name}"
                    if wanted_name else
                    f"back to checking every {interval:.0f}s"
                ),
            )
        ]

    def _next_window(self, now: float) -> Any:
        """The window this bot should already be awake for: one that is
        open, or one that is arming.  Asks the controller's plan across
        every source it knows, because a bot ticks for all of them."""
        plan = getattr(self._snipe, "plan", None)
        if plan is None:
            return None
        candidates = [
            w for w in plan.windows
            if w.arms_at <= now < w.closes_at
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda w: (w.interval_s, w.name))

    # -- polling -----------------------------------------------------------

    def _poll(self, sku: SourceSku, now: float) -> List[Event]:
        """One look at one listing, through the scheduler.

        The injected fetcher and parser are wrapped, not replaced: the
        wrappers re-raise so that
        :meth:`~jarvis_poke.sources.PollScheduler.poll_once` still records
        the failure against the source and widens its backoff.  All the
        wrappers add is a note of *what* went wrong, so the owner sees a
        line in the feed instead of a bot that quietly stops finding
        things.
        """
        trouble: List[str] = []

        def fetch(url: str, headers: Dict[str, str], policy: Any) -> Any:
            try:
                result = self._fetcher(url, headers, policy)
            except Exception as exc:  # noqa: BLE001 - reported, then re-raised
                trouble.append(f"the fetcher raised {type(exc).__name__}")
                raise
            if isinstance(result, FetchResult) and not result.ok:
                reason = _safe_reason(result.reason)
                status = f"HTTP {result.status}" if result.status else "no response"
                trouble.append(f"{status}{f' ({reason})' if reason else ''}")
            return result

        def parse(entry: SourceSku, body: str, at: float) -> Any:
            try:
                return self._parser(entry, body, at)
            except Exception as exc:  # noqa: BLE001 - reported, then re-raised
                trouble.append(f"the parser raised {type(exc).__name__}")
                raise

        self._polls += 1
        observation = self._scheduler.poll_once(sku, fetch, parse, now)
        if isinstance(observation, Observation):
            self._observations += 1
            self._history.append(observation)

        if not trouble:
            return []
        self._poll_failures += 1
        # NOTICE, not ERROR: one bad poll is a line in the feed, not a
        # reason to buzz a phone (see the module docstring on severity).
        return [
            self.event(
                Severity.NOTICE,
                f"Could not read {self._name(sku.product_id)} at "
                f"{self._source_label(sku.source)}: {'; '.join(trouble)}",
                href=self.info.href,
                product_id=sku.product_id,
                source=sku.source,
            )
        ]

    # -- deciding ----------------------------------------------------------

    def _decide(self, product_id: str, now: float) -> List[Event]:
        """Evaluate one product and turn the verdict into events."""
        observations = self._recent(product_id, now)
        verdict = self._engine.evaluate(product_id, observations)
        if not isinstance(verdict, Verdict):
            raise PokeBotError(
                f"engine.evaluate returned {type(verdict).__name__}, not a Verdict"
            )
        return self._events_for(verdict)

    def _recent(self, product_id: str, now: float) -> List[Observation]:
        """Evidence about what is on sale *now*.

        The engine takes the newest row per listing, so handing it the
        whole history would let a three-week-old "in stock at $39.99"
        decide a purchase.  The window is
        :attr:`observation_ttl_s`; ``for_product``'s ``since`` is an
        inclusive lower bound, which is what we want -- a row stamped
        exactly at the edge is still evidence.
        """
        return list(
            self._history.for_product(product_id, since=now - self.observation_ttl_s)
        )

    def _events_for(self, verdict: Verdict) -> List[Event]:
        """The whole attention lifecycle for one product, in one place.

        The rules, restated as code:

        * a BUY at a key that is not open raises ``ACTION`` and opens it;
        * a BUY at the key already open raises nothing -- contracts.py has
          the badge collapse repeats, and re-raising would also re-alert
          the moment anything cleared the key;
        * any verdict that no longer *evidences* the open key resolves it:
          gone from the shelf, or on the shelf at a different price.  A
          SKIP taken inside the rule's cooldown still names the same
          listing at the same landed price, so it keeps the key open --
          the offer has not changed, only our willingness to shout about
          it again;
        * WATCH and NO_STOCK earn at most one ``NOTICE``, and only when
          they are news.  A resolution is itself that notice, so the two
          never double up.  A SKIP says nothing: "rule disabled, budget
          gone, cooldown" is not an event, it is a state, and it is on the
          card.
        """
        events: List[Event] = []
        pid = verdict.product_id
        key = attention_key_for(verdict)
        open_key = self._open.get(pid)
        previous = self._current.get(pid, {})

        # The key is still evidenced only when this verdict names the same
        # listing at the same landed price *and* that listing is one the
        # engine considered buyable.
        still_open = open_key is not None and key == open_key

        if open_key is not None and not still_open:
            events.append(self._resolve(pid, open_key, verdict))
            open_key = None

        if verdict.action is Action.BUY:
            if open_key is None and key is not None:
                self._open[pid] = key
                self._open_since[key] = verdict.at
                events.append(self._buy_event(verdict, key))
        elif verdict.action in (Action.WATCH, Action.NO_STOCK) and not events:
            note = self._change_note(verdict, previous)
            if note is not None:
                events.append(note)

        self._current[pid] = {
            "action": verdict.action.value,
            "source": verdict.source,
            "landed": verdict.landed,
            "discount_pct": verdict.discount_pct,
            "at": verdict.at,
        }
        return events

    def _buy_event(self, verdict: Verdict, key: str) -> Event:
        """The one event in this package that asks the owner for a decision.

        ``ACTION``, because contracts.py defines it as "the owner should
        decide something: this drives the badge".  ``href`` is the
        *listing*, so the alert deep links to the page the owner completes
        the purchase on -- which is where this tool's involvement ends.
        """
        landed = int(verdict.landed or 0)
        discount = _pct(verdict.discount_pct)
        text = (
            f"Buy: {self._name(verdict.product_id)} at {fmt_cents(landed)} "
            f"from {self._source_label(verdict.source or '')}"
        )
        if discount and verdict.market:
            text += f", {discount} the {fmt_cents(verdict.market)} market price"
        elif discount:
            text += f", {discount}"
        else:
            text += " (no market reference yet; your ceiling decided)"
        return self.event(
            Severity.ACTION,
            text,
            attention_key=key,
            href=verdict.url or self.info.href,
            product_id=verdict.product_id,
            source=verdict.source,
            sku=verdict.sku,
            url=verdict.url,
            price_cents=landed,
            market_cents=verdict.market,
            discount_pct=verdict.discount_pct,
            quantity=verdict.quantity,
        )

    def _resolve(self, product_id: str, key: str, verdict: Verdict) -> Event:
        """Close one open request, in the shape the framework expects.

        ``NOTICE`` -- below ACTION, so ``Event.wants_attention`` is false
        and the supervisor does not re-open the key -- carrying that same
        key and ``resolved=True``.  That is verbatim the convention
        ``jarvis_bots/templates/bot.py.tmpl`` sets: "a supervisor that
        closes keys on the flag and one that closes them on the severity
        agree".  :class:`~jarvis_bots.supervisor.Supervisor` closes on the
        flag (``RESOLVED_FLAG``), so this event empties the badge by
        itself.
        """
        self._open.pop(product_id, None)
        self._open_since.pop(key, None)
        if verdict.action is Action.NO_STOCK:
            why = "it is out of stock"
        elif verdict.landed is not None:
            why = f"the price is now {fmt_cents(int(verdict.landed))}"
        else:
            why = "the listing no longer clears your rules"
        event = self.event(
            Severity.NOTICE,
            f"No longer a buy: {self._name(product_id)} -- {why}",
            attention_key=key,
            href=self.info.href,
            product_id=product_id,
            resolved=True,
        )
        self._resolved.append(event)
        return event

    def _change_note(
        self, verdict: Verdict, previous: Mapping[str, Any]
    ) -> Optional[Event]:
        """A WATCH or NO_STOCK worth a line, or ``None``.

        News is a change of action, or a change of the price we are
        watching.  Repeating "still out of stock" every five minutes is
        not news; :mod:`jarvis_bots.templates` puts it plainly: "A quiet
        bot should be quiet."
        """
        if (
            previous.get("action") == verdict.action.value
            and previous.get("landed") == verdict.landed
            and previous.get("source") == verdict.source
        ):
            return None
        name = self._name(verdict.product_id)
        if verdict.action is Action.NO_STOCK:
            text = f"Out of stock: {name}"
        else:
            landed = fmt_cents(int(verdict.landed or 0))
            rule = self._rules.get(verdict.product_id)
            ceiling = f", over your {fmt_cents(rule.max_price)} ceiling" if rule else ""
            text = (
                f"Watching {name}: {landed} at "
                f"{self._source_label(verdict.source or '')}{ceiling}"
            )
        return self.event(
            Severity.NOTICE,
            text,
            href=verdict.url or self.info.href,
            product_id=verdict.product_id,
            source=verdict.source,
            price_cents=verdict.landed,
        )

    # -- source health -----------------------------------------------------

    def _source_health(self, now: float) -> List[Event]:
        """One event when the scheduler pauses a source, one when it is back.

        A paused source is the failure the owner cannot see any other way:
        the bot goes on ticking, healthily, and simply stops finding
        anything at that retailer.  Raised at ERROR because it is worth a
        push, and *once per episode* because the pause lasts an hour and
        this bot ticks twelve times inside one.
        """
        pause_state = getattr(self._scheduler, "pause_state", None)
        if not callable(pause_state):
            return []
        events: List[Event] = []
        for source, row in sorted(pause_state(now).items()):
            paused = bool(row.get("paused"))
            was = self._source_paused.get(source, False)
            self._source_paused[source] = paused
            if paused and not was:
                seconds = int(row.get("seconds_remaining") or 0)
                reason = _safe_reason(row.get("reason")) or "repeated failures"
                events.append(
                    self.event(
                        Severity.ERROR,
                        f"Not watching {self._source_label(source)} for the next "
                        f"{seconds // 60} minutes: {reason}",
                        href=self.info.href,
                        source=source,
                    )
                )
            elif was and not paused:
                events.append(
                    self.event(
                        Severity.NOTICE,
                        f"Watching {self._source_label(source)} again",
                        href=self.info.href,
                        source=source,
                    )
                )
        return events

    def _fault_event(self, exc: BaseException) -> Event:
        """An unexpected internal failure, reported rather than raised.

        ERROR the first time a given fault appears, NOTICE for a repeat of
        the same one: a bot that is broken should say so, and then stop
        shouting.  Only the type name is kept -- the same rule
        ``jarvis_poke.sources`` applies to a fetcher that raises.
        """
        signature = type(exc).__name__
        repeat = signature == self._last_fault
        self._last_fault = signature
        return self.event(
            Severity.NOTICE if repeat else Severity.ERROR,
            f"The Pokemon watcher could not finish a pass ({signature}). "
            f"Nothing was bought and nothing was missed permanently; the next "
            f"pass will look again.",
            href=self.info.href,
            fault=signature,
        )

    # -- the card ----------------------------------------------------------

    def status(self) -> BotStatus:
        """What the launcher renders.  Cheap, and it does not raise.

        IDLE until the first tick, RUNNING after: PAUSED and QUARANTINED
        are the supervisor's facts about this bot, not the bot's, and
        :meth:`Supervisor.launcher_state` overrides the state it owns.

        The three figures are the ones that answer "is this worth opening"
        at a glance: how much is being watched, how much of the budget is
        left, and how close anything currently is to a deal.  Every value
        is a pre-formatted string -- contracts.py: "the bot knows how its
        own numbers should read, and the page should not be doing money
        maths" -- and the money is formatted from integer cents by
        ``jarvis_poke.contracts.fmt_cents``, never divided here.
        """
        try:
            watched = len(self._rules.enabled_rules())
            remaining = fmt_cents(int(self._rules.remaining()))
        except Exception as exc:  # noqa: BLE001 - a card must never take the page down
            return BotStatus(
                BotState.RUNNING if self._ticks else BotState.IDLE,
                detail=f"rules unavailable ({type(exc).__name__})",
            )

        best = self.best_discount()
        stats = (
            self.stat("Watching", f"{watched} product{'' if watched == 1 else 's'}"),
            self.stat("Budget left", remaining),
            self.stat("Best discount", _pct(best) if best is not None else "none yet"),
        )
        return BotStatus(
            BotState.RUNNING if self._ticks else BotState.IDLE,
            stats=stats,
            detail=self._detail(),
        )

    def best_discount(self) -> Optional[float]:
        """The deepest discount among listings currently on the shelf.

        Only BUY and WATCH count: a discount on something out of stock is
        not a discount on anything.  ``None`` when nothing has a usable
        market reference yet, which is honest -- contracts.py's engine
        makes the same distinction, reporting "no idea" as an absent
        number rather than a zero.
        """
        live = [
            row["discount_pct"]
            for row in self._current.values()
            if row.get("action") in (Action.BUY.value, Action.WATCH.value)
            and isinstance(row.get("discount_pct"), (int, float))
        ]
        return max(live) if live else None

    def _detail(self) -> str:
        open_now = len(self._open)
        if open_now:
            return f"{open_now} waiting on you"
        if not self._ticks:
            return "not looked yet"
        return f"{self._observations} listing reads over {self._ticks} passes"

    # -- attention, for the app --------------------------------------------

    def open_attention_keys(self) -> List[str]:
        """The keys this bot believes are open, sorted.

        The supervisor owns the badge; this is the bot's own record.  The
        two agree by construction now that the supervisor closes a key on
        the resolution event this bot raises.  Exposed so a wiring can
        check that.
        """
        return sorted(self._open.values())

    def drain_resolved(self) -> List[Event]:
        """The resolution events from the most recent tick.

        For a caller that wants to close keys without sifting every event
        of a round.  Cleared at the start of each tick, not by reading, so
        reading it twice gives the same answer.
        """
        return list(self._resolved)

    # -- the framework's optional half --------------------------------------

    def on_pause(self) -> None:
        """contracts.py: "Paused means paused ... a badge asking you to act
        on something you switched off is a lie."

        The supervisor clears this bot's attention items when it is paused.
        If the bot kept its own record it would then believe those requests
        were still open and would never re-raise them on resume, so the
        badge would stay empty over a live deal.  Forgetting them here is
        what makes resume honest: the next tick re-raises whatever is still
        a BUY.
        """
        self._open.clear()
        self._open_since.clear()
        self._resolved = []

    def snapshot(self) -> Dict[str, Any]:
        """JSON-able state: "State is the bot's, persistence is ours."

        What is here, and why:

        * **watch states** -- the engine's cooldowns and alert counts.
          Lose these and a restart re-alerts every open deal.
        * **open attention keys** -- so a restart can still *resolve* a
          request it raised before the restart, instead of leaving it in
          the badge forever.
        * **the last verdict per product** -- what the card shows, so a
          freshly restarted bot is not blank for five minutes.
        * **the scheduler's own snapshot** -- "everything a restart needs
          to stay as polite as it was".  It is included even though
          :class:`~jarvis_poke.sources.PollScheduler` has its own store
          seam, because the supervisor takes this snapshot after the round
          and the scheduler writes its store during one, so this copy is
          never the older of the two.

        What is deliberately *not* here: the price history, which is large,
        is the ``HistoryStore`` seam's job, and is a record of the world
        rather than of this bot; and the engine's reservations, which refer
        to deep links already on a phone and which the engine itself offers
        no way to restore. Both are noted in the report for this lane.
        """
        states = getattr(self._engine, "watch_states", {}) or {}
        return {
            "version": SNAPSHOT_VERSION,
            "ticks": self._ticks,
            "polls": self._polls,
            "poll_failures": self._poll_failures,
            "observations": self._observations,
            "last_tick_at": self._last_tick_at,
            "watch_states": {
                pid: _watch_state_to_obj(state) for pid, state in sorted(states.items())
            },
            "attention": [
                {
                    "product_id": pid,
                    "key": key,
                    "since": self._open_since.get(key, 0.0),
                }
                for pid, key in sorted(self._open.items())
            ],
            "current": {pid: dict(row) for pid, row in sorted(self._current.items())},
            "source_paused": dict(sorted(self._source_paused.items())),
            "scheduler": _scheduler_snapshot(self._scheduler),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Take back what :meth:`snapshot` returned.

        A snapshot from a newer build is refused rather than half-read: a
        watch state read half-way is a forgotten cooldown, and a forgotten
        cooldown is the same alert again. Anything else missing is treated
        as absent, so a snapshot from an older build still opens.

        The engine's ``watch_states`` mapping is updated *in place*,
        because :class:`~jarvis_poke.engine.DecisionEngine` documents that
        it "is used in place, so a caller that keeps it in a store sees the
        engine's updates" -- replacing the object would leave the engine
        writing to the old one.
        """
        if not isinstance(snapshot, Mapping) or not snapshot:
            return
        version = snapshot.get("version", SNAPSHOT_VERSION)
        if not isinstance(version, int) or version > SNAPSHOT_VERSION:
            raise PokeBotError(
                f"poke bot snapshot version {version!r} is newer than this build "
                f"understands (version {SNAPSHOT_VERSION})"
            )

        self._ticks = int(snapshot.get("ticks") or 0)
        self._polls = int(snapshot.get("polls") or 0)
        self._poll_failures = int(snapshot.get("poll_failures") or 0)
        self._observations = int(snapshot.get("observations") or 0)
        self._last_tick_at = float(snapshot.get("last_tick_at") or 0.0)

        states = getattr(self._engine, "watch_states", None)
        if isinstance(states, dict):
            states.clear()
            for pid, row in (snapshot.get("watch_states") or {}).items():
                states[str(pid)] = _watch_state_from_obj(str(pid), row)

        self._open = {}
        self._open_since = {}
        for row in snapshot.get("attention") or []:
            pid, key = str(row.get("product_id") or ""), str(row.get("key") or "")
            if pid and key:
                self._open[pid] = key
                self._open_since[key] = float(row.get("since") or 0.0)

        self._current = {
            str(pid): dict(row)
            for pid, row in (snapshot.get("current") or {}).items()
            if isinstance(row, Mapping)
        }
        self._source_paused = {
            str(name): bool(flag)
            for name, flag in (snapshot.get("source_paused") or {}).items()
        }

        scheduler_state = snapshot.get("scheduler")
        restore = getattr(self._scheduler, "restore", None)
        if scheduler_state and callable(restore):
            restore(scheduler_state)

    # -- small helpers ------------------------------------------------------

    def _name(self, product_id: str) -> str:
        """The owner's name for a product, falling back to its id.

        A name is what makes an alert readable; an id is what makes it
        unambiguous when the catalog has drifted out from under a rule.
        """
        product = self._catalog.find(product_id) if self._catalog is not None else None
        return getattr(product, "name", None) or product_id

    def _source_label(self, source: str) -> str:
        label = getattr(self._catalog, "source_label", None)
        return label(source) if callable(label) else source

    def _check_shared_clock(self) -> None:
        """Refuse an engine running on a different clock.

        :meth:`DecisionEngine.evaluate` takes no ``now``; it reads its own
        injected clock and stamps the verdict with it.  This bot stamps its
        events from its own.  Two clocks would put the cooldown, the
        market window and the event feed on different timelines, and the
        symptom -- an alert that repeats, or one that never comes -- would
        look like a rules bug.  Construction is the one moment both can be
        compared, so it is compared here.
        """
        engine_now = getattr(self._engine, "now", None)
        if not callable(engine_now):
            return
        try:
            drift = abs(float(engine_now()) - self.now())
        except Exception:  # noqa: BLE001 - a stub engine is not a wiring error
            return
        if drift > 5.0:
            raise PokeBotError(
                f"the engine's clock and the bot's disagree by {drift:.0f}s: inject "
                f"the same callable into both, or the rule cooldowns and the event "
                f"feed are on different timelines"
            )


# --------------------------------------------------------------------------
# module-private helpers
# --------------------------------------------------------------------------


def _needs(obj: Any, methods: Sequence[str], what: str) -> Any:
    """Check a collaborator at wiring time, naming what is missing."""
    if obj is None:
        raise PokeBotError(f"PokeBot needs a {what}")
    missing = [name for name in methods if not callable(getattr(obj, name, None))]
    if missing:
        raise PokeBotError(
            f"{what} has no {', '.join(f'{m}()' for m in missing)}: got "
            f"{type(obj).__name__}"
        )
    return obj


def _watch_state_to_obj(state: Any) -> Dict[str, Any]:
    """One :class:`~jarvis_poke.contracts.WatchState` as plain JSON."""
    action = getattr(state, "last_action", None)
    return {
        "last_alert_at": float(getattr(state, "last_alert_at", 0.0) or 0.0),
        "last_action": getattr(action, "value", None),
        "alerts_sent": int(getattr(state, "alerts_sent", 0) or 0),
        "last_seen_in_stock": float(getattr(state, "last_seen_in_stock", 0.0) or 0.0),
    }


def _watch_state_from_obj(product_id: str, row: Any) -> WatchState:
    """The inverse.  An unknown action becomes ``None`` rather than a guess."""
    row = row if isinstance(row, Mapping) else {}
    raw = row.get("last_action")
    try:
        action = Action(raw) if raw is not None else None
    except ValueError:
        action = None
    return WatchState(
        product_id=product_id,
        last_alert_at=float(row.get("last_alert_at") or 0.0),
        last_action=action,
        alerts_sent=int(row.get("alerts_sent") or 0),
        last_seen_in_stock=float(row.get("last_seen_in_stock") or 0.0),
    )


def _scheduler_snapshot(scheduler: Any) -> Optional[Dict[str, Any]]:
    snapshot = getattr(scheduler, "snapshot", None)
    if not callable(snapshot):
        return None
    taken = snapshot()
    return taken if isinstance(taken, dict) else None


# --------------------------------------------------------------------------
# a smoke run: python3 -m jarvis_bots.bots.poke_bot
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    from jarvis_bots.registry import BotRegistry
    from jarvis_bots.supervisor import Supervisor
    from jarvis_poke.catalog import Catalog
    from jarvis_poke.contracts import Budget, FetchPolicy, Rule, Stock
    from jarvis_poke.engine import DecisionEngine
    from jarvis_poke.prices import PriceHistory
    from jarvis_poke.rules import RuleSet
    from jarvis_poke.sources import MemoryPollStore, PollScheduler, load_policies

    catalog = Catalog.load()
    policies = load_policies()
    at = {"now": 1_700_000_000.0}
    clock = lambda: at["now"]  # noqa: E731 - the injected clock, in one line

    watched = [p.id for p in catalog.products()[:3]]
    rules = RuleSet(
        [Rule(product_id=pid, max_price=5500, cooldown_s=1800.0) for pid in watched],
        budget=Budget(total=50_000),
    )
    history = PriceHistory()
    engine = DecisionEngine(catalog, rules, history, clock)
    scheduler = PollScheduler(catalog, policies, clock, MemoryPollStore())

    # A dull fake shop: cheap for a while, then a price rise, then gone.
    # Deterministic -- no random, no hash, no wall clock.
    step = {"n": 0}

    def fetcher(url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
        if step["n"] == 7:
            return FetchResult(ok=False, status=503, reason="upstream busy")
        return FetchResult(ok=True, status=200, body="<fake/>")

    def parser(sku: SourceSku, body: str, moment: float) -> Observation:
        n = step["n"]
        stock = Stock.OUT_OF_STOCK if n >= 10 else Stock.IN_STOCK
        price = 5200 if n < 5 else 4200
        return Observation(
            product_id=sku.product_id, source=sku.source, sku=sku.sku, at=moment,
            stock=stock, price=None if stock is Stock.OUT_OF_STOCK else price,
            shipping=0, url=sku.url,
        )

    bot = PokeBot(catalog, rules, history, engine, scheduler, fetcher, parser, clock)
    supervisor = Supervisor(BotRegistry([bot]), clock)

    print(f"{bot!r}  {bot.info.name} -> {bot.info.href}")
    for _ in range(13):
        supervisor.run_round(at["now"])
        card = supervisor.launcher_state(at["now"])["bots"][0]
        last = card["last_event"]
        figures = "; ".join(f"{s['label']}: {s['value']}" for s in card["stats"])
        elapsed = int(at["now"] - 1_700_000_000)
        print(
            f"  t+{elapsed:>6}s  {card['state']:<7} badge={card['attention']}  {figures}"
        )
        if last:
            print(f"              last: {last['text']}")
        close_resolved(supervisor, bot.drain_resolved())
        step["n"] += 1
        at["now"] += 1800.0

    print(f"  open keys: {bot.open_attention_keys()}")
    print(f"  badge now: {supervisor.badge_status()}")
    restored = PokeBot(catalog, rules, history, engine, scheduler, fetcher, parser, clock)
    restored.restore(bot.snapshot())
    print(f"  snapshot round trip: watch states {len(restored.snapshot()['watch_states'])}")
