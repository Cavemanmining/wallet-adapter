"""The framework gate: proof that bot isolation and the launcher contract hold.

Contract: :mod:`jarvis_bots.contracts`.  Its module docstring makes five
promises about what the supervisor does *for* a bot, and every one of them
is the kind of thing that quietly stops being true as a package grows --
nothing fails loudly when a broken bot starts hammering, or when a paused
bot keeps buzzing the owner's phone.  So this module builds a fleet of
synthetic bots that behave badly on purpose, runs real
:class:`jarvis_bots.supervisor.Supervisor` rounds against them, and then
argues with the result.

The five rules and where each is tested
---------------------------------------
=============================  =====================================
contracts.py rule              checked by
=============================  =====================================
"A bot never blocks another"   :meth:`_Gate._check_health` (the
                               healthy bot's schedule and its
                               untouched health record)
"A sick bot backs off"         :meth:`_Gate._check_health` (the
                               observed gap between a failing bot's
                               own ticks, against
                               :func:`_expected_gap`)
"Paused means paused"          :meth:`_Gate._check_attention`,
                               :meth:`_Gate._check_alerts` and the
                               paused-tick check in
                               :meth:`_Gate._absorb`
"Events are the only output"   :meth:`_Gate._check_alerts` -- every
                               push is traced back to one event, and
                               the fake alert service is the only
                               place a push can appear
"State is the bot's,           :meth:`_Gate._check_persistence` -- the
persistence is ours"           snapshot handed to an injected dict
                               store, every round, unfiltered -- and
                               :meth:`_Gate._check_restart`, which
                               reads it back into a fleet that has
                               never ticked
=============================  =====================================

and the launcher's own contract, ``jarvis_bots/web/README.md``, is
validated key by key in :meth:`_Gate._check_launcher` every round.

Two things the run does that are easy to leave out, and that a gate is
worthless without
-----------------------------------------------------------------------
It pushes at :data:`GATE_ALERT_MIN`, which is *below* ACTION.  At the
ACTION default every event that could earn a push was also one that
opened a badge item, so the alert rules and the badge rules were the same
rule and no sub-ACTION event ever reached the alert path at all -- which
is where the alert memory can be poisoned and a later real push silently
suppressed.

And a question is closed by **both** mechanisms: the app calling
:meth:`~jarvis_bots.supervisor.Supervisor.clear_attention`, and a bot
emitting :data:`~jarvis_bots.supervisor.RESOLVED_FLAG`.  Only the first
used to happen, so ``_close_attention`` was called zero times in a
240-round run, every way of breaking the bot-side close left the gate
green, and "attention cleared one round late" was caught or missed
depending purely on which of the two paths you broke -- the invisible one
being the one both shipped bots use.

The fleet
---------
Nine bots, registered in this order, so that the healthy one is *last*:
if isolation is broken, the bots in front of it take it down with them and
the schedule check says so.

``always_fails``  raises every tick.  Drives quarantine and backoff.
``flaky``         raises on a seeded pattern, never three ticks running,
                  so it must recover and must never be quarantined.
``malformed``     returns things that are not a sequence of its own
                  events, including one stamped with ``healthy``'s id.
``flapper``       raises and clears attention repeatedly on one stable
                  key, so re-alerting is visible.
``burster``       emits a burst of five to twelve events in one tick,
                  plus a keyless ACTION notice and two keyed ones.
``pauser``        holds one open request; paused and resumed mid-run.
``resolver``      opens a request and then *closes it itself*, with the
                  framework's own bot-side signal (``RESOLVED_FLAG``),
                  on the same key string the flapper uses.
``slowpoke``      advances the injected clock past ``SLOW_TICK_S``
                  inside its tick -- a slow tick with no sleeping.
``healthy``       never fails, ticks on schedule, and is the control.

What the gate asserts
---------------------
Each is a :class:`Problem` with a stable ``kind``:

``isolation_broken``      the healthy bot missed a round it was due, or
                          picked up a failure from a neighbour
``round_crashed``         ``run_round`` raised instead of isolating
``quarantine_wrong``      ``is_quarantined`` disagreed with "at or past
                          ``QUARANTINE_AFTER_FAILURES`` consecutive
                          failures", or the first quarantine did not
                          land on exactly that failure
``backoff_narrowed``      a failing bot's next attempt was sooner than
                          its base interval
``backoff_wrong``         the gap was not what :func:`_expected_gap`
                          says, given the failure count
``schedule_wrong``        a successful tick did not rearm at
                          ``now + interval_s``
``paused_ticked``         a paused bot was ticked
``paused_attention``      a paused bot held an item of attention
``paused_alert``          a paused bot's event reached the phone
``attention_wrong``       the open set, or its count, did not match the
                          distinct keys the fleet actually opened
``attention_unbounded``   more open items than the fleet has keys
``attention_since_moved`` ``since`` was restamped while the question
                          stayed open
``realerted``             a second push for a key that was already open
``alert_missing``         a newly opened key, or a keyless ACTION event,
                          earned no push
``alert_unexpected``      a push nothing in the round asked for
``badge_wrong``           ``badge_status`` did not match the computed
                          truth
``launcher_invalid``      ``launcher_state`` broke web/README.md
``launcher_wrong``        a card contradicted the computed truth
``malformed_kept``        a malformed event became an event, an item of
                          attention or a push
``health_wrong``          the supervisor's counters disagreed with what
                          the bots recorded doing
``alert_memory_leak``     the record of what has been pushed outlived
                          the questions it was about
``state_not_json``        the snapshot handed to the store was not
                          JSON-able, or a bot's snapshot raised
``paused_wrong``          the persisted pause flag did not match the
                          owner's switch
``round_report_wrong``    ``RoundReport``'s counts did not add up
``slow_wrong``            the slow tick was not reported, or a prompt
                          one was
``scenario_thin``         the run did not contain a hazard the gate
                          claims to test, so a pass would be vacuous
``crashed``               the run itself raised

Truth is computed from the *bots' own logs*, never from the supervisor's
bookkeeping: each synthetic bot records the round it was entered on,
whether it raised, and what it returned, and the gate derives the failure
counts, the open keys and the badge from that.  A check that asked the
supervisor what it thought and then agreed with it would prove nothing.

Shown to fail before it is trusted
----------------------------------
``inject_defect`` swaps the supervisor for a subclass that breaks exactly
one rule -- see :data:`DEFECTS` -- and every one must be reported.
``python3 -m jarvis_bots.validate --show-defects`` exits 1 if any is
missed.

    no_isolation         a failing tick aborts the round
    no_quarantine        failures never quarantine, so a broken bot
                         keeps hammering at its base interval
    narrowing_backoff    failures make the bot retry *sooner*
    tick_paused          pause is ignored while a round is running
    attention_leak       every repeat of a key opens a new item
    realert_every_round  the alert memory is dropped, so one open
                         question pushes every round
    state_drift          the badge and the launcher payload stop
                         matching the contract

The boundary
------------
This is a test harness for a scheduler.  Nothing here buys, pays or
transacts, and nothing it exercises can: the widest thing a synthetic bot
does is return an event with an ``href``, which becomes a line in a fake
alert log.  No sockets, no ``time.time()`` (the clock is a cell this
module advances by hand, and the slow bot is slow because it advances that
cell), no ``random`` (every draw comes from
:class:`lucifer_gen.seed.SeedFields`), and no money, so the integer-cents
rule has nothing to violate here.

Entry point
-----------
:func:`run_gate` returns a :class:`GateReport`.  ``python3 -m
jarvis_bots.validate`` prints one; ``jarvis_bots.cli gate`` runs the same
call.
"""

from __future__ import annotations

import argparse
import collections
import collections.abc
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from lucifer_gen.seed import SeedFields, format_seed, parse_seed

from jarvis_bots.base import BaseBot
from jarvis_bots.contracts import (
    QUARANTINE_AFTER_FAILURES,
    QUARANTINE_S,
    SLOW_TICK_S,
    BotInfo,
    BotState,
    BotStatus,
    Event,
    Severity,
    backoff_interval,
)
from jarvis_bots.registry import BotRegistry
from jarvis_bots.supervisor import RESOLVED_FLAG, Supervisor

__all__ = [
    "BASE_INTERVAL_S",
    "DEFAULT_ROUNDS",
    "DEFAULT_SEED",
    "DEFECTS",
    "GATE_EPOCH",
    "MIN_ROUNDS",
    "ROUND_S",
    "GateError",
    "GateReport",
    "Problem",
    "main",
    "run_gate",
]


# --------------------------------------------------------------------------
# Constants of the run
# --------------------------------------------------------------------------

#: Where the gate's clock starts.  A fixed number, because a gate that
#: starts "now" is a gate whose failures cannot be reproduced.
GATE_EPOCH = 1_700_000_000.0

#: Seconds between supervisor rounds.  A divisor of every bot interval
#: below, so "due" lands exactly on a round and the expected schedule is a
#: statement about the contract rather than about rounding.
ROUND_S = 60.0

#: The interval most of the fleet declares.  Five rounds apart.
BASE_INTERVAL_S = 300.0

#: The burster and the slow bot are rarer, so a round is not all noise.
BURST_INTERVAL_S = 600.0
SLOW_INTERVAL_S = 900.0

#: How far past :data:`~jarvis_bots.contracts.SLOW_TICK_S` the slow bot
#: pushes the injected clock inside its tick.  Less than ``ROUND_S``, so
#: the clock never has to be wound backwards between rounds.
SLOW_BY_S = SLOW_TICK_S + 5.0

#: Rounds below this cannot contain what the gate claims to test: the
#: always-failing bot first quarantines at round 70 and its first probe
#: runs at round 100, a second probe at 130.  Asking for fewer is asking
#: for a vacuous pass, so it raises instead.
MIN_ROUNDS = 140
DEFAULT_ROUNDS = 240

DEFAULT_SEED = 0xB075_6A7E_0000_0001

#: The severity the gate's supervisor pushes at.
#:
#: NOTICE, not the ACTION default, and that is the point.  At ACTION the
#: run never exercised the threshold from below: every event that could
#: possibly push was one that also opened a badge item, so the alert rules
#: and the badge rules were the same rule and a bug in either looked like a
#: bug in both.  It also made a whole class of defect invisible -- a
#: sub-ACTION event carrying an attention key (a *resolution*, most of all)
#: could poison the alert memory and permanently suppress the real push
#: that followed, and the gate could not see it because no sub-ACTION event
#: ever reached the alert path.  contracts.py calls NOTICE "worth a line in
#: the feed" and ``Supervisor.alert_min_severity`` is a documented public
#: argument, so this is a supported wiring, not a contrivance.
#:
#: DEBUG and INFO stay below it, so "below the threshold does not push" is
#: still checked on every round by every event that is not worth a push.
GATE_ALERT_MIN = Severity.NOTICE

#: The defects :func:`run_gate` can inject.  Each is a rule in
#: contracts.py broken in one place, and each must be reported.
DEFECTS: Tuple[str, ...] = (
    "no_isolation",
    "no_quarantine",
    "narrowing_backoff",
    "tick_paused",
    "attention_leak",
    "realert_every_round",
    "state_drift",
    # The bot-side close path (RESOLVED_FLAG).  Every one of these was
    # missed before the resolver bot existed: the fleet only ever closed a
    # key through the app-side clear_attention(), so "attention cleared one
    # round late" was caught or missed depending purely on which of the two
    # closing mechanisms was broken -- and the invisible one is the one
    # both shipped bots use.
    "never_resolves",
    "loose_resolution",
    "cross_bot_close",
    "late_resolution",
    # The alert memory outliving the question it was about, which is what
    # makes a restock that comes back a second time silent.
    "resolution_poisons_alerts",
    # "State is the bot's, persistence is ours", broken at the seam.
    "state_not_restored",
)

#: Float comparisons here are on sums of constants and are exact in
#: practice, but a gate that fails on the last bit of a float is a gate
#: nobody trusts.
EPS = 1e-6

#: At most this many problems of one kind are recorded; the rest are
#: counted.  A single broken rule can fail on every one of 500 rounds, and
#: a report nobody can read is a report nobody reads.
MAX_PER_KIND = 4

#: The id the malformed bot stamps its poison event with: another bot's.
#: contracts.py files attention, the card and the push under the event's
#: ``bot_id``, so an event that lies about who it came from is how one bot
#: nags the owner in another's name.
POISON_BOT_ID = "healthy"
POISON_KEY = "poison"

#: The launcher's documented card keys, in web/README.md's order.
CARD_KEYS: Tuple[str, ...] = (
    "id",
    "name",
    "blurb",
    "kind",
    "state",
    "attention",
    "href",
    "can_pause",
    "stats",
    "last_event",
)

#: web/README.md: "`state` is `ok`, `warn` or `error`" for the badge, and
#: a card's state is BotState.ui.
BADGE_STATES = frozenset({"ok", "warn", "error"})
CARD_STATES = frozenset(s.ui for s in BotState)


class GateError(ValueError):
    """The gate was asked for a run it cannot build: fewer rounds than the
    hazards it claims to test need, or a defect name it does not know."""


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One failed assertion.

    ``kind`` is stable and is what the tests match on; ``detail`` is for a
    person; ``bot_id`` and ``round_index`` say where.  There is no money
    anywhere in this gate, so no value here is ever a formatted amount.
    """

    kind: str
    detail: str
    bot_id: str = ""
    round_index: int = -1

    def __str__(self) -> str:
        where = []
        if self.bot_id:
            where.append(f"bot={self.bot_id}")
        if self.round_index >= 0:
            where.append(f"round={self.round_index}")
        tail = f" [{' '.join(where)}]" if where else ""
        return f"{self.kind}: {self.detail}{tail}"


@dataclass
class GateReport:
    """What :func:`run_gate` found.

    ``counts`` is JSON-friendly and every value is an ``int``; it is also
    the evidence that the run contained what it claims to (quarantines,
    probes, pauses, bursts, rejections, alerts).  ``problems`` is in
    detection order, so ``first_failure`` is the most upstream one.
    """

    n_rounds: int
    seed: int
    inject_defect: Optional[str]
    counts: Dict[str, int] = field(default_factory=dict)
    problems: List[Problem] = field(default_factory=list)
    suppressed: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def first_failure(self) -> Optional[Problem]:
        return self.problems[0] if self.problems else None

    @property
    def repro(self) -> str:
        return (
            f"run_gate(n_rounds={self.n_rounds}, seed={format_seed(self.seed)}, "
            f"inject_defect={self.inject_defect!r})"
        )

    def kinds(self) -> List[str]:
        """Distinct problem kinds, in first-seen order."""
        seen: List[str] = []
        for problem in self.problems:
            if problem.kind not in seen:
                seen.append(problem.kind)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_rounds": self.n_rounds,
            "seed": format_seed(self.seed),
            "inject_defect": self.inject_defect,
            "ok": self.ok,
            "counts": dict(self.counts),
            "suppressed": self.suppressed,
            "problems": [dataclasses.asdict(p) for p in self.problems],
        }

    def summary(self, max_problems: int = 8) -> str:
        return "\n".join(self.lines_for(max_problems))

    def lines_for(self, max_problems: int = 8) -> List[str]:
        lines = [
            f"gate: rounds={self.n_rounds} seed={format_seed(self.seed)} "
            f"defect={self.inject_defect!r}",
            "  counts: " + " ".join(f"{k}={v}" for k, v in sorted(self.counts.items())),
        ]
        if self.ok:
            lines.append("  result: OK")
            return lines
        lines.append(
            f"  result: FAIL ({len(self.problems)} problem(s)"
            f"{f' +{self.suppressed} suppressed' if self.suppressed else ''}; "
            f"kinds: {', '.join(self.kinds())})"
        )
        for problem in self.problems[:max_problems]:
            lines.append(f"    - {problem}")
        if len(self.problems) > max_problems:
            lines.append(f"    ... {len(self.problems) - max_problems} more")
        lines.append(f"  repro: {self.repro}")
        return lines

    @property
    def lines(self) -> Tuple[str, ...]:
        """What ``jarvis_bots.cli gate`` prints.  The CLI reads ``ok`` and
        ``lines`` off whatever the gate returns; this is that seam."""
        return tuple(self.lines_for())


# --------------------------------------------------------------------------
# Injected seams: the clock and the alert service
# --------------------------------------------------------------------------


class _Clock:
    """The injected clock, as a cell the gate moves by hand.

    contracts.py: "Time is injected everywhere. Nothing here calls
    time.time()."  Holding the clock in a cell is also what makes a slow
    tick testable without sleeping: :class:`_SlowBot` advances this, and
    the supervisor -- which times a tick with the same injected callable --
    sees a tick that took longer than ``SLOW_TICK_S``.
    """

    __slots__ = ("t",)

    def __init__(self, at: float) -> None:
        self.t = float(at)

    def __call__(self) -> float:
        return self.t

    def set(self, at: float) -> None:
        self.t = float(at)

    def advance(self, by: float) -> None:
        self.t += float(by)


class _RecordingAlerts:
    """A fake ``jarvis_alerts.api.AlertService``: it records, it never sends.

    contracts.py: "A bot does not send alerts itself; it returns events and
    the supervisor decides what is worth a push."  This is the far side of
    that decision, injected exactly as ``jarvis_alerts`` takes an injected
    sender, so the *only* place a push can appear in this run is a row in
    :attr:`published`.  No socket is opened by anything here.
    """

    def __init__(self) -> None:
        self.published: List[Dict[str, Any]] = []

    def publish(
        self,
        profile_id: str,
        kind: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        priority: Any = "normal",
        dedupe_key: Optional[str] = None,
    ) -> str:
        row = {
            "profile_id": profile_id,
            "kind": kind,
            "title": title,
            "body": body,
            "data": dict(data or {}),
            "priority": str(priority),
            "dedupe_key": dedupe_key,
        }
        self.published.append(row)
        return f"alert-{len(self.published)}"


# --------------------------------------------------------------------------
# The synthetic fleet
# --------------------------------------------------------------------------


def _round_of(now: float) -> int:
    """Which round a tick's ``now`` belongs to.

    ``run_round`` captures one ``at`` and hands the same value to every
    bot, so this is exact even in the round where the slow bot moves the
    clock on underneath the others.
    """
    return int(round((now - GATE_EPOCH) / ROUND_S))


class _GateBot(BaseBot):
    """A synthetic bot that keeps its own log.

    The log is the gate's source of truth: the round it was entered on,
    whether it raised, and what it returned.  Everything the gate then
    asserts about health, attention and the badge is derived from these
    lists, so a check never has to ask the supervisor what it believes and
    then agree with it.
    """

    #: Attention keys this bot can ever open.  The gate sums these across
    #: the fleet to get a hard ceiling on the badge.
    keys: Tuple[str, ...] = ()

    def __init__(self, clock: _Clock, info: BotInfo) -> None:
        super().__init__(clock, info)
        self.rounds: List[int] = []
        self.outcomes: List[str] = []  # "ok" or "raise", one per tick
        self.returned: List[Any] = []  # exactly what tick() handed back
        self.break_status = False
        self._state = BotState.IDLE
        #: Real state, because the rule is "State is the bot's, persistence
        #: is ours" and a fleet of stateless bots checks that rule against
        #: eight empty dicts.  Deliberately not flat: a nested list and a
        #: nested object are what a JSON-ability check has to be pointed at
        #: to mean anything, and they are what a real bot's watchlist looks
        #: like.
        self.ticks_recorded = 0
        self.notes: List[str] = []
        self.marks: Dict[str, Any] = {}

    def snapshot(self) -> Dict[str, Any]:
        """contracts.py: "A bot hands over a JSON-able snapshot and gets it
        back on the next start."
        """
        return {
            "ticks_recorded": self.ticks_recorded,
            "notes": list(self.notes[-4:]),
            "marks": dict(self.marks),
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.ticks_recorded = int(snapshot.get("ticks_recorded", 0))
        self.notes = list(snapshot.get("notes") or [])
        self.marks = dict(snapshot.get("marks") or {})

    def state_fingerprint(self) -> Dict[str, Any]:
        """What a restored copy of this bot must match, exactly."""
        return self.snapshot()

    # -- the protocol's required half ---------------------------------------

    def tick(self, now: float) -> Any:
        """contracts.py: "Do one unit of work. ... Raising is allowed and is
        handled."  Log first, so a raise is still on the record."""
        index = len(self.rounds)
        self.rounds.append(_round_of(now))
        self._state = BotState.RUNNING
        # State moves before the work, so a tick that raises still changed
        # something: a bot whose state only ever advances on a clean tick
        # would make "the snapshot survives a restart" true for a reason
        # that has nothing to do with the seam.
        self.ticks_recorded += 1
        self.marks["last_round"] = self.rounds[-1]
        try:
            events = self._work(self.rounds[-1], index)
        except BaseException:
            self.notes.append(f"raised at {self.rounds[-1]}")
            self.notes = self.notes[-4:]
            self.outcomes.append("raise")
            self.returned.append(None)
            raise
        self.notes.append(f"ok at {self.rounds[-1]}")
        self.notes = self.notes[-4:]
        self.outcomes.append("ok")
        self.returned.append(events)
        return events

    def status(self) -> BotStatus:
        """The launcher card.  ``break_status`` makes it raise on the rounds
        the gate chooses, to prove web/README.md's page still renders: "A
        ``status()`` that raises does not take the page down with it"."""
        if self.break_status:
            raise RuntimeError("status() is broken this round, on purpose")
        return BotStatus(self._state, (self.stat("Ticks", len(self.rounds)),))

    # -- what a subclass writes ----------------------------------------------

    def _work(self, round_index: int, index: int) -> Any:
        raise NotImplementedError


class _HealthyBot(_GateBot):
    """The control: never fails, never surprises, emits a NOTICE now and
    then.  contracts.py's "A bot never blocks another" is the claim that
    this bot's transcript is identical whatever the rest of the fleet does."""

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        if index % 3 == 0:
            return [self.event(Severity.NOTICE, f"all quiet at round {round_index}")]
        return []


class _AlwaysFailsBot(_GateBot):
    """Raises every tick.  Drives "A sick bot backs off": the interval
    widens per :func:`~jarvis_bots.contracts.backoff_interval` and then the
    bot is quarantined."""

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        raise RuntimeError(f"always_fails: tick {index} at round {round_index}")


class _FlakyBot(_GateBot):
    """Raises on a seeded pattern that never has three failures running, so
    it must recover on its own and must never reach
    ``QUARANTINE_AFTER_FAILURES``.  The interesting case: intermittent is
    not the same as broken, and the framework has to tell them apart."""

    def __init__(self, clock: _Clock, info: BotInfo, pattern: Tuple[bool, ...]) -> None:
        super().__init__(clock, info)
        self._pattern = pattern

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        if self._pattern[index % len(self._pattern)]:
            raise ValueError(f"flaky: tick {index} did not come back")
        return [self.event(Severity.INFO, f"flaky recovered at round {round_index}")]


class _PauserBot(_GateBot):
    """Holds one standing request for a decision, every tick.  The gate
    pauses and resumes it mid-run, so "Paused means paused" is tested on a
    bot that would otherwise be shouting."""

    keys = ("pauser-open",)

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        return [
            self.event(
                Severity.ACTION,
                f"something needs deciding (round {round_index})",
                attention_key="pauser-open",
                href="/bots/pauser",
            )
        ]


#: One key *string*, held by two different bots at once.
#:
#: The badge keys on ``(bot_id, key)``, and every synthetic bot used to
#: pick a key nobody else used, which made "whose question is this?"
#: untestable: a supervisor that closed a key under whatever bot asked --
#: one bot emptying another's badge -- behaved identically to a correct
#: one.  The flapper and the resolver now share this string and close it
#: by different mechanisms.
SHARED_KEY = "shared-question"


class _FlapperBot(_GateBot):
    """Raises attention on one stable key, every tick.  The gate closes that
    key on a seeded cadence and the bot opens it again, which is the exact
    shape contracts.py cares about: "one restock nagging across ten ticks is
    one item of attention, not ten", and a genuinely new question later is
    news again.

    Its key is :data:`SHARED_KEY`, the same string the resolver uses, and
    the two must never close each other's."""

    keys = (SHARED_KEY,)

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        return [
            self.event(
                Severity.ACTION,
                f"still open (round {round_index})",
                attention_key=SHARED_KEY,
                href="/bots/flapper",
            )
        ]


class _BursterBot(_GateBot):
    """Emits a burst in one tick: noise, one keyless ACTION notice (which
    must push every time, because nothing collapses it) and two keyed ones
    (which must push once each, and then not again)."""

    keys = ("burst-a", "burst-b")

    def __init__(self, clock: _Clock, info: BotInfo, sizes: Tuple[int, ...]) -> None:
        super().__init__(clock, info)
        self.sizes = sizes

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        size = self.sizes[index % len(self.sizes)]
        out = [
            self.event(
                Severity.DEBUG if i % 2 else Severity.INFO,
                f"burst {round_index}.{i}",
            )
            for i in range(size)
        ]
        out.append(
            self.event(
                Severity.ACTION,
                f"one-off notice at round {round_index}",
                href="/bots/burster",
            )
        )
        # The top of the scale, keyless.  Every other severity was emitted
        # by some bot and ERROR by none, which left one entry of the
        # supervisor's severity -> priority table unexercised: a missing
        # one raises inside _publish, is swallowed into last_alert_error,
        # and the owner simply never hears about the loudest thing a bot
        # can say.
        out.append(
            self.event(
                Severity.ERROR,
                f"something went badly wrong at round {round_index}",
                href="/bots/burster",
            )
        )
        out.append(
            self.event(
                Severity.ACTION,
                f"burst A still open (round {round_index})",
                attention_key="burst-a",
                href="/bots/burster",
            )
        )
        out.append(
            self.event(
                Severity.ACTION,
                f"burst B still open (round {round_index})",
                attention_key="burst-b",
                href="/bots/burster",
            )
        )
        return out


class _ResolverBot(_GateBot):
    """Opens a request for a decision and then *closes it itself*.

    The one bot in the fleet that uses the framework's own bot-side close
    (:data:`~jarvis_bots.supervisor.RESOLVED_FLAG`): a NOTICE carrying the
    same key and ``resolved=True``.  Until it existed, the gate's only way
    a key ever closed was the driver calling ``clear_attention`` -- the
    app-side API -- so ``_close_attention`` was called zero times in a
    240-round run and ``_is_resolution`` returned False every single time.
    Four separate ways of breaking the close path left the gate green,
    including the exact defect the gate names ("attention cleared one round
    late"), because that defect was only ever injected on the other path.

    Both shipped bots close this way (``PokeBot._resolve`` and
    ``templates/bot.py.tmpl``), so this is the path a second bot's author
    will actually take.

    The cadence alternates on a fixed period rather than a seeded one, so
    the number of open/close episodes does not depend on the seed: a
    question that is never closed and one that is never re-opened are both
    vacuous.
    """

    keys = (SHARED_KEY,)

    #: Ticks the question stays open, then ticks it stays closed.
    OPEN_FOR = 2
    SHUT_FOR = 2

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        phase = index % (self.OPEN_FOR + self.SHUT_FOR)
        if phase == 0:
            return [
                self.event(
                    Severity.ACTION,
                    f"decide something (round {round_index})",
                    attention_key=SHARED_KEY,
                    href="/bots/resolver",
                )
            ]
        if phase < self.OPEN_FOR:
            # Still open: a repeat of the same key, which must collapse --
            # and, in the same tick, a sub-ACTION event carrying that key
            # with the flag set to something merely *truthy*. contracts
            # requires ``is True`` exactly, so this must not close
            # anything; a supervisor that accepts truthy empties the badge
            # here, while the question is still open, which is the whole
            # point of testing it while the question is still open.
            return [
                self.event(
                    Severity.ACTION,
                    f"still needs deciding (round {round_index})",
                    attention_key=SHARED_KEY,
                    href="/bots/resolver",
                ),
                self.event(
                    Severity.INFO,
                    f"resolved is a string here, not True (round {round_index})",
                    attention_key=SHARED_KEY,
                    href="/bots/resolver",
                    **{RESOLVED_FLAG: "no"},
                ),
            ]
        if phase == self.OPEN_FOR:
            # Closed by the bot, in the shape the framework documents:
            # below ACTION so it cannot re-open, same key, resolved=True.
            return [
                self.event(
                    Severity.NOTICE,
                    f"no longer a question (round {round_index})",
                    attention_key=SHARED_KEY,
                    href="/bots/resolver",
                    **{RESOLVED_FLAG: True},
                )
            ]
        # Shut, and saying so in a way that must not re-open anything: the
        # same key at a sub-ACTION severity with no flag at all.
        return [
            self.event(
                Severity.NOTICE,
                f"a line in the feed about the same thing (round {round_index})",
                attention_key=SHARED_KEY,
                href="/bots/resolver",
            )
        ]


class _SlowBot(_GateBot):
    """A tick that takes longer than ``SLOW_TICK_S`` without sleeping.

    contracts.py: "A tick that runs longer than this is reported as an
    anomaly; the framework cannot kill it, but a bot hogging the round
    should be visible."  The supervisor times a tick with the *injected*
    clock, so moving that clock on is a slow tick in every way that
    matters, and the gate stays deterministic and instant.
    """

    def __init__(self, clock: _Clock, info: BotInfo, by: float) -> None:
        super().__init__(clock, info)
        self._cell = clock
        self._by = float(by)

    def _work(self, round_index: int, index: int) -> Sequence[Event]:
        self._cell.advance(self._by)
        return [self.event(Severity.INFO, f"slow tick at round {round_index}")]


class _MalformedBot(_GateBot):
    """Returns things that are not a sequence of its own events.

    Four bad shapes, each alternating with a clean tick so the bot stays
    out of quarantine and keeps producing them.  The fourth is the one that
    matters most: an event stamped with :data:`POISON_BOT_ID`.  contracts.py
    files attention, the card and the push under ``event.bot_id``, so an
    accepted foreign event is one bot nagging the owner in another's name.
    """

    SHAPES: Tuple[str, ...] = (
        "not_a_sequence",
        "clean",
        "non_event_member",
        "clean",
        "bare_event",
        "clean",
        "foreign_id",
        "clean",
    )

    def _work(self, round_index: int, index: int) -> Any:
        shape = self.SHAPES[index % len(self.SHAPES)]
        if shape == "clean":
            return [self.event(Severity.NOTICE, f"well-formed at round {round_index}")]
        if shape == "not_a_sequence":
            return 42
        if shape == "non_event_member":
            return [self.event(Severity.INFO, "one real event"), "not an event"]
        if shape == "bare_event":
            return self.event(Severity.INFO, "a bare event, not a sequence")
        return [
            Event(
                bot_id=POISON_BOT_ID,
                at=self.now(),
                severity=Severity.ACTION,
                text="an event wearing another bot's name",
                attention_key=POISON_KEY,
                href="/bots/healthy",
            )
        ]


def _closes_a_key(event: Event) -> bool:
    """True when this event says one open request for a decision is closed.

    :data:`~jarvis_bots.supervisor.RESOLVED_FLAG`'s rule, restated here
    independently of ``supervisor._is_resolution`` so the gate can say what
    *should* have closed without borrowing the code that decides what did.
    All three conditions, each ruling out a different mistake:

    * a non-empty key -- there is no closing a question never asked;
    * the flag is ``True`` exactly, not merely truthy, so ``resolved="no"``
      or ``resolved=0`` in a payload cannot empty the badge;
    * below ACTION, so an event asking for a decision is never also
      reporting one made.
    """
    return (
        bool(event.attention_key)
        and event.data.get(RESOLVED_FLAG) is True
        and not event.wants_attention
    )


def _clean_events(bot_id: str, returned: Any) -> Optional[Tuple[Event, ...]]:
    """What a tick's return value is worth, per the contract, or ``None``.

    contracts.py types :meth:`Bot.tick` as returning ``Sequence[Event]``,
    and files every event under ``event.bot_id``.  This states that rule
    once, independently of the supervisor's own ``_checked_events``, so the
    gate can say what *should* have happened without borrowing the code
    that decides what did.
    """
    if returned is None:
        return ()
    if isinstance(returned, Event):
        return None  # a bare Event is not a sequence of them
    if isinstance(returned, (str, bytes)) or not isinstance(
        returned, collections.abc.Sequence
    ):
        return None
    out: List[Event] = []
    for item in returned:
        if not isinstance(item, Event) or item.bot_id != bot_id:
            return None
        out.append(item)
    return tuple(out)


# --------------------------------------------------------------------------
# The defects
# --------------------------------------------------------------------------


class _NoIsolation(Supervisor):
    """Breaks "A bot never blocks another": a failing tick takes the round
    with it, so every bot registered behind it is skipped."""

    def _record_failure(self, bot, health, at, exc, report):  # type: ignore[no-untyped-def]
        super()._record_failure(bot, health, at, exc, report)
        raise exc


class _NoQuarantine(Supervisor):
    """Breaks the second half of "A sick bot backs off": failures are
    recorded but never quarantine, so a broken bot keeps hammering."""

    def _record_failure(self, bot, health, at, exc, report):  # type: ignore[no-untyped-def]
        super()._record_failure(bot, health, at, exc, report)
        if health.quarantined_until > 0.0:
            health.quarantined_until = 0.0
            health.next_due_at = at + float(bot.info.interval_s)
            report.quarantined = max(0, report.quarantined - 1)


class _NarrowingBackoff(Supervisor):
    """Breaks the first half of "A sick bot backs off": each failure makes
    the next attempt sooner than the healthy interval, which is exactly the
    hammering the rule exists to prevent."""

    def _record_failure(self, bot, health, at, exc, report):  # type: ignore[no-untyped-def]
        super()._record_failure(bot, health, at, exc, report)
        health.next_due_at = at + float(bot.info.interval_s) / 4.0


class _TickPaused(Supervisor):
    """Breaks "Paused means paused": pause still shows in the UI, but a
    round ignores it, so a switched-off bot ticks and pushes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._in_round = False

    def is_paused(self, bot_id: str) -> bool:
        return False if self._in_round else super().is_paused(bot_id)

    def run_round(self, now: Optional[float] = None):  # type: ignore[no-untyped-def]
        self._in_round = True
        try:
            return super().run_round(now)
        finally:
            self._in_round = False


class _AttentionLeak(Supervisor):
    """Breaks the badge rule: every repeat of a key opens a *new* item, so
    "one restock nagging across ten ticks" becomes ten, and the count grows
    for as long as the process runs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._leaked = 0

    def _open_attention(self, event: Event):  # type: ignore[no-untyped-def]
        self._leaked += 1
        return super()._open_attention(
            dataclasses.replace(event, attention_key=f"{event.attention_key}#{self._leaked}")
        )


class _RealertEveryRound(Supervisor):
    """Breaks alert policy: the memory of what has already been pushed is
    dropped, so one open question buzzes the phone every round."""

    def _maybe_alert(self, bot, event, at):  # type: ignore[no-untyped-def]
        if event.attention_key:
            self._alerted.pop((bot.info.id, event.attention_key), None)
        return super()._maybe_alert(bot, event, at)


class _StateDrift(Supervisor):
    """Breaks web/README.md: the badge counts bots rather than keys and
    never says error, and the cards drop a key the page reads."""

    def badge_status(self) -> Dict[str, Any]:
        return {"attention": len(self.bots_wanting_attention()), "state": "ok"}

    def launcher_state(self, now: Optional[float] = None, *, include_detail: bool = False):  # type: ignore[no-untyped-def]
        state = super().launcher_state(now, include_detail=include_detail)
        for card in state["bots"]:
            card.pop("can_pause", None)
        return state


class _NeverResolves(Supervisor):
    """Breaks the bot-side close: a bot can never close its own key, so
    every badge item is permanent and the owner learns to ignore the
    badge.  ``RESOLVED_FLAG``'s own docstring says this is why the signal
    exists."""

    def _close_attention(self, event: Event) -> bool:
        return False


class _LooseResolution(Supervisor):
    """Breaks the other half: the flag is believed when it is merely
    truthy, so an event carrying ``resolved="no"`` or ``resolved=0``
    empties the badge.  ``_is_resolution`` documents ``is True`` as "the
    one test that cannot be passed by accident"."""

    def _apply_events(self, bot, events, at, report):  # type: ignore[no-untyped-def]
        loosened = []
        for event in events:
            if (
                event.attention_key
                and not event.wants_attention
                and event.data.get(RESOLVED_FLAG) is not None
                and event.data.get(RESOLVED_FLAG) is not True
                and bool(event.data.get(RESOLVED_FLAG))
            ):
                loosened.append(
                    dataclasses.replace(
                        event, data={**event.data, RESOLVED_FLAG: True}
                    )
                )
            else:
                loosened.append(event)
        return super()._apply_events(bot, tuple(loosened), at, report)


class _CrossBotClose(Supervisor):
    """Breaks whose question it is: a resolution closes the key under
    whatever bot asks, so one bot can empty another's badge."""

    def _close_attention(self, event: Event) -> bool:
        key = str(event.attention_key)
        closed = False
        for owner, held in [pair for pair in self._attention if pair[1] == key]:
            closed = self.clear_attention(owner, held) or closed
        return closed


class _LateResolution(Supervisor):
    """The defect the gate already names -- "attention cleared one round
    late" -- injected on the resolution path instead of the
    ``clear_attention`` one.  Identical latency, and before the resolver
    bot existed the gate saw one and not the other."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._deferred: List[Event] = []

    def _close_attention(self, event: Event) -> bool:
        self._deferred.append(event)
        return True

    def run_round(self, now: Optional[float] = None):  # type: ignore[no-untyped-def]
        due, self._deferred = self._deferred, []
        for event in due:
            self.clear_attention(event.bot_id, str(event.attention_key))
        return super().run_round(now)


class _ResolutionPoisonsAlerts(Supervisor):
    """Breaks the alert memory at the moment a question closes: the key is
    filed as "already pushed" by the resolution itself, so the *next* time
    that question opens the badge goes to 1 and the phone stays silent,
    for the rest of the process's life.

    This is the shape of a real bug this package shipped: ``clear_attention``
    dropped the memory and ``_maybe_alert``, publishing the resolution in
    the same iteration, put it straight back."""

    def _apply_events(self, bot, events, at, report):  # type: ignore[no-untyped-def]
        out = super()._apply_events(bot, events, at, report)
        for event in events:
            if event.attention_key and not event.wants_attention:
                self._alerted[(event.bot_id, str(event.attention_key))] = float(at)
        return out


class _StateNotRestored(Supervisor):
    """Breaks "State is the bot's, persistence is ours" at the seam: the
    snapshot is written, and handed back empty.  A bot comes up having
    forgotten its watchlist, and nothing anywhere says so."""

    def _restore_one(self, bot, bot_id, raw):  # type: ignore[no-untyped-def]
        return super()._restore_one(bot, bot_id, {**raw, "snapshot": {}})


_DEFECT_CLASSES: Dict[str, Any] = {
    "never_resolves": _NeverResolves,
    "loose_resolution": _LooseResolution,
    "cross_bot_close": _CrossBotClose,
    "late_resolution": _LateResolution,
    "resolution_poisons_alerts": _ResolutionPoisonsAlerts,
    "state_not_restored": _StateNotRestored,
    "no_isolation": _NoIsolation,
    "no_quarantine": _NoQuarantine,
    "narrowing_backoff": _NarrowingBackoff,
    "tick_paused": _TickPaused,
    "attention_leak": _AttentionLeak,
    "realert_every_round": _RealertEveryRound,
    "state_drift": _StateDrift,
}


# --------------------------------------------------------------------------
# The rules, stated independently of the supervisor
# --------------------------------------------------------------------------


def _expected_gap(base_s: float, consecutive_failures: int) -> float:
    """How long after a failing tick the next attempt may be, at the
    earliest.

    Below ``QUARANTINE_AFTER_FAILURES`` this is
    :func:`~jarvis_bots.contracts.backoff_interval` with its own documented
    floor re-applied ("Never narrower than base"; the contract's function
    loses that promise above ``BACKOFF_CAP_S``, which is why the floor is
    spelled out here as well as in the supervisor).

    At and above the threshold the bot is quarantined, and contracts.py
    defines ``QUARANTINE_S`` as "how long a quarantine lasts before one
    probe tick is allowed through" -- so the quarantine holds the probe
    back, but it may not *release* it sooner than the back-off this
    failure earned.  Reaching quarantine used to shorten the gap (2400s
    after the third failure, 1800s after the fourth) on any bot whose
    widened interval had passed ``QUARANTINE_S``, which is the one thing
    ``BotInfo.interval_s`` forbids: "The supervisor may widen this when the
    bot is failing, never narrow it."  So the expectation is the larger of
    the two, at every failure count, and the curve is monotonic.
    """
    if consecutive_failures <= 0:
        return float(base_s)
    widened = max(float(base_s), backoff_interval(float(base_s), consecutive_failures))
    if consecutive_failures >= QUARANTINE_AFTER_FAILURES:
        return max(widened, QUARANTINE_S)
    return widened


def _fail_pattern(stream: Any, length: int) -> Tuple[bool, ...]:
    """A seeded failure pattern with no three failures in a row, wrap
    included, and at least one run of two.

    The constraint is the point: the flaky bot must stay below
    ``QUARANTINE_AFTER_FAILURES`` however the rounds fall, so "quarantined"
    and "sometimes fails" cannot be confused for one another.
    """
    pattern = [stream.chance(0.45) for _ in range(length)]
    for _ in range(2):  # twice, so the wrap-around is settled too
        for i in range(length):
            if pattern[i] and pattern[i - 1] and pattern[i - 2]:
                pattern[i] = False
    if not any(pattern):
        pattern[1] = True
    if all(pattern):  # unreachable given the clearing above, kept honest
        pattern[0] = False
    if not any(pattern[i] and pattern[i - 1] for i in range(length)):
        pattern[2] = pattern[3] = True
        pattern[4] = False
    return tuple(pattern)


def _every_few(stream: Any, start: int, stop: int, lo: int, hi: int) -> FrozenSet[int]:
    """Round indices spaced a seeded ``lo``..``hi`` apart."""
    out: List[int] = []
    at = start + stream.randint(lo, hi)
    while at < stop:
        out.append(at)
        at += stream.randint(lo, hi)
    return frozenset(out)


# --------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------


class _Gate:
    """One run: build the fleet, drive ``n_rounds`` rounds, argue."""

    def __init__(self, n_rounds: int, seed: int, inject_defect: Optional[str]) -> None:
        if not isinstance(n_rounds, int) or isinstance(n_rounds, bool):
            raise GateError(f"n_rounds must be an int; got {n_rounds!r}")
        if n_rounds < MIN_ROUNDS:
            raise GateError(
                f"n_rounds must be at least {MIN_ROUNDS}: the always-failing bot "
                f"first probes out of quarantine at round 100 and a shorter run "
                f"would pass vacuously; got {n_rounds}"
            )
        if inject_defect is not None and inject_defect not in DEFECTS:
            raise GateError(
                f"unknown defect {inject_defect!r}; known: {', '.join(DEFECTS)}"
            )
        self.n_rounds = n_rounds
        self.seed = int(seed)
        self.inject_defect = inject_defect
        self.fields = SeedFields.parse(self.seed)

        self.problems: List[Problem] = []
        self.suppressed = 0
        self._per_kind: Dict[str, int] = collections.defaultdict(int)
        self.counts: Dict[str, int] = collections.defaultdict(int)

        self.clock = _Clock(GATE_EPOCH)
        self.alerts = _RecordingAlerts()
        #: A plain dict is a MutableMapping, which the supervisor accepts as
        #: a store, so "persistence is ours" is exercised on every pause
        #: without this module opening a file.
        self.store: Dict[str, Any] = {}

        self._build_fleet()
        self._build_script()

        supervisor_class = _DEFECT_CLASSES.get(inject_defect or "", Supervisor)
        self.sup: Supervisor = supervisor_class(
            self.registry,
            self.clock,
            store=self.store,
            alerts=self.alerts,
            alert_min_severity=GATE_ALERT_MIN,
        )

        # -- the gate's own truth, derived only from the bots' logs --------
        self.paused_now: Set[str] = set()
        self.open_keys: Set[Tuple[str, str]] = set()
        self.since: Dict[Tuple[str, str], float] = {}
        self.episodes: Dict[Tuple[str, str], int] = collections.defaultdict(int)
        self.fails: Dict[str, int] = {b.info.id: 0 for b in self.bots}
        self.total_fails: Dict[str, int] = {b.info.id: 0 for b in self.bots}
        self.total_attempts: Dict[str, int] = {b.info.id: 0 for b in self.bots}
        self.first_quarantine_at: Dict[str, int] = {}
        self.max_attention = 0
        #: The gate's own copy of the supervisor's alert memory: the keys a
        #: push has already gone out for and that are still open.  Written
        #: when a key opens, dropped when it closes, by either mechanism.
        self.alerted: Set[Tuple[str, str]] = set()
        #: Every severity the fleet has actually emitted.
        self._severities_seen: Set[Severity] = set()
        #: Keys closed by their own bot's resolution event, and keys closed
        #: by the app calling clear_attention.  Both counted, because the
        #: gate used to exercise only the second.
        self.closed_by_bot = 0
        self.closed_by_app = 0
        self.reopened_after_resolution = 0
        self.shapes_rejected: Set[str] = set()
        self._expect_alerts: List[Tuple[str, Optional[str]]] = []
        self._ticked_this_round: Set[str] = set()
        self._failed_this_round: Set[str] = set()
        self._events_this_round = 0

    # -- construction ---------------------------------------------------------

    def _build_fleet(self) -> None:
        """Build the live fleet and the registry the run ticks."""
        self.bots, self.registry, self.max_keys = self._make_fleet(self.clock)
        self.by_id: Dict[str, _GateBot] = {b.info.id: b for b in self.bots}

    def _make_fleet(
        self, clock: _Clock
    ) -> Tuple[List["_GateBot"], BotRegistry, int]:
        """Nine bots, healthy one last.

        Registration order is the tick order (registry.py: "Order is
        registration order"), so putting the control at the end means a
        supervisor that lets one bot take down a round fails the schedule
        check rather than passing by luck.

        Called twice: once for the fleet the run drives, and once per
        restart check for a *fresh* fleet that has never ticked, so what
        comes back out of the store is compared against something that
        could only have got its state from the store.  Every draw is
        ``SeedFields.stream(label)``, which is a pure function of the seed
        and the label, so the second fleet is built from the same numbers
        as the first.
        """

        def info(bot_id: str, name: str, kind: str, interval: float) -> BotInfo:
            return BotInfo(
                id=bot_id,
                name=name,
                blurb=f"Synthetic {bot_id} bot, for the framework gate.",
                kind=kind,
                interval_s=interval,
                href=f"/bots/{bot_id}",
            )

        flaky = _FlakyBot(
            clock,
            info("flaky", "Flaky bot", "radar", BASE_INTERVAL_S),
            _fail_pattern(self.fields.stream("bots.flaky.pattern"), 24),
        )
        burster = _BursterBot(
            clock,
            info("burster", "Bursting bot", "grid", BURST_INTERVAL_S),
            tuple(
                self.fields.stream("bots.burster.sizes").randint(5, 12) for _ in range(16)
            ),
        )
        bots: List[_GateBot] = [
            _AlwaysFailsBot(
                clock, info("always_fails", "Always-failing bot", "bot", BASE_INTERVAL_S)
            ),
            flaky,
            _MalformedBot(
                clock, info("malformed", "Malformed-event bot", "bot", BASE_INTERVAL_S)
            ),
            _FlapperBot(clock, info("flapper", "Flapping bot", "radar", BASE_INTERVAL_S)),
            burster,
            _PauserBot(clock, info("pauser", "Pausable bot", "cart", BASE_INTERVAL_S)),
            _ResolverBot(
                clock, info("resolver", "Self-closing bot", "radar", BASE_INTERVAL_S)
            ),
            _SlowBot(
                clock,
                info("slowpoke", "Slow bot", "bot", SLOW_INTERVAL_S),
                SLOW_BY_S,
            ),
            _HealthyBot(clock, info("healthy", "Healthy bot", "grid", BASE_INTERVAL_S)),
        ]
        #: Every key the fleet can ever open.  The badge may never exceed it.
        max_keys = len({(b.info.id, k) for b in bots for k in b.keys})
        return bots, BotRegistry(bots), max_keys

    def _build_script(self) -> None:
        """The driver's own seeded decisions: when the owner pauses, when a
        question gets answered, when a card's ``status()`` breaks."""
        pause_stream = self.fields.stream("bots.script.pause")
        self.pause_at = pause_stream.randint(20, 40)
        hold = pause_stream.randint(12, 24)
        self.resume_at = self.pause_at + hold
        self.flap_clears = _every_few(
            self.fields.stream("bots.script.flap"), 8, self.n_rounds, 6, 14
        )
        self.broken_status = _every_few(
            self.fields.stream("bots.script.status"), 12, self.n_rounds, 17, 41
        )
        #: Rounds on which the gate restarts from the store.  Not every
        #: round: a restart rebuilds the whole fleet, and the check is
        #: about the seam, not about how often it is crossed.
        self.restart_at = _every_few(
            self.fields.stream("bots.script.restart"), 25, self.n_rounds, 29, 53
        )

    # -- problem recording -----------------------------------------------------

    def _fail(self, kind: str, detail: str, bot_id: str = "", round_index: int = -1) -> None:
        self._per_kind[kind] += 1
        if self._per_kind[kind] > MAX_PER_KIND:
            self.suppressed += 1
            return
        self.problems.append(Problem(kind, detail, bot_id, round_index))

    # -- the run ----------------------------------------------------------------

    def run(self) -> GateReport:
        try:
            self._drive()
            self._check_scenario()
        except GateError:
            raise
        except Exception as exc:  # noqa: BLE001 - the gate reports, never throws
            self._fail("crashed", f"the gate itself raised: {type(exc).__name__}: {exc}")
        return GateReport(
            n_rounds=self.n_rounds,
            seed=self.seed,
            inject_defect=self.inject_defect,
            counts=dict(sorted(self.counts.items())),
            problems=list(self.problems),
            suppressed=self.suppressed,
        )

    def _drive(self) -> None:
        for r in range(self.n_rounds):
            at = GATE_EPOCH + r * ROUND_S
            self.clock.set(at)
            self._commands(r)

            seen = {b.info.id: len(b.rounds) for b in self.bots}
            before = len(self.alerts.published)
            report = None
            try:
                report = self.sup.run_round()
            except Exception as exc:  # noqa: BLE001 - that is the defect
                self.counts["round_crashes"] += 1
                self._fail(
                    "round_crashed",
                    f"run_round raised instead of isolating the tick: "
                    f"{type(exc).__name__}: {exc}",
                    round_index=r,
                )
            fresh_alerts = self.alerts.published[before:]

            self._absorb(r, at, seen)
            self._check_alerts(r, fresh_alerts)
            self._apply_clears(r)
            self._check_attention(r)
            self._check_persistence(r)
            if r in self.restart_at:
                self._check_restart(r)
            self._check_badge(r)
            self._check_launcher(r)
            self._check_health(r, at, report, len(fresh_alerts))
            self.counts["rounds"] += 1

    def _commands(self, r: int) -> None:
        """What the owner and the app do between rounds: the pause button,
        an answered question, and a card whose ``status()`` is about to
        break."""
        if r == self.pause_at:
            self.sup.pause("pauser")
            self.paused_now.add("pauser")
            # contracts.py: "A paused bot is not ticked and reports no
            # attention" -- so the truth drops its keys the moment the owner
            # flips the switch, and the supervisor had better agree.
            for pair in [p for p in self.open_keys if p[0] == "pauser"]:
                self.open_keys.discard(pair)
                self.since.pop(pair, None)
                self.alerted.discard(pair)
            self.counts["pauses"] += 1
        if r == self.resume_at:
            self.sup.resume("pauser")
            self.paused_now.discard("pauser")
            self.counts["resumes"] += 1
        if self.pause_at <= r < self.resume_at:
            self.counts["paused_rounds"] += 1
        self.by_id["malformed"].break_status = r in self.broken_status
        if r in self.broken_status:
            self.counts["status_breaks"] += 1

    def _apply_clears(self, r: int) -> None:
        """The app answering the flapper's standing question."""
        if r not in self.flap_clears:
            return
        pair = ("flapper", SHARED_KEY)
        closed = self.sup.clear_attention(*pair)
        if pair in self.open_keys:
            self.open_keys.discard(pair)
            self.since.pop(pair, None)
            self.alerted.discard(pair)
            self.counts["attention_cleared"] += 1
            self.closed_by_app += 1
            if not closed:
                self._fail(
                    "attention_wrong",
                    "clear_attention() said nothing was open for a key the fleet "
                    "had opened and never closed",
                    bot_id="flapper",
                    round_index=r,
                )

    # -- truth, from the bots' own logs -----------------------------------------

    def _absorb(self, r: int, at: float, seen: Dict[str, int]) -> None:
        """Read what each bot recorded doing this round and move the gate's
        model forward.  Nothing here reads the supervisor."""
        self._expect_alerts = []
        self._ticked_this_round = set()
        self._failed_this_round = set()
        self._events_this_round = 0

        for bot in self.bots:
            bot_id = bot.info.id
            entered = bot.rounds[seen[bot_id] :]
            if not entered:
                continue
            if len(entered) != 1 or entered[0] != r:
                self._fail(
                    "round_report_wrong",
                    f"entered tick {len(entered)} time(s) in one round, at rounds "
                    f"{entered}",
                    bot_id=bot_id,
                    round_index=r,
                )
            index = seen[bot_id]
            outcome = bot.outcomes[index]
            returned = bot.returned[index]
            clean = None if outcome == "raise" else _clean_events(bot_id, returned)
            self._ticked_this_round.add(bot_id)
            self.total_attempts[bot_id] += 1
            self.counts["ticks"] += 1

            if bot_id in self.paused_now:
                self._fail(
                    "paused_ticked",
                    "a paused bot was ticked; contracts.py: 'A paused bot is not "
                    "ticked'",
                    bot_id=bot_id,
                    round_index=r,
                )

            if clean is None:
                self.fails[bot_id] += 1
                self.total_fails[bot_id] += 1
                self._failed_this_round.add(bot_id)
                self.counts["failures"] += 1
                if outcome == "raise":
                    self.counts["raises"] += 1
                else:
                    self.counts["rejections"] += 1
                    if isinstance(bot, _MalformedBot):
                        self.shapes_rejected.add(
                            _MalformedBot.SHAPES[index % len(_MalformedBot.SHAPES)]
                        )
                if (
                    self.fails[bot_id] >= QUARANTINE_AFTER_FAILURES
                    and bot_id not in self.first_quarantine_at
                ):
                    self.first_quarantine_at[bot_id] = self.fails[bot_id]
                    self.counts["quarantines"] += 1
                if self.fails[bot_id] > QUARANTINE_AFTER_FAILURES:
                    self.counts["probes"] += 1
                continue

            self.fails[bot_id] = 0
            self._events_this_round += len(clean)
            self.counts["events"] += len(clean)
            if len(clean) >= 5:
                self.counts["bursts"] += 1
            if bot_id in self.paused_now:
                # A paused bot should not have run at all; its events are not
                # part of the truth, and the mismatch is already reported.
                continue
            for event in clean:
                self._absorb_event(event, r)

    def _absorb_event(self, event: Event, r: int) -> None:
        """One event, against contracts.py's badge and alert rules.

        Three things an event can be, in the order the contract reads them:
        it opens a request for a decision, it closes one, or it is a line
        in the feed.  Then, independently, it is worth a push or it is not.

        The alert rule, stated here rather than borrowed: a push goes out
        for every event at or above ``GATE_ALERT_MIN`` *except* one whose
        attention key has already been pushed about and is still open --
        which is what stops one restock alerting every round.  The memory
        belongs to the open question, so it is written when the question
        opens and dropped when it closes, by either mechanism.
        """
        if event.wants_attention:
            pair = (event.bot_id, str(event.attention_key))
            if pair not in self.open_keys:
                self.open_keys.add(pair)
                self.since[pair] = event.at
                self.episodes[pair] += 1
                self.counts["attention_opened"] += 1
                if self.episodes[pair] > 1:
                    self.reopened_after_resolution += 1
            # A repeat of a key already pushed about does not push again;
            # a key that is open but was never pushed about (the alert
            # failed, the bot was paused when it opened) still can.
            if pair not in self.alerted:
                self._expect_alerts.append(pair)
                self.alerted.add(pair)
            return

        self._severities_seen.add(event.severity)
        self.counts["severities_seen"] = len(self._severities_seen)

        if event.attention_key and not event.wants_attention:
            self.counts["sub_action_keyed"] += 1

        if _closes_a_key(event):
            pair = (event.bot_id, str(event.attention_key))
            if pair in self.open_keys:
                self.open_keys.discard(pair)
                self.since.pop(pair, None)
                self.counts["attention_cleared"] += 1
                self.closed_by_bot += 1
            # The memory goes with the question even if the question was
            # not open here: "if the same question is asked again later it
            # is news again and earns a fresh push".
            self.alerted.discard(pair)

        if event.severity >= GATE_ALERT_MIN:
            key = str(event.attention_key) if event.attention_key else None
            pair = (event.bot_id, key) if key is not None else None
            if pair is None or (event.bot_id, key) not in self.alerted:
                self._expect_alerts.append((event.bot_id, key))

    # -- checks -------------------------------------------------------------------

    def _check_alerts(self, r: int, fresh: List[Dict[str, Any]]) -> None:
        """"Events are the only output": every push traces to one event, a
        newly opened key pushes exactly once, and a key that was already
        open pushes not at all."""
        actual: "collections.Counter[Tuple[str, Optional[str]]]" = collections.Counter()
        for row in fresh:
            data = row["data"]
            bot_id = str(data.get("bot_id", ""))
            key = data.get("attention_key")
            actual[(bot_id, key if key is None else str(key))] += 1
            self.counts["alerts"] += 1
            if bot_id in self.paused_now:
                self._fail(
                    "paused_alert",
                    "a paused bot's event reached the phone; contracts.py: 'a badge "
                    "asking you to act on something you switched off is a lie'",
                    bot_id=bot_id,
                    round_index=r,
                )
            if key == POISON_KEY:
                self._fail(
                    "malformed_kept",
                    f"an event stamped {POISON_BOT_ID!r} by another bot was pushed",
                    bot_id=bot_id,
                    round_index=r,
                )
            if row["kind"] != "bot_event" or not row["profile_id"]:
                self._fail(
                    "alert_unexpected",
                    f"alert with kind={row['kind']!r} profile={row['profile_id']!r}",
                    bot_id=bot_id,
                    round_index=r,
                )

        expected: "collections.Counter[Tuple[str, Optional[str]]]" = collections.Counter(
            self._expect_alerts
        )
        for pair, n in (actual - expected).items():
            bot_id, key = pair
            if key is not None:
                self._fail(
                    "realerted",
                    f"{n} extra push(es) for attention key {key!r}, which nothing "
                    f"newly opened this round; contracts.py collapses repeats of a "
                    f"key into one item, and one push",
                    bot_id=bot_id,
                    round_index=r,
                )
            else:
                self._fail(
                    "alert_unexpected",
                    f"{n} push(es) no event in this round asked for",
                    bot_id=bot_id,
                    round_index=r,
                )
        for pair, n in (expected - actual).items():
            bot_id, key = pair
            self._fail(
                "alert_missing",
                f"{n} push(es) missing for "
                + (f"newly opened key {key!r}" if key else "a keyless ACTION event"),
                bot_id=bot_id,
                round_index=r,
            )
        if self.sup.last_alert_error:
            self._fail(
                "alert_unexpected",
                f"the alert service reported an error: {self.sup.last_alert_error}",
                round_index=r,
            )

    def _check_attention(self, r: int) -> None:
        """The badge rule: "The badge counts distinct open keys"."""
        items = self.sup.attention_items()
        got = {(item.bot_id, item.key) for item in items}
        count = self.sup.attention_count()
        self.max_attention = max(self.max_attention, count)

        if len(items) != len(got):
            self._fail(
                "attention_wrong",
                f"{len(items)} items over {len(got)} distinct keys: repeats did not "
                f"collapse",
                round_index=r,
            )
        if got != self.open_keys:
            extra = sorted(f"{b}/{k}" for b, k in got - self.open_keys)
            missing = sorted(f"{b}/{k}" for b, k in self.open_keys - got)
            self._fail(
                "attention_wrong",
                f"open set differs from the keys the fleet opened: "
                f"extra={extra[:4]} missing={missing[:4]}",
                round_index=r,
            )
        if count != len(self.open_keys):
            self._fail(
                "attention_wrong",
                f"attention_count()={count}, distinct open keys={len(self.open_keys)}",
                round_index=r,
            )
        if count > self.max_keys:
            self._fail(
                "attention_unbounded",
                f"attention_count()={count} but the whole fleet declares only "
                f"{self.max_keys} key(s): the badge is growing without bound",
                round_index=r,
            )
        per_bot = sum(self.sup.attention_count(b.info.id) for b in self.bots)
        if per_bot != count:
            self._fail(
                "attention_wrong",
                f"per-bot counts sum to {per_bot} but attention_count() is {count}",
                round_index=r,
            )
        for bot_id in sorted(self.paused_now):
            if self.sup.attention_count(bot_id) != 0:
                self._fail(
                    "paused_attention",
                    "a paused bot still holds attention; contracts.py: 'a paused bot "
                    "... reports no attention'",
                    bot_id=bot_id,
                    round_index=r,
                )
        for item in items:
            pair = (item.bot_id, item.key)
            want = self.since.get(pair)
            if want is not None and abs(item.since - want) > EPS:
                self._fail(
                    "attention_since_moved",
                    f"since={item.since!r} for a question open since {want!r}: a "
                    f"repeat restamped it",
                    bot_id=item.bot_id,
                    round_index=r,
                )
        if self.sup.attention_count(POISON_BOT_ID) != 0:
            self._fail(
                "malformed_kept",
                "a foreign-stamped event filed attention under another bot",
                bot_id=POISON_BOT_ID,
                round_index=r,
            )

    def _check_persistence(self, r: int) -> None:
        """The unfiltered view, through the seam that has to survive a restart.

        :meth:`Supervisor.attention_items` filters out paused bots at query
        time, which makes "a paused bot reports no attention" true by
        construction and therefore untestable from the outside.
        :meth:`Supervisor.save_state` serialises the *raw* dictionaries, so
        this is where the gate can tell "cleared on pause" apart from "hidden
        on pause" -- and where contracts.py's "A bot hands over a JSON-able
        snapshot" is actually checked to be JSON-able.
        """
        state = self.sup.save_state()
        try:
            json.dumps(state)
        except (TypeError, ValueError) as exc:
            self._fail(
                "state_not_json",
                f"the snapshot the store is handed is not JSON: {exc}",
                round_index=r,
            )
        if self.sup.last_state_error:
            self._fail(
                "state_not_json",
                f"a snapshot failed: {self.sup.last_state_error}",
                round_index=r,
            )
        held = {(str(row["bot_id"]), str(row["key"])) for row in state["attention"]}
        for bot_id in sorted(self.paused_now):
            kept = sorted(k for b, k in held if b == bot_id)
            if kept:
                self._fail(
                    "paused_attention",
                    f"a paused bot still holds {kept} in the persisted state; "
                    f"contracts.py: 'a paused bot ... reports no attention', and a "
                    f"hidden item comes back at the next restart",
                    bot_id=bot_id,
                    round_index=r,
                )
        if held != self.open_keys:
            extra = sorted(f"{b}/{k}" for b, k in held - self.open_keys)
            missing = sorted(f"{b}/{k}" for b, k in self.open_keys - held)
            self._fail(
                "attention_wrong",
                f"the persisted open set differs from the keys the fleet opened: "
                f"extra={extra[:4]} missing={missing[:4]}",
                round_index=r,
            )
        alerted = {(str(row[0]), str(row[1])) for row in state["alerted"]}
        stale = sorted(f"{b}/{k}" for b, k in alerted - held)
        if stale:
            self._fail(
                "alert_memory_leak",
                f"the alert memory outlived the questions it was about: {stale[:4]}; "
                f"it grows for as long as the process runs and re-suppresses a push "
                f"the owner should get",
                round_index=r,
            )
        for bot in self.bots:
            persisted = bool(state["bots"][bot.info.id]["paused"])
            if persisted != (bot.info.id in self.paused_now):
                self._fail(
                    "paused_wrong",
                    f"persisted paused={persisted}, the owner's switch says "
                    f"{bot.info.id in self.paused_now}; contracts.py writes a pause "
                    f"through so it survives a restart",
                    bot_id=bot.info.id,
                    round_index=r,
                )

    def _check_restart(self, r: int) -> None:
        """Actually restart: build a fresh fleet and a fresh supervisor over
        what is in the store, and see whether anything came back.

        The store was written to on every round and never once read.  That
        made "State is the bot's, persistence is ours" a claim about a dict
        nobody opened: a supervisor that serialised everything correctly
        and restored *nothing* passed, and so did one whose bots had no
        state to lose -- which, until ``_GateBot`` grew a snapshot, was all
        of them.

        The clone is a fleet that has never ticked, so every field it has
        can only have come through the seam.  It is built with the injected
        defect class as well, because the seam is a place a defect can
        live.
        """
        self.counts["restarts"] += 1
        supervisor_class = _DEFECT_CLASSES.get(self.inject_defect or "", Supervisor)
        clone_clock = _Clock(self.clock())
        clones, registry, _keys = self._make_fleet(clone_clock)
        store = json.loads(json.dumps(self.store))
        fresh = supervisor_class(
            registry,
            clone_clock,
            store=store,
            alerts=_RecordingAlerts(),
            alert_min_severity=GATE_ALERT_MIN,
        )
        if not fresh.load_state():
            self._fail(
                "state_not_restored",
                "load_state() found nothing in the store the supervisor has been "
                "writing to every round",
                round_index=r,
            )
            return

        by_id = {b.info.id: b for b in clones}
        for bot in self.bots:
            bot_id = bot.info.id
            want = bot.state_fingerprint()
            if not want:
                self._fail(
                    "scenario_thin",
                    "a gate bot has no state, so the persistence check is "
                    "comparing empty dicts",
                    bot_id=bot_id,
                    round_index=r,
                )
                continue
            got = by_id[bot_id].state_fingerprint()
            if got != want:
                self._fail(
                    "state_not_restored",
                    f"after a restart the bot's own state is {got!r}, not the "
                    f"{want!r} it handed over; contracts.py: 'A bot hands over a "
                    f"JSON-able snapshot and gets it back on the next start'",
                    bot_id=bot_id,
                    round_index=r,
                )
            live, back = self.sup.health(bot_id), fresh.health(bot_id)
            for field_name in (
                "consecutive_failures",
                "total_failures",
                "next_due_at",
                "quarantined_until",
            ):
                if getattr(live, field_name) != getattr(back, field_name):
                    self._fail(
                        "state_not_restored",
                        f"health.{field_name} came back "
                        f"{getattr(back, field_name)!r}, not "
                        f"{getattr(live, field_name)!r}; a restart that resets a "
                        f"failing bot's back-off is a restart that starts it "
                        f"hammering again",
                        bot_id=bot_id,
                        round_index=r,
                    )
            if fresh.is_paused(bot_id) != (bot_id in self.paused_now):
                self._fail(
                    "paused_wrong",
                    f"paused={fresh.is_paused(bot_id)} after a restart, the "
                    f"owner's switch says {bot_id in self.paused_now}",
                    bot_id=bot_id,
                    round_index=r,
                )
        got_keys = {(i.bot_id, i.key) for i in fresh.attention_items()}
        if got_keys != self.open_keys:
            extra = sorted(f"{b}/{k}" for b, k in got_keys - self.open_keys)
            missing = sorted(f"{b}/{k}" for b, k in self.open_keys - got_keys)
            self._fail(
                "state_not_restored",
                f"the open questions did not survive a restart: extra="
                f"{extra[:4]} missing={missing[:4]}",
                round_index=r,
            )
        for item in fresh.attention_items():
            want_since = self.since.get((item.bot_id, item.key))
            if want_since is not None and abs(item.since - want_since) > EPS:
                self._fail(
                    "attention_since_moved",
                    f"a restart restamped since to {item.since!r} for a question "
                    f"open since {want_since!r}",
                    bot_id=item.bot_id,
                    round_index=r,
                )
        if fresh.badge_status() != self.sup.badge_status():
            self._fail(
                "state_not_restored",
                f"the badge reads {fresh.badge_status()} after a restart and "
                f"{self.sup.badge_status()} before it",
                round_index=r,
            )

    def _quarantined_truth(self) -> Set[str]:
        """Who is quarantined, from the bots' own failure runs.

        contracts.py quarantines after ``QUARANTINE_AFTER_FAILURES``
        consecutive failures, and the supervisor documents that "A bot is out
        of quarantine when a tick of it succeeds, not when a timer elapses".
        Between rounds, those two sentences are exactly this set.
        """
        return {b for b, n in self.fails.items() if n >= QUARANTINE_AFTER_FAILURES}

    def _check_badge(self, r: int) -> None:
        """``GET /api/bots/status``, against web/README.md and the truth."""
        badge = self.sup.badge_status()
        if set(badge) != {"attention", "state"}:
            self._fail(
                "badge_wrong",
                f"badge keys are {sorted(badge)}; web/README.md documents "
                f"['attention', 'state']",
                round_index=r,
            )
            return
        attention, state = badge["attention"], badge["state"]
        if not isinstance(attention, int) or isinstance(attention, bool):
            self._fail(
                "badge_wrong", f"attention is {type(attention).__name__}, not an int",
                round_index=r,
            )
        elif attention != len(self.open_keys):
            self._fail(
                "badge_wrong",
                f"attention={attention}, distinct open keys={len(self.open_keys)}",
                round_index=r,
            )
        if state not in BADGE_STATES:
            self._fail(
                "badge_wrong", f"state={state!r} is not one of {sorted(BADGE_STATES)}",
                round_index=r,
            )
            return
        quarantined = self._quarantined_truth()
        want = "error" if quarantined else ("warn" if self.paused_now else "ok")
        if state != want:
            self._fail(
                "badge_wrong",
                f"state={state!r} but quarantined={sorted(quarantined)} "
                f"paused={sorted(self.paused_now)} makes it {want!r}",
                round_index=r,
            )

    def _check_launcher(self, r: int) -> None:
        """``GET /api/bots/``, key by key against web/README.md, then against
        the truth.  Both halves matter: a payload can be perfectly shaped and
        still say a paused bot is running."""
        # Read the clock at the moment of the call, not the round's start:
        # the slow bot moves it on inside its tick, and generated_at is
        # documented as when the payload was generated.
        called_at = int(self.clock())
        payload = self.sup.launcher_state()
        if set(payload) != {"generated_at", "bots"}:
            self._fail(
                "launcher_invalid",
                f"top-level keys are {sorted(payload)}; web/README.md documents "
                f"['bots', 'generated_at']",
                round_index=r,
            )
            return
        generated = payload["generated_at"]
        if not isinstance(generated, int) or isinstance(generated, bool):
            self._fail(
                "launcher_invalid",
                f"generated_at is {type(generated).__name__}; the README's example "
                f"is whole seconds",
                round_index=r,
            )
        elif generated != called_at:
            self._fail(
                "launcher_wrong",
                f"generated_at={generated} but the injected clock read {called_at}; "
                f"the page measures relative times against it",
                round_index=r,
            )
        cards = payload["bots"]
        if not isinstance(cards, list) or len(cards) != len(self.bots):
            self._fail(
                "launcher_invalid",
                f"bots is {type(cards).__name__} of "
                f"{len(cards) if isinstance(cards, list) else '?'}; "
                f"{len(self.bots)} are registered",
                round_index=r,
            )
            return
        try:
            if json.loads(json.dumps(payload)) != payload:
                raise ValueError("did not survive a JSON round trip")
        except (TypeError, ValueError) as exc:
            self._fail("launcher_invalid", f"payload is not JSON: {exc}", round_index=r)

        for bot, card in zip(self.bots, cards):
            self._check_card(r, bot, card)

    def _check_card(self, r: int, bot: _GateBot, card: Any) -> None:
        bot_id = bot.info.id
        if not isinstance(card, dict) or set(card) != set(CARD_KEYS):
            got = sorted(card) if isinstance(card, dict) else type(card).__name__
            self._fail(
                "launcher_invalid",
                f"card keys are {got}; web/README.md documents {list(CARD_KEYS)}",
                bot_id=bot_id,
                round_index=r,
            )
            return
        for name in ("id", "name", "blurb", "kind", "state", "href"):
            if not isinstance(card[name], str):
                self._fail(
                    "launcher_invalid",
                    f"{name} is {type(card[name]).__name__}, not a string",
                    bot_id=bot_id,
                    round_index=r,
                )
        if not isinstance(card["can_pause"], bool):
            self._fail(
                "launcher_invalid",
                f"can_pause is {type(card['can_pause']).__name__}, not a bool",
                bot_id=bot_id,
                round_index=r,
            )
        attention = card["attention"]
        if not isinstance(attention, int) or isinstance(attention, bool) or attention < 0:
            self._fail(
                "launcher_invalid",
                f"attention={attention!r} is not a count",
                bot_id=bot_id,
                round_index=r,
            )
        if card["state"] not in CARD_STATES:
            self._fail(
                "launcher_invalid",
                f"state={card['state']!r} is not one of {sorted(CARD_STATES)}",
                bot_id=bot_id,
                round_index=r,
            )
        stats = card["stats"]
        if not isinstance(stats, list) or any(
            not isinstance(s, dict)
            or set(s) != {"label", "value"}
            or not isinstance(s["label"], str)
            or not isinstance(s["value"], str)
            for s in stats
        ):
            self._fail(
                "launcher_invalid",
                "stats must be a list of {label, value} strings",
                bot_id=bot_id,
                round_index=r,
            )
        last = card["last_event"]
        if last is not None:
            if (
                not isinstance(last, dict)
                or set(last) != {"at", "text"}
                or not isinstance(last["at"], int)
                or isinstance(last["at"], bool)
                or not isinstance(last["text"], str)
            ):
                self._fail(
                    "launcher_invalid",
                    f"last_event must be null or {{at:int, text:str}}; got {last!r}",
                    bot_id=bot_id,
                    round_index=r,
                )

        # -- and now against the truth ------------------------------------
        if card["id"] != bot_id or card["name"] != bot.info.name:
            self._fail(
                "launcher_wrong",
                f"card identifies as {card['id']!r}/{card['name']!r}",
                bot_id=bot_id,
                round_index=r,
            )
        if card["kind"] != bot.info.kind or card["href"] != bot.info.href:
            self._fail(
                "launcher_wrong",
                f"kind/href are {card['kind']!r}/{card['href']!r}, not "
                f"{bot.info.kind!r}/{bot.info.href!r}",
                bot_id=bot_id,
                round_index=r,
            )
        if card["can_pause"] is not bool(bot.info.can_pause):
            self._fail(
                "launcher_wrong",
                f"can_pause={card['can_pause']!r} contradicts BotInfo",
                bot_id=bot_id,
                round_index=r,
            )
        want_attention = len([1 for b, _k in self.open_keys if b == bot_id])
        if isinstance(attention, int) and attention != want_attention:
            self._fail(
                "launcher_wrong",
                f"attention={attention} but the bot holds {want_attention} open key(s)",
                bot_id=bot_id,
                round_index=r,
            )
        want_state = self._card_state_truth(bot)
        if card["state"] != want_state:
            self._fail(
                "launcher_wrong",
                f"state={card['state']!r}, computed {want_state!r}",
                bot_id=bot_id,
                round_index=r,
            )
        if bot.break_status and stats:
            self._fail(
                "launcher_wrong",
                "a card whose status() raised still carried stats",
                bot_id=bot_id,
                round_index=r,
            )

    def _card_state_truth(self, bot: _GateBot) -> str:
        """What the card should say, computed rather than asked for.

        The supervisor's facts outrank the bot's own: paused first (the
        owner's decision), then quarantined, then a ``status()`` that raised,
        then whatever the bot reports about itself.
        """
        bot_id = bot.info.id
        if bot_id in self.paused_now:
            return BotState.PAUSED.ui
        if self.fails[bot_id] >= QUARANTINE_AFTER_FAILURES:
            return BotState.QUARANTINED.ui
        if bot.break_status:
            return BotState.QUARANTINED.ui
        return (BotState.RUNNING if bot.rounds else BotState.IDLE).ui

    def _check_health(
        self, r: int, at: float, report: Any, n_alerts: int
    ) -> None:
        """Health, backoff and the round's own arithmetic."""
        for bot in self.bots:
            bot_id = bot.info.id
            health = self.sup.health(bot_id)
            base = float(bot.info.interval_s)

            if health.total_ticks != self.total_attempts[bot_id]:
                self._fail(
                    "health_wrong",
                    f"total_ticks={health.total_ticks}, the bot recorded "
                    f"{self.total_attempts[bot_id]} attempt(s)",
                    bot_id=bot_id,
                    round_index=r,
                )
            if health.total_failures != self.total_fails[bot_id]:
                self._fail(
                    "health_wrong",
                    f"total_failures={health.total_failures}, the bot recorded "
                    f"{self.total_fails[bot_id]}",
                    bot_id=bot_id,
                    round_index=r,
                )
            if health.consecutive_failures != self.fails[bot_id]:
                self._fail(
                    "health_wrong",
                    f"consecutive_failures={health.consecutive_failures}, the bot "
                    f"recorded {self.fails[bot_id]}",
                    bot_id=bot_id,
                    round_index=r,
                )
            want_quarantined = self.fails[bot_id] >= QUARANTINE_AFTER_FAILURES
            if self.sup.is_quarantined(bot_id) != want_quarantined:
                self._fail(
                    "quarantine_wrong",
                    f"is_quarantined={self.sup.is_quarantined(bot_id)} after "
                    f"{self.fails[bot_id]} consecutive failure(s); "
                    f"QUARANTINE_AFTER_FAILURES is {QUARANTINE_AFTER_FAILURES}",
                    bot_id=bot_id,
                    round_index=r,
                )

            if bot_id not in self._ticked_this_round:
                continue
            gap = health.next_due_at - at
            if bot_id in self._failed_this_round:
                if gap < base - EPS:
                    self._fail(
                        "backoff_narrowed",
                        f"next attempt in {gap:.0f}s after failure "
                        f"{self.fails[bot_id]}, narrower than the base interval "
                        f"{base:.0f}s; contracts.py: 'Never narrower than base'",
                        bot_id=bot_id,
                        round_index=r,
                    )
                want = _expected_gap(base, self.fails[bot_id])
                if abs(gap - want) > EPS:
                    self._fail(
                        "backoff_wrong",
                        f"next attempt in {gap:.0f}s after failure "
                        f"{self.fails[bot_id]}; backoff_interval and QUARANTINE_S "
                        f"make it {want:.0f}s",
                        bot_id=bot_id,
                        round_index=r,
                    )
            else:
                if abs(gap - base) > EPS:
                    self._fail(
                        "schedule_wrong",
                        f"a successful tick rearmed at +{gap:.0f}s, not the declared "
                        f"interval_s={base:.0f}s",
                        bot_id=bot_id,
                        round_index=r,
                    )
                if health.last_error:
                    self._fail(
                        "health_wrong",
                        f"last_error survived a successful tick: "
                        f"{health.last_error!r}",
                        bot_id=bot_id,
                        round_index=r,
                    )

        self._check_healthy_control(r)
        if report is None:
            return
        n = report.ticked + report.failed + report.skipped
        if n != len(self.bots):
            self._fail(
                "round_report_wrong",
                f"ticked+failed+skipped={n} over {len(self.bots)} registered bots",
                round_index=r,
            )
        if report.failed != len(self._failed_this_round):
            self._fail(
                "round_report_wrong",
                f"failed={report.failed}, the bots recorded "
                f"{len(self._failed_this_round)}",
                round_index=r,
            )
        if report.ticked != len(self._ticked_this_round) - len(self._failed_this_round):
            self._fail(
                "round_report_wrong",
                f"ticked={report.ticked}, the bots recorded "
                f"{len(self._ticked_this_round) - len(self._failed_this_round)} "
                f"successful tick(s)",
                round_index=r,
            )
        if report.events != self._events_this_round:
            self._fail(
                "round_report_wrong",
                f"events={report.events}, the bots returned "
                f"{self._events_this_round}",
                round_index=r,
            )
        if report.alerts != n_alerts:
            self._fail(
                "round_report_wrong",
                f"alerts={report.alerts}, the alert service recorded {n_alerts}",
                round_index=r,
            )
        if abs(report.at - at) > EPS:
            self._fail(
                "round_report_wrong",
                f"report.at={report.at!r}, the round ran at {at!r}",
                round_index=r,
            )
        want_slow = {"slowpoke"} if "slowpoke" in self._ticked_this_round else set()
        if set(report.slow) != want_slow:
            self._fail(
                "slow_wrong",
                f"slow={sorted(report.slow)}; the only tick past SLOW_TICK_S="
                f"{SLOW_TICK_S:.0f}s this round was {sorted(want_slow)}",
                round_index=r,
            )
        if want_slow:
            self.counts["slow_rounds"] += 1

    def _check_healthy_control(self, r: int) -> None:
        """"A bot never blocks another", stated as a schedule.

        The healthy bot never fails, so its due times are a one-line
        consequence of ``interval_s`` and nothing else in the fleet.  Any
        failure recorded against it, or any round it was due and did not run,
        is a neighbour reaching across the isolation boundary.
        """
        bot = self.by_id["healthy"]
        health = self.sup.health("healthy")
        if health.total_failures or health.consecutive_failures or health.last_error:
            self._fail(
                "isolation_broken",
                f"the control bot picked up {health.total_failures} failure(s) from "
                f"its neighbours: {health.last_error!r}",
                bot_id="healthy",
                round_index=r,
            )
        if self.sup.is_quarantined("healthy") or self.sup.is_paused("healthy"):
            self._fail(
                "isolation_broken",
                "the control bot was quarantined or paused by its neighbours",
                bot_id="healthy",
                round_index=r,
            )
        want = self._healthy_schedule(r + 1)
        if bot.rounds != want:
            missed = [x for x in want if x not in bot.rounds][:4]
            extra = [x for x in bot.rounds if x not in want][:4]
            self._fail(
                "isolation_broken",
                f"the control bot ticked on {bot.rounds[-4:]} where the interval says "
                f"{want[-4:]} (missed={missed} unexpected={extra})",
                bot_id="healthy",
                round_index=r,
            )

    def _healthy_schedule(self, through: int) -> List[int]:
        """Which rounds a never-failing bot on ``BASE_INTERVAL_S`` is due on.

        Health starts at ``next_due_at = 0.0`` and the gate's clock starts
        well above it, so the first round is one of them.
        """
        due = 0.0
        out: List[int] = []
        for r in range(through):
            at = GATE_EPOCH + r * ROUND_S
            if due <= at:
                out.append(r)
                due = at + BASE_INTERVAL_S
        return out

    # -- the anti-vacuity guard ----------------------------------------------------

    def _check_scenario(self) -> None:
        """A green gate that never met a hazard is worse than no gate.

        Every line here is a thing the run *claims* to have exercised; if the
        script drifts and stops producing one, that is a failure of the gate,
        reported as such rather than passed over.
        """
        self.counts["max_attention"] = self.max_attention
        self.counts["alert_rows"] = len(self.alerts.published)
        self.counts["bots"] = len(self.bots)
        self.counts["attention_keys_declared"] = self.max_keys
        self.counts["episodes"] = sum(self.episodes.values())

        def want(condition: bool, detail: str) -> None:
            if not condition:
                self._fail("scenario_thin", detail)

        want(self.counts["quarantines"] >= 1, "no bot was ever quarantined")
        want(self.counts["probes"] >= 1, "no probe tick ran out of a quarantine")
        want(
            self.first_quarantine_at.get("always_fails") == QUARANTINE_AFTER_FAILURES,
            f"the always-failing bot first quarantined at failure "
            f"{self.first_quarantine_at.get('always_fails')!r}, not at "
            f"{QUARANTINE_AFTER_FAILURES}",
        )
        want(
            "flaky" not in self.first_quarantine_at and self.total_fails["flaky"] >= 2,
            f"the intermittent bot was not exercised as intermittent: "
            f"{self.total_fails['flaky']} failure(s), quarantined="
            f"{'flaky' in self.first_quarantine_at}",
        )
        pauser_ticks_in_window = [
            r for r in self.by_id["pauser"].rounds if self.pause_at <= r < self.resume_at
        ]
        want(
            self.counts["pauses"] == 1 and self.counts["resumes"] == 1,
            f"pause/resume did not both happen: {self.counts['pauses']}/"
            f"{self.counts['resumes']}",
        )
        want(
            (self.resume_at - self.pause_at) * ROUND_S >= 2 * BASE_INTERVAL_S,
            "the pause window was shorter than two of the paused bot's intervals, so "
            "'never ticked while paused' would hold vacuously",
        )
        want(not pauser_ticks_in_window, "the paused bot ticked inside the pause window")
        want(self.counts["attention_cleared"] >= 2, "no question was ever answered")
        want(
            self.episodes.get(("flapper", SHARED_KEY), 0) >= 3,
            f"the flapping key opened "
            f"{self.episodes.get(('flapper', SHARED_KEY), 0)} time(s); it should flap",
        )
        want(self.counts["bursts"] >= 5, f"only {self.counts['bursts']} burst(s)")
        want(
            self.counts.get("severities_seen", 0) == len(Severity),
            f"the fleet emitted {self.counts.get('severities_seen', 0)} of "
            f"{len(Severity)} severities; an unmapped one raises inside the "
            f"publish and is swallowed into last_alert_error",
        )
        want(self.counts["slow_rounds"] >= 1, "no slow tick was reported")
        want(self.counts["status_breaks"] >= 1, "no card's status() ever broke")
        missing_shapes = [
            s for s in set(_MalformedBot.SHAPES) - {"clean"} if s not in self.shapes_rejected
        ]
        want(
            not missing_shapes,
            f"malformed shapes never rejected: {sorted(missing_shapes)}",
        )
        want(self.counts["alerts"] >= 8, f"only {self.counts['alerts']} push(es)")

        # The bot-side close path.  Every one of these was zero before the
        # resolver bot existed, and the gate said OK.
        self.counts["closed_by_bot"] = self.closed_by_bot
        self.counts["closed_by_app"] = self.closed_by_app
        self.counts["reopened"] = self.reopened_after_resolution
        want(
            self.closed_by_bot >= 3,
            f"only {self.closed_by_bot} question(s) were closed by the bot that "
            f"asked them (RESOLVED_FLAG); both shipped bots close this way, and "
            f"a gate that never sees it cannot tell a working close path from a "
            f"dead one",
        )
        want(
            self.closed_by_app >= 2,
            f"only {self.closed_by_app} question(s) were closed by the app "
            f"calling clear_attention; both closing mechanisms have to be "
            f"exercised or one of them is untested",
        )
        want(
            self.reopened_after_resolution >= 2,
            f"a closed question re-opened only "
            f"{self.reopened_after_resolution} time(s); "
            f"clear_attention promises it 'is news again and earns a fresh "
            f"push', and a run that never re-opens one cannot see that "
            f"promise broken",
        )
        # A sub-ACTION event carrying a live attention key is what poisons
        # the alert memory, and it only reaches the alert path at all
        # because GATE_ALERT_MIN is below ACTION.
        want(
            self.counts.get("sub_action_keyed", 0) >= 4,
            f"only {self.counts.get('sub_action_keyed', 0)} event(s) below "
            f"ACTION carried an attention key, so the alert memory was never "
            f"offered a chance to outlive the question it was about",
        )
        want(
            GATE_ALERT_MIN < Severity.ACTION,
            "the run pushed only at ACTION, so every event that could push was "
            "also one that opened a badge item and the alert rules were never "
            "tested apart from the badge rules",
        )
        want(
            self.counts.get("restarts", 0) >= 2,
            f"the store was read back {self.counts.get('restarts', 0)} time(s); "
            f"a persistence rule checked only by writing is not checked",
        )
        stateful = [b.info.id for b in self.bots if b.state_fingerprint()]
        want(
            len(stateful) == len(self.bots),
            f"only {len(stateful)} of {len(self.bots)} bots have any state, so "
            f"'a bot hands over a JSON-able snapshot' is being checked against "
            f"empty dicts",
        )
        want(self.max_attention >= 3, f"the badge never reached 3 (max {self.max_attention})")
        want(
            self.counts["round_crashes"] == 0 or self.inject_defect is not None,
            "a healthy run should never see run_round raise",
        )


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def run_gate(
    n_rounds: int = DEFAULT_ROUNDS,
    seed: int = DEFAULT_SEED,
    inject_defect: Optional[str] = None,
    *,
    defect: Optional[str] = None,
) -> GateReport:
    """Run ``n_rounds`` supervisor rounds against the synthetic fleet.

    Everything is a pure function of ``(n_rounds, seed, inject_defect)``:
    the clock is a cell this module advances, the only randomness comes from
    :class:`lucifer_gen.seed.SeedFields`, and no module in the run opens a
    socket.

    ``inject_defect`` is one of :data:`DEFECTS` and must be reported -- the
    gate is shown to fail before it is trusted.  ``defect`` is the same
    argument under the name ``jarvis_bots.cli gate`` supplies; passing both
    is an error rather than a guess.

    Raises :class:`GateError` for a run too short to contain the hazards it
    claims to test, or for a defect name it does not know.
    """
    if defect is not None:
        if inject_defect is not None and inject_defect != defect:
            raise GateError(
                f"inject_defect={inject_defect!r} and defect={defect!r} disagree"
            )
        inject_defect = defect
    return _Gate(n_rounds, seed, inject_defect).run()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m jarvis_bots.validate",
        description="Run the framework gate: bot isolation and the launcher contract.",
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--seed", default=format_seed(DEFAULT_SEED))
    parser.add_argument("--defect", choices=list(DEFECTS), default=None)
    parser.add_argument("--json", action="store_true", help="the report as JSON")
    parser.add_argument(
        "--show-defects",
        action="store_true",
        help="run every defect and exit 1 if any goes unreported",
    )
    args = parser.parse_args(argv)

    try:
        seed = parse_seed(args.seed)
    except Exception:  # noqa: BLE001 - one line, never echo the value
        print("error: --seed must be an integer, decimal or 0x hex", file=sys.stderr)
        return 2

    try:
        if args.show_defects:
            missed = 0
            for defect in DEFECTS:
                report = run_gate(args.rounds, seed, defect)
                caught = not report.ok
                missed += 0 if caught else 1
                kinds = ", ".join(report.kinds()) if report.problems else "nothing"
                print(f"{'caught ' if caught else 'MISSED '} {defect:<20} {kinds}")
            healthy = run_gate(args.rounds, seed)
            print(healthy.summary())
            if not healthy.ok:
                missed += 1
            return 0 if missed == 0 else 1
        report = run_gate(args.rounds, seed, args.defect)
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
