"""The delivery worker: the loop that turns outbox rows into push sends.

Design: contracts.py, module docstring. This file is the *worker* side of
design point 1 (the durable outbox: rows are leased, not popped, so a
worker that dies mid-send hands the row back after a visibility timeout),
the caller of design point 2 (sending goes through the subscription
registry and an injected transport), and the producer for design point 3
(every delivery attempt is recorded; retries are at-least-once and the
client collapses duplicates by alert id).

One pass of :meth:`Worker.run_once` is::

    outbox.revive_exhausted(now)                     # dead letters whose outage may be over
    rows = outbox.lease(now, batch, lease_s)
    for each row:
        subscription = outbox.subscription(profile_id, device_id)
        transport    = transports[subscription.transport]
        result       = transport.send(subscription, outbox.alert(alert_id))
        outbox.mark(row_id, result, now, jitter)     # records the attempt

The send happens *before* the mark on purpose: if the process dies between
the two, the lease expires, the row is sent again and the client dedupes.
That is the at-least-once guarantee contracts.py asks for; the other order
(mark, then send) would be at-most-once and could lose the alert.

Three guards keep "at least once" from becoming "many times":

* a row is not sent once its lease has run out (``clock() >= lease_until``
  at send time): another worker may already hold it.  Such a row is
  counted as ``expired`` and comes back through the normal reclaim;
* a send whose *mark* raised (the store was unreachable for a moment) is
  remembered in the worker and applied at the start of its next pass, or
  when the row is next leased to it, instead of being sent again.  Only
  the process dying between send and mark still produces a repeat;
* a row whose device turns out to be pruned (a sibling row in the same
  batch was the device's PRUNE_AFTER_FAILURES-th failure) is *released*
  back to the store untouched (``OutboxPort.release``), not dead-lettered:
  the device is retried after its cooldown and the row goes with it.

Division of labour with the outbox
----------------------------------
The outbox owns the retry policy: ``mark`` records the attempt, applies
``backoff_seconds`` / ``MAX_ATTEMPTS`` / ``PRUNE_AFTER_FAILURES`` and
returns the row's new state, and ``revive_exhausted`` puts a row that
died only because an outage outlasted its budget (``DeadReason.EXHAUSTED``)
back after ``EXHAUSTED_RETRY_COOLDOWN_S``, while its alert is younger
than ``ALERT_MAX_AGE_S``.  The worker only decides *what result to
report* for a row, turns the returned state into counts, and calls
``revive_exhausted`` at the start of every pass so that no human has to
notice an outage for its alerts to go out once it clears (``revived`` in
the report).  The worker does supply the backoff jitter, drawn from its
injected ``jitter`` callable exactly once per backoff that will happen --
counted within the row's current budget (``OutboxRow.attempts_base``), so
a re-queued row gets its jitter too -- so a seeded
``lucifer_gen.seed.Stream`` is consumed in a reproducible order.

Rows the worker dead-letters itself, all through ``mark`` with a permanent
failure so the attempt is recorded like any other, tagged with the
``DeadReason`` the outbox stores (never EXHAUSTED: none of these is an
outage, so none is revived):

    "no subscription"   the device vanished or was reported gone by the
                        transport between lease and lookup (the outbox
                        never leases such rows otherwise); a device merely
                        *pruned* in that window gets its row released
                        instead, reason "device pruned"   (NO_SUBSCRIPTION)
    "no transport"      ``subscription.transport`` names nothing injected
                        (NO_TRANSPORT)
    "no alert"          the row's alert is missing from the store (PERMANENT)

A transport that raises is reported as a retryable failure with reason
``"transport raised <Type>"``; only the type name, never the message.

Privacy (contracts.py, Subscription.blob "never logged"): nothing here
logs, prints or embeds the blob or a transport's error text. Log lines and
reasons carry row ids, (profile_id, device_id) and exception *type names*.

Determinism: the worker never calls ``time.time()`` or ``random`` in its
logic. The clock and the jitter are injected callables; the tests drive them
with :class:`SimClock` and a seed stream. ``run_forever`` waits on the stop
event, which is the loop, not the logic.

Store interface
---------------
:class:`OutboxPort` lists every call the worker makes; ``jarvis_alerts.outbox.
Outbox`` satisfies it and :class:`MemoryStore` is an in-memory twin with the
same semantics, used by the tests and by ``python3 -m jarvis_alerts.worker``.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Set, Tuple

from .contracts import (
    ALERT_MAX_AGE_S,
    BACKOFF_CAP_S,
    EXHAUSTED_RETRY_COOLDOWN_S,
    LEASE_S,
    MAX_ATTEMPTS,
    PRUNE_AFTER_FAILURES,
    PRUNE_COOLDOWN_S,
    Alert,
    DeadReason,
    DeliveryAttempt,
    OutboxRow,
    RowState,
    SendResult,
    Subscription,
    Transport,
    backoff_seconds,
    dead_reason_for,
)

log = logging.getLogger(__name__)

Clock = Callable[[], float]
Jitter = Callable[[], float]


def no_jitter() -> float:
    """The default jitter source: none, so backoff is half the nominal delay."""
    return 0.0


# ---------------------------------------------------------------------------
# What the worker asks of the store.
# ---------------------------------------------------------------------------


class OutboxPort(Protocol):
    """The outbox as the worker sees it; ``jarvis_alerts.outbox.Outbox`` is one.

    The store holds the subscriptions too (contracts.py stores the blob as
    the app's own data next to the rows), which is why the brief's worker
    signature has no separate registry.
    """

    def lease(self, now: float, limit: int, lease_s: float = LEASE_S) -> List[OutboxRow]:
        """Atomically take up to ``limit`` rows that are due at ``now``:
        PENDING with ``next_due <= now`` and a live subscription, after
        reclaiming rows whose lease expired. Each comes back LEASED with
        ``lease_until = now + lease_s`` and goes to no other caller."""
        ...

    def mark(self, row_id: int, result: SendResult, now: float,
             jitter: Optional[float] = None) -> RowState:
        """Record the attempt, apply the retry policy, return the new state.
        ``jitter`` in [0, 1] is used for the backoff when one is needed."""
        ...

    def release(self, row_id: int, reason: str = "") -> RowState:
        """Hand a LEASED row back untouched: PENDING again, no attempt
        recorded, ``last_reason = reason``.  Returns the row's state."""
        ...

    def revive_exhausted(self, now: float) -> int:
        """Put every row DEAD as ``EXHAUSTED`` for at least
        ``EXHAUSTED_RETRY_COOLDOWN_S``, of an alert younger than
        ``ALERT_MAX_AGE_S`` at ``now`` and of a reachable device, back to
        PENDING due now with a fresh budget; return how many.  Rows dead
        for any other reason are never touched."""
        ...

    def subscription(self, profile_id: str, device_id: str) -> Optional[Subscription]:
        """The device's subscription, gone or not, or None if unknown."""
        ...

    def alert(self, alert_id: str) -> Optional[Alert]: ...

    def stats(self, dead_limit: int = 50) -> Dict[str, Any]:
        """Counts per state; :func:`drain` reads ``pending``, ``leased``,
        ``pending_unreachable`` (PENDING rows with no live subscription)
        and ``dead_revivable`` (DEAD rows ``revive_exhausted`` will put
        back once their cooldown has passed)."""
        ...


# ---------------------------------------------------------------------------
# Report and simulation clock.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunReport:
    """What one :meth:`Worker.run_once` did (or, from :func:`drain`, a sum).

    ``leased == delivered + retried + dead + errors + expired`` always holds.

    ``retried``   rows the store put back to PENDING with a backoff; a row
                  whose device was pruned this pass is counted here too
                  (the store parks it PENDING until the device's cooldown),
                  as is a row released because its device was found pruned.
    ``pruned``    distinct subscriptions marked gone this pass, by a gone
                  result or by reaching ``PRUNE_AFTER_FAILURES``, or found
                  pruned after the lease.
    ``errors``    rows whose *store* call raised after the send; they stay
                  LEASED, the worker remembers the result and applies it on
                  its next pass (or the lease expires and the row comes back).
    ``expired``   rows whose lease had already run out when their turn to
                  be sent came (a batch too large for the lease); not sent,
                  they come back through the reclaim.
    ``revived``   dead letters the store put back to PENDING at the start
                  of the pass (``OutboxPort.revive_exhausted``): rows that
                  died only because an outage outlasted their retry budget
                  and have waited out ``EXHAUSTED_RETRY_COOLDOWN_S``.  Not
                  part of ``leased``'s sum: a revived row is leased in the
                  same pass when it is due, and counted there like any other.
    ``rounds``    1 for ``run_once``; the number of passes for ``drain``.
    ``next_due``  the earliest instant a row this pass left PENDING or
                  LEASED becomes leasable again, or None; a row that died
                  EXHAUSTED counts as leasable again at ``now +
                  EXHAUSTED_RETRY_COOLDOWN_S``, when the store may revive
                  it. Feeds ``drain``.
    """

    leased: int
    delivered: int
    retried: int
    dead: int
    pruned: int
    errors: int = 0
    rounds: int = 1
    next_due: Optional[float] = None
    expired: int = 0
    revived: int = 0

    @staticmethod
    def empty() -> "RunReport":
        return RunReport(0, 0, 0, 0, 0, errors=0, rounds=0)

    def __add__(self, other: "RunReport") -> "RunReport":
        if not isinstance(other, RunReport):
            return NotImplemented
        return RunReport(
            leased=self.leased + other.leased,
            delivered=self.delivered + other.delivered,
            retried=self.retried + other.retried,
            dead=self.dead + other.dead,
            pruned=self.pruned + other.pruned,
            errors=self.errors + other.errors,
            rounds=self.rounds + other.rounds,
            next_due=_min_due(self.next_due, other.next_due),
            expired=self.expired + other.expired,
            revived=self.revived + other.revived,
        )


def _min_due(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


class SimClock:
    """A clock the caller moves by hand. Callable so it fits ``Clock``.

    :func:`drain` needs ``advance``; the worker itself only calls the clock.
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = float(start)
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("a clock does not run backwards")
        with self._lock:
            self._now += float(seconds)


# ---------------------------------------------------------------------------
# The worker.
# ---------------------------------------------------------------------------

_DELIVERED, _RETRIED, _DEAD, _ERROR, _EXPIRED = "delivered", "retried", "dead", "error", "expired"
_DEVICE_PRUNED = "device pruned"

_NO_SUBSCRIPTION = SendResult(ok=False, retryable=False, reason="no subscription",
                              dead_reason=DeadReason.NO_SUBSCRIPTION)
_NO_TRANSPORT = SendResult(ok=False, retryable=False, reason="no transport",
                           dead_reason=DeadReason.NO_TRANSPORT)
_NO_ALERT = SendResult(ok=False, retryable=False, reason="no alert")


@dataclass(frozen=True)
class _Settled:
    """One row's fate, for the report."""

    outcome: str                                  # _DELIVERED / _RETRIED / _DEAD / _ERROR / _EXPIRED
    pruned: Optional[Tuple[str, str]] = None      # (profile_id, device_id) marked gone
    due: Optional[float] = None                   # when the row is leasable again


class Worker:
    """Leases due outbox rows and pushes them through the transports.

    ``outbox``      an :class:`OutboxPort`
    ``transports``  transports keyed by the name subscriptions use
                    (``Subscription.transport``: "webpush", "fcm", "fake", ...)
    ``clock``       returns unix seconds; injected so tests can drive it
    ``jitter``      returns a float in [0, 1) for :func:`backoff_seconds`;
                    an app wires ``SeedFields.parse(seed).stream("alerts.backoff").random``
    ``batch``       rows per pass
    ``lease_s``     visibility timeout for a leased row
    """

    def __init__(
        self,
        outbox: OutboxPort,
        transports: Dict[str, Transport],
        clock: Clock,
        jitter: Jitter,
        batch: int = 50,
        lease_s: float = LEASE_S,
    ) -> None:
        if batch < 1:
            raise ValueError("batch must be at least 1")
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        self.outbox = outbox
        self.transports: Dict[str, Transport] = dict(transports)
        self.clock = clock
        self.jitter = jitter
        self.batch = int(batch)
        self.lease_s = float(lease_s)
        # Sends whose mark raised: row -> result, applied before the next
        # lease (see the module docstring).  Guarded because ``run_forever``
        # may be driven from a thread while a test inspects it.
        self._unmarked: Dict[int, Tuple[OutboxRow, SendResult]] = {}
        self._unmarked_lock = threading.Lock()

    @property
    def unmarked(self) -> int:
        """How many sends this worker still has to report to the store."""
        with self._unmarked_lock:
            return len(self._unmarked)

    # -- one pass ---------------------------------------------------------

    def run_once(self) -> RunReport:
        """Revive cooled-down dead letters, lease one batch of due rows and
        settle every one of them.

        A failure in one row (a raising transport, a missing subscription,
        even a store error while marking) never stops the rest of the batch.
        Only a failure of ``revive_exhausted`` or ``lease`` itself
        propagates: with no rows in hand there is nothing to protect, and
        the caller should know the store is unreachable.
        """
        self._apply_unmarked()
        now = self.clock()
        revived = int(self.outbox.revive_exhausted(now))
        rows = self.outbox.lease(now, self.batch, self.lease_s)
        counts = {_DELIVERED: 0, _RETRIED: 0, _DEAD: 0, _ERROR: 0, _EXPIRED: 0}
        pruned: Set[Tuple[str, str]] = set()
        next_due: Optional[float] = None
        for row in rows:
            settled = self._settle(row)
            counts[settled.outcome] += 1
            if settled.pruned is not None:
                pruned.add(settled.pruned)
            next_due = _min_due(next_due, settled.due)
        return RunReport(
            leased=len(rows),
            delivered=counts[_DELIVERED],
            retried=counts[_RETRIED],
            dead=counts[_DEAD],
            pruned=len(pruned),
            errors=counts[_ERROR],
            next_due=next_due,
            expired=counts[_EXPIRED],
            revived=revived,
        )

    def _apply_unmarked(self) -> None:
        """Report the sends whose mark failed last time.  A mark that fails
        again stays remembered; nothing is sent here."""
        with self._unmarked_lock:
            pending = list(self._unmarked.items())
            self._unmarked.clear()
        for row_id, (row, result) in pending:
            try:
                self._mark(row, result, None)
            except Exception as exc:  # noqa: BLE001 - the store is still unreachable
                log.warning("row %d for (%s, %s): store call raised %s again; result kept",
                            row.row_id, row.profile_id, row.device_id, type(exc).__name__)

    def _settle(self, row: OutboxRow) -> _Settled:
        """Guard around :meth:`_settle_unguarded`: a store error on this row
        is logged by type and counted, and the row stays LEASED until its
        lease expires. The blob is not in scope here, so it cannot leak."""
        try:
            return self._settle_unguarded(row)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            log.warning(
                "row %d for (%s, %s): store call raised %s; row stays leased",
                row.row_id, row.profile_id, row.device_id, type(exc).__name__,
            )
            return _Settled(_ERROR, due=row.lease_until)

    def _settle_unguarded(self, row: OutboxRow) -> _Settled:
        """Look up, send, mark. Every path ends in one ``mark`` or ``release``."""
        subscription = self.outbox.subscription(row.profile_id, row.device_id)

        with self._unmarked_lock:
            remembered = self._unmarked.pop(row.row_id, None)
        if remembered is not None:
            # This row was sent already and only the mark failed; report
            # that send instead of making another.
            return self._mark(row, remembered[1], subscription)

        if subscription is None or subscription.gone:
            if subscription is not None and subscription.pruned:
                return self._release(row, _DEVICE_PRUNED)
            return self._mark(row, _NO_SUBSCRIPTION, None)

        transport = self.transports.get(subscription.transport)
        if transport is None:
            return self._mark(row, _NO_TRANSPORT, subscription)

        alert = self.outbox.alert(row.alert_id)
        if alert is None:
            return self._mark(row, _NO_ALERT, subscription)

        if self.clock() >= row.lease_until:
            # The batch outlived the lease: another worker may hold this
            # row by now, so it is not sent.  The reclaim brings it back.
            log.warning("row %d for (%s, %s): lease expired before send; not sent",
                        row.row_id, row.profile_id, row.device_id)
            return _Settled(_EXPIRED, due=row.lease_until)

        return self._mark(row, self._send(transport, subscription, alert), subscription)

    def _release(self, row: OutboxRow, reason: str) -> _Settled:
        """Hand a row back untouched; no attempt is recorded."""
        self.outbox.release(row.row_id, reason)
        return _Settled(_RETRIED, pruned=(row.profile_id, row.device_id))

    def _send(self, transport: Transport, subscription: Subscription, alert: Alert) -> SendResult:
        """Call the transport; turn an exception into a retryable failure.

        Only the exception's *type name* goes into the reason. Its message
        might quote the endpoint or the blob, and the reason is stored.
        """
        try:
            result = transport.send(subscription, alert)
        except Exception as exc:  # noqa: BLE001 - a raising transport must not kill the batch
            log.warning(
                "transport %r raised %s for (%s, %s)",
                getattr(transport, "name", "?"), type(exc).__name__,
                subscription.profile_id, subscription.device_id,
            )
            return SendResult(ok=False, retryable=True, reason=f"transport raised {type(exc).__name__}")
        if not isinstance(result, SendResult):
            return SendResult(
                ok=False, retryable=True,
                reason=f"transport returned {type(result).__name__}",
            )
        return result

    def _mark(self, row: OutboxRow, result: SendResult,
              subscription: Optional[Subscription]) -> _Settled:
        """Hand the result to the store and read its decision back.

        The jitter is drawn only when the store will actually back off
        (a retryable failure short of ``MAX_ATTEMPTS`` within the row's
        current budget, which starts at ``attempts_base``), so a seeded
        stream advances exactly once per backoff. ``next_due`` for the
        report is computed the way the store computes it from the same
        jitter.
        """
        pid, did = row.profile_id, row.device_id
        budget_no = row.attempts + 1 - row.attempts_base
        now = self.clock()
        will_back_off = (
            not result.ok and not result.gone and result.retryable
            and budget_no < MAX_ATTEMPTS
        )
        jitter = self._jitter() if will_back_off else None

        try:
            state = self.outbox.mark(row.row_id, result, now, jitter)
        except Exception:
            with self._unmarked_lock:
                self._unmarked[row.row_id] = (row, result)
            raise

        if state is RowState.DELIVERED:
            return _Settled(_DELIVERED)
        if state is RowState.DEAD:
            if dead_reason_for(result, budget_spent=True) is DeadReason.EXHAUSTED:
                # An outage, not a verdict: the store revives the row once
                # the cooldown has passed (if its alert is still young).
                return _Settled(_DEAD, due=now + EXHAUSTED_RETRY_COOLDOWN_S)
            return _Settled(_DEAD, pruned=(pid, did) if result.gone else None)
        if state is RowState.PENDING:
            # A retryable failure that the store backed off. If it was the
            # device's PRUNE_AFTER_FAILURES-th in a row the store has also
            # marked the device gone and parked this row; the snapshot tells
            # us when that is possible, the store tells us whether it happened.
            if subscription is not None and subscription.failures + 1 >= PRUNE_AFTER_FAILURES:
                fresh = self.outbox.subscription(pid, did)
                if fresh is None or fresh.gone:
                    return _Settled(_RETRIED, pruned=(pid, did), due=now + PRUNE_COOLDOWN_S)
            due = now + backoff_seconds(max(1, budget_no), jitter if jitter is not None else 0.0)
            return _Settled(_RETRIED, due=due)
        # LEASED (or anything else) from mark means the store did not settle
        # the row; it stays with its lease and comes back when that expires.
        log.warning("row %d for (%s, %s): mark left the row %s", row.row_id, pid, did, state)
        return _Settled(_ERROR, due=row.lease_until)

    def _jitter(self) -> float:
        """The injected jitter, clamped into [0, 1] so a sloppy callable
        cannot make the store reject the mark and strand the row."""
        return min(1.0, max(0.0, float(self.jitter())))

    # -- the loop ---------------------------------------------------------

    def run_forever(
        self,
        stop: threading.Event,
        idle_s: float = 1.0,
        on_error: Optional[Callable[[BaseException], None]] = None,
    ) -> None:
        """Run passes until ``stop`` is set.

        After an empty batch the loop waits ``idle_s`` on ``stop.wait`` (not
        ``time.sleep``), so ``stop.set()`` returns promptly; after a
        non-empty batch it goes straight round again. An exception from
        ``run_once`` (the store itself failing) is re-raised unless
        ``on_error`` is given, in which case it is reported there and the
        loop idles before trying again.
        """
        while not stop.is_set():
            try:
                report = self.run_once()
            except Exception as exc:  # noqa: BLE001 - see docstring
                if on_error is None:
                    raise
                on_error(exc)
                stop.wait(idle_s)
                continue
            if report.leased == 0:
                stop.wait(idle_s)


# ---------------------------------------------------------------------------
# Simulation helper.
# ---------------------------------------------------------------------------


def drain(
    outbox: OutboxPort,
    transports: Dict[str, Transport],
    clock: SimClock,
    jitter: Jitter,
    max_rounds: int = 1000,
    *,
    batch: int = 50,
    lease_s: float = LEASE_S,
) -> RunReport:
    """SIMULATION HELPER for tests and the gate; not for production.

    Runs :meth:`Worker.run_once` until no row is outstanding (nothing LEASED,
    nothing PENDING with a live subscription, nothing DEAD that the store
    will revive) or ``max_rounds`` passes have run. Between rounds the clock
    is advanced to the earliest ``next_due`` the worker scheduled, so every
    backoff in the retry policy -- and the ``EXHAUSTED_RETRY_COOLDOWN_S``
    before a dead letter is revived -- is honoured exactly without real
    waiting. ``clock`` must therefore be a :class:`SimClock` or anything
    callable with ``advance(seconds)``.

    Rows whose timing this drain never saw (retries scheduled before it
    started, or rows leased by a worker that died) are not exposed by the
    port, so when nothing is scheduled but rows are outstanding the clock
    steps past the longest possible wait, ``max(BACKOFF_CAP_S, lease_s)``;
    at most two such steps make any row due.  When only revivable dead
    letters are outstanding the step is the cooldown itself, after which
    every one of them is due.

    Returns the sum of the per-round reports; ``rounds`` says how many ran.
    Convergence is ``outbox.stats()`` showing nothing outstanding afterwards;
    ``max_rounds`` being hit means it did not.
    """
    if max_rounds < 1:
        raise ValueError("max_rounds must be at least 1")
    advance = getattr(clock, "advance", None)
    if advance is None:
        raise TypeError("drain needs a clock with advance(seconds), such as SimClock")

    worker = Worker(outbox, transports, clock, jitter, batch=batch, lease_s=lease_s)
    total = RunReport.empty()
    scheduled: List[float] = []
    for _ in range(max_rounds):
        report = worker.run_once()
        total += report
        now = clock()
        if report.next_due is not None:
            scheduled.append(report.next_due)
        scheduled = [t for t in scheduled if t > now]
        if report.leased >= batch:
            continue  # a full batch: more may be due at this very instant
        in_flight, revivable = _outstanding(outbox)
        if not in_flight and not revivable:
            break
        if scheduled:
            advance(min(scheduled) - now)
        elif in_flight:
            advance(max(BACKOFF_CAP_S, lease_s))
        else:
            advance(EXHAUSTED_RETRY_COOLDOWN_S)
    # A per-round next_due means nothing for the sum; the clock already moved.
    return replace(total, next_due=None)


def _outstanding(outbox: OutboxPort) -> Tuple[bool, bool]:
    """``(in_flight, revivable)``: is any row LEASED or PENDING and
    reachable, and is any DEAD row one the store will revive?"""
    stats = outbox.stats(dead_limit=0)
    reachable_pending = int(stats["pending"]) - int(stats.get("pending_unreachable", 0))
    in_flight = int(stats["leased"]) > 0 or reachable_pending > 0
    return in_flight, int(stats.get("dead_revivable", 0)) > 0


# ---------------------------------------------------------------------------
# Reference in-memory store with the outbox's semantics.
# ---------------------------------------------------------------------------


class MemoryStore:
    """In-memory :class:`OutboxPort` with the semantics of ``outbox.Outbox``.

    Not durable, so not the outbox contracts.py point 1 asks for; it is the
    executable statement of what the worker relies on, and the tests run
    the same cases against it and against the sqlite outbox. Mirrored from
    the outbox: ``lease`` revives devices whose prune cooldown has passed,
    reclaims rows whose ``lease_until < now``, hands out only rows whose
    device has a live subscription (and, given ``transports``, a wired
    one), orders by priority then ``next_due`` then ``row_id``; ``mark``
    applies the retry policy, records ``dead_reason`` and ``dead_at`` on
    every DEAD transition, is a no-op on a settled row except that a gone
    result still marks the device gone; ``release`` hands a leased row
    back; ``requeue`` gives a dead row a fresh budget; ``revive_exhausted``
    does the same by itself for EXHAUSTED rows past their cooldown whose
    alert is young enough and whose device is reachable; ``register``
    resets ``failures``, ``gone`` and ``pruned``; ``publish`` and
    ``backfill`` fan out to live and pruned devices. Left out: the publish
    dedupe window and its repair, which the worker never touches.
    Thread-safe, so ``run_forever`` can be driven from a thread. Rows
    handed out are copies.
    """

    def __init__(self, clock: Clock, jitter: Jitter = no_jitter) -> None:
        self._clock = clock
        self._jitter = jitter
        self.alerts: Dict[str, Alert] = {}
        self.subscriptions: Dict[Tuple[str, str], Subscription] = {}
        self.rows: Dict[int, OutboxRow] = {}
        self.attempts: List[DeliveryAttempt] = []
        self._gone_at: Dict[Tuple[str, str], float] = {}
        self._revivals: Dict[int, int] = {}
        self._next_row_id = 1
        self._lock = threading.RLock()

    # -- registry (the app's side) --------------------------------------

    def register(self, sub: Subscription, backfill_since: Optional[float] = None, *,
                 supersede_same_blob: bool = False) -> int:
        with self._lock:
            if supersede_same_blob:
                for key in [k for k, s in self.subscriptions.items()
                            if k[0] == sub.profile_id and k[1] != sub.device_id and s.blob == sub.blob]:
                    del self.subscriptions[key]
            self.subscriptions[(sub.profile_id, sub.device_id)] = replace(
                sub, failures=0, gone=False, pruned=False)
            self._gone_at.pop((sub.profile_id, sub.device_id), None)
            if backfill_since is None:
                return 0
            return self._backfill(sub.profile_id, float(backfill_since))

    def unregister(self, profile_id: str, device_id: str) -> bool:
        with self._lock:
            return self.subscriptions.pop((profile_id, device_id), None) is not None

    def subscription(self, profile_id: str, device_id: str) -> Optional[Subscription]:
        with self._lock:
            return self.subscriptions.get((profile_id, device_id))

    def pruned_subscriptions(self) -> List[Tuple[str, str]]:
        with self._lock:
            return sorted(k for k, s in self.subscriptions.items() if s.gone and s.pruned)

    def _live(self, profile_id: str, device_id: str) -> bool:
        sub = self.subscriptions.get((profile_id, device_id))
        return sub is not None and not sub.gone

    def _reachable(self, profile_id: str, device_id: str) -> bool:
        sub = self.subscriptions.get((profile_id, device_id))
        return sub is not None and (not sub.gone or sub.pruned)

    # -- publishing (the app's side) ------------------------------------

    def publish(self, alert: Alert) -> int:
        """Store the alert and create one PENDING row per reachable
        subscription of its profile, due now. Returns the rows created."""
        with self._lock:
            if alert.id in self.alerts:
                raise ValueError(f"alert {alert.id!r} is already stored")
            self.alerts[alert.id] = alert
            now = self._clock()
            created = 0
            for (pid, did), sub in sorted(self.subscriptions.items()):
                if pid == alert.profile_id and self._reachable(pid, did):
                    self._new_row(alert.id, pid, did, now)
                    created += 1
            return created

    def _new_row(self, alert_id: str, pid: str, did: str, now: float) -> None:
        self.rows[self._next_row_id] = OutboxRow(
            row_id=self._next_row_id, alert_id=alert_id, profile_id=pid, device_id=did,
            state=RowState.PENDING, attempts=0, next_due=now, lease_until=0.0,
        )
        self._next_row_id += 1

    def backfill(self, profile_id: str, since: float) -> int:
        with self._lock:
            return self._backfill(profile_id, float(since))

    def _backfill(self, profile_id: str, since: float) -> int:
        now = self._clock()
        have = {(r.alert_id, r.device_id) for r in self.rows.values()}
        created = 0
        alerts = sorted((a for a in self.alerts.values() if a.profile_id == profile_id and a.created_at > since),
                        key=lambda a: (a.created_at, a.id))
        for a in alerts:
            for (pid, did), _sub in sorted(self.subscriptions.items()):
                if pid == profile_id and self._reachable(pid, did) and (a.id, did) not in have:
                    self._new_row(a.id, pid, did, now)
                    have.add((a.id, did))
                    created += 1
        return created

    # -- OutboxPort ------------------------------------------------------

    def lease(self, now: float, limit: int, lease_s: float = LEASE_S,
              transports: Optional[Iterable[str]] = None) -> List[OutboxRow]:
        if lease_s <= 0:
            raise ValueError("lease_s must be positive")
        if limit <= 0:
            return []
        names = None if transports is None else {str(n) for n in transports}
        if names is not None and not names:
            return []
        with self._lock:
            for key, sub in list(self.subscriptions.items()):
                if sub.gone and sub.pruned and self._gone_at.get(key, 0.0) + PRUNE_COOLDOWN_S <= now:
                    self.subscriptions[key] = replace(sub, gone=False, pruned=False, failures=0)
                    self._gone_at.pop(key, None)
            for r in self.rows.values():
                if r.state is RowState.LEASED and r.lease_until < now:
                    r.state, r.lease_until, r.last_reason = RowState.PENDING, 0.0, "lease expired"
            due = [
                r for r in self.rows.values()
                if r.state is RowState.PENDING and r.next_due <= now
                and self._live(r.profile_id, r.device_id)
                and (names is None or self.subscriptions[(r.profile_id, r.device_id)].transport in names)
            ]
            due.sort(key=lambda r: (-int(self.alerts[r.alert_id].priority), r.next_due, r.row_id))
            out: List[OutboxRow] = []
            for r in due[:limit]:
                r.state = RowState.LEASED
                r.lease_until = now + lease_s
                out.append(replace(r))
            return out

    def release(self, row_id: int, reason: str = "") -> RowState:
        with self._lock:
            row = self.rows.get(row_id)
            if row is None:
                raise LookupError(f"outbox row {row_id!r} does not exist")
            if row.state is not RowState.LEASED:
                return row.state
            row.state, row.lease_until, row.last_reason = RowState.PENDING, 0.0, str(reason or "")
            return RowState.PENDING

    def mark(self, row_id: int, result: SendResult, now: float,
             jitter: Optional[float] = None) -> RowState:
        with self._lock:
            row = self.rows.get(row_id)
            if row is None:
                raise LookupError(f"outbox row {row_id!r} does not exist")
            key = (row.profile_id, row.device_id)
            if row.state in (RowState.DELIVERED, RowState.DEAD):
                if result.gone and not result.ok:
                    self._set_gone(key, now, pruned=False)
                return row.state
            attempt_no = row.attempts + 1
            budget_no = attempt_no - row.attempts_base
            self.attempts.append(DeliveryAttempt(
                row_id=row_id, attempt=attempt_no, at=float(now), ok=bool(result.ok),
                retryable=bool(result.retryable), gone=bool(result.gone), reason=str(result.reason or ""),
            ))
            next_due = row.next_due
            dead_reason = dead_reason_for(result, budget_spent=budget_no >= MAX_ATTEMPTS)
            if result.ok:
                new_state = RowState.DELIVERED
                self._update_sub(key, failures=0)
            elif result.gone:
                new_state = RowState.DEAD
                self._set_gone(key, now, pruned=False)
            elif result.retryable:
                if dead_reason is not None:
                    new_state = RowState.DEAD
                else:
                    new_state = RowState.PENDING
                    drawn = self._jitter() if jitter is None else jitter
                    if not 0.0 <= drawn <= 1.0:
                        raise ValueError(f"jitter must be in [0, 1], got {drawn!r}")
                    next_due = float(now) + backoff_seconds(budget_no, float(drawn))
                sub = self.subscriptions.get(key)
                if sub is not None:
                    failures = sub.failures + 1
                    self._update_sub(key, failures=failures)
                    if failures >= PRUNE_AFTER_FAILURES:
                        self._set_gone(key, now, pruned=True)
            else:
                new_state = RowState.DEAD
            row.state, row.attempts, row.next_due = new_state, attempt_no, next_due
            row.lease_until, row.last_reason = 0.0, str(result.reason or "")
            dead = new_state is RowState.DEAD
            row.dead_reason = dead_reason if dead else None
            row.dead_at = float(now) if dead else 0.0
            return new_state

    def requeue(self, row_id: int) -> bool:
        with self._lock:
            row = self.rows.get(row_id)
            if row is None:
                raise LookupError(f"outbox row {row_id!r} does not exist")
            return self._requeue(row, self._clock()) > 0

    def requeue_dead(self, profile_id: Optional[str] = None, device_id: Optional[str] = None) -> int:
        if profile_id is None and device_id is not None:
            raise ValueError("device_id needs a profile_id")
        with self._lock:
            n = 0
            now = self._clock()
            for row in sorted(self.rows.values(), key=lambda r: r.row_id):
                if profile_id is not None and row.profile_id != profile_id:
                    continue
                if device_id is not None and row.device_id != device_id:
                    continue
                n += self._requeue(row, now)
            return n

    def revive_exhausted(self, now: float) -> int:
        with self._lock:
            n = 0
            for row in sorted(self.rows.values(), key=lambda r: r.row_id):
                if self._revivable(row, now) and row.dead_at + EXHAUSTED_RETRY_COOLDOWN_S <= now:
                    n += self._requeue(row, now, "requeued: exhausted, cooldown passed", revival=True)
            return n

    def _revivable(self, row: OutboxRow, now: float) -> bool:
        """DEAD as EXHAUSTED, alert young enough at ``now``, device
        reachable; the cooldown is the caller's question."""
        if row.state is not RowState.DEAD or row.dead_reason is not DeadReason.EXHAUSTED:
            return False
        alert = self.alerts.get(row.alert_id)
        if alert is None or not alert.created_at + ALERT_MAX_AGE_S > now:
            return False
        return self._reachable(row.profile_id, row.device_id)

    def _requeue(self, row: OutboxRow, now: float, reason: str = "requeued", *, revival: bool = False) -> int:
        if row.state is not RowState.DEAD:
            return 0
        row.attempts_base = row.attempts
        row.state, row.next_due, row.lease_until = RowState.PENDING, float(now), 0.0
        row.last_reason, row.dead_reason, row.dead_at = reason, None, 0.0
        if revival:
            self._revivals[row.row_id] = self._revivals.get(row.row_id, 0) + 1
        return 1

    def _update_sub(self, key: Tuple[str, str], **changes: Any) -> None:
        sub = self.subscriptions.get(key)
        if sub is not None:
            self.subscriptions[key] = replace(sub, **changes)

    def _set_gone(self, key: Tuple[str, str], now: float, *, pruned: bool) -> None:
        sub = self.subscriptions.get(key)
        if sub is None or (pruned and sub.gone):
            return
        self.subscriptions[key] = replace(sub, gone=True, pruned=pruned)
        self._gone_at[key] = float(now)

    def alert(self, alert_id: str) -> Optional[Alert]:
        with self._lock:
            return self.alerts.get(alert_id)

    def backlog_by_transport(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for r in self.rows.values():
                if r.state is RowState.PENDING and self._live(r.profile_id, r.device_id):
                    name = self.subscriptions[(r.profile_id, r.device_id)].transport
                    counts[name] = counts.get(name, 0) + 1
            return dict(sorted(counts.items()))

    def stats(self, dead_limit: int = 50) -> Dict[str, Any]:
        with self._lock:
            now = self._clock()
            counts = {state.value: 0 for state in RowState}
            unreachable = exhausted = revivable = 0
            for r in self.rows.values():
                counts[r.state.value] += 1
                if r.state is RowState.PENDING and not self._live(r.profile_id, r.device_id):
                    unreachable += 1
                if r.state is RowState.DEAD and r.dead_reason is DeadReason.EXHAUSTED:
                    exhausted += 1
                    revivable += int(self._revivable(r, now))
            dead = [r for r in self.rows.values() if r.state is RowState.DEAD]
            dead.sort(key=lambda r: -r.row_id)
            return {
                **counts,
                "pending_unreachable": unreachable,
                "dead_exhausted": exhausted,
                "dead_revivable": revivable,
                "dead_permanent": counts[RowState.DEAD.value] - exhausted,
                "revived": sum(self._revivals.values()),
                "alerts": len(self.alerts),
                "subscriptions_live": sum(1 for s in self.subscriptions.values() if not s.gone),
                "subscriptions_gone": sum(1 for s in self.subscriptions.values() if s.gone),
                "dead_letters": [
                    dataclasses.asdict(r) | {
                        "state": r.state.value,
                        "dead_reason": None if r.dead_reason is None else r.dead_reason.value,
                    }
                    for r in dead[:max(0, dead_limit)]
                ],
            }

    # -- inspection, for tests -------------------------------------------

    def row(self, row_id: int) -> Optional[OutboxRow]:
        with self._lock:
            r = self.rows.get(row_id)
            return None if r is None else replace(r)

    def rows_for(self, alert_id: str) -> List[OutboxRow]:
        with self._lock:
            return [replace(r) for r in sorted(self.rows.values(), key=lambda r: r.row_id)
                    if r.alert_id == alert_id]

    def attempts_for(self, row_id: int) -> List[DeliveryAttempt]:
        with self._lock:
            if row_id not in self.rows:
                raise LookupError(f"outbox row {row_id!r} does not exist")
            return [a for a in self.attempts if a.row_id == row_id]


__all__ = [
    "Clock", "Jitter", "OutboxPort", "RunReport", "SimClock", "Worker", "drain",
    "MemoryStore", "no_jitter",
]


# ---------------------------------------------------------------------------
# ``python3 -m jarvis_alerts.worker``: the same scenario on both stores.
# ---------------------------------------------------------------------------


def _demo() -> int:
    """Drive a flaky transport through drain on MemoryStore and, when the
    sqlite outbox is importable, on it too; the two reports must agree.
    Prints counts, row ids and reasons only, never a blob."""
    from lucifer_gen.seed import SeedFields

    class FlakyTransport:
        name = "fake"

        def __init__(self) -> None:
            # phone-explodes: an outage that outlasts the retry budget, so
            # its row dies EXHAUSTED and is revived after the cooldown.
            self.left = {"phone-flaky": 2, "phone-explodes": MAX_ATTEMPTS}

        def send(self, subscription: Subscription, alert: Alert) -> SendResult:
            device = subscription.device_id
            if device == "phone-gone":
                return SendResult(ok=False, gone=True, reason="410")
            if self.left.get(device, 0) > 0:
                self.left[device] -= 1
                if device == "phone-explodes":
                    raise ConnectionError("network unreachable")
                return SendResult(ok=False, retryable=True, reason="503")
            return SendResult(ok=True)

    def scenario(make_store) -> Tuple[RunReport, Dict[str, Any], List[OutboxRow]]:
        clock = SimClock(1_700_000_000.0)
        stream = SeedFields.parse(0xC0FFEE).stream("alerts.backoff")
        store = make_store(clock)
        for device in ("laptop", "phone-flaky", "phone-gone", "phone-explodes", "phone-lost"):
            store.register(Subscription("owner", device, "fake", blob="{...}", created_at=clock()))
        alert = Alert("a1", "owner", "render_done", "Render finished", "crypt.png is ready", clock())
        store.publish(alert)
        store.unregister("owner", "phone-lost")  # its row is parked, never leased
        report = drain(store, {"fake": FlakyTransport()}, clock, stream.random, max_rounds=50)
        rows = [store.row(i) for i in range(1, 6)]
        return report, store.stats(dead_limit=0), rows

    stores = {"MemoryStore": lambda clock: MemoryStore(clock)}
    try:
        from .outbox import Outbox
        stores["Outbox(sqlite)"] = lambda clock: Outbox(":memory:", clock=clock)
    except ImportError:
        pass

    results = {}
    for name, make in stores.items():
        report, stats, rows = scenario(make)
        results[name] = report
        print(f"{name}: {report}")
        print(f"  outstanding: leased={stats['leased']} pending={stats['pending']} "
              f"unreachable={stats['pending_unreachable']}")
        for row in rows:
            print(f"  row {row.row_id} ({row.profile_id}, {row.device_id}): "
                  f"{row.state.value} after {row.attempts} attempt(s) {row.last_reason!r}")
    agree = len(set(results.values())) == 1
    print("stores agree:", agree)
    return 0 if agree else 1


if __name__ == "__main__":
    raise SystemExit(_demo())
