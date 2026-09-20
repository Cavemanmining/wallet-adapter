"""The delivery gate: run a miniature deployment and hold it to contracts.py.

Design: :mod:`jarvis_alerts.contracts`, module docstring.  It names three
things delivery needs, and this module checks each of them end to end by
driving the real pieces -- the sqlite :class:`~jarvis_alerts.outbox.Outbox`,
the :class:`~jarvis_alerts.worker.Worker` and a scripted
:class:`~jarvis_alerts.transports.FakeTransport` -- through a scenario that
is reproducible from one seed:

1. A durable outbox (design point 1).  "Rows are leased, not popped, so a
   worker that dies mid-send hands the row back after a visibility
   timeout."  The gate kills a worker every ``crash_every`` rounds by
   leasing a batch and abandoning it -- no send, no mark -- and at the end
   demands that no row is left LEASED and no row is left PENDING behind a
   live device.  It also *watches the lease itself* through an observing
   store between the worker and the outbox (:class:`_Observed`): a row
   handed out while an earlier lease on it is still live, a row of a
   gone device, a row before its ``next_due``, or an expired lease the
   outbox had room to re-offer but did not, is each a finding in the
   round it happens.  An outbox with no leasing at all, or one that
   reclaims without a timeout, is told apart from a correct one this way.
   Besides the crashes, one *slow* worker holds the gone devices' rows
   for ``STALL_LEASE_S`` while those devices go gone, so that every run
   contains a row that comes back from a lease onto a gone device and
   must be parked, never handed out; a second, brief hold of the next
   wave's row makes it come back together with the wave after, so the
   device goes gone on the first of the two in one batch and the worker
   is seen to leave the second alone.
2. A push transport (design point 2).  The transport is injected and the
   registry decides who gets what.  The gate's FakeTransport script fails
   about 15 percent of sends transiently, rejects one device per ten
   profiles permanently, reports one device per eight profiles gone
   partway through, and fails one device per twelve profiles transiently
   on its first alert until the retry budget is spent -- an outage that
   outlasts the budget and then *clears*: the row dies EXHAUSTED and the
   outbox must bring it back by itself ``EXHAUSTED_RETRY_COOLDOWN_S``
   later, when the next send succeeds.  One more device per twelve
   profiles fails an alert published a day before the waves for good:
   that row dies EXHAUSTED too, but its alert is past ``ALERT_MAX_AGE_S``
   by the time the cooldown has passed, so it must be seen to *stay*
   dead.  Some devices register late and are backfilled, one of them
   after a device of the same profile has gone gone.
3. Idempotency (design point 3).  "Retries are at-least-once, so every
   alert carries a dedupe key and every delivery attempt is recorded."
   The gate demands that every (alert, device) pair that should have gone
   out is DELIVERED exactly once, that deduped publishes stored nothing
   (and repaired exactly what the outbox's contract says they repair),
   that every DELIVERED row has an ok send behind it, that the count of
   DELIVERED rows equals the count of ok sends the transport saw, and
   that every ``mark`` the worker made is one recorded attempt.  The
   retry policy at the foot of contracts.py is checked at every ``mark``:
   the state the store returns and the ``next_due`` it writes are
   compared with what the policy says for that attempt and jitter.

Entry point
-----------
:func:`run_gate` returns a :class:`GateReport`: ``counts`` for the
operator, ``problems`` (each a :class:`Problem` carrying the ids at fault
and a ``repro`` call that reproduces the run) and ``ok``.
``python3 -m jarvis_alerts.validate`` prints one.

The gate is shown to fail before it passes: ``inject_defect`` perturbs the
run in one of eight ways (:data:`DEFECTS`) that a correct outbox never
allows, the tests check that each is reported, and ``python3 -m
jarvis_alerts.validate --show-defects`` exits 1 if any is not:

    skip_reclaim    the crashed worker's lease never expires, so its rows
                    are never reclaimed (an outbox that forgets step 1 of
                    ``lease``): rows stay LEASED and alerts go missing
    double_lease    ``lease`` hands every row out twice in one batch (a
                    lease that is not atomic): rows are sent twice, the
                    second mark is a no-op, ok sends exceed DELIVERED rows
    lose_on_crash   the crashing worker marks its rows DELIVERED and dies
                    before sending (mark-then-send, the at-most-once order
                    worker.py warns about): DELIVERED rows with no send
    no_lease        rows are handed out but not held: the store puts each
                    one straight back to PENDING, so a crashed worker's
                    rows are re-offered at once and a second worker would
                    send them too
    no_backoff      a retryable failure is due again immediately instead
                    of after ``backoff_seconds``
    lease_gone      rows of a device the transport reported gone are
                    leased and sent anyway
    attempts_lost   only the last attempt of a row is kept and the row's
                    counter never passes 1
    no_revive       ``revive_exhausted`` does nothing: a row whose outage
                    outlasted the retry budget stays DEAD once the outage
                    is over (the defect this package's cooldown exists to
                    prevent): reported by the observer the round its
                    revival was due, by the ledger at the end, and as a
                    run that never converges

What the expected set is
------------------------
The gate keeps its own ledger of which (profile, device, alert) triples
must reach a device, built as it acts, never derived from the store:

* publishing a stored alert adds every device of the profile that is
  registered and has not gone gone;
* registering a late device and calling ``Outbox.backfill(profile, now -
  BACKFILL_WINDOW_S)`` adds the profile's stored alerts newer than that;
* a deduped publish stores nothing; it *repairs* the alert it repeats
  (``Outbox.publish``): the ledger predicts from the rows in the store
  which live devices lack a row for it and which of its rows are DEAD,
  adds the former and expects the latter re-queued, and the outbox must
  report exactly that count.

Each expected triple must end DELIVERED, except on the scripted devices:
a *permanent* device's rows must all be DEAD with a reason; a *gone*
device's rows are DELIVERED before the gone moment, DEAD with a reason at
or after it, or parked PENDING (the outbox never leases rows for a gone
device); an *exhausting* device's first-wave row must be DELIVERED after
exactly ``MAX_ATTEMPTS + 1`` attempts, revived exactly once by the outbox
and no earlier than ``EXHAUSTED_RETRY_COOLDOWN_S`` after its death; a
*stale* device's prelude row must be DEAD as EXHAUSTED after exactly
``MAX_ATTEMPTS`` attempts and never revived.  Every DEAD row must carry a
``dead_reason`` that matches its last recorded attempt and a ``dead_at``
equal to that attempt's time.  Any row outside the ledger is reported.
Revival itself is watched at every pass (:class:`_Observed`): a row that
was due for revival and is still DEAD afterwards, a row that was revived
although it was not due (wrong reason, alert too old, cooldown not yet
passed, device gone), a revived row without a fresh budget, and a count
that does not match are each a finding in that round.

Dedupe is probed on both sides of its window at every size: after the
waves, every profile repeats its most recent key (must collapse, with
the repair predicted above); after a further ``DEDUPE_WINDOW_S`` of sim
time, every profile repeats its oldest key (must go out).

Randomness and time
-------------------
Every draw comes from ``SeedFields.parse(seed).stream(label)`` of
:mod:`lucifer_gen.seed`, one labelled stream per concern (devices,
publishing, transport failures, backoff jitter), so a run is a pure
function of its arguments.  Time is a :class:`SimClock` advanced by
``TICK_S`` per round, plus two jumps: ``ALERT_MAX_AGE_S`` after the
prelude (so the stale device's alert is a day old when the waves start)
and ``DEDUPE_WINDOW_S`` before the stale dedupe probe; nothing here reads
the wall clock.  Waiting out ``EXHAUSTED_RETRY_COOLDOWN_S`` is done in
ticks, not a jump, so the observer sees every pass in which the store
must *not* revive the row yet.

Privacy (contracts.py, ``Subscription.blob`` "never logged"): the fake
blobs are written once at registration and never read back by the gate;
problems, counts and the summary name subscriptions by (profile_id,
device_id) only.

Judgement calls the assignment left open, recorded here:

* "No row is left LEASED or PENDING" is read as: none LEASED, and none
  PENDING behind a *live* device.  The outbox parks the PENDING rows of a
  device that went gone (``Outbox.unregister`` docstring, ``lease`` step
  2) and they are the documented outcome, so they are counted as
  ``rows_pending_parked`` and checked to belong to a gone device.
* "Dead letters exist only for the permanent-failure device" is read with
  the outbox's own semantics: a gone result makes that row DEAD too, and a
  row of the same device leased in the same batch dead-letters as "no
  subscription" (worker.py).  DEAD rows are therefore allowed on
  permanent devices and on gone devices at or after the gone moment, and
  every one must carry a reason.  Anything else DEAD is reported.
* Late devices are backfilled, because ``Outbox.backfill`` exists for
  exactly that case, with a window of ``BACKFILL_WINDOW_S`` so the
  ``since`` filter is exercised: an alert older than the window must
  *not* reach the late device, and one inside it must.
* Random transient failures on one row are capped at ``MAX_ATTEMPTS - 1``,
  so no row dead-letters by exhausting the retry policy by chance; that
  would be a legitimate DEAD row -- one the outbox would revive a
  cooldown later -- the gate could not tell from a bug.  The cap is
  applied after the draw, so the 15 percent rate holds.  The policy's
  end is exercised on purpose instead, by the exhausting devices, whose
  row the ledger expects revived and DELIVERED, and by the stale devices,
  whose row it expects kept DEAD.  Random failures on one device are also
  capped so that, with the exhausting row's forced ones, no device can
  reach ``PRUNE_AFTER_FAILURES``: a prune parks rows for a cooldown,
  which the gate does not model.
* The exhausting profile's first-wave alert carries no dedupe key.  A
  keyed repeat inside the window would *repair* the DEAD row (re-queue
  it at once), and the row would be DELIVERED without the outbox's own
  revival ever being needed -- and ``no_revive`` would go unseen on such
  a seed.  Without a key, only ``revive_exhausted`` can bring it back.
* The stale device's alert is aged by a clock jump, not by a backdated
  ``created_at``: it is published, its row is driven to exhaustion in
  ticks, and then the clock moves ``ALERT_MAX_AGE_S`` before the waves,
  so the outage really did outlast the maximum age.  The prelude runs to
  "nothing in flight" without waiting for revivals, or the row would be
  revived before it could age.
* "Nothing outstanding" (convergence) includes DEAD rows the outbox will
  revive: a run ends only once every revivable row has come back and
  been settled, so a store that never revives is a run that never
  converges.  The stall detector allows for the cooldown accordingly.
* A simulated crash happens *before* the send, so ok sends and DELIVERED
  rows stay equal in a healthy run; ``lose_on_crash`` is the variant
  that acknowledges without sending.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from jarvis_alerts.contracts import (
    ALERT_MAX_AGE_S,
    BACKOFF_CAP_S,
    EXHAUSTED_RETRY_COOLDOWN_S,
    LEASE_S,
    MAX_ATTEMPTS,
    PRUNE_AFTER_FAILURES,
    Alert,
    DeadReason,
    OutboxRow,
    Priority,
    RowState,
    SendResult,
    Subscription,
    backoff_seconds,
)
from jarvis_alerts.outbox import DEDUPE_WINDOW_S, Outbox
from jarvis_alerts.transports import FakeTransport
from jarvis_alerts.worker import RunReport, SimClock, Worker
from lucifer_gen.seed import SeedFields, Stream, format_seed

__all__ = [
    "DEFECTS",
    "GateReport",
    "Problem",
    "run_gate",
    "main",
    "T0",
    "TICK_S",
    "ROUNDS_PER_WAVE",
    "BACKFILL_WINDOW_S",
    "BATCH",
    "TRANSIENT_P",
    "PERMANENT_EVERY",
    "GONE_EVERY",
    "EXHAUST_EVERY",
    "STALE_EVERY",
    "LATE_P",
]

#: Sim time the run starts at (unix seconds); only its differences matter.
T0 = 1_700_000_000.0
#: Sim seconds the clock advances after every round.
TICK_S = 5.0
#: Worker rounds (or crashes) between two publish waves.
ROUNDS_PER_WAVE = 4
#: How far back a late device is backfilled: the two previous waves, not three.
BACKFILL_WINDOW_S = 50.0
#: Rows a worker pass leases.
BATCH = 50
#: The wave after whose publish a slow worker leases the gone devices' rows
#: and holds them for STALL_LEASE_S; the devices go gone two waves later,
#: while it still holds them.  Needs n_alerts > STALL_WAVE + 2; smaller
#: runs script the gone moment by call count instead.
STALL_WAVE = 2
STALL_LEASE_S = 120.0
#: The brief hold of the wave after STALL_WAVE: shorter than the gap to the
#: next wave's publish (ROUNDS_PER_WAVE * TICK_S = 20 s), so the held row is
#: reclaimed in the very round the next wave's row is first due and the two
#: are leased in one batch.
SIBLING_HOLD_S = 15.0
#: Share of sends the transport fails transiently.
TRANSIENT_P = 0.15
#: One device of every PERMANENT_EVERY-th profile rejects every send.
PERMANENT_EVERY = 10
#: One device of every GONE_EVERY-th profile (offset by one) goes gone partway.
GONE_EVERY = 8
#: One device of every EXHAUST_EVERY-th profile (offset by two) fails its
#: first-wave alert transiently until the retry budget is spent; then the
#: outage is over and the revived row goes out.
EXHAUST_EVERY = 12
#: One device of every STALE_EVERY-th profile (offset by five) fails the
#: prelude alert -- published ALERT_MAX_AGE_S before the waves -- for
#: good; its row is DEAD as exhausted and must never be revived.
STALE_EVERY = 12
#: Random transient failures in a row on one device stop here, so that with
#: the exhausting row's MAX_ATTEMPTS forced ones a device never reaches
#: PRUNE_AFTER_FAILURES.
DEVICE_STREAK_CAP = PRUNE_AFTER_FAILURES - 1 - MAX_ATTEMPTS
#: Chance an ordinary device registers late.
LATE_P = 0.3
#: Rounds without any change before the run is declared stalled: longer
#: than the longest wait the retry policy can impose (the backoff cap, or
#: the cooldown before a dead letter is revived, plus one lease).
STALL_ROUNDS = int((max(BACKOFF_CAP_S, EXHAUSTED_RETRY_COOLDOWN_S) + LEASE_S) / TICK_S) + 2
#: Hard cap on rounds, a backstop behind the stall detector.
MAX_ROUNDS = 20_000
#: The defects ``inject_defect`` accepts; see the module docstring.
DEFECTS: Tuple[str, ...] = (
    "skip_reclaim", "double_lease", "lose_on_crash",
    "no_lease", "no_backoff", "lease_gone", "attempts_lost", "no_revive",
)

KINDS = ("render_done", "gpu_missing", "portal_open", "sigil_dropped", "descent_cleared")
PRIORITIES = (Priority.LOW, Priority.NORMAL, Priority.HIGH)
PRIORITY_WEIGHTS = (2, 5, 3)
TRANSPORT_NAME = "fake"
#: The gone-scripted devices register under this transport name so the slow
#: worker can lease exactly their rows (``Outbox.lease(transports=...)``);
#: the worker serves both names with the same FakeTransport.
GONE_TRANSPORT_NAME = "fake-gone"

_OK = SendResult(ok=True)
_TRANSIENT = SendResult(ok=False, retryable=True, reason="gate: transient failure (503)")
_EXHAUST = SendResult(ok=False, retryable=True, reason="gate: transient failure (503), exhausting")
_GONE = SendResult(ok=False, gone=True, reason="gate: endpoint gone (410)")
_PERMANENT = SendResult(ok=False, retryable=False, reason="gate: permanent rejection (403)")
#: The reason worker.py writes when it finds the device gone after the lease.
_NO_SUBSCRIPTION_REASON = "no subscription"

DeviceKey = Tuple[str, str]           # (profile_id, device_id)
Triple = Tuple[str, str, str]         # (profile_id, device_id, alert_id)


# ---------------------------------------------------------------------------
# Report types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One failed assertion.  ``kind`` is stable for tests; ``detail`` is
    for a person; the ids say where; ``repro`` is the call that reproduces
    the whole run.  Never carries a blob: none is in scope anywhere."""

    kind: str
    detail: str
    profile_id: str = ""
    device_id: str = ""
    alert_id: str = ""
    row_id: Optional[int] = None
    repro: str = ""

    def __str__(self) -> str:
        where = []
        if self.profile_id:
            where.append(f"profile={self.profile_id}")
        if self.device_id:
            where.append(f"device={self.device_id}")
        if self.alert_id:
            where.append(f"alert={self.alert_id}")
        if self.row_id is not None:
            where.append(f"row={self.row_id}")
        tail = f" [{' '.join(where)}]" if where else ""
        return f"{self.kind}: {self.detail}{tail}"


@dataclass
class GateReport:
    """What :func:`run_gate` found.

    ``counts`` is JSON-friendly and every value is an int.  ``problems``
    is in detection order, so ``first_failure`` is the earliest, most
    upstream one.  ``converged`` says whether the run ended with nothing
    outstanding (as opposed to stalling or hitting ``MAX_ROUNDS``).
    """

    n_profiles: int
    n_alerts: int
    seed: int
    crash_every: int
    inject_defect: Optional[str]
    converged: bool
    counts: Dict[str, int] = field(default_factory=dict)
    problems: List[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def first_failure(self) -> Optional[Problem]:
        return self.problems[0] if self.problems else None

    @property
    def repro(self) -> str:
        return _repro(self.n_profiles, self.n_alerts, self.seed, self.crash_every, self.inject_defect)

    def kinds(self) -> List[str]:
        """Distinct problem kinds, in first-seen order."""
        seen: List[str] = []
        for p in self.problems:
            if p.kind not in seen:
                seen.append(p.kind)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_profiles": self.n_profiles,
            "n_alerts": self.n_alerts,
            "seed": format_seed(self.seed),
            "crash_every": self.crash_every,
            "inject_defect": self.inject_defect,
            "converged": self.converged,
            "ok": self.ok,
            "counts": dict(self.counts),
            "problems": [p.__dict__ for p in self.problems],
        }

    def summary(self, max_problems: int = 5) -> str:
        lines = [
            f"gate: profiles={self.n_profiles} alerts={self.n_alerts} seed={format_seed(self.seed)} "
            f"crash_every={self.crash_every} defect={self.inject_defect!r} "
            f"converged={'yes' if self.converged else 'NO'}",
            "  counts: " + " ".join(f"{k}={v}" for k, v in self.counts.items()),
        ]
        if self.ok:
            lines.append("  result: OK")
        else:
            lines.append(f"  result: FAIL ({len(self.problems)} problem(s); kinds: {', '.join(self.kinds())})")
            for p in self.problems[:max_problems]:
                lines.append(f"    - {p}")
            if len(self.problems) > max_problems:
                lines.append(f"    ... {len(self.problems) - max_problems} more")
            lines.append(f"  repro: {self.repro}")
        return "\n".join(lines)


def _repro(n_profiles: int, n_alerts: int, seed: int, crash_every: int, defect: Optional[str]) -> str:
    return (
        f"run_gate(n_profiles={n_profiles}, n_alerts={n_alerts}, seed={format_seed(seed)}, "
        f"crash_every={crash_every}, inject_defect={defect!r})"
    )


# ---------------------------------------------------------------------------
# The scenario: devices and the transport script.
# ---------------------------------------------------------------------------


@dataclass
class _Device:
    """One device of the scenario and what the gate learns about it."""

    profile_id: str
    device_id: str
    reg_wave: int                      # 0: before wave 0; n_alerts: after the last wave
    permanent: bool = False            # every send is rejected permanently
    gone_after: Optional[int] = None   # sends answered normally before "gone"; None with ``stalled``
    stalled: bool = False              # a slow worker holds its rows from STALL_WAVE; gone once armed
    gone_armed: bool = False           # the next send of one of ``gone_alert_ids`` says gone
    exhaust: bool = False              # the first-wave alert fails until the budget is spent
    stale: bool = False                # the prelude alert fails for ever; its row must stay dead
    registered_at: Optional[float] = None
    gone_index: Optional[int] = None   # transport call index of the first gone result
    gone_at: Optional[float] = None    # sim time of that call

    @property
    def key(self) -> DeviceKey:
        return (self.profile_id, self.device_id)

    @property
    def went_gone(self) -> bool:
        return self.gone_index is not None

    @property
    def gone_scripted(self) -> bool:
        return self.gone_after is not None or self.stalled

    @property
    def gone_alert_ids(self) -> Tuple[str, str]:
        """The briefly held row and the one it comes back with: whichever
        of the two is sent first takes a stalled device gone."""
        return (f"{self.profile_id}-a{STALL_WAVE + 1:03d}", f"{self.profile_id}-a{STALL_WAVE + 2:03d}")

    @property
    def exhaust_alert_id(self) -> Optional[str]:
        return f"{self.profile_id}-a000" if self.exhaust else None

    @property
    def stale_alert_id(self) -> Optional[str]:
        return f"{self.profile_id}-a-old" if self.stale else None


class _Script:
    """The FakeTransport script (design point 2: the transport is injected).

    Called as ``script(subscription, alert, index)`` by
    :class:`~jarvis_alerts.transports.FakeTransport`; ``index`` is the
    call's position in ``transport.calls``.  Per call, in this order:

    1. a permanent device is rejected, not retryable, not gone;
    2. a gone device that has had its ``gone_after`` normal answers, or a
       stalled device sent one of its two ``gone_alert_ids`` once armed,
       says gone from then on, and the first such call is recorded on the
       device;
    3. an exhausting device's first-wave alert fails transiently on each of
       its first ``MAX_ATTEMPTS`` sends, no draw, and a stale device's
       prelude alert fails transiently on every send, no draw;
    4. otherwise a transient failure with probability ``TRANSIENT_P`` from
       the seeded stream, unless this (device, alert) has already failed
       ``MAX_ATTEMPTS - 1`` times (so no row dead-letters by chance) or the
       device is ``DEVICE_STREAK_CAP`` consecutive failures deep (so no
       device is pruned); the draw happens either way;
    5. otherwise ok.

    Reads only ``profile_id``, ``device_id`` and ``alert.id`` from its
    arguments; the blob is never touched.  ``results[i]`` is the result of
    call ``i`` so the gate can pair every call with its answer.
    """

    def __init__(self, devices: Dict[DeviceKey, _Device], stream: Stream, clock: Callable[[], float]) -> None:
        self._devices = devices
        self._stream = stream
        self._clock = clock
        self.results: List[SendResult] = []
        self.calls_to: Counter = Counter()
        self.streak: Dict[Triple, int] = {}
        self.device_streak: Dict[DeviceKey, int] = {}   # consecutive transient failures, as the outbox counts
        self.unknown_devices: List[DeviceKey] = []

    def __call__(self, subscription: Subscription, alert: Alert, index: int) -> SendResult:
        key = (subscription.profile_id, subscription.device_id)
        self.calls_to[key] += 1
        device = self._devices.get(key)
        if device is None:
            # A send to a device the scenario never registered is a bug in
            # the registry; refuse permanently so the row dead-letters with
            # a reason, and remember it for the report.
            self.unknown_devices.append(key)
            result = _PERMANENT
        elif device.permanent:
            result = _PERMANENT
        elif device.went_gone or (device.gone_armed and alert.id in device.gone_alert_ids) or (
                device.gone_after is not None and self.calls_to[key] > device.gone_after):
            if device.gone_index is None:
                device.gone_index = index
                device.gone_at = self._clock()
            result = _GONE
        else:
            triple = (key[0], key[1], alert.id)
            streak = self.streak.get(triple, 0)
            if device.exhaust and alert.id == device.exhaust_alert_id and streak < MAX_ATTEMPTS:
                self.streak[triple] = streak + 1
                result = _EXHAUST
            elif device.stale and alert.id == device.stale_alert_id:
                self.streak[triple] = streak + 1
                result = _EXHAUST
            else:
                fail = self._stream.chance(TRANSIENT_P)
                if fail and streak < MAX_ATTEMPTS - 1 and self.device_streak.get(key, 0) < DEVICE_STREAK_CAP:
                    self.streak[triple] = streak + 1
                    result = _TRANSIENT
                else:
                    result = _OK
            if result.ok:
                self.device_streak[key] = 0
            elif result.retryable:
                self.device_streak[key] = self.device_streak.get(key, 0) + 1
        self.results.append(result)
        return result


class _DefectiveStore:
    """A deliberately broken outbox for ``inject_defect``.  Wraps the real
    outbox for the worker and the crash path and perturbs one thing
    (see the module docstring); everything else is delegated unchanged.

    ``double_lease`` hands every leased row out twice in the same batch,
    as a copy, which is what two workers would see if ``lease`` did not
    take the write lock up front.  ``no_lease`` releases every leased row
    on the spot, so the store never holds it.  ``no_backoff``,
    ``lease_gone`` and ``attempts_lost`` reach into the outbox's sqlite
    file to do what a buggy store would do inside its own transaction.
    ``no_revive`` answers ``revive_exhausted`` with 0 without calling the
    outbox: dead letters stay dead.
    """

    def __init__(self, inner: Outbox, defect: str) -> None:
        self._inner = inner
        self._defect = defect

    def lease(self, now: float, limit: int, lease_s: float = LEASE_S,
              transports: Optional[Set[str]] = None) -> List[OutboxRow]:
        if transports is None:
            rows = self._inner.lease(now, limit, lease_s)
        else:
            rows = self._inner.lease(now, limit, lease_s, transports=transports)
        if self._defect == "double_lease":
            return rows + [replace(r) for r in rows]
        if self._defect == "no_lease":
            for r in rows:
                self._inner.release(r.row_id, "")
            return rows
        if self._defect == "lease_gone" and len(rows) < limit:
            conn = self._inner._conn
            extra = conn.execute(
                """
                SELECT o.row_id FROM outbox o
                JOIN subscriptions s ON s.profile_id = o.profile_id AND s.device_id = o.device_id
                WHERE o.state = 'pending' AND o.next_due <= ? AND s.gone = 1
                ORDER BY o.row_id LIMIT ?
                """, (now, limit - len(rows)),
            ).fetchall()
            for (rid,) in extra:
                conn.execute("UPDATE outbox SET state = 'leased', lease_until = ? WHERE row_id = ?",
                             (now + lease_s, rid))
                rows.append(self._inner.row(rid))
        return rows

    def mark(self, row_id: int, result: SendResult, now: float, jitter: Optional[float] = None) -> RowState:
        state = self._inner.mark(row_id, result, now, jitter)
        conn = self._inner._conn
        if self._defect == "no_backoff" and state is RowState.PENDING:
            conn.execute("UPDATE outbox SET next_due = ? WHERE row_id = ?", (now, row_id))
        elif self._defect == "attempts_lost":
            conn.execute("DELETE FROM attempts WHERE row_id = ? AND attempt < "
                         "(SELECT max(attempt) FROM attempts WHERE row_id = ?)", (row_id, row_id))
            conn.execute("UPDATE attempts SET attempt = 1 WHERE row_id = ?", (row_id,))
            conn.execute("UPDATE outbox SET attempts = 1 WHERE row_id = ?", (row_id,))
        return state

    def release(self, row_id: int, reason: str = "") -> RowState:
        return self._inner.release(row_id, reason)

    def revive_exhausted(self, now: float) -> int:
        if self._defect == "no_revive":
            return 0
        return self._inner.revive_exhausted(now)

    def subscription(self, profile_id: str, device_id: str) -> Optional[Subscription]:
        return self._inner.subscription(profile_id, device_id)

    def alert(self, alert_id: str) -> Optional[Alert]:
        return self._inner.alert(alert_id)

    def stats(self, dead_limit: int = 50) -> Dict[str, Any]:
        return self._inner.stats(dead_limit)


AddProblem = Callable[..., None]


class _Observed:
    """The store as the worker (and the crash path) see it, watched.

    Sits between the worker and the store under test and checks, at every
    call, what the outbox contract promises about *that* call, against the
    real outbox's rows read back afterwards:

    ``lease``    no row comes back while a lease this observer saw on it is
                 still live (``lease_not_exclusive``); no row of a device
                 without a live subscription (``leased_for_gone_device``);
                 no row before its ``next_due`` (``leased_before_due``);
                 and when the call returned fewer rows than ``limit``, every
                 row whose lease had expired and that is due behind a live
                 device came back (``reclaim_skipped``) -- a reclaim that
                 merely happens late is a finding, not just one that never
                 happens; and the batch is in priority order, with no
                 higher-priority due row left behind when it was full
                 (``lease_order``).
    ``mark``     on an unsettled row: the state returned and stored is the
                 policy's for this attempt and result, the attempt counter
                 went up by one, and a backoff landed at ``now +
                 backoff_seconds(n, jitter)`` (``mark_policy_mismatch``,
                 ``backoff_not_applied``); on a settled row: nothing on the
                 row changed.  Every mark on an unsettled row is counted per
                 row, for the end-of-run comparison with ``attempts_for``.
    ``release``  a LEASED row is PENDING afterwards with no new attempt.
    ``revive_exhausted``
                 judged against the contract in contracts.py, from the
                 DEAD rows read back before and after the call: every row
                 DEAD as EXHAUSTED for at least ``EXHAUSTED_RETRY_COOLDOWN_S``,
                 of an alert younger than ``ALERT_MAX_AGE_S`` and of a
                 reachable device is PENDING afterwards, due now, with
                 ``attempts_base == attempts`` and no dead reason
                 (``revive_skipped``, ``revive_mismatch``); no other DEAD
                 row left DEAD (``revived_wrong_row``); the count returned
                 is the number that did (``revive_count``).  A skipped row
                 is reported once, the round its revival was due.

    ``base`` holds, per row, the attempts at its last re-queue (the gate
    sets it when a repair re-queues a row, the observer when a revival
    does), so the policy is judged on the row's current budget as the
    outbox defines it.  ``revived`` counts revivals per row.
    """

    def __init__(self, inner: Any, outbox: Outbox, add: AddProblem) -> None:
        self._inner = inner
        self._outbox = outbox
        self._add = add
        self.held: Dict[int, float] = {}
        self.marks: Counter = Counter()
        self.releases = 0
        self.base: Dict[int, int] = {}
        self.revived: Counter = Counter()
        self._revive_reported: Set[int] = set()

    @property
    def total_marks(self) -> int:
        return sum(self.marks.values())

    def lease(self, now: float, limit: int, lease_s: float = LEASE_S,
              transports: Optional[Set[str]] = None) -> List[OutboxRow]:
        if transports is None:
            rows = self._inner.lease(now, limit, lease_s)
        else:
            rows = self._inner.lease(now, limit, lease_s, transports=transports)
        seen: Set[int] = set()
        for row in rows:
            ids = dict(profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
            until = self.held.get(row.row_id)
            if row.row_id in seen:
                self._add("lease_not_exclusive", "the same row was handed out twice in one batch", **ids)
            elif until is not None and not until < now:
                self._add("lease_not_exclusive", f"handed out at {now - T0:.0f}s while a lease on it runs "
                          f"until {until - T0:.0f}s", **ids)
            seen.add(row.row_id)
            sub = self._outbox.subscription(row.profile_id, row.device_id)
            if sub is None or sub.gone:
                self._add("leased_for_gone_device", "a row was handed out for a device with no live subscription", **ids)
            if row.next_due > now:
                self._add("leased_before_due", f"handed out at {now - T0:.0f}s, due at {row.next_due - T0:.0f}s", **ids)
            self.held[row.row_id] = now + lease_s
        self._check_order(rows, now, limit, transports)
        if len(rows) < limit:
            for row_id, until in list(self.held.items()):
                if row_id in seen or not until < now:
                    continue
                stored = self._outbox.row(row_id)
                if stored is None or stored.state in (RowState.DELIVERED, RowState.DEAD):
                    self.held.pop(row_id, None)
                    continue
                sub = self._outbox.subscription(stored.profile_id, stored.device_id)
                live = sub is not None and not sub.gone
                if transports is not None and (sub is None or sub.transport not in transports):
                    continue   # a filtered lease is allowed to leave other transports' rows
                if live and stored.next_due <= now:
                    self._add("reclaim_skipped", f"lease expired at {until - T0:.0f}s, due, device live, "
                              f"and the lease at {now - T0:.0f}s had room ({len(rows)} of {limit}) but "
                              f"did not re-offer it (row is {stored.state.value})",
                              profile_id=stored.profile_id, device_id=stored.device_id,
                              alert_id=stored.alert_id, row_id=row_id)
                    self.held.pop(row_id, None)   # report once
        return rows

    def _check_order(self, rows: List[OutboxRow], now: float, limit: int,
                     transports: Optional[Set[str]]) -> None:
        """Priority high first, then ``next_due``, then ``row_id``; a full
        batch leaves no higher-priority due row of a live device (of a
        transport it asked for) behind."""
        priority: Dict[str, int] = {}

        def prio(row: OutboxRow) -> int:
            if row.alert_id not in priority:
                alert = self._outbox.alert(row.alert_id)
                priority[row.alert_id] = int(alert.priority) if alert is not None else 0
            return priority[row.alert_id]

        keys = [(-prio(r), r.next_due, r.row_id) for r in rows]
        if keys != sorted(keys):
            first = next(i for i in range(1, len(keys)) if keys[i] < keys[i - 1])
            r = rows[first]
            self._add("lease_order", f"row {first} of the batch is out of order (priority {prio(r)}, "
                      f"due {r.next_due - T0:.0f}s) after row {first - 1} (priority {prio(rows[first - 1])}, "
                      f"due {rows[first - 1].next_due - T0:.0f}s)",
                      profile_id=r.profile_id, device_id=r.device_id, alert_id=r.alert_id, row_id=r.row_id)
        if rows and len(rows) >= limit:
            lowest = min(prio(r) for r in rows)
            names = sorted(transports) if transports is not None else None
            transport_filter = "" if names is None else f" AND s.transport IN ({', '.join('?' * len(names))})"
            left = self._outbox._conn.execute(
                f"""
                SELECT o.row_id, o.profile_id, o.device_id, o.alert_id, a.priority FROM outbox o
                JOIN alerts a ON a.id = o.alert_id
                JOIN subscriptions s ON s.profile_id = o.profile_id AND s.device_id = o.device_id
                WHERE o.state = 'pending' AND o.next_due <= ? AND s.gone = 0 AND a.priority > ?{transport_filter}
                ORDER BY o.row_id LIMIT 1
                """, (now, lowest, *(names or [])),
            ).fetchone()
            if left is not None:
                self._add("lease_order", f"a full batch of {len(rows)} took a priority-{lowest} row and left a "
                          f"due priority-{int(left[4])} row behind",
                          profile_id=str(left[1]), device_id=str(left[2]), alert_id=str(left[3]), row_id=int(left[0]))

    def mark(self, row_id: int, result: SendResult, now: float, jitter: Optional[float] = None) -> RowState:
        before = self._outbox.row(row_id)
        state = self._inner.mark(row_id, result, now, jitter)
        after = self._outbox.row(row_id)
        if before is None or after is None:
            return state
        ids = dict(profile_id=before.profile_id, device_id=before.device_id, alert_id=before.alert_id, row_id=row_id)
        if before.state in (RowState.DELIVERED, RowState.DEAD):
            if state is not before.state or after.state is not before.state or after.attempts != before.attempts:
                self._add("mark_policy_mismatch", f"a mark on a {before.state.value} row changed it: returned "
                          f"{state.value}, stored {after.state.value} with {after.attempts} attempts", **ids)
            return state
        self.marks[row_id] += 1
        n = before.attempts + 1 - self.base.get(row_id, 0)
        if result.ok:
            expected = RowState.DELIVERED
        elif result.gone:
            expected = RowState.DEAD
        elif result.retryable:
            expected = RowState.PENDING if n < MAX_ATTEMPTS else RowState.DEAD
        else:
            expected = RowState.DEAD
        if state is not expected or after.state is not expected:
            self._add("mark_policy_mismatch", f"attempt {n} of the budget, result "
                      f"{'ok' if result.ok else 'gone' if result.gone else 'transient' if result.retryable else 'permanent'}: "
                      f"policy says {expected.value}, store returned {state.value} and holds {after.state.value}", **ids)
        if after.attempts != before.attempts + 1:
            self._add("mark_policy_mismatch", f"attempts went {before.attempts} -> {after.attempts} on one mark", **ids)
        if expected is RowState.PENDING and jitter is not None:
            due = now + backoff_seconds(n, jitter)
            if abs(after.next_due - due) > 1e-6:
                self._add("backoff_not_applied", f"failure {n} should be due at {due - T0:.3f}s, "
                          f"the row is due at {after.next_due - T0:.3f}s", **ids)
        if state is not RowState.LEASED:
            self.held.pop(row_id, None)
        return state

    def release(self, row_id: int, reason: str = "") -> RowState:
        before = self._outbox.row(row_id)
        state = self._inner.release(row_id, reason)
        after = self._outbox.row(row_id)
        if before is not None and after is not None and before.state is RowState.LEASED:
            self.releases += 1
            if state is not RowState.PENDING or after.state is not RowState.PENDING or after.attempts != before.attempts:
                self._add("release_mismatch", f"released row is {after.state.value} with {after.attempts} attempts",
                          profile_id=before.profile_id, device_id=before.device_id,
                          alert_id=before.alert_id, row_id=row_id)
            self.held.pop(row_id, None)
        return state

    def revive_exhausted(self, now: float) -> int:
        before = self._dead_rows()
        n = self._inner.revive_exhausted(now)
        left = set(before) - self._dead_ids()          # rows that left DEAD in this call
        due = {row_id for row_id, r in before.items() if self._due_for_revival(r, now)}
        for row_id in sorted(due):
            pid, did, aid, attempts, _reason, dead_at, created_at, _reachable = before[row_id]
            ids = dict(profile_id=pid, device_id=did, alert_id=aid, row_id=row_id)
            after = self._outbox.row(row_id)
            if row_id not in left or after is None or after.state is not RowState.PENDING:
                if row_id not in self._revive_reported:
                    self._revive_reported.add(row_id)
                    self._add("revive_skipped", f"DEAD as exhausted since {dead_at - T0:.0f}s, its cooldown of "
                              f"{EXHAUSTED_RETRY_COOLDOWN_S:.0f}s passed at {now - T0:.0f}s, its alert "
                              f"{now - created_at:.0f}s old (limit {ALERT_MAX_AGE_S:.0f}s), device reachable, "
                              f"but revive_exhausted left it {'missing' if after is None else after.state.value}", **ids)
                continue
            self.revived[row_id] += 1
            self.base[row_id] = after.attempts
            if (after.attempts != attempts or after.attempts_base != after.attempts
                    or abs(after.next_due - now) > 1e-6 or after.dead_reason is not None or after.dead_at != 0.0):
                self._add("revive_mismatch", f"revived row has {after.attempts} attempts (had {attempts}), "
                          f"attempts_base {after.attempts_base}, due at {after.next_due - T0:.0f}s (now {now - T0:.0f}s), "
                          f"dead_reason {after.dead_reason}", **ids)
        for row_id in sorted(left - due):
            pid, did, aid, _attempts, reason, dead_at, created_at, reachable = before[row_id]
            if reason != DeadReason.EXHAUSTED.value:
                why = f"it was DEAD as {reason or 'nothing'}"
            elif not created_at + ALERT_MAX_AGE_S > now:
                why = f"its alert was {now - created_at:.0f}s old, past {ALERT_MAX_AGE_S:.0f}s"
            elif not dead_at + EXHAUSTED_RETRY_COOLDOWN_S <= now:
                why = f"it died at {dead_at - T0:.0f}s, only {now - dead_at:.0f}s before"
            elif not reachable:
                why = "its device had no reachable subscription"
            else:
                why = "no reason the gate can see"
            self._add("revived_wrong_row", f"a DEAD row left DEAD in revive_exhausted although it was not due: {why}",
                      profile_id=pid, device_id=did, alert_id=aid, row_id=row_id)
        if n != len(left):
            self._add("revive_count", f"revive_exhausted returned {n}; {len(left)} rows left DEAD")
        return n

    @staticmethod
    def _due_for_revival(r: Tuple[Any, ...], now: float) -> bool:
        _pid, _did, _aid, _attempts, reason, dead_at, created_at, reachable = r
        return (reason == DeadReason.EXHAUSTED.value and dead_at + EXHAUSTED_RETRY_COOLDOWN_S <= now
                and created_at + ALERT_MAX_AGE_S > now and bool(reachable))

    def _dead_rows(self) -> Dict[int, Tuple[Any, ...]]:
        """Every DEAD row with what the contract judges a revival by."""
        recs = self._outbox._conn.execute(
            """
            SELECT o.row_id, o.profile_id, o.device_id, o.alert_id, o.attempts, o.dead_reason, o.dead_at,
                   a.created_at,
                   CASE WHEN s.device_id IS NULL THEN 0 WHEN s.gone = 0 OR s.pruned = 1 THEN 1 ELSE 0 END
            FROM outbox o
            JOIN alerts a ON a.id = o.alert_id
            LEFT JOIN subscriptions s ON s.profile_id = o.profile_id AND s.device_id = o.device_id
            WHERE o.state = 'dead'
            """
        ).fetchall()
        return {int(r[0]): (str(r[1]), str(r[2]), str(r[3]), int(r[4]), str(r[5]), float(r[6]),
                            float(r[7]), int(r[8])) for r in recs}

    def _dead_ids(self) -> Set[int]:
        return {int(r[0]) for r in self._outbox._conn.execute("SELECT row_id FROM outbox WHERE state = 'dead'")}

    def subscription(self, profile_id: str, device_id: str) -> Optional[Subscription]:
        return self._inner.subscription(profile_id, device_id)

    def alert(self, alert_id: str) -> Optional[Alert]:
        return self._inner.alert(alert_id)

    def stats(self, dead_limit: int = 50) -> Dict[str, Any]:
        return self._inner.stats(dead_limit)


# ---------------------------------------------------------------------------
# The gate itself.
# ---------------------------------------------------------------------------


class _Gate:
    """One run.  Build with the arguments of :func:`run_gate`, then ``run()``.

    Phases: register the initial devices; when a stale device exists,
    publish the prelude alert on its profiles, run rounds until nothing
    is in flight (the stale row is DEAD by then) and move the clock
    ``ALERT_MAX_AGE_S``; for each wave, register that wave's late devices
    (with backfill), publish one alert per profile, run ``ROUNDS_PER_WAVE``
    rounds; register the devices due after the last wave; run rounds to
    convergence (which waits for every revivable dead letter to come back
    and settle); check.  A round is one ``Worker.run_once`` or a crash.

    Crash cadence: one crash in every window of ``crash_every`` rounds, at
    a position inside the window drawn from the seeded stream
    ``alerts.gate.crash``.  Not at a fixed offset, because the outbox only
    reclaims a lease once ``lease_until < now``, and a crasher whose period
    equals the ticks to expiry wakes exactly when its own rows come back,
    re-leases them and abandons them again, for ever.  That livelock would
    measure the gate's own tick alignment, not the outbox.
    """

    def __init__(
        self,
        n_profiles: int,
        n_alerts: int,
        seed: int,
        crash_every: int,
        defect: Optional[str],
        db_path: Optional[str],
    ) -> None:
        self.n_profiles = n_profiles
        self.n_alerts = n_alerts
        self.seed = seed
        self.crash_every = crash_every
        self.defect = defect
        self.repro = _repro(n_profiles, n_alerts, seed, crash_every, defect)

        fields = SeedFields.parse(seed)
        self.pub_stream = fields.stream("alerts.gate.publish")
        self.clock = SimClock(T0)
        # The worker supplies jitter on every backoff, so the store's own
        # stream should never be drawn; it gets its own label regardless so
        # that, if it ever is, the other streams are undisturbed.
        self.outbox = Outbox(
            db_path if db_path is not None else ":memory:",
            clock=self.clock,
            jitter=fields.stream("alerts.gate.backoff.store").random,
        )
        self.devices: Dict[DeviceKey, _Device] = {}
        self.profiles: List[str] = []
        self._build_devices(fields.stream("alerts.gate.devices"))

        self.script = _Script(self.devices, fields.stream("alerts.gate.transport"), self.clock)
        self.transport = FakeTransport(self.script, name=TRANSPORT_NAME)
        self.problems: List[Problem] = []
        inner: Any = self.outbox
        if defect in ("double_lease", "no_lease", "no_backoff", "lease_gone", "attempts_lost", "no_revive"):
            inner = _DefectiveStore(self.outbox, defect)
        self.store = _Observed(inner, self.outbox, self._add)
        self.worker = Worker(
            self.store,
            {TRANSPORT_NAME: self.transport, GONE_TRANSPORT_NAME: self.transport},
            clock=self.clock,
            jitter=fields.stream("alerts.gate.backoff").random,
            batch=BATCH,
            lease_s=LEASE_S,
        )

        # The ledger (see the module docstring).
        self.expected: Set[Triple] = set()
        self.expected_order: List[Triple] = []
        self.stored_alerts: List[Tuple[str, str, float]] = []   # (alert_id, profile_id, created_at)
        self.deduped: List[str] = []
        self.key_last_created: Dict[str, Dict[str, Tuple[float, str]]] = {}   # key -> (created_at, alert_id)
        self.publishes = 0
        self.probes = 0
        self.prelude_publishes = 0
        self.rows_backfilled = 0
        self.rows_repaired = 0
        self.rows_requeued = 0
        self.requeued: Counter = Counter()          # row_id -> times a repair re-queued it

        self.rounds = 0
        self.crashes = 0
        self.stalls = 0
        self.rows_stalled = 0
        self.rows_abandoned = 0
        self.crash_stream = fields.stream("alerts.gate.crash")
        self._crash_window = -1
        self._crash_offset = 0
        self.worker_total = RunReport.empty()
        self.converged = False

    # -- scenario -----------------------------------------------------------

    def _build_devices(self, stream: Stream) -> None:
        """1 to 3 devices per profile; one permanent device on every
        PERMANENT_EVERY-th profile, one gone-partway device on every
        GONE_EVERY-th (offset so the two never coincide), one exhausting
        device on every EXHAUST_EVERY-th (offset again), one stale device
        on every STALE_EVERY-th (offset once more); the rest register
        late with chance LATE_P at a wave drawn from the stream.  The
        scripted devices are always registered from the start.  In a
        profile with a gone device, the first ordinary device instead
        registers after the last wave, so its backfill runs while a device
        of its profile is gone (the outbox must not create rows for that
        one)."""
        for i in range(self.n_profiles):
            pid = f"p{i:03d}"
            self.profiles.append(pid)
            n_dev = stream.randint(1, 3)
            if i % GONE_EVERY == 1:
                n_dev = max(n_dev, 2)      # a gone device and one to register after it went gone
            devs = [_Device(pid, f"d{j}", reg_wave=0) for j in range(n_dev)]
            special: Set[int] = set()
            if i % PERMANENT_EVERY == 0:
                j = stream.randint(0, n_dev - 1)
                devs[j].permanent = True
                special.add(j)
            has_gone = False
            if i % GONE_EVERY == 1:
                candidates = [j for j in range(n_dev) if j not in special]
                if candidates:
                    j = stream.choice(candidates)
                    if self.n_alerts > STALL_WAVE + 2:
                        devs[j].stalled = True
                    else:
                        devs[j].gone_after = stream.randint(1, max(1, self.n_alerts // 2))
                    special.add(j)
                    has_gone = True
            if i % EXHAUST_EVERY == 2:
                candidates = [j for j in range(n_dev) if j not in special]
                if candidates:
                    j = stream.choice(candidates)
                    devs[j].exhaust = True
                    special.add(j)
            if i % STALE_EVERY == 5:
                candidates = [j for j in range(n_dev) if j not in special]
                if candidates:
                    j = stream.choice(candidates)
                    devs[j].stale = True
                    special.add(j)
            if has_gone:
                ordinary = [j for j in range(n_dev) if j not in special]
                if ordinary:
                    devs[ordinary[0]].reg_wave = self.n_alerts
                    special.add(ordinary[0])
            for j, d in enumerate(devs):
                if j not in special and stream.chance(LATE_P):
                    d.reg_wave = stream.randint(1, self.n_alerts)
            for d in devs:
                self.devices[d.key] = d

    def _devices_of(self, pid: str) -> List[_Device]:
        return [d for d in self.devices.values() if d.profile_id == pid]

    # -- problems -----------------------------------------------------------

    def _add(self, kind: str, detail: str, *, profile_id: str = "", device_id: str = "",
             alert_id: str = "", row_id: Optional[int] = None) -> None:
        self.problems.append(Problem(kind, detail, profile_id, device_id, alert_id, row_id, self.repro))

    # -- registration and publishing ---------------------------------------

    def _register(self, d: _Device) -> None:
        """Register a device; a late one is backfilled over the last
        BACKFILL_WINDOW_S and the ledger is extended the same way."""
        now = self.clock()
        blob = json.dumps({"endpoint": f"fake://{d.profile_id}/{d.device_id}"})
        transport = GONE_TRANSPORT_NAME if d.gone_scripted else TRANSPORT_NAME
        self.outbox.register(Subscription(d.profile_id, d.device_id, transport, blob, created_at=now))
        del blob
        d.registered_at = now
        if d.reg_wave == 0:
            return
        since = now - BACKFILL_WINDOW_S
        created = self.outbox.backfill(d.profile_id, since)
        self.rows_backfilled += created
        new = [
            (d.profile_id, d.device_id, aid)
            for aid, pid, t in self.stored_alerts
            if pid == d.profile_id and t > since and (d.profile_id, d.device_id, aid) not in self.expected
        ]
        for triple in new:
            self._expect(triple)
        if created != len(new):
            self._add(
                "backfill_count",
                f"backfill created {created} rows, the ledger expected {len(new)} "
                f"(alerts newer than {since - T0:.0f}s of sim time)",
                profile_id=d.profile_id, device_id=d.device_id,
            )

    def _expect(self, triple: Triple) -> None:
        if triple not in self.expected:
            self.expected.add(triple)
            self.expected_order.append(triple)

    def _dedupe_choice(self, pid: str, kind: str, wave: int, now: float, mode: str) -> Tuple[Optional[str], bool]:
        """Pick this publish's dedupe_key and predict whether the outbox
        must collapse it: yes exactly when the profile stored an alert with
        the same key less than DEDUPE_WINDOW_S ago (Outbox.publish).
        ``mode`` "random" draws from the stream; "recent" repeats the most
        recent key and "stale" the oldest (the two probes); a profile
        without keys probes nothing (``(None, False)`` with no draw)."""
        keys = self.key_last_created.setdefault(pid, {})
        if mode == "recent":
            if not keys:
                return None, False
            key = max(keys, key=lambda k: (keys[k][0], k))
        elif mode == "stale":
            if not keys:
                return None, False
            key = min(keys, key=lambda k: (keys[k][0], k))
        else:
            roll = self.pub_stream.random()
            stalled = [d for d in self._devices_of(pid) if d.stalled]
            if stalled and (STALL_WAVE <= wave <= STALL_WAVE + 2 or wave >= self.n_alerts - 2):
                # The stalled device must have a row to be held, rows to go
                # gone on while the slow worker still holds that one, and
                # the late device of its profile must have alerts to be
                # backfilled with after the gone moment; no collapse there.
                return None, False
            if wave == 0 and any(d.exhaust for d in self._devices_of(pid)):
                # The exhausting row must come back by revival alone: a
                # keyed repeat would repair it first (module docstring).
                return None, False
            recent = [k for k, (t, _) in keys.items() if t > now - DEDUPE_WINDOW_S]
            stale = [k for k, (t, _) in keys.items() if t <= now - DEDUPE_WINDOW_S]
            if roll < 0.10 and recent:
                key = max(recent, key=lambda k: (keys[k][0], k))     # the most recent key: a collision
            elif roll < 0.15 and stale:
                key = min(stale, key=lambda k: (keys[k][0], k))      # a key outside the window: must go out
            elif roll < 0.45:
                return None, False
            else:
                key = f"{kind}:{wave}"
        expect_dupe = key in keys and keys[key][0] > now - DEDUPE_WINDOW_S
        return key, expect_dupe

    def _reachable(self, pid: str) -> List[_Device]:
        """Devices the outbox fans out to, as the ledger sees them."""
        return [d for d in self._devices_of(pid) if d.registered_at is not None and not d.went_gone]

    def _predict_repair(self, pid: str, earlier: str) -> Tuple[List[Triple], List[OutboxRow]]:
        """What a repeat of ``earlier``'s key must repair (Outbox.publish):
        the reachable devices without a row for it, and its DEAD rows on
        reachable devices."""
        rows = {r.device_id: r for r in self.outbox.rows_for(earlier)}
        missing: List[Triple] = []
        dead: List[OutboxRow] = []
        for d in self._reachable(pid):
            row = rows.get(d.device_id)
            if row is None:
                missing.append((pid, d.device_id, earlier))
            elif row.state is RowState.DEAD:
                dead.append(row)
        return missing, dead

    def _publish_wave(self, wave: int, mode: str = "random") -> None:
        now = self.clock()
        for pid in self.profiles:
            aid = f"{pid}-a{wave:03d}"
            if mode == "random":
                kind = self.pub_stream.choice(KINDS)
                priority = self.pub_stream.weighted_choice(PRIORITIES, PRIORITY_WEIGHTS)
            else:
                kind, priority = KINDS[0], Priority.NORMAL
            key, expect_dupe = self._dedupe_choice(pid, kind, wave, now, mode)
            if mode != "random":
                if key is None:
                    continue
                self.probes += 1
            alert = Alert(
                id=aid, profile_id=pid, kind=kind,
                title=f"{kind} for {pid}", body=f"wave {wave}",
                created_at=now, priority=priority, dedupe_key=key,
                data={"url": f"/alerts/{aid}", "wave": wave},
            )
            self.publishes += 1
            if expect_dupe:
                earlier = self.key_last_created[pid][key][1]
                missing, dead = self._predict_repair(pid, earlier)
            else:
                earlier, missing, dead = "", [], []
            created = self.outbox.publish(alert)
            stored = self.outbox.alert(aid) is not None
            if expect_dupe and stored:
                self._add("dedupe_missed", f"key {key!r} was stored again inside the {DEDUPE_WINDOW_S:.0f}s window",
                          profile_id=pid, alert_id=aid)
            if not expect_dupe and not stored:
                self._add("publish_lost", f"publish returned {created} and the alert is not stored",
                          profile_id=pid, alert_id=aid)
            if not stored:
                self.deduped.append(aid)
                self._settle_repair(pid, aid, earlier, created, missing, dead)
                continue
            if key is not None:
                self.key_last_created[pid][key] = (now, aid)
            self.stored_alerts.append((aid, pid, now))
            live = self._reachable(pid)
            for d in live:
                self._expect((pid, d.device_id, aid))
            if created != len(live):
                self._add("fanout_count", f"publish created {created} rows, {len(live)} devices were live",
                          profile_id=pid, alert_id=aid)

    def _publish_prelude(self) -> None:
        """One unkeyed alert on every profile with a stale device, before
        the clock jumps ``ALERT_MAX_AGE_S``: the stale device fails it for
        good, every other device of the profile gets it now.  Booked in
        the ledger like a wave's alert (rows are expected on the devices
        registered at this point; nothing is backfilled with it later,
        because it is outside every backfill window)."""
        now = self.clock()
        for pid in self.profiles:
            stale = [d for d in self._devices_of(pid) if d.stale]
            if not stale:
                continue
            aid = stale[0].stale_alert_id
            alert = Alert(
                id=aid, profile_id=pid, kind=KINDS[0], title=f"{KINDS[0]} for {pid}",
                body="prelude", created_at=now, priority=Priority.NORMAL, dedupe_key=None,
                data={"url": f"/alerts/{aid}", "wave": -1},
            )
            self.publishes += 1
            self.prelude_publishes += 1
            created = self.outbox.publish(alert)
            if self.outbox.alert(aid) is None:
                self._add("publish_lost", f"publish returned {created} and the alert is not stored",
                          profile_id=pid, alert_id=aid)
                continue
            self.stored_alerts.append((aid, pid, now))
            live = self._reachable(pid)
            for d in live:
                self._expect((pid, d.device_id, aid))
            if created != len(live):
                self._add("fanout_count", f"publish created {created} rows, {len(live)} devices were live",
                          profile_id=pid, alert_id=aid)

    def _settle_repair(self, pid: str, aid: str, earlier: str, created: int,
                       missing: List[Triple], dead: List[OutboxRow]) -> None:
        """Book what a deduped publish repaired and hold the outbox to it."""
        expected = len(missing) + len(dead)
        if created != expected:
            self._add("dedupe_repair_count", f"a deduped publish reported {created} rows repaired; "
                      f"{len(missing)} devices lacked a row for the earlier alert and {len(dead)} of its rows were dead",
                      profile_id=pid, alert_id=aid)
        for triple in missing:
            self._expect(triple)
        for row in dead:
            after = self.outbox.row(row.row_id)
            if after is None or after.state is not RowState.PENDING:
                self._add("dedupe_repair_count", "a dead row of the repeated alert was not re-queued",
                          profile_id=pid, device_id=row.device_id, alert_id=earlier, row_id=row.row_id)
                continue
            self.requeued[row.row_id] += 1
            self.store.base[row.row_id] = after.attempts
        self.rows_repaired += len(missing)
        self.rows_requeued += len(dead)

    # -- rounds -------------------------------------------------------------

    def _round(self) -> None:
        self.rounds += 1
        if self.crash_every and self._is_crash_round(self.rounds):
            self._crash()
        else:
            self.worker_total += self.worker.run_once()
        self.clock.advance(TICK_S)

    def _is_crash_round(self, round_no: int) -> bool:
        """One crash per window of ``crash_every`` rounds, at an offset
        drawn once per window (see the class docstring for why)."""
        window, offset = divmod(round_no - 1, self.crash_every)
        if window != self._crash_window:
            self._crash_window = window
            self._crash_offset = self.crash_stream.randint(0, self.crash_every - 1)
        return offset == self._crash_offset

    def _crash(self) -> None:
        """A worker that leases a batch and dies: no send, no mark.  The
        outbox must hand the rows back once the lease expires (design
        point 1).  ``skip_reclaim`` makes the lease effectively infinite;
        ``lose_on_crash`` acknowledges the rows before dying."""
        lease_s = 1e12 if self.defect == "skip_reclaim" else LEASE_S
        rows = self.store.lease(self.clock(), BATCH, lease_s)
        self.crashes += 1
        self.rows_abandoned += len(rows)
        if self.defect == "lose_on_crash":
            for row in rows:
                self.store.mark(row.row_id, _OK, self.clock())

    def _stall(self, hold_s: float) -> None:
        """A slow worker: leases every due row of the gone-scripted devices
        and holds them for ``hold_s`` without sending (it is not dead, so
        it may still mark them; it never does).  At STALL_WAVE the hold is
        long; at the wave after it is brief, so that row is reclaimed in
        the round the following wave's row is first due and the two are
        leased in one batch.  The devices are armed at that following
        wave's publish: whichever of the two rows is sent first says gone
        (a row of any later wave sent before them -- a crash can delay the
        pair by a lease -- is answered normally), and the other row of the
        batch must be left unsent by the worker (it is dead-lettered "no
        subscription").  When the long hold runs out its row comes back
        PENDING onto a gone device and must be parked -- design point 1
        meeting the registry."""
        lease_s = 1e12 if self.defect == "skip_reclaim" else hold_s
        rows = self.store.lease(self.clock(), BATCH, lease_s, transports={GONE_TRANSPORT_NAME})
        self.stalls += 1
        self.rows_stalled += len(rows)

    def _outstanding(self, wait_revival: bool = True) -> bool:
        """Anything LEASED, PENDING behind a live device or -- unless told
        not to wait for it -- DEAD in a way the outbox will undo."""
        s = self.outbox.stats(dead_limit=0)
        if int(s["leased"]) > 0 or int(s["pending"]) - int(s["pending_unreachable"]) > 0:
            return True
        return wait_revival and int(s.get("dead_revivable", 0)) > 0

    def _signature(self) -> Tuple[int, ...]:
        s = self.outbox.stats(dead_limit=0)
        return (s["pending"], s["leased"], s["delivered"], s["dead"], len(self.transport.calls))

    def _converge(self, wait_revival: bool = True) -> None:
        stall = 0
        last: Optional[Tuple[int, ...]] = None
        while self.rounds < MAX_ROUNDS:
            if not self._outstanding(wait_revival):
                self.converged = True
                return
            sig = self._signature()
            if sig == last:
                stall += 1
                if stall >= STALL_ROUNDS:
                    return
            else:
                stall, last = 0, sig
            self._round()

    # -- the run ------------------------------------------------------------

    def run(self) -> GateReport:
        try:
            for d in self.devices.values():
                if d.reg_wave == 0:
                    self._register(d)
            if any(d.stale for d in self.devices.values()):
                # The prelude: the stale devices' alert goes out, their row
                # dies exhausted, and a day passes before the waves begin.
                self._publish_prelude()
                self._converge(wait_revival=False)
                self.converged = False
                self.clock.advance(ALERT_MAX_AGE_S)
            for wave in range(self.n_alerts):
                for d in self.devices.values():
                    if d.reg_wave == wave and wave > 0:
                        self._register(d)
                self._publish_wave(wave)
                if wave in (STALL_WAVE, STALL_WAVE + 1):
                    self._stall(STALL_LEASE_S if wave == STALL_WAVE else SIBLING_HOLD_S)
                if wave == STALL_WAVE + 2:
                    for d in self.devices.values():
                        if d.stalled:
                            d.gone_armed = True   # the first of the pair to be sent says gone: see _stall
                for _ in range(ROUNDS_PER_WAVE):
                    self._round()
            for d in self.devices.values():
                if d.reg_wave == self.n_alerts:
                    self._register(d)
            # Dedupe probes: inside the window every profile's most recent
            # key must collapse (and repair); a window later its oldest key
            # must go out.
            self._publish_wave(self.n_alerts, mode="recent")
            self._converge()
            if self.converged:
                self.converged = False
                self.clock.advance(DEDUPE_WINDOW_S)
                self._publish_wave(self.n_alerts + 1, mode="stale")
                self._converge()
            counts = self._check()
        finally:
            self.outbox.close()
        return GateReport(
            n_profiles=self.n_profiles, n_alerts=self.n_alerts, seed=self.seed,
            crash_every=self.crash_every, inject_defect=self.defect,
            converged=self.converged, counts=counts, problems=list(self.problems),
        )

    # -- the assertions -----------------------------------------------------

    def _check(self) -> Dict[str, int]:
        """Every assertion, in the order that puts the most upstream
        finding first.  Returns the counts.  Read together with the module
        docstring's "What the expected set is"."""
        add = self._add
        if self.worker_total.errors:
            add("worker_store_errors", f"the worker reported {self.worker_total.errors} store errors")
        if self.worker_total.expired:
            add("worker_expired_leases", f"the worker skipped {self.worker_total.expired} rows whose lease had expired")
        if len(self.script.results) != len(self.transport.calls):
            add("transport_log_mismatch", f"{len(self.transport.calls)} calls but {len(self.script.results)} results")
        for key in self.script.unknown_devices[:1]:
            add("sent_to_unknown_device", "the transport was asked to send to a device the scenario never registered",
                profile_id=key[0], device_id=key[1])

        # Deduped publishes stored nothing and created no rows.
        for aid in self.deduped:
            pid = aid.split("-a")[0]
            if self.outbox.alert(aid) is not None:
                add("dedupe_stored", "a deduped alert is in the store", profile_id=pid, alert_id=aid)
            rows = self.outbox.rows_for(aid)
            if rows:
                add("dedupe_created_rows", f"a deduped alert has {len(rows)} outbox rows",
                    profile_id=pid, alert_id=aid, row_id=rows[0].row_id)

        # Every row of every stored alert, by pair and in row order.
        created_at: Dict[str, float] = {aid: t for aid, _, t in self.stored_alerts}
        rows_by_triple: Dict[Triple, List[OutboxRow]] = {}
        all_rows: List[OutboxRow] = []
        for aid, _pid, _t in self.stored_alerts:
            for row in self.outbox.rows_for(aid):
                rows_by_triple.setdefault((row.profile_id, row.device_id, row.alert_id), []).append(row)
                all_rows.append(row)
        all_rows.sort(key=lambda r: r.row_id)
        by_state = Counter(r.state for r in all_rows)

        # Nothing LEASED; PENDING only parked behind a gone device.
        parked = 0
        for row in all_rows:
            if row.state is RowState.LEASED:
                add("row_left_leased", f"still LEASED until {row.lease_until - T0:.0f}s after {row.attempts} attempt(s), "
                    f"last reason {row.last_reason!r}", profile_id=row.profile_id, device_id=row.device_id,
                    alert_id=row.alert_id, row_id=row.row_id)
            elif row.state is RowState.PENDING:
                d = self.devices.get((row.profile_id, row.device_id))
                sub = self.outbox.subscription(row.profile_id, row.device_id)
                live = sub is not None and not sub.gone
                if live or d is None or not d.went_gone or created_at[row.alert_id] > (d.gone_at or 0.0):
                    add("row_left_pending", f"still PENDING (device {'live' if live else 'gone'}) after "
                        f"{row.attempts} attempt(s), due at {row.next_due - T0:.0f}s, last reason {row.last_reason!r}",
                        profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
                else:
                    parked += 1

        # Every ok send in the transport log, by triple and by index.
        ok_calls: Counter = Counter()
        attempts_total = 0
        for i, (pid, did, aid) in enumerate(self.transport.calls):
            if i < len(self.script.results) and self.script.results[i].ok:
                ok_calls[(pid, did, aid)] += 1

        # DELIVERED rows have an ok send behind them; attempt records add up.
        for row in all_rows:
            triple = (row.profile_id, row.device_id, row.alert_id)
            attempts = self.outbox.attempts_for(row.row_id)
            attempts_total += len(attempts)
            if len(attempts) != row.attempts:
                add("attempt_count_mismatch", f"row says {row.attempts} attempts, {len(attempts)} are recorded",
                    profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
            marks = self.store.marks.get(row.row_id, 0)
            if len(attempts) != marks:
                add("attempt_record_mismatch", f"the worker marked this row {marks} time(s), {len(attempts)} attempts are recorded",
                    profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
            if row.state is RowState.DELIVERED:
                if not attempts or not attempts[-1].ok:
                    add("delivered_without_ok_attempt", "DELIVERED but the last recorded attempt is not ok",
                        profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
                if ok_calls.get(triple, 0) == 0:
                    add("delivered_without_send", "DELIVERED but the transport never returned ok for this "
                        "(device, alert): the row was acknowledged without a send",
                        profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
            if row.state is RowState.DEAD:
                if not row.last_reason:
                    add("dead_without_reason", "DEAD row carries no reason",
                        profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
                want = _dead_reason_of(attempts[-1]) if attempts else None
                if row.dead_reason is None or (want is not None and (row.dead_reason is not want
                                                                      or row.dead_at != attempts[-1].at)):
                    add("dead_reason_mismatch", f"DEAD row carries dead_reason "
                        f"{'none' if row.dead_reason is None else row.dead_reason.value} at {row.dead_at - T0:.0f}s; "
                        f"its last attempt says {'nothing' if want is None else want.value}"
                        f"{'' if not attempts else f' at {attempts[-1].at - T0:.0f}s'}",
                        profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)
            elif row.dead_reason is not None or row.dead_at:
                add("dead_reason_mismatch", f"a {row.state.value} row carries dead_reason "
                    f"{row.dead_reason} at {row.dead_at - T0:.0f}s",
                    profile_id=row.profile_id, device_id=row.device_id, alert_id=row.alert_id, row_id=row.row_id)

        # No (device, alert) was sent ok twice; ok sends equal DELIVERED rows.
        for triple, n in sorted(ok_calls.items()):
            if n > 1:
                rows = rows_by_triple.get(triple, [])
                add("duplicate_ok_send", f"the transport returned ok {n} times for one (device, alert)",
                    profile_id=triple[0], device_id=triple[1], alert_id=triple[2],
                    row_id=rows[0].row_id if rows else None)
        delivered_rows = by_state[RowState.DELIVERED]
        ok_total = sum(ok_calls.values())
        if delivered_rows != ok_total:
            add("ok_call_count_mismatch", f"{delivered_rows} DELIVERED rows but {ok_total} ok sends")
        if attempts_total != self.store.total_marks:
            add("attempt_total_mismatch", f"{attempts_total} attempts recorded, the worker made {self.store.total_marks} marks")

        # Nothing sent to a gone device after it went gone.
        for i, (pid, did, aid) in enumerate(self.transport.calls):
            d = self.devices.get((pid, did))
            if d is not None and d.went_gone and i > d.gone_index:
                add("sent_after_gone", f"transport call {i} to a device reported gone at call {d.gone_index}",
                    profile_id=pid, device_id=did, alert_id=aid)
                break

        # Every expected triple ended the way its device allows.
        exhausted_revived = exhausted_stale = 0
        for triple in self.expected_order:
            pid, did, aid = triple
            rows = rows_by_triple.get(triple, [])
            d = self.devices[(pid, did)]
            if not rows:
                add("missing_row", "no outbox row exists for a (device, alert) that should have gone out",
                    profile_id=pid, device_id=did, alert_id=aid)
                continue
            if len(rows) > 1:
                add("duplicate_rows", f"{len(rows)} outbox rows for one (device, alert)",
                    profile_id=pid, device_id=did, alert_id=aid, row_id=rows[0].row_id)
            row = rows[0]
            state = row.state
            if d.permanent:
                if state is not RowState.DEAD:
                    add("permanent_not_dead", f"row for a permanently rejecting device is {state.value} "
                        f"after {row.attempts} attempt(s)", profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)
            elif d.exhaust and aid == d.exhaust_alert_id:
                revivals = self.store.revived.get(row.row_id, 0)
                requeues = self.requeued.get(row.row_id, 0)
                attempts = self.outbox.attempts_for(row.row_id)
                cause = "none" if row.dead_reason is None else row.dead_reason.value
                if (state is not RowState.DELIVERED or row.attempts != MAX_ATTEMPTS + 1
                        or revivals != 1 or requeues != 0):
                    add("exhaustion_mismatch", f"the exhausting row should be delivered after exactly "
                        f"{MAX_ATTEMPTS + 1} attempts, revived once by the outbox {EXHAUSTED_RETRY_COOLDOWN_S:.0f}s "
                        f"after its budget ran out; it is {state.value} after {row.attempts}, revived {revivals} "
                        f"time(s), re-queued {requeues} time(s), last reason {row.last_reason!r}, dead_reason {cause}",
                        profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)
                elif (len(attempts) != MAX_ATTEMPTS + 1 or any(a.ok or not a.retryable for a in attempts[:MAX_ATTEMPTS])
                      or not attempts[-1].ok):
                    add("exhaustion_mismatch", f"the exhausting row's history should be {MAX_ATTEMPTS} transient "
                        f"failures then one ok send; {len(attempts)} attempts are recorded",
                        profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)
                elif attempts[MAX_ATTEMPTS].at < attempts[MAX_ATTEMPTS - 1].at + EXHAUSTED_RETRY_COOLDOWN_S:
                    add("exhaustion_mismatch", f"the exhausting row was sent again "
                        f"{attempts[MAX_ATTEMPTS].at - attempts[MAX_ATTEMPTS - 1].at:.0f}s after its budget ran out, "
                        f"before the {EXHAUSTED_RETRY_COOLDOWN_S:.0f}s cooldown",
                        profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)
                else:
                    exhausted_revived += 1
            elif d.stale and aid == d.stale_alert_id:
                revivals = self.store.revived.get(row.row_id, 0)
                age = self.clock() - created_at[aid]
                cause = "none" if row.dead_reason is None else row.dead_reason.value
                if (state is not RowState.DEAD or row.attempts != MAX_ATTEMPTS or revivals
                        or row.dead_reason is not DeadReason.EXHAUSTED or row.last_reason != _EXHAUST.reason):
                    add("stale_exhaustion_mismatch", f"the stale row (alert {age:.0f}s old, past "
                        f"{ALERT_MAX_AGE_S:.0f}s) should be dead as exhausted after exactly {MAX_ATTEMPTS} attempts "
                        f"and never revived; it is {state.value} after {row.attempts}, revived {revivals} time(s), "
                        f"last reason {row.last_reason!r}, dead_reason {cause}",
                        profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)
                else:
                    exhausted_stale += 1
            elif d.went_gone:
                gone_at = d.gone_at or 0.0
                if state is RowState.DELIVERED:
                    last = self.outbox.attempts_for(row.row_id)[-1:]
                    if last and last[0].at > gone_at:
                        add("delivered_after_gone", f"delivered at {last[0].at - T0:.0f}s, device gone at {gone_at - T0:.0f}s",
                            profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)
                elif state is RowState.DEAD:
                    last = self.outbox.attempts_for(row.row_id)[-1:]
                    explained = bool(last) and (last[0].gone or last[0].reason == _NO_SUBSCRIPTION_REASON) \
                        and last[0].at >= gone_at
                    if not explained:
                        add("dead_letter_unexplained", f"DEAD on a gone device but not by the gone result: "
                            f"reason {row.last_reason!r}", profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)
                # PENDING (parked) and LEASED were judged above.
            elif state is not RowState.DELIVERED:
                add("not_delivered", f"row is {state.value} after {row.attempts} attempt(s), last reason {row.last_reason!r}",
                    profile_id=pid, device_id=did, alert_id=aid, row_id=row.row_id)

        # No row outside the ledger.
        for triple in sorted(rows_by_triple, key=lambda t: rows_by_triple[t][0].row_id):
            if triple not in self.expected:
                row = rows_by_triple[triple][0]
                d = self.devices.get((triple[0], triple[1]))
                why = "unknown device" if d is None else (
                    "device was gone" if d.went_gone and created_at[triple[2]] > (d.gone_at or 0.0)
                    else "device was not registered, or the alert is outside its backfill window")
                add("unexpected_row", f"a {row.state.value} row exists for a (device, alert) that should not: {why}",
                    profile_id=triple[0], device_id=triple[1], alert_id=triple[2], row_id=row.row_id)

        # Dead letters only on permanent, gone, exhausting or stale devices
        # (an exhausting device's row is expected back, so a DEAD one there
        # is the exhaustion_mismatch above; here it is not a second finding).
        for row in all_rows:
            if row.state is RowState.DEAD:
                d = self.devices.get((row.profile_id, row.device_id))
                if d is None or not (d.permanent or d.went_gone or row.alert_id == d.exhaust_alert_id
                                     or row.alert_id == d.stale_alert_id):
                    add("unexpected_dead_letter", f"DEAD on an ordinary device after {row.attempts} attempt(s), "
                        f"reason {row.last_reason!r}", profile_id=row.profile_id, device_id=row.device_id,
                        alert_id=row.alert_id, row_id=row.row_id)

        # The store's own counts agree with the enumeration, and its
        # revival count with what the worker was told and the observer saw.
        stats = self.outbox.stats(dead_limit=0)
        for state in RowState:
            if int(stats[state.value]) != by_state[state]:
                add("stats_mismatch", f"stats say {stats[state.value]} {state.value} rows, enumeration found {by_state[state]}")
        dead_exhausted = sum(1 for r in all_rows if r.state is RowState.DEAD and r.dead_reason is DeadReason.EXHAUSTED)
        if int(stats.get("dead_exhausted", -1)) != dead_exhausted:
            add("stats_mismatch", f"stats say {stats.get('dead_exhausted')} dead_exhausted rows, enumeration found {dead_exhausted}")
        if int(stats.get("dead_exhausted", 0)) + int(stats.get("dead_permanent", 0)) != int(stats["dead"]):
            add("stats_mismatch", f"stats say dead_exhausted {stats.get('dead_exhausted')} + dead_permanent "
                f"{stats.get('dead_permanent')} != dead {stats['dead']}")
        observed_revived = sum(self.store.revived.values())
        if int(stats.get("revived", -1)) != observed_revived or self.worker_total.revived != observed_revived:
            add("stats_mismatch", f"stats say {stats.get('revived')} revivals, the worker reported "
                f"{self.worker_total.revived}, the observer saw {observed_revived}")

        # The run itself: reported last, after the rows that explain it.
        if not self.converged:
            add("not_converged", f"rows were still outstanding after {self.rounds} rounds "
                f"({self.clock() - T0:.0f}s of sim time)")

        results = Counter()
        for r in self.script.results:
            results["ok" if r.ok else "gone" if r.gone else "transient" if r.retryable else "permanent"] += 1
        counts = {
            "profiles": self.n_profiles,
            "devices": len(self.devices),
            "devices_late": sum(1 for d in self.devices.values() if d.reg_wave > 0),
            "devices_permanent": sum(1 for d in self.devices.values() if d.permanent),
            "devices_gone_scripted": sum(1 for d in self.devices.values() if d.gone_scripted),
            "devices_stalled": sum(1 for d in self.devices.values() if d.stalled),
            "devices_went_gone": sum(1 for d in self.devices.values() if d.went_gone),
            "devices_exhaust": sum(1 for d in self.devices.values() if d.exhaust),
            "devices_stale": sum(1 for d in self.devices.values() if d.stale),
            "rows_exhausted": exhausted_revived + exhausted_stale,
            "rows_exhausted_revived": exhausted_revived,
            "rows_exhausted_stale": exhausted_stale,
            "rows_revived": sum(self.store.revived.values()),
            "publishes": self.publishes,
            "publishes_prelude": self.prelude_publishes,
            "probes": self.probes,
            "alerts_stored": len(self.stored_alerts),
            "alerts_deduped": len(self.deduped),
            "pairs_expected": len(self.expected),
            "rows": len(all_rows),
            "rows_backfilled": self.rows_backfilled,
            "rows_repaired": self.rows_repaired,
            "rows_requeued": self.rows_requeued,
            "rows_delivered": by_state[RowState.DELIVERED],
            "rows_dead": by_state[RowState.DEAD],
            "rows_pending_parked": parked,
            "rows_pending_live": by_state[RowState.PENDING] - parked,
            "rows_leased": by_state[RowState.LEASED],
            "attempts": attempts_total,
            "marks": self.store.total_marks,
            "releases": self.store.releases,
            "transport_calls": len(self.transport.calls),
            "transport_ok": results["ok"],
            "transport_transient": results["transient"],
            "transport_gone": results["gone"],
            "transport_permanent": results["permanent"],
            "worker_leased": self.worker_total.leased,
            "worker_delivered": self.worker_total.delivered,
            "worker_retried": self.worker_total.retried,
            "worker_dead": self.worker_total.dead,
            "worker_pruned": self.worker_total.pruned,
            "worker_errors": self.worker_total.errors,
            "worker_expired": self.worker_total.expired,
            "worker_revived": self.worker_total.revived,
            "crashes": self.crashes,
            "rows_abandoned": self.rows_abandoned,
            "stalls": self.stalls,
            "rows_stalled": self.rows_stalled,
            "rounds": self.rounds,
            "sim_seconds": int(self.clock() - T0),
            "converged": int(self.converged),
            "problems": len(self.problems),
        }
        return counts


def _dead_reason_of(last: Any) -> DeadReason:
    """The reason contracts.py assigns a row whose last attempt is ``last``:
    the reading the outbox must have made when it dead-lettered it."""
    if last.gone:
        return DeadReason.GONE
    if last.retryable:
        return DeadReason.EXHAUSTED
    if last.reason == _NO_SUBSCRIPTION_REASON:
        return DeadReason.NO_SUBSCRIPTION
    if last.reason == "no transport":
        return DeadReason.NO_TRANSPORT
    return DeadReason.PERMANENT


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def run_gate(
    n_profiles: int,
    n_alerts: int,
    seed: int,
    crash_every: int = 7,
    inject_defect: Optional[str] = None,
    *,
    db_path: Optional[str] = None,
) -> GateReport:
    """Run the delivery gate and report.  See the module docstring.

    ``n_profiles`` profiles get 1 to 3 devices each and ``n_alerts``
    publishes each; ``seed`` makes the run reproducible; one round in
    every ``crash_every`` is a simulated worker crash, at a seeded
    position in each window (0 or ``None`` disables crashes; 1 is refused
    because a worker that crashes on every round can never deliver
    anything).  ``inject_defect`` is one of
    :data:`DEFECTS` or ``None``.  ``db_path`` lets the outbox live in a
    file instead of ``":memory:"``; the logic is the same, only slower.

    Never raises on a failed assertion: those are the report's
    ``problems``.  Raises ``ValueError`` on bad arguments.
    """
    if n_profiles < 1:
        raise ValueError("n_profiles must be at least 1")
    if n_alerts < 1:
        raise ValueError("n_alerts must be at least 1")
    crash_every = int(crash_every or 0)
    if crash_every == 1:
        raise ValueError("crash_every=1 would crash the worker on every round; use 0 to disable crashes")
    if crash_every < 0:
        raise ValueError("crash_every must be 0 or at least 2")
    if inject_defect is not None and inject_defect not in DEFECTS:
        raise ValueError(f"inject_defect must be one of {DEFECTS} or None, not {inject_defect!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an int")
    return _Gate(n_profiles, n_alerts, seed, crash_every, inject_defect, db_path).run()


def main(argv: Optional[Sequence[str]] = None) -> int:
    """``python3 -m jarvis_alerts.validate``: print a report; exit 0 iff ok.

    ``--show-defects`` also runs each injected defect and prints its first
    finding, which is how the gate is shown to fail before it passes; a
    defect the gate does not catch makes the exit status 1 as well, so a
    CI wired to the status cannot pass on a gate that has gone blind.
    """
    parser = argparse.ArgumentParser(prog="python3 -m jarvis_alerts.validate", description=__doc__.split("\n\n")[0])
    parser.add_argument("--profiles", type=int, default=50)
    parser.add_argument("--alerts", type=int, default=20)
    parser.add_argument("--seed", type=lambda s: int(s, 0), default=0x5EED_A1E7_0000_0001,
                        help="an int, decimal or 0x-hex (default 0x5EEDA1E70000000001)")
    parser.add_argument("--crash-every", type=int, default=7)
    parser.add_argument("--defect", choices=DEFECTS, default=None)
    parser.add_argument("--show-defects", action="store_true", help="also run every defect and show it is caught")
    parser.add_argument("--json", action="store_true", help="print the report as JSON instead of a summary")
    args = parser.parse_args(argv)

    report = run_gate(args.profiles, args.alerts, args.seed, args.crash_every, args.defect)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.summary())
    uncaught = 0
    if args.show_defects:
        for defect in DEFECTS:
            r = run_gate(args.profiles, args.alerts, args.seed, args.crash_every, defect)
            first = r.first_failure
            uncaught += int(r.ok)
            print(f"defect {defect}: {'caught' if not r.ok else 'NOT CAUGHT'}; "
                  f"{len(r.problems)} problem(s); first: {first}")
    return 0 if report.ok and not uncaught else 1


if __name__ == "__main__":
    raise SystemExit(main())
