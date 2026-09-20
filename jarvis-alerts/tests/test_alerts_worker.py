"""Tests for the alert delivery worker.

Design: jarvis_alerts/contracts.py (durable outbox with leases, injected
transport, at-least-once with recorded attempts) and the retry policy at
the bottom of that file.

Every behavioural test runs against both stores through the ``store``
fixture: ``jarvis_alerts.worker.MemoryStore`` always, and the sqlite
``jarvis_alerts.outbox.Outbox`` when that module is importable (it is a
sibling written separately, so the import is guarded). The transport is a
local scripted ``FakeTransport``; nothing here depends on transports.py.

The clock is a ``SimClock`` and the jitter is a constant or a
``lucifer_gen.seed.Stream``; no test sleeps except the ``run_forever`` ones,
and those wait on a thread join. The stores are built with a jitter source
that *raises*, which proves the worker supplies its own jitter to every
backoff instead of letting the store draw.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Dict, Iterator, List, Tuple, Union

# Runnable as `pytest tests/test_alerts_worker.py` or
# `python3 tests/test_alerts_worker.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_alerts.contracts import (
    ALERT_MAX_AGE_S,
    BACKOFF_CAP_S,
    EXHAUSTED_RETRY_COOLDOWN_S,
    LEASE_S,
    MAX_ATTEMPTS,
    PRUNE_AFTER_FAILURES,
    PRUNE_COOLDOWN_S,
    Alert,
    DeadReason,
    Priority,
    RowState,
    SendResult,
    Subscription,
    backoff_seconds,
)
from jarvis_alerts.worker import MemoryStore, RunReport, SimClock, Worker, drain
from lucifer_gen.seed import SeedFields

try:
    from jarvis_alerts.outbox import Outbox
except ImportError:  # sibling not written yet, or broken: test the worker anyway
    Outbox = None  # type: ignore[assignment,misc]

T0 = 1_700_000_000.0
BLOB = '{"endpoint":"https://push.example/SECRET-ENDPOINT-TOKEN","keys":{"auth":"AUTHSECRET"}}'
PID = "owner"

Scripted = Union[SendResult, BaseException]


class FakeTransport:
    """A transport driven by a per-device script.

    ``script[device_id]`` is a list of SendResults or exceptions handed out
    in order; once the list is used up every send succeeds. Calls are
    recorded as (profile_id, device_id, alert_id): the blob is never kept.
    """

    name = "fake"

    def __init__(self, script: Dict[str, List[Scripted]] | None = None) -> None:
        self.script: Dict[str, List[Scripted]] = {k: list(v) for k, v in (script or {}).items()}
        self.calls: List[Tuple[str, str, str]] = []

    def send(self, subscription: Subscription, alert: Alert) -> SendResult:
        self.calls.append((subscription.profile_id, subscription.device_id, alert.id))
        queue = self.script.get(subscription.device_id)
        if queue:
            step = queue.pop(0)
            if isinstance(step, BaseException):
                raise step
            return step
        return SendResult(ok=True)

    def calls_for(self, device_id: str) -> int:
        return sum(1 for _, d, _ in self.calls if d == device_id)


RETRY = SendResult(ok=False, retryable=True, reason="503")
GONE = SendResult(ok=False, gone=True, reason="410")
PERMANENT = SendResult(ok=False, retryable=False, reason="400 bad payload")


def subscription(device: str, transport: str = "fake") -> Subscription:
    return Subscription(PID, device, transport, blob=BLOB, created_at=T0)


def alert(alert_id: str = "a1", priority: Priority = Priority.NORMAL,
          created_at: float = T0) -> Alert:
    return Alert(alert_id, PID, "render_done", "Render finished", "crypt.png is ready", created_at,
                 priority=priority)


def old_alert(alert_id: str = "a1") -> Alert:
    """An alert already ALERT_MAX_AGE_S old, so a row of it that spends its
    retry budget is DEAD for good: ``revive_exhausted`` never touches it
    (contracts.py, retry policy).  Tests that want the policy's end without
    the automatic second budget publish this one."""
    return alert(alert_id, created_at=T0 - ALERT_MAX_AGE_S)


def zero_jitter() -> float:
    return 0.0


def store_must_not_draw() -> float:
    raise AssertionError("the store drew jitter; the worker must supply it")


def make_worker(store, transport: FakeTransport, clock: SimClock, **kw) -> Worker:
    return Worker(store, {"fake": transport}, clock, zero_jitter, **kw)


def check_invariant(report: RunReport) -> None:
    assert report.leased == report.delivered + report.retried + report.dead + report.errors


def states(store, alert_id: str = "a1") -> Dict[str, RowState]:
    return {r.device_id: r.state for r in store.rows_for(alert_id)}


BACKENDS = ["memory"] + (["sqlite"] if Outbox is not None else [])


@pytest.fixture
def clock() -> SimClock:
    return SimClock(T0)


@pytest.fixture(params=BACKENDS)
def store(request: pytest.FixtureRequest, clock: SimClock) -> Iterator[object]:
    if request.param == "memory":
        yield MemoryStore(clock, jitter=store_must_not_draw)
        return
    outbox = Outbox(":memory:", clock=clock, jitter=store_must_not_draw)
    try:
        yield outbox
    finally:
        outbox.close()


def registered(store, *devices: str) -> None:
    for d in devices:
        store.register(subscription(d))


# ---------------------------------------------------------------------------
# One pass.
# ---------------------------------------------------------------------------


def test_every_due_row_is_attempted_exactly_once_per_run(store, clock: SimClock) -> None:
    devices = [f"dev{i}" for i in range(5)]
    registered(store, *devices)
    assert store.publish(alert()) == 5
    transport = FakeTransport()

    report = make_worker(store, transport, clock).run_once()

    assert report == RunReport(leased=5, delivered=5, retried=0, dead=0, pruned=0)
    check_invariant(report)
    assert sorted(d for _, d, _ in transport.calls) == devices
    assert set(states(store).values()) == {RowState.DELIVERED}
    # Nothing is due any more: a second pass finds nothing and sends nothing.
    assert make_worker(store, transport, clock).run_once().leased == 0
    assert len(transport.calls) == 5
    # Every delivery attempt was recorded (contracts.py, point 3).
    for row in store.rows_for("a1"):
        (attempt,) = store.attempts_for(row.row_id)
        assert attempt.ok and attempt.attempt == 1 and attempt.at == T0


def test_batch_limits_rows_per_pass(store, clock: SimClock) -> None:
    registered(store, *(f"dev{i}" for i in range(5)))
    store.publish(alert())
    worker = make_worker(store, FakeTransport(), clock, batch=2)

    assert [worker.run_once().leased for _ in range(4)] == [2, 2, 1, 0]


def test_high_priority_rows_go_first(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert("normal"))
    store.publish(alert("high", priority=Priority.HIGH))
    transport = FakeTransport()

    make_worker(store, transport, clock, batch=1).run_once()

    assert [a for _, _, a in transport.calls] == ["high"]


def test_retryable_failure_backs_off_then_succeeds(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())
    transport = FakeTransport({"phone": [RETRY]})
    worker = Worker(store, {"fake": transport}, clock, lambda: 0.5)

    first = worker.run_once()
    expected_delay = backoff_seconds(1, 0.5)
    assert first == RunReport(leased=1, delivered=0, retried=1, dead=0, pruned=0,
                              next_due=T0 + expected_delay)
    (row,) = store.rows_for("a1")
    assert row.state is RowState.PENDING
    assert row.next_due == T0 + expected_delay
    assert row.attempts == 1
    assert row.last_reason == "503"
    assert store.subscription(PID, "phone").failures == 1

    # Not due yet: nothing happens, however often we ask.
    clock.advance(expected_delay - 0.001)
    assert worker.run_once().leased == 0
    assert transport.calls_for("phone") == 1

    # Due: attempt 2 succeeds, the failure count resets.
    clock.advance(0.001)
    second = worker.run_once()
    assert second == RunReport(leased=1, delivered=1, retried=0, dead=0, pruned=0)
    (row,) = store.rows_for("a1")
    assert row.state is RowState.DELIVERED and row.attempts == 2
    assert [(a.attempt, a.ok) for a in store.attempts_for(row.row_id)] == [(1, False), (2, True)]
    assert store.subscription(PID, "phone").failures == 0


def test_jitter_is_drawn_once_per_backoff_and_never_otherwise(store, clock: SimClock) -> None:
    registered(store, "ok", "gone", "perm", "retry", "boom")
    store.publish(alert())
    transport = FakeTransport({
        "gone": [GONE], "perm": [PERMANENT], "retry": [RETRY], "boom": [RuntimeError("x")],
    })
    draws: List[int] = []

    def counting_jitter() -> float:
        draws.append(1)
        return 0.0

    report = Worker(store, {"fake": transport}, clock, counting_jitter).run_once()

    assert report == RunReport(leased=5, delivered=1, retried=2, dead=2, pruned=1,
                               next_due=T0 + backoff_seconds(1, 0.0))
    assert len(draws) == 2  # "retry" and "boom" backed off; nothing else drew


def test_final_failure_at_max_attempts_draws_no_jitter(store, clock: SimClock) -> None:
    """The death itself schedules nothing, so it draws no jitter.  The alert
    is one past ALERT_MAX_AGE_S: a younger one would be revived a cooldown
    later and spend a second budget, with MAX_ATTEMPTS - 1 more draws."""
    registered(store, "phone")
    store.publish(old_alert())
    draws: List[int] = []

    def counting_jitter() -> float:
        draws.append(1)
        return 0.0

    report = drain(store, {"fake": FakeTransport({"phone": [RETRY] * 100})}, clock,
                   counting_jitter, max_rounds=100)

    assert len(draws) == MAX_ATTEMPTS - 1
    assert report.revived == 0


def test_gone_prunes_subscription_and_dead_letters_the_rest(store, clock: SimClock) -> None:
    registered(store, "phone", "laptop")
    store.publish(alert("a1"))
    store.publish(alert("a2"))  # phone and laptop again: four rows
    transport = FakeTransport({"phone": [GONE]})
    worker = make_worker(store, transport, clock)

    # One pass leases all four. The first phone send says gone; the second
    # phone row then finds its subscription gone and is dead-lettered
    # without a send.
    report = worker.run_once()

    assert report == RunReport(leased=4, delivered=2, retried=0, dead=2, pruned=1)
    check_invariant(report)
    assert store.subscription(PID, "phone").gone is True
    assert transport.calls_for("phone") == 1
    phone_rows = [r for a in ("a1", "a2") for r in store.rows_for(a) if r.device_id == "phone"]
    assert sorted(r.last_reason for r in phone_rows) == ["410", "no subscription"]
    assert all(r.state is RowState.DEAD for r in phone_rows)


def test_raising_transport_does_not_kill_the_batch(store, clock: SimClock) -> None:
    registered(store, "a", "boom", "c")
    store.publish(alert())
    # The exception message quotes the blob, as a careless transport might.
    transport = FakeTransport({"boom": [RuntimeError(f"push failed for {BLOB}")]})

    report = make_worker(store, transport, clock).run_once()

    assert report == RunReport(leased=3, delivered=2, retried=1, dead=0, pruned=0,
                               next_due=T0 + backoff_seconds(1, 0.0))
    assert len(transport.calls) == 3
    (retry,) = [r for r in store.rows_for("a1") if r.state is RowState.PENDING]
    assert retry.device_id == "boom"
    assert retry.last_reason == "transport raised RuntimeError"
    assert retry.next_due == T0 + backoff_seconds(1, 0.0)
    # It is a retryable failure: the next pass, once due, succeeds.
    clock.advance(backoff_seconds(1, 0.0))
    assert make_worker(store, transport, clock).run_once().delivered == 1


def test_transport_returning_garbage_is_a_retryable_failure(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())

    class Liar:
        name = "fake"

        def send(self, subscription, alert):
            return "sure"

    report = Worker(store, {"fake": Liar()}, clock, zero_jitter).run_once()

    assert report.retried == 1
    assert store.rows_for("a1")[0].last_reason == "transport returned str"


def test_subscription_vanishing_between_lease_and_lookup_dead_letters(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())
    transport = FakeTransport()
    store.subscription = lambda pid, did: None  # unregistered a moment after the lease

    report = make_worker(store, transport, clock).run_once()

    assert report == RunReport(leased=1, delivered=0, retried=0, dead=1, pruned=0)
    (row,) = store.rows_for("a1")
    assert row.state is RowState.DEAD and row.last_reason == "no subscription"
    assert transport.calls == []
    (attempt,) = store.attempts_for(row.row_id)
    assert (attempt.ok, attempt.retryable, attempt.gone) == (False, False, False)


def test_unregistered_device_rows_are_parked_not_sent(store, clock: SimClock) -> None:
    registered(store, "phone", "laptop")
    store.publish(alert())
    assert store.unregister(PID, "phone") is True
    transport = FakeTransport()

    report = make_worker(store, transport, clock).run_once()

    assert report == RunReport(leased=1, delivered=1, retried=0, dead=0, pruned=0)
    assert states(store) == {"phone": RowState.PENDING, "laptop": RowState.DELIVERED}
    assert store.stats()["pending_unreachable"] == 1
    assert transport.calls_for("phone") == 0


def test_unknown_transport_dead_letters(store, clock: SimClock) -> None:
    store.register(subscription("phone", transport="carrier-pigeon"))
    store.publish(alert())

    report = make_worker(store, FakeTransport(), clock).run_once()

    assert report == RunReport(leased=1, delivered=0, retried=0, dead=1, pruned=0)
    assert store.rows_for("a1")[0].last_reason == "no transport"


def test_missing_alert_dead_letters(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())
    store.alert = lambda alert_id: None
    transport = FakeTransport()

    report = make_worker(store, transport, clock).run_once()

    assert report.dead == 1 and store.rows_for("a1")[0].last_reason == "no alert"
    assert transport.calls == []


def test_permanent_failure_dead_letters_with_transport_reason(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())

    report = make_worker(store, FakeTransport({"phone": [PERMANENT]}), clock).run_once()

    assert report == RunReport(leased=1, delivered=0, retried=0, dead=1, pruned=0)
    assert store.rows_for("a1")[0].last_reason == "400 bad payload"
    sub = store.subscription(PID, "phone")
    assert sub.gone is False and sub.failures == 0


def test_max_attempts_dead_letters(store, clock: SimClock) -> None:
    """MAX_ATTEMPTS transient failures in one budget kill the row as
    EXHAUSTED.  The alert is past ALERT_MAX_AGE_S, which is the case where
    that death is final: nothing is revivable afterwards, so the drain
    converges instead of waiting out the cooldown and trying again."""
    registered(store, "phone")
    store.publish(old_alert())
    transport = FakeTransport({"phone": [RETRY] * 100})

    report = drain(store, {"fake": transport}, clock, zero_jitter, max_rounds=100)

    assert report == RunReport(leased=MAX_ATTEMPTS, delivered=0, retried=MAX_ATTEMPTS - 1,
                               dead=1, pruned=0, rounds=MAX_ATTEMPTS, revived=0)
    (row,) = store.rows_for("a1")
    assert row.state is RowState.DEAD and row.attempts == MAX_ATTEMPTS
    assert row.last_reason == "503" and row.dead_reason is DeadReason.EXHAUSTED
    assert [a.attempt for a in store.attempts_for(row.row_id)] == list(range(1, MAX_ATTEMPTS + 1))
    assert transport.calls_for("phone") == MAX_ATTEMPTS
    assert clock() == T0 + sum(backoff_seconds(n, 0.0) for n in range(1, MAX_ATTEMPTS))
    stats = store.stats()
    assert (stats["dead_exhausted"], stats["dead_revivable"], stats["revived"]) == (1, 0, 0)


def test_exhausted_row_is_revived_after_the_cooldown_and_reported(store, clock: SimClock) -> None:
    """The worker calls ``revive_exhausted`` at the start of every pass and
    reports the count.  An outage that outlasts the retry budget therefore
    costs a cooldown, not the alert: the row dies EXHAUSTED, the pass
    EXHAUSTED_RETRY_COOLDOWN_S later puts it back and leases it in the same
    breath, and the send that follows succeeds.  ``revived`` is not part of
    ``leased``'s sum."""
    registered(store, "phone")
    store.publish(alert())
    transport = FakeTransport({"phone": [RETRY] * MAX_ATTEMPTS})   # the outage clears afterwards
    worker = make_worker(store, transport, clock)

    for n in range(1, MAX_ATTEMPTS):
        assert worker.run_once() == RunReport(leased=1, delivered=0, retried=1, dead=0, pruned=0,
                                              next_due=clock() + backoff_seconds(n, 0.0), revived=0)
        clock.advance(backoff_seconds(n, 0.0))
    died_at = clock()
    # The death schedules nothing itself; what it promises is the cooldown.
    assert worker.run_once() == RunReport(leased=1, delivered=0, retried=0, dead=1, pruned=0,
                                          next_due=died_at + EXHAUSTED_RETRY_COOLDOWN_S, revived=0)
    (row,) = store.rows_for("a1")
    assert row.state is RowState.DEAD and row.dead_reason is DeadReason.EXHAUSTED

    # A pass one second early revives nothing and leases nothing.
    clock.advance(EXHAUSTED_RETRY_COOLDOWN_S - 1.0)
    empty = worker.run_once()
    assert empty == RunReport(leased=0, delivered=0, retried=0, dead=0, pruned=0, revived=0)
    assert store.rows_for("a1")[0].state is RowState.DEAD
    assert transport.calls_for("phone") == MAX_ATTEMPTS

    clock.advance(1.0)
    report = worker.run_once()
    assert report == RunReport(leased=1, delivered=1, retried=0, dead=0, pruned=0, revived=1)
    check_invariant(report)
    assert states(store) == {"phone": RowState.DELIVERED}
    assert transport.calls_for("phone") == MAX_ATTEMPTS + 1
    # The whole history is still there, under one contiguous run of numbers.
    assert [a.attempt for a in store.attempts_for(row.row_id)] == list(range(1, MAX_ATTEMPTS + 2))
    stats = store.stats()
    assert (stats["dead"], stats["revived"], stats["delivered"]) == (0, 1, 1)
    assert worker.run_once().revived == 0                       # nothing left to revive


def test_consecutive_failures_prune_the_subscription(store, clock: SimClock) -> None:
    """Three alerts for one device that always fails. Failures accumulate
    per device across rows, so the 20th failure lands on the second row in
    round 7; the third row then finds the device pruned and is *released*
    -- handed back untouched, no attempt, not dead-lettered -- because a
    prune is a cooldown, not a verdict on the endpoint.  After the
    cooldown all three rows go out."""
    registered(store, "phone")
    for i in range(3):
        store.publish(alert(f"a{i}"))
    transport = FakeTransport({"phone": [RETRY] * 100})

    report = drain(store, {"fake": transport}, clock, zero_jitter, max_rounds=100)

    rounds = (PRUNE_AFTER_FAILURES + 2) // 3  # 7
    assert report == RunReport(leased=3 * rounds, delivered=0, retried=PRUNE_AFTER_FAILURES + 1,
                               dead=0, pruned=1, rounds=rounds)
    check_invariant(report)
    sub = store.subscription(PID, "phone")
    assert sub.gone is True and sub.pruned is True and sub.failures == PRUNE_AFTER_FAILURES
    assert states(store, "a0") == {"phone": RowState.PENDING}   # parked until the cooldown
    assert states(store, "a1") == {"phone": RowState.PENDING}
    assert states(store, "a2") == {"phone": RowState.PENDING}
    (released,) = store.rows_for("a2")
    assert released.last_reason == "device pruned" and released.attempts == 6
    assert store.attempts_for(released.row_id)[-1].retryable      # the release itself recorded nothing
    assert store.stats()["pending_unreachable"] == 3
    assert transport.calls_for("phone") == PRUNE_AFTER_FAILURES

    # The push service recovers; after the cooldown the device is probed again
    # and every parked row, the released one included, is delivered.
    transport.script["phone"] = []
    clock.advance(PRUNE_COOLDOWN_S)
    report = make_worker(store, transport, clock).run_once()
    assert report == RunReport(leased=3, delivered=3, retried=0, dead=0, pruned=0)
    assert store.subscription(PID, "phone").failures == 0 and not store.subscription(PID, "phone").gone


def test_store_error_while_marking_leaves_row_leased_and_batch_alive(store, clock: SimClock) -> None:
    registered(store, "a", "b", "c")
    store.publish(alert())
    transport = FakeTransport()
    real_mark = store.mark

    def flaky_mark(row_id, result, now, jitter=None):
        if store.row(row_id).device_id == "b":
            raise OSError("disk full")
        return real_mark(row_id, result, now, jitter)

    store.mark = flaky_mark
    worker = make_worker(store, transport, clock, lease_s=10.0)

    report = worker.run_once()

    assert report == RunReport(leased=3, delivered=2, retried=0, dead=0, pruned=0, errors=1,
                               next_due=T0 + 10.0)
    check_invariant(report)
    assert states(store)["b"] is RowState.LEASED
    assert worker.unmarked == 1
    # The send happened; the worker remembers its result and reports it at
    # the start of its next pass once the store works again, instead of
    # letting the lease expire and sending the alert a second time.
    store.mark = real_mark
    assert worker.run_once().leased == 0
    assert worker.unmarked == 0
    assert states(store)["b"] is RowState.DELIVERED
    assert transport.calls_for("b") == 1
    (row_b,) = [r for r in store.rows_for("a1") if r.device_id == "b"]
    assert row_b.attempts == 1 and store.attempts_for(row_b.row_id)[0].ok
    clock.advance(10.0 + 1.0)
    assert worker.run_once() == RunReport(0, 0, 0, 0, 0)
    assert transport.calls_for("b") == 1


def test_remembered_result_is_applied_when_the_row_is_leased_again(store, clock: SimClock) -> None:
    """The mark keeps failing for a while; the lease expires and the row
    comes back to this worker, which reports the old send instead of
    sending again."""
    registered(store, "phone")
    store.publish(alert())
    transport = FakeTransport()
    real_mark = store.mark
    store.mark = lambda *a, **k: (_ for _ in ()).throw(OSError("database is locked"))
    worker = make_worker(store, transport, clock, lease_s=10.0)
    for _ in range(5):
        worker.run_once()
        clock.advance(11.0)
    assert transport.calls_for("phone") == 1              # one send, however often the mark failed
    assert worker.unmarked == 1
    store.mark = real_mark
    worker.run_once()
    assert states(store) == {"phone": RowState.DELIVERED}
    assert transport.calls_for("phone") == 1
    assert [a.ok for a in store.attempts_for(store.rows_for("a1")[0].row_id)] == [True]


def test_rows_are_not_sent_after_their_lease_expired(store, clock: SimClock) -> None:
    """A batch that outlives its lease: the rows whose turn comes after
    ``lease_until`` are skipped, not sent (another worker may hold them),
    and come back through the reclaim."""
    registered(store, *(f"dev{i}" for i in range(5)))
    store.publish(alert())
    transport = FakeTransport()
    slow_transport = FakeTransport()

    def slow_send(subscription, alert):
        clock.advance(4.0)                                 # each send takes 4 s of a 10 s lease
        return transport.send(subscription, alert)

    slow_transport.send = slow_send
    worker = make_worker(store, slow_transport, clock, lease_s=10.0)

    report = worker.run_once()

    assert report == RunReport(leased=5, delivered=3, retried=0, dead=0, pruned=0,
                               next_due=T0 + 10.0, expired=2)
    assert report.leased == report.delivered + report.retried + report.dead + report.errors + report.expired
    assert len(transport.calls) == 3
    assert sum(1 for s in states(store).values() if s is RowState.LEASED) == 2
    assert all(len(store.attempts_for(r.row_id)) == (1 if r.state is RowState.DELIVERED else 0)
               for r in store.rows_for("a1"))
    clock.advance(1.0)                                     # the leases are past; the two come back
    report = worker.run_once()
    assert report.leased == 2 and report.delivered == 2 and report.expired == 0
    assert len(transport.calls) == 5 and set(states(store).values()) == {RowState.DELIVERED}


def test_lease_failure_propagates(store, clock: SimClock) -> None:
    def broken_lease(now, limit, lease_s=LEASE_S):
        raise OSError("database is locked")

    store.lease = broken_lease
    with pytest.raises(OSError):
        make_worker(store, FakeTransport(), clock).run_once()


def test_constructor_rejects_nonsense(clock: SimClock) -> None:
    store = MemoryStore(clock)
    with pytest.raises(ValueError):
        make_worker(store, FakeTransport(), clock, batch=0)
    with pytest.raises(ValueError):
        make_worker(store, FakeTransport(), clock, lease_s=0)


# ---------------------------------------------------------------------------
# Privacy: the blob never leaves the subscription.
# ---------------------------------------------------------------------------


def test_blob_never_appears_in_reasons_attempts_or_logs(
    store, clock: SimClock, caplog: pytest.LogCaptureFixture
) -> None:
    registered(store, "ok", "boom", "gone", "perm", "lost")
    store.publish(alert())
    store.unregister(PID, "lost")
    transport = FakeTransport({
        "boom": [ValueError(BLOB)],
        "gone": [GONE],
        "perm": [PERMANENT],
    })
    with caplog.at_level(logging.DEBUG, logger="jarvis_alerts.worker"):
        report = drain(store, {"fake": transport}, clock, zero_jitter, max_rounds=20)
    assert report.delivered == 2 and report.dead == 2

    secret_markers = ("SECRET-ENDPOINT-TOKEN", "AUTHSECRET", "push.example")
    haystacks = [r.last_reason for r in store.rows_for("a1")]
    haystacks += [a.reason for r in store.rows_for("a1") for a in store.attempts_for(r.row_id)]
    haystacks += [rec.getMessage() for rec in caplog.records]
    haystacks += [repr(transport.calls), repr(report), repr(store.stats())]
    for text in haystacks:
        for marker in secret_markers:
            assert marker not in text, "subscription blob leaked"
    assert any("raised ValueError" in rec.getMessage() for rec in caplog.records)


# ---------------------------------------------------------------------------
# The loop.
# ---------------------------------------------------------------------------


def test_run_forever_stops_within_200ms_of_stop(store, clock: SimClock) -> None:
    worker = make_worker(store, FakeTransport(), clock)
    stop = threading.Event()
    thread = threading.Thread(target=worker.run_forever, args=(stop,), kwargs={"idle_s": 30.0})
    thread.start()
    time.sleep(0.05)  # let it enter its idle wait
    assert thread.is_alive()

    started = time.monotonic()
    stop.set()
    thread.join(timeout=0.2)
    assert not thread.is_alive()
    assert time.monotonic() - started < 0.2


def test_run_forever_delivers_without_waiting_for_idle(store, clock: SimClock) -> None:
    registered(store, "dev0", "dev1", "dev2")
    store.publish(alert())
    transport = FakeTransport()
    worker = make_worker(store, transport, clock, batch=1)  # three passes, no idle between
    stop = threading.Event()
    thread = threading.Thread(target=worker.run_forever, args=(stop,), kwargs={"idle_s": 30.0})
    thread.start()
    deadline = time.monotonic() + 1.0
    while len(transport.calls) < 3 and time.monotonic() < deadline:
        time.sleep(0.005)
    stop.set()
    thread.join(timeout=0.2)
    assert not thread.is_alive()
    assert set(states(store).values()) == {RowState.DELIVERED}
    assert len(transport.calls) == 3


def test_run_forever_reports_store_errors_and_keeps_going(store, clock: SimClock) -> None:
    calls = {"n": 0}
    real_lease = store.lease

    def lease_fails_once(now, limit, lease_s=LEASE_S):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("database is locked")
        return real_lease(now, limit, lease_s)

    store.lease = lease_fails_once
    registered(store, "phone")
    store.publish(alert())
    errors: List[BaseException] = []
    transport = FakeTransport()
    worker = make_worker(store, transport, clock)
    stop = threading.Event()
    thread = threading.Thread(
        target=worker.run_forever, args=(stop,), kwargs={"idle_s": 0.01, "on_error": errors.append}
    )
    thread.start()
    deadline = time.monotonic() + 1.0
    while not transport.calls and time.monotonic() < deadline:
        time.sleep(0.005)
    stop.set()
    thread.join(timeout=0.2)
    assert not thread.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], OSError)
    assert states(store) == {"phone": RowState.DELIVERED}


def test_run_forever_without_on_error_raises(store, clock: SimClock) -> None:
    def broken_lease(now, limit, lease_s=LEASE_S):
        raise OSError("database is locked")

    store.lease = broken_lease
    with pytest.raises(OSError):
        make_worker(store, FakeTransport(), clock).run_forever(threading.Event(), idle_s=0.01)


# ---------------------------------------------------------------------------
# drain: the simulation helper.
# ---------------------------------------------------------------------------


def test_drain_converges_and_reports_exact_counts(store, clock: SimClock) -> None:
    """The "outage" device spends its whole budget and dies EXHAUSTED; the
    drain waits out EXHAUSTED_RETRY_COOLDOWN_S, the store revives the row
    and the send that follows succeeds.  The gone device's row is DEAD for
    another reason and is never revived, which is why this converges."""
    registered(store, "a", "b", "c", "flaky", "gone", "outage", "ghost")
    assert store.publish(alert()) == 7
    store.unregister(PID, "ghost")  # parked: never leased, never counted
    transport = FakeTransport({
        "flaky": [RETRY, RETRY],           # succeeds on attempt 3
        "gone": [GONE],
        "outage": [RETRY] * MAX_ATTEMPTS,  # dies after MAX_ATTEMPTS, then the outage clears
    })

    report = drain(store, {"fake": transport}, clock, zero_jitter, max_rounds=100)

    # a, b, c: 1 each; flaky: 3; gone: 1; outage: MAX_ATTEMPTS + 1 after its
    # revival.  ``dead`` counts what each pass *did*, so the outage row's
    # death is in it even though a later pass undid it; ``revived`` says so,
    # and the store's own ``dead`` below is the one row still DEAD.
    assert report == RunReport(
        leased=3 + 3 + 1 + MAX_ATTEMPTS + 1,
        delivered=5,
        retried=2 + (MAX_ATTEMPTS - 1),
        dead=2,
        pruned=1,
        rounds=MAX_ATTEMPTS + 1,  # one pass per attempt of the outage row, plus the revived one
        revived=1,
    )
    check_invariant(report)
    assert states(store) == {
        "a": RowState.DELIVERED, "b": RowState.DELIVERED, "c": RowState.DELIVERED,
        "flaky": RowState.DELIVERED, "gone": RowState.DEAD,
        "outage": RowState.DELIVERED, "ghost": RowState.PENDING,
    }
    stats = store.stats()
    assert (stats["leased"], stats["pending"], stats["pending_unreachable"]) == (0, 1, 1)
    assert (stats["dead"], stats["dead_exhausted"], stats["revived"]) == (1, 0, 1)
    assert len(transport.calls) == 3 + 3 + 1 + MAX_ATTEMPTS + 1
    # The revived send waited the cooldown, not a backoff.
    (row,) = [r for r in store.rows_for("a1") if r.device_id == "outage"]
    history = store.attempts_for(row.row_id)
    assert [a.attempt for a in history] == list(range(1, MAX_ATTEMPTS + 2))   # history kept
    assert history[-1].at == history[-2].at + EXHAUSTED_RETRY_COOLDOWN_S


def test_drain_advances_the_clock_by_exactly_the_backoff(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())
    failures = 3
    transport = FakeTransport({"phone": [RETRY] * failures})
    jitter_value = 0.25

    report = drain(store, {"fake": transport}, clock, lambda: jitter_value, max_rounds=100)

    assert report.delivered == 1 and report.retried == failures and report.rounds == failures + 1
    expected_elapsed = sum(backoff_seconds(n, jitter_value) for n in range(1, failures + 1))
    assert clock() == pytest.approx(T0 + expected_elapsed)
    # Attempts were made exactly at the successive due times.
    ats = [a.at for a in store.attempts_for(store.rows_for("a1")[0].row_id)]
    assert ats[0] == T0
    for n in range(1, failures + 1):
        assert ats[n] == pytest.approx(ats[n - 1] + backoff_seconds(n, jitter_value))


def test_drain_stops_at_max_rounds(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())

    report = drain(store, {"fake": FakeTransport({"phone": [RETRY] * 100})}, clock, zero_jitter, max_rounds=3)

    assert report.rounds == 3 and report.retried == 3 and report.dead == 0
    assert store.stats()["pending"] == 1  # did not converge, and the store says so


def test_drain_on_empty_store_is_one_idle_round(store, clock: SimClock) -> None:
    assert drain(store, {"fake": FakeTransport()}, clock, zero_jitter) == RunReport(0, 0, 0, 0, 0, rounds=1)
    assert clock() == T0


def test_drain_steps_past_retries_it_did_not_schedule(store, clock: SimClock) -> None:
    """A retry scheduled by an earlier worker is invisible to drain; it
    steps past the backoff cap rather than spinning or stopping early."""
    registered(store, "phone")
    store.publish(alert())
    transport = FakeTransport({"phone": [RETRY]})
    make_worker(store, transport, clock).run_once()  # backed off to T0 + 1

    report = drain(store, {"fake": transport}, clock, zero_jitter, max_rounds=10)

    assert report == RunReport(leased=1, delivered=1, retried=0, dead=0, pruned=0, rounds=2)
    assert clock() == T0 + max(BACKOFF_CAP_S, LEASE_S)


def test_drain_needs_an_advanceable_clock(clock: SimClock) -> None:
    with pytest.raises(TypeError):
        drain(MemoryStore(clock), {"fake": FakeTransport()}, lambda: T0, zero_jitter)  # type: ignore[arg-type]


def test_jitter_from_seed_stream_is_deterministic() -> None:
    def run(seed: int) -> List[float]:
        clock = SimClock(T0)
        store = MemoryStore(clock, jitter=store_must_not_draw)
        registered(store, "phone")
        store.publish(alert())
        jitter = SeedFields.parse(seed).stream("alerts.backoff").random
        drain(store, {"fake": FakeTransport({"phone": [RETRY] * 4})}, clock, jitter, max_rounds=50)
        return [a.at for a in store.attempts_for(1)]

    same = run(0xC0FFEE)
    assert same == run(0xC0FFEE)
    assert same != run(0xBEEF)
    # Every gap obeys the policy envelope: half to the full cap-limited delay.
    for n, (before, after) in enumerate(zip(same, same[1:]), start=1):
        assert backoff_seconds(n, 0.0) <= after - before <= backoff_seconds(n, 1.0)


# ---------------------------------------------------------------------------
# Crash recovery: leases are visibility timeouts, not ownership.
# ---------------------------------------------------------------------------


def test_abandoned_lease_is_recovered_after_it_expires(store, clock: SimClock) -> None:
    registered(store, "a", "b", "c")
    store.publish(alert())
    transport = FakeTransport()

    # Worker 1 leases everything and dies before sending or marking.
    crashed = store.lease(clock(), 50, LEASE_S)
    assert len(crashed) == 3 and all(r.state is RowState.LEASED for r in crashed)

    # Worker 2 comes up while the leases are live: it sees nothing.
    worker2 = make_worker(store, transport, clock)
    clock.advance(LEASE_S - 1.0)
    assert worker2.run_once().leased == 0
    assert transport.calls == []

    # Once the leases expire the rows are visible again and get delivered.
    clock.advance(2.0)
    report = worker2.run_once()
    assert report == RunReport(leased=3, delivered=3, retried=0, dead=0, pruned=0)
    assert len(transport.calls) == 3
    # An abandoned lease is not an attempt: the rows were delivered first time.
    assert all(r.attempts == 1 for r in store.rows_for("a1"))


def test_drain_recovers_abandoned_leases_by_stepping_past_them(store, clock: SimClock) -> None:
    registered(store, "phone")
    store.publish(alert())
    store.lease(clock(), 50, LEASE_S)  # crashed worker

    report = drain(store, {"fake": FakeTransport()}, clock, zero_jitter, max_rounds=10)

    assert report == RunReport(leased=1, delivered=1, retried=0, dead=0, pruned=0, rounds=2)
    assert states(store) == {"phone": RowState.DELIVERED}


def test_late_mark_from_a_crashed_worker_is_a_no_op(store, clock: SimClock) -> None:
    """A worker that comes back from the dead after its row was re-leased
    and delivered cannot overwrite the newer state (outbox semantics the
    worker relies on: mark on a settled row returns that state)."""
    registered(store, "phone")
    store.publish(alert())
    (stale,) = store.lease(clock(), 50, LEASE_S)
    clock.advance(LEASE_S + 1.0)
    assert make_worker(store, FakeTransport(), clock).run_once().delivered == 1

    late = store.mark(stale.row_id, SendResult(ok=False, retryable=False, reason="late"), clock())

    assert late is RowState.DELIVERED
    (row,) = store.rows_for("a1")
    assert row.state is RowState.DELIVERED and row.attempts == 1 and row.last_reason == ""


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
