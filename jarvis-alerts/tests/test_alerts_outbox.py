"""Tests for the durable outbox, part 1 of the design in jarvis_alerts/contracts.py.

Every test drives time through a hand-cranked clock and jitter through an
explicit number or a seeded ``lucifer_gen`` stream, so the retry schedule is
checked to the float.  The two-worker test uses real threads on one file,
once with two connections and once with a shared instance, and asserts the
1000 rows are handed out exactly once between them.  The privacy test seeds
every blob with a canary and greps it out of every row, stat, dead letter and
exception the module can produce.
"""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List

# Runnable as `pytest tests/test_alerts_outbox.py` or
# `python3 tests/test_alerts_outbox.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_alerts.contracts import (
    ALERT_MAX_AGE_S,
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
from jarvis_alerts.outbox import (
    DEDUPE_WINDOW_S,
    SCHEMA_VERSION,
    DuplicateAlert,
    Outbox,
    SchemaError,
    UnknownRow,
)
from lucifer_gen.seed import SeedFields

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

CANARY = "BLOB-CANARY-7f3a9c"

OK = SendResult(ok=True)
RETRY = SendResult(ok=False, retryable=True, reason="503")
GONE = SendResult(ok=False, gone=True, reason="410")
FATAL = SendResult(ok=False, reason="payload too large")


class FakeClock:
    """``clock()`` returns ``now``; tests move it by assignment."""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def sub(profile: str = "p", device: str = "phone", *, created_at: float = 1.0,
        transport: str = "fake", blob: str | None = None, **extra: Any) -> Subscription:
    return Subscription(
        profile_id=profile,
        device_id=device,
        transport=transport,
        blob=blob if blob is not None else json.dumps({"endpoint": f"https://push/{CANARY}/{device}"}),
        created_at=created_at,
        **extra,
    )


def alert(id: str, profile: str = "p", *, created_at: float = 1000.0,
          priority: Priority = Priority.NORMAL, dedupe_key: str | None = None,
          kind: str = "render_done", data: Dict[str, Any] | None = None) -> Alert:
    return Alert(
        id=id,
        profile_id=profile,
        kind=kind,
        title=f"title {id}",
        body=f"body {id}",
        created_at=created_at,
        priority=priority,
        dedupe_key=dedupe_key,
        data=data if data is not None else {"alert": id},
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock(1000.0)


@pytest.fixture
def path(tmp_path: Path) -> Path:
    return tmp_path / "outbox.sqlite"


@pytest.fixture
def outbox(path: Path, clock: FakeClock):
    box = Outbox(path, clock)
    yield box
    box.close()


def lease_one(box: Outbox, now: float):
    rows = box.lease(now=now, limit=1)
    assert len(rows) == 1, rows
    return rows[0]


def exhaust_one(box: Outbox, now: float, step: float = 1000.0) -> float:
    """Spend one whole retry budget on the one row that is due: MAX_ATTEMPTS
    transient failures, ``step`` apart (past every backoff and lease).
    Returns the instant of the death, which is the row's ``dead_at``."""
    for n in range(MAX_ATTEMPTS):
        if n:
            now += step
        row = lease_one(box, now)
        state = box.mark(row.row_id, RETRY, now=now, jitter=0.0)
    assert state is RowState.DEAD, state
    return now


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------


def test_migrate_sets_wal_and_foreign_keys_and_is_idempotent(outbox: Outbox, path: Path, clock: FakeClock) -> None:
    assert outbox._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert outbox._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    assert outbox.schema_version() == SCHEMA_VERSION == 1
    outbox.migrate()  # second run is a no-op
    tables = {r[0] for r in outbox._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"alerts", "subscriptions", "outbox", "attempts", "schema_version"} <= tables
    with Outbox(path, clock) as again:  # reopening an existing file works
        assert again.schema_version() == SCHEMA_VERSION
    assert not outbox._conn.in_transaction


def test_wrong_schema_version_raises(path: Path, clock: FakeClock) -> None:
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version (id, version) VALUES (1, 99)")
    raw.commit()
    raw.close()
    with pytest.raises(SchemaError, match="version 99"):
        Outbox(path, clock)


def test_timestamps_are_stored_as_real(outbox: Outbox) -> None:
    outbox.register(sub(created_at=5))          # an int in, a REAL stored
    outbox.publish(alert("a", created_at=7))
    assert outbox._conn.execute("SELECT typeof(created_at) FROM subscriptions").fetchone()[0] == "real"
    assert outbox._conn.execute("SELECT typeof(created_at) FROM alerts").fetchone()[0] == "real"
    assert outbox._conn.execute("SELECT typeof(next_due), typeof(lease_until) FROM outbox").fetchone() == ("real", "real")
    assert outbox.subscription("p", "phone").created_at == 5.0
    assert outbox.alert("a").created_at == 7.0


# --------------------------------------------------------------------------
# Subscriptions
# --------------------------------------------------------------------------


def test_register_upserts_and_reregistering_clears_gone_and_failures(outbox: Outbox, clock: FakeClock) -> None:
    outbox.register(sub(device="phone", blob="one", created_at=1.0))
    outbox.publish(alert("a"))
    row = lease_one(outbox, clock.now)
    outbox.mark(row.row_id, RETRY, now=clock.now)
    outbox.publish(alert("b"))
    row_b = lease_one(outbox, clock.now + 100)
    assert outbox.mark(row_b.row_id, GONE, now=clock.now + 100) is RowState.DEAD
    before = outbox.subscription("p", "phone")
    assert before is not None and before.gone and before.failures == 1
    assert outbox.subscriptions_for("p") == []

    # The stored copy of failures/gone on the argument is ignored: a
    # registration is the device saying it is reachable.
    outbox.register(sub(device="phone", blob="two", transport="webpush", created_at=9.0, failures=5, gone=True))
    after = outbox.subscription("p", "phone")
    assert after == Subscription("p", "phone", "webpush", "two", 9.0, failures=0, gone=False)
    assert outbox.subscriptions_for("p") == [after]
    assert outbox._conn.execute("SELECT count(*) FROM subscriptions").fetchone()[0] == 1


def test_subscriptions_for_excludes_gone_and_is_ordered_by_device(outbox: Outbox) -> None:
    for device in ("zed", "amy", "kim"):
        outbox.register(sub(device=device))
    outbox.register(sub(profile="other", device="phone"))
    assert [s.device_id for s in outbox.subscriptions_for("p")] == ["amy", "kim", "zed"]
    assert [s.device_id for s in outbox.subscriptions_for("other")] == ["phone"]
    assert outbox.subscriptions_for("nobody") == []


def test_unregister_returns_whether_it_existed(outbox: Outbox) -> None:
    outbox.register(sub())
    assert outbox.unregister("p", "phone") is True
    assert outbox.unregister("p", "phone") is False
    assert outbox.subscription("p", "phone") is None


@pytest.mark.parametrize(
    "bad",
    [
        dict(profile=""),
        dict(device=""),
        dict(transport=""),
        dict(created_at="soon"),
    ],
)
def test_register_rejects_malformed_subscriptions(outbox: Outbox, bad: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        outbox.register(sub(**bad))
    assert outbox._conn.execute("SELECT count(*) FROM subscriptions").fetchone()[0] == 0


# --------------------------------------------------------------------------
# Publish, dedupe, backfill
# --------------------------------------------------------------------------


def test_publish_fans_out_one_row_per_live_subscription(outbox: Outbox, clock: FakeClock) -> None:
    for device in ("phone", "laptop", "tablet", "watch"):
        outbox.register(sub(device=device))
    outbox.register(sub(profile="other", device="phone"))
    outbox.unregister("p", "watch")
    assert outbox.publish(alert("a0")) == 3
    tablet = next(r for r in outbox.lease(now=clock.now, limit=10) if r.device_id == "tablet")
    assert outbox.mark(tablet.row_id, GONE, now=clock.now) is RowState.DEAD

    clock.now = 2000.0
    assert outbox.publish(alert("a1", priority=Priority.HIGH)) == 2
    rows = outbox.rows_for("a1")
    assert [r.device_id for r in rows] == ["laptop", "phone"]
    for row in rows:
        assert row.alert_id == "a1" and row.profile_id == "p"
        assert row.state is RowState.PENDING
        assert row.attempts == 0
        assert row.next_due == 2000.0
        assert row.lease_until == 0.0
        assert row.last_reason == ""
    assert outbox.rows_for("nope") == []
    assert outbox.stats()["alerts"] == 2


def test_publish_with_no_subscriptions_still_stores_the_alert(outbox: Outbox) -> None:
    a = alert("lonely", profile="nobody", priority=Priority.HIGH, dedupe_key="k",
              data={"nested": {"x": [1, 2.5, None, "s"]}, "flag": True})
    assert outbox.publish(a) == 0
    assert outbox.alert("lonely") == a
    assert outbox.alert("missing") is None
    assert outbox.rows_for("lonely") == []


def test_publish_same_id_twice_raises(outbox: Outbox) -> None:
    outbox.register(sub())
    assert outbox.publish(alert("a")) == 1
    with pytest.raises(DuplicateAlert, match="'a'"):
        outbox.publish(alert("a", kind="different"))
    assert not outbox._conn.in_transaction
    assert outbox.alert("a").kind == "render_done"
    assert len(outbox.rows_for("a")) == 1


def test_publish_rejects_bad_alerts_before_writing(outbox: Outbox) -> None:
    outbox.register(sub())
    with pytest.raises(TypeError):
        outbox.publish(alert("bad-data", data={"when": object()}))
    with pytest.raises(ValueError):
        outbox.publish(alert(""))
    assert outbox.stats()["alerts"] == 0
    assert not outbox._conn.in_transaction


def test_dedupe_within_window_and_not_outside(outbox: Outbox, clock: FakeClock) -> None:
    outbox.register(sub())
    outbox.register(sub(profile="other"))
    assert DEDUPE_WINDOW_S == 300.0

    clock.now = 1000.0
    assert outbox.publish(alert("a1", dedupe_key="render:crypt", created_at=1000.0)) == 1
    clock.now = 1299.999
    assert outbox.publish(alert("a2", dedupe_key="render:crypt", created_at=1299.999)) == 0
    assert outbox.alert("a2") is None            # dropped, not stored
    assert outbox.rows_for("a2") == []
    clock.now = 1300.0                           # the window is half-open
    assert outbox.publish(alert("a3", dedupe_key="render:crypt", created_at=1300.0)) == 1
    clock.now = 1300.5                           # a3 now anchors the window again
    assert outbox.publish(alert("a4", dedupe_key="render:crypt", created_at=1300.5)) == 0

    # Another profile, another key, or no key at all: never deduped.
    assert outbox.publish(alert("b1", profile="other", dedupe_key="render:crypt", created_at=1300.5)) == 1
    assert outbox.publish(alert("a5", dedupe_key="render:other", created_at=1300.5)) == 1
    assert outbox.publish(alert("a6", created_at=1300.5)) == 1
    assert outbox.publish(alert("a7", created_at=1300.5)) == 1
    assert outbox.stats()["alerts"] == 6


def test_backfill_only_creates_rows_the_device_lacks(outbox: Outbox, clock: FakeClock) -> None:
    outbox.register(sub(device="phone"))
    for i, t in enumerate((10.0, 20.0, 30.0), start=1):
        clock.now = t
        assert outbox.publish(alert(f"a{i}", created_at=t)) == 1
    clock.now = 25.0
    outbox.publish(alert("z1", profile="other", created_at=25.0))  # no subs, stored only

    clock.now = 40.0
    outbox.register(sub(device="laptop", created_at=40.0))
    # a1 was created at exactly `since` and is excluded: newer than means >.
    assert outbox.backfill("p", since=10.0) == 2
    assert [r.device_id for r in outbox.rows_for("a1")] == ["phone"]
    assert [r.device_id for r in outbox.rows_for("a2")] == ["phone", "laptop"]
    assert [r.device_id for r in outbox.rows_for("a3")] == ["phone", "laptop"]
    laptop_rows = [r for a in ("a2", "a3") for r in outbox.rows_for(a) if r.device_id == "laptop"]
    assert all(r.state is RowState.PENDING and r.attempts == 0 and r.next_due == 40.0 for r in laptop_rows)

    # Idempotent, and a delivered or dead row counts as "has a row".
    assert outbox.backfill("p", since=10.0) == 0
    phone_a2 = next(r for r in outbox.rows_for("a2") if r.device_id == "phone")
    leased = {r.row_id for r in outbox.lease(now=50.0, limit=10)}
    assert phone_a2.row_id in leased
    outbox.mark(phone_a2.row_id, OK, now=50.0)
    assert outbox.backfill("p", since=0.0) == 1          # only a1 for laptop was missing
    assert [r.device_id for r in outbox.rows_for("a1")] == ["phone", "laptop"]

    # Gone devices and other profiles get nothing.
    gone_row = next(r for r in outbox.rows_for("a1") if r.device_id == "laptop")
    leased = {r.row_id for r in outbox.lease(now=60.0, limit=10)}
    assert gone_row.row_id in leased
    outbox.mark(gone_row.row_id, GONE, now=60.0)
    outbox.publish(alert("a4", created_at=60.0))
    assert outbox.backfill("p", since=0.0) == 0
    assert outbox.backfill("other", since=0.0) == 0
    assert outbox.rows_for("z1") == []


# --------------------------------------------------------------------------
# Lease
# --------------------------------------------------------------------------


def test_lease_returns_due_rows_only_in_priority_order(outbox: Outbox, clock: FakeClock) -> None:
    outbox.register(sub())
    plan = [
        ("low", Priority.LOW, 1.0),
        ("high-early", Priority.HIGH, 2.0),
        ("normal", Priority.NORMAL, 3.0),
        ("high-late", Priority.HIGH, 4.0),
        ("future", Priority.HIGH, 500.0),
    ]
    for id, priority, t in plan:
        clock.now = t
        outbox.publish(alert(id, priority=priority, created_at=t))

    assert outbox.lease(now=100.0, limit=0) == []
    first = outbox.lease(now=100.0, limit=2, lease_s=30.0)
    assert [r.alert_id for r in first] == ["high-early", "high-late"]
    for row in first:
        assert row.state is RowState.LEASED
        assert row.lease_until == 130.0
        assert outbox.row(row.row_id) == row              # what was returned is what was stored
    rest = outbox.lease(now=100.0, limit=10)
    assert [r.alert_id for r in rest] == ["normal", "low"]
    assert rest[0].lease_until == 100.0 + LEASE_S
    assert outbox.lease(now=100.0, limit=10) == []       # nothing due is left
    for row in first + rest:                             # settle them, or they would be reclaimed below
        outbox.mark(row.row_id, OK, now=101.0)
    assert outbox.lease(now=499.0, limit=10) == []
    assert [r.alert_id for r in outbox.lease(now=500.0, limit=10)] == ["future"]
    with pytest.raises(ValueError):
        outbox.lease(now=500.0, limit=1, lease_s=0)


def test_lease_reclaims_expired_leases(outbox: Outbox, clock: FakeClock) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    row = lease_one(outbox, 1000.0)
    assert row.lease_until == 1000.0 + LEASE_S
    assert outbox.lease(now=1000.0 + LEASE_S, limit=10) == []      # not yet: lease_until < now is strict
    assert outbox.stats()["leased"] == 1

    again = lease_one(outbox, 1000.0 + LEASE_S + 0.5)              # crashed worker's row comes back
    assert again.row_id == row.row_id
    assert again.state is RowState.LEASED
    assert again.attempts == 0                                     # a reclaim is not an attempt
    assert again.last_reason == "lease expired"
    assert again.lease_until == 1000.0 + LEASE_S + 0.5 + LEASE_S
    assert outbox.attempts_for(row.row_id) == []
    assert outbox.stats()["leased"] == 1


def test_lease_skips_devices_without_a_live_subscription(outbox: Outbox, clock: FakeClock) -> None:
    outbox.register(sub(device="phone"))
    outbox.register(sub(device="laptop"))
    outbox.publish(alert("a"))
    outbox.unregister("p", "laptop")
    assert [r.device_id for r in outbox.lease(now=1000.0, limit=10)] == ["phone"]
    stats = outbox.stats()
    assert stats["pending"] == 1 and stats["pending_unreachable"] == 1
    outbox.register(sub(device="laptop"))
    assert [r.device_id for r in outbox.lease(now=1000.0, limit=10)] == ["laptop"]
    assert outbox.stats()["pending_unreachable"] == 0


@pytest.mark.parametrize("shared", [False, True], ids=["two-connections", "shared-instance"])
def test_two_workers_leasing_concurrently_get_disjoint_rows(path: Path, clock: FakeClock, shared: bool) -> None:
    """1000 rows, two threads racing on lease; every row goes to exactly one.

    ``two-connections`` is the real deployment shape (two processes on one
    file) and proves the guarantee comes from ``BEGIN IMMEDIATE``, not from
    a Python lock; ``shared-instance`` proves one connection can be shared.
    """
    box = Outbox(path, clock)
    for d in range(10):
        box.register(sub(device=f"d{d:02d}"))
    for a in range(100):
        assert box.publish(alert(f"a{a:03d}", created_at=1000.0 + a)) == 10
    assert box.stats()["pending"] == 1000

    barrier = threading.Barrier(2)
    got: Dict[int, List[int]] = {0: [], 1: []}
    calls: Dict[int, int] = {0: 0, 1: 0}
    errors: List[BaseException] = []

    def worker(i: int) -> None:
        mine = box if shared else Outbox(path, clock)
        try:
            barrier.wait()
            while True:
                rows = mine.lease(now=2000.0, limit=7, lease_s=30.0)
                calls[i] += 1
                if not rows:
                    break
                got[i].extend(r.row_id for r in rows)
        except BaseException as exc:  # pragma: no cover - reported via `errors`
            errors.append(exc)
        finally:
            if not shared:
                mine.close()

    threads = [threading.Thread(target=worker, args=(i,), name=f"worker-{i}") for i in (0, 1)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads)
    assert errors == []
    assert len(got[0]) == len(set(got[0])) and len(got[1]) == len(set(got[1]))
    assert not (set(got[0]) & set(got[1])), "a row was leased twice"
    assert len(got[0]) + len(got[1]) == 1000
    assert set(got[0]) | set(got[1]) == set(range(1, 1001))
    assert box.stats()["leased"] == 1000
    assert box.lease(now=2000.0, limit=10) == []
    box.close()


# --------------------------------------------------------------------------
# Mark
# --------------------------------------------------------------------------


def test_mark_ok_delivers_and_resets_failures(outbox: Outbox) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    row = lease_one(outbox, 1000.0)
    assert outbox.mark(row.row_id, RETRY, now=1001.0, jitter=0.0) is RowState.PENDING
    assert outbox.subscription("p", "phone").failures == 1

    row = lease_one(outbox, 5000.0)
    assert outbox.mark(row.row_id, SendResult(ok=True, reason="201"), now=5001.0) is RowState.DELIVERED
    stored = outbox.row(row.row_id)
    assert stored.state is RowState.DELIVERED
    assert stored.attempts == 2
    assert stored.lease_until == 0.0
    assert stored.last_reason == "201"
    assert outbox.subscription("p", "phone").failures == 0
    attempts = outbox.attempts_for(row.row_id)
    assert [(a.attempt, a.at, a.ok, a.retryable, a.gone, a.reason) for a in attempts] == [
        (1, 1001.0, False, True, False, "503"),
        (2, 5001.0, True, False, False, "201"),
    ]
    assert all(a.row_id == row.row_id for a in attempts)
    assert outbox.stats()["delivered"] == 1
    assert outbox.dead_letters() == []


def test_mark_gone_kills_the_row_and_the_subscription(outbox: Outbox) -> None:
    outbox.register(sub(device="phone"))
    outbox.register(sub(device="laptop"))
    outbox.publish(alert("a"))
    outbox.publish(alert("b"))
    phone_a = next(r for r in outbox.lease(now=1000.0, limit=10) if r.alert_id == "a" and r.device_id == "phone")
    assert outbox.mark(phone_a.row_id, GONE, now=1001.0) is RowState.DEAD

    stored = outbox.row(phone_a.row_id)
    assert stored.state is RowState.DEAD and stored.attempts == 1 and stored.last_reason == "410"
    phone = outbox.subscription("p", "phone")
    assert phone.gone is True
    assert [s.device_id for s in outbox.subscriptions_for("p")] == ["laptop"]
    assert [r.row_id for r in outbox.dead_letters()] == [phone_a.row_id]
    # The rest of that device's rows are parked, not handed out again.
    for row in outbox.lease(now=1000.0, limit=10):
        assert row.device_id == "laptop"
    outbox.publish(alert("c"))
    assert [r.device_id for r in outbox.rows_for("c")] == ["laptop"]
    stats = outbox.stats()
    assert stats["dead"] == 1 and stats["subscriptions_gone"] == 1 and stats["subscriptions_live"] == 1


def test_mark_retryable_backs_off_until_max_attempts(outbox: Outbox) -> None:
    assert MAX_ATTEMPTS == 8
    outbox.register(sub())
    outbox.publish(alert("a"))
    now = 1000.0
    jitter = 0.25
    for n in range(1, MAX_ATTEMPTS + 1):
        row = lease_one(outbox, now)
        assert row.attempts == n - 1
        state = outbox.mark(row.row_id, RETRY, now=now, jitter=jitter)
        stored = outbox.row(row.row_id)
        assert stored.attempts == n
        assert stored.lease_until == 0.0
        assert stored.last_reason == "503"
        if n < MAX_ATTEMPTS:
            assert state is stored.state is RowState.PENDING
            # The n-th failure waits the n-th delay: 2 * 2**(n-1), capped at 300, times 0.625.
            assert stored.next_due == now + backoff_seconds(n, jitter)
            assert stored.next_due == now + min(300.0, 2.0 * 2 ** (n - 1)) * 0.625
            assert outbox.lease(now=stored.next_due - 0.001, limit=10) == []   # not due yet
            now = stored.next_due
        else:
            assert state is stored.state is RowState.DEAD
            assert stored.next_due == now                                     # untouched on the way to DEAD
    assert [a.attempt for a in outbox.attempts_for(row.row_id)] == list(range(1, MAX_ATTEMPTS + 1))
    assert outbox.subscription("p", "phone").failures == MAX_ATTEMPTS
    assert outbox.subscription("p", "phone").gone is False
    assert outbox.lease(now=now + 1e6, limit=10) == []
    assert [r.row_id for r in outbox.dead_letters()] == [row.row_id]
    assert outbox.stats()["dead"] == 1


def test_mark_non_retryable_failure_is_dead_and_leaves_the_device_alone(outbox: Outbox) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    row = lease_one(outbox, 1000.0)
    assert outbox.mark(row.row_id, FATAL, now=1001.0) is RowState.DEAD
    stored = outbox.row(row.row_id)
    assert stored.state is RowState.DEAD and stored.attempts == 1
    assert stored.last_reason == "payload too large"
    phone = outbox.subscription("p", "phone")
    assert phone.failures == 0 and phone.gone is False
    attempt, = outbox.attempts_for(row.row_id)
    assert (attempt.ok, attempt.retryable, attempt.gone) == (False, False, False)


def test_prune_after_failures_boundary(outbox: Outbox) -> None:
    """The 20th consecutive transient failure marks the device gone; the
    19th does not; a success in between resets the count; re-registering
    brings the device and its parked rows back.  (A prune is not for ever:
    the device is probed again after PRUNE_COOLDOWN_S, see
    test_pruned_device_is_probed_again_after_the_cooldown; here the clock
    stays inside the cooldown.)"""
    assert PRUNE_AFTER_FAILURES == 20
    outbox.register(sub())
    for i in range(4):                       # 4 rows x up to 7 failures each stays under MAX_ATTEMPTS
        outbox.publish(alert(f"a{i}"))

    now = 1000.0

    def fail_once() -> RowState:
        nonlocal now
        now += 1000.0                        # past every backoff, so the least-recently-failed row is next
        row = lease_one(outbox, now)
        return outbox.mark(row.row_id, RETRY, now=now, jitter=0.0)

    for i in range(PRUNE_AFTER_FAILURES - 1):
        assert fail_once() is RowState.PENDING
    phone = outbox.subscription("p", "phone")
    assert phone.failures == PRUNE_AFTER_FAILURES - 1 and phone.gone is False

    # A success resets the streak, so it takes 20 more to prune.
    now += 1000.0
    for row in outbox.lease(now=now, limit=10):
        outbox.mark(row.row_id, OK, now=now)
    assert outbox.subscription("p", "phone").failures == 0
    assert outbox.stats()["delivered"] == 4
    for i in range(4):                       # fresh rows: the old ones carry 5 attempts each
        outbox.publish(alert(f"b{i}"))
    for i in range(PRUNE_AFTER_FAILURES - 1):
        assert fail_once() is RowState.PENDING
    assert outbox.subscription("p", "phone").gone is False

    assert fail_once() is RowState.PENDING   # the row itself follows the retry rule...
    phone = outbox.subscription("p", "phone")
    assert phone.failures == PRUNE_AFTER_FAILURES and phone.gone is True   # ...but the device is pruned
    assert phone.pruned is True
    assert outbox.subscriptions_for("p") == []
    assert outbox.pruned_subscriptions() == [("p", "phone")]
    assert outbox.lease(now=now + PRUNE_COOLDOWN_S - 1, limit=10) == []
    stats = outbox.stats()
    assert stats["pending"] == 4 and stats["pending_unreachable"] == 4 and stats["dead"] == 0
    assert all(r.attempts < MAX_ATTEMPTS for i in range(4) for r in outbox.rows_for(f"b{i}"))

    outbox.register(sub())
    assert outbox.subscription("p", "phone").failures == 0
    assert len(outbox.lease(now=now + PRUNE_COOLDOWN_S - 1, limit=10)) == 4


def test_mark_on_a_terminal_row_is_ignored(outbox: Outbox) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    outbox.publish(alert("b"))
    a, b = outbox.lease(now=1000.0, limit=10)
    assert outbox.mark(a.row_id, OK, now=1001.0) is RowState.DELIVERED
    assert outbox.mark(b.row_id, FATAL, now=1001.0) is RowState.DEAD
    assert outbox.mark(a.row_id, RETRY, now=1002.0) is RowState.DELIVERED
    assert outbox.mark(b.row_id, OK, now=1002.0) is RowState.DEAD
    assert outbox.row(a.row_id).attempts == 1 and len(outbox.attempts_for(a.row_id)) == 1
    assert outbox.row(b.row_id).attempts == 1 and len(outbox.attempts_for(b.row_id)) == 1
    assert outbox.subscription("p", "phone").failures == 0


def test_a_late_mark_after_the_lease_expired_still_counts(outbox: Outbox) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    outbox.publish(alert("b"))
    a, b = outbox.lease(now=1000.0, limit=10, lease_s=30.0)
    # Worker A stalls; its rows are reclaimed. Row a is re-leased by worker B,
    # row b sits PENDING. A then wakes and reports both as delivered.
    (a_again,) = outbox.lease(now=1031.0, limit=1)
    assert a_again.row_id == a.row_id
    assert outbox.mark(a.row_id, OK, now=1032.0) is RowState.DELIVERED
    assert outbox.row(b.row_id).state is RowState.PENDING
    assert outbox.mark(b.row_id, OK, now=1032.0) is RowState.DELIVERED
    assert outbox.lease(now=1033.0, limit=10) == []
    assert outbox.stats()["delivered"] == 2


def test_mark_and_attempts_for_unknown_row_raise(outbox: Outbox) -> None:
    with pytest.raises(UnknownRow, match="42"):
        outbox.mark(42, OK, now=1.0)
    with pytest.raises(UnknownRow, match="42"):
        outbox.attempts_for(42)
    assert outbox.row(42) is None
    assert not outbox._conn.in_transaction


def test_jitter_comes_from_the_injected_stream_only_when_a_backoff_is_needed(path: Path, clock: FakeClock) -> None:
    def make(tag: str, seed: int, counter: List[int]) -> Outbox:
        stream = SeedFields.parse(seed).stream("alerts.backoff")

        def draw() -> float:
            counter[0] += 1
            return stream.random()

        box = Outbox(path.with_name(f"{tag}.sqlite"), clock, jitter=draw)
        box.register(sub())
        for i in range(6):
            box.publish(alert(f"a{i}"))
        return box

    def run(box: Outbox) -> List[float]:
        due: List[float] = []
        now = 1000.0
        rows = box.lease(now=now, limit=10)
        box.mark(rows[0].row_id, OK, now=now)                 # no draw
        box.mark(rows[1].row_id, GONE, now=now)               # no draw (and prunes nothing else here)
        box.register(sub())                                   # device back
        box.mark(rows[2].row_id, FATAL, now=now)              # no draw
        box.mark(rows[3].row_id, RETRY, now=now, jitter=0.5)  # explicit number: no draw
        due.append(box.row(rows[3].row_id).next_due)
        for row in rows[4:]:
            box.mark(row.row_id, RETRY, now=now)              # drawn from the stream
            due.append(box.row(row.row_id).next_due)
        return due

    count_a, count_b, count_c = [0], [0], [0]
    box_a, box_b, box_c = make("a", 0xBEEF, count_a), make("b", 0xBEEF, count_b), make("c", 0xF00D, count_c)
    try:
        due_a, due_b, due_c = run(box_a), run(box_b), run(box_c)
    finally:
        box_a.close(), box_b.close(), box_c.close()
    assert count_a == count_b == count_c == [2]               # only the two real backoffs drew
    assert due_a == due_b                                     # same seed, same schedule
    assert due_a[0] == 1000.0 + backoff_seconds(1, 0.5)
    assert due_a[1:] != due_c[1:]                             # different seed, different schedule
    for d in due_a[1:] + due_c[1:]:
        assert 1000.0 + backoff_seconds(1, 0.0) <= d < 1000.0 + backoff_seconds(1, 1.0)


def test_invalid_jitter_rolls_the_mark_back(outbox: Outbox) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    row = lease_one(outbox, 1000.0)
    with pytest.raises(ValueError, match="jitter"):
        outbox.mark(row.row_id, RETRY, now=1001.0, jitter=1.5)
    assert not outbox._conn.in_transaction
    stored = outbox.row(row.row_id)
    assert stored.state is RowState.LEASED and stored.attempts == 0
    assert outbox.attempts_for(row.row_id) == []
    assert outbox.subscription("p", "phone").failures == 0


# --------------------------------------------------------------------------
# Privacy: the blob never leaks
# --------------------------------------------------------------------------


def test_blob_never_appears_in_rows_stats_dead_letters_or_exceptions(
    outbox: Outbox, path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    texts: List[str] = []

    def capture(*values: Any) -> None:
        texts.extend(str(v) for v in values)
        texts.extend(repr(v) for v in values)

    outbox.register(sub(device="phone"))
    outbox.register(sub(device="laptop"))
    outbox.register(sub(device="tablet"))
    assert outbox.subscription("p", "phone").blob.count(CANARY) == 1   # the canary is really in there
    capture(outbox)
    capture(outbox.publish(alert("a", dedupe_key="k")))
    rows = outbox.lease(now=1000.0, limit=10)
    capture(rows, *rows)
    outbox.mark(rows[0].row_id, OK, now=1001.0)
    outbox.mark(rows[1].row_id, RETRY, now=1001.0, jitter=0.5)
    outbox.mark(rows[2].row_id, GONE, now=1001.0)
    capture(outbox.stats(), json.dumps(outbox.stats()), outbox.dead_letters(), *outbox.dead_letters())
    capture(outbox.row(rows[1].row_id), outbox.rows_for("a"), outbox.attempts_for(rows[1].row_id))
    capture(outbox.alert("a"))

    with pytest.raises(ValueError) as bad_device:
        outbox.register(sub(device="", blob=CANARY))
    with pytest.raises(ValueError) as bad_blob:
        outbox.register(Subscription("p", "watch", "fake", {"blob": CANARY}, 1.0))  # type: ignore[arg-type]
    with pytest.raises(DuplicateAlert) as dup:
        outbox.publish(alert("a"))
    with pytest.raises(UnknownRow) as unknown:
        outbox.mark(999, OK, now=1.0)
    with pytest.raises(ValueError) as jitter:
        outbox.mark(rows[1].row_id, RETRY, now=1.0, jitter=7)

    def explode(self: Outbox, cur: sqlite3.Cursor, a: Alert, now: float) -> int:
        raise RuntimeError(f"fan-out failed for {a.id} at {now}")

    monkeypatch.setattr(Outbox, "_fan_out", explode)
    with pytest.raises(RuntimeError) as injected:
        outbox.publish(alert("b"))
    monkeypatch.undo()
    for info in (bad_device, bad_blob, dup, unknown, jitter, injected):
        capture(info.value, info.value.args, info.getrepr(style="long"))

    assert len(texts) > 20
    for text in texts:
        assert CANARY not in text, text


# --------------------------------------------------------------------------
# Atomicity: an interrupted mutation leaves no partial rows
# --------------------------------------------------------------------------


def test_interrupted_publish_leaves_no_partial_rows(
    outbox: Outbox, path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox.register(sub(device="phone"))
    outbox.register(sub(device="laptop"))
    outbox.publish(alert("before"))

    def explode(self: Outbox, cur: sqlite3.Cursor, a: Alert, now: float) -> int:
        # By now the alert row is inside the open transaction; it must not survive.
        assert cur.execute("SELECT count(*) FROM alerts WHERE id = ?", (a.id,)).fetchone()[0] == 1
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(Outbox, "_fan_out", explode)
    with pytest.raises(RuntimeError, match="disk on fire"):
        outbox.publish(alert("broken", dedupe_key="k"))
    assert not outbox._conn.in_transaction
    assert outbox.alert("broken") is None
    assert outbox.rows_for("broken") == []
    assert outbox.stats()["alerts"] == 1 and outbox.stats()["pending"] == 2

    with Outbox(path, clock) as fresh:                       # the file agrees
        assert fresh.alert("broken") is None
        assert fresh._conn.execute("SELECT count(*) FROM outbox").fetchone()[0] == 2

    monkeypatch.undo()                                       # still usable, and not deduped against the ghost
    assert outbox.publish(alert("broken", dedupe_key="k")) == 2


def test_interrupted_mark_leaves_no_partial_rows(
    outbox: Outbox, path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    row = lease_one(outbox, 1000.0)

    def explode(self: Outbox, cur: sqlite3.Cursor, *args: Any) -> None:
        # The attempt record and the failure bump are already inside the transaction.
        assert cur.execute("SELECT count(*) FROM attempts").fetchone()[0] == 1
        assert cur.execute("SELECT failures FROM subscriptions").fetchone()[0] == 1
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(Outbox, "_update_row", explode)
    with pytest.raises(RuntimeError, match="disk on fire"):
        outbox.mark(row.row_id, RETRY, now=1001.0, jitter=0.0)
    assert not outbox._conn.in_transaction
    assert outbox.attempts_for(row.row_id) == []
    stored = outbox.row(row.row_id)
    assert stored.state is RowState.LEASED and stored.attempts == 0 and stored.lease_until == row.lease_until
    assert outbox.subscription("p", "phone").failures == 0
    with Outbox(path, clock) as fresh:
        assert fresh._conn.execute("SELECT count(*) FROM attempts").fetchone()[0] == 0
        assert fresh.row(row.row_id).state is RowState.LEASED

    monkeypatch.undo()
    assert outbox.mark(row.row_id, RETRY, now=1001.0, jitter=0.0) is RowState.PENDING
    assert len(outbox.attempts_for(row.row_id)) == 1


# --------------------------------------------------------------------------
# Stats
# --------------------------------------------------------------------------


def test_stats_counts_every_state_and_lists_dead_letters_newest_first(outbox: Outbox) -> None:
    empty = outbox.stats()
    assert empty == {
        "pending": 0, "leased": 0, "delivered": 0, "dead": 0, "pending_unreachable": 0,
        "dead_exhausted": 0, "dead_revivable": 0, "dead_permanent": 0, "revived": 0,
        "alerts": 0, "subscriptions_live": 0, "subscriptions_gone": 0, "dead_letters": [],
    }
    outbox.register(sub(device="phone"))
    outbox.register(sub(device="laptop"))
    for i in range(3):
        outbox.publish(alert(f"a{i}"))
    rows = outbox.lease(now=1000.0, limit=4)                 # 4 of 6 leased
    outbox.mark(rows[0].row_id, OK, now=1001.0)
    outbox.mark(rows[1].row_id, FATAL, now=1001.0)
    outbox.mark(rows[2].row_id, SendResult(ok=False, reason="bad key"), now=1002.0)
    stats = outbox.stats()
    assert {k: stats[k] for k in ("pending", "leased", "delivered", "dead")} == {
        "pending": 2, "leased": 1, "delivered": 1, "dead": 2,
    }
    assert stats["alerts"] == 3 and stats["subscriptions_live"] == 2 and stats["subscriptions_gone"] == 0
    # Both deaths are non-retryable, so neither is one the outbox will undo.
    assert {k: stats[k] for k in ("dead_exhausted", "dead_revivable", "dead_permanent", "revived")} == {
        "dead_exhausted": 0, "dead_revivable": 0, "dead_permanent": 2, "revived": 0,
    }
    assert [d["row_id"] for d in stats["dead_letters"]] == [rows[2].row_id, rows[1].row_id]
    assert stats["dead_letters"][0] == {
        "row_id": rows[2].row_id, "alert_id": rows[2].alert_id, "profile_id": "p",
        "device_id": rows[2].device_id, "state": "dead", "attempts": 1, "next_due": 1000.0,
        "lease_until": 0.0, "last_reason": "bad key",
        "dead_reason": "permanent", "dead_at": 1002.0, "attempts_base": 0,
    }
    assert [r.row_id for r in outbox.dead_letters(limit=1)] == [rows[2].row_id]
    assert outbox.dead_letters(limit=0) == []
    assert [d["row_id"] for d in outbox.stats(dead_limit=1)["dead_letters"]] == [rows[2].row_id]
    json.dumps(stats)                                        # JSON-serialisable end to end


# --------------------------------------------------------------------------
# Nothing is lost silently: prune cooldown, release, requeue, dedupe repair
# --------------------------------------------------------------------------


def prune(outbox: Outbox, now: float) -> float:
    """Drive the one registered device to PRUNE_AFTER_FAILURES; return the
    time of the pruning failure."""
    for _ in range(PRUNE_AFTER_FAILURES):
        now += 1000.0
        row = lease_one(outbox, now)
        outbox.mark(row.row_id, RETRY, now=now, jitter=0.0)
    assert outbox.subscription("p", "phone").pruned
    return now


def test_pruned_device_is_probed_again_after_the_cooldown(outbox: Outbox, clock: FakeClock) -> None:
    """A device pruned by failure count is treated as gone only for
    PRUNE_COOLDOWN_S; an alert published meanwhile still gets a row
    (parked), and the next lease after the cooldown revives the device
    with a clean failure count and hands its rows out.  A transport-
    reported gone never expires."""
    outbox.register(sub())
    for i in range(4):
        outbox.publish(alert(f"a{i}"))
    pruned_at = prune(outbox, 1000.0)

    clock.now = pruned_at + 10.0
    assert outbox.publish(alert("during")) == 1                    # a row, parked: not silently dropped
    (row,) = outbox.rows_for("during")
    assert row.state is RowState.PENDING
    stats = outbox.stats()
    assert stats["pending"] == 5 and stats["pending_unreachable"] == 5
    assert outbox.lease(now=pruned_at + PRUNE_COOLDOWN_S - 0.001, limit=10) == []

    revived = outbox.lease(now=pruned_at + PRUNE_COOLDOWN_S, limit=10)
    assert sorted(r.alert_id for r in revived) == ["a0", "a1", "a2", "a3", "during"]
    phone = outbox.subscription("p", "phone")
    assert phone.gone is False and phone.pruned is False and phone.failures == 0
    assert outbox.pruned_subscriptions() == []
    for r in revived:
        outbox.mark(r.row_id, OK, now=pruned_at + PRUNE_COOLDOWN_S)
    assert outbox.stats()["delivered"] == 5

    # Gone by the transport: no cooldown, and a later prune cannot downgrade it.
    outbox.publish(alert("z"))
    row = lease_one(outbox, pruned_at + PRUNE_COOLDOWN_S + 1)
    outbox.mark(row.row_id, GONE, now=pruned_at + PRUNE_COOLDOWN_S + 1)
    assert outbox.lease(now=pruned_at + 10 * PRUNE_COOLDOWN_S, limit=10) == []
    gone = outbox.subscription("p", "phone")
    assert gone.gone is True and gone.pruned is False
    assert outbox.publish(alert("after-gone")) == 0                 # no row for a device the transport killed


def test_release_hands_a_leased_row_back_without_an_attempt(outbox: Outbox) -> None:
    outbox.register(sub())
    outbox.publish(alert("a"))
    row = lease_one(outbox, 1000.0)
    assert outbox.release(row.row_id, "device pruned") is RowState.PENDING
    stored = outbox.row(row.row_id)
    assert stored.state is RowState.PENDING and stored.attempts == 0 and stored.lease_until == 0.0
    assert stored.last_reason == "device pruned" and stored.next_due == row.next_due
    assert outbox.attempts_for(row.row_id) == []
    assert outbox.subscription("p", "phone").failures == 0
    again = lease_one(outbox, 1000.0)                                # due again at once
    assert again.row_id == row.row_id
    outbox.mark(row.row_id, OK, now=1001.0)
    assert outbox.release(row.row_id) is RowState.DELIVERED          # settled rows are left alone
    with pytest.raises(UnknownRow):
        outbox.release(99)


def test_requeue_gives_a_dead_row_a_fresh_budget(outbox: Outbox, clock: FakeClock) -> None:
    outbox.register(sub(device="phone"))
    outbox.register(sub(device="laptop"))
    outbox.publish(alert("a"))
    now = 1000.0
    for r in outbox.lease(now=now, limit=10):
        if r.device_id == "laptop":
            outbox.mark(r.row_id, OK, now=now)
        else:
            phone_row = r
    for n in range(MAX_ATTEMPTS):                                    # exhaust the budget on the phone
        if n:
            (phone_row,) = outbox.lease(now=now, limit=10)
        outbox.mark(phone_row.row_id, RETRY, now=now, jitter=0.0)
        now += 1000.0
    assert outbox.row(phone_row.row_id).state is RowState.DEAD
    outbox.publish(alert("b"))
    for r in outbox.lease(now=now, limit=10):
        assert outbox.requeue(r.row_id) is False                     # LEASED and DELIVERED rows are left alone
        assert outbox.row(r.row_id).state is RowState.LEASED
        outbox.mark(r.row_id, OK, now=now)
    laptop_row = next(r for r in outbox.rows_for("a") if r.device_id == "laptop")
    assert outbox.requeue(laptop_row.row_id) is False

    clock.now = now
    assert outbox.requeue(phone_row.row_id) is True
    stored = outbox.row(phone_row.row_id)
    assert stored.state is RowState.PENDING and stored.next_due == now and stored.last_reason == "requeued"
    assert stored.attempts == MAX_ATTEMPTS                           # history is kept ...
    assert len(outbox.attempts_for(phone_row.row_id)) == MAX_ATTEMPTS
    row = lease_one(outbox, now)
    assert row.row_id == phone_row.row_id
    assert outbox.mark(row.row_id, RETRY, now=now, jitter=0.0) is RowState.PENDING   # ... but the budget is fresh
    assert outbox.row(row.row_id).next_due == now + backoff_seconds(1, 0.0)   # and so is the backoff schedule
    assert outbox.row(row.row_id).attempts == MAX_ATTEMPTS + 1
    for n in range(2, MAX_ATTEMPTS + 1):
        now += 1000.0
        row = lease_one(outbox, now)
        state = outbox.mark(row.row_id, RETRY, now=now, jitter=0.0)
    assert state is RowState.DEAD and outbox.row(row.row_id).attempts == 2 * MAX_ATTEMPTS

    # requeue_dead: by device, by profile, all; unknown ids raise.
    assert outbox.requeue_dead("p", "nobody") == 0
    assert outbox.requeue_dead("p", "phone") == 1
    assert outbox.row(row.row_id).state is RowState.PENDING
    with pytest.raises(ValueError):
        outbox.requeue_dead(None, "phone")
    with pytest.raises(UnknownRow):
        outbox.requeue(12345)


# --------------------------------------------------------------------------
# Automatic revival: an outage that outlasted the retry budget
# --------------------------------------------------------------------------


def test_exhausted_row_is_revived_after_the_cooldown_but_not_before(outbox: Outbox, clock: FakeClock) -> None:
    """The retry budget spans about four minutes, which is shorter than an
    ordinary push-service outage, so exhaustion is not a verdict on the
    alert (contracts.py, retry policy): the row dies EXHAUSTED and the
    outbox itself puts it back EXHAUSTED_RETRY_COOLDOWN_S later -- then,
    and not one instant earlier."""
    outbox.register(sub())
    outbox.publish(alert("a", created_at=1000.0))
    died_at = exhaust_one(outbox, 1000.0)

    (row,) = outbox.rows_for("a")
    assert row.state is RowState.DEAD and row.dead_reason is DeadReason.EXHAUSTED
    assert row.dead_at == died_at and row.last_reason == "503"
    clock.now = died_at
    stats = outbox.stats()
    assert (stats["dead"], stats["dead_exhausted"], stats["dead_permanent"]) == (1, 1, 0)
    assert stats["dead_revivable"] == 1 and stats["revived"] == 0

    assert outbox.revive_exhausted(died_at + EXHAUSTED_RETRY_COOLDOWN_S - 0.001) == 0
    assert outbox.row(row.row_id).state is RowState.DEAD
    assert outbox.lease(now=died_at + EXHAUSTED_RETRY_COOLDOWN_S, limit=10) == []   # DEAD is not leasable

    back = died_at + EXHAUSTED_RETRY_COOLDOWN_S
    assert outbox.revive_exhausted(back) == 1
    stored = outbox.row(row.row_id)
    assert stored.state is RowState.PENDING and stored.next_due == back and stored.lease_until == 0.0
    assert stored.dead_reason is None and stored.dead_at == 0.0
    assert stored.last_reason == "requeued: exhausted, cooldown passed"
    assert outbox.revive_exhausted(back) == 0                    # and only once

    clock.now = back
    stats = outbox.stats()
    assert (stats["dead"], stats["dead_exhausted"], stats["dead_revivable"]) == (0, 0, 0)
    assert stats["pending"] == 1 and stats["pending_unreachable"] == 0 and stats["revived"] == 1

    # The outage is over: the revived row goes out on the next pass.
    leased = lease_one(outbox, back)
    assert leased.row_id == row.row_id
    assert outbox.mark(leased.row_id, OK, now=back) is RowState.DELIVERED
    assert outbox.stats()["revived"] == 1                        # the count is of revivals, for good


def test_revival_keeps_the_attempt_history_and_gives_a_fresh_budget(outbox: Outbox, clock: FakeClock) -> None:
    """A revival is a re-queue (:meth:`Outbox.requeue`), so the recorded
    attempts stay as history while the budget counts again from zero."""
    outbox.register(sub())
    outbox.publish(alert("a", created_at=1000.0))
    died_at = exhaust_one(outbox, 1000.0)
    (row,) = outbox.rows_for("a")
    history = outbox.attempts_for(row.row_id)
    assert [a.attempt for a in history] == list(range(1, MAX_ATTEMPTS + 1))

    back = died_at + EXHAUSTED_RETRY_COOLDOWN_S
    assert outbox.revive_exhausted(back) == 1
    stored = outbox.row(row.row_id)
    assert stored.attempts == MAX_ATTEMPTS                       # nothing is forgotten ...
    assert stored.attempts_base == MAX_ATTEMPTS                  # ... and the new budget counts from here
    assert outbox.attempts_for(row.row_id) == history

    # The first failure of the new budget waits the *first* backoff again,
    # and it takes another whole MAX_ATTEMPTS of them to die.
    leased = lease_one(outbox, back)
    assert outbox.mark(leased.row_id, RETRY, now=back, jitter=0.0) is RowState.PENDING
    assert outbox.row(row.row_id).next_due == back + backoff_seconds(1, 0.0)
    now = back
    for _ in range(2, MAX_ATTEMPTS + 1):
        now += 1000.0
        leased = lease_one(outbox, now)
        state = outbox.mark(leased.row_id, RETRY, now=now, jitter=0.0)
    assert state is RowState.DEAD
    stored = outbox.row(row.row_id)
    assert stored.attempts == 2 * MAX_ATTEMPTS and stored.dead_reason is DeadReason.EXHAUSTED
    assert [a.attempt for a in outbox.attempts_for(row.row_id)] == list(range(1, 2 * MAX_ATTEMPTS + 1))
    assert outbox.stats()["revived"] == 1


def test_revival_stops_at_the_alert_maximum_age(outbox: Outbox, clock: FakeClock) -> None:
    """An alert older than ALERT_MAX_AGE_S is never auto-requeued: its row
    stays DEAD as EXHAUSTED, the dead-letter list keeps it, and only the
    operator's requeue brings it back.  The two rows here straddle the
    boundary at one instant, so the limit is checked to the second."""
    outbox.register(sub())
    outbox.publish(alert("old", created_at=1000.0))
    old_died = exhaust_one(outbox, 1000.0)
    (old_row,) = outbox.rows_for("old")

    young_created = old_died + 1000.0
    clock.now = young_created
    outbox.publish(alert("young", created_at=young_created))
    young_died = exhaust_one(outbox, young_created)
    (young_row,) = outbox.rows_for("young")

    deadline = 1000.0 + ALERT_MAX_AGE_S          # the first instant the old alert is too old
    assert deadline > young_died + EXHAUSTED_RETRY_COOLDOWN_S    # both cooldowns passed long ago
    clock.now = deadline
    stats = outbox.stats()
    assert (stats["dead_exhausted"], stats["dead_revivable"]) == (2, 1)      # only the young one

    assert outbox.revive_exhausted(deadline) == 1
    assert outbox.row(young_row.row_id).state is RowState.PENDING
    assert outbox.row(old_row.row_id).state is RowState.DEAD

    # Not later either: the age is a limit, not another delay.
    assert outbox.revive_exhausted(deadline + 10 * EXHAUSTED_RETRY_COOLDOWN_S) == 0
    stale = outbox.row(old_row.row_id)
    assert stale.state is RowState.DEAD and stale.dead_reason is DeadReason.EXHAUSTED
    clock.now = deadline + 10 * EXHAUSTED_RETRY_COOLDOWN_S
    stats = outbox.stats()
    assert [d["row_id"] for d in stats["dead_letters"]] == [old_row.row_id]
    assert (stats["dead_exhausted"], stats["dead_revivable"], stats["revived"]) == (1, 0, 1)

    # The operator's way back still works, whatever the age.
    assert outbox.requeue(old_row.row_id) is True
    assert outbox.row(old_row.row_id).state is RowState.PENDING
    assert outbox.stats()["revived"] == 1                        # a requeue is not a revival


def test_only_exhaustion_is_revived_the_other_dead_reasons_wait_for_requeue(
    outbox: Outbox, clock: FakeClock
) -> None:
    """PERMANENT, GONE, NO_SUBSCRIPTION and NO_TRANSPORT are final for the
    automatic path (contracts.py, ``DeadReason.revivable``), however long
    one waits and however young the alert still is."""
    reasons = {
        "perm": FATAL,
        "gone": GONE,
        "nosub": SendResult(ok=False, reason="no subscription", dead_reason=DeadReason.NO_SUBSCRIPTION),
        "notrans": SendResult(ok=False, reason="no transport", dead_reason=DeadReason.NO_TRANSPORT),
    }
    for device in reasons:
        outbox.register(sub(device=device))
    outbox.publish(alert("a", created_at=1000.0))
    for row in outbox.lease(now=1000.0, limit=10):
        outbox.mark(row.row_id, reasons[row.device_id], now=1000.0)

    assert {r.device_id: r.dead_reason for r in outbox.rows_for("a")} == {
        "perm": DeadReason.PERMANENT, "gone": DeadReason.GONE,
        "nosub": DeadReason.NO_SUBSCRIPTION, "notrans": DeadReason.NO_TRANSPORT,
    }
    assert not any(r.dead_reason.revivable for r in outbox.rows_for("a"))
    clock.now = 1000.0
    stats = outbox.stats()
    assert (stats["dead"], stats["dead_exhausted"], stats["dead_revivable"], stats["dead_permanent"]) \
        == (4, 0, 0, 4)

    later = 1000.0 + ALERT_MAX_AGE_S - 1.0       # cooldown long passed, alert still young enough
    assert outbox.revive_exhausted(later) == 0
    assert all(r.state is RowState.DEAD for r in outbox.rows_for("a"))
    clock.now = later
    assert outbox.stats()["revived"] == 0

    # Only the operator brings these back.
    assert outbox.requeue_dead("p") == 4
    assert all(r.state is RowState.PENDING for r in outbox.rows_for("a"))
    assert outbox.stats()["revived"] == 0


def test_dedupe_repeat_repairs_the_earlier_alert(outbox: Outbox, clock: FakeClock) -> None:
    """A keyed repeat inside the window is still not stored, but it makes
    the earlier alert reach every reachable device: dead rows are
    re-queued and missing rows are created."""
    outbox.register(sub(device="phone"))
    clock.now = 1000.0
    assert outbox.publish(alert("a", dedupe_key="render:crypt", created_at=1000.0)) == 1
    (row,) = outbox.rows_for("a")
    outbox.lease(now=1000.0, limit=10)
    outbox.mark(row.row_id, FATAL, now=1001.0)                       # dead on the phone
    outbox.register(sub(device="laptop", created_at=1050.0))         # a device with no row for "a"
    outbox.register(sub(device="tablet", created_at=1050.0))
    tablet_row_before = outbox.rows_for("a")
    assert [r.device_id for r in tablet_row_before] == ["phone"]

    clock.now = 1100.0
    assert outbox.publish(alert("b", dedupe_key="render:crypt", created_at=1100.0)) == 3
    assert outbox.alert("b") is None and outbox.rows_for("b") == []  # not stored, no rows of its own
    rows = {r.device_id: r for r in outbox.rows_for("a")}
    assert set(rows) == {"phone", "laptop", "tablet"}
    assert rows["phone"].state is RowState.PENDING and rows["phone"].attempts == 1
    assert rows["phone"].last_reason.startswith("requeued")
    assert rows["laptop"].state is RowState.PENDING and rows["laptop"].attempts == 0 and rows["laptop"].next_due == 1100.0
    assert outbox.stats()["alerts"] == 1

    # Once everything is covered a repeat repairs nothing and returns 0.
    for r in outbox.lease(now=1100.0, limit=10):
        outbox.mark(r.row_id, OK, now=1101.0)
    clock.now = 1200.0
    assert outbox.publish(alert("c", dedupe_key="render:crypt", created_at=1200.0)) == 0
    assert outbox.alert("c") is None
    # A device the transport reported gone is not repaired to.
    (phone_row,) = [r for r in outbox.rows_for("a") if r.device_id == "phone"]
    outbox.publish(alert("z", created_at=1200.0))
    for r in outbox.lease(now=1200.0, limit=10):
        if r.device_id == "phone":
            outbox.mark(r.row_id, GONE, now=1201.0)
    assert outbox.publish(alert("d", dedupe_key="render:crypt", created_at=1250.0)) == 0
    # Outside the window it is a new alert again.
    clock.now = 1500.0
    assert outbox.publish(alert("e", dedupe_key="render:crypt", created_at=1500.0)) == 2


def test_register_can_supersede_a_device_holding_the_same_blob(outbox: Outbox) -> None:
    """One browser that lost its device id but kept its push subscription
    must not become two devices (two pushes per alert)."""
    outbox.register(sub(device="old", blob='{"endpoint": "https://push/one"}'))
    outbox.register(sub(device="other", blob='{"endpoint": "https://push/two"}'))
    outbox.register(sub(profile="q", device="old", blob='{"endpoint": "https://push/one"}'))
    outbox.register(sub(device="new", blob='{"endpoint": "https://push/one"}'))          # default: both stay
    assert [s.device_id for s in outbox.subscriptions_for("p")] == ["new", "old", "other"]
    outbox.register(sub(device="new", blob='{"endpoint": "https://push/one"}'), supersede_same_blob=True)
    assert [s.device_id for s in outbox.subscriptions_for("p")] == ["new", "other"]
    assert outbox.subscription("q", "old") is not None                                   # other profiles untouched
    assert outbox.publish(alert("a")) == 2


def test_register_with_backfill_since_runs_in_the_same_transaction(outbox: Outbox, clock: FakeClock) -> None:
    clock.now = 100.0
    outbox.publish(alert("old", created_at=100.0))
    clock.now = 200.0
    outbox.publish(alert("recent", created_at=200.0))
    clock.now = 250.0
    assert outbox.register(sub(device="phone", created_at=250.0), backfill_since=150.0) == 1
    assert [r.alert_id for r in outbox.rows_for("recent")] == ["recent"]
    assert outbox.rows_for("old") == []
    assert outbox.register(sub(device="phone", created_at=250.0), backfill_since=150.0) == 0   # idempotent
    assert outbox.register(sub(device="laptop", created_at=250.0)) == 0                       # no backfill asked


def test_lease_can_be_limited_to_transports(outbox: Outbox) -> None:
    outbox.register(sub(device="phone", transport="fake"))
    outbox.register(sub(device="laptop", transport="webpush"))
    outbox.register(sub(device="tv", transport="apns"))
    outbox.publish(alert("a"))
    assert outbox.backlog_by_transport() == {"apns": 1, "fake": 1, "webpush": 1}
    assert outbox.lease(now=1000.0, limit=10, transports=set()) == []
    rows = outbox.lease(now=1000.0, limit=10, transports={"fake", "fcm"})
    assert [r.device_id for r in rows] == ["phone"]
    assert outbox.stats()["leased"] == 1 and outbox.stats()["pending"] == 2          # the others were not touched
    assert outbox.backlog_by_transport() == {"apns": 1, "webpush": 1}
    rows = outbox.lease(now=1000.0, limit=10)                                        # no filter: everything due
    assert sorted(r.device_id for r in rows) == ["laptop", "tv"]
    assert outbox.backlog_by_transport() == {}


def test_gone_result_on_a_settled_row_still_marks_the_device_gone(outbox: Outbox) -> None:
    """A stale duplicate send that comes back 410 says something true
    about the endpoint even though the row is already settled."""
    outbox.register(sub())
    outbox.publish(alert("a"))
    row = lease_one(outbox, 1000.0)
    outbox.mark(row.row_id, OK, now=1001.0)
    assert outbox.mark(row.row_id, GONE, now=1002.0) is RowState.DELIVERED
    assert outbox.row(row.row_id).attempts == 1 and len(outbox.attempts_for(row.row_id)) == 1
    assert outbox.subscription("p", "phone").gone is True
    assert outbox.publish(alert("b")) == 0


def test_database_file_and_its_companions_are_private(path: Path, clock: FakeClock) -> None:
    if not hasattr(os, "fchmod"):
        pytest.skip("POSIX permissions only")
    box = Outbox(path, clock)
    box.register(sub())
    try:
        for name in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
            if name.exists():
                assert stat.S_IMODE(name.stat().st_mode) == 0o600, name
    finally:
        box.close()
    with Outbox(path, clock) as again:                               # an existing file is left as it is
        assert again.subscription("p", "phone") is not None


def test_blob_with_a_lone_surrogate_is_refused_by_name_only(outbox: Outbox) -> None:
    """json.loads lets a "\ud800" escape through; sqlite would raise a
    UnicodeEncodeError whose args are the whole blob."""
    blob = json.loads('{"endpoint": "https://push/' + CANARY + '", "auth": "\\ud800"}')
    with pytest.raises(ValueError) as info:
        outbox.register(sub(blob=json.dumps(blob, ensure_ascii=False)))
    for text in (str(info.value), repr(info.value), repr(info.value.args), repr(info.value.__cause__)):
        assert CANARY not in text and "ud800" not in text
    assert outbox.subscription("p", "phone") is None
    with pytest.raises(ValueError):
        outbox.publish(alert("a", kind="k\ud800"))
    assert outbox.stats()["alerts"] == 0


def test_migrate_adds_the_newer_columns_to_an_older_file(path: Path, clock: FakeClock) -> None:
    """A file written before gone_at/pruned/attempts_base existed is
    upgraded in place; the version stamp stays 1 because the columns are
    additive."""
    raw = sqlite3.connect(path)
    raw.executescript("""
        CREATE TABLE schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL);
        INSERT INTO schema_version (id, version) VALUES (1, 1);
        CREATE TABLE alerts (id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, kind TEXT NOT NULL, title TEXT NOT NULL,
            body TEXT NOT NULL, priority INTEGER NOT NULL, dedupe_key TEXT, data TEXT NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE subscriptions (profile_id TEXT NOT NULL, device_id TEXT NOT NULL, transport TEXT NOT NULL,
            blob TEXT NOT NULL, created_at REAL NOT NULL, failures INTEGER NOT NULL DEFAULT 0,
            gone INTEGER NOT NULL DEFAULT 0, PRIMARY KEY (profile_id, device_id));
        CREATE TABLE outbox (row_id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id TEXT NOT NULL REFERENCES alerts (id),
            profile_id TEXT NOT NULL, device_id TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            next_due REAL NOT NULL, lease_until REAL NOT NULL DEFAULT 0, last_reason TEXT NOT NULL DEFAULT '',
            UNIQUE (alert_id, device_id));
        CREATE TABLE attempts (row_id INTEGER NOT NULL, attempt INTEGER NOT NULL, "at" REAL NOT NULL, ok INTEGER NOT NULL,
            retryable INTEGER NOT NULL, gone INTEGER NOT NULL, reason TEXT NOT NULL DEFAULT '', PRIMARY KEY (row_id, attempt));
        INSERT INTO subscriptions VALUES ('p', 'phone', 'fake', '{}', 1.0, 3, 1);
        INSERT INTO alerts VALUES ('a', 'p', 'k', 't', 'b', 1, NULL, '{}', 1.0);
        INSERT INTO outbox (alert_id, profile_id, device_id, state, attempts, next_due) VALUES ('a', 'p', 'phone', 'dead', 8, 1.0);
    """)
    raw.commit()
    raw.close()
    with Outbox(path, clock) as box:
        assert box.schema_version() == SCHEMA_VERSION == 1
        phone = box.subscription("p", "phone")
        assert phone.gone is True and phone.pruned is False and phone.failures == 3   # an old gone is a transport gone
        assert box.requeue(1) is True                                                  # attempts_base defaulted to 0
        cols = {r[1] for r in box._conn.execute("PRAGMA table_info(outbox)")}
        assert "attempts_base" in cols
        box.migrate()                                                                  # idempotent


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
