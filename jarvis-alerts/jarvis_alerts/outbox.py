"""The durable outbox: part 1 of the design in :mod:`jarvis_alerts.contracts`.

An alert is written here before anyone tries to send it, so a closed app, a
dropped socket or a crashed worker loses nothing.  One ``sqlite3`` file holds
four tables:

    alerts          what was published, immutable, one row per alert id
    subscriptions   one row per (profile_id, device_id); the blob is stored
                    as the app's own opaque data and never leaves this module
                    except through :meth:`Outbox.subscriptions_for` and
                    :meth:`Outbox.subscription`, which a transport needs
    outbox          one row per (alert, device) on its way out, in one of the
                    four :class:`RowState` states
    attempts        every delivery attempt ever made, for idempotency and
                    for the operator

Rows are *leased*, not popped (:meth:`Outbox.lease`): a worker gets a row for
``lease_s`` seconds and reports back with :meth:`Outbox.mark`, or hands it
back untouched with :meth:`Outbox.release`.  A worker that dies mid-send
never reports, its lease runs out, and the next ``lease`` hands the row to
someone else.  Retries follow the policy constants in contracts: attempt
``n`` fails -> wait ``backoff_seconds(n, jitter)``, give up after
``MAX_ATTEMPTS``, and a device that fails ``PRUNE_AFTER_FAILURES`` times in a
row is treated as gone -- for ``PRUNE_COOLDOWN_S``, after which ``lease``
probes it again.  A device the *transport* reports gone stays gone until it
registers again.

Nothing is ever lost silently.  An alert published while a device is
pruned still gets a row for it (parked, visible in :meth:`Outbox.stats` as
``pending_unreachable``); a repeat of a dedupe key repairs the earlier
alert instead of being discarded when that alert has not reached every
device; a dead letter can be put back with :meth:`Outbox.requeue`; and a
row that died only because an outage outlasted its retry budget
(``DeadReason.EXHAUSTED``, recorded on every DEAD transition with the
time of death) is put back *by the outbox itself*: :meth:`Outbox.revive_exhausted`,
which the worker calls every pass, re-queues every such row
``EXHAUSTED_RETRY_COOLDOWN_S`` after it died, with a fresh budget and its
attempt history kept, for as long as its alert is younger than
``ALERT_MAX_AGE_S`` and its device is reachable.  Rows dead for any other
reason (``PERMANENT``, ``GONE``, ``NO_SUBSCRIPTION``, ``NO_TRANSPORT``) are
never touched by it.

Determinism.  The outbox never reads the wall clock or ``random`` itself.  It
takes a ``clock`` callable at construction and ``now`` explicitly on ``lease``
and ``mark``; backoff jitter comes from a ``jitter`` callable (in production
``SeedFields.parse(seed).stream("alerts.backoff").random`` from
:mod:`lucifer_gen.seed`) or from the number the caller hands to ``mark``.

Privacy.  Nothing here logs, prints or puts a subscription blob in an
exception.  Subscriptions are referred to by (profile_id, device_id) only;
:class:`OutboxRow`, :meth:`Outbox.stats` and :meth:`Outbox.dead_letters`
carry no blob at all.  A new database file is created mode 0600, and
sqlite gives its ``-wal``/``-shm`` companions the same mode.

Concurrency.  Every mutation is one ``BEGIN IMMEDIATE`` transaction, so two
workers -- two threads sharing one :class:`Outbox`, or two connections to the
same file -- can never lease the same row.  A shared instance is additionally
guarded by a lock because one ``sqlite3`` connection is not itself
thread-safe.
"""

from __future__ import annotations

import contextlib
import json
import os
import sqlite3
import threading
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from jarvis_alerts.contracts import (
    ALERT_MAX_AGE_S,
    EXHAUSTED_RETRY_COOLDOWN_S,
    LEASE_S,
    MAX_ATTEMPTS,
    PRUNE_AFTER_FAILURES,
    PRUNE_COOLDOWN_S,
    Alert,
    DeadReason,
    DeliveryAttempt,
    OutboxRow,
    Priority,
    RowState,
    SendResult,
    Subscription,
    backoff_seconds,
    dead_reason_for,
)

__all__ = [
    "DEDUPE_WINDOW_S",
    "SCHEMA_VERSION",
    "DuplicateAlert",
    "Outbox",
    "OutboxError",
    "SchemaError",
    "UnknownRow",
    "no_jitter",
]

#: Two alerts for the same profile with the same ``dedupe_key`` published
#: less than this many seconds apart collapse into one.  See :meth:`Outbox.publish`.
DEDUPE_WINDOW_S = 300.0

#: Schema version 1.  Columns added since the first release are *additive*
#: (``NOT NULL DEFAULT``): :meth:`Outbox.migrate` adds them to an older file
#: in place, and an older reader ignores them, so the version stays 1.  It
#: bumps only for a change an older reader could misread.
SCHEMA_VERSION = 1

#: How long a connection waits for another writer before giving up.  Leases
#: are short transactions, so this is only ever hit under real contention.
BUSY_TIMEOUT_S = 30.0

#: Mode of a database file this module creates.  The file holds every
#: device's subscription blob, which is the owner's data.
DB_FILE_MODE = 0o600

Clock = Callable[[], float]
Jitter = Callable[[], float]


def no_jitter() -> float:
    """The default jitter source: none.  Backoff is then exactly half the
    nominal delay (``backoff_seconds`` scales by ``0.5 + 0.5 * jitter``)."""
    return 0.0


class OutboxError(Exception):
    """Base class for everything this module raises on purpose."""


class SchemaError(OutboxError):
    """The file was written by a different, unsupported schema version."""


class UnknownRow(OutboxError):
    """``mark`` or ``attempts_for`` named a row id that does not exist."""


class DuplicateAlert(OutboxError):
    """``publish`` was given an alert id that is already stored."""


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

_STATE_VALUES = ", ".join(f"'{s.value}'" for s in RowState)

_SCHEMA: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        id      INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS alerts (
        id          TEXT PRIMARY KEY,
        profile_id  TEXT NOT NULL,
        kind        TEXT NOT NULL,
        title       TEXT NOT NULL,
        body        TEXT NOT NULL,
        priority    INTEGER NOT NULL,
        dedupe_key  TEXT,
        data        TEXT NOT NULL,
        created_at  REAL NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS alerts_dedupe ON alerts (profile_id, dedupe_key, created_at)",
    "CREATE INDEX IF NOT EXISTS alerts_profile_created ON alerts (profile_id, created_at)",
    """
    CREATE TABLE IF NOT EXISTS subscriptions (
        profile_id  TEXT NOT NULL,
        device_id   TEXT NOT NULL,
        transport   TEXT NOT NULL,
        blob        TEXT NOT NULL,
        created_at  REAL NOT NULL,
        failures    INTEGER NOT NULL DEFAULT 0,
        gone        INTEGER NOT NULL DEFAULT 0,
        gone_at     REAL NOT NULL DEFAULT 0,
        pruned      INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (profile_id, device_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS outbox (
        row_id        INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_id      TEXT NOT NULL REFERENCES alerts (id),
        profile_id    TEXT NOT NULL,
        device_id     TEXT NOT NULL,
        state         TEXT NOT NULL CHECK (state IN ({_STATE_VALUES})),
        attempts      INTEGER NOT NULL DEFAULT 0,
        next_due      REAL NOT NULL,
        lease_until   REAL NOT NULL DEFAULT 0,
        last_reason   TEXT NOT NULL DEFAULT '',
        attempts_base INTEGER NOT NULL DEFAULT 0,
        dead_reason   TEXT NOT NULL DEFAULT '',
        dead_at       REAL NOT NULL DEFAULT 0,
        revivals      INTEGER NOT NULL DEFAULT 0,
        UNIQUE (alert_id, device_id)
    )
    """,
    "CREATE INDEX IF NOT EXISTS outbox_state_due ON outbox (state, next_due)",
    "CREATE INDEX IF NOT EXISTS outbox_device ON outbox (profile_id, device_id, state)",
    """
    CREATE TABLE IF NOT EXISTS attempts (
        row_id     INTEGER NOT NULL REFERENCES outbox (row_id) ON DELETE CASCADE,
        attempt    INTEGER NOT NULL,
        "at"       REAL NOT NULL,
        ok         INTEGER NOT NULL,
        retryable  INTEGER NOT NULL,
        gone       INTEGER NOT NULL,
        reason     TEXT NOT NULL DEFAULT '',
        PRIMARY KEY (row_id, attempt)
    )
    """,
)

#: Columns added after the first release, (table, column, definition).
#: ``migrate`` adds any that an existing file lacks.
_ADDED_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("subscriptions", "gone_at", "REAL NOT NULL DEFAULT 0"),
    ("subscriptions", "pruned", "INTEGER NOT NULL DEFAULT 0"),
    ("outbox", "attempts_base", "INTEGER NOT NULL DEFAULT 0"),
    ("outbox", "dead_reason", "TEXT NOT NULL DEFAULT ''"),
    ("outbox", "dead_at", "REAL NOT NULL DEFAULT 0"),
    ("outbox", "revivals", "INTEGER NOT NULL DEFAULT 0"),
)

#: Fills ``dead_reason`` / ``dead_at`` for the DEAD rows of a file written
#: before those columns existed, from each row's last recorded attempt
#: (the same reading :func:`contracts.dead_reason_for` makes of a result).
#: A DEAD row with no attempt on record is PERMANENT: nothing says it was an
#: outage, so it waits for the operator rather than being retried blindly.
_BACKFILL_DEAD_REASON = f"""
    UPDATE outbox SET
        dead_reason = COALESCE((
            SELECT CASE
                WHEN a.gone THEN '{DeadReason.GONE.value}'
                WHEN a.retryable THEN '{DeadReason.EXHAUSTED.value}'
                WHEN a.reason = 'no subscription' THEN '{DeadReason.NO_SUBSCRIPTION.value}'
                WHEN a.reason = 'no transport' THEN '{DeadReason.NO_TRANSPORT.value}'
                ELSE '{DeadReason.PERMANENT.value}' END
            FROM attempts a WHERE a.row_id = outbox.row_id ORDER BY a.attempt DESC LIMIT 1
        ), '{DeadReason.PERMANENT.value}'),
        dead_at = COALESCE((
            SELECT a."at" FROM attempts a WHERE a.row_id = outbox.row_id ORDER BY a.attempt DESC LIMIT 1
        ), 0)
    WHERE state = '{RowState.DEAD.value}' AND dead_reason = ''
"""

_ROW_COLUMNS = (
    "row_id, alert_id, profile_id, device_id, state, attempts, next_due, lease_until, last_reason, "
    "dead_reason, dead_at, attempts_base"
)
_SUB_COLUMNS = "profile_id, device_id, transport, blob, created_at, failures, gone, pruned"
#: A subscription the outbox will hand rows out for.
_LIVE = "s.gone = 0"
#: A subscription an alert is still fanned out to: live, or pruned (its
#: rows wait for the cooldown).  Never one the transport reported gone.
_REACHABLE = "(s.gone = 0 OR s.pruned = 1)"


def _revivable_where(table: str) -> str:
    """The rows :meth:`Outbox.revive_exhausted` puts back, as a WHERE
    fragment over ``table`` (``outbox`` or an alias of it), with three
    parameters: ``(EXHAUSTED_RETRY_COOLDOWN_S, now, ALERT_MAX_AGE_S, now)``
    -- DEAD as EXHAUSTED, dead for at least the cooldown, alert younger
    than the maximum age, device reachable (live or pruned)."""
    return f"""{table}.state = '{RowState.DEAD.value}'
        AND {table}.dead_reason = '{DeadReason.EXHAUSTED.value}'
        AND {table}.dead_at + ? <= ?
        AND EXISTS (SELECT 1 FROM alerts a WHERE a.id = {table}.alert_id AND a.created_at + ? > ?)
        AND EXISTS (
            SELECT 1 FROM subscriptions s
            WHERE s.profile_id = {table}.profile_id AND s.device_id = {table}.device_id AND {_REACHABLE}
        )"""


def _revivable_params(now: float) -> Tuple[float, float, float, float]:
    return (EXHAUSTED_RETRY_COOLDOWN_S, float(now), ALERT_MAX_AGE_S, float(now))


def _row_from_record(rec: Sequence[Any]) -> OutboxRow:
    return OutboxRow(
        row_id=int(rec[0]),
        alert_id=str(rec[1]),
        profile_id=str(rec[2]),
        device_id=str(rec[3]),
        state=RowState(rec[4]),
        attempts=int(rec[5]),
        next_due=float(rec[6]),
        lease_until=float(rec[7]),
        last_reason=str(rec[8]),
        dead_reason=DeadReason(rec[9]) if rec[9] else None,
        dead_at=float(rec[10]),
        attempts_base=int(rec[11]),
    )


def _row_as_dict(row: OutboxRow) -> Dict[str, Any]:
    """JSON-friendly view of a row for :meth:`Outbox.stats`.  No blob here
    by construction: :class:`OutboxRow` never carries one."""
    return {
        "row_id": row.row_id,
        "alert_id": row.alert_id,
        "profile_id": row.profile_id,
        "device_id": row.device_id,
        "state": row.state.value,
        "attempts": row.attempts,
        "next_due": row.next_due,
        "lease_until": row.lease_until,
        "last_reason": row.last_reason,
        "dead_reason": None if row.dead_reason is None else row.dead_reason.value,
        "dead_at": row.dead_at,
        "attempts_base": row.attempts_base,
    }


def _storable_text(value: Any) -> bool:
    """``str`` that sqlite can bind: no lone surrogates.  (``json.loads`` lets
    a ``"\\ud800"`` escape through; binding it raises ``UnicodeEncodeError``,
    whose ``args`` would carry the whole text.)"""
    if not isinstance(value, str):
        return False
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _check_subscription(sub: Subscription) -> None:
    """Reject a malformed registration.  The message names the subscription
    by (profile_id, device_id) only; the blob is never formatted."""
    where = f"subscription (profile_id={sub.profile_id!r}, device_id={sub.device_id!r})"
    for name in ("profile_id", "device_id", "transport"):
        value = getattr(sub, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{where}: {name} must be a non-empty string")
        if not _storable_text(value):
            raise ValueError(f"{where}: {name} is not valid UTF-8 text")
    if not isinstance(sub.blob, str):
        raise ValueError(f"{where}: blob must be a str (opaque JSON text)")
    if not _storable_text(sub.blob):
        raise ValueError(f"{where}: blob is not valid UTF-8 text (content not shown)")
    if isinstance(sub.created_at, bool) or not isinstance(sub.created_at, (int, float)):
        raise ValueError(f"{where}: created_at must be a number")


def _check_alert(alert: Alert) -> str:
    """Validate an alert and return its ``data`` serialised as JSON.
    Serialising up front means a bad payload fails before any write."""
    for name in ("id", "profile_id", "kind"):
        value = getattr(alert, name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"alert {alert.id!r}: {name} must be a non-empty string")
    for name in ("id", "profile_id", "kind", "title", "body"):
        if not _storable_text(getattr(alert, name)):
            raise ValueError(f"alert {alert.id!r}: {name} must be valid UTF-8 text")
    if isinstance(alert.created_at, bool) or not isinstance(alert.created_at, (int, float)):
        raise ValueError(f"alert {alert.id!r}: created_at must be a number")
    Priority(alert.priority)  # raises ValueError on an unknown level
    if alert.dedupe_key is not None and not _storable_text(alert.dedupe_key):
        raise ValueError(f"alert {alert.id!r}: dedupe_key must be a string or None")
    return json.dumps(alert.data, sort_keys=True, separators=(",", ":"))


def _check_jitter(value: float) -> float:
    """``backoff_seconds`` wants jitter in [0, 1).  ``Stream.random`` delivers
    exactly that; 1.0 is tolerated so a caller can ask for the maximum."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"jitter must be a number in [0, 1], not {type(value).__name__}")
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"jitter must be in [0, 1], got {value!r}")
    return float(value)


def _create_private_file(path: str) -> None:
    """Create ``path`` mode 0600 if it does not exist yet, so the blobs it
    will hold are never world-readable.  sqlite gives the ``-wal`` and
    ``-shm`` files the database file's mode."""
    if path in (":memory:", "") or path.startswith("file:"):
        return
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, DB_FILE_MODE)
    except FileExistsError:
        return
    except OSError:
        return  # sqlite will report the real problem when it opens the path
    os.close(fd)


# --------------------------------------------------------------------------
# The outbox
# --------------------------------------------------------------------------


class Outbox:
    """One ``sqlite3`` file of alerts, subscriptions and outbox rows.

    ``path`` may be a filesystem path or ``":memory:"`` (private to this
    instance; two workers need a file).  ``clock`` supplies "now" for
    :meth:`publish`, :meth:`backfill` and :meth:`requeue`; :meth:`lease` and
    :meth:`mark` take ``now`` explicitly because a worker measures time once
    per batch.  ``jitter`` is drawn once per backoff by :meth:`mark` when the
    caller does not pass a value; wire it to a :class:`lucifer_gen.seed.Stream`::

        stream = SeedFields.parse(seed).stream("alerts.backoff")
        outbox = Outbox(path, clock=time.time, jitter=stream.random)

    The connection is in autocommit mode so that every transaction boundary
    is an explicit ``BEGIN`` / ``COMMIT`` / ``ROLLBACK`` in this class.
    """

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        clock: Clock,
        jitter: Jitter = no_jitter,
    ) -> None:
        self.path = os.fspath(path)
        self._clock = clock
        self._jitter = jitter
        self._lock = threading.Lock()
        _create_private_file(self.path)
        self._conn = sqlite3.connect(
            self.path,
            timeout=BUSY_TIMEOUT_S,
            isolation_level=None,
            check_same_thread=False,
        )
        # Pragmas must run outside a transaction; autocommit mode does that.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    def __repr__(self) -> str:
        return f"Outbox({self.path!r})"

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "Outbox":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- transactions -------------------------------------------------------

    @contextlib.contextmanager
    def _transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Cursor]:
        """Run the block in one transaction under the instance lock.

        ``IMMEDIATE`` takes the write lock up front, so a mutation never has
        to upgrade a read lock halfway through and fail after it has already
        changed rows.  Reads use ``DEFERRED`` and get a consistent snapshot.
        An exception anywhere -- including from ``COMMIT`` itself -- rolls
        the whole transaction back, so no mutation can leave partial rows.
        """
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(f"BEGIN {mode}")
            try:
                yield cur
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    try:
                        self._conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass  # the original exception is the one worth raising
                raise
            finally:
                cur.close()

    # -- schema -------------------------------------------------------------

    def migrate(self) -> None:
        """Create any missing tables and columns and stamp :data:`SCHEMA_VERSION`.

        Idempotent.  A file from before a column was added gets it in place
        (``ALTER TABLE ... ADD COLUMN`` with a default), which an older
        reader of the same file ignores.  A file stamped with another
        version raises :class:`SchemaError`: there is no upgrade path for
        that, and guessing one would silently corrupt the delivery record.

        A file from before ``dead_reason`` existed has its DEAD rows
        classified from their last recorded attempt (a retryable last
        failure is EXHAUSTED, and so is revived like any other), so the
        fix for an outage that outlasted the budget reaches rows that
        died before the upgrade too.  A DEAD row with no attempt on record
        is PERMANENT and waits for the operator.
        """
        with self._transaction() as cur:
            for statement in _SCHEMA:
                cur.execute(statement)
            added: List[str] = []
            for table, column, definition in _ADDED_COLUMNS:
                present = {str(r[1]) for r in cur.execute(f"PRAGMA table_info({table})")}
                if column not in present:
                    cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                    added.append(column)
            if "dead_reason" in added:
                cur.execute(_BACKFILL_DEAD_REASON)
            row = cur.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO schema_version (id, version) VALUES (1, ?)", (SCHEMA_VERSION,)
                )
            elif int(row[0]) != SCHEMA_VERSION:
                raise SchemaError(
                    f"{self.path}: schema version {row[0]} is not the supported "
                    f"version {SCHEMA_VERSION}"
                )

    def schema_version(self) -> int:
        row = self._conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        if row is None:
            raise SchemaError(f"{self.path}: schema_version row is missing")
        return int(row[0])

    # -- subscriptions ------------------------------------------------------

    def register(
        self,
        sub: Subscription,
        backfill_since: Optional[float] = None,
        *,
        supersede_same_blob: bool = False,
    ) -> int:
        """Upsert one device's subscription (design part 2, the registry).

        Re-registering a (profile_id, device_id) replaces transport, blob and
        created_at and clears ``gone``, ``pruned`` and ``failures``: a fresh
        registration is the device saying it is reachable again, whatever
        the stored copy of ``sub`` says.  ``failures``, ``gone`` and
        ``pruned`` on the argument are therefore ignored.

        ``backfill_since``: when given, :meth:`backfill` runs for the
        profile in the same transaction, so an alert published concurrently
        either sees the new subscription (fan-out) or is seen by the
        backfill; there is no gap between the two.  Returns the rows that
        backfill created (0 when ``backfill_since`` is ``None``).

        ``supersede_same_blob``: also forget any *other* device id of the
        same profile registered with byte-identical blob text.  A browser
        that lost its device id (site data cleared) but kept its push
        subscription would otherwise be two devices and get every push
        twice.  Off by default because tests and fixtures may share one
        placeholder blob between devices; the reference subscribe endpoint
        in ``client/README.md`` turns it on.
        """
        _check_subscription(sub)
        with self._transaction() as cur:
            if supersede_same_blob:
                cur.execute(
                    "DELETE FROM subscriptions WHERE profile_id = ? AND device_id != ? AND blob = ?",
                    (sub.profile_id, sub.device_id, sub.blob),
                )
            cur.execute(
                """
                INSERT INTO subscriptions
                    (profile_id, device_id, transport, blob, created_at, failures, gone, gone_at, pruned)
                VALUES (?, ?, ?, ?, ?, 0, 0, 0, 0)
                ON CONFLICT (profile_id, device_id) DO UPDATE SET
                    transport  = excluded.transport,
                    blob       = excluded.blob,
                    created_at = excluded.created_at,
                    failures   = 0,
                    gone       = 0,
                    gone_at    = 0,
                    pruned     = 0
                """,
                (sub.profile_id, sub.device_id, sub.transport, sub.blob, float(sub.created_at)),
            )
            if backfill_since is None:
                return 0
            return self._backfill(cur, sub.profile_id, float(backfill_since), self._clock())

    def unregister(self, profile_id: str, device_id: str) -> bool:
        """Forget a device.  Returns whether it was registered.

        The device's outbox rows are kept as history.  Its PENDING rows are
        parked: :meth:`lease` only hands out rows whose device has a live
        subscription, so they wait -- visible in :meth:`stats` as
        ``pending_unreachable`` -- until the device registers again or for
        ever.  A LEASED row stays with its worker, whose ``mark`` decides.
        """
        with self._transaction() as cur:
            cur.execute(
                "DELETE FROM subscriptions WHERE profile_id = ? AND device_id = ?",
                (profile_id, device_id),
            )
            return cur.rowcount > 0

    def subscriptions_for(self, profile_id: str) -> List[Subscription]:
        """Every live (not gone, not pruned) subscription of a profile, by
        device_id.

        This and :meth:`subscription` are the only places the blob comes
        out, because a transport cannot send without it.
        """
        with self._transaction("DEFERRED") as cur:
            rows = cur.execute(
                f"""
                SELECT {_SUB_COLUMNS}
                FROM subscriptions WHERE profile_id = ? AND gone = 0 ORDER BY device_id
                """,
                (profile_id,),
            ).fetchall()
        return [self._subscription_from_record(r) for r in rows]

    def subscription(self, profile_id: str, device_id: str) -> Optional[Subscription]:
        """One device's subscription, gone or not, or ``None`` if unknown."""
        with self._transaction("DEFERRED") as cur:
            return self._load_subscription(cur, profile_id, device_id)

    def pruned_subscriptions(self) -> List[Tuple[str, str]]:
        """(profile_id, device_id) of every device currently pruned (gone
        by failure count, waiting for its cooldown), sorted."""
        with self._transaction("DEFERRED") as cur:
            recs = cur.execute(
                "SELECT profile_id, device_id FROM subscriptions WHERE gone = 1 AND pruned = 1 "
                "ORDER BY profile_id, device_id"
            ).fetchall()
        return [(str(p), str(d)) for p, d in recs]

    @staticmethod
    def _subscription_from_record(rec: Sequence[Any]) -> Subscription:
        return Subscription(
            profile_id=str(rec[0]),
            device_id=str(rec[1]),
            transport=str(rec[2]),
            blob=str(rec[3]),
            created_at=float(rec[4]),
            failures=int(rec[5]),
            gone=bool(rec[6]),
            pruned=bool(rec[7]),
        )

    def _load_subscription(
        self, cur: sqlite3.Cursor, profile_id: str, device_id: str
    ) -> Optional[Subscription]:
        rec = cur.execute(
            f"SELECT {_SUB_COLUMNS} FROM subscriptions WHERE profile_id = ? AND device_id = ?",
            (profile_id, device_id),
        ).fetchone()
        return None if rec is None else self._subscription_from_record(rec)

    # -- publishing ---------------------------------------------------------

    def publish(self, alert: Alert) -> int:
        """Store an alert and fan it out; return the outbox rows created.

        Dedupe window.  When ``alert.dedupe_key`` is set and this profile
        already has an alert with the same key whose ``created_at`` is
        later than ``clock() - DEDUPE_WINDOW_S`` (300 s), the new alert is
        not stored.  The window is half-open, so a repeat exactly 300 s
        after the original goes out again.  The comparison uses the outbox
        clock against the stored alert's ``created_at``, which the app
        supplied from its own clock; the two are expected to agree.

        A repeat is not discarded blindly, though: it *repairs* the earlier
        alert.  For every reachable device of the profile (live or pruned)
        that has no row for the earlier alert, a PENDING row is created;
        every DEAD row of the earlier alert on such a device is re-queued
        with a fresh attempt budget (see :meth:`requeue`).  The return
        value is the number of rows created or re-queued, 0 when the
        earlier alert already covers every device.  So a keyed alert that
        died in an outage, or that a device registered later never got, is
        re-driven by the app's natural reaction -- firing it again.

        Otherwise the alert is inserted and one PENDING row is created per
        reachable subscription of the profile, due immediately (``next_due
        = clock()``); a pruned device's row is parked until its cooldown.
        A profile with no such subscription still gets its alert stored --
        also returning 0 -- so that :meth:`backfill` can deliver it to a
        device registered later.  ``alert(id)`` tells the zero cases apart.

        Publishing an id that is already stored raises
        :class:`DuplicateAlert`: ids are the app's, and the same id with
        different content would otherwise be silently lost.
        """
        data_json = _check_alert(alert)
        now = self._clock()
        with self._transaction() as cur:
            earlier = self._duplicate_of(cur, alert, now)
            if earlier is not None:
                return self._repair(cur, earlier, alert.profile_id, now)
            self._insert_alert(cur, alert, data_json)
            return self._fan_out(cur, alert, now)

    def _duplicate_of(self, cur: sqlite3.Cursor, alert: Alert, now: float) -> Optional[str]:
        """The id of the newest stored alert this one repeats, or ``None``."""
        if alert.dedupe_key is None:
            return None
        hit = cur.execute(
            """
            SELECT id FROM alerts
            WHERE profile_id = ? AND dedupe_key = ? AND created_at > ?
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """,
            (alert.profile_id, alert.dedupe_key, now - DEDUPE_WINDOW_S),
        ).fetchone()
        return None if hit is None else str(hit[0])

    def _insert_alert(self, cur: sqlite3.Cursor, alert: Alert, data_json: str) -> None:
        try:
            cur.execute(
                """
                INSERT INTO alerts
                    (id, profile_id, kind, title, body, priority, dedupe_key, data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    alert.id,
                    alert.profile_id,
                    alert.kind,
                    alert.title,
                    alert.body,
                    int(alert.priority),
                    alert.dedupe_key,
                    data_json,
                    float(alert.created_at),
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateAlert(f"alert {alert.id!r} is already stored") from exc

    def _fan_out(self, cur: sqlite3.Cursor, alert: Alert, now: float) -> int:
        """One PENDING row per reachable subscription of the alert's profile.
        Kept as its own step so a test can break it and prove the alert
        insert rolls back with it."""
        cur.execute(
            f"""
            INSERT INTO outbox
                (alert_id, profile_id, device_id, state, attempts, next_due, lease_until, last_reason)
            SELECT ?, s.profile_id, s.device_id, ?, 0, ?, 0, ''
            FROM subscriptions s WHERE s.profile_id = ? AND {_REACHABLE}
            ORDER BY s.device_id
            """,
            (alert.id, RowState.PENDING.value, now, alert.profile_id),
        )
        return cur.rowcount

    def _repair(self, cur: sqlite3.Cursor, alert_id: str, profile_id: str, now: float) -> int:
        """Make ``alert_id`` reach every reachable device: see :meth:`publish`."""
        requeued = self._requeue(
            cur, now,
            f"""alert_id = ? AND EXISTS (
                  SELECT 1 FROM subscriptions s
                  WHERE s.profile_id = outbox.profile_id AND s.device_id = outbox.device_id
                    AND {_REACHABLE}
              )""",
            (alert_id,),
            reason="requeued: dedupe key repeated",
        )
        cur.execute(
            f"""
            INSERT INTO outbox
                (alert_id, profile_id, device_id, state, attempts, next_due, lease_until, last_reason)
            SELECT ?, s.profile_id, s.device_id, ?, 0, ?, 0, ''
            FROM subscriptions s
            WHERE s.profile_id = ? AND {_REACHABLE}
              AND NOT EXISTS (
                  SELECT 1 FROM outbox o WHERE o.alert_id = ? AND o.device_id = s.device_id
              )
            ORDER BY s.device_id
            """,
            (alert_id, RowState.PENDING.value, now, profile_id, alert_id),
        )
        return requeued + cur.rowcount

    def backfill(self, profile_id: str, since: float) -> int:
        """Create rows for stored alerts a reachable device has none for.

        For every alert of ``profile_id`` with ``created_at > since`` and
        every reachable subscription (live or pruned) of that profile,
        insert a PENDING row due now unless a row for that (alert, device)
        already exists in any state.  This is how a device registered
        after an alert was published still receives it.  Returns the rows
        created.  The dedupe window does not apply: these alerts were
        already accepted.
        """
        now = self._clock()
        with self._transaction() as cur:
            return self._backfill(cur, profile_id, float(since), now)

    @staticmethod
    def _backfill(cur: sqlite3.Cursor, profile_id: str, since: float, now: float) -> int:
        cur.execute(
            f"""
            INSERT INTO outbox
                (alert_id, profile_id, device_id, state, attempts, next_due, lease_until, last_reason)
            SELECT a.id, a.profile_id, s.device_id, ?, 0, ?, 0, ''
            FROM alerts a
            JOIN subscriptions s ON s.profile_id = a.profile_id AND {_REACHABLE}
            WHERE a.profile_id = ? AND a.created_at > ?
              AND NOT EXISTS (
                  SELECT 1 FROM outbox o
                  WHERE o.alert_id = a.id AND o.device_id = s.device_id
              )
            ORDER BY a.created_at, a.id, s.device_id
            """,
            (RowState.PENDING.value, now, profile_id, since),
        )
        return cur.rowcount

    # -- leasing ------------------------------------------------------------

    def lease(
        self,
        now: float,
        limit: int,
        lease_s: float = LEASE_S,
        transports: Optional[Iterable[str]] = None,
    ) -> List[OutboxRow]:
        """Hand out up to ``limit`` due rows, each held until ``now + lease_s``.

        In one ``BEGIN IMMEDIATE`` transaction:

        0. Revive: every subscription pruned by failure count whose
           ``PRUNE_COOLDOWN_S`` has passed becomes live again with
           ``failures`` reset, so its parked rows are probed once more.
        1. Reclaim: every LEASED row whose ``lease_until < now`` goes back
           to PENDING (its worker died or stalled).  Its ``attempts`` are
           untouched, because no ``mark`` was recorded for it; its
           ``last_reason`` becomes ``"lease expired"`` so an operator can see
           what happened.  A worker that crashes before ``mark`` therefore
           re-leases the row without limit; a worker whose *mark* fails
           remembers the result and applies it on the next pass
           (:class:`jarvis_alerts.worker.Worker`).
        2. Select PENDING rows with ``next_due <= now`` whose device has a
           live subscription -- and, when ``transports`` is given, one whose
           transport is named in it -- ordered by alert priority (high
           first), then ``next_due``, then ``row_id``; take ``limit`` of them.
        3. Move them to LEASED with ``lease_until = now + lease_s``.

        Because the write lock is held from step 1, two callers -- threads
        or separate connections -- can never receive the same row.  Rows
        whose device has no live subscription are never handed out; they
        wait for a re-registration or a cooldown.  ``transports`` lets a
        process that can only send through some transports leave the other
        rows untouched for a process that can (an empty collection leases
        nothing).
        """
        if lease_s <= 0:
            raise ValueError(f"lease_s must be positive, got {lease_s!r}")
        if limit <= 0:
            return []
        names: Optional[List[str]] = None
        if transports is not None:
            names = sorted({str(t) for t in transports})
            if not names:
                return []
        until = float(now) + float(lease_s)
        with self._transaction() as cur:
            cur.execute(
                """
                UPDATE subscriptions SET gone = 0, pruned = 0, failures = 0, gone_at = 0
                WHERE gone = 1 AND pruned = 1 AND gone_at + ? <= ?
                """,
                (PRUNE_COOLDOWN_S, now),
            )
            cur.execute(
                """
                UPDATE outbox SET state = ?, lease_until = 0, last_reason = 'lease expired'
                WHERE state = ? AND lease_until < ?
                """,
                (RowState.PENDING.value, RowState.LEASED.value, now),
            )
            transport_filter = ""
            params: List[Any] = [RowState.PENDING.value, now]
            if names is not None:
                transport_filter = f" AND s.transport IN ({', '.join('?' * len(names))})"
                params.extend(names)
            params.append(int(limit))
            picked = cur.execute(
                f"""
                SELECT {", ".join("o." + c.strip() for c in _ROW_COLUMNS.split(","))}
                FROM outbox o
                JOIN alerts a ON a.id = o.alert_id
                JOIN subscriptions s ON s.profile_id = o.profile_id AND s.device_id = o.device_id
                WHERE o.state = ? AND o.next_due <= ? AND {_LIVE}{transport_filter}
                ORDER BY a.priority DESC, o.next_due ASC, o.row_id ASC
                LIMIT ?
                """,
                params,
            ).fetchall()
            cur.executemany(
                "UPDATE outbox SET state = ?, lease_until = ? WHERE row_id = ?",
                [(RowState.LEASED.value, until, rec[0]) for rec in picked],
            )
        rows = []
        for rec in picked:
            row = _row_from_record(rec)
            row.state = RowState.LEASED
            row.lease_until = until
            rows.append(row)
        return rows

    def release(self, row_id: int, reason: str = "") -> RowState:
        """Hand a LEASED row back without an attempt: it returns to PENDING
        at once, keeping its ``next_due`` and ``attempts``, with
        ``last_reason`` set to ``reason``.  For a worker that leased a row
        it then finds it must not send (the device was pruned by a sibling
        row in the same batch, say).  A row in any other state is left
        alone; the state is returned either way.  Unknown ids raise
        :class:`UnknownRow`.
        """
        with self._transaction() as cur:
            row = self._load_row(cur, row_id)
            if row.state is not RowState.LEASED:
                return row.state
            cur.execute(
                "UPDATE outbox SET state = ?, lease_until = 0, last_reason = ? WHERE row_id = ?",
                (RowState.PENDING.value, str(reason or ""), row_id),
            )
            return RowState.PENDING

    # -- marking ------------------------------------------------------------

    def mark(
        self,
        row_id: int,
        result: SendResult,
        now: float,
        jitter: Optional[float] = None,
    ) -> RowState:
        """Record one delivery attempt and move the row.  Returns the new state.

        Transitions (design part 3, at-least-once with a full record):

        * every call on an unsettled row appends a :class:`DeliveryAttempt`
          and bumps the row's ``attempts``, so ``len(attempts_for(row)) ==
          row.attempts`` always;
        * ``ok``            -> DELIVERED; the device's ``failures`` reset to 0;
        * ``gone``          -> DEAD as ``GONE``; the device is marked gone
          (by the transport);
        * ``retryable``     -> DEAD as ``EXHAUSTED`` once the row's budget
          is spent, else PENDING with ``next_due = now + backoff_seconds(n,
          jitter)`` where ``n`` counts this failure within the budget (the
          n-th failure waits the n-th delay); the budget is ``MAX_ATTEMPTS``
          attempts since the row was created or last re-queued
          (``attempts_base``).  The device's ``failures`` goes up by one
          and at ``PRUNE_AFTER_FAILURES`` the device is marked gone and
          ``pruned``, for ``PRUNE_COOLDOWN_S``;
        * anything else     -> DEAD as ``PERMANENT``, or as the
          ``NO_SUBSCRIPTION`` / ``NO_TRANSPORT`` the worker tagged the
          result with; the device's ``failures`` is untouched, because a
          permanent, non-gone error (bad payload, say) says nothing about
          the endpoint.

        Every DEAD transition records ``dead_reason`` and ``dead_at``
        (:func:`contracts.dead_reason_for`); an EXHAUSTED row is the one
        :meth:`revive_exhausted` brings back.

        ``jitter`` is a number in [0, 1]; when ``None`` the outbox draws one
        from its jitter source, and only when a backoff is actually needed,
        so a seeded stream is consumed in a reproducible order.

        A late ``mark`` -- the worker's lease had already expired -- is
        still applied when the row is PENDING, because the send did happen
        and recording it is what prevents a needless repeat.  A ``mark`` on
        a DELIVERED or DEAD row leaves the row alone and returns that
        state; the client collapses the duplicate by alert id.  One thing
        is still taken from it: a ``gone`` result marks the device gone,
        because that is a fact about the endpoint, not the row.  An
        unknown ``row_id`` raises :class:`UnknownRow`.
        """
        with self._transaction() as cur:
            row = self._load_row(cur, row_id)
            if row.state in (RowState.DELIVERED, RowState.DEAD):
                if result.gone and not result.ok:
                    self._set_gone(cur, row.profile_id, row.device_id, now, pruned=False)
                return row.state
            attempt_no = row.attempts + 1
            budget_no = attempt_no - row.attempts_base
            self._record_attempt(cur, row_id, attempt_no, now, result)

            next_due = row.next_due
            dead_reason = dead_reason_for(result, budget_spent=budget_no >= MAX_ATTEMPTS)
            if result.ok:
                new_state = RowState.DELIVERED
                self._set_failures(cur, row.profile_id, row.device_id, 0)
            elif result.gone:
                new_state = RowState.DEAD
                self._set_gone(cur, row.profile_id, row.device_id, now, pruned=False)
            elif result.retryable:
                if dead_reason is not None:
                    new_state = RowState.DEAD
                else:
                    new_state = RowState.PENDING
                    drawn = self._jitter() if jitter is None else jitter
                    next_due = float(now) + backoff_seconds(budget_no, _check_jitter(drawn))
                failures = self._bump_failures(cur, row.profile_id, row.device_id)
                if failures is not None and failures >= PRUNE_AFTER_FAILURES:
                    self._set_gone(cur, row.profile_id, row.device_id, now, pruned=True)
            else:
                new_state = RowState.DEAD

            self._update_row(cur, row_id, new_state, attempt_no, next_due, result.reason,
                             dead_reason, now)
            return new_state

    def requeue(self, row_id: int) -> bool:
        """Put a DEAD row back to PENDING, due now, with a fresh budget of
        ``MAX_ATTEMPTS`` attempts and a fresh backoff schedule; the recorded
        attempts stay as history.  The operator's way back from a dead
        letter (``cli requeue``), whatever its ``dead_reason``.  Returns
        whether the row was DEAD; any other state is left alone.  Unknown
        ids raise :class:`UnknownRow`.
        """
        now = self._clock()
        with self._transaction() as cur:
            self._load_row(cur, row_id)
            return self._requeue(cur, now, "row_id = ?", (row_id,)) > 0

    def requeue_dead(self, profile_id: Optional[str] = None, device_id: Optional[str] = None) -> int:
        """:meth:`requeue` every DEAD row, or those of one profile, or of one
        device.  Returns how many were re-queued."""
        where, params = "1 = 1", []
        if profile_id is not None:
            where, params = "profile_id = ?", [profile_id]
            if device_id is not None:
                where, params = "profile_id = ? AND device_id = ?", [profile_id, device_id]
        elif device_id is not None:
            raise ValueError("device_id needs a profile_id")
        now = self._clock()
        with self._transaction() as cur:
            return self._requeue(cur, now, where, params)

    def revive_exhausted(self, now: float) -> int:
        """Re-queue, as :meth:`requeue` does, every row that is DEAD as
        ``EXHAUSTED``, died at least ``EXHAUSTED_RETRY_COOLDOWN_S`` before
        ``now``, belongs to an alert younger than ``ALERT_MAX_AGE_S`` at
        ``now`` (``created_at + ALERT_MAX_AGE_S > now``) and whose device
        is reachable (live, or pruned and waiting for its cooldown).
        Returns how many.  The worker calls this at the start of every
        pass, so an outage that outlasts the retry budget costs a cooldown,
        not the alert.  Rows DEAD for any other reason, rows of an alert
        past the maximum age and rows of a device the transport reported
        gone (until it registers again) are never touched.  Each revival
        is counted on the row; :meth:`stats` sums them as ``revived``.
        """
        with self._transaction() as cur:
            return self._requeue(
                cur, now, _revivable_where("outbox"), _revivable_params(now),
                reason="requeued: exhausted, cooldown passed", revival=True,
            )

    @staticmethod
    def _requeue(
        cur: sqlite3.Cursor,
        now: float,
        where: str,
        params: Sequence[Any],
        *,
        reason: str = "requeued",
        revival: bool = False,
    ) -> int:
        """DEAD -> PENDING for the rows ``where`` selects: due now, budget
        reset (``attempts_base = attempts``), dead reason and time cleared,
        history kept.  ``revival`` counts it as an automatic one."""
        cur.execute(
            f"""
            UPDATE outbox
            SET state = ?, next_due = ?, lease_until = 0, attempts_base = attempts,
                last_reason = ?, dead_reason = '', dead_at = 0,
                revivals = revivals + ?
            WHERE state = ? AND {where}
            """,
            (RowState.PENDING.value, float(now), str(reason), int(revival), RowState.DEAD.value, *params),
        )
        return cur.rowcount

    def _load_row(self, cur: sqlite3.Cursor, row_id: int) -> OutboxRow:
        rec = cur.execute(
            f"SELECT {_ROW_COLUMNS} FROM outbox WHERE row_id = ?", (row_id,)
        ).fetchone()
        if rec is None:
            raise UnknownRow(f"outbox row {row_id!r} does not exist")
        return _row_from_record(rec)

    def _record_attempt(
        self, cur: sqlite3.Cursor, row_id: int, attempt_no: int, now: float, result: SendResult
    ) -> None:
        cur.execute(
            """
            INSERT INTO attempts (row_id, attempt, "at", ok, retryable, gone, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row_id,
                attempt_no,
                float(now),
                int(bool(result.ok)),
                int(bool(result.retryable)),
                int(bool(result.gone)),
                str(result.reason or ""),
            ),
        )

    def _update_row(
        self,
        cur: sqlite3.Cursor,
        row_id: int,
        state: RowState,
        attempts: int,
        next_due: float,
        reason: str,
        dead_reason: Optional[DeadReason] = None,
        now: float = 0.0,
    ) -> None:
        """The last step of ``mark``; a test breaks it to prove the attempt
        record and the subscription change roll back with it.  A DEAD
        transition records why and when; any other clears both."""
        dead = state is RowState.DEAD
        if dead and dead_reason is None:
            raise ValueError("a DEAD transition needs a dead_reason")
        cur.execute(
            """
            UPDATE outbox
            SET state = ?, attempts = ?, next_due = ?, lease_until = 0, last_reason = ?,
                dead_reason = ?, dead_at = ?
            WHERE row_id = ?
            """,
            (
                state.value, attempts, float(next_due), str(reason or ""),
                dead_reason.value if dead else "", float(now) if dead else 0.0, row_id,
            ),
        )

    @staticmethod
    def _set_failures(cur: sqlite3.Cursor, profile_id: str, device_id: str, value: int) -> None:
        cur.execute(
            "UPDATE subscriptions SET failures = ? WHERE profile_id = ? AND device_id = ?",
            (value, profile_id, device_id),
        )

    @staticmethod
    def _bump_failures(cur: sqlite3.Cursor, profile_id: str, device_id: str) -> Optional[int]:
        """``failures += 1``; returns the new count, or ``None`` when the
        device is no longer registered."""
        cur.execute(
            "UPDATE subscriptions SET failures = failures + 1 WHERE profile_id = ? AND device_id = ?",
            (profile_id, device_id),
        )
        rec = cur.execute(
            "SELECT failures FROM subscriptions WHERE profile_id = ? AND device_id = ?",
            (profile_id, device_id),
        ).fetchone()
        return None if rec is None else int(rec[0])

    @staticmethod
    def _set_gone(cur: sqlite3.Cursor, profile_id: str, device_id: str, now: float, *, pruned: bool) -> None:
        """Mark a device gone.  By the transport (``pruned=False``): always,
        and it overrides a prune.  By failure count (``pruned=True``): only
        a live device, so a transport-reported gone is never downgraded to
        one that expires."""
        if pruned:
            cur.execute(
                "UPDATE subscriptions SET gone = 1, pruned = 1, gone_at = ? "
                "WHERE profile_id = ? AND device_id = ? AND gone = 0",
                (float(now), profile_id, device_id),
            )
        else:
            cur.execute(
                "UPDATE subscriptions SET gone = 1, pruned = 0, gone_at = ? "
                "WHERE profile_id = ? AND device_id = ?",
                (float(now), profile_id, device_id),
            )

    # -- inspection ---------------------------------------------------------

    def row(self, row_id: int) -> Optional[OutboxRow]:
        with self._transaction("DEFERRED") as cur:
            rec = cur.execute(
                f"SELECT {_ROW_COLUMNS} FROM outbox WHERE row_id = ?", (row_id,)
            ).fetchone()
        return None if rec is None else _row_from_record(rec)

    def rows_for(self, alert_id: str) -> List[OutboxRow]:
        """Every outbox row of one alert, oldest first."""
        with self._transaction("DEFERRED") as cur:
            recs = cur.execute(
                f"SELECT {_ROW_COLUMNS} FROM outbox WHERE alert_id = ? ORDER BY row_id",
                (alert_id,),
            ).fetchall()
        return [_row_from_record(r) for r in recs]

    def alert(self, alert_id: str) -> Optional[Alert]:
        with self._transaction("DEFERRED") as cur:
            rec = cur.execute(
                """
                SELECT id, profile_id, kind, title, body, priority, dedupe_key, data, created_at
                FROM alerts WHERE id = ?
                """,
                (alert_id,),
            ).fetchone()
        if rec is None:
            return None
        return Alert(
            id=str(rec[0]),
            profile_id=str(rec[1]),
            kind=str(rec[2]),
            title=str(rec[3]),
            body=str(rec[4]),
            created_at=float(rec[8]),
            priority=Priority(int(rec[5])),
            dedupe_key=None if rec[6] is None else str(rec[6]),
            data=json.loads(rec[7]),
        )

    def attempts_for(self, row_id: int) -> List[DeliveryAttempt]:
        """Every recorded attempt on a row, in order.  Unknown row ids raise
        :class:`UnknownRow` so a typo is not mistaken for "never tried"."""
        with self._transaction("DEFERRED") as cur:
            self._load_row(cur, row_id)
            recs = cur.execute(
                """
                SELECT row_id, attempt, "at", ok, retryable, gone, reason
                FROM attempts WHERE row_id = ? ORDER BY attempt
                """,
                (row_id,),
            ).fetchall()
        return [
            DeliveryAttempt(
                row_id=int(r[0]),
                attempt=int(r[1]),
                at=float(r[2]),
                ok=bool(r[3]),
                retryable=bool(r[4]),
                gone=bool(r[5]),
                reason=str(r[6]),
            )
            for r in recs
        ]

    def dead_letters(self, limit: int = 50) -> List[OutboxRow]:
        """DEAD rows, newest first, each with its ``dead_reason`` and
        ``dead_at``.  No blob: rows never carry one."""
        with self._transaction("DEFERRED") as cur:
            return self._dead_letters(cur, limit)

    @staticmethod
    def _dead_letters(cur: sqlite3.Cursor, limit: int) -> List[OutboxRow]:
        if limit <= 0:
            return []
        recs = cur.execute(
            f"SELECT {_ROW_COLUMNS} FROM outbox WHERE state = ? ORDER BY row_id DESC LIMIT ?",
            (RowState.DEAD.value, int(limit)),
        ).fetchall()
        return [_row_from_record(r) for r in recs]

    def backlog_by_transport(self) -> Dict[str, int]:
        """PENDING rows behind a live subscription, counted per transport
        name, sorted by name.  A worker process that cannot send through
        some transport reads here how much it is leaving for one that can.
        """
        with self._transaction("DEFERRED") as cur:
            recs = cur.execute(
                f"""
                SELECT s.transport, count(*) FROM outbox o
                JOIN subscriptions s ON s.profile_id = o.profile_id AND s.device_id = o.device_id
                WHERE o.state = ? AND {_LIVE}
                GROUP BY s.transport ORDER BY s.transport
                """,
                (RowState.PENDING.value,),
            ).fetchall()
        return {str(name): int(n) for name, n in recs}

    def stats(self, dead_limit: int = 50) -> Dict[str, Any]:
        """Counts per row state plus a dead-letter list, JSON-serialisable.

        Keys: one per :class:`RowState` value (``pending``, ``leased``,
        ``delivered``, ``dead``); ``pending_unreachable`` -- PENDING rows
        whose device has no live subscription (unregistered, gone or
        pruned) and so will not be leased right now; ``dead_exhausted`` --
        DEAD rows whose budget ran out on transient failures
        (``DeadReason.EXHAUSTED``); ``dead_revivable`` -- those of them
        :meth:`revive_exhausted` will put back once their cooldown has
        passed (alert younger than ``ALERT_MAX_AGE_S`` at the outbox clock,
        device reachable); ``dead_permanent`` -- every other DEAD row,
        which only ``requeue`` brings back; ``revived`` -- how many times
        :meth:`revive_exhausted` has put a row back, in total; ``alerts``;
        ``subscriptions_live``; ``subscriptions_gone`` (pruned ones
        included; :meth:`pruned_subscriptions` lists those); and
        ``dead_letters``, the newest ``dead_limit`` DEAD rows as dicts.
        ``dead == dead_exhausted + dead_permanent`` always.  The blob is
        never part of it.
        """
        now = self._clock()
        with self._transaction("DEFERRED") as cur:
            counts = {state.value: 0 for state in RowState}
            for state, n in cur.execute("SELECT state, count(*) FROM outbox GROUP BY state"):
                counts[str(state)] = int(n)
            unreachable = cur.execute(
                """
                SELECT count(*) FROM outbox o
                LEFT JOIN subscriptions s
                    ON s.profile_id = o.profile_id AND s.device_id = o.device_id
                WHERE o.state = ? AND (s.device_id IS NULL OR s.gone = 1)
                """,
                (RowState.PENDING.value,),
            ).fetchone()[0]
            exhausted = cur.execute(
                "SELECT count(*) FROM outbox WHERE state = ? AND dead_reason = ?",
                (RowState.DEAD.value, DeadReason.EXHAUSTED.value),
            ).fetchone()[0]
            # Revivable regardless of the cooldown: the cooldown is timing,
            # the age and the device are what decide whether it ever comes back.
            revivable = cur.execute(
                f"SELECT count(*) FROM outbox o WHERE {_revivable_where('o')}",
                (0.0, float("inf"), ALERT_MAX_AGE_S, now),
            ).fetchone()[0]
            revived = cur.execute("SELECT coalesce(sum(revivals), 0) FROM outbox").fetchone()[0]
            alerts = cur.execute("SELECT count(*) FROM alerts").fetchone()[0]
            live, gone = cur.execute(
                "SELECT sum(gone = 0), sum(gone = 1) FROM subscriptions"
            ).fetchone()
            dead = self._dead_letters(cur, dead_limit)
        return {
            **counts,
            "pending_unreachable": int(unreachable),
            "dead_exhausted": int(exhausted),
            "dead_revivable": int(revivable),
            "dead_permanent": int(counts[RowState.DEAD.value]) - int(exhausted),
            "revived": int(revived),
            "alerts": int(alerts),
            "subscriptions_live": int(live or 0),
            "subscriptions_gone": int(gone or 0),
            "dead_letters": [_row_as_dict(r) for r in dead],
        }


# --------------------------------------------------------------------------
# Smoke run: python3 -m jarvis_alerts.outbox
# --------------------------------------------------------------------------


def _demo() -> Dict[str, Any]:
    """Walk one alert through publish, lease and mark on an in-memory file
    with a hand-driven clock and a seeded jitter stream, and return stats."""
    from lucifer_gen.seed import SeedFields

    ticks = [1000.0]
    stream = SeedFields.parse(0xC0FFEE).stream("alerts.backoff")
    outbox = Outbox(":memory:", clock=lambda: ticks[0], jitter=stream.random)
    outbox.register(Subscription("owner", "phone", "fake", '{"opaque": true}', 999.0))
    outbox.register(Subscription("owner", "laptop", "fake", '{"opaque": true}', 999.0))
    outbox.publish(Alert("a1", "owner", "render_done", "Render finished", "crypt.png is ready",
                         1000.0, Priority.HIGH, dedupe_key="render:crypt"))
    outbox.publish(Alert("a2", "owner", "render_done", "Render finished", "duplicate",
                         1000.0, Priority.HIGH, dedupe_key="render:crypt"))  # deduped
    leased = outbox.lease(now=1000.0, limit=10)
    outbox.mark(leased[0].row_id, SendResult(ok=True), now=1001.0)
    outbox.mark(leased[1].row_id, SendResult(ok=False, retryable=True, reason="503"), now=1001.0)
    ticks[0] = 1010.0
    retry = outbox.lease(now=1010.0, limit=10)
    outbox.mark(retry[0].row_id, SendResult(ok=False, gone=True, reason="410"), now=1011.0)
    return outbox.stats()


if __name__ == "__main__":
    print(json.dumps(_demo(), indent=2, sort_keys=True))
