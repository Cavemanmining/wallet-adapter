"""Durable state for the Pokemon buying assistant: one ``sqlite3`` file.

Design: :mod:`jarvis_poke.contracts`.  Every type this module writes is
defined there -- :class:`~jarvis_poke.contracts.Product`,
:class:`~jarvis_poke.contracts.SourceSku`,
:class:`~jarvis_poke.contracts.Observation`,
:class:`~jarvis_poke.contracts.Rule`, :class:`~jarvis_poke.contracts.Budget`,
:class:`~jarvis_poke.contracts.WatchState` and
:class:`~jarvis_poke.contracts.Verdict` -- plus the scheduler bookkeeping
from :mod:`jarvis_poke.sources` (``SourceState``/``SkuState``: the last
attempt, the conditional validators, the error run and the pause).

What this file is for
---------------------
The monitor has to survive a restart without becoming impolite or
forgetful.  Three things have to come back exactly as they went in:

* the rate gate and the pause (``source_state``) and the conditional
  validators (``sku_state``), or a restart re-polls every host at once and
  re-downloads pages a 304 would have covered -- contracts.py's
  "Politeness is a design constraint, not an afterthought";
* the market history (``observations``), or the deal scorer has nothing to
  score against and every listing looks like a bargain;
* what was decided and why (``verdicts``), which is the audit trail for a
  tool whose whole output is "act now".

Money is integer cents
----------------------
contracts.py: "Floats are never used for money".  That is enforced here
rather than hoped for: :func:`_cents` refuses anything that is not an
``int`` on the way in, so a ``4999.0`` that crept out of a parser fails at
the write with the column named, not three weeks later as a budget that is
one cent wrong.  Money columns are ``INTEGER``; timestamps are ``REAL``.

Atomicity
---------
Every save replaces one logical unit in a single ``BEGIN IMMEDIATE``
transaction: the products, the listings, the rules, the watch states, the
poll state.  A save is a *reconciliation*, not a truncate-and-refill -- it
deletes the keys that disappeared and upserts the rest -- so a foreign key
into the row being replaced never dangles mid-transaction.  Each table has
its own ``_insert_*`` seam, which is what the tests break to prove that an
exception halfway through leaves the previous state untouched.

The verdict log is append-only, and not merely by convention: two triggers
make ``UPDATE`` and ``DELETE`` on ``verdicts`` fail inside sqlite, so a
later bug -- or a later maintainer -- cannot quietly rewrite the record of
what the tool told its owner to buy.

No network, no clock, no randomness
-----------------------------------
This module opens a file and nothing else.  ``clock`` is injected, exactly
as in the rest of the package; nothing here calls ``time.time()`` or draws
a random number.

File mode
---------
The file is created ``0600`` before sqlite opens it, the same way
``jarvis_alerts.outbox`` creates its outbox: a watchlist is a list of what
its owner is about to spend money on, and the ``-wal`` and ``-shm`` files
inherit the mode.
"""

from __future__ import annotations

import collections.abc as _abc
import contextlib
import dataclasses
import enum
import json
import math
import os
import sqlite3
import threading

try:  # POSIX only; the cross-process poll lock degrades to in-process without it
    import fcntl
except ImportError:  # pragma: no cover - not POSIX
    fcntl = None  # type: ignore[assignment]
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from jarvis_poke.contracts import (
    Action,
    Budget,
    Cents,
    Observation,
    Product,
    ProductKind,
    Rule,
    SourceSku,
    Stock,
    Verdict,
    WatchState,
)

__all__ = [
    "BUSY_TIMEOUT_S",
    "DB_FILE_MODE",
    "POLL_STATE_VERSION_KEY",
    "SCHEMA_VERSION",
    "PokeStore",
    "SchemaError",
    "SqlitePollStore",
    "StoreError",
    "assert_round_trip_equal",
    "first_difference",
    "round_trip_equal",
]

#: Schema version 1.  It bumps only for a change an older reader could
#: misread; additive columns are handled by :meth:`PokeStore.migrate`.
SCHEMA_VERSION = 1

#: How long a connection waits for another writer before giving up.  Saves
#: are short transactions, so this is only ever hit under real contention.
BUSY_TIMEOUT_S = 30.0

#: Mode of a database file this module creates.  The same choice, for the
#: same reason, as ``jarvis_alerts.outbox.DB_FILE_MODE``.
DB_FILE_MODE = 0o600

#: ``meta`` key holding the scheduler snapshot's own version number, so a
#: snapshot written by a newer :mod:`jarvis_poke.sources` is refused by that
#: module's ``restore`` rather than half-read here.
POLL_STATE_VERSION_KEY = "poll_state_version"

Clock = Callable[[], float]


class StoreError(ValueError):
    """A bad value on the way in, or a bad row on the way out.

    ``ValueError`` to match the rest of the package (``CatalogError``,
    ``SourcesError``, ``RuleError``, ``EngineError``).
    """


class SchemaError(StoreError):
    """The file was written by a different, unsupported schema version."""


# --------------------------------------------------------------------------
# Value checks.  contracts.py: "Money is integer cents".
# --------------------------------------------------------------------------


def _cents(value: Any, what: str, *, allow_none: bool = False) -> Optional[Cents]:
    """``value`` as integer cents, or raise :class:`StoreError`.

    A ``float`` is refused even when it is integral (``4999.0``).  There is
    no safe way to accept one: by the time a price has been through a float
    it may already be ``4998.999999999999``, and the column it is headed
    for is an ``INTEGER``, so sqlite would silently take the truncation.
    ``bool`` is refused because ``True`` is an ``int`` and a price of 1 cent
    is not what the caller meant.
    """
    if value is None:
        if allow_none:
            return None
        raise StoreError(f"{what} is required and must be integer cents")
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreError(
            f"{what} must be integer cents (contracts.py: money is never a float); "
            f"got {type(value).__name__} {value!r}"
        )
    return value


def _timestamp(value: Any, what: str) -> float:
    """``value`` as a unix-seconds ``float``.  Rejects NaN and infinity."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StoreError(f"{what} must be a number of unix seconds; got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise StoreError(f"{what} must be finite; got {value!r}")
    return out


def _fraction(value: Any, what: str, *, allow_none: bool = False) -> Optional[float]:
    """A percentage or ratio: a real float, not money.  NaN is refused."""
    if value is None:
        if allow_none:
            return None
        raise StoreError(f"{what} is required")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StoreError(f"{what} must be a number; got {value!r}")
    out = float(value)
    if not math.isfinite(out):
        raise StoreError(f"{what} must be finite; got {value!r}")
    return out


def _text(value: Any, what: str, *, allow_none: bool = False) -> Optional[str]:
    if value is None and allow_none:
        return None
    if not isinstance(value, str):
        raise StoreError(f"{what} must be a string; got {type(value).__name__} {value!r}")
    return value


def _count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StoreError(f"{what} must be an int; got {type(value).__name__} {value!r}")
    if value < 0:
        raise StoreError(f"{what} must not be negative; got {value!r}")
    return value


def _flag(value: Any, what: str) -> int:
    if not isinstance(value, bool):
        raise StoreError(f"{what} must be a bool; got {type(value).__name__} {value!r}")
    return 1 if value else 0


def _member(kind: Any, value: Any, what: str, *, allow_none: bool = False) -> Any:
    """An enum member from its stored ``value`` string."""
    if value is None:
        if allow_none:
            return None
        raise StoreError(f"{what} is required")
    try:
        return kind(value)
    except ValueError:
        raise StoreError(f"{what}: {value!r} is not a {kind.__name__}") from None


def _json_list(value: Any, what: str) -> str:
    if not isinstance(value, (list, tuple)):
        raise StoreError(f"{what} must be a sequence; got {type(value).__name__}")
    out: List[str] = []
    for index, item in enumerate(value):
        text = _text(item, f"{what}[{index}]")
        assert text is not None
        out.append(text)
    return json.dumps(out)


def _read_json_list(raw: Any, what: str) -> Tuple[str, ...]:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        raise StoreError(f"{what} is not the JSON this module wrote") from None
    if not isinstance(parsed, list) or not all(isinstance(x, str) for x in parsed):
        raise StoreError(f"{what} is not a JSON array of strings")
    return tuple(parsed)


def _create_private_file(path: str) -> None:
    """Create ``path`` mode 0600 if it does not exist yet, so the watchlist
    it will hold is never world-readable.  sqlite gives the ``-wal`` and
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
# Schema
# --------------------------------------------------------------------------

_STOCK_VALUES = ", ".join(f"'{s.value}'" for s in Stock)
_ACTION_VALUES = ", ".join(f"'{a.value}'" for a in Action)
_KIND_VALUES = ", ".join(f"'{k.value}'" for k in ProductKind)

#: Columns of ``source_state``, in ``jarvis_poke.sources.SourceState`` field
#: order.  A field that module grows later is kept in ``extra`` (below), so
#: a round trip stays exact without this file having to move in lockstep.
_SOURCE_STATE_COLUMNS: Tuple[str, ...] = (
    "source", "last_attempt_at", "next_due_at", "paused_until", "pause_reason",
    "consecutive_errors", "attempts", "ok", "not_modified", "errors",
    "parse_errors", "refusals", "pauses", "observations", "last_status",
    "last_reason", "last_error_at",
)

#: Columns of ``sku_state``, in ``jarvis_poke.sources.SkuState`` field order.
#: ``etag`` and ``last_modified`` are the conditional-request validators;
#: losing them across a restart means re-downloading every page a 304 would
#: have covered, which is the impoliteness this table exists to prevent.
_SKU_STATE_COLUMNS: Tuple[str, ...] = (
    "source", "product_id", "last_attempt_at", "next_due_at", "etag",
    "last_modified", "attempts", "errors", "consecutive_errors",
    "not_modified", "observations", "last_status", "last_outcome", "draws",
)

#: What a state column holds when a snapshot does not mention it.  These
#: mirror the ``DEFAULT`` clauses in the schema below, and the key columns
#: (``source``, ``product_id``) are deliberately absent: a row that does not
#: say which host or listing it is about is a bug, not a default.
_SOURCE_STATE_DEFAULTS: Dict[str, Any] = {
    "last_attempt_at": 0.0, "next_due_at": 0.0, "paused_until": 0.0,
    "pause_reason": "", "consecutive_errors": 0, "attempts": 0, "ok": 0,
    "not_modified": 0, "errors": 0, "parse_errors": 0, "refusals": 0,
    "pauses": 0, "observations": 0, "last_status": 0, "last_reason": "",
    "last_error_at": 0.0,
}

_SKU_STATE_DEFAULTS: Dict[str, Any] = {
    "last_attempt_at": 0.0, "next_due_at": 0.0, "etag": None,
    "last_modified": None, "attempts": 0, "errors": 0,
    "consecutive_errors": 0, "not_modified": 0, "observations": 0,
    "last_status": 0, "last_outcome": "", "draws": 0,
}

_SCHEMA: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        id      INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS products (
        id       TEXT PRIMARY KEY,
        name     TEXT NOT NULL,
        set_code TEXT NOT NULL,
        kind     TEXT NOT NULL CHECK (kind IN ({_KIND_VALUES})),
        msrp     INTEGER,
        released TEXT,
        upc      TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS source_skus (
        source     TEXT NOT NULL,
        product_id TEXT NOT NULL REFERENCES products (id) ON DELETE CASCADE,
        sku        TEXT NOT NULL,
        url        TEXT NOT NULL,
        PRIMARY KEY (source, product_id)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS observations (
        obs_id             INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id         TEXT NOT NULL,
        source             TEXT NOT NULL,
        sku                TEXT NOT NULL,
        at                 REAL NOT NULL,
        stock              TEXT NOT NULL CHECK (stock IN ({_STOCK_VALUES})),
        price              INTEGER,
        shipping           INTEGER NOT NULL DEFAULT 0,
        per_customer_limit INTEGER,
        url                TEXT NOT NULL DEFAULT '',
        note               TEXT NOT NULL DEFAULT ''
    )
    """,
    # One look at one listing at one moment: re-saving the same look (a
    # restart replaying its last poll) must not double the history the
    # market reference is computed from.
    """
    CREATE UNIQUE INDEX IF NOT EXISTS observations_identity
        ON observations (source, product_id, sku, at)
    """,
    "CREATE INDEX IF NOT EXISTS observations_product_at ON observations (product_id, at)",
    """
    CREATE TABLE IF NOT EXISTS rules (
        product_id       TEXT PRIMARY KEY,
        max_price        INTEGER NOT NULL,
        quantity         INTEGER NOT NULL,
        min_discount_pct REAL NOT NULL,
        allowed_sources  TEXT NOT NULL,
        include_shipping INTEGER NOT NULL,
        cooldown_s       REAL NOT NULL,
        enabled          INTEGER NOT NULL
    )
    """,
    # One budget, one row.  contracts.py has a single Budget: "A ceiling
    # across everything, so a good week cannot empty an account".
    """
    CREATE TABLE IF NOT EXISTS budget (
        id       INTEGER PRIMARY KEY CHECK (id = 1),
        total    INTEGER NOT NULL,
        spent    INTEGER NOT NULL,
        window_s REAL NOT NULL
    )
    """,
    # Money promised to a BUY alert the owner has not resolved yet.  It
    # has to outlive the process: every ``decide`` run is a fresh process
    # and the deep link is still sitting on someone's phone.  Added with
    # CREATE TABLE IF NOT EXISTS and no SCHEMA_VERSION bump on purpose --
    # an older file simply gains the table, and bumping the version would
    # make every existing store unreadable for want of an upgrade path.
    """
    CREATE TABLE IF NOT EXISTS reservations (
        product_id TEXT PRIMARY KEY,
        amount     INTEGER NOT NULL CHECK (amount > 0),
        at         REAL NOT NULL DEFAULT 0
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS watch_states (
        product_id         TEXT PRIMARY KEY,
        last_alert_at      REAL NOT NULL DEFAULT 0,
        last_action        TEXT CHECK (last_action IS NULL OR last_action IN ({_ACTION_VALUES})),
        alerts_sent        INTEGER NOT NULL DEFAULT 0,
        last_seen_in_stock REAL NOT NULL DEFAULT 0
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS verdicts (
        verdict_id   INTEGER PRIMARY KEY AUTOINCREMENT,
        product_id   TEXT NOT NULL,
        action       TEXT NOT NULL CHECK (action IN ({_ACTION_VALUES})),
        at           REAL NOT NULL,
        source       TEXT,
        sku          TEXT,
        price        INTEGER,
        landed       INTEGER,
        market       INTEGER,
        discount_pct REAL,
        quantity     INTEGER NOT NULL DEFAULT 0,
        url          TEXT NOT NULL DEFAULT '',
        reasons      TEXT NOT NULL DEFAULT '[]'
    )
    """,
    "CREATE INDEX IF NOT EXISTS verdicts_product_at ON verdicts (product_id, at)",
    "CREATE INDEX IF NOT EXISTS verdicts_at ON verdicts (at)",
    # Append-only, enforced by sqlite rather than by good manners.  The log
    # is the audit trail of a tool that tells its owner to spend money.
    """
    CREATE TRIGGER IF NOT EXISTS verdicts_no_update
        BEFORE UPDATE ON verdicts
        BEGIN SELECT RAISE(ABORT, 'verdicts is append-only'); END
    """,
    """
    CREATE TRIGGER IF NOT EXISTS verdicts_no_delete
        BEFORE DELETE ON verdicts
        BEGIN SELECT RAISE(ABORT, 'verdicts is append-only'); END
    """,
    """
    CREATE TABLE IF NOT EXISTS source_state (
        source            TEXT PRIMARY KEY,
        last_attempt_at   REAL NOT NULL DEFAULT 0,
        next_due_at       REAL NOT NULL DEFAULT 0,
        paused_until      REAL NOT NULL DEFAULT 0,
        pause_reason      TEXT NOT NULL DEFAULT '',
        consecutive_errors INTEGER NOT NULL DEFAULT 0,
        attempts          INTEGER NOT NULL DEFAULT 0,
        ok                INTEGER NOT NULL DEFAULT 0,
        not_modified      INTEGER NOT NULL DEFAULT 0,
        errors            INTEGER NOT NULL DEFAULT 0,
        parse_errors      INTEGER NOT NULL DEFAULT 0,
        refusals          INTEGER NOT NULL DEFAULT 0,
        pauses            INTEGER NOT NULL DEFAULT 0,
        observations      INTEGER NOT NULL DEFAULT 0,
        last_status       INTEGER NOT NULL DEFAULT 0,
        last_reason       TEXT NOT NULL DEFAULT '',
        last_error_at     REAL NOT NULL DEFAULT 0,
        extra             TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sku_state (
        source            TEXT NOT NULL,
        product_id        TEXT NOT NULL,
        last_attempt_at   REAL NOT NULL DEFAULT 0,
        next_due_at       REAL NOT NULL DEFAULT 0,
        etag              TEXT,
        last_modified     TEXT,
        attempts          INTEGER NOT NULL DEFAULT 0,
        errors            INTEGER NOT NULL DEFAULT 0,
        consecutive_errors INTEGER NOT NULL DEFAULT 0,
        not_modified      INTEGER NOT NULL DEFAULT 0,
        observations      INTEGER NOT NULL DEFAULT 0,
        last_status       INTEGER NOT NULL DEFAULT 0,
        last_outcome      TEXT NOT NULL DEFAULT '',
        draws             INTEGER NOT NULL DEFAULT 0,
        extra             TEXT NOT NULL DEFAULT '{}',
        PRIMARY KEY (source, product_id)
    )
    """,
)


# --------------------------------------------------------------------------
# The store
# --------------------------------------------------------------------------


class PokeStore:
    """One ``sqlite3`` file holding the whole watchlist's state.

    ``path`` may be a filesystem path or ``":memory:"`` (private to this
    instance).  ``clock`` returns unix seconds and is required: contracts.py
    forbids a default, because the only possible default is ``time.time``.
    It is used for the relative forms of :meth:`prune_observations` and
    :meth:`verdicts`, and for nothing else -- every method that writes a
    timestamp takes it from the object being written.

    The connection is opened in autocommit mode so every transaction
    boundary is an explicit ``BEGIN`` / ``COMMIT`` / ``ROLLBACK`` here, and
    guarded by a lock so a monitor loop and an alert worker can share one
    instance.
    """

    def __init__(
        self,
        path: Union[str, "os.PathLike[str]"],
        clock: Clock,
    ) -> None:
        if not callable(clock):
            raise StoreError(
                "PokeStore needs an injected clock: a callable returning unix "
                "seconds (this package never calls time.time())"
            )
        self.path = os.fspath(path)
        self._clock = clock
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
        return f"PokeStore({self.path!r})"

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "PokeStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def now(self) -> float:
        """The injected clock, as a float."""
        return float(self._clock())

    # -- transactions -------------------------------------------------------

    @contextlib.contextmanager
    def _transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Cursor]:
        """Run the block in one transaction under the instance lock.

        ``IMMEDIATE`` takes the write lock up front, so a save never has to
        upgrade a read lock halfway through and fail after it has already
        deleted rows.  An exception anywhere -- including from ``COMMIT``
        itself -- rolls the whole transaction back, which is what makes a
        half-written save impossible.
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

    def _query(self, sql: str, params: Sequence[Any] = ()) -> List[Tuple[Any, ...]]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)))

    # -- schema -------------------------------------------------------------

    def migrate(self) -> None:
        """Create any missing tables and stamp :data:`SCHEMA_VERSION`.

        Idempotent.  A file stamped with another version raises
        :class:`SchemaError`: there is no upgrade path, and guessing one
        would corrupt a spending record quietly.
        """
        with self._transaction() as cur:
            for statement in _SCHEMA:
                cur.execute(statement)
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

    # -- meta ---------------------------------------------------------------

    def set_meta(self, key: str, value: str) -> None:
        """Set one small named string (versions, a last-run marker)."""
        _text(key, "meta key")
        _text(value, "meta value")
        with self._transaction() as cur:
            cur.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def meta(self, key: str) -> Optional[str]:
        rows = self._query("SELECT value FROM meta WHERE key = ?", (key,))
        return None if not rows else str(rows[0][0])

    # -- products -----------------------------------------------------------

    def _insert_product(self, cur: sqlite3.Cursor, product: Product) -> None:
        """One product row.  A seam: the rollback test breaks this."""
        if not isinstance(product, Product):
            raise StoreError(f"not a Product: {product!r}")
        cur.execute(
            "INSERT INTO products (id, name, set_code, kind, msrp, released, upc) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET name = excluded.name, "
            "set_code = excluded.set_code, kind = excluded.kind, "
            "msrp = excluded.msrp, released = excluded.released, upc = excluded.upc",
            (
                _text(product.id, "product.id"),
                _text(product.name, "product.name"),
                _text(product.set_code, "product.set_code"),
                _enum_value(product.kind, ProductKind, "product.kind"),
                _cents(product.msrp, "product.msrp", allow_none=True),
                _text(product.released, "product.released", allow_none=True),
                _text(product.upc, "product.upc", allow_none=True),
            ),
        )

    def save_products(self, products: Iterable[Product]) -> int:
        """Replace the product table with ``products``; return how many.

        A reconciliation, not a truncate: ids that disappeared are deleted
        (their listings cascade with them, because a listing for a product
        the catalog no longer carries is not a listing), the rest are
        upserted.  Observations and verdicts keep no foreign key into this
        table on purpose -- a product dropped from the catalog for a week
        must not take its price history with it.
        """
        items = list(products)
        ids = [_text(p.id, "product.id") for p in items if isinstance(p, Product)]
        _no_duplicates(ids, "product id")
        with self._transaction() as cur:
            keep = set(ids)
            existing = {str(r[0]) for r in cur.execute("SELECT id FROM products")}
            for gone in sorted(existing - keep):
                cur.execute("DELETE FROM products WHERE id = ?", (gone,))
            for product in items:
                self._insert_product(cur, product)
        return len(items)

    def load_products(self) -> List[Product]:
        """Every product, sorted by id."""
        rows = self._query(
            "SELECT id, name, set_code, kind, msrp, released, upc FROM products ORDER BY id"
        )
        return [
            Product(
                id=str(r[0]),
                name=str(r[1]),
                set_code=str(r[2]),
                kind=_member(ProductKind, r[3], "products.kind"),
                msrp=_cents(r[4], "products.msrp", allow_none=True),
                released=None if r[5] is None else str(r[5]),
                upc=None if r[6] is None else str(r[6]),
            )
            for r in rows
        ]

    def load_product(self, product_id: str) -> Optional[Product]:
        for product in self.load_products():
            if product.id == product_id:
                return product
        return None

    # -- listings -----------------------------------------------------------

    def _insert_sku(self, cur: sqlite3.Cursor, sku: SourceSku) -> None:
        """One listing row.  A seam: the rollback test breaks this."""
        if not isinstance(sku, SourceSku):
            raise StoreError(f"not a SourceSku: {sku!r}")
        cur.execute(
            "INSERT INTO source_skus (source, product_id, sku, url) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (source, product_id) DO UPDATE SET "
            "sku = excluded.sku, url = excluded.url",
            (
                _text(sku.source, "sku.source"),
                _text(sku.product_id, "sku.product_id"),
                _text(sku.sku, "sku.sku"),
                _text(sku.url, "sku.url"),
            ),
        )

    def save_source_skus(self, skus: Iterable[SourceSku]) -> int:
        """Replace the listing table; return how many.

        Every listing's product must already be stored: the foreign key is
        the point, since a listing whose product id has drifted is a poll
        that can never produce a usable observation.  Save the products
        first, or use :meth:`save_catalog`.
        """
        items = list(skus)
        keys = [
            (_text(s.source, "sku.source"), _text(s.product_id, "sku.product_id"))
            for s in items
            if isinstance(s, SourceSku)
        ]
        _no_duplicates(keys, "(source, product_id)")
        with self._transaction() as cur:
            keep = set(keys)
            existing = {
                (str(r[0]), str(r[1]))
                for r in cur.execute("SELECT source, product_id FROM source_skus")
            }
            for gone in sorted(existing - keep):
                cur.execute(
                    "DELETE FROM source_skus WHERE source = ? AND product_id = ?", gone
                )
            for sku in items:
                try:
                    self._insert_sku(cur, sku)
                except sqlite3.IntegrityError as exc:
                    raise StoreError(
                        f"listing {sku.source}/{sku.product_id} names a product that is "
                        f"not stored: {exc}"
                    ) from None
        return len(items)

    def load_source_skus(self, source: Optional[str] = None) -> List[SourceSku]:
        """Every listing, or one source's, sorted by (source, product_id)."""
        sql = "SELECT source, product_id, sku, url FROM source_skus"
        params: Tuple[Any, ...] = ()
        if source is not None:
            sql += " WHERE source = ?"
            params = (source,)
        sql += " ORDER BY source, product_id"
        return [
            SourceSku(source=str(r[0]), product_id=str(r[1]), sku=str(r[2]), url=str(r[3]))
            for r in self._query(sql, params)
        ]

    def save_catalog(
        self,
        products: Iterable[Product],
        skus: Iterable[SourceSku] = (),
    ) -> Tuple[int, int]:
        """Replace products *and* listings in one transaction.

        Two separate saves would briefly have listings pointing at the old
        catalog; this cannot.  Accepts a :class:`jarvis_poke.catalog.Catalog`
        in place of ``products`` (its ``products()`` and ``skus()`` are
        used), which is the usual call.
        """
        if hasattr(products, "products") and hasattr(products, "skus"):
            catalog = products
            products = catalog.products()       # type: ignore[union-attr]
            skus = catalog.skus()               # type: ignore[union-attr]
        product_list = list(products)
        sku_list = list(skus)
        product_ids = [_text(p.id, "product.id") for p in product_list if isinstance(p, Product)]
        _no_duplicates(product_ids, "product id")
        sku_keys = [
            (_text(s.source, "sku.source"), _text(s.product_id, "sku.product_id"))
            for s in sku_list
            if isinstance(s, SourceSku)
        ]
        _no_duplicates(sku_keys, "(source, product_id)")
        with self._transaction() as cur:
            keep_skus = set(sku_keys)
            for row in list(cur.execute("SELECT source, product_id FROM source_skus")):
                if (str(row[0]), str(row[1])) not in keep_skus:
                    cur.execute(
                        "DELETE FROM source_skus WHERE source = ? AND product_id = ?",
                        (row[0], row[1]),
                    )
            keep_products = set(product_ids)
            for row in list(cur.execute("SELECT id FROM products")):
                if str(row[0]) not in keep_products:
                    cur.execute("DELETE FROM products WHERE id = ?", (row[0],))
            for product in product_list:
                self._insert_product(cur, product)
            for sku in sku_list:
                try:
                    self._insert_sku(cur, sku)
                except sqlite3.IntegrityError as exc:
                    raise StoreError(
                        f"listing {sku.source}/{sku.product_id} names a product that is "
                        f"not in the catalog being saved: {exc}"
                    ) from None
        return len(product_list), len(sku_list)

    # -- observations -------------------------------------------------------

    def _insert_observation(self, cur: sqlite3.Cursor, obs: Observation) -> int:
        """One observation row, ignored if that exact look is already
        stored.  Returns the rows written (0 or 1).  A seam for the
        rollback test."""
        if not isinstance(obs, Observation):
            raise StoreError(f"not an Observation: {obs!r}")
        cur.execute(
            "INSERT OR IGNORE INTO observations "
            "(product_id, source, sku, at, stock, price, shipping, "
            " per_customer_limit, url, note) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _text(obs.product_id, "observation.product_id"),
                _text(obs.source, "observation.source"),
                _text(obs.sku, "observation.sku"),
                _timestamp(obs.at, "observation.at"),
                _enum_value(obs.stock, Stock, "observation.stock"),
                _cents(obs.price, "observation.price", allow_none=True),
                _cents(obs.shipping, "observation.shipping"),
                None if obs.per_customer_limit is None
                else _count(obs.per_customer_limit, "observation.per_customer_limit"),
                _text(obs.url, "observation.url"),
                _text(obs.note, "observation.note"),
            ),
        )
        return int(cur.rowcount or 0)

    def save_observations(self, observations: Iterable[Observation]) -> int:
        """Append observations; return how many rows were new.

        The history is append-only in spirit and idempotent in fact: one
        look at one listing at one moment is one row, so replaying a poll
        after a restart cannot double-count a price into the market
        reference.  The batch is one transaction -- a pass over ten
        listings lands whole or not at all.
        """
        written = 0
        with self._transaction() as cur:
            for obs in observations:
                written += self._insert_observation(cur, obs)
        return written

    def save_observation(self, observation: Observation) -> int:
        """One observation; returns 1, or 0 if that look was already stored."""
        return self.save_observations([observation])

    def load_observations(
        self,
        product_id: Optional[str] = None,
        *,
        source: Optional[str] = None,
        sku: Optional[str] = None,
        since: Optional[float] = None,
        until: Optional[float] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Observation]:
        """The history, oldest first, narrowed by whatever is given.

        ``since`` is inclusive and ``until`` exclusive, so consecutive
        windows neither overlap nor drop a row.  ``newest_first`` with a
        ``limit`` is how you ask for the last N.
        """
        sql = (
            "SELECT product_id, source, sku, at, stock, price, shipping, "
            "per_customer_limit, url, note FROM observations"
        )
        where: List[str] = []
        params: List[Any] = []
        if product_id is not None:
            where.append("product_id = ?")
            params.append(product_id)
        if source is not None:
            where.append("source = ?")
            params.append(source)
        if sku is not None:
            where.append("sku = ?")
            params.append(sku)
        if since is not None:
            where.append("at >= ?")
            params.append(_timestamp(since, "since"))
        if until is not None:
            where.append("at < ?")
            params.append(_timestamp(until, "until"))
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY at DESC, obs_id DESC" if newest_first else " ORDER BY at, obs_id"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(_count(limit, "limit"))
        return [
            Observation(
                product_id=str(r[0]),
                source=str(r[1]),
                sku=str(r[2]),
                at=float(r[3]),
                stock=_member(Stock, r[4], "observations.stock"),
                price=_cents(r[5], "observations.price", allow_none=True),
                shipping=_cents(r[6], "observations.shipping"),
                per_customer_limit=None if r[7] is None else int(r[7]),
                url=str(r[8]),
                note=str(r[9]),
            )
            for r in self._query(sql, params)
        ]

    def prune_observations(
        self,
        before: Optional[float] = None,
        *,
        window_s: Optional[float] = None,
        now: Optional[float] = None,
        keep_latest_per_listing: bool = True,
    ) -> int:
        """Delete observations older than the window; return how many.

        Give either ``before`` (an absolute cutoff) or ``window_s`` (that
        many seconds back from ``now``, or from the injected clock).  An
        observation exactly at the cutoff is kept, matching
        :meth:`load_observations`'s inclusive ``since``.

        ``keep_latest_per_listing`` (the default) spares the newest row of
        every listing whatever its age.  That row is the *current* state of
        that listing: the scheduler reuses it on a 304, and dropping it
        would make a quiet listing look unknown and pull a fresh fetch out
        of a host that had told us nothing changed.  Pass ``False`` for a
        hard cutoff.
        """
        if (before is None) == (window_s is None):
            raise StoreError("prune_observations takes exactly one of before= or window_s=")
        if before is None:
            span = _timestamp(window_s, "window_s")
            if span < 0:
                raise StoreError("window_s must not be negative")
            cutoff = (self.now() if now is None else _timestamp(now, "now")) - span
        else:
            cutoff = _timestamp(before, "before")
        with self._transaction() as cur:
            keep: set = set()
            if keep_latest_per_listing:
                listings = list(
                    cur.execute("SELECT DISTINCT source, product_id, sku FROM observations")
                )
                for source, product_id, sku in listings:
                    row = cur.execute(
                        "SELECT obs_id FROM observations "
                        "WHERE source = ? AND product_id = ? AND sku = ? "
                        "ORDER BY at DESC, obs_id DESC LIMIT 1",
                        (source, product_id, sku),
                    ).fetchone()
                    if row is not None:
                        keep.add(int(row[0]))
            doomed = [
                int(r[0])
                for r in cur.execute(
                    "SELECT obs_id FROM observations WHERE at < ?", (cutoff,)
                )
                if int(r[0]) not in keep
            ]
            cur.executemany("DELETE FROM observations WHERE obs_id = ?", [(i,) for i in doomed])
        return len(doomed)

    # -- rules and budget ---------------------------------------------------

    def _insert_rule(self, cur: sqlite3.Cursor, rule: Rule) -> None:
        """One rule row.  A seam: the rollback test breaks this."""
        if not isinstance(rule, Rule):
            raise StoreError(f"not a Rule: {rule!r}")
        cur.execute(
            "INSERT INTO rules (product_id, max_price, quantity, min_discount_pct, "
            "allowed_sources, include_shipping, cooldown_s, enabled) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (product_id) DO UPDATE SET "
            "max_price = excluded.max_price, quantity = excluded.quantity, "
            "min_discount_pct = excluded.min_discount_pct, "
            "allowed_sources = excluded.allowed_sources, "
            "include_shipping = excluded.include_shipping, "
            "cooldown_s = excluded.cooldown_s, enabled = excluded.enabled",
            (
                _text(rule.product_id, "rule.product_id"),
                _cents(rule.max_price, "rule.max_price"),
                _count(rule.quantity, "rule.quantity"),
                _fraction(rule.min_discount_pct, "rule.min_discount_pct"),
                _json_list(rule.allowed_sources, "rule.allowed_sources"),
                _flag(rule.include_shipping, "rule.include_shipping"),
                _timestamp(rule.cooldown_s, "rule.cooldown_s"),
                _flag(rule.enabled, "rule.enabled"),
            ),
        )

    def save_rules(self, rules: Iterable[Rule]) -> int:
        """Replace the rule table; return how many.

        No foreign key to ``products``: contracts.py has the engine SKIP a
        rule whose product the catalog does not know, with a reason, and
        that only works if such a rule can be stored in the first place.
        """
        items = list(rules)
        ids = [_text(r.product_id, "rule.product_id") for r in items if isinstance(r, Rule)]
        _no_duplicates(ids, "rule product id")
        with self._transaction() as cur:
            keep = set(ids)
            existing = {str(r[0]) for r in cur.execute("SELECT product_id FROM rules")}
            for gone in sorted(existing - keep):
                cur.execute("DELETE FROM rules WHERE product_id = ?", (gone,))
            for rule in items:
                self._insert_rule(cur, rule)
        return len(items)

    def load_rules(self) -> List[Rule]:
        """Every rule, enabled or not, sorted by product id."""
        rows = self._query(
            "SELECT product_id, max_price, quantity, min_discount_pct, allowed_sources, "
            "include_shipping, cooldown_s, enabled FROM rules ORDER BY product_id"
        )
        return [
            Rule(
                product_id=str(r[0]),
                max_price=_cents(r[1], "rules.max_price"),
                quantity=int(r[2]),
                min_discount_pct=float(r[3]),
                allowed_sources=_read_json_list(r[4], "rules.allowed_sources"),
                include_shipping=bool(r[5]),
                cooldown_s=float(r[6]),
                enabled=bool(r[7]),
            )
            for r in rows
        ]

    def _insert_budget(self, cur: sqlite3.Cursor, budget: Budget) -> None:
        """The one budget row.  A seam: the rollback test breaks this."""
        if not isinstance(budget, Budget):
            raise StoreError(f"not a Budget: {budget!r}")
        cur.execute(
            "INSERT INTO budget (id, total, spent, window_s) VALUES (1, ?, ?, ?) "
            "ON CONFLICT (id) DO UPDATE SET total = excluded.total, "
            "spent = excluded.spent, window_s = excluded.window_s",
            (
                _cents(budget.total, "budget.total"),
                _cents(budget.spent, "budget.spent"),
                _timestamp(budget.window_s, "budget.window_s"),
            ),
        )

    def save_budget(self, budget: Budget) -> None:
        """Replace the budget."""
        with self._transaction() as cur:
            self._insert_budget(cur, budget)

    def load_budget(self) -> Optional[Budget]:
        """The stored budget, or ``None`` if the owner never set one."""
        rows = self._query("SELECT total, spent, window_s FROM budget WHERE id = 1")
        if not rows:
            return None
        total, spent, window_s = rows[0]
        return Budget(
            total=_cents(total, "budget.total"),
            spent=_cents(spent, "budget.spent"),
            window_s=float(window_s),
        )

    def save_rule_set(self, rule_set: Any) -> int:
        """Replace the rules *and* the budget in one transaction.

        Takes a :class:`jarvis_poke.rules.RuleSet` (anything with
        ``rules()`` and ``budget()``).  Rules and the ledger they spend
        from are one logical unit: a restart that found new rules against
        an old budget could commit to more than the owner allowed.

        Reservations are stored too, in the same transaction.  They used
        to be dropped, on the theory that "after a restart nobody is
        holding a deep link open" -- which is wrong for the way this tool
        is actually run: ``decide`` is a fresh process every time, and the
        alert it sent last night is still on the owner's phone.  Dropping
        them meant the ledger reset to nothing-promised on every
        invocation, so the per-verdict cap was the only live control and
        the owner's monthly ceiling never moved at all.  They are cleared
        deliberately, by :meth:`jarvis_poke.rules.RuleSet.release_for`
        (the owner ignored the link) or :meth:`RuleSet.commit_for` (they
        bought it), not by a restart.
        """
        # ``RuleSet.rules`` is a method and ``RuleSet.budget`` a property;
        # accept either shape so a caller's own rule holder also works.
        raw_rules = getattr(rule_set, "rules")
        raw_budget = getattr(rule_set, "budget")
        rules = list(raw_rules() if callable(raw_rules) else raw_rules)
        budget = raw_budget() if callable(raw_budget) else raw_budget
        ids = [_text(r.product_id, "rule.product_id") for r in rules if isinstance(r, Rule)]
        _no_duplicates(ids, "rule product id")
        with self._transaction() as cur:
            keep = set(ids)
            existing = {str(r[0]) for r in cur.execute("SELECT product_id FROM rules")}
            for gone in sorted(existing - keep):
                cur.execute("DELETE FROM rules WHERE product_id = ?", (gone,))
            for rule in rules:
                self._insert_rule(cur, rule)
            self._insert_budget(cur, budget)
            holds = getattr(rule_set, "reservations", None)
            if callable(holds):
                self._replace_reservations(cur, holds())
        return len(rules)

    # -- reservations -------------------------------------------------------

    def _replace_reservations(
        self, cur: sqlite3.Cursor, holds: Mapping[str, int]
    ) -> int:
        rows = {}
        for key, amount in dict(holds).items():
            product_id = _text(key, "reservation.product_id")
            cents = _cents(amount, "reservation.amount")
            if cents < 0:
                raise StoreError(f"reservation for {product_id!r} is negative: {cents}")
            if cents:
                rows[product_id] = cents
        cur.execute("DELETE FROM reservations")
        for product_id, cents in sorted(rows.items()):
            cur.execute(
                "INSERT INTO reservations (product_id, amount, at) VALUES (?, ?, ?)",
                (product_id, cents, 0.0),
            )
        return len(rows)

    def save_reservations(self, holds: Mapping[str, int]) -> int:
        """Replace every stored reservation.  Returns how many were kept.

        A reservation is cents promised to a BUY alert the owner has not
        acted on.  It is the only part of the ledger that can survive a
        run without the owner saying anything, which is why it is stored
        rather than recomputed.
        """
        with self._transaction() as cur:
            return self._replace_reservations(cur, holds)

    def load_reservations(self) -> Dict[str, int]:
        """``{product_id: cents}`` for every stored reservation."""
        return {
            str(row[0]): int(row[1])
            for row in self._query(
                "SELECT product_id, amount FROM reservations ORDER BY product_id"
            )
        }

    def load_rule_set(self) -> Any:
        """The stored rules and budget as a :class:`RuleSet`."""
        from jarvis_poke.rules import RuleSet  # local: keeps store.py importable alone

        budget = self.load_budget()
        rule_set = RuleSet(self.load_rules(), budget if budget is not None else Budget(total=0))
        holds = self.load_reservations()
        if holds:
            loader = getattr(rule_set, "load_reservations", None)
            if callable(loader):
                loader(holds)
        return rule_set

    # -- watch states -------------------------------------------------------

    def _insert_watch_state(self, cur: sqlite3.Cursor, state: WatchState) -> None:
        """One watch-state row.  A seam: the rollback test breaks this."""
        if not isinstance(state, WatchState):
            raise StoreError(f"not a WatchState: {state!r}")
        cur.execute(
            "INSERT INTO watch_states (product_id, last_alert_at, last_action, "
            "alerts_sent, last_seen_in_stock) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT (product_id) DO UPDATE SET "
            "last_alert_at = excluded.last_alert_at, last_action = excluded.last_action, "
            "alerts_sent = excluded.alerts_sent, "
            "last_seen_in_stock = excluded.last_seen_in_stock",
            (
                _text(state.product_id, "watch_state.product_id"),
                _timestamp(state.last_alert_at, "watch_state.last_alert_at"),
                None if state.last_action is None
                else _enum_value(state.last_action, Action, "watch_state.last_action"),
                _count(state.alerts_sent, "watch_state.alerts_sent"),
                _timestamp(state.last_seen_in_stock, "watch_state.last_seen_in_stock"),
            ),
        )

    def save_watch_states(
        self, states: Union[Mapping[str, WatchState], Iterable[WatchState]]
    ) -> int:
        """Replace the watch states; return how many.

        Takes the engine's ``watch_states`` mapping directly, or any
        iterable of :class:`WatchState`.
        """
        items = list(states.values()) if isinstance(states, _abc.Mapping) else list(states)
        ids = [
            _text(s.product_id, "watch_state.product_id")
            for s in items
            if isinstance(s, WatchState)
        ]
        _no_duplicates(ids, "watch state product id")
        with self._transaction() as cur:
            keep = set(ids)
            existing = {str(r[0]) for r in cur.execute("SELECT product_id FROM watch_states")}
            for gone in sorted(existing - keep):
                cur.execute("DELETE FROM watch_states WHERE product_id = ?", (gone,))
            for state in items:
                self._insert_watch_state(cur, state)
        return len(items)

    def load_watch_states(self) -> Dict[str, WatchState]:
        """The watch states by product id, ready to hand to the engine."""
        rows = self._query(
            "SELECT product_id, last_alert_at, last_action, alerts_sent, last_seen_in_stock "
            "FROM watch_states ORDER BY product_id"
        )
        out: Dict[str, WatchState] = {}
        for r in rows:
            out[str(r[0])] = WatchState(
                product_id=str(r[0]),
                last_alert_at=float(r[1]),
                last_action=_member(Action, r[2], "watch_states.last_action", allow_none=True),
                alerts_sent=int(r[3]),
                last_seen_in_stock=float(r[4]),
            )
        return out

    # -- verdicts: the append-only log --------------------------------------

    def _insert_verdict(self, cur: sqlite3.Cursor, verdict: Verdict) -> int:
        """One verdict row; returns its id.  A seam for the rollback test."""
        if not isinstance(verdict, Verdict):
            raise StoreError(f"not a Verdict: {verdict!r}")
        cur.execute(
            "INSERT INTO verdicts (product_id, action, at, source, sku, price, landed, "
            "market, discount_pct, quantity, url, reasons) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                _text(verdict.product_id, "verdict.product_id"),
                _enum_value(verdict.action, Action, "verdict.action"),
                _timestamp(verdict.at, "verdict.at"),
                _text(verdict.source, "verdict.source", allow_none=True),
                _text(verdict.sku, "verdict.sku", allow_none=True),
                _cents(verdict.price, "verdict.price", allow_none=True),
                _cents(verdict.landed, "verdict.landed", allow_none=True),
                _cents(verdict.market, "verdict.market", allow_none=True),
                _fraction(verdict.discount_pct, "verdict.discount_pct", allow_none=True),
                _count(verdict.quantity, "verdict.quantity"),
                _text(verdict.url, "verdict.url"),
                _json_list(verdict.reasons, "verdict.reasons"),
            ),
        )
        return int(cur.lastrowid or 0)

    def append_verdict(self, verdict: Verdict) -> int:
        """Append one verdict to the log; return its row id.

        The log is never updated and never deleted from -- two triggers in
        the schema make sqlite refuse both -- because it is the record of
        what this tool told its owner to do with money.
        """
        with self._transaction() as cur:
            return self._insert_verdict(cur, verdict)

    def append_verdicts(self, verdicts: Iterable[Verdict]) -> List[int]:
        """Append a pass's worth of verdicts in one transaction."""
        with self._transaction() as cur:
            return [self._insert_verdict(cur, v) for v in verdicts]

    def verdicts(
        self,
        product_id: Optional[str] = None,
        *,
        since: Optional[float] = None,
        until: Optional[float] = None,
        within_s: Optional[float] = None,
        actions: Optional[Iterable[Action]] = None,
        limit: Optional[int] = None,
        newest_first: bool = False,
    ) -> List[Verdict]:
        """Query the log by product and time; oldest first by default.

        ``since`` is inclusive, ``until`` exclusive.  ``within_s`` is the
        relative form -- the last that many seconds by the injected clock --
        and cannot be combined with ``since``.  Ordering is by ``at`` then
        by row id, so two verdicts stamped the same instant still come back
        in the order they were written.
        """
        if within_s is not None:
            if since is not None:
                raise StoreError("verdicts takes since= or within_s=, not both")
            span = _timestamp(within_s, "within_s")
            if span < 0:
                raise StoreError("within_s must not be negative")
            since = self.now() - span
        sql = (
            "SELECT product_id, action, at, source, sku, price, landed, market, "
            "discount_pct, quantity, url, reasons FROM verdicts"
        )
        where: List[str] = []
        params: List[Any] = []
        if product_id is not None:
            where.append("product_id = ?")
            params.append(product_id)
        if since is not None:
            where.append("at >= ?")
            params.append(_timestamp(since, "since"))
        if until is not None:
            where.append("at < ?")
            params.append(_timestamp(until, "until"))
        if actions is not None:
            wanted = [_enum_value(a, Action, "action") for a in actions]
            if not wanted:
                return []
            where.append("action IN (" + ", ".join("?" for _ in wanted) + ")")
            params.extend(wanted)
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += (
            " ORDER BY at DESC, verdict_id DESC" if newest_first
            else " ORDER BY at, verdict_id"
        )
        if limit is not None:
            sql += " LIMIT ?"
            params.append(_count(limit, "limit"))
        return [
            Verdict(
                product_id=str(r[0]),
                action=_member(Action, r[1], "verdicts.action"),
                at=float(r[2]),
                source=None if r[3] is None else str(r[3]),
                sku=None if r[4] is None else str(r[4]),
                price=_cents(r[5], "verdicts.price", allow_none=True),
                landed=_cents(r[6], "verdicts.landed", allow_none=True),
                market=_cents(r[7], "verdicts.market", allow_none=True),
                discount_pct=None if r[8] is None else float(r[8]),
                quantity=int(r[9]),
                url=str(r[10]),
                reasons=_read_json_list(r[11], "verdicts.reasons"),
            )
            for r in self._query(sql, params)
        ]

    def last_verdict(self, product_id: str) -> Optional[Verdict]:
        """The newest verdict for one product, or ``None``."""
        found = self.verdicts(product_id, limit=1, newest_first=True)
        return found[0] if found else None

    # -- scheduler state ----------------------------------------------------

    def _insert_source_state(self, cur: sqlite3.Cursor, row: Mapping[str, Any]) -> None:
        """One ``source_state`` row from a snapshot mapping.  A seam."""
        values, extra = _split_state(
            row, _SOURCE_STATE_COLUMNS, _SOURCE_STATE_DEFAULTS, "source_state"
        )
        columns = ", ".join(_SOURCE_STATE_COLUMNS) + ", extra"
        marks = ", ".join("?" for _ in range(len(_SOURCE_STATE_COLUMNS) + 1))
        updates = ", ".join(
            f"{c} = excluded.{c}" for c in _SOURCE_STATE_COLUMNS[1:] + ("extra",)
        )
        cur.execute(
            f"INSERT INTO source_state ({columns}) VALUES ({marks}) "
            f"ON CONFLICT (source) DO UPDATE SET {updates}",
            tuple(values) + (json.dumps(extra, sort_keys=True),),
        )

    def _insert_sku_state(self, cur: sqlite3.Cursor, row: Mapping[str, Any]) -> None:
        """One ``sku_state`` row from a snapshot mapping.  A seam."""
        values, extra = _split_state(
            row, _SKU_STATE_COLUMNS, _SKU_STATE_DEFAULTS, "sku_state"
        )
        columns = ", ".join(_SKU_STATE_COLUMNS) + ", extra"
        marks = ", ".join("?" for _ in range(len(_SKU_STATE_COLUMNS) + 1))
        updates = ", ".join(
            f"{c} = excluded.{c}" for c in _SKU_STATE_COLUMNS[2:] + ("extra",)
        )
        cur.execute(
            f"INSERT INTO sku_state ({columns}) VALUES ({marks}) "
            f"ON CONFLICT (source, product_id) DO UPDATE SET {updates}",
            tuple(values) + (json.dumps(extra, sort_keys=True),),
        )

    def save_poll_snapshot(self, snapshot: Mapping[str, Any]) -> Tuple[int, int]:
        """Store a :meth:`jarvis_poke.sources.PollScheduler.snapshot`.

        Both tables are replaced in one transaction: the per-host gate and
        pause (``source_state``) and the per-listing validators
        (``sku_state``).  A half-applied poll state is the one outcome that
        must never happen -- it is how a restart forgets a pause and
        hammers a host that had already told us to stop.
        """
        sources = snapshot.get("sources") or {}
        skus = snapshot.get("skus") or []
        if not isinstance(sources, _abc.Mapping):
            raise StoreError("snapshot['sources'] must be a mapping of source -> state")
        if not isinstance(skus, (list, tuple)):
            raise StoreError("snapshot['skus'] must be a sequence of states")
        source_rows = [dict(row, source=name) for name, row in sources.items()]
        sku_rows = [dict(row) for row in skus]
        source_keys = {str(r["source"]) for r in source_rows}
        sku_keys = {(str(r.get("source")), str(r.get("product_id"))) for r in sku_rows}
        version = snapshot.get("version")
        with self._transaction() as cur:
            for row in list(cur.execute("SELECT source FROM source_state")):
                if str(row[0]) not in source_keys:
                    cur.execute("DELETE FROM source_state WHERE source = ?", (row[0],))
            for row in list(cur.execute("SELECT source, product_id FROM sku_state")):
                if (str(row[0]), str(row[1])) not in sku_keys:
                    cur.execute(
                        "DELETE FROM sku_state WHERE source = ? AND product_id = ?",
                        (row[0], row[1]),
                    )
            for row in source_rows:
                self._insert_source_state(cur, row)
            for row in sku_rows:
                self._insert_sku_state(cur, row)
            if version is not None:
                cur.execute(
                    "INSERT INTO meta (key, value) VALUES (?, ?) "
                    "ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                    (POLL_STATE_VERSION_KEY, json.dumps(version)),
                )
        return len(source_rows), len(sku_rows)

    def load_poll_snapshot(self) -> Optional[Dict[str, Any]]:
        """The stored scheduler snapshot, or ``None`` when there is none.

        ``None`` rather than an empty snapshot, because
        ``PollScheduler.restore`` of an empty snapshot and never restoring
        at all are the same thing, and ``None`` is what a
        :class:`~jarvis_poke.sources.PollStore` returns for "nothing yet".
        """
        source_rows = self._query(
            "SELECT " + ", ".join(_SOURCE_STATE_COLUMNS) + ", extra FROM source_state "
            "ORDER BY source"
        )
        sku_rows = self._query(
            "SELECT " + ", ".join(_SKU_STATE_COLUMNS) + ", extra FROM sku_state "
            "ORDER BY source, product_id"
        )
        if not source_rows and not sku_rows:
            return None
        sources: Dict[str, Dict[str, Any]] = {}
        for row in source_rows:
            data = _join_state(row, _SOURCE_STATE_COLUMNS)
            sources[str(data["source"])] = data
        skus = [_join_state(row, _SKU_STATE_COLUMNS) for row in sku_rows]
        raw_version = self.meta(POLL_STATE_VERSION_KEY)
        snapshot: Dict[str, Any] = {"sources": sources, "skus": skus}
        if raw_version is not None:
            snapshot["version"] = json.loads(raw_version)
        return snapshot

    def save_source_states(self, states: Union[Mapping[str, Any], Iterable[Any]]) -> int:
        """Replace ``source_state`` from ``SourceState`` objects."""
        items = list(states.values()) if isinstance(states, _abc.Mapping) else list(states)
        rows = [_state_as_row(s, "SourceState") for s in items]
        keys = {str(r["source"]) for r in rows}
        with self._transaction() as cur:
            for row in list(cur.execute("SELECT source FROM source_state")):
                if str(row[0]) not in keys:
                    cur.execute("DELETE FROM source_state WHERE source = ?", (row[0],))
            for row in rows:
                self._insert_source_state(cur, row)
        return len(rows)

    def load_source_states(self) -> Dict[str, Any]:
        """``source_state`` as ``jarvis_poke.sources.SourceState`` objects."""
        module = _sources_module()
        rows = self._query(
            "SELECT " + ", ".join(_SOURCE_STATE_COLUMNS) + ", extra FROM source_state "
            "ORDER BY source"
        )
        out: Dict[str, Any] = {}
        for row in rows:
            data = _join_state(row, _SOURCE_STATE_COLUMNS)
            state = _state_from_row(module.SourceState, data)
            out[state.source] = state
        return out

    def save_sku_states(self, states: Union[Mapping[Any, Any], Iterable[Any]]) -> int:
        """Replace ``sku_state`` from ``SkuState`` objects (the validators)."""
        items = list(states.values()) if isinstance(states, _abc.Mapping) else list(states)
        rows = [_state_as_row(s, "SkuState") for s in items]
        keys = {(str(r["source"]), str(r["product_id"])) for r in rows}
        with self._transaction() as cur:
            for row in list(cur.execute("SELECT source, product_id FROM sku_state")):
                if (str(row[0]), str(row[1])) not in keys:
                    cur.execute(
                        "DELETE FROM sku_state WHERE source = ? AND product_id = ?",
                        (row[0], row[1]),
                    )
            for row in rows:
                self._insert_sku_state(cur, row)
        return len(rows)

    def load_sku_states(self) -> Dict[Tuple[str, str], Any]:
        """``sku_state`` as ``SkuState`` objects, keyed (source, product_id)."""
        module = _sources_module()
        rows = self._query(
            "SELECT " + ", ".join(_SKU_STATE_COLUMNS) + ", extra FROM sku_state "
            "ORDER BY source, product_id"
        )
        out: Dict[Tuple[str, str], Any] = {}
        for row in rows:
            data = _join_state(row, _SKU_STATE_COLUMNS)
            state = _state_from_row(module.SkuState, data)
            out[(state.source, state.product_id)] = state
        return out

    def poll_store(self) -> "SqlitePollStore":
        """A :class:`~jarvis_poke.sources.PollStore` view of this file, to
        hand to ``PollScheduler`` so it saves its politeness state here."""
        return SqlitePollStore(self)


class SqlitePollStore:
    """A :class:`jarvis_poke.sources.PollStore` backed by a
    :class:`PokeStore`: ``load()``, ``save(snapshot)`` and ``lock()``.

    ``lock()`` is what makes the poll gate hold across *processes*.  The
    CLI refuses a long-running poller -- "a long-running poller belongs
    in the app" -- so two ``poll --once`` runs overlapping is the
    ordinary case: a cron entry that overran, a cron entry plus the app,
    the app's monitor loop plus an alert worker.  Without a
    cross-process lock each of them read the schedule, each decided the
    host was due, and each fetched, which is N times the agreed rate
    arriving as one burst -- the signature a retailer's WAF bans.

    It is a separate ``flock`` file rather than a sqlite transaction
    because the scheduler saves *inside* the locked region, and a store
    transaction held open across that would deadlock against its own
    connection lock.
    """

    def __init__(self, store: PokeStore) -> None:
        self.store = store
        self._lock = threading.RLock()
        self._depth = 0
        self._handle: Any = None

    def load(self) -> Optional[Dict[str, Any]]:
        return self.store.load_poll_snapshot()

    def save(self, snapshot: Dict[str, Any]) -> None:
        self.store.save_poll_snapshot(snapshot)

    def lock_path(self) -> Optional[str]:
        """The advisory lock file, or ``None`` for an in-memory store.

        An in-memory store is private to one process by construction, so
        there is nothing to exclude and no file worth creating.
        """
        if self.store.path in ("", ":memory:") or self.store.path.startswith("file::memory:"):
            return None
        return self.store.path + ".pollock"

    @contextlib.contextmanager
    def lock(self) -> "Iterator[None]":
        """Exclude other pollers, in this process and in every other.

        Re-entrant: the scheduler holds it across read-decide-claim and
        the nested ``save`` that claim makes.
        """
        with self._lock:
            path = self.lock_path()
            if path is None or fcntl is None:
                # No file to lock (``:memory:``) or no ``fcntl`` on this
                # platform: the in-process lock is all there is, and it
                # is still better than nothing.
                yield
                return
            if self._depth == 0:
                handle = open(path, "a+b")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                except OSError:
                    handle.close()
                    raise
                self._handle = handle
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1
                if self._depth == 0 and self._handle is not None:
                    handle, self._handle = self._handle, None
                    try:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    finally:
                        handle.close()

    def __repr__(self) -> str:
        return f"SqlitePollStore({self.store.path!r})"


# --------------------------------------------------------------------------
# Row helpers
# --------------------------------------------------------------------------


def _enum_value(member: Any, kind: Any, what: str) -> str:
    if not isinstance(member, kind):
        raise StoreError(f"{what} must be a {kind.__name__}; got {member!r}")
    return str(member.value)


def _no_duplicates(keys: Sequence[Any], what: str) -> None:
    seen = set()
    for key in keys:
        if key in seen:
            raise StoreError(f"duplicate {what}: {key!r}")
        seen.add(key)


def _sources_module() -> Any:
    """:mod:`jarvis_poke.sources`, imported late.

    Late so that a file of poll state can be opened, read and pruned by a
    process that never builds a scheduler, and so this module does not
    import a sibling at import time just to name two dataclasses.
    """
    from jarvis_poke import sources  # local import on purpose

    return sources


def _state_as_row(state: Any, what: str) -> Dict[str, Any]:
    if not dataclasses.is_dataclass(state) or isinstance(state, type):
        raise StoreError(f"not a {what} dataclass: {state!r}")
    return dataclasses.asdict(state)


def _split_state(
    row: Mapping[str, Any],
    columns: Sequence[str],
    defaults: Mapping[str, Any],
    what: str,
) -> Tuple[List[Any], Dict[str, Any]]:
    """Split a state mapping into this schema's columns and the rest.

    A column the row does not mention takes its default, so a hand-written
    or older snapshot still saves; a *key* column has no default and its
    absence is an error.  Anything :mod:`jarvis_poke.sources` grows later
    lands in ``extra`` as JSON, so a snapshot round-trips exactly without
    this file having to move in lockstep with a sibling module.
    """
    data = dict(row)
    values: List[Any] = []
    for column in columns:
        if column not in data:
            if column not in defaults:
                raise StoreError(f"{what} row is missing {column!r}")
            values.append(defaults[column])
            continue
        values.append(_state_value(data.pop(column), f"{what}.{column}"))
    try:
        json.dumps(data, sort_keys=True)
    except (TypeError, ValueError):
        raise StoreError(f"{what} row has a value that is not JSON: {sorted(data)}") from None
    return values, data


def _join_state(row: Sequence[Any], columns: Sequence[str]) -> Dict[str, Any]:
    """The inverse of :func:`_split_state`: columns plus the ``extra`` JSON."""
    data = {column: row[index] for index, column in enumerate(columns)}
    extra = json.loads(row[len(columns)])
    if not isinstance(extra, dict):
        raise StoreError("state extra column is not a JSON object")
    data.update(extra)
    return data


def _state_value(value: Any, what: str) -> Any:
    """Scheduler state is numbers, strings, ``None`` -- and no floats where
    the column is an ``INTEGER``, which sqlite would silently truncate."""
    if value is None or (isinstance(value, (str, int, float)) and not isinstance(value, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise StoreError(f"{what} must be finite; got {value!r}")
        return value
    if isinstance(value, bool):
        return 1 if value else 0
    raise StoreError(f"{what} must be a number, a string or None; got {type(value).__name__}")


def _state_from_row(cls: Any, row: Mapping[str, Any]) -> Any:
    """Build a state dataclass, dropping fields it does not have.

    The same forgiveness as ``jarvis_poke.sources._from_row``: a column
    from an older or newer build is ignored rather than fatal.
    """
    fields = set(getattr(cls, "__dataclass_fields__", {}))
    return cls(**{k: v for k, v in row.items() if k in fields})


# --------------------------------------------------------------------------
# Round-trip comparison: what the tests and the gate assert with
# --------------------------------------------------------------------------


def _sort_key(key: Any) -> Tuple[int, Any]:
    """A total order over the dictionary keys these entities can have, so a
    report is deterministic even for mixed key types."""
    if isinstance(key, enum.Enum):
        return (0, str(key.value))
    if isinstance(key, bool):
        return (1, int(key))
    if isinstance(key, (int, float)):
        return (1, key)
    if isinstance(key, str):
        return (2, key)
    if isinstance(key, tuple):
        return (3, repr(key))
    return (4, repr(key))


def _diff(a: Any, b: Any, path: str) -> Optional[str]:
    """The first place ``a`` and ``b`` differ, walking in a fixed order.

    Dataclass fields in declaration order, sequences by index, mappings in
    sorted key order.  ``list`` and ``tuple`` are treated alike, because a
    ``Rule``'s ``allowed_sources`` is a tuple that JSON brings back as a
    list and that is not a difference worth failing a gate over.

    ``int`` and ``float`` are *not* treated alike, which is the one place
    this walker is stricter than its cousin in ``lucifer_descent.store``:
    money is integer cents (contracts.py), so a ``4999`` that comes back as
    ``4999.0`` is exactly the bug the round trip is there to catch.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        if type(a) is not type(b) or a != b:
            return f"{path}: {a!r} != {b!r}"
        return None

    if isinstance(a, enum.Enum) or isinstance(b, enum.Enum):
        if a is not b:
            return f"{path}: {a!r} != {b!r}"
        return None

    if dataclasses.is_dataclass(a) and not isinstance(a, type):
        if type(a) is not type(b):
            return f"{path}: {type(a).__name__} != {type(b).__name__}"
        for f in dataclasses.fields(a):
            found = _diff(getattr(a, f.name), getattr(b, f.name), f"{path}.{f.name}")
            if found is not None:
                return found
        return None

    if isinstance(a, _abc.Mapping):
        if not isinstance(b, _abc.Mapping):
            return f"{path}: mapping != {type(b).__name__}"
        missing = sorted(set(a) - set(b), key=_sort_key)
        if missing:
            return f"{path}: key {missing[0]!r} is missing from the second"
        extra = sorted(set(b) - set(a), key=_sort_key)
        if extra:
            return f"{path}: key {extra[0]!r} is only in the second"
        for key in sorted(a, key=_sort_key):
            found = _diff(a[key], b[key], f"{path}[{key!r}]")
            if found is not None:
                return found
        return None

    if isinstance(a, (set, frozenset)):
        if not isinstance(b, (set, frozenset)):
            return f"{path}: set != {type(b).__name__}"
        missing = sorted(set(a) - set(b), key=_sort_key)
        if missing:
            return f"{path}: {missing[0]!r} is missing from the second"
        extra = sorted(set(b) - set(a), key=_sort_key)
        if extra:
            return f"{path}: {extra[0]!r} is only in the second"
        return None

    if isinstance(a, (list, tuple)):
        if not isinstance(b, (list, tuple)):
            return f"{path}: sequence != {type(b).__name__}"
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for index, (x, y) in enumerate(zip(a, b)):
            found = _diff(x, y, f"{path}[{index}]")
            if found is not None:
                return found
        return None

    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"

    if isinstance(a, float) and math.isnan(a) and math.isnan(b):
        return None
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def first_difference(a: Any, b: Any, path: str = "value") -> Optional[str]:
    """Describe the first field where two entities differ, or ``None``.

    The path is spelled from the root down, e.g.
    ``value.reasons[2]: 'under the cap' != 'over the cap'`` or
    ``value.price: 4999 != 4999.0``, so a failed round trip points at one
    field instead of dumping two objects side by side.
    """
    return _diff(a, b, path)


def round_trip_equal(a: Any, b: Any) -> bool:
    """True when what went into the store came back out identical.

    Works on anything this module stores: a :class:`Product`, an
    :class:`Observation`, a :class:`Verdict` with its reasons, a
    :class:`Budget` with its spend, a list of them, or the dicts of watch
    and poll state.  Use :func:`first_difference` for the reason when this
    is ``False``.
    """
    return first_difference(a, b) is None


def assert_round_trip_equal(a: Any, b: Any) -> None:
    """Raise :class:`AssertionError` naming the first difference, if any."""
    found = first_difference(a, b)
    if found is not None:
        raise AssertionError(f"round trip differs at {found}")


# --------------------------------------------------------------------------
# Smoke run: python3 -m jarvis_poke.store
# --------------------------------------------------------------------------


def _demo() -> Dict[str, Any]:
    """Save one of everything into a real file, read it back, prune, and
    report.  No network, no sleeping, a hand-driven clock."""
    import tempfile

    ticks = [1_700_000_000.0]
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "poke.sqlite3")
        with PokeStore(path, clock=lambda: ticks[0]) as store:
            product = Product(
                id="sv08-surging-sparks-etb",
                name="Placeholder Elite Trainer Box",
                set_code="SV08",
                kind=ProductKind.ELITE_TRAINER_BOX,
                msrp=4999,
            )
            sku = SourceSku(
                source="examplemart",
                product_id=product.id,
                sku="EM-SV08-ETB",
                url="https://examplemart.example.com/p/sv08-surging-sparks-etb",
            )
            store.save_catalog([product], [sku])
            for age in range(5):
                store.save_observation(
                    Observation(
                        product_id=product.id,
                        source=sku.source,
                        sku=sku.sku,
                        at=ticks[0] - age * 86400.0,
                        stock=Stock.IN_STOCK,
                        price=4499 + age,
                        shipping=0,
                        url=sku.url,
                    )
                )
            store.save_rules([Rule(product_id=product.id, max_price=4999, quantity=2)])
            store.save_budget(Budget(total=50_000, spent=4_499))
            store.save_watch_states(
                {product.id: WatchState(product_id=product.id, last_alert_at=ticks[0],
                                        last_action=Action.BUY, alerts_sent=1)}
            )
            verdict = Verdict(
                product_id=product.id,
                action=Action.BUY,
                at=ticks[0],
                source=sku.source,
                sku=sku.sku,
                price=4499,
                landed=4499,
                market=5200,
                discount_pct=13.5,
                quantity=2,
                url=sku.url,
                reasons=("in stock at examplemart", "13.5% under the market", "within budget"),
            )
            store.append_verdict(verdict)
            store.save_poll_snapshot(
                {
                    "version": 1,
                    "sources": {"examplemart": {"source": "examplemart",
                                                "last_attempt_at": ticks[0],
                                                "paused_until": 0.0,
                                                "consecutive_errors": 0}},
                    "skus": [{"source": "examplemart", "product_id": product.id,
                              "etag": 'W/"7"', "last_modified": None,
                              "last_attempt_at": ticks[0]}],
                }
            )
            pruned = store.prune_observations(window_s=2 * 86400.0)
            back = store.verdicts(product.id)[0]
            return {
                "path_mode": oct(os.stat(path).st_mode & 0o777),
                "schema_version": store.schema_version(),
                "products": len(store.load_products()),
                "skus": len(store.load_source_skus()),
                "observations_left": len(store.load_observations()),
                "pruned": pruned,
                "budget_remaining": (store.load_budget() or Budget(0)).remaining,
                "verdict_round_trip": round_trip_equal(verdict, back),
                "verdict_difference": first_difference(verdict, back),
                "poll_sources": sorted((store.load_poll_snapshot() or {}).get("sources", {})),
                "etag_survived": store.load_poll_snapshot()["skus"][0]["etag"],
            }


if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    print(json.dumps(_demo(), indent=2, sort_keys=True))
