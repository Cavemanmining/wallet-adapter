"""Persistence for the Descent: a :class:`ProfileState` in, the same one out.

Spec: docs/WORLD_BIBLE.md section 03 -- "a profile owns one web, one passive
ledger, one Sigil stash and its fragments"; section 02 for the 64-bit seed
that a Sigil carries and a portal turns into a map seed.

Two backends sit behind one interface, :class:`DescentStore`:

* :class:`MemoryStore` keeps profiles in a dict.  It deep-copies on the way
  in and on the way out, so a caller that keeps mutating its own state after
  ``save`` cannot reach into the store, and a loaded state is the caller's to
  change.
* :class:`SqliteStore` keeps them in one standard-library ``sqlite3`` file in
  WAL mode with foreign keys on.  Every ``save`` is a single ``BEGIN
  IMMEDIATE ... COMMIT`` that deletes the profile's old rows and inserts the
  new ones; the profile row cascades to every child table, so a crash or an
  exception anywhere inside the transaction rolls the whole profile back to
  what it was before.  There is no moment at which a reader can see half a
  profile.

Three rules the schema follows, and why:

* **Seeds are TEXT, never INTEGER.**  A seed is 64-bit *unsigned* (section
  02).  SQLite's INTEGER is signed 64-bit, so a seed with its top bit set
  would come back negative -- and, worse, ``seed & MASK64`` would "fix" it
  silently downstream.  Every seed column holds the ``0x%016X`` string that
  :func:`lucifer_gen.seed.format_seed` produces and
  :func:`lucifer_gen.seed.parse_seed` reads back.
* **Colour is never stored.**  ``node_states`` holds the :class:`NodeState`
  value only; the colour is a pure function of it (section 03: "colour is
  derived from state, never stored").
* **Order is data.**  The ledger is replayed in order, and the web's node and
  edge tuples are compared field by field by :func:`round_trip_equal`, so
  ``web_nodes`` and ``web_edges`` carry a ``seq`` column and the ledger is
  keyed by its own ``seq``.  Dictionaries (states, stash, fragments) have no
  meaningful order and are rebuilt in sorted key order, which also keeps the
  loaded state deterministic.

Two more rules, on the way *out* of storage and on the way back in:

* **Reads are strict.**  A seed column is read only as the exact
  ``0x`` + 16 upper-case hex digits :func:`format_seed` writes; an INTEGER
  that crept in, a negative number, a 65-bit value or a decimal string is
  refused with :class:`ValueError` instead of being masked into a value
  that was never stored.  Integer columns must hold integers, and state and
  event columns must name a member of their enum.
* **Saves are compare-and-swap.**  Every profile row carries a ``revision``
  that grows by one on each save.  A state that came out of :meth:`load`
  remembers the revision it was loaded at, and :meth:`save` refuses it with
  :class:`StaleState` when the stored revision has moved on -- two shells on
  one profile, or a state kept across another save, cannot silently roll a
  spent Sigil back into the stash.  A state that was never loaded (revision
  0, e.g. a freshly created profile) replaces whatever is stored, and so
  does a state loaded from a *different* store (the stamp names the store
  it came from, so copying a profile between files is a plain save, not a
  conflict).  The revision is bookkeeping, not profile content: it lives
  on the state object as a private attribute, never in a dataclass field,
  so :func:`round_trip_equal` ignores it.

The store draws no randomness at all; nothing here needs a
:class:`lucifer_gen.seed.Stream`.
"""

from __future__ import annotations

import abc
import contextlib
import copy
import dataclasses
import enum
import json
import math
import os
import re
import sqlite3
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from lucifer_descent.contracts import (
    Event,
    Instance,
    LedgerEntry,
    Mechanic,
    NodeState,
    Pinnacle,
    ProfileState,
    Sigil,
    Web,
    WebEdge,
    WebNode,
)
from lucifer_gen.seed import MASK64, format_seed

__all__ = [
    "SCHEMA_VERSION",
    "SchemaError",
    "StaleState",
    "revision_of",
    "DescentStore",
    "MemoryStore",
    "SqliteStore",
    "validate_state",
    "first_difference",
    "round_trip_equal",
    "assert_round_trip_equal",
]

#: Bumped whenever the table layout changes.  :meth:`SqliteStore.migrate`
#: refuses a file written by a different version rather than guessing.
#: Version 2 added ``ledger.has_boss``, ``ledger.elite_total`` (the probe's
#: facts on every ``OPEN``) and ``profiles.revision``; a version-1 file has
#: no facts to migrate, so it is refused rather than half-upgraded.
SCHEMA_VERSION = 2


class SchemaError(RuntimeError):
    """The database file's schema version is not one this module can read."""


class StaleState(RuntimeError):
    """The profile was saved by someone else since this state was loaded."""


#: Where a loaded state remembers ``(store key, revision)``; not a dataclass field.
_REVISION_ATTR = "_store_revision"


def revision_of(state: ProfileState, store: Optional["DescentStore"] = None) -> int:
    """The revision ``state`` was loaded or last saved at; 0 if never stored.

    With ``store`` given, 0 also when the stamp belongs to another store.
    """
    stamp = getattr(state, _REVISION_ATTR, None)
    if stamp is None:
        return 0
    key, revision = stamp
    if store is not None and key != store._revision_key():
        return 0
    return int(revision)


def _stamp_revision(state: ProfileState, key: object, revision: int) -> None:
    setattr(state, _REVISION_ATTR, (key, revision))


#: The one shape a stored seed may have: what :func:`format_seed` writes.
_SEED_TEXT = re.compile(r"^0x[0-9A-F]{16}$")


def _read_seed(value: Any, what: str) -> int:
    """Parse a seed column strictly; see the module docstring."""
    if not isinstance(value, str) or not _SEED_TEXT.match(value):
        raise ValueError(f"{what}: stored seed {value!r} is not 0x + 16 upper-case hex digits")
    return int(value[2:], 16)


def _read_int(value: Any, what: str) -> int:
    """An INTEGER column must come back as an int, not a float, str or bool."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what}: stored value {value!r} is not an integer")
    return value


def _read_bool(value: Any, what: str) -> bool:
    if value not in (0, 1) or isinstance(value, bool) and value not in (False, True):
        raise ValueError(f"{what}: stored value {value!r} is not 0 or 1")
    return bool(value)


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


class DescentStore(abc.ABC):
    """Where a profile's Descent lives between sessions.

    Spec section 03: a profile owns exactly one web, one passive ledger, one
    Sigil stash and its fragments, so the unit of storage is the whole
    :class:`ProfileState` and nothing smaller.  ``save`` replaces the
    profile's previous state wholesale; there is no partial update.
    """

    @abc.abstractmethod
    def load(self, profile_id: str) -> Optional[ProfileState]:
        """The profile's last saved state, or ``None`` if it was never saved."""

    @abc.abstractmethod
    def save(self, state: ProfileState) -> None:
        """Replace whatever was saved for ``state.profile_id`` with ``state``.

        Either the whole new state is stored or nothing changes; a backend
        must never leave a profile half-written.  Raises :class:`ValueError`
        for a state that could not be read back identically (see
        :func:`validate_state`), before touching storage.
        """

    @abc.abstractmethod
    def list_profiles(self) -> List[str]:
        """Every saved profile id, sorted."""

    @abc.abstractmethod
    def delete(self, profile_id: str) -> bool:
        """Forget a profile entirely.  True if there was one to forget."""

    def _revision_key(self) -> object:
        """What a loaded state's revision stamp names as its origin.

        Two store objects over the same underlying data must share a key,
        so a state loaded through one connection and saved through another
        is still held to the revision it was loaded at.
        """
        return id(self)


# --------------------------------------------------------------------------
# Validation shared by both backends
# --------------------------------------------------------------------------


def _check_int(value: Any, what: str, minimum: Optional[int] = None) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{what} must be an int, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be at least {minimum}, got {value}")


def _check_seed(value: Any, what: str) -> None:
    _check_int(value, what)
    if not 0 <= value <= MASK64:
        raise ValueError(f"{what} is not a 64-bit unsigned value: {value!r}")


def _check_sigil(sigil: Sigil, what: str) -> None:
    _check_int(sigil.tier, f"{what}.tier")
    _check_seed(sigil.seed, f"{what}.seed")


def validate_state(state: ProfileState) -> None:
    """Refuse a state that no backend could hand back unchanged.

    Both backends call this before storing anything, so a state that one of
    them would reject is rejected by the other too and the two never drift.
    The checks are exactly the assumptions the SQLite schema makes:

    * every seed (profile, stash, live Sigil, map) fits in 64 unsigned bits,
      the only shape a seed column can hold (section 02);
    * node ids are unique, and every node-state, edge end and live instance
      names a node that exists -- the composite foreign keys say the same;
    * a stash entry's key is its Sigil's id, because the table has one column
      for both;
    * ledger ``seq`` values strictly increase along the list, because the
      ledger is stored keyed by ``seq`` and read back ordered by it;
    * every INTEGER column's value is an ``int`` (a float tier or a bool
      tick would be silently rewritten by the column's affinity);
    * layout coordinates are finite, because SQLite stores NaN as NULL.
    """
    if not isinstance(state.profile_id, str) or not state.profile_id:
        raise ValueError("profile_id must be a non-empty string")

    web = state.web
    _check_seed(web.profile_seed, "web.profile_seed")

    ids: List[int] = []
    for index, node in enumerate(web.nodes):
        if node.id in ids:
            raise ValueError(f"web.nodes[{index}]: duplicate node id {node.id}")
        ids.append(node.id)
        for axis, value in (("x", node.x), ("y", node.y)):
            if not math.isfinite(value):
                raise ValueError(f"web.nodes[{index}].{axis} is not finite: {value!r}")
    id_set = set(ids)
    if web.origin_id not in id_set:
        raise ValueError(f"web.origin_id {web.origin_id} is not a node")
    for index, edge in enumerate(web.edges):
        for end in (edge.a, edge.b):
            if end not in id_set:
                raise ValueError(f"web.edges[{index}] touches unknown node {end}")

    for node_id in state.states:
        if node_id not in id_set:
            raise ValueError(f"states has an entry for unknown node {node_id}")

    for key, sigil in state.stash.items():
        if key != sigil.id:
            raise ValueError(f"stash key {key!r} does not match its Sigil id {sigil.id!r}")
        _check_sigil(sigil, f"stash[{key!r}]")

    _check_int(state.passive_points, "passive_points")
    for pinnacle, count in state.fragments.items():
        _check_int(count, f"fragments[{pinnacle!r}]")

    inst = state.instance
    if inst is not None:
        if inst.node_id not in id_set:
            raise ValueError(f"instance.node_id {inst.node_id} is not a node")
        _check_sigil(inst.sigil, "instance.sigil")
        _check_seed(inst.map_seed, "instance.map_seed")
        if not isinstance(inst.has_boss, bool):
            raise ValueError(f"instance.has_boss must be a bool, got {type(inst.has_boss).__name__}")
        _check_int(inst.elite_total, "instance.elite_total")
        _check_int(inst.elite_killed, "instance.elite_killed")
        _check_int(inst.opened_tick, "instance.opened_tick")

    last_seq: Optional[int] = None
    for index, entry in enumerate(state.history):
        _check_int(entry.seq, f"history[{index}].seq")
        _check_int(entry.node_id, f"history[{index}].node_id")
        _check_int(entry.tick, f"history[{index}].tick")
        if entry.has_boss is not None and not isinstance(entry.has_boss, bool):
            raise ValueError(f"history[{index}].has_boss must be a bool or None")
        if entry.elite_total is not None:
            _check_int(entry.elite_total, f"history[{index}].elite_total")
        if last_seq is not None and entry.seq <= last_seq:
            raise ValueError(
                f"history[{index}].seq {entry.seq} does not follow {last_seq}; "
                "ledger seq must strictly increase"
            )
        last_seq = entry.seq


# --------------------------------------------------------------------------
# MemoryStore
# --------------------------------------------------------------------------


class MemoryStore(DescentStore):
    """A dict of profiles, for tests and for the table's in-session cache.

    Copies on both sides of the boundary, so the store behaves like the
    SQLite one: what you saved is what you get back, however you mutated
    your own object in between.
    """

    def __init__(self) -> None:
        self._profiles: Dict[str, ProfileState] = {}
        self._revisions: Dict[str, int] = {}

    def load(self, profile_id: str) -> Optional[ProfileState]:
        stored = self._profiles.get(profile_id)
        if stored is None:
            return None
        loaded = copy.deepcopy(stored)
        _stamp_revision(loaded, self._revision_key(), self._revisions[profile_id])
        return loaded

    def save(self, state: ProfileState) -> None:
        validate_state(state)
        current = self._revisions.get(state.profile_id, 0)
        _check_not_stale(state, self, current)
        self._profiles[state.profile_id] = copy.deepcopy(state)
        self._revisions[state.profile_id] = current + 1
        _stamp_revision(state, self._revision_key(), current + 1)

    def list_profiles(self) -> List[str]:
        return sorted(self._profiles)

    def delete(self, profile_id: str) -> bool:
        # The revision goes with the profile, as the SQLite row does; a state
        # loaded before the delete is then stale, since what it was built on
        # is gone, and a never-loaded state can start the profile again.
        self._revisions.pop(profile_id, None)
        return self._profiles.pop(profile_id, None) is not None


def _check_not_stale(state: ProfileState, store: DescentStore, stored_revision: int) -> None:
    """A loaded state may only be saved over the revision it was loaded at."""
    mine = revision_of(state, store)
    if mine != 0 and mine != stored_revision:
        raise StaleState(
            f"profile {state.profile_id!r} is at revision {stored_revision} in the store "
            f"but this state was loaded at revision {mine}; reload it and redo the change"
        )


# --------------------------------------------------------------------------
# SqliteStore
# --------------------------------------------------------------------------

# ``before``, ``after`` and ``count`` are SQL keywords or function names, so
# they are quoted everywhere they appear.
_SCHEMA: Tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        id      INTEGER PRIMARY KEY CHECK (id = 1),
        version INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS profiles (
        profile_id         TEXT    PRIMARY KEY,
        profile_seed       TEXT    NOT NULL,
        origin_id          INTEGER NOT NULL,
        web_version        INTEGER NOT NULL,
        passive_points     INTEGER NOT NULL,
        unlocked_pinnacles TEXT    NOT NULL,
        revision           INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS web_nodes (
        profile_id TEXT    NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
        seq        INTEGER NOT NULL,
        id         INTEGER NOT NULL,
        tier       INTEGER NOT NULL,
        ring_index INTEGER NOT NULL,
        template   TEXT    NOT NULL,
        mechanic   TEXT,
        pinnacle   TEXT,
        glyph      TEXT,
        x          REAL    NOT NULL,
        y          REAL    NOT NULL,
        PRIMARY KEY (profile_id, seq),
        UNIQUE (profile_id, id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS web_edges (
        profile_id TEXT    NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
        seq        INTEGER NOT NULL,
        a          INTEGER NOT NULL,
        b          INTEGER NOT NULL,
        PRIMARY KEY (profile_id, seq),
        FOREIGN KEY (profile_id, a) REFERENCES web_nodes(profile_id, id) ON DELETE CASCADE,
        FOREIGN KEY (profile_id, b) REFERENCES web_nodes(profile_id, id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS node_states (
        profile_id TEXT    NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
        node_id    INTEGER NOT NULL,
        state      TEXT    NOT NULL,
        PRIMARY KEY (profile_id, node_id),
        FOREIGN KEY (profile_id, node_id) REFERENCES web_nodes(profile_id, id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stash (
        profile_id TEXT    NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
        sigil_id   TEXT    NOT NULL,
        tier       INTEGER NOT NULL,
        seed       TEXT    NOT NULL,
        PRIMARY KEY (profile_id, sigil_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fragments (
        profile_id TEXT    NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
        pinnacle   TEXT    NOT NULL,
        "count"    INTEGER NOT NULL,
        PRIMARY KEY (profile_id, pinnacle)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS instance (
        profile_id   TEXT    PRIMARY KEY REFERENCES profiles(profile_id) ON DELETE CASCADE,
        node_id      INTEGER NOT NULL,
        sigil_id     TEXT    NOT NULL,
        sigil_tier   INTEGER NOT NULL,
        sigil_seed   TEXT    NOT NULL,
        map_seed     TEXT    NOT NULL,
        has_boss     INTEGER NOT NULL,
        elite_total  INTEGER NOT NULL,
        elite_killed INTEGER NOT NULL,
        opened_tick  INTEGER NOT NULL,
        FOREIGN KEY (profile_id, node_id) REFERENCES web_nodes(profile_id, id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS ledger (
        profile_id TEXT    NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
        seq        INTEGER NOT NULL,
        node_id    INTEGER NOT NULL,
        event      TEXT    NOT NULL,
        "before"   TEXT    NOT NULL,
        "after"    TEXT    NOT NULL,
        sigil_id   TEXT,
        tick       INTEGER NOT NULL,
        has_boss   INTEGER,
        elite_total INTEGER,
        PRIMARY KEY (profile_id, seq)
    )
    """,
)


def _enum_or_none(kind: Any, value: Optional[str]) -> Any:
    return None if value is None else kind(value)


def _value_or_none(member: Optional[enum.Enum]) -> Optional[str]:
    return None if member is None else member.value


class SqliteStore(DescentStore):
    """One ``sqlite3`` file holding every profile.

    ``path`` may be a filesystem path or ``":memory:"``.  The connection is
    opened in autocommit mode so that transaction boundaries are the explicit
    ``BEGIN IMMEDIATE`` / ``COMMIT`` / ``ROLLBACK`` statements in this class
    and nothing is left to the driver's implicit behaviour.  WAL mode lets
    the 170HX pre-roller read a profile while the table is saving another,
    and ``foreign_keys=ON`` makes the profile row's cascades real.
    """

    def __init__(self, path: Union[str, "os.PathLike[str]"]) -> None:
        self.path = os.fspath(path)
        # An in-memory database is private to its connection, so its key is
        # this object; a file is shared by every connection to it.
        self._key: object = id(self) if self.path == ":memory:" else os.path.realpath(self.path)
        self._conn = sqlite3.connect(self.path, isolation_level=None)
        # Both pragmas must run outside a transaction; autocommit mode does that.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self.migrate()

    # -- lifecycle ----------------------------------------------------------

    def _revision_key(self) -> object:
        return self._key

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteStore":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- transactions -------------------------------------------------------

    @contextlib.contextmanager
    def _transaction(self, mode: str = "IMMEDIATE") -> Iterator[sqlite3.Cursor]:
        """Run the block inside one transaction; roll back on any exception.

        ``IMMEDIATE`` takes the write lock up front so a save never has to
        upgrade a read lock halfway through and fail with ``SQLITE_BUSY``
        after it has already deleted the old rows.  ``DEFERRED`` is used for
        reads, which get a consistent snapshot under WAL.
        """
        cur = self._conn.cursor()
        cur.execute(f"BEGIN {mode}")
        try:
            yield cur
            # Inside the try on purpose: a COMMIT that fails (disk full at
            # the last moment, an interrupted WAL write) must roll back too,
            # or the connection is left inside an open transaction and every
            # later BEGIN fails with "cannot start a transaction within a
            # transaction".
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
        """Create any missing tables and record :data:`SCHEMA_VERSION`.

        Idempotent: opening an existing file of the same version is a no-op
        beyond the ``IF NOT EXISTS`` statements.  A file stamped with a
        different version raises :class:`SchemaError`; there is no upgrade
        path and pretending otherwise would corrupt profiles quietly.  In
        particular a version-1 file has ``OPEN`` entries without the probe
        facts version 2 records, and those cannot be recovered.
        """
        with self._transaction() as cur:
            for statement in _SCHEMA:
                cur.execute(statement)
            row = cur.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
            if row is None:
                cur.execute(
                    "INSERT INTO schema_version (id, version) VALUES (1, ?)",
                    (SCHEMA_VERSION,),
                )
            elif row[0] != SCHEMA_VERSION:
                raise SchemaError(
                    f"{self.path}: schema version {row[0]} is not the supported "
                    f"version {SCHEMA_VERSION}"
                )

    def schema_version(self) -> int:
        row = self._conn.execute("SELECT version FROM schema_version WHERE id = 1").fetchone()
        if row is None:
            raise SchemaError(f"{self.path}: schema_version row is missing")
        return int(row[0])

    # -- DescentStore -------------------------------------------------------

    def list_profiles(self) -> List[str]:
        rows = self._conn.execute("SELECT profile_id FROM profiles ORDER BY profile_id").fetchall()
        return [row[0] for row in rows]

    def delete(self, profile_id: str) -> bool:
        with self._transaction() as cur:
            cur.execute("DELETE FROM profiles WHERE profile_id = ?", (profile_id,))
            return cur.rowcount > 0

    def save(self, state: ProfileState) -> None:
        """Replace the profile's rows in one transaction.

        The old profile row is deleted first, which cascades through every
        child table, then the new rows go in parents-first so the foreign
        keys are satisfied at each statement.  The per-table ``_insert_*``
        methods are the seams a test can break to prove the rollback: an
        exception from any of them leaves the previous profile untouched.
        """
        validate_state(state)
        with self._transaction() as cur:
            row = cur.execute(
                "SELECT revision FROM profiles WHERE profile_id = ?", (state.profile_id,)
            ).fetchone()
            current = _read_int(row[0], "profiles.revision") if row is not None else 0
            _check_not_stale(state, self, current)
            cur.execute("DELETE FROM profiles WHERE profile_id = ?", (state.profile_id,))
            self._insert_profile(cur, state, current + 1)
            self._insert_web(cur, state)
            self._insert_states(cur, state)
            self._insert_stash(cur, state)
            self._insert_fragments(cur, state)
            self._insert_instance(cur, state)
            self._insert_ledger(cur, state)
        _stamp_revision(state, self._key, current + 1)

    def load(self, profile_id: str) -> Optional[ProfileState]:
        with self._transaction("DEFERRED") as cur:
            row = cur.execute(
                "SELECT profile_seed, origin_id, web_version, passive_points, unlocked_pinnacles, "
                "revision FROM profiles WHERE profile_id = ?",
                (profile_id,),
            ).fetchone()
            if row is None:
                return None
            profile_seed, origin_id, web_version, passive_points, unlocked_json, revision = row
            web = Web(
                profile_seed=_read_seed(profile_seed, "profiles.profile_seed"),
                origin_id=_read_int(origin_id, "profiles.origin_id"),
                nodes=self._load_nodes(cur, profile_id),
                edges=self._load_edges(cur, profile_id),
                version=_read_int(web_version, "profiles.web_version"),
            )
            state = ProfileState(
                profile_id=profile_id,
                web=web,
                states=self._load_states(cur, profile_id),
                stash=self._load_stash(cur, profile_id),
                passive_points=_read_int(passive_points, "profiles.passive_points"),
                fragments=self._load_fragments(cur, profile_id),
                unlocked_pinnacles=frozenset(Pinnacle(v) for v in json.loads(unlocked_json)),
                instance=self._load_instance(cur, profile_id),
                history=self._load_ledger(cur, profile_id),
            )
        _stamp_revision(state, self._key, _read_int(revision, "profiles.revision"))
        return state

    # -- writers, one per table ---------------------------------------------

    def _insert_profile(self, cur: sqlite3.Cursor, state: ProfileState, revision: int) -> None:
        cur.execute(
            "INSERT INTO profiles "
            "(profile_id, profile_seed, origin_id, web_version, passive_points, "
            "unlocked_pinnacles, revision) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                state.profile_id,
                format_seed(state.web.profile_seed),
                state.web.origin_id,
                state.web.version,
                state.passive_points,
                json.dumps(sorted(p.value for p in state.unlocked_pinnacles)),
                revision,
            ),
        )

    def _insert_web(self, cur: sqlite3.Cursor, state: ProfileState) -> None:
        pid = state.profile_id
        cur.executemany(
            "INSERT INTO web_nodes "
            "(profile_id, seq, id, tier, ring_index, template, mechanic, pinnacle, glyph, x, y) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    pid, seq, n.id, n.tier, n.ring_index, n.template,
                    _value_or_none(n.mechanic), _value_or_none(n.pinnacle),
                    _value_or_none(n.glyph), float(n.x), float(n.y),
                )
                for seq, n in enumerate(state.web.nodes)
            ],
        )
        cur.executemany(
            "INSERT INTO web_edges (profile_id, seq, a, b) VALUES (?, ?, ?, ?)",
            [(pid, seq, e.a, e.b) for seq, e in enumerate(state.web.edges)],
        )

    def _insert_states(self, cur: sqlite3.Cursor, state: ProfileState) -> None:
        cur.executemany(
            "INSERT INTO node_states (profile_id, node_id, state) VALUES (?, ?, ?)",
            [
                (state.profile_id, node_id, state.states[node_id].value)
                for node_id in sorted(state.states)
            ],
        )

    def _insert_stash(self, cur: sqlite3.Cursor, state: ProfileState) -> None:
        cur.executemany(
            "INSERT INTO stash (profile_id, sigil_id, tier, seed) VALUES (?, ?, ?, ?)",
            [
                (state.profile_id, key, state.stash[key].tier, format_seed(state.stash[key].seed))
                for key in sorted(state.stash)
            ],
        )

    def _insert_fragments(self, cur: sqlite3.Cursor, state: ProfileState) -> None:
        cur.executemany(
            'INSERT INTO fragments (profile_id, pinnacle, "count") VALUES (?, ?, ?)',
            [
                (state.profile_id, pinnacle.value, state.fragments[pinnacle])
                for pinnacle in sorted(state.fragments, key=lambda p: p.value)
            ],
        )

    def _insert_instance(self, cur: sqlite3.Cursor, state: ProfileState) -> None:
        inst = state.instance
        if inst is None:
            return
        cur.execute(
            "INSERT INTO instance "
            "(profile_id, node_id, sigil_id, sigil_tier, sigil_seed, map_seed, "
            "has_boss, elite_total, elite_killed, opened_tick) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                state.profile_id, inst.node_id, inst.sigil.id, inst.sigil.tier,
                format_seed(inst.sigil.seed), format_seed(inst.map_seed),
                1 if inst.has_boss else 0, inst.elite_total, inst.elite_killed,
                inst.opened_tick,
            ),
        )

    def _insert_ledger(self, cur: sqlite3.Cursor, state: ProfileState) -> None:
        cur.executemany(
            "INSERT INTO ledger "
            '(profile_id, seq, node_id, event, "before", "after", sigil_id, tick, '
            "has_boss, elite_total) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    state.profile_id, e.seq, e.node_id, e.event.value,
                    e.before.value, e.after.value, e.sigil_id, e.tick,
                    None if e.has_boss is None else (1 if e.has_boss else 0),
                    e.elite_total,
                )
                for e in state.history
            ],
        )

    # -- readers, one per table ---------------------------------------------

    @staticmethod
    def _load_nodes(cur: sqlite3.Cursor, pid: str) -> Tuple[WebNode, ...]:
        rows = cur.execute(
            "SELECT id, tier, ring_index, template, mechanic, pinnacle, glyph, x, y "
            "FROM web_nodes WHERE profile_id = ? ORDER BY seq",
            (pid,),
        ).fetchall()
        return tuple(
            WebNode(
                id=_read_int(r[0], "web_nodes.id"), tier=_read_int(r[1], "web_nodes.tier"),
                ring_index=_read_int(r[2], "web_nodes.ring_index"), template=r[3],
                mechanic=_enum_or_none(Mechanic, r[4]),
                pinnacle=_enum_or_none(Pinnacle, r[5]),
                glyph=_enum_or_none(Pinnacle, r[6]),
                x=float(r[7]), y=float(r[8]),
            )
            for r in rows
        )

    @staticmethod
    def _load_edges(cur: sqlite3.Cursor, pid: str) -> Tuple[WebEdge, ...]:
        rows = cur.execute(
            "SELECT a, b FROM web_edges WHERE profile_id = ? ORDER BY seq", (pid,)
        ).fetchall()
        return tuple(WebEdge(_read_int(a, "web_edges.a"), _read_int(b, "web_edges.b")) for a, b in rows)

    @staticmethod
    def _load_states(cur: sqlite3.Cursor, pid: str) -> Dict[int, NodeState]:
        rows = cur.execute(
            "SELECT node_id, state FROM node_states WHERE profile_id = ? ORDER BY node_id", (pid,)
        ).fetchall()
        return {_read_int(node_id, "node_states.node_id"): NodeState(value) for node_id, value in rows}

    @staticmethod
    def _load_stash(cur: sqlite3.Cursor, pid: str) -> Dict[str, Sigil]:
        rows = cur.execute(
            "SELECT sigil_id, tier, seed FROM stash WHERE profile_id = ? ORDER BY sigil_id", (pid,)
        ).fetchall()
        return {
            sid: Sigil(id=sid, tier=_read_int(tier, f"stash[{sid!r}].tier"), seed=_read_seed(seed, f"stash[{sid!r}].seed"))
            for sid, tier, seed in rows
        }

    @staticmethod
    def _load_fragments(cur: sqlite3.Cursor, pid: str) -> Dict[Pinnacle, int]:
        rows = cur.execute(
            'SELECT pinnacle, "count" FROM fragments WHERE profile_id = ? ORDER BY pinnacle', (pid,)
        ).fetchall()
        return {Pinnacle(value): _read_int(count, f"fragments[{value!r}]") for value, count in rows}

    @staticmethod
    def _load_instance(cur: sqlite3.Cursor, pid: str) -> Optional[Instance]:
        row = cur.execute(
            "SELECT node_id, sigil_id, sigil_tier, sigil_seed, map_seed, "
            "has_boss, elite_total, elite_killed, opened_tick "
            "FROM instance WHERE profile_id = ?",
            (pid,),
        ).fetchone()
        if row is None:
            return None
        return Instance(
            node_id=_read_int(row[0], "instance.node_id"),
            sigil=Sigil(
                id=row[1], tier=_read_int(row[2], "instance.sigil_tier"),
                seed=_read_seed(row[3], "instance.sigil_seed"),
            ),
            map_seed=_read_seed(row[4], "instance.map_seed"),
            has_boss=_read_bool(row[5], "instance.has_boss"),
            elite_total=_read_int(row[6], "instance.elite_total"),
            elite_killed=_read_int(row[7], "instance.elite_killed"),
            opened_tick=_read_int(row[8], "instance.opened_tick"),
        )

    @staticmethod
    def _load_ledger(cur: sqlite3.Cursor, pid: str) -> List[LedgerEntry]:
        rows = cur.execute(
            'SELECT seq, node_id, event, "before", "after", sigil_id, tick, has_boss, elite_total '
            "FROM ledger WHERE profile_id = ? ORDER BY seq",
            (pid,),
        ).fetchall()
        return [
            LedgerEntry(
                seq=_read_int(r[0], "ledger.seq"), node_id=_read_int(r[1], "ledger.node_id"),
                event=Event(r[2]), before=NodeState(r[3]), after=NodeState(r[4]),
                sigil_id=r[5], tick=_read_int(r[6], "ledger.tick"),
                has_boss=None if r[7] is None else _read_bool(r[7], "ledger.has_boss"),
                elite_total=None if r[8] is None else _read_int(r[8], "ledger.elite_total"),
            )
            for r in rows
        ]


# --------------------------------------------------------------------------
# Round-trip comparison
# --------------------------------------------------------------------------


def _sort_key(key: Any) -> Tuple[int, Any]:
    """A total order over the dictionary keys a ProfileState can have.

    Keys are ints (node ids), strs (Sigil ids) or enums (Pinnacles); the rank
    keeps different kinds apart so mixed keys never raise ``TypeError``.
    """
    if isinstance(key, enum.Enum):
        return (0, str(key.value))
    if isinstance(key, (int, float)) and not isinstance(key, bool):
        return (1, key)
    if isinstance(key, str):
        return (2, key)
    return (3, repr(key))


def _diff(a: Any, b: Any, path: str) -> Optional[str]:
    """The first place ``a`` and ``b`` differ, walking in a fixed order.

    Dataclass fields are walked in declaration order, sequences by index,
    dictionaries and sets in sorted key order, so the report is
    deterministic.  ``list`` and ``tuple`` are treated alike, as are ``set``
    and ``frozenset``, and an ``int`` may equal a ``float``; everything else
    must match in type as well as value.  Enum members compare by identity.
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

    if isinstance(a, dict):
        if not isinstance(b, dict):
            return f"{path}: dict != {type(b).__name__}"
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
        missing = sorted(a - b, key=_sort_key)
        if missing:
            return f"{path}: {missing[0]!r} is missing from the second"
        extra = sorted(b - a, key=_sort_key)
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

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a != b and not (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)):
            return f"{path}: {a!r} != {b!r}"
        return None

    if type(a) is not type(b):
        return f"{path}: {type(a).__name__} != {type(b).__name__}"
    if a != b:
        return f"{path}: {a!r} != {b!r}"
    return None


def first_difference(a: ProfileState, b: ProfileState) -> Optional[str]:
    """Describe the first field where two states differ, or ``None``.

    The path is spelled from the state down, e.g.
    ``state.web.nodes[3].tier: 4 != 5`` or
    ``state.history: length 7 != 6``, so a failed gate points at one field.
    """
    return _diff(a, b, "state")


def round_trip_equal(a: ProfileState, b: ProfileState) -> bool:
    """True when two states are identical field by field.

    Covers the web (node and edge order included), node states, stash,
    passive points, fragments, unlocked Pinnacles, the live instance and the
    ledger in order.  Use :func:`first_difference` for the reason when this
    is False.
    """
    return first_difference(a, b) is None


def assert_round_trip_equal(a: ProfileState, b: ProfileState) -> None:
    """Raise :class:`AssertionError` naming the first difference, if any."""
    found = first_difference(a, b)
    if found is not None:
        raise AssertionError(f"states differ at {found}")
