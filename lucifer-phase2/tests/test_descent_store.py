"""Tests for the Descent profile store.

Spec: docs/WORLD_BIBLE.md section 03 (a profile owns one web, one passive
ledger, one Sigil stash and its fragments) and section 02 (64-bit seeds).

The web is hand-built so a failure points at the store, not at the web
generator.  Every behavioural test runs against both backends through the
``backend`` fixture; the SQLite-only tests look at the file itself: seed
columns are TEXT, WAL and foreign keys are on, and an exception injected
into the middle of a save rolls the whole profile back.
"""

from __future__ import annotations

import copy
import sqlite3
import sys
from pathlib import Path
from typing import Callable, List

# Runnable as `pytest tests/test_descent_store.py` or
# `python3 tests/test_descent_store.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

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
from lucifer_descent.store import (
    SCHEMA_VERSION,
    DescentStore,
    MemoryStore,
    SchemaError,
    SqliteStore,
    assert_round_trip_equal,
    first_difference,
    round_trip_equal,
    validate_state,
)
from lucifer_gen.seed import MASK64

# --------------------------------------------------------------------------
# Fixtures: a small web and a busy profile on it
# --------------------------------------------------------------------------

TOP_BIT = 1 << 63


def make_web(profile_seed: int = 0x1234_5678_9ABC_DEF0) -> Web:
    """A diamond with a tail out to a tier-15 arena.

    Node ids are deliberately *not* in tuple order (2 before 1) so that a
    store which sorted nodes by id instead of keeping their order would be
    caught by :func:`round_trip_equal`.
    """
    nodes = (
        WebNode(id=0, tier=0, ring_index=0, template="origin", x=0.0, y=0.0),
        WebNode(id=2, tier=1, ring_index=1, template="crypt", mechanic=Mechanic.DIG, x=-3.5, y=1.25),
        WebNode(id=1, tier=1, ring_index=0, template="crypt", mechanic=Mechanic.BREACH, x=3.0, y=1.0),
        WebNode(id=3, tier=2, ring_index=0, template="ramparts", mechanic=Mechanic.SHRINE, x=0.0, y=7.0),
        WebNode(id=4, tier=3, ring_index=0, template="ramparts", mechanic=Mechanic.RITUAL, x=0.5, y=12.0),
        WebNode(
            id=5, tier=15, ring_index=0, template="ramparts",
            glyph=Pinnacle.ARBITER, x=-1.0, y=99.5,
        ),
        WebNode(
            id=6, tier=15, ring_index=1, template="ramparts",
            pinnacle=Pinnacle.MONOLITH, glyph=Pinnacle.MONOLITH, x=1.0, y=99.5,
        ),
    )
    edges = (
        WebEdge(0, 1), WebEdge(0, 2), WebEdge(1, 3), WebEdge(2, 3),
        WebEdge(3, 4), WebEdge(4, 5), WebEdge(5, 6), WebEdge(4, 6),
    )
    return Web(profile_seed=profile_seed, origin_id=0, nodes=nodes, edges=edges, version=1)


def make_state(profile_id: str = "alice", profile_seed: int = 0x1234_5678_9ABC_DEF0) -> ProfileState:
    """A profile mid-run: one live instance, a ledger, fragments and a stash."""
    live = Sigil(id="sig-live", tier=5, seed=0x0000_00AB_CDEF_0123)
    return ProfileState(
        profile_id=profile_id,
        web=make_web(profile_seed),
        states={
            0: NodeState.CLEARED,
            1: NodeState.ACTIVE,
            2: NodeState.FAILED,
            3: NodeState.LOCKED,
            4: NodeState.LOCKED,
            5: NodeState.LOCKED,
            6: NodeState.LOCKED,
        },
        stash={
            "sig-a": Sigil(id="sig-a", tier=1, seed=0x11),
            "sig-b": Sigil(id="sig-b", tier=15, seed=0x22),
            "sig-c": Sigil(id="sig-c", tier=7, seed=0x33),
        },
        passive_points=3,
        fragments={Pinnacle.ARBITER: 2, Pinnacle.MONOLITH: 0},
        unlocked_pinnacles=frozenset({Pinnacle.ARBITER}),
        instance=Instance(
            node_id=1, sigil=live, map_seed=live.seed, has_boss=False,
            elite_total=10, elite_killed=4, opened_tick=77,
        ),
        history=[
            LedgerEntry(1, 1, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE, None, 0),
            LedgerEntry(2, 2, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE, None, 0),
            LedgerEntry(3, 2, Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "sig-old", 10),
            LedgerEntry(4, 2, Event.DIED, NodeState.ACTIVE, NodeState.FAILED, "sig-old", 50),
            LedgerEntry(5, 1, Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "sig-live", 77),
        ],
    )


def advance(state: ProfileState) -> ProfileState:
    """A later version of ``state``: the boss died, a point was earned."""
    later = copy.deepcopy(state)
    later.states[1] = NodeState.CLEARED
    later.states[3] = NodeState.REACHABLE
    later.passive_points += 1
    later.instance = None
    later.stash.pop("sig-c")
    later.fragments[Pinnacle.MONOLITH] = 1
    later.history.append(
        LedgerEntry(6, 1, Event.BOSS_KILLED, NodeState.ACTIVE, NodeState.CLEARED, "sig-live", 90)
    )
    later.history.append(
        LedgerEntry(7, 3, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE, None, 90)
    )
    return later


StoreFactory = Callable[[], DescentStore]


@pytest.fixture(params=["memory", "sqlite"])
def backend(request: pytest.FixtureRequest, tmp_path: Path) -> StoreFactory:
    """A factory that returns a store over the same underlying data each call.

    For SQLite that means a fresh connection to the same file, which is how
    "saving twice then loading gives the latest" is checked across
    connections and not only within one.
    """
    opened: List[SqliteStore] = []
    if request.param == "memory":
        shared = MemoryStore()

        def factory() -> DescentStore:
            return shared
    else:
        path = tmp_path / "descent.sqlite"

        def factory() -> DescentStore:
            store = SqliteStore(path)
            opened.append(store)
            return store

    yield factory
    for store in opened:
        store.close()


@pytest.fixture
def sqlite_path(tmp_path: Path) -> Path:
    return tmp_path / "descent.sqlite"


# --------------------------------------------------------------------------
# Round trips
# --------------------------------------------------------------------------


def test_round_trip_full_profile(backend: StoreFactory) -> None:
    state = make_state()
    store = backend()
    store.save(state)
    loaded = backend().load("alice")
    assert loaded is not None
    assert first_difference(state, loaded) is None
    assert round_trip_equal(state, loaded)
    assert_round_trip_equal(state, loaded)
    # The pieces the spec cares about, spelled out.
    assert loaded.instance is not None and loaded.instance.sigil == state.instance.sigil
    assert [e.seq for e in loaded.history] == [1, 2, 3, 4, 5]
    assert loaded.fragments == {Pinnacle.ARBITER: 2, Pinnacle.MONOLITH: 0}
    assert set(loaded.stash) == {"sig-a", "sig-b", "sig-c"}
    assert loaded.unlocked_pinnacles == frozenset({Pinnacle.ARBITER})
    assert [n.id for n in loaded.web.nodes] == [0, 2, 1, 3, 4, 5, 6]


def test_round_trip_profile_without_instance_or_history(backend: StoreFactory) -> None:
    state = make_state("bob")
    state.instance = None
    state.history = []
    state.stash = {}
    state.fragments = {}
    state.unlocked_pinnacles = frozenset()
    backend().save(state)
    loaded = backend().load("bob")
    assert loaded is not None
    assert round_trip_equal(state, loaded), first_difference(state, loaded)
    assert loaded.instance is None and loaded.history == [] and loaded.stash == {}


def test_loaded_state_is_independent_of_the_store(backend: StoreFactory) -> None:
    """Mutating what you saved, or what you loaded, must not reach the store."""
    state = make_state()
    store = backend()
    store.save(state)
    state.passive_points = 999
    state.history.clear()
    loaded = backend().load("alice")
    assert loaded is not None
    assert loaded.passive_points == 3 and len(loaded.history) == 5
    loaded.states[0] = NodeState.FAILED
    again = backend().load("alice")
    assert again is not None and again.states[0] is NodeState.CLEARED


def test_seeds_with_top_bit_set_survive(backend: StoreFactory) -> None:
    """Section 02: seeds are 64-bit unsigned; the sign bit must not flip."""
    state = make_state("carol", profile_seed=MASK64)
    state.stash["sig-top"] = Sigil(id="sig-top", tier=3, seed=TOP_BIT)
    state.stash["sig-a"] = Sigil(id="sig-a", tier=1, seed=TOP_BIT | 1)
    live = Sigil(id="sig-live", tier=5, seed=TOP_BIT | 0xDEAD_BEEF)
    state.instance = Instance(
        node_id=1, sigil=live, map_seed=MASK64 - 1, has_boss=True, elite_total=0
    )
    backend().save(state)
    loaded = backend().load("carol")
    assert loaded is not None
    assert loaded.web.profile_seed == MASK64
    assert loaded.stash["sig-top"].seed == TOP_BIT
    assert loaded.stash["sig-a"].seed == TOP_BIT | 1
    assert loaded.instance is not None
    assert loaded.instance.sigil.seed == TOP_BIT | 0xDEAD_BEEF
    assert loaded.instance.map_seed == MASK64 - 1
    assert all(s >= 0 for s in (loaded.web.profile_seed, loaded.instance.map_seed))
    assert round_trip_equal(state, loaded), first_difference(state, loaded)


def test_save_twice_then_load_gives_latest(backend: StoreFactory) -> None:
    first = make_state()
    second = advance(first)
    backend().save(first)
    backend().save(second)
    loaded = backend().load("alice")
    assert loaded is not None
    assert round_trip_equal(second, loaded), first_difference(second, loaded)
    assert not round_trip_equal(first, loaded)
    assert loaded.instance is None and len(loaded.history) == 7
    assert "sig-c" not in loaded.stash


def test_list_and_delete(backend: StoreFactory) -> None:
    store = backend()
    assert store.list_profiles() == []
    assert store.load("nobody") is None
    for pid in ("zed", "alice", "mia"):
        store.save(make_state(pid))
    assert backend().list_profiles() == ["alice", "mia", "zed"]
    assert backend().delete("mia") is True
    assert backend().delete("mia") is False
    assert backend().delete("never-saved") is False
    assert backend().list_profiles() == ["alice", "zed"]
    assert backend().load("mia") is None
    remaining = backend().load("zed")
    assert remaining is not None and round_trip_equal(make_state("zed"), remaining)


def test_profiles_do_not_bleed_into_each_other(backend: StoreFactory) -> None:
    a = make_state("a", profile_seed=1)
    b = advance(make_state("b", profile_seed=2))
    store = backend()
    store.save(a)
    store.save(b)
    la, lb = backend().load("a"), backend().load("b")
    assert la is not None and lb is not None
    assert round_trip_equal(a, la), first_difference(a, la)
    assert round_trip_equal(b, lb), first_difference(b, lb)


# --------------------------------------------------------------------------
# Validation shared by both backends
# --------------------------------------------------------------------------


def test_validation_rejects_states_that_cannot_round_trip(backend: StoreFactory) -> None:
    store = backend()

    bad = make_state()
    bad.stash["wrong-key"] = Sigil(id="sig-x", tier=1, seed=1)
    with pytest.raises(ValueError, match="stash key"):
        store.save(bad)

    bad = make_state()
    bad.stash["sig-huge"] = Sigil(id="sig-huge", tier=1, seed=1 << 64)
    with pytest.raises(ValueError, match="64-bit"):
        store.save(bad)

    bad = make_state()
    bad.instance = Instance(node_id=1, sigil=Sigil("s", 1, 1), map_seed=-1, has_boss=True, elite_total=1)
    with pytest.raises(ValueError, match="map_seed"):
        store.save(bad)

    bad = make_state()
    bad.history.append(LedgerEntry(5, 0, Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE))
    with pytest.raises(ValueError, match="strictly increase"):
        store.save(bad)

    bad = make_state()
    bad.states[42] = NodeState.LOCKED
    with pytest.raises(ValueError, match="unknown node 42"):
        store.save(bad)

    bad = make_state()
    bad.instance = Instance(node_id=42, sigil=Sigil("s", 1, 1), map_seed=1, has_boss=True, elite_total=1)
    with pytest.raises(ValueError, match="instance.node_id"):
        store.save(bad)

    bad = make_state()
    bad.web = Web(bad.web.profile_seed, 0, bad.web.nodes, bad.web.edges + (WebEdge(0, 99),))
    with pytest.raises(ValueError, match="unknown node 99"):
        store.save(bad)

    bad = make_state()
    bad.web = Web(bad.web.profile_seed, 0, bad.web.nodes + (bad.web.nodes[1],), bad.web.edges)
    with pytest.raises(ValueError, match="duplicate node id"):
        store.save(bad)

    # Nothing above touched the store.
    assert store.list_profiles() == []
    validate_state(make_state())  # and a good state passes


# --------------------------------------------------------------------------
# SQLite specifics
# --------------------------------------------------------------------------


def test_sqlite_pragmas_and_schema_version(sqlite_path: Path) -> None:
    with SqliteStore(sqlite_path) as store:
        conn = store._conn
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert store.schema_version() == SCHEMA_VERSION
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    assert {
        "schema_version", "profiles", "web_nodes", "web_edges", "node_states",
        "stash", "fragments", "instance", "ledger",
    } <= tables


def test_sqlite_migrate_is_idempotent_and_refuses_other_versions(sqlite_path: Path) -> None:
    store = SqliteStore(sqlite_path)
    store.save(make_state())
    store.migrate()
    store.migrate()
    assert store.schema_version() == SCHEMA_VERSION
    store.close()

    reopened = SqliteStore(sqlite_path)  # migrate on open must keep data
    loaded = reopened.load("alice")
    assert loaded is not None and round_trip_equal(make_state(), loaded)
    reopened.close()

    raw = sqlite3.connect(sqlite_path)
    raw.execute("UPDATE schema_version SET version = ? WHERE id = 1", (SCHEMA_VERSION + 1,))
    raw.commit()
    raw.close()
    with pytest.raises(SchemaError):
        SqliteStore(sqlite_path)


def test_sqlite_stores_seeds_as_hex_text(sqlite_path: Path) -> None:
    """The high bit would flip in a signed INTEGER column; TEXT keeps it."""
    state = make_state(profile_seed=MASK64)
    state.stash["sig-top"] = Sigil(id="sig-top", tier=2, seed=TOP_BIT)
    with SqliteStore(sqlite_path) as store:
        store.save(state)
        conn = store._conn
        seed, kind = conn.execute("SELECT profile_seed, typeof(profile_seed) FROM profiles").fetchone()
        assert (seed, kind) == ("0xFFFFFFFFFFFFFFFF", "text")
        seed, kind = conn.execute(
            "SELECT seed, typeof(seed) FROM stash WHERE sigil_id = 'sig-top'"
        ).fetchone()
        assert (seed, kind) == ("0x8000000000000000", "text")
        for column in ("sigil_seed", "map_seed"):
            kind = conn.execute(f"SELECT typeof({column}) FROM instance").fetchone()[0]
            assert kind == "text"


def test_sqlite_delete_cascades_to_every_child_table(sqlite_path: Path) -> None:
    children = ("web_nodes", "web_edges", "node_states", "stash", "fragments", "instance", "ledger")
    with SqliteStore(sqlite_path) as store:
        store.save(make_state("alice"))
        store.save(make_state("bob"))
        conn = store._conn
        for table in children:
            assert conn.execute(f"SELECT count(*) FROM {table} WHERE profile_id = 'alice'").fetchone()[0] > 0
        assert store.delete("alice") is True
        for table in children:
            assert conn.execute(f"SELECT count(*) FROM {table} WHERE profile_id = 'alice'").fetchone()[0] == 0
            assert conn.execute(f"SELECT count(*) FROM {table} WHERE profile_id = 'bob'").fetchone()[0] > 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_sqlite_save_replaces_rows_without_leaving_orphans(sqlite_path: Path) -> None:
    with SqliteStore(sqlite_path) as store:
        store.save(make_state())
        store.save(advance(make_state()))
        conn = store._conn
        assert conn.execute("SELECT count(*) FROM profiles").fetchone()[0] == 1
        assert conn.execute("SELECT count(*) FROM ledger").fetchone()[0] == 7
        assert conn.execute("SELECT count(*) FROM stash").fetchone()[0] == 2
        assert conn.execute("SELECT count(*) FROM instance").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM web_nodes").fetchone()[0] == 7
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_sqlite_interrupted_save_leaves_previous_profile_intact(
    sqlite_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-save must never leave a half-written profile.

    The ledger insert is the last step of ``save``, so by the time it raises
    the old rows are gone and every other table already holds the new
    profile inside the open transaction.  The rollback has to undo all of
    it, and the file on disk (read through a second connection) has to show
    the old profile, not an empty or mixed one.
    """
    first = make_state()
    second = advance(first)
    store = SqliteStore(sqlite_path)
    store.save(first)

    def explode(self: SqliteStore, cur: sqlite3.Cursor, state: ProfileState) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(SqliteStore, "_insert_ledger", explode)
    with pytest.raises(RuntimeError, match="disk on fire"):
        store.save(second)
    assert not store._conn.in_transaction

    same_connection = store.load("alice")
    assert same_connection is not None
    assert round_trip_equal(first, same_connection), first_difference(first, same_connection)

    with SqliteStore(sqlite_path) as fresh:
        on_disk = fresh.load("alice")
        assert on_disk is not None
        assert round_trip_equal(first, on_disk), first_difference(first, on_disk)
        assert fresh.list_profiles() == ["alice"]
        assert fresh._conn.execute("SELECT count(*) FROM ledger").fetchone()[0] == 5
        assert fresh._conn.execute("SELECT count(*) FROM instance").fetchone()[0] == 1

    # The store is still usable once the fault is gone.
    monkeypatch.undo()
    store.save(second)
    recovered = store.load("alice")
    assert recovered is not None and round_trip_equal(second, recovered)
    store.close()


def test_sqlite_interrupted_first_save_leaves_no_profile(
    sqlite_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure in the middle of a brand-new profile's save leaves nothing."""
    store = SqliteStore(sqlite_path)

    def explode(self: SqliteStore, cur: sqlite3.Cursor, state: ProfileState) -> None:
        raise sqlite3.OperationalError("simulated I/O error")

    monkeypatch.setattr(SqliteStore, "_insert_stash", explode)
    with pytest.raises(sqlite3.OperationalError):
        store.save(make_state())
    assert store.list_profiles() == []
    assert store.load("alice") is None
    assert store._conn.execute("SELECT count(*) FROM web_nodes").fetchone()[0] == 0
    store.close()


def test_sqlite_constraint_violation_rolls_back_too(sqlite_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An IntegrityError from the database itself is handled like any other."""
    first = make_state()
    store = SqliteStore(sqlite_path)
    store.save(first)

    original = SqliteStore._insert_states

    def duplicate_row(self: SqliteStore, cur: sqlite3.Cursor, state: ProfileState) -> None:
        original(self, cur, state)
        cur.execute(
            "INSERT INTO node_states (profile_id, node_id, state) VALUES (?, ?, ?)",
            (state.profile_id, 0, NodeState.LOCKED.value),
        )

    monkeypatch.setattr(SqliteStore, "_insert_states", duplicate_row)
    with pytest.raises(sqlite3.IntegrityError):
        store.save(advance(first))
    loaded = store.load("alice")
    assert loaded is not None and round_trip_equal(first, loaded)
    store.close()


def test_sqlite_in_memory_database_works(tmp_path: Path) -> None:
    with SqliteStore(":memory:") as store:
        store.save(make_state())
        loaded = store.load("alice")
        assert loaded is not None and round_trip_equal(make_state(), loaded)


# --------------------------------------------------------------------------
# round_trip_equal and first_difference
# --------------------------------------------------------------------------


def test_round_trip_equal_on_deep_copy() -> None:
    state = make_state()
    assert round_trip_equal(state, copy.deepcopy(state))
    assert first_difference(state, copy.deepcopy(state)) is None


@pytest.mark.parametrize(
    "mutate, expected",
    [
        (lambda s: setattr(s, "passive_points", 4), "state.passive_points: 3 != 4"),
        (lambda s: s.states.__setitem__(2, NodeState.LOCKED), "state.states[2]"),
        (lambda s: s.history.pop(), "state.history: length 5 != 4"),
        (lambda s: s.history.__setitem__(4, LedgerEntry(5, 1, Event.OPEN, NodeState.FAILED, NodeState.ACTIVE, "sig-live", 77)),
         "state.history[4].before"),
        (lambda s: setattr(s.instance, "elite_killed", 5), "state.instance.elite_killed: 4 != 5"),
        (lambda s: setattr(s, "instance", None), "state.instance"),
        (lambda s: s.stash.__setitem__("sig-a", Sigil("sig-a", 1, 0x12)), "state.stash['sig-a'].seed"),
        (lambda s: s.stash.pop("sig-b"), "state.stash: key 'sig-b' is missing from the second"),
        (lambda s: s.fragments.__setitem__(Pinnacle.MONOLITH, 3), "state.fragments[<Pinnacle.MONOLITH: 'monolith'>]"),
        (lambda s: setattr(s, "unlocked_pinnacles", frozenset()), "state.unlocked_pinnacles"),
        (lambda s: setattr(s, "web", Web(s.web.profile_seed ^ 1, 0, s.web.nodes, s.web.edges)),
         "state.web.profile_seed"),
        (lambda s: setattr(s, "web", Web(s.web.profile_seed, 0, s.web.nodes[::-1], s.web.edges)),
         "state.web.nodes[0].id: 0 != 6"),
        (lambda s: setattr(s, "web", Web(s.web.profile_seed, 0, s.web.nodes, s.web.edges[:-1])),
         "state.web.edges: length 8 != 7"),
        (lambda s: setattr(s, "web", Web(
            s.web.profile_seed, 0,
            s.web.nodes[:5] + (WebNode(5, 15, 0, "ramparts", glyph=Pinnacle.MONOLITH, x=-1.0, y=99.5),) + s.web.nodes[6:],
            s.web.edges)),
         "state.web.nodes[5].glyph"),
    ],
)
def test_first_difference_names_the_field(mutate, expected: str) -> None:
    a = make_state()
    b = copy.deepcopy(a)
    mutate(b)
    found = first_difference(a, b)
    assert found is not None and found.startswith(expected), found
    assert not round_trip_equal(a, b)
    with pytest.raises(AssertionError, match="states differ at"):
        assert_round_trip_equal(a, b)


def test_first_difference_treats_list_and_tuple_alike_but_not_bool_and_int() -> None:
    a = make_state()
    b = copy.deepcopy(a)
    b.history = tuple(b.history)  # type: ignore[assignment]
    assert first_difference(a, b) is None
    b = copy.deepcopy(a)
    b.instance.has_boss = 0  # type: ignore[assignment]
    assert first_difference(a, b) == "state.instance.has_boss: False != 0"


# --------------------------------------------------------------------------
# Hardening after adversarial review: compare-and-swap saves, strict reads,
# a COMMIT that fails, probe facts on the ledger
# --------------------------------------------------------------------------

from lucifer_descent.store import StaleState, revision_of  # noqa: E402


def test_save_is_compare_and_swap(backend: StoreFactory) -> None:
    """Two copies of one profile: the second to save on a stale copy is refused."""
    store = backend()
    fresh = make_state()
    assert revision_of(fresh) == 0
    store.save(fresh)
    assert revision_of(fresh, store) == 1
    a = backend().load("alice")
    b = backend().load("alice")
    assert revision_of(a, store) == revision_of(b, store) == 1
    a.passive_points += 1
    backend().save(a)
    assert revision_of(a, store) == 2
    b.stash["sig-z"] = Sigil("sig-z", 1, 1)
    with pytest.raises(StaleState, match="revision 2"):
        backend().save(b)
    loaded = backend().load("alice")
    assert loaded.passive_points == 4 and "sig-z" not in loaded.stash, "the stale save changed nothing"
    # Reload, redo, and it goes through.
    b = backend().load("alice")
    b.stash["sig-z"] = Sigil("sig-z", 1, 1)
    backend().save(b)
    assert "sig-z" in backend().load("alice").stash
    # A never-loaded state replaces whatever is there, and keeps the count going.
    replacement = make_state()
    backend().save(replacement)
    assert revision_of(replacement, store) == 4
    with pytest.raises(StaleState):
        backend().save(b)  # b was at 3
    # After a delete the old copies are stale too.
    assert backend().delete("alice")
    with pytest.raises(StaleState):
        backend().save(replacement)


def test_revision_stamp_is_not_profile_content_and_is_per_store(backend: StoreFactory, tmp_path: Path) -> None:
    state = make_state()
    backend().save(state)
    loaded = backend().load("alice")
    assert round_trip_equal(state, loaded) and round_trip_equal(make_state(), loaded)
    assert first_difference(state, copy.deepcopy(loaded)) is None
    # A state loaded from one file saves into another file as a plain copy.
    other = SqliteStore(tmp_path / "other.sqlite")
    other.save(loaded)
    assert revision_of(loaded, other) == 1
    assert round_trip_equal(other.load("alice"), state)
    other.close()


def test_sqlite_ledger_carries_probe_facts(sqlite_path: Path) -> None:
    state = make_state()
    state.history[4] = LedgerEntry(5, 1, Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "sig-live", 77, False, 10)
    with SqliteStore(sqlite_path) as store:
        store.save(state)
        rows = store._conn.execute("SELECT seq, has_boss, elite_total FROM ledger ORDER BY seq").fetchall()
        assert rows == [(1, None, None), (2, None, None), (3, None, None), (4, None, None), (5, 0, 10)]
        loaded = store.load("alice")
    assert loaded is not None and round_trip_equal(state, loaded), first_difference(state, loaded)
    assert loaded.history[4].has_boss is False and loaded.history[4].elite_total == 10


@pytest.mark.parametrize(
    "table, column, value",
    [
        ("stash", "seed", -1),
        ("stash", "seed", "0x1FFFFFFFFFFFFFFFF"),
        ("stash", "seed", "18446744073709551615"),
        ("stash", "seed", "0xffffffffffffffff"),
        ("stash", "seed", 5),
        ("instance", "map_seed", -2),
        ("instance", "sigil_seed", "12"),
        ("profiles", "profile_seed", -12345),
        ("stash", "tier", 1.5),
        ("instance", "elite_total", "seven"),
        ("ledger", "tick", 2.5),
        ("ledger", "has_boss", 7),
    ],
)
def test_sqlite_reads_are_strict(sqlite_path: Path, table: str, column: str, value: object) -> None:
    """A seed column is read only in the exact form it is written; integers must be integers."""
    with SqliteStore(sqlite_path) as store:
        store.save(make_state())
    raw = sqlite3.connect(sqlite_path)
    raw.execute(f"UPDATE {table} SET {column} = ?", (value,))
    raw.commit()
    raw.close()
    with SqliteStore(sqlite_path) as store:
        with pytest.raises(ValueError):
            store.load("alice")


def test_sqlite_commit_failure_rolls_back_and_frees_the_connection(sqlite_path: Path) -> None:
    first = make_state()
    second = advance(first)
    store = SqliteStore(sqlite_path)
    store.save(first)
    real = store._conn

    class Conn:
        def execute(self, sql, *args):
            if sql == "COMMIT":
                raise sqlite3.OperationalError("simulated: commit failed")
            return real.execute(sql, *args)

        def __getattr__(self, name):
            return getattr(real, name)

    store._conn = Conn()  # type: ignore[assignment]
    with pytest.raises(sqlite3.OperationalError, match="commit failed"):
        store.save(second)
    store._conn = real
    assert not real.in_transaction
    loaded = store.load("alice")
    assert loaded is not None and round_trip_equal(first, loaded)
    store.save(second)  # the connection is usable again
    assert round_trip_equal(second, store.load("alice"))
    store.close()


def test_validation_rejects_non_integer_fields(backend: StoreFactory) -> None:
    store = backend()
    bad = make_state()
    bad.stash["sig-f"] = Sigil(id="sig-f", tier=1.5, seed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tier must be an int"):
        store.save(bad)
    bad = make_state()
    bad.instance.elite_killed = True  # type: ignore[assignment]
    with pytest.raises(ValueError, match="elite_killed"):
        store.save(bad)
    bad = make_state()
    bad.history[0] = LedgerEntry(1, 1, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE, None, 0.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tick"):
        store.save(bad)
    assert store.list_profiles() == []


def test_schema_version_is_two_and_version_one_files_are_refused(sqlite_path: Path) -> None:
    assert SCHEMA_VERSION == 2
    raw = sqlite3.connect(sqlite_path)
    raw.execute("CREATE TABLE schema_version (id INTEGER PRIMARY KEY CHECK (id = 1), version INTEGER NOT NULL)")
    raw.execute("INSERT INTO schema_version (id, version) VALUES (1, 1)")
    raw.commit()
    raw.close()
    with pytest.raises(SchemaError, match="version 1"):
        SqliteStore(sqlite_path)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
