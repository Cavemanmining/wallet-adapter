"""Tests for the greybox tile database and room library.

Spec: docs/WORLD_BIBLE.md stages 3 and 4.

The load-bearing claim here is coverage.  Stage 4 must never be told "no tile
fits", and stage 3 must never be told "no room fits", so most of this file is
exhaustive enumeration rather than spot checks.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

# Runnable as `pytest tests/test_tiles.py` or `python3 tests/test_tiles.py`
# from anywhere, without depending on how the package root reaches sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_gen.contracts import (
    ALL_SLOTS,
    E,
    N,
    NO_SLOTS,
    OPPOSITE,
    S,
    SIDES,
    W,
    EdgeSig,
    Placement,
    SideSpec,
    Tile,
    TileClass,
    neighbour,
    sides_compatible,
)
from lucifer_gen.rooms import NoFittingRoom, RoomLibrary, prove_side_coverage
from lucifer_gen.seed import SeedFields
from lucifer_gen.tiles import (
    COVERAGE_ALPHABET,
    DUNGEON_ALPHABET,
    FULL_ALPHABET,
    HERO_PERIOD,
    HeroBudget,
    NoFittingTile,
    TileDatabase,
    facing,
    open_side,
    prove_class_coverage,
    prove_coverage,
    side_satisfies,
    wall_side,
)

CLASSES = (TileClass.DUNGEON, TileClass.OUTDOOR, TileClass.BOTH)
SIGS = tuple(EdgeSig)


@pytest.fixture(scope="module")
def db() -> TileDatabase:
    return TileDatabase.load()


@pytest.fixture(scope="module")
def rooms() -> RoomLibrary:
    return RoomLibrary.load()


def stream(label: str = "tile-test", seed: int = 0x0F1E2D3C4B5A6978):
    return SeedFields.parse(seed).stream(label)


# --------------------------------------------------------------------------
# The database itself
# --------------------------------------------------------------------------


def test_database_loads_and_is_identified(db):
    assert db.version == "greybox@1"
    assert db.ref == db.version
    assert len(db) >= 20
    # blake2b always: a digest two hosts compute differently cannot spot an
    # edit, which is the only thing content_digest is for.
    assert db.content_digest().startswith("blake2b:")


def test_every_tile_id_survives_packing(db):
    """Stage 4 packs a tile into one byte; ids must respect that."""
    for tile in db.tiles:
        packed = Placement(tile.id, 3, True).packed()
        assert Placement.unpack(*packed) == Placement(tile.id, 3, True)


def test_filler_is_impassable_and_carries_its_collision_data(db):
    """Stage 4: untouched cells get filler with a navmesh exclusion."""
    filler = db.filler
    assert filler.walkable is False
    assert all(side.sig is EdgeSig.WALL for side in filler.sides)
    meta = db.meta(filler.id)
    assert meta["navmesh"] == "exclude"
    assert meta["collision_margin_m"] == 1.0
    assert db.is_walkable(db.filler_placement()) is False


def test_filler_is_never_chosen_while_a_walkable_tile_fits(db):
    """Filler is placed deliberately, not drawn; its weight keeps it out."""
    s = stream("tile-filler")
    for _ in range(500):
        placement = db.find([wall_side()] * 4, TileClass.DUNGEON, s)
        assert placement.tile_id != db.filler_tile_id


def test_dungeon_family_lists_all_sixteen_combinations_explicitly(db):
    """The spec asks for the 16 OPEN/WALL tiles outright, not by rotation."""
    literal = {
        tuple(side.sig for side in tile.sides)
        for tile in db.tiles
        if tile.tile_class is TileClass.DUNGEON and not tile.hero
    }
    expected = set(itertools.product((EdgeSig.OPEN, EdgeSig.WALL), repeat=4))
    assert expected <= literal
    assert len(expected) == 16


def test_open_sides_expose_every_connection_slot(db):
    """Full slots is what makes any slot subset satisfiable."""
    for tile in db.tiles:
        for side in tile.sides:
            if side.sig is EdgeSig.OPEN:
                assert side.slots == ALL_SLOTS, tile.name
            else:
                assert side.slots == NO_SLOTS, tile.name


def test_at_least_two_hero_variants(db):
    heroes = db.heroes()
    assert len(heroes) >= 2
    assert all(tile.hero for tile in heroes)


def test_outdoor_signatures_are_present(db):
    """Stage 3's outdoor class produces cliffs and water."""
    seen = {side.sig for tile in db.tiles for side in tile.sides}
    assert {EdgeSig.CLIFF_UP, EdgeSig.CLIFF_DOWN, EdgeSig.WATER} <= seen


# --------------------------------------------------------------------------
# Coverage: the hard requirement
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tile_class", CLASSES)
def test_full_coverage(db, tile_class):
    """Every request a class can actually receive has a fitting tile.

    For OUTDOOR and BOTH that is the whole alphabet: 1296 signature
    combinations (five signatures plus "unconstrained" on four sides) and 4096
    slot combinations.  For DUNGEON it is the 81 OPEN/WALL/unconstrained
    combinations plus the same slot sweep, because ``tileize.surface_of``
    turns a dungeon plan's cliff and water cells into sealed void -- an
    underground map is never asked for an escarpment.

    That distinction is load-bearing.  Requiring DUNGEON to answer the full
    alphabet was only satisfiable by marking the outdoor terrain family
    ``"both"``, which put hillside meshes underground; see
    ``test_a_dungeon_map_is_built_only_from_dungeon_meshes``.
    """
    alphabet = COVERAGE_ALPHABET[tile_class]
    proved = prove_class_coverage(db, tile_class, allow_hero=False)
    assert proved == len(alphabet) ** 4 + (ALL_SLOTS + 1) ** 4


def test_dungeon_open_wall_coverage(db):
    """The 16 OPEN/WALL cases the spec calls out by name, plus wildcards."""
    proved = prove_coverage(
        db,
        TileClass.DUNGEON,
        alphabet=DUNGEON_ALPHABET,
        slot_sweep=False,
        allow_hero=False,
    )
    assert proved == len(DUNGEON_ALPHABET) ** 4


@pytest.mark.parametrize("tile_class", CLASSES)
def test_coverage_still_holds_with_heroes_allowed(db, tile_class):
    assert prove_class_coverage(db, tile_class, allow_hero=True) > 0


def test_dungeon_class_is_not_total_over_terrain_signatures(db):
    """The honest limit of the dungeon family, asserted rather than papered over.

    No dungeon tile carries a cliff or a water side, and none needs to.  This
    test exists so that "make DUNGEON answer CLIFF_UP" is a deliberate data
    decision rather than something a future edit can achieve by accident --
    the only way to satisfy it with the shipped data is to hand dungeon maps
    outdoor geometry again.
    """
    with pytest.raises(NoFittingTile):
        prove_coverage(db, TileClass.DUNGEON, alphabet=FULL_ALPHABET)


def test_tile_meshes_stay_inside_their_class(db):
    """A tile's mesh family and its class must agree.

    ``greybox/terrain/*`` is open-air geometry.  Any tile carrying it must be
    OUTDOOR, or a dungeon request can draw it -- which is exactly what
    happened while the 122 terrain tiles were labelled ``"both"``.
    """
    for tile in db.tiles:
        if tile.mesh.startswith("greybox/terrain/"):
            assert tile.tile_class is TileClass.OUTDOOR, tile.name
        elif tile.mesh.startswith("greybox/dungeon/"):
            assert tile.tile_class is TileClass.DUNGEON, tile.name


def test_a_dungeon_map_is_built_only_from_dungeon_meshes(db, rooms):
    """No underground map may contain an open-air mesh.

    The gate cannot see this: ``validate_map`` judges cells by kind and
    signature and the renderer colours by ``CellKind``, so a crypt built out
    of hillside passes every check while looking wrong the moment it is
    greyboxed.  At 12-to-2 weights with one terrain tile per request the leak
    was routine, not rare -- about one matched cell in seven, and both hero
    cells of ``crypt`` seed 0 were "open ground around a leaning monolith".
    """
    from lucifer_gen.pipeline import generate, resolve_template

    template = resolve_template("crypt")
    assert template.tile_class is TileClass.DUNGEON
    allowed = ("greybox/dungeon/", "greybox/filler")
    for seed in (0, 1, 0xDEADBEEF, 0xFFFFFFFFFFFFFFFF):
        gmap = generate(template, db, rooms, seed)
        for row in gmap.tiles.cells:
            for placement in row:
                mesh = db.by_id(placement.tile_id).mesh
                assert mesh.startswith(allowed), (seed, mesh)


def test_unsatisfiable_request_is_reported_clearly(db):
    """A request no data could serve must raise, not return nonsense."""
    # The shipped database is total by construction, so to see the failure
    # path at all we have to starve it down to the one tile that can never
    # present an open side.
    impossible = SideSpec(EdgeSig.OPEN, ALL_SLOTS)
    tiny = TileDatabase(
        [t for t in db.tiles if t.id == db.filler_tile_id],
        filler_tile_id=db.filler_tile_id,
    )
    with pytest.raises(NoFittingTile) as excinfo:
        tiny.find([impossible, None, None, None], TileClass.DUNGEON, stream())
    assert "N=OPEN:111" in str(excinfo.value)


# --------------------------------------------------------------------------
# Matching correctness
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tile_class", CLASSES)
def test_every_returned_placement_actually_fits(db, tile_class):
    """Re-derive the chosen tile's sides and compare against the request."""
    s = stream(f"tile-fit-{tile_class.value}")
    alphabet = COVERAGE_ALPHABET[tile_class]
    checked = 0
    for combo in itertools.product(alphabet, repeat=4):
        placement = db.find(combo, tile_class, s, allow_hero=True)
        sides = db.sides_of(placement)
        for side, want in zip(SIDES, combo):
            if want is None:
                continue
            assert side_satisfies(sides[side], want), (
                db.by_id(placement.tile_id).name,
                placement,
                side,
                want,
            )
        checked += 1
    assert checked == len(alphabet) ** 4


def test_class_filter_is_respected(db):
    s = stream("tile-class")
    for tile_class in (TileClass.DUNGEON, TileClass.OUTDOOR):
        for _ in range(200):
            placement = db.find([None] * 4, tile_class, s)
            actual = db.by_id(placement.tile_id).tile_class
            assert actual in (tile_class, TileClass.BOTH)


def test_slot_request_is_a_superset_rule():
    """A tile satisfies a request when it offers at least the slots asked for."""
    full = SideSpec(EdgeSig.OPEN, ALL_SLOTS)
    middle = SideSpec(EdgeSig.OPEN, 0b010)
    assert side_satisfies(full, middle)
    assert not side_satisfies(middle, full)
    assert side_satisfies(middle, SideSpec(EdgeSig.OPEN, NO_SLOTS))
    assert not side_satisfies(full, wall_side())


@pytest.mark.parametrize("sig", SIGS)
def test_facing_composes_with_the_contract_compatibility_rule(db, sig):
    """Two tiles chosen through ``facing`` really may sit side by side.

    Pick a tile whose east side is ``sig``, ask ``facing`` what its eastern
    neighbour's west side must be, pick that tile, then check the pair against
    ``contracts.sides_compatible`` -- the rule the rest of the pipeline trusts.
    """
    s = stream(f"tile-facing-{sig.name}")
    left_request = [None, SideSpec(sig, ALL_SLOTS if sig is EdgeSig.OPEN else NO_SLOTS),
                    None, None]
    left = db.find(left_request, TileClass.BOTH, s)
    left_sides = db.sides_of(left)
    right_request = [None, None, None, facing(left_sides[E])]
    right = db.find(right_request, TileClass.BOTH, s)
    right_sides = db.sides_of(right)
    assert sides_compatible(left_sides[E], right_sides[W])


def test_transform_search_uses_rotation(db):
    """One orbit representative has to cover all four turns of a pattern.

    The terrain family stores one tile per rotation/flip orbit, so this is the
    mechanism that turns 120 tiles into 625 reachable side combinations. The
    pattern below is fully asymmetric, so exactly one tile can serve it and
    only the rotation can differ.
    """
    pattern = [
        open_side(),
        wall_side(),
        SideSpec(EdgeSig.WATER),
        SideSpec(EdgeSig.CLIFF_UP),
    ]
    seen = {}
    for rot in range(4):
        rotated = [pattern[(i - rot) % 4] for i in SIDES]
        placement = db.find(rotated, TileClass.OUTDOOR, stream(f"tile-rot{rot}"))
        assert db.placement_fits(placement, rotated)
        seen[placement.rot] = placement.tile_id
    assert sorted(seen) == [0, 1, 2, 3], "rotation is doing no work"
    assert len(set(seen.values())) == 1, "one tile should cover the whole orbit"


def test_transform_search_uses_flip(db):
    """The mirror of that pattern is not any rotation of it, so flip must run."""
    pattern = [
        open_side(),
        wall_side(),
        SideSpec(EdgeSig.WATER),
        SideSpec(EdgeSig.CLIFF_UP),
    ]
    mirrored = [pattern[0], pattern[3], pattern[2], pattern[1]]
    rotations = {tuple(pattern[(i - r) % 4] for i in SIDES) for r in range(4)}
    assert tuple(mirrored) not in rotations, "the fixture stopped being asymmetric"
    placement = db.find(mirrored, TileClass.OUTDOOR, stream("tile-flip"))
    assert placement.flip is True
    assert db.placement_fits(placement, mirrored)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_find_is_deterministic_for_a_fixed_stream(db):
    request = [open_side(), wall_side(), open_side(), wall_side()]
    a = stream("tile-det")
    b = stream("tile-det")
    seq_a = [db.find(request, TileClass.DUNGEON, a, allow_hero=True) for _ in range(50)]
    seq_b = [db.find(request, TileClass.DUNGEON, b, allow_hero=True) for _ in range(50)]
    assert seq_a == seq_b
    assert len({(p.tile_id, p.rot, p.flip) for p in seq_a}) > 1, "stream is stuck"


def test_different_seeds_diverge(db):
    request = [open_side()] * 4
    a = stream("tile-div", seed=0x1111111111111111)
    b = stream("tile-div", seed=0x2222222222222222)
    seq_a = [db.find(request, TileClass.BOTH, a, allow_hero=True) for _ in range(60)]
    seq_b = [db.find(request, TileClass.BOTH, b, allow_hero=True) for _ in range(60)]
    assert seq_a != seq_b


def test_only_the_tile_field_moves_tile_choices(db):
    """Seed bits 32-63 own tile selection; routing bits must not disturb it."""
    request = [open_side(), None, open_side(), None]
    base = 0xABCDEF0123456789
    changed_routing = base ^ (0xFFFF << 8)  # inside bits 8-31
    a = SeedFields.parse(base).stream("tiles")
    b = SeedFields.parse(changed_routing).stream("tiles")
    assert [db.find(request, TileClass.BOTH, a) for _ in range(40)] == [
        db.find(request, TileClass.BOTH, b) for _ in range(40)
    ]


def test_candidate_order_is_stable(db):
    request = [open_side(), None, None, wall_side()]
    once = db.candidates(request, TileClass.OUTDOOR)
    twice = TileDatabase.load().candidates(request, TileClass.OUTDOOR)
    assert [(o.tile.id, o.rot, o.flip) for o in once] == [
        (o.tile.id, o.rot, o.flip) for o in twice
    ]


# --------------------------------------------------------------------------
# The hero budget
# --------------------------------------------------------------------------


def test_hero_budget_refuses_the_first_nineteen_cells():
    budget = HeroBudget()
    assert budget.period == HERO_PERIOD
    for _ in range(HERO_PERIOD - 1):
        assert budget.allows_hero() is False
        budget.record(False)
    assert budget.allows_hero() is True


def test_hero_budget_holds_at_one_per_twenty_over_a_long_run(db):
    """Drive 2000 cells through find() and watch the rate, not just the total."""
    s = stream("tile-hero-run")
    budget = HeroBudget()
    request = [open_side()] * 4
    hero_indices = []
    for index in range(2000):
        allow = budget.allows_hero()
        placement = db.find(request, TileClass.DUNGEON, s, allow_hero=allow)
        is_hero = db.by_id(placement.tile_id).hero
        assert not (is_hero and not allow), "a hero slipped through a closed budget"
        if is_hero:
            hero_indices.append(index)
        budget.record(is_hero)
        assert budget.within_budget

    assert budget.cells == 2000
    assert budget.heroes * HERO_PERIOD <= budget.cells
    assert budget.heroes > 0, "heroes never appear; the weighting is broken"
    # Spacing, the other reading of "once per 20 cells".
    # Spacing implies the sliding-window reading too: if consecutive heroes
    # are never closer than 20 cells, no 20-cell window holds two.
    gaps = [b - a for a, b in zip(hero_indices, hero_indices[1:])]
    assert all(gap >= HERO_PERIOD for gap in gaps)


def test_hero_budget_survives_an_unbudgeted_hero():
    """Stage 5 overwrites tiles; the running rate must still be honest."""
    budget = HeroBudget(period=5)
    budget.record(True)  # a set piece, not something find() offered
    assert budget.allows_hero() is False
    for _ in range(4):
        budget.record(False)
    assert budget.allows_hero() is False  # 2 * 5 > 5 + 1
    while not budget.allows_hero():
        budget.record(False)
    assert budget.heroes * budget.period <= budget.cells


def test_heroes_are_excluded_when_not_allowed(db):
    s = stream("tile-nohero")
    for _ in range(400):
        placement = db.find([open_side()] * 4, TileClass.BOTH, s, allow_hero=False)
        assert db.by_id(placement.tile_id).hero is False


# --------------------------------------------------------------------------
# Rooms (stage 3)
# --------------------------------------------------------------------------


def test_room_library_loads(rooms):
    assert rooms.version == "greybox_rooms@1"
    assert len(rooms) >= 8
    sizes = {(r.w, r.h) for r in rooms.rooms}
    assert (2, 2) in sizes
    assert (6, 5) in sizes


def test_room_library_serves_every_subset_of_sides(rooms):
    assert prove_side_coverage(rooms) == 15
    # ...and still does when only a 2x2 footprint will fit.
    assert prove_side_coverage(rooms, 2, 2) == 15


def test_every_subset_has_an_exact_match(rooms):
    """A doorway opening onto rock is a bug; the library avoids needing one."""
    for size in (1, 2, 3, 4):
        for subset in itertools.combinations(SIDES, size):
            exact = rooms.candidates(set(subset), exact=True)
            assert exact, subset
            assert all(room.sides == set(subset) for room in exact)


def test_candidates_respects_the_size_budget(rooms):
    for room in rooms.candidates({N}, 3, 3):
        assert room.w <= 3 and room.h <= 3


def test_choose_is_deterministic_and_fits(rooms):
    for size in (1, 2, 3, 4):
        for subset in itertools.combinations(SIDES, size):
            a = rooms.choose(set(subset), stream("place-room"))
            b = rooms.choose(set(subset), stream("place-room"))
            assert a == b
            assert set(subset) <= a.sides


def test_choose_raises_when_the_budget_is_impossible(rooms):
    with pytest.raises(NoFittingRoom):
        rooms.choose({N, E, S, W}, stream("place-room"), w_max=1, h_max=1)


def test_placement_is_centred_clamped_and_never_scaled(rooms):
    grid = 8
    for room in rooms.rooms:
        for cell in itertools.product(range(grid), repeat=2):
            if room.w > grid or room.h > grid:
                continue
            placement = rooms.place(room, "node", cell, grid)
            assert (placement.w, placement.h) == (room.w, room.h), "rooms never scale"
            cells = placement.cells()
            assert len(cells) == room.w * room.h
            for x, y in cells:
                assert 0 <= x < grid and 0 <= y < grid
            # Clamping slides the room towards the node, so the node never
            # ends up outside the room it belongs to.
            assert cell in cells


def test_odd_rooms_centre_exactly_on_the_node(rooms):
    grid = 48
    for room in rooms.rooms:
        if room.w % 2 == 1 and room.h % 2 == 1:
            placement = rooms.place(room, "node", (24, 24), grid)
            assert placement.origin == (24 - room.w // 2, 24 - room.h // 2)


def test_doorways_sit_on_the_footprint_boundary(rooms):
    grid = 48
    for room in rooms.rooms:
        placement = rooms.place(room, "node", (24, 24), grid)
        ox, oy = placement.origin
        footprint = set(placement.cells())
        for side, cell in rooms.doorways(placement):
            assert cell in footprint
            x, y = cell
            if side == N:
                assert y == oy
            elif side == E:
                assert x == ox + room.w - 1
            elif side == S:
                assert y == oy + room.h - 1
            elif side == W:
                assert x == ox


def test_socket_offsets_run_clockwise(rooms):
    """North counts west to east, south counts east to west, and so on."""
    room = rooms.by_id("vault_6x5_nesw")
    origin = (10, 10)
    from lucifer_gen.rooms import DoorwaySocket

    assert room.socket_cell(origin, DoorwaySocket(N, 0)) == (10, 10)
    assert room.socket_cell(origin, DoorwaySocket(N, 5)) == (15, 10)
    assert room.socket_cell(origin, DoorwaySocket(E, 0)) == (15, 10)
    assert room.socket_cell(origin, DoorwaySocket(E, 4)) == (15, 14)
    assert room.socket_cell(origin, DoorwaySocket(S, 0)) == (15, 14)
    assert room.socket_cell(origin, DoorwaySocket(S, 5)) == (10, 14)
    assert room.socket_cell(origin, DoorwaySocket(W, 0)) == (10, 14)
    assert room.socket_cell(origin, DoorwaySocket(W, 4)) == (10, 10)


def test_room_too_large_for_the_grid_is_refused(rooms):
    with pytest.raises(ValueError):
        rooms.place(rooms.by_id("vault_6x5_nesw"), "node", (2, 2), 4)


# --------------------------------------------------------------------------
# The whole thing, the way stage 4 will use it
# --------------------------------------------------------------------------


def test_a_whole_grid_tiles_without_a_bad_seam(db):
    """Tile a grid the way stage 4 will and check every seam in it.

    Requirements are derived from a floor mask, so neighbouring cells agree on
    the shared edge by construction; what this proves is that the database can
    answer all of them and that the answers really do abut, judged by
    ``contracts.sides_compatible`` rather than by this module's own rule.
    """
    size = 24
    floor = {
        (x, y)
        for y in range(size)
        for x in range(size)
        if (x * x + y * 3) % 7 < 3 or (4 <= x <= size - 5 and y in (6, 12, 18))
    }
    s = stream("tiles", seed=0xFEEDFACECAFEBEEF)
    budget = HeroBudget()
    placed = {}

    for y in range(size):
        for x in range(size):
            cell = (x, y)
            if cell not in floor:
                placed[cell] = db.filler_placement()
                budget.record(False)
                continue
            required = [
                open_side() if neighbour(cell, side) in floor else wall_side()
                for side in SIDES
            ]
            placement = db.find(required, TileClass.DUNGEON, s, budget.allows_hero())
            assert db.placement_fits(placement, required)
            placed[cell] = placement
            budget.record(db.by_id(placement.tile_id).hero)

    for (x, y), placement in placed.items():
        mine = db.sides_of(placement)
        for side in (E, S):  # each seam is visited once
            other = placed.get(neighbour((x, y), side))
            if other is None:
                continue
            theirs = db.sides_of(other)
            assert sides_compatible(mine[side], theirs[OPPOSITE[side]]), (x, y, side)

    assert budget.cells == size * size
    assert budget.within_budget
    assert all(db.is_walkable(placed[c]) for c in floor)
    assert not any(db.is_walkable(placed[c]) for c in placed if c not in floor)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# facing(): a requirement must promise only what it can deliver
# --------------------------------------------------------------------------


def test_facing_refuses_an_open_side_with_no_slots(db):
    """``facing`` used to return a requirement that guaranteed a broken seam.

    An OPEN side offering no connection slot can never be joined: no tile is
    compatible with it.  The naive translation is ``SideSpec(OPEN, 0)``, which
    ``side_satisfies`` reads as the wildcard "open, slots don't matter" -- so
    ``find`` happily returned a placement whose seam then failed
    ``contracts.sides_compatible``.  The two rules must not disagree about
    what ``facing`` promised.
    """
    with pytest.raises(ValueError):
        facing(SideSpec(EdgeSig.OPEN, NO_SLOTS))

    # Every side the real database can present still translates fine, and the
    # promise holds: satisfying the requirement implies a compatible seam.
    s = stream("tile-facing-promise")
    for tile in db.tiles:
        for side_spec in tile.sides:
            request = facing(side_spec)
            placement = db.find([request, None, None, None], TileClass.BOTH, s)
            assert sides_compatible(db.sides_of(placement)[N], side_spec)


def test_a_tile_with_an_unjoinable_open_side_is_refused():
    """The contradiction is rejected at load, not discovered at a seam."""
    bad = Tile(
        id=7,
        name="unjoinable",
        tile_class=TileClass.BOTH,
        sides=(
            SideSpec(EdgeSig.OPEN, NO_SLOTS),
            SideSpec(EdgeSig.WALL),
            SideSpec(EdgeSig.WALL),
            SideSpec(EdgeSig.WALL),
        ),
    )
    filler = Tile(
        id=0,
        name="filler",
        tile_class=TileClass.BOTH,
        sides=tuple(SideSpec(EdgeSig.WALL) for _ in range(4)),
        walkable=False,
    )
    with pytest.raises(ValueError) as excinfo:
        TileDatabase([filler, bad], filler_tile_id=0)
    assert "connection slot" in str(excinfo.value)
