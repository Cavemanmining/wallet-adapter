"""Tests for stage 4, tileization.

Spec: docs/WORLD_BIBLE.md stage 4.

The load-bearing claim is the seam rule: every pair of orthogonally adjacent
placed tiles must satisfy ``contracts.sides_compatible``.  That is a property
of the whole grid rather than of any one call, so most of this file is sweeps
over many seeds and many plan shapes rather than spot checks.

Terrain plans are built here directly instead of by running stage 3, so a
failure points at tileization and nothing else.  The random plans are the
harshest input available: every cell kind next to every other cell kind,
including combinations a real stage 3 would never emit.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable as `pytest tests/test_tileize.py` or `python3 tests/test_tileize.py`
# from anywhere, without depending on how the package root reaches sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_gen.contracts import (
    OPPOSITE,
    SIDES,
    CellKind,
    EdgeSig,
    Placement,
    TerrainPlan,
    TileClass,
    neighbour,
    sides_compatible,
)
from lucifer_gen.seed import SeedFields
from lucifer_gen.tiles import NoFittingTile, TileDatabase
from lucifer_gen.tileize import (
    COLLISION_MARGIN_M,
    SIG_BETWEEN,
    Box,
    SeamError,
    Surface,
    TileizeError,
    assert_seams,
    check_seams,
    debug_check,
    required_sides,
    surface_of,
    tileize,
)

SEEDS = 200
FLOOR_KINDS = (
    CellKind.CORRIDOR,
    CellKind.ROOM,
    CellKind.SET_PIECE,
    CellKind.APPROACH,
)
OUTDOOR_KINDS = FLOOR_KINDS + (CellKind.EMPTY, CellKind.WATER, CellKind.CLIFF)
DUNGEON_KINDS = FLOOR_KINDS + (CellKind.EMPTY,)


@pytest.fixture(scope="module")
def db() -> TileDatabase:
    return TileDatabase.load()


# --------------------------------------------------------------------------
# Plan builders (stage 3 stand-ins)
# --------------------------------------------------------------------------


def noisy_plan(grid: int, seed: int, kinds=OUTDOOR_KINDS) -> TerrainPlan:
    """Every cell drawn independently: maximal adjacency coverage.

    Deterministic, and it exercises pairs a real stage 3 would never produce,
    which is exactly what a seam rule should survive.
    """
    stream = SeedFields.parse(seed).stream("tile-plan-noise")
    plan = TerrainPlan.blank(grid)
    for y in range(grid):
        for x in range(grid):
            plan.set_kind((x, y), stream.choice(kinds))
    return plan


def carved_plan(grid: int, seed: int, *, outdoor: bool = True) -> TerrainPlan:
    """A plan shaped the way stage 3 shapes one: rooms joined by corridors.

    A handful of rooms, corridors walked between them in an L, and for the
    outdoor flavour a water pool and a cliff ledge dropped alongside.
    """
    stream = SeedFields.parse(seed).stream("tile-plan-carve")
    plan = TerrainPlan.blank(grid)
    centres = []
    for _ in range(4):
        cx = stream.randint(3, grid - 4)
        cy = stream.randint(3, grid - 4)
        centres.append((cx, cy))
        w, h = stream.randint(2, 4), stream.randint(2, 4)
        for dy in range(h):
            for dx in range(w):
                plan.set_kind((cx + dx - w // 2, cy + dy - h // 2), CellKind.ROOM)
    for (ax, ay), (bx, by) in zip(centres, centres[1:]):
        for x in range(min(ax, bx), max(ax, bx) + 1):
            if plan.kind((x, ay)) is CellKind.EMPTY:
                plan.set_kind((x, ay), CellKind.CORRIDOR)
        for y in range(min(ay, by), max(ay, by) + 1):
            if plan.kind((bx, y)) is CellKind.EMPTY:
                plan.set_kind((bx, y), CellKind.CORRIDOR)
    if outdoor:
        px, py = stream.randint(0, grid - 3), stream.randint(0, grid - 3)
        for dy in range(3):
            for dx in range(3):
                plan.set_kind((px + dx, py + dy), CellKind.WATER)
        ly = stream.randint(0, grid - 1)
        for x in range(grid):
            if plan.kind((x, ly)) is CellKind.EMPTY:
                plan.set_kind((x, ly), CellKind.CLIFF)
    return plan


def full_floor_plan(grid: int) -> TerrainPlan:
    """Every cell walkable: the densest possible tile-matching workload."""
    plan = TerrainPlan.blank(grid)
    for y in range(grid):
        for x in range(grid):
            plan.set_kind((x, y), CellKind.CORRIDOR)
    return plan


# --------------------------------------------------------------------------
# Walkability and filler
# --------------------------------------------------------------------------


def test_floor_is_walkable_and_untouched_is_filler(db):
    """Spec stage 4: untouched cells get impassable filler, floor does not."""
    plan = carved_plan(24, 0xA11CE)
    grid = tileize(plan, db, TileClass.OUTDOOR, 0x0123456789ABCDEF)

    seen_floor = seen_void = 0
    for y in range(plan.grid):
        for x in range(plan.grid):
            cell = (x, y)
            placement = grid.at(cell)
            assert placement is not None, f"{cell} was left empty"
            if plan.is_floor(cell):
                seen_floor += 1
                assert grid.is_walkable(cell)
                assert db.by_id(placement.tile_id).walkable
                assert placement.tile_id != db.filler_tile_id
            else:
                seen_void += 1
                assert not grid.is_walkable(cell)
    assert seen_floor > 0 and seen_void > 0

    # Untouched (EMPTY) cells specifically must carry the filler tile.
    for y in range(plan.grid):
        for x in range(plan.grid):
            if plan.kind((x, y)) is CellKind.EMPTY:
                assert grid.at((x, y)).tile_id == db.filler_tile_id
                assert not grid.is_walkable((x, y))


def test_filler_records_navmesh_exclusion_and_collision_hull(db):
    """Spec stage 4: filler carries an exclusion and a hull 1 m beyond the mesh."""
    plan = carved_plan(16, 0xBEE5)
    grid = tileize(plan, db, TileClass.OUTDOOR, 0xFEEDFACE)

    recorded = {f.cell for f in grid.filler}
    expected = {
        (x, y)
        for y in range(plan.grid)
        for x in range(plan.grid)
        if surface_of(plan, (x, y), TileClass.OUTDOOR) is Surface.VOID
    }
    assert recorded == expected
    assert len(grid.filler) == len(recorded), "filler records must not repeat"
    # Row-major, so two runs and any consumer see the same order.
    assert [f.cell for f in grid.filler] == sorted(recorded, key=lambda c: (c[1], c[0]))

    for entry in grid.filler:
        x, y = entry.cell
        assert entry.mesh == Box.of_cell(entry.cell, db.cell_m)
        assert entry.navmesh_exclusion is entry.mesh
        assert entry.margin_m == COLLISION_MARGIN_M == 1.0
        assert entry.hull.min_x == pytest.approx(entry.mesh.min_x - 1.0)
        assert entry.hull.min_y == pytest.approx(entry.mesh.min_y - 1.0)
        assert entry.hull.max_x == pytest.approx(entry.mesh.max_x + 1.0)
        assert entry.hull.max_y == pytest.approx(entry.mesh.max_y + 1.0)
        # The hull stands proud of the mesh on every side, never inside it.
        assert entry.hull.min_x < entry.mesh.min_x
        assert entry.hull.max_y > entry.mesh.max_y

    assert len(grid.navmesh_exclusions()) == len(grid.filler)
    assert len(grid.collision_hulls()) == len(grid.filler)
    assert grid.filler_cells() == tuple(f.cell for f in grid.filler)


def test_filler_metadata_comes_from_the_database(db):
    """The 1 m margin is the default; the database may name its own."""
    assert db.meta(db.filler_tile_id).get("navmesh") == "exclude"
    assert float(db.meta(db.filler_tile_id)["collision_margin_m"]) == 1.0


# --------------------------------------------------------------------------
# Seams
# --------------------------------------------------------------------------


def test_requirement_table_is_self_complementary():
    """Seams hold because the requirements are complements by construction."""
    from lucifer_gen.contracts import SIG_COMPLEMENT

    for here in Surface:
        for there in Surface:
            mine = SIG_BETWEEN[(here, there)]
            theirs = SIG_BETWEEN[(there, here)]
            assert SIG_COMPLEMENT[mine] is theirs


@pytest.mark.parametrize("tile_class", [TileClass.OUTDOOR, TileClass.DUNGEON])
def test_requirements_agree_across_every_seam(db, tile_class):
    """What a cell must show east matches what its neighbour must show west."""
    kinds = OUTDOOR_KINDS if tile_class is TileClass.OUTDOOR else DUNGEON_KINDS
    for seed in range(20):
        plan = noisy_plan(10, seed, kinds)
        for y in range(plan.grid):
            for x in range(plan.grid):
                here = (x, y)
                mine = required_sides(plan, here, tile_class)
                for side in SIDES:
                    there = neighbour(here, side)
                    if not plan.inside(there):
                        assert mine[side].sig is EdgeSig.WALL
                        continue
                    theirs = required_sides(plan, there, tile_class)
                    assert sides_compatible(mine[side], theirs[OPPOSITE[side]])


def test_zero_seam_mismatches_over_many_seeds(db):
    """The headline invariant, swept over 200 seeds and both tile classes."""
    mismatches = 0
    grids = 0
    for seed in range(SEEDS):
        for tile_class, kinds in (
            (TileClass.OUTDOOR, OUTDOOR_KINDS),
            (TileClass.DUNGEON, DUNGEON_KINDS),
        ):
            plan = noisy_plan(10, seed, kinds)
            # verify=False so the sweep counts mismatches instead of stopping
            # at the first grid that has any.
            grid = tileize(plan, db, tile_class, seed, verify=False)
            mismatches += len(check_seams(grid, db))
            grids += 1
    assert grids == SEEDS * 2
    assert mismatches == 0


def test_zero_seam_mismatches_on_realistic_plans(db):
    """Same invariant on plans shaped the way stage 3 shapes them."""
    for seed in range(SEEDS):
        plan = carved_plan(20, seed, outdoor=seed % 2 == 0)
        tile_class = TileClass.OUTDOOR if seed % 2 == 0 else TileClass.DUNGEON
        grid = tileize(plan, db, tile_class, seed ^ 0xABCDEF, verify=False)
        assert check_seams(grid, db) == []


def test_border_of_the_map_is_sealed(db):
    """Outside the grid reads as void, so the rim must present walls."""
    plan = full_floor_plan(12)
    grid = tileize(plan, db, TileClass.OUTDOOR, 0x99)
    for y in range(plan.grid):
        for x in range(plan.grid):
            for side in SIDES:
                if plan.inside(neighbour((x, y), side)):
                    continue
                shown = db.sides_of(grid.at((x, y)))[side]
                assert shown.sig is EdgeSig.WALL


def test_seam_checker_catches_a_broken_grid(db):
    """The debug check must be able to fail, or it proves nothing."""
    plan = full_floor_plan(8)
    grid = tileize(plan, db, TileClass.OUTDOOR, 0x1234)
    assert check_seams(grid, db) == []

    # Drop sealed filler into the middle of open ground: its neighbours are
    # all still showing OPEN, so four seams must break.
    grid.put((4, 4), db.filler_placement(), walkable=False)
    bad = check_seams(grid, db)
    assert len(bad) == 4
    with pytest.raises(SeamError):
        assert_seams(grid, db)
    assert any("(4, 4)" in str(m) for m in bad)


def test_verify_flag_runs_the_check(db):
    """tileize verifies by default; the flag only turns the check off."""
    plan = carved_plan(14, 0x55)
    checked = tileize(plan, db, TileClass.OUTDOOR, 0x77, verify=True)
    unchecked = tileize(plan, db, TileClass.OUTDOOR, 0x77, verify=False)
    assert checked.cells == unchecked.cells


# --------------------------------------------------------------------------
# Cliffs and water
# --------------------------------------------------------------------------


def test_cliff_and_water_signatures_face_the_right_way(db):
    """Floor looks down at a cliff; the cliff looks up at the floor."""
    plan = TerrainPlan.blank(5)
    plan.set_kind((1, 2), CellKind.CORRIDOR)
    plan.set_kind((2, 2), CellKind.CLIFF)
    plan.set_kind((3, 2), CellKind.CORRIDOR)
    plan.set_kind((2, 3), CellKind.WATER)

    from lucifer_gen.contracts import E, N, S, W

    floor = required_sides(plan, (1, 2), TileClass.OUTDOOR)
    assert floor[E].sig is EdgeSig.CLIFF_DOWN
    cliff = required_sides(plan, (2, 2), TileClass.OUTDOOR)
    assert cliff[W].sig is EdgeSig.CLIFF_UP
    assert cliff[E].sig is EdgeSig.CLIFF_UP
    assert cliff[S].sig is EdgeSig.WATER
    water = required_sides(plan, (2, 3), TileClass.OUTDOOR)
    assert water[N].sig is EdgeSig.WATER

    grid = tileize(plan, db, TileClass.OUTDOOR, 0xC11FF)
    assert db.sides_of(grid.at((1, 2)))[E].sig is EdgeSig.CLIFF_DOWN
    assert db.sides_of(grid.at((2, 2)))[W].sig is EdgeSig.CLIFF_UP
    assert not grid.is_walkable((2, 2)), "a cliff cell is terrain, not floor"
    assert not grid.is_walkable((2, 3)), "a water cell is terrain, not floor"
    # Terrain cells are still tiled, never replaced by filler.
    assert grid.at((2, 2)).tile_id != db.filler_tile_id
    assert grid.at((2, 3)).tile_id != db.filler_tile_id
    assert (2, 2) not in grid.filler_cells()


def test_dungeon_degrades_cliff_and_water_to_filler(db):
    """A dungeon plan has no escarpments; stage 4 seals them rather than crash."""
    plan = TerrainPlan.blank(5)
    plan.set_kind((2, 2), CellKind.CORRIDOR)
    plan.set_kind((3, 2), CellKind.CLIFF)
    plan.set_kind((2, 3), CellKind.WATER)

    assert surface_of(plan, (3, 2), TileClass.DUNGEON) is Surface.VOID
    assert surface_of(plan, (3, 2), TileClass.OUTDOOR) is Surface.CLIFF

    grid = tileize(plan, db, TileClass.DUNGEON, 0xD00D)
    assert grid.at((3, 2)).tile_id == db.filler_tile_id
    assert grid.at((2, 3)).tile_id == db.filler_tile_id
    assert check_seams(grid, db) == []


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def test_same_seed_gives_an_identical_grid(db):
    plan = carved_plan(24, 0x2468)
    a = tileize(plan, db, TileClass.OUTDOOR, 0x0FEDCBA987654321)
    b = tileize(plan, db, TileClass.OUTDOOR, 0x0FEDCBA987654321)
    assert a.cells == b.cells
    assert a.walkable == b.walkable
    assert a.filler == b.filler
    assert a.hero_cells == b.hero_cells
    assert a.packed_cells() == b.packed_cells()


def test_seed_fields_and_int_seeds_agree(db):
    plan = carved_plan(12, 0x31337)
    seed = 0xCAFEBABEDEADBEEF
    a = tileize(plan, db, TileClass.OUTDOOR, seed)
    b = tileize(plan, db, TileClass.OUTDOOR, SeedFields.parse(seed))
    assert a.packed_cells() == b.packed_cells()


def test_only_the_tile_field_moves_the_tiling(db):
    """Seed bits 0-31 belong to earlier stages; they must not perturb stage 4."""
    plan = carved_plan(16, 0x4242)
    base = 0x0000000100000000
    same_tiles = tileize(plan, db, TileClass.OUTDOOR, base)
    other_low_bits = tileize(plan, db, TileClass.OUTDOOR, base | 0x00000000FFFFFFFF)
    assert same_tiles.packed_cells() == other_low_bits.packed_cells()

    other_tile_field = tileize(plan, db, TileClass.OUTDOOR, 0x0000000200000000)
    assert other_tile_field.packed_cells() != same_tiles.packed_cells()


def test_different_seeds_generally_differ(db):
    plan = full_floor_plan(16)
    packed = {
        tileize(plan, db, TileClass.OUTDOOR, s << 32).packed_cells() for s in range(8)
    }
    assert len(packed) > 1


# --------------------------------------------------------------------------
# Hero budget
# --------------------------------------------------------------------------


def test_hero_variants_stay_within_one_in_twenty(db):
    """Spec stage 4: a hero variant appears at most once per 20 cells."""
    period = db.hero_period
    assert period == 20
    plan = full_floor_plan(48)
    for seed in range(24):
        grid = tileize(plan, db, TileClass.OUTDOOR, seed << 32)
        heroes = grid.hero_count
        # Measured against the cells the matcher actually chose, which is the
        # stricter of the two readings: filler is never a hero variant, so it
        # must not be allowed to buy budget for one.
        assert heroes * period <= grid.matched_cells, (
            f"seed {seed}: {heroes} heroes in {grid.matched_cells} matched cells"
        )
        # And against the whole grid, which is the looser reading.
        assert heroes * period <= plan.grid * plan.grid
        placed = [
            cell
            for cell in grid.hero_cells
            if db.by_id(grid.at(cell).tile_id).hero
        ]
        assert placed == list(grid.hero_cells)


def test_hero_variants_do_appear(db):
    """The budget must ration heroes, not ban them."""
    plan = full_floor_plan(48)
    total = sum(tileize(plan, db, TileClass.DUNGEON, s << 32).hero_count for s in range(4))
    assert total > 0


def test_hero_budget_holds_on_sparse_plans(db):
    """A map with few floor cells should get few heroes, or none."""
    for seed in range(40):
        plan = carved_plan(16, seed)
        grid = tileize(plan, db, TileClass.OUTDOOR, seed << 32)
        assert grid.hero_count * db.hero_period <= grid.matched_cells


# --------------------------------------------------------------------------
# The matcher is never stuck
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tile_class", [TileClass.OUTDOOR, TileClass.DUNGEON, TileClass.BOTH]
)
def test_matcher_never_fails_to_find_a_tile(db, tile_class):
    """Stage 4 must never be told "no tile fits", on any plan it can be handed."""
    kinds = DUNGEON_KINDS if tile_class is TileClass.DUNGEON else OUTDOOR_KINDS
    for seed in range(40):
        plan = noisy_plan(12, seed, kinds)
        try:
            tileize(plan, db, tile_class, seed, verify=False)
        except NoFittingTile as exc:  # pragma: no cover - the failure path
            pytest.fail(f"{tile_class.value} seed {seed}: {exc}")


def test_full_invariant_sweep(db):
    """Every stage 4 invariant, re-derived from the output, over many plans."""
    for seed in range(60):
        for tile_class, kinds in (
            (TileClass.OUTDOOR, OUTDOOR_KINDS),
            (TileClass.DUNGEON, DUNGEON_KINDS),
        ):
            plan = noisy_plan(10, seed, kinds)
            grid = tileize(plan, db, tile_class, seed << 32)
            assert debug_check(grid, plan, db) == []


# --------------------------------------------------------------------------
# Shape of the output
# --------------------------------------------------------------------------


def test_output_is_a_tile_grid_and_packs_two_bytes_per_cell(db):
    from lucifer_gen.contracts import TileGrid

    plan = carved_plan(12, 0x600D)
    grid = tileize(plan, db, TileClass.OUTDOOR, 0x5EED)
    assert isinstance(grid, TileGrid)
    assert grid.tile_class is TileClass.OUTDOOR
    assert grid.tileset_ref == db.version
    assert grid.cell_m == db.cell_m

    blob = grid.packed_cells()
    assert len(blob) == 2 * plan.grid * plan.grid
    first = Placement.unpack(blob[0], blob[1])
    assert first == grid.at((0, 0))
    assert grid.matched_cells + len(grid.filler) == plan.grid * plan.grid


def test_rejects_a_bad_plan_or_tile_class(db):
    with pytest.raises(TileizeError):
        tileize(TerrainPlan.blank(0), db, TileClass.OUTDOOR, 1)
    with pytest.raises(TypeError):
        tileize(TerrainPlan.blank(4), db, "outdoor", 1)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
