"""Tests for the Phase 2 gate checks.

Spec: docs/WORLD_BIBLE.md, the Phase 2 gate -- no navmesh island, no seam
mismatch, over a thousand seeds.

The grids here are hand-built from ASCII art rather than generated, so a
failure points at the checker and not at stage 3 or 4.  ``build_grid`` makes
a grid that is correct by construction: every cell presents OPEN towards a
walkable neighbour and WALL towards anything else, which is exactly the rule
``contracts.sides_compatible`` enforces.  A test that wants a broken grid
therefore has to break it on purpose, one placement at a time, and can then
say precisely which seam should be reported.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Runnable as `pytest tests/test_validate.py` or `python3 tests/test_validate.py`
# from anywhere, without depending on how the package root reaches sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_gen.contracts import (
    ALL_SLOTS,
    E,
    N,
    S,
    W,
    CellKind,
    EdgeSig,
    GeneratedMap,
    GraphTemplate,
    Placement,
    Role,
    RoutedEdge,
    RoutedLayout,
    RoutedNode,
    SetPiecePlacement,
    Shape,
    SideSpec,
    SpawnPack,
    TemplateEdge,
    TemplateNode,
    TerrainPlan,
    Tile,
    TileClass,
    TileGrid,
)
from lucifer_gen.seed import format_seed
from lucifer_gen.validate import (
    Seam,
    SuiteReport,
    ValidationReport,
    build_map,
    find_navmesh_islands,
    find_seam_mismatches,
    find_stage4_breaks,
    find_transform_breaks,
    flood,
    geometric_sides,
    geometrically_fits,
    reaches,
    run_suite,
    seam_is_open,
    suite_seed,
    validate_map,
    walkable_cells,
)

# --------------------------------------------------------------------------
# A tiny tile database: one filler plus every open/wall combination
# --------------------------------------------------------------------------

OPEN = SideSpec(EdgeSig.OPEN, ALL_SLOTS)
WALL = SideSpec(EdgeSig.WALL)

FILLER_ID = 0


def floor_id(mask: int) -> int:
    """Tile id for a floor open on exactly the sides named by ``mask``.

    Bit ``i`` of the mask is side ``i`` in N, E, S, W order.
    """
    return 16 + mask


def _make_tiles() -> dict:
    tiles = {
        FILLER_ID: Tile(
            id=FILLER_ID,
            name="filler",
            tile_class=TileClass.BOTH,
            sides=(WALL, WALL, WALL, WALL),
            walkable=False,
        )
    }
    for mask in range(16):
        sides = tuple(OPEN if mask & (1 << side) else WALL for side in (N, E, S, W))
        tiles[floor_id(mask)] = Tile(
            id=floor_id(mask),
            name=f"floor{mask:04b}",
            tile_class=TileClass.BOTH,
            sides=sides,
        )
    return tiles


TILES = _make_tiles()

DELTA = {N: (0, -1), E: (1, 0), S: (0, 1), W: (-1, 0)}


def build_grid(rows, order="row-major") -> TileGrid:
    """A seam-clean grid from ASCII art: ``.`` is floor, ``#`` is filler.

    ``order`` only changes the sequence the cells are written in, never the
    result; the order-independence test leans on that.
    """
    n = len(rows)
    assert all(len(r) == n for r in rows), "the grid is square"
    grid = TileGrid.blank(n)

    def is_floor(x: int, y: int) -> bool:
        return 0 <= x < n and 0 <= y < n and rows[y][x] == "."

    cells = [(x, y) for y in range(n) for x in range(n)]
    if order == "reversed":
        cells.reverse()
    elif order == "columns":
        cells = [(x, y) for x in range(n) for y in range(n)]

    for x, y in cells:
        if is_floor(x, y):
            mask = 0
            for side, (dx, dy) in DELTA.items():
                if is_floor(x + dx, y + dy):
                    mask |= 1 << side
            grid.put((x, y), Placement(floor_id(mask)), walkable=True)
        else:
            grid.put((x, y), Placement(FILLER_ID), walkable=False)
    return grid


HEALTHY = [
    "######",
    "#....#",
    "#.##.#",
    "#.##.#",
    "#....#",
    "######",
]

# A wide left-hand room and a right-hand column with no way across.
POCKET = [
    "######",
    "#..#.#",
    "#..#.#",
    "#..#.#",
    "#..#.#",
    "######",
]


def test_the_helper_builds_a_clean_grid():
    """Guard the guard: the hand-built healthy grid must itself be clean."""
    grid = build_grid(HEALTHY)
    assert find_seam_mismatches(grid, TILES) == []
    assert find_navmesh_islands(grid, (1, 1)) == []
    assert len(walkable_cells(grid)) == 12


# --------------------------------------------------------------------------
# Gate check 1: navmesh islands
# --------------------------------------------------------------------------


def test_finds_exactly_the_disconnected_pocket():
    grid = build_grid(POCKET)
    islands = find_navmesh_islands(grid, (1, 1))
    assert len(islands) == 1
    assert islands[0] == frozenset({(4, 1), (4, 2), (4, 3), (4, 4)})
    # ...and the cells on the entrance's own side are not called islands.
    assert (1, 1) not in islands[0]


def test_healthy_grid_has_no_islands_from_any_walkable_start():
    grid = build_grid(HEALTHY)
    for start in walkable_cells(grid):
        assert find_navmesh_islands(grid, start) == []


def test_connectivity_is_four_connected_not_diagonal():
    """Two floors touching only at a corner are two regions, as the navmesh
    bakes them: an actor cannot slip between two diagonal cells."""
    rows = [
        "####",
        "#.##",
        "##.#",
        "####",
    ]
    grid = build_grid(rows)
    islands = find_navmesh_islands(grid, (1, 1))
    assert islands == [frozenset({(2, 2)})]


def test_two_islands_come_back_sorted_by_their_smallest_cell():
    rows = [
        "#######",
        "#.#.#.#",
        "#######",
    ] + ["#######"] * 4
    grid = build_grid(rows)
    islands = find_navmesh_islands(grid, (1, 1))
    assert islands == [frozenset({(3, 1)}), frozenset({(5, 1)})]


def test_unwalkable_start_makes_every_region_an_island():
    grid = build_grid(HEALTHY)
    islands = find_navmesh_islands(grid, (0, 0))  # a filler cell
    assert len(islands) == 1
    assert islands[0] == frozenset(walkable_cells(grid))
    # Off the grid entirely reads the same way.
    assert find_navmesh_islands(grid, (-3, 99)) == islands


def test_flood_fill_survives_a_grid_sized_snake():
    """A serpentine corridor touching ~half of a 48x48 grid.

    A recursive fill would recurse once per cell and blow the default limit
    of 1000 long before the end of it; this is the case the spec calls
    pathological and the reason the fill is iterative.
    """
    n = 48
    rows = []
    for y in range(n):
        if y % 2 == 0:
            rows.append("." * n)
        elif (y // 2) % 2 == 0:
            rows.append("#" * (n - 1) + ".")
        else:
            rows.append("." + "#" * (n - 1))
    grid = build_grid(rows)
    assert len(walkable_cells(grid)) > 1000
    assert find_navmesh_islands(grid, (0, 0)) == []
    assert reaches(grid, (0, 0), (0, n - 1))


# --------------------------------------------------------------------------
# Gate check 2: seams
# --------------------------------------------------------------------------


def test_finds_exactly_the_incompatible_pair():
    rows = [
        "####",
        "#..#",
        "####",
        "####",
    ]
    grid = build_grid(rows)
    assert find_seam_mismatches(grid, TILES) == []

    # Wall off the west side of (2, 1): its neighbour still shows OPEN east.
    grid.put((2, 1), Placement(floor_id(0)), walkable=True)
    bad = find_seam_mismatches(grid, TILES)
    assert len(bad) == 1
    cell_a, cell_b, side, reason = bad[0]
    assert (cell_a, cell_b, side) == ((1, 1), (2, 1), E)
    assert "OPEN" in reason and "WALL" in reason
    # It is a plain tuple as well as a named one.
    assert bad[0] == ((1, 1), (2, 1), E, reason)
    assert isinstance(bad[0], tuple) and isinstance(bad[0], Seam)


def test_open_sides_that_share_no_slot_are_a_mismatch():
    """Complementary signatures are not enough: the doorways must line up."""
    west = Tile(
        id=1,
        name="east door in the first third",
        tile_class=TileClass.BOTH,
        sides=(WALL, SideSpec(EdgeSig.OPEN, 0b001), WALL, WALL),
    )
    east = Tile(
        id=2,
        name="west door in the first third",
        tile_class=TileClass.BOTH,
        sides=(WALL, WALL, WALL, SideSpec(EdgeSig.OPEN, 0b001)),
    )
    tiles = {1: west, 2: east}
    grid = TileGrid.blank(2)
    grid.put((0, 0), Placement(1), walkable=True)
    grid.put((1, 0), Placement(2), walkable=True)

    bad = find_seam_mismatches(grid, tiles)
    assert len(bad) == 1
    assert bad[0].cell_a == (0, 0) and bad[0].cell_b == (1, 0) and bad[0].side == E
    assert "connection slot" in bad[0].reason

    # The same two doors at the far third of each side do meet.
    tiles[2] = Tile(
        id=2,
        name="west door in the last third",
        tile_class=TileClass.BOTH,
        sides=(WALL, WALL, WALL, SideSpec(EdgeSig.OPEN, 0b100)),
    )
    assert find_seam_mismatches(grid, tiles) == []


def test_each_pair_is_judged_once_not_twice():
    """Every seam of a 2x2 block is broken; there are four seams, not eight.

    The cells alternate open-all-round and walled-all-round, so each of the
    four interior seams shows OPEN against WALL.
    """
    grid = TileGrid.blank(2)
    for cell in ((0, 0), (1, 1)):
        grid.put(cell, Placement(floor_id(0b1111)), walkable=True)
    for cell in ((1, 0), (0, 1)):
        grid.put(cell, Placement(floor_id(0)), walkable=True)

    bad = find_seam_mismatches(grid, TILES)
    pairs = {(s.cell_a, s.cell_b) for s in bad}
    assert len(bad) == len(pairs) == 4
    # Each seam is reported from its west or north cell only.
    assert all(side in (E, S) for _, _, side, _ in bad)
    assert ((1, 0), (0, 0)) not in pairs


def test_unplaced_cells_are_skipped():
    grid = TileGrid.blank(2)
    grid.put((0, 0), Placement(floor_id(0b1111)), walkable=True)
    # (1, 0) and the rest stay None: nothing to disagree with.
    assert find_seam_mismatches(grid, TILES) == []


def test_an_unknown_tile_id_is_reported_rather_than_raised():
    grid = build_grid(["..", ".."])
    grid.put((0, 0), Placement(200), walkable=True)
    bad = find_seam_mismatches(grid, TILES)
    assert len(bad) == 2  # its east and south seams
    assert all("200" in s.reason for s in bad)


def test_a_real_tile_database_resolves_the_same_way():
    """The checker reads a TileDatabase through ``sides_of`` when it has one."""
    tiles_mod = pytest.importorskip("lucifer_gen.tiles")
    db = tiles_mod.TileDatabase.load()
    grid = TileGrid.blank(2)
    filler = db.filler_placement()
    for cell in ((0, 0), (1, 0), (0, 1), (1, 1)):
        grid.put(cell, filler, walkable=False)
    assert find_seam_mismatches(grid, db) == []

    # A tile that opens east, against sealed filler, must be caught.
    open_tile = next(t for t in db.tiles if t.sides[E].sig is EdgeSig.OPEN)
    grid.put((0, 0), Placement(open_tile.id), walkable=True)
    assert find_seam_mismatches(grid, db)


# --------------------------------------------------------------------------
# Order independence and determinism
# --------------------------------------------------------------------------


@pytest.mark.parametrize("order", ["row-major", "reversed", "columns"])
def test_both_checks_ignore_the_order_cells_were_written_in(order):
    grid = build_grid(POCKET, order=order)
    reference = build_grid(POCKET)
    assert find_navmesh_islands(grid, (1, 1)) == find_navmesh_islands(reference, (1, 1))
    assert find_seam_mismatches(grid, TILES) == find_seam_mismatches(reference, TILES)


def test_repeated_runs_give_identical_answers():
    grid = build_grid(POCKET)
    grid.put((1, 1), Placement(floor_id(0)), walkable=True)  # break some seams too
    first = (find_navmesh_islands(grid, (2, 1)), find_seam_mismatches(grid, TILES))
    for _ in range(5):
        assert (find_navmesh_islands(grid, (2, 1)), find_seam_mismatches(grid, TILES)) == first


def test_tiles_may_be_a_mapping_or_a_sequence():
    grid = build_grid(POCKET)
    grid.put((1, 1), Placement(floor_id(0)), walkable=True)
    as_mapping = find_seam_mismatches(grid, TILES)
    as_sequence = find_seam_mismatches(grid, sorted(TILES.values(), key=lambda t: -t.id))
    assert as_mapping == as_sequence
    assert as_mapping  # the broken grid really does report something


# --------------------------------------------------------------------------
# validate_map
# --------------------------------------------------------------------------

CORRIDOR = [
    "#######",
    "#.....#",
    "#######",
    "#######",
    "#######",
    "#######",
    "#######",
]


def _template(grid: int = 7) -> GraphTemplate:
    return GraphTemplate(
        id="unit",
        version=1,
        tile_class=TileClass.DUNGEON,
        shape=Shape.I,
        grid=grid,
        cell_m=4.0,
        nodes=(
            TemplateNode(id="in", role=Role.ENTRANCE, anchor="shape.start"),
            TemplateNode(id="out", role=Role.EXIT, anchor="shape.end"),
        ),
        edges=(TemplateEdge(a="in", b="out"),),
        tileset="unit_v1",
    )


def _map(rows=CORRIDOR, **overrides) -> GeneratedMap:
    """A minimal but complete GeneratedMap over a straight corridor."""
    grid = build_grid(rows)
    n = len(rows)
    path = [(x, 1) for x in range(1, n - 1)]
    template = _template(n)
    routed = RoutedLayout(
        seed=7,
        template=template,
        grid=n,
        nodes={
            "in": RoutedNode(id="in", role=Role.ENTRANCE, cell=(1, 1)),
            "out": RoutedNode(id="out", role=Role.EXIT, cell=(n - 2, 1)),
        },
        edges=[RoutedEdge(a="in", b="out", path=path)],
    )
    terrain = TerrainPlan.blank(n)
    for cell in path:
        terrain.set_kind(cell, CellKind.CORRIDOR)
    fields = dict(
        seed=7,
        template=template,
        tileset_ref="unit@1",
        routed=routed,
        terrain=terrain,
        tiles=grid,
        set_pieces=[],
        spawns=[],
        exit_cell=(n - 2, 1),
        checkpoints=[],
    )
    fields.update(overrides)
    return GeneratedMap(**fields)


def test_a_sound_map_validates_clean():
    report = validate_map(_map(), TILES)
    assert report.ok, report.summary()
    assert report.problems == () and report.islands == () and report.seams == ()
    assert report.counts() == {}
    assert report.start_cell == (1, 1)
    assert report.template_ref == "unit@1" and report.tileset_ref == "unit@1"
    assert format_seed(7) in report.summary()
    assert "clean" in report.summary()


def test_an_unwalkable_exit_is_reported():
    gmap = _map()
    gmap.exit_cell = (0, 0)  # filler
    report = validate_map(gmap, TILES)
    assert not report.ok
    assert [p.kind for p in report.problems] == ["exit-not-walkable"]


def test_an_out_of_bounds_exit_is_reported():
    gmap = _map()
    gmap.exit_cell = (99, 99)
    assert [p.kind for p in validate_map(gmap, TILES).problems] == ["exit-out-of-bounds"]


def test_an_unreachable_exit_is_reported_alongside_its_island():
    rows = [
        "#######",
        "#.#...#",
        "#######",
        "#######",
        "#######",
        "#######",
        "#######",
    ]
    report = validate_map(_map(rows), TILES)
    kinds = [p.kind for p in report.problems]
    assert "exit-unreachable" in kinds
    assert report.islands == (frozenset({(3, 1), (4, 1), (5, 1)}),)
    assert report.island_cells == 3
    assert report.counts()["navmesh-island"] == 1
    assert "FAILED" in report.summary()


def test_a_set_piece_hanging_off_the_grid_is_reported():
    piece = SetPiecePlacement(id="boss", cell=(4, 1), rot=0, w=5, h=5)
    report = validate_map(_map(set_pieces=[piece]), TILES)
    assert [p.kind for p in report.problems] == ["set-piece-out-of-bounds"]
    assert "boss" in report.problems[0].message

    inside = SetPiecePlacement(id="boss", cell=(1, 1), rot=0, w=2, h=1)
    assert validate_map(_map(set_pieces=[inside]), TILES).ok


def test_spawns_must_sit_on_walkable_non_approach_floor():
    good = SpawnPack(pack="ghouls", cell=(3, 1), count=4)
    assert validate_map(_map(spawns=[good]), TILES).ok

    on_filler = SpawnPack(pack="ghouls", cell=(3, 3), count=4)
    assert [p.kind for p in validate_map(_map(spawns=[on_filler]), TILES).problems] == [
        "spawn-not-walkable"
    ]

    off_grid = SpawnPack(pack="ghouls", cell=(50, 50), count=4)
    assert [p.kind for p in validate_map(_map(spawns=[off_grid]), TILES).problems] == [
        "spawn-out-of-bounds"
    ]

    gmap = _map(spawns=[good])
    gmap.terrain.set_kind((3, 1), CellKind.APPROACH)
    report = validate_map(gmap, TILES)
    assert [p.kind for p in report.problems] == ["spawn-on-approach"]


def test_a_missing_or_unwalkable_entrance_is_reported():
    gmap = _map()
    gmap.routed.nodes.pop("in")
    kinds = [p.kind for p in validate_map(gmap, TILES).problems]
    assert kinds == ["entrance-missing"]

    gmap = _map()
    gmap.routed.nodes["in"] = RoutedNode(id="in", role=Role.ENTRANCE, cell=(0, 0))
    report = validate_map(gmap, TILES)
    assert "entrance-not-walkable" in [p.kind for p in report.problems]
    # Nothing is reachable from a wall, so the whole corridor reads as an island.
    assert report.island_cells == 5


def test_a_broken_seam_shows_up_in_the_report():
    gmap = _map()
    gmap.tiles.put((3, 1), Placement(floor_id(0)), walkable=True)
    report = validate_map(gmap, TILES)
    assert len(report.seams) == 2  # its west and east seams, from each owner
    assert report.counts()["seam-mismatch"] == 2
    assert not report.ok


# --------------------------------------------------------------------------
# run_suite
# --------------------------------------------------------------------------


def test_suite_seeds_are_reproducible_and_spread_across_the_word():
    start = 0xDEAD_BEEF
    assert suite_seed(start, 0) == start  # index 0 reruns a reported seed
    assert suite_seed(start, 7) == suite_seed(start, 7)
    assert suite_seed(start, 7) != suite_seed(start + 1, 7)
    drawn = [suite_seed(start, i) for i in range(64)]
    assert len(set(drawn)) == 64
    # The tile field lives in bits 32-63; a counting sweep would never move it.
    assert len({s >> 32 for s in drawn}) > 32
    assert all(0 <= s < 1 << 64 for s in drawn)


def test_the_suite_counts_and_names_the_first_failure():
    healthy = _map()
    broken = _map()
    broken.exit_cell = (0, 0)
    seeds = [11, 22, 33, 44]
    seen = []

    def generate(seed):
        return broken if seed in (22, 44) else healthy

    def on_result(index, seed, report):
        seen.append((index, seed, None if report is None else report.ok))

    report = run_suite(
        _template(), TILES, None, seeds=seeds, generate=generate, on_result=on_result
    )
    assert isinstance(report, SuiteReport)
    assert (report.requested, report.checked, report.clean, report.failed) == (4, 4, 2, 2)
    assert report.crashed == 0
    assert report.problems == {"exit-not-walkable": 2}
    assert not report.ok

    failure = report.first_failure
    assert failure is not None and failure.seed == 22 and failure.index == 1
    assert not failure.crashed
    assert format_seed(22) in failure.repro()
    assert "reproduce with" in failure.summary()
    assert seen == [(0, 11, True), (1, 22, False), (2, 33, True), (3, 44, False)]


def test_the_suite_treats_a_raising_seed_as_that_seed_failing():
    def generate(seed):
        if seed == 2:
            raise RuntimeError("stage 3 fell over")
        return _map()

    report = run_suite(_template(), TILES, None, seeds=[1, 2, 3], generate=generate)
    assert (report.checked, report.clean, report.failed, report.crashed) == (3, 2, 1, 1)
    assert report.first_failure is not None and report.first_failure.seed == 2
    assert report.first_failure.crashed
    assert "stage 3 fell over" in report.first_failure.error
    assert "stage 3 fell over" in report.first_failure.summary()


def test_stop_early_stops_at_the_first_failure():
    def generate(seed):
        gmap = _map()
        if seed >= 2:
            gmap.exit_cell = (0, 0)
        return gmap

    report = run_suite(
        _template(), TILES, None, seeds=[1, 2, 3, 4], generate=generate, stop_early=True
    )
    assert report.checked == 2 and report.failed == 1
    assert report.first_failure.seed == 2


def test_a_clean_sweep_reports_ok_and_keeps_no_failure():
    report = run_suite(
        _template(), TILES, None, n_seeds=5, start_seed=1, generate=lambda seed: _map()
    )
    assert report.ok and report.first_failure is None
    assert report.checked == report.clean == 5
    assert "5 clean" in report.summary()


def test_the_suite_holds_one_map_at_a_time():
    """Streaming, not batching: the generator is driven one seed at a time and
    no map outlives its iteration."""
    import weakref

    alive = []

    def generate(seed):
        gmap = _map()
        alive.append(weakref.ref(gmap))
        # Every previously generated map must already be collectable.
        assert sum(1 for ref in alive if ref() is not None) == 1
        return gmap

    report = run_suite(_template(), TILES, None, n_seeds=6, generate=generate)
    assert report.checked == 6


# --------------------------------------------------------------------------
# End to end, against the real stages
# --------------------------------------------------------------------------


def _pipeline():
    """The real generator, or a skip if a sibling stage is unavailable."""
    template = pytest.importorskip("lucifer_gen.template")
    tiles_mod = pytest.importorskip("lucifer_gen.tiles")
    rooms_mod = pytest.importorskip("lucifer_gen.rooms")
    pytest.importorskip("lucifer_gen.route")
    pytest.importorskip("lucifer_gen.translate")
    pytest.importorskip("lucifer_gen.tileize")
    pytest.importorskip("lucifer_gen.setpieces")
    pytest.importorskip("lucifer_gen.spawn")
    return (
        template.load_builtin_template("crypt"),
        tiles_mod.TileDatabase.load(),
        rooms_mod.RoomLibrary.load(),
    )


def test_generated_maps_clear_the_gate():
    template, db, rooms = _pipeline()
    report = run_suite(template, db, rooms, n_seeds=12, start_seed=0x5EED)
    assert report.ok, report.summary()
    assert report.checked == 12 and report.islands == 0 and report.seams == 0


def test_a_generated_map_reports_its_own_provenance():
    template, db, rooms = _pipeline()
    gmap = build_map(template, db, rooms, 0x1234_5678_9ABC_DEF0)
    report = validate_map(gmap, db)
    assert isinstance(report, ValidationReport)
    assert report.ok, report.summary()
    assert report.template_ref == template.ref
    assert report.tileset_ref == db.version
    assert report.grid == template.grid
    assert report.seed == gmap.seed


# --------------------------------------------------------------------------
# A sealed seam: the defect neither gate check could see
# --------------------------------------------------------------------------


def _seal_between(grid: TileGrid, a, b) -> None:
    """Wall off the seam between two adjacent floor cells, nothing else.

    Both cells keep their walkable flag -- which is exactly the situation
    stage 4 and stage 5 produce, since that flag is a copy of the terrain
    plan's floor mask and is never re-derived from the tile that was placed.
    The result is still seam-clean: ``WALL`` meets ``WALL``.
    """
    (ax, ay), (bx, by) = a, b
    side = {(1, 0): E, (-1, 0): W, (0, 1): S, (0, -1): N}[(bx - ax, by - ay)]
    opposite = {N: S, E: W, S: N, W: E}[side]
    for cell, drop in ((a, side), (b, opposite)):
        old = grid.at(cell)
        mask = old.tile_id - 16
        grid.put(cell, Placement(floor_id(mask & ~(1 << drop))), walkable=True)


def test_a_walled_seam_between_two_floor_cells_is_an_island():
    """The critical gate hole: a sealed corridor used to pass both checks.

    ``TileGrid.walkable`` is a copy of stage 3's floor mask, so flooding it
    alone says the two halves are connected; ``sides_compatible`` is happy
    because a wall meeting a wall is perfectly consistent.  Only reading the
    two together -- a step needs an *open* seam -- sees the wall.
    """
    grid = build_grid(CORRIDOR)
    n = grid.grid
    mid = n // 2
    _seal_between(grid, (mid, 1), (mid + 1, 1))

    # Still seam-clean, and bare adjacency still calls the corridor connected.
    assert find_seam_mismatches(grid, TILES) == []
    assert find_navmesh_islands(grid, (1, 1)) == []

    # Reading the tiles, the far half is unreachable.
    islands = find_navmesh_islands(grid, (1, 1), TILES)
    assert len(islands) == 1
    assert (n - 2, 1) in islands[0]
    assert (1, 1) not in islands[0]
    assert not reaches(grid, (1, 1), (n - 2, 1), tiles=TILES)
    assert reaches(grid, (1, 1), (n - 2, 1))  # the old, blind reading
    assert len(flood(grid, (1, 1), tiles=TILES)) < len(flood(grid, (1, 1)))


def test_validate_map_fails_a_sealed_map():
    """The gate's own entry point must catch what the pieces catch."""
    gmap = _map()
    n = gmap.tiles.grid
    mid = n // 2
    _seal_between(gmap.tiles, (mid, 1), (mid + 1, 1))
    report = validate_map(gmap, TILES)
    assert not report.ok
    counts = report.counts()
    assert counts.get("navmesh-island") == 1
    assert counts.get("exit-unreachable") == 1
    assert "seam-mismatch" not in counts  # every seam is still self-consistent


def test_seam_is_open_needs_an_open_edge_on_both_sides():
    assert seam_is_open(OPEN, OPEN)
    assert not seam_is_open(WALL, WALL)
    # Slot k of one side faces slot 2-k of the other, so two sides that each
    # only offer their own first third never meet.
    assert not seam_is_open(
        SideSpec(EdgeSig.OPEN, 0b001), SideSpec(EdgeSig.OPEN, 0b001)
    )
    assert seam_is_open(SideSpec(EdgeSig.OPEN, 0b100), SideSpec(EdgeSig.OPEN, 0b001))


# --------------------------------------------------------------------------
# The transform algebra, read twice
# --------------------------------------------------------------------------


def test_the_geometric_model_agrees_with_the_shipped_algebra():
    """Every orientation of every tile, both readings, on the real database.

    This is the cross-check the gate used to lack entirely: ``Tile.transformed``
    had exactly one implementation and every "independent" checker called it.
    """
    tiles_mod = pytest.importorskip("lucifer_gen.tiles")
    db = tiles_mod.TileDatabase.load()
    checked = 0
    for tile in db.tiles:
        for flip in (False, True):
            for rot in range(4):
                assert tuple(tile.transformed(rot, flip)) == geometric_sides(
                    tile, rot, flip
                ), (tile.name, rot, flip)
                checked += 1
    assert checked == len(db.tiles) * 8


def test_the_geometric_model_agrees_on_asymmetric_slot_masks():
    """Slot handling, which the shipped data cannot exercise.

    Every open side in the greybox database offers all three slots, so
    ``_reverse_slots`` is the identity on real data and no map-level check can
    ever see a bug in it.  A synthetic tile with lopsided slots can.
    """
    lopsided = Tile(
        id=99,
        name="lopsided",
        tile_class=TileClass.BOTH,
        sides=(
            SideSpec(EdgeSig.OPEN, 0b001),
            SideSpec(EdgeSig.OPEN, 0b110),
            SideSpec(EdgeSig.WALL),
            SideSpec(EdgeSig.OPEN, 0b100),
        ),
    )
    for flip in (False, True):
        for rot in range(4):
            assert tuple(lopsided.transformed(rot, flip)) == geometric_sides(
                lopsided, rot, flip
            ), (rot, flip)


def test_the_two_compatibility_readings_agree_everywhere():
    from lucifer_gen.contracts import sides_compatible

    specs = [SideSpec(sig, mask) for sig in EdgeSig for mask in range(ALL_SLOTS + 1)]
    for a in specs:
        for b in specs:
            assert sides_compatible(a, b) == geometrically_fits(a, b), (a, b)


class _WrongWayRound(Tile):
    """A tile whose transform turns anticlockwise: the likeliest algebra bug."""

    def transformed(self, rot, flip):
        sides = self.sides
        if flip:
            sides = (
                sides[N].mirrored(),
                sides[W].mirrored(),
                sides[S].mirrored(),
                sides[E].mirrored(),
            )
        rot &= 3
        if rot:
            sides = tuple(sides[(i + rot) % 4] for i in (N, E, S, W))
        return tuple(sides)


def test_a_broken_transform_is_reported():
    """A sign error in the rotation must not be able to ship green."""
    liar = _WrongWayRound(
        id=50,
        name="liar",
        tile_class=TileClass.BOTH,
        sides=(OPEN, WALL, WALL, WALL),
    )
    table = dict(TILES)
    table[liar.id] = liar
    grid = TileGrid.blank(1)
    grid.put((0, 0), Placement(liar.id, rot=1), walkable=True)

    problems = find_transform_breaks(grid, table)
    assert len(problems) == 1
    assert problems[0].kind == "transform-mismatch"
    assert "liar" in problems[0].message

    # ...and rot=0 is not a false positive.
    clean = TileGrid.blank(1)
    clean.put((0, 0), Placement(liar.id, rot=0), walkable=True)
    assert find_transform_breaks(clean, table) == []


def test_a_seam_rule_that_disagrees_with_geometry_is_reported(monkeypatch):
    """If ``sides_compatible`` itself were wrong, both old readings agreed."""
    import lucifer_gen.validate as validate_module

    monkeypatch.setattr(validate_module, "sides_compatible", lambda a, b: True)
    grid = build_grid(HEALTHY)
    grid.put((1, 1), Placement(floor_id(0)), walkable=True)  # sealed on all sides
    seams = find_seam_mismatches(grid, TILES)
    assert seams, "a wall facing a doorway must still be reported"
    assert any("geometric" in seam.reason for seam in seams)


# --------------------------------------------------------------------------
# Stage 4's own invariants, which the gate never used to run
# --------------------------------------------------------------------------


def test_the_gate_runs_stage_4_s_invariants():
    """``tileize.debug_check`` is wired into the gate, not just into a unit test.

    Before this, nothing outside two unit tests called it, so a 1000-seed run
    never checked border sealing, per-cell requirement satisfaction or the
    hero budget on a single real map.
    """
    template, db, rooms = _pipeline()
    gmap = build_map(template, db, rooms, 0xC0FFEE)
    assert find_stage4_breaks(gmap, db) == []
    assert validate_map(gmap, db).ok

    # A cell on the border showing OPEN outward has no neighbour, so it makes
    # neither a seam nor an island: this check is the only one that sees it.
    grid = gmap.tiles
    edge = None
    for x in range(grid.grid):
        if grid.at((x, 0)) is not None:
            edge = (x, 0)
            break
    assert edge is not None
    leak = next(t for t in db.tiles if t.sides[N].sig is EdgeSig.OPEN and not t.hero)
    grid.put(edge, Placement(leak.id, rot=0, flip=False), walkable=False)

    problems = find_stage4_breaks(gmap, db)
    assert problems and all(p.kind == "stage4-invariant" for p in problems)
    assert any("off the edge of the map" in p.message for p in problems)
    assert not validate_map(gmap, db).ok


def test_stage4_breaks_are_skipped_for_a_hand_built_grid():
    """A ``TileGrid`` with no stage 4 bookkeeping has nothing to re-derive."""
    assert find_stage4_breaks(_map(), TILES) == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
