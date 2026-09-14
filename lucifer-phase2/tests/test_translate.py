"""Tests for stage 3: translating a routed layout into terrain.

Spec: docs/WORLD_BIBLE.md stage 3.

The bulk of the file is property testing over 200 seeds of each tile class,
because stage 3's contract is a set of invariants rather than a fixed answer:
every node stands on floor, the floor is 4-connected from the entrance, nothing
leaves the grid, rooms are placed but never scaled, and the same seed always
gives the same terrain.
"""

from __future__ import annotations

import functools
import math
import pathlib
import sys

import pytest

# Allow ``pytest tests/test_translate.py`` from anywhere, not only ``python -m
# pytest`` from the package root.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lucifer_gen import translate as T  # noqa: E402
from lucifer_gen.contracts import (  # noqa: E402
    SIDE_DELTA,
    CellKind,
    Role,
    RoutedEdge,
    RoutedLayout,
    RoutedNode,
    TileClass,
)
from lucifer_gen.rooms import RoomLibrary  # noqa: E402
from lucifer_gen.route import MARGIN, route  # noqa: E402
from lucifer_gen.seed import MASK64, SeedFields  # noqa: E402
from lucifer_gen.template import load_builtin_template, template_from_dict  # noqa: E402

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

#: 200 well spread seeds; the golden-ratio step keeps every seed field moving.
SEEDS = tuple((i * 0x9E3779B97F4A7C15) & MASK64 for i in range(200))

#: The shipped template of each class.
DUNGEON = "crypt"
OUTDOOR = "ashen_ramparts"
BOTH_CLASSES = (DUNGEON, OUTDOOR)

#: An outdoor template that names no water anywhere, so its banks are absent.
DRY_RIDGE = {
    "id": "dry_ridge",
    "version": 1,
    "class": "outdoor",
    "shape": "I",
    "grid": 48,
    "cell_m": 4,
    "tileset": "ash_v1",
    "landmarks": ["ash_beacon"],
    "nodes": [
        {"id": "in", "role": "entrance", "anchor": "shape.start"},
        {"id": "mid", "role": "side", "anchor": "shape.centre"},
        {"id": "out", "role": "exit", "anchor": "shape.end"},
    ],
    "edges": [{"a": "in", "b": "mid"}, {"a": "mid", "b": "out"}],
}


@functools.lru_cache(maxsize=None)
def library() -> RoomLibrary:
    return RoomLibrary.load()


@functools.lru_cache(maxsize=None)
def plans(name: str):
    """``(seed, routed, plan)`` for every seed, built once per template."""
    template = load_builtin_template(name)
    out = []
    for seed in SEEDS:
        routed = route(template, seed)
        out.append((seed, routed, T.translate(routed, library(), seed)))
    return tuple(out)


def floor_reachable(plan, start):
    """Every floor cell 4-connected to ``start``, by an independent walk."""
    if not plan.is_floor(start):
        return set()
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in SIDE_DELTA:
            cell = (x + dx, y + dy)
            if cell not in seen and plan.is_floor(cell):
                seen.add(cell)
                stack.append(cell)
    return seen


def signature(plan):
    """Everything stage 3 decides, in a comparable form."""
    return (
        tuple(tuple(int(k) for k in row) for row in plan.kinds),
        tuple(
            (r.room_id, r.node_id, r.origin, r.w, r.h, r.rot, r.flip)
            for r in plan.rooms
        ),
        tuple(tuple(p for p in spline) for spline in plan.splines),
    )


def kinds_present(plan):
    return {CellKind(k) for row in plan.kinds for k in row}


def corridor_footprints(routed, plan, fields):
    """Recompute, per edge, the cells stage 3 carves as corridor.

    A dungeon corridor follows the routed path; an outdoor ridge follows the
    rasterised spline, so the two classes start from different spines and both
    widen through :func:`translate.corridor_cells`.
    """
    outdoor = routed.template.tile_class is TileClass.OUTDOOR
    out = []
    for edge in routed.edges:
        width, widen_first = T.corridor_profile(fields, edge)
        if outdoor:
            spine = T.rasterise(
                T.bezier_spline(T.smooth_centres(edge.path)), plan.grid, clamp=True
            )
        else:
            spine = list(edge.path)
        out.append((edge, width, spine, T.corridor_cells(spine, width, widen_first, plan.grid)))
    return out


# --------------------------------------------------------------------------
# The invariants, over 200 seeds of each class
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", BOTH_CLASSES)
def test_every_node_cell_is_floor(name):
    for seed, routed, plan in plans(name):
        for node in routed.nodes.values():
            assert plan.is_floor(node.cell), (
                f"{name} seed {seed:#x}: node {node.id} at {node.cell} "
                f"is {plan.kind(node.cell).name}, not floor"
            )


@pytest.mark.parametrize("name", BOTH_CLASSES)
def test_floor_is_four_connected_from_the_entrance(name):
    for seed, routed, plan in plans(name):
        entrance = routed.node_of_role(Role.ENTRANCE)
        assert entrance is not None
        reached = floor_reachable(plan, entrance.cell)
        for node in routed.nodes.values():
            assert node.cell in reached, (
                f"{name} seed {seed:#x}: node {node.id} at {node.cell} is cut off "
                f"from the entrance at {entrance.cell}"
            )


@pytest.mark.parametrize("name", BOTH_CLASSES)
def test_everything_stays_in_bounds(name):
    for seed, routed, plan in plans(name):
        grid = plan.grid
        assert grid == routed.grid
        assert len(plan.kinds) == grid
        assert all(len(row) == grid for row in plan.kinds)
        for placement in plan.rooms:
            for cell in placement.cells():
                assert plan.inside(cell), (
                    f"{name} seed {seed:#x}: room {placement.room_id} "
                    f"spills out of the grid at {cell}"
                )
        # Stage 2 keeps its paths inside a one-cell margin and stage 3 keeps
        # the same border, so the outer ring is left for stage 4's filler.
        for y, row in enumerate(plan.kinds):
            for x, kind in enumerate(row):
                if kind is not CellKind.EMPTY:
                    assert MARGIN <= x <= grid - 1 - MARGIN, (name, seed, x, y)
                    assert MARGIN <= y <= grid - 1 - MARGIN, (name, seed, x, y)


@pytest.mark.parametrize("name", BOTH_CLASSES)
def test_translate_is_deterministic(name):
    lib = library()
    for seed, routed, plan in plans(name):
        again = T.translate(routed, lib, seed)
        assert signature(again) == signature(plan), f"{name} seed {seed:#x}"
        # ... and from scratch, so nothing leaks between stage 2 and stage 3.
        fresh = T.translate(route(routed.template, seed), lib, seed)
        assert signature(fresh) == signature(plan), f"{name} seed {seed:#x}"


@pytest.mark.parametrize("name", BOTH_CLASSES)
def test_translate_ignores_node_dict_order(name):
    """A dict is ordered, so stage 3 must not read anything off that order."""
    lib = library()
    for seed, routed, plan in plans(name)[:25]:
        shuffled = RoutedLayout(
            seed=routed.seed,
            template=routed.template,
            grid=routed.grid,
            nodes={k: routed.nodes[k] for k in reversed(list(routed.nodes))},
            edges=list(routed.edges),
        )
        assert signature(T.translate(shuffled, lib, seed)) == signature(plan)


def test_rooms_are_never_scaled():
    lib = library()
    for seed, routed, plan in plans(DUNGEON):
        assert len(plan.rooms) == len(routed.nodes)
        assert sorted(r.node_id for r in plan.rooms) == sorted(routed.nodes)
        for placement in plan.rooms:
            room = lib.by_id(placement.room_id)  # a real library room
            assert (placement.w, placement.h) == (room.w, room.h), (
                f"seed {seed:#x}: room {room.id} was placed {placement.w}x"
                f"{placement.h}, library says {room.w}x{room.h}"
            )
            assert (placement.rot, placement.flip) == (0, False)
            for cell in placement.cells():
                assert plan.kind(cell) is CellKind.ROOM


def test_rooms_serve_every_incident_direction():
    """The spec filters the library by the directions the edges leave in."""
    lib = library()
    for seed, routed, plan in plans(DUNGEON):
        for placement in plan.rooms:
            wanted = T.incident_sides(routed, placement.node_id)
            room = lib.by_id(placement.room_id)
            assert room.serves(wanted), (
                f"seed {seed:#x}: room {room.id} has doorways {sorted(room.sides)} "
                f"but node {placement.node_id} needs {sorted(wanted)}"
            )
            assert placement.origin[0] <= routed.nodes[placement.node_id].cell[0]
            assert placement.origin[1] <= routed.nodes[placement.node_id].cell[1]


def test_repair_pass_is_never_needed_on_a_fresh_plan():
    """Both classes are built connected, so the safety net stays idle."""
    for name in BOTH_CLASSES:
        for seed, routed, plan in plans(name):
            assert T.repair_connectivity(plan, routed) == 0, f"{name} {seed:#x}"


# --------------------------------------------------------------------------
# Corridor width
# --------------------------------------------------------------------------


def test_corridor_widths_are_one_or_two_and_both_occur():
    seen = set()
    for name in BOTH_CLASSES:
        for seed, routed, _plan in plans(name):
            fields = SeedFields.parse(seed)
            for edge in routed.edges:
                width, widen_first = T.corridor_profile(fields, edge)
                assert width in (T.MIN_CORRIDOR_WIDTH, T.MAX_CORRIDOR_WIDTH)
                assert isinstance(widen_first, bool)
                seen.add((width, widen_first))
    assert seen == {(1, True), (1, False), (2, True), (2, False)}


@pytest.mark.parametrize("name", BOTH_CLASSES)
def test_width_two_corridors_never_leave_the_grid(name):
    """Every cell a width-2 corridor claims is inside the grid, and is floor."""
    widened = 0
    for seed, routed, plan in plans(name):
        grid = plan.grid
        for edge, width, spine, cells in corridor_footprints(
            routed, plan, SeedFields.parse(seed)
        ):
            assert all(plan.inside(c) for c in cells), (name, seed, edge.a, edge.b)
            assert set(spine) <= set(cells)
            for cell in cells:
                assert plan.is_floor(cell), (
                    f"{name} seed {seed:#x}: corridor cell {cell} of edge "
                    f"{edge.a}-{edge.b} reads {plan.kind(cell).name}"
                )
            if width == 2:
                widened += 1
                assert len(cells) > len(spine), "a width of 2 has to widen somewhere"
    assert widened, "no seed drew a width-2 corridor"


@pytest.mark.parametrize("widen_first", [True, False])
@pytest.mark.parametrize("width", [1, 2])
def test_corridor_cells_hugging_every_border(width, widen_first):
    """A corridor pinned against each border narrows rather than overflowing."""
    grid = 48
    lo, hi = MARGIN, grid - 1 - MARGIN
    runs = [
        [(x, lo) for x in range(lo, lo + 12)],  # along the north margin
        [(x, hi) for x in range(hi, hi - 12, -1)],  # along the south margin
        [(lo, y) for y in range(lo, lo + 12)],  # along the west margin
        [(hi, y) for y in range(hi, hi - 12, -1)],  # along the east margin
    ]
    for path in runs:
        cells = T.corridor_cells(path, width, widen_first, grid)
        assert set(path) <= set(cells), "the routed path itself must survive"
        assert len(cells) == len(set(cells)), "cells are reported once"
        for x, y in cells:
            assert 0 <= x < grid and 0 <= y < grid
            assert lo <= x <= hi and lo <= y <= hi
        if width == 2:
            # One of the two perpendiculars is off the map, so the corridor
            # widens to the other side and keeps its full width.
            assert len(cells) == 2 * len(path)


def test_corridor_cells_widen_perpendicular_to_travel():
    grid = 48
    path = [(10, 10), (11, 10), (12, 10)]  # heading east
    for widen_first in (True, False):
        extra = set(T.corridor_cells(path, 2, widen_first, grid)) - set(path)
        rows = {cell[1] for cell in extra}
        assert rows in ({9}, {11}), "widening must be north or south of an east run"
        assert len(extra) == len(path)


# --------------------------------------------------------------------------
# Dungeon specifics
# --------------------------------------------------------------------------


def test_dungeon_plans_carry_rooms_and_no_splines():
    for _seed, _routed, plan in plans(DUNGEON):
        assert plan.rooms
        assert plan.splines == []
        assert kinds_present(plan) <= {CellKind.EMPTY, CellKind.CORRIDOR, CellKind.ROOM}


def test_dungeon_rooms_win_over_corridors():
    """Where a room overlaps a corridor the cell reads as room."""
    lib = library()
    hits = 0
    for seed, routed, plan in plans(DUNGEON)[:50]:
        fields = SeedFields.parse(seed)
        corridor = set()
        for edge in routed.edges:
            width, widen_first = T.corridor_profile(fields, edge)
            corridor |= set(T.corridor_cells(edge.path, width, widen_first, plan.grid))
        for placement in plan.rooms:
            for cell in placement.cells():
                if cell in corridor:
                    hits += 1
                    assert plan.kind(cell) is CellKind.ROOM
    assert hits, "the node cell alone should put every room over a corridor"


def test_a_dungeon_needs_a_room_library():
    routed = route(load_builtin_template(DUNGEON), SEEDS[1])
    with pytest.raises(ValueError):
        T.translate(routed, None, SEEDS[1])


def test_incident_sides_read_the_first_step_out_of_the_node():
    """North out of ``a``, and west out of ``b`` at the far end."""
    template = load_builtin_template(DUNGEON)
    layout = RoutedLayout(
        seed=0,
        template=template,
        grid=template.grid,
        nodes={
            "a": RoutedNode("a", Role.ENTRANCE, (10, 10)),
            "b": RoutedNode("b", Role.EXIT, (12, 7)),
        },
        edges=[
            RoutedEdge("a", "b", [(10, 10), (10, 9), (10, 8), (10, 7), (11, 7), (12, 7)])
        ],
    )
    assert T.incident_sides(layout, "a") == {0}  # the path leaves north
    assert T.incident_sides(layout, "b") == {3}  # and arrives from the west


# --------------------------------------------------------------------------
# Outdoor specifics
# --------------------------------------------------------------------------


def test_outdoor_plans_carry_splines_and_no_room_placements():
    for seed, routed, plan in plans(OUTDOOR):
        assert plan.rooms == [], "a clearing has no library room behind it"
        assert len(plan.splines) == len(routed.edges) * T.SPLINES_PER_EDGE_WATER
        for spline in plan.splines:
            assert len(spline) >= 2
            assert all(isinstance(p, tuple) and len(p) == 2 for p in spline)
        present = kinds_present(plan)
        assert CellKind.CORRIDOR in present  # the ridge
        assert CellKind.ROOM in present  # the clearings
        assert CellKind.CLIFF in present  # the offset cliff lines
        assert CellKind.WATER in present  # ashen_ramparts declares a cistern


def test_outdoor_ridge_spline_starts_and_ends_on_the_node_cells():
    for seed, routed, plan in plans(OUTDOOR):
        for i, edge in enumerate(routed.edges):
            ridge = plan.splines[i * T.SPLINES_PER_EDGE_WATER]
            first = (math.floor(ridge[0][0]), math.floor(ridge[0][1]))
            last = (math.floor(ridge[-1][0]), math.floor(ridge[-1][1]))
            assert first == edge.path[0], (seed, edge.a, edge.b)
            assert last == edge.path[-1], (seed, edge.a, edge.b)


def test_outdoor_cliffs_never_eat_the_floor():
    """Scenery only fills what the ridge and the clearings left empty."""
    for seed, routed, plan in plans(OUTDOOR):
        for _edge, _width, _spine, cells in corridor_footprints(
            routed, plan, SeedFields.parse(seed)
        ):
            for cell in cells:
                assert plan.kind(cell) in (CellKind.CORRIDOR, CellKind.ROOM), (
                    f"seed {seed:#x}: ridge cell {cell} was overwritten by "
                    f"{plan.kind(cell).name}"
                )
        for node in routed.nodes.values():
            assert plan.kind(node.cell) is CellKind.ROOM  # its clearing
        # Stage 3 never sets these; stage 5 does.
        assert not kinds_present(plan) & {CellKind.SET_PIECE, CellKind.APPROACH}


def test_a_template_with_no_water_gets_no_banks():
    template = template_from_dict(DRY_RIDGE, source="dry_ridge")
    assert not T.declares_water(template)
    for seed in SEEDS[:25]:
        routed = route(template, seed)
        plan = T.translate(routed, None, seed)  # outdoor needs no library
        assert len(plan.splines) == len(routed.edges) * T.SPLINES_PER_EDGE
        assert CellKind.WATER not in kinds_present(plan)
        entrance = routed.node_of_role(Role.ENTRANCE)
        reached = floor_reachable(plan, entrance.cell)
        assert all(n.cell in reached for n in routed.nodes.values())


def test_declares_water_reads_the_authored_strings():
    assert T.declares_water(load_builtin_template(OUTDOOR))  # the cistern node
    assert not T.declares_water(load_builtin_template(DUNGEON))
    wet = dict(DRY_RIDGE, id="wet_ridge", landmarks=["broken_moat_bridge"])
    assert T.declares_water(template_from_dict(wet, source="wet"))


# --------------------------------------------------------------------------
# The spline maths
# --------------------------------------------------------------------------


def test_smoothing_keeps_the_endpoints_and_averages_the_middle():
    path = [(0, 0), (1, 0), (1, 1), (2, 1)]
    smoothed = T.smooth_centres(path)
    assert smoothed[0] == (0.5, 0.5)
    assert smoothed[-1] == (2.5, 1.5)
    assert smoothed[1] == pytest.approx(((0.5 + 1.5 + 1.5) / 3, (0.5 + 0.5 + 1.5) / 3))
    assert len(smoothed) == len(path)
    assert T.smooth_centres([(3, 4)]) == [(3.5, 4.5)]


def test_the_bezier_passes_through_every_point():
    points = [(0.5, 0.5), (3.5, 0.5), (3.5, 4.5), (7.5, 4.5)]
    curve = T.bezier_spline(points, samples=8)
    assert curve[0] == points[0]
    assert curve[-1] == points[-1]
    assert len(curve) == (len(points) - 1) * 8 + 1
    for i, point in enumerate(points):
        assert curve[i * 8] == pytest.approx(point), f"point {i} is not on the curve"


def test_offsets_sit_at_the_asked_distance():
    points = [(float(x) + 0.5, 4.5) for x in range(10)]  # a straight run east
    left = T.offset_spline(points, 2.0)
    right = T.offset_spline(points, -2.0)
    for base, a, b in zip(points, left, right):
        assert math.hypot(a[0] - base[0], a[1] - base[1]) == pytest.approx(2.0)
        assert math.hypot(b[0] - base[0], b[1] - base[1]) == pytest.approx(2.0)
        assert a[1] != b[1], "the two offsets fall on opposite sides"
    assert T.offset_spline([(1.0, 1.0)], 2.0) == []


def test_rasterise_returns_a_four_connected_chain():
    points = T.bezier_spline(T.smooth_centres([(5, 5), (6, 5), (7, 5), (7, 6), (7, 7)]))
    cells = T.rasterise(points, 48, clamp=True)
    assert cells[0] == (5, 5)
    assert cells[-1] == (7, 7)
    for a, b in zip(cells, cells[1:]):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1, f"{a} and {b} are not adjacent"
    assert len(cells) == len(set(cells))


def test_rasterise_clamps_a_ridge_but_drops_a_stray_cliff():
    grid = 48
    stray = [(-4.0, 5.5), (-3.0, 5.5), (1.5, 5.5)]
    assert T.rasterise(stray, grid, clamp=True)[0] == (MARGIN, 5)
    dropped = T.rasterise(stray, grid)
    assert dropped == [(1, 5)]


# --------------------------------------------------------------------------
# The repair pass
# --------------------------------------------------------------------------


def test_repair_carves_the_missing_link():
    seed = SEEDS[3]
    template = load_builtin_template(DUNGEON)
    routed = route(template, seed)
    plan = T.translate(routed, library(), seed)
    entrance = routed.node_of_role(Role.ENTRANCE)

    # Cut the map in half along a column the first corridor crosses.
    column = routed.edges[0].path[len(routed.edges[0].path) // 2][0]
    node_cells = {n.cell for n in routed.nodes.values()}
    wiped = 0
    for y in range(plan.grid):
        cell = (column, y)
        if plan.is_floor(cell) and cell not in node_cells:
            plan.set_kind(cell, CellKind.EMPTY)
            wiped += 1
    assert wiped, "the cut has to remove something to be a test"

    before = floor_reachable(plan, entrance.cell)
    stranded = [n.id for n in routed.nodes.values() if n.cell not in before]
    assert stranded, "the cut has to strand a node to be a test"

    carved = T.repair_connectivity(plan, routed)
    assert carved > 0
    after = floor_reachable(plan, entrance.cell)
    assert all(n.cell in after for n in routed.nodes.values())
    assert T.repair_connectivity(plan, routed) == 0, "the repair is idempotent"


def test_repair_floors_an_entrance_left_empty():
    seed = SEEDS[7]
    routed = route(load_builtin_template(DUNGEON), seed)
    plan = T.translate(routed, library(), seed)
    entrance = routed.node_of_role(Role.ENTRANCE)
    for cell in [entrance.cell] + [
        (entrance.cell[0] + dx, entrance.cell[1] + dy) for dx, dy in SIDE_DELTA
    ]:
        plan.set_kind(cell, CellKind.EMPTY)
    assert T.repair_connectivity(plan, routed) > 0
    assert plan.is_floor(entrance.cell)
    reached = floor_reachable(plan, entrance.cell)
    assert all(n.cell in reached for n in routed.nodes.values())
