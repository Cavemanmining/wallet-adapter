"""Stage 3: translate a routed layout into terrain.

Spec: docs/WORLD_BIBLE.md stage 3, "Translate edges into terrain".

Stage 2 hands over an abstract flow: node cells and, between them, four
connected A* paths.  This stage decides which cells are *floor*, and of what
kind, and it does so differently for the two tile classes:

Dungeon
    Every routed edge is carved as a corridor one or two cells wide.  Every
    node becomes a room taken from :class:`~lucifer_gen.rooms.RoomLibrary`,
    filtered to rooms whose doorway sockets face the directions the incident
    edges leave in, placed centred on the node cell and **never scaled**.
    Corridor cells are :data:`CellKind.CORRIDOR`, room cells
    :data:`CellKind.ROOM`; where the two meet the room wins, which is why the
    corridors are carved first and the rooms stamped over them.

Outdoor
    Every routed edge becomes a cubic Bezier spline through the cell centres,
    smoothed over a three-cell window, rasterised into a walkable ridge one or
    two cells wide.  Fixed-distance perpendicular offsets of that spline give
    the cliff lines on both sides, and, for a template that declares water, a
    bank line further out again.  Nodes become open clearings: a small
    rectangle of :data:`CellKind.ROOM` centred on the node cell.

Nothing here reserves anything for the set pieces; stage 5 overwrites what it
needs and re-runs the filler around itself.

Ordering, and what wins
-----------------------
The floor is laid down before the scenery, and scenery is only ever written
into cells that are still empty.  So the passes run: corridors/ridges, then
rooms/clearings, then cliffs, then banks.  That makes the floor set a pure
function of the routed paths -- rasterising a cliff can never eat a corridor
-- which is what lets the connectivity guarantee below be structural rather
than lucky.

Randomness
----------
Every draw comes from ``SeedFields.stream`` on a label beginning ``route`` or
``place``, so stage 3 is funded by the same seed field as stage 2 (bits 8-31)
and cannot perturb tile selection or spawns.  See the judgement calls below.

Judgement calls the spec did not settle
---------------------------------------
*Which seed field pays for stage 3.*  Section 02 gives stage 3 no field of its
own: bits 8-31 are "node jitter and edge routing", bits 32-63 are "tile
selection, filler, spawns".  Corridor widths and room choices are geometry
that follows the routed paths, so they draw on ``route.``/``place.`` labels and
move with the routing bits.  Re-rolling the tile field therefore re-tiles a map
without moving a wall, which is the behaviour the field split is there for.

*Doorway alignment.*  The spec filters rooms by socket direction, and this
stage honours that, but it does not walk the corridor over to the exact socket
cell: the room interior is all floor, so the corridor meets the room whatever
offset the socket sits at, and stage 5 is the stage that snaps doorways.

*Room size budget.*  Nodes are only guaranteed six cells apart, so two 6x5
rooms centred on neighbouring nodes could overlap.  Each room is therefore
capped at the Chebyshev distance to the nearest other node, which is exactly
the condition that makes two centred footprints disjoint, and floored at 2x2
so a candidate always exists.  If even that comes back empty the budget is
dropped rather than failing the stage: an overlap is survivable, a map that
will not generate is not.

*Margin.*  Stage 2 keeps everything inside a one-cell border
(:data:`~lucifer_gen.route.MARGIN`).  Stage 3 keeps the same border: widening,
smoothing overshoot, rooms and cliffs are all held inside it, so the outer ring
is always untouched and stage 4 can fill it with impassable filler.  Rooms are
*shifted* back inside, never resized.

*Outdoor clearings are not recorded in* ``TerrainPlan.rooms``.  That list
carries :class:`RoomPlacement` records whose ``room_id`` names a library room;
an outdoor clearing has no library room behind it, and a later stage that
looked its id up would fail.  Clearings are marked in ``kinds`` only.

*Water.*  No field on :class:`~lucifer_gen.contracts.GraphTemplate` says
"this map has water", so a template declares water by naming it: the tileset,
a landmark, a node id or a set piece containing one of :data:`WATER_TOKENS`.
``ashen_ramparts`` declares water through its ``cistern`` node.

*Spline bookkeeping.*  ``TerrainPlan.splines`` is an unlabelled list, so the
splines are appended in a fixed order -- per routed edge, in layout order:
ridge, cliff A, cliff B, and then bank A, bank B when the template declares
water.  :data:`SPLINES_PER_EDGE` and :data:`SPLINES_PER_EDGE_WATER` give the
stride.  Points are in cell coordinates, where the centre of cell ``(x, y)``
is ``(x + 0.5, y + 0.5)``.
"""

from __future__ import annotations

import heapq
import math
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .contracts import (
    SIDES,
    SIDE_DELTA,
    Cell,
    CellKind,
    GraphTemplate,
    Role,
    RoomPlacement,
    RoutedEdge,
    RoutedLayout,
    TerrainPlan,
    TileClass,
)
from .rooms import NoFittingRoom, Room, RoomLibrary
from .route import MARGIN
from .seed import SeedFields, Stream

__all__ = [
    "MIN_CORRIDOR_WIDTH",
    "MAX_CORRIDOR_WIDTH",
    "SMOOTH_WINDOW",
    "BEZIER_SAMPLES",
    "CLIFF_OFFSET_M",
    "BANK_OFFSET_M",
    "CLEARING_MIN",
    "CLEARING_MAX",
    "WATER_TOKENS",
    "SPLINES_PER_EDGE",
    "SPLINES_PER_EDGE_WATER",
    "Point",
    "translate",
    "declares_water",
    "incident_sides",
    "corridor_profile",
    "corridor_cells",
    "smooth_centres",
    "bezier_spline",
    "offset_spline",
    "rasterise",
    "repair_connectivity",
]

#: A corridor, or an outdoor ridge, is this many cells wide.
MIN_CORRIDOR_WIDTH = 1
MAX_CORRIDOR_WIDTH = 2

#: Spec stage 3: the spline is "smoothed over a 3-cell window".
SMOOTH_WINDOW = 3

#: Samples taken along each cubic Bezier segment when rasterising.  Eight is
#: comfortably more than one per cell for a segment that spans one cell, so no
#: cell the curve crosses is stepped over.
BEZIER_SAMPLES = 8

#: Fixed offsets, in metres, of the cliff and bank splines from the ridge.
CLIFF_OFFSET_M = 8.0
BANK_OFFSET_M = 12.0

#: An outdoor clearing is a rectangle between these sizes, per axis.
CLEARING_MIN = 3
CLEARING_MAX = 5

#: Lower-case substrings that make a template's authored strings declare water.
WATER_TOKENS = (
    "water",
    "moat",
    "river",
    "lake",
    "canal",
    "cistern",
    "aqueduct",
    "reservoir",
    "lagoon",
    "marsh",
    "flood",
    "pond",
    "creek",
)

#: How many splines an outdoor edge contributes to ``TerrainPlan.splines``.
SPLINES_PER_EDGE = 3  # ridge, cliff A, cliff B
SPLINES_PER_EDGE_WATER = 5  # ... plus bank A, bank B

Point = Tuple[float, float]

#: Reverse of ``SIDE_DELTA``: the side a unit step travels through.
_SIDE_OF: Dict[Cell, int] = {SIDE_DELTA[s]: s for s in SIDES}


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _fields(seed) -> SeedFields:
    """Accept a raw seed or an already decoded :class:`SeedFields`."""
    if isinstance(seed, SeedFields):
        return seed
    return SeedFields.parse(int(seed))


def _band(grid: int) -> Tuple[int, int]:
    """The inclusive cell range stage 3 may write to, matching stage 2."""
    return MARGIN, grid - 1 - MARGIN


def _in_band(cell: Cell, grid: int) -> bool:
    lo, hi = _band(grid)
    return lo <= cell[0] <= hi and lo <= cell[1] <= hi


def _clamp_to_band(cell: Cell, grid: int) -> Cell:
    lo, hi = _band(grid)
    return (min(max(cell[0], lo), hi), min(max(cell[1], lo), hi))


def _centre(cell: Cell) -> Point:
    """The centre of a cell, in cell coordinates."""
    return (cell[0] + 0.5, cell[1] + 0.5)


def _walk(a: Cell, b: Cell) -> List[Cell]:
    """Four-connected cells from ``a`` to ``b``, x first then y, inclusive.

    Used to join two cells a rasterised curve visited in sequence, so the
    result is a chain with no diagonal gaps in it.
    """
    cells = [a]
    x, y = a
    while x != b[0]:
        x += 1 if b[0] > x else -1
        cells.append((x, y))
    while y != b[1]:
        y += 1 if b[1] > y else -1
        cells.append((x, y))
    return cells


def _entrance_node(routed: RoutedLayout):
    """The entrance, or -- defensively -- the first node by id."""
    node = routed.node_of_role(Role.ENTRANCE)
    if node is not None:
        return node
    return routed.nodes[sorted(routed.nodes)[0]]


# --------------------------------------------------------------------------
# Template questions
# --------------------------------------------------------------------------


def declares_water(template: GraphTemplate) -> bool:
    """True when the template names water anywhere an author could name it.

    See the module docstring: there is no dedicated field, so the tileset, the
    landmarks, the node ids and the set piece ids are searched for one of
    :data:`WATER_TOKENS`.
    """
    haystack = [template.tileset, *template.landmarks]
    for node in template.nodes:
        haystack.append(node.id)
        if node.set_piece:
            haystack.append(node.set_piece)
    blob = " ".join(haystack).lower()
    return any(token in blob for token in WATER_TOKENS)


def incident_sides(routed: RoutedLayout, node_id: str) -> Set[int]:
    """Which sides of a node its routed edges leave through.

    Spec stage 3: the direction of an incident edge is read from the first
    step of the routed path *leaving* the node, which is the last step of the
    stored path when the node is the far end of the edge.
    """
    sides: Set[int] = set()
    for edge in routed.incident_edges(node_id):
        side = _leaving_side(edge, node_id)
        if side is not None:
            sides.add(side)
    return sides


def _leaving_side(edge: RoutedEdge, node_id: str) -> Optional[int]:
    path = edge.path
    if len(path) < 2:
        return None
    if edge.a == node_id:
        here, there = path[0], path[1]
    elif edge.b == node_id:
        here, there = path[-1], path[-2]
    else:
        return None
    return _SIDE_OF.get((there[0] - here[0], there[1] - here[1]))


# --------------------------------------------------------------------------
# Corridors (and outdoor ridges, which widen the same way)
# --------------------------------------------------------------------------


def corridor_profile(fields: SeedFields, edge: RoutedEdge) -> Tuple[int, bool]:
    """The width of one edge's corridor, and which side a width of 2 widens to.

    Drawn from a stream labelled per edge, so adding an edge to a template
    cannot shift the widths of the edges around it.
    """
    stream = fields.stream(f"route.width:{edge.a}-{edge.b}")
    width = stream.randint(MIN_CORRIDOR_WIDTH, MAX_CORRIDOR_WIDTH)
    widen_first = stream.chance(0.5)
    return width, widen_first


def corridor_cells(
    path: Sequence[Cell], width: int, widen_first: bool, grid: int
) -> List[Cell]:
    """Every cell a corridor of ``width`` along ``path`` occupies.

    A width of 2 adds, for each step, the cell perpendicular to the local
    direction of travel -- the direction of the next step, or of the previous
    one at the far end.  ``widen_first`` picks which of the two perpendiculars
    is tried first; if that cell falls outside the margin the other side is
    used, and if both do the corridor stays one cell wide there.  So a width-2
    corridor never leaves the grid, it only narrows against the border.

    The returned list is in path order, deduplicated, and contains only cells
    inside the margin.
    """
    out: List[Cell] = []
    seen: Set[Cell] = set()

    def add(cell: Cell) -> None:
        if cell not in seen and _in_band(cell, grid):
            seen.add(cell)
            out.append(cell)

    n = len(path)
    for i, cell in enumerate(path):
        add(cell)
        if width < 2 or n < 2:
            continue
        if i + 1 < n:  # the direction of the step about to be taken
            ahead = path[i + 1]
            dx, dy = ahead[0] - cell[0], ahead[1] - cell[1]
        else:  # at the far end, the direction of the step just taken
            behind = path[i - 1]
            dx, dy = cell[0] - behind[0], cell[1] - behind[1]
        if dx == 0 and dy == 0:
            continue
        first = (-dy, dx)
        second = (dy, -dx)
        if not widen_first:
            first, second = second, first
        for offset in (first, second):
            candidate = (cell[0] + offset[0], cell[1] + offset[1])
            if _in_band(candidate, grid):
                add(candidate)
                break
    return out


# --------------------------------------------------------------------------
# Splines (outdoor)
# --------------------------------------------------------------------------


def smooth_centres(path: Sequence[Cell], window: int = SMOOTH_WINDOW) -> List[Point]:
    """Cell centres along ``path``, smoothed with a moving window.

    Spec stage 3 asks for a 3-cell window.  The two endpoints are left where
    they are, so the spline still begins and ends on the node cells.
    """
    points = [_centre(c) for c in path]
    if window < 3 or len(points) < 3:
        return points
    half = window // 2
    out: List[Point] = []
    for i, point in enumerate(points):
        if i < half or i >= len(points) - half:
            out.append(point)
            continue
        lo, hi = i - half, i + half + 1
        chunk = points[lo:hi]
        out.append(
            (
                sum(p[0] for p in chunk) / len(chunk),
                sum(p[1] for p in chunk) / len(chunk),
            )
        )
    return out


def bezier_spline(points: Sequence[Point], samples: int = BEZIER_SAMPLES) -> List[Point]:
    """A cubic Bezier spline that passes through every one of ``points``.

    Each segment's two inner control points are the Catmull-Rom tangents,
    ``p1 + (p2 - p0) / 6`` and ``p2 - (p3 - p1) / 6``, which is the standard
    way of writing an interpolating curve as cubic Beziers: the curve touches
    the cell centres rather than merely being pulled toward them.
    """
    if len(points) < 2:
        return list(points)
    out: List[Point] = []
    last = len(points) - 1
    for i in range(last):
        p0 = points[i - 1] if i > 0 else points[i]
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2] if i + 2 <= last else points[i + 1]
        b1 = (p1[0] + (p2[0] - p0[0]) / 6.0, p1[1] + (p2[1] - p0[1]) / 6.0)
        b2 = (p2[0] - (p3[0] - p1[0]) / 6.0, p2[1] - (p3[1] - p1[1]) / 6.0)
        for k in range(samples):
            t = k / samples
            u = 1.0 - t
            w0 = u * u * u
            w1 = 3 * u * u * t
            w2 = 3 * u * t * t
            w3 = t * t * t
            out.append(
                (
                    w0 * p1[0] + w1 * b1[0] + w2 * b2[0] + w3 * p2[0],
                    w0 * p1[1] + w1 * b1[1] + w2 * b2[1] + w3 * p2[1],
                )
            )
    out.append(points[-1])
    return out


def offset_spline(points: Sequence[Point], distance: float) -> List[Point]:
    """``points`` pushed ``distance`` cells along their left-hand normal.

    A negative distance gives the other side.  The normal is taken from a
    central difference of the neighbouring points; where consecutive points
    coincide the previous normal is carried forward, so a stalled tangent
    never produces a spike.
    """
    if len(points) < 2:
        return []
    out: List[Point] = []
    normal = (0.0, 0.0)
    for i, point in enumerate(points):
        ahead = points[min(i + 1, len(points) - 1)]
        behind = points[max(i - 1, 0)]
        tx, ty = ahead[0] - behind[0], ahead[1] - behind[1]
        length = math.hypot(tx, ty)
        if length > 1e-9:
            normal = (-ty / length, tx / length)
        out.append((point[0] + normal[0] * distance, point[1] + normal[1] * distance))
    return out


def rasterise(points: Sequence[Point], grid: int, *, clamp: bool = False) -> List[Cell]:
    """The cells a polyline passes through, joined into a 4-connected chain.

    With ``clamp`` the points are pulled back inside the margin, which is what
    a walkable ridge wants: the chain stays unbroken even where the smoothed
    curve overshoots the border.  Without it, cells outside the margin are
    dropped and no join is drawn across the gap, which is what a cliff or bank
    wants: an offset that leaves the map should stop, not smear along the edge.
    """
    out: List[Cell] = []
    seen: Set[Cell] = set()
    previous: Optional[Cell] = None
    for px, py in points:
        cell = (int(math.floor(px)), int(math.floor(py)))
        if clamp:
            cell = _clamp_to_band(cell, grid)
        if cell == previous:
            continue
        if previous is not None and _in_band(previous, grid) and _in_band(cell, grid):
            step_cells = _walk(previous, cell)[1:]
        elif _in_band(cell, grid):
            step_cells = [cell]
        else:
            step_cells = []
        for c in step_cells:
            if c not in seen:
                seen.add(c)
                out.append(c)
        previous = cell
    return out


# --------------------------------------------------------------------------
# Rooms and clearings
# --------------------------------------------------------------------------


def _footprint_budget(node_id: str, cells: Dict[str, Cell], grid: int) -> int:
    """The largest footprint that cannot reach a neighbouring node.

    Two footprints centred ``d`` cells apart in Chebyshev distance, each at
    most ``d`` cells across, are disjoint: their extents along the axis that
    realises ``d`` sum to ``d - 1``.  Floored at 2 because the library's
    smallest room is 2x2, and capped by the width of the margin band.
    """
    lo, hi = _band(grid)
    span = hi - lo + 1
    here = cells[node_id]
    distances = [
        max(abs(here[0] - other[0]), abs(here[1] - other[1]))
        for key, other in cells.items()
        if key != node_id
    ]
    budget = min(distances) if distances else span
    return max(2, min(budget, span))


def _place_room(
    library: RoomLibrary, room: Room, node_id: str, node_cell: Cell, grid: int
) -> RoomPlacement:
    """Centre a room on its node and slide it inside the margin.

    The library already centres and clamps to the grid; this shifts it one
    further step to respect stage 2's border.  It is a translation, never a
    resize: ``w`` and ``h`` come straight off the library room.
    """
    placement = library.place(room, node_id, node_cell, grid)
    lo, hi = _band(grid)
    if room.w <= hi - lo + 1 and room.h <= hi - lo + 1:
        ox = min(max(placement.origin[0], lo), hi - room.w + 1)
        oy = min(max(placement.origin[1], lo), hi - room.h + 1)
        placement = RoomPlacement(
            room_id=room.id,
            node_id=node_id,
            origin=(ox, oy),
            w=room.w,
            h=room.h,
        )
    return placement


def _choose_room(
    library: RoomLibrary, sides: Set[int], stream: Stream, budget: int
) -> Room:
    """A room serving ``sides`` within ``budget``, or the best the library has.

    Dropping the budget rather than failing keeps stage 3 total; see the
    module docstring.
    """
    try:
        return library.choose(sides, stream, budget, budget)
    except NoFittingRoom:
        return library.choose(sides, stream, None, None)


# --------------------------------------------------------------------------
# The two classes
# --------------------------------------------------------------------------


def _translate_dungeon(
    plan: TerrainPlan,
    routed: RoutedLayout,
    library: RoomLibrary,
    fields: SeedFields,
) -> None:
    """Corridors first, then rooms stamped over them (spec stage 3, dungeon)."""
    grid = routed.grid
    for edge in routed.edges:
        width, widen_first = corridor_profile(fields, edge)
        for cell in corridor_cells(edge.path, width, widen_first, grid):
            plan.set_kind(cell, CellKind.CORRIDOR)

    cells = {node_id: node.cell for node_id, node in routed.nodes.items()}
    for node_id in sorted(routed.nodes):
        node = routed.nodes[node_id]
        budget = _footprint_budget(node_id, cells, grid)
        room = _choose_room(
            library,
            incident_sides(routed, node_id),
            fields.stream(f"place.room:{node_id}"),
            budget,
        )
        placement = _place_room(library, room, node_id, node.cell, grid)
        plan.rooms.append(placement)
        for cell in placement.cells():
            plan.set_kind(cell, CellKind.ROOM)


def _clearing_cells(
    node_id: str, node_cell: Cell, cells: Dict[str, Cell], grid: int, fields: SeedFields
) -> List[Cell]:
    """A small rectangle centred on an outdoor node (spec stage 3, outdoor)."""
    stream = fields.stream(f"place.clearing:{node_id}")
    budget = _footprint_budget(node_id, cells, grid)
    w = min(stream.randint(CLEARING_MIN, CLEARING_MAX), budget)
    h = min(stream.randint(CLEARING_MIN, CLEARING_MAX), budget)
    lo, hi = _band(grid)
    ox = min(max(node_cell[0] - (w - 1) // 2, lo), hi - w + 1)
    oy = min(max(node_cell[1] - (h - 1) // 2, lo), hi - h + 1)
    return [(ox + dx, oy + dy) for dy in range(h) for dx in range(w)]


def _translate_outdoor(
    plan: TerrainPlan, routed: RoutedLayout, fields: SeedFields
) -> None:
    """Ridges, clearings, then cliffs and banks (spec stage 3, outdoor)."""
    grid = routed.grid
    template = routed.template
    cell_m = template.cell_m if template.cell_m > 0 else 4.0
    cliff_distance = CLIFF_OFFSET_M / cell_m
    bank_distance = BANK_OFFSET_M / cell_m
    water = declares_water(template)

    # One pass to build the geometry, so ``plan.splines`` is in edge order.
    ridges: List[Tuple[List[Cell], int, bool]] = []
    cliffs: List[List[Point]] = []
    banks: List[List[Point]] = []
    for edge in routed.edges:
        width, widen_first = corridor_profile(fields, edge)
        spline = bezier_spline(smooth_centres(edge.path))
        chain = rasterise(spline, grid, clamp=True)
        ridges.append((chain, width, widen_first))
        plan.splines.append(spline)
        for sign in (1.0, -1.0):
            cliff = offset_spline(spline, sign * cliff_distance)
            cliffs.append(cliff)
            plan.splines.append(cliff)
        if water:
            for sign in (1.0, -1.0):
                bank = offset_spline(spline, sign * bank_distance)
                banks.append(bank)
                plan.splines.append(bank)

    for chain, width, widen_first in ridges:
        for cell in corridor_cells(chain, width, widen_first, grid):
            plan.set_kind(cell, CellKind.CORRIDOR)

    cells = {node_id: node.cell for node_id, node in routed.nodes.items()}
    for node_id in sorted(routed.nodes):
        for cell in _clearing_cells(
            node_id, routed.nodes[node_id].cell, cells, grid, fields
        ):
            plan.set_kind(cell, CellKind.ROOM)

    # Scenery only ever fills cells the floor did not claim.
    for points, kind in (
        *((c, CellKind.CLIFF) for c in cliffs),
        *((b, CellKind.WATER) for b in banks),
    ):
        for cell in rasterise(points, grid):
            if plan.kind(cell) is CellKind.EMPTY:
                plan.set_kind(cell, kind)


# --------------------------------------------------------------------------
# Connectivity repair
# --------------------------------------------------------------------------


def _flood(plan: TerrainPlan, start: Cell) -> Set[Cell]:
    """Every floor cell 4-connected to ``start``."""
    if not plan.is_floor(start):
        return set()
    seen = {start}
    stack = [start]
    while stack:
        x, y = stack.pop()
        for dx, dy in SIDE_DELTA:
            nxt = (x + dx, y + dy)
            if nxt not in seen and plan.is_floor(nxt):
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _cheapest_link(plan: TerrainPlan, sources: Set[Cell], goal: Cell) -> List[Cell]:
    """The path to ``goal`` that carves the fewest new cells.

    A uniform-cost search out of the reached floor: stepping onto floor is
    free, stepping onto anything else costs one cell of carving.  Ties break on
    the cell coordinates, so the repair is a pure function of the plan.
    """
    grid = plan.grid
    allowed_band = _in_band(goal, grid)
    heap: List[Tuple[int, Cell]] = []
    best: Dict[Cell, int] = {}
    came: Dict[Cell, Cell] = {}
    for cell in sorted(sources):
        best[cell] = 0
        heap.append((0, cell))
    heapq.heapify(heap)
    while heap:
        cost, cell = heapq.heappop(heap)
        if cost > best.get(cell, cost):
            continue
        if cell == goal:
            path = [cell]
            while cell in came:
                cell = came[cell]
                path.append(cell)
            path.reverse()
            return path
        for dx, dy in SIDE_DELTA:
            nxt = (cell[0] + dx, cell[1] + dy)
            if not plan.inside(nxt):
                continue
            if allowed_band and not _in_band(nxt, grid) and nxt != goal:
                continue
            step = 0 if plan.is_floor(nxt) else 1
            ncost = cost + step
            if ncost < best.get(nxt, ncost + 1):
                best[nxt] = ncost
                came[nxt] = cell
                heapq.heappush(heap, (ncost, nxt))
    return []


def repair_connectivity(plan: TerrainPlan, routed: RoutedLayout) -> int:
    """Carve the shortest missing link to any node the floor does not reach.

    Spec stage 3 requires the floor to be 4-connected from the entrance to
    every other node.  Both classes are built to satisfy that structurally --
    a corridor is a connected chain and a room always covers its node cell --
    so this pass is a safety net for rasterisation surprises.  It returns the
    number of cells it had to carve, which is 0 on a healthy plan.
    """
    entrance = _entrance_node(routed)
    carved = 0
    if not plan.is_floor(entrance.cell):
        plan.set_kind(entrance.cell, CellKind.CORRIDOR)
        carved += 1
    reached = _flood(plan, entrance.cell)
    for node_id in sorted(routed.nodes):
        cell = routed.nodes[node_id].cell
        if cell in reached:
            continue
        for step in _cheapest_link(plan, reached, cell):
            if not plan.is_floor(step):
                plan.set_kind(step, CellKind.CORRIDOR)
                carved += 1
        reached = _flood(plan, entrance.cell)
    return carved


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def translate(
    routed: RoutedLayout, rooms: Optional[RoomLibrary], seed: int
) -> TerrainPlan:
    """Run stage 3 for one routed layout.

    ``rooms`` is required for the dungeon class and ignored by the outdoor
    one.  A template whose class is ``BOTH`` is translated as a dungeon: it
    has a room library to hand, and rooms are the richer of the two readings.
    """
    fields = _fields(seed)
    plan = TerrainPlan.blank(routed.grid)
    if routed.template.tile_class is TileClass.OUTDOOR:
        _translate_outdoor(plan, routed, fields)
    else:
        if rooms is None:
            raise ValueError(
                f"template {routed.template.ref} is {routed.template.tile_class.value}; "
                "stage 3 needs a room library to place its rooms"
            )
        _translate_dungeon(plan, routed, rooms, fields)
    repair_connectivity(plan, routed)
    return plan
