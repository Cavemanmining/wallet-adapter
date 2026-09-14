"""Stage 2: shuffle the shape, place the nodes, route the edges.

Spec: docs/WORLD_BIBLE.md stage 2, "Shuffle shape and route edges".

The stage turns an authored graph template into a :class:`contracts.RoutedLayout`
-- the abstract flow of one map -- in four steps:

1. Read the seed transform: rotation from bits 0-1, mirror from bit 2, and the
   entrance/exit swap from bit 3 (honoured only for U, C and I).  The transform
   is applied to the shape's anchors in normalised space by :mod:`shapes`.
2. Decide which optional edges survive: each is kept with probability 1/2,
   drawn from seed bits 4-7 -- the set-piece field, since an optional set
   piece exists exactly when the optional edge that reaches it does.
3. Place every surviving node at its anchor plus a jitter of up to 3 cells,
   keeping a minimum separation of 6 cells.
4. Route every surviving edge with A*: cost 1 to enter a cell, plus 4 if that
   cell is orthogonally adjacent to a path that has already been routed, which
   is what makes corridors spread out instead of braiding together.

Randomness comes from ``SeedFields.stream``.  The geometry of this stage --
node jitter and the A* paths -- draws on labels beginning ``route`` or
``place`` and is funded by seed bits 8-31; the optional-edge coin flips draw
on ``setpiece`` labels and are funded by bits 4-7.  Nothing this stage does
can perturb tile selection or spawns, and changing bits 4-7 changes which
branches exist without shifting a single routing draw.

Judgement calls the spec did not settle
---------------------------------------
*Separation metric.*  "A minimum separation of 6 cells" is read as Euclidean
distance, compared in squared integer form so it is exact.

*Contested anchors.*  The spec resolves a separation failure by pushing a
jittered node back toward its anchor.  That works only while the anchors
themselves are at least 6 cells apart; ``crypt.json`` deliberately pins both
the boss and the exit to ``shape.end``, where walking back toward the anchor
makes the crowding worse, not better.  So the push is tried first, exactly as
specified, and only if the whole walk fails does the node fall back to a
deterministic outward ring search around the anchor -- the nearest cell, in a
fixed scan order, that satisfies separation and the margin.

*Routing order.*  Edges are routed in template order.  The adjacency penalty
makes the result depend on that order, and an authored order is more
predictable for level designers than a shuffled one; the seed still varies the
result through node jitter (bits 8-31) and through which optional edges
survive (bits 4-7).

*Adjacency penalty.*  The +4 is charged on entering a cell that has a routed
path orthogonally adjacent to it, whether or not the entered cell is itself
part of a path, which is the literal reading of the spec line.

*Keeping the map playable.*  The spec demands the entrance still reach the
exit after optional edges are dropped.  Stage 1 only proves that for dropping
one optional edge at a time, so this stage checks the stronger property
directly -- every non-optional node, plus the exit, reachable from the
entrance -- and reinstates dropped edges in template order until it holds.
"""

from __future__ import annotations

import heapq
from typing import (
    Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Union,
)

from .contracts import (
    SIDES,
    SIDE_DELTA,
    Cell,
    GraphTemplate,
    Role,
    RoutedEdge,
    RoutedLayout,
    RoutedNode,
    TemplateEdge,
    TemplateNode,
    neighbour,
)
from .seed import SeedFields
from .shapes import anchor_cell
from .template import validate_template

__all__ = [
    "MIN_SEPARATION",
    "MAX_JITTER",
    "ADJACENCY_PENALTY",
    "STEP_COST",
    "MARGIN",
    "OPTIONAL_KEEP_CHANCE",
    "route",
    "place_nodes",
    "astar_path",
    "separation_ok",
]

#: Nodes must sit at least this many cells apart (Euclidean).
MIN_SEPARATION = 6
_MIN_SEPARATION_SQ = MIN_SEPARATION * MIN_SEPARATION

#: A node may be jittered this far from its anchor along each axis.
MAX_JITTER = 3

#: Extra cost for entering a cell that touches an already-routed path.
ADJACENCY_PENALTY = 4

#: Base cost of entering any cell.
STEP_COST = 1

#: Cells kept clear along every border; nothing is placed or routed there.
MARGIN = 1

#: Probability that an optional edge survives.
OPTIONAL_KEEP_CHANCE = 0.5


# --------------------------------------------------------------------------
# Small graph helpers
# --------------------------------------------------------------------------


def _adjacency(
    node_ids: Sequence[str], edges: Sequence[TemplateEdge]
) -> Dict[str, List[str]]:
    """Undirected adjacency with sorted neighbour lists, for a stable walk."""
    adj: Dict[str, List[str]] = {n: [] for n in node_ids}
    for edge in edges:
        adj[edge.a].append(edge.b)
        adj[edge.b].append(edge.a)
    for key in adj:
        adj[key].sort()
    return adj


def _reachable(
    entrance: str, node_ids: Sequence[str], edges: Sequence[TemplateEdge]
) -> Set[str]:
    """The ids reachable from ``entrance`` over ``edges``."""
    adj = _adjacency(node_ids, edges)
    seen = {entrance}
    stack = [entrance]
    while stack:
        current = stack.pop()
        for nxt in adj[current]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _entrance_id(template: GraphTemplate) -> str:
    for node in template.nodes:
        if node.role is Role.ENTRANCE:
            return node.id
    # validate_template() has already proved there is exactly one.
    raise ValueError(f"template {template.ref} has no entrance")


def _must_survive(template: GraphTemplate) -> Set[str]:
    """Nodes the finished layout has to contain and connect to the entrance.

    Every non-optional node, plus the exit even if an author marked it
    optional: stage 5 and stage 6 have nothing to work with without one.
    """
    needed = {n.id for n in template.nodes if not n.optional}
    for node in template.nodes:
        if node.role is Role.EXIT:
            needed.add(node.id)
    return needed


# --------------------------------------------------------------------------
# Step 2: which optional edges survive
# --------------------------------------------------------------------------


def _choose_edges(
    template: GraphTemplate, fields: SeedFields
) -> List[TemplateEdge]:
    """Flip a coin for each optional edge, then repair what the flips broke.

    Each optional edge draws from its own labelled stream, so adding an edge to
    a template does not shift the coin flips of the edges around it -- and so
    turning one branch on or off leaves every other edge's draws alone.

    The label is ``setpiece.optional:...``, which ``SeedFields.stream`` routes
    to seed bits 4-7.  That is the field ``seed.py`` documents as deciding
    "which optional set piece is present", and an optional set piece is
    present exactly when the optional edge reaching its node survives.  The
    flip used to be labelled ``route.optional:...`` and drawn from bits 8-31,
    which left bits 4-7 with no observable effect at all and made "same
    layout, different optional room" impossible to ask for.
    """
    kept: List[int] = []
    dropped: List[int] = []
    for i, edge in enumerate(template.edges):
        if not edge.optional:
            kept.append(i)
            continue
        stream = fields.stream(f"setpiece.optional:{edge.a}-{edge.b}")
        if stream.chance(OPTIONAL_KEEP_CHANCE):
            kept.append(i)
        else:
            dropped.append(i)

    entrance = _entrance_id(template)
    node_ids = sorted(n.id for n in template.nodes)
    needed = _must_survive(template)

    while dropped:
        reach = _reachable(
            entrance, node_ids, [template.edges[i] for i in kept]
        )
        if needed <= reach:
            break
        # Reinstate the first dropped edge that reaches something new. The
        # full template graph is connected (stage 1 proved it), so while a
        # needed node is still out of reach some dropped edge must cross the
        # frontier; the loop therefore terminates.
        for slot, i in enumerate(dropped):
            edge = template.edges[i]
            if (edge.a in reach) != (edge.b in reach):
                kept.append(dropped.pop(slot))
                break
        else:  # pragma: no cover - unreachable for a validated template
            kept.extend(dropped)
            dropped = []

    return [template.edges[i] for i in sorted(kept)]


def _live_graph(
    template: GraphTemplate, kept: Sequence[TemplateEdge]
) -> Tuple[List[TemplateNode], List[TemplateEdge]]:
    """Drop optional nodes the surviving edges no longer reach.

    An optional node whose edges were all dropped is omitted from the layout,
    and so is one left stranded on an island of its own: the layout must be a
    single connected graph hanging off the entrance.
    """
    entrance = _entrance_id(template)
    node_ids = sorted(n.id for n in template.nodes)
    live_ids = _reachable(entrance, node_ids, kept)
    nodes = [n for n in template.nodes if n.id in live_ids]
    edges = [e for e in kept if e.a in live_ids and e.b in live_ids]
    return nodes, edges


# --------------------------------------------------------------------------
# Step 3: node placement
# --------------------------------------------------------------------------


def separation_ok(cell: Cell, taken: Iterable[Cell]) -> bool:
    """True when ``cell`` is at least :data:`MIN_SEPARATION` from every other."""
    x, y = cell
    for tx, ty in taken:
        dx, dy = x - tx, y - ty
        if dx * dx + dy * dy < _MIN_SEPARATION_SQ:
            return False
    return True


def _clamp(cell: Cell, grid: int) -> Cell:
    lo, hi = MARGIN, grid - 1 - MARGIN
    return (min(max(cell[0], lo), hi), min(max(cell[1], lo), hi))


def _walk_towards(start: Cell, anchor: Cell) -> Iterator[Cell]:
    """Every cell from ``start`` back to ``anchor``, one step at a time.

    This is the spec's remedy for a jitter that crowded two nodes together:
    push the node back toward its anchor until the separation is satisfied.
    """
    x, y = start
    ax, ay = anchor
    yield (x, y)
    while (x, y) != (ax, ay):
        if x != ax:
            x += 1 if ax > x else -1
        if y != ay:
            y += 1 if ay > y else -1
        yield (x, y)


def _rings(anchor: Cell, grid: int) -> Iterator[Cell]:
    """Cells around ``anchor`` in widening square rings, in a fixed scan order.

    The fallback for anchors two nodes have to share, where walking back toward
    the anchor can never separate them.
    """
    ax, ay = anchor
    lo, hi = MARGIN, grid - 1 - MARGIN
    for radius in range(1, grid):
        ring: List[Cell] = []
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                x, y = ax + dx, ay + dy
                if lo <= x <= hi and lo <= y <= hi:
                    ring.append((x, y))
        yield from ring  # already in (dy, dx) order, so deterministic


def _settle(start: Cell, anchor: Cell, taken: Sequence[Cell], grid: int) -> Cell:
    """Find the cell a node actually occupies, honouring separation."""
    for candidate in _walk_towards(start, anchor):
        if separation_ok(candidate, taken):
            return candidate
    for candidate in _rings(anchor, grid):
        if separation_ok(candidate, taken):
            return candidate
    raise ValueError(
        f"no cell on a {grid}x{grid} grid keeps {MIN_SEPARATION} cells from "
        f"every other node; the template asks for too many nodes"
    )


def place_nodes(
    template: GraphTemplate,
    nodes: Sequence[TemplateNode],
    fields: SeedFields,
) -> Dict[str, Cell]:
    """Place each node at its transformed anchor plus jitter (spec stage 2).

    Nodes are settled in template order, each against the ones already placed,
    so the result depends only on the template and the seed's routing bits.
    """
    grid = template.grid
    placed: Dict[str, Cell] = {}
    taken: List[Cell] = []
    for node in nodes:
        anchor = anchor_cell(
            template.shape,
            node.anchor,
            grid,
            rotation=fields.rotation,
            mirror=fields.mirror,
            swap_ends=fields.swap_ends,
            margin=MARGIN,
        )
        jitter = fields.stream(f"place.jitter:{node.id}")
        dx = jitter.randint(-MAX_JITTER, MAX_JITTER)
        dy = jitter.randint(-MAX_JITTER, MAX_JITTER)
        start = _clamp((anchor[0] + dx, anchor[1] + dy), grid)
        cell = _settle(start, anchor, taken, grid)
        placed[node.id] = cell
        taken.append(cell)
    return placed


# --------------------------------------------------------------------------
# Step 4: routing
# --------------------------------------------------------------------------


def astar_path(
    start: Cell,
    goal: Cell,
    grid: int,
    penalty: Optional[Sequence[Sequence[int]]] = None,
) -> List[Cell]:
    """A* over the grid, four-connected, inclusive of both endpoints.

    Entering a cell costs :data:`STEP_COST` plus ``penalty[y][x]``, which
    stage 2 uses to charge :data:`ADJACENCY_PENALTY` next to corridors that
    have already been routed.  The Manhattan heuristic never overestimates
    because the penalty is never negative.

    Everything is kept inside a :data:`MARGIN` cell border, and ties are broken
    by (cost so far, cell) so the path is a pure function of its inputs.
    """
    lo, hi = MARGIN, grid - 1 - MARGIN
    for cell in (start, goal):
        if not (lo <= cell[0] <= hi and lo <= cell[1] <= hi):
            raise ValueError(f"cell {cell} is outside the {MARGIN}-cell margin")
    if start == goal:
        return [start]

    gx, gy = goal

    def heuristic(cell: Cell) -> int:
        return abs(cell[0] - gx) + abs(cell[1] - gy)

    best: Dict[Cell, int] = {start: 0}
    came: Dict[Cell, Cell] = {}
    heap: List[Tuple[int, int, Cell]] = [(heuristic(start), 0, start)]
    while heap:
        _f, g, cell = heapq.heappop(heap)
        if cell == goal:
            path = [cell]
            while cell in came:
                cell = came[cell]
                path.append(cell)
            path.reverse()
            return path
        if g > best.get(cell, g):
            continue  # a cheaper route to this cell was already expanded
        for side in SIDES:
            dx, dy = SIDE_DELTA[side]
            nxt = (cell[0] + dx, cell[1] + dy)
            if not (lo <= nxt[0] <= hi and lo <= nxt[1] <= hi):
                continue
            step = STEP_COST
            if penalty is not None:
                step += penalty[nxt[1]][nxt[0]]
            ng = g + step
            if ng < best.get(nxt, ng + 1):
                best[nxt] = ng
                came[nxt] = cell
                heapq.heappush(heap, (ng + heuristic(nxt), ng, nxt))
    # The grid has no obstacles, only a margin, so this cannot happen.
    raise ValueError(f"no route from {start} to {goal}")  # pragma: no cover


def _mark_routed(path: Sequence[Cell], penalty: List[List[int]], grid: int) -> None:
    """Charge the adjacency penalty around a path that has just been routed."""
    for cell in path:
        for side in SIDES:
            nx, ny = neighbour(cell, side)
            if 0 <= nx < grid and 0 <= ny < grid:
                penalty[ny][nx] = ADJACENCY_PENALTY


def _route_edges(
    edges: Sequence[TemplateEdge], cells: Mapping[str, Cell], grid: int
) -> List[RoutedEdge]:
    """Route every surviving edge, in template order, spreading corridors out."""
    penalty: List[List[int]] = [[0] * grid for _ in range(grid)]
    routed: List[RoutedEdge] = []
    for edge in edges:
        path = astar_path(cells[edge.a], cells[edge.b], grid, penalty)
        routed.append(RoutedEdge(a=edge.a, b=edge.b, path=path))
        _mark_routed(path, penalty, grid)
    return routed


# --------------------------------------------------------------------------
# The stage
# --------------------------------------------------------------------------


def route(template: GraphTemplate, seed: Union[int, SeedFields]) -> RoutedLayout:
    """Run stage 2 for one template and one seed.

    The template is validated first: every later step assumes stage 1's
    guarantees, in particular that there is exactly one entrance and that the
    graph is connected.

    ``seed`` may be a raw 64-bit int or an already decoded
    :class:`~lucifer_gen.seed.SeedFields`, which is what the other stages
    accept, so a pipeline can decode the seed once and hand the same fields to
    every stage.
    """
    validate_template(template)
    fields = seed if isinstance(seed, SeedFields) else SeedFields.parse(seed)

    kept = _choose_edges(template, fields)
    nodes, edges = _live_graph(template, kept)
    cells = place_nodes(template, nodes, fields)

    return RoutedLayout(
        seed=fields.raw,
        template=template,
        grid=template.grid,
        nodes={
            node.id: RoutedNode(
                id=node.id,
                role=node.role,
                cell=cells[node.id],
                set_piece=node.set_piece,
            )
            for node in nodes
        },
        edges=_route_edges(edges, cells, template.grid),
    )
