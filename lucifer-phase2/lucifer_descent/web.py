"""Web generation for the Descent: one spider-web node graph per profile.

Spec: docs/WORLD_BIBLE.md section 03 -- "the web is a planar graph generated
once per profile from a profile seed", "edges are never removed", "tier =
min(15, graph distance from the origin)", "tier 15 nodes ring the outer edge
and are the only ones that can be Pinnacle arenas or carry a Pinnacle glyph".

Shape
-----
The web is built as a spider web so that planarity and the tier rule hold by
construction rather than by search:

* the origin (id 0) sits at tier 0 in the centre;
* ring ``r`` (1..rings) holds ``6 + 2r`` nodes at radius ``RING_SPACING * r``,
  evenly spaced with a per-ring phase and a small angular jitter;
* every ring node is joined to its two angular neighbours on the same ring
  and to the single nearest node (by angle) on ring ``r - 1``; ring 1 joins
  the origin.

Edges only ever join a ring to itself or to the ring just inside, and every
node keeps an inner link, so a node on ring ``r`` is at graph distance
exactly ``r`` from the origin, which is its tier.  Same-ring edges are then
thinned at random for variety.  That cannot break the tier rule or
connectivity, because the inner links are never dropped and they alone form
a spanning tree rooted at the origin; the thinning pass still checks both
outright rather than trusting the argument.

Why the drawing is planar
-------------------------
With ``JITTER_FRACTION`` at most 0.1 of a ring's spacing, the angular gap
between neighbours on any ring stays within 0.8 and 1.2 spacings.  Seen from
the origin, each inner link then covers an arc strictly inside the arc
between its node's two same-ring neighbours: the nearest inner node is at
most 0.6 inner spacings away, which is less than 0.8 outer spacings for
every ring size the web uses (``6 + 2r >= 10`` once there is an inner ring
other than the origin).  Same-ring chords cover disjoint arcs, inner links
cover arcs nested between them, a chord on ring ``r`` never dips below
radius ``r - 1`` and a link from ring ``r`` to ``r - 1`` never does either,
so no two edges that do not share a node can meet.  :func:`is_planar_layout`
checks the finished drawing anyway, and the gate runs it over many seeds.

Randomness
----------
Every draw comes from a :class:`lucifer_gen.seed.Stream`.  The labels used
here start with ``web.``, which :meth:`SeedFields.stream` does not recognise,
so each stream is fed the whole 64-bit profile seed.  That is intended: the
field split in ``seed.py`` describes *map* seeds and a profile seed has no
such structure.  Each stage still has its own label, so the mechanic roll
cannot shift the edge thinning and vice versa.
"""

from __future__ import annotations

import math
import sys
from collections import deque
from typing import Dict, List, Optional, Sequence, Set, Tuple

from lucifer_descent.contracts import (
    MAX_TIER,
    Mechanic,
    Pinnacle,
    Web,
    WebEdge,
    WebNode,
)
from lucifer_gen.seed import SeedFields, Stream

# --------------------------------------------------------------------------
# Tunables. Each one is a judgement call the spec leaves open; see the report.
# --------------------------------------------------------------------------

#: Layout units between consecutive rings, so ring ``r`` sits at radius
#: ``RING_SPACING * r``. Purely cosmetic; the table view scales the drawing.
RING_SPACING = 10.0

#: Angular jitter as a fraction of the ring's even spacing, in each direction.
#: Must stay at or below 0.1 for the planarity argument in the module
#: docstring to hold.
JITTER_FRACTION = 0.1

#: Chance that any one same-ring edge is dropped for variety.
DROP_CHANCE = 0.25

#: Chance that a node carries a mechanic.
MECHANIC_CHANCE = 0.35

#: Glyph-bearing (fragment source) nodes placed on the outer ring per Pinnacle.
GLYPHS_PER_PINNACLE = 4

#: No node may end the thinning pass with fewer edges than this.
MIN_DEGREE = 2

#: Fewest rings a web may have: the outer ring must hold two arenas plus
#: ``2 * GLYPHS_PER_PINNACLE`` glyph nodes, and ring 2 (10 nodes) is the
#: first that can.
MIN_RINGS = 2

#: Mechanics in declaration order; ``Stream.choice`` picks uniformly.
MECHANICS: Tuple[Mechanic, ...] = tuple(Mechanic)

#: Pinnacles in declaration order: the Arbiter of Cinders, then the Blind
#: Monolith.
PINNACLES: Tuple[Pinnacle, ...] = tuple(Pinnacle)

#: Template table keyed by tier band: ``(low_tier, high_tier, names)``.
#: A band with several names alternates them around the ring by
#: ``ring_index``. Spec: the two shipped generator templates are ``crypt``
#: and ``ashen_ramparts``; low tiers are crypts, high tiers ramparts, and the
#: middle band mixes them.
TEMPLATE_BANDS: Tuple[Tuple[int, int, Tuple[str, ...]], ...] = (
    (0, 4, ("crypt",)),
    (5, 10, ("crypt", "ashen_ramparts")),
    (11, MAX_TIER, ("ashen_ramparts",)),
)

Point = Tuple[float, float]

_TWO_PI = 2.0 * math.pi

#: Tolerance for the orientation tests in :func:`segments_cross`. Layout
#: coordinates are rounded to three decimals and span a few hundred units, so
#: genuine crossings are many orders of magnitude clearer than this.
_EPS = 1e-9


# --------------------------------------------------------------------------
# Ring bookkeeping
# --------------------------------------------------------------------------


def ring_size(ring: int) -> int:
    """Nodes on a ring. Spec: ring ``r`` holds ``6 + 2r``; the origin is one.

    Ring 0 is the origin alone.
    """
    if ring < 0:
        raise ValueError(f"ring out of range: {ring}")
    if ring == 0:
        return 1
    return 6 + 2 * ring


def first_id_of_ring(ring: int) -> int:
    """Id of ``ring_index`` 0 on ``ring``.

    Ids are dense integers assigned in ring order, origin first, so the
    first id on ring ``r`` is one plus the size of rings 1 to ``r - 1``.
    """
    if ring < 0:
        raise ValueError(f"ring out of range: {ring}")
    return sum(ring_size(r) for r in range(ring))


def node_count(rings: int) -> int:
    """Total nodes in a web with ``rings`` rings, origin included."""
    return first_id_of_ring(rings + 1)


def template_for(tier: int, ring_index: int) -> str:
    """The generator template for a node, from :data:`TEMPLATE_BANDS`.

    Deterministic and stream-free: two profiles with nodes at the same tier
    and ring position get the same template. The variety comes from the
    Sigil's seed at portal-open time, not from here.
    """
    for low, high, names in TEMPLATE_BANDS:
        if low <= tier <= high:
            return names[ring_index % len(names)]
    raise ValueError(f"no template band covers tier {tier}")


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------


def generate_web(profile_seed: int, rings: int = MAX_TIER) -> Web:
    """Build the one web a profile owns from its profile seed.

    Spec: "The web is a planar graph generated once per profile from a
    profile seed", "tier = min(15, graph distance from the origin)", "tier 15
    nodes ring the outer edge".

    ``rings`` is the number of rings around the origin; the production value
    is ``MAX_TIER`` so that the outer ring is tier 15. Smaller webs are
    allowed for tests, in which case the arenas and glyphs still sit on the
    outermost ring, whose tier is then ``rings``. ``rings`` above ``MAX_TIER``
    is refused: tier would be clamped and stop matching the ring.

    The result is fully determined by ``profile_seed`` (masked to 64 bits)
    and ``rings``; ``Web.profile_seed`` holds the masked value so that
    ``generate_web(web.profile_seed)`` reproduces ``web``.
    """
    if not isinstance(profile_seed, int) or isinstance(profile_seed, bool):
        raise TypeError(f"profile_seed must be an int, got {type(profile_seed).__name__}")
    if not MIN_RINGS <= rings <= MAX_TIER:
        raise ValueError(f"rings must be between {MIN_RINGS} and {MAX_TIER}, got {rings}")

    fields = SeedFields.parse(profile_seed)

    # Stage A: polar positions. angles[r][k] for ring r >= 1; the origin has
    # no angle.
    angles: Dict[int, List[float]] = {}
    for ring in range(1, rings + 1):
        angles[ring] = _ring_angles(fields.stream(f"web.angles:ring{ring}"), ring_size(ring))

    # Stage B: the full edge set, before thinning.
    adjacency: Dict[int, Set[int]] = {i: set() for i in range(node_count(rings))}
    same_ring_edges: List[Tuple[int, int]] = []   # in ring order, for thinning
    for ring in range(1, rings + 1):
        base = first_id_of_ring(ring)
        count = ring_size(ring)
        for k in range(count):
            node_id = base + k
            # Spec: "every node on ring r gets an edge to its two angular
            # neighbours on the same ring".
            right = base + (k + 1) % count
            same_ring_edges.append((node_id, right))
            _link(adjacency, node_id, right)
            # "... and to at least one nearest node on ring r-1 (ring 1
            # connects to the origin)". Exactly one: a second link could
            # cover an arc reaching past the next same-ring neighbour and
            # cross its chord, which would break the planarity argument.
            if ring == 1:
                inner = 0
            else:
                inner = first_id_of_ring(ring - 1) + _nearest_index(
                    angles[ring][k], angles[ring - 1]
                )
            _link(adjacency, node_id, inner)

    # Stage C: thin the same-ring edges. Inner links are never candidates.
    _thin_same_ring_edges(adjacency, same_ring_edges, fields.stream("web.edges"))

    # Stage D: outer-ring Pinnacle arenas and glyphs.
    arenas, glyphs = _assign_pinnacles(rings, fields.stream("web.pinnacles"))

    # Stage E: mechanics. Every node rolls so the sequence of draws does not
    # depend on where the arenas landed; an arena's roll is then discarded.
    mechanics = _roll_mechanics(node_count(rings), fields.stream("web.mechanics"))

    # Stage F: assemble the nodes in id order.
    nodes: List[WebNode] = [
        WebNode(
            id=0,
            tier=0,
            ring_index=0,
            template=template_for(0, 0),
            mechanic=mechanics[0],
            x=0.0,
            y=0.0,
        )
    ]
    for ring in range(1, rings + 1):
        base = first_id_of_ring(ring)
        radius = RING_SPACING * ring
        for k in range(ring_size(ring)):
            node_id = base + k
            theta = angles[ring][k]
            pinnacle = arenas.get(node_id)
            nodes.append(
                WebNode(
                    id=node_id,
                    tier=min(MAX_TIER, ring),
                    ring_index=k,
                    template=template_for(min(MAX_TIER, ring), k),
                    # Judgement call: an arena is the Pinnacle fight itself
                    # and carries no side mechanic.
                    mechanic=None if pinnacle is not None else mechanics[node_id],
                    pinnacle=pinnacle,
                    glyph=glyphs.get(node_id),
                    x=_round_coord(radius * math.cos(theta)),
                    y=_round_coord(radius * math.sin(theta)),
                )
            )

    edges = tuple(
        WebEdge(a, b)
        for a, b in sorted(
            (a, b) for a in adjacency for b in adjacency[a] if a < b
        )
    )
    return Web(profile_seed=fields.raw, origin_id=0, nodes=tuple(nodes), edges=edges)


def _ring_angles(stream: Stream, count: int) -> List[float]:
    """Angles for one ring: even spacing, one random phase, small jitter.

    Spec: "evenly spaced angles with a small deterministic angular jitter".
    The phase stops every ring's node 0 lining up on the +x axis, which would
    draw a visible seam. Jitter is bounded by :data:`JITTER_FRACTION` of the
    spacing in each direction, which keeps neighbours in angular order and is
    what the planarity argument relies on.
    """
    spacing = _TWO_PI / count
    phase = stream.random() * spacing
    out: List[float] = []
    for k in range(count):
        jitter = (stream.random() * 2.0 - 1.0) * JITTER_FRACTION * spacing
        out.append((phase + k * spacing + jitter) % _TWO_PI)
    return out


def _angular_distance(a: float, b: float) -> float:
    """Shortest arc between two angles, in radians, in ``[0, pi]``."""
    d = abs(a - b) % _TWO_PI
    return min(d, _TWO_PI - d)


def _nearest_index(theta: float, inner_angles: Sequence[float]) -> int:
    """Index of the inner-ring node nearest in angle; ties go to the lower index."""
    best = 0
    best_d = _angular_distance(theta, inner_angles[0])
    for i in range(1, len(inner_angles)):
        d = _angular_distance(theta, inner_angles[i])
        if d < best_d:
            best, best_d = i, d
    return best


def _link(adjacency: Dict[int, Set[int]], a: int, b: int) -> None:
    adjacency[a].add(b)
    adjacency[b].add(a)


def _thin_same_ring_edges(
    adjacency: Dict[int, Set[int]],
    candidates: Sequence[Tuple[int, int]],
    stream: Stream,
) -> None:
    """Drop same-ring edges with :data:`DROP_CHANCE` each, under two guards.

    Spec (assignment): "deterministically drop some same-ring edges (never
    inner-ring links) with about 25 percent chance each, but never drop an
    edge whose removal would disconnect the graph, and never let any node's
    degree fall below 2".

    Every candidate draws its coin in ring order so that a guard refusing one
    drop does not shift the draws of the edges after it.
    """
    for a, b in candidates:
        if not stream.chance(DROP_CHANCE):
            continue
        if len(adjacency[a]) - 1 < MIN_DEGREE or len(adjacency[b]) - 1 < MIN_DEGREE:
            continue
        adjacency[a].discard(b)
        adjacency[b].discard(a)
        if not _still_connected(adjacency, a, b):
            _link(adjacency, a, b)


def _still_connected(adjacency: Dict[int, Set[int]], start: int, goal: int) -> bool:
    """True if ``goal`` is reachable from ``start`` in the current graph.

    Removing an edge disconnects the graph exactly when its two ends lose
    every other path between them, so one search from ``start`` answers it.
    The answer does not depend on iteration order, so the sets are walked
    as-is.
    """
    seen = {start}
    queue = deque([start])
    while queue:
        u = queue.popleft()
        if u == goal:
            return True
        for v in adjacency[u]:
            if v not in seen:
                seen.add(v)
                queue.append(v)
    return False


def _assign_pinnacles(
    rings: int, stream: Stream
) -> Tuple[Dict[int, Pinnacle], Dict[int, Pinnacle]]:
    """Choose the arenas and glyph nodes on the outer ring.

    Spec: "Two Pinnacles ... Each arena is unlocked by collecting fragments
    from three cleared tier-15 nodes bearing that Pinnacle's glyph", and the
    outer ring is "the only ones that can be Pinnacle arenas or carry a
    Pinnacle glyph". Assignment: exactly one arena per Pinnacle, at least
    :data:`GLYPHS_PER_PINNACLE` glyph nodes per Pinnacle that are not arenas,
    spaced around the ring.

    Layout: the first Pinnacle's arena lands at a random ring index and the
    second's diametrically opposite. Each Pinnacle's glyphs are then aimed at
    ``GLYPHS_PER_PINNACLE`` evenly spaced indices (stride ``n // 4``), the
    second Pinnacle's offset half a stride from the first's so the two glyph
    sets interleave. An aimed index that is already taken slides to the
    nearest free one, alternating sides, so nothing is ever double-booked.

    Returns ``(arenas, glyphs)`` keyed by node id.
    """
    count = ring_size(rings)
    base = first_id_of_ring(rings)
    needed = len(PINNACLES) * (1 + GLYPHS_PER_PINNACLE)
    if count < needed:
        raise ValueError(f"outer ring of {count} cannot hold {needed} Pinnacle nodes")

    taken: Set[int] = set()
    arenas: Dict[int, Pinnacle] = {}
    glyphs: Dict[int, Pinnacle] = {}

    first_arena = stream.randint(0, count - 1)
    for i, pinnacle in enumerate(PINNACLES):
        index = (first_arena + i * (count // len(PINNACLES))) % count
        index = _nearest_free(index, count, taken)
        taken.add(index)
        arenas[base + index] = pinnacle

    stride = max(1, count // GLYPHS_PER_PINNACLE)
    glyph_offset = stream.randint(1, max(1, stride - 1))
    for i, pinnacle in enumerate(PINNACLES):
        start = first_arena + glyph_offset + i * (stride // 2)
        for k in range(GLYPHS_PER_PINNACLE):
            index = _nearest_free((start + k * stride) % count, count, taken)
            taken.add(index)
            glyphs[base + index] = pinnacle
    return arenas, glyphs


def _nearest_free(index: int, count: int, taken: Set[int]) -> int:
    """The ring index closest to ``index`` not in ``taken``, ``+1`` before ``-1``."""
    for step in range(count):
        for candidate in ((index + step) % count, (index - step) % count):
            if candidate not in taken:
                return candidate
    raise ValueError("ring is full")


def _roll_mechanics(count: int, stream: Stream) -> List[Optional[Mechanic]]:
    """One roll per node id: :data:`MECHANIC_CHANCE` to carry one, uniform kind.

    Spec: "A node may carry one mechanic: breach, ritual, dig or shrine,
    decided at web generation and deterministic."
    """
    out: List[Optional[Mechanic]] = []
    for _ in range(count):
        if stream.chance(MECHANIC_CHANCE):
            out.append(stream.choice(MECHANICS))
        else:
            out.append(None)
    return out


def _round_coord(value: float) -> float:
    """Three decimals, and no negative zero, so serialised layouts are stable."""
    return round(value, 3) + 0.0


# --------------------------------------------------------------------------
# Provenance: is this web the one its seed generates?
# --------------------------------------------------------------------------


def rings_of(web: Web) -> int:
    """The ring count a web of this size was generated with.

    :func:`node_count` is strictly increasing in ``rings``, so a node count
    names at most one ring count.  Raises :class:`ValueError` when the count
    matches no web this generator can produce.
    """
    for rings in range(MIN_RINGS, MAX_TIER + 1):
        if node_count(rings) == len(web.nodes):
            return rings
    raise ValueError(f"{len(web.nodes)} nodes is not a web of {MIN_RINGS}..{MAX_TIER} rings")


def regenerated(web: Web) -> Web:
    """The web :func:`generate_web` builds from ``web``'s own seed and size."""
    return generate_web(web.profile_seed, rings_of(web))


def matches_seed(web: Web) -> bool:
    """True when ``web`` is, field for field, what its seed generates.

    Spec: "the web is a planar graph generated once per profile from a
    profile seed" -- so a stored web is either that web or it has been
    altered.  Persisted webs carry their nodes and edges, not only the
    seed, which is what lets a saved file be edited; this is the check that
    catches it.  A web of a size the generator cannot make, or of another
    ``version``, is not a match.
    """
    try:
        return regenerated(web) == web
    except ValueError:
        return False


# --------------------------------------------------------------------------
# Graph queries used by the gate and the engine
# --------------------------------------------------------------------------


def adjacency_of(web: Web) -> Dict[int, List[int]]:
    """Neighbour lists for every node, sorted, in one pass over the edges.

    ``Web.neighbours`` scans every edge per call; this builds the whole map
    once for callers that walk the graph.
    """
    adj: Dict[int, List[int]] = {n.id: [] for n in web.nodes}
    for e in web.edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)
    for key in adj:
        adj[key].sort()
    return adj


def bfs_tiers(web: Web) -> Dict[int, int]:
    """True graph distance from the origin for every reachable node.

    Spec: "tier = min(15, graph distance from the origin)". This returns the
    unclamped distance so the gate can compare it with ``WebNode.tier``;
    with ``rings <= MAX_TIER`` the two are equal. Nodes with no path to the
    origin are absent from the result, which the gate should treat as a
    failure.
    """
    adj = adjacency_of(web)
    dist: Dict[int, int] = {web.origin_id: 0}
    queue = deque([web.origin_id])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def _orientation(p: Point, q: Point, r: Point) -> int:
    """Sign of the turn p -> q -> r: 1 anticlockwise, -1 clockwise, 0 collinear."""
    cross = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    if cross > _EPS:
        return 1
    if cross < -_EPS:
        return -1
    return 0


def _within_box(p: Point, q: Point, r: Point) -> bool:
    """True if ``r``, known to be collinear with ``p``-``q``, lies between them."""
    return (
        min(p[0], q[0]) - _EPS <= r[0] <= max(p[0], q[0]) + _EPS
        and min(p[1], q[1]) - _EPS <= r[1] <= max(p[1], q[1]) + _EPS
    )


def segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True if closed segments ``a``-``b`` and ``c``-``d`` share any point.

    Touching counts: an endpoint of one segment lying on the other, or
    collinear overlap, is an intersection. Callers checking planarity skip
    edges that share a node before calling this, so a shared endpoint never
    reaches here as a false positive.
    """
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _within_box(a, b, c):
        return True
    if o2 == 0 and _within_box(a, b, d):
        return True
    if o3 == 0 and _within_box(c, d, a):
        return True
    if o4 == 0 and _within_box(c, d, b):
        return True
    return False


def find_crossing(web: Web) -> Optional[Tuple[WebEdge, WebEdge]]:
    """The first pair of non-adjacent edges that meet in the x, y layout.

    Spec: "The web is a planar graph". Two edges that share a node are
    allowed to meet at that node and are skipped. The search sweeps edges
    sorted by their left-most x and only tests pairs whose bounding boxes
    overlap, so a web-sized graph is checked in a few thousand tests rather
    than the full quadratic count.

    Returns ``None`` when the layout is planar.
    """
    pos: Dict[int, Point] = {n.id: (n.x, n.y) for n in web.nodes}
    boxed = []
    for e in web.edges:
        p, q = pos[e.a], pos[e.b]
        boxed.append(
            (
                min(p[0], q[0]),
                max(p[0], q[0]),
                min(p[1], q[1]),
                max(p[1], q[1]),
                e,
                p,
                q,
            )
        )
    boxed.sort(key=lambda item: (item[0], item[4].a, item[4].b))
    for i, (xmin, xmax, ymin, ymax, edge, p, q) in enumerate(boxed):
        for j in range(i + 1, len(boxed)):
            other = boxed[j]
            if other[0] > xmax + _EPS:
                break
            if other[3] < ymin - _EPS or other[2] > ymax + _EPS:
                continue
            o = other[4]
            if o.a in (edge.a, edge.b) or o.b in (edge.a, edge.b):
                continue
            if segments_cross(p, q, other[5], other[6]):
                return edge, o
    return None


def is_planar_layout(web: Web) -> bool:
    """True when no two non-adjacent edges intersect in the x, y layout."""
    return find_crossing(web) is None


__all__ = [
    "RING_SPACING", "JITTER_FRACTION", "DROP_CHANCE", "MECHANIC_CHANCE",
    "GLYPHS_PER_PINNACLE", "MIN_DEGREE", "MIN_RINGS", "MECHANICS", "PINNACLES",
    "TEMPLATE_BANDS", "ring_size", "first_id_of_ring", "node_count",
    "template_for", "generate_web", "rings_of", "regenerated", "matches_seed",
    "adjacency_of", "bfs_tiers",
    "segments_cross", "find_crossing", "is_planar_layout",
]


# --------------------------------------------------------------------------
# Self-check entry point: python3 -m lucifer_descent.web [seed] [rings]
# --------------------------------------------------------------------------


def _main(argv: Sequence[str]) -> int:
    seed = int(argv[1], 0) if len(argv) > 1 else 0x5EED
    rings = int(argv[2]) if len(argv) > 2 else MAX_TIER
    web = generate_web(seed, rings)
    tiers = bfs_tiers(web)
    mismatched = [n.id for n in web.nodes if tiers.get(n.id) != n.tier]
    degrees = adjacency_of(web)
    outer = web.nodes_at_tier(min(MAX_TIER, rings))
    print(f"profile seed 0x{web.profile_seed:016X}, {rings} rings")
    print(f"nodes {len(web.nodes)}, edges {len(web.edges)}")
    print(f"tier == bfs distance for all nodes: {not mismatched}")
    print(f"connected: {len(tiers) == len(web.nodes)}")
    print(f"planar layout: {is_planar_layout(web)}")
    print(f"min degree: {min(len(v) for v in degrees.values())}")
    print(f"mechanic nodes: {sum(1 for n in web.nodes if n.mechanic)} of {len(web.nodes)}")
    for pinnacle in PINNACLES:
        arena = [n.id for n in outer if n.pinnacle is pinnacle]
        glyph = [n.id for n in outer if n.glyph is pinnacle]
        print(f"{pinnacle.value}: arena {arena}, glyphs {glyph}")
    return 0 if not mismatched else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))

