"""The Phase 4 gate: does a Descent web, profile and ledger obey section 03?

Spec: docs/WORLD_BIBLE.md section 03 (the web, node states, Sigils, the two
Pinnacles) and the reconnect rule in section 07.

Four entry points, each returning a list of :class:`Problem` so a caller can
count, print or assert on them, plus one driver:

* :func:`check_web`     -- the static graph a profile was given.
* :func:`check_state`   -- one profile's mutable state against its web.
* :func:`check_replay`  -- the ledger rebuilds the same profile.
* :func:`simulate`      -- a deterministic random walk of *legal* play.
* :func:`run_suite`     -- all of the above over many profile seeds, plus a
  SQLite round trip, streamed one profile at a time.

The gate keeps its own answers for the rules it checks.  Its BFS is the
textbook one (there is only one way to write it, so the value of running
it twice is small), but its segment-intersection test is genuinely
different from the generator's: it works in exact integer arithmetic on
the layout's decimal grid, with no epsilon, so a rounding-sensitive
crossing the generator's floating-point sweep could miss cannot hide here.
When :mod:`lucifer_descent.web` also exposes ``bfs_tiers`` and
``is_planar_layout`` they are run as well and any *disagreement* with the
gate's answer is itself reported (``helper_disagreement``).

The gate also asks two questions the engine does not: is the web the one
its seed generates (``web_not_from_seed``), and is every Sigil in the
stash, on the live instance and in the ledger a Sigil this profile's mint
produces (``*_sigil_not_minted``, ``ledger_sigil_*``)?  A persisted profile
carries its web and its Sigils as data, so those are what a save-file
editor changes, and the gate is where a loaded profile is held to them.

Randomness
----------
Every draw in :func:`simulate` comes from a :class:`lucifer_gen.seed.Stream`
opened through :meth:`SeedFields.stream` on the profile seed (the walk) or
on the map seed (the stub map probe).  Nothing here touches ``random``, the
clock, or unordered iteration that could reach the output; every set and
dict is sorted before it is walked.

Problem kinds
-------------
``Problem.kind`` is one of the ``K_*`` constants below.  A ``node_id`` is
attached when one node is at fault, otherwise it is ``None`` and the detail
names what is.

Judgement calls the assignment left open, recorded here:

* The ledger's first ``seq`` may be 0 or 1.  The assignment says ``0..n-1``;
  the engine that writes every ledger numbers from 1.  The gate accepts a
  dense, consecutive ledger from either base and reports the base in the
  problem detail when it is anything else.
* ``check_state`` also checks a few invariants the assignment did not list
  but section 03 implies: the origin is ``CLEARED``; an ``ACTIVE``,
  ``FAILED`` or ``CLEARED`` node other than the origin has a ``CLEARED``
  neighbour (it was reachable once, and edges are never removed); the live
  instance's Sigil has left the stash, seeded the map, and can open its
  node; a Pinnacle with three fragments is unlocked.
* ``check_replay`` compares passive points, fragments and unlocked
  Pinnacles as well as the states dict, because the engine promises to
  rebuild all four from the ledger.  It does not compare the stash or the
  live instance: the engine documents that the ledger does not carry them.
  Given a ``map_probe`` it also re-derives every ``OPEN`` entry's recorded
  probe facts from the Sigil that funded it (recovered from the id through
  :func:`lucifer_descent.sigils.unmint`) and reports a disagreement.
* ``check_state`` holds the stash and the live instance to what it can
  see: every Sigil is a genuine mint of the profile seed, no stashed id
  was ever spent, the instance agrees with the ledger's ``OPEN`` of its
  node, ticks never run backwards, cleared nodes form one component with
  the origin, and no arena was opened while its Pinnacle was locked.
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from lucifer_descent import web as webmod
from lucifer_descent.contracts import (
    ELITE_CLEAR_FRACTION,
    FRAGMENTS_TO_UNLOCK,
    MAX_TIER,
    SIGIL_CONSUMING_EVENTS,
    TRANSITIONS,
    Event,
    NodeState,
    Pinnacle,
    ProfileState,
    Sigil,
    Web,
    WebEdge,
    WebNode,
)
from lucifer_descent.engine import DescentEngine, MapProbe
from lucifer_descent.sigils import is_genuine, mint_sigil, node_is_openable, roll_drops, tier_from_id, unmint
from lucifer_descent.store import SqliteStore, first_difference, round_trip_equal
from lucifer_descent.web import generate_web, rings_of
from lucifer_gen.seed import MASK64, SeedFields, Stream, format_seed

__all__ = [
    "Problem",
    "SuiteReport",
    "check_web",
    "check_state",
    "check_replay",
    "simulate",
    "run_suite",
    "bfs_distances",
    "segments_cross",
    "find_crossings",
    "MIN_DEGREE",
    "MIN_GLYPHS_PER_PINNACLE",
    "ARENAS_PER_PINNACLE",
]

# --------------------------------------------------------------------------
# Tunables
# --------------------------------------------------------------------------

#: Assignment: "no node degree below 2".
MIN_DEGREE = 2

#: Assignment: "exactly one arena and at least 4 glyphs per Pinnacle".
ARENAS_PER_PINNACLE = 1
MIN_GLYPHS_PER_PINNACLE = 4

#: Ledger bases the gate accepts; see the module docstring.
LEDGER_BASES = (0, 1)

#: Walk probabilities for :func:`simulate`.  Fixed, as the assignment asks.
P_PUSH_OUTWARD = 0.7   # pick among the highest-tier openable nodes
P_DIE = 0.15
P_ABANDON = 0.05
P_TIMEOUT = 0.05
#: How far above the node's tier a freshly minted Sigil may be, at most.
SIGIL_TIER_HEADROOM = 2

#: The stub map probe's answers: chance of a boss, range of elite packs.
STUB_BOSS_CHANCE = 0.5
STUB_ELITES_MIN = 2
STUB_ELITES_MAX = 12

#: The gate's geometry has no tolerance: coordinates are lifted to a common
#: integer grid (see :func:`_integer_grid`) and every orientation test is
#: exact.  That is what makes it a different test from the generator's.

# --------------------------------------------------------------------------
# Problem kinds
# --------------------------------------------------------------------------

# check_web
K_DUPLICATE_ID = "node_id_duplicate"
K_IDS_NOT_DENSE = "node_ids_not_dense"
K_ORIGIN_MISSING = "origin_missing"
K_EDGE_SELF_LOOP = "edge_self_loop"
K_EDGE_UNKNOWN_NODE = "edge_unknown_node"
K_EDGE_DUPLICATE = "edge_duplicate"
K_DEGREE = "degree_below_min"
K_TIER_ABOVE_MAX = "tier_above_max"
K_TIER_MISMATCH = "tier_mismatch"
K_DISCONNECTED = "disconnected"
K_EDGE_CROSSING = "edge_crossing"
K_ARENA_COUNT = "arena_count"
K_ARENA_TIER = "arena_off_outer_ring"
K_ARENA_IS_GLYPH = "arena_is_glyph"
K_GLYPH_COUNT = "glyph_count"
K_GLYPH_TIER = "glyph_off_outer_ring"
K_HELPER_DISAGREEMENT = "helper_disagreement"
K_HELPER_ERROR = "helper_error"
K_WEB_NOT_FROM_SEED = "web_not_from_seed"

# check_state
K_STATE_MISSING = "state_missing"
K_STATE_EXTRA = "state_extra"
K_STATE_INVALID = "state_invalid"
K_ORIGIN_NOT_CLEARED = "origin_not_cleared"
K_REACHABLE_NO_CLEARED = "reachable_without_cleared_neighbour"
K_LOCKED_WITH_CLEARED = "locked_with_cleared_neighbour"
K_UNLOCKED_NO_CLEARED = "unlocked_without_cleared_neighbour"
K_MULTIPLE_ACTIVE = "multiple_active"
K_ACTIVE_NO_INSTANCE = "active_without_instance"
K_INSTANCE_NOT_ACTIVE = "instance_node_not_active"
K_INSTANCE_UNKNOWN_NODE = "instance_unknown_node"
K_INSTANCE_SIGIL_IN_STASH = "instance_sigil_in_stash"
K_INSTANCE_SEED = "instance_seed_mismatch"
K_INSTANCE_SIGIL_WEAK = "instance_sigil_too_weak"
K_INSTANCE_ARENA_LOCKED = "instance_arena_locked"
K_STASH_KEY = "stash_key_mismatch"
K_PINNACLE_EARLY = "pinnacle_unlocked_early"
K_PINNACLE_LATE = "pinnacle_not_unlocked"
K_FRAGMENTS_NEGATIVE = "fragments_negative"
K_PASSIVE_NEGATIVE = "passive_points_negative"
K_LEDGER_BASE = "ledger_base"
K_LEDGER_GAP = "ledger_gap"
K_LEDGER_UNKNOWN_NODE = "ledger_unknown_node"
K_LEDGER_ILLEGAL = "ledger_illegal_transition"
K_LEDGER_TICK_ORDER = "ledger_tick_order"
K_LEDGER_SIGIL_MALFORMED = "ledger_sigil_malformed"
K_LEDGER_SIGIL_WEAK = "ledger_sigil_too_weak"
K_LEDGER_SIGIL_REUSED = "ledger_sigil_reused"
K_CLEARED_ISLAND = "cleared_not_connected_to_origin"
K_ARENA_OPENED_LOCKED = "arena_opened_while_locked"
K_STASH_NOT_MINTED = "stash_sigil_not_minted"
K_STASH_SPENT = "stash_sigil_spent"
K_INSTANCE_NOT_MINTED = "instance_sigil_not_minted"
K_INSTANCE_LEDGER = "instance_disagrees_with_ledger"
K_INSTANCE_KILLS = "instance_kills_invalid"

# check_replay
K_LEDGER_MISSING_FACT = "ledger_missing_fact"
K_REPLAY_ERROR = "replay_error"
K_REPLAY_MISMATCH = "replay_mismatch"
K_REPLAY_PASSIVE = "replay_passive_points"
K_REPLAY_FRAGMENTS = "replay_fragments"
K_REPLAY_UNLOCKED = "replay_unlocked_pinnacles"
K_REPLAY_LEDGER = "replay_ledger"
K_LEDGER_FACT_MISMATCH = "ledger_fact_mismatch"

# run_suite
K_STORE_ROUND_TRIP = "store_round_trip"
K_EXCEPTION = "exception"   # suffixed with the phase: "exception:simulate"


@dataclass(frozen=True)
class Problem:
    """One rule the gate found broken.

    ``kind`` is a ``K_*`` constant, ``node_id`` the node at fault when there
    is exactly one, ``detail`` a sentence a person can act on.
    """

    kind: str
    node_id: Optional[int]
    detail: str

    def __str__(self) -> str:
        where = f" node {self.node_id}" if self.node_id is not None else ""
        return f"[{self.kind}]{where}: {self.detail}"


Point = Tuple[float, float]


# --------------------------------------------------------------------------
# Graph helpers: the gate's own, independent of lucifer_descent.web
# --------------------------------------------------------------------------


def _adjacency(node_ids: Sequence[int], edges: Sequence[WebEdge]) -> Dict[int, List[int]]:
    """Sorted neighbour lists over ``edges``, all of whose ends are in ``node_ids``."""
    adj: Dict[int, List[int]] = {nid: [] for nid in node_ids}
    for edge in edges:
        adj[edge.a].append(edge.b)
        adj[edge.b].append(edge.a)
    for nid in adj:
        adj[nid].sort()
    return adj


def bfs_distances(origin_id: int, adjacency: Dict[int, List[int]]) -> Dict[int, int]:
    """Unclamped graph distance from the origin for every node it can reach.

    Spec: "tier = min(15, graph distance from the origin)".  The distance is
    returned unclamped so the caller can tell a node at distance 16 (which
    breaks "tier 15 nodes ring the outer edge") from one at 15.  Nodes with
    no path from the origin are absent, which is how :func:`check_web`
    detects a disconnected web.
    """
    dist: Dict[int, int] = {origin_id: 0}
    queue = deque([origin_id])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


IntPoint = Tuple[int, int]


def _integer_grid(points: Iterable[Point]) -> List[IntPoint]:
    """Lift coordinates onto one integer grid, exactly.

    Each coordinate is read as the decimal its ``repr`` spells (``12.345``
    is 12345/1000, not the nearest binary float), the least common
    denominator over every coordinate is found, and everything is scaled
    by it.  Layouts from :func:`generate_web` are three-decimal, so the
    scale is 1000; a hand-built layout with more decimals gets a larger
    one.  Orientation tests on the result are exact integer arithmetic.
    """
    exact = [(Fraction(repr(float(x))), Fraction(repr(float(y)))) for x, y in points]
    scale = 1
    for x, y in exact:
        scale = lcm(scale, x.denominator, y.denominator)
    return [(int(x * scale), int(y * scale)) for x, y in exact]


def _orientation(p: IntPoint, q: IntPoint, r: IntPoint) -> int:
    """Sign of the turn p -> q -> r: 1 anticlockwise, -1 clockwise, 0 collinear.  Exact."""
    cross = (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    return (cross > 0) - (cross < 0)


def _on_segment(p: IntPoint, q: IntPoint, r: IntPoint) -> bool:
    """True if ``r``, already known collinear with ``p``-``q``, lies between them."""
    return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[1] <= max(p[1], q[1])


def _segments_cross_exact(a: IntPoint, b: IntPoint, c: IntPoint, d: IntPoint) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a, b, c):
        return True
    if o2 == 0 and _on_segment(a, b, d):
        return True
    if o3 == 0 and _on_segment(c, d, a):
        return True
    if o4 == 0 and _on_segment(c, d, b):
        return True
    return False


def segments_cross(a: Point, b: Point, c: Point, d: Point) -> bool:
    """True when closed segments ``a``-``b`` and ``c``-``d`` share any point.

    Spec: "The web is a planar graph."  Touching counts as crossing -- an
    endpoint on the other segment, or collinear overlap -- because two edges
    that meet anywhere but at a shared node are not a planar drawing.  The
    caller skips edge pairs that share a node before calling this.  The test
    is exact: the four points are lifted to an integer grid first.
    """
    ia, ib, ic, id_ = _integer_grid((a, b, c, d))
    return _segments_cross_exact(ia, ib, ic, id_)


def find_crossings(
    positions: Dict[int, Point], edges: Sequence[WebEdge]
) -> List[Tuple[WebEdge, WebEdge]]:
    """Every pair of edges that meet without sharing a node, in a fixed order.

    Spec: "The web is a planar graph."  A sweep over edges sorted by their
    left-most x only tests pairs whose bounding boxes overlap, so a
    web-sized layout is a few thousand tests rather than the full quadratic
    count.  Every test is exact integer arithmetic on one common grid.  The
    order of the result is a function of the input alone.
    """
    ids = sorted(positions)
    grid = dict(zip(ids, _integer_grid(positions[nid] for nid in ids)))
    boxed = []
    for edge in edges:
        p, q = grid[edge.a], grid[edge.b]
        boxed.append(
            (min(p[0], q[0]), max(p[0], q[0]), min(p[1], q[1]), max(p[1], q[1]), edge, p, q)
        )
    boxed.sort(key=lambda item: (item[0], item[4].a, item[4].b))
    found: List[Tuple[WebEdge, WebEdge]] = []
    for i, (xmin, xmax, ymin, ymax, edge, p, q) in enumerate(boxed):
        for j in range(i + 1, len(boxed)):
            oxmin, oxmax, oymin, oymax, other, r, s = boxed[j]
            if oxmin > xmax:
                break
            if oymax < ymin or oymin > ymax:
                continue
            if other.a in (edge.a, edge.b) or other.b in (edge.a, edge.b):
                continue
            if _segments_cross_exact(p, q, r, s):
                found.append((edge, other))
    return found


# --------------------------------------------------------------------------
# check_web
# --------------------------------------------------------------------------


def check_web(web: Web) -> List[Problem]:
    """Every structural rule section 03 puts on a generated web.

    In order: node ids unique and dense from 0; the origin exists; edges are
    well formed (no self loops, no unknown ends, no duplicates); no degree
    below :data:`MIN_DEGREE`; ``tier == graph distance`` for every node
    ("tier = min(15, graph distance from the origin)") with no node beyond
    tier 15 ("tier 15 nodes ring the outer edge"); the web is connected;
    the ``x, y`` layout is planar ("the web is a planar graph"); and per
    Pinnacle exactly one arena and at least :data:`MIN_GLYPHS_PER_PINNACLE`
    glyph nodes, all on tier 15 ("the only ones that can be Pinnacle arenas
    or carry a Pinnacle glyph"), with no arena doubling as a glyph.  Last,
    provenance: the web must be, field for field, what :func:`generate_web`
    builds from its own ``profile_seed`` at its own size ("generated once
    per profile from a profile seed"), which is the check that catches a
    persisted web edited after the fact.

    When :mod:`lucifer_descent.web` exposes ``bfs_tiers`` or
    ``is_planar_layout`` they are run too and a disagreement with the gate's
    own answer is reported, but only once the structure is sound enough for
    them not to crash.
    """
    problems: List[Problem] = []

    # --- nodes -----------------------------------------------------------
    nodes_by_id: Dict[int, WebNode] = {}
    for node in web.nodes:
        if node.id in nodes_by_id:
            problems.append(Problem(K_DUPLICATE_ID, node.id, f"node id {node.id} appears twice"))
            continue
        nodes_by_id[node.id] = node
    ids = sorted(nodes_by_id)
    if ids != list(range(len(ids))):
        problems.append(
            Problem(K_IDS_NOT_DENSE, None, f"{len(ids)} node ids are not 0..{len(ids) - 1}")
        )
    origin_ok = web.origin_id in nodes_by_id
    if not origin_ok:
        problems.append(Problem(K_ORIGIN_MISSING, None, f"origin {web.origin_id} is not a node"))

    # --- edges -----------------------------------------------------------
    good_edges: List[WebEdge] = []
    seen_pairs: Set[Tuple[int, int]] = set()
    edges_ok = True
    for edge in web.edges:
        if edge.a == edge.b:
            problems.append(Problem(K_EDGE_SELF_LOOP, edge.a, f"edge {edge.a}-{edge.b} is a self loop"))
            edges_ok = False
            continue
        if edge.a not in nodes_by_id or edge.b not in nodes_by_id:
            problems.append(
                Problem(K_EDGE_UNKNOWN_NODE, None, f"edge {edge.a}-{edge.b} touches an unknown node")
            )
            edges_ok = False
            continue
        pair = (min(edge.a, edge.b), max(edge.a, edge.b))
        if pair in seen_pairs:
            problems.append(Problem(K_EDGE_DUPLICATE, None, f"edge {pair[0]}-{pair[1]} appears twice"))
            edges_ok = False
            continue
        seen_pairs.add(pair)
        good_edges.append(edge)
    adjacency = _adjacency(ids, good_edges)

    # --- degree ----------------------------------------------------------
    for nid in ids:
        if len(adjacency[nid]) < MIN_DEGREE:
            problems.append(
                Problem(K_DEGREE, nid, f"degree {len(adjacency[nid])} is below {MIN_DEGREE}")
            )

    # --- tiers and connectivity ------------------------------------------
    for nid in ids:
        node = nodes_by_id[nid]
        if node.tier > MAX_TIER:
            problems.append(Problem(K_TIER_ABOVE_MAX, nid, f"tier {node.tier} exceeds {MAX_TIER}"))
    distances: Dict[int, int] = {}
    if origin_ok:
        distances = bfs_distances(web.origin_id, adjacency)
        for nid in ids:
            node = nodes_by_id[nid]
            if nid not in distances:
                problems.append(Problem(K_DISCONNECTED, nid, "no path from the origin"))
            elif node.tier != distances[nid]:
                problems.append(
                    Problem(
                        K_TIER_MISMATCH,
                        nid,
                        f"tier {node.tier} but graph distance from the origin is {distances[nid]}",
                    )
                )

    # --- planarity -------------------------------------------------------
    positions: Dict[int, Point] = {nid: (nodes_by_id[nid].x, nodes_by_id[nid].y) for nid in ids}
    crossings = find_crossings(positions, good_edges)
    for edge, other in crossings:
        problems.append(
            Problem(
                K_EDGE_CROSSING,
                None,
                f"edge {edge.a}-{edge.b} crosses edge {other.a}-{other.b} in the layout",
            )
        )

    # --- cross-check the generator's own helpers, if it has them ---------
    structure_ok = origin_ok and edges_ok and len(nodes_by_id) == len(web.nodes)
    if structure_ok:
        problems.extend(_cross_check_helpers(web, distances, not crossings))

    # --- Pinnacles -------------------------------------------------------
    for pinnacle in sorted(Pinnacle, key=lambda p: p.value):
        arenas = [nodes_by_id[nid] for nid in ids if nodes_by_id[nid].pinnacle is pinnacle]
        glyphs = [nodes_by_id[nid] for nid in ids if nodes_by_id[nid].glyph is pinnacle]
        if len(arenas) != ARENAS_PER_PINNACLE:
            problems.append(
                Problem(
                    K_ARENA_COUNT,
                    None,
                    f"{pinnacle.value} has {len(arenas)} arena(s), expected {ARENAS_PER_PINNACLE}: "
                    f"{[a.id for a in arenas]}",
                )
            )
        for arena in arenas:
            if arena.tier != MAX_TIER:
                problems.append(
                    Problem(K_ARENA_TIER, arena.id, f"{pinnacle.value} arena is tier {arena.tier}, not {MAX_TIER}")
                )
            if arena.glyph is not None:
                problems.append(
                    Problem(K_ARENA_IS_GLYPH, arena.id, f"{pinnacle.value} arena also carries a {arena.glyph.value} glyph")
                )
        if len(glyphs) < MIN_GLYPHS_PER_PINNACLE:
            problems.append(
                Problem(
                    K_GLYPH_COUNT,
                    None,
                    f"{pinnacle.value} has {len(glyphs)} glyph node(s), fewer than {MIN_GLYPHS_PER_PINNACLE}",
                )
            )
        for glyph in glyphs:
            if glyph.tier != MAX_TIER:
                problems.append(
                    Problem(K_GLYPH_TIER, glyph.id, f"{pinnacle.value} glyph is on tier {glyph.tier}, not {MAX_TIER}")
                )

    # --- provenance --------------------------------------------------------
    problems.extend(_check_provenance(web))
    return problems


def _check_provenance(web: Web) -> List[Problem]:
    """``web`` is what its seed generates, or it has been altered."""
    try:
        rings = rings_of(web)
    except ValueError as exc:
        return [Problem(K_WEB_NOT_FROM_SEED, None, f"{exc}")]
    expected = generate_web(web.profile_seed, rings)
    if expected == web:
        return []
    if expected.version != web.version:
        detail = f"web is version {web.version}, the generator makes version {expected.version}"
    elif expected.origin_id != web.origin_id:
        detail = f"origin is {web.origin_id}, the generator puts it at {expected.origin_id}"
    else:
        by_id = {n.id: n for n in expected.nodes}
        bad_nodes = sorted(n.id for n in web.nodes if by_id.get(n.id) != n)
        bad_edges = sorted(set(web.edges) ^ set(expected.edges), key=lambda e: (e.a, e.b))
        detail = (
            f"web differs from generate_web({format_seed(web.profile_seed)}, {rings}) at "
            f"{len(bad_nodes)} node(s) {bad_nodes[:6]} and {len(bad_edges)} edge(s) "
            f"{[(e.a, e.b) for e in bad_edges[:6]]}"
        )
    return [Problem(K_WEB_NOT_FROM_SEED, None, detail)]


def _cross_check_helpers(web: Web, distances: Dict[int, int], planar: bool) -> List[Problem]:
    """Run ``web.bfs_tiers`` and ``web.is_planar_layout`` when they exist.

    The gate's own answer stays authoritative; these only add a problem when
    the helper *disagrees*, or raises, so the two implementations police each
    other.
    """
    problems: List[Problem] = []
    helper_bfs = getattr(webmod, "bfs_tiers", None)
    if callable(helper_bfs):
        try:
            theirs = helper_bfs(web)
        except Exception as exc:  # a crashing helper is a finding, not a gate failure
            problems.append(Problem(K_HELPER_ERROR, None, f"web.bfs_tiers raised {type(exc).__name__}: {exc}"))
        else:
            if dict(theirs) != distances:
                differing = sorted(k for k in set(theirs) | set(distances) if theirs.get(k) != distances.get(k))
                problems.append(
                    Problem(
                        K_HELPER_DISAGREEMENT,
                        None,
                        f"web.bfs_tiers disagrees with the gate's BFS at nodes {differing[:10]}",
                    )
                )
    helper_planar = getattr(webmod, "is_planar_layout", None)
    if callable(helper_planar):
        try:
            theirs_planar = bool(helper_planar(web))
        except Exception as exc:
            problems.append(
                Problem(K_HELPER_ERROR, None, f"web.is_planar_layout raised {type(exc).__name__}: {exc}")
            )
        else:
            if theirs_planar != planar:
                problems.append(
                    Problem(
                        K_HELPER_DISAGREEMENT,
                        None,
                        f"web.is_planar_layout says {theirs_planar}, the gate's sweep says {planar}",
                    )
                )
    return problems


# --------------------------------------------------------------------------
# check_state
# --------------------------------------------------------------------------


def check_state(state: ProfileState) -> List[Problem]:
    """Every invariant one profile's mutable state must satisfy against its web.

    Spec rules, in the order they are checked:

    * every node in the web has a state and nothing else does;
    * the origin is ``CLEARED`` (something must be, or nothing is reachable);
    * "A node is reachable only if it shares an edge with a cleared node":
      a ``REACHABLE`` node has a ``CLEARED`` neighbour, a ``LOCKED`` node has
      none, and an ``ACTIVE``/``FAILED``/``CLEARED`` node other than the
      origin has one (it was reachable once; edges are never removed);
    * one instance at a time: an ``ACTIVE`` node exists iff ``instance`` is
      set and names it, and there is at most one;
    * the live Sigil left the stash, "the Sigil's own item seed becomes the
      map seed", ``sigil.tier >= node.tier``, and an arena is only live if
      its Pinnacle is unlocked;
    * "Each arena is unlocked by collecting fragments from three cleared
      tier-15 nodes": unlocked Pinnacles have at least
      :data:`FRAGMENTS_TO_UNLOCK` fragments, and vice versa, and an arena
      that is or was open (``ACTIVE``, ``FAILED``, ``CLEARED``) has its
      Pinnacle unlocked, since fragments never go away;
    * cleared nodes form one component with the origin: every cleared node
      was reachable once, which needed a cleared neighbour, and so on back
      to the origin -- an island of cleared nodes was never reached;
    * every Sigil in the stash and on the instance is a genuine mint of the
      profile seed (:func:`sigils.is_genuine`) and no stashed id was ever
      spent; the instance agrees with the ledger's ``OPEN`` of its node
      (Sigil, tick, probe facts), its kill count is not negative, and on a
      bossless map it is below the threshold;
    * the ledger is dense and consecutive (see the module docstring on its
      base), every entry is a row of :data:`contracts.TRANSITIONS`, ticks
      never decrease, and every ``OPEN``'s Sigil id is well formed, of a
      tier that can open its node, and used by no other ``OPEN``.
    """
    problems: List[Problem] = []
    web = state.web
    nodes_by_id: Dict[int, WebNode] = {n.id: n for n in web.nodes}
    web_ids = sorted(nodes_by_id)
    web_id_set = set(web_ids)
    states = state.states

    # --- coverage --------------------------------------------------------
    for nid in web_ids:
        if nid not in states:
            problems.append(Problem(K_STATE_MISSING, nid, "web node has no state"))
    for nid in sorted(states):
        if nid not in web_id_set:
            problems.append(Problem(K_STATE_EXTRA, nid, "state names a node that is not in the web"))
        elif not isinstance(states[nid], NodeState):
            problems.append(Problem(K_STATE_INVALID, nid, f"state is {states[nid]!r}, not a NodeState"))
    known = [nid for nid in web_ids if isinstance(states.get(nid), NodeState)]
    known_set = set(known)

    def state_of(nid: int) -> Optional[NodeState]:
        return states.get(nid) if nid in known_set else None

    # --- neighbour rules -------------------------------------------------
    valid_edges = [e for e in web.edges if e.a in web_id_set and e.b in web_id_set and e.a != e.b]
    adjacency = _adjacency(web_ids, valid_edges)
    origin = web.origin_id
    if origin in known_set and states[origin] is not NodeState.CLEARED:
        problems.append(
            Problem(K_ORIGIN_NOT_CLEARED, origin, f"origin is {states[origin].value}; it must be cleared by genesis")
        )
    for nid in known:
        s = states[nid]
        has_cleared = any(state_of(m) is NodeState.CLEARED for m in adjacency[nid])
        if s is NodeState.REACHABLE and not has_cleared:
            problems.append(Problem(K_REACHABLE_NO_CLEARED, nid, "reachable but no neighbour is cleared"))
        elif s is NodeState.LOCKED and has_cleared:
            problems.append(Problem(K_LOCKED_WITH_CLEARED, nid, "locked although a neighbour is cleared"))
        elif s in (NodeState.ACTIVE, NodeState.FAILED, NodeState.CLEARED) and nid != origin and not has_cleared:
            problems.append(
                Problem(K_UNLOCKED_NO_CLEARED, nid, f"{s.value} but no neighbour is cleared, so it was never reachable")
            )
    # Cleared nodes must be one component with the origin.
    if origin in known_set and states[origin] is NodeState.CLEARED:
        component = {origin}
        queue = deque([origin])
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                if v not in component and state_of(v) is NodeState.CLEARED:
                    component.add(v)
                    queue.append(v)
        for nid in known:
            if states[nid] is NodeState.CLEARED and nid not in component:
                problems.append(
                    Problem(K_CLEARED_ISLAND, nid, "cleared but not connected to the origin through cleared nodes")
                )
    # An arena that is or was open needed its Pinnacle unlocked.
    for nid in known:
        node = nodes_by_id[nid]
        if node.pinnacle is None or node.pinnacle in state.unlocked_pinnacles:
            continue
        if states[nid] in (NodeState.ACTIVE, NodeState.FAILED, NodeState.CLEARED):
            problems.append(
                Problem(
                    K_ARENA_OPENED_LOCKED, nid,
                    f"{node.pinnacle.value} arena is {states[nid].value} but the {node.pinnacle.value} is not unlocked",
                )
            )

    # --- the live instance -----------------------------------------------
    active = [nid for nid in known if states[nid] is NodeState.ACTIVE]
    if len(active) > 1:
        problems.append(Problem(K_MULTIPLE_ACTIVE, None, f"{len(active)} nodes are active: {active}"))
    inst = state.instance
    if inst is None:
        for nid in active:
            problems.append(Problem(K_ACTIVE_NO_INSTANCE, nid, "active but the profile has no live instance"))
    else:
        if inst.node_id not in web_id_set:
            problems.append(Problem(K_INSTANCE_UNKNOWN_NODE, inst.node_id, "instance names a node that is not in the web"))
        elif state_of(inst.node_id) is not NodeState.ACTIVE:
            problems.append(
                Problem(
                    K_INSTANCE_NOT_ACTIVE,
                    inst.node_id,
                    f"instance is live here but the node is {getattr(state_of(inst.node_id), 'value', None)}",
                )
            )
        for nid in active:
            if nid != inst.node_id:
                problems.append(Problem(K_ACTIVE_NO_INSTANCE, nid, f"active but the live instance is on node {inst.node_id}"))
        if inst.sigil.id in state.stash:
            problems.append(Problem(K_INSTANCE_SIGIL_IN_STASH, inst.node_id, f"live sigil {inst.sigil.id!r} is still in the stash"))
        if inst.map_seed != inst.sigil.seed:
            problems.append(
                Problem(
                    K_INSTANCE_SEED,
                    inst.node_id,
                    f"map seed {format_seed(inst.map_seed)} is not the sigil's seed {format_seed(inst.sigil.seed)}",
                )
            )
        node = nodes_by_id.get(inst.node_id)
        if node is not None:
            if inst.sigil.tier < node.tier:
                problems.append(
                    Problem(K_INSTANCE_SIGIL_WEAK, node.id, f"sigil tier {inst.sigil.tier} is below node tier {node.tier}")
                )
            if node.pinnacle is not None and node.pinnacle not in state.unlocked_pinnacles:
                problems.append(
                    Problem(K_INSTANCE_ARENA_LOCKED, node.id, f"{node.pinnacle.value} arena is live but not unlocked")
                )
        if not is_genuine(web.profile_seed, inst.sigil):
            problems.append(
                Problem(K_INSTANCE_NOT_MINTED, inst.node_id, f"live sigil {inst.sigil.id!r} is not a mint of this profile")
            )
        problems.extend(_check_instance_against_ledger(state))

    # --- stash and rewards -----------------------------------------------
    spent = {e.sigil_id for e in state.history if e.sigil_id is not None}
    for key in sorted(state.stash):
        if key != state.stash[key].id:
            problems.append(Problem(K_STASH_KEY, None, f"stash key {key!r} holds sigil {state.stash[key].id!r}"))
        if not is_genuine(web.profile_seed, state.stash[key]):
            problems.append(Problem(K_STASH_NOT_MINTED, None, f"stashed sigil {key!r} is not a mint of this profile"))
        if key in spent:
            problems.append(Problem(K_STASH_SPENT, None, f"stashed sigil {key!r} was already spent by the ledger"))
    if state.passive_points < 0:
        problems.append(Problem(K_PASSIVE_NEGATIVE, None, f"passive points are {state.passive_points}"))
    for pinnacle in sorted(Pinnacle, key=lambda p: p.value):
        count = state.fragments.get(pinnacle, 0)
        unlocked = pinnacle in state.unlocked_pinnacles
        if count < 0:
            problems.append(Problem(K_FRAGMENTS_NEGATIVE, None, f"{pinnacle.value} has {count} fragments"))
        if unlocked and count < FRAGMENTS_TO_UNLOCK:
            problems.append(
                Problem(K_PINNACLE_EARLY, None, f"{pinnacle.value} is unlocked with {count} of {FRAGMENTS_TO_UNLOCK} fragments")
            )
        if not unlocked and count >= FRAGMENTS_TO_UNLOCK:
            problems.append(
                Problem(K_PINNACLE_LATE, None, f"{pinnacle.value} has {count} fragments but is not unlocked")
            )

    # --- the ledger ------------------------------------------------------
    problems.extend(_check_ledger(state, web_id_set))
    return problems


def _check_instance_against_ledger(state: ProfileState) -> List[Problem]:
    """The live instance is what the ledger's last ``OPEN`` of its node created."""
    problems: List[Problem] = []
    inst = state.instance
    assert inst is not None
    nid = inst.node_id
    opened = next((e for e in reversed(state.history) if e.node_id == nid), None)
    if opened is None or opened.event is not Event.OPEN:
        problems.append(Problem(K_INSTANCE_LEDGER, nid, "the ledger's last event on this node is not an open"))
    else:
        if opened.sigil_id != inst.sigil.id:
            problems.append(
                Problem(K_INSTANCE_LEDGER, nid, f"instance sigil {inst.sigil.id!r} but the open at seq {opened.seq} spent {opened.sigil_id!r}")
            )
        if opened.tick != inst.opened_tick:
            problems.append(
                Problem(K_INSTANCE_LEDGER, nid, f"instance opened at tick {inst.opened_tick}, the ledger says {opened.tick}")
            )
        if opened.has_boss is None or opened.elite_total is None:
            problems.append(Problem(K_INSTANCE_LEDGER, nid, f"the open at seq {opened.seq} records no map facts"))
        elif (opened.has_boss, opened.elite_total) != (inst.has_boss, inst.elite_total):
            problems.append(
                Problem(
                    K_INSTANCE_LEDGER, nid,
                    f"instance says boss {inst.has_boss}, {inst.elite_total} packs; "
                    f"the open at seq {opened.seq} says boss {opened.has_boss}, {opened.elite_total} packs",
                )
            )
    if inst.elite_killed < 0:
        problems.append(Problem(K_INSTANCE_KILLS, nid, f"{inst.elite_killed} elite kills"))
    elif not inst.has_boss and inst.elite_fraction() >= ELITE_CLEAR_FRACTION:
        problems.append(
            Problem(
                K_INSTANCE_KILLS, nid,
                f"{inst.elite_killed}/{inst.elite_total} elite kills on a bossless map is past the threshold, "
                "yet the node is still active",
            )
        )
    return problems


def _check_ledger(state: ProfileState, web_id_set: Set[int]) -> List[Problem]:
    """The ledger is dense, consecutive, and made only of legal transitions.

    Spec: "The transition table in contracts.py is the only legal set of
    moves."  Each entry's ``(before, event)`` must be a row of the table and
    its ``after`` must be that row's result.  A gap is reported once, where
    it opens, rather than at every entry after it; so is the first tick
    that runs backwards.  Every ``OPEN``'s Sigil id must be well formed,
    carry a tier that can open the node, and fund no other ``OPEN``.
    """
    problems: List[Problem] = []
    history = state.history
    if not history:
        return problems
    nodes_by_id: Dict[int, WebNode] = {n.id: n for n in state.web.nodes}
    first = history[0].seq
    if first not in LEDGER_BASES:
        problems.append(Problem(K_LEDGER_BASE, None, f"ledger starts at seq {first}, expected one of {LEDGER_BASES}"))
    previous = first - 1
    last_tick: Optional[int] = None
    tick_reported = False
    opened_by: Dict[str, int] = {}
    for index, entry in enumerate(history):
        if entry.seq != previous + 1:
            problems.append(
                Problem(K_LEDGER_GAP, None, f"entry {index} has seq {entry.seq}, expected {previous + 1}")
            )
        previous = entry.seq
        if last_tick is not None and entry.tick < last_tick and not tick_reported:
            problems.append(
                Problem(K_LEDGER_TICK_ORDER, None, f"entry seq {entry.seq} has tick {entry.tick}, after tick {last_tick}")
            )
            tick_reported = True
        last_tick = entry.tick
        if entry.event is Event.OPEN and entry.sigil_id is not None:
            sid = entry.sigil_id
            if sid in opened_by:
                problems.append(
                    Problem(K_LEDGER_SIGIL_REUSED, entry.node_id, f"entry seq {entry.seq} spends sigil {sid!r} again, first spent at seq {opened_by[sid]}")
                )
            else:
                opened_by[sid] = entry.seq
            try:
                sigil_tier = tier_from_id(sid)
            except ValueError as exc:
                problems.append(Problem(K_LEDGER_SIGIL_MALFORMED, entry.node_id, f"entry seq {entry.seq}: {exc}"))
            else:
                node = nodes_by_id.get(entry.node_id)
                if node is not None and sigil_tier < node.tier:
                    problems.append(
                        Problem(K_LEDGER_SIGIL_WEAK, entry.node_id, f"entry seq {entry.seq}: sigil {sid!r} is tier {sigil_tier}, node is tier {node.tier}")
                    )
        if entry.node_id not in web_id_set:
            problems.append(Problem(K_LEDGER_UNKNOWN_NODE, entry.node_id, f"entry seq {entry.seq} names a node not in the web"))
        row = TRANSITIONS.get((entry.before, entry.event))
        if row is None:
            problems.append(
                Problem(
                    K_LEDGER_ILLEGAL,
                    entry.node_id,
                    f"entry seq {entry.seq}: {entry.event.value} is not allowed while {entry.before.value}",
                )
            )
        elif row is not entry.after:
            problems.append(
                Problem(
                    K_LEDGER_ILLEGAL,
                    entry.node_id,
                    f"entry seq {entry.seq}: {entry.before.value} + {entry.event.value} leads to "
                    f"{row.value}, ledger says {entry.after.value}",
                )
            )
    return problems


# --------------------------------------------------------------------------
# check_replay
# --------------------------------------------------------------------------

#: Anything with ``replay(profile_id, web, history) -> ProfileState``.
ReplayEngine = object
EngineFactory = Callable[[], ReplayEngine]


def _resolve_engine(engine_factory: Optional[object]) -> object:
    """Accept a factory, an engine object, or ``None`` for :class:`DescentEngine`.

    ``DescentEngine`` itself is both callable and a replayer, so a caller may
    pass the class, a zero-argument factory returning it, or any object of
    their own that exposes ``replay(profile_id, web, history)``.
    """
    if engine_factory is None:
        return DescentEngine
    if hasattr(engine_factory, "replay"):
        return engine_factory
    return engine_factory()


def check_replay(
    state: ProfileState,
    engine_factory: Optional[EngineFactory] = None,
    *,
    map_probe: Optional[Callable[[str, int, int], MapProbe]] = None,
) -> List[Problem]:
    """Rebuild the profile from its web and ledger and compare.

    Spec: the ledger exists "so a profile can be audited or replayed".  A
    fresh profile is built on the same web and the recorded entries are
    replayed through the engine, which holds every entry to the rules the
    live engine applied and to the facts the ledger recorded (the ``OPEN``
    entry carries what the probe said), so no map probe is needed to
    replay.  Then the ``states`` dict must be identical, and so must the
    passive points, fragments and unlocked Pinnacles the engine promises to
    rebuild.

    Ways the ledger can fall short that are reported rather than skipped:

    * an ``OPEN`` or Sigil-consuming entry with no ``sigil_id``, or an
      ``OPEN`` with no probe facts, lacks a fact the audit needs
      (``ledger_missing_fact``);
    * the engine refuses the ledger -- wrong ``before``, illegal move, a
      node the web lacks, a propagation out of order, an arena opened while
      locked, a Sigil spent twice, a clear the map does not allow -- which
      surfaces as ``replay_error`` with the engine's reason;
    * with a ``map_probe``, each ``OPEN``'s recorded facts are re-derived:
      the Sigil that funded it is recovered from its id
      (:func:`sigils.unmint`, then :func:`sigils.mint_sigil` for its seed)
      and the probe is asked again about that template, seed and tier
      (``ledger_fact_mismatch``).  A malformed id is ``check_state``'s
      finding and is skipped here.
    """
    problems: List[Problem] = []
    for entry in state.history:
        if (entry.event is Event.OPEN or entry.event in SIGIL_CONSUMING_EVENTS) and entry.sigil_id is None:
            problems.append(
                Problem(
                    K_LEDGER_MISSING_FACT,
                    entry.node_id,
                    f"entry seq {entry.seq}: {entry.event.value} records no sigil id",
                )
            )
        if entry.event is Event.OPEN and (entry.has_boss is None or entry.elite_total is None):
            problems.append(
                Problem(K_LEDGER_MISSING_FACT, entry.node_id, f"entry seq {entry.seq}: open records no map facts")
            )
    if map_probe is not None:
        problems.extend(_check_ledger_facts(state, map_probe))

    engine = _resolve_engine(engine_factory)
    try:
        rebuilt = engine.replay(state.profile_id, state.web, list(state.history))
    except Exception as exc:  # the engine's refusal is the finding
        problems.append(Problem(K_REPLAY_ERROR, None, f"{type(exc).__name__}: {exc}"))
        return problems

    for nid in sorted(set(state.states) | set(rebuilt.states)):
        mine = state.states.get(nid)
        theirs = rebuilt.states.get(nid)
        if mine is not theirs:
            problems.append(
                Problem(
                    K_REPLAY_MISMATCH,
                    nid,
                    f"profile says {getattr(mine, 'value', None)}, replay says {getattr(theirs, 'value', None)}",
                )
            )
    if rebuilt.passive_points != state.passive_points:
        problems.append(
            Problem(K_REPLAY_PASSIVE, None, f"profile has {state.passive_points} passive points, replay {rebuilt.passive_points}")
        )
    for pinnacle in sorted(Pinnacle, key=lambda p: p.value):
        mine_count = state.fragments.get(pinnacle, 0)
        theirs_count = rebuilt.fragments.get(pinnacle, 0)
        if mine_count != theirs_count:
            problems.append(
                Problem(K_REPLAY_FRAGMENTS, None, f"{pinnacle.value}: profile has {mine_count} fragments, replay {theirs_count}")
            )
    if frozenset(rebuilt.unlocked_pinnacles) != frozenset(state.unlocked_pinnacles):
        problems.append(
            Problem(
                K_REPLAY_UNLOCKED,
                None,
                f"profile unlocked {sorted(p.value for p in state.unlocked_pinnacles)}, "
                f"replay {sorted(p.value for p in rebuilt.unlocked_pinnacles)}",
            )
        )
    if list(rebuilt.history) != list(state.history):
        length = min(len(rebuilt.history), len(state.history))
        where = next((i for i in range(length) if rebuilt.history[i] != state.history[i]), length)
        problems.append(
            Problem(K_REPLAY_LEDGER, None, f"replayed ledger differs from the profile's at index {where}")
        )
    return problems


def _check_ledger_facts(
    state: ProfileState, map_probe: Callable[[str, int, int], MapProbe]
) -> List[Problem]:
    """Every ``OPEN``'s recorded probe facts are what the probe says today."""
    problems: List[Problem] = []
    nodes_by_id: Dict[int, WebNode] = {n.id: n for n in state.web.nodes}
    for entry in state.history:
        if entry.event is not Event.OPEN or entry.sigil_id is None:
            continue
        if entry.has_boss is None or entry.elite_total is None:
            continue  # reported as a missing fact already
        node = nodes_by_id.get(entry.node_id)
        if node is None:
            continue
        try:
            counter, tier = unmint(state.web.profile_seed, entry.sigil_id)
        except ValueError:
            continue  # check_state reports the malformed id
        sigil = mint_sigil(state.web.profile_seed, counter, tier)
        probe = map_probe(node.template, sigil.seed, sigil.tier)
        if (probe.has_boss, probe.elite_total) != (entry.has_boss, entry.elite_total):
            problems.append(
                Problem(
                    K_LEDGER_FACT_MISMATCH,
                    entry.node_id,
                    f"entry seq {entry.seq} records boss {entry.has_boss}, {entry.elite_total} packs; "
                    f"the probe says boss {probe.has_boss}, {probe.elite_total} packs for sigil {entry.sigil_id!r}",
                )
            )
    return problems


# --------------------------------------------------------------------------
# simulate: a deterministic random walk of legal play
# --------------------------------------------------------------------------


class _Clock:
    """The game clock the engine reads; the walk advances it once per step."""

    __slots__ = ("now",)

    def __init__(self) -> None:
        self.now = 0

    def read(self) -> int:
        return self.now


def stub_map_probe(template: str, map_seed: int, sigil_tier: int) -> MapProbe:
    """A map probe that never runs the generator.

    Deterministic in its three inputs: the stream is opened on the *map*
    seed under a ``spawn.`` label (the tile/spawn field of the seed, per
    ``seed.py``) that carries the template and Sigil tier, so the same
    portal always answers the same way and a different Sigil tier -- which
    "feeds spawn density" -- answers differently.
    """
    stream = SeedFields.parse(map_seed).stream(f"spawn.probe:{template}:{sigil_tier}")
    has_boss = stream.chance(STUB_BOSS_CHANCE)
    elite_total = stream.randint(STUB_ELITES_MIN, STUB_ELITES_MAX)
    return MapProbe(has_boss=has_boss, elite_total=elite_total)


def _usable_sigils(state: ProfileState, node: WebNode) -> List[Sigil]:
    """Stash Sigils that may open ``node`` (``sigil.tier >= node.tier``), by id."""
    return [state.stash[key] for key in sorted(state.stash) if state.stash[key].can_open(node)]


def simulate(
    profile_seed: int,
    steps: int,
    rng_label: str,
    *,
    web: Optional[Web] = None,
    end_active: bool = False,
    map_probe: Optional[Callable[[str, int, int], MapProbe]] = None,
) -> ProfileState:
    """Play ``steps`` legal turns on a fresh profile and return its state.

    Each step: pick an openable node (``REACHABLE`` or ``FAILED``, and not a
    locked arena); with :data:`P_PUSH_OUTWARD` restrict the pick to the
    highest tier on offer so the walk actually reaches tier 15; find a
    stash Sigil that can open it or mint one of tier ``node.tier`` up to
    ``node.tier + SIGIL_TIER_HEADROOM`` (never above 15); open the portal;
    then resolve it with fixed probabilities -- die, abandon, time out
    (spec: all three "behave like death"), or clear by boss kill or by
    reporting elite kills until the 80 percent threshold trips.  A clear
    rolls the sustain drops into the stash via :func:`sigils.roll_drops`.

    Every call the walk makes is legal by construction: the node is chosen
    from those the engine will accept, the Sigil is minted to fit, the
    instance is resolved before the next open, and elite kills are reported
    only until the engine says the node cleared.  No exception is caught.

    ``end_active`` leaves the last step's portal open so the returned state
    carries a live instance.  ``web`` may supply the profile's web when the
    caller already generated it (it must have the same profile seed).
    ``map_probe`` defaults to :func:`stub_map_probe`; pass
    :func:`lucifer_descent.engine.default_map_probe` to drive the real
    generator.  The walk stops early, legally, if nothing is openable.

    The whole walk is a function of ``(profile_seed, steps, rng_label,
    end_active)`` and the probe: the walk's draws come from one stream
    labelled ``rng_label`` on the profile seed, Sigils from
    :func:`sigils.mint_sigil` under a running counter, drops from streams
    labelled ``<rng_label>.drop:<seq>``.
    """
    if isinstance(profile_seed, bool) or not isinstance(profile_seed, int):
        raise TypeError(f"profile_seed must be an int, got {type(profile_seed).__name__}")
    if steps < 0:
        raise ValueError(f"steps must be non-negative, got {steps}")
    profile_seed &= MASK64
    if web is None:
        web = generate_web(profile_seed)
    elif web.profile_seed != profile_seed:
        raise ValueError(
            f"web was generated from {format_seed(web.profile_seed)}, not {format_seed(profile_seed)}"
        )
    probe = stub_map_probe if map_probe is None else map_probe

    profile_id = f"sim:{format_seed(profile_seed)}"
    state = DescentEngine.new_profile(profile_id, web, tick=0)
    clock = _Clock()
    engine = DescentEngine(state, probe, clock.read)
    walk: Stream = SeedFields.parse(profile_seed).stream(rng_label)
    nodes_by_id: Dict[int, WebNode] = {n.id: n for n in web.nodes}
    counter = 0  # the profile's mint counter; advanced by every mint

    for step in range(steps):
        clock.now = step + 1
        candidates = [nid for nid in sorted(state.states) if node_is_openable(state, nodes_by_id[nid])]
        if not candidates:
            break
        top = max(nodes_by_id[nid].tier for nid in candidates)
        if walk.chance(P_PUSH_OUTWARD):
            pool = [nid for nid in candidates if nodes_by_id[nid].tier == top]
        else:
            pool = candidates
        node = nodes_by_id[walk.choice(pool)]

        usable = _usable_sigils(state, node)
        if usable:
            sigil = walk.choice(usable)
        else:
            tier = min(MAX_TIER, max(1, node.tier) + walk.randint(0, SIGIL_TIER_HEADROOM))
            sigil = mint_sigil(profile_seed, counter, tier)
            counter += 1
            engine.add_sigil(profile_id, sigil)

        engine.open_portal(profile_id, node.id, sigil.id)
        if end_active and step == steps - 1:
            break

        instance = state.instance
        assert instance is not None  # open_portal just created it
        roll = walk.random()
        if roll < P_DIE:
            fail = engine.report_death
        elif roll < P_DIE + P_ABANDON:
            fail = engine.report_abandon
        elif roll < P_DIE + P_ABANDON + P_TIMEOUT:
            fail = engine.report_timeout
        else:
            fail = None

        cleared = False
        if fail is None:
            if instance.has_boss:
                engine.report_boss_kill(profile_id)
                cleared = True
            else:
                # Spec: "80 percent of elite packs dead when the map has no
                # boss."  By the last pack the fraction is 1.0, so this loop
                # always ends in a clear within elite_total reports.
                for _ in range(instance.elite_total + 1):
                    if engine.report_elite_kill(profile_id) is NodeState.CLEARED:
                        cleared = True
                        break
                assert cleared, "an elites-only map must clear by its last pack"
        else:
            # Some packs may fall before the run fails; on a bossless map
            # that can tip the node over the threshold first, which is a
            # legal clear rather than a failure.
            kills = walk.randint(0, instance.elite_total)
            for _ in range(kills):
                if engine.report_elite_kill(profile_id) is NodeState.CLEARED:
                    cleared = True
                    break
            if not cleared:
                fail(profile_id)

        if cleared:
            seq = state.history[-1].seq
            drops = roll_drops(profile_seed, counter, node.tier, f"{rng_label}.drop:{seq}")
            counter += len(drops)
            for drop in drops:
                engine.add_sigil(profile_id, drop)
    return state


# --------------------------------------------------------------------------
# run_suite
# --------------------------------------------------------------------------


@dataclass
class SuiteReport:
    """What :func:`run_suite` found over many profiles.

    ``problem_counts`` is keyed by :attr:`Problem.kind`; ``first_failure``
    holds every problem of the first seed that had any, so one report is
    enough to start debugging.  Only counters and that one list are kept:
    states are checked and dropped one at a time.
    """

    n_profiles: int
    start_seed: int
    steps_per_profile: int
    profiles_run: int = 0
    profiles_failed: int = 0
    problem_counts: Dict[str, int] = field(default_factory=dict)
    first_failure_seed: Optional[int] = None
    first_failure: List[Problem] = field(default_factory=list)
    nodes_checked: int = 0
    ledger_entries: int = 0
    profiles_with_instance: int = 0
    profiles_with_unlocked_pinnacle: int = 0

    @property
    def ok(self) -> bool:
        return self.profiles_run == self.n_profiles and self.profiles_failed == 0

    def summary(self) -> str:
        lines = [
            f"gate: {self.profiles_run}/{self.n_profiles} profiles from {format_seed(self.start_seed)}, "
            f"{self.steps_per_profile} steps each: {'OK' if self.ok else 'FAILED'}",
            f"  nodes checked {self.nodes_checked}, ledger entries replayed {self.ledger_entries}, "
            f"profiles ending with a live instance {self.profiles_with_instance}, "
            f"with an unlocked Pinnacle {self.profiles_with_unlocked_pinnacle}",
        ]
        for kind in sorted(self.problem_counts):
            lines.append(f"  {kind}: {self.problem_counts[kind]}")
        if self.first_failure_seed is not None:
            lines.append(f"  first failing seed {format_seed(self.first_failure_seed)}:")
            for problem in self.first_failure[:20]:
                lines.append(f"    {problem}")
            if len(self.first_failure) > 20:
                lines.append(f"    ... and {len(self.first_failure) - 20} more")
        return "\n".join(lines)


def run_suite(
    n_profiles: int,
    start_seed: int,
    steps_per_profile: int = 200,
    *,
    rng_label: str = "suite.walk",
    db_path: Optional[str] = None,
) -> SuiteReport:
    """Generate, play, check, replay and round-trip ``n_profiles`` profiles.

    Seeds are ``start_seed, start_seed + 1, ...`` masked to 64 bits.  For
    each: :func:`generate_web` then :func:`check_web`; :func:`simulate`
    (every third profile ends with a live instance so the instance path is
    covered); :func:`check_state`; :func:`check_replay` with the stub probe,
    so every recorded map fact is re-derived; then a
    :class:`SqliteStore` save/load on ``db_path`` (a temporary file by
    default) compared with :func:`round_trip_equal`.  An exception in any
    phase is recorded as ``exception:<phase>`` and the later phases that
    needed its result are skipped, so one bad seed never stops the suite.

    Profiles are processed and discarded one at a time; the report holds
    counters and the first failing seed's problems only.
    """
    if n_profiles < 0:
        raise ValueError(f"n_profiles must be non-negative, got {n_profiles}")
    if db_path is None:
        with tempfile.TemporaryDirectory(prefix="lucifer-gate-") as tmp:
            return _run_suite(n_profiles, start_seed, steps_per_profile, rng_label, os.path.join(tmp, "gate.sqlite3"))
    return _run_suite(n_profiles, start_seed, steps_per_profile, rng_label, db_path)


def _run_suite(n_profiles: int, start_seed: int, steps: int, rng_label: str, db_path: str) -> SuiteReport:
    report = SuiteReport(n_profiles=n_profiles, start_seed=start_seed & MASK64, steps_per_profile=steps)
    with SqliteStore(db_path) as store:
        for index in range(n_profiles):
            seed = (start_seed + index) & MASK64
            problems = _check_one_profile(seed, steps, rng_label, index % 3 == 2, store, report)
            report.profiles_run += 1
            if problems:
                report.profiles_failed += 1
                for problem in problems:
                    report.problem_counts[problem.kind] = report.problem_counts.get(problem.kind, 0) + 1
                if report.first_failure_seed is None:
                    report.first_failure_seed = seed
                    report.first_failure = list(problems)
    return report


def _phase(problems: List[Problem], phase: str, action: Callable[[], object]) -> Tuple[bool, object]:
    """Run one phase, turning an exception into a problem.  Returns (ok, result)."""
    try:
        return True, action()
    except Exception as exc:
        problems.append(Problem(f"{K_EXCEPTION}:{phase}", None, f"{type(exc).__name__}: {exc}"))
        return False, None


def _check_one_profile(
    seed: int, steps: int, rng_label: str, end_active: bool, store: SqliteStore, report: SuiteReport
) -> List[Problem]:
    problems: List[Problem] = []

    ok, web = _phase(problems, "generate_web", lambda: generate_web(seed))
    if not ok:
        return problems
    report.nodes_checked += len(web.nodes)
    _phase(problems, "check_web", lambda: problems.extend(check_web(web)))

    ok, state = _phase(
        problems, "simulate", lambda: simulate(seed, steps, rng_label, web=web, end_active=end_active)
    )
    if not ok:
        return problems
    report.ledger_entries += len(state.history)
    if state.instance is not None:
        report.profiles_with_instance += 1
    if state.unlocked_pinnacles:
        report.profiles_with_unlocked_pinnacle += 1

    _phase(problems, "check_state", lambda: problems.extend(check_state(state)))
    _phase(
        problems, "check_replay",
        lambda: problems.extend(check_replay(state, map_probe=stub_map_probe)),
    )

    def round_trip() -> None:
        # Saved under the gate's own id, and never over an existing profile:
        # a caller's ``db_path`` may hold profiles of its own.
        copy_ = dataclasses.replace(state, profile_id=f"gate:{format_seed(seed)}")
        if store.load(copy_.profile_id) is not None:
            problems.append(
                Problem(K_STORE_ROUND_TRIP, None, f"profile {copy_.profile_id!r} already exists in the store")
            )
            return
        store.save(copy_)
        loaded = store.load(copy_.profile_id)
        store.delete(copy_.profile_id)
        if loaded is None:
            problems.append(Problem(K_STORE_ROUND_TRIP, None, "profile did not come back from the store"))
        elif not round_trip_equal(copy_, loaded):
            problems.append(Problem(K_STORE_ROUND_TRIP, None, f"loaded state differs at {first_difference(copy_, loaded)}"))

    _phase(problems, "store_round_trip", round_trip)
    return problems


# --------------------------------------------------------------------------
# Self-check entry point: python3 -m lucifer_descent.validate [n] [seed] [steps]
# --------------------------------------------------------------------------


def _main(argv: Sequence[str]) -> int:
    n_profiles = int(argv[1]) if len(argv) > 1 else 10
    start_seed = int(argv[2], 0) if len(argv) > 2 else 0x5EED
    steps = int(argv[3]) if len(argv) > 3 else 200
    report = run_suite(n_profiles, start_seed, steps)
    print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
