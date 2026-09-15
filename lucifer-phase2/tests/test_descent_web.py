"""Tests for the Descent web generator.

Spec: docs/WORLD_BIBLE.md section 03. The structural invariants (tier equals
graph distance, planarity, connectivity, outer-ring Pinnacles) are checked
over several hundred profile seeds; the geometry helpers get hand-built
cases.
"""

from __future__ import annotations

import math
import pathlib
import sys
from collections import Counter
from typing import Dict, List

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lucifer_descent.contracts import (  # noqa: E402
    MAX_TIER,
    Mechanic,
    Pinnacle,
    Web,
    WebEdge,
    WebNode,
)
from lucifer_descent import web as W  # noqa: E402
from lucifer_gen.seed import MASK64  # noqa: E402
from lucifer_gen.template import builtin_template_names  # noqa: E402

#: 300 small seeds plus a few with every bit pattern in play.
SEEDS = tuple(range(300)) + (
    0xDEADBEEFCAFEF00D,
    MASK64,
    0x8000000000000000,
    12345678901234567,
)

OUTER = MAX_TIER
OUTER_FIRST_ID = W.first_id_of_ring(OUTER)
OUTER_SIZE = W.ring_size(OUTER)


@pytest.fixture(scope="module")
def webs() -> Dict[int, Web]:
    return {seed: W.generate_web(seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def degrees(webs: Dict[int, Web]) -> Dict[int, Dict[int, int]]:
    return {
        seed: {node_id: len(nb) for node_id, nb in W.adjacency_of(web).items()}
        for seed, web in webs.items()
    }


def _cyclic_gaps(indices: List[int], count: int) -> List[int]:
    """Gaps between consecutive ring indices going round the ring once."""
    ordered = sorted(indices)
    return [
        (ordered[(i + 1) % len(ordered)] - ordered[i]) % count
        for i in range(len(ordered))
    ]


# --------------------------------------------------------------------------
# Structure over many seeds
# --------------------------------------------------------------------------


class TestStructure:
    def test_tier_equals_bfs_distance(self, webs):
        for seed, web in webs.items():
            dist = W.bfs_tiers(web)
            for node in web.nodes:
                assert dist.get(node.id) == node.tier, (seed, node.id)

    def test_connected(self, webs):
        for seed, web in webs.items():
            assert len(W.bfs_tiers(web)) == len(web.nodes), seed

    def test_planar_layout(self, webs):
        for seed, web in webs.items():
            assert W.find_crossing(web) is None, seed
            assert W.is_planar_layout(web), seed

    def test_outer_ring_is_tier_15_and_nothing_beyond(self, webs):
        for seed, web in webs.items():
            assert len(web.nodes) == W.node_count(OUTER), seed
            assert max(n.tier for n in web.nodes) == OUTER, seed
            outer_ids = [n.id for n in web.nodes if n.id >= OUTER_FIRST_ID]
            assert outer_ids == list(range(OUTER_FIRST_ID, OUTER_FIRST_ID + OUTER_SIZE))
            for node in web.nodes:
                assert (node.tier == OUTER) == (node.id >= OUTER_FIRST_ID), (seed, node.id)
            assert len(web.nodes_at_tier(OUTER)) == OUTER_SIZE == 6 + 2 * OUTER

    def test_ring_sizes_and_dense_ids(self, webs):
        for seed, web in webs.items():
            assert [n.id for n in web.nodes] == list(range(len(web.nodes))), seed
            assert web.origin_id == 0 and web.nodes[0].tier == 0
            for ring in range(1, OUTER + 1):
                on_ring = web.nodes_at_tier(ring)
                assert len(on_ring) == 6 + 2 * ring, (seed, ring)
                assert [n.ring_index for n in on_ring] == list(range(len(on_ring)))
                assert [n.id for n in on_ring] == list(
                    range(W.first_id_of_ring(ring), W.first_id_of_ring(ring + 1))
                )

    def test_edges_are_canonical(self, webs):
        for seed, web in webs.items():
            ids = {n.id for n in web.nodes}
            pairs = [(e.a, e.b) for e in web.edges]
            assert pairs == sorted(pairs), seed
            assert len(set(pairs)) == len(pairs), seed
            for a, b in pairs:
                assert a < b and a in ids and b in ids, (seed, a, b)

    def test_edges_join_same_or_adjacent_rings_only(self, webs):
        for seed, web in webs.items():
            tier = {n.id: n.tier for n in web.nodes}
            for e in web.edges:
                assert abs(tier[e.a] - tier[e.b]) <= 1, (seed, e)

    def test_every_node_keeps_an_inner_link(self, webs):
        for seed, web in webs.items():
            tier = {n.id: n.tier for n in web.nodes}
            adj = W.adjacency_of(web)
            for node in web.nodes:
                if node.id == web.origin_id:
                    continue
                assert any(tier[v] == node.tier - 1 for v in adj[node.id]), (seed, node.id)

    def test_no_node_below_min_degree(self, degrees):
        for seed, per_node in degrees.items():
            assert min(per_node.values()) >= W.MIN_DEGREE, seed
            assert min(per_node.values()) >= 2, seed

    def test_layout_radius_matches_tier(self, webs):
        for seed, web in webs.items():
            for node in web.nodes:
                radius = math.hypot(node.x, node.y)
                assert abs(radius - W.RING_SPACING * node.tier) < 0.01, (seed, node.id)

    def test_layout_positions_distinct(self, webs):
        for seed, web in webs.items():
            positions = {(n.x, n.y) for n in web.nodes}
            assert len(positions) == len(web.nodes), seed


# --------------------------------------------------------------------------
# Pinnacles and glyphs
# --------------------------------------------------------------------------


class TestPinnacles:
    def test_one_arena_and_enough_glyphs_per_pinnacle(self, webs):
        for seed, web in webs.items():
            for pinnacle in Pinnacle:
                arenas = [n for n in web.nodes if n.pinnacle is pinnacle]
                glyphs = [n for n in web.nodes if n.glyph is pinnacle]
                assert len(arenas) == 1, (seed, pinnacle)
                assert len(glyphs) >= 4, (seed, pinnacle)
                assert all(n.tier == OUTER for n in arenas + glyphs), (seed, pinnacle)

    def test_arenas_never_bear_glyphs(self, webs):
        for seed, web in webs.items():
            for node in web.nodes:
                if node.pinnacle is not None:
                    assert node.glyph is None, (seed, node.id)

    def test_pinnacle_marks_only_on_outer_ring(self, webs):
        for seed, web in webs.items():
            for node in web.nodes:
                if node.tier != OUTER:
                    assert node.pinnacle is None and node.glyph is None, (seed, node.id)

    def test_arenas_carry_no_mechanic(self, webs):
        for seed, web in webs.items():
            for node in web.nodes:
                if node.pinnacle is not None:
                    assert node.mechanic is None, (seed, node.id)

    def test_glyphs_are_spaced_around_the_ring(self, webs):
        for seed, web in webs.items():
            for pinnacle in Pinnacle:
                idx = [n.ring_index for n in web.nodes if n.glyph is pinnacle]
                gaps = _cyclic_gaps(idx, OUTER_SIZE)
                assert min(gaps) >= 3, (seed, pinnacle, idx)
                assert max(gaps) <= OUTER_SIZE // 2, (seed, pinnacle, idx)

    def test_arenas_sit_apart(self, webs):
        for seed, web in webs.items():
            idx = [n.ring_index for n in web.nodes if n.pinnacle is not None]
            assert min(_cyclic_gaps(idx, OUTER_SIZE)) >= OUTER_SIZE // 4, (seed, idx)

    def test_arena_positions_vary_with_seed(self, webs):
        arenas = {
            next(n.ring_index for n in web.nodes if n.pinnacle is Pinnacle.ARBITER)
            for web in webs.values()
        }
        assert len(arenas) > OUTER_SIZE // 2


# --------------------------------------------------------------------------
# Determinism and variety
# --------------------------------------------------------------------------


class TestDeterminism:
    def test_same_seed_same_web(self, webs):
        for seed, web in webs.items():
            again = W.generate_web(seed)
            assert again.edges == web.edges, seed
            assert len(again.edges) == len(web.edges), seed
            assert again.nodes == web.nodes, seed
            assert again == web, seed

    def test_reproducible_from_stored_seed(self, webs):
        for seed, web in webs.items():
            assert W.generate_web(web.profile_seed) == web, seed

    def test_seed_is_masked_to_64_bits(self):
        assert W.generate_web(5) == W.generate_web(5 + (1 << 64))
        assert W.generate_web(5).profile_seed == 5
        assert W.generate_web(MASK64 + 6).profile_seed == 5

    def test_different_seeds_give_different_edge_sets(self, webs):
        seeds = list(SEEDS)
        for first, second in zip(seeds, seeds[1:]):
            assert set(webs[first].edges) != set(webs[second].edges), (first, second)
            assert [(n.x, n.y) for n in webs[first].nodes] != [
                (n.x, n.y) for n in webs[second].nodes
            ]

    def test_edge_count_varies_across_seeds(self, webs):
        assert len({len(web.edges) for web in webs.values()}) > 1

    def test_thinning_keeps_edge_count_in_expected_band(self, webs):
        # 330 same-ring edges before thinning, about a quarter dropped, and
        # the 330 inner links always kept.
        inner = W.node_count(OUTER) - 1
        for seed, web in webs.items():
            same_ring = len(web.edges) - inner
            assert 0.55 * inner <= same_ring <= 0.9 * inner, (seed, same_ring)


# --------------------------------------------------------------------------
# Mechanics and templates
# --------------------------------------------------------------------------


class TestMechanicsAndTemplates:
    def test_mechanic_rate_in_aggregate(self, webs):
        carrying = sum(1 for web in webs.values() for n in web.nodes if n.mechanic)
        total = sum(len(web.nodes) for web in webs.values())
        rate = carrying / total
        assert 0.25 <= rate <= 0.45, rate

    def test_every_mechanic_kind_is_used_evenly(self, webs):
        kinds = Counter(n.mechanic for web in webs.values() for n in web.nodes if n.mechanic)
        assert set(kinds) == set(Mechanic)
        low, high = min(kinds.values()), max(kinds.values())
        assert high <= 1.25 * low, kinds

    def test_mechanic_is_a_single_enum_member(self, webs):
        for web in webs.values():
            for node in web.nodes:
                assert node.mechanic is None or isinstance(node.mechanic, Mechanic)

    def test_templates_are_shipped_generator_templates(self, webs):
        shipped = set(builtin_template_names())
        for _, _, names in W.TEMPLATE_BANDS:
            assert set(names) <= shipped, names
        for web in webs.values():
            for node in web.nodes:
                assert node.template in shipped, node

    def test_template_bands_cover_every_tier_once(self):
        covered = Counter()
        for low, high, _ in W.TEMPLATE_BANDS:
            for tier in range(low, high + 1):
                covered[tier] += 1
        assert all(covered[t] == 1 for t in range(0, MAX_TIER + 1)), covered

    def test_template_follows_tier_band(self, webs):
        web = next(iter(webs.values()))
        for node in web.nodes:
            assert node.template == W.template_for(node.tier, node.ring_index)
        assert {n.template for n in web.nodes_at_tier(1)} == {"crypt"}
        assert {n.template for n in web.nodes_at_tier(OUTER)} == {"ashen_ramparts"}
        assert {n.template for n in web.nodes_at_tier(7)} == {"crypt", "ashen_ramparts"}


# --------------------------------------------------------------------------
# Argument handling and small webs
# --------------------------------------------------------------------------


class TestArguments:
    @pytest.mark.parametrize("rings", [0, 1, MAX_TIER + 1, 40])
    def test_rings_out_of_range(self, rings):
        with pytest.raises(ValueError):
            W.generate_web(1, rings)

    @pytest.mark.parametrize("bad", ["12", 1.5, None, True])
    def test_seed_must_be_int(self, bad):
        with pytest.raises(TypeError):
            W.generate_web(bad)  # type: ignore[arg-type]

    @pytest.mark.parametrize("rings", [2, 3, 5, 9])
    def test_small_webs_keep_the_invariants(self, rings):
        for seed in range(20):
            web = W.generate_web(seed, rings)
            dist = W.bfs_tiers(web)
            assert len(dist) == len(web.nodes) == W.node_count(rings)
            assert all(dist[n.id] == n.tier for n in web.nodes)
            assert max(n.tier for n in web.nodes) == rings
            assert W.is_planar_layout(web)
            adj = W.adjacency_of(web)
            assert min(len(v) for v in adj.values()) >= W.MIN_DEGREE
            for pinnacle in Pinnacle:
                assert sum(1 for n in web.nodes if n.pinnacle is pinnacle) == 1
                assert sum(1 for n in web.nodes if n.glyph is pinnacle) >= 4
            for node in web.nodes:
                if node.pinnacle is not None or node.glyph is not None:
                    assert node.tier == rings
                    assert not (node.pinnacle is not None and node.glyph is not None)

    def test_ring_bookkeeping(self):
        assert W.ring_size(0) == 1
        assert [W.ring_size(r) for r in (1, 2, 15)] == [8, 10, 36]
        assert W.first_id_of_ring(0) == 0
        assert W.first_id_of_ring(1) == 1
        assert W.first_id_of_ring(2) == 9
        assert W.node_count(15) == 331
        with pytest.raises(ValueError):
            W.ring_size(-1)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


class TestSegmentsCross:
    def test_proper_crossing(self):
        assert W.segments_cross((0, 0), (2, 2), (0, 2), (2, 0))

    def test_parallel_apart(self):
        assert not W.segments_cross((0, 0), (2, 0), (0, 1), (2, 1))

    def test_skew_apart(self):
        assert not W.segments_cross((0, 0), (1, 1), (2, 0), (3, 1))
        # The line x + y = 3 meets the diagonal at (1.5, 1.5), past this end.
        assert not W.segments_cross((0, 0), (1, 1), (3, 0), (0, 3))
        assert W.segments_cross((0, 0), (2, 2), (3, 0), (0, 3))

    def test_touch_at_endpoint_counts(self):
        assert W.segments_cross((0, 0), (2, 0), (2, 0), (3, 5))

    def test_t_junction_counts(self):
        assert W.segments_cross((0, 0), (4, 0), (2, -1), (2, 0))
        assert W.segments_cross((0, 0), (4, 0), (2, 0), (2, 3))

    def test_collinear_overlap_counts(self):
        assert W.segments_cross((0, 0), (3, 0), (2, 0), (5, 0))

    def test_collinear_disjoint(self):
        assert not W.segments_cross((0, 0), (1, 0), (2, 0), (3, 0))

    def test_symmetric(self):
        cases = [
            ((0, 0), (2, 2), (0, 2), (2, 0)),
            ((0, 0), (1, 0), (2, 0), (3, 0)),
            ((0, 0), (4, 0), (2, -1), (2, 0)),
        ]
        for a, b, c, d in cases:
            assert W.segments_cross(a, b, c, d) == W.segments_cross(c, d, a, b)
            assert W.segments_cross(a, b, c, d) == W.segments_cross(b, a, d, c)


def _tiny_web(edges, positions) -> Web:
    nodes = tuple(
        WebNode(id=i, tier=0, ring_index=0, template="crypt", x=x, y=y)
        for i, (x, y) in enumerate(positions)
    )
    return Web(
        profile_seed=0,
        origin_id=0,
        nodes=nodes,
        edges=tuple(WebEdge(a, b) for a, b in edges),
    )


class TestGraphQueries:
    def test_bfs_tiers_hand_built(self):
        # 0-1-2-3 chain, 4 hangs off 1, 5 is unreachable.
        web = _tiny_web(
            [(0, 1), (1, 2), (2, 3), (1, 4)],
            [(0, 0), (1, 0), (2, 0), (3, 0), (1, 1), (9, 9)],
        )
        assert W.bfs_tiers(web) == {0: 0, 1: 1, 2: 2, 3: 3, 4: 2}

    def test_bfs_uses_shortest_path(self):
        # A triangle plus a long way round.
        web = _tiny_web(
            [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (0, 4)],
            [(0, 0), (1, 0), (1, 1), (0, 2), (-1, 1)],
        )
        assert W.bfs_tiers(web) == {0: 0, 1: 1, 2: 1, 3: 2, 4: 1}

    def test_adjacency_of_is_sorted(self):
        web = _tiny_web([(0, 3), (0, 1), (2, 0)], [(0, 0), (1, 0), (0, 1), (1, 1)])
        assert W.adjacency_of(web) == {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}

    def test_planar_square(self):
        web = _tiny_web(
            [(0, 1), (1, 2), (2, 3), (3, 0)],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        )
        assert W.is_planar_layout(web)
        assert W.find_crossing(web) is None

    def test_crossing_diagonals_detected(self):
        web = _tiny_web(
            [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3)],
            [(0, 0), (1, 0), (1, 1), (0, 1)],
        )
        assert not W.is_planar_layout(web)
        found = W.find_crossing(web)
        assert found is not None
        assert {found[0], found[1]} == {WebEdge(0, 2), WebEdge(1, 3)}

    def test_shared_endpoint_is_not_a_crossing(self):
        web = _tiny_web([(0, 1), (0, 2), (0, 3)], [(0, 0), (1, 0), (0, 1), (-1, 0)])
        assert W.is_planar_layout(web)

    def test_edge_through_a_foreign_node_is_a_crossing(self):
        # Edge 0-2 passes straight through node 1, which edge 1-3 uses.
        web = _tiny_web([(0, 2), (1, 3)], [(0, 0), (1, 0), (2, 0), (1, 5)])
        assert not W.is_planar_layout(web)


# --------------------------------------------------------------------------
# Provenance: a web is the one its seed generates, or it is not
# --------------------------------------------------------------------------


def test_matches_seed_and_rings_of():
    import dataclasses
    from lucifer_descent.web import generate_web, matches_seed, regenerated, rings_of
    for rings in (2, 5, 15):
        web = generate_web(0x5EED, rings)
        assert rings_of(web) == rings
        assert regenerated(web) == web
        assert matches_seed(web)
    web = generate_web(0x5EED)
    node = web.nodes[7]
    edited = dataclasses.replace(web, nodes=tuple(dataclasses.replace(n, template="volcano") if n is node else n for n in web.nodes))
    assert not matches_seed(edited)
    assert not matches_seed(dataclasses.replace(web, edges=web.edges[:-1]))
    assert not matches_seed(dataclasses.replace(web, profile_seed=web.profile_seed ^ 1))
    assert not matches_seed(dataclasses.replace(web, nodes=web.nodes[:-1]))
    with pytest.raises(ValueError):
        rings_of(dataclasses.replace(web, nodes=web.nodes[:-1]))
