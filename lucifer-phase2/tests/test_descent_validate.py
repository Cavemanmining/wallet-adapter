"""Tests for the Phase 4 gate, :mod:`lucifer_descent.validate`.

Spec: docs/WORLD_BIBLE.md section 03 and the reconnect rule in section 07.

Every defect is injected deliberately into a *healthy* generated web or a
*healthy* simulated profile, and each test first shows the healthy input
does not report that kind, then that the damaged one does: the gate is
shown to fail before it is shown to pass.  The all-clear tests sit at the
end of the file for the same reason.
"""

from __future__ import annotations

import copy
import dataclasses
import sys
from pathlib import Path
from typing import List, Optional, Set

# Runnable as `pytest tests/test_descent_validate.py` or
# `python3 -m pytest tests/test_descent_validate.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_descent import validate as V
from lucifer_descent import web as webmod
from lucifer_descent.contracts import (
    FRAGMENTS_TO_UNLOCK,
    MAX_TIER,
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
from lucifer_descent.engine import DescentEngine
from lucifer_descent.store import round_trip_equal
from lucifer_descent.web import generate_web

SEED = 0x5EED_0000_0000_0001
STEPS = 200
LABEL = "test.walk"


# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def healthy_web() -> Web:
    return generate_web(SEED)


@pytest.fixture(scope="module")
def healthy_state() -> ProfileState:
    return V.simulate(SEED, STEPS, LABEL)


@pytest.fixture(scope="module")
def healthy_active_state() -> ProfileState:
    return V.simulate(SEED, STEPS, LABEL, end_active=True)


def kinds(problems: List[V.Problem]) -> Set[str]:
    return {p.kind for p in problems}


def of_kind(problems: List[V.Problem], kind: str) -> List[V.Problem]:
    return [p for p in problems if p.kind == kind]


def assert_caught(before: List[V.Problem], after: List[V.Problem], kind: str, node_id: Optional[int] = None) -> None:
    """The healthy input has no ``kind``; the damaged one has it (at ``node_id``)."""
    assert kind not in kinds(before), f"healthy input already reports {kind}: {of_kind(before, kind)}"
    hits = of_kind(after, kind)
    assert hits, f"{kind} was not reported; got {sorted(kinds(after))}"
    if node_id is not None:
        assert any(p.node_id == node_id for p in hits), f"{kind} not attributed to node {node_id}: {hits}"


def replace_node(web: Web, node_id: int, **changes) -> Web:
    nodes = tuple(dataclasses.replace(n, **changes) if n.id == node_id else n for n in web.nodes)
    return dataclasses.replace(web, nodes=nodes)


def fresh(state: ProfileState) -> ProfileState:
    return copy.deepcopy(state)


def first_of_state(state: ProfileState, wanted: NodeState, exclude: Set[int] = frozenset()) -> int:
    for nid in sorted(state.states):
        if state.states[nid] is wanted and nid not in exclude:
            return nid
    raise AssertionError(f"no {wanted.value} node in the state")


# --------------------------------------------------------------------------
# check_web: defects
# --------------------------------------------------------------------------


def test_wrong_tier_is_caught(healthy_web: Web) -> None:
    victim = webmod.first_id_of_ring(5)  # a ring-5 node, tier 5
    damaged = replace_node(healthy_web, victim, tier=3)
    assert_caught(V.check_web(healthy_web), V.check_web(damaged), V.K_TIER_MISMATCH, victim)


def test_disconnected_web_is_caught(healthy_web: Web) -> None:
    # Cut two adjacent outer-ring nodes off from everything but each other.
    base = webmod.first_id_of_ring(MAX_TIER)
    u, v = base, base + 1
    kept = tuple(
        e for e in healthy_web.edges
        if not ({e.a, e.b} & {u, v}) or {e.a, e.b} == {u, v}
    )
    if not any({e.a, e.b} == {u, v} for e in kept):
        kept = kept + (WebEdge(u, v),)
    damaged = dataclasses.replace(healthy_web, edges=kept)
    problems = V.check_web(damaged)
    assert_caught(V.check_web(healthy_web), problems, V.K_DISCONNECTED, u)
    assert any(p.node_id == v for p in of_kind(problems, V.K_DISCONNECTED))


def test_crossing_edge_is_caught_in_generated_web(healthy_web: Web) -> None:
    # Swap the origin's position with an outer-ring node's: the origin's
    # ring-1 links now span every ring and must cut through their chords.
    outer = healthy_web.node(webmod.first_id_of_ring(MAX_TIER))
    origin = healthy_web.node(healthy_web.origin_id)
    damaged = replace_node(healthy_web, origin.id, x=outer.x, y=outer.y)
    damaged = replace_node(damaged, outer.id, x=origin.x, y=origin.y)
    assert_caught(V.check_web(healthy_web), V.check_web(damaged), V.K_EDGE_CROSSING)


def test_crossing_edge_is_caught_in_hand_built_web() -> None:
    # A square whose two diagonals cross at the centre.
    square = Web(
        profile_seed=1,
        origin_id=0,
        nodes=(
            WebNode(id=0, tier=0, ring_index=0, template="t", x=0.0, y=0.0),
            WebNode(id=1, tier=1, ring_index=0, template="t", x=1.0, y=0.0),
            WebNode(id=2, tier=1, ring_index=1, template="t", x=1.0, y=1.0),
            WebNode(id=3, tier=1, ring_index=2, template="t", x=0.0, y=1.0),
        ),
        edges=(WebEdge(0, 1), WebEdge(1, 2), WebEdge(2, 3), WebEdge(3, 0), WebEdge(0, 2), WebEdge(1, 3)),
    )
    problems = V.check_web(square)
    hits = of_kind(problems, V.K_EDGE_CROSSING)
    assert len(hits) == 1 and "0-2" in hits[0].detail and "1-3" in hits[0].detail
    # Without the diagonals the same square is planar (it still fails other
    # rules, but not this one).
    planar = dataclasses.replace(square, edges=square.edges[:4])
    assert V.K_EDGE_CROSSING not in kinds(V.check_web(planar))


def test_segments_cross_primitive() -> None:
    assert V.segments_cross((0, 0), (2, 2), (0, 2), (2, 0))          # proper crossing
    assert V.segments_cross((0, 0), (2, 0), (1, 0), (1, 1))          # endpoint touching
    assert V.segments_cross((0, 0), (2, 0), (1, 0), (3, 0))          # collinear overlap
    assert not V.segments_cross((0, 0), (1, 0), (2, 0), (3, 0))      # collinear, apart
    assert not V.segments_cross((0, 0), (1, 1), (0, 1), (-1, 2))     # parallel-ish, apart
    assert not V.segments_cross((0, 0), (1, 0), (0, 1), (1, 1))      # parallel


def test_tier_above_max_is_caught(healthy_web: Web) -> None:
    victim = webmod.first_id_of_ring(MAX_TIER)
    damaged = replace_node(healthy_web, victim, tier=MAX_TIER + 1)
    problems = V.check_web(damaged)
    assert_caught(V.check_web(healthy_web), problems, V.K_TIER_ABOVE_MAX, victim)
    assert V.K_TIER_MISMATCH in kinds(problems)  # 16 is also not its distance


def test_arena_count_is_caught(healthy_web: Web) -> None:
    arena = next(n for n in healthy_web.nodes if n.pinnacle is Pinnacle.ARBITER)
    no_arena = replace_node(healthy_web, arena.id, pinnacle=None)
    assert_caught(V.check_web(healthy_web), V.check_web(no_arena), V.K_ARENA_COUNT)
    spare = next(n for n in healthy_web.nodes if n.tier == MAX_TIER and n.pinnacle is None and n.glyph is None)
    two_arenas = replace_node(healthy_web, spare.id, pinnacle=Pinnacle.ARBITER)
    assert_caught(V.check_web(healthy_web), V.check_web(two_arenas), V.K_ARENA_COUNT)


def test_arena_that_is_also_a_glyph_is_caught(healthy_web: Web) -> None:
    arena = next(n for n in healthy_web.nodes if n.pinnacle is Pinnacle.MONOLITH)
    damaged = replace_node(healthy_web, arena.id, glyph=Pinnacle.MONOLITH)
    assert_caught(V.check_web(healthy_web), V.check_web(damaged), V.K_ARENA_IS_GLYPH, arena.id)


def test_arena_off_the_outer_ring_is_caught(healthy_web: Web) -> None:
    arena = next(n for n in healthy_web.nodes if n.pinnacle is Pinnacle.ARBITER)
    inner = healthy_web.node(webmod.first_id_of_ring(3))
    damaged = replace_node(healthy_web, arena.id, pinnacle=None)
    damaged = replace_node(damaged, inner.id, pinnacle=Pinnacle.ARBITER)
    assert_caught(V.check_web(healthy_web), V.check_web(damaged), V.K_ARENA_TIER, inner.id)


def test_glyph_off_the_outer_ring_and_too_few_glyphs_are_caught(healthy_web: Web) -> None:
    glyph = next(n for n in healthy_web.nodes if n.glyph is Pinnacle.ARBITER)
    inner = healthy_web.node(webmod.first_id_of_ring(7))
    moved = replace_node(healthy_web, inner.id, glyph=Pinnacle.ARBITER)
    assert_caught(V.check_web(healthy_web), V.check_web(moved), V.K_GLYPH_TIER, inner.id)
    stripped = replace_node(healthy_web, glyph.id, glyph=None)
    assert_caught(V.check_web(healthy_web), V.check_web(stripped), V.K_GLYPH_COUNT)


def test_degree_below_two_is_caught(healthy_web: Web) -> None:
    victim = webmod.first_id_of_ring(MAX_TIER) + 3
    neighbours = healthy_web.neighbours(victim)
    keep = neighbours[0]
    kept = tuple(e for e in healthy_web.edges if victim not in (e.a, e.b) or {e.a, e.b} == {victim, keep})
    damaged = dataclasses.replace(healthy_web, edges=kept)
    assert_caught(V.check_web(healthy_web), V.check_web(damaged), V.K_DEGREE, victim)


def test_sparse_node_ids_are_caught(healthy_web: Web) -> None:
    last = healthy_web.nodes[-1]
    renumbered = replace_node(healthy_web, last.id, id=last.id + 7)
    edges = tuple(
        WebEdge(last.id + 7 if e.a == last.id else e.a, last.id + 7 if e.b == last.id else e.b)
        for e in healthy_web.edges
    )
    damaged = dataclasses.replace(renumbered, edges=edges)
    assert_caught(V.check_web(healthy_web), V.check_web(damaged), V.K_IDS_NOT_DENSE)


def test_malformed_edges_are_caught(healthy_web: Web) -> None:
    damaged = dataclasses.replace(
        healthy_web, edges=healthy_web.edges + (WebEdge(0, 0), WebEdge(0, 9999), healthy_web.edges[0])
    )
    problems = V.check_web(damaged)
    for kind in (V.K_EDGE_SELF_LOOP, V.K_EDGE_UNKNOWN_NODE, V.K_EDGE_DUPLICATE):
        assert_caught(V.check_web(healthy_web), problems, kind)


def test_generator_helper_that_disagrees_is_reported(healthy_web: Web, monkeypatch: pytest.MonkeyPatch) -> None:
    """The gate's own BFS and sweep stay authoritative; a lying helper is a finding."""
    monkeypatch.setattr(webmod, "bfs_tiers", lambda web: {n.id: 0 for n in web.nodes})
    monkeypatch.setattr(webmod, "is_planar_layout", lambda web: False)
    problems = V.check_web(healthy_web)
    hits = of_kind(problems, V.K_HELPER_DISAGREEMENT)
    assert len(hits) == 2
    assert V.K_TIER_MISMATCH not in kinds(problems) and V.K_EDGE_CROSSING not in kinds(problems)
    monkeypatch.setattr(webmod, "bfs_tiers", lambda web: 1 / 0)
    assert V.K_HELPER_ERROR in kinds(V.check_web(healthy_web))


def test_check_web_without_helpers_uses_its_own(healthy_web: Web, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(webmod, "bfs_tiers")
    monkeypatch.delattr(webmod, "is_planar_layout")
    assert V.check_web(healthy_web) == []
    victim = webmod.first_id_of_ring(2)
    assert of_kind(V.check_web(replace_node(healthy_web, victim, tier=9)), V.K_TIER_MISMATCH)


# --------------------------------------------------------------------------
# check_state: defects
# --------------------------------------------------------------------------


def test_state_missing_a_node_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    victim = first_of_state(damaged, NodeState.LOCKED)
    del damaged.states[victim]
    assert_caught(V.check_state(healthy_state), V.check_state(damaged), V.K_STATE_MISSING, victim)


def test_state_with_an_extra_node_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    damaged.states[9999] = NodeState.LOCKED
    assert_caught(V.check_state(healthy_state), V.check_state(damaged), V.K_STATE_EXTRA, 9999)


def test_reachable_node_without_cleared_neighbour_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    # Every LOCKED node in a healthy profile has no CLEARED neighbour, so
    # promoting one to REACHABLE breaks exactly the rule under test.
    victim = first_of_state(damaged, NodeState.LOCKED)
    damaged.states[victim] = NodeState.REACHABLE
    assert_caught(V.check_state(healthy_state), V.check_state(damaged), V.K_REACHABLE_NO_CLEARED, victim)


def test_locked_node_with_cleared_neighbour_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    victim = first_of_state(damaged, NodeState.REACHABLE)
    damaged.states[victim] = NodeState.LOCKED
    assert_caught(V.check_state(healthy_state), V.check_state(damaged), V.K_LOCKED_WITH_CLEARED, victim)


def test_origin_not_cleared_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    damaged.states[damaged.web.origin_id] = NodeState.LOCKED
    assert_caught(V.check_state(healthy_state), V.check_state(damaged), V.K_ORIGIN_NOT_CLEARED, damaged.web.origin_id)


def test_two_active_nodes_are_caught(healthy_active_state: ProfileState) -> None:
    damaged = fresh(healthy_active_state)
    assert damaged.instance is not None
    second = first_of_state(damaged, NodeState.REACHABLE)
    damaged.states[second] = NodeState.ACTIVE
    problems = V.check_state(damaged)
    assert_caught(V.check_state(healthy_active_state), problems, V.K_MULTIPLE_ACTIVE)
    assert_caught(V.check_state(healthy_active_state), problems, V.K_ACTIVE_NO_INSTANCE, second)


def test_active_node_without_instance_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    assert damaged.instance is None
    victim = first_of_state(damaged, NodeState.REACHABLE)
    damaged.states[victim] = NodeState.ACTIVE
    assert_caught(V.check_state(healthy_state), V.check_state(damaged), V.K_ACTIVE_NO_INSTANCE, victim)


def test_instance_whose_node_is_not_active_is_caught(healthy_active_state: ProfileState) -> None:
    damaged = fresh(healthy_active_state)
    damaged.states[damaged.instance.node_id] = NodeState.FAILED
    assert_caught(
        V.check_state(healthy_active_state), V.check_state(damaged), V.K_INSTANCE_NOT_ACTIVE, damaged.instance.node_id
    )


def test_instance_sigil_rules_are_caught(healthy_active_state: ProfileState) -> None:
    before = V.check_state(healthy_active_state)
    in_stash = fresh(healthy_active_state)
    in_stash.stash[in_stash.instance.sigil.id] = in_stash.instance.sigil
    assert_caught(before, V.check_state(in_stash), V.K_INSTANCE_SIGIL_IN_STASH)

    wrong_seed = fresh(healthy_active_state)
    wrong_seed.instance.map_seed = (wrong_seed.instance.map_seed + 1) & ((1 << 64) - 1)
    assert_caught(before, V.check_state(wrong_seed), V.K_INSTANCE_SEED)

    weak = fresh(healthy_active_state)
    node = weak.web.node(weak.instance.node_id)
    assert node.tier > 1
    weak.instance.sigil = Sigil(id=weak.instance.sigil.id, tier=1, seed=weak.instance.sigil.seed)
    assert_caught(before, V.check_state(weak), V.K_INSTANCE_SIGIL_WEAK)


def test_pinnacle_fragment_rules_are_caught(healthy_state: ProfileState) -> None:
    before = V.check_state(healthy_state)
    early = fresh(healthy_state)
    early.fragments[Pinnacle.ARBITER] = FRAGMENTS_TO_UNLOCK - 1
    early.unlocked_pinnacles = frozenset({Pinnacle.ARBITER})
    assert_caught(before, V.check_state(early), V.K_PINNACLE_EARLY)

    late = fresh(healthy_state)
    late.fragments[Pinnacle.MONOLITH] = FRAGMENTS_TO_UNLOCK
    late.unlocked_pinnacles = frozenset()
    assert_caught(before, V.check_state(late), V.K_PINNACLE_LATE)


def test_ledger_gap_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    assert len(damaged.history) > 10
    del damaged.history[5]
    problems = V.check_state(damaged)
    assert_caught(V.check_state(healthy_state), problems, V.K_LEDGER_GAP)
    assert len(of_kind(problems, V.K_LEDGER_GAP)) == 1  # reported once, where it opens


def test_ledger_starting_at_the_wrong_base_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    damaged.history = [dataclasses.replace(e, seq=e.seq + 5) for e in damaged.history]
    problems = V.check_state(damaged)
    assert_caught(V.check_state(healthy_state), problems, V.K_LEDGER_BASE)
    assert V.K_LEDGER_GAP not in kinds(problems)  # still dense, just misbased


def test_ledger_entry_with_illegal_transition_is_caught(healthy_state: ProfileState) -> None:
    before = V.check_state(healthy_state)
    # (CLEARED, OPEN) is not a row of the table at all.
    damaged = fresh(healthy_state)
    index = next(i for i, e in enumerate(damaged.history) if e.event is Event.OPEN)
    entry = damaged.history[index]
    assert (NodeState.CLEARED, Event.OPEN) not in TRANSITIONS
    damaged.history[index] = dataclasses.replace(entry, before=NodeState.CLEARED)
    assert_caught(before, V.check_state(damaged), V.K_LEDGER_ILLEGAL, entry.node_id)
    # A real row whose recorded result is not what the table says.
    wrong_after = fresh(healthy_state)
    wrong_after.history[index] = dataclasses.replace(entry, after=NodeState.CLEARED)
    assert_caught(before, V.check_state(wrong_after), V.K_LEDGER_ILLEGAL, entry.node_id)


def test_ledger_entry_naming_an_unknown_node_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    damaged.history[3] = dataclasses.replace(damaged.history[3], node_id=9999)
    assert_caught(V.check_state(healthy_state), V.check_state(damaged), V.K_LEDGER_UNKNOWN_NODE, 9999)


# --------------------------------------------------------------------------
# check_replay: defects
# --------------------------------------------------------------------------


def test_truncated_ledger_no_longer_replays(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    damaged.history = damaged.history[:-3]
    # check_state cannot see this: the ledger is still dense and legal.
    assert V.K_LEDGER_GAP not in kinds(V.check_state(damaged))
    problems = V.check_replay(damaged)
    assert_caught(V.check_replay(healthy_state), problems, V.K_REPLAY_MISMATCH)
    # Every transition changes a state, so even one lost entry shows up in
    # the states dict; the truncated ledger itself still replays verbatim.
    assert V.K_REPLAY_LEDGER not in kinds(problems)


def test_ledger_missing_the_sigil_fact_is_reported(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    index = next(i for i, e in enumerate(damaged.history) if e.event is Event.OPEN)
    damaged.history[index] = dataclasses.replace(damaged.history[index], sigil_id=None)
    problems = V.check_replay(damaged)
    assert_caught(V.check_replay(healthy_state), problems, V.K_LEDGER_MISSING_FACT, damaged.history[index].node_id)
    # The states still replay: the fact is missing, not the move.
    assert V.K_REPLAY_MISMATCH not in kinds(problems)


def test_ledger_the_engine_refuses_is_reported(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    index = next(i for i, e in enumerate(damaged.history) if e.event is Event.BOSS_KILLED)
    other = first_of_state(damaged, NodeState.LOCKED)
    damaged.history[index] = dataclasses.replace(damaged.history[index], node_id=other)
    problems = V.check_replay(damaged)
    assert_caught(V.check_replay(healthy_state), problems, V.K_REPLAY_ERROR)
    assert "LedgerMismatch" in of_kind(problems, V.K_REPLAY_ERROR)[0].detail


def test_replay_compares_rewards_too(healthy_state: ProfileState) -> None:
    before = V.check_replay(healthy_state)
    points = fresh(healthy_state)
    points.passive_points += 1
    assert_caught(before, V.check_replay(points), V.K_REPLAY_PASSIVE)
    frags = fresh(healthy_state)
    frags.fragments[Pinnacle.ARBITER] = frags.fragments.get(Pinnacle.ARBITER, 0) + 1
    assert_caught(before, V.check_replay(frags), V.K_REPLAY_FRAGMENTS)
    unlocked = fresh(healthy_state)
    unlocked.unlocked_pinnacles = frozenset()
    if healthy_state.unlocked_pinnacles:
        assert_caught(before, V.check_replay(unlocked), V.K_REPLAY_UNLOCKED)


def test_replay_goes_through_the_engine_factory(healthy_state: ProfileState) -> None:
    class Flipping:
        """An engine whose replay disagrees with the profile on one node."""

        @staticmethod
        def replay(profile_id, web, history):
            rebuilt = DescentEngine.replay(profile_id, web, history)
            victim = first_of_state(rebuilt, NodeState.LOCKED)
            rebuilt.states[victim] = NodeState.CLEARED
            return rebuilt

    class Broken:
        @staticmethod
        def replay(profile_id, web, history):
            raise RuntimeError("no engine here")

    assert V.check_replay(healthy_state, lambda: DescentEngine) == []
    assert V.check_replay(healthy_state, DescentEngine) == []
    assert V.K_REPLAY_MISMATCH in kinds(V.check_replay(healthy_state, lambda: Flipping()))
    assert V.K_REPLAY_MISMATCH in kinds(V.check_replay(healthy_state, Flipping))
    hits = of_kind(V.check_replay(healthy_state, Broken), V.K_REPLAY_ERROR)
    assert hits and "no engine here" in hits[0].detail


# --------------------------------------------------------------------------
# simulate
# --------------------------------------------------------------------------


def test_simulate_is_deterministic_and_label_sensitive(healthy_state: ProfileState) -> None:
    again = V.simulate(SEED, STEPS, LABEL)
    assert round_trip_equal(healthy_state, again)
    other = V.simulate(SEED, STEPS, "another.walk")
    assert not round_trip_equal(healthy_state, other)
    assert not round_trip_equal(healthy_state, V.simulate(SEED + 1, STEPS, LABEL))


def test_simulate_accepts_its_own_web_and_refuses_another(healthy_web: Web, healthy_state: ProfileState) -> None:
    assert round_trip_equal(V.simulate(SEED, STEPS, LABEL, web=healthy_web), healthy_state)
    with pytest.raises(ValueError):
        V.simulate(SEED + 1, 10, LABEL, web=healthy_web)


def test_simulate_plays_only_legal_moves(healthy_state: ProfileState) -> None:
    for entry in healthy_state.history:
        assert TRANSITIONS[(entry.before, entry.event)] is entry.after
    assert healthy_state.instance is None
    assert not any(s is NodeState.ACTIVE for s in healthy_state.states.values())
    events = {e.event for e in healthy_state.history}
    assert {Event.OPEN, Event.BOSS_KILLED, Event.ELITES_MET, Event.DIED, Event.NEIGHBOUR_CLEARED} <= events


def test_simulate_reaches_the_outer_ring_and_the_pinnacles(healthy_state: ProfileState) -> None:
    cleared_tiers = {healthy_state.web.node(n).tier for n in healthy_state.cleared_ids()}
    assert MAX_TIER in cleared_tiers
    assert healthy_state.unlocked_pinnacles, "the walk should collect three fragments in 200 steps"
    assert healthy_state.passive_points > 0
    assert any(s is NodeState.FAILED for s in healthy_state.states.values())


def test_simulate_can_end_with_a_live_instance(healthy_active_state: ProfileState) -> None:
    inst = healthy_active_state.instance
    assert inst is not None
    assert healthy_active_state.states[inst.node_id] is NodeState.ACTIVE
    assert healthy_active_state.history[-1].event is Event.OPEN
    assert inst.sigil.id not in healthy_active_state.stash


def test_simulate_argument_checks() -> None:
    with pytest.raises(TypeError):
        V.simulate("1", 5, LABEL)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        V.simulate(SEED, -1, LABEL)
    zero = V.simulate(SEED, 0, LABEL)
    assert zero.cleared_ids() == [zero.web.origin_id]


# --------------------------------------------------------------------------
# run_suite: failures are counted and attributed
# --------------------------------------------------------------------------


def test_run_suite_counts_problems_and_remembers_the_first_failing_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    real = V.generate_web
    bad_seed = SEED + 1

    def sabotaged(profile_seed: int, *args, **kwargs) -> Web:
        web = real(profile_seed, *args, **kwargs)
        if profile_seed == bad_seed:
            return replace_node(web, webmod.first_id_of_ring(4), tier=2)
        return web

    monkeypatch.setattr(V, "generate_web", sabotaged)
    report = V.run_suite(3, SEED, 60)
    assert not report.ok
    assert report.profiles_run == 3 and report.profiles_failed == 1
    assert report.problem_counts == {V.K_TIER_MISMATCH: 1}
    assert report.first_failure_seed == bad_seed
    assert [p.kind for p in report.first_failure] == [V.K_TIER_MISMATCH]
    assert "FAILED" in report.summary() and V.K_TIER_MISMATCH in report.summary()


def test_run_suite_records_an_exception_as_a_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    real = V.simulate

    def exploding(profile_seed: int, *args, **kwargs) -> ProfileState:
        if profile_seed == SEED:
            raise RuntimeError("walk fell over")
        return real(profile_seed, *args, **kwargs)

    monkeypatch.setattr(V, "simulate", exploding)
    report = V.run_suite(2, SEED, 40)
    assert report.profiles_failed == 1
    assert report.problem_counts == {f"{V.K_EXCEPTION}:simulate": 1}
    assert "walk fell over" in report.first_failure[0].detail


def test_run_suite_reports_a_store_round_trip_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(V, "round_trip_equal", lambda a, b: False)
    report = V.run_suite(1, SEED, 40)
    assert report.problem_counts == {V.K_STORE_ROUND_TRIP: 1}


# --------------------------------------------------------------------------
# Hardening after adversarial review: provenance, forged progression, the
# stash and instance, the ledger's facts
# --------------------------------------------------------------------------

from lucifer_descent.contracts import LedgerEntry  # noqa: E402
from lucifer_descent.sigils import mint_sigil  # noqa: E402


def test_web_not_from_seed_is_caught(healthy_web: Web) -> None:
    before = V.check_web(healthy_web)
    assert V.K_WEB_NOT_FROM_SEED not in kinds(before)
    victim = next(n for n in healthy_web.nodes if n.tier == 5 and n.mechanic is None)
    with_mechanic = replace_node(healthy_web, victim.id, mechanic=V.Web.__mro__ and __import__("lucifer_descent.contracts").contracts.Mechanic.DIG)
    problems = V.check_web(with_mechanic)
    assert_caught(before, problems, V.K_WEB_NOT_FROM_SEED)
    assert kinds(problems) == {V.K_WEB_NOT_FROM_SEED}, "a mechanic edit breaks no structural rule; only provenance sees it"
    assert str(victim.id) in of_kind(problems, V.K_WEB_NOT_FROM_SEED)[0].detail
    retemplated = dataclasses.replace(healthy_web, nodes=tuple(dataclasses.replace(n, template="volcano") for n in healthy_web.nodes))
    assert_caught(before, V.check_web(retemplated), V.K_WEB_NOT_FROM_SEED)
    reseeded = dataclasses.replace(healthy_web, profile_seed=healthy_web.profile_seed ^ 1)
    assert_caught(before, V.check_web(reseeded), V.K_WEB_NOT_FROM_SEED)
    reversioned = dataclasses.replace(healthy_web, version=2)
    assert "version" in of_kind(V.check_web(reversioned), V.K_WEB_NOT_FROM_SEED)[0].detail
    # A smaller generated web is its own seed's web too.
    assert V.K_WEB_NOT_FROM_SEED not in kinds(V.check_web(generate_web(SEED, 3)))


def test_cleared_island_is_caught(healthy_state: ProfileState) -> None:
    damaged = fresh(healthy_state)
    nodes = {n.id: n for n in damaged.web.nodes}
    # Two adjacent LOCKED nodes, cleared together: each has a cleared
    # neighbour (the other), so only the global rule can see them.
    locked = [nid for nid in sorted(damaged.states) if damaged.states[nid] is NodeState.LOCKED]
    u = next(
        nid for nid in locked
        if any(m in locked for m in damaged.web.neighbours(nid))
        and all(damaged.states[m] is not NodeState.CLEARED for m in damaged.web.neighbours(nid))
    )
    v = next(
        m for m in damaged.web.neighbours(u)
        if m in locked and all(damaged.states[k] is not NodeState.CLEARED for k in damaged.web.neighbours(m))
    )
    assert nodes[u].tier > 1 and nodes[v].tier > 1
    damaged.states[u] = damaged.states[v] = NodeState.CLEARED
    problems = V.check_state(damaged)
    assert_caught(V.check_state(healthy_state), problems, V.K_CLEARED_ISLAND, u)
    assert any(p.node_id == v for p in of_kind(problems, V.K_CLEARED_ISLAND))


def test_arena_opened_while_locked_is_caught() -> None:
    early = V.simulate(SEED, 12, LABEL)   # nothing unlocked yet
    assert not early.unlocked_pinnacles
    before = V.check_state(early)
    for state_ in (NodeState.FAILED, NodeState.ACTIVE, NodeState.CLEARED):
        damaged = fresh(early)
        arena = next(n for n in damaged.web.nodes if n.pinnacle is Pinnacle.ARBITER)
        damaged.states[arena.id] = state_
        assert_caught(before, V.check_state(damaged), V.K_ARENA_OPENED_LOCKED, arena.id)
    # Once unlocked, a cleared arena is fine.
    late = V.simulate(SEED, STEPS, LABEL)
    unlocked = next(p for p in Pinnacle if p in late.unlocked_pinnacles)
    arena = next(n for n in late.web.nodes if n.pinnacle is unlocked)
    fine = fresh(late)
    fine.states[arena.id] = NodeState.CLEARED
    assert V.K_ARENA_OPENED_LOCKED not in kinds(V.check_state(fine))


def test_stash_and_instance_sigils_must_be_mints_of_the_profile(healthy_active_state: ProfileState) -> None:
    before = V.check_state(healthy_active_state)
    seed = healthy_active_state.web.profile_seed
    forged = fresh(healthy_active_state)
    for i in range(3):
        sid = f"sg-f{i:07x}"
        forged.stash[sid] = Sigil(id=sid, tier=15, seed=i)
    problems = V.check_state(forged)
    assert_caught(before, problems, V.K_STASH_NOT_MINTED)
    assert len(of_kind(problems, V.K_STASH_NOT_MINTED)) == 3
    # A genuine mint with its tier field altered is not genuine either.
    key = sorted(healthy_active_state.stash)[0]
    retiered = fresh(healthy_active_state)
    real = retiered.stash[key]
    retiered.stash[key] = Sigil(id=real.id, tier=15 if real.tier != 15 else 14, seed=real.seed)
    assert_caught(before, V.check_state(retiered), V.K_STASH_NOT_MINTED)
    # A genuine mint that the ledger already spent.
    spent_id = next(e.sigil_id for e in healthy_active_state.history if e.event is Event.OPEN)
    from lucifer_descent.sigils import unmint
    counter, tier = unmint(seed, spent_id)
    respent = fresh(healthy_active_state)
    respent.stash[spent_id] = mint_sigil(seed, counter, tier)
    problems = V.check_state(respent)
    assert_caught(before, problems, V.K_STASH_SPENT)
    assert V.K_STASH_NOT_MINTED not in kinds(problems)
    # The live instance's Sigil swapped for one that was never minted.
    swapped = fresh(healthy_active_state)
    other = Sigil(id="sg-f0000099", tier=15, seed=0x8000000000000001)
    swapped.instance.sigil = other
    swapped.instance.map_seed = other.seed
    problems = V.check_state(swapped)
    assert_caught(before, problems, V.K_INSTANCE_NOT_MINTED)
    assert_caught(before, problems, V.K_INSTANCE_LEDGER)
    # And a fresh mint of a counter the profile never used is still a mint.
    unused = mint_sigil(seed, 10**6, 3)
    minted = fresh(healthy_active_state)
    minted.stash[unused.id] = unused
    assert V.K_STASH_NOT_MINTED not in kinds(V.check_state(minted))


def test_instance_must_agree_with_the_ledger(healthy_active_state: ProfileState) -> None:
    before = V.check_state(healthy_active_state)
    flipped = fresh(healthy_active_state)
    flipped.instance.has_boss = not flipped.instance.has_boss
    problems = V.check_state(flipped)
    assert_caught(before, problems, V.K_INSTANCE_LEDGER, flipped.instance.node_id)
    ticked = fresh(healthy_active_state)
    ticked.instance.opened_tick = -5
    assert_caught(before, V.check_state(ticked), V.K_INSTANCE_LEDGER)
    packs = fresh(healthy_active_state)
    packs.instance.elite_total += 50
    assert_caught(before, V.check_state(packs), V.K_INSTANCE_LEDGER)
    negative = fresh(healthy_active_state)
    negative.instance.elite_killed = -1
    assert_caught(before, V.check_state(negative), V.K_INSTANCE_KILLS)
    past = fresh(healthy_active_state)
    past.instance.has_boss = False
    past.instance.elite_killed = past.instance.elite_total
    assert_caught(before, V.check_state(past), V.K_INSTANCE_KILLS)
    no_open = fresh(healthy_active_state)
    no_open.history.pop()
    assert_caught(before, V.check_state(no_open), V.K_INSTANCE_LEDGER)


def test_ledger_ticks_and_sigil_ids_are_checked(healthy_state: ProfileState) -> None:
    before = V.check_state(healthy_state)
    prefix = len(DescentEngine.new_profile("x", healthy_state.web).history)
    backwards = fresh(healthy_state)
    backwards.history = backwards.history[:prefix] + [
        dataclasses.replace(e, tick=10**6 - e.seq) for e in backwards.history[prefix:]
    ]
    problems = V.check_state(backwards)
    assert_caught(before, problems, V.K_LEDGER_TICK_ORDER)
    assert len(of_kind(problems, V.K_LEDGER_TICK_ORDER)) == 1, "reported once, where it first turns"

    opens = [i for i, e in enumerate(healthy_state.history) if e.event is Event.OPEN]
    assert len(opens) >= 2
    reused = fresh(healthy_state)
    first_id = reused.history[opens[0]].sigil_id
    for i in opens[1:2]:
        reused.history[i] = dataclasses.replace(reused.history[i], sigil_id=first_id)
    assert_caught(before, V.check_state(reused), V.K_LEDGER_SIGIL_REUSED)

    malformed = fresh(healthy_state)
    malformed.history[opens[0]] = dataclasses.replace(malformed.history[opens[0]], sigil_id="sg-never-minted")
    assert_caught(before, V.check_state(malformed), V.K_LEDGER_SIGIL_MALFORMED)

    nodes = {n.id: n for n in healthy_state.web.nodes}
    deep = next(i for i in opens if nodes[healthy_state.history[i].node_id].tier >= 2)
    weak = fresh(healthy_state)
    weak_id = "sg-1" + weak.history[deep].sigil_id[4:]  # tier nibble 1
    weak.history[deep] = dataclasses.replace(weak.history[deep], sigil_id=weak_id)
    assert_caught(before, V.check_state(weak), V.K_LEDGER_SIGIL_WEAK, healthy_state.history[deep].node_id)


def test_replay_rederives_the_recorded_map_facts_with_a_probe(healthy_state: ProfileState) -> None:
    before = V.check_replay(healthy_state, map_probe=V.stub_map_probe)
    assert before == []
    damaged = fresh(healthy_state)
    index = next(i for i, e in enumerate(damaged.history) if e.event is Event.OPEN and e.has_boss)
    entry = damaged.history[index]
    damaged.history[index] = dataclasses.replace(entry, elite_total=entry.elite_total + 1)
    problems = V.check_replay(damaged, map_probe=V.stub_map_probe)
    assert_caught(before, problems, V.K_LEDGER_FACT_MISMATCH, entry.node_id)
    assert V.K_REPLAY_ERROR not in kinds(problems), "the ledger is self-consistent; only the probe disagrees"
    assert V.K_LEDGER_FACT_MISMATCH not in kinds(V.check_replay(damaged)), "without a probe the facts are taken as given"
    # Missing facts are a missing fact.
    missing = fresh(healthy_state)
    missing.history[index] = dataclasses.replace(entry, has_boss=None, elite_total=None)
    assert_caught(before, V.check_replay(missing), V.K_LEDGER_MISSING_FACT, entry.node_id)


def test_forged_progression_is_refused_end_to_end() -> None:
    """The reviewers' island: three glyph nodes cleared out of nowhere, with
    well-formed ids and plausible facts, on an otherwise honest profile."""
    base = V.simulate(SEED, 12, LABEL)
    nodes = {n.id: n for n in base.web.nodes}
    assert not base.unlocked_pinnacles
    forged = fresh(base)

    def append(node_id: int, event: Event, sigil_id=None, facts=(None, None)) -> None:
        before_ = forged.states[node_id]
        after = TRANSITIONS[(before_, event)]
        forged.history.append(LedgerEntry(len(forged.history) + 1, node_id, event, before_, after, sigil_id, 999, *facts))
        forged.states[node_id] = after

    glyphs = [n for n in base.web.nodes if n.glyph is Pinnacle.ARBITER][:3]
    for k, g in enumerate(glyphs):
        buddy = next(m for m in base.web.neighbours(g.id) if nodes[m].tier == MAX_TIER and nodes[m].pinnacle is None)
        for j, nid in enumerate((g.id, buddy)):
            if forged.states[nid] is NodeState.LOCKED:
                append(nid, Event.NEIGHBOUR_CLEARED)
            if forged.states[nid] is NodeState.REACHABLE:
                sid = mint_sigil(base.web.profile_seed, 5000 + 2 * k + j, 15).id
                append(nid, Event.OPEN, sid, (True, 1))
                append(nid, Event.BOSS_KILLED, sid)
                for m in base.web.neighbours(nid):
                    if forged.states[m] is NodeState.LOCKED:
                        append(m, Event.NEIGHBOUR_CLEARED)
    forged.fragments[Pinnacle.ARBITER] = 3
    forged.unlocked_pinnacles = frozenset({Pinnacle.ARBITER})
    state_kinds = kinds(V.check_state(forged))
    replay_kinds = kinds(V.check_replay(forged, map_probe=V.stub_map_probe))
    assert V.K_CLEARED_ISLAND in state_kinds
    assert V.K_REPLAY_ERROR in replay_kinds
    assert "no clear to propagate" in of_kind(V.check_replay(forged), V.K_REPLAY_ERROR)[0].detail


def test_run_suite_leaves_a_callers_profiles_alone(tmp_path: Path) -> None:
    from lucifer_descent.store import SqliteStore
    path = str(tmp_path / "user.sqlite3")
    mine = V.simulate(SEED, 20, "my.own.walk")
    with SqliteStore(path) as store:
        store.save(mine)
    report = V.run_suite(1, SEED, 20, db_path=path)
    assert report.ok, report.summary()
    with SqliteStore(path) as store:
        assert store.list_profiles() == [mine.profile_id]
        assert round_trip_equal(store.load(mine.profile_id), mine)


def test_gate_geometry_is_exact() -> None:
    """A near miss a float-with-epsilon test would call collinear-and-touching."""
    assert not V.segments_cross((0, 0), (1, 1), (0.5, 0.4999999999), (1, 0))
    assert V.segments_cross((0, 0), (1, 1), (0.5, 0.5000000001), (1, 0))
    assert V.segments_cross((0, 0), (1, 1), (0.5, 0.5), (1, 0))   # touching, exactly
    assert not V.segments_cross((0, 0), (2, 0), (2.0000000001, 0), (3, 0))
    assert V.segments_cross((0, 0), (2, 0), (2.0, 0), (3, 0))


# --------------------------------------------------------------------------
# And finally: healthy input passes the whole gate
# --------------------------------------------------------------------------


def test_healthy_web_reports_zero_problems(healthy_web: Web) -> None:
    assert V.check_web(healthy_web) == []


def test_healthy_profile_reports_zero_problems(healthy_state: ProfileState, healthy_active_state: ProfileState) -> None:
    assert V.check_state(healthy_state) == []
    assert V.check_replay(healthy_state) == []
    assert V.check_state(healthy_active_state) == []
    assert V.check_replay(healthy_active_state) == []


def test_run_suite_passes_and_is_deterministic(tmp_path: Path) -> None:
    first = V.run_suite(4, SEED, 120, db_path=str(tmp_path / "gate.sqlite3"))
    assert first.ok, first.summary()
    assert first.problem_counts == {} and first.first_failure_seed is None and first.first_failure == []
    assert first.profiles_run == 4 and first.nodes_checked == 4 * len(generate_web(SEED).nodes)
    assert first.profiles_with_instance == 1  # every third profile ends live
    second = V.run_suite(4, SEED, 120)
    assert first == second
    assert "OK" in first.summary()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
