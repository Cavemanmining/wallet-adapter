"""Tests for the Descent rules engine.

Spec: docs/WORLD_BIBLE.md section 03, and the reconnect rule in section 07.

Every test drives :class:`DescentEngine` through its public surface with a
stub map probe (no generator involved), except one smoke test of the default
probe against the real pipeline.  The webs are hand-built so a failure points
at a rule, not at the web generator: a small diamond for the everyday rules,
and a chain out to tier 15 with glyph nodes and an arena hanging off its end
for the Pinnacle rules.
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from typing import List, Tuple

# Runnable as `pytest tests/test_descent_engine.py` or
# `python3 tests/test_descent_engine.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_descent.contracts import (
    ELITE_CLEAR_FRACTION,
    FRAGMENTS_TO_UNLOCK,
    MAX_TIER,
    STATE_COLOUR,
    Event,
    IllegalTransition,
    LedgerEntry,
    Mechanic,
    NodeState,
    Pinnacle,
    PortalOpened,
    Sigil,
    Web,
    WebEdge,
    WebNode,
    next_state,
)
from lucifer_descent.engine import (
    DescentEngine,
    DuplicateSigil,
    InstanceAlreadyActive,
    LedgerMismatch,
    MalformedWeb,
    MapProbe,
    NoActiveInstance,
    NodeNotOpenable,
    PinnacleLocked,
    PortalCandidate,
    SigilTooWeak,
    UnknownNode,
    UnknownSigil,
    WrongProfile,
    default_map_probe,
)

PID = "profile-A"


# --------------------------------------------------------------------------
# Fixtures: a stub probe, a counting clock, two hand-built webs
# --------------------------------------------------------------------------


class StubProbe:
    """Answers with a fixed MapProbe and records every call it receives."""

    def __init__(self, has_boss: bool = True, elite_total: int = 5) -> None:
        self.has_boss = has_boss
        self.elite_total = elite_total
        self.calls: List[Tuple[str, int, int]] = []

    def __call__(self, template: str, map_seed: int, sigil_tier: int) -> MapProbe:
        self.calls.append((template, map_seed, sigil_tier))
        return MapProbe(has_boss=self.has_boss, elite_total=self.elite_total)


class Clock:
    """A tick source that advances by one on every read."""

    def __init__(self) -> None:
        self.now = 100

    def __call__(self) -> int:
        self.now += 1
        return self.now


def diamond_web() -> Web:
    """origin 0 at tier 0; 1 and 2 at tier 1; 3 at tier 2 (mechanic); 4 at tier 3.

        0 -- 1 -- 3 -- 4
         \\-- 2 --/
    """
    nodes = (
        WebNode(id=0, tier=0, ring_index=0, template="crypt"),
        WebNode(id=1, tier=1, ring_index=0, template="crypt"),
        WebNode(id=2, tier=1, ring_index=1, template="ashen_ramparts"),
        WebNode(id=3, tier=2, ring_index=0, template="crypt", mechanic=Mechanic.BREACH),
        WebNode(id=4, tier=3, ring_index=0, template="crypt"),
    )
    edges = (WebEdge(0, 1), WebEdge(0, 2), WebEdge(1, 3), WebEdge(2, 3), WebEdge(3, 4))
    return Web(profile_seed=1, origin_id=0, nodes=nodes, edges=edges)


ARENA_ID = 100
ARBITER_GLYPHS = (101, 102, 103)
MONOLITH_GLYPHS = (201, 202)


def pinnacle_web() -> Web:
    """A chain 0-1-...-14 (tier = distance), then off node 14 at tier 15:
    three Arbiter glyph nodes, two Monolith glyph nodes, and the Arbiter arena.
    """
    nodes = [WebNode(id=i, tier=min(MAX_TIER, i), ring_index=0, template="crypt") for i in range(15)]
    edges = [WebEdge(i, i + 1) for i in range(14)]
    for k, nid in enumerate(ARBITER_GLYPHS):
        nodes.append(WebNode(id=nid, tier=MAX_TIER, ring_index=k, template="crypt", glyph=Pinnacle.ARBITER))
        edges.append(WebEdge(14, nid))
    for k, nid in enumerate(MONOLITH_GLYPHS):
        nodes.append(WebNode(id=nid, tier=MAX_TIER, ring_index=3 + k, template="crypt", glyph=Pinnacle.MONOLITH))
        edges.append(WebEdge(14, nid))
    nodes.append(WebNode(id=ARENA_ID, tier=MAX_TIER, ring_index=5, template="crypt", pinnacle=Pinnacle.ARBITER))
    edges.append(WebEdge(14, ARENA_ID))
    return Web(profile_seed=2, origin_id=0, nodes=tuple(nodes), edges=tuple(edges))


def make_engine(web: Web, probe=None, clock=None) -> Tuple[DescentEngine, StubProbe, Clock]:
    probe = probe if probe is not None else StubProbe()
    clock = clock if clock is not None else Clock()
    state = DescentEngine.new_profile(PID, web)
    return DescentEngine(state, probe, clock), probe, clock


def sigil(sid: str, tier: int, seed: int = 0xABCD) -> Sigil:
    return Sigil(id=sid, tier=tier, seed=seed)


def clear_with_boss(engine: DescentEngine, node_id: int, sid: str, tier: int = MAX_TIER) -> None:
    """Open ``node_id`` with a fresh Sigil and kill the boss."""
    engine.add_sigil(PID, sigil(sid, tier, seed=zlib.crc32(sid.encode("utf-8"))))
    engine.open_portal(PID, node_id, sid)
    engine.report_boss_kill(PID)


# --------------------------------------------------------------------------
# A new profile
# --------------------------------------------------------------------------


def test_new_profile_origin_cleared_ring_one_reachable_rest_locked():
    engine, _, _ = make_engine(diamond_web())
    s = engine.state
    assert s.profile_id == PID
    assert s.states == {
        0: NodeState.CLEARED,
        1: NodeState.REACHABLE,
        2: NodeState.REACHABLE,
        3: NodeState.LOCKED,
        4: NodeState.LOCKED,
    }
    assert s.reachable_ids() == [1, 2]
    assert s.cleared_ids() == [0]
    assert s.instance is None and s.stash == {} and s.passive_points == 0
    assert s.fragments == {} and s.unlocked_pinnacles == frozenset()


def test_new_profile_records_ring_one_propagation_in_ledger():
    engine, _, _ = make_engine(diamond_web())
    h = engine.state.history
    assert [(e.seq, e.node_id, e.event, e.before, e.after) for e in h] == [
        (1, 1, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE),
        (2, 2, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE),
    ]
    assert all(e.sigil_id is None for e in h)
    assert DescentEngine.new_profile(PID, diamond_web(), tick=7).history[0].tick == 7


def test_colour_is_derived_from_state():
    engine, _, _ = make_engine(diamond_web())
    assert engine.colour_of(0) == "green"
    assert engine.colour_of(1) == "blue"
    assert engine.colour_of(3) == "grey"
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    assert engine.colour_of(1) == "amber"
    engine.report_death(PID)
    assert engine.colour_of(1) == "red"
    assert engine.colour_of(1) == STATE_COLOUR[engine.state.state_of(1)]
    with pytest.raises(UnknownNode):
        engine.colour_of(999)


def test_malformed_webs_are_refused():
    n = WebNode(id=0, tier=0, ring_index=0, template="crypt")
    with pytest.raises(MalformedWeb):
        DescentEngine.new_profile(PID, Web(1, origin_id=5, nodes=(n,), edges=()))
    with pytest.raises(MalformedWeb):
        DescentEngine.new_profile(PID, Web(1, origin_id=0, nodes=(n, n), edges=()))
    with pytest.raises(MalformedWeb):
        DescentEngine.new_profile(PID, Web(1, origin_id=0, nodes=(n,), edges=(WebEdge(0, 9),)))
    with pytest.raises(MalformedWeb):
        DescentEngine.new_profile(PID, Web(1, origin_id=0, nodes=(n,), edges=(WebEdge(0, 0),)))


# --------------------------------------------------------------------------
# Opening a portal
# --------------------------------------------------------------------------


def test_open_consumes_sigil_and_makes_node_active():
    engine, probe, clock = make_engine(diamond_web())
    engine.add_sigil(PID, sigil("s1", 3, seed=0xBEEF))
    opened = engine.open_portal(PID, 1, "s1")

    assert opened == PortalOpened(
        node_id=1, template="crypt", map_seed=0xBEEF, sigil_tier=3, node_tier=1,
        mechanic=None, pinnacle=None,
    )
    assert "s1" not in engine.state.stash
    assert engine.state.state_of(1) is NodeState.ACTIVE
    inst = engine.state.instance
    assert inst is not None
    assert inst.node_id == 1 and inst.sigil.id == "s1"
    assert inst.map_seed == 0xBEEF, "the Sigil's own seed becomes the map seed"
    assert inst.has_boss is True and inst.elite_total == 5 and inst.elite_killed == 0
    assert inst.opened_tick == 101
    # The probe was asked about this template, this seed, at the Sigil's tier.
    assert probe.calls == [("crypt", 0xBEEF, 3)]
    last = engine.state.history[-1]
    assert (last.node_id, last.event, last.before, last.after, last.sigil_id, last.tick) == (
        1, Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "s1", 101,
    )


def test_open_uses_sigil_tier_for_density_not_node_tier():
    engine, probe, _ = make_engine(diamond_web())
    engine.add_sigil(PID, sigil("big", 12))
    engine.open_portal(PID, 1, "big")
    assert probe.calls[0][2] == 12


def test_open_refusals_in_order_with_distinct_exceptions():
    engine, probe, _ = make_engine(diamond_web())
    engine.add_sigil(PID, sigil("weak", 1))
    engine.add_sigil(PID, sigil("ok", 5))

    with pytest.raises(WrongProfile):
        engine.open_portal("someone-else", 1, "ok")
    with pytest.raises(UnknownNode):
        engine.open_portal(PID, 42, "nope")          # node checked before sigil
    with pytest.raises(NodeNotOpenable):
        engine.open_portal(PID, 3, "nope")           # locked; state before sigil
    with pytest.raises(NodeNotOpenable):
        engine.open_portal(PID, 0, "ok")             # cleared is not openable
    with pytest.raises(UnknownSigil):
        engine.open_portal(PID, 1, "nope")
    assert probe.calls == [], "no refusal reaches the probe"
    assert set(engine.state.stash) == {"weak", "ok"}, "a refusal never consumes a Sigil"

    engine.open_portal(PID, 1, "ok")
    with pytest.raises(InstanceAlreadyActive):
        engine.open_portal(PID, 2, "weak")
    assert "weak" in engine.state.stash
    assert engine.state.state_of(2) is NodeState.REACHABLE


def test_sigil_tier_below_node_tier_is_refused():
    engine, _, _ = make_engine(diamond_web())
    clear_with_boss(engine, 1, "a", tier=1)
    assert engine.state.state_of(3) is NodeState.REACHABLE  # tier 2
    engine.add_sigil(PID, sigil("t1", 1))
    with pytest.raises(SigilTooWeak):
        engine.open_portal(PID, 3, "t1")
    assert "t1" in engine.state.stash
    engine.add_sigil(PID, sigil("t2", 2))
    engine.open_portal(PID, 3, "t2")
    assert engine.state.state_of(3) is NodeState.ACTIVE


def test_probe_failure_does_not_eat_the_sigil():
    def exploding(template, map_seed, sigil_tier):
        raise RuntimeError("generator fell over")

    engine, _, _ = make_engine(diamond_web(), probe=exploding)
    engine.add_sigil(PID, sigil("s", 1))
    with pytest.raises(RuntimeError):
        engine.open_portal(PID, 1, "s")
    assert "s" in engine.state.stash
    assert engine.state.instance is None
    assert engine.state.state_of(1) is NodeState.REACHABLE


def test_add_sigil_checks_profile_and_rejects_duplicate_ids():
    engine, _, _ = make_engine(diamond_web())
    with pytest.raises(WrongProfile):
        engine.add_sigil("other", sigil("s", 1))
    engine.add_sigil(PID, sigil("s", 1))
    with pytest.raises(DuplicateSigil):
        engine.add_sigil(PID, sigil("s", 9))
    assert engine.state.stash["s"].tier == 1


# --------------------------------------------------------------------------
# Clearing
# --------------------------------------------------------------------------


def test_boss_kill_clears_and_unlocks_only_locked_neighbours():
    engine, _, _ = make_engine(diamond_web())
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    before = len(engine.state.history)

    assert engine.report_boss_kill(PID) is NodeState.CLEARED
    s = engine.state
    assert s.state_of(1) is NodeState.CLEARED
    assert s.instance is None, "the instance is destroyed on clear"
    assert s.state_of(3) is NodeState.REACHABLE, "the locked neighbour opens up"
    assert s.state_of(0) is NodeState.CLEARED, "the cleared neighbour is untouched"
    assert s.state_of(4) is NodeState.LOCKED, "two steps away stays locked"
    new = s.history[before:]
    assert [(e.node_id, e.event) for e in new] == [
        (1, Event.BOSS_KILLED), (3, Event.NEIGHBOUR_CLEARED),
    ]
    assert new[0].sigil_id == "s" and new[1].sigil_id is None
    assert new[0].tick == new[1].tick, "one clear is one moment in the ledger"


def test_propagation_never_touches_reachable_or_failed_neighbours():
    engine, _, _ = make_engine(diamond_web())
    # Fail node 2 so it is FAILED, then clear node 3 (whose neighbours are 1, 2, 4).
    clear_with_boss(engine, 1, "a", tier=1)
    engine.add_sigil(PID, sigil("b", 1))
    engine.open_portal(PID, 2, "b")
    engine.report_death(PID)
    assert engine.state.state_of(2) is NodeState.FAILED
    clear_with_boss(engine, 3, "c", tier=2)
    assert engine.state.state_of(2) is NodeState.FAILED, "failed neighbour left alone"
    assert engine.state.state_of(1) is NodeState.CLEARED
    assert engine.state.state_of(4) is NodeState.REACHABLE
    touched = [e.node_id for e in engine.state.history if e.event is Event.NEIGHBOUR_CLEARED]
    assert touched == [1, 2, 3, 4], "each node is made reachable exactly once"


def test_elites_only_map_clears_at_threshold_and_never_below():
    engine, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=False, elite_total=5))
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    # 80 percent of 5 packs is 4; the first three kills must not clear it.
    for _ in range(3):
        assert engine.report_elite_kill(PID) is None
        assert engine.state.state_of(1) is NodeState.ACTIVE
    assert engine.state.instance.elite_killed == 3
    assert engine.state.instance.elite_fraction() < ELITE_CLEAR_FRACTION
    assert engine.report_elite_kill(PID) is NodeState.CLEARED
    assert engine.state.state_of(1) is NodeState.CLEARED
    assert engine.state.instance is None
    assert engine.state.history[-2].event is Event.ELITES_MET
    assert engine.state.history[-1].event is Event.NEIGHBOUR_CLEARED  # node 3


@pytest.mark.parametrize("total,needed", [(1, 1), (4, 4), (7, 6), (10, 8), (12, 10)])
def test_elite_threshold_rounds_up(total, needed):
    engine, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=False, elite_total=total))
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    for _ in range(needed - 1):
        assert engine.report_elite_kill(PID) is None
    assert engine.report_elite_kill(PID) is NodeState.CLEARED


def test_boss_map_never_clears_via_elites():
    engine, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=True, elite_total=2))
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    for _ in range(50):
        assert engine.report_elite_kill(PID) is None
    assert engine.state.state_of(1) is NodeState.ACTIVE
    assert engine.state.instance.elite_killed == 50
    assert engine.report_boss_kill(PID) is NodeState.CLEARED


# --------------------------------------------------------------------------
# Failing
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "report,event",
    [("report_death", Event.DIED), ("report_abandon", Event.ABANDON), ("report_timeout", Event.TIMEOUT)],
)
def test_death_abandon_and_timeout_fail_the_node_and_consume_the_sigil(report, event):
    engine, _, _ = make_engine(diamond_web())
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    assert getattr(engine, report)(PID) is NodeState.FAILED
    s = engine.state
    assert s.state_of(1) is NodeState.FAILED
    assert s.instance is None
    assert "s" not in s.stash, "the Sigil is gone"
    assert s.state_of(3) is NodeState.LOCKED, "failing unlocks nothing"
    last = s.history[-1]
    assert (last.event, last.before, last.after, last.sigil_id) == (
        event, NodeState.ACTIVE, NodeState.FAILED, "s",
    )


def test_failed_node_reopens_with_equal_or_higher_tier_not_lower():
    engine, _, _ = make_engine(diamond_web())
    clear_with_boss(engine, 1, "a", tier=1)           # node 3 (tier 2) reachable
    engine.add_sigil(PID, sigil("first", 2))
    engine.open_portal(PID, 3, "first")
    engine.report_death(PID)
    assert engine.state.state_of(3) is NodeState.FAILED

    engine.add_sigil(PID, sigil("low", 1))
    with pytest.raises(SigilTooWeak):
        engine.open_portal(PID, 3, "low")
    assert "low" in engine.state.stash

    engine.add_sigil(PID, sigil("equal", 2))
    engine.open_portal(PID, 3, "equal")
    assert engine.state.state_of(3) is NodeState.ACTIVE
    assert "equal" not in engine.state.stash
    engine.report_abandon(PID)
    assert engine.state.state_of(3) is NodeState.FAILED

    engine.add_sigil(PID, sigil("higher", 9))
    engine.open_portal(PID, 3, "higher")
    assert engine.state.state_of(3) is NodeState.ACTIVE
    assert engine.report_boss_kill(PID) is NodeState.CLEARED


def test_reports_without_an_instance_are_refused():
    engine, _, _ = make_engine(diamond_web())
    for report in ("report_elite_kill", "report_boss_kill", "report_death", "report_abandon", "report_timeout"):
        with pytest.raises(NoActiveInstance):
            getattr(engine, report)(PID)
        with pytest.raises(WrongProfile):
            getattr(engine, report)("other")
    assert engine.state.history[-1].seq == 2, "nothing was recorded"


# --------------------------------------------------------------------------
# Cleared is terminal
# --------------------------------------------------------------------------


def test_cleared_node_rejects_every_event():
    engine, _, _ = make_engine(diamond_web())
    clear_with_boss(engine, 1, "a", tier=1)
    n = len(engine.state.history)
    for event in Event:
        with pytest.raises(IllegalTransition):
            engine._apply(1, event)
    assert engine.state.state_of(1) is NodeState.CLEARED
    assert len(engine.state.history) == n, "a refused move leaves no entry"
    # And the origin, cleared by genesis, is just as final through the public API.
    engine.add_sigil(PID, sigil("s", 15))
    for node_id in (0, 1):
        with pytest.raises(NodeNotOpenable):
            engine.open_portal(PID, node_id, "s")


def test_locked_node_rejects_open_and_instance_events():
    engine, _, _ = make_engine(diamond_web())
    for event in (Event.OPEN, Event.BOSS_KILLED, Event.ELITES_MET, Event.DIED, Event.ABANDON, Event.TIMEOUT):
        with pytest.raises(IllegalTransition):
            engine._apply(3, event)
    assert engine.state.state_of(3) is NodeState.LOCKED


# --------------------------------------------------------------------------
# Rewards: passive points, fragments, Pinnacles
# --------------------------------------------------------------------------


def test_passive_point_granted_once_per_mechanic_clear():
    engine, _, _ = make_engine(diamond_web())
    clear_with_boss(engine, 1, "a", tier=1)
    assert engine.state.passive_points == 0, "node 1 has no mechanic"
    clear_with_boss(engine, 3, "b", tier=2)
    assert engine.state.passive_points == 1, "node 3 is a breach"
    clear_with_boss(engine, 2, "c", tier=1)
    clear_with_boss(engine, 4, "d", tier=3)
    assert engine.state.passive_points == 1, "only the one mechanic node in this web"


def test_failing_a_mechanic_node_grants_nothing():
    engine, _, _ = make_engine(diamond_web())
    clear_with_boss(engine, 1, "a", tier=1)
    engine.add_sigil(PID, sigil("b", 2))
    engine.open_portal(PID, 3, "b")
    engine.report_death(PID)
    assert engine.state.passive_points == 0


def walk_chain_to_14(engine: DescentEngine) -> None:
    for i in range(1, 15):
        clear_with_boss(engine, i, f"chain{i}")
    assert engine.state.state_of(14) is NodeState.CLEARED
    for nid in ARBITER_GLYPHS + MONOLITH_GLYPHS + (ARENA_ID,):
        assert engine.state.state_of(nid) is NodeState.REACHABLE


def test_fragments_accrue_once_per_glyph_clear_and_unlock_the_arena():
    engine, _, _ = make_engine(pinnacle_web())
    walk_chain_to_14(engine)
    assert engine.state.fragments == {}

    engine.add_sigil(PID, sigil("arena0", 15))
    with pytest.raises(PinnacleLocked):
        engine.open_portal(PID, ARENA_ID, "arena0")
    assert "arena0" in engine.state.stash

    # Monolith fragments do not count towards the Arbiter.
    for k, nid in enumerate(MONOLITH_GLYPHS):
        clear_with_boss(engine, nid, f"mono{k}")
    assert engine.state.fragments == {Pinnacle.MONOLITH: 2}
    assert engine.state.unlocked_pinnacles == frozenset()
    with pytest.raises(PinnacleLocked):
        engine.open_portal(PID, ARENA_ID, "arena0")

    # Two Arbiter fragments: still locked.
    clear_with_boss(engine, ARBITER_GLYPHS[0], "arb0")
    clear_with_boss(engine, ARBITER_GLYPHS[1], "arb1")
    assert engine.state.fragments[Pinnacle.ARBITER] == 2
    with pytest.raises(PinnacleLocked):
        engine.open_portal(PID, ARENA_ID, "arena0")

    # The third unlocks it.
    clear_with_boss(engine, ARBITER_GLYPHS[2], "arb2")
    assert engine.state.fragments[Pinnacle.ARBITER] == FRAGMENTS_TO_UNLOCK
    assert engine.state.unlocked_pinnacles == frozenset({Pinnacle.ARBITER})
    opened = engine.open_portal(PID, ARENA_ID, "arena0")
    assert opened.pinnacle is Pinnacle.ARBITER and opened.node_tier == MAX_TIER
    assert engine.state.state_of(ARENA_ID) is NodeState.ACTIVE
    engine.report_boss_kill(PID)
    assert engine.state.state_of(ARENA_ID) is NodeState.CLEARED
    assert engine.state.fragments[Pinnacle.ARBITER] == FRAGMENTS_TO_UNLOCK, "an arena is not a glyph"
    assert engine.state.passive_points == 0, "no mechanic nodes in this web"


def test_a_failed_glyph_node_gives_no_fragment_until_actually_cleared():
    engine, _, _ = make_engine(pinnacle_web())
    walk_chain_to_14(engine)
    nid = ARBITER_GLYPHS[0]
    engine.add_sigil(PID, sigil("x", 15))
    engine.open_portal(PID, nid, "x")
    engine.report_timeout(PID)
    assert engine.state.fragments == {}
    clear_with_boss(engine, nid, "y")
    assert engine.state.fragments == {Pinnacle.ARBITER: 1}


# --------------------------------------------------------------------------
# Pre-roll candidates
# --------------------------------------------------------------------------


def test_portal_candidates_enumerate_openable_node_sigil_seed_triples():
    engine, _, _ = make_engine(pinnacle_web())
    walk_chain_to_14(engine)
    engine.add_sigil(PID, sigil("s15", 15, seed=15))
    engine.add_sigil(PID, sigil("s14", 14, seed=14))
    engine.add_sigil(PID, sigil("a01", 15, seed=1))
    with pytest.raises(WrongProfile):
        engine.portal_candidates("other")

    got = engine.portal_candidates(PID)
    openable = sorted(ARBITER_GLYPHS + MONOLITH_GLYPHS)   # arena excluded: locked
    assert [(c.node_id, c.sigil_id) for c in got] == [
        (nid, sid) for nid in openable for sid in ("a01", "s15")
    ], "tier-14 sigil unusable at tier 15; arena hidden while locked; sorted"
    assert all(c.template == "crypt" and c.sigil_tier == 15 for c in got)
    assert {c.sigil_id: c.map_seed for c in got} == {"a01": 1, "s15": 15}
    assert isinstance(got[0], PortalCandidate)

    # Failed nodes are candidates again; active ones are not.
    engine.open_portal(PID, ARBITER_GLYPHS[0], "a01")
    assert ARBITER_GLYPHS[0] not in {c.node_id for c in engine.portal_candidates(PID)}
    engine.report_death(PID)
    assert ARBITER_GLYPHS[0] in {c.node_id for c in engine.portal_candidates(PID)}


def test_portal_candidates_on_fresh_profile():
    engine, _, _ = make_engine(diamond_web())
    assert engine.portal_candidates(PID) == []
    engine.add_sigil(PID, sigil("s", 1, seed=77))
    assert engine.portal_candidates(PID) == [
        PortalCandidate(1, "s", 77, "crypt", 1),
        PortalCandidate(2, "s", 77, "ashen_ramparts", 1),
    ]


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------


def play_a_session(engine: DescentEngine) -> None:
    """A mixed run over the pinnacle web: clears, fails, elites, retries."""
    clear_with_boss(engine, 1, "a")
    engine.add_sigil(PID, sigil("b", 15))
    engine.open_portal(PID, 2, "b")
    engine.report_death(PID)
    engine.add_sigil(PID, sigil("c", 15))
    engine.open_portal(PID, 2, "c")
    engine.report_abandon(PID)
    for i in range(2, 15):
        clear_with_boss(engine, i, f"chain{i}")
    for k, nid in enumerate(ARBITER_GLYPHS):
        clear_with_boss(engine, nid, f"arb{k}")
    engine.add_sigil(PID, sigil("m", 15))
    engine.open_portal(PID, MONOLITH_GLYPHS[0], "m")
    engine.report_timeout(PID)
    engine.add_sigil(PID, sigil("arena", 15))
    engine.open_portal(PID, ARENA_ID, "arena")


def test_ledger_is_append_only_and_sequential():
    engine, _, _ = make_engine(pinnacle_web())
    snapshot = list(engine.state.history)
    play_a_session(engine)
    h = engine.state.history
    assert h[: len(snapshot)] == snapshot, "earlier entries are never rewritten"
    assert [e.seq for e in h] == list(range(1, len(h) + 1))
    assert all(isinstance(e, LedgerEntry) for e in h)
    ticks = [e.tick for e in h]
    assert ticks == sorted(ticks), "the clock only moves forward"
    # Every entry is a legal move of the table.
    assert all(next_state(e.before, e.event) is e.after for e in h)


def test_replay_reproduces_states_and_rewards():
    engine, _, _ = make_engine(pinnacle_web())
    play_a_session(engine)
    live = engine.state
    assert live.state_of(ARENA_ID) is NodeState.ACTIVE

    rebuilt = DescentEngine.replay(PID, pinnacle_web(), live.history)
    assert rebuilt.states == live.states
    assert rebuilt.history == live.history
    assert rebuilt.passive_points == live.passive_points
    assert rebuilt.fragments == live.fragments
    assert rebuilt.unlocked_pinnacles == live.unlocked_pinnacles == frozenset({Pinnacle.ARBITER})
    # The ledger does not carry the stash or the instance; those are the caller's.
    assert rebuilt.stash == {} and rebuilt.instance is None


def test_replay_over_a_fresh_profile_is_identity():
    fresh = DescentEngine.new_profile(PID, diamond_web())
    again = DescentEngine.replay(PID, diamond_web(), fresh.history)
    assert again.states == fresh.states and again.history == fresh.history


def test_replay_rejects_a_tampered_ledger():
    engine, _, _ = make_engine(diamond_web())
    clear_with_boss(engine, 1, "a", tier=1)
    h = list(engine.state.history)
    open_entry = next(e for e in h if e.event is Event.OPEN)

    # A gap in seq.
    with pytest.raises(LedgerMismatch):
        DescentEngine.replay(PID, diamond_web(), [e for e in h if e.seq != open_entry.seq])
    # A move the table forbids: DIED straight from REACHABLE.
    bad = LedgerEntry(open_entry.seq, 1, Event.DIED, NodeState.REACHABLE, NodeState.FAILED, "a")
    with pytest.raises(LedgerMismatch):
        DescentEngine.replay(PID, diamond_web(), [e if e.seq != open_entry.seq else bad for e in h])
    # A wrong 'before'.
    lied = LedgerEntry(open_entry.seq, 1, Event.OPEN, NodeState.FAILED, NodeState.ACTIVE, "a")
    with pytest.raises(LedgerMismatch):
        DescentEngine.replay(PID, diamond_web(), [e if e.seq != open_entry.seq else lied for e in h])
    # A genesis entry that disagrees with the fresh profile.
    forged = LedgerEntry(1, 2, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE)
    with pytest.raises(LedgerMismatch):
        DescentEngine.replay(PID, diamond_web(), [forged] + h[1:])


def test_engine_is_deterministic_for_the_same_script():
    runs = []
    for _ in range(2):
        engine, _, _ = make_engine(pinnacle_web())
        play_a_session(engine)
        runs.append(engine.state)
    assert runs[0].states == runs[1].states
    assert runs[0].history == runs[1].history
    assert runs[0].fragments == runs[1].fragments


# --------------------------------------------------------------------------
# The default probe against the real generator
# --------------------------------------------------------------------------


def test_default_probe_sees_the_boss_in_both_shipped_templates():
    crypt = default_map_probe("crypt", 12345, 3)
    assert crypt.has_boss is True
    assert isinstance(crypt.elite_total, int) and crypt.elite_total >= 0
    assert default_map_probe("crypt", 12345, 3) == crypt, "the probe is a pure function"
    # Its boss set piece is 'ramparts_warlord_v1', which does not say 'boss'.
    assert default_map_probe("ashen_ramparts", 7, 5).has_boss is True


# --------------------------------------------------------------------------
# Hardening after adversarial review: replay re-derives, ids are spent once,
# empty maps clear on open, the instance is held to the ledger
# --------------------------------------------------------------------------

from lucifer_descent.engine import CorruptInstance  # noqa: E402


def entries(engine: DescentEngine) -> List[LedgerEntry]:
    return list(engine.state.history)


def test_replay_accepts_a_profile_created_at_any_tick():
    for tick in (0, 7, 10**9):
        state = DescentEngine.new_profile(PID, diamond_web(), tick=tick)
        engine = DescentEngine(state, StubProbe(), Clock())
        clear_with_boss(engine, 1, "a", tier=1)
        rebuilt = DescentEngine.replay(PID, diamond_web(), state.history)
        assert rebuilt.states == state.states and rebuilt.history == state.history


def test_add_sigil_refuses_the_live_sigil_and_a_spent_one():
    engine, _, _ = make_engine(diamond_web())
    key = sigil("only-key", 3, seed=0xBEEF)
    engine.add_sigil(PID, key)
    engine.open_portal(PID, 1, "only-key")
    with pytest.raises(DuplicateSigil, match="funding the live instance"):
        engine.add_sigil(PID, engine.state.instance.sigil)
    engine.report_boss_kill(PID)
    with pytest.raises(DuplicateSigil, match="already spent"):
        engine.add_sigil(PID, key)
    with pytest.raises(DuplicateSigil):
        engine.add_sigil(PID, sigil("only-key", 15))  # same id, any tier
    assert engine.spent_sigil_ids() == {"only-key"}
    # A fresh engine over the same state knows what the ledger spent.
    again = DescentEngine(engine.state, StubProbe(), Clock())
    with pytest.raises(DuplicateSigil):
        again.add_sigil(PID, key)
    assert [e.sigil_id for e in engine.state.history if e.sigil_id] == ["only-key", "only-key"]


def test_open_records_the_probe_facts_on_the_ledger():
    engine, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=False, elite_total=7))
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    opened = engine.state.history[-1]
    assert (opened.event, opened.has_boss, opened.elite_total) == (Event.OPEN, False, 7)
    assert all(e.has_boss is None and e.elite_total is None for e in engine.state.history[:-1])
    engine.report_death(PID)
    assert engine.state.history[-1].has_boss is None, "facts travel on OPEN only"


def test_empty_bossless_map_clears_on_open():
    probe = StubProbe(has_boss=True, elite_total=5)
    engine, _, clock = make_engine(diamond_web(), probe=probe)
    engine.add_sigil(PID, sigil("s", 2))
    clear_with_boss(engine, 1, "a", tier=1)             # node 3 (mechanic) reachable
    probe.has_boss, probe.elite_total = False, 0         # the next map is empty
    opened = engine.open_portal(PID, 3, "s")
    assert opened.node_id == 3
    assert engine.state.state_of(3) is NodeState.CLEARED
    assert engine.state.instance is None
    assert engine.state.passive_points == 1, "clearing on open still rewards"
    assert engine.state.state_of(4) is NodeState.REACHABLE, "and still propagates"
    tail = engine.state.history[-3:]
    assert [(e.node_id, e.event) for e in tail] == [
        (3, Event.OPEN), (3, Event.ELITES_MET), (4, Event.NEIGHBOUR_CLEARED),
    ]
    assert len({e.tick for e in tail}) == 1, "one moment in the ledger"
    assert tail[0].sigil_id == tail[1].sigil_id == "s"
    with pytest.raises(NoActiveInstance):
        engine.report_elite_kill(PID)
    # A boss map with no elites does not clear on open.
    other, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=True, elite_total=0))
    other.add_sigil(PID, sigil("s", 1))
    other.open_portal(PID, 1, "s")
    assert other.state.state_of(1) is NodeState.ACTIVE and other.state.instance is not None
    # And replay holds the ledger to it.
    rebuilt = DescentEngine.replay(PID, diamond_web(), engine.state.history)
    assert rebuilt.states == engine.state.states


def test_map_probe_refuses_nonsense():
    for bad in (dict(has_boss=1, elite_total=1), dict(has_boss=True, elite_total=-1), dict(has_boss=True, elite_total=2.0)):
        with pytest.raises(ValueError):
            MapProbe(**bad)


def _played(web: Web = None, probe: StubProbe = None) -> Tuple[DescentEngine, List[LedgerEntry]]:
    engine, _, _ = make_engine(web or diamond_web(), probe=probe)
    clear_with_boss(engine, 1, "a", tier=1)
    return engine, entries(engine)


def _swap(h: List[LedgerEntry], seq: int, **changes) -> List[LedgerEntry]:
    import dataclasses
    return [dataclasses.replace(e, **changes) if e.seq == seq else e for e in h]


def test_replay_holds_the_clear_to_the_recorded_map():
    engine, h = _played()
    open_seq = next(e.seq for e in h if e.event is Event.OPEN)
    # The map had a boss and the boss was killed: fine.  Say it had none.
    with pytest.raises(LedgerMismatch, match="has no boss"):
        DescentEngine.replay(PID, diamond_web(), _swap(h, open_seq, has_boss=False, elite_total=3))
    # An elite clear on a boss map.
    elites, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=False, elite_total=1))
    elites.add_sigil(PID, sigil("e", 1))
    elites.open_portal(PID, 1, "e")
    elites.report_elite_kill(PID)
    h2 = entries(elites)
    open2 = next(e.seq for e in h2 if e.event is Event.OPEN)
    with pytest.raises(LedgerMismatch, match="has a boss"):
        DescentEngine.replay(PID, diamond_web(), _swap(h2, open2, has_boss=True))
    # Facts missing altogether.
    with pytest.raises(LedgerMismatch, match="no map facts"):
        DescentEngine.replay(PID, diamond_web(), _swap(h, open_seq, has_boss=None, elite_total=None))
    # A bossless empty map must clear at once.
    with pytest.raises(LedgerMismatch, match="must clear at once"):
        DescentEngine.replay(PID, diamond_web(), _swap(h, open_seq, has_boss=False, elite_total=0))


def test_replay_rejects_forged_propagation():
    engine, h = _played()
    n = len(h)
    far = LedgerEntry(n + 1, 4, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE, None, h[-1].tick)
    with pytest.raises(LedgerMismatch, match="no clear to propagate"):
        DescentEngine.replay(PID, diamond_web(), h + [far])
    # The clear of node 1 propagates to node 3; put something else first.
    clear_seq = next(e.seq for e in h if e.event is Event.BOSS_KILLED)
    prop = h[clear_seq]  # the entry after the clear: NEIGHBOUR_CLEARED on 3
    assert prop.event is Event.NEIGHBOUR_CLEARED and prop.node_id == 3
    import dataclasses
    interloper = dataclasses.replace(prop, node_id=4)
    with pytest.raises(LedgerMismatch, match="expected neighbour_cleared on node 3"):
        DescentEngine.replay(PID, diamond_web(), h[:clear_seq] + [interloper])
    # Same batch, wrong tick.
    with pytest.raises(LedgerMismatch, match="carries tick"):
        DescentEngine.replay(PID, diamond_web(), _swap(h, prop.seq, tick=prop.tick + 1))
    # The genesis batch: node 2 before node 1, or one of them at another tick.
    with pytest.raises(LedgerMismatch):
        DescentEngine.replay(PID, diamond_web(), [h[1], h[0]] + h[2:])
    with pytest.raises(LedgerMismatch, match="carries tick"):
        DescentEngine.replay(PID, diamond_web(), _swap(h, 2, tick=99))


def test_replay_rejects_forged_opens():
    engine, h = _played(pinnacle_web())
    n = len(h)
    tick = h[-1].tick
    # An arena opened while its Pinnacle is locked, on a ledger that walked
    # the chain honestly.
    walker, _, _ = make_engine(pinnacle_web())
    walk_chain_to_14(walker)
    hw = entries(walker)
    m = len(hw)
    forged = hw + [
        LedgerEntry(m + 1, ARENA_ID, Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "x", tick, True, 1),
    ]
    with pytest.raises(LedgerMismatch, match="opened while locked"):
        DescentEngine.replay(PID, pinnacle_web(), forged)
    # A second instance while one is live.
    two = hw + [
        LedgerEntry(m + 1, ARBITER_GLYPHS[0], Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "x", tick, True, 1),
        LedgerEntry(m + 2, ARBITER_GLYPHS[1], Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "y", tick, True, 1),
    ]
    with pytest.raises(LedgerMismatch, match="already active"):
        DescentEngine.replay(PID, pinnacle_web(), two)
    # A Sigil spent twice, and a consuming event with the wrong Sigil.
    spent = next(e.sigil_id for e in hw if e.event is Event.OPEN)
    twice = hw + [
        LedgerEntry(m + 1, ARBITER_GLYPHS[0], Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, spent, tick, True, 1),
    ]
    with pytest.raises(LedgerMismatch, match="already spent"):
        DescentEngine.replay(PID, pinnacle_web(), twice)
    wrong_key = hw + [
        LedgerEntry(m + 1, ARBITER_GLYPHS[0], Event.OPEN, NodeState.REACHABLE, NodeState.ACTIVE, "x", tick, True, 1),
        LedgerEntry(m + 2, ARBITER_GLYPHS[0], Event.DIED, NodeState.ACTIVE, NodeState.FAILED, "y", tick),
    ]
    with pytest.raises(LedgerMismatch, match="funded by 'x'"):
        DescentEngine.replay(PID, pinnacle_web(), wrong_key)
    # No sigil at all.
    open_seq = next(e.seq for e in h if e.event is Event.OPEN)
    with pytest.raises(LedgerMismatch, match="no sigil id"):
        DescentEngine.replay(PID, pinnacle_web(), _swap(h, open_seq, sigil_id=None))


def test_replay_of_a_ledger_cut_mid_propagation_stops_at_the_cut():
    engine, h = _played()
    # h ends with BOSS_KILLED on 1 then NEIGHBOUR_CLEARED on 3; drop the last.
    assert h[-1].event is Event.NEIGHBOUR_CLEARED
    cut = DescentEngine.replay(PID, diamond_web(), h[:-1])
    assert cut.state_of(1) is NodeState.CLEARED and cut.state_of(3) is NodeState.LOCKED
    assert cut.history == h[:-1], "nothing is invented"
    # An empty ledger is the cut before genesis propagation.
    empty = DescentEngine.replay(PID, diamond_web(), [])
    assert empty.states == {0: NodeState.CLEARED, 1: NodeState.LOCKED, 2: NodeState.LOCKED, 3: NodeState.LOCKED, 4: NodeState.LOCKED}


def test_replay_rejects_seq_that_is_not_the_next_one():
    engine, h = _played()
    for bad in (0, -1, -5, 99):
        with pytest.raises(LedgerMismatch, match="arrived when"):
            DescentEngine.replay(PID, diamond_web(), [LedgerEntry(bad, 1, Event.NEIGHBOUR_CLEARED, NodeState.LOCKED, NodeState.REACHABLE)])
    with pytest.raises(LedgerMismatch):
        DescentEngine.replay(PID, diamond_web(), [h[0], h[1], h[0], h[1]] + h[2:])


def test_reports_refuse_an_instance_that_disagrees_with_the_ledger():
    def live() -> DescentEngine:
        engine, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=True, elite_total=40))
        engine.add_sigil(PID, sigil("s", 1))
        engine.open_portal(PID, 1, "s")
        return engine

    engine = live()
    engine.state.instance.has_boss = False
    engine.state.instance.elite_total = 0
    with pytest.raises(CorruptInstance, match="ledger's open"):
        engine.report_elite_kill(PID)
    assert engine.state.state_of(1) is NodeState.ACTIVE, "nothing was applied"

    engine = live()
    engine.state.instance.sigil = sigil("other", 1)
    with pytest.raises(CorruptInstance, match="funded by"):
        engine.report_boss_kill(PID)

    engine = live()
    engine.state.instance.opened_tick += 1
    with pytest.raises(CorruptInstance, match="tick"):
        engine.report_death(PID)

    engine = live()
    engine.state.instance.node_id = 2
    with pytest.raises(CorruptInstance, match="the node is reachable"):
        engine.report_abandon(PID)

    engine = live()
    engine.state.instance.elite_killed = -1
    with pytest.raises(CorruptInstance, match="elite kills"):
        engine.report_timeout(PID)

    # A bossless instance already past its threshold cannot come from play.
    engine, _, _ = make_engine(diamond_web(), probe=StubProbe(has_boss=False, elite_total=5))
    engine.add_sigil(PID, sigil("s", 1))
    engine.open_portal(PID, 1, "s")
    engine.state.instance.elite_killed = 4
    with pytest.raises(CorruptInstance, match="past the threshold"):
        engine.report_elite_kill(PID)

    # The honest instance still plays.
    engine = live()
    assert engine.report_elite_kill(PID) is None
    assert engine.report_boss_kill(PID) is NodeState.CLEARED


def test_engine_reads_the_web_once():
    import dataclasses
    engine, _, _ = make_engine(diamond_web())
    web = engine.state.web
    engine.state.web = dataclasses.replace(web, edges=web.edges + (WebEdge(1, 4),))
    clear_with_boss(engine, 1, "a", tier=1)
    assert engine.state.state_of(4) is NodeState.LOCKED, "propagation used the construction-time web"
    assert engine.state.state_of(3) is NodeState.REACHABLE
    with pytest.raises(MalformedWeb):
        DescentEngine(DescentEngine.new_profile(PID, diamond_web()).__class__(
            profile_id=PID, web=Web(1, 0, (WebNode(0, 0, 0, "t"),), (WebEdge(0, 3),)), states={0: NodeState.CLEARED}
        ), StubProbe(), Clock())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
