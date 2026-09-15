"""Tests for Sigil minting, the sustain drop rule and the pre-roll enumerator.

Spec: docs/WORLD_BIBLE.md section 03.

Three claims carry the weight here.

*Minting is a pure function and ids never collide.*  The same triple mints
the same Sigil, and 100 000 mints under one profile give 100 000 distinct
ids.  A random 32-bit id would fail that with about 69 percent probability
(birthday bound), so the set check is the proof that ids are a permutation
of the counter and not a draw.

*The sustain rule has the documented expectation.*  Over 20 000 rolls the
count is uniform on 1..3, the tier offsets sit at 0.2 / 0.5 / 0.3, and the
drift is +0.1 per Sigil.  Tolerances are set at four to five standard errors
so the test is stable yet would catch a wrong weight.

*The pre-roll enumerator applies exactly the node and Sigil rules.*  A hand
built chain web puts every node state and both kinds of Pinnacle node in
play, so each rule has a node that exercises it and one that does not.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# Runnable as `pytest tests/test_descent_sigils.py` or
# `python3 tests/test_descent_sigils.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_descent.contracts import (
    MAX_TIER,
    NodeState,
    Pinnacle,
    ProfileState,
    Sigil,
    Web,
    WebEdge,
    WebNode,
)
from lucifer_descent import sigils as S
from lucifer_gen.seed import MASK64

PROFILE = 0xC0FFEE_D00D_1234
OTHER_PROFILE = 0xBAD_5EED

ID_PATTERN = re.compile(r"^sg-[0-9a-f]{8}$")


# --------------------------------------------------------------------------
# Minting
# --------------------------------------------------------------------------


def test_mint_is_deterministic_and_well_formed():
    a = S.mint_sigil(PROFILE, 42, 7)
    b = S.mint_sigil(PROFILE, 42, 7)
    assert a == b
    assert isinstance(a, Sigil)
    assert a.tier == 7
    assert ID_PATTERN.match(a.id), a.id
    assert 0 <= a.seed <= MASK64
    # The id is a pure function too, reachable without minting.
    assert S.sigil_id(PROFILE, 42, 7) == a.id
    # The tier rides in the first hex digit and reads back.
    assert S.tier_from_id(a.id) == 7
    assert a.id[3] == "7"


def test_mint_masks_the_profile_seed_to_64_bits():
    # The same seed with bits above 64 set names the same profile, as it
    # does for the web generator.
    assert S.mint_sigil(PROFILE, 1, 3) == S.mint_sigil(PROFILE | (1 << 70), 1, 3)
    assert S.mint_sigil(MASK64, 0, 15) == S.mint_sigil(-1, 0, 15)


def test_mint_differs_across_each_input():
    base = S.mint_sigil(PROFILE, 10, 5)
    by_counter = S.mint_sigil(PROFILE, 11, 5)
    by_tier = S.mint_sigil(PROFILE, 10, 6)
    by_profile = S.mint_sigil(OTHER_PROFILE, 10, 5)
    assert base.id != by_counter.id and base.seed != by_counter.seed
    assert base.id != by_tier.id and base.seed != by_tier.seed
    assert base.id != by_profile.id and base.seed != by_profile.seed


def test_ids_are_keyed_per_profile():
    # The permutation key comes from the profile seed, so the same counter
    # reads differently almost everywhere in another profile.  A bijection
    # under one key may agree with another key's on a few points; 99 percent
    # disagreement over a thousand counters is far past chance.
    differ = sum(
        S.sigil_id(PROFILE, c, 4) != S.sigil_id(OTHER_PROFILE, c, 4) for c in range(1000)
    )
    assert differ >= 990


def test_no_id_collision_over_100k_mints():
    ids = set()
    seeds = set()
    for counter in range(100_000):
        sigil = S.mint_sigil(PROFILE, counter, 1 + counter % MAX_TIER)
        ids.add(sigil.id)
        seeds.add(sigil.seed)
    assert len(ids) == 100_000
    # Seeds are 64-bit draws and need not be unique, but for this fixed
    # profile they are, and the run is deterministic, so pin that too.
    assert len(seeds) == 100_000


def test_same_counter_every_tier_is_distinct():
    # Injective in (counter, tier), not just in counter: the tier nibble.
    ids = {S.sigil_id(PROFILE, 5, tier) for tier in range(1, MAX_TIER + 1)}
    assert len(ids) == MAX_TIER
    assert sorted(S.tier_from_id(i) for i in ids) == list(range(1, MAX_TIER + 1))


def test_permutation_is_a_bijection_on_a_prefix():
    key_in, key_out = S._profile_key(PROFILE)
    n = 1 << 17
    assert len({S._permute28(v, key_in, key_out) for v in range(n)}) == n
    # Every output stays inside 28 bits so the tier nibble is never touched.
    assert all(S._permute28(v, key_in, key_out) <= S.MAX_COUNTER for v in range(0, n, 97))
    assert S._permute28(S.MAX_COUNTER, key_in, key_out) <= S.MAX_COUNTER


@pytest.mark.parametrize("tier", [0, 16, -1, 100])
def test_mint_refuses_tier_out_of_range(tier):
    with pytest.raises(ValueError):
        S.mint_sigil(PROFILE, 0, tier)


@pytest.mark.parametrize("counter", [-1, S.MAX_COUNTER + 1])
def test_mint_refuses_counter_out_of_range(counter):
    with pytest.raises(ValueError):
        S.mint_sigil(PROFILE, counter, 1)


def test_mint_accepts_the_counter_limits():
    assert S.mint_sigil(PROFILE, 0, 1).id != S.mint_sigil(PROFILE, S.MAX_COUNTER, 1).id


@pytest.mark.parametrize("bad", [1.5, "3", True, None])
def test_mint_refuses_non_int_arguments(bad):
    with pytest.raises(TypeError):
        S.mint_sigil(PROFILE, bad, 1)
    with pytest.raises(TypeError):
        S.mint_sigil(PROFILE, 0, bad)
    with pytest.raises(TypeError):
        S.mint_sigil(bad, 0, 1)


@pytest.mark.parametrize(
    "text",
    ["", "sg-", "sg-0000000", "sg-000000000", "SG-1a2b3c4d", "sg-1A2B3C4D",
     "sg-g1a2b3c4", "sg-01a2b3c4", "xx-1a2b3c4d", 12345],
)
def test_tier_from_id_refuses_malformed(text):
    with pytest.raises(ValueError):
        S.tier_from_id(text)


# --------------------------------------------------------------------------
# Drops: the sustain rule
# --------------------------------------------------------------------------


def _roll_many(n: int, cleared_tier: int, label: str) -> Tuple[List[List[Sigil]], int]:
    """``n`` rolls, advancing the counter by ``len(drops)`` as the docstring asks."""
    counter = 0
    rolls: List[List[Sigil]] = []
    for i in range(n):
        drops = S.roll_drops(PROFILE, counter, cleared_tier, f"{label}:{i}")
        counter += len(drops)
        rolls.append(drops)
    return rolls, counter


def test_drop_count_and_tier_distribution():
    n = 20_000
    t = 8  # away from both clamps
    rolls, counter = _roll_many(n, t, "dist")

    counts = Counter(len(r) for r in rolls)
    assert set(counts) <= {1, 2, 3}
    # Uniform on 1..3: each third within 4.5 standard errors (0.0033).
    for k in (1, 2, 3):
        assert abs(counts[k] / n - 1 / 3) < 0.015, counts
    # Mean count 2.0 within 5 standard errors (0.0058).
    assert abs(counter / n - S.EXPECTED_DROPS_PER_CLEAR) < 0.03

    sigils = [s for r in rolls for s in r]
    total = len(sigils)
    assert total == counter
    offsets = Counter(s.tier - t for s in sigils)
    assert set(offsets) <= {-1, 0, 1}
    # Weights 2, 5, 3 -> 0.2, 0.5, 0.3 within ~5 standard errors (0.0025).
    assert abs(offsets[-1] / total - 0.2) < 0.012, offsets
    assert abs(offsets[0] / total - 0.5) < 0.012, offsets
    assert abs(offsets[1] / total - 0.3) < 0.012, offsets
    # The documented drift: E[tier] = t + 0.1.
    drift = sum(s.tier - t for s in sigils) / total
    assert abs(drift - S.EXPECTED_TIER_DRIFT) < 0.012
    # The "sustains" half: 80 percent of drops can reopen the tier just cleared.
    sustain = sum(1 for s in sigils if s.tier >= t) / total
    assert abs(sustain - 0.8) < 0.012

    # Every dropped Sigil is a distinct item.
    assert len({s.id for s in sigils}) == total


def test_drop_tiers_clamp_at_both_ends():
    n = 5_000
    low, _ = _roll_many(n, 1, "low")
    low_tiers = Counter(s.tier for r in low for s in r)
    assert set(low_tiers) == {1, 2}
    total = sum(low_tiers.values())
    # The -1 offset folds onto tier 1: P(1) = 0.7, P(2) = 0.3.
    assert abs(low_tiers[1] / total - 0.7) < 0.03
    assert abs(low_tiers[2] / total - 0.3) < 0.03

    high, _ = _roll_many(n, MAX_TIER, "high")
    high_tiers = Counter(s.tier for r in high for s in r)
    assert set(high_tiers) == {MAX_TIER - 1, MAX_TIER}
    total = sum(high_tiers.values())
    # The +1 offset folds onto tier 15: P(15) = 0.8, P(14) = 0.2.
    assert abs(high_tiers[MAX_TIER] / total - 0.8) < 0.03
    assert abs(high_tiers[MAX_TIER - 1] / total - 0.2) < 0.03


def test_clamp_tier():
    assert S.clamp_tier(0) == 1
    assert S.clamp_tier(1) == 1
    assert S.clamp_tier(8) == 8
    assert S.clamp_tier(15) == 15
    assert S.clamp_tier(16) == 15
    assert S.clamp_tier(-40) == 1


def test_drops_are_deterministic_and_label_sensitive():
    a = S.roll_drops(PROFILE, 100, 6, "clear:17")
    b = S.roll_drops(PROFILE, 100, 6, "clear:17")
    assert a == b
    assert 1 <= len(a) <= 3
    assert all(isinstance(s, Sigil) for s in a)

    # The Sigils are the mint of (counter, counter + 1, ...): the drop roll
    # adds nothing to a Sigil that minting would not.
    for i, sigil in enumerate(a):
        assert sigil == S.mint_sigil(PROFILE, 100 + i, sigil.tier)

    # Another label is another roll; over many labels the rolls differ.
    others = [S.roll_drops(PROFILE, 100, 6, f"clear:{k}") for k in range(18, 48)]
    assert any([len(o) for o in others] != [len(a)] or
               [s.tier for s in o] != [s.tier for s in a] for o in others)

    # Another counter is other Sigils, even under the same label.
    c = S.roll_drops(PROFILE, 200, 6, "clear:17")
    assert [s.tier for s in c] == [s.tier for s in a]
    assert {s.id for s in c}.isdisjoint({s.id for s in a})


def test_roll_drops_refuses_bad_arguments():
    with pytest.raises(ValueError):
        S.roll_drops(PROFILE, 0, 0, "x")  # tier 0 is the origin, never cleared
    with pytest.raises(ValueError):
        S.roll_drops(PROFILE, 0, 16, "x")
    with pytest.raises(ValueError):
        S.roll_drops(PROFILE, -1, 5, "x")
    with pytest.raises(ValueError):
        S.roll_drops(PROFILE, 0, 5, "")
    with pytest.raises(ValueError):
        S.roll_drops(PROFILE, 0, 5, None)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        S.roll_drops(PROFILE, 0, 5.0, "x")  # type: ignore[arg-type]


def test_roll_drops_refuses_counter_with_no_room():
    # At the very top of the counter space a roll may not fit; it must refuse
    # rather than mint past MAX_COUNTER or silently drop fewer.
    counter = S.MAX_COUNTER  # room for exactly one
    seen_refusal = False
    for i in range(50):
        try:
            drops = S.roll_drops(PROFILE, counter, 5, f"edge:{i}")
        except ValueError:
            seen_refusal = True
            continue
        assert len(drops) == 1
    assert seen_refusal


# --------------------------------------------------------------------------
# Pre-roll enumerator and the stash summary
# --------------------------------------------------------------------------

# A chain 0-1-2-...-14 with four tier-15 nodes hanging off node 14: a plain
# one, an Arbiter arena, a Monolith arena and an Arbiter glyph source.
PLAIN15, ARENA_A, ARENA_M, GLYPH_A = 15, 16, 17, 18


def _chain_web() -> Web:
    nodes = [WebNode(id=i, tier=i, ring_index=0, template="crypt") for i in range(15)]
    nodes += [
        WebNode(id=PLAIN15, tier=15, ring_index=0, template="crypt"),
        WebNode(id=ARENA_A, tier=15, ring_index=1, template="crypt", pinnacle=Pinnacle.ARBITER),
        WebNode(id=ARENA_M, tier=15, ring_index=2, template="crypt", pinnacle=Pinnacle.MONOLITH),
        WebNode(id=GLYPH_A, tier=15, ring_index=3, template="crypt", glyph=Pinnacle.ARBITER),
    ]
    edges = [WebEdge(i, i + 1) for i in range(14)]
    edges += [WebEdge(14, n) for n in (PLAIN15, ARENA_A, ARENA_M, GLYPH_A)]
    return Web(profile_seed=PROFILE, origin_id=0, nodes=tuple(nodes), edges=tuple(edges))


def _state(stash: List[Sigil], **overrides) -> ProfileState:
    web = _chain_web()
    states = {n.id: NodeState.LOCKED for n in web.nodes}
    states[0] = NodeState.CLEARED
    states.update({int(k[1:]): v for k, v in overrides.items()})
    return ProfileState(
        profile_id="p1",
        web=web,
        states=states,
        stash={s.id: s for s in stash},
    )


@pytest.fixture
def stash() -> Dict[str, Sigil]:
    return {
        "t1": S.mint_sigil(PROFILE, 0, 1),
        "t5": S.mint_sigil(PROFILE, 1, 5),
        "t15a": S.mint_sigil(PROFILE, 2, 15),
        "t15b": S.mint_sigil(PROFILE, 3, 15),
    }


def test_prewarm_applies_state_tier_and_arena_rules(stash):
    t1, t5, t15a, t15b = stash["t1"], stash["t5"], stash["t15a"], stash["t15b"]
    state = _state(
        list(stash.values()),
        n1=NodeState.REACHABLE,
        n2=NodeState.LOCKED,
        n3=NodeState.CLEARED,
        n4=NodeState.ACTIVE,
        n5=NodeState.FAILED,
        n15=NodeState.REACHABLE,
        n16=NodeState.REACHABLE,   # Arbiter arena, Arbiter locked
        n17=NodeState.REACHABLE,   # Monolith arena, Monolith unlocked
        n18=NodeState.FAILED,      # Arbiter glyph source: not an arena, not gated
    )
    state.unlocked_pinnacles = frozenset({Pinnacle.MONOLITH})

    got = S.prewarm_candidates(state)

    def triples(node_id: int, *sigils: Sigil):
        return sorted((node_id, s.id, s.seed) for s in sigils)

    expected = (
        triples(1, t1, t5, t15a, t15b)      # tier 1: every Sigil qualifies
        + triples(5, t5, t15a, t15b)        # tier 5, FAILED: retry allowed
        + triples(PLAIN15, t15a, t15b)      # tier 15: only tier-15 Sigils
        + triples(ARENA_M, t15a, t15b)      # unlocked arena
        + triples(GLYPH_A, t15a, t15b)      # glyph node is never gated
    )
    assert got == expected
    # LOCKED (2), CLEARED (3), ACTIVE (4) and the locked arena (16) are absent.
    assert {n for n, _, _ in got}.isdisjoint({2, 3, 4, ARENA_A})
    # The map seed is the Sigil's own seed, always.
    for node_id, sigil_id, map_seed in got:
        assert state.stash[sigil_id].seed == map_seed
        assert state.stash[sigil_id].tier >= state.web.node(node_id).tier
        assert state.states[node_id] in (NodeState.REACHABLE, NodeState.FAILED)


def test_prewarm_unlocking_a_pinnacle_adds_its_arena(stash):
    state = _state(list(stash.values()), n16=NodeState.REACHABLE, n17=NodeState.REACHABLE)
    assert S.prewarm_candidates(state) == []

    state.unlocked_pinnacles = frozenset({Pinnacle.ARBITER})
    got = S.prewarm_candidates(state)
    assert {n for n, _, _ in got} == {ARENA_A}
    assert len(got) == 2  # the two tier-15 Sigils

    state.unlocked_pinnacles = frozenset({Pinnacle.ARBITER, Pinnacle.MONOLITH})
    assert {n for n, _, _ in S.prewarm_candidates(state)} == {ARENA_A, ARENA_M}


def test_prewarm_is_sorted_and_stable(stash):
    state = _state(list(stash.values()), n1=NodeState.REACHABLE, n5=NodeState.REACHABLE)
    got = S.prewarm_candidates(state)
    assert got == sorted(got)
    assert got == S.prewarm_candidates(state)
    # Insertion order of the stash dict must not leak into the output.
    reversed_stash = dict(reversed(list(state.stash.items())))
    state.stash = reversed_stash
    assert S.prewarm_candidates(state) == got


def test_prewarm_ignores_the_active_instance_rule(stash):
    # Pre-rolling while a run is in progress is the point; the enumerator
    # does not look at ``state.instance`` at all.
    from lucifer_descent.contracts import Instance

    state = _state(list(stash.values()), n1=NodeState.REACHABLE, n4=NodeState.ACTIVE)
    state.instance = Instance(node_id=4, sigil=stash["t5"], map_seed=1, has_boss=True, elite_total=0)
    got = S.prewarm_candidates(state)
    assert {n for n, _, _ in got} == {1}
    assert len(got) == 4


def test_prewarm_with_empty_stash_or_nothing_openable(stash):
    assert S.prewarm_candidates(_state([], n1=NodeState.REACHABLE)) == []
    assert S.prewarm_candidates(_state(list(stash.values()))) == []


def test_prewarm_too_weak_sigils_open_nothing():
    weak = [S.mint_sigil(PROFILE, 0, 2), S.mint_sigil(PROFILE, 1, 4)]
    state = _state(weak, n5=NodeState.REACHABLE, n15=NodeState.FAILED)
    assert S.prewarm_candidates(state) == []
    just_enough = [S.mint_sigil(PROFILE, 2, 5)]
    state = _state(just_enough, n5=NodeState.REACHABLE, n15=NodeState.FAILED)
    assert [(n, t) for n, _, t in S.prewarm_candidates(state)] == [(5, just_enough[0].seed)]


def test_prewarm_refuses_a_state_naming_an_unknown_node(stash):
    state = _state(list(stash.values()), n1=NodeState.REACHABLE)
    state.states[999] = NodeState.REACHABLE
    with pytest.raises(KeyError):
        S.prewarm_candidates(state)


def test_node_is_openable_predicate(stash):
    state = _state([], n1=NodeState.REACHABLE, n5=NodeState.FAILED, n16=NodeState.REACHABLE)
    web = state.web
    assert S.node_is_openable(state, web.node(1))
    assert S.node_is_openable(state, web.node(5))
    assert not S.node_is_openable(state, web.node(0))    # CLEARED
    assert not S.node_is_openable(state, web.node(2))    # LOCKED
    assert not S.node_is_openable(state, web.node(ARENA_A))  # arena, locked
    state.unlocked_pinnacles = frozenset({Pinnacle.ARBITER})
    assert S.node_is_openable(state, web.node(ARENA_A))


def test_stash_summary(stash):
    state = _state(list(stash.values()))
    assert S.stash_summary(state) == {1: 1, 5: 1, 15: 2}
    assert list(S.stash_summary(state)) == [1, 5, 15]  # ascending
    assert sum(S.stash_summary(state).values()) == len(state.stash)
    assert S.stash_summary(_state([])) == {}

    many = [S.mint_sigil(PROFILE, c, 1 + c % 3) for c in range(30)]
    assert S.stash_summary(_state(many)) == {1: 10, 2: 10, 3: 10}


# --------------------------------------------------------------------------
# Against a generated web, when the sibling generator is present
# --------------------------------------------------------------------------


def test_prewarm_on_a_generated_web():
    W = pytest.importorskip("lucifer_descent.web")
    if not hasattr(W, "generate_web"):
        pytest.skip("web generator has no generate_web")
    web = W.generate_web(PROFILE)
    states = {n.id: NodeState.LOCKED for n in web.nodes}
    states[web.origin_id] = NodeState.CLEARED
    for nb in web.neighbours(web.origin_id):
        states[nb] = NodeState.REACHABLE
    # And make the whole outer ring openable so arenas and glyphs are in play.
    for n in web.nodes_at_tier(MAX_TIER):
        states[n.id] = NodeState.FAILED
    stash = [S.mint_sigil(PROFILE, c, 1 + c % MAX_TIER) for c in range(45)]
    state = ProfileState(profile_id="p", web=web, states=states, stash={s.id: s for s in stash})

    got = S.prewarm_candidates(state)
    assert got == sorted(got)
    assert got, "ring 1 is reachable and the stash has tier-1 Sigils"
    nodes = {n.id: n for n in web.nodes}
    for node_id, sigil_id, map_seed in got:
        node = nodes[node_id]
        assert states[node_id] in (NodeState.REACHABLE, NodeState.FAILED)
        assert state.stash[sigil_id].tier >= node.tier
        assert state.stash[sigil_id].seed == map_seed
        assert node.pinnacle is None  # nothing unlocked yet
    # Every openable non-arena node with a qualifying Sigil is present.
    expected_nodes = {
        n.id for n in web.nodes
        if states[n.id] in (NodeState.REACHABLE, NodeState.FAILED) and n.pinnacle is None
    }
    assert {n for n, _, _ in got} == expected_nodes

    state.unlocked_pinnacles = frozenset(Pinnacle)
    with_arenas = {n for n, _, _ in S.prewarm_candidates(state)}
    arenas = {n.id for n in web.nodes if n.pinnacle is not None}
    assert arenas and arenas <= with_arenas


# --------------------------------------------------------------------------
# unmint / is_genuine: a Sigil can be checked against its mint
# --------------------------------------------------------------------------


def test_unmint_inverts_sigil_id():
    import random
    rnd = random.Random(7)
    for _ in range(2000):
        seed = rnd.getrandbits(64)
        counter = rnd.randint(0, S.MAX_COUNTER)
        tier = rnd.randint(S.MIN_TIER, MAX_TIER)
        assert S.unmint(seed, S.sigil_id(seed, counter, tier)) == (counter, tier)
    assert S.unmint(PROFILE, S.sigil_id(PROFILE, S.MAX_COUNTER, 15)) == (S.MAX_COUNTER, 15)
    assert S.unmint(PROFILE, S.sigil_id(PROFILE, 0, 1)) == (0, 1)
    for bad in ("sg-0000000", "sg-00000000", "xx-12345678", "SG-12345678", 42):
        with pytest.raises(ValueError):
            S.unmint(PROFILE, bad)


def test_is_genuine_checks_id_tier_and_seed():
    sigil = S.mint_sigil(PROFILE, 12, 7)
    assert S.is_genuine(PROFILE, sigil)
    assert not S.is_genuine(PROFILE ^ 1, sigil), "another profile's mint"
    assert not S.is_genuine(PROFILE, S.Sigil(sigil.id, 7, (sigil.seed + 1) & S.MASK64))
    assert not S.is_genuine(PROFILE, S.Sigil(sigil.id, 8, sigil.seed))
    assert not S.is_genuine(PROFILE, S.Sigil("hand-made", 1, 1))
    assert not S.is_genuine(PROFILE, S.Sigil("sg-f0000001", 15, 1)), "well-formed id, seed not drawn by the mint"
    drops = S.roll_drops(PROFILE, 100, 5, "t.drop")
    assert all(S.is_genuine(PROFILE, d) for d in drops)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
