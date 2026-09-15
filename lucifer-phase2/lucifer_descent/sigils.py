"""Sigils: minting, the sustain drop rule, and the pre-roll enumerator.

Spec: docs/WORLD_BIBLE.md section 03 -- "Sigils are consumable keys tiered 1
to 15. Inserting one opens a portal into a reachable node; the Sigil's own
item seed becomes the map seed", and "the 170HX pre-rolls layouts for
reachable nodes so portals open instantly".

Three things live here and nothing else:

* :func:`mint_sigil` -- one Sigil from ``(profile_seed, counter, tier)``.
* :func:`roll_drops` -- the sustain rule: what clearing a tier-``t`` node
  yields, returned as freshly minted Sigils.
* :func:`prewarm_candidates` and :func:`stash_summary` -- read-only views over
  a :class:`ProfileState` for the 170HX and the table UI.

Nothing here mutates a profile.  The mint counter that keeps ids unique
belongs to whoever owns the profile (the engine, or storage); this module only
says how far it must advance after each call.

Randomness
----------
Every draw comes from a :class:`lucifer_gen.seed.Stream`, opened through
:meth:`SeedFields.stream` exactly as the web generator does.  The labels
here start with ``sigil.``, which the seed module does not recognise, so
each stream is fed the whole 64-bit profile seed.  That is intended: the
field split in ``seed.py`` describes *map* seeds, and a profile seed has no
such structure.  Each purpose still has its own label, so the drop roll for
one clear cannot shift the seed of an unrelated mint.

Why an id is a permutation and not a draw
-----------------------------------------
A Sigil id is ``sg-`` plus eight hex digits, 32 bits.  Drawing those bits at
random would collide: by the birthday bound, 100 000 random 32-bit ids share a
value with probability about ``1 - exp(-n^2 / 2^33)``, roughly 69 percent, and
the engine refuses a duplicate id into a stash.  So the id is not drawn, it is
*computed*: the top nibble is the tier and the low 28 bits are a keyed
bijection of the counter (:func:`_permute28`).  Two mints that differ in
counter or in tier therefore differ in id by construction, not by luck.  The
Stream's job for the id is to choose the per-profile key, so the same counter
reads differently in every profile.  The map seed, which need not be unique,
is a plain 64-bit draw from a per-mint stream.
"""

from __future__ import annotations

import functools
import sys
from typing import Dict, List, Sequence, Tuple

from lucifer_descent.contracts import MAX_TIER, NodeState, ProfileState, Sigil, WebNode
from lucifer_gen.seed import MASK64, SeedFields, Stream

__all__ = [
    "MIN_TIER",
    "COUNTER_BITS",
    "MAX_COUNTER",
    "ID_PREFIX",
    "DROP_COUNT_MIN",
    "DROP_COUNT_MAX",
    "DROP_TIER_OFFSETS",
    "DROP_TIER_WEIGHTS",
    "EXPECTED_DROPS_PER_CLEAR",
    "EXPECTED_TIER_DRIFT",
    "OPENABLE_STATES",
    "clamp_tier",
    "sigil_id",
    "tier_from_id",
    "unmint",
    "is_genuine",
    "mint_sigil",
    "roll_drops",
    "node_is_openable",
    "prewarm_candidates",
    "stash_summary",
]

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

#: Spec: "Sigils are consumable keys tiered 1 to 15."  ``MAX_TIER`` is the
#: contracts' 15; this is the other end.
MIN_TIER = 1

#: Bits of a Sigil id given to the counter.  The other four hold the tier
#: (1..15 fits a nibble with 0 left over, so a valid id never starts ``sg-0``).
COUNTER_BITS = 28
MAX_COUNTER = (1 << COUNTER_BITS) - 1
_MASK28 = MAX_COUNTER

ID_PREFIX = "sg-"
ID_HEX_DIGITS = 8

#: The sustain rule: clearing a node yields 1 to 3 Sigils ...
DROP_COUNT_MIN = 1
DROP_COUNT_MAX = 3
#: ... each of tier t-1, t or t+1 with weights 2, 5, 3.
DROP_TIER_OFFSETS: Tuple[int, ...] = (-1, 0, 1)
DROP_TIER_WEIGHTS: Tuple[int, ...] = (2, 5, 3)

#: E[count] for a uniform draw on 1..3.
EXPECTED_DROPS_PER_CLEAR = (DROP_COUNT_MIN + DROP_COUNT_MAX) / 2.0
#: E[tier - t] for one dropped Sigil, away from the clamps: (-2 + 0 + 3) / 10.
EXPECTED_TIER_DRIFT = sum(o * w for o, w in zip(DROP_TIER_OFFSETS, DROP_TIER_WEIGHTS)) / float(
    sum(DROP_TIER_WEIGHTS)
)

#: Node states a portal may be opened into.  Spec: a Sigil opens "a reachable
#: node", and after death "the node returns to reachable, not cleared.
#: Retrying costs another Sigil"; the transition table spells that as both
#: ``REACHABLE -> OPEN`` and ``FAILED -> OPEN``.
OPENABLE_STATES = frozenset({NodeState.REACHABLE, NodeState.FAILED})

_ID_KEY_LABEL = "sigil.id-key"
_MINT_LABEL = "sigil.mint:{counter}:{tier}"


# --------------------------------------------------------------------------
# Argument checks
# --------------------------------------------------------------------------


def _check_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{what} must be an int, got {type(value).__name__}")
    return value


def _check_tier(tier: object, what: str = "tier") -> int:
    tier = _check_int(tier, what)
    if not MIN_TIER <= tier <= MAX_TIER:
        raise ValueError(f"{what} must be between {MIN_TIER} and {MAX_TIER}, got {tier}")
    return tier


def _check_counter(counter: object) -> int:
    counter = _check_int(counter, "counter")
    if not 0 <= counter <= MAX_COUNTER:
        raise ValueError(f"counter must be between 0 and {MAX_COUNTER}, got {counter}")
    return counter


def _profile_stream(profile_seed: object, label: str) -> Stream:
    """A ``sigil.*`` stream fed the whole profile seed, masked to 64 bits.

    Goes through :meth:`SeedFields.stream` so the masking and the routing rule
    are the seed module's, the same way :func:`lucifer_descent.web.generate_web`
    opens its ``web.*`` streams.
    """
    _check_int(profile_seed, "profile_seed")
    return SeedFields.parse(profile_seed).stream(label)


def clamp_tier(tier: int) -> int:
    """Clamp a tier into 1..15.  Spec: drop tiers are "clamped to 1..15"."""
    return max(MIN_TIER, min(MAX_TIER, tier))


# --------------------------------------------------------------------------
# Ids
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=256)
def _profile_key(profile_seed: int) -> Tuple[int, int]:
    """The two 28-bit keys that make this profile's id permutation its own.

    A pure function of the profile seed, so caching it changes nothing but
    the cost: without the cache every mint would seed a fresh stream just to
    recompute the same pair.
    """
    stream = _profile_stream(profile_seed, _ID_KEY_LABEL)
    return stream.randint(0, _MASK28), stream.randint(0, _MASK28)


def _permute28(value: int, key_in: int, key_out: int) -> int:
    """A keyed bijection on 28-bit integers.

    Every step is invertible on its own, so the composition is too:

    * ``x ^ key`` for a constant key;
    * ``x ^= x >> s`` -- the top ``s`` bits are unchanged and each lower bit
      is recovered from one already known;
    * ``x * m mod 2**28`` for odd ``m`` -- odd numbers are units modulo a
      power of two.

    The constants and shifts are MurmurHash3's 32-bit finaliser, kept only
    for their mixing; any odd multipliers would do for the bijection.
    """
    h = (value ^ key_in) & _MASK28
    h ^= h >> 16
    h = (h * 0x85EBCA6B) & _MASK28
    h ^= h >> 13
    h = (h * 0xC2B2AE35) & _MASK28
    h ^= h >> 16
    return (h ^ key_out) & _MASK28


#: Multiplicative inverses of the two odd constants modulo 2**28, so that
#: :func:`_unpermute28` can undo :func:`_permute28` step by step.
_INV_MUL_A = pow(0x85EBCA6B, -1, 1 << COUNTER_BITS)
_INV_MUL_B = pow(0xC2B2AE35, -1, 1 << COUNTER_BITS)


def _unpermute28(value: int, key_in: int, key_out: int) -> int:
    """The inverse of :func:`_permute28`: the same steps, undone in reverse.

    ``x ^= x >> 16`` is its own inverse on 28 bits (a second shift by 16
    clears everything).  ``x ^= x >> 13`` is undone by applying it and then
    ``x ^= x >> 26``, since ``x ^ (x >> 13) ^ (x >> 13) ^ (x >> 26)`` leaves
    only ``x ^ (x >> 26)`` and one more fold clears that too.  Each multiply
    is undone by its inverse modulo ``2**28``.
    """
    h = (value ^ key_out) & _MASK28
    h ^= h >> 16
    h = (h * _INV_MUL_B) & _MASK28
    h ^= h >> 13
    h ^= h >> 26
    h = (h * _INV_MUL_A) & _MASK28
    h ^= h >> 16
    return (h ^ key_in) & _MASK28


def sigil_id(profile_seed: int, counter: int, tier: int) -> str:
    """The id a Sigil minted from ``(profile_seed, counter, tier)`` carries.

    ``sg-`` then eight lowercase hex digits: the first is the tier, the other
    seven are :func:`_permute28` of the counter under this profile's key.
    Injective in ``(counter, tier)`` for one profile, so two different
    counters never collide whatever their tiers (see the module docstring).
    """
    tier = _check_tier(tier)
    counter = _check_counter(counter)
    key_in, key_out = _profile_key(profile_seed)
    value = (tier << COUNTER_BITS) | _permute28(counter, key_in, key_out)
    return f"{ID_PREFIX}{value:0{ID_HEX_DIGITS}x}"


def tier_from_id(sigil_id_text: str) -> int:
    """Read the tier nibble back out of an id, refusing anything malformed.

    A storage layer can use this to check a stashed Sigil's ``tier`` matches
    its ``id`` before trusting it; "identity is checked on every portal open".
    """
    if (
        not isinstance(sigil_id_text, str)
        or not sigil_id_text.startswith(ID_PREFIX)
        or len(sigil_id_text) != len(ID_PREFIX) + ID_HEX_DIGITS
    ):
        raise ValueError(f"malformed sigil id: {sigil_id_text!r}")
    digits = sigil_id_text[len(ID_PREFIX):]
    try:
        value = int(digits, 16)
    except ValueError:
        raise ValueError(f"malformed sigil id: {sigil_id_text!r}") from None
    if digits != digits.lower():
        raise ValueError(f"malformed sigil id: {sigil_id_text!r}")
    tier = value >> COUNTER_BITS
    if not MIN_TIER <= tier <= MAX_TIER:
        raise ValueError(f"sigil id carries tier {tier}: {sigil_id_text!r}")
    return tier


def unmint(profile_seed: int, sigil_id_text: str) -> Tuple[int, int]:
    """Recover the ``(counter, tier)`` that :func:`sigil_id` built an id from.

    The id's low 28 bits are a keyed bijection of the counter, so every
    well-formed id inverts to exactly one counter for a given profile seed;
    the tier is the top nibble.  Raises :class:`ValueError` for a malformed
    id, exactly as :func:`tier_from_id` does.

    An id alone therefore proves nothing about provenance -- every 28-bit
    value is *some* counter's image.  What can be checked is the whole
    Sigil: see :func:`is_genuine`.
    """
    tier = tier_from_id(sigil_id_text)
    value = int(sigil_id_text[len(ID_PREFIX):], 16) & _MASK28
    key_in, key_out = _profile_key(profile_seed)
    return _unpermute28(value, key_in, key_out), tier


def is_genuine(profile_seed: int, sigil: Sigil) -> bool:
    """True when ``sigil`` is exactly what this profile's mint produces.

    Recovers the counter and tier from the id, mints that triple again and
    compares the whole Sigil -- id, tier *and* seed.  The seed is what makes
    this meaningful: the id can be inverted to a counter for any well-formed
    text, but the seed is a draw from the stream labelled with that counter
    and tier, so a Sigil whose seed was not drawn there is not a mint of this
    profile.  A malformed id is simply not genuine.
    """
    try:
        counter, tier = unmint(profile_seed, sigil.id)
    except ValueError:
        return False
    if tier != sigil.tier:
        return False
    return mint_sigil(profile_seed, counter, tier) == sigil


# --------------------------------------------------------------------------
# Minting
# --------------------------------------------------------------------------


def mint_sigil(profile_seed: int, counter: int, tier: int) -> Sigil:
    """Mint the one Sigil that ``(profile_seed, counter, tier)`` names.

    Spec: "Sigils are consumable keys tiered 1 to 15 ... the Sigil's own item
    seed becomes the map seed."  A Sigil is a value: minting the same triple
    twice returns equal Sigils, and there is no other way to make one.

    * ``id`` is :func:`sigil_id` -- unique per counter within a profile, by
      construction.
    * ``seed`` is a full 64-bit draw from the stream labelled
      ``sigil.mint:<counter>:<tier>``, so it is fixed the moment the Sigil is
      minted, long before any node is chosen for it.  That is what lets the
      170HX pre-roll a layout for a Sigil still sitting in the stash.

    ``counter`` is the profile's mint counter, 0..``MAX_COUNTER`` (2**28 - 1,
    about 268 million mints per profile).  ``tier`` must already be 1..15;
    this function does not clamp, because a caller who means "clamp" should
    say so (:func:`clamp_tier`) and a caller who does not has a bug.
    """
    tier = _check_tier(tier)
    counter = _check_counter(counter)
    ident = sigil_id(profile_seed, counter, tier)
    stream = _profile_stream(profile_seed, _MINT_LABEL.format(counter=counter, tier=tier))
    seed = stream.randint(0, MASK64)
    return Sigil(id=ident, tier=tier, seed=seed)


# --------------------------------------------------------------------------
# Drops: the sustain rule
# --------------------------------------------------------------------------


def roll_drops(
    profile_seed: int, counter: int, cleared_tier: int, stream_label: str
) -> List[Sigil]:
    """What clearing a node of tier ``cleared_tier`` drops, freshly minted.

    Spec (the sustain rule): "Clearing a node of tier t yields 1 to 3 Sigils,
    each of tier t-1, t or t+1 (clamped to 1..15) with weights 2, 5, 3, so on
    average a player sustains their tier and slowly climbs."

    The count is uniform on 1..3 and the tier offsets are drawn from the
    stream labelled ``stream_label``, which the caller must make unique per
    clear (the ledger sequence number of the clearing event is the natural
    choice); the same label rolls the same count and offsets again.  Each
    dropped Sigil is then :func:`mint_sigil`-ed with counters ``counter``,
    ``counter + 1``, ... in order, so **the caller advances its mint counter
    by ``len(result)``**.

    Expected values, away from the clamps (``2 <= t <= 14``):

    * ``E[count] = 2`` Sigils per clear (``EXPECTED_DROPS_PER_CLEAR``);
    * per Sigil, ``P(t-1) = 0.2``, ``P(t) = 0.5``, ``P(t+1) = 0.3``, so
      ``E[tier] = t + 0.1`` (``EXPECTED_TIER_DRIFT``);
    * ``P(tier >= t) = 0.8``, so a clear returns on average ``1.6`` Sigils
      able to reopen a node of the tier just cleared, against the one it
      cost: that is the "sustains" half.  The ``+0.1`` drift and the 0.3
      chance of a tier-up per Sigil are the "slowly climbs" half.

    At the clamps the ``-1`` (at ``t = 1``) or ``+1`` (at ``t = 15``) offset
    folds onto ``t``: ``E[tier] = 1.3`` at tier 1 and ``14.8`` at tier 15.

    ``cleared_tier`` must be 1..15.  Tier 0 is the origin, which the player
    never clears (it is cleared by genesis), so a tier-0 roll is a bug and is
    refused rather than clamped.
    """
    cleared_tier = _check_tier(cleared_tier, "cleared_tier")
    counter = _check_counter(counter)
    if not isinstance(stream_label, str) or not stream_label:
        raise ValueError("stream_label must be a non-empty string")

    stream = _profile_stream(profile_seed, stream_label)
    count = stream.randint(DROP_COUNT_MIN, DROP_COUNT_MAX)
    if counter + count - 1 > MAX_COUNTER:
        raise ValueError(
            f"counter {counter} leaves no room for {count} drops (max {MAX_COUNTER})"
        )

    drops: List[Sigil] = []
    for i in range(count):
        offset = stream.weighted_choice(DROP_TIER_OFFSETS, DROP_TIER_WEIGHTS)
        drops.append(mint_sigil(profile_seed, counter + i, clamp_tier(cleared_tier + offset)))
    return drops


# --------------------------------------------------------------------------
# Views over a profile: what the 170HX pre-rolls, what the table shows
# --------------------------------------------------------------------------


def node_is_openable(state: ProfileState, node: WebNode) -> bool:
    """Could *some* Sigil open ``node`` right now, ignoring the stash?

    Two rules, both node-side:

    * its state is ``REACHABLE`` or ``FAILED`` (the only states with an
      ``OPEN`` row in the transition table);
    * if it is a Pinnacle arena, that Pinnacle is unlocked.  Spec: "Each
      arena is unlocked by collecting fragments from three cleared tier-15
      nodes bearing that Pinnacle's glyph."  A glyph node is not an arena and
      is not gated.

    The one-instance rule is deliberately not here: pre-rolling while a run
    is in progress is the whole point of pre-rolling.
    """
    if state.states[node.id] not in OPENABLE_STATES:
        return False
    if node.pinnacle is not None and node.pinnacle not in state.unlocked_pinnacles:
        return False
    return True


def prewarm_candidates(state: ProfileState) -> List[Tuple[int, str, int]]:
    """Every ``(node_id, sigil_id, map_seed)`` the 170HX could pre-roll.

    Spec: "The 170HX pre-rolls layouts for reachable nodes so portals open
    instantly."  One triple per openable node (:func:`node_is_openable`) and
    stash Sigil that may open it (the tier rule, :meth:`Sigil.can_open`:
    ``sigil.tier >= node.tier``).  ``map_seed`` is the Sigil's own seed,
    because "the Sigil's own item seed becomes the map seed".

    Sorted by node id then Sigil id, so the list is a pure function of the
    state and two pre-rollers given the same profile agree on the order.
    ``LOCKED``, ``ACTIVE`` and ``CLEARED`` nodes contribute nothing, nor does
    an arena whose Pinnacle is still locked.

    A state naming a node the web does not have is malformed and raises
    ``KeyError`` rather than being quietly skipped.
    """
    nodes_by_id: Dict[int, WebNode] = {node.id: node for node in state.web.nodes}
    sigils: List[Tuple[str, Sigil]] = [(key, state.stash[key]) for key in sorted(state.stash)]

    out: List[Tuple[int, str, int]] = []
    for node_id in sorted(state.states):
        try:
            node = nodes_by_id[node_id]
        except KeyError:
            raise KeyError(f"profile state names node {node_id}, which is not in its web") from None
        if not node_is_openable(state, node):
            continue
        for key, sigil in sigils:
            if sigil.can_open(node):
                out.append((node.id, key, sigil.seed))
    return out


def stash_summary(state: ProfileState) -> Dict[int, int]:
    """How many Sigils of each tier the stash holds, keyed by tier, ascending.

    Only tiers with at least one Sigil appear, so an empty stash gives ``{}``
    and ``sum(summary.values()) == len(state.stash)`` always holds.  Read a
    missing tier with ``summary.get(tier, 0)``.
    """
    counts: Dict[int, int] = {}
    for key in sorted(state.stash):
        tier = state.stash[key].tier
        counts[tier] = counts.get(tier, 0) + 1
    return {tier: counts[tier] for tier in sorted(counts)}


# --------------------------------------------------------------------------
# A small demonstration, so ``python3 -m lucifer_descent.sigils`` shows a mint
# --------------------------------------------------------------------------


def _main(argv: Sequence[str]) -> int:
    profile_seed = int(argv[0], 0) if argv else 0x1234
    counter = 0
    print(f"profile seed 0x{profile_seed & MASK64:016X}")
    for tier in (1, 8, 15):
        sigil = mint_sigil(profile_seed, counter, tier)
        counter += 1
        print(f"  mint tier {tier:2d}: {sigil.id}  seed 0x{sigil.seed:016X}")
    for cleared_tier in (1, 8, 15):
        drops = roll_drops(profile_seed, counter, cleared_tier, f"demo.clear:{cleared_tier}")
        counter += len(drops)
        tiers = ", ".join(str(d.tier) for d in drops)
        print(f"  clear tier {cleared_tier:2d} drops {len(drops)} sigil(s) of tier {tiers}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main(sys.argv[1:]))
