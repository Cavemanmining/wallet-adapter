"""Stage 6, first half: monster spawning.

Spec: docs/WORLD_BIBLE.md stage 6, "Spawn and sync".

    density = base_density * tier_multiplier * sigil_modifiers; packs of 3 to
    8 placed at least 12 m apart and never within 20 m of the entrance; elite
    packs flagged for the HUD.

The density formula
-------------------
``density`` is a *packs per walkable cell* figure::

    density        = base_density * tier_multiplier(tier) * sigil_product
    expected_packs = density * walkable_cell_count

``base_density`` and ``elite_rate`` come off the template's
:class:`~lucifer_gen.contracts.SpawnRules` (0.018 and 0.12 by default, so a
48x48 dungeon with ~500 walkable cells expects ~9 packs).

``tier_multiplier`` is 1.0 at tier 1 and rises linearly to 2.0 at tier 15::

    tier_multiplier(t) = 1.0 + (clamp(t, 1, 15) - 1) / 14

The spec only draws that line between tiers 1 and 15, so the tier is clamped
to that range rather than extrapolated; see "Judgement calls" below.

``sigil_modifiers`` is the *product* of every active sigil's multiplier.  It
is accepted as a bare float, as a sequence of floats, or as a mapping of sigil
name to float; a mapping is multiplied out in sorted-key order so that
floating point rounding cannot depend on dictionary insertion order.

Turning an expected count into a whole number
---------------------------------------------
``expected_packs`` is rarely an integer.  Truncating it would bias every map
downward, so the fractional part is spent as one coin flip off the spawn
stream: 9.4 expected packs means 9 packs plus a 40% chance of a tenth.  The
expectation is therefore exactly ``density * walkable``, which is what makes
the density testable over many seeds.  The same trick sets the elite count.

Placement, and why it terminates
--------------------------------
Candidates are every cell that is walkable in the tile grid, is not an
``APPROACH`` or ``SET_PIECE`` cell in the terrain plan, and is at least 20 m
from the entrance.  They are collected in row-major order and then shuffled
with the spawn stream, so the scan order is a pure function of the seed.

The scan is greedy: walk the shuffled candidates, keep one if it is at least
12 m from every pack already placed, stop when the target is reached.  Each
candidate is examined at most once, so the loop is bounded by the candidate
count even with no other cap.  On top of that the scan gives up after

    ATTEMPT_FLOOR + ATTEMPTS_PER_PACK * target   ( = 64 + 12 * target )

examined candidates, which bounds the work on a large map whose geometry
cannot hold the target -- a corridor-only map, say, where nearly every
candidate is rejected for spacing.  A small map therefore returns fewer packs
than the density asks for rather than looping; that is deliberate, spacing is
a hard constraint and density is a target.

Distances
---------
Distances are Euclidean between cell centres in metres, using the template's
``cell_m`` (4 m), so "12 m apart" is 3 cells and "20 m from the entrance" is
5 cells exactly as the spec says.  Comparisons are made on squared distances
in cell units with a small epsilon, so a pack sitting at exactly 3 cells is
accepted rather than lost to floating point.

Randomness
----------
Every draw -- shuffle, pack size, pack family, elite choice -- comes off a
single stream labelled ``"spawn-place"``.  ``seed.SeedFields.stream`` routes
any label starting with ``spawn`` to the tile field (bits 32-63), which is
the field the spec assigns to stage 6, so re-rolling routing bits cannot move
a monster.

Judgement calls the spec did not settle
---------------------------------------
* **Tier outside 1..15** is clamped, not extrapolated.
* **Pack names**: nothing in the template or tile database names monster
  packs, so this module carries a small family table chosen by tile class
  (:data:`DUNGEON_PACKS` / :data:`OUTDOOR_PACKS`).  Swap it for real content
  data when there is any; the placement logic does not care.
* **Elite flagging** sets ``elite=True`` and leaves the pack name alone, so
  the HUD and the content table stay decoupled.
* **Walkability** is read from the tile grid (stage 4's answer, and what the
  navmesh will agree with) rather than re-derived from the terrain plan; the
  plan is consulted only for the APPROACH / SET_PIECE exclusions.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from .contracts import (
    Cell,
    CellKind,
    GraphTemplate,
    Role,
    RoutedLayout,
    SpawnPack,
    SpawnRules,
    TerrainPlan,
    TileClass,
    TileGrid,
)
from .seed import SeedFields, Stream

#: Spawn draws come off this label, which seed.py routes to the tile field.
SPAWN_STREAM_LABEL = "spawn-place"

#: The tier band the spec draws its line across.
MIN_TIER = 1
MAX_TIER = 15
TIER_MIN_MULTIPLIER = 1.0
TIER_MAX_MULTIPLIER = 2.0

#: Spec stage 6: "packs of 3 to 8".
PACK_MIN = 3
PACK_MAX = 8

#: Spec stage 6: "at least 12 m apart", "never within 20 m of the entrance".
PACK_SPACING_M = 12.0
ENTRANCE_EXCLUSION_M = 20.0

#: Attempt cap, in candidates examined: floor + per-pack allowance.
ATTEMPT_FLOOR = 64
ATTEMPTS_PER_PACK = 12

#: Cell kinds a pack may never stand on (spec stage 5: the boss approach runs
#: at zero ambient density, and set piece interiors are hand-authored).
BLOCKED_KINDS: Tuple[CellKind, ...] = (CellKind.APPROACH, CellKind.SET_PIECE)

#: Slack for squared-distance comparisons, so an exactly-3-cell gap passes.
_EPS = 1e-9

#: Placeholder content tables; see "Judgement calls" in the module docstring.
DUNGEON_PACKS: Tuple[str, ...] = (
    "bonepicker_swarm",
    "chain_thrall_gang",
    "crypt_ghoul_band",
    "ossuary_stalkers",
    "rot_choir",
)
OUTDOOR_PACKS: Tuple[str, ...] = (
    "ash_wolf_pack",
    "bone_kite_flight",
    "cinder_wraith_flock",
    "rampart_skirmishers",
    "scorched_marauders",
)


class SpawnError(ValueError):
    """Raised when the inputs to stage 6 cannot describe a spawnable map."""


# --------------------------------------------------------------------------
# Density
# --------------------------------------------------------------------------


def tier_multiplier(tier: Union[int, float]) -> float:
    """1.0 at tier 1, rising linearly to 2.0 at tier 15.

    Spec stage 6.  The tier is clamped to 1..15: the spec defines the line
    only across that band, and silently doubling again at tier 29 would be an
    invention rather than an implementation.
    """
    t = float(tier)
    if t < MIN_TIER:
        t = float(MIN_TIER)
    elif t > MAX_TIER:
        t = float(MAX_TIER)
    span = float(MAX_TIER - MIN_TIER)
    return TIER_MIN_MULTIPLIER + (t - MIN_TIER) / span * (
        TIER_MAX_MULTIPLIER - TIER_MIN_MULTIPLIER
    )


def sigil_product(
    sigil_modifiers: Union[None, float, int, Sequence[float], Mapping[str, float]]
) -> float:
    """Fold the active sigils into one multiplier.

    Accepts ``None`` (no sigils, 1.0), a bare number, a sequence of numbers,
    or a mapping of sigil name to number.  A mapping is multiplied in sorted
    key order so the result does not depend on insertion order.
    """
    if sigil_modifiers is None:
        return 1.0
    if isinstance(sigil_modifiers, bool):  # bool is an int; refuse the trap
        raise SpawnError("sigil_modifiers must be a number, not a bool")
    if isinstance(sigil_modifiers, (int, float)):
        values: List[float] = [float(sigil_modifiers)]
    elif isinstance(sigil_modifiers, Mapping):
        values = [float(sigil_modifiers[k]) for k in sorted(sigil_modifiers)]
    elif isinstance(sigil_modifiers, Iterable):
        values = [float(v) for v in sigil_modifiers]
    else:
        raise SpawnError(f"cannot read sigil modifiers from {sigil_modifiers!r}")

    product = 1.0
    for v in values:
        if v < 0.0:
            raise SpawnError(f"negative sigil modifier: {v}")
        product *= v
    return product


def spawn_density(
    rules: SpawnRules,
    tier: Union[int, float],
    sigil_modifiers: Union[None, float, int, Sequence[float], Mapping[str, float]],
) -> float:
    """``base_density * tier_multiplier * sigil_modifiers`` -- packs per cell."""
    if rules.base_density < 0.0:
        raise SpawnError(f"negative base density: {rules.base_density}")
    return rules.base_density * tier_multiplier(tier) * sigil_product(sigil_modifiers)


def _draw_whole(stream: Stream, expected: float) -> int:
    """Round ``expected`` to a whole count, spending the fraction as a flip.

    Keeps the expectation exact, which is what lets a density or an elite rate
    be asserted over many seeds instead of merely eyeballed.
    """
    if expected <= 0.0:
        return 0
    whole = int(math.floor(expected))
    frac = expected - whole
    if frac > 0.0 and stream.chance(frac):
        whole += 1
    return whole


# --------------------------------------------------------------------------
# Candidates
# --------------------------------------------------------------------------


def _entrance_cell(routed: Optional[RoutedLayout]) -> Optional[Cell]:
    if routed is None:
        return None
    node = routed.node_of_role(Role.ENTRANCE)
    return None if node is None else node.cell


def walkable_cells(tile_grid: TileGrid) -> List[Cell]:
    """Every cell stage 4 marked walkable, row-major.

    This is the denominator of the density: the whole walkable floor, before
    any exclusion, because the spec's "walkable cell count" is a property of
    the map rather than of where a pack happens to be allowed.  Walkability is
    stage 4's answer, which is also what the navmesh will agree with; the
    terrain plan is consulted only for the kind exclusions in
    :func:`candidate_cells`.
    """
    out: List[Cell] = []
    for y in range(tile_grid.grid):
        for x in range(tile_grid.grid):
            if tile_grid.walkable[y][x]:
                out.append((x, y))
    return out


def candidate_cells(
    plan: Optional[TerrainPlan],
    tile_grid: TileGrid,
    entrance: Optional[Cell],
    cell_m: float,
) -> List[Cell]:
    """Walkable cells a pack is allowed to stand on, row-major.

    Excludes APPROACH and SET_PIECE cells (spec stage 5) and anything within
    :data:`ENTRANCE_EXCLUSION_M` of the entrance (spec stage 6).
    """
    exclusion_sq = (ENTRANCE_EXCLUSION_M / cell_m) ** 2
    out: List[Cell] = []
    for cell in walkable_cells(tile_grid):
        if plan is not None and plan.kind(cell) in BLOCKED_KINDS:
            continue
        if entrance is not None:
            dx = cell[0] - entrance[0]
            dy = cell[1] - entrance[1]
            if dx * dx + dy * dy + _EPS < exclusion_sq:
                continue
        out.append(cell)
    return out


def _pack_names(tile_class: TileClass) -> Tuple[str, ...]:
    """The content table for a tile class; sorted, so choice is reproducible."""
    if tile_class is TileClass.DUNGEON:
        return DUNGEON_PACKS
    if tile_class is TileClass.OUTDOOR:
        return OUTDOOR_PACKS
    return tuple(sorted(set(DUNGEON_PACKS) | set(OUTDOOR_PACKS)))


def _as_fields(seed: Union[int, SeedFields]) -> SeedFields:
    return seed if isinstance(seed, SeedFields) else SeedFields.parse(int(seed))


# --------------------------------------------------------------------------
# Stage 6 entry point
# --------------------------------------------------------------------------


def place_spawns(
    plan: Optional[TerrainPlan],
    tile_grid: TileGrid,
    routed: Optional[RoutedLayout],
    template: GraphTemplate,
    tier: Union[int, float],
    sigil_modifiers: Union[None, float, int, Sequence[float], Mapping[str, float]],
    seed: Union[int, SeedFields],
) -> List[SpawnPack]:
    """Stage 6: place and flag the monster packs.

    Spec: docs/WORLD_BIBLE.md stage 6, "Spawn and sync".

    Returns the packs in placement order, which is deterministic for a given
    seed.  Every returned pack is on a walkable, non-APPROACH, non-SET_PIECE
    cell, at least 12 m from every other pack and at least 20 m from the
    entrance, and holds 3 to 8 monsters.  The count is the density target
    rounded as described in the module docstring, or fewer if the geometry
    cannot hold that many -- spacing wins over density.

    ``routed`` may be ``None``, or carry no entrance node, in which case the
    entrance exclusion simply does not apply; likewise a ``None`` ``plan``
    drops the APPROACH / SET_PIECE exclusions.  Both are conveniences for
    driving stage 6 from a partial pipeline, not licences for a real map to
    skip them.
    """
    if tile_grid is None or tile_grid.grid <= 0:
        raise SpawnError("stage 6 needs a tile grid with at least one cell")
    cell_m = float(getattr(template, "cell_m", 4.0))
    if cell_m <= 0.0:
        raise SpawnError(f"cell size must be positive, got {cell_m}")

    fields = _as_fields(seed)
    stream = fields.stream(SPAWN_STREAM_LABEL)

    rules: SpawnRules = template.spawn
    density = spawn_density(rules, tier, sigil_modifiers)

    entrance = _entrance_cell(routed)
    walkable = walkable_cells(tile_grid)
    target = _draw_whole(stream, density * len(walkable))
    if target <= 0:
        return []

    candidates = candidate_cells(plan, tile_grid, entrance, cell_m)
    if not candidates:
        return []

    names = _pack_names(template.tile_class)
    spacing_sq = (PACK_SPACING_M / cell_m) ** 2
    budget = ATTEMPT_FLOOR + ATTEMPTS_PER_PACK * target

    placed: List[SpawnPack] = []
    for attempts, cell in enumerate(stream.shuffled(candidates), start=1):
        if len(placed) >= target or attempts > budget:
            break
        if not _far_enough(cell, placed, spacing_sq):
            continue
        placed.append(
            SpawnPack(
                pack=stream.choice(names),
                cell=cell,
                count=stream.randint(PACK_MIN, PACK_MAX),
                elite=False,
            )
        )

    _flag_elites(placed, rules.elite_rate, stream)
    return placed


def _far_enough(cell: Cell, placed: Sequence[SpawnPack], spacing_sq: float) -> bool:
    """True when ``cell`` clears every placed pack by the spacing rule."""
    for pack in placed:
        dx = cell[0] - pack.cell[0]
        dy = cell[1] - pack.cell[1]
        if dx * dx + dy * dy + _EPS < spacing_sq:
            return False
    return True


def _flag_elites(placed: List[SpawnPack], elite_rate: float, stream: Stream) -> None:
    """Flag ``elite_rate`` of the placed packs for the HUD, in place.

    The count is drawn the same way as the pack count, so over many maps the
    share of elites converges on ``elite_rate`` exactly.  Which packs are
    promoted is a shuffle of the indices, not a per-pack coin flip, so a map
    can never come out all-elite by luck.
    """
    if not placed or elite_rate <= 0.0:
        return
    if elite_rate < 0.0:
        raise SpawnError(f"negative elite rate: {elite_rate}")
    wanted = min(len(placed), _draw_whole(stream, elite_rate * len(placed)))
    if wanted <= 0:
        return
    for index in sorted(stream.shuffled(range(len(placed)))[:wanted]):
        placed[index].elite = True


__all__ = [
    "ATTEMPTS_PER_PACK",
    "ATTEMPT_FLOOR",
    "BLOCKED_KINDS",
    "DUNGEON_PACKS",
    "ENTRANCE_EXCLUSION_M",
    "MAX_TIER",
    "MIN_TIER",
    "OUTDOOR_PACKS",
    "PACK_MAX",
    "PACK_MIN",
    "PACK_SPACING_M",
    "SPAWN_STREAM_LABEL",
    "SpawnError",
    "candidate_cells",
    "place_spawns",
    "sigil_product",
    "spawn_density",
    "tier_multiplier",
    "walkable_cells",
]
