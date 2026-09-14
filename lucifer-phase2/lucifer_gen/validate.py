"""The Phase 2 gate checks: navmesh islands, seam mismatches, map report.

Spec: docs/WORLD_BIBLE.md -- the Phase 2 gate that every generated map must
clear before it may ship, run over 1000 seeds by the CLI.

Four properties carry the gate:

**No navmesh island.**  Every walkable cell must be reachable on foot from
the entrance.  A pocket of floor that nothing connects to is a map the player
can see but never stand on, and it is the failure mode stage 3's connectivity
repair and stage 5's overwrite-then-refill are most likely to introduce.

Reachability is judged *through the tiles*, not through
``TileGrid.walkable``.  That flag is written by stage 4 and stage 5 as a
straight copy of the terrain plan's floor mask, so flooding it alone would
only ever re-ask whether stage 3's plan is connected -- it would call two
adjacent floor cells connected even when the tiles placed on them present a
solid wall to one another.  A step is therefore allowed only across a seam
both tiles leave open and whose connection slots line up
(:func:`seam_is_open`), which is the same question the client's navmesh baker
asks.

**No seam mismatch.**  Every orthogonally adjacent pair of placed tiles must
fit together, judged from the sides the tiles *actually* present after
rotation and flip -- not from the requirement they were chosen against.

**No transform mismatch.**  Every placement's sides, as
``contracts.Tile.transformed`` computes them, must agree with
:func:`geometric_sides`, which derives the same four sides by carrying each
connection slot to its new position as a point on the cell boundary.

The last two are one argument.  This module used to say it re-implemented the
seam walk so that "two independent readings of the same rule that agree are
evidence"; only the ``for`` loop was ever independent.  Both walks asked
``Tile.transformed`` what a placement presents and ``sides_compatible``
whether two sides fit -- the same two functions the matcher chose the tiles
with -- so a sign error in the rotation, a mirror on the wrong axis, or a
reversed slot mask would have agreed with itself on every seam of every map.
The geometric model above is a genuinely separate derivation, and
:func:`find_seam_mismatches` now judges each seam by both rules and reports a
disagreement between them as loudly as a failure of either.

**No stage 4 invariant broken.**  :func:`find_stage4_breaks` runs
``tileize.debug_check`` over the finished map, which is where the border
sealing, the per-cell requirement satisfaction and the hero budget are
checked.  A cell showing OPEN off the edge of the grid has no neighbour, so
it produces neither a seam nor an island; nothing else in the gate can see it.

Around those live the cheap sanity checks (:func:`validate_map`) and the seed
sweep (:func:`run_suite`).

Judgement calls the spec did not settle
---------------------------------------
* An island is reported as a ``frozenset`` of cells, and the returned list is
  sorted by each island's smallest cell, so the answer never depends on the
  order the grid happened to be walked in.
* ``find_navmesh_islands`` with an unwalkable or out-of-bounds ``start_cell``
  reports *every* walkable component as an island, rather than raising.  That
  is the honest reading: if the entrance is not standable, nothing on the map
  is reachable from it.  ``validate_map`` additionally records the start cell
  itself as a problem, so the cause is never mistaken for the symptom.
* A placement naming a tile the database does not define is reported as a
  seam mismatch on each seam it touches, with a reason saying so, rather than
  raising.  The gate's job is to report, and a grid full of unknown ids is a
  louder signal than one traceback.
* Suite seeds are derived with ``hashlib.blake2b`` and never with blake3,
  even where the wheel is installed: the seeds a gate run covers must not
  depend on what is installed on the machine running it.  (Hashing the
  *layout* is a different question, and ``layout.py`` answers it there.)
* ``run_suite`` treats any exception raised while generating or validating a
  map as that seed's failure and keeps going, because "seed 0x... crashes
  stage 3" is exactly the kind of finding a 1000-seed gate exists to make.
"""

from __future__ import annotations

import collections.abc
import hashlib
import traceback
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
    Tuple,
)

from .contracts import (
    E,
    OPPOSITE,
    S,
    SIDE_NAMES,
    SIDES,
    Cell,
    CellKind,
    EdgeSig,
    GeneratedMap,
    GraphTemplate,
    Placement,
    Role,
    SideSpec,
    Tile,
    TileGrid,
    neighbour,
    sides_compatible,
)
from .seed import MASK64, format_seed

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps imports light
    from .rooms import RoomLibrary
    from .tiles import TileDatabase

#: The two sides a cell is responsible for, so each seam is tested once: its
#: east and south neighbours. The west and north seams belong to those cells.
OWNED_SIDES: Tuple[int, int] = (E, S)

#: Domain tag for :func:`suite_seed`; bump it and a gate run covers new seeds.
SUITE_DOMAIN = "lucifer.validate.suite/1"

Island = FrozenSet[Cell]


class Seam(NamedTuple):
    """One failing seam: ``(cell_a, cell_b, side, reason)``.

    ``side`` is the side *of ``cell_a``* that faces ``cell_b``, so it is
    always east or south -- see :data:`OWNED_SIDES`.  This is a tuple, and
    compares equal to the plain 4-tuple the gate spec asks for; the names are
    for the reader.
    """

    cell_a: Cell
    cell_b: Cell
    side: int
    reason: str

    def __str__(self) -> str:  # pragma: no cover - diagnostic text
        return f"{self.cell_a}->{self.cell_b} {SIDE_NAMES[self.side]}: {self.reason}"


@dataclass(frozen=True)
class Problem:
    """One sanity check that failed, named by a stable slug."""

    kind: str
    message: str
    cell: Optional[Cell] = None

    def __str__(self) -> str:  # pragma: no cover - diagnostic text
        where = f" at {self.cell}" if self.cell is not None else ""
        return f"{self.kind}{where}: {self.message}"


# --------------------------------------------------------------------------
# An independent reading of the transform algebra
# --------------------------------------------------------------------------
#
# Everything below re-derives, from geometry alone, what ``contracts`` derives
# from index arithmetic.  That is the whole point: the gate's seam walk used to
# ask ``Tile.transformed`` what a placement presents and ``sides_compatible``
# whether two sides fit, which is the same oracle stage 4 chose the tiles with.
# Two readings that share an oracle are one reading asserted twice, so a sign
# error in the rotation, a mirror on the wrong axis, or a slot mask reversed
# the wrong way would have agreed with itself and shipped green.
#
# The model: a cell is the square [0, 6] x [0, 6] with x east and y south, so
# the three connection slots of a side sit at the odd coordinates 1, 3 and 5
# along it and every transform stays on integers.  Slot order runs clockwise
# (north west-to-east, east north-to-south, south east-to-west, west
# south-to-north), which is what fixes where slot k of a side physically is.

_CELL = 6

#: ``(side, slot) -> the lattice point where that slot sits``.
_SLOT_POINT: Dict[Tuple[int, int], Tuple[int, int]] = {}
for _k in range(3):
    _a = 1 + 2 * _k  # 1, 3, 5 along the side, in the side's clockwise order
    _SLOT_POINT[(0, _k)] = (_a, 0)                    # N runs west to east
    _SLOT_POINT[(1, _k)] = (_CELL, _a)                # E runs north to south
    _SLOT_POINT[(2, _k)] = (_CELL - _a, _CELL)        # S runs east to west
    _SLOT_POINT[(3, _k)] = (0, _CELL - _a)            # W runs south to north
del _k, _a

#: The inverse: which side and slot a lattice point is.
_POINT_SLOT: Dict[Tuple[int, int], Tuple[int, int]] = {
    point: key for key, point in _SLOT_POINT.items()
}


def _turn(point: Tuple[int, int], rot: int, flip: bool) -> Tuple[int, int]:
    """Move one point of the cell boundary under flip-then-rotate.

    The flip mirrors across the vertical axis, ``x -> 6 - x``.  A clockwise
    quarter turn in this coordinate system (y runs *south*) is
    ``(x, y) -> (6 - y, x)``: it carries the east edge onto the south edge,
    which is what "clockwise" means on screen.
    """
    x, y = point
    if flip:
        x = _CELL - x
    for _ in range(rot & 3):
        x, y = _CELL - y, x
    return (x, y)


def geometric_sides(tile: Tile, rot: int, flip: bool) -> Tuple[SideSpec, ...]:
    """What ``tile`` presents after ``flip`` then ``rot``, derived geometrically.

    Independent of :meth:`contracts.Tile.transformed` and of
    ``contracts._reverse_slots``: each of the tile's twelve connection slots is
    carried to its new position as a *point*, and the side it lands on is read
    off the lattice.  A tile side's signature travels with its slots.
    """
    sigs: List[Optional[EdgeSig]] = [None, None, None, None]
    slots = [0, 0, 0, 0]
    for side in SIDES:
        spec = tile.sides[side]
        for k in range(3):
            landed_side, landed_slot = _POINT_SLOT[
                _turn(_SLOT_POINT[(side, k)], rot, flip)
            ]
            sigs[landed_side] = spec.sig
            if (spec.slots >> k) & 1:
                slots[landed_side] |= 1 << landed_slot
    return tuple(
        SideSpec(sigs[side] or EdgeSig.OPEN, slots[side]) for side in SIDES
    )


#: What a side must meet across a seam, stated here rather than imported, so a
#: broken ``SIG_COMPLEMENT`` cannot make both readings wrong together.  A rise
#: meets a fall; everything else meets its own kind.
_MEETS: Dict[EdgeSig, EdgeSig] = {
    EdgeSig.OPEN: EdgeSig.OPEN,
    EdgeSig.WALL: EdgeSig.WALL,
    EdgeSig.WATER: EdgeSig.WATER,
    EdgeSig.CLIFF_UP: EdgeSig.CLIFF_DOWN,
    EdgeSig.CLIFF_DOWN: EdgeSig.CLIFF_UP,
}


def geometrically_fits(a_side: SideSpec, b_side: SideSpec) -> bool:
    """Whether two facing sides meet, judged from where their slots are.

    The two sides run along the shared edge in opposite directions, so slot
    ``k`` of one is physically the same third of the edge as slot ``2 - k`` of
    the other.  Deriving that from the geometry, rather than calling
    ``contracts._reverse_slots``, is what makes this an independent reading of
    :func:`contracts.sides_compatible`.
    """
    if _MEETS[a_side.sig] is not b_side.sig:
        return False
    if a_side.sig is not EdgeSig.OPEN:
        return True
    return any(
        ((a_side.slots >> k) & 1) and ((b_side.slots >> (2 - k)) & 1)
        for k in range(3)
    )


def seam_is_open(a_side: SideSpec, b_side: SideSpec) -> bool:
    """True when an actor may actually step across this seam.

    Both tiles must present an open edge *and* share a connection slot: an
    open side whose doorway does not line up with the neighbour's is a wall as
    far as the navmesh is concerned.
    """
    return a_side.sig is EdgeSig.OPEN and geometrically_fits(a_side, b_side)


def _tile_lookup(tiles: Any) -> Callable[[int], Optional[Tile]]:
    """``tile id -> Tile`` for a database, a mapping, or an iterable of tiles."""
    by_id = getattr(tiles, "by_id", None)
    if callable(by_id):

        def lookup(tile_id: int) -> Optional[Tile]:
            try:
                return by_id(tile_id)
            except LookupError:
                return None

        return lookup

    table = _tile_table(tiles)
    return lambda tile_id: table.get(tile_id)


def placed_placements(tile_grid: TileGrid) -> List[Placement]:
    """Every distinct placement in the grid, in first-seen row-major order."""
    seen: Set[Placement] = set()
    out: List[Placement] = []
    for y in range(tile_grid.grid):
        for x in range(tile_grid.grid):
            placement = tile_grid.at((x, y))
            if placement is not None and placement not in seen:
                seen.add(placement)
                out.append(placement)
    return out


def find_transform_breaks(tile_grid: TileGrid, tiles: Any) -> List[Problem]:
    """Placements whose shipped sides disagree with the geometric model.

    Gate check 3.  A 48x48 grid holds only a few hundred distinct placements,
    so this costs nothing, and it is the only thing in the package that
    corroborates what ``rot`` and ``flip`` *mean* -- the two numbers stage 6
    packs into the client blob.
    """
    lookup = _tile_lookup(tiles)
    sides_of = side_resolver(tiles)
    problems: List[Problem] = []
    for placement in placed_placements(tile_grid):
        tile = lookup(placement.tile_id)
        if tile is None:
            continue  # reported as a seam mismatch, with a clearer message
        try:
            shipped = tuple(sides_of(placement))
        except _UnknownTile:
            continue
        derived = geometric_sides(tile, placement.rot, placement.flip)
        if shipped != derived:
            shown = " ".join(
                f"{SIDE_NAMES[s]}={shipped[s].sig.name}:{shipped[s].slots:03b}"
                f"/{derived[s].sig.name}:{derived[s].slots:03b}"
                for s in SIDES
                if shipped[s] != derived[s]
            )
            problems.append(
                Problem(
                    "transform-mismatch",
                    f"tile {tile.name!r} at rot={placement.rot} "
                    f"flip={placement.flip}: the transform algebra and the "
                    f"geometric model disagree ({shown}, shipped/geometric)",
                )
            )
    return problems


# --------------------------------------------------------------------------
# Walkability and flood fill
# --------------------------------------------------------------------------


def walkable_cells(tile_grid: TileGrid) -> List[Cell]:
    """Every walkable cell, row-major, so callers get a stable order."""
    return [
        (x, y)
        for y in range(tile_grid.grid)
        for x in range(tile_grid.grid)
        if tile_grid.is_walkable((x, y))
    ]


#: A ``(cell, side) -> bool`` predicate: may an actor cross this seam?
StepRule = Callable[[Cell, int], bool]


def step_rule(tile_grid: TileGrid, tiles: Any) -> StepRule:
    """The honest "can an actor cross here" test for one grid.

    A step is possible only when both cells are walkable **and** the two tiles
    actually leave the shared edge open -- see :func:`seam_is_open`.  This is
    the correction the gate most needed: ``TileGrid.walkable`` is written by
    stage 4 and stage 5 as a straight copy of the terrain plan's floor mask
    (``walkable = here is Surface.FLOOR``), so flooding it alone asks whether
    stage 3's *plan* is connected and never once looks at the geometry the
    client bakes a navmesh from.  A wall standing between two floor cells is
    perfectly self-consistent -- ``WALL`` meets ``WALL`` -- so it also slips
    past the seam check; only reading both together catches it.
    """
    sides_of = side_resolver(tiles)

    def allowed(cell: Cell, side: int) -> bool:
        here = tile_grid.at(cell)
        there = tile_grid.at(neighbour(cell, side))
        if here is None or there is None:
            return False
        try:
            return seam_is_open(sides_of(here)[side], sides_of(there)[OPPOSITE[side]])
        except _UnknownTile:
            return False

    return allowed


def flood(
    tile_grid: TileGrid,
    start: Cell,
    stop: Optional[Cell] = None,
    *,
    tiles: Any = None,
    allowed: Optional[StepRule] = None,
) -> Set[Cell]:
    """The walkable cells an actor can reach from ``start``, found iteratively.

    Iterative on purpose: a 48x48 grid is 2304 cells, and a snaking corridor
    that visits most of them would push a recursive fill past CPython's
    default recursion limit.  A gate check must not be the thing that breaks
    on the pathological map it exists to catch.

    ``stop`` lets a caller that only wants a yes/no answer bail out as soon as
    the goal is seen; the returned set is then partial by design.

    ``tiles`` (a tile database, or anything :func:`side_resolver` accepts)
    makes the walk *edge-aware*: a step is taken only across a seam both tiles
    leave open.  ``allowed`` passes a prepared :data:`StepRule` instead, which
    is what :func:`find_navmesh_islands` does so the resolver cache is built
    once for the whole map.  With neither -- the reading a hand-built
    ``TileGrid`` with no database behind it can support -- adjacency alone
    decides, which is strictly weaker and cannot see a sealed seam.
    """
    if allowed is None and tiles is not None:
        allowed = step_rule(tile_grid, tiles)
    if not tile_grid.is_walkable(start):
        return set()
    seen: Set[Cell] = {start}
    stack: List[Cell] = [start]
    while stack:
        cell = stack.pop()
        if stop is not None and cell == stop:
            break
        for side in SIDES:
            nxt = neighbour(cell, side)
            # is_walkable is bounds-checked, so the edge of the grid needs no
            # special case here.
            if nxt in seen or not tile_grid.is_walkable(nxt):
                continue
            if allowed is not None and not allowed(cell, side):
                continue
            seen.add(nxt)
            stack.append(nxt)
    return seen


def reaches(
    tile_grid: TileGrid,
    start: Cell,
    goal: Cell,
    *,
    tiles: Any = None,
    allowed: Optional[StepRule] = None,
) -> bool:
    """True when an actor can walk from ``start`` to ``goal``."""
    if not tile_grid.is_walkable(start) or not tile_grid.is_walkable(goal):
        return False
    if start == goal:
        return True
    return goal in flood(tile_grid, start, stop=goal, tiles=tiles, allowed=allowed)


def find_navmesh_islands(
    tile_grid: TileGrid, start_cell: Cell, tiles: Any = None
) -> List[Island]:
    """Walkable regions that ``start_cell`` cannot reach.

    Gate check 1.  Returns one frozenset per disconnected region, sorted by
    the region's smallest cell; empty on a healthy map.  Connectivity is
    4-connected, matching the navmesh the client bakes from these tiles:
    actors do not step diagonally between two cells that share only a corner.

    Pass ``tiles`` -- :func:`validate_map` always does -- and a step is also
    required to cross an open seam, so a floor cell walled off by the tiles
    that were placed on it counts as an island.  Without it the walk falls
    back to bare adjacency, which is only what a caller holding a hand-built
    grid and no tile database can be asked for.
    """
    allowed = step_rule(tile_grid, tiles) if tiles is not None else None
    reached = flood(tile_grid, start_cell, allowed=allowed)
    seen: Set[Cell] = set(reached)
    islands: List[Island] = []
    for cell in walkable_cells(tile_grid):
        if cell in seen:
            continue
        region = flood(tile_grid, cell, allowed=allowed)
        seen |= region
        islands.append(frozenset(region))
    islands.sort(key=min)  # regions are disjoint, so the minimum is unique
    return islands


# --------------------------------------------------------------------------
# Seams
# --------------------------------------------------------------------------


class _UnknownTile(LookupError):
    """A placement naming a tile id the database does not define."""


def _tile_table(tiles: Any) -> Mapping[int, Tile]:
    """Coerce whatever the caller called a tile database into id -> Tile."""
    if isinstance(tiles, collections.abc.Mapping):
        return tiles
    if isinstance(tiles, collections.abc.Iterable):
        table: Dict[int, Tile] = {}
        for tile in tiles:
            table[tile.id] = tile
        return table
    raise TypeError(
        "tiles must be a TileDatabase, a mapping of id to Tile, or an "
        f"iterable of Tile, got {type(tiles).__name__}"
    )


def side_resolver(tiles: Any) -> Callable[[Placement], Tuple[SideSpec, ...]]:
    """A cached ``Placement -> four SideSpecs`` reader for ``tiles``.

    Accepts a :class:`~lucifer_gen.tiles.TileDatabase` (anything exposing
    ``sides_of``), a mapping of tile id to :class:`~contracts.Tile`, or an
    iterable of tiles -- the last two so a test can hand-build three tiles
    without standing up a database.  A 48x48 grid reuses a few hundred
    distinct placements thousands of times, hence the cache.

    Raises :class:`_UnknownTile` for an id the source does not define.
    """
    native = getattr(tiles, "sides_of", None)
    if callable(native):

        def base(placement: Placement) -> Tuple[SideSpec, ...]:
            try:
                return tuple(native(placement))
            except LookupError as exc:
                raise _UnknownTile(placement.tile_id) from exc

    else:
        table = _tile_table(tiles)

        def base(placement: Placement) -> Tuple[SideSpec, ...]:
            tile = table.get(placement.tile_id)
            if tile is None:
                raise _UnknownTile(placement.tile_id)
            return tuple(tile.transformed(placement.rot, placement.flip))

    cache: Dict[Placement, Tuple[SideSpec, ...]] = {}

    def resolve(placement: Placement) -> Tuple[SideSpec, ...]:
        hit = cache.get(placement)
        if hit is None:
            hit = base(placement)
            cache[placement] = hit
        return hit

    return resolve


def _seam_reason(a_side: SideSpec, b_side: SideSpec, side: int) -> str:
    """Say, in one line, why these two facing sides may not meet."""
    here, there = SIDE_NAMES[side], SIDE_NAMES[OPPOSITE[side]]
    if _MEETS[a_side.sig] is not b_side.sig:
        return (
            f"{here} shows {a_side.sig.name} which must meet "
            f"{_MEETS[a_side.sig].name}, but {there} shows "
            f"{b_side.sig.name}"
        )
    # Both sides are open and complementary, so the only rule left is slots.
    return (
        f"{here} and {there} are both OPEN but share no connection slot: "
        f"{a_side.slots:03b} against {b_side.slots:03b} read back along the "
        "shared edge"
    )


def find_seam_mismatches(tile_grid: TileGrid, tiles: Any) -> List[Seam]:
    """Every adjacent pair of placed tiles that does not fit together.

    Gate check 2.  Walks the grid row-major and tests only each cell's east
    and south seams, so a pair is judged once rather than twice from both
    ends.  Cells with no placement are skipped: an empty cell has no sides to
    disagree with, and a half-built grid is stage 4's problem, not a seam's.

    Every seam is judged twice: once by ``contracts.sides_compatible``, the
    rule stage 4 chose the tiles with, and once by :func:`geometrically_fits`,
    which re-derives the same rule from where the connection slots physically
    are.  A seam is reported when either reading rejects it, and a
    disagreement between them says so in the reason -- that is the only way a
    bug *inside* the compatibility rule can surface, since asking the same
    function twice can only ever agree with itself.

    Returns ``(cell_a, cell_b, side, reason)`` tuples in row-major order of
    ``cell_a``, east seam before south seam.
    """
    sides_of = side_resolver(tiles)
    bad: List[Seam] = []
    for y in range(tile_grid.grid):
        for x in range(tile_grid.grid):
            here = (x, y)
            a = tile_grid.at(here)
            if a is None:
                continue
            for side in OWNED_SIDES:
                there = neighbour(here, side)
                b = tile_grid.at(there)
                if b is None:
                    continue
                try:
                    a_side = sides_of(a)[side]
                    b_side = sides_of(b)[OPPOSITE[side]]
                except _UnknownTile as exc:
                    bad.append(
                        Seam(
                            here,
                            there,
                            side,
                            f"tile id {exc.args[0]} is not in the tile database",
                        )
                    )
                    continue
                shipped = sides_compatible(a_side, b_side)
                derived = geometrically_fits(a_side, b_side)
                if shipped != derived:
                    # The two readings of the *rule* disagree, which is a
                    # louder finding than either verdict: one of
                    # contracts.sides_compatible and the geometric model is
                    # wrong, and the matcher trusts the first of them.
                    bad.append(
                        Seam(
                            here,
                            there,
                            side,
                            "contracts.sides_compatible says "
                            f"{'fits' if shipped else 'does not fit'} but the "
                            "geometric reading says "
                            f"{'fits' if derived else 'does not fit'}: "
                            + _seam_reason(a_side, b_side, side),
                        )
                    )
                elif not shipped:
                    bad.append(Seam(here, there, side, _seam_reason(a_side, b_side, side)))
    return bad


# --------------------------------------------------------------------------
# The whole-map report
# --------------------------------------------------------------------------


@dataclass
class ValidationReport:
    """What the gate found in one map.

    ``ok`` is the gate's verdict: no islands, no seam mismatches, no failed
    sanity check.  Everything else on the report exists so that a failure can
    be read without re-running the generator.
    """

    seed: int
    template_ref: str = ""
    tileset_ref: str = ""
    grid: int = 0
    start_cell: Optional[Cell] = None
    islands: Tuple[Island, ...] = ()
    seams: Tuple[Seam, ...] = ()
    problems: Tuple[Problem, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.islands and not self.seams and not self.problems

    @property
    def island_cells(self) -> int:
        """How much floor the player can never stand on."""
        return sum(len(i) for i in self.islands)

    def counts(self) -> Dict[str, int]:
        """Failures by kind, sorted, for a suite to add up."""
        out: Dict[str, int] = {}
        if self.islands:
            out["navmesh-island"] = len(self.islands)
        if self.seams:
            out["seam-mismatch"] = len(self.seams)
        for problem in self.problems:
            out[problem.kind] = out.get(problem.kind, 0) + 1
        return dict(sorted(out.items()))

    def summary(self, limit: int = 5) -> str:
        """A short, deterministic report a human can act on."""
        head = (
            f"seed {format_seed(self.seed)} template {self.template_ref or '?'} "
            f"tiles {self.tileset_ref or '?'} grid {self.grid}: "
            + ("clean" if self.ok else "FAILED")
        )
        lines = [head]
        if self.islands:
            lines.append(
                f"  {len(self.islands)} navmesh island(s), {self.island_cells} "
                f"cell(s) unreachable from {self.start_cell}"
            )
            for island in self.islands[:limit]:
                cells = sorted(island)
                shown = ", ".join(str(c) for c in cells[:4])
                more = "" if len(cells) <= 4 else f", +{len(cells) - 4} more"
                lines.append(f"    {len(cells)} cell(s): {shown}{more}")
        if self.seams:
            lines.append(f"  {len(self.seams)} seam mismatch(es)")
            for seam in self.seams[:limit]:
                lines.append(f"    {seam}")
        for problem in self.problems[:limit]:
            lines.append(f"  {problem}")
        hidden = max(0, len(self.problems) - limit)
        if hidden:
            lines.append(f"  (+{hidden} more problem(s))")
        return "\n".join(lines)


def _entrance_cell(generated_map: GeneratedMap) -> Optional[Cell]:
    routed = getattr(generated_map, "routed", None)
    if routed is None:
        return None
    node = routed.node_of_role(Role.ENTRANCE)
    return None if node is None else node.cell


def find_stage4_breaks(generated_map: GeneratedMap, tiles: Any) -> List[Problem]:
    """Re-derive stage 4's own invariants over the finished map.

    Gate check 4.  ``tileize.debug_check`` already knows how to prove that
    every cell holds a placement, that each placement satisfies the
    requirement *its own cell* imposes, that filler records match void cells,
    that ``matched_cells`` and ``hero_cells`` are accurate, that the hero
    budget holds, and -- the one no other check can see -- that the border of
    the map presents a wall outward.  A cell showing OPEN off the edge of the
    grid has no neighbour, so it raises no seam and no island; before this was
    wired in, the 1000-seed gate never looked at any of it, because nothing
    outside two unit tests ever called the function.

    Returns an empty list when the map is not one stage 4 produced (a
    hand-built ``TileGrid``, or a map with no terrain plan), since there is
    then no plan to re-derive the requirements from.
    """
    grid = getattr(generated_map, "tiles", None)
    plan = getattr(generated_map, "terrain", None)
    if grid is None or plan is None:
        return []
    if not hasattr(grid, "matched_cells") or not hasattr(grid, "filler"):
        return []  # a plain TileGrid: stage 4 kept no bookkeeping to check
    if getattr(plan, "grid", None) != grid.grid:
        return []  # mismatched shapes are build_layout_description's finding

    from .tileize import debug_check  # deferred: tileize imports the database

    template = getattr(generated_map, "template", None)
    tile_class = getattr(template, "tile_class", None) or getattr(
        grid, "tile_class", None
    )
    return [
        Problem("stage4-invariant", message)
        for message in debug_check(grid, plan, tiles, tile_class)
    ]


def validate_map(generated_map: GeneratedMap, tiles: Any) -> ValidationReport:
    """Run the whole gate over one finished map.

    ``tiles`` is the tile *database* (stage 4's input); the map's own
    ``tiles`` attribute is the placed :class:`~contracts.TileGrid`.

    The two gate checks run first, then the cheap sanity checks the spec asks
    for: the exit cell is walkable, every set piece lies inside the grid,
    every spawn sits on a walkable cell that is not boss-approach floor, and
    the entrance can walk to the exit.  Islands are measured from the
    entrance, since "reachable" only means anything from where the player
    starts.
    """
    tile_grid: TileGrid = generated_map.tiles
    problems: List[Problem] = []

    entrance = _entrance_cell(generated_map)
    exit_cell = generated_map.exit_cell

    # The island start: the entrance if there is one, else the exit, so a
    # partial map still gets a useful answer instead of no answer.
    start = entrance if entrance is not None else exit_cell
    if entrance is None:
        problems.append(
            Problem("entrance-missing", "the routed layout has no entrance node")
        )
    elif not tile_grid.is_walkable(entrance):
        problems.append(
            Problem("entrance-not-walkable", "the entrance cell is not walkable", entrance)
        )

    islands = (
        find_navmesh_islands(tile_grid, start, tiles) if start is not None else []
    )
    seams = find_seam_mismatches(tile_grid, tiles)
    problems.extend(find_transform_breaks(tile_grid, tiles))
    problems.extend(find_stage4_breaks(generated_map, tiles))

    # -- exit ------------------------------------------------------------
    if exit_cell is None:
        problems.append(Problem("exit-missing", "the map names no exit cell"))
    elif not tile_grid.inside(exit_cell):
        problems.append(
            Problem("exit-out-of-bounds", "the exit cell is off the grid", exit_cell)
        )
    elif not tile_grid.is_walkable(exit_cell):
        problems.append(
            Problem("exit-not-walkable", "the exit cell is not walkable", exit_cell)
        )
    elif entrance is not None and not reaches(
        tile_grid, entrance, exit_cell, tiles=tiles
    ):
        problems.append(
            Problem(
                "exit-unreachable",
                f"no walkable path from the entrance at {entrance} to the exit",
                exit_cell,
            )
        )

    # -- set pieces ------------------------------------------------------
    for piece in generated_map.set_pieces or ():
        x, y = piece.cell
        far = (x + max(1, piece.w) - 1, y + max(1, piece.h) - 1)
        if not tile_grid.inside(piece.cell) or not tile_grid.inside(far):
            problems.append(
                Problem(
                    "set-piece-out-of-bounds",
                    f"set piece {piece.id!r} covers {piece.w}x{piece.h} from "
                    f"{piece.cell} to {far}, which leaves a {tile_grid.grid}-cell grid",
                    piece.cell,
                )
            )

    # -- spawns ----------------------------------------------------------
    terrain = getattr(generated_map, "terrain", None)
    for pack in generated_map.spawns or ():
        cell = pack.cell
        if not tile_grid.inside(cell):
            problems.append(
                Problem("spawn-out-of-bounds", f"pack {pack.pack!r} is off the grid", cell)
            )
            continue
        if not tile_grid.is_walkable(cell):
            problems.append(
                Problem(
                    "spawn-not-walkable",
                    f"pack {pack.pack!r} stands on an unwalkable cell",
                    cell,
                )
            )
        if terrain is not None and terrain.kind(cell) is CellKind.APPROACH:
            problems.append(
                Problem(
                    "spawn-on-approach",
                    f"pack {pack.pack!r} sits on boss approach floor, which the "
                    "spec keeps at zero ambient density",
                    cell,
                )
            )

    return ValidationReport(
        seed=int(getattr(generated_map, "seed", 0)),
        template_ref=getattr(getattr(generated_map, "template", None), "ref", ""),
        tileset_ref=str(getattr(generated_map, "tileset_ref", "")),
        grid=tile_grid.grid,
        start_cell=start,
        islands=tuple(islands),
        seams=tuple(seams),
        problems=tuple(problems),
    )


# --------------------------------------------------------------------------
# The seed sweep (the 1000-seed gate)
# --------------------------------------------------------------------------


def suite_seed(start_seed: int, index: int) -> int:
    """The ``index``-th seed of a suite starting at ``start_seed``.

    Index 0 is ``start_seed`` itself, so re-running a reported failure is
    ``run_suite(..., n_seeds=1, start_seed=<that seed>)``.  Later indices are
    hashed rather than counted up: the seed's fields are spread across all 64
    bits (rotation in bits 0-1, tiles in bits 32-63, see ``seed.py``), and a
    sweep of ``start + i`` would leave everything above bit 10 fixed, testing
    one tiling a thousand times.

    blake2b, always -- never blake3 even where the wheel exists -- so two
    machines running the gate cover the same seeds.
    """
    if index == 0:
        return int(start_seed) & MASK64
    digest = hashlib.blake2b(
        f"{SUITE_DOMAIN}:{int(start_seed) & MASK64}:{int(index)}".encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") & MASK64


def build_map(
    template: GraphTemplate,
    tiles: "TileDatabase",
    rooms: Optional["RoomLibrary"],
    seed: int,
    *,
    tier: float = 1.0,
    sigil_modifiers: Any = None,
) -> GeneratedMap:
    """Run stages 1 to 6 for one seed and collect them into a GeneratedMap.

    Thin wrapper over :func:`lucifer_gen.pipeline.generate`, which owns the
    stage order: the gate must measure the pipeline the CLI ships, not a
    second copy of it that can drift.  The import is deferred so that this
    module stays importable -- and its unit tests runnable -- while a sibling
    stage is mid-rewrite.
    """
    from .pipeline import generate

    return generate(
        template, tiles, rooms, seed, tier=tier, sigil_modifiers=sigil_modifiers
    )


def validate_seed(
    template: GraphTemplate,
    tiles: "TileDatabase",
    rooms: Optional["RoomLibrary"],
    seed: int,
    *,
    tier: float = 1.0,
    sigil_modifiers: Any = None,
) -> ValidationReport:
    """Generate one map and gate it. The unit :func:`run_suite` repeats."""
    return validate_map(
        build_map(
            template, tiles, rooms, seed, tier=tier, sigil_modifiers=sigil_modifiers
        ),
        tiles,
    )


@dataclass
class SuiteFailure:
    """The first seed that failed, with enough detail to reproduce it."""

    index: int
    seed: int
    template_ref: str
    tileset_ref: str
    report: Optional[ValidationReport] = None
    error: str = ""  # the traceback, when generation or validation raised

    @property
    def crashed(self) -> bool:
        return bool(self.error)

    def repro(self) -> str:
        """The one-liner that reruns exactly this map."""
        return (
            "validate_seed(template, tiles, rooms, "
            f"seed={format_seed(self.seed)})  # template "
            f"{self.template_ref or '?'}, tiles {self.tileset_ref or '?'}, "
            f"suite index {self.index}"
        )

    def summary(self) -> str:
        lines = [f"first failing seed: {format_seed(self.seed)} (index {self.index})"]
        if self.report is not None:
            lines.append(self.report.summary())
        if self.error:
            lines.append("  raised while generating or validating:")
            lines.extend(f"    {ln}" for ln in self.error.rstrip().splitlines())
        lines.append(f"  reproduce with: {self.repro()}")
        return "\n".join(lines)


@dataclass
class SuiteReport:
    """Aggregate counts over a seed sweep. No map is kept alive to build it."""

    template_ref: str
    tileset_ref: str
    requested: int
    checked: int = 0
    clean: int = 0
    failed: int = 0
    crashed: int = 0
    islands: int = 0  # total islands found across every map
    island_maps: int = 0  # maps carrying at least one island
    island_cells: int = 0
    seams: int = 0
    seam_maps: int = 0
    problems: Dict[str, int] = field(default_factory=dict)
    first_failure: Optional[SuiteFailure] = None

    @property
    def ok(self) -> bool:
        return self.checked > 0 and self.failed == 0

    def summary(self) -> str:
        lines = [
            f"{self.template_ref or '?'} / {self.tileset_ref or '?'}: "
            f"{self.checked} of {self.requested} seed(s) checked, "
            f"{self.clean} clean, {self.failed} failed "
            f"({self.crashed} of them raised)",
            f"  navmesh islands: {self.islands} across {self.island_maps} map(s), "
            f"{self.island_cells} unreachable cell(s)",
            f"  seam mismatches: {self.seams} across {self.seam_maps} map(s)",
        ]
        for kind, count in sorted(self.problems.items()):
            lines.append(f"  {kind}: {count}")
        if self.first_failure is not None:
            lines.append(self.first_failure.summary())
        return "\n".join(lines)


def run_suite(
    template: GraphTemplate,
    tiles: "TileDatabase",
    rooms: Optional["RoomLibrary"] = None,
    n_seeds: int = 1000,
    start_seed: int = 0,
    *,
    tier: float = 1.0,
    sigil_modifiers: Any = None,
    seeds: Optional[Iterable[int]] = None,
    generate: Optional[Callable[[int], GeneratedMap]] = None,
    on_result: Optional[Callable[[int, int, Optional[ValidationReport]], None]] = None,
    stop_early: bool = False,
) -> SuiteReport:
    """Gate ``n_seeds`` maps and return the aggregate counts.

    This is the CLI's 1000-seed gate.  It **streams**: one map is generated,
    validated, folded into the counters, and dropped before the next seed
    starts, so peak memory is one map rather than a thousand.  The only map
    detail that outlives its iteration is the first failure's report, kept so
    the run can say what went wrong without a second pass.

    ``seeds`` overrides the derived sweep with an explicit list -- handy for
    re-running a known-bad set.  ``generate`` overrides how a map is built,
    for a caller that owns a pipeline of its own.  ``on_result`` is called as
    ``(index, seed, report)`` after every seed, with ``report`` ``None`` when
    that seed raised; it is the hook a CLI prints progress from.
    ``stop_early`` stops at the first failure instead of counting them all.
    """
    if generate is None:

        def generate(seed: int) -> GeneratedMap:  # noqa: F811 - deliberate default
            return build_map(
                template, tiles, rooms, seed, tier=tier, sigil_modifiers=sigil_modifiers
            )

    if seeds is None:
        requested = max(0, int(n_seeds))
        stream: Iterator[int] = (suite_seed(start_seed, i) for i in range(requested))
    else:
        materialised = list(seeds)
        requested = len(materialised)
        stream = iter(materialised)

    report = SuiteReport(
        template_ref=getattr(template, "ref", ""),
        tileset_ref=str(getattr(tiles, "version", "")),
        requested=requested,
    )

    for index, seed in enumerate(stream):
        one: Optional[ValidationReport] = None
        error = ""
        try:
            one = validate_map(generate(seed), tiles)
        except Exception:  # noqa: BLE001 - a crash is this seed's verdict
            error = traceback.format_exc(limit=12)

        report.checked += 1
        if on_result is not None:
            on_result(index, seed, one)

        if one is not None and one.ok:
            report.clean += 1
        else:
            report.failed += 1
            if one is None:
                report.crashed += 1
            else:
                if one.islands:
                    report.island_maps += 1
                    report.islands += len(one.islands)
                    report.island_cells += one.island_cells
                if one.seams:
                    report.seam_maps += 1
                    report.seams += len(one.seams)
                for kind, count in one.counts().items():
                    report.problems[kind] = report.problems.get(kind, 0) + count
            if report.first_failure is None:
                report.first_failure = SuiteFailure(
                    index=index,
                    seed=seed,
                    template_ref=report.template_ref,
                    tileset_ref=report.tileset_ref,
                    report=one,
                    error=error,
                )
            if stop_early:
                break
        # `one` and the map behind it fall out of scope here; nothing but
        # counters and the first failure survives into the next seed.

    report.problems = dict(sorted(report.problems.items()))
    return report


__all__ = [
    "Island",
    "OWNED_SIDES",
    "Problem",
    "Seam",
    "StepRule",
    "SuiteFailure",
    "SuiteReport",
    "SUITE_DOMAIN",
    "ValidationReport",
    "build_map",
    "find_navmesh_islands",
    "find_seam_mismatches",
    "find_stage4_breaks",
    "find_transform_breaks",
    "flood",
    "geometric_sides",
    "geometrically_fits",
    "placed_placements",
    "reaches",
    "run_suite",
    "seam_is_open",
    "side_resolver",
    "step_rule",
    "suite_seed",
    "validate_map",
    "validate_seed",
    "walkable_cells",
]
