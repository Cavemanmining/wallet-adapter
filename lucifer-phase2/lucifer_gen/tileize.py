"""Stage 4: tileization -- turning a :class:`TerrainPlan` into a ``TileGrid``.

Spec: docs/WORLD_BIBLE.md stage 4, "Tileization".

    Compute the required edge signature for each cell from its neighbours,
    pick uniformly among fitting tiles using the tile seed field with rotation
    and flip, and weight variants so a hero variant appears at most once per
    20 cells.  Cells the routing never touched get impassable filler carrying
    a navmesh exclusion and a collision hull 1 m beyond the mesh.

How the requirement for a cell is derived
-----------------------------------------
Stage 3 hands over a *kind* per cell; stage 4 turns that into a *surface*
(:class:`Surface`) and then reads the signature a cell must present toward
each neighbour out of one symmetric table, :data:`SIG_BETWEEN`.

The table is stated pairwise rather than "from the floor's point of view" on
purpose.  The seam rule in ``contracts.sides_compatible`` is a statement about
a pair of tiles, so the only way to guarantee it by construction -- rather
than by hoping -- is for the requirement placed on cell A's east side and the
requirement placed on its eastern neighbour's west side to be complements of
one another *by definition*.  :func:`_validate_sig_table` checks exactly that
at import time, so a future edit to the table cannot quietly break seams.

Outside the grid counts as :attr:`Surface.VOID`, which is why the border of
the map is sealed with walls.

What each surface means
-----------------------
``FLOOR``   any of CORRIDOR, ROOM, SET_PIECE, APPROACH -- walkable ground.
``CLIFF``   an outdoor escarpment cell.  It presents ``CLIFF_UP`` toward the
            floor beside it, so the floor presents ``CLIFF_DOWN``: the same
            escarpment named from its two sides, per ``SIG_COMPLEMENT``.
``WATER``   an outdoor water cell; water meets water.
``VOID``    a cell routing never touched.  It gets the filler tile, is not
            walkable, and is walled on every side.

``CLIFF`` and ``WATER`` only exist for the outdoor tile classes.  A dungeon
plan that contains them is a stage 3 error; rather than crash, stage 4
degrades them to ``VOID`` (sealed filler) -- see :data:`TERRAIN_CLASSES`.

Randomness
----------
Every draw comes off a single :class:`~lucifer_gen.seed.Stream` labelled
``"tile-select"``, which ``seed.SeedFields.stream`` routes to the tile field
(bits 32-63).  Cells are visited in row-major order, so the sequence of draws
-- and therefore the whole grid -- is a pure function of the seed, the plan
and the database.  Filler cells consume no randomness at all.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

from .contracts import (
    ALL_SLOTS,
    E,
    NO_SLOTS,
    OPPOSITE,
    S,
    SIDE_NAMES,
    SIDES,
    SIG_COMPLEMENT,
    Cell,
    CellKind,
    EdgeSig,
    Placement,
    SideSpec,
    TerrainPlan,
    TileClass,
    TileGrid,
    neighbour,
    sides_compatible,
)
from .seed import SeedFields, Stream
from .tiles import HeroBudget, NoFittingTile, TileDatabase

#: The stream stage 4 draws from.  The prefix matters: ``SeedFields.stream``
#: routes any label starting with "tile" to the tile field of the seed.
TILE_STREAM_LABEL = "tile-select"

#: Spec stage 4: filler carries "a collision hull 1 m beyond the mesh".
#: The database may override it per tile via ``collision_margin_m``.
COLLISION_MARGIN_M = 1.0

#: Which cell kinds stage 3 considers walkable ground.  Mirrors
#: ``TerrainPlan.is_floor`` but as a set, so a cell can be classified once.
FLOOR_KINDS = frozenset(
    {CellKind.CORRIDOR, CellKind.ROOM, CellKind.SET_PIECE, CellKind.APPROACH}
)

#: Tile classes whose databases carry cliff and water signatures.  For
#: ``DUNGEON`` those cell kinds are treated as untouched void instead.
TERRAIN_CLASSES = frozenset({TileClass.OUTDOOR, TileClass.BOTH})


class TileizeError(RuntimeError):
    """Stage 4 was handed a plan or a database it cannot honour."""


# --------------------------------------------------------------------------
# Surfaces and the signature table
# --------------------------------------------------------------------------


class Surface(enum.Enum):
    """What stage 4 treats a cell as, once the tile class is known."""

    VOID = "void"
    FLOOR = "floor"
    CLIFF = "cliff"
    WATER = "water"


#: What surface ``here`` presents along an edge it shares with ``there``.
#:
#: Read it as ``SIG_BETWEEN[(here, there)]``.  Every entry has its mirror,
#: and the two are complements under ``contracts.SIG_COMPLEMENT`` -- that is
#: the whole reason seams are correct by construction.
SIG_BETWEEN: Dict[Tuple[Surface, Surface], EdgeSig] = {
    # Ground meeting ground is the only thing an actor may walk across.
    (Surface.FLOOR, Surface.FLOOR): EdgeSig.OPEN,
    # An escarpment seen from below and from above.
    (Surface.FLOOR, Surface.CLIFF): EdgeSig.CLIFF_DOWN,
    (Surface.CLIFF, Surface.FLOOR): EdgeSig.CLIFF_UP,
    # A shoreline: both sides show the waterline.
    (Surface.FLOOR, Surface.WATER): EdgeSig.WATER,
    (Surface.WATER, Surface.FLOOR): EdgeSig.WATER,
    (Surface.CLIFF, Surface.WATER): EdgeSig.WATER,
    (Surface.WATER, Surface.CLIFF): EdgeSig.WATER,
    (Surface.WATER, Surface.WATER): EdgeSig.WATER,
    # Two cliff cells sit at the same height, so no escarpment runs between
    # them; the edge is simply closed.  (Judgement call: the plan records no
    # up/down direction, so the only self-consistent choice is a wall.)
    (Surface.CLIFF, Surface.CLIFF): EdgeSig.WALL,
    # Anything meeting untouched void is sealed, because the filler tile is
    # walled on all four sides.
    (Surface.FLOOR, Surface.VOID): EdgeSig.WALL,
    (Surface.VOID, Surface.FLOOR): EdgeSig.WALL,
    (Surface.CLIFF, Surface.VOID): EdgeSig.WALL,
    (Surface.VOID, Surface.CLIFF): EdgeSig.WALL,
    (Surface.WATER, Surface.VOID): EdgeSig.WALL,
    (Surface.VOID, Surface.WATER): EdgeSig.WALL,
    (Surface.VOID, Surface.VOID): EdgeSig.WALL,
}


def _validate_sig_table() -> None:
    """Check at import that the table is total and self-complementary.

    Total: every ordered pair of surfaces has an entry.  Complementary: what
    A shows B is the complement of what B shows A.  Together these make
    ``sides_compatible`` true for every seam stage 4 can emit, before a
    single tile is looked up.
    """
    for here in Surface:
        for there in Surface:
            key = (here, there)
            if key not in SIG_BETWEEN:
                raise TileizeError(f"SIG_BETWEEN is missing {here.value}->{there.value}")
            mine = SIG_BETWEEN[key]
            theirs = SIG_BETWEEN[(there, here)]
            if SIG_COMPLEMENT[mine] is not theirs:
                raise TileizeError(
                    f"SIG_BETWEEN is asymmetric: {here.value}->{there.value} is "
                    f"{mine.name} but {there.value}->{here.value} is {theirs.name}"
                )


_validate_sig_table()


def _spec(sig: EdgeSig) -> SideSpec:
    """A side spec for one signature, with full slots where slots mean anything.

    Only an open side can be joined, so it asks for all three connection
    slots (the greybox database offers all three on every open side).  Every
    other signature carries no slots; ``SideSpec`` enforces that anyway.
    """
    return SideSpec(sig, ALL_SLOTS if sig is EdgeSig.OPEN else NO_SLOTS)


#: Precomputed, because the loop asks for these tens of thousands of times.
_SPEC_BETWEEN: Dict[Tuple[Surface, Surface], SideSpec] = {
    key: _spec(sig) for key, sig in SIG_BETWEEN.items()
}

#: The requirement a fully sealed cell (the filler) must meet.
SEALED: Tuple[SideSpec, SideSpec, SideSpec, SideSpec] = tuple(  # type: ignore[assignment]
    _spec(EdgeSig.WALL) for _ in SIDES
)


def surface_of(plan: TerrainPlan, cell: Cell, tile_class: TileClass) -> Surface:
    """Classify one cell of the plan, honouring the tile class.

    Cells outside the grid read as :attr:`Surface.VOID`, which is what seals
    the border of the map.
    """
    if not plan.inside(cell):
        return Surface.VOID
    kind = plan.kind(cell)
    if kind in FLOOR_KINDS:
        return Surface.FLOOR
    if tile_class in TERRAIN_CLASSES:
        if kind is CellKind.CLIFF:
            return Surface.CLIFF
        if kind is CellKind.WATER:
            return Surface.WATER
    return Surface.VOID


def required_sides(
    plan: TerrainPlan,
    cell: Cell,
    tile_class: TileClass,
    here: Optional[Surface] = None,
) -> Tuple[SideSpec, SideSpec, SideSpec, SideSpec]:
    """The four side specs a tile in ``cell`` must present, in N, E, S, W order.

    ``here`` is the cell's own surface, passed in when the caller has already
    computed it.  Each side is decided purely by the pair (this surface, the
    neighbour's surface), so the requirement computed here and the one
    computed for the neighbour are complements by construction.
    """
    if here is None:
        here = surface_of(plan, cell, tile_class)
    return tuple(  # type: ignore[return-value]
        _SPEC_BETWEEN[(here, surface_of(plan, neighbour(cell, side), tile_class))]
        for side in SIDES
    )


# --------------------------------------------------------------------------
# Filler bookkeeping: navmesh exclusions and collision hulls
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in world metres, x running east and y south."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def grown(self, margin: float) -> "Box":
        """This box pushed out by ``margin`` metres on every side."""
        return Box(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin,
        )

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    @staticmethod
    def of_cell(cell: Cell, cell_m: float) -> "Box":
        x, y = cell
        return Box(x * cell_m, y * cell_m, (x + 1) * cell_m, (y + 1) * cell_m)


@dataclass(frozen=True)
class FillerCell:
    """One untouched cell, with the navigation data the runtime needs.

    Spec stage 4: filler carries "a navmesh exclusion and a collision hull
    1 m beyond the mesh".  ``mesh`` is the visible footprint of the filler
    tile; the navmesh is cut away over exactly that footprint, while the
    collision hull stands ``margin_m`` metres proud of it so an actor is
    stopped before it can clip the geometry.
    """

    cell: Cell
    mesh: Box
    hull: Box
    margin_m: float = COLLISION_MARGIN_M

    @property
    def navmesh_exclusion(self) -> Box:
        """The region to cut out of the navmesh: the visible mesh footprint."""
        return self.mesh


# --------------------------------------------------------------------------
# The grid stage 4 returns
# --------------------------------------------------------------------------


@dataclass
class TileizedGrid(TileGrid):
    """A :class:`TileGrid` plus everything else stage 4 learned.

    It *is* a ``TileGrid`` -- stages 5 and 6 can treat it as one and never
    look at the extra fields -- so the promised return type holds while the
    filler's navmesh and collision data still has somewhere honest to live.

    ``filler`` is ordered row-major, the same order the cells were emitted
    in, so two runs with the same inputs compare equal element for element.
    """

    tile_class: TileClass = TileClass.BOTH
    tileset_ref: str = ""
    cell_m: float = 4.0
    filler: Tuple[FillerCell, ...] = ()
    hero_cells: Tuple[Cell, ...] = ()
    matched_cells: int = 0

    def navmesh_exclusions(self) -> Tuple[Box, ...]:
        """Every region the navmesh baker must leave out."""
        return tuple(f.navmesh_exclusion for f in self.filler)

    def collision_hulls(self) -> Tuple[Box, ...]:
        """Every filler collision hull, already grown past its mesh."""
        return tuple(f.hull for f in self.filler)

    def filler_cells(self) -> Tuple[Cell, ...]:
        return tuple(f.cell for f in self.filler)

    @property
    def hero_count(self) -> int:
        return len(self.hero_cells)

    def packed_cells(self) -> bytes:
        """The client's ``cells`` blob: two bytes per cell, row-major.

        Tile id first, then rotation in bits 0-1 and flip in bit 2, exactly
        as ``Placement.packed`` defines it.  Stage 6 base64-encodes this.
        """
        out = bytearray()
        for row in self.cells:
            for placement in row:
                if placement is None:
                    raise TileizeError("cannot pack a grid with an empty cell")
                tile_byte, flags = placement.packed()
                out.append(tile_byte)
                out.append(flags)
        return bytes(out)


# --------------------------------------------------------------------------
# Seam checking (the debug check the spec asks for)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SeamMismatch:
    """Two adjacent placements that ``sides_compatible`` rejects."""

    a: Cell
    side: int
    b: Cell
    a_side: SideSpec
    b_side: SideSpec

    def __str__(self) -> str:  # pragma: no cover - diagnostic text
        return (
            f"{self.a} {SIDE_NAMES[self.side]} shows "
            f"{self.a_side.sig.name}:{self.a_side.slots:03b} but {self.b} "
            f"{SIDE_NAMES[OPPOSITE[self.side]]} shows "
            f"{self.b_side.sig.name}:{self.b_side.slots:03b}"
        )


class SeamError(AssertionError):
    """Raised by :func:`assert_seams` when the grid does not tile cleanly."""

    def __init__(self, mismatches: Sequence[SeamMismatch]) -> None:
        self.mismatches = tuple(mismatches)
        shown = "; ".join(str(m) for m in self.mismatches[:5])
        more = "" if len(self.mismatches) <= 5 else f" (+{len(self.mismatches) - 5} more)"
        super().__init__(f"{len(self.mismatches)} seam mismatch(es): {shown}{more}")


def check_seams(grid: TileGrid, tiles: TileDatabase) -> List[SeamMismatch]:
    """Every seam in the grid that breaks ``contracts.sides_compatible``.

    Each interior seam is tested once, from its west or north cell, using the
    sides the placements *actually* present after rotation and flip -- not the
    requirement they were chosen against.  That is the point: it re-derives
    the answer from the placed geometry, so a bug in the requirement table,
    the matcher or the transform algebra all show up here.
    """
    bad: List[SeamMismatch] = []
    # Cache: a placement's resolved sides, since a 48x48 grid reuses a few
    # hundred distinct placements thousands of times.
    resolved: Dict[Placement, Tuple[SideSpec, ...]] = {}

    def sides_of(placement: Placement) -> Tuple[SideSpec, ...]:
        hit = resolved.get(placement)
        if hit is None:
            hit = tiles.sides_of(placement)
            resolved[placement] = hit
        return hit

    for y in range(grid.grid):
        for x in range(grid.grid):
            here = (x, y)
            a = grid.at(here)
            if a is None:
                continue
            for side in (E, S):
                there = neighbour(here, side)
                b = grid.at(there)
                if b is None:
                    continue
                a_side = sides_of(a)[side]
                b_side = sides_of(b)[OPPOSITE[side]]
                if not sides_compatible(a_side, b_side):
                    bad.append(SeamMismatch(here, side, there, a_side, b_side))
    return bad


def assert_seams(grid: TileGrid, tiles: TileDatabase) -> None:
    """Raise :class:`SeamError` unless every adjacent pair of tiles fits."""
    bad = check_seams(grid, tiles)
    if bad:
        raise SeamError(bad)


def debug_check(
    grid: TileizedGrid,
    plan: TerrainPlan,
    tiles: TileDatabase,
    tile_class: Optional[TileClass] = None,
) -> List[str]:
    """Re-derive every stage 4 invariant from the output and list the breaks.

    Returns a list of human-readable problems, empty when the grid is sound.
    Checked here rather than trusted:

    * every cell holds a placement;
    * floor cells are walkable, everything else is not;
    * untouched cells hold the filler tile and appear in ``filler``;
    * every placement actually satisfies the requirement its cell imposes;
    * the border of the map presents a wall outward;
    * no seam breaks ``sides_compatible``;
    * heroes stay inside the one-per-``hero_period`` budget.
    """
    if tile_class is None:
        tile_class = grid.tile_class
    problems: List[str] = []

    if grid.grid != plan.grid:
        problems.append(f"grid is {grid.grid} but the plan is {plan.grid}")
        return problems

    filler_lookup = {f.cell for f in grid.filler}
    heroes = 0
    matched = 0

    for y in range(plan.grid):
        for x in range(plan.grid):
            cell = (x, y)
            placement = grid.at(cell)
            if placement is None:
                problems.append(f"{cell} has no placement")
                continue
            here = surface_of(plan, cell, tile_class)
            want = required_sides(plan, cell, tile_class, here)
            if not tiles.placement_fits(placement, want):
                problems.append(f"{cell} tile does not meet its own requirement")
            tile = tiles.by_id(placement.tile_id)
            if here is Surface.FLOOR:
                if not grid.is_walkable(cell):
                    problems.append(f"floor cell {cell} is not walkable")
                if not tile.walkable:
                    problems.append(f"floor cell {cell} holds unwalkable {tile.name!r}")
            else:
                if grid.is_walkable(cell):
                    problems.append(f"non-floor cell {cell} is walkable")
            if here is Surface.VOID:
                if placement.tile_id != tiles.filler_tile_id:
                    problems.append(f"untouched cell {cell} is not filler")
                if cell not in filler_lookup:
                    problems.append(f"untouched cell {cell} has no filler record")
            else:
                matched += 1
                if cell in filler_lookup:
                    problems.append(f"cell {cell} is not void but is recorded as filler")
            if tile.hero:
                heroes += 1

            # The border must be sealed: outward-facing sides are walls.
            for side in SIDES:
                if not plan.inside(neighbour(cell, side)):
                    shown = tiles.sides_of(placement)[side]
                    if shown.sig is not EdgeSig.WALL:
                        problems.append(
                            f"{cell} shows {shown.sig.name} off the edge of the map"
                        )

    if matched != grid.matched_cells:
        problems.append(
            f"matched_cells says {grid.matched_cells} but {matched} cells were matched"
        )
    if heroes != len(grid.hero_cells):
        problems.append(
            f"hero_cells lists {len(grid.hero_cells)} but {heroes} hero tiles are placed"
        )
    period = tiles.hero_period
    if heroes * period > matched:
        problems.append(
            f"hero budget blown: {heroes} heroes in {matched} matched cells "
            f"(limit one per {period})"
        )

    problems.extend(str(m) for m in check_seams(grid, tiles))
    return problems


# --------------------------------------------------------------------------
# Stage 4 proper
# --------------------------------------------------------------------------


def _as_fields(seed: Union[int, SeedFields]) -> SeedFields:
    return seed if isinstance(seed, SeedFields) else SeedFields.parse(int(seed))


def tileize(
    plan: TerrainPlan,
    tiles: TileDatabase,
    tile_class: TileClass,
    seed: Union[int, SeedFields],
    *,
    verify: bool = True,
) -> TileizedGrid:
    """Stage 4: choose a concrete tile, rotation and flip for every cell.

    Spec: docs/WORLD_BIBLE.md stage 4.

    Walks the grid in row-major order.  For each cell it derives the four
    side specs the cell must present (see :func:`required_sides`), then either

    * drops the impassable filler tile, if routing never touched the cell,
      recording its navmesh exclusion and collision hull; or
    * asks the database for a fitting tile, drawing from the tile seed field,
      with hero variants allowed only while the budget has room.

    ``verify`` runs :func:`assert_seams` before returning.  It is on by
    default and costs about one dictionary lookup per seam: a grid that does
    not tile is a bug worth failing loudly on, not one worth shipping.

    Returns a :class:`TileizedGrid`, which is a ``TileGrid`` carrying the
    filler navmesh data alongside.
    """
    if plan.grid <= 0:
        raise TileizeError(f"a plan needs at least one cell, got grid={plan.grid}")
    if not isinstance(tile_class, TileClass):
        raise TypeError(f"tile_class must be a TileClass, got {tile_class!r}")

    fields = _as_fields(seed)
    stream: Stream = fields.stream(TILE_STREAM_LABEL)
    budget = HeroBudget(tiles.hero_period)

    cell_m = float(tiles.cell_m)
    filler_placement = tiles.filler_placement()
    filler_meta = tiles.meta(tiles.filler_tile_id)
    filler_margin = float(filler_meta.get("collision_margin_m", COLLISION_MARGIN_M))
    # The filler seals a cell on all four sides; if the database's filler does
    # not, every seam against it would fail, so say so plainly and early.
    filler_seals = tiles.placement_fits(filler_placement, SEALED)

    grid = TileizedGrid(
        grid=plan.grid,
        cells=[[None] * plan.grid for _ in range(plan.grid)],
        walkable=[[False] * plan.grid for _ in range(plan.grid)],
        tile_class=tile_class,
        tileset_ref=tiles.version,
        cell_m=cell_m,
    )

    filler_cells: List[FillerCell] = []
    hero_cells: List[Cell] = []
    matched = 0

    for y in range(plan.grid):
        for x in range(plan.grid):
            cell = (x, y)
            here = surface_of(plan, cell, tile_class)

            if here is Surface.VOID:
                if not filler_seals:
                    raise TileizeError(
                        f"filler tile {tiles.filler.name!r} is not walled on all "
                        "four sides, so it cannot seal an untouched cell"
                    )
                grid.put(cell, filler_placement, walkable=False)
                mesh = Box.of_cell(cell, cell_m)
                filler_cells.append(
                    FillerCell(
                        cell=cell,
                        mesh=mesh,
                        hull=mesh.grown(filler_margin),
                        margin_m=filler_margin,
                    )
                )
                # Filler is not chosen, so it neither draws from the stream
                # nor spends hero budget.
                continue

            want = required_sides(plan, cell, tile_class, here)
            try:
                placement = tiles.find(
                    want, tile_class, stream, allow_hero=budget.allows_hero()
                )
            except NoFittingTile as exc:
                raise NoFittingTile(f"cell {cell} ({here.value}): {exc}") from exc

            tile = tiles.by_id(placement.tile_id)
            if here is Surface.FLOOR and not tile.walkable:
                raise TileizeError(
                    f"cell {cell} is floor but the matcher returned unwalkable "
                    f"tile {tile.name!r}"
                )
            budget.record(tile.hero)
            matched += 1
            if tile.hero:
                hero_cells.append(cell)
            # Cliff and water are terrain the actor does not stand on, so only
            # true floor is marked walkable.
            grid.put(cell, placement, walkable=here is Surface.FLOOR)

    grid.filler = tuple(filler_cells)
    grid.hero_cells = tuple(hero_cells)
    grid.matched_cells = matched

    if verify:
        assert_seams(grid, tiles)

    return grid


__all__ = [
    "COLLISION_MARGIN_M",
    "FLOOR_KINDS",
    "SEALED",
    "SIG_BETWEEN",
    "TERRAIN_CLASSES",
    "TILE_STREAM_LABEL",
    "Box",
    "FillerCell",
    "SeamError",
    "SeamMismatch",
    "Surface",
    "TileizeError",
    "TileizedGrid",
    "assert_seams",
    "check_seams",
    "debug_check",
    "required_sides",
    "surface_of",
    "tileize",
]
