"""Shared types and geometry for the Lucifer map generator.

This module is the keystone: every stage builds against these types, so the
transform algebra and signature-compatibility rules live here and nowhere else.

Spec: docs/WORLD_BIBLE.md sections 04 and 05.

Coordinate system
-----------------
Cells are (x, y) with x running east and y running south, origin top-left.
The grid is square, ``grid`` cells on a side, each ``cell_m`` metres.

Sides are indexed 0..3 in clockwise order starting north: N, E, S, W.
Rotation is in 90-degree clockwise steps, so rotating a tile by one step
moves the value on side i to side (i + 1) % 4.

Connection slots
----------------
Each side carries a 3-bit mask naming which thirds of that edge a path may
join at. Slot order is clockwise-consistent so that rotation never has to
reorder bits: north runs west to east, east runs north to south, south runs
east to west, west runs south to north. Bit 0 is the first third in that
direction, bit 2 the last. A mirror reverses the order within a side, which
is why ``_reverse_slots`` exists.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Cell = Tuple[int, int]

# --------------------------------------------------------------------------
# Sides
# --------------------------------------------------------------------------

N, E, S, W = 0, 1, 2, 3
SIDES = (N, E, S, W)
SIDE_NAMES = ("N", "E", "S", "W")

#: Unit step in cell coordinates for each side, in N/E/S/W order.
SIDE_DELTA: Tuple[Cell, ...] = ((0, -1), (1, 0), (0, 1), (-1, 0))

#: The side of the neighbour that faces this side.
OPPOSITE: Tuple[int, ...] = (S, W, N, E)


def neighbour(cell: Cell, side: int) -> Cell:
    """Return the cell adjacent to ``cell`` across ``side``."""
    dx, dy = SIDE_DELTA[side]
    return (cell[0] + dx, cell[1] + dy)


# --------------------------------------------------------------------------
# Edge signatures
# --------------------------------------------------------------------------


class EdgeSig(enum.IntEnum):
    """What the tile presents along one of its four sides."""

    OPEN = 0
    WALL = 1
    CLIFF_UP = 2
    CLIFF_DOWN = 3
    WATER = 4


#: Two tiles may sit side by side only if each side maps to the other's.
#: Open meets open, wall meets wall, water meets water, and a cliff that
#: rises meets one that falls, because they describe the same escarpment
#: seen from its two sides.
SIG_COMPLEMENT: Dict[EdgeSig, EdgeSig] = {
    EdgeSig.OPEN: EdgeSig.OPEN,
    EdgeSig.WALL: EdgeSig.WALL,
    EdgeSig.CLIFF_UP: EdgeSig.CLIFF_DOWN,
    EdgeSig.CLIFF_DOWN: EdgeSig.CLIFF_UP,
    EdgeSig.WATER: EdgeSig.WATER,
}

ALL_SLOTS = 0b111
NO_SLOTS = 0b000


def _reverse_slots(mask: int) -> int:
    """Mirror a 3-bit slot mask end for end."""
    return ((mask & 0b001) << 2) | (mask & 0b010) | ((mask & 0b100) >> 2)


@dataclass(frozen=True)
class SideSpec:
    """One side of a tile: what it presents and where a path may join."""

    sig: EdgeSig
    slots: int = ALL_SLOTS

    def __post_init__(self) -> None:
        if not 0 <= self.slots <= ALL_SLOTS:
            raise ValueError(f"slot mask out of range: {self.slots}")
        if self.sig is not EdgeSig.OPEN and self.slots != NO_SLOTS:
            # Only an open side can be joined, so keep the data honest.
            object.__setattr__(self, "slots", NO_SLOTS)

    def mirrored(self) -> "SideSpec":
        return SideSpec(self.sig, _reverse_slots(self.slots))


def sides_compatible(a: SideSpec, b: SideSpec) -> bool:
    """True when tile side ``a`` may abut neighbour side ``b``.

    Signatures must complement one another, and two open sides must also
    share at least one connection slot, otherwise the floor would meet a
    doorway that does not line up.
    """
    if SIG_COMPLEMENT[a.sig] is not b.sig:
        return False
    if a.sig is EdgeSig.OPEN:
        # b's slots are read from its own clockwise direction, which runs
        # opposite to a's along the shared edge, so reverse before masking.
        return bool(a.slots & _reverse_slots(b.slots))
    return True


# --------------------------------------------------------------------------
# Tiles and their transforms
# --------------------------------------------------------------------------


class TileClass(enum.Enum):
    DUNGEON = "dungeon"
    OUTDOOR = "outdoor"
    BOTH = "both"


@dataclass(frozen=True)
class Tile:
    """A 4 m square of greybox geometry.

    ``sides`` is in N, E, S, W order. ``walkable`` says whether an actor may
    stand on the tile at all; filler is never walkable.
    """

    id: int
    name: str
    tile_class: TileClass
    sides: Tuple[SideSpec, SideSpec, SideSpec, SideSpec]
    walkable: bool = True
    weight: int = 1
    hero: bool = False
    mesh: str = "greybox/plain"

    def transformed(self, rot: int, flip: bool) -> Tuple[SideSpec, ...]:
        """Return this tile's sides after a flip then a rotation.

        The flip mirrors across the vertical axis, which swaps east and west
        and reverses the slot order on every side. The rotation then advances
        each side clockwise by ``rot`` quarter turns.
        """
        s = self.sides
        if flip:
            s = (
                s[N].mirrored(),
                s[W].mirrored(),
                s[S].mirrored(),
                s[E].mirrored(),
            )
        rot &= 3
        if rot:
            s = tuple(s[(i - rot) % 4] for i in SIDES)
        return tuple(s)


@dataclass(frozen=True)
class Placement:
    """A tile as it sits in the grid."""

    tile_id: int
    rot: int = 0
    flip: bool = False

    def packed(self) -> Tuple[int, int]:
        """Two bytes: tile id, then rotation in bits 0-1 and flip in bit 2."""
        if not 0 <= self.tile_id <= 0xFF:
            raise ValueError(f"tile id does not fit in one byte: {self.tile_id}")
        return self.tile_id & 0xFF, (self.rot & 0b11) | (0b100 if self.flip else 0)

    @staticmethod
    def unpack(tile_byte: int, flags: int) -> "Placement":
        return Placement(tile_byte, flags & 0b11, bool(flags & 0b100))


# --------------------------------------------------------------------------
# Templates (stage 1)
# --------------------------------------------------------------------------


class Role(enum.Enum):
    ENTRANCE = "entrance"
    EXIT = "exit"
    BOSS = "boss"
    MECHANIC = "mechanic"
    SIDE = "side"
    CHECKPOINT = "checkpoint"


class Shape(enum.Enum):
    U = "U"
    C = "C"
    I = "I"
    DIAMOND = "Diamond"
    SPIRAL = "Spiral"
    HUB = "Hub"


#: Only these shapes have an entrance and exit that may trade places; the
#: others would become a different shape if their ends were swapped.
SWAPPABLE_SHAPES = frozenset({Shape.U, Shape.C, Shape.I})


@dataclass(frozen=True)
class TemplateNode:
    id: str
    role: Role
    anchor: str
    optional: bool = False
    set_piece: Optional[str] = None


@dataclass(frozen=True)
class TemplateEdge:
    a: str
    b: str
    optional: bool = False


@dataclass(frozen=True)
class SpawnRules:
    base_density: float
    elite_rate: float


@dataclass(frozen=True)
class GraphTemplate:
    id: str
    version: int
    tile_class: TileClass
    shape: Shape
    grid: int
    cell_m: float
    nodes: Tuple[TemplateNode, ...]
    edges: Tuple[TemplateEdge, ...]
    tileset: str
    landmarks: Tuple[str, ...] = ()
    spawn: SpawnRules = SpawnRules(0.018, 0.12)

    def node(self, node_id: str) -> TemplateNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    @property
    def ref(self) -> str:
        return f"{self.id}@{self.version}"


# --------------------------------------------------------------------------
# Stage 2 output
# --------------------------------------------------------------------------


@dataclass
class RoutedNode:
    id: str
    role: Role
    cell: Cell
    set_piece: Optional[str] = None


@dataclass
class RoutedEdge:
    a: str
    b: str
    path: List[Cell]


@dataclass
class RoutedLayout:
    """The abstract flow of a map: where the rooms are and how they connect."""

    seed: int
    template: GraphTemplate
    grid: int
    nodes: Dict[str, RoutedNode]
    edges: List[RoutedEdge]

    def node_of_role(self, role: Role) -> Optional[RoutedNode]:
        for n in self.nodes.values():
            if n.role is role:
                return n
        return None

    def incident_edges(self, node_id: str) -> List[RoutedEdge]:
        return [e for e in self.edges if e.a == node_id or e.b == node_id]


# --------------------------------------------------------------------------
# Stage 3 output
# --------------------------------------------------------------------------


class CellKind(enum.IntEnum):
    """What stage 3 decided a cell is, before any tile is chosen."""

    EMPTY = 0
    CORRIDOR = 1
    ROOM = 2
    SET_PIECE = 3
    APPROACH = 4  # boss approach floor family, see spec 02
    WATER = 5
    CLIFF = 6


@dataclass
class RoomPlacement:
    room_id: str
    node_id: str
    origin: Cell  # top-left cell
    w: int
    h: int
    rot: int = 0
    flip: bool = False

    def cells(self) -> List[Cell]:
        return [
            (self.origin[0] + dx, self.origin[1] + dy)
            for dy in range(self.h)
            for dx in range(self.w)
        ]


@dataclass
class TerrainPlan:
    """Stage 3: which cells are floor, and of what kind."""

    grid: int
    kinds: List[List[CellKind]]
    rooms: List[RoomPlacement] = field(default_factory=list)
    splines: List[List[Tuple[float, float]]] = field(default_factory=list)

    @staticmethod
    def blank(grid: int) -> "TerrainPlan":
        return TerrainPlan(
            grid=grid,
            kinds=[[CellKind.EMPTY] * grid for _ in range(grid)],
        )

    def kind(self, cell: Cell) -> CellKind:
        x, y = cell
        if not self.inside(cell):
            return CellKind.EMPTY
        return self.kinds[y][x]

    def set_kind(self, cell: Cell, kind: CellKind) -> None:
        x, y = cell
        if self.inside(cell):
            self.kinds[y][x] = kind

    def inside(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.grid and 0 <= y < self.grid

    def is_floor(self, cell: Cell) -> bool:
        return self.kind(cell) in (
            CellKind.CORRIDOR,
            CellKind.ROOM,
            CellKind.SET_PIECE,
            CellKind.APPROACH,
        )


# --------------------------------------------------------------------------
# Stage 4/5 output
# --------------------------------------------------------------------------


@dataclass
class TileGrid:
    """Stage 4: a concrete tile, rotation and flip for every cell."""

    grid: int
    cells: List[List[Optional[Placement]]]
    walkable: List[List[bool]]

    @staticmethod
    def blank(grid: int) -> "TileGrid":
        return TileGrid(
            grid=grid,
            cells=[[None] * grid for _ in range(grid)],
            walkable=[[False] * grid for _ in range(grid)],
        )

    def inside(self, cell: Cell) -> bool:
        x, y = cell
        return 0 <= x < self.grid and 0 <= y < self.grid

    def at(self, cell: Cell) -> Optional[Placement]:
        if not self.inside(cell):
            return None
        return self.cells[cell[1]][cell[0]]

    def put(self, cell: Cell, placement: Placement, walkable: bool) -> None:
        x, y = cell
        self.cells[y][x] = placement
        self.walkable[y][x] = walkable

    def is_walkable(self, cell: Cell) -> bool:
        if not self.inside(cell):
            return False
        return self.walkable[cell[1]][cell[0]]


@dataclass
class SetPiecePlacement:
    id: str
    cell: Cell  # top-left cell of the footprint
    rot: int = 0
    w: int = 1
    h: int = 1


# --------------------------------------------------------------------------
# Stage 6 output
# --------------------------------------------------------------------------


@dataclass
class SpawnPack:
    pack: str
    cell: Cell
    count: int
    elite: bool = False


@dataclass(frozen=True)
class MapMarker:
    """A named point of interest stage 5 stamped into the finished map.

    The thin ``SetPiecePlacement`` the client blob carries names only a piece,
    an origin and a rotation, so the *tells* a set piece plants -- the exit
    brazier, the landmark the seed's set-piece field chose out of the
    template's list -- had nowhere to live once stage 5 returned.  They were
    computed and dropped on the floor by :func:`pipeline.generate`, which made
    the landmark choice unobservable from the pipeline's own output.  This is
    where they live now.
    """

    kind: str
    cell: Cell
    name: str = ""
    piece_id: str = ""
    node_id: str = ""


@dataclass
class GeneratedMap:
    """Everything the six stages produce, before serialisation."""

    seed: int
    template: GraphTemplate
    tileset_ref: str
    routed: RoutedLayout
    terrain: TerrainPlan
    tiles: TileGrid
    set_pieces: List[SetPiecePlacement]
    spawns: List[SpawnPack]
    exit_cell: Cell
    checkpoints: List[Cell]
    markers: List[MapMarker] = field(default_factory=list)


__all__ = [
    "Cell", "N", "E", "S", "W", "SIDES", "SIDE_NAMES", "SIDE_DELTA",
    "OPPOSITE", "neighbour", "EdgeSig", "SIG_COMPLEMENT", "ALL_SLOTS",
    "NO_SLOTS", "SideSpec", "sides_compatible", "TileClass", "Tile",
    "Placement", "Role", "Shape", "SWAPPABLE_SHAPES", "TemplateNode",
    "TemplateEdge", "SpawnRules", "GraphTemplate", "RoutedNode",
    "RoutedEdge", "RoutedLayout", "CellKind", "RoomPlacement", "TerrainPlan",
    "TileGrid", "SetPiecePlacement", "SpawnPack", "MapMarker", "GeneratedMap",
]
