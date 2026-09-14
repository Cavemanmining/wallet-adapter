"""Stage 5: set pieces -- stamping fixed interiors over the tiled grid.

Spec: docs/WORLD_BIBLE.md stage 5, plus the two navigational tells of
section 02.

    Boss arena, exit checkpoint and mechanic room are fixed interiors; only
    the doorway moves, snapping so the entrance socket meets the routed edge
    reaching that node.  They overwrite overlapped tiles and re-run filler
    around themselves.  Enforce the tells: the exit sits within 15 m of a
    checkpoint brazier and a landmark, and the last 30 m before a boss uses
    the approach floor family with zero ambient density.

What "fixed interior" means here
--------------------------------
A set piece is a square block of cell kinds, drawn as rows of glyphs in
``data/setpieces_greybox.json``.  The glyphs are structural, not decorative:
``.`` is floor, ``#`` is solid, and ``D`` is the one entrance socket, which is
floor and must lie on the perimeter.  Stamping never edits the interior -- the
only freedom is where the block lands (its origin) and which quarter turn it
lands at.  Two maps that place ``crypt_boss_v2`` therefore contain the same
arena, cell for cell, which is the point of a set piece.

How a piece is snapped to its corridor
--------------------------------------
Every set-piece node is reached by one *arriving* edge: the incident routed
edge whose far end is closest to the entrance in the graph, so it is the
corridor the player walks in along.  Walk that edge's path from the node
outwards.  Each step gives a candidate anchor -- a corridor cell -- and the
travel direction ``t`` towards the node.  The door then sits one step further
along ``t``, so the socket must be on side ``OPPOSITE[t]``, which fixes the
rotation, which fixes the origin.  The first candidate that fits inside the
grid and clashes with nothing already stamped wins; failing that the piece is
nudged one cell further back along the path, and so on.  That is the whole of
"snapping so the entrance socket meets the routed edge", and it is why a
placement is a pure function of the routed layout.

Why solid cells never touch a piece's perimeter
-----------------------------------------------
A routed path may run along the side of a footprint, or cross it.  If the
perimeter could be solid, stamping could wall a corridor off and sever the
map.  The library therefore rejects a piece with a solid perimeter cell
(:func:`_validate_piece`), and the greybox pieces keep their pillars set in
from the walls.  Interior pillars are safe: floor still rings them, so the
footprint stays walkable end to end.

Randomness
----------
Stage 5 makes no random choices about *where* a piece goes -- the geometry
above decides that -- so placement is deterministic without drawing at all.
Re-tileizing the cells the stamp dirtied does draw, from a stream labelled
``"tile-setpiece"``, which ``seed.SeedFields.stream`` routes to the tile field
(bits 32-63), the same field stage 4 used.  Cells are re-tileized in row-major
order, so the result is a pure function of the seed, the plan and the tile
database.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from .contracts import (
    OPPOSITE,
    SIDE_DELTA,
    SIDE_NAMES,
    SIDES,
    Cell,
    CellKind,
    Placement,
    Role,
    RoutedEdge,
    RoutedLayout,
    RoutedNode,
    SetPiecePlacement,
    SideSpec,
    TerrainPlan,
    TileClass,
    TileGrid,
    neighbour,
    sides_compatible,
)
from .seed import SeedFields, Stream
from .tileize import (
    COLLISION_MARGIN_M,
    Box,
    FillerCell,
    SeamError,
    SeamMismatch,
    Surface,
    TileizedGrid,
    required_sides,
    surface_of,
)
from .tiles import TileDatabase

#: Where the checked-in greybox set pieces live.
DEFAULT_SETPIECE_DATA = Path(__file__).resolve().parent / "data" / "setpieces_greybox.json"

#: The stream stage 5 re-tileizes from.  The "tile" prefix matters:
#: ``SeedFields.stream`` routes it to the tile field, so stamping a set piece
#: cannot perturb routing or node jitter.
RETILE_STREAM_LABEL = "tile-setpiece"

#: Spec section 02: "the last 30 m before a boss uses the approach floor
#: family".  At 4 m per cell that is 7.5 cells, and half a cell of approach
#: floor is not a thing, so take the ceiling: 8 cells.
BOSS_APPROACH_M = 30.0
BOSS_APPROACH_CELLS = 8

#: Spec section 02: "the exit sits within 15 m of a checkpoint brazier and a
#: landmark".  Distances are between cell centres, in metres.
EXIT_TELL_RADIUS_M = 15.0

#: Marker kinds a piece may carry.  An exit piece must carry a brazier and a
#: landmark -- they are the tell spec section 02 demands -- and may name the
#: cell the player actually leaves through, which otherwise defaults to the
#: middle of the footprint.
MARKER_BRAZIER = "brazier"
MARKER_LANDMARK = "landmark"
MARKER_EXIT = "exit"

#: Which glyph in the JSON interior means which cell kind.  ``D`` is floor
#: like ``.``; it is spelled differently only so the socket is visible in the
#: data and can be cross-checked against the declared socket side and offset.
KIND_BY_GLYPH: Mapping[str, CellKind] = {
    ".": CellKind.SET_PIECE,
    "D": CellKind.SET_PIECE,
    "#": CellKind.EMPTY,
}
DOOR_GLYPH = "D"

#: Pieces are stamped in this order, most constrained first: the boss arena is
#: the largest footprint and the one whose approach has to be kept clear, so it
#: picks its ground before the smaller pieces have to work around it.  Nodes
#: with the same role are ordered by id, never by dict order.
ROLE_ORDER: Mapping[Role, int] = {
    Role.BOSS: 0,
    Role.EXIT: 1,
    Role.MECHANIC: 2,
    Role.CHECKPOINT: 3,
    Role.SIDE: 4,
    Role.ENTRANCE: 5,
}

#: Stage 2 keeps routing one cell in from the edge of the map.  A footprint is
#: allowed outside that band -- it only has to be inside the grid -- but a
#: candidate that stays inside it is preferred, so pieces do not press up
#: against the sealed border unless nothing else fits.
ROUTING_MARGIN = 1


class SetPieceError(RuntimeError):
    """Stage 5 was handed data or a layout it cannot honour."""


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def rotate_local(cell: Cell, n: int, rot: int) -> Cell:
    """Rotate a cell inside an ``n`` by ``n`` block by ``rot`` quarter turns.

    Clockwise, matching the rotation convention in ``contracts``: one step
    moves what was on side ``i`` to side ``(i + 1) % 4``.  North-west goes to
    north-east, and so on round.
    """
    x, y = cell
    rot &= 3
    if rot == 0:
        return (x, y)
    if rot == 1:
        return (n - 1 - y, x)
    if rot == 2:
        return (n - 1 - x, n - 1 - y)
    return (y, n - 1 - x)


def cells_apart_m(a: Cell, b: Cell, cell_m: float) -> float:
    """Distance between two cell centres, in metres."""
    return math.hypot(a[0] - b[0], a[1] - b[1]) * cell_m


def _direction(a: Cell, b: Cell) -> int:
    """The side of ``a`` that ``b`` lies across; ``b`` must be a neighbour."""
    step = (b[0] - a[0], b[1] - a[1])
    for side in SIDES:
        if SIDE_DELTA[side] == step:
            return side
    raise SetPieceError(f"{a} and {b} are not orthogonally adjacent")


# --------------------------------------------------------------------------
# The library
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Socket:
    """Where the routed edge enters a set piece: a side and an offset.

    The offset is read in the side's clockwise direction -- north west to
    east, east north to south, south east to west, west south to north -- the
    same convention connection slots use in ``contracts``.  This is the room
    library's ``DoorwaySocket`` idea, kept separate because a set piece turns
    (:meth:`SetPiece.socket_local` takes a rotation) while a room never does,
    and because stage 5 must work for outdoor templates, which have no room
    library at all.
    """

    side: int
    offset: int

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{SIDE_NAMES[self.side]}@{self.offset}"


@dataclass(frozen=True)
class SetPieceMarker:
    """A named point of interest inside a piece, in unrotated local cells."""

    kind: str
    cell: Cell


@dataclass(frozen=True)
class SetPiece:
    """One fixed interior: a square of cell kinds plus its entrance socket.

    ``kinds`` is indexed ``kinds[y][x]`` in unrotated local coordinates, with
    y running south and x east, the same as the grid.
    """

    id: str
    w: int
    h: int
    socket: Socket
    kinds: Tuple[Tuple[CellKind, ...], ...]
    markers: Tuple[SetPieceMarker, ...] = ()
    role: str = ""
    note: str = ""

    # -- shape -------------------------------------------------------------

    @property
    def size(self) -> int:
        """The side length; pieces are square so a turn keeps the footprint."""
        return self.w

    def kind_local(self, cell: Cell) -> CellKind:
        return self.kinds[cell[1]][cell[0]]

    def local_cells(self) -> List[Cell]:
        """Every local cell, row-major, so stamping order never varies."""
        return [(x, y) for y in range(self.h) for x in range(self.w)]

    def is_floor_local(self, cell: Cell) -> bool:
        return self.kind_local(cell) is not CellKind.EMPTY

    # -- the socket --------------------------------------------------------

    def socket_local_base(self) -> Cell:
        """The socket's cell in the unrotated block."""
        side, k = self.socket.side, self.socket.offset
        if side == 0:  # N: west to east along the top row
            return (k, 0)
        if side == 1:  # E: north to south down the right column
            return (self.w - 1, k)
        if side == 2:  # S: east to west along the bottom row
            return (self.w - 1 - k, self.h - 1)
        if side == 3:  # W: south to north up the left column
            return (0, self.h - 1 - k)
        raise SetPieceError(f"piece {self.id!r} has a socket on side {side}")

    def socket_side(self, rot: int) -> int:
        """Which side the socket faces once the piece is turned."""
        return (self.socket.side + (rot & 3)) % 4

    def socket_local(self, rot: int) -> Cell:
        """The socket's cell in the turned block."""
        return rotate_local(self.socket_local_base(), self.size, rot)

    def rotation_for_side(self, side: int) -> int:
        """The quarter turn that puts the socket on ``side``."""
        return (side - self.socket.side) % 4

    # -- placement ---------------------------------------------------------

    def origin_for_socket(self, socket_cell: Cell, rot: int) -> Cell:
        """The footprint's top-left cell that puts the socket on a given cell."""
        lx, ly = self.socket_local(rot)
        return (socket_cell[0] - lx, socket_cell[1] - ly)

    def footprint(self, origin: Cell) -> List[Cell]:
        """Every grid cell the piece covers, row-major."""
        ox, oy = origin
        return [(ox + x, oy + y) for y in range(self.h) for x in range(self.w)]

    def stamp_kinds(self, origin: Cell, rot: int) -> List[Tuple[Cell, CellKind]]:
        """``(grid cell, kind)`` for the whole interior at this placement.

        The interior itself never changes; the rotation only decides which
        grid cell each of its cells lands on.
        """
        ox, oy = origin
        n = self.size
        out: List[Tuple[Cell, CellKind]] = []
        for local in self.local_cells():
            rx, ry = rotate_local(local, n, rot)
            out.append(((ox + rx, oy + ry), self.kind_local(local)))
        out.sort(key=lambda item: (item[0][1], item[0][0]))
        return out

    def marker_cell(self, marker: SetPieceMarker, origin: Cell, rot: int) -> Cell:
        rx, ry = rotate_local(marker.cell, self.size, rot)
        return (origin[0] + rx, origin[1] + ry)

    def markers_of(self, kind: str) -> Tuple[SetPieceMarker, ...]:
        return tuple(m for m in self.markers if m.kind == kind)

    def carries(self, kind: str) -> bool:
        return any(m.kind == kind for m in self.markers)


def _parse_piece(raw: Mapping[str, object]) -> SetPiece:
    """Read one piece from JSON and check it before it can be placed."""
    piece_id = str(raw["id"])
    rows = list(raw["interior"])  # type: ignore[arg-type]
    w, h = int(raw["w"]), int(raw["h"])
    if len(rows) != h:
        raise SetPieceError(f"piece {piece_id!r} declares h={h} but drew {len(rows)} rows")
    kinds: List[Tuple[CellKind, ...]] = []
    door_cells: List[Cell] = []
    for y, row in enumerate(rows):
        text = str(row)
        if len(text) != w:
            raise SetPieceError(
                f"piece {piece_id!r} row {y} is {len(text)} cells wide, expected {w}"
            )
        line: List[CellKind] = []
        for x, glyph in enumerate(text):
            if glyph not in KIND_BY_GLYPH:
                raise SetPieceError(
                    f"piece {piece_id!r} row {y} column {x} uses unknown glyph {glyph!r}"
                )
            if glyph == DOOR_GLYPH:
                door_cells.append((x, y))
            line.append(KIND_BY_GLYPH[glyph])
        kinds.append(tuple(line))

    socket_raw = raw["socket"]
    if isinstance(socket_raw, Mapping):
        socket = Socket(int(socket_raw["side"]), int(socket_raw["offset"]))
    else:
        side, offset = socket_raw  # type: ignore[misc]
        socket = Socket(int(side), int(offset))

    markers = tuple(
        SetPieceMarker(str(m["kind"]), (int(m["at"][0]), int(m["at"][1])))  # type: ignore[index]
        for m in raw.get("markers", ())  # type: ignore[union-attr]
    )

    piece = SetPiece(
        id=piece_id,
        w=w,
        h=h,
        socket=socket,
        kinds=tuple(kinds),
        markers=markers,
        role=str(raw.get("role", "")),
        note=str(raw.get("note", "")),
    )
    _validate_piece(piece, door_cells)
    return piece


def _validate_piece(piece: SetPiece, door_cells: Sequence[Cell]) -> None:
    """Every rule a piece must keep for stamping to be safe.

    Square, because a quarter turn must not change the footprint.  Exactly one
    door glyph, sitting where the declared socket says and on the perimeter,
    because the socket is what the corridor meets.  No solid cell on the
    perimeter, so stamping can never wall off a corridor running past the
    footprint (see the module docstring).  Floor 4-connected, so no part of the
    interior is unreachable.  Markers on floor and inside the block.
    """
    where = f"set piece {piece.id!r}"
    if piece.w != piece.h:
        raise SetPieceError(f"{where} is {piece.w}x{piece.h}; pieces must be square")
    if piece.w < 2:
        raise SetPieceError(f"{where} is too small to have an interior and a socket")
    if piece.socket.side not in SIDES:
        raise SetPieceError(f"{where} has a socket on side {piece.socket.side}")
    span = piece.w if piece.socket.side in (0, 2) else piece.h
    if not 0 <= piece.socket.offset < span:
        raise SetPieceError(
            f"{where} has socket offset {piece.socket.offset} off a side of {span} cells"
        )

    declared = piece.socket_local_base()
    if len(door_cells) != 1:
        raise SetPieceError(f"{where} draws {len(door_cells)} door glyphs, expected one")
    if door_cells[0] != declared:
        raise SetPieceError(
            f"{where} draws its door at {door_cells[0]} but declares socket "
            f"{piece.socket}, which is cell {declared}"
        )
    if not piece.is_floor_local(declared):
        raise SetPieceError(f"{where} has a solid entrance socket")

    for cell in piece.local_cells():
        x, y = cell
        on_perimeter = x in (0, piece.w - 1) or y in (0, piece.h - 1)
        if on_perimeter and not piece.is_floor_local(cell):
            raise SetPieceError(
                f"{where} puts a solid cell at {cell} on its perimeter; a solid "
                "perimeter could wall off a corridor running past the footprint"
            )

    floors = [c for c in piece.local_cells() if piece.is_floor_local(c)]
    if not floors:
        raise SetPieceError(f"{where} has no floor at all")
    seen = {floors[0]}
    queue = deque([floors[0]])
    while queue:
        cell = queue.popleft()
        for side in SIDES:
            nxt = neighbour(cell, side)
            if (
                0 <= nxt[0] < piece.w
                and 0 <= nxt[1] < piece.h
                and nxt not in seen
                and piece.is_floor_local(nxt)
            ):
                seen.add(nxt)
                queue.append(nxt)
    if len(seen) != len(floors):
        stranded = sorted(set(floors) - seen)
        raise SetPieceError(f"{where} has floor nothing can reach: {stranded}")

    for marker in piece.markers:
        mx, my = marker.cell
        if not (0 <= mx < piece.w and 0 <= my < piece.h):
            raise SetPieceError(f"{where} puts marker {marker.kind!r} outside itself")
        if not piece.is_floor_local(marker.cell):
            raise SetPieceError(f"{where} puts marker {marker.kind!r} on solid ground")
    if piece.role == "exit":
        for kind in (MARKER_BRAZIER, MARKER_LANDMARK):
            if not piece.carries(kind):
                raise SetPieceError(f"{where} is an exit piece with no {kind} marker")


class SetPieceLibrary:
    """The set pieces stage 5 may stamp, indexed by id."""

    def __init__(
        self,
        pieces: Iterable[SetPiece],
        *,
        name: str = "greybox_setpieces",
        revision: int = 1,
        cell_m: float = 4.0,
    ) -> None:
        self.pieces: Tuple[SetPiece, ...] = tuple(sorted(pieces, key=lambda p: p.id))
        if not self.pieces:
            raise SetPieceError("a set piece library needs at least one piece")
        self.name = name
        self.revision = int(revision)
        self.cell_m = float(cell_m)
        self._by_id: Dict[str, SetPiece] = {}
        for piece in self.pieces:
            if piece.id in self._by_id:
                raise SetPieceError(f"duplicate set piece id {piece.id!r}")
            self._by_id[piece.id] = piece

    @staticmethod
    def load(path: Optional[Path] = None) -> "SetPieceLibrary":
        """Load a library from JSON; defaults to the checked-in greybox set."""
        path = Path(path) if path is not None else DEFAULT_SETPIECE_DATA
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return SetPieceLibrary(
            [_parse_piece(raw) for raw in doc["pieces"]],
            name=str(doc.get("id", "greybox_setpieces")),
            revision=int(doc.get("version", 1)),
            cell_m=float(doc.get("cell_m", 4.0)),
        )

    @property
    def version(self) -> str:
        """The reference stage 6 may hash, e.g. ``greybox_setpieces@1``."""
        return f"{self.name}@{self.revision}"

    @property
    def ref(self) -> str:
        return self.version

    def __len__(self) -> int:
        return len(self.pieces)

    def __contains__(self, piece_id: object) -> bool:
        return piece_id in self._by_id

    def ids(self) -> Tuple[str, ...]:
        return tuple(p.id for p in self.pieces)

    def by_id(self, piece_id: str) -> SetPiece:
        try:
            return self._by_id[piece_id]
        except KeyError:
            raise SetPieceError(
                f"no set piece {piece_id!r} in {self.version}; have {', '.join(self.ids())}"
            ) from None

    def get(self, piece_id: str) -> Optional[SetPiece]:
        return self._by_id.get(piece_id)


_DEFAULT_LIBRARY: Optional[SetPieceLibrary] = None


def default_library() -> SetPieceLibrary:
    """The checked-in greybox library, parsed once and shared."""
    global _DEFAULT_LIBRARY
    if _DEFAULT_LIBRARY is None:
        _DEFAULT_LIBRARY = SetPieceLibrary.load()
    return _DEFAULT_LIBRARY


# --------------------------------------------------------------------------
# What stage 5 returns
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PlacedMarker:
    """One marker as it sits in the grid."""

    kind: str
    cell: Cell
    piece_id: str
    node_id: str
    name: str = ""


@dataclass(frozen=True)
class PlacementDetail:
    """Everything stage 5 decided about one piece, for stages 6 and for tests.

    ``SetPiecePlacement`` in ``contracts`` is deliberately thin -- the client
    only needs id, origin, rotation and size -- so the reasoning behind the
    placement lives here rather than being thrown away.
    """

    node_id: str
    piece_id: str
    role: Role
    placement: SetPiecePlacement
    origin: Cell
    rot: int
    socket_cell: Cell
    socket_side: int
    anchor_cell: Cell  # the corridor cell the socket opens onto
    edge: Tuple[str, str]  # the arriving routed edge
    footprint: Tuple[Cell, ...]
    markers: Tuple[PlacedMarker, ...] = ()
    nudged: int = 0  # cells stepped back along the arriving path
    clamped: bool = False

    @property
    def centre(self) -> Cell:
        """The middle cell of the footprint; odd sizes make this exact."""
        return (
            self.origin[0] + (self.placement.w - 1) // 2,
            self.origin[1] + (self.placement.h - 1) // 2,
        )


class SetPieceResult(List[SetPiecePlacement]):
    """Stage 5's answer: the placements, and everything they imply.

    It *is* the ``list[SetPiecePlacement]`` the stage promises, so a caller
    that only wants placements can ignore the rest, while the checkpoint cells,
    the markers and the boss approach -- which stage 6 and the client layout
    description both need -- still have somewhere honest to live.
    """

    def __init__(self, placements: Iterable[SetPiecePlacement] = ()) -> None:
        super().__init__(placements)
        self.details: List[PlacementDetail] = []
        self.markers: List[PlacedMarker] = []
        self.checkpoints: List[Cell] = []
        self.approach_cells: List[Cell] = []
        self.retileized: List[Cell] = []
        self.exit_cell: Optional[Cell] = None
        self.skipped: List[Tuple[str, str]] = []  # (node id, unknown piece id)
        self.notes: List[str] = []

    # -- lookups -----------------------------------------------------------

    def detail_of_node(self, node_id: str) -> Optional[PlacementDetail]:
        for detail in self.details:
            if detail.node_id == node_id:
                return detail
        return None

    def detail_of_role(self, role: Role) -> Optional[PlacementDetail]:
        for detail in self.details:
            if detail.role is role:
                return detail
        return None

    def markers_of(self, kind: str) -> List[PlacedMarker]:
        return [m for m in self.markers if m.kind == kind]

    def footprint_cells(self) -> FrozenSet[Cell]:
        return frozenset(c for d in self.details for c in d.footprint)

    def ambient_exclusions(self) -> FrozenSet[Cell]:
        """Cells stage 6 must not drop an ambient pack on.

        The spec only demands it of the boss approach ("zero ambient
        density"); the set-piece interiors are added because a fixed interior
        is authored content, and a wandering pack inside the boss arena or on
        the exit platform is exactly what a set piece exists to prevent.
        """
        return self.footprint_cells() | frozenset(self.approach_cells)


# --------------------------------------------------------------------------
# Reading the routed layout
# --------------------------------------------------------------------------


def _as_fields(seed: Union[int, SeedFields]) -> SeedFields:
    return seed if isinstance(seed, SeedFields) else SeedFields.parse(int(seed))


def _graph_distances(routed: RoutedLayout) -> Dict[str, int]:
    """Hops from the entrance to every node, over the edges that survived.

    Neighbours are visited in sorted order, so the answer never depends on
    dict or set iteration order.
    """
    entrance = routed.node_of_role(Role.ENTRANCE)
    adjacency: Dict[str, List[str]] = {node_id: [] for node_id in routed.nodes}
    for edge in routed.edges:
        if edge.a in adjacency and edge.b in adjacency:
            adjacency[edge.a].append(edge.b)
            adjacency[edge.b].append(edge.a)
    for neighbours in adjacency.values():
        neighbours.sort()

    start = entrance.id if entrance is not None else min(routed.nodes)
    dist = {start: 0}
    queue = deque([start])
    while queue:
        here = queue.popleft()
        for there in adjacency.get(here, ()):
            if there not in dist:
                dist[there] = dist[here] + 1
                queue.append(there)
    return dist


def arriving_edge(
    routed: RoutedLayout, node_id: str, dist: Optional[Mapping[str, int]] = None
) -> Optional[RoutedEdge]:
    """The edge the player arrives by: the incident edge nearest the entrance.

    Ties -- two neighbours the same number of hops away -- are broken by the
    shorter routed path and then by the neighbour's id, so the choice is a
    pure function of the layout.
    """
    if dist is None:
        dist = _graph_distances(routed)
    incident = routed.incident_edges(node_id)
    if not incident:
        return None
    far = len(routed.nodes) + 1

    def key(edge: RoutedEdge) -> Tuple[int, int, str]:
        other = edge.b if edge.a == node_id else edge.a
        return (dist.get(other, far), len(edge.path), other)

    return min(incident, key=key)


def _path_towards(edge: RoutedEdge, node_id: str) -> List[Cell]:
    """The edge's path oriented so it ends at ``node_id``'s cell."""
    if edge.b == node_id:
        return list(edge.path)
    return list(reversed(edge.path))


# --------------------------------------------------------------------------
# Choosing a placement
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Candidate:
    """One way to snap a piece: an anchor on the path and what follows from it."""

    anchor: Cell
    socket_cell: Cell
    rot: int
    origin: Cell
    footprint: Tuple[Cell, ...]
    nudged: int


def _candidates(piece: SetPiece, path: Sequence[Cell]) -> List[_Candidate]:
    """Every snap of ``piece`` onto ``path``, nearest the node first.

    Walking the path backwards from the node, each step gives an anchor (a
    corridor cell) and the travel direction towards the node.  The socket sits
    one cell further along that direction, so it faces back down the corridor,
    which is what "the entrance socket meets the routed edge" means.  Stepping
    further back is the deterministic nudge the spec asks for when a placement
    would not fit or would collide.
    """
    out: List[_Candidate] = []
    for nudged, index in enumerate(range(len(path) - 2, -1, -1)):
        anchor = path[index]
        ahead = path[index + 1]
        if anchor == ahead:
            continue  # a degenerate path should not stop stage 5
        try:
            travel = _direction(anchor, ahead)
        except SetPieceError:
            continue
        rot = piece.rotation_for_side(OPPOSITE[travel])
        origin = piece.origin_for_socket(ahead, rot)
        out.append(
            _Candidate(
                anchor=anchor,
                socket_cell=ahead,
                rot=rot,
                origin=origin,
                footprint=tuple(piece.footprint(origin)),
                nudged=nudged,
            )
        )
    return out


def _inside(footprint: Sequence[Cell], grid: int, margin: int = 0) -> bool:
    lo, hi = margin, grid - 1 - margin
    return all(lo <= x <= hi and lo <= y <= hi for x, y in footprint)


def _choose_candidate(
    candidates: Sequence[_Candidate],
    plan: TerrainPlan,
    reserved: Set[Cell],
    node_cells: Set[Cell],
) -> Optional[_Candidate]:
    """Pick the first candidate that holds, in three passes of falling strictness.

    Hard rules, never relaxed: the footprint is inside the grid, it touches
    nothing already stamped, and the anchor it opens onto is floor and outside
    the footprint.  The two softenings are preferences -- stay inside the
    routing margin, and do not swallow *another* node's cell -- which a cramped
    corner of the map may make impossible.  The piece's own node is not in
    ``node_cells``: standing on it is the whole point of the placement.

    Returns the first candidate that holds, or ``None`` when even the hard
    rules cannot be met on this path.
    """

    def hard_ok(cand: _Candidate) -> bool:
        if not _inside(cand.footprint, plan.grid):
            return False
        cells = set(cand.footprint)
        if cells & reserved:
            return False
        if cand.anchor in cells:
            return False
        return plan.is_floor(cand.anchor)

    passes = (
        lambda c: hard_ok(c)
        and _inside(c.footprint, plan.grid, ROUTING_MARGIN)
        and not (set(c.footprint) & node_cells),
        lambda c: hard_ok(c) and not (set(c.footprint) & node_cells),
        hard_ok,
    )
    for predicate in passes:
        for cand in candidates:
            if predicate(cand):
                return cand
    return None


def _centred_fallback(piece: SetPiece, cell: Cell, plan: TerrainPlan) -> _Candidate:
    """The piece centred on one cell, shoved inside the grid.

    For the pathological case of an arriving edge with no direction to it --
    two nodes on the same cell -- where there is no corridor to snap to.
    """
    half = (piece.size - 1) // 2
    origin = (cell[0] - half, cell[1] - half)
    return _clamped_fallback(
        piece,
        _Candidate(
            anchor=cell,
            socket_cell=cell,
            rot=0,
            origin=origin,
            footprint=tuple(piece.footprint(origin)),
            nudged=0,
        ),
        plan,
    )


def _clamped_fallback(
    piece: SetPiece, cand: _Candidate, plan: TerrainPlan
) -> _Candidate:
    """Last resort: shove a candidate inside the grid, socket be damned.

    Only reached when no anchor on the whole arriving path yields a legal
    placement -- a map so cramped that the alternative is dropping the set
    piece entirely.  The caller records it on the placement as ``clamped`` so
    the breach is visible rather than silent.
    """
    n = piece.size
    ox = min(max(cand.origin[0], 0), plan.grid - n)
    oy = min(max(cand.origin[1], 0), plan.grid - n)
    origin = (ox, oy)
    socket_local = piece.socket_local(cand.rot)
    return _Candidate(
        anchor=cand.anchor,
        socket_cell=(origin[0] + socket_local[0], origin[1] + socket_local[1]),
        rot=cand.rot,
        origin=origin,
        footprint=tuple(piece.footprint(origin)),
        nudged=cand.nudged,
    )


# --------------------------------------------------------------------------
# The boss approach
# --------------------------------------------------------------------------


def approach_cell_count(cell_m: float) -> int:
    """How many cells of corridor make up the 30 m boss approach.

    30 m is 7.5 cells at 4 m per cell, so use the ceiling: 8 cells.
    """
    if cell_m <= 0:
        raise SetPieceError(f"cell_m must be positive, got {cell_m}")
    return int(math.ceil(BOSS_APPROACH_M / cell_m))


def _approach_cells(
    plan: TerrainPlan,
    path: Sequence[Cell],
    anchor: Cell,
    blocked: Set[Cell],
    wanted: int,
) -> List[Cell]:
    """The last ``wanted`` cells of corridor before a boss, nearest the door first.

    Walks back down the arriving path from the anchor, which is the corridor
    the player actually comes in along.  A short path -- the arena may have
    swallowed most of it -- is topped up by a breadth-first walk over floor
    outwards from the cells already chosen, so the count the spec asks for is
    met whenever that much floor exists.  Both walks skip cells inside a
    stamped footprint: a set piece's interior is fixed and must not be
    repainted as approach floor.
    """
    chosen: List[Cell] = []
    seen: Set[Cell] = set()

    def take(cell: Cell) -> bool:
        if cell in seen or cell in blocked or not plan.is_floor(cell):
            return False
        seen.add(cell)
        chosen.append(cell)
        return len(chosen) >= wanted

    index = list(path).index(anchor) if anchor in path else len(path) - 1
    for cell in reversed(list(path)[: index + 1]):
        if take(cell):
            return chosen

    # Top up outwards from what we have, nearest first, ties by cell so the
    # answer never depends on iteration order.
    frontier = deque(chosen)
    while frontier and len(chosen) < wanted:
        cell = frontier.popleft()
        for side in SIDES:
            nxt = neighbour(cell, side)
            if nxt in seen or nxt in blocked or not plan.is_floor(nxt):
                continue
            if take(nxt):
                return chosen
            frontier.append(nxt)
    return chosen


# --------------------------------------------------------------------------
# Re-tileizing what the stamp dirtied
# --------------------------------------------------------------------------


def _dirty_halo(cells: Iterable[Cell], grid: int) -> Set[Cell]:
    """The cells whose tile may have to change when ``cells`` change kind.

    A tile's requirement is decided by its own kind and its four neighbours'
    kinds (see ``tileize.required_sides``), so the halo is exactly the changed
    cells plus their orthogonal neighbours -- no more, no less.
    """
    dirty: Set[Cell] = set()
    for cell in cells:
        if 0 <= cell[0] < grid and 0 <= cell[1] < grid:
            dirty.add(cell)
            for side in SIDES:
                nxt = neighbour(cell, side)
                if 0 <= nxt[0] < grid and 0 <= nxt[1] < grid:
                    dirty.add(nxt)
    return dirty


def _retileize(
    plan: TerrainPlan,
    grid: TileGrid,
    tiles: TileDatabase,
    tile_class: TileClass,
    stream: Stream,
    cells: Iterable[Cell],
) -> List[Cell]:
    """Re-pick the tile for every cell in ``cells``, in row-major order.

    This is stage 4's inner loop run again over a patch: untouched cells get
    the sealed filler, everything else gets a tile that meets the requirement
    its new neighbourhood imposes.  Hero variants are excluded, because the
    hero budget was spent in stage 4 and re-rolling a patch must not be able to
    push the map over "at most one per 20 cells"; excluding them can only lower
    the count.
    """
    touched = sorted(set(cells), key=lambda c: (c[1], c[0]))
    filler = tiles.filler_placement()
    for cell in touched:
        if not plan.inside(cell):
            continue
        here = surface_of(plan, cell, tile_class)
        if here is Surface.VOID:
            grid.put(cell, filler, walkable=False)
            continue
        want = required_sides(plan, cell, tile_class, here)
        placement = tiles.find(want, tile_class, stream, allow_hero=False)
        grid.put(cell, placement, walkable=here is Surface.FLOOR)
    return touched


def _update_bookkeeping(
    grid: TileGrid, tiles: TileDatabase, cells: Sequence[Cell]
) -> None:
    """Refresh a :class:`TileizedGrid`'s filler, hero and matched records.

    Stamping turns floor into filler and filler into floor, so the summaries
    stage 4 built are stale for exactly the cells that were re-tileized -- and
    for no others, which is why this walks ``cells`` rather than the grid.  The
    filler and hero lists are re-emitted row-major, the order stage 4 uses, so
    two runs compare element for element.  A plain ``TileGrid`` carries none of
    these, so there is nothing to do for one.
    """
    if not isinstance(grid, TileizedGrid):
        return
    margin = float(
        tiles.meta(tiles.filler_tile_id).get("collision_margin_m", COLLISION_MARGIN_M)
    )
    filler: Dict[Cell, FillerCell] = {f.cell: f for f in grid.filler}
    heroes: Set[Cell] = set(grid.hero_cells)
    for cell in cells:
        placement = grid.at(cell)
        if placement is None:  # pragma: no cover - every cell is placed
            continue
        if placement.tile_id == tiles.filler_tile_id:
            mesh = Box.of_cell(cell, grid.cell_m)
            filler[cell] = FillerCell(
                cell=cell, mesh=mesh, hull=mesh.grown(margin), margin_m=margin
            )
            heroes.discard(cell)
            continue
        filler.pop(cell, None)
        if tiles.by_id(placement.tile_id).hero:
            heroes.add(cell)
        else:
            heroes.discard(cell)

    row_major = lambda cell: (cell[1], cell[0])  # noqa: E731
    grid.filler = tuple(filler[c] for c in sorted(filler, key=row_major))
    grid.hero_cells = tuple(sorted(heroes, key=row_major))
    # Filler is the one tile stage 4 never matches a cell to, so everything
    # else in the grid is a matched cell.
    grid.matched_cells = grid.grid * grid.grid - len(grid.filler)


def assert_local_seams(
    grid: TileGrid, tiles: TileDatabase, cells: Iterable[Cell]
) -> None:
    """Raise :class:`SeamError` unless every seam touching ``cells`` fits.

    Stage 4 checks the whole grid; stage 5 only rewrote a patch, so only the
    seams on that patch's boundary -- and inside it -- can have broken.  Each
    of the four sides of each cell is tested against what the neighbour
    actually presents, re-derived from the placed geometry rather than from
    the requirement the tile was chosen against.
    """
    resolved: Dict[Placement, Tuple[SideSpec, ...]] = {}

    def sides_of(placement: Placement) -> Tuple[SideSpec, ...]:
        hit = resolved.get(placement)
        if hit is None:
            hit = tiles.sides_of(placement)
            resolved[placement] = hit
        return hit

    bad: List[SeamMismatch] = []
    for cell in cells:
        here = grid.at(cell)
        if here is None:
            continue
        for side in SIDES:
            there = neighbour(cell, side)
            other = grid.at(there)
            if other is None:
                continue
            mine = sides_of(here)[side]
            theirs = sides_of(other)[OPPOSITE[side]]
            if not sides_compatible(mine, theirs):
                bad.append(SeamMismatch(cell, side, there, mine, theirs))
    if bad:
        raise SeamError(bad)


# --------------------------------------------------------------------------
# Stage 5 proper
# --------------------------------------------------------------------------


def place_set_pieces(
    routed: RoutedLayout,
    plan: TerrainPlan,
    tile_grid: Optional[TileGrid],
    tiles: TileDatabase,
    seed: Union[int, SeedFields],
    *,
    library: Optional[SetPieceLibrary] = None,
    tile_class: Optional[TileClass] = None,
    strict: bool = False,
    verify: bool = True,
) -> SetPieceResult:
    """Stage 5: stamp every node's set piece, then enforce the two tells.

    Spec: docs/WORLD_BIBLE.md stage 5 and section 02.

    For each node carrying a ``set_piece``, in boss-then-exit-then-mechanic
    order, this

    * finds the routed edge the player arrives by,
    * turns the piece so its entrance socket faces back down that corridor,
    * snaps the footprint so the socket sits on the corridor's last cell,
      nudging back along the path if that would leave the grid or touch
      another set piece,
    * stamps the fixed interior over ``plan``, floor cells becoming
      ``CellKind.SET_PIECE`` and solid cells ``CellKind.EMPTY``,
    * and re-tileizes the footprint and its halo, so the filler is re-run
      around the piece and every seam still holds.

    Then the tells: the exit piece's brazier and landmark are placed and
    checked against the 15 m radius, and the 8 cells (30 m) of corridor before
    the boss are re-marked ``CellKind.APPROACH`` for stage 6 to leave empty.

    ``plan`` and ``tile_grid`` are modified in place; ``tile_grid`` may be
    ``None`` when a caller only wants the plan updated.  ``strict`` turns an
    unknown piece id from a recorded skip into an error.  ``verify`` re-checks
    the seams stamping could have broken, the same guard stage 4 runs over the
    whole grid.
    """
    fields = _as_fields(seed)
    library = library if library is not None else default_library()
    tile_class = tile_class if tile_class is not None else routed.template.tile_class
    cell_m = float(routed.template.cell_m)
    result = SetPieceResult()

    dist = _graph_distances(routed)
    node_cells = {n.cell for n in routed.nodes.values()}
    reserved: Set[Cell] = set()
    dirty: Set[Cell] = set()

    ordered = sorted(
        (n for n in routed.nodes.values() if n.set_piece),
        key=lambda n: (ROLE_ORDER.get(n.role, len(ROLE_ORDER)), n.id),
    )

    for node in ordered:
        piece = library.get(str(node.set_piece))
        if piece is None:
            message = (
                f"node {node.id!r} asks for set piece {node.set_piece!r}, "
                f"which {library.version} does not define"
            )
            if strict:
                raise SetPieceError(message)
            result.skipped.append((node.id, str(node.set_piece)))
            result.notes.append(message)
            continue

        detail = _place_one(
            node=node,
            piece=piece,
            routed=routed,
            plan=plan,
            dist=dist,
            reserved=reserved,
            node_cells=node_cells,
            fields=fields,
        )
        if detail is None:
            result.notes.append(f"node {node.id!r} has no routed edge to arrive by")
            continue

        # Stamp the fixed interior, then reserve the ground so a later piece
        # cannot land on it and the anchor stays floor for good.
        for cell, kind in piece.stamp_kinds(detail.origin, detail.rot):
            plan.set_kind(cell, kind)
        reserved.update(detail.footprint)
        reserved.add(detail.anchor_cell)
        dirty.update(detail.footprint)

        result.append(detail.placement)
        result.details.append(detail)
        result.markers.extend(detail.markers)
        if detail.clamped:
            result.notes.append(
                f"set piece {piece.id!r} at node {node.id!r} had to be clamped "
                "inside the grid; its socket may not meet the corridor"
            )

    _enforce_exit_tell(routed, result, cell_m)
    dirty |= _enforce_boss_approach(routed, plan, result, dist, cell_m)
    result.checkpoints = _checkpoint_cells(routed, result)

    if tile_grid is not None and dirty:
        stream = fields.stream(RETILE_STREAM_LABEL)
        result.retileized = _retileize(
            plan, tile_grid, tiles, tile_class, stream, _dirty_halo(dirty, plan.grid)
        )
        _update_bookkeeping(tile_grid, tiles, result.retileized)
        if verify:
            assert_local_seams(tile_grid, tiles, result.retileized)

    return result


def _place_one(
    *,
    node: RoutedNode,
    piece: SetPiece,
    routed: RoutedLayout,
    plan: TerrainPlan,
    dist: Mapping[str, int],
    reserved: Set[Cell],
    node_cells: Set[Cell],
    fields: SeedFields,
) -> Optional[PlacementDetail]:
    """Work out where and how one piece lands, without touching the plan yet."""
    edge = arriving_edge(routed, node.id, dist)
    if edge is None:
        return None
    # The piece's own node is the one cell it is *meant* to stand on, so only
    # the other nodes count as ground to keep clear.
    others = {cell for cell in node_cells if cell != node.cell}
    candidates = _candidates(piece, _path_towards(edge, node.id))
    chosen = _choose_candidate(candidates, plan, reserved, others) if candidates else None
    if chosen is not None:
        candidate, clamped = chosen, False
    elif candidates:
        # Nothing on the whole path fits: keep the piece, keep it in the grid,
        # and let the caller say so.
        candidate, clamped = _clamped_fallback(piece, candidates[0], plan), True
    else:
        candidate, clamped = _centred_fallback(piece, node.cell, plan), True

    placement = SetPiecePlacement(
        id=piece.id,
        cell=candidate.origin,
        rot=candidate.rot,
        w=piece.w,
        h=piece.h,
    )
    markers = tuple(
        PlacedMarker(
            kind=marker.kind,
            cell=piece.marker_cell(marker, candidate.origin, candidate.rot),
            piece_id=piece.id,
            node_id=node.id,
            name=_marker_name(marker, routed, fields),
        )
        for marker in piece.markers
    )
    return PlacementDetail(
        node_id=node.id,
        piece_id=piece.id,
        role=node.role,
        placement=placement,
        origin=candidate.origin,
        rot=candidate.rot,
        socket_cell=candidate.socket_cell,
        socket_side=piece.socket_side(candidate.rot),
        anchor_cell=candidate.anchor,
        edge=(edge.a, edge.b),
        footprint=candidate.footprint,
        markers=markers,
        nudged=candidate.nudged,
        clamped=clamped,
    )


def _marker_name(
    marker: SetPieceMarker, routed: RoutedLayout, fields: SeedFields
) -> str:
    """Name a marker, drawing a landmark from the template's list.

    Judgement call: the spec says the exit needs "a landmark" without saying
    which, and the template lists the landmarks the art pass has built.  Bits
    4-7 of the seed are the set-piece field, so they choose, which keeps the
    landmark stable for a seed and independent of routing and tiles.
    """
    if marker.kind != MARKER_LANDMARK:
        return marker.kind
    landmarks = tuple(routed.template.landmarks)
    if not landmarks:
        return MARKER_LANDMARK
    return landmarks[fields.set_piece_choice % len(landmarks)]


def _enforce_exit_tell(
    routed: RoutedLayout, result: SetPieceResult, cell_m: float
) -> None:
    """Spec 02: the exit sits within 15 m of a brazier and of a landmark.

    The markers ride inside the exit piece, so the radius holds by
    construction; this re-measures it from the placed cells anyway, because a
    tell nobody checks is a tell that quietly stops being true when the piece
    is redrawn.
    """
    detail = result.detail_of_role(Role.EXIT)
    if detail is None:
        exit_node = routed.node_of_role(Role.EXIT)
        if exit_node is not None:
            result.exit_cell = exit_node.cell
            result.notes.append(
                f"exit node {exit_node.id!r} carries no set piece, so it has no "
                "brazier or landmark of its own"
            )
        return

    named = [m for m in detail.markers if m.kind == MARKER_EXIT]
    result.exit_cell = named[0].cell if named else detail.centre
    for kind in (MARKER_BRAZIER, MARKER_LANDMARK):
        placed = [m for m in detail.markers if m.kind == kind]
        if not placed:
            raise SetPieceError(
                f"exit set piece {detail.piece_id!r} carries no {kind} marker, so "
                "the exit tell cannot be met"
            )
        nearest = min(cells_apart_m(result.exit_cell, m.cell, cell_m) for m in placed)
        if nearest > EXIT_TELL_RADIUS_M:
            raise SetPieceError(
                f"exit at {result.exit_cell} is {nearest:.1f} m from its nearest "
                f"{kind}, past the {EXIT_TELL_RADIUS_M:.0f} m tell"
            )


def _enforce_boss_approach(
    routed: RoutedLayout,
    plan: TerrainPlan,
    result: SetPieceResult,
    dist: Mapping[str, int],
    cell_m: float,
) -> Set[Cell]:
    """Spec 02: re-mark the last 30 m of corridor before the boss.

    Returns the cells changed, so the caller can re-tileize them.  They keep
    their floor status -- ``CellKind.APPROACH`` is floor to
    ``TerrainPlan.is_floor`` -- so the marking never breaks a route; what it
    carries is the instruction to stage 6 to leave the ground empty.
    """
    detail = result.detail_of_role(Role.BOSS)
    if detail is None:
        return set()
    edge = arriving_edge(routed, detail.node_id, dist)
    if edge is None:  # pragma: no cover - a placed piece always had an edge
        return set()

    wanted = approach_cell_count(cell_m)
    cells = _approach_cells(
        plan=plan,
        path=_path_towards(edge, detail.node_id),
        anchor=detail.anchor_cell,
        blocked=set(result.footprint_cells()),
        wanted=wanted,
    )
    for cell in cells:
        plan.set_kind(cell, CellKind.APPROACH)
    result.approach_cells = list(cells)
    if len(cells) < wanted:
        result.notes.append(
            f"only {len(cells)} of {wanted} approach cells before the boss: the "
            "corridor reaching it is shorter than 30 m"
        )
    return set(cells)


def _checkpoint_cells(routed: RoutedLayout, result: SetPieceResult) -> List[Cell]:
    """Where the runtime may check a player in.

    Every brazier a set piece placed, in the order the pieces were stamped,
    followed by the cells of any node the template gave the checkpoint role.
    Duplicates are dropped, order is never taken from a set.
    """
    cells: List[Cell] = []
    seen: Set[Cell] = set()
    for marker in result.markers:
        if marker.kind == MARKER_BRAZIER and marker.cell not in seen:
            seen.add(marker.cell)
            cells.append(marker.cell)
    for node in sorted(routed.nodes.values(), key=lambda n: n.id):
        if node.role is Role.CHECKPOINT and node.cell not in seen:
            seen.add(node.cell)
            cells.append(node.cell)
    return cells


# --------------------------------------------------------------------------
# Re-derived checks (what the tests and a debug build lean on)
# --------------------------------------------------------------------------


def check_placements(
    routed: RoutedLayout,
    plan: TerrainPlan,
    result: SetPieceResult,
    library: Optional[SetPieceLibrary] = None,
    *,
    cell_m: Optional[float] = None,
) -> List[str]:
    """Re-derive every stage 5 invariant from the output and list the breaks.

    Returns a list of human-readable problems, empty when the stamping is
    sound.  Checked here rather than trusted: footprints inside the grid and
    disjoint, each socket on the perimeter and opening onto a floor cell of its
    own arriving edge, the interior stamped exactly as the library draws it,
    the exit tells inside 15 m, and the boss approach the right length and
    floor.
    """
    library = library if library is not None else default_library()
    cell_m = float(cell_m if cell_m is not None else routed.template.cell_m)
    problems: List[str] = []
    seen: Dict[Cell, str] = {}

    for detail in result.details:
        piece = library.by_id(detail.piece_id)
        where = f"{detail.piece_id} at node {detail.node_id}"

        if not _inside(detail.footprint, plan.grid):
            problems.append(f"{where} is not fully inside the {plan.grid}-cell grid")
        for cell in detail.footprint:
            if cell in seen:
                problems.append(f"{where} overlaps {seen[cell]} at {cell}")
            seen[cell] = where

        # The socket sits on the footprint's perimeter, facing the corridor.
        expected = piece.socket_local(detail.rot)
        want = (detail.origin[0] + expected[0], detail.origin[1] + expected[1])
        if want != detail.socket_cell:
            problems.append(f"{where} records socket {detail.socket_cell}, expected {want}")
        outward = neighbour(detail.socket_cell, detail.socket_side)
        if outward != detail.anchor_cell:
            problems.append(
                f"{where} opens onto {outward} but anchors on {detail.anchor_cell}"
            )
        if not plan.is_floor(detail.anchor_cell):
            problems.append(f"{where} opens onto {detail.anchor_cell}, which is not floor")
        if detail.anchor_cell in detail.footprint:
            problems.append(f"{where} opens onto its own footprint at {detail.anchor_cell}")

        edge = next(
            (
                e
                for e in routed.edges
                if (e.a, e.b) == detail.edge or (e.b, e.a) == detail.edge
            ),
            None,
        )
        if edge is None:
            problems.append(f"{where} names an arriving edge that is not routed")
        elif detail.anchor_cell not in edge.path and not detail.clamped:
            problems.append(
                f"{where} anchors on {detail.anchor_cell}, which is not on its "
                f"arriving edge {detail.edge[0]}-{detail.edge[1]}"
            )

        # The interior is fixed: every cell must read back as the library draws it.
        for cell, kind in piece.stamp_kinds(detail.origin, detail.rot):
            if plan.kind(cell) is kind:
                continue
            if kind is CellKind.SET_PIECE and plan.kind(cell) is CellKind.APPROACH:
                problems.append(f"{where} had {cell} repainted as approach floor")
            else:
                problems.append(
                    f"{where} should have {kind.name} at {cell}, plan has "
                    f"{plan.kind(cell).name}"
                )

        for marker in detail.markers:
            if marker.cell not in detail.footprint:
                problems.append(f"{where} put its {marker.kind} outside the footprint")

    # --- the exit tell ---------------------------------------------------
    exit_detail = result.detail_of_role(Role.EXIT)
    if exit_detail is not None:
        if result.exit_cell is None:
            problems.append("an exit set piece was placed but no exit cell was recorded")
        else:
            for kind in (MARKER_BRAZIER, MARKER_LANDMARK):
                placed = [m for m in result.markers if m.kind == kind]
                if not placed:
                    problems.append(f"no {kind} was placed for the exit")
                    continue
                nearest = min(cells_apart_m(result.exit_cell, m.cell, cell_m) for m in placed)
                if nearest > EXIT_TELL_RADIUS_M:
                    problems.append(
                        f"the nearest {kind} is {nearest:.1f} m from the exit, past "
                        f"the {EXIT_TELL_RADIUS_M:.0f} m tell"
                    )

    for cell in result.checkpoints:
        if not plan.is_floor(cell):
            problems.append(f"checkpoint {cell} is not on floor")

    # --- the boss approach ------------------------------------------------
    boss_detail = result.detail_of_role(Role.BOSS)
    marked = [
        (x, y)
        for y in range(plan.grid)
        for x in range(plan.grid)
        if plan.kind((x, y)) is CellKind.APPROACH
    ]
    if boss_detail is None:
        if marked:
            problems.append(f"{len(marked)} approach cells marked with no boss placed")
    else:
        wanted = approach_cell_count(cell_m)
        if len(marked) != wanted:
            problems.append(
                f"{len(marked)} cells of approach floor before the boss, expected {wanted}"
            )
        if sorted(marked) != sorted(result.approach_cells):
            problems.append("the approach cells in the plan are not the ones reported")
        footprints = result.footprint_cells()
        for cell in result.approach_cells:
            if cell in footprints:
                problems.append(f"approach cell {cell} is inside a set piece")

    return problems


__all__ = [
    "BOSS_APPROACH_CELLS",
    "BOSS_APPROACH_M",
    "DEFAULT_SETPIECE_DATA",
    "EXIT_TELL_RADIUS_M",
    "KIND_BY_GLYPH",
    "MARKER_BRAZIER",
    "MARKER_EXIT",
    "MARKER_LANDMARK",
    "RETILE_STREAM_LABEL",
    "ROLE_ORDER",
    "PlacedMarker",
    "PlacementDetail",
    "SetPiece",
    "SetPieceError",
    "SetPieceLibrary",
    "SetPieceMarker",
    "SetPieceResult",
    "Socket",
    "approach_cell_count",
    "arriving_edge",
    "assert_local_seams",
    "cells_apart_m",
    "check_placements",
    "default_library",
    "place_set_pieces",
    "rotate_local",
]
