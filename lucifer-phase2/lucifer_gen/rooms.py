"""The greybox room library used by stage 3 for the dungeon class.

Spec: docs/WORLD_BIBLE.md stage 3, "Terrain".

Stage 3 turns each routed node into a room.  It knows which directions the
incident edges arrive from, and asks this module for rooms whose doorway
sockets face those directions.  **Rooms are never scaled**, so the library
carries enough shapes that one always fits rather than stretching one that
does not: there is a room for each of the fifteen non-empty subsets of
{N, E, S, W}, and the smallest of them is 2x2, so even a cramped node has a
candidate.

Doorway sockets
---------------
A socket is a side plus an offset along that side.  Offsets run in the same
clockwise direction as the connection slots in ``contracts``: north runs west
to east, east north to south, south east to west, west south to north.  Offset
0 is therefore the first cell in that direction, which means a socket keeps its
meaning if the room is ever mirrored or turned, even though this library never
does either.

Placement
---------
:meth:`RoomLibrary.place` centres a room's footprint on the node cell and
clamps it inside the grid, returning a
:class:`~lucifer_gen.contracts.RoomPlacement` with ``rot=0`` and ``flip=False``.
Clamping shifts the room; it never resizes it.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple

from .contracts import SIDE_NAMES, SIDES, Cell, RoomPlacement
from .seed import Stream

#: Where the checked-in greybox room library lives.
DEFAULT_ROOM_DATA = Path(__file__).resolve().parent / "data" / "rooms_greybox.json"


class NoFittingRoom(LookupError):
    """Raised when no room in the library can serve a request."""


@dataclass(frozen=True)
class DoorwaySocket:
    """Where an edge may enter a room: a side and an offset along it."""

    side: int
    offset: int

    def __str__(self) -> str:  # pragma: no cover - debugging aid
        return f"{SIDE_NAMES[self.side]}@{self.offset}"


@dataclass(frozen=True)
class Room:
    """One greybox room: a fixed footprint in cells and its doorways."""

    id: str
    w: int
    h: int
    sockets: Tuple[DoorwaySocket, ...]
    note: str = ""

    @property
    def sides(self) -> FrozenSet[int]:
        """Which sides carry at least one doorway."""
        return frozenset(s.side for s in self.sockets)

    @property
    def area(self) -> int:
        return self.w * self.h

    def side_length(self, side: int) -> int:
        """How many cells long a side is; north and south run along ``w``."""
        return self.w if side in (0, 2) else self.h

    def sockets_on(self, side: int) -> Tuple[DoorwaySocket, ...]:
        return tuple(s for s in self.sockets if s.side == side)

    def serves(self, required_sides: Iterable[int]) -> bool:
        """True when every required direction has a doorway."""
        return set(required_sides) <= self.sides

    def socket_cell(self, origin: Cell, socket: DoorwaySocket) -> Cell:
        """The grid cell a doorway occupies, for a footprint at ``origin``.

        Offsets are read in each side's clockwise direction, which is why the
        south and west cases count backwards.
        """
        ox, oy = origin
        side, k = socket.side, socket.offset
        if side == 0:  # N, west to east along the top row
            return (ox + k, oy)
        if side == 1:  # E, north to south down the right column
            return (ox + self.w - 1, oy + k)
        if side == 2:  # S, east to west along the bottom row
            return (ox + self.w - 1 - k, oy + self.h - 1)
        if side == 3:  # W, south to north up the left column
            return (ox, oy + self.h - 1 - k)
        raise ValueError(f"not a side: {side}")

    def doorways(self, origin: Cell) -> List[Tuple[int, Cell]]:
        """Every doorway as ``(side, cell)`` for a footprint at ``origin``."""
        return [(s.side, self.socket_cell(origin, s)) for s in self.sockets]


def _parse_room(raw: dict) -> Room:
    sockets = []
    for entry in raw["sockets"]:
        if isinstance(entry, dict):
            side, offset = int(entry["side"]), int(entry["offset"])
        else:
            side, offset = int(entry[0]), int(entry[1])
        sockets.append(DoorwaySocket(side, offset))
    room = Room(
        id=str(raw["id"]),
        w=int(raw["w"]),
        h=int(raw["h"]),
        sockets=tuple(sorted(sockets, key=lambda s: (s.side, s.offset))),
        note=str(raw.get("note", "")),
    )
    _validate(room)
    return room


def _validate(room: Room) -> None:
    if room.w < 1 or room.h < 1:
        raise ValueError(f"room {room.id!r} has a degenerate footprint")
    if not room.sockets:
        raise ValueError(f"room {room.id!r} has no doorway, nothing could reach it")
    seen = set()
    for socket in room.sockets:
        if socket.side not in SIDES:
            raise ValueError(f"room {room.id!r} has a socket on side {socket.side}")
        length = room.side_length(socket.side)
        if not 0 <= socket.offset < length:
            raise ValueError(
                f"room {room.id!r} socket {socket} falls off a side {length} cells long"
            )
        if (socket.side, socket.offset) in seen:
            raise ValueError(f"room {room.id!r} has two doorways at {socket}")
        seen.add((socket.side, socket.offset))


class RoomLibrary:
    """The rooms stage 3 may place, with the queries stage 3 actually makes."""

    def __init__(
        self,
        rooms: Iterable[Room],
        *,
        name: str = "greybox_rooms",
        revision: int = 1,
    ) -> None:
        # Ordered smallest first, then by id: a stable order that also happens
        # to be the order stage 3 wants to consider things in.
        self.rooms: Tuple[Room, ...] = tuple(
            sorted(rooms, key=lambda r: (r.area, r.w, r.h, r.id))
        )
        if not self.rooms:
            raise ValueError("a room library needs at least one room")
        self.name = name
        self.revision = int(revision)
        self._by_id: Dict[str, Room] = {}
        for room in self.rooms:
            if room.id in self._by_id:
                raise ValueError(f"duplicate room id {room.id!r}")
            self._by_id[room.id] = room

    @staticmethod
    def load(path: Optional[Path] = None) -> "RoomLibrary":
        """Load a library from JSON; defaults to the checked-in greybox set."""
        path = Path(path) if path is not None else DEFAULT_ROOM_DATA
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        return RoomLibrary(
            [_parse_room(raw) for raw in doc["rooms"]],
            name=str(doc.get("id", "greybox_rooms")),
            revision=int(doc.get("version", 1)),
        )

    @property
    def version(self) -> str:
        """e.g. ``greybox_rooms@1``, matching how templates spell a ref."""
        return f"{self.name}@{self.revision}"

    @property
    def ref(self) -> str:
        return self.version

    def __len__(self) -> int:
        return len(self.rooms)

    def by_id(self, room_id: str) -> Room:
        return self._by_id[room_id]

    @property
    def max_w(self) -> int:
        return max(r.w for r in self.rooms)

    @property
    def max_h(self) -> int:
        return max(r.h for r in self.rooms)

    # -- queries -----------------------------------------------------------

    def candidates(
        self,
        required_sides: Set[int],
        w_max: Optional[int] = None,
        h_max: Optional[int] = None,
        *,
        exact: bool = False,
    ) -> List[Room]:
        """Rooms with a doorway on every required side that fit in the budget.

        ``w_max``/``h_max`` cap the footprint; ``None`` means no cap.  By
        default a room may have spare doorways on sides nothing arrives at,
        which is what keeps the library total -- the all-sides 2x2 chamber
        answers every request.  Pass ``exact=True`` to demand the socket set
        equal the required set, which avoids doorways opening onto rock but
        can come back empty, so treat it as a preference and fall back.

        The result is ordered smallest footprint first, ties broken by id, so
        it is stable across runs.
        """
        wanted = set(required_sides)
        for side in wanted:
            if side not in SIDES:
                raise ValueError(f"not a side: {side}")
        out = []
        for room in self.rooms:
            if w_max is not None and room.w > w_max:
                continue
            if h_max is not None and room.h > h_max:
                continue
            if exact:
                if room.sides != wanted:
                    continue
            elif not wanted <= room.sides:
                continue
            out.append(room)
        return out

    def choose(
        self,
        required_sides: Set[int],
        stream: Stream,
        w_max: Optional[int] = None,
        h_max: Optional[int] = None,
        *,
        prefer_exact: bool = True,
    ) -> Room:
        """Pick one fitting room deterministically from ``stream``.

        Rooms whose doorways match the incident edges exactly are preferred,
        because a doorway opening onto nothing reads as a bug; when there are
        none, any room that covers the required sides will do.
        """
        pool: List[Room] = []
        if prefer_exact:
            pool = self.candidates(required_sides, w_max, h_max, exact=True)
        if not pool:
            pool = self.candidates(required_sides, w_max, h_max)
        if not pool:
            names = ",".join(SIDE_NAMES[s] for s in sorted(required_sides)) or "-"
            raise NoFittingRoom(
                f"no room has doorways on {names} within {w_max}x{h_max}"
            )
        return stream.choice(pool)

    # -- placement ---------------------------------------------------------

    def place(
        self,
        room: Room,
        node_id: str,
        node_cell: Cell,
        grid: int,
    ) -> RoomPlacement:
        """Centre ``room`` on ``node_cell`` and clamp it inside the grid.

        The node cell sits at footprint index ``(w - 1) // 2`` across and
        ``(h - 1) // 2`` down, so an odd room is centred exactly and an even
        one sits half a cell north-west of centre.  If the footprint would
        overhang the grid it slides back in; it is never scaled, which is the
        whole point of carrying this many shapes.
        """
        if room.w > grid or room.h > grid:
            raise ValueError(
                f"room {room.id!r} is {room.w}x{room.h}, larger than the {grid} grid"
            )
        nx, ny = node_cell
        ox = nx - (room.w - 1) // 2
        oy = ny - (room.h - 1) // 2
        ox = max(0, min(ox, grid - room.w))
        oy = max(0, min(oy, grid - room.h))
        return RoomPlacement(
            room_id=room.id,
            node_id=node_id,
            origin=(ox, oy),
            w=room.w,
            h=room.h,
            rot=0,
            flip=False,
        )

    def doorways(self, placement: RoomPlacement) -> List[Tuple[int, Cell]]:
        """Every doorway of a placed room as ``(side, cell)``."""
        return self._by_id[placement.room_id].doorways(placement.origin)


def prove_side_coverage(
    library: RoomLibrary,
    w_max: Optional[int] = None,
    h_max: Optional[int] = None,
) -> int:
    """Check every non-empty subset of sides can be served; return the count.

    Stage 3 must always be able to place something, so this is the room-side
    twin of ``tiles.prove_coverage``.  Raises :class:`NoFittingRoom` naming the
    first subset that comes back empty.
    """
    proved = 0
    for size in (1, 2, 3, 4):
        for subset in itertools.combinations(SIDES, size):
            if not library.candidates(set(subset), w_max, h_max):
                names = ",".join(SIDE_NAMES[s] for s in subset)
                raise NoFittingRoom(f"no room serves {names} within {w_max}x{h_max}")
            proved += 1
    return proved


__all__ = [
    "DEFAULT_ROOM_DATA",
    "DoorwaySocket",
    "NoFittingRoom",
    "Room",
    "RoomLibrary",
    "prove_side_coverage",
]
