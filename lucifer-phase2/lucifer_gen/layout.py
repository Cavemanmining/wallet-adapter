"""Stage 6, second half: the client layout description and its hash.

Spec: docs/WORLD_BIBLE.md stage 6, "Spawn and sync" -- "hash the layout,
seed, template version and tile database version into ``layout_hash``" -- and
the client layout description fields:

    seed, template (id@version), tiles (id@version), layout_hash, grid,
    cells (base64, 2 bytes per cell: tile_id then rot in bits 0-1 and flip in
    bit 2), set_pieces, spawns, exit_cell, checkpoints

The description is the wire format between the generator and the client, so
two things matter more than anything else here:

**It round-trips exactly.**  :func:`parse_layout_description` returns the very
:class:`~lucifer_gen.contracts.Placement` list that was packed, so a client
that rebuilds the grid can be checked against the server's, cell for cell.
The packing itself is ``Placement.packed`` / ``Placement.unpack`` from
contracts; this module only lays those bytes out row-major and base64s them.

**The same map hashes to the same string everywhere.**  ``layout_hash`` is
always ``blake2b``, which ships with Python, and it still carries the
``"blake2b:<hex>"`` prefix so the algorithm is named rather than assumed.

It used to prefer the optional ``blake3`` wheel and fall back to blake2b,
arguing that naming the algorithm made the difference "diagnosable".  It made
it *visible*, not harmless: ``layout_hash`` is the one value a client uses to
decide "am I looking at the same map as the server", and that answer became a
function of which wheels happened to be installed on each machine.  A build
box with the wheel and a designer's laptop without it produced two different
hashes for the same bytes, and ``verify_layout_description`` reported a
mismatch where there was no layout divergence at all.  ``validate.suite_seed``
already made this call for the gate's seed sweep, for the same reason; this is
the same rule applied to the value that actually ships.  (An optional wheel is
still welcome to accelerate anything that is not an identity.)

Each hashed field is length-prefixed (see :func:`_framed`), so no two
different field sets can concatenate to the same bytes: ``template="a@1",
tiles="b@2"`` cannot collide with ``template="a@1b", tiles="@2"``.

Hash inputs
-----------
The packed cells, the seed, the template ref and the tile database ref, in
that order, under the version tag :data:`HASH_DOMAIN`.  Nothing else: not the
spawns, not the set piece list.  That is the spec's list, and it is also the
useful one -- the hash answers "is the client's tile grid the one this seed
and this content build produce", and spawns are derived from those same
inputs rather than being independent state.  The grid size needs no separate
entry because it is exactly ``len(cells) // 2`` under a square grid, and the
length is framed.

Judgement calls the spec did not settle
---------------------------------------
* Cells are serialised in **row-major** order (all of row y=0 west to east,
  then y=1, ...), matching ``TileGrid.cells`` and stage 4's emission order.
* ``seed`` is emitted as ``0x``-prefixed uppercase hex, 16 digits, via
  ``seed.format_seed``; the spec asks for the prefix and the case and this
  keeps the width stable too.
* ``tiles`` is taken from ``GeneratedMap.tileset_ref`` -- stage 4 sets it from
  ``TileDatabase.version``.  The template's ``tileset`` field names an art
  set, which is a different thing and is not what the hash wants.
* Cells are emitted as two-element ``[x, y]`` lists, the JSON-friendly form of
  the ``(x, y)`` tuples used in code.
* ``verify_layout_description`` re-packs from the *parsed* placements rather
  than re-hashing the description's own base64 string, so it proves the round
  trip and the hash in one pass.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from typing import Any, Dict, List, Mapping, Sequence, Tuple, Union

from .contracts import (
    GeneratedMap,
    Placement,
    SetPiecePlacement,
    SpawnPack,
    TileGrid,
)
from .seed import format_seed, parse_seed

#: The one hash algorithm this module may use.  blake2b ships with CPython, so
#: the value a client compares against never depends on an optional wheel.
HASH_ALGORITHM = "blake2b"

#: Domain tag, bumped if the hashed field set ever changes.
HASH_DOMAIN = b"lucifer/layout_hash/1"

#: Digest length in bytes for both algorithms, so the hex is the same width.
HASH_SIZE = 32

#: The fields a description carries, in the order the spec lists them.
DESCRIPTION_FIELDS: Tuple[str, ...] = (
    "seed",
    "template",
    "tiles",
    "layout_hash",
    "grid",
    "cells",
    "set_pieces",
    "spawns",
    "exit_cell",
    "checkpoints",
)

#: Bytes per cell in the packed blob: tile id, then flags.
BYTES_PER_CELL = 2


class LayoutError(ValueError):
    """Raised when a map or a description cannot be serialised or read back."""


class LayoutMismatch(LayoutError):
    """Raised when a description's ``layout_hash`` does not match its cells."""


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def hash_algorithm() -> str:
    """The algorithm every ``layout_hash`` names: always ``"blake2b"``.

    Deliberately not a lookup of what is installed.  See the module docstring:
    a map's identity must not change because a host has or lacks a wheel.
    """
    return HASH_ALGORITHM


def _framed(*chunks: bytes) -> bytes:
    """Concatenate chunks, each prefixed with its 8-byte big-endian length.

    Framing is what makes the hash unambiguous: without it, moving a character
    from the end of one field to the start of the next would not change the
    hashed bytes.
    """
    out = bytearray()
    for chunk in chunks:
        out += len(chunk).to_bytes(8, "big")
        out += chunk
    return bytes(out)


def _digest(payload: bytes) -> str:
    return (
        f"{HASH_ALGORITHM}:"
        + hashlib.blake2b(payload, digest_size=HASH_SIZE).hexdigest()
    )


def layout_hash(
    cells: bytes,
    seed: Union[int, str],
    template: str,
    tiles: str,
) -> str:
    """Hash the packed cells, seed, template ref and tile database ref.

    Spec stage 6: "hash the layout, seed, template version and tile database
    version into ``layout_hash``".

    ``cells`` is the packed blob -- two bytes per cell, row-major -- *before*
    base64, so the hash is over the layout itself rather than over an encoding
    of it.  ``template`` and ``tiles`` are ``id@version`` refs.  The returned
    string is prefixed with the algorithm used, always ``"blake2b:9f2c..."``.
    """
    if not isinstance(cells, (bytes, bytearray, memoryview)):
        raise LayoutError(f"cells must be bytes, got {type(cells).__name__}")
    payload = _framed(
        HASH_DOMAIN,
        bytes(cells),
        format_seed(parse_seed(seed)).encode("utf-8"),
        str(template).encode("utf-8"),
        str(tiles).encode("utf-8"),
    )
    return _digest(payload)


# --------------------------------------------------------------------------
# Cell packing
# --------------------------------------------------------------------------


def pack_cells(tile_grid: TileGrid) -> bytes:
    """Two bytes per cell, row-major: tile id, then rot in bits 0-1, flip in 2.

    Exactly ``Placement.packed`` per cell, laid out in the order
    ``TileGrid.cells`` stores them.  An unfilled cell is an error: stage 4
    promises a placement everywhere, filler included.
    """
    grid = tile_grid.grid
    if grid <= 0:
        raise LayoutError(f"a layout needs at least one cell, got grid={grid}")
    if len(tile_grid.cells) != grid:
        raise LayoutError(
            f"tile grid claims grid={grid} but holds {len(tile_grid.cells)} rows"
        )
    out = bytearray()
    for y, row in enumerate(tile_grid.cells):
        if len(row) != grid:
            raise LayoutError(f"row {y} has {len(row)} cells, expected {grid}")
        for x, placement in enumerate(row):
            if placement is None:
                raise LayoutError(f"cell ({x}, {y}) has no tile placed")
            tile_byte, flags = placement.packed()
            out.append(tile_byte)
            out.append(flags)
    return bytes(out)


def unpack_cells(blob: bytes, grid: int) -> List[Placement]:
    """Read a packed blob back into ``grid * grid`` placements, row-major."""
    if grid <= 0:
        raise LayoutError(f"a layout needs at least one cell, got grid={grid}")
    expected = BYTES_PER_CELL * grid * grid
    if len(blob) != expected:
        raise LayoutError(
            f"cells blob is {len(blob)} bytes, expected {expected} for a "
            f"{grid}x{grid} grid"
        )
    return [
        Placement.unpack(blob[i], blob[i + 1])
        for i in range(0, expected, BYTES_PER_CELL)
    ]


def _encode_cells(blob: bytes) -> str:
    return base64.b64encode(blob).decode("ascii")


def _decode_cells(text: Any) -> bytes:
    if not isinstance(text, str):
        raise LayoutError(f"'cells' must be a base64 string, got {type(text).__name__}")
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise LayoutError(f"'cells' is not valid base64: {exc}") from exc


# --------------------------------------------------------------------------
# Building the description
# --------------------------------------------------------------------------


def _cell_pair(cell: Sequence[int], what: str) -> List[int]:
    try:
        x, y = cell
    except (TypeError, ValueError) as exc:
        raise LayoutError(f"{what} must be an (x, y) cell, got {cell!r}") from exc
    return [int(x), int(y)]


def _set_piece_dict(piece: SetPiecePlacement) -> Dict[str, Any]:
    return {
        "id": str(piece.id),
        "cell": _cell_pair(piece.cell, "set piece cell"),
        "rot": int(piece.rot) & 0b11,
        "w": int(piece.w),
        "h": int(piece.h),
    }


def _spawn_dict(pack: SpawnPack) -> Dict[str, Any]:
    return {
        "pack": str(pack.pack),
        "cell": _cell_pair(pack.cell, "spawn cell"),
        "count": int(pack.count),
        "elite": bool(pack.elite),
    }


def build_layout_description(generated_map: GeneratedMap) -> Dict[str, Any]:
    """Stage 6: the client layout description, as a JSON-ready dict.

    Spec: docs/WORLD_BIBLE.md stage 6.  The keys are exactly the spec's ten
    fields, inserted in the spec's order so a dumped description reads the way
    the document does.

    The ``layout_hash`` is computed here from the packed cells, the seed, the
    template ref and the tile database ref, so the description always carries
    a hash of itself rather than one passed in from elsewhere.
    """
    tiles_grid = generated_map.tiles
    blob = pack_cells(tiles_grid)
    grid = tiles_grid.grid

    terrain = getattr(generated_map, "terrain", None)
    if terrain is not None and terrain.grid != grid:
        raise LayoutError(
            f"tile grid is {grid}x{grid} but the terrain plan is "
            f"{terrain.grid}x{terrain.grid}"
        )

    seed = parse_seed(generated_map.seed)
    template_ref = str(generated_map.template.ref)
    tiles_ref = str(generated_map.tileset_ref or "")
    if not tiles_ref:
        raise LayoutError("the map has no tileset ref to hash or report")

    return {
        "seed": format_seed(seed),
        "template": template_ref,
        "tiles": tiles_ref,
        "layout_hash": layout_hash(blob, seed, template_ref, tiles_ref),
        "grid": grid,
        "cells": _encode_cells(blob),
        "set_pieces": [_set_piece_dict(p) for p in generated_map.set_pieces],
        "spawns": [_spawn_dict(s) for s in generated_map.spawns],
        "exit_cell": _cell_pair(generated_map.exit_cell, "exit_cell"),
        "checkpoints": [
            _cell_pair(c, "checkpoint") for c in generated_map.checkpoints
        ],
    }


# --------------------------------------------------------------------------
# Reading it back
# --------------------------------------------------------------------------


def _require(description: Mapping[str, Any], key: str) -> Any:
    if key not in description:
        raise LayoutError(f"description is missing {key!r}")
    return description[key]


def _grid_of(description: Mapping[str, Any]) -> int:
    grid = _require(description, "grid")
    if isinstance(grid, bool) or not isinstance(grid, int):
        raise LayoutError(f"'grid' must be an int, got {grid!r}")
    if grid <= 0:
        raise LayoutError(f"'grid' must be positive, got {grid}")
    return grid


def parse_layout_description(description: Mapping[str, Any]) -> List[Placement]:
    """Read a description's ``cells`` back into placements, row-major.

    The inverse of the ``cells`` half of :func:`build_layout_description`:
    ``grid * grid`` placements, index ``y * grid + x``.  This is what lets a
    client rebuild be compared against the generator's own grid.
    """
    if not isinstance(description, Mapping):
        raise LayoutError(
            f"description must be a mapping, got {type(description).__name__}"
        )
    grid = _grid_of(description)
    return unpack_cells(_decode_cells(_require(description, "cells")), grid)


def recompute_layout_hash(description: Mapping[str, Any]) -> str:
    """The hash a description *should* carry, derived from its parsed cells.

    Deliberately re-packs from :func:`parse_layout_description` rather than
    hashing the base64 string as received: a description whose blob decodes to
    the same placements but whose bytes differ (an alternative base64 spelling,
    say) must still verify, and a round trip that loses information must not.
    """
    out = bytearray()
    for placement in parse_layout_description(description):
        tile_byte, flags = placement.packed()
        out.append(tile_byte)
        out.append(flags)
    return layout_hash(
        bytes(out),
        parse_seed(_require(description, "seed")),
        str(_require(description, "template")),
        str(_require(description, "tiles")),
    )


def verify_layout_description(description: Mapping[str, Any]) -> bool:
    """True when the description's ``layout_hash`` matches its own contents.

    Recomputes the hash from the parsed cells, seed, template ref and tile
    database ref and compares, algorithm prefix included.  Because the
    algorithm is fixed (see the module docstring), a description that fails to
    verify says something about the *layout*, never about the host that
    produced it.
    """
    stored = _require(description, "layout_hash")
    return isinstance(stored, str) and stored == recompute_layout_hash(description)


def assert_layout_description(description: Mapping[str, Any]) -> None:
    """Raise :class:`LayoutMismatch` unless the description verifies."""
    stored = _require(description, "layout_hash")
    expected = recompute_layout_hash(description)
    if stored != expected:
        raise LayoutMismatch(
            f"layout_hash mismatch: description says {stored!r}, "
            f"its contents hash to {expected!r}"
        )


__all__ = [
    "BYTES_PER_CELL",
    "DESCRIPTION_FIELDS",
    "HASH_ALGORITHM",
    "HASH_DOMAIN",
    "HASH_SIZE",
    "LayoutError",
    "LayoutMismatch",
    "assert_layout_description",
    "build_layout_description",
    "hash_algorithm",
    "layout_hash",
    "pack_cells",
    "parse_layout_description",
    "recompute_layout_hash",
    "unpack_cells",
    "verify_layout_description",
]
