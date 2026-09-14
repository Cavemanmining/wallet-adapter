"""A minimal but correct PNG writer, standard library only.

Spec: docs/WORLD_BIBLE.md -- this module is not one of the six generation
stages.  It is the output plumbing underneath :mod:`lucifer_gen.render`,
which draws the debug map images the tools and the CLI emit.

Only :mod:`zlib` and :mod:`struct` are used, so the generator never grows a
dependency on an imaging library just to dump a picture.

What is written
---------------
An 8-bit truecolour (RGB, colour type 2), non-interlaced PNG:

    signature | IHDR | IDAT | IEND

Every scanline is prefixed with filter byte 0 (``None``).  Filtering would
compress a little better, but the images are flat blocks of colour that zlib
already handles well, and an unfiltered stream is trivially verifiable: a
reader can pull pixels straight out of the inflated bytes.  Each chunk
carries its own CRC-32 over the type tag and the data, per the PNG spec.

Determinism
-----------
The byte stream depends only on the pixels and the compression level: no
timestamps, no text chunks, no adaptive filtering.  Rendering the same map
twice therefore produces identical files, which is what the render tests
assert.

:func:`decode_png` reads back what this module writes (and any other 8-bit
truecolour PNG, including filtered ones).  It exists so callers and tests can
round-trip without an image library present.
"""

from __future__ import annotations

import struct
import zlib
from typing import Iterator, List, Sequence, Tuple, Union

__all__ = [
    "PNG_SIGNATURE",
    "PngError",
    "encode_png",
    "write_png",
    "iter_chunks",
    "decode_png",
]

#: The eight bytes every PNG file starts with.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

BIT_DEPTH = 8
COLOUR_TYPE_RGB = 2
CHANNELS = 3
FILTER_NONE = 0

#: zlib level used for IDAT.  Fixed so output is reproducible.
DEFAULT_LEVEL = 9

RGB = Tuple[int, int, int]
Row = Union[bytes, bytearray, memoryview, Sequence[RGB], Sequence[int]]
Rows = Union[bytes, bytearray, memoryview, Sequence[Row]]


class PngError(ValueError):
    """Raised for malformed pixel input or malformed PNG bytes."""


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


def _chunk(tag: bytes, data: bytes) -> bytes:
    """One PNG chunk: length, type, data, CRC-32 of type plus data."""
    if len(tag) != 4:
        raise PngError(f"chunk type must be four bytes, got {tag!r}")
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def _pixel_bytes(pixel, where: str) -> bytes:
    try:
        r, g, b = pixel
    except (TypeError, ValueError):
        raise PngError(f"{where}: expected an (r, g, b) triple, got {pixel!r}") from None
    try:
        return bytes((r, g, b))
    except (ValueError, TypeError):
        raise PngError(f"{where}: channel out of range 0..255 in {pixel!r}") from None


def _row_bytes(row: Row, width: int, index: int) -> bytes:
    """Normalise one scanline to ``width * 3`` raw bytes.

    A row may be bytes-like, a sequence of ``(r, g, b)`` triples, or a flat
    sequence of channel ints.  The three forms are distinguished by type, not
    by length, so a one-pixel row is never ambiguous.
    """
    want = width * CHANNELS
    if isinstance(row, (bytes, bytearray, memoryview)):
        raw = bytes(row)
        if len(raw) != want:
            raise PngError(
                f"row {index} has {len(raw)} bytes, expected {want} "
                f"({width} pixels x {CHANNELS} channels)"
            )
        return raw

    try:
        items = list(row)
    except TypeError:
        raise PngError(f"row {index} is not a sequence of pixels") from None

    if items and isinstance(items[0], int):
        if len(items) != want:
            raise PngError(
                f"row {index} has {len(items)} channel values, expected {want}"
            )
        try:
            return bytes(items)
        except (ValueError, TypeError):
            raise PngError(f"row {index}: channel out of range 0..255") from None

    if len(items) != width:
        raise PngError(f"row {index} has {len(items)} pixels, expected {width}")
    out = bytearray(want)
    for x, pixel in enumerate(items):
        out[x * CHANNELS : x * CHANNELS + CHANNELS] = _pixel_bytes(
            pixel, f"row {index} pixel {x}"
        )
    return bytes(out)


def _raw_scanlines(width: int, height: int, rgb_rows: Rows) -> bytes:
    """Build the pre-compression stream: filter byte 0 then pixels, per row."""
    stride = width * CHANNELS
    out = bytearray()

    if isinstance(rgb_rows, (bytes, bytearray, memoryview)):
        flat = bytes(rgb_rows)
        if len(flat) != stride * height:
            raise PngError(
                f"flat pixel buffer has {len(flat)} bytes, expected "
                f"{stride * height} for {width}x{height}"
            )
        for y in range(height):
            out.append(FILTER_NONE)
            out += flat[y * stride : (y + 1) * stride]
        return bytes(out)

    rows = list(rgb_rows)
    if len(rows) != height:
        raise PngError(f"got {len(rows)} rows, expected {height}")
    for y, row in enumerate(rows):
        out.append(FILTER_NONE)
        out += _row_bytes(row, width, y)
    return bytes(out)


def encode_png(
    width: int,
    height: int,
    rgb_rows: Rows,
    *,
    level: int = DEFAULT_LEVEL,
) -> bytes:
    """Return the complete bytes of an 8-bit truecolour PNG.

    ``rgb_rows`` is either a flat bytes-like buffer of ``width * height * 3``
    channel bytes, or a sequence of ``height`` rows (each bytes-like, a
    sequence of ``(r, g, b)`` triples, or a flat sequence of channel ints).
    """
    if not isinstance(width, int) or isinstance(width, bool):
        raise PngError(f"width must be an int, got {width!r}")
    if not isinstance(height, int) or isinstance(height, bool):
        raise PngError(f"height must be an int, got {height!r}")
    if width <= 0 or height <= 0:
        raise PngError(f"image must have positive size, got {width}x{height}")

    raw = _raw_scanlines(width, height, rgb_rows)

    ihdr = struct.pack(
        ">IIBBBBB",
        width,
        height,
        BIT_DEPTH,
        COLOUR_TYPE_RGB,
        0,  # compression method: deflate, the only one defined
        0,  # filter method: adaptive, the only one defined
        0,  # interlace: none
    )
    idat = zlib.compress(raw, level)
    return b"".join(
        (
            PNG_SIGNATURE,
            _chunk(b"IHDR", ihdr),
            _chunk(b"IDAT", idat),
            _chunk(b"IEND", b""),
        )
    )


def write_png(
    path,
    width: int,
    height: int,
    rgb_rows: Rows,
    *,
    level: int = DEFAULT_LEVEL,
) -> int:
    """Write an 8-bit truecolour PNG to ``path``; return the bytes written."""
    data = encode_png(width, height, rgb_rows, level=level)
    with open(path, "wb") as handle:
        handle.write(data)
    return len(data)


# --------------------------------------------------------------------------
# Reading back (verification, not a general-purpose decoder)
# --------------------------------------------------------------------------


def iter_chunks(data: bytes) -> Iterator[Tuple[bytes, bytes]]:
    """Yield ``(type, data)`` for each chunk, checking the signature and CRCs.

    Raises :class:`PngError` on a bad signature, a truncated chunk or a CRC
    mismatch, so a caller can use it as a structural validator.
    """
    blob = bytes(data)
    if not blob.startswith(PNG_SIGNATURE):
        raise PngError("not a PNG: bad signature")
    pos = len(PNG_SIGNATURE)
    seen_end = False
    while pos < len(blob):
        if seen_end:
            raise PngError("trailing bytes after IEND")
        if pos + 8 > len(blob):
            raise PngError("truncated chunk header")
        (length,) = struct.unpack(">I", blob[pos : pos + 4])
        tag = blob[pos + 4 : pos + 8]
        start = pos + 8
        end = start + length
        if end + 4 > len(blob):
            raise PngError(f"truncated {tag!r} chunk")
        payload = blob[start:end]
        (stored,) = struct.unpack(">I", blob[end : end + 4])
        actual = zlib.crc32(tag + payload) & 0xFFFFFFFF
        if stored != actual:
            raise PngError(
                f"CRC mismatch in {tag!r}: stored {stored:#010x}, computed {actual:#010x}"
            )
        yield tag, payload
        if tag == b"IEND":
            seen_end = True
        pos = end + 4
    if not seen_end:
        raise PngError("no IEND chunk")


def _unfilter(raw: bytes, width: int, height: int) -> List[bytes]:
    """Undo the per-scanline filters of an 8-bit truecolour image."""
    stride = width * CHANNELS
    if len(raw) != (stride + 1) * height:
        raise PngError(
            f"inflated data is {len(raw)} bytes, expected {(stride + 1) * height}"
        )
    rows: List[bytes] = []
    prior = bytearray(stride)
    pos = 0
    for y in range(height):
        ftype = raw[pos]
        pos += 1
        line = bytearray(raw[pos : pos + stride])
        pos += stride
        if ftype == 0:
            pass
        elif ftype == 1:  # Sub
            for i in range(CHANNELS, stride):
                line[i] = (line[i] + line[i - CHANNELS]) & 0xFF
        elif ftype == 2:  # Up
            for i in range(stride):
                line[i] = (line[i] + prior[i]) & 0xFF
        elif ftype == 3:  # Average
            for i in range(stride):
                left = line[i - CHANNELS] if i >= CHANNELS else 0
                line[i] = (line[i] + ((left + prior[i]) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for i in range(stride):
                left = line[i - CHANNELS] if i >= CHANNELS else 0
                up = prior[i]
                upleft = prior[i - CHANNELS] if i >= CHANNELS else 0
                p = left + up - upleft
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - upleft)
                if pa <= pb and pa <= pc:
                    pred = left
                elif pb <= pc:
                    pred = up
                else:
                    pred = upleft
                line[i] = (line[i] + pred) & 0xFF
        else:
            raise PngError(f"row {y}: unknown filter type {ftype}")
        rows.append(bytes(line))
        prior = line
    return rows


def decode_png(data: bytes) -> Tuple[int, int, List[bytes]]:
    """Return ``(width, height, rows)`` for an 8-bit truecolour PNG.

    Each row is ``width * 3`` raw bytes.  Chunk CRCs are checked on the way
    through, so a successful decode also proves the file is well formed.
    """
    width = height = 0
    header_seen = False
    idat = bytearray()
    for tag, payload in iter_chunks(data):
        if tag == b"IHDR":
            if header_seen:
                raise PngError("more than one IHDR")
            if len(payload) != 13:
                raise PngError(f"IHDR is {len(payload)} bytes, expected 13")
            width, height, depth, colour, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if depth != BIT_DEPTH or colour != COLOUR_TYPE_RGB:
                raise PngError(
                    f"unsupported image: bit depth {depth}, colour type {colour}"
                )
            if comp != 0 or filt != 0 or interlace != 0:
                raise PngError("unsupported compression, filter or interlace method")
            header_seen = True
        elif tag == b"IDAT":
            if not header_seen:
                raise PngError("IDAT before IHDR")
            idat += payload
    if not header_seen:
        raise PngError("no IHDR chunk")
    if not idat:
        raise PngError("no IDAT data")
    return width, height, _unfilter(zlib.decompress(bytes(idat)), width, height)
