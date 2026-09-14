"""Top-down debug renders of a generated map.

Spec: docs/WORLD_BIBLE.md -- this is tooling, not one of the six generation
stages.  It draws what the stages produced so a human can look at a seed and
say "that is wrong" without loading the map in the engine:

* :func:`render_layout` draws the finished map -- stage 3 terrain kinds,
  stage 4 filler, stage 5 set pieces and tells, stage 6 spawns -- with the
  seed and the template and tileset refs stamped into the image.
* :func:`render_routed` draws only the stage 2 result, the node graph and its
  routed edges, which is what the Phase 1 CLI showed.

Both write an 8-bit truecolour PNG through :mod:`lucifer_gen.png` and return
the exact bytes written, so a caller can compare two renders without touching
the filesystem.

Reading the picture
-------------------
Cell colour comes from the stage 3 terrain plan (:class:`CellKind`), because
that is the record of what the generator *decided* a cell is; the stage 4
tile grid is consulted only to tell impassable filler from untouched space.
Markers are drawn on top in this order -- spawns, set pieces, checkpoints,
exit, entrance -- so the landmarks a reader looks for first are never buried
under a spawn dot.  Every marker paints its own centre pixel last, so a
marker stays visible (and its legend colour stays findable) even at the
smallest useful scale.

Geometry
--------
The map occupies ``grid * scale`` pixels square, inset by :data:`MARGIN` on
every side, with a caption band of :func:`caption_height` pixels below it for
the seed and reference stamp.  :func:`image_size` is the one formula; both
renderers and the tests use it rather than recomputing.

Determinism
-----------
Nothing here is random: the image is a pure function of the map, the palette
and the scale.  No stage seed stream is drawn from, because drawing one would
be a lie -- rendering makes no choices the seed could vary.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Sequence, Tuple

from .contracts import (
    Cell,
    CellKind,
    GeneratedMap,
    Role,
    RoutedLayout,
    SetPiecePlacement,
    SpawnPack,
)
from .png import encode_png
from .seed import format_seed

__all__ = [
    "PALETTE",
    "LEGEND",
    "KIND_KEY",
    "ROLE_KEY",
    "MARGIN",
    "GRID_PERIOD",
    "FONT",
    "image_size",
    "caption_height",
    "caption_top",
    "text_scale",
    "Canvas",
    "render_layout",
    "render_routed",
]

RGB = Tuple[int, int, int]

# --------------------------------------------------------------------------
# Layout constants
# --------------------------------------------------------------------------

#: Blank pixels around the map square, so edge cells and markers are not
#: clipped by the image border.
MARGIN = 8

#: A thin rule is drawn every this many cells, in both directions, so a
#: reader can count distances off the picture (8 cells is 32 m at 4 m cells).
GRID_PERIOD = 8


def text_scale(scale: int) -> int:
    """Pixel size of one font dot at a given cell scale (at least 1)."""
    return max(1, int(scale) // 6)


def caption_height(scale: int) -> int:
    """Height of the caption band: padding, two text lines and a gap."""
    t = text_scale(scale)
    return 2 * (3 * t) + 2 * (5 * t) + (2 * t)


def image_size(grid: int, scale: int) -> Tuple[int, int]:
    """Pixel size of a render of a ``grid``-cell map at ``scale`` px/cell."""
    grid = int(grid)
    scale = int(scale)
    if grid <= 0:
        raise ValueError(f"grid must be positive, got {grid}")
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    side = grid * scale
    return side + 2 * MARGIN, side + 2 * MARGIN + caption_height(scale)


# --------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------

#: Every colour the renderers can paint, by legend name.  Terrain families sit
#: apart in hue (sand corridors, pale rooms, violet set pieces, ember boss
#: approach, blue water, grey cliff, near-black filler) so a colour-blind
#: reader still has lightness to go on, and every marker is brighter than the
#: ground it can land on -- spawn rose against the ember approach floor, a
#: white set-piece outline against the violet interior -- so nothing an
#: overlay marks can swallow the marker.
PALETTE: Dict[str, RGB] = {
    # ground
    "background": (20, 21, 26),
    "filler": (46, 48, 58),
    "corridor": (198, 152, 92),
    "room": (236, 218, 170),
    "set_piece": (168, 82, 176),
    "approach": (214, 84, 52),
    "water": (44, 104, 184),
    "cliff": (128, 118, 104),
    # furniture
    "grid_line": (78, 82, 96),
    "text": (226, 230, 238),
    # markers
    "entrance": (64, 220, 120),
    "exit": (248, 226, 64),
    "checkpoint": (72, 208, 236),
    "set_piece_marker": (255, 255, 255),
    "spawn": (240, 60, 122),
    "spawn_elite": (255, 176, 32),
    # stage 2 node roles
    "node_boss": (226, 60, 60),
    "node_mechanic": (170, 120, 240),
    "node_side": (200, 204, 212),
}

#: Reading order for a legend, and the set of names the tests sweep.
LEGEND: Tuple[str, ...] = (
    "corridor",
    "room",
    "set_piece",
    "approach",
    "filler",
    "cliff",
    "water",
    "entrance",
    "exit",
    "checkpoint",
    "set_piece_marker",
    "spawn",
    "spawn_elite",
    "grid_line",
)

#: Stage 3 cell kinds to palette names.  EMPTY is resolved per cell: it is
#: filler where stage 4 laid an impassable tile, background where it did not.
KIND_KEY: Dict[CellKind, str] = {
    CellKind.CORRIDOR: "corridor",
    CellKind.ROOM: "room",
    CellKind.SET_PIECE: "set_piece",
    CellKind.APPROACH: "approach",
    CellKind.WATER: "water",
    CellKind.CLIFF: "cliff",
    CellKind.EMPTY: "background",
}

#: Stage 1 roles to palette names, for the stage 2 render.
ROLE_KEY: Dict[Role, str] = {
    Role.ENTRANCE: "entrance",
    Role.EXIT: "exit",
    Role.CHECKPOINT: "checkpoint",
    Role.BOSS: "node_boss",
    Role.MECHANIC: "node_mechanic",
    Role.SIDE: "node_side",
}


# --------------------------------------------------------------------------
# A 3x5 block font, defined here so no font file or library is needed
# --------------------------------------------------------------------------

_GLYPHS: Dict[str, Tuple[str, str, str, str, str]] = {
    " ": ("...", "...", "...", "...", "..."),
    "A": (".#.", "#.#", "###", "#.#", "#.#"),
    "B": ("##.", "#.#", "##.", "#.#", "##."),
    "C": (".##", "#..", "#..", "#..", ".##"),
    "D": ("##.", "#.#", "#.#", "#.#", "##."),
    "E": ("###", "#..", "##.", "#..", "###"),
    "F": ("###", "#..", "##.", "#..", "#.."),
    "G": (".##", "#..", "#.#", "#.#", ".##"),
    "H": ("#.#", "#.#", "###", "#.#", "#.#"),
    "I": ("###", ".#.", ".#.", ".#.", "###"),
    "J": ("..#", "..#", "..#", "#.#", ".#."),
    "K": ("#.#", "#.#", "##.", "#.#", "#.#"),
    "L": ("#..", "#..", "#..", "#..", "###"),
    "M": ("#.#", "###", "###", "#.#", "#.#"),
    "N": ("##.", "#.#", "#.#", "#.#", "#.#"),
    "O": (".#.", "#.#", "#.#", "#.#", ".#."),
    "P": ("##.", "#.#", "##.", "#..", "#.."),
    "Q": (".#.", "#.#", "#.#", "##.", ".##"),
    "R": ("##.", "#.#", "##.", "#.#", "#.#"),
    "S": (".##", "#..", ".#.", "..#", "##."),
    "T": ("###", ".#.", ".#.", ".#.", ".#."),
    "U": ("#.#", "#.#", "#.#", "#.#", "###"),
    "V": ("#.#", "#.#", "#.#", ".#.", ".#."),
    "W": ("#.#", "#.#", "###", "###", "#.#"),
    "X": ("#.#", "#.#", ".#.", "#.#", "#.#"),
    "Y": ("#.#", "#.#", ".#.", ".#.", ".#."),
    "Z": ("###", "..#", ".#.", "#..", "###"),
    "0": ("###", "#.#", "#.#", "#.#", "###"),
    "1": (".#.", "##.", ".#.", ".#.", "###"),
    "2": ("##.", "..#", ".#.", "#..", "###"),
    "3": ("##.", "..#", ".#.", "..#", "##."),
    "4": ("#.#", "#.#", "###", "..#", "..#"),
    "5": ("###", "#..", "##.", "..#", "##."),
    "6": (".##", "#..", "###", "#.#", "###"),
    "7": ("###", "..#", ".#.", ".#.", ".#."),
    "8": ("###", "#.#", "###", "#.#", "###"),
    "9": ("###", "#.#", "###", "..#", "##."),
    ".": ("...", "...", "...", "...", ".#."),
    ",": ("...", "...", "...", ".#.", "#.."),
    "-": ("...", "...", "###", "...", "..."),
    "_": ("...", "...", "...", "...", "###"),
    "+": ("...", ".#.", "###", ".#.", "..."),
    ":": ("...", ".#.", "...", ".#.", "..."),
    "/": ("..#", "..#", ".#.", "#..", "#.."),
    "@": ("###", "#.#", "###", "#..", ".##"),
    "#": ("#.#", "###", "#.#", "###", "#.#"),
    "(": (".#.", "#..", "#..", "#..", ".#."),
    ")": (".#.", "..#", "..#", "..#", ".#."),
    "?": ("##.", "..#", ".#.", "...", ".#."),
    "!": (".#.", ".#.", ".#.", "...", ".#."),
    "*": ("#.#", ".#.", "###", ".#.", "#.#"),
    "=": ("...", "###", "...", "###", "..."),
    "%": ("#.#", "..#", ".#.", "#..", "#.#"),
}

#: Anything not in the font is drawn as this, rather than silently vanishing.
FALLBACK_GLYPH = "?"

GLYPH_W = 3
GLYPH_H = 5
GLYPH_GAP = 1  # blank columns between glyphs, in font dots

#: Public view of the font, so tools and tests can check coverage.
FONT: Mapping[str, Tuple[str, ...]] = _GLYPHS


def glyph_for(char: str) -> Tuple[str, ...]:
    """The 5 row strings for ``char``; uppercase, with a visible fallback."""
    return _GLYPHS.get(char.upper(), _GLYPHS[FALLBACK_GLYPH])


def text_width(text: str, scale: int) -> int:
    """Pixel width of ``text`` drawn at font-dot size ``scale``."""
    if not text:
        return 0
    return (len(text) * (GLYPH_W + GLYPH_GAP) - GLYPH_GAP) * scale


def fit_text(text: str, max_px: int, scale: int) -> str:
    """Truncate ``text`` to what fits in ``max_px`` pixels.

    The caption band is sized from the scale alone so the image geometry is
    predictable, which means a very long template ref has to give way; it is
    cut rather than shrunk so the rest of the stamp stays the same size.
    """
    if text_width(text, scale) <= max_px:
        return text
    advance = (GLYPH_W + GLYPH_GAP) * scale
    if advance <= 0:
        return ""
    # Last glyph needs GLYPH_W * scale, the ones before it a full advance.
    fits = (max_px + GLYPH_GAP * scale) // advance
    return text[: max(0, fits)]


# --------------------------------------------------------------------------
# Canvas
# --------------------------------------------------------------------------


class Canvas:
    """A flat RGB pixel buffer with the few drawing primitives needed here.

    Coordinates are pixels, x east and y south, matching the cell grid.  All
    primitives clip silently at the borders, so a marker near the edge of the
    grid costs no special cases at the call sites.
    """

    __slots__ = ("width", "height", "pixels")

    def __init__(self, width: int, height: int, background: RGB) -> None:
        if width <= 0 or height <= 0:
            raise ValueError(f"canvas must be positive, got {width}x{height}")
        self.width = int(width)
        self.height = int(height)
        self.pixels = bytearray(bytes(background) * (self.width * self.height))

    # -- primitives --------------------------------------------------------

    def set_pixel(self, x: int, y: int, colour: RGB) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            i = (y * self.width + x) * 3
            self.pixels[i : i + 3] = bytes(colour)

    def get_pixel(self, x: int, y: int) -> RGB:
        # Bounds-checked, unlike the drawing primitives: a negative index
        # would otherwise quietly read some other row's pixel.
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise IndexError(f"({x}, {y}) is outside a {self.width}x{self.height} canvas")
        i = (y * self.width + x) * 3
        return tuple(self.pixels[i : i + 3])  # type: ignore[return-value]

    def fill_rect(self, x: int, y: int, w: int, h: int, colour: RGB) -> None:
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(self.width, x + w), min(self.height, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        run = bytes(colour) * (x1 - x0)
        for row in range(y0, y1):
            i = (row * self.width + x0) * 3
            self.pixels[i : i + len(run)] = run

    def rect_outline(self, x: int, y: int, w: int, h: int, colour: RGB, thickness: int = 1) -> None:
        t = max(1, thickness)
        self.fill_rect(x, y, w, t, colour)
        self.fill_rect(x, y + h - t, w, t, colour)
        self.fill_rect(x, y, t, h, colour)
        self.fill_rect(x + w - t, y, t, h, colour)

    def disc(self, cx: int, cy: int, radius: int, colour: RGB) -> None:
        r = max(0, radius)
        rr = r * r
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= rr:
                    self.set_pixel(cx + dx, cy + dy, colour)

    def diamond(self, cx: int, cy: int, radius: int, colour: RGB, hollow: bool = False) -> None:
        r = max(0, radius)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                d = abs(dx) + abs(dy)
                if d == r or (not hollow and d < r):
                    self.set_pixel(cx + dx, cy + dy, colour)

    def cross(self, cx: int, cy: int, radius: int, colour: RGB, thickness: int = 1) -> None:
        r = max(0, radius)
        t = max(1, thickness)
        self.fill_rect(cx - r, cy - t // 2, 2 * r + 1, t, colour)
        self.fill_rect(cx - t // 2, cy - r, t, 2 * r + 1, colour)

    # -- text --------------------------------------------------------------

    def text(self, x: int, y: int, message: str, colour: RGB, scale: int = 1) -> int:
        """Draw ``message`` with its top-left dot at ``(x, y)``.

        Returns the x pixel just past the last glyph, so captions can be
        composed left to right.
        """
        s = max(1, int(scale))
        pen = x
        for char in message:
            rows = glyph_for(char)
            for gy, row in enumerate(rows):
                for gx, dot in enumerate(row):
                    if dot != ".":
                        self.fill_rect(pen + gx * s, y + gy * s, s, s, colour)
            pen += (GLYPH_W + GLYPH_GAP) * s
        return pen - GLYPH_GAP * s

    # -- output ------------------------------------------------------------

    def write(self, path) -> bytes:
        """Write the canvas to ``path`` as a PNG and return the file bytes.

        The bytes are returned as well as written so a caller can compare two
        renders -- the determinism tests do -- without reading them back off
        disk and without encoding twice.
        """
        data = encode_png(self.width, self.height, self.pixels)
        with open(path, "wb") as handle:
            handle.write(data)
        return data


# --------------------------------------------------------------------------
# Shared drawing
# --------------------------------------------------------------------------


def _cell_origin(cell: Cell, scale: int) -> Tuple[int, int]:
    """Top-left pixel of a cell, including the margin."""
    return MARGIN + int(cell[0]) * scale, MARGIN + int(cell[1]) * scale


def _cell_centre(cell: Cell, scale: int) -> Tuple[int, int]:
    px, py = _cell_origin(cell, scale)
    return px + scale // 2, py + scale // 2


def _new_canvas(grid: int, scale: int) -> Canvas:
    width, height = image_size(grid, scale)
    return Canvas(width, height, PALETTE["background"])


def _draw_grid_lines(canvas: Canvas, grid: int, scale: int) -> None:
    """A one-pixel rule every :data:`GRID_PERIOD` cells, plus the border.

    Drawn after the terrain and before the markers: a reader needs the ruler
    over the ground but under anything they are trying to locate.
    """
    colour = PALETTE["grid_line"]
    span = grid * scale
    for k in range(0, grid + 1, GRID_PERIOD):
        offset = k * scale
        canvas.fill_rect(MARGIN + offset, MARGIN, 1, span + 1, colour)
        canvas.fill_rect(MARGIN, MARGIN + offset, span + 1, 1, colour)
    if grid % GRID_PERIOD:  # close the box when the grid is not a multiple
        canvas.fill_rect(MARGIN + span, MARGIN, 1, span + 1, colour)
        canvas.fill_rect(MARGIN, MARGIN + span, span + 1, 1, colour)


def _marker_radius(scale: int) -> int:
    return max(1, scale // 2 - 1)


def _draw_entrance(canvas: Canvas, cell: Cell, scale: int) -> None:
    """A filled square, ringed in background so it reads off pale room floor."""
    cx, cy = _cell_centre(cell, scale)
    r = _marker_radius(scale)
    colour = PALETTE["entrance"]
    canvas.fill_rect(cx - r, cy - r, 2 * r + 1, 2 * r + 1, colour)
    canvas.rect_outline(cx - r - 1, cy - r - 1, 2 * r + 3, 2 * r + 3, PALETTE["background"])
    canvas.set_pixel(cx, cy, colour)


def _draw_exit(canvas: Canvas, cell: Cell, scale: int) -> None:
    cx, cy = _cell_centre(cell, scale)
    colour = PALETTE["exit"]
    canvas.diamond(cx, cy, _marker_radius(scale) + 1, colour)
    canvas.set_pixel(cx, cy, colour)


def _draw_checkpoint(canvas: Canvas, cell: Cell, scale: int) -> None:
    cx, cy = _cell_centre(cell, scale)
    colour = PALETTE["checkpoint"]
    canvas.cross(cx, cy, _marker_radius(scale) + 1, colour, thickness=max(1, scale // 6))
    canvas.set_pixel(cx, cy, colour)


def _draw_spawn(canvas: Canvas, pack: SpawnPack, scale: int) -> None:
    """Ordinary packs are a rose disc; elites an amber disc inside a ring.

    Shape as well as colour, because the elite flag drives the HUD and a
    reader should not have to trust their monitor's gamma to see it.
    """
    cx, cy = _cell_centre(pack.cell, scale)
    r = max(1, scale // 3)
    if pack.elite:
        elite = PALETTE["spawn_elite"]
        canvas.diamond(cx, cy, r + 2, elite, hollow=True)
        canvas.disc(cx, cy, r, elite)
        canvas.set_pixel(cx, cy, elite)
    else:
        colour = PALETTE["spawn"]
        canvas.disc(cx, cy, r, colour)
        canvas.set_pixel(cx, cy, colour)


def _draw_set_piece(canvas: Canvas, piece: SetPiecePlacement, scale: int) -> None:
    """Outline the stored footprint and mark its centre.

    ``rot`` orients the interior, not the footprint rectangle, so the outline
    uses ``w`` and ``h`` exactly as stage 5 recorded them.
    """
    px, py = _cell_origin(piece.cell, scale)
    w = max(1, int(piece.w)) * scale
    h = max(1, int(piece.h)) * scale
    colour = PALETTE["set_piece_marker"]
    canvas.rect_outline(px, py, w, h, colour, thickness=max(1, scale // 8))
    canvas.set_pixel(px + w // 2, py + h // 2, colour)


def caption_top(grid: int, scale: int) -> int:
    """First pixel row of the caption band, below the map and its margin."""
    return 2 * MARGIN + int(grid) * int(scale)


def _stamp(canvas: Canvas, grid: int, scale: int, lines: Sequence[str]) -> None:
    """Write the caption band: seed on the first line, refs on the second.

    The band is ``3t`` padding, a ``5t`` line, a ``2t`` gap, a second ``5t``
    line and ``3t`` padding, which is exactly :func:`caption_height`.  Text is
    upper-cased because the font has one case, and truncated to the width
    because the band's size must depend on the scale alone.
    """
    t = text_scale(scale)
    available = canvas.width - 2 * MARGIN
    y = caption_top(grid, scale) + 3 * t
    for index, line in enumerate(lines[:2]):
        canvas.text(
            MARGIN,
            y + index * (GLYPH_H * t + 2 * t),
            fit_text(line.upper(), available, t),
            PALETTE["text"],
            t,
        )


# --------------------------------------------------------------------------
# Stage 2 render
# --------------------------------------------------------------------------


def render_routed(routed: RoutedLayout, path, scale: int = 12) -> bytes:
    """Draw the stage 2 result: routed edges and the nodes they connect.

    Spec: docs/WORLD_BIBLE.md stage 2.  Every cell on a routed path is painted
    in the corridor colour, and each node is drawn in its role colour, so the
    picture answers the two questions stage 2 can get wrong: did the nodes
    land where the shape wanted them, and did the corridors spread out.
    """
    grid = int(getattr(routed, "grid", 0) or 0)
    if grid <= 0:
        raise ValueError("routed layout has no grid size to draw")
    canvas = _new_canvas(grid, scale)

    path_colour = PALETTE["corridor"]
    for edge in routed.edges:
        for cell in edge.path:
            px, py = _cell_origin(cell, scale)
            canvas.fill_rect(px, py, scale, scale, path_colour)

    _draw_grid_lines(canvas, grid, scale)

    # Sorted by id so the picture does not depend on dict insertion order.
    for node in sorted(routed.nodes.values(), key=lambda n: n.id):
        colour = PALETTE[ROLE_KEY.get(node.role, "node_side")]
        cx, cy = _cell_centre(node.cell, scale)
        r = max(1, scale // 2)
        canvas.fill_rect(cx - r, cy - r, 2 * r + 1, 2 * r + 1, colour)
        canvas.rect_outline(cx - r, cy - r, 2 * r + 1, 2 * r + 1, PALETTE["background"])
        canvas.set_pixel(cx, cy, colour)

    template = getattr(routed, "template", None)
    template_ref = getattr(template, "ref", "") if template is not None else ""
    shape = getattr(getattr(template, "shape", None), "value", "")
    _stamp(
        canvas,
        grid,
        scale,
        (
            f"SEED {format_seed(int(routed.seed))}",
            f"TPL {template_ref} SHAPE {shape} GRID {grid}",
        ),
    )
    return canvas.write(path)


# --------------------------------------------------------------------------
# Full map render
# --------------------------------------------------------------------------


def _grid_of(generated_map: GeneratedMap) -> int:
    for holder in (
        getattr(generated_map, "tiles", None),
        getattr(generated_map, "terrain", None),
        getattr(generated_map, "routed", None),
    ):
        grid = getattr(holder, "grid", None)
        if isinstance(grid, int) and grid > 0:
            return grid
    raise ValueError("map carries no grid size to draw")


def _filler_id(tiles) -> Optional[int]:
    """The tile id stage 4 uses for impassable filler, if the db offers one."""
    value = getattr(tiles, "filler_tile_id", None)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _is_filler(generated_map: GeneratedMap, tiles, cell: Cell) -> bool:
    """True when stage 4 laid an impassable tile on an untouched cell.

    Filler is read off the tile grid, not the terrain plan, because the plan
    does not model it: to stage 3 those cells are simply EMPTY.  The grid's
    walkable table is the signal; where a caller passes something without one,
    the database's filler id is the fallback.
    """
    tile_grid = getattr(generated_map, "tiles", None)
    if tile_grid is None or not hasattr(tile_grid, "at"):
        return False
    placement = tile_grid.at(cell)
    if placement is None:
        return False
    if hasattr(tile_grid, "is_walkable"):
        # Walkable floor on a cell stage 3 called empty is not filler; leave
        # it as background rather than mislabel it.
        return not tile_grid.is_walkable(cell)
    filler_id = _filler_id(tiles)
    return filler_id is None or placement.tile_id == filler_id


def _terrain_key(generated_map: GeneratedMap, tiles, cell: Cell) -> str:
    terrain = getattr(generated_map, "terrain", None)
    kind = terrain.kind(cell) if terrain is not None else CellKind.EMPTY
    if kind is not CellKind.EMPTY:
        return KIND_KEY.get(kind, "background")
    return "filler" if _is_filler(generated_map, tiles, cell) else "background"


def render_layout(generated_map: GeneratedMap, tiles, path, scale: int = 12) -> bytes:
    """Draw the finished map and write it to ``path``; return the PNG bytes.

    ``tiles`` is the stage 4 tile database (or ``None``).  It is read only for
    the filler tile id and the tileset ref stamped into the caption; the
    picture itself comes from the map, so a render never depends on the
    database agreeing with what was already placed.

    Spec: stages 3 to 6 are all visible here -- terrain kinds as ground
    colour, set pieces outlined, the exit and checkpoints marked so the stage
    5 tells can be eyeballed, and stage 6 packs as dots with elites apart.
    """
    grid = _grid_of(generated_map)
    canvas = _new_canvas(grid, scale)

    # Ground: stage 3 kinds, with stage 4 filler filling in the EMPTY cells.
    background = PALETTE["background"]
    for y in range(grid):
        for x in range(grid):
            colour = PALETTE[_terrain_key(generated_map, tiles, (x, y))]
            if colour == background:
                continue  # the canvas already is this colour
            px, py = _cell_origin((x, y), scale)
            canvas.fill_rect(px, py, scale, scale, colour)

    _draw_grid_lines(canvas, grid, scale)

    # Overlay, back to front: spawns first, landmarks last.
    for pack in getattr(generated_map, "spawns", ()) or ():
        _draw_spawn(canvas, pack, scale)
    for piece in getattr(generated_map, "set_pieces", ()) or ():
        _draw_set_piece(canvas, piece, scale)
    for checkpoint in getattr(generated_map, "checkpoints", ()) or ():
        _draw_checkpoint(canvas, checkpoint, scale)
    exit_cell = getattr(generated_map, "exit_cell", None)
    if exit_cell is not None:
        _draw_exit(canvas, exit_cell, scale)
    routed = getattr(generated_map, "routed", None)
    entrance = routed.node_of_role(Role.ENTRANCE) if routed is not None else None
    if entrance is not None:
        _draw_entrance(canvas, entrance.cell, scale)

    template = getattr(generated_map, "template", None)
    template_ref = getattr(template, "ref", "") if template is not None else ""
    tiles_ref = getattr(generated_map, "tileset_ref", "") or getattr(tiles, "ref", "")
    _stamp(
        canvas,
        grid,
        scale,
        (
            f"SEED {format_seed(int(generated_map.seed))}",
            f"TPL {template_ref} TILES {tiles_ref} GRID {grid}",
        ),
    )
    return canvas.write(path)
