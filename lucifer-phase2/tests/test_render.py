"""Tests for the PNG writer and the map renders.

Spec: docs/WORLD_BIBLE.md -- rendering is tooling rather than one of the six
stages, so what is asserted here is not "the map is right" but "the picture is
a faithful, valid, reproducible view of whatever map it was handed".

Three claims carry the weight.

*The bytes are a real PNG.*  Nothing in this project can open an image, so the
file is parsed here from first principles: signature, chunk framing, a CRC-32
recomputed over every chunk, the IHDR fields, and the filter byte on each
scanline.  If the writer drifts, these fail before an artist ever tries to
open the file.

*The render is deterministic.*  Two renders of one map are compared byte for
byte, including across freshly rebuilt inputs, because the images go in bug
reports next to a seed and must be diffable.

*The legend means something.*  Every colour in :data:`render.LEGEND` is
asserted present when its feature is present *and absent when it is not* --
the second half is what stops the palette from decaying into decoration.
"""

from __future__ import annotations

import struct
import sys
import zlib
from pathlib import Path

# Runnable as `pytest tests/test_render.py` or `python3 tests/test_render.py`
# from anywhere, without depending on how the package root reaches sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_gen import render
from lucifer_gen.contracts import (
    CellKind,
    GeneratedMap,
    GraphTemplate,
    Placement,
    Role,
    RoutedEdge,
    RoutedLayout,
    RoutedNode,
    SetPiecePlacement,
    Shape,
    SpawnPack,
    TemplateEdge,
    TemplateNode,
    TerrainPlan,
    TileClass,
    TileGrid,
)
from lucifer_gen.png import PNG_SIGNATURE, PngError, encode_png, write_png
from lucifer_gen.render import MARGIN, PALETTE, caption_height, image_size, render_layout, render_routed

# --------------------------------------------------------------------------
# An independent PNG reader, so the writer is never checked against itself
# --------------------------------------------------------------------------


def parse_chunks(data):
    """Return ``[(tag, payload)]``, failing the test on any framing error."""
    assert data[:8] == PNG_SIGNATURE, "missing or wrong PNG signature"
    chunks = []
    pos = 8
    while pos < len(data):
        assert pos + 8 <= len(data), "truncated chunk header"
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        payload = data[pos + 8 : pos + 8 + length]
        assert len(payload) == length, f"truncated {tag!r} payload"
        (stored,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        expected = zlib.crc32(tag + payload) & 0xFFFFFFFF
        assert stored == expected, f"bad CRC on {tag!r}: {stored:#x} != {expected:#x}"
        chunks.append((tag, payload))
        pos += 12 + length
    return chunks


def parse_header(data):
    """Return the seven IHDR fields of the first chunk."""
    tag, payload = parse_chunks(data)[0]
    assert tag == b"IHDR", "first chunk must be IHDR"
    assert len(payload) == 13
    return struct.unpack(">IIBBBBB", payload)


def parse_pixels(data):
    """Return ``(width, height, rows)`` with rows as lists of (r, g, b).

    Asserts on the way that every scanline carries filter byte 0, which is
    both a spec requirement of this writer and what makes this reader simple
    enough to trust.
    """
    width, height, depth, colour, comp, filt, interlace = parse_header(data)
    assert (depth, colour, comp, filt, interlace) == (8, 2, 0, 0, 0)
    idat = b"".join(p for tag, p in parse_chunks(data) if tag == b"IDAT")
    raw = zlib.decompress(idat)
    stride = width * 3
    assert len(raw) == (stride + 1) * height, "wrong amount of pixel data"
    rows = []
    for y in range(height):
        start = y * (stride + 1)
        assert raw[start] == 0, f"scanline {y} is not filter 0"
        line = raw[start + 1 : start + 1 + stride]
        rows.append([tuple(line[i : i + 3]) for i in range(0, stride, 3)])
    return width, height, rows


def colours_in(data):
    """The set of distinct RGB triples in an image."""
    _, _, rows = parse_pixels(data)
    return {px for row in rows for px in row}


# --------------------------------------------------------------------------
# Fixtures: synthetic maps built from contracts only
# --------------------------------------------------------------------------


class FakeTileDatabase:
    """The two attributes :func:`render_layout` is allowed to read."""

    def __init__(self, filler_tile_id=0, ref="greybox@1"):
        self.filler_tile_id = filler_tile_id
        self.ref = ref


GRID = 24


def make_template(grid=GRID, version=3):
    return GraphTemplate(
        id="crypt",
        version=version,
        tile_class=TileClass.DUNGEON,
        shape=Shape.U,
        grid=grid,
        cell_m=4.0,
        nodes=(
            TemplateNode("start", Role.ENTRANCE, "a"),
            TemplateNode("mid", Role.CHECKPOINT, "b"),
            TemplateNode("boss", Role.BOSS, "c", set_piece="boss_arena"),
            TemplateNode("end", Role.EXIT, "d", set_piece="exit_checkpoint"),
        ),
        edges=(
            TemplateEdge("start", "mid"),
            TemplateEdge("mid", "boss"),
            TemplateEdge("boss", "end"),
        ),
        tileset="greybox",
    )


def make_routed(grid=GRID, seed=0xDEADBEEF, template=None):
    template = template or make_template(grid)
    across = [(x, 2) for x in range(2, 21)]
    down = [(20, y) for y in range(2, 21)]
    return RoutedLayout(
        seed=seed,
        template=template,
        grid=grid,
        nodes={
            "start": RoutedNode("start", Role.ENTRANCE, (2, 2)),
            "mid": RoutedNode("mid", Role.CHECKPOINT, (12, 2)),
            "boss": RoutedNode("boss", Role.BOSS, (20, 12), set_piece="boss_arena"),
            "end": RoutedNode("end", Role.EXIT, (20, 20), set_piece="exit_checkpoint"),
        },
        edges=[
            RoutedEdge("start", "mid", across[:11]),
            RoutedEdge("mid", "boss", across[10:] + down[:11]),
            RoutedEdge("boss", "end", down[10:]),
        ],
    )


def make_map(
    grid=GRID,
    seed=0xDEADBEEF,
    *,
    water=True,
    cliff=True,
    approach=True,
    elite=True,
    spawns=True,
    set_pieces=True,
    checkpoints=True,
    filler=True,
    version=3,
):
    """A hand-built map exercising whichever features a test wants present.

    Built from contracts alone -- no stage is run -- so a failure here points
    at the renderer and nothing else.
    """
    template = make_template(grid, version=version)
    routed = make_routed(grid, seed, template)

    terrain = TerrainPlan.blank(grid)
    for edge in routed.edges:
        for cell in edge.path:
            terrain.set_kind(cell, CellKind.CORRIDOR)
    for dx in range(3):
        for dy in range(3):
            terrain.set_kind((2 + dx, 2 + dy), CellKind.ROOM)
            if set_pieces:
                terrain.set_kind((19 + dx, 19 + dy), CellKind.SET_PIECE)
    if approach:
        for x in range(15, 19):
            terrain.set_kind((x, 2), CellKind.APPROACH)
    if water:
        terrain.set_kind((10, 10), CellKind.WATER)
    if cliff:
        terrain.set_kind((11, 10), CellKind.CLIFF)

    tile_grid = TileGrid.blank(grid)
    for y in range(grid):
        for x in range(grid):
            walkable = terrain.is_floor((x, y))
            if walkable:
                tile_grid.put((x, y), Placement(1, 0, False), True)
            elif filler:
                tile_grid.put((x, y), Placement(0, 0, False), False)

    packs = []
    if spawns:
        packs.append(SpawnPack("ghouls", (8, 2), 4, False))
        if elite:
            packs.append(SpawnPack("wight", (20, 14), 3, True))

    return GeneratedMap(
        seed=seed,
        template=template,
        tileset_ref="greybox@1",
        routed=routed,
        terrain=terrain,
        tiles=tile_grid,
        set_pieces=[SetPiecePlacement("boss_arena", (19, 19), 0, 3, 3)] if set_pieces else [],
        spawns=packs,
        exit_cell=(20, 20),
        checkpoints=[(12, 2)] if checkpoints else [],
    )


# --------------------------------------------------------------------------
# png.py: structure
# --------------------------------------------------------------------------


def test_written_file_is_a_structurally_valid_png(tmp_path):
    path = tmp_path / "tiny.png"
    rows = [[(255, 0, 0), (0, 255, 0)], [(0, 0, 255), (16, 32, 48)]]
    written = write_png(path, 2, 2, rows)

    data = path.read_bytes()
    assert written == len(data)
    chunks = parse_chunks(data)  # recomputes every CRC
    assert [tag for tag, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    assert parse_header(data) == (2, 2, 8, 2, 0, 0, 0)
    assert chunks[-1][1] == b"", "IEND carries no data"


def test_pixels_survive_the_round_trip(tmp_path):
    rows = [
        [(x * 7 % 256, y * 11 % 256, (x + y) % 256) for x in range(9)]
        for y in range(5)
    ]
    path = tmp_path / "grad.png"
    write_png(path, 9, 5, rows)
    width, height, back = parse_pixels(path.read_bytes())
    assert (width, height) == (9, 5)
    assert back == rows


def test_the_three_input_shapes_agree():
    rows = [[(1, 2, 3), (4, 5, 6)], [(7, 8, 9), (10, 11, 12)]]
    flat_rows = [bytes([1, 2, 3, 4, 5, 6]), bytes([7, 8, 9, 10, 11, 12])]
    whole = bytearray(range(1, 13))
    assert encode_png(2, 2, rows) == encode_png(2, 2, flat_rows) == encode_png(2, 2, whole)


def test_malformed_pixel_input_is_refused():
    with pytest.raises(PngError):
        encode_png(0, 1, b"")
    with pytest.raises(PngError):
        encode_png(2, 1, [[(1, 2, 3)]])  # row too short
    with pytest.raises(PngError):
        encode_png(1, 2, [[(1, 2, 3)]])  # too few rows
    with pytest.raises(PngError):
        encode_png(1, 1, [[(0, 0, 300)]])  # channel out of range
    with pytest.raises(PngError):
        encode_png(1, 1, b"\x00\x00")  # flat buffer too short


def test_a_corrupted_byte_is_caught_by_the_crc():
    data = bytearray(encode_png(4, 4, bytes(48)))
    data[len(data) // 2] ^= 0xFF  # somewhere inside the IDAT payload
    with pytest.raises(AssertionError):
        parse_chunks(bytes(data))


# --------------------------------------------------------------------------
# render.py: geometry
# --------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [1, 4, 6, 12, 20])
def test_image_is_the_grid_plus_the_declared_margin(tmp_path, scale):
    generated = make_map()
    data = render_layout(generated, FakeTileDatabase(), tmp_path / "m.png", scale=scale)

    expected_w = GRID * scale + 2 * MARGIN
    expected_h = GRID * scale + 2 * MARGIN + caption_height(scale)
    assert image_size(GRID, scale) == (expected_w, expected_h)

    width, height, *_ = parse_header(data)
    assert (width, height) == (expected_w, expected_h)
    assert parse_pixels(data)[:2] == (expected_w, expected_h)


def test_routed_render_uses_the_same_geometry(tmp_path):
    routed = make_routed()
    data = render_routed(routed, tmp_path / "r.png", 9)
    assert parse_header(data)[:2] == image_size(GRID, 9)


def test_a_bigger_grid_makes_a_bigger_image(tmp_path):
    small = render_layout(make_map(grid=16), None, tmp_path / "s.png", scale=8)
    large = render_layout(make_map(grid=32), None, tmp_path / "l.png", scale=8)
    assert parse_header(small)[:2] == image_size(16, 8)
    assert parse_header(large)[:2] == image_size(32, 8)


def test_nonsense_geometry_is_refused():
    with pytest.raises(ValueError):
        image_size(0, 12)
    with pytest.raises(ValueError):
        image_size(48, 0)


# --------------------------------------------------------------------------
# render.py: determinism
# --------------------------------------------------------------------------


def test_two_renders_of_one_map_are_byte_identical(tmp_path):
    generated = make_map()
    db = FakeTileDatabase()
    first = render_layout(generated, db, tmp_path / "a.png", scale=12)
    second = render_layout(generated, db, tmp_path / "b.png", scale=12)
    assert first == second
    assert (tmp_path / "a.png").read_bytes() == (tmp_path / "b.png").read_bytes() == first


def test_rebuilding_the_same_map_renders_the_same_bytes(tmp_path):
    first = render_layout(make_map(), FakeTileDatabase(), tmp_path / "a.png")
    second = render_layout(make_map(), FakeTileDatabase(), tmp_path / "b.png")
    assert first == second


def test_the_routed_render_is_deterministic(tmp_path):
    first = render_routed(make_routed(), tmp_path / "a.png", 12)
    second = render_routed(make_routed(), tmp_path / "b.png", 12)
    assert first == second


def test_a_different_seed_changes_the_picture(tmp_path):
    """The seed is stamped into the image, so it must move the bytes."""
    one = render_layout(make_map(seed=0xDEADBEEF), None, tmp_path / "a.png")
    two = render_layout(make_map(seed=0x0BADC0DE), None, tmp_path / "b.png")
    assert one != two


def test_a_different_template_version_changes_the_picture(tmp_path):
    one = render_layout(make_map(version=3), None, tmp_path / "a.png")
    two = render_layout(make_map(version=4), None, tmp_path / "b.png")
    assert one != two


# --------------------------------------------------------------------------
# render.py: the legend
# --------------------------------------------------------------------------


def test_every_legend_colour_appears_when_its_feature_is_present(tmp_path):
    data = render_layout(make_map(), FakeTileDatabase(), tmp_path / "m.png", scale=12)
    present = colours_in(data)
    missing = [name for name in render.LEGEND if PALETTE[name] not in present]
    assert not missing, f"legend colours never painted: {missing}"


@pytest.mark.parametrize("scale", [4, 7, 12, 16])
def test_markers_survive_small_scales(tmp_path, scale):
    """A marker that vanishes at scale 4 is a marker a reader cannot trust."""
    data = render_layout(make_map(), None, tmp_path / "m.png", scale=scale)
    present = colours_in(data)
    for name in ("entrance", "exit", "checkpoint", "set_piece_marker", "spawn", "spawn_elite"):
        assert PALETTE[name] in present, f"{name} marker lost at scale {scale}"


def test_absent_features_do_not_paint_their_colour(tmp_path):
    data = render_layout(
        make_map(water=False, cliff=False, approach=False, set_pieces=False),
        None,
        tmp_path / "plain.png",
    )
    present = colours_in(data)
    for name in ("water", "cliff", "approach", "set_piece", "set_piece_marker"):
        assert PALETTE[name] not in present, f"{name} painted for a map without it"
    # ...while the features that remain are still drawn.
    for name in ("corridor", "room", "filler", "entrance", "exit", "checkpoint"):
        assert PALETTE[name] in present


def test_elites_are_marked_differently_from_ordinary_packs(tmp_path):
    with_elite = colours_in(render_layout(make_map(elite=True), None, tmp_path / "e.png"))
    without = colours_in(render_layout(make_map(elite=False), None, tmp_path / "n.png"))
    assert PALETTE["spawn_elite"] != PALETTE["spawn"]
    assert PALETTE["spawn_elite"] in with_elite
    assert PALETTE["spawn_elite"] not in without
    assert PALETTE["spawn"] in without


def test_no_spawns_means_no_spawn_colours(tmp_path):
    present = colours_in(render_layout(make_map(spawns=False), None, tmp_path / "q.png"))
    assert PALETTE["spawn"] not in present
    assert PALETTE["spawn_elite"] not in present


def test_filler_is_distinct_from_untouched_background(tmp_path):
    filled = colours_in(render_layout(make_map(filler=True), None, tmp_path / "f.png"))
    bare = colours_in(render_layout(make_map(filler=False), None, tmp_path / "b.png"))
    assert PALETTE["filler"] != PALETTE["background"]
    assert PALETTE["filler"] in filled
    assert PALETTE["filler"] not in bare
    assert PALETTE["background"] in bare


def test_terrain_kinds_land_on_the_cells_they_belong_to(tmp_path):
    """Spot-check the mapping from cell coordinates to pixels."""
    scale = 12
    generated = make_map()
    data = render_layout(generated, None, tmp_path / "m.png", scale=scale)
    _, _, rows = parse_pixels(data)

    def sample(cell):
        # A pixel just inside the cell, away from grid rules and markers.
        x = MARGIN + cell[0] * scale + 2
        y = MARGIN + cell[1] * scale + 2
        return rows[y][x]

    assert sample((10, 10)) == PALETTE["water"]
    assert sample((11, 10)) == PALETTE["cliff"]
    assert sample((16, 2)) == PALETTE["approach"]
    assert sample((20, 19)) == PALETTE["set_piece"]
    assert sample((3, 3)) == PALETTE["room"]
    assert sample((6, 2)) == PALETTE["corridor"]
    assert sample((0, 12)) == PALETTE["filler"]


# --------------------------------------------------------------------------
# render.py: grid rules and the stamp
# --------------------------------------------------------------------------


def test_a_rule_is_drawn_every_eight_cells(tmp_path):
    scale = 12
    data = render_layout(make_map(), None, tmp_path / "m.png", scale=scale)
    _, _, rows = parse_pixels(data)
    inside_y = MARGIN + 13 * scale + 5  # a row with no marker on it

    for k in range(0, GRID + 1, render.GRID_PERIOD):
        x = MARGIN + k * scale
        assert rows[inside_y][x] == PALETTE["grid_line"], f"no rule at cell column {k}"
    # Between rules there is ground, not more rule.
    assert rows[inside_y][MARGIN + 4 * scale] != PALETTE["grid_line"]


def test_the_caption_band_carries_text(tmp_path):
    scale = 12
    data = render_layout(make_map(), None, tmp_path / "m.png", scale=scale)
    _, height, rows = parse_pixels(data)
    top = render.caption_top(GRID, scale)
    band = [px for row in rows[top:height] for px in row]
    assert PALETTE["text"] in band, "nothing was stamped in the caption band"
    # The band is caption only: no map ground bleeds into it.
    assert PALETTE["corridor"] not in band


def test_the_font_covers_everything_the_stamp_can_print():
    generated = make_map()
    lines = (
        "SEED 0X00000000DEADBEEF",
        f"TPL {generated.template.ref} TILES {generated.tileset_ref} GRID {GRID}",
        "SHAPE " + generated.template.shape.value,
    )
    for line in lines:
        for char in line.upper():
            assert char in render.FONT, f"font has no glyph for {char!r}"
    assert render.FALLBACK_GLYPH in render.FONT
    for char, glyph in render.FONT.items():
        assert len(glyph) == render.GLYPH_H, f"{char!r} is not {render.GLYPH_H} rows"
        assert all(len(row) == render.GLYPH_W for row in glyph), f"{char!r} is not 3 wide"


def test_an_over_long_stamp_is_truncated_not_overflowed(tmp_path):
    """A long template ref must not spill outside the image."""
    generated = make_map()
    generated.tileset_ref = "x" * 400
    data = render_layout(generated, None, tmp_path / "m.png", scale=12)
    width, height, rows = parse_pixels(data)
    assert (width, height) == image_size(GRID, 12)
    # The right-hand margin column stays background.
    assert all(row[width - 1] == PALETTE["background"] for row in rows)


# --------------------------------------------------------------------------
# render.py: the stage 2 picture
# --------------------------------------------------------------------------


def test_the_routed_render_shows_paths_and_every_node(tmp_path):
    scale = 12
    routed = make_routed()
    data = render_routed(routed, tmp_path / "r.png", scale)
    _, _, rows = parse_pixels(data)
    present = {px for row in rows for px in row}

    assert PALETTE["corridor"] in present, "routed paths were not drawn"
    assert PALETTE["grid_line"] in present
    for node in routed.nodes.values():
        key = render.ROLE_KEY[node.role]
        assert PALETTE[key] in present, f"{node.role} node not drawn"
        cx = MARGIN + node.cell[0] * scale + scale // 2
        cy = MARGIN + node.cell[1] * scale + scale // 2
        assert rows[cy][cx] == PALETTE[key], f"{node.id} is not at its own cell"

    # A cell on a routed path, off any node.
    x = MARGIN + 6 * scale + 2
    y = MARGIN + 2 * scale + 2
    assert rows[y][x] == PALETTE["corridor"]


def test_a_routed_layout_without_a_grid_is_refused(tmp_path):
    routed = make_routed()
    routed.grid = 0
    with pytest.raises(ValueError):
        render_routed(routed, tmp_path / "r.png", 12)


def test_render_layout_tolerates_no_tile_database(tmp_path):
    """The database is read for two optional fields; None must still render."""
    with_db = render_layout(make_map(), FakeTileDatabase(), tmp_path / "a.png")
    without = render_layout(make_map(), None, tmp_path / "b.png")
    assert with_db == without  # the ref stamped comes from the map either way
    assert parse_header(without)[:2] == image_size(GRID, 12)


# --------------------------------------------------------------------------
# render.py: the awkward inputs
# --------------------------------------------------------------------------


def test_the_canvas_clips_drawing_and_bounds_checks_reads():
    canvas = render.Canvas(4, 4, PALETTE["background"])
    canvas.fill_rect(-10, -10, 3, 3, PALETTE["exit"])  # wholly off canvas
    canvas.fill_rect(3, 3, 10, 10, PALETTE["exit"])  # clipped to one pixel
    canvas.disc(-2, -2, 1, PALETTE["exit"])
    assert canvas.get_pixel(3, 3) == PALETTE["exit"]
    assert canvas.get_pixel(0, 0) == PALETTE["background"]
    with pytest.raises(IndexError):
        canvas.get_pixel(-1, 0)
    with pytest.raises(IndexError):
        canvas.get_pixel(0, 4)


def test_markers_off_the_edge_of_the_grid_still_render(tmp_path):
    """Stage 5 and 6 own their bounds; the renderer must not be the thing
    that explodes when one of them hands over a cell near or past the edge."""
    generated = make_map()
    generated.set_pieces = [SetPiecePlacement("edge", (GRID - 1, GRID - 1), 0, 4, 4)]
    generated.spawns = [SpawnPack("stray", (GRID + 3, 2), 3, True)]
    generated.checkpoints = [(0, 0)]
    generated.exit_cell = (GRID - 1, 0)
    data = render_layout(generated, None, tmp_path / "m.png", scale=8)
    assert parse_header(data)[:2] == image_size(GRID, 8)


def test_a_map_without_an_entrance_node_still_renders(tmp_path):
    generated = make_map()
    generated.routed.nodes = {
        node_id: node
        for node_id, node in generated.routed.nodes.items()
        if node.role is not Role.ENTRANCE
    }
    data = render_layout(generated, None, tmp_path / "m.png", scale=8)
    assert parse_header(data)[:2] == image_size(GRID, 8)
    assert PALETTE["entrance"] not in colours_in(data)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
