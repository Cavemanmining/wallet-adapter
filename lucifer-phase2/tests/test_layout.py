"""Tests for stage 6: spawning, the client layout description and its hash.

Spec: docs/WORLD_BIBLE.md stage 6.

Two claims carry the weight here.

*The description round-trips exactly.*  A client rebuilds the grid from the
``cells`` blob, so a single byte lost or reordered is a desync that no later
stage can detect.  The round trip is therefore swept over 200+ synthetic
grids covering every tile id, rotation and flip rather than spot-checked.

*The hash is sensitive to exactly the right things.*  It must move when the
cells, the seed, the template version or the tile database version move, and
must not move otherwise -- including across a rebuild of the same map.

Spawn placement is asserted as properties over many seeds (spacing, entrance
exclusion, blocked kinds, pack size) plus two statistical claims (density and
elite rate) that only mean anything in aggregate.  The synthetic inputs are
built here directly rather than by running stages 1 to 5, so a failure points
at stage 6 and nothing else.
"""

from __future__ import annotations

import base64
import json
import math
import random
import sys
from pathlib import Path

# Runnable as `pytest tests/test_layout.py` or `python3 tests/test_layout.py`
# from anywhere, without depending on how the package root reaches sys.path.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_gen.contracts import (
    CellKind,
    GeneratedMap,
    GraphTemplate,
    Placement,
    Role,
    RoutedLayout,
    RoutedNode,
    SetPiecePlacement,
    Shape,
    SpawnPack,
    SpawnRules,
    TemplateEdge,
    TemplateNode,
    TerrainPlan,
    TileClass,
    TileGrid,
)
from lucifer_gen.layout import (
    BYTES_PER_CELL,
    DESCRIPTION_FIELDS,
    LayoutError,
    LayoutMismatch,
    assert_layout_description,
    build_layout_description,
    hash_algorithm,
    layout_hash,
    pack_cells,
    parse_layout_description,
    recompute_layout_hash,
    unpack_cells,
    verify_layout_description,
)
from lucifer_gen.seed import SeedFields, format_seed, parse_seed
from lucifer_gen.spawn import (
    ATTEMPTS_PER_PACK,
    ATTEMPT_FLOOR,
    BLOCKED_KINDS,
    ENTRANCE_EXCLUSION_M,
    PACK_MAX,
    PACK_MIN,
    PACK_SPACING_M,
    SpawnError,
    candidate_cells,
    place_spawns,
    sigil_product,
    spawn_density,
    tier_multiplier,
    walkable_cells,
)

#: How many synthetic grids the round-trip sweep covers.  The spec asks for
#: at least 200.
ROUND_TRIPS = 240

#: Seeds used by the statistical spawn sweeps.  Only the tile field (bits
#: 32-63) feeds stage 6, so the seeds must differ up there to differ at all.
SPAWN_SEEDS = tuple((i << 32) | 0xABCD for i in range(1, 201))


# --------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------


def make_template(
    *,
    tile_class: TileClass = TileClass.DUNGEON,
    grid: int = 24,
    cell_m: float = 4.0,
    base_density: float = 0.018,
    elite_rate: float = 0.12,
    version: int = 3,
) -> GraphTemplate:
    return GraphTemplate(
        id="synthetic",
        version=version,
        tile_class=tile_class,
        shape=Shape.I,
        grid=grid,
        cell_m=cell_m,
        nodes=(
            TemplateNode("in", Role.ENTRANCE, "shape.start"),
            TemplateNode("out", Role.EXIT, "shape.end"),
        ),
        edges=(TemplateEdge("in", "out"),),
        tileset="synthetic_v1",
        spawn=SpawnRules(base_density=base_density, elite_rate=elite_rate),
    )


def make_routed(template: GraphTemplate, entrance=(1, 1), exit_cell=None) -> RoutedLayout:
    nodes = {"in": RoutedNode("in", Role.ENTRANCE, entrance)}
    if exit_cell is not None:
        nodes["out"] = RoutedNode("out", Role.EXIT, exit_cell)
    return RoutedLayout(
        seed=0, template=template, grid=template.grid, nodes=nodes, edges=[]
    )


def open_map(grid: int, kind: CellKind = CellKind.ROOM):
    """A fully walkable square plan plus a matching tile grid."""
    plan = TerrainPlan.blank(grid)
    tiles = TileGrid.blank(grid)
    for y in range(grid):
        for x in range(grid):
            plan.set_kind((x, y), kind)
            tiles.put((x, y), Placement(1, 0, False), True)
    return plan, tiles


def random_tile_grid(grid: int, rng: random.Random) -> TileGrid:
    """A tile grid whose every cell is a random, legal placement."""
    tiles = TileGrid.blank(grid)
    for y in range(grid):
        for x in range(grid):
            placement = Placement(
                rng.randrange(0, 256), rng.randrange(0, 4), bool(rng.getrandbits(1))
            )
            tiles.put((x, y), placement, bool(rng.getrandbits(1)))
    return tiles


def make_map(
    rng: random.Random,
    *,
    grid: int = 8,
    seed: int = 0x0123456789ABCDEF,
    template: GraphTemplate = None,
    tileset_ref: str = "greybox@1",
) -> GeneratedMap:
    template = template or make_template(grid=grid)
    tiles = random_tile_grid(grid, rng)
    plan = TerrainPlan.blank(grid)
    return GeneratedMap(
        seed=seed,
        template=template,
        tileset_ref=tileset_ref,
        routed=make_routed(template),
        terrain=plan,
        tiles=tiles,
        set_pieces=[
            SetPiecePlacement("crypt_boss_v2", (2, 3), rot=1, w=4, h=4),
            SetPiecePlacement("exit_brazier", (5, 6), rot=0, w=2, h=2),
        ],
        spawns=[
            SpawnPack("crypt_ghoul_band", (4, 4), 5, elite=False),
            SpawnPack("rot_choir", (7, 1), 8, elite=True),
        ],
        exit_cell=(5, 6),
        checkpoints=[(1, 1), (5, 6)],
    )


# --------------------------------------------------------------------------
# Packing and round-tripping
# --------------------------------------------------------------------------


def test_packed_blob_is_two_bytes_per_cell():
    for grid in (1, 2, 5, 16, 48):
        rng = random.Random(grid)
        tiles = random_tile_grid(grid, rng)
        blob = pack_cells(tiles)
        assert len(blob) == BYTES_PER_CELL * grid * grid


def test_description_payload_length_before_encoding():
    """The base64 must decode to exactly 2 * grid * grid bytes."""
    for grid in (1, 4, 9, 48):
        gm = make_map(random.Random(grid), grid=grid)
        desc = build_layout_description(gm)
        raw = base64.b64decode(desc["cells"], validate=True)
        assert len(raw) == BYTES_PER_CELL * grid * grid
        assert len(raw) == BYTES_PER_CELL * desc["grid"] * desc["grid"]


def test_round_trip_is_exact_over_many_grids():
    """Spec stage 6: a client rebuild must be verifiable cell for cell."""
    rng = random.Random(20250913)
    for i in range(ROUND_TRIPS):
        grid = 1 + (i % 12)
        gm = make_map(rng, grid=grid, seed=rng.getrandbits(64))
        desc = build_layout_description(gm)

        placements = parse_layout_description(desc)
        assert len(placements) == grid * grid

        for y in range(grid):
            for x in range(grid):
                assert placements[y * grid + x] == gm.tiles.at((x, y)), (i, x, y)

        # And the description verifies against its own hash.
        assert verify_layout_description(desc)
        assert_layout_description(desc)


def test_round_trip_survives_json():
    gm = make_map(random.Random(7), grid=12)
    desc = build_layout_description(gm)
    reloaded = json.loads(json.dumps(desc))
    assert reloaded == desc
    assert parse_layout_description(reloaded) == parse_layout_description(desc)
    assert verify_layout_description(reloaded)


def test_unpack_rejects_a_wrong_sized_blob():
    with pytest.raises(LayoutError):
        unpack_cells(b"\x01\x02", 2)
    with pytest.raises(LayoutError):
        unpack_cells(b"", 0)


def test_pack_rejects_an_unfilled_cell():
    tiles = TileGrid.blank(2)
    tiles.put((0, 0), Placement(1), True)
    with pytest.raises(LayoutError):
        pack_cells(tiles)


def test_every_placement_value_survives_the_round_trip():
    """All 256 ids x 4 rotations x 2 flips, exhaustively."""
    values = [
        Placement(tile_id, rot, flip)
        for tile_id in range(256)
        for rot in range(4)
        for flip in (False, True)
    ]
    grid = 46  # 46 * 46 = 2116 >= 2048 placements
    tiles = TileGrid.blank(grid)
    for index in range(grid * grid):
        tiles.put(
            (index % grid, index // grid), values[index % len(values)], True
        )
    blob = pack_cells(tiles)
    assert unpack_cells(blob, grid) == [
        values[i % len(values)] for i in range(grid * grid)
    ]


# --------------------------------------------------------------------------
# The description's shape
# --------------------------------------------------------------------------


def test_description_has_exactly_the_spec_fields_in_order():
    desc = build_layout_description(make_map(random.Random(1)))
    assert tuple(desc) == DESCRIPTION_FIELDS


def test_description_field_values():
    template = make_template(version=9)
    gm = make_map(
        random.Random(3), grid=8, seed=0xDEADBEEFCAFE1234, template=template
    )
    desc = build_layout_description(gm)

    assert desc["seed"] == "0xDEADBEEFCAFE1234"
    assert desc["seed"] == format_seed(gm.seed)
    assert parse_seed(desc["seed"]) == gm.seed
    assert desc["template"] == "synthetic@9"
    assert desc["tiles"] == "greybox@1"
    assert desc["grid"] == 8
    assert desc["layout_hash"].split(":", 1)[0] == hash_algorithm()
    assert desc["exit_cell"] == [5, 6]
    assert desc["checkpoints"] == [[1, 1], [5, 6]]
    assert desc["set_pieces"][0] == {
        "id": "crypt_boss_v2",
        "cell": [2, 3],
        "rot": 1,
        "w": 4,
        "h": 4,
    }
    assert desc["spawns"][1] == {
        "pack": "rot_choir",
        "cell": [7, 1],
        "count": 8,
        "elite": True,
    }


def test_description_rejects_an_inconsistent_map():
    gm = make_map(random.Random(5), grid=6)
    gm.terrain = TerrainPlan.blank(7)
    with pytest.raises(LayoutError):
        build_layout_description(gm)

    gm = make_map(random.Random(5), grid=6)
    gm.tileset_ref = ""
    with pytest.raises(LayoutError):
        build_layout_description(gm)


def test_parse_rejects_a_broken_description():
    desc = build_layout_description(make_map(random.Random(11), grid=4))

    missing = dict(desc)
    del missing["cells"]
    with pytest.raises(LayoutError):
        parse_layout_description(missing)

    wrong_grid = dict(desc, grid=5)
    with pytest.raises(LayoutError):
        parse_layout_description(wrong_grid)

    not_base64 = dict(desc, cells="not base64!!")
    with pytest.raises(LayoutError):
        parse_layout_description(not_base64)

    with pytest.raises(LayoutError):
        parse_layout_description(["not", "a", "mapping"])


# --------------------------------------------------------------------------
# The hash
# --------------------------------------------------------------------------


def test_hash_names_its_algorithm():
    value = layout_hash(b"\x00\x01", 1, "t@1", "tiles@1")
    algo, _, hexdigest = value.partition(":")
    assert algo == "blake2b"
    assert algo == hash_algorithm()
    assert len(hexdigest) == 64
    int(hexdigest, 16)  # must be hex


def test_hash_does_not_depend_on_an_optional_wheel():
    """The same bytes must hash the same way on every host.

    ``layout_hash`` is the value a client compares to decide whether it is
    looking at the same map as the server, so it may not vary with what is
    installed.  It used to prefer an optional ``blake3`` wheel and fall back
    to blake2b, which meant a build box and a laptop disagreed about a map
    they both generated identically.
    """
    import lucifer_gen.layout as layout_module

    assert not hasattr(layout_module, "_blake3")
    assert layout_module.HASH_ALGORITHM == "blake2b"
    assert hash_algorithm() == "blake2b"

    gm = make_map(random.Random(77), grid=6)
    desc = build_layout_description(gm)
    assert desc["layout_hash"].startswith("blake2b:")
    assert len(desc["layout_hash"].split(":", 1)[1]) == 64
    assert verify_layout_description(desc)

    # And it is the plain blake2b of the framed payload, computable by any
    # host with nothing but the standard library.
    import hashlib

    from lucifer_gen.layout import HASH_DOMAIN, HASH_SIZE, _framed, pack_cells
    from lucifer_gen.seed import format_seed

    payload = _framed(
        HASH_DOMAIN,
        pack_cells(gm.tiles),
        format_seed(gm.seed).encode("utf-8"),
        gm.template.ref.encode("utf-8"),
        gm.tileset_ref.encode("utf-8"),
    )
    expected = hashlib.blake2b(payload, digest_size=HASH_SIZE).hexdigest()
    assert desc["layout_hash"] == "blake2b:" + expected


def test_hash_is_stable_for_the_same_inputs():
    a = layout_hash(b"\x01\x02\x03\x04", 0x1234, "crypt@3", "greybox@1")
    b = layout_hash(bytearray(b"\x01\x02\x03\x04"), "0x1234", "crypt@3", "greybox@1")
    assert a == b

    rng_seed = 99
    first = build_layout_description(make_map(random.Random(rng_seed), grid=10))
    second = build_layout_description(make_map(random.Random(rng_seed), grid=10))
    assert first == second
    assert first["layout_hash"] == second["layout_hash"]


def test_hash_moves_when_any_hashed_input_moves():
    base_cells = bytes(range(32))
    base = layout_hash(base_cells, 0x1122334455667788, "crypt@3", "greybox@1")

    # cells
    for index in (0, 7, 31):
        mutated = bytearray(base_cells)
        mutated[index] ^= 0x01
        assert layout_hash(bytes(mutated), 0x1122334455667788, "crypt@3", "greybox@1") != base

    # seed
    assert layout_hash(base_cells, 0x1122334455667789, "crypt@3", "greybox@1") != base

    # template version, and template id
    assert layout_hash(base_cells, 0x1122334455667788, "crypt@4", "greybox@1") != base
    assert layout_hash(base_cells, 0x1122334455667788, "cryp@3", "greybox@1") != base

    # tile database version, and its id
    assert layout_hash(base_cells, 0x1122334455667788, "crypt@3", "greybox@2") != base
    assert layout_hash(base_cells, 0x1122334455667788, "crypt@3", "greybo@1") != base


def test_hash_framing_defeats_field_run_together():
    """Moving a character across a field boundary must change the hash."""
    cells = b"\x00\x00"
    assert layout_hash(cells, 1, "a@1", "b@2") != layout_hash(cells, 1, "a@1b", "@2")


def test_description_hash_tracks_each_field():
    grid = 9
    rng_seed = 4242
    base = build_layout_description(make_map(random.Random(rng_seed), grid=grid))

    # A different tile in one cell.
    gm = make_map(random.Random(rng_seed), grid=grid)
    old = gm.tiles.at((3, 3))
    gm.tiles.put((3, 3), Placement((old.tile_id + 1) % 256, old.rot, old.flip), True)
    assert build_layout_description(gm)["layout_hash"] != base["layout_hash"]

    # Only the rotation differs.
    gm = make_map(random.Random(rng_seed), grid=grid)
    old = gm.tiles.at((0, 0))
    gm.tiles.put((0, 0), Placement(old.tile_id, (old.rot + 1) % 4, old.flip), True)
    assert build_layout_description(gm)["layout_hash"] != base["layout_hash"]

    # Only the flip differs.
    gm = make_map(random.Random(rng_seed), grid=grid)
    old = gm.tiles.at((1, 0))
    gm.tiles.put((1, 0), Placement(old.tile_id, old.rot, not old.flip), True)
    assert build_layout_description(gm)["layout_hash"] != base["layout_hash"]

    # Seed.
    gm = make_map(random.Random(rng_seed), grid=grid, seed=0x0123456789ABCDEE)
    assert build_layout_description(gm)["layout_hash"] != base["layout_hash"]

    # Template version.
    gm = make_map(
        random.Random(rng_seed), grid=grid, template=make_template(grid=grid, version=4)
    )
    assert build_layout_description(gm)["layout_hash"] != base["layout_hash"]

    # Tile database version.
    gm = make_map(random.Random(rng_seed), grid=grid, tileset_ref="greybox@2")
    assert build_layout_description(gm)["layout_hash"] != base["layout_hash"]

    # Spawns and set pieces are *not* hashed: the spec lists cells, seed,
    # template and tile database, and spawns derive from those same inputs.
    gm = make_map(random.Random(rng_seed), grid=grid)
    gm.spawns = []
    gm.set_pieces = []
    assert build_layout_description(gm)["layout_hash"] == base["layout_hash"]


def test_verify_catches_tampering():
    desc = build_layout_description(make_map(random.Random(13), grid=10))
    assert verify_layout_description(desc)

    raw = bytearray(base64.b64decode(desc["cells"]))
    raw[5] ^= 0xFF
    tampered = dict(desc, cells=base64.b64encode(bytes(raw)).decode("ascii"))
    assert not verify_layout_description(tampered)
    with pytest.raises(LayoutMismatch):
        assert_layout_description(tampered)

    for field, value in (
        ("seed", "0x0000000000000001"),
        ("template", "synthetic@99"),
        ("tiles", "greybox@99"),
        ("layout_hash", "blake2b:" + "0" * 64),
    ):
        assert not verify_layout_description(dict(desc, **{field: value}))


def test_recompute_matches_build():
    desc = build_layout_description(make_map(random.Random(21), grid=11))
    assert recompute_layout_hash(desc) == desc["layout_hash"]


# --------------------------------------------------------------------------
# Density arithmetic
# --------------------------------------------------------------------------


def test_tier_multiplier_is_linear_from_one_to_two():
    assert tier_multiplier(1) == pytest.approx(1.0)
    assert tier_multiplier(15) == pytest.approx(2.0)
    assert tier_multiplier(8) == pytest.approx(1.5)
    for tier in range(1, 15):
        step = tier_multiplier(tier + 1) - tier_multiplier(tier)
        assert step == pytest.approx(1.0 / 14.0)
    # Clamped outside the band the spec defines.
    assert tier_multiplier(0) == pytest.approx(1.0)
    assert tier_multiplier(-5) == pytest.approx(1.0)
    assert tier_multiplier(99) == pytest.approx(2.0)


def test_sigil_modifiers_fold_to_a_product():
    assert sigil_product(None) == 1.0
    assert sigil_product(1.25) == pytest.approx(1.25)
    assert sigil_product([1.5, 2.0]) == pytest.approx(3.0)
    assert sigil_product({"wrath": 2.0, "famine": 0.5}) == pytest.approx(1.0)
    # Mapping order must not matter.
    assert sigil_product({"a": 1.1, "b": 1.3}) == sigil_product({"b": 1.3, "a": 1.1})
    with pytest.raises(SpawnError):
        sigil_product([-1.0])
    with pytest.raises(SpawnError):
        sigil_product(True)


def test_density_formula():
    template = make_template(base_density=0.02)
    assert spawn_density(template.spawn, 1, None) == pytest.approx(0.02)
    assert spawn_density(template.spawn, 15, None) == pytest.approx(0.04)
    assert spawn_density(template.spawn, 15, 0.5) == pytest.approx(0.02)
    assert spawn_density(template.spawn, 8, [2.0, 1.5]) == pytest.approx(0.02 * 1.5 * 3.0)


# --------------------------------------------------------------------------
# Spawn placement
# --------------------------------------------------------------------------


def spawn_on_open_map(seed, *, grid=32, tier=1, sigils=None, template=None, entrance=(0, 0)):
    template = template or make_template(grid=grid)
    plan, tiles = open_map(grid)
    routed = make_routed(template, entrance=entrance)
    packs = place_spawns(plan, tiles, routed, template, tier, sigils, seed)
    return plan, tiles, routed, template, packs


def test_placement_obeys_every_hard_rule():
    grid = 32
    for seed in SPAWN_SEEDS[:40]:
        plan, tiles, routed, template, packs = spawn_on_open_map(seed, grid=grid)
        cell_m = template.cell_m
        entrance = routed.nodes["in"].cell
        for i, pack in enumerate(packs):
            assert PACK_MIN <= pack.count <= PACK_MAX
            assert tiles.is_walkable(pack.cell)
            assert plan.kind(pack.cell) not in BLOCKED_KINDS
            assert math.dist(pack.cell, entrance) * cell_m >= ENTRANCE_EXCLUSION_M - 1e-9
            for other in packs[i + 1 :]:
                gap = math.dist(pack.cell, other.cell) * cell_m
                assert gap >= PACK_SPACING_M - 1e-9, (pack, other, gap)
            assert isinstance(pack.pack, str) and pack.pack


def test_placement_avoids_approach_and_set_piece_cells():
    grid = 24
    template = make_template(grid=grid)
    plan, tiles = open_map(grid)
    for y in range(grid):
        for x in range(grid):
            if x < grid // 2:
                plan.set_kind((x, y), CellKind.APPROACH)
            elif y >= grid - 3:
                plan.set_kind((x, y), CellKind.SET_PIECE)
    routed = make_routed(template, entrance=(0, 0))
    seen = 0
    for seed in SPAWN_SEEDS[:40]:
        packs = place_spawns(plan, tiles, routed, template, 15, None, seed)
        seen += len(packs)
        for pack in packs:
            assert plan.kind(pack.cell) not in BLOCKED_KINDS
            assert pack.cell[0] >= grid // 2 and pack.cell[1] < grid - 3
    assert seen > 0, "the fixture excluded everything; the test proved nothing"


def test_placement_is_deterministic_and_seed_sensitive():
    a = spawn_on_open_map(SPAWN_SEEDS[0])[-1]
    b = spawn_on_open_map(SPAWN_SEEDS[0])[-1]
    assert a == b

    # Passing SeedFields rather than an int must give the same answer.
    c = spawn_on_open_map(SeedFields.parse(SPAWN_SEEDS[0]))[-1]
    assert a == c

    # Different tile fields must generally give different layouts.
    differing = sum(
        1 for seed in SPAWN_SEEDS[1:21] if spawn_on_open_map(seed)[-1] != a
    )
    assert differing >= 18

    # Routing bits (8-31) belong to stage 2 and must not move a monster.
    same_tiles = SPAWN_SEEDS[0] ^ 0x00FF_FF00
    assert spawn_on_open_map(same_tiles)[-1] == a


def test_entrance_exclusion_holds_for_entrances_all_over_the_map():
    grid = 20
    template = make_template(grid=grid)
    plan, tiles = open_map(grid)
    for entrance in ((0, 0), (10, 10), (19, 0), (19, 19), (5, 14)):
        routed = make_routed(template, entrance=entrance)
        for seed in SPAWN_SEEDS[:20]:
            for pack in place_spawns(plan, tiles, routed, template, 15, None, seed):
                gap = math.dist(pack.cell, entrance) * template.cell_m
                assert gap >= ENTRANCE_EXCLUSION_M - 1e-9, (entrance, pack, gap)


def test_missing_entrance_only_drops_the_exclusion():
    grid = 16
    template = make_template(grid=grid)
    plan, tiles = open_map(grid)
    empty = RoutedLayout(seed=0, template=template, grid=grid, nodes={}, edges=[])
    packs = place_spawns(plan, tiles, empty, template, 15, None, SPAWN_SEEDS[0])
    assert packs  # nothing is excluded, so the density target should be met
    none_routed = place_spawns(plan, tiles, None, template, 15, None, SPAWN_SEEDS[0])
    assert none_routed == packs


def test_density_is_met_on_a_map_with_room_for_it():
    """The mean pack count should sit on density * walkable cells."""
    grid = 48
    template = make_template(grid=grid)
    plan, tiles = open_map(grid)
    routed = make_routed(template, entrance=(0, 0))
    walkable = len(walkable_cells(tiles))
    expected = spawn_density(template.spawn, 1, None) * walkable

    counts = []
    for seed in SPAWN_SEEDS:
        counts.append(len(place_spawns(plan, tiles, routed, template, 1, None, seed)))
    mean = sum(counts) / len(counts)
    # Expected is ~41 packs on 2304 cells; spacing is 3 cells, so the map has
    # ample room and the only variance is the fractional coin flip.
    assert mean == pytest.approx(expected, abs=1.0), (mean, expected)
    assert min(counts) >= math.floor(expected) - 1


def test_density_scales_with_tier_and_sigils():
    grid = 48
    template = make_template(grid=grid)
    plan, tiles = open_map(grid)
    routed = make_routed(template, entrance=(0, 0))

    def mean_for(tier, sigils):
        total = 0
        for seed in SPAWN_SEEDS[:60]:
            total += len(place_spawns(plan, tiles, routed, template, tier, sigils, seed))
        return total / 60.0

    at_1 = mean_for(1, None)
    at_15 = mean_for(15, None)
    halved = mean_for(1, 0.5)
    assert at_15 == pytest.approx(2.0 * at_1, rel=0.15)
    assert halved == pytest.approx(0.5 * at_1, rel=0.2)


def test_zero_density_places_nothing():
    grid = 16
    template = make_template(grid=grid, base_density=0.0)
    plan, tiles = open_map(grid)
    routed = make_routed(template, entrance=(0, 0))
    assert place_spawns(plan, tiles, routed, template, 15, None, SPAWN_SEEDS[0]) == []

    live = make_template(grid=grid)
    assert place_spawns(plan, tiles, routed, live, 15, 0.0, SPAWN_SEEDS[0]) == []


def test_elite_rate_holds_in_aggregate():
    grid = 48
    rate = 0.12
    template = make_template(grid=grid, elite_rate=rate)
    plan, tiles = open_map(grid)
    routed = make_routed(template, entrance=(0, 0))

    packs = 0
    elites = 0
    for seed in SPAWN_SEEDS:
        placed = place_spawns(plan, tiles, routed, template, 15, None, seed)
        packs += len(placed)
        elites += sum(1 for p in placed if p.elite)
    assert packs > 5000, packs
    share = elites / packs
    # ~8000 packs at p=0.12 has a standard error near 0.004, so 0.02 is wide.
    assert share == pytest.approx(rate, abs=0.02), (share, packs, elites)


def test_elite_rate_of_zero_and_one():
    grid = 32
    plan, tiles = open_map(grid)

    none_template = make_template(grid=grid, elite_rate=0.0)
    routed = make_routed(none_template, entrance=(0, 0))
    for seed in SPAWN_SEEDS[:20]:
        placed = place_spawns(plan, tiles, routed, none_template, 15, None, seed)
        assert not any(p.elite for p in placed)

    all_template = make_template(grid=grid, elite_rate=1.0)
    for seed in SPAWN_SEEDS[:20]:
        placed = place_spawns(plan, tiles, routed, all_template, 15, None, seed)
        assert placed and all(p.elite for p in placed)


def test_terminates_on_small_and_hostile_maps():
    """A map with no room must return quickly rather than spin."""
    for grid in range(1, 7):
        template = make_template(grid=grid)
        plan, tiles = open_map(grid)
        routed = make_routed(template, entrance=(0, 0))
        for seed in SPAWN_SEEDS[:10]:
            packs = place_spawns(plan, tiles, routed, template, 15, None, seed)
            for i, pack in enumerate(packs):
                for other in packs[i + 1 :]:
                    assert math.dist(pack.cell, other.cell) * 4.0 >= PACK_SPACING_M - 1e-9

    # A one-cell-wide corridor: dense target, almost no legal room.
    grid = 48
    template = make_template(grid=grid, base_density=0.2)
    plan = TerrainPlan.blank(grid)
    tiles = TileGrid.blank(grid)
    for x in range(grid):
        plan.set_kind((x, 24), CellKind.CORRIDOR)
        tiles.put((x, 24), Placement(1), True)
    routed = make_routed(template, entrance=(0, 24))
    packs = place_spawns(plan, tiles, routed, template, 15, None, SPAWN_SEEDS[0])
    assert packs
    assert len(packs) <= grid // 3 + 1
    for i, pack in enumerate(packs):
        for other in packs[i + 1 :]:
            assert abs(pack.cell[0] - other.cell[0]) >= 3


def test_no_walkable_cells_places_nothing():
    grid = 8
    template = make_template(grid=grid)
    plan = TerrainPlan.blank(grid)
    tiles = TileGrid.blank(grid)
    for y in range(grid):
        for x in range(grid):
            tiles.put((x, y), Placement(0), False)
    routed = make_routed(template, entrance=(0, 0))
    assert place_spawns(plan, tiles, routed, template, 15, None, SPAWN_SEEDS[0]) == []


def test_candidates_are_row_major_and_exclusions_apply():
    grid = 12
    template = make_template(grid=grid)
    plan, tiles = open_map(grid)
    plan.set_kind((11, 11), CellKind.SET_PIECE)
    cells = candidate_cells(plan, tiles, (0, 0), template.cell_m)
    assert cells == sorted(cells, key=lambda c: (c[1], c[0]))
    assert (11, 11) not in cells
    assert (0, 0) not in cells and (4, 0) not in cells  # inside 20 m
    assert (5, 0) in cells  # exactly 20 m away, which is not "within" it


def test_pack_families_follow_the_tile_class():
    grid = 32
    plan, tiles = open_map(grid)
    dungeon = make_template(grid=grid, tile_class=TileClass.DUNGEON)
    outdoor = make_template(grid=grid, tile_class=TileClass.OUTDOOR)
    routed = make_routed(dungeon, entrance=(0, 0))

    dungeon_names = {
        p.pack
        for seed in SPAWN_SEEDS[:20]
        for p in place_spawns(plan, tiles, routed, dungeon, 15, None, seed)
    }
    outdoor_names = {
        p.pack
        for seed in SPAWN_SEEDS[:20]
        for p in place_spawns(plan, tiles, routed, outdoor, 15, None, seed)
    }
    assert dungeon_names and outdoor_names
    assert not (dungeon_names & outdoor_names)


def test_rejects_a_degenerate_grid_or_cell_size():
    template = make_template(grid=4)
    plan, tiles = open_map(4)
    with pytest.raises(SpawnError):
        place_spawns(plan, TileGrid.blank(0), None, template, 1, None, 1)
    bad_cell_m = make_template(grid=4, cell_m=0.0)
    with pytest.raises(SpawnError):
        place_spawns(plan, tiles, None, bad_cell_m, 1, None, 1)


def test_attempt_cap_is_documented_and_bounded():
    """The cap is a function of the target, not of the map size."""
    assert ATTEMPT_FLOOR > 0 and ATTEMPTS_PER_PACK > 0
    grid = 48
    template = make_template(grid=grid, base_density=0.001)
    plan, tiles = open_map(grid)
    routed = make_routed(template, entrance=(0, 0))
    packs = place_spawns(plan, tiles, routed, template, 1, None, SPAWN_SEEDS[0])
    target_upper = math.ceil(0.001 * grid * grid)
    assert len(packs) <= target_upper


# --------------------------------------------------------------------------
# The two halves together
# --------------------------------------------------------------------------


def test_spawns_survive_into_the_description():
    grid = 32
    template = make_template(grid=grid)
    plan, tiles = open_map(grid)
    routed = make_routed(template, entrance=(0, 0), exit_cell=(31, 31))
    packs = place_spawns(plan, tiles, routed, template, 9, [1.25], SPAWN_SEEDS[3])
    assert packs

    gm = GeneratedMap(
        seed=SPAWN_SEEDS[3],
        template=template,
        tileset_ref="greybox@1",
        routed=routed,
        terrain=plan,
        tiles=tiles,
        set_pieces=[SetPiecePlacement("exit_brazier", (30, 30), 0, 2, 2)],
        spawns=packs,
        exit_cell=(31, 31),
        checkpoints=[(31, 31)],
    )
    desc = build_layout_description(gm)
    assert len(desc["spawns"]) == len(packs)
    assert desc["spawns"][0]["pack"] == packs[0].pack
    assert [d["elite"] for d in desc["spawns"]] == [p.elite for p in packs]
    assert verify_layout_description(desc)
    assert json.loads(json.dumps(desc)) == desc


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
