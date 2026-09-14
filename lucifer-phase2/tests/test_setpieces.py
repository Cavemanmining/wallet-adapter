"""Tests for stage 5, set pieces.

Spec: docs/WORLD_BIBLE.md stage 5 and the two navigational tells of section 02.

Stage 5's contract is a handful of invariants rather than a fixed answer, so
most of this file sweeps 200 seeds of three templates -- the shipped dungeon,
the shipped outdoor map, and a hub template written here so the mechanic room
is exercised too -- and re-derives each invariant from the output:

* every footprint lies inside the grid and no two footprints touch;
* each entrance socket sits on its piece's perimeter and opens onto a floor
  cell of the very edge the player arrives by;
* the interior is the one the library draws, cell for cell, in every instance;
* the exit is within 15 m of both its brazier and its landmark;
* exactly eight cells (30 m) of approach floor precede the boss;
* the tiles still tile: no seam breaks, no cell is left unplaced;
* and the whole thing is a pure function of the seed.

The 200 runs per template are built once and shared.  The tile grid is checked
while it exists and then reduced to its packed bytes, so 600 full maps do not
have to be held in memory at once.
"""

from __future__ import annotations

import copy
import functools
import json
import pathlib
import sys
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pytest

# Runnable as ``pytest tests/test_setpieces.py`` from anywhere, not only as
# ``python -m pytest`` from the package root.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lucifer_gen import setpieces as SP  # noqa: E402
from lucifer_gen.contracts import (  # noqa: E402
    SIDES,
    Cell,
    CellKind,
    Role,
    RoutedLayout,
    SetPiecePlacement,
    TerrainPlan,
    neighbour,
)
from lucifer_gen.rooms import RoomLibrary  # noqa: E402
from lucifer_gen.route import route  # noqa: E402
from lucifer_gen.seed import MASK64  # noqa: E402
from lucifer_gen.template import load_builtin_template, template_from_dict  # noqa: E402
from lucifer_gen.tiles import TileDatabase  # noqa: E402
from lucifer_gen.tileize import debug_check, tileize  # noqa: E402
from lucifer_gen.translate import translate  # noqa: E402

#: 200 well spread seeds; the golden-ratio step keeps every seed field moving.
SEEDS: Tuple[int, ...] = tuple((i * 0x9E3779B97F4A7C15) & MASK64 for i in range(200))

#: A hub-shaped dungeon whose mechanic room is required rather than optional,
#: so all three greybox pieces are stamped on every seed.  The shipped crypt
#: template leaves its mechanic node optional and gives it no set piece.
RITUAL_KEEP = {
    "id": "ritual_keep",
    "version": 1,
    "class": "dungeon",
    "shape": "Hub",
    "grid": 48,
    "cell_m": 4,
    "tileset": "crypt_v5",
    "landmarks": ["sunken_bell", "iron_orrery"],
    "nodes": [
        {"id": "gate", "role": "entrance", "anchor": "shape.spoke_n"},
        {"id": "hall", "role": "side", "anchor": "shape.centre"},
        {
            "id": "ritual",
            "role": "mechanic",
            "anchor": "shape.spoke_w",
            "set_piece": "mechanic_ritual",
        },
        {
            "id": "warden",
            "role": "boss",
            "anchor": "shape.spoke_e",
            "set_piece": "crypt_boss_v2",
        },
        {
            "id": "postern",
            "role": "exit",
            "anchor": "shape.spoke_s",
            "set_piece": "exit_brazier",
        },
    ],
    "edges": [
        {"a": "gate", "b": "hall"},
        {"a": "hall", "b": "ritual"},
        {"a": "hall", "b": "warden"},
        {"a": "warden", "b": "postern"},
    ],
}

TEMPLATES = ("crypt", "ashen_ramparts", "ritual_keep")


# --------------------------------------------------------------------------
# Building the sweeps
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def rooms() -> RoomLibrary:
    return RoomLibrary.load()


@functools.lru_cache(maxsize=None)
def tiles() -> TileDatabase:
    return TileDatabase.load()


@functools.lru_cache(maxsize=None)
def library() -> SP.SetPieceLibrary:
    return SP.SetPieceLibrary.load()


@functools.lru_cache(maxsize=None)
def get_template(name: str):
    if name == "ritual_keep":
        return template_from_dict(RITUAL_KEEP, source="tests/ritual_keep")
    return load_builtin_template(name)


@dataclass
class Run:
    """One whole map, with the tile grid boiled down to what tests need."""

    seed: int
    name: str
    routed: RoutedLayout
    plan: TerrainPlan
    result: SP.SetPieceResult
    tile_problems: Tuple[str, ...]
    cells: bytes

    @property
    def why(self) -> str:
        return f"{self.name} seed 0x{self.seed:016X}"


def generate(name: str, seed: int, *, verify_tiles: bool = True) -> Run:
    """Stages 2 to 5 for one seed, exactly as the pipeline runs them.

    ``verify_tiles`` runs stage 4's own ``debug_check`` over the stamped grid.
    It is the slowest thing here, so the determinism sweep -- which compares
    two runs cell for cell anyway -- turns it off for its second run.
    """
    template = get_template(name)
    routed = route(template, seed)
    plan = translate(routed, rooms(), seed)
    grid = tileize(plan, tiles(), template.tile_class, seed)
    result = SP.place_set_pieces(routed, plan, grid, tiles(), seed)
    # Checked here, where the grid is still in hand; the verdict travels on.
    problems = debug_check(grid, plan, tiles(), template.tile_class) if verify_tiles else []
    return Run(
        seed=seed,
        name=name,
        routed=routed,
        plan=plan,
        result=result,
        tile_problems=tuple(problems),
        cells=grid.packed_cells(),
    )


@functools.lru_cache(maxsize=None)
def runs(name: str) -> Tuple[Run, ...]:
    return tuple(generate(name, seed) for seed in SEEDS)


def details(run: Run) -> List[SP.PlacementDetail]:
    return list(run.result.details)


def path_of(run: Run, edge_ids: Tuple[str, str]) -> Sequence[Cell]:
    for edge in run.routed.edges:
        if (edge.a, edge.b) == edge_ids or (edge.b, edge.a) == edge_ids:
            return edge.path
    raise AssertionError(f"no routed edge {edge_ids}")


def reachable_floor(plan: TerrainPlan, start: Cell) -> set:
    """Every floor cell four-connected to ``start``, walked independently."""
    seen = {start}
    stack = [start]
    while stack:
        cell = stack.pop()
        for side in SIDES:
            nxt = neighbour(cell, side)
            if nxt not in seen and plan.is_floor(nxt):
                seen.add(nxt)
                stack.append(nxt)
    return seen


# --------------------------------------------------------------------------
# The library itself
# --------------------------------------------------------------------------


def test_library_carries_the_three_greybox_pieces():
    lib = library()
    for piece_id in ("crypt_boss_v2", "exit_brazier", "mechanic_ritual"):
        assert piece_id in lib, f"{piece_id} is missing from {lib.version}"
    assert lib.by_id("crypt_boss_v2").size == 7
    assert lib.by_id("exit_brazier").size == 3
    assert lib.by_id("mechanic_ritual").size == 5


def test_every_piece_is_square_with_one_socket_on_its_perimeter():
    for piece in library().pieces:
        assert piece.w == piece.h, f"{piece.id} is not square"
        x, y = piece.socket_local_base()
        assert x in (0, piece.w - 1) or y in (0, piece.h - 1), (
            f"{piece.id}'s socket is not on its perimeter"
        )
        assert piece.is_floor_local(piece.socket_local_base())
        # A solid perimeter could wall off a corridor running past the piece.
        for cell in piece.local_cells():
            cx, cy = cell
            if cx in (0, piece.w - 1) or cy in (0, piece.h - 1):
                assert piece.is_floor_local(cell), f"{piece.id} has a solid perimeter"


def test_the_exit_piece_carries_both_tells():
    exit_piece = library().by_id("exit_brazier")
    assert exit_piece.carries(SP.MARKER_BRAZIER)
    assert exit_piece.carries(SP.MARKER_LANDMARK)


def test_a_piece_with_a_misdrawn_socket_is_rejected(tmp_path):
    """The library validates rather than trusting; prove it on bad data."""
    bad = {
        "id": "broken",
        "version": 1,
        "pieces": [
            {
                "id": "wrong_door",
                "w": 3,
                "h": 3,
                "socket": [0, 1],
                "interior": ["...", ".D.", "..."],  # door in the middle
            }
        ],
    }
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SP.SetPieceError):
        SP.SetPieceLibrary.load(path)


def test_a_solid_perimeter_is_rejected(tmp_path):
    bad = {
        "id": "broken",
        "version": 1,
        "pieces": [
            {
                "id": "walled",
                "w": 3,
                "h": 3,
                "socket": [0, 1],
                "interior": [".D#", "...", "..."],
            }
        ],
    }
    path = tmp_path / "walled.json"
    path.write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(SP.SetPieceError):
        SP.SetPieceLibrary.load(path)


# --------------------------------------------------------------------------
# Geometry units
# --------------------------------------------------------------------------


def test_rotate_local_turns_clockwise_and_four_turns_is_identity():
    n = 5
    cells = [(x, y) for y in range(n) for x in range(n)]
    for rot in range(4):
        turned = [SP.rotate_local(c, n, rot) for c in cells]
        assert sorted(turned) == sorted(cells), "a rotation must be a bijection"
    # North-west corner walks round the corners clockwise.
    assert SP.rotate_local((0, 0), n, 1) == (n - 1, 0)
    assert SP.rotate_local((0, 0), n, 2) == (n - 1, n - 1)
    assert SP.rotate_local((0, 0), n, 3) == (0, n - 1)
    for cell in cells:
        once = cell
        for _ in range(4):
            once = SP.rotate_local(once, n, 1)
        assert once == cell


def test_rotating_a_piece_moves_its_socket_round_the_sides():
    piece = library().by_id("crypt_boss_v2")
    for rot in range(4):
        side = piece.socket_side(rot)
        assert side == (piece.socket.side + rot) % 4
        assert piece.rotation_for_side(side) == rot
        lx, ly = piece.socket_local(rot)
        assert lx in (0, piece.w - 1) or ly in (0, piece.h - 1)


def test_thirty_metres_of_approach_is_eight_cells():
    """30 m is 7.5 cells at 4 m per cell, so the ceiling: 8."""
    assert SP.approach_cell_count(4.0) == 8 == SP.BOSS_APPROACH_CELLS
    assert SP.approach_cell_count(3.0) == 10
    assert SP.approach_cell_count(5.0) == 6


# --------------------------------------------------------------------------
# The sweeps
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", TEMPLATES)
def test_every_seed_places_every_declared_piece(name):
    for run in runs(name):
        wanted = sorted(n.id for n in run.routed.nodes.values() if n.set_piece)
        got = sorted(d.node_id for d in details(run))
        assert got == wanted, run.why
        assert run.result.skipped == [], run.why
        assert isinstance(run.result, list)
        assert all(isinstance(p, SetPiecePlacement) for p in run.result)
        assert [d.placement for d in details(run)] == list(run.result), run.why


@pytest.mark.parametrize("name", TEMPLATES)
def test_footprints_are_inside_the_grid(name):
    for run in runs(name):
        for detail in details(run):
            for cell in detail.footprint:
                assert run.plan.inside(cell), f"{run.why}: {detail.piece_id} at {cell}"
            assert len(detail.footprint) == detail.placement.w * detail.placement.h


@pytest.mark.parametrize("name", TEMPLATES)
def test_set_pieces_never_overlap(name):
    for run in runs(name):
        taken: Dict[Cell, str] = {}
        for detail in details(run):
            for cell in detail.footprint:
                assert cell not in taken, (
                    f"{run.why}: {detail.piece_id} overlaps {taken[cell]} at {cell}"
                )
                taken[cell] = detail.piece_id


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_socket_meets_the_arriving_corridor(name):
    """The tell of a correct snap: the door opens onto the edge walked in on."""
    for run in runs(name):
        for detail in details(run):
            piece = library().by_id(detail.piece_id)
            assert not detail.clamped, f"{run.why}: {detail.piece_id} had to be clamped"

            # The socket is where the piece says it is, and on the perimeter.
            local = piece.socket_local(detail.rot)
            assert detail.socket_cell == (
                detail.origin[0] + local[0],
                detail.origin[1] + local[1],
            ), run.why
            assert detail.socket_cell in detail.footprint
            assert run.plan.kind(detail.socket_cell) is CellKind.SET_PIECE, run.why

            # It faces the anchor, the anchor is floor, and the anchor is a
            # cell of the routed edge that reaches this node.
            assert neighbour(detail.socket_cell, detail.socket_side) == detail.anchor_cell
            assert detail.anchor_cell not in detail.footprint, run.why
            assert run.plan.is_floor(detail.anchor_cell), run.why
            assert detail.anchor_cell in path_of(run, detail.edge), run.why
            assert detail.node_id in detail.edge, run.why


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_interior_is_identical_in_every_instance(name):
    """A set piece is set: same cells, same kinds, wherever it lands."""
    signatures: Dict[str, Tuple[CellKind, ...]] = {}
    for run in runs(name):
        for detail in details(run):
            piece = library().by_id(detail.piece_id)
            drawn = tuple(
                piece.kind_local(local) for local in piece.local_cells()
            )
            stamped = tuple(
                run.plan.kind(
                    (
                        detail.origin[0] + SP.rotate_local(local, piece.size, detail.rot)[0],
                        detail.origin[1] + SP.rotate_local(local, piece.size, detail.rot)[1],
                    )
                )
                for local in piece.local_cells()
            )
            assert stamped == drawn, f"{run.why}: {detail.piece_id} was not stamped whole"
            first = signatures.setdefault(detail.piece_id, stamped)
            assert stamped == first, f"{run.why}: {detail.piece_id} differs between maps"


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_exit_sits_within_fifteen_metres_of_its_tells(name):
    for run in runs(name):
        detail = run.result.detail_of_role(Role.EXIT)
        if detail is None:  # every shipped template has an exit piece
            continue
        exit_cell = run.result.exit_cell
        assert exit_cell is not None, run.why
        cell_m = run.routed.template.cell_m
        for kind in (SP.MARKER_BRAZIER, SP.MARKER_LANDMARK):
            placed = run.result.markers_of(kind)
            assert placed, f"{run.why}: no {kind} placed"
            nearest = min(SP.cells_apart_m(exit_cell, m.cell, cell_m) for m in placed)
            assert nearest <= SP.EXIT_TELL_RADIUS_M, (
                f"{run.why}: nearest {kind} is {nearest:.1f} m away"
            )
            for marker in placed:
                assert marker.cell in detail.footprint, run.why
                assert run.plan.is_floor(marker.cell), run.why
        # The brazier is what stage 6 and the client call a checkpoint.
        braziers = [m.cell for m in run.result.markers_of(SP.MARKER_BRAZIER)]
        assert set(braziers) <= set(run.result.checkpoints), run.why


@pytest.mark.parametrize("name", TEMPLATES)
def test_exactly_eight_cells_of_approach_precede_the_boss(name):
    for run in runs(name):
        detail = run.result.detail_of_role(Role.BOSS)
        marked = [
            (x, y)
            for y in range(run.plan.grid)
            for x in range(run.plan.grid)
            if run.plan.kind((x, y)) is CellKind.APPROACH
        ]
        if detail is None:
            assert not marked, run.why
            continue
        assert len(run.result.approach_cells) == SP.BOSS_APPROACH_CELLS, run.why
        assert sorted(marked) == sorted(run.result.approach_cells), run.why
        # The approach starts at the arena door and stays out of the arena.
        assert run.result.approach_cells[0] == detail.anchor_cell, run.why
        footprints = run.result.footprint_cells()
        for cell in run.result.approach_cells:
            assert cell not in footprints, run.why
            assert run.plan.is_floor(cell), run.why
        assert len(set(run.result.approach_cells)) == SP.BOSS_APPROACH_CELLS, run.why
        # Stage 6 must not spawn on any of them.
        assert set(run.result.approach_cells) <= run.result.ambient_exclusions()


@pytest.mark.parametrize("name", TEMPLATES)
def test_checkpoints_are_floor_and_include_the_checkpoint_nodes(name):
    for run in runs(name):
        for cell in run.result.checkpoints:
            assert run.plan.is_floor(cell), run.why
        assert len(set(run.result.checkpoints)) == len(run.result.checkpoints), run.why
        for node in run.routed.nodes.values():
            if node.role is Role.CHECKPOINT:
                assert node.cell in run.result.checkpoints, run.why


@pytest.mark.parametrize("name", TEMPLATES)
def test_stage_five_re_derives_clean(name):
    """``check_placements`` re-reads the plan and must find nothing wrong."""
    for run in runs(name):
        problems = SP.check_placements(run.routed, run.plan, run.result, library())
        assert problems == [], f"{run.why}: {problems[:3]}"


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_grid_still_tiles_after_stamping(name):
    """Stamping re-runs filler and re-tileizes, so stage 4's checks must hold."""
    for run in runs(name):
        assert run.tile_problems == (), f"{run.why}: {list(run.tile_problems)[:3]}"


@pytest.mark.parametrize("name", TEMPLATES)
def test_the_floor_stays_connected(name):
    """A stamped piece must never strand part of the map behind it."""
    for run in runs(name):
        entrance = run.routed.node_of_role(Role.ENTRANCE)
        assert entrance is not None
        reached = reachable_floor(run.plan, entrance.cell)
        for detail in details(run):
            piece = library().by_id(detail.piece_id)
            floor = [
                cell
                for cell, kind in piece.stamp_kinds(detail.origin, detail.rot)
                if kind is not CellKind.EMPTY
            ]
            missing = [c for c in floor if c not in reached]
            assert not missing, (
                f"{run.why}: {detail.piece_id} has floor the entrance cannot reach: "
                f"{missing[:3]}"
            )
        for cell in run.result.checkpoints:
            assert cell in reached, run.why


@pytest.mark.parametrize("name", TEMPLATES)
def test_placements_are_deterministic(name):
    """Same seed, same map: stage 5 draws nothing that is not derived."""
    for run in runs(name):
        again = generate(name, run.seed, verify_tiles=False)
        assert [
            (p.id, p.cell, p.rot, p.w, p.h) for p in again.result
        ] == [(p.id, p.cell, p.rot, p.w, p.h) for p in run.result], run.why
        assert again.result.checkpoints == run.result.checkpoints, run.why
        assert again.result.approach_cells == run.result.approach_cells, run.why
        assert again.result.exit_cell == run.result.exit_cell, run.why
        assert [
            (m.kind, m.cell, m.name) for m in again.result.markers
        ] == [(m.kind, m.cell, m.name) for m in run.result.markers], run.why
        assert again.plan.kinds == run.plan.kinds, run.why
        assert again.cells == run.cells, run.why


def test_a_different_seed_moves_the_pieces():
    """Determinism must not be constancy: the layout really does vary."""
    origins = {
        tuple((d.origin, d.rot) for d in details(run)) for run in runs("crypt")
    }
    assert len(origins) > 20, "200 seeds produced barely any distinct placements"


# --------------------------------------------------------------------------
# Behaviour around the edges of the contract
# --------------------------------------------------------------------------


def _pipeline(template, seed):
    routed = route(template, seed)
    plan = translate(routed, rooms(), seed)
    grid = tileize(plan, tiles(), template.tile_class, seed)
    return routed, plan, grid


def test_an_unknown_piece_is_skipped_or_raises_when_strict():
    data = copy.deepcopy(RITUAL_KEEP)
    for node in data["nodes"]:
        if node["id"] == "ritual":
            node["set_piece"] = "no_such_piece_v9"
    template = template_from_dict(data, source="tests/unknown_piece")
    seed = SEEDS[3]

    routed, plan, grid = _pipeline(template, seed)
    result = SP.place_set_pieces(routed, plan, grid, tiles(), seed)
    assert result.skipped == [("ritual", "no_such_piece_v9")]
    assert [d.node_id for d in result.details] == ["warden", "postern"]
    assert any("no_such_piece_v9" in note for note in result.notes)

    routed, plan, grid = _pipeline(template, seed)
    with pytest.raises(SP.SetPieceError):
        SP.place_set_pieces(routed, plan, grid, tiles(), seed, strict=True)


def test_the_tile_grid_is_optional():
    """A caller may stamp the plan alone, before any tiles exist."""
    template = get_template("crypt")
    seed = SEEDS[7]
    routed = route(template, seed)
    plan = translate(routed, rooms(), seed)
    result = SP.place_set_pieces(routed, plan, None, tiles(), seed)
    assert result.details, "nothing was stamped"
    assert result.retileized == []
    for detail in result.details:
        for cell, kind in library().by_id(detail.piece_id).stamp_kinds(
            detail.origin, detail.rot
        ):
            assert plan.kind(cell) is kind


def test_pieces_are_stamped_boss_first():
    """Order is fixed by role, so the largest footprint picks its ground first."""
    template = get_template("ritual_keep")
    roles = []
    for seed in SEEDS[:10]:
        routed, plan, grid = _pipeline(template, seed)
        result = SP.place_set_pieces(routed, plan, grid, tiles(), seed)
        roles.append([d.role for d in result.details])
    assert all(order == [Role.BOSS, Role.EXIT, Role.MECHANIC] for order in roles)


def test_the_landmark_is_named_from_the_template():
    template = get_template("ritual_keep")
    seen = set()
    for seed in SEEDS[:16]:
        routed, plan, grid = _pipeline(template, seed)
        result = SP.place_set_pieces(routed, plan, grid, tiles(), seed)
        for marker in result.markers_of(SP.MARKER_LANDMARK):
            assert marker.name in template.landmarks
            seen.add(marker.name)
    assert seen, "no landmark was named"


def test_the_markers_survive_the_pipeline():
    """Stage 5's tells must reach the map, not die inside the stage.

    ``pipeline.generate`` used to keep only the placements, the checkpoints
    and the exit cell, so the landmark the seed's set-piece field chooses was
    computed and dropped -- the one consumer of
    ``SeedFields.set_piece_choice`` produced nothing anyone could observe.
    """
    from lucifer_gen.contracts import MapMarker
    from lucifer_gen.pipeline import generate, resolve_rooms, resolve_template

    template = resolve_template("crypt")
    rooms = resolve_rooms(None, template)
    gmap = generate(template, tiles(), rooms, 0)

    assert gmap.markers and all(isinstance(m, MapMarker) for m in gmap.markers)
    kinds = {m.kind for m in gmap.markers}
    assert {SP.MARKER_EXIT, SP.MARKER_BRAZIER, SP.MARKER_LANDMARK} <= kinds

    landmarks = [m for m in gmap.markers if m.kind == SP.MARKER_LANDMARK]
    assert landmarks and all(m.name in template.landmarks for m in landmarks)
    assert all(0 <= m.cell[0] < gmap.tiles.grid for m in gmap.markers)

    # The landmark is a function of the set-piece field, and of nothing else.
    def landmark_of(seed):
        return [
            m.name
            for m in generate(template, tiles(), rooms, seed).markers
            if m.kind == SP.MARKER_LANDMARK
        ]

    assert landmark_of(0) == landmark_of(0x0000_0000_0000_0100)  # routing moved


if __name__ == "__main__":  # pragma: no cover - convenience runner
    raise SystemExit(pytest.main([__file__, "-q"]))
