"""Tests for stages 1 and 2: shapes, graph templates, node placement, routing.

Spec: docs/WORLD_BIBLE.md stages 1 and 2.
"""

from __future__ import annotations

import copy
import pathlib
import sys
from fractions import Fraction

import pytest

# Allow ``pytest tests/test_route.py`` from anywhere, not only ``python -m
# pytest`` from the package root.
_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lucifer_gen import shapes  # noqa: E402
from lucifer_gen.contracts import (  # noqa: E402
    GraphTemplate,
    Role,
    Shape,
    SWAPPABLE_SHAPES,
    TemplateNode,
    TileClass,
)
from lucifer_gen.route import (  # noqa: E402
    ADJACENCY_PENALTY,
    MARGIN,
    MAX_JITTER,
    MIN_SEPARATION,
    astar_path,
    route,
)
from lucifer_gen.seed import MASK64, SeedFields  # noqa: E402
from lucifer_gen.template import (  # noqa: E402
    TemplateError,
    builtin_template_names,
    load_builtin_template,
    template_from_dict,
    template_to_dict,
)

# --------------------------------------------------------------------------
# Fixtures and helpers
# --------------------------------------------------------------------------

#: 200 well spread seeds. The golden-ratio step keeps every seed field moving,
#: so rotation, mirror, swap and the routing bits are all exercised.
SEEDS = tuple((i * 0x9E3779B97F4A7C15) & MASK64 for i in range(200))

#: One seed per (rotation, mirror, swap) combination, on a fixed routing field.
TRANSFORM_SEEDS = tuple(
    rot | (mirror << 2) | (swap << 3) | (0x5A5A5A << 8)
    for rot in range(4)
    for mirror in (0, 1)
    for swap in (0, 1)
)

BUILTINS = ("crypt", "ashen_ramparts")


@pytest.fixture(scope="module")
def crypt() -> GraphTemplate:
    return load_builtin_template("crypt")


@pytest.fixture(scope="module")
def ramparts() -> GraphTemplate:
    return load_builtin_template("ashen_ramparts")


def layout_signature(layout):
    """Everything stage 2 decides, in a comparable form."""
    return (
        tuple(sorted((k, v.cell, v.role.value) for k, v in layout.nodes.items())),
        tuple((e.a, e.b, tuple(e.path)) for e in layout.edges),
    )


def good_template_dict() -> dict:
    """A minimal valid template; each bad-case test breaks one thing in it."""
    return {
        "id": "probe",
        "version": 1,
        "class": "dungeon",
        "shape": "I",
        "grid": 48,
        "cell_m": 4,
        "tileset": "probe_v1",
        "nodes": [
            {"id": "in", "role": "entrance", "anchor": "shape.start"},
            {"id": "mid", "role": "side", "anchor": "shape.bend"},
            {"id": "side", "role": "side", "anchor": "shape.pocket", "optional": True},
            {"id": "out", "role": "exit", "anchor": "shape.end"},
        ],
        "edges": [
            {"a": "in", "b": "mid"},
            {"a": "mid", "b": "out"},
            {"a": "mid", "b": "side", "optional": True},
        ],
    }


def connected_ids(layout) -> set:
    """Node ids reachable from the entrance through the routed edges."""
    adj: dict = {}
    for edge in layout.edges:
        adj.setdefault(edge.a, []).append(edge.b)
        adj.setdefault(edge.b, []).append(edge.a)
    entrance = layout.node_of_role(Role.ENTRANCE)
    assert entrance is not None
    seen = {entrance.id}
    stack = [entrance.id]
    while stack:
        current = stack.pop()
        for nxt in sorted(adj.get(current, ())):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


# --------------------------------------------------------------------------
# Stage 2, part one: shapes
# --------------------------------------------------------------------------


class TestShapes:
    def test_every_shape_defines_the_required_anchors(self):
        for shape in Shape:
            macro = shapes.get_shape(shape)
            for name in shapes.REQUIRED_ANCHORS:
                point = macro.anchor(name)
                assert 0 <= point.x <= 1 and 0 <= point.y <= 1
            # The five the spec names are reachable as attributes too.
            assert macro.start == macro.anchors["start"]
            assert macro.bend == macro.anchors["bend"]
            assert macro.pocket == macro.anchors["pocket"]
            assert macro.end == macro.anchors["end"]
            assert macro.centre == macro.anchors["centre"]

    def test_hub_exposes_spoke_anchors(self):
        hub = shapes.get_shape(Shape.HUB)
        assert len(hub.spokes) == 8
        for name in hub.spokes:
            assert name in hub.anchors
        for other in Shape:
            if other is not Shape.HUB:
                assert shapes.get_shape(other).spokes == ()

    def test_anchors_are_exact_rationals(self):
        # Floats would make the transform algebra inexact; see shapes.py.
        for shape in Shape:
            for point in shapes.get_shape(shape).anchors.values():
                assert isinstance(point.x, Fraction)
                assert isinstance(point.y, Fraction)

    @pytest.mark.parametrize("shape", list(Shape))
    def test_half_turn_applied_twice_is_identity(self, shape):
        for name, point in sorted(shapes.get_shape(shape).anchors.items()):
            once = shapes.transform_point(point, rotation=2)
            twice = shapes.transform_point(once, rotation=2)
            assert twice == point, f"{shape.value}.{name}"
            assert once != point or point == shapes.Point(
                Fraction(1, 2), Fraction(1, 2)
            )

    @pytest.mark.parametrize("shape", list(Shape))
    def test_mirror_applied_twice_is_identity(self, shape):
        for name, point in sorted(shapes.get_shape(shape).anchors.items()):
            once = shapes.transform_point(point, mirror=True)
            assert shapes.transform_point(once, mirror=True) == point, name

    @pytest.mark.parametrize("shape", list(Shape))
    def test_four_quarter_turns_are_identity(self, shape):
        for point in shapes.get_shape(shape).anchors.values():
            turned = point
            for _ in range(4):
                turned = shapes.transform_point(turned, rotation=1)
            assert turned == point

    def test_quarter_turn_is_clockwise(self):
        # North-west corner goes to the north-east corner under a clockwise
        # quarter turn, with y running south.
        corner = shapes.Point(Fraction(0), Fraction(0))
        assert shapes.transform_point(corner, rotation=1) == shapes.Point(
            Fraction(1), Fraction(0)
        )

    def test_rotation_is_modular(self):
        point = shapes.get_shape(Shape.U).start
        assert shapes.transform_point(point, rotation=5) == shapes.transform_point(
            point, rotation=1
        )

    def test_transform_is_applied_before_scaling(self):
        # Rotating in grid space would round twice; rotating the normalised
        # point puts the cell exactly where the mirrored anchor says.
        for shape in Shape:
            for name in shapes.REQUIRED_ANCHORS:
                cell = shapes.anchor_cell(shape, name, 48, rotation=1)
                point = shapes.transform_point(
                    shapes.get_shape(shape).anchor(name), rotation=1
                )
                assert cell == shapes.point_to_cell(point, 48)

    def test_anchor_names_accept_the_template_prefix(self):
        assert shapes.anchor_point(Shape.U, "shape.bend") == shapes.anchor_point(
            Shape.U, "bend"
        )
        assert shapes.has_anchor(Shape.U, "shape.pocket")
        assert not shapes.has_anchor(Shape.U, "shape.nowhere")
        with pytest.raises(KeyError):
            shapes.get_shape(Shape.U).anchor("nowhere")

    def test_swap_ends_only_applies_to_swappable_shapes(self):
        for shape in Shape:
            plain = shapes.anchor_point(shape, "start")
            swapped = shapes.anchor_point(shape, "start", swap_ends=True)
            if shape in SWAPPABLE_SHAPES:
                assert swapped == shapes.anchor_point(shape, "end")
                assert swapped != plain
            else:
                assert swapped == plain
        assert SWAPPABLE_SHAPES == {Shape.U, Shape.C, Shape.I}

    def test_swap_is_its_own_inverse(self):
        for shape in SWAPPABLE_SHAPES:
            for name in ("start", "end"):
                point = shapes.anchor_point(shape, name, swap_ends=True)
                other = "end" if name == "start" else "start"
                assert point == shapes.anchor_point(shape, other)

    def test_cells_respect_the_margin_and_the_grid(self):
        for grid in (8, 16, 48):
            for shape in Shape:
                for name in shapes.get_shape(shape).anchor_names():
                    for rotation in range(4):
                        for mirror in (False, True):
                            x, y = shapes.anchor_cell(
                                shape, name, grid, rotation=rotation, mirror=mirror
                            )
                            assert MARGIN <= x <= grid - 1 - MARGIN
                            assert MARGIN <= y <= grid - 1 - MARGIN

    def test_point_to_cell_rejects_a_grid_with_no_room(self):
        with pytest.raises(ValueError):
            shapes.point_to_cell(shapes.Point(Fraction(1, 2), Fraction(1, 2)), 2)

    def test_primary_axis_follows_the_transform(self):
        assert shapes.shape_axis(Shape.I) is shapes.Axis.VERTICAL
        assert shapes.shape_axis(Shape.I, rotation=1) is shapes.Axis.HORIZONTAL
        assert shapes.shape_axis(Shape.I, rotation=2) is shapes.Axis.VERTICAL
        # A mirror leaves an orthogonal axis alone but flips a diagonal one.
        assert shapes.shape_axis(Shape.I, mirror=True) is shapes.Axis.VERTICAL
        assert shapes.shape_axis(Shape.DIAMOND) is shapes.Axis.DIAGONAL_SE
        assert shapes.shape_axis(Shape.DIAMOND, mirror=True) is shapes.Axis.DIAGONAL_NE
        for shape in Shape:
            for rotation in range(4):
                for mirror in (False, True):
                    axis = shapes.shape_axis(shape, rotation, mirror)
                    # Four quarter turns bring any axis home.
                    assert (
                        shapes.transform_axis(shapes.transform_axis(axis, 2), 2) is axis
                    )

    def test_anchors_within_a_shape_are_far_enough_apart(self):
        # Distinct anchors must be at least the minimum separation apart on a
        # 48 grid, or a template using both could never satisfy stage 2.
        for shape in Shape:
            macro = shapes.get_shape(shape)
            names = macro.anchor_names()
            for i, a in enumerate(names):
                for b in names[i + 1 :]:
                    pa, pb = macro.anchors[a], macro.anchors[b]
                    if pa == pb:
                        continue  # deliberate alias, e.g. Hub start == spoke_n
                    ca = shapes.point_to_cell(pa, 48)
                    cb = shapes.point_to_cell(pb, 48)
                    d2 = (ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2
                    assert d2 >= MIN_SEPARATION**2, f"{shape.value}: {a} vs {b}"


# --------------------------------------------------------------------------
# Stage 1: templates
# --------------------------------------------------------------------------


class TestTemplateLoading:
    def test_builtins_load_and_validate(self):
        for name in BUILTINS:
            assert name in builtin_template_names()
            template = load_builtin_template(name)
            assert template.ref == f"{template.id}@{template.version}"

    def test_crypt_matches_the_spec(self, crypt):
        assert crypt.tile_class is TileClass.DUNGEON
        assert crypt.shape is Shape.U
        assert (crypt.grid, crypt.cell_m) == (48, 4)
        assert crypt.tileset == "crypt_v5"
        assert crypt.landmarks == ("bell_tower_fallen", "chain_gantry")
        assert crypt.spawn.base_density == pytest.approx(0.018)
        assert crypt.spawn.elite_rate == pytest.approx(0.12)
        assert [n.id for n in crypt.nodes] == ["in", "b1", "mech", "boss", "out"]
        assert crypt.node("mech").optional and crypt.node("mech").role is Role.MECHANIC
        assert crypt.node("boss").set_piece == "crypt_boss_v2"
        assert crypt.node("out").set_piece == "exit_brazier"
        assert [(e.a, e.b, e.optional) for e in crypt.edges] == [
            ("in", "b1", False),
            ("b1", "boss", False),
            ("b1", "mech", True),
            ("boss", "out", False),
        ]

    def test_ramparts_is_the_outdoor_counterpart(self, ramparts):
        assert ramparts.tile_class is TileClass.OUTDOOR
        assert ramparts.shape is Shape.C
        assert ramparts.tileset == "ramparts_v2"
        assert ramparts.landmarks and ramparts.landmarks != ()
        roles = {n.role for n in ramparts.nodes}
        assert {Role.ENTRANCE, Role.EXIT, Role.BOSS, Role.MECHANIC} <= roles

    def test_round_trip_through_a_dict(self):
        for name in BUILTINS:
            template = load_builtin_template(name)
            assert template_from_dict(template_to_dict(template)) == template

    def test_every_node_anchor_exists_on_its_shape(self):
        for name in BUILTINS:
            template = load_builtin_template(name)
            for node in template.nodes:
                assert shapes.has_anchor(template.shape, node.anchor)

    def test_unknown_builtin_is_reported(self):
        with pytest.raises(TemplateError, match="no built-in template"):
            load_builtin_template("no_such_template")

    def test_missing_field_is_reported(self):
        data = good_template_dict()
        del data["tileset"]
        with pytest.raises(TemplateError, match="tileset"):
            template_from_dict(data)

    def test_bad_json_file_is_reported(self, tmp_path):
        from lucifer_gen.template import load_template

        path = tmp_path / "broken.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(TemplateError, match="invalid JSON"):
            load_template(path)


class TestTemplateValidation:
    """Every rejection the spec asks stage 1 to make."""

    def test_the_probe_template_is_valid(self):
        template_from_dict(good_template_dict())

    def test_unknown_role(self):
        data = good_template_dict()
        data["nodes"][1]["role"] = "quartermaster"
        with pytest.raises(TemplateError, match="unknown role"):
            template_from_dict(data)

    def test_unknown_shape(self):
        data = good_template_dict()
        data["shape"] = "Pretzel"
        with pytest.raises(TemplateError, match="unknown shape"):
            template_from_dict(data)

    def test_unknown_class(self):
        data = good_template_dict()
        data["class"] = "underwater"
        with pytest.raises(TemplateError, match="unknown class"):
            template_from_dict(data)

    def test_unknown_role_on_a_hand_built_template(self):
        # validate_template() defends itself even when it is handed a
        # GraphTemplate that never went through the JSON loader.
        from lucifer_gen.template import validate_template

        template = load_builtin_template("crypt")
        broken = copy.deepcopy(template)
        nodes = list(broken.nodes)
        nodes[1] = TemplateNode(id="b1", role="side", anchor="shape.bend")  # type: ignore[arg-type]
        broken = GraphTemplate(
            id=broken.id,
            version=broken.version,
            tile_class=broken.tile_class,
            shape=broken.shape,
            grid=broken.grid,
            cell_m=broken.cell_m,
            nodes=tuple(nodes),
            edges=broken.edges,
            tileset=broken.tileset,
        )
        with pytest.raises(TemplateError, match="unknown role"):
            validate_template(broken)

    def test_duplicate_node_ids(self):
        data = good_template_dict()
        data["nodes"][2]["id"] = "mid"
        with pytest.raises(TemplateError, match="duplicate node id"):
            template_from_dict(data)

    def test_edge_names_a_missing_node(self):
        data = good_template_dict()
        data["edges"].append({"a": "mid", "b": "ghost"})
        with pytest.raises(TemplateError, match="unknown node 'ghost'"):
            template_from_dict(data)

    def test_no_entrance(self):
        data = good_template_dict()
        data["nodes"][0]["role"] = "side"
        with pytest.raises(TemplateError, match="exactly one entrance, found 0"):
            template_from_dict(data)

    def test_two_entrances(self):
        data = good_template_dict()
        data["nodes"][1]["role"] = "entrance"
        with pytest.raises(TemplateError, match="exactly one entrance, found 2"):
            template_from_dict(data)

    def test_two_bosses(self):
        data = good_template_dict()
        data["nodes"][1]["role"] = "boss"
        data["nodes"][2]["role"] = "boss"
        data["nodes"][2]["optional"] = False
        with pytest.raises(TemplateError, match="at most one boss"):
            template_from_dict(data)

    def test_one_boss_is_fine(self):
        data = good_template_dict()
        data["nodes"][1]["role"] = "boss"
        template_from_dict(data)

    def test_disconnected_graph(self):
        data = good_template_dict()
        data["edges"] = [{"a": "in", "b": "mid"}, {"a": "mid", "b": "side"}]
        with pytest.raises(TemplateError, match="disconnected"):
            template_from_dict(data)

    def test_optional_edge_whose_removal_disconnects_the_graph(self):
        data = good_template_dict()
        # 'out' is required, so its only link may not be optional.
        data["edges"][1]["optional"] = True
        with pytest.raises(TemplateError, match="cannot be dropped"):
            template_from_dict(data)

    def test_optional_edge_to_an_optional_node_is_allowed(self):
        # The mechanic room in crypt.json hangs off exactly such an edge.
        data = good_template_dict()
        assert data["edges"][2]["optional"] and data["nodes"][2]["optional"]
        template_from_dict(data)

    # Checks beyond the spec's list, documented in template.py.
    def test_unknown_anchor(self):
        data = good_template_dict()
        data["nodes"][1]["anchor"] = "shape.turret"
        with pytest.raises(TemplateError, match="does not define"):
            template_from_dict(data)

    def test_self_loop_edge(self):
        data = good_template_dict()
        data["edges"].append({"a": "mid", "b": "mid"})
        with pytest.raises(TemplateError, match="self-loop"):
            template_from_dict(data)

    def test_duplicate_edge(self):
        data = good_template_dict()
        data["edges"].append({"a": "mid", "b": "in"})
        with pytest.raises(TemplateError, match="duplicate edge"):
            template_from_dict(data)

    def test_two_exits(self):
        data = good_template_dict()
        data["nodes"][1]["role"] = "exit"
        with pytest.raises(TemplateError, match="at most one exit"):
            template_from_dict(data)

    def test_no_nodes(self):
        data = good_template_dict()
        data["nodes"] = []
        data["edges"] = []
        with pytest.raises(TemplateError, match="no nodes"):
            template_from_dict(data)


# --------------------------------------------------------------------------
# Stage 2: routing
# --------------------------------------------------------------------------


class TestAStar:
    def test_shortest_path_on_an_empty_grid(self):
        path = astar_path((5, 5), (9, 8), 48)
        assert path[0] == (5, 5) and path[-1] == (9, 8)
        assert len(path) == 1 + abs(9 - 5) + abs(8 - 5)  # Manhattan optimal

    def test_path_is_contiguous_and_orthogonal(self):
        path = astar_path((2, 40), (44, 3), 48)
        for a, b in zip(path, path[1:]):
            assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1

    def test_degenerate_path(self):
        assert astar_path((4, 4), (4, 4), 48) == [(4, 4)]

    def test_endpoints_must_respect_the_margin(self):
        with pytest.raises(ValueError, match="margin"):
            astar_path((0, 4), (4, 4), 48)

    def test_adjacency_penalty_pushes_the_path_aside(self):
        # A wall of penalty across the straight line costs 4 per cell entered,
        # so a detour of fewer than four extra steps has to win.
        grid = 48
        penalty = [[0] * grid for _ in range(grid)]
        for y in range(2, 20):
            penalty[y][10] = ADJACENCY_PENALTY
        path = astar_path((10, 2), (10, 19), grid, penalty)
        assert path[0] == (10, 2) and path[-1] == (10, 19)
        assert any(cell[0] != 10 for cell in path), "penalty was ignored"

    def test_zero_penalty_matches_no_penalty(self):
        grid = 48
        blank = [[0] * grid for _ in range(grid)]
        assert astar_path((3, 3), (30, 21), grid, blank) == astar_path(
            (3, 3), (30, 21), grid
        )


class TestRouteInvariants:
    @pytest.mark.parametrize("name", BUILTINS)
    def test_nodes_keep_their_separation(self, name):
        template = load_builtin_template(name)
        for seed in SEEDS:
            layout = route(template, seed)
            cells = [node.cell for node in layout.nodes.values()]
            assert len(set(cells)) == len(cells)
            for i, a in enumerate(cells):
                for b in cells[i + 1 :]:
                    d2 = (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2
                    assert d2 >= MIN_SEPARATION**2, (name, hex(seed), a, b)

    @pytest.mark.parametrize("name", BUILTINS)
    def test_everything_stays_inside_the_margin(self, name):
        template = load_builtin_template(name)
        lo, hi = MARGIN, template.grid - 1 - MARGIN
        for seed in SEEDS:
            layout = route(template, seed)
            assert layout.grid == template.grid
            for node in layout.nodes.values():
                x, y = node.cell
                assert lo <= x <= hi and lo <= y <= hi
            for edge in layout.edges:
                for x, y in edge.path:
                    assert lo <= x <= hi and lo <= y <= hi, (name, hex(seed))

    @pytest.mark.parametrize("name", BUILTINS)
    def test_paths_join_the_cells_of_their_nodes(self, name):
        template = load_builtin_template(name)
        for seed in SEEDS:
            layout = route(template, seed)
            for edge in layout.edges:
                assert edge.a in layout.nodes and edge.b in layout.nodes
                assert edge.path[0] == layout.nodes[edge.a].cell
                assert edge.path[-1] == layout.nodes[edge.b].cell
                for p, q in zip(edge.path, edge.path[1:]):
                    assert abs(p[0] - q[0]) + abs(p[1] - q[1]) == 1

    @pytest.mark.parametrize("name", BUILTINS)
    def test_entrance_reaches_the_exit(self, name):
        template = load_builtin_template(name)
        for seed in SEEDS:
            layout = route(template, seed)
            exit_node = layout.node_of_role(Role.EXIT)
            assert exit_node is not None
            reached = connected_ids(layout)
            assert exit_node.id in reached, (name, hex(seed))
            # Nothing is left floating either.
            assert reached == set(layout.nodes), (name, hex(seed))

    @pytest.mark.parametrize("name", BUILTINS)
    def test_required_nodes_always_survive(self, name):
        template = load_builtin_template(name)
        required = {n.id for n in template.nodes if not n.optional}
        for seed in SEEDS:
            layout = route(template, seed)
            assert required <= set(layout.nodes)
            assert set(layout.nodes) <= {n.id for n in template.nodes}

    def test_jitter_never_exceeds_three_cells_unless_separation_forces_it(self, crypt):
        # Nodes that do not share an anchor stay within the jitter budget; the
        # two crypt nodes that do share shape.end are the documented exception.
        shared = {"boss", "out"}
        for seed in SEEDS:
            fields = SeedFields.parse(seed)
            layout = route(crypt, seed)
            for node in layout.nodes.values():
                if node.id in shared:
                    continue
                anchor = shapes.anchor_cell(
                    crypt.shape,
                    crypt.node(node.id).anchor,
                    crypt.grid,
                    rotation=fields.rotation,
                    mirror=fields.mirror,
                    swap_ends=fields.swap_ends,
                    margin=MARGIN,
                )
                assert abs(node.cell[0] - anchor[0]) <= MAX_JITTER
                assert abs(node.cell[1] - anchor[1]) <= MAX_JITTER


class TestOptionalContent:
    def test_a_dropped_optional_node_leaves_the_layout(self, crypt):
        present = absent = 0
        for seed in SEEDS:
            layout = route(crypt, seed)
            if "mech" in layout.nodes:
                present += 1
                assert any(
                    {e.a, e.b} == {"b1", "mech"} for e in layout.edges
                ), "a surviving optional node needs its edge"
            else:
                absent += 1
                assert not any(
                    "mech" in (e.a, e.b) for e in layout.edges
                ), "an omitted node must not be named by an edge"
        # A fair coin over 200 seeds: both outcomes must show up.
        assert present > 50 and absent > 50, (present, absent)

    def test_optional_edges_are_reinstated_when_the_map_would_break(self):
        # Either optional edge may go, but not both: 'a' is a required node.
        template = template_from_dict(
            {
                "id": "redundant",
                "version": 1,
                "class": "dungeon",
                "shape": "I",
                "grid": 48,
                "cell_m": 4,
                "tileset": "probe_v1",
                "nodes": [
                    {"id": "in", "role": "entrance", "anchor": "shape.start"},
                    {"id": "a", "role": "side", "anchor": "shape.pocket"},
                    {"id": "out", "role": "exit", "anchor": "shape.end"},
                ],
                "edges": [
                    {"a": "in", "b": "a", "optional": True},
                    {"a": "a", "b": "out", "optional": True},
                    {"a": "in", "b": "out"},
                ],
            }
        )
        sizes = set()
        for seed in SEEDS:
            layout = route(template, seed)
            assert "a" in layout.nodes
            assert connected_ids(layout) == set(layout.nodes), hex(seed)
            sizes.add(len(layout.edges))
        # Both the "all three kept" and the "one reinstated" cases occur.
        assert sizes == {2, 3}, sizes


class TestDeterminism:
    @pytest.mark.parametrize("name", BUILTINS)
    def test_the_same_seed_gives_the_same_layout(self, name):
        template = load_builtin_template(name)
        for seed in SEEDS:
            assert layout_signature(route(template, seed)) == layout_signature(
                route(template, seed)
            )

    @pytest.mark.parametrize("name", BUILTINS)
    def test_a_reloaded_template_gives_the_same_layout(self, name):
        a = load_builtin_template(name)
        b = load_builtin_template(name)
        seed = SEEDS[7]
        assert layout_signature(route(a, seed)) == layout_signature(route(b, seed))

    @pytest.mark.parametrize("name", BUILTINS)
    def test_different_routing_bits_give_a_different_layout(self, name):
        template = load_builtin_template(name)
        for seed in SEEDS:
            other = seed ^ (1 << 8)  # the lowest routing bit
            assert layout_signature(route(template, seed)) != layout_signature(
                route(template, other)
            ), hex(seed)

    @pytest.mark.parametrize("name", BUILTINS)
    def test_seeds_spread_over_many_distinct_layouts(self, name):
        template = load_builtin_template(name)
        distinct = {layout_signature(route(template, seed)) for seed in SEEDS}
        assert len(distinct) == len(SEEDS)

    @pytest.mark.parametrize("name", BUILTINS)
    def test_tile_bits_do_not_disturb_stage_two(self, name):
        # Seed bits 32-63 belong to stages 4 and 6; stage 2 must ignore them.
        template = load_builtin_template(name)
        for seed in SEEDS[:40]:
            other = seed ^ ((0xFFFFFFFF << 32) & MASK64)
            assert layout_signature(route(template, seed)) == layout_signature(
                route(template, other)
            )

    def test_the_layout_carries_its_seed_and_template(self, crypt):
        seed = SEEDS[3]
        layout = route(crypt, seed)
        assert layout.seed == (seed & MASK64)
        assert layout.template is crypt


class TestTransforms:
    def test_every_transform_moves_the_map(self, crypt):
        signatures = {}
        for seed in TRANSFORM_SEEDS:
            fields = SeedFields.parse(seed)
            key = (fields.rotation, fields.mirror, fields.swap_ends)
            signatures[key] = layout_signature(route(crypt, seed))
        assert len(signatures) == 16
        assert len(set(signatures.values())) == 16

    def test_a_half_turn_of_the_grid_maps_the_nodes_onto_each_other(self):
        # The seed transform is exact, so with jitter held still by a fixed
        # routing field a 180 degree rotation is a pure point reflection.
        template = load_builtin_template("crypt")
        grid = template.grid
        base = route(template, 0x0000 | (0x31415 << 8))
        turned = route(template, 0x0002 | (0x31415 << 8))
        for node_id, node in base.nodes.items():
            if node_id in {"boss", "out"}:
                continue  # settled by the ring fallback, not a pure anchor
            anchor = shapes.anchor_cell(
                template.shape, template.node(node_id).anchor, grid, margin=MARGIN
            )
            spun = shapes.anchor_cell(
                template.shape,
                template.node(node_id).anchor,
                grid,
                rotation=2,
                margin=MARGIN,
            )
            assert spun == (grid - 1 - anchor[0], grid - 1 - anchor[1])
            offset = (node.cell[0] - anchor[0], node.cell[1] - anchor[1])
            spun_offset = (
                turned.nodes[node_id].cell[0] - spun[0],
                turned.nodes[node_id].cell[1] - spun[1],
            )
            # Same jitter draw, applied at the rotated anchor.
            assert offset == spun_offset

    @pytest.mark.parametrize("shape", [s.value for s in Shape])
    def test_every_shape_can_be_routed(self, shape):
        data = good_template_dict()
        data["shape"] = shape
        if shape == "Hub":
            data["nodes"][1]["anchor"] = "shape.spoke_e"
            data["nodes"][2]["anchor"] = "shape.spoke_w"
        template = template_from_dict(data)
        for seed in SEEDS[:25]:
            layout = route(template, seed)
            assert connected_ids(layout) == set(layout.nodes)


# --------------------------------------------------------------------------
# Which seed field funds the optional-edge coin flips
# --------------------------------------------------------------------------


def _with_field(seed: int, offset: int, width: int, value: int) -> int:
    mask = ((1 << width) - 1) << offset
    return ((seed & ~mask) | ((value << offset) & mask)) & MASK64


def test_optional_branches_are_funded_by_the_set_piece_field():
    """``seed.py`` says bits 4-7 decide which optional set piece is present.

    They now do.  The coin flip used to be labelled ``route.optional:...`` and
    therefore drawn from bits 8-31, which left bits 4-7 with no observable
    effect on any map -- 1 seed in 16 was a duplicate -- while the routing
    field silently owned the decision the doc attributes to the set-piece
    field.
    """
    template = load_builtin_template("crypt")
    optional = [n.id for n in template.nodes if n.optional]
    assert optional == ["mech"]

    base = 0x0123_4567_89AB_CDEF
    presence = {
        value: "mech" in route(template, _with_field(base, 4, 4, value)).nodes
        for value in range(16)
    }
    assert True in presence.values() and False in presence.values()

    # And the routing field no longer decides it: with the set-piece field
    # pinned, every routing value gives the same answer.
    for pinned in (0b0000, 0b0001):
        seed = _with_field(base, 4, 4, pinned)
        answers = {
            "mech" in route(template, _with_field(seed, 8, 24, r)).nodes
            for r in range(24)
        }
        assert len(answers) == 1, (pinned, answers)


def test_flipping_a_branch_leaves_the_other_edges_routed_identically():
    """"Same layout, one room more" is the use case bits 4-7 exist for.

    Each optional edge draws from its own labelled stream, so turning one on
    or off must not shift any other edge's path -- which was impossible while
    the flip came out of the routing field, since changing it rerouted every
    corridor on the map.
    """
    template = load_builtin_template("crypt")
    base = 0x0123_4567_89AB_CDEF

    with_room = without_room = None
    for value in range(16):
        layout = route(template, _with_field(base, 4, 4, value))
        if "mech" in layout.nodes and with_room is None:
            with_room = layout
        elif "mech" not in layout.nodes and without_room is None:
            without_room = layout
    assert with_room is not None and without_room is not None

    shared = set(with_room.nodes) & set(without_room.nodes)
    assert "mech" not in shared and len(shared) >= 4
    for node_id in shared:
        assert with_room.nodes[node_id].cell == without_room.nodes[node_id].cell

    def paths(layout):
        return {
            (e.a, e.b): tuple(e.path)
            for e in layout.edges
            if e.a in shared and e.b in shared
        }

    assert paths(with_room) == paths(without_room)
