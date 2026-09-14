"""Lucifer procedural map generator.

Spec: docs/WORLD_BIBLE.md.  One 64-bit seed, six stages, one map:

    >>> from lucifer_gen import generate, load_builtin_template, TileDatabase, RoomLibrary
    >>> tiles, rooms = TileDatabase.load(), RoomLibrary.load()
    >>> gmap = generate(load_builtin_template("crypt"), tiles, rooms, 0x1234)
    >>> description = build_layout_description(gmap)   # doctest: +SKIP

This module re-exports the public surface of the stage modules, so callers
name a stage's package rather than its file.  The stage modules themselves
stay importable directly (``lucifer_gen.route``, ``lucifer_gen.tileize``, ...)
for anyone who wants one stage without paying for the rest.

Layout of the package, by spec stage:

    stage 1  :mod:`lucifer_gen.template`, :mod:`lucifer_gen.shapes`
    stage 2  :mod:`lucifer_gen.route`
    stage 3  :mod:`lucifer_gen.translate`, :mod:`lucifer_gen.rooms`
    stage 4  :mod:`lucifer_gen.tileize`, :mod:`lucifer_gen.tiles`
    stage 5  :mod:`lucifer_gen.setpieces`
    stage 6  :mod:`lucifer_gen.spawn`, :mod:`lucifer_gen.layout`

    all six :mod:`lucifer_gen.pipeline` (:func:`generate`) and
            :mod:`lucifer_gen.cli` (``python3 -m lucifer_gen.cli``)

    tools   :mod:`lucifer_gen.validate` (the seed gate),
            :mod:`lucifer_gen.render` and :mod:`lucifer_gen.png` (pictures)

Three stage entry points are spelled the same as the module they live in --
``route``, ``translate`` and ``tileize`` -- so re-exporting the functions here
would shadow the modules and break ``from lucifer_gen import translate``.  The
modules win: ``lucifer_gen.route`` is the module and ``lucifer_gen.route.route``
the stage, the way ``datetime.datetime`` reads.  Every other stage entry point
(:func:`place_set_pieces`, :func:`place_spawns`, :func:`generate`, ...) is
exported here directly, because none of them collides.
"""

from __future__ import annotations

__version__ = "0.2.0"

# The stage modules, imported so ``lucifer_gen.route`` and friends are always
# the module even when only the package has been imported.  ``cli`` is left
# out on purpose: importing the package should not pull in argparse.
from . import (  # noqa: F401
    contracts,
    layout,
    pipeline,
    png,
    render,
    rooms,
    route,
    seed,
    setpieces,
    shapes,
    spawn,
    template,
    tileize,
    tiles,
    translate,
    validate,
)

# -- the shared vocabulary -------------------------------------------------
from .contracts import (
    ALL_SLOTS,
    Cell,
    CellKind,
    E,
    EdgeSig,
    GeneratedMap,
    GraphTemplate,
    MapMarker,
    N,
    NO_SLOTS,
    OPPOSITE,
    Placement,
    Role,
    RoomPlacement,
    RoutedEdge,
    RoutedLayout,
    RoutedNode,
    S,
    SIDES,
    SIDE_DELTA,
    SIDE_NAMES,
    SWAPPABLE_SHAPES,
    SetPiecePlacement,
    Shape,
    SideSpec,
    SpawnPack,
    SpawnRules,
    TemplateEdge,
    TemplateNode,
    TerrainPlan,
    Tile,
    TileClass,
    TileGrid,
    W,
    neighbour,
    sides_compatible,
)
from .seed import SeedFields, Stream, format_seed, parse_seed

# -- stage 1 ---------------------------------------------------------------
from .shapes import MacroShape, anchor_cell, get_shape, transform_point
from .template import (
    TemplateError,
    builtin_template_names,
    load_builtin_template,
    load_template,
    template_from_dict,
    template_to_dict,
    validate_template,
)

# -- stage 2 ---------------------------------------------------------------
# ``route.route`` itself is reached through the module; see the note above.
from .route import astar_path, place_nodes

# -- stage 3 ---------------------------------------------------------------
from .rooms import NoFittingRoom, Room, RoomLibrary

# -- stage 4 ---------------------------------------------------------------
from .tileize import SeamError, TileizedGrid, TileizeError
from .tiles import HeroBudget, NoFittingTile, TileDatabase

# -- stage 5 ---------------------------------------------------------------
from .setpieces import (
    SetPiece,
    SetPieceError,
    SetPieceLibrary,
    SetPieceResult,
    place_set_pieces,
)

# -- stage 6 ---------------------------------------------------------------
from .layout import (
    LayoutError,
    LayoutMismatch,
    build_layout_description,
    layout_hash,
    pack_cells,
    parse_layout_description,
    unpack_cells,
    verify_layout_description,
)
from .spawn import SpawnError, place_spawns, spawn_density

# -- the whole thing -------------------------------------------------------
from .pipeline import generate

# -- tools -----------------------------------------------------------------
from .png import PngError, encode_png, write_png
from .render import render_layout, render_routed
from .validate import (
    SuiteReport,
    ValidationReport,
    find_navmesh_islands,
    find_seam_mismatches,
    find_stage4_breaks,
    find_transform_breaks,
    geometric_sides,
    geometrically_fits,
    run_suite,
    seam_is_open,
    validate_map,
    validate_seed,
)

__all__ = [
    "__version__",
    # the stage modules (``lucifer_gen.route.route`` is stage 2's entry point)
    "contracts", "layout", "pipeline", "png", "render", "rooms", "route",
    "seed", "setpieces", "shapes", "spawn", "template", "tileize", "tiles",
    "translate", "validate",
    # contracts
    "ALL_SLOTS", "Cell", "CellKind", "E", "EdgeSig", "GeneratedMap",
    "GraphTemplate", "MapMarker", "N", "NO_SLOTS", "OPPOSITE", "Placement",
    "Role",
    "RoomPlacement", "RoutedEdge", "RoutedLayout", "RoutedNode", "S", "SIDES",
    "SIDE_DELTA", "SIDE_NAMES", "SWAPPABLE_SHAPES", "SetPiecePlacement",
    "Shape", "SideSpec", "SpawnPack", "SpawnRules", "TemplateEdge",
    "TemplateNode", "TerrainPlan", "Tile", "TileClass", "TileGrid", "W",
    "neighbour", "sides_compatible",
    # seed
    "SeedFields", "Stream", "format_seed", "parse_seed",
    # stage 1
    "MacroShape", "TemplateError", "anchor_cell", "builtin_template_names",
    "get_shape", "load_builtin_template", "load_template", "template_from_dict",
    "template_to_dict", "transform_point", "validate_template",
    # stage 2
    "astar_path", "place_nodes",
    # stage 3
    "NoFittingRoom", "Room", "RoomLibrary",
    # stage 4
    "HeroBudget", "NoFittingTile", "SeamError", "TileDatabase", "TileizeError",
    "TileizedGrid",
    # stage 5
    "SetPiece", "SetPieceError", "SetPieceLibrary", "SetPieceResult",
    "place_set_pieces",
    # stage 6
    "LayoutError", "LayoutMismatch", "SpawnError", "build_layout_description",
    "layout_hash", "pack_cells", "parse_layout_description", "place_spawns",
    "spawn_density", "unpack_cells", "verify_layout_description",
    # pipeline
    "generate",
    # tools
    "PngError", "SuiteReport", "ValidationReport", "encode_png",
    "find_navmesh_islands", "find_seam_mismatches", "find_stage4_breaks",
    "find_transform_breaks", "geometric_sides", "geometrically_fits",
    "render_layout", "render_routed", "run_suite", "seam_is_open",
    "validate_map", "validate_seed", "write_png",
]
