"""The whole generator, stage 1 to stage 6, in one call.

Spec: docs/WORLD_BIBLE.md, the six stages.

    1  load and validate the graph template            :mod:`template`
    2  shuffle the shape and route the edges           :mod:`route`
    3  translate routed edges into terrain             :mod:`translate`
    4  tileize the terrain and fill the untouched      :mod:`tileize`
    5  stamp the set pieces and enforce the tells      :mod:`setpieces`
    6  place the spawn packs; hash the layout          :mod:`spawn`, :mod:`layout`

Every stage is a pure function of its input plus the seed, and every draw of
randomness comes from ``SeedFields(...).stream(label)``, so one seed is one
map on every machine.  Stage 6's hash is not computed here: it belongs to the
client layout description, which :func:`lucifer_gen.layout.build_layout_description`
builds from the :class:`~lucifer_gen.contracts.GeneratedMap` this returns.

The stage order is load-bearing in two places that are easy to get wrong:

* stage 5 runs *after* stage 4, because stamping a fixed interior overwrites
  tiles and has to re-run the filler around itself; and
* stage 6 runs *after* stage 5, because the boss approach and the set-piece
  interiors are exclusions stage 6 reads off the terrain plan.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional, Sequence, Union

from .contracts import GeneratedMap, GraphTemplate, MapMarker, Role, TileClass
from .rooms import RoomLibrary
from .route import route
from .seed import SeedFields, parse_seed
from .setpieces import SetPieceLibrary, SetPieceResult, place_set_pieces
from .spawn import place_spawns
from .template import load_builtin_template, load_template
from .tileize import tileize
from .tiles import TileDatabase
from .translate import translate

__all__ = [
    "generate",
    "resolve_template",
    "resolve_tiles",
    "resolve_rooms",
]

SigilModifiers = Union[None, float, int, Sequence[float], Mapping[str, float]]


# --------------------------------------------------------------------------
# Convenience resolution, so a caller may pass a name or nothing at all
# --------------------------------------------------------------------------


def resolve_template(template: Union[GraphTemplate, str, "os.PathLike[str]"]) -> GraphTemplate:
    """Accept a loaded template, a built-in name, or a path to a JSON file.

    Stage 1.  Loading validates: connected graph, exactly one entrance, at
    most one boss, and no optional edge whose removal would disconnect it.
    """
    if isinstance(template, GraphTemplate):
        return template
    text = os.fspath(template) if hasattr(template, "__fspath__") else str(template)
    if os.sep in text or text.lower().endswith(".json"):
        return load_template(text)
    return load_builtin_template(text)


def resolve_tiles(tiles: Optional[TileDatabase]) -> TileDatabase:
    """The tile database, loading the shipped greybox set when none is given."""
    return tiles if tiles is not None else TileDatabase.load()


def resolve_rooms(
    rooms: Optional[RoomLibrary], template: GraphTemplate
) -> Optional[RoomLibrary]:
    """The room library stage 3 needs; outdoor templates need none.

    A dungeon (or ``BOTH``) template is translated into rooms and corridors,
    so it must have a library; an outdoor one is translated into splines and
    never opens the library at all, so leaving it ``None`` there is honest
    rather than lazy.
    """
    if rooms is not None:
        return rooms
    if template.tile_class is TileClass.OUTDOOR:
        return None
    return RoomLibrary.load()


# --------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------


def generate(
    template: Union[GraphTemplate, str],
    tiles: Optional[TileDatabase],
    rooms: Optional[RoomLibrary],
    seed: Union[int, str],
    tier: Union[int, float] = 1,
    sigil_modifiers: SigilModifiers = 1.0,
    *,
    set_pieces: Optional[SetPieceLibrary] = None,
    verify: bool = True,
) -> GeneratedMap:
    """Run stages 1 to 6 for one seed and return the assembled map.

    ``template`` may be a :class:`GraphTemplate`, the name of a built-in one,
    or a path to a template JSON file.  ``tiles`` and ``rooms`` may be
    ``None``, in which case the shipped greybox databases are loaded.

    ``verify`` keeps stage 4's seam assertion and stage 5's local re-check on;
    turning it off buys a few milliseconds per map and gives up the guarantee
    that every seam in the grid fits, so leave it on outside a benchmark.
    """
    graph = resolve_template(template)          # stage 1
    db = resolve_tiles(tiles)
    library = resolve_rooms(rooms, graph)
    fields = SeedFields.parse(parse_seed(seed))

    routed = route(graph, fields.raw)                                   # stage 2
    plan = translate(routed, library, fields.raw)                       # stage 3
    tile_grid = tileize(plan, db, graph.tile_class, fields, verify=verify)  # stage 4
    pieces: SetPieceResult = place_set_pieces(                          # stage 5
        routed,
        plan,
        tile_grid,
        db,
        fields,
        library=set_pieces,
        tile_class=graph.tile_class,
        verify=verify,
    )
    spawns = place_spawns(                                              # stage 6
        plan, tile_grid, routed, graph, tier, sigil_modifiers, fields
    )

    return GeneratedMap(
        seed=fields.raw,
        template=graph,
        tileset_ref=getattr(tile_grid, "tileset_ref", None) or db.version,
        routed=routed,
        terrain=plan,
        tiles=tile_grid,
        set_pieces=list(pieces),
        spawns=list(spawns),
        exit_cell=_exit_cell(routed, pieces),
        checkpoints=list(pieces.checkpoints),
        markers=_markers(pieces),
    )


def _markers(pieces: SetPieceResult):
    """Carry stage 5's tells out of the stage and into the map.

    Stage 5 names every marker it stamps, including the landmark it picks from
    the template's list using the seed's set-piece field.  Keeping only the
    checkpoints and the exit cell -- as this function's absence used to --
    threw the rest away the moment stage 5 returned, so the only consumer of
    ``SeedFields.set_piece_choice`` produced nothing anyone could observe.
    """
    return [
        MapMarker(
            kind=marker.kind,
            cell=marker.cell,
            name=marker.name,
            piece_id=marker.piece_id,
            node_id=marker.node_id,
        )
        for marker in pieces.markers
    ]


def _exit_cell(routed, pieces: SetPieceResult):
    """Where the player leaves: the exit piece's marked cell, or its node.

    Stage 5 names the cell when the exit carries an ``exit`` marker.  A
    template with no exit node at all still has to answer, so fall back to the
    entrance and finally to the origin rather than returning ``None`` into a
    field the client layout description declares mandatory.
    """
    if pieces.exit_cell is not None:
        return pieces.exit_cell
    for role in (Role.EXIT, Role.ENTRANCE):
        node = routed.node_of_role(role)
        if node is not None:
            return node.cell
    return (0, 0)
