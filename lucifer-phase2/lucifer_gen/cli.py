"""Command line front end: ``python3 -m lucifer_gen.cli``.

Four subcommands, one per thing a human wants from the generator:

    render    the finished map as a PNG (stages 1 to 6)
    graph     the routed layout as a PNG (stage 2 only, before terrain)
    validate  the seed gate: N maps, navmesh islands and seam mismatches
    describe  the client layout description, as JSON on stdout

Every command takes ``--template`` (a built-in name or a path to a template
JSON file) and, where it makes sense, ``--seed``, which accepts ``0x...`` hex
or plain decimal.  Exit status is 0 on success, 1 when the gate finds a
failure or a file cannot be written, and 2 for a usage error, so a CI job can
simply run the gate and check ``$?``.

Spec: docs/WORLD_BIBLE.md, all six stages; the gate is the "1000 seeds, zero
navmesh islands, zero seam mismatches" acceptance check.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Optional, Sequence, Tuple

from .contracts import GeneratedMap
from .layout import build_layout_description
from .pipeline import generate, resolve_rooms, resolve_template, resolve_tiles
from .render import render_layout, render_routed
from .route import route
from .seed import format_seed, parse_seed
from .template import TemplateError, builtin_template_names
from .tiles import TileDatabase
from .validate import ValidationReport, run_suite

__all__ = ["main", "build_parser"]

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_common(parser: argparse.ArgumentParser, *, seed: bool = True) -> None:
    parser.add_argument(
        "--template",
        default="crypt",
        help="built-in template name (%s) or a path to a template JSON file"
        % ", ".join(builtin_template_names()),
    )
    parser.add_argument(
        "--tiles",
        default=None,
        help="path to a tile database JSON file (default: the shipped greybox set)",
    )
    parser.add_argument(
        "--rooms",
        default=None,
        help="path to a room library JSON file (default: the shipped greybox set)",
    )
    if seed:
        parser.add_argument(
            "--seed",
            default="0",
            help="64-bit map seed, hex (0x...) or decimal (default: 0)",
        )


def _add_spawn_knobs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tier", type=float, default=1.0, help="difficulty tier, 1 to 15 (default: 1)"
    )
    parser.add_argument(
        "--sigil",
        type=float,
        action="append",
        default=None,
        dest="sigils",
        metavar="MULT",
        help="a sigil spawn-density multiplier; repeat for several",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m lucifer_gen.cli",
        description="Lucifer procedural map generator.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    render = subs.add_parser("render", help="draw the finished tiled map as a PNG")
    _add_common(render)
    _add_spawn_knobs(render)
    render.add_argument("--out", default="map.png", help="PNG to write")
    render.add_argument(
        "--scale", type=int, default=12, help="pixels per cell (default: 12)"
    )

    graph = subs.add_parser("graph", help="draw the stage 2 routed layout as a PNG")
    _add_common(graph)
    graph.add_argument("--out", default="graph.png", help="PNG to write")
    graph.add_argument(
        "--scale", type=int, default=12, help="pixels per cell (default: 12)"
    )

    gate = subs.add_parser("validate", help="run the seed gate over many maps")
    _add_common(gate, seed=False)
    _add_spawn_knobs(gate)
    gate.add_argument(
        "--seeds", type=int, default=1000, help="how many maps to check (default: 1000)"
    )
    gate.add_argument(
        "--start-seed",
        default="0",
        dest="start_seed",
        help="first seed of the sweep, hex or decimal (default: 0)",
    )
    gate.add_argument(
        "--stop-early",
        action="store_true",
        help="stop at the first failing seed instead of counting them all",
    )
    gate.add_argument(
        "--progress",
        type=int,
        default=0,
        metavar="N",
        help="print a progress line to stderr every N seeds (0: never)",
    )

    describe = subs.add_parser(
        "describe", help="print the client layout description as JSON"
    )
    _add_common(describe)
    _add_spawn_knobs(describe)
    describe.add_argument(
        "--indent", type=int, default=2, help="JSON indent; 0 for one line"
    )
    describe.add_argument(
        "--out", default=None, help="write the JSON here instead of to stdout"
    )

    return parser


# --------------------------------------------------------------------------
# Shared loading
# --------------------------------------------------------------------------


def _load(args: argparse.Namespace):
    """Resolve the template, tile database and room library from the flags."""
    from .rooms import RoomLibrary

    template = resolve_template(args.template)
    tiles = resolve_tiles(
        TileDatabase.load(args.tiles) if args.tiles else None
    )
    rooms = (
        RoomLibrary.load(args.rooms)
        if args.rooms
        else resolve_rooms(None, template)
    )
    return template, tiles, rooms


def _sigils(args: argparse.Namespace) -> float:
    """Fold repeated ``--sigil`` flags into the one multiplier stage 6 wants."""
    values = getattr(args, "sigils", None)
    if not values:
        return 1.0
    product = 1.0
    for value in values:
        product *= float(value)
    return product


def _generate(args: argparse.Namespace) -> Tuple["GeneratedMap", "TileDatabase"]:
    """Build one map, and hand back the tile database it was built from.

    The database comes back with the map because the renderer wants it and
    loading it twice would read the same JSON twice for no reason.
    """
    template, tiles, rooms = _load(args)
    gmap = generate(
        template,
        tiles,
        rooms,
        parse_seed(args.seed),
        tier=getattr(args, "tier", 1.0),
        sigil_modifiers=_sigils(args),
    )
    return gmap, tiles


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------


def cmd_render(args: argparse.Namespace) -> int:
    gmap, tiles = _generate(args)
    data = render_layout(gmap, tiles, args.out, scale=args.scale)
    print(
        f"{args.out}: {gmap.template.ref} seed {format_seed(gmap.seed)} "
        f"{gmap.tiles.grid}x{gmap.tiles.grid} cells, "
        f"{len(gmap.set_pieces)} set piece(s), {len(gmap.spawns)} pack(s), "
        f"{len(data)} bytes"
    )
    return EXIT_OK


def cmd_graph(args: argparse.Namespace) -> int:
    template, _, _ = _load(args)
    routed = route(template, parse_seed(args.seed))
    data = render_routed(routed, args.out, scale=args.scale)
    print(
        f"{args.out}: {template.ref} seed {format_seed(routed.seed)} "
        f"{len(routed.nodes)} node(s), {len(routed.edges)} edge(s), "
        f"{len(data)} bytes"
    )
    return EXIT_OK


def cmd_describe(args: argparse.Namespace) -> int:
    gmap, _tiles = _generate(args)
    description = build_layout_description(gmap)
    indent = args.indent if args.indent and args.indent > 0 else None
    text = json.dumps(description, indent=indent, sort_keys=False)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
        print(f"{args.out}: {len(text)} bytes, {description['layout_hash']}")
    else:
        print(text)
    return EXIT_OK


def cmd_validate(args: argparse.Namespace) -> int:
    template, tiles, rooms = _load(args)
    start = parse_seed(args.start_seed)
    every = max(0, int(args.progress))

    def on_result(index: int, seed: int, report: Optional[ValidationReport]) -> None:
        if every and (index + 1) % every == 0:
            verdict = "ok" if (report is not None and report.ok) else "FAILED"
            print(
                f"  [{index + 1}/{args.seeds}] {format_seed(seed)} {verdict}",
                file=sys.stderr,
                flush=True,
            )

    began = time.monotonic()
    suite = run_suite(
        template,
        tiles,
        rooms,
        n_seeds=args.seeds,
        start_seed=start,
        tier=args.tier,
        sigil_modifiers=_sigils(args),
        on_result=on_result if every else None,
        stop_early=args.stop_early,
    )
    elapsed = time.monotonic() - began

    print(suite.summary())
    print(
        f"  totals: islands={suite.islands} seams={suite.seams} "
        f"clean={suite.clean}/{suite.checked} in {elapsed:.1f}s"
    )
    if suite.ok:
        print("GATE PASSED")
        return EXIT_OK
    print("GATE FAILED", file=sys.stderr)
    return EXIT_FAIL


_COMMANDS = {
    "render": cmd_render,
    "graph": cmd_graph,
    "validate": cmd_validate,
    "describe": cmd_describe,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return _COMMANDS[args.command](args)
    except (TemplateError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover - the entry point itself
    raise SystemExit(main())
