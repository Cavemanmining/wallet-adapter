"""Command line front end for the Descent: ``python3 -m lucifer_descent.cli``.

Spec: docs/WORLD_BIBLE.md section 03 (the web, Sigils, states, Pinnacles)
and the reconnect rule in section 07.  This is the table as a shell tool: one
sqlite file holds every profile, and each command loads one profile, drives
:class:`lucifer_descent.engine.DescentEngine` exactly as the game would, saves
the result and prints a single line saying what happened.

    new     --profile ID --seed 0x...        create the web and profile
    show    --profile ID                     tiers, state counts, stash, points,
                                             fragments, live instance
    grant   --profile ID --tier N [--count K]   mint Sigils into the stash (dev tool)
    open    --profile ID --node N --sigil ID open a portal (real generator probe)
    event   --profile ID --kind boss|elite|died|abandon|timeout
    render  --profile ID --out web.png       the table view as a PNG
    check   --profile ID                     audit a stored profile with the gate
    gate    --profiles N --start-seed S      run lucifer_descent.validate.run_suite

``--db PATH`` goes before the subcommand and defaults to ``descent.sqlite3``
in the current directory.  Exit status is 0 on success, 1 when the engine or
the gate refuses (one line on stderr, no traceback), 2 for a usage error.

The map probe
-------------
``open`` must know whether the map behind the portal has a boss and how many
elite packs it holds (spec: "Completion: boss dead, or 80 percent of elite
packs dead when the map has no boss").  By default that is
:func:`lucifer_descent.engine.default_map_probe`, which runs the real
generator.  Two hooks replace it so the CLI can be exercised without the
generator, in this order of precedence:

1. the module flag :data:`MAP_PROBE` -- a test sets
   ``lucifer_descent.cli.MAP_PROBE = my_callable`` in-process;
2. the environment variable :data:`PROBE_ENV` (``LUCIFER_DESCENT_PROBE``):
   ``boss`` or ``boss:<elites>`` answers "has a boss, that many elite
   packs"; ``noboss:<elites>`` answers "no boss"; ``real`` or unset means
   the generator.  :func:`parse_probe_spec` is the grammar.

Ticks
-----
The engine stamps a game-clock tick on every ledger entry.  The shell has no
game clock, so ``--tick N`` sets it explicitly and, when omitted, the tick is
the number of ledger entries already recorded -- monotone, and a pure
function of the saved profile, so a replayed script gives the same ledger.

Trust
-----
A profile is loaded from a file anyone can edit.  Every command that plays
on a profile first checks that its stored web is the web its seed generates
(:func:`lucifer_descent.web.matches_seed`) and refuses otherwise; the engine
itself checks the live instance against the ledger before applying a report
and refuses a Sigil id the ledger has already seen.  ``check`` runs the full
gate (``check_web``, ``check_state``, ``check_replay`` with the resolved map
probe) over one stored profile and lists every problem.  Saves are
compare-and-swap: a second shell that loaded the same profile earlier gets
``error: profile ... is at revision N`` instead of silently rolling the
first shell's moves back.

Randomness
----------
The CLI draws none itself.  ``new`` hands the profile seed to
:func:`lucifer_descent.web.generate_web`; ``grant`` mints through
:func:`lucifer_descent.sigils.mint_sigil` with a counter chosen by scanning
(:func:`next_free_counter`), so the ids minted depend only on the profile
seed and what the profile already holds or has consumed.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from lucifer_descent.contracts import (
    FRAGMENTS_TO_UNLOCK,
    MAX_TIER,
    IllegalTransition,
    NodeState,
    Pinnacle,
    ProfileState,
    Sigil,
)
from lucifer_descent.engine import DescentEngine, DescentError, MapProbe, default_map_probe
from lucifer_descent.render import DEFAULT_SIZE, render_web
from lucifer_descent.sigils import MIN_TIER, mint_sigil, sigil_id, stash_summary
from lucifer_descent.store import SchemaError, SqliteStore, StaleState
from lucifer_descent.web import generate_web, matches_seed
from lucifer_gen.seed import format_seed, parse_seed

__all__ = [
    "DEFAULT_DB",
    "PROBE_ENV",
    "MAP_PROBE",
    "EXIT_OK",
    "EXIT_FAIL",
    "EXIT_USAGE",
    "CliError",
    "parse_probe_spec",
    "resolve_probe",
    "next_free_counter",
    "build_parser",
    "main",
]

DEFAULT_DB = "descent.sqlite3"
PROBE_ENV = "LUCIFER_DESCENT_PROBE"

#: In-process override for the map probe; ``None`` means consult the
#: environment and then fall back to the real generator.
MAP_PROBE: Optional[Callable[[str, int, int], MapProbe]] = None

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

Probe = Callable[[str, int, int], MapProbe]

#: ``--kind`` values for ``event`` and the engine method each one calls.
EVENT_KINDS: Tuple[str, ...] = ("boss", "elite", "died", "abandon", "timeout")


class CliError(Exception):
    """A problem the shell can explain in one line: missing profile, bad flag."""


# --------------------------------------------------------------------------
# The map probe hooks
# --------------------------------------------------------------------------


def parse_probe_spec(spec: str) -> Probe:
    """Turn a :data:`PROBE_ENV` value into a probe callable.

    ``real`` (or blank) is the generator.  ``boss`` and ``noboss`` build a
    fixed :class:`MapProbe`; an optional ``:<n>`` sets ``elite_total``
    (default 0).  Anything else is an error, loudly, so a typo in the
    environment never quietly runs the generator.
    """
    text = spec.strip().lower()
    if text in ("", "real"):
        return default_map_probe
    head, _, tail = text.partition(":")
    if head not in ("boss", "noboss"):
        raise CliError(
            f"{PROBE_ENV}={spec!r} is not understood; use real, boss[:elites] or noboss[:elites]"
        )
    elites = 0
    if tail:
        try:
            elites = int(tail)
        except ValueError:
            raise CliError(f"{PROBE_ENV}={spec!r}: elite count {tail!r} is not an integer") from None
        if elites < 0:
            raise CliError(f"{PROBE_ENV}={spec!r}: elite count must not be negative")
    fixed = MapProbe(has_boss=(head == "boss"), elite_total=elites)

    def stub(template: str, map_seed: int, sigil_tier: int) -> MapProbe:
        return fixed

    return stub


def resolve_probe() -> Probe:
    """The probe ``open`` will use: module flag, then environment, then real."""
    if MAP_PROBE is not None:
        return MAP_PROBE
    return parse_probe_spec(os.environ.get(PROBE_ENV, ""))


# --------------------------------------------------------------------------
# Minting for ``grant``
# --------------------------------------------------------------------------


def _used_sigil_ids(state: ProfileState) -> Set[str]:
    """Every Sigil id the profile holds, is running, or has ever consumed.

    The same set :meth:`DescentEngine.add_sigil` refuses; it is scanned here
    only to pick a counter whose id the engine will accept, so ``grant``
    never has to retry.
    """
    used: Set[str] = set(state.stash)
    if state.instance is not None:
        used.add(state.instance.sigil.id)
    for entry in state.history:
        if entry.sigil_id is not None:
            used.add(entry.sigil_id)
    return used


def next_free_counter(state: ProfileState, tier: int, used: Set[str], start: int = 0) -> int:
    """The smallest mint counter at or after ``start`` whose id is not in ``used``.

    The profile does not persist a mint counter (see the report), so the
    counter is recovered by scanning: :func:`sigil_id` is injective in
    ``(counter, tier)`` for one profile seed, so the first counter whose id is
    unused yields a Sigil the engine will accept, and the scan is a pure
    function of the saved state.
    """
    counter = start
    while sigil_id(state.web.profile_seed, counter, tier) in used:
        counter += 1
    return counter


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def _load(store: SqliteStore, profile_id: str) -> ProfileState:
    """Load a profile and refuse one whose web is not the web its seed makes.

    The web decides rewards and reachability, and the file it sits in can be
    edited; regenerating it from the seed costs a few milliseconds and makes
    such an edit unplayable.  ``check`` reports the same condition as a
    gate problem instead of refusing, so it loads through the store directly.
    """
    state = store.load(profile_id)
    if state is None:
        raise CliError(f"no profile {profile_id!r} in {store.path}")
    if not matches_seed(state.web):
        raise CliError(
            f"profile {profile_id!r}: the stored web is not the web seed "
            f"{format_seed(state.web.profile_seed)} generates; run `check --profile {profile_id}`"
        )
    return state


def _tick(args: argparse.Namespace, state: Optional[ProfileState]) -> int:
    if args.tick is not None:
        return int(args.tick)
    return len(state.history) if state is not None else 0


def _engine(state: ProfileState, tick: int, probe: Optional[Probe] = None) -> DescentEngine:
    return DescentEngine(state, probe if probe is not None else _no_probe, lambda: tick)


def _no_probe(template: str, map_seed: int, sigil_tier: int) -> MapProbe:
    raise CliError("this command does not open portals")


def _state_counts(state: ProfileState) -> Dict[NodeState, int]:
    counts = {s: 0 for s in NodeState}
    for node_id in sorted(state.states):
        counts[state.states[node_id]] += 1
    return counts


def _yes(flag: bool) -> str:
    return "yes" if flag else "no"


def _ids(ids: Sequence[int]) -> str:
    return " ".join(str(i) for i in ids) if ids else "-"


def _fragments_text(state: ProfileState) -> str:
    parts = [
        f"{p.value} {state.fragments.get(p, 0)}/{FRAGMENTS_TO_UNLOCK}" for p in Pinnacle
    ]
    unlocked = sorted(p.value for p in state.unlocked_pinnacles)
    return "  ".join(parts) + "  unlocked: " + (", ".join(unlocked) if unlocked else "none")


def _instance_text(state: ProfileState) -> str:
    inst = state.instance
    if inst is None:
        return "none"
    node = state.web.node(inst.node_id)
    return (
        f"node {inst.node_id} ({node.template} tier {node.tier}) sigil {inst.sigil.id} "
        f"tier {inst.sigil.tier} map {format_seed(inst.map_seed)} boss {_yes(inst.has_boss)} "
        f"elites {inst.elite_killed}/{inst.elite_total} tick {inst.opened_tick}"
    )


# --------------------------------------------------------------------------
# Commands.  Each returns an exit status and prints its own result.
# --------------------------------------------------------------------------


def cmd_new(args: argparse.Namespace, store: SqliteStore) -> int:
    """Spec: "The web is a planar graph generated once per profile from a
    profile seed."  Refuses to regenerate over an existing profile unless
    ``--force``: the web is generated *once*."""
    if store.load(args.profile) is not None and not args.force:
        raise CliError(f"profile {args.profile!r} already exists in {store.path} (use --force to replace it)")
    seed = parse_seed(args.seed)
    web = generate_web(seed, args.rings)
    state = DescentEngine.new_profile(args.profile, web, tick=_tick(args, None))
    store.save(state)
    print(
        f"new: profile {args.profile} seed {format_seed(web.profile_seed)} "
        f"nodes {len(web.nodes)} edges {len(web.edges)} rings {args.rings} "
        f"reachable {_ids(state.reachable_ids())}"
    )
    return EXIT_OK


def cmd_show(args: argparse.Namespace, store: SqliteStore) -> int:
    state = _load(store, args.profile)
    web = state.web
    counts = _state_counts(state)
    tiers = sorted({n.tier for n in web.nodes})

    print(
        f"profile {state.profile_id}  seed {format_seed(web.profile_seed)}  "
        f"web v{web.version}  nodes {len(web.nodes)}  edges {len(web.edges)}  "
        f"tiers {tiers[0]}..{tiers[-1]}"
    )
    header = f"{'tier':>4} {'nodes':>5} " + " ".join(f"{s.value:>9}" for s in NodeState)
    print(header)
    for tier in tiers:
        nodes = web.nodes_at_tier(tier)
        per = {s: 0 for s in NodeState}
        for n in nodes:
            per[state.states[n.id]] += 1
        print(f"{tier:>4} {len(nodes):>5} " + " ".join(f"{per[s]:>9}" for s in NodeState))
    print("states: " + "  ".join(f"{s.value} {counts[s]}" for s in NodeState))
    print(f"reachable: {_ids(state.reachable_ids())}")
    failed = sorted(i for i, s in state.states.items() if s is NodeState.FAILED)
    if failed:
        print(f"failed: {_ids(failed)}")

    summary = stash_summary(state)
    by_tier = "  ".join(f"tier {t} x{summary[t]}" for t in sorted(summary))
    print(f"stash: {len(state.stash)} sigil(s)" + (f"  {by_tier}" if by_tier else ""))
    for key in sorted(state.stash):
        sigil = state.stash[key]
        print(f"  {sigil.id} tier {sigil.tier} seed {format_seed(sigil.seed)}")
    print(f"points: {state.passive_points}")
    print(f"fragments: {_fragments_text(state)}")
    print(f"instance: {_instance_text(state)}")
    print(f"ledger: {len(state.history)} entries")
    return EXIT_OK


def cmd_grant(args: argparse.Namespace, store: SqliteStore) -> int:
    """Dev tool: mint ``--count`` Sigils of ``--tier`` straight into the stash.

    Spec: "Sigils are consumable keys tiered 1 to 15."  Minting goes through
    :func:`lucifer_descent.sigils.mint_sigil`, so a granted Sigil is the same
    value a drop with that counter would have produced.
    """
    if not MIN_TIER <= args.tier <= MAX_TIER:
        raise CliError(f"--tier must be between {MIN_TIER} and {MAX_TIER}, got {args.tier}")
    if args.count < 1:
        raise CliError(f"--count must be at least 1, got {args.count}")
    state = _load(store, args.profile)
    engine = _engine(state, _tick(args, state))
    used = _used_sigil_ids(state)
    minted: List[Sigil] = []
    counter = 0
    for _ in range(args.count):
        counter = next_free_counter(state, args.tier, used, counter)
        sigil = mint_sigil(state.web.profile_seed, counter, args.tier)
        engine.add_sigil(args.profile, sigil)
        used.add(sigil.id)
        minted.append(sigil)
        counter += 1
    store.save(state)
    print(
        f"grant: {len(minted)} tier-{args.tier} sigil(s) minted into {args.profile}: "
        + ", ".join(s.id for s in minted)
    )
    return EXIT_OK


def cmd_open(args: argparse.Namespace, store: SqliteStore) -> int:
    """Spec: "Inserting one opens a portal into a reachable node; the Sigil's
    own item seed becomes the map seed", identity checked on every open."""
    state = _load(store, args.profile)
    engine = _engine(state, _tick(args, state), resolve_probe())
    opened = engine.open_portal(args.profile, args.node, args.sigil)
    store.save(state)
    inst = state.instance
    if inst is not None:
        facts = f"boss {_yes(inst.has_boss)}; elite packs {inst.elite_total}"
    else:
        # The engine clears a bossless map with no elite packs on open.
        facts = "boss no; elite packs 0; cleared on open"
    print(
        f"open: node {opened.node_id} ({opened.template} tier {opened.node_tier}) "
        f"with sigil {args.sigil} tier {opened.sigil_tier}; map seed {format_seed(opened.map_seed)}; "
        + facts
        + (f"; mechanic {opened.mechanic.value}" if opened.mechanic else "")
        + (f"; pinnacle {opened.pinnacle.value}" if opened.pinnacle else "")
    )
    return EXIT_OK


def cmd_event(args: argparse.Namespace, store: SqliteStore) -> int:
    """Report what happened inside the live instance.

    Spec: "Completion: boss dead, or 80 percent of elite packs dead when the
    map has no boss"; "Death consumes the Sigil and destroys the instance";
    abandoning and the 60 s reconnect timeout (section 07) behave like death.
    """
    state = _load(store, args.profile)
    engine = _engine(state, _tick(args, state))
    inst = state.instance
    node_id = inst.node_id if inst is not None else None
    sigil_id_text = inst.sigil.id if inst is not None else None
    points_before = state.passive_points
    fragments_before = dict(state.fragments)
    unlocked_before = set(state.unlocked_pinnacles)
    reachable_before = set(state.reachable_ids())

    if args.kind == "boss":
        after: Optional[NodeState] = engine.report_boss_kill(args.profile)
    elif args.kind == "elite":
        after = engine.report_elite_kill(args.profile)
    elif args.kind == "died":
        after = engine.report_death(args.profile)
    elif args.kind == "abandon":
        after = engine.report_abandon(args.profile)
    elif args.kind == "timeout":
        after = engine.report_timeout(args.profile)
    else:  # argparse's choices make this unreachable
        raise CliError(f"unknown event kind {args.kind!r}")
    store.save(state)

    parts = [f"event {args.kind}: node {node_id}"]
    if after is None:
        live = state.instance
        parts.append(
            f"still active, elites {live.elite_killed}/{live.elite_total}" if live else "no change"
        )
    else:
        parts.append(after.value)
        if after is not NodeState.ACTIVE:
            parts.append(f"sigil {sigil_id_text} consumed")
    gained = state.passive_points - points_before
    if gained:
        parts.append(f"+{gained} passive point(s), total {state.passive_points}")
    for pinnacle in Pinnacle:
        if state.fragments.get(pinnacle, 0) != fragments_before.get(pinnacle, 0):
            parts.append(f"{pinnacle.value} fragments {state.fragments[pinnacle]}/{FRAGMENTS_TO_UNLOCK}")
    for pinnacle in sorted(set(state.unlocked_pinnacles) - unlocked_before, key=lambda p: p.value):
        parts.append(f"{pinnacle.value} arena unlocked")
    newly = sorted(set(state.reachable_ids()) - reachable_before)
    if newly:
        parts.append(f"newly reachable {_ids(newly)}")
    print("; ".join(parts))
    return EXIT_OK


def cmd_render(args: argparse.Namespace, store: SqliteStore) -> int:
    state = _load(store, args.profile)
    written = render_web(state, args.out, size=args.size)
    print(f"render: wrote {args.out} ({args.size}x{args.size}, {written} bytes) for {args.profile}")
    return EXIT_OK


def cmd_check(args: argparse.Namespace, store: SqliteStore) -> int:
    """Audit one stored profile with the gate and list every problem.

    ``check_web`` (structure and provenance), ``check_state`` (the profile
    against its web, stash and ledger) and ``check_replay`` with the
    resolved map probe (so every recorded map fact is re-derived) all run;
    exit status is 1 when anything is reported.  This is the command to
    run on a profile whose file may have been touched by something other
    than this tool.
    """
    try:
        from lucifer_descent.validate import check_replay, check_state, check_web
    except ImportError as exc:
        raise CliError(f"check unavailable: cannot import lucifer_descent.validate ({exc})")
    state = store.load(args.profile)
    if state is None:
        raise CliError(f"no profile {args.profile!r} in {store.path}")
    problems = list(check_web(state.web))
    problems += check_state(state)
    problems += check_replay(state, map_probe=resolve_probe())
    for problem in problems:
        print(f"  {problem}")
    verdict = "OK" if not problems else f"{len(problems)} problem(s)"
    print(f"check: {args.profile} {verdict}; ledger {len(state.history)} entries")
    return EXIT_OK if not problems else EXIT_FAIL


def cmd_gate(args: argparse.Namespace, store: Optional[SqliteStore]) -> int:
    """Run the Descent gate and exit non-zero on failure.  Needs no store.

    ``lucifer_descent.validate.run_suite`` is imported lazily, because the
    gate is the only command that needs it.  Its result is read by
    :func:`_gate_outcome`, which accepts the obvious shapes a suite can
    return -- see there -- so the two modules are not welded together.
    """
    try:
        from lucifer_descent.validate import run_suite
    except ImportError as exc:
        raise CliError(f"gate unavailable: cannot import lucifer_descent.validate.run_suite ({exc})")
    if args.profiles < 1:
        raise CliError(f"--profiles must be at least 1, got {args.profiles}")
    start_seed = parse_seed(args.start_seed)
    # ``run_suite(n_profiles, start_seed, ...)``: the two inputs go
    # positionally, which is the one calling convention the suite and the
    # tests' stub share.  Catching ``TypeError`` here to retry under other
    # keyword names would also swallow a genuine ``TypeError`` from inside
    # the suite, so there is no fallback.
    result = run_suite(args.profiles, start_seed)
    ok, summary = _gate_outcome(result)
    verdict = "PASS" if ok else "FAIL"
    print(f"gate: {verdict} profiles {args.profiles} start-seed {format_seed(start_seed)}")
    if summary:
        # A SuiteReport's summary is several lines; keep it below the verdict.
        print(summary)
    return EXIT_OK if ok else EXIT_FAIL


def _gate_outcome(result: object) -> Tuple[bool, str]:
    """Read pass/fail and a short summary out of whatever ``run_suite`` returned.

    Accepted, in this order: a ``bool``; an object or mapping with an ``ok``
    or ``passed`` flag (and an optional ``summary`` string or method, or
    ``failures``); a ``(bool, ...)`` tuple; a list of failures (empty means
    pass); an int (0 means pass, as an exit status or a failure count);
    ``None`` (the suite raises on failure).  Anything else is treated as a
    pass only if it is falsy-free -- that is, as a failure -- so an
    unrecognised report can never wave a broken web through.
    """
    if isinstance(result, bool):
        return result, ""
    if result is None:
        return True, ""

    def flag(name: str):
        if isinstance(result, dict):
            return result.get(name)
        return getattr(result, name, None)

    for name in ("ok", "passed"):
        value = flag(name)
        if isinstance(value, bool):
            summary = flag("summary")
            if callable(summary):
                summary = summary()
            if not isinstance(summary, str):
                failures = flag("failures")
                summary = f"failures {len(failures)}" if isinstance(failures, (list, tuple)) else ""
            return value, summary
    if isinstance(result, tuple) and result and isinstance(result[0], bool):
        rest = " ".join(str(x) for x in result[1:])
        return result[0], rest
    if isinstance(result, (list, tuple)):
        return len(result) == 0, f"failures {len(result)}"
    if isinstance(result, int):
        return result == 0, f"result {result}"
    return False, f"unrecognised result {type(result).__name__}"


COMMANDS: Dict[str, Callable[[argparse.Namespace, SqliteStore], int]] = {
    "new": cmd_new,
    "show": cmd_show,
    "grant": cmd_grant,
    "open": cmd_open,
    "event": cmd_event,
    "render": cmd_render,
    "check": cmd_check,
    "gate": cmd_gate,
}


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------


def _add_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", required=True, help="profile id")


def _add_tick(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tick",
        type=int,
        default=None,
        help="game-clock tick stamped on the ledger (default: the ledger length)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m lucifer_descent.cli",
        description="The Descent table, as a shell tool over one sqlite file.",
    )
    parser.add_argument(
        "--db",
        default=DEFAULT_DB,
        help=f"sqlite file holding the profiles (default: ./{DEFAULT_DB})",
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p = sub.add_parser("new", help="create a profile and generate its web")
    _add_profile(p)
    p.add_argument("--seed", required=True, help="64-bit profile seed, hex (0x...) or decimal")
    p.add_argument(
        "--rings",
        type=int,
        default=MAX_TIER,
        help=f"rings around the origin, 2..{MAX_TIER} (default {MAX_TIER}; smaller for quick tests)",
    )
    p.add_argument("--force", action="store_true", help="replace an existing profile of that id")
    _add_tick(p)

    p = sub.add_parser("show", help="print the profile's web, stash, points, fragments, instance")
    _add_profile(p)

    p = sub.add_parser("grant", help="mint Sigils into the stash (dev tool)")
    _add_profile(p)
    p.add_argument("--tier", type=int, required=True, help=f"Sigil tier, {MIN_TIER}..{MAX_TIER}")
    p.add_argument("--count", type=int, default=1, help="how many to mint (default 1)")
    _add_tick(p)

    p = sub.add_parser("open", help="insert a Sigil and open a portal into a node")
    _add_profile(p)
    p.add_argument("--node", type=int, required=True, help="node id")
    p.add_argument("--sigil", required=True, help="Sigil id from the stash")
    _add_tick(p)

    p = sub.add_parser("event", help="report what happened in the live instance")
    _add_profile(p)
    p.add_argument("--kind", required=True, choices=EVENT_KINDS, help="what happened")
    _add_tick(p)

    p = sub.add_parser("render", help="write the table view as a PNG")
    _add_profile(p)
    p.add_argument("--out", required=True, help="output PNG path")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE, help=f"image side in pixels (default {DEFAULT_SIZE})")

    p = sub.add_parser("check", help="audit a stored profile with the gate; exit 1 on any problem")
    _add_profile(p)

    p = sub.add_parser("gate", help="run the Descent validation suite; exit 1 on failure")
    p.add_argument("--profiles", type=int, required=True, help="how many profiles to generate")
    p.add_argument("--start-seed", required=True, help="first profile seed, hex (0x...) or decimal")

    return parser


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Parse, run one command, and turn every expected failure into one line.

    Engine refusals (:class:`DescentError`, :class:`IllegalTransition`),
    shell-level problems (:class:`CliError`), bad values (a seed or tier out
    of range) and storage trouble all print ``error: ...`` on stderr and
    return 1.  Genuine bugs still raise, because hiding them would not help.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    try:
        if args.command == "gate":
            # The gate plays on its own temporary database; opening the
            # profile file would only risk a schema error unrelated to it.
            return handler(args, None)  # type: ignore[arg-type]
        with SqliteStore(args.db) as store:
            return handler(args, store)
    except (CliError, DescentError, IllegalTransition, ValueError, SchemaError, StaleState) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAIL
    except (sqlite3.Error, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
