"""Command line front end: ``python3 -m jarvis_bots.cli <command>``.

Design: :mod:`jarvis_bots.contracts`.  Two jobs live here.

**Adding a bot is one command.**  ``new-bot`` renders a working module, a
detail page and a passing test through :mod:`jarvis_bots.scaffold`.  It needs
nothing running and nothing configured::

    python3 -m jarvis_bots.cli new-bot --id stock-watcher \\
        --name "Stock watcher" --kind radar --dest jarvis_bots/bots

**Driving the supervisor from a shell.**::

    list        registered bots with state and next due
    status      badge_status JSON
    state       launcher_state JSON, so the page can be driven with no server
    pause/resume --id ID
    round       run one supervisor round and print the report
    gate        --rounds N --seed S [--defect NAME]

``state`` is there for the page: ``python3 -m jarvis_bots.cli state
> state.json`` and a page pointed at that file shows real figures with no
service behind it -- the honest version of the demo mode the page ships with.

Where the bots come from
------------------------
This package composes nothing by itself, so the operator says what to run::

    --bots MODULE[:ATTR]   ATTR defaults to BOTS. It may be a mapping
                           {id: bot-or-factory} (BotRegistry.register_from),
                           an iterable of bots, or a callable taking the
                           clock and returning either.
    --state PATH           a JSON file the supervisor loads before the
                           command and writes after one that changed
                           something. Without it, a pause lasts as long as
                           the process, which is to say it does not.

With no ``--bots`` the registry is empty and ``list`` says so.  That is the
truth about a framework with nothing configured, and better than inventing a
bot to fill the table.

This file is the composition root
---------------------------------
Which means it is the one place allowed to hand ``time.time`` to anything
(contracts: "Time is injected everywhere.  Nothing here calls time.time()" --
everything *below* this file takes the clock as an argument).  The seed for
``gate`` becomes a ``lucifer_gen.seed`` stream, the only randomness the
package may draw.  Nothing here opens a socket.

Binding to the rest of the package
----------------------------------
``jarvis_bots.supervisor``, ``jarvis_bots.registry``, ``jarvis_bots.api`` and
``jarvis_bots.validate`` are written alongside this file, so the CLI binds to
them by *shape*: it looks for a factory or class under a short list of names
and passes only the arguments the thing it found declares (:func:`_call`).
Where a lane is missing or names nothing recognisable the command fails the
way every other failure here does -- one line, exit 1 -- instead of a
traceback about an attribute.  ``new-bot`` touches none of that and works on
its own.

The boundary
------------
No command buys, pays or transacts, and none ever will: the widest thing this
CLI does with a bot's finding is print it with a link.

Exit status: 0 on success, 1 on any error (one line on stderr, never a
traceback), 2 for a usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import importlib
import inspect
import json
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, TextIO

from jarvis_bots.scaffold import KNOWN_KINDS, ScaffoldError, new_bot

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2


class CliError(Exception):
    """An error the user should read as one line, not as a stack."""


# ---------------------------------------------------------------------------
# binding to the rest of the package
# ---------------------------------------------------------------------------

_SUPERVISOR_MODULE = "jarvis_bots.supervisor"
_REGISTRY_MODULE = "jarvis_bots.registry"
_API_MODULE = "jarvis_bots.api"
_VALIDATE_MODULE = "jarvis_bots.validate"

_SUPERVISOR_FACTORIES = ("build_supervisor", "default_supervisor", "make_supervisor")
_SUPERVISOR_CLASSES = ("Supervisor", "BotSupervisor")
_REGISTRY_FACTORIES = ("default_registry", "build_registry", "make_registry")
_REGISTRY_CLASSES = ("BotRegistry", "Registry")
_STORE_CLASSES = ("JsonFileStore", "FileStore", "JsonStore")
_STATE_FUNCS = ("launcher_state", "bots_state", "page_state", "state")
_BADGE_FUNCS = ("badge_status", "badge", "status_payload")
_DETAIL_FUNCS = ("bot_detail", "detail", "bot_state")
_ROUND_METHODS = ("run_round", "round", "run_once", "step")
_SAVE_METHODS = ("save_state", "save")
_LOAD_METHODS = ("load_state", "load")
_GATE_FUNCS = ("run_gate", "gate", "main")


def _import(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise CliError(f"{module_name} is not available: {exc}") from exc


def _first(module, names: Sequence[str], want_class: bool = False):
    """The first of ``names`` the module defines, as a class or as a plain
    callable.  Preference order, not alphabetical: the first name is the one
    the package is expected to use."""
    for name in names:
        found = getattr(module, name, None)
        if found is None:
            continue
        if want_class and not isinstance(found, type):
            continue
        if not want_class and (isinstance(found, type) or not callable(found)):
            continue
        return found
    return None


def _call(target: Callable[..., Any], available: Mapping[str, Any]) -> Any:
    """Call ``target`` with whichever of ``available`` it declares.

    The lanes this CLI drives are written next to it, so their exact
    signatures are not knowable here.  What *is* knowable is the vocabulary
    -- a clock, a registry, a store, a ``now`` -- so the CLI reads the
    signature, supplies the names it asks for, skips what it has a default
    for, and refuses clearly when it needs something with no candidate.  One
    readable line instead of a TypeError from inside somebody's constructor.
    """
    try:
        sig = inspect.signature(target)
    except (TypeError, ValueError):  # builtins and other odd callables
        return target()

    args: List[Any] = []
    kwargs: Dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name == "self" or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if name in available:
            if param.kind is inspect.Parameter.POSITIONAL_ONLY:
                args.append(available[name])
            else:
                kwargs[name] = available[name]
        elif param.default is not inspect.Parameter.empty:
            continue
        else:
            label = getattr(target, "__qualname__", repr(target))
            raise CliError(
                f"cannot call {label}: it requires {name!r} and the CLI has "
                f"nothing to bind to it (it can supply: {', '.join(sorted(available))})"
            )
    return target(*args, **kwargs)


def _method(obj: Any, names: Sequence[str]):
    for name in names:
        found = getattr(obj, name, None)
        if callable(found):
            return found
    return None


# -- the bots the operator named --------------------------------------------


def _resolve_bots_spec(spec: str, clock: Callable[[], float]) -> Any:
    """``module[:attr]`` -> the mapping or iterable of bots it names.

    A callable attribute is called with the clock, because a bot cannot be
    built without one and a config that hard-codes ``time.time`` to get round
    that is exactly what the contract forbids.
    """
    module_name, _, attr = spec.partition(":")
    attr = attr or "BOTS"
    module = _import(module_name.strip())
    value = getattr(module, attr, None)
    if value is None:
        raise CliError(f"{module_name} has no {attr!r} to register")
    if callable(value) and not isinstance(value, (Mapping, list, tuple)):
        value = _call(value, {"clock": clock, "now": clock})
    return value


def _build_registry(clock: Callable[[], float], spec: Optional[str]):
    """The registry, empty unless ``--bots`` named something.

    ``None`` is not an error when the lane offers no registry at all: a
    supervisor may own its own.
    """
    try:
        module = _import(_REGISTRY_MODULE)
    except CliError:
        if spec:
            raise
        return None
    target = _first(module, _REGISTRY_FACTORIES) or _first(
        module, _REGISTRY_CLASSES, want_class=True
    )
    if target is None:
        if spec:
            raise CliError(f"{_REGISTRY_MODULE} defines no registry to fill")
        return None
    try:
        registry = _call(target, {"clock": clock, "now": clock})
    except CliError:
        if spec:
            raise
        return None
    if spec:
        bots = _resolve_bots_spec(spec, clock)
        register_from = getattr(registry, "register_from", None)
        if isinstance(bots, Mapping) and callable(register_from):
            register_from(bots)
        else:
            register = getattr(registry, "register", None)
            if not callable(register):
                raise CliError(f"{type(registry).__name__} has no register()")
            for bot in (bots.values() if isinstance(bots, Mapping) else bots):
                register(bot)
    return registry


def _build_store(path: Optional[str]):
    if not path:
        return None
    module = _import(_SUPERVISOR_MODULE)
    store_cls = _first(module, _STORE_CLASSES, want_class=True)
    if store_cls is None:
        raise CliError(
            f"--state needs one of {', '.join(_STORE_CLASSES)} in {_SUPERVISOR_MODULE}"
        )
    return _call(store_cls, {"path": path, "filename": path})


def _build_supervisor(args: argparse.Namespace):
    """The supervisor for this command, with its state loaded if asked."""
    clock = args.clock
    module = _import(_SUPERVISOR_MODULE)
    available: Dict[str, Any] = {"clock": clock, "now": clock, "time_fn": clock}
    registry = _build_registry(clock, getattr(args, "bots", None))
    if registry is not None:
        available["registry"] = registry
        available["bots"] = registry
    store = _build_store(getattr(args, "state_path", None))
    if store is not None:
        available["store"] = store

    target = _first(module, _SUPERVISOR_FACTORIES) or _first(
        module, _SUPERVISOR_CLASSES, want_class=True
    )
    if target is None:
        raise CliError(
            f"{_SUPERVISOR_MODULE} defines none of "
            + ", ".join(_SUPERVISOR_FACTORIES + _SUPERVISOR_CLASSES)
        )
    supervisor = _call(target, available)

    if store is not None:
        loader = _method(supervisor, _LOAD_METHODS)
        if loader is not None:
            # Nothing saved yet is the normal first run, not a failure.
            with contextlib.suppress(Exception):
                loader()
    return supervisor


def _save(supervisor, args: argparse.Namespace) -> None:
    """Persist after a command that changed something, if there is anywhere
    to persist to.  A pause that forgets itself on exit is not a pause."""
    if not getattr(args, "state_path", None):
        return
    saver = _method(supervisor, _SAVE_METHODS)
    if saver is None:
        raise CliError("the supervisor cannot save state: no save_state()")
    saver()


# -- the two payloads the page reads ----------------------------------------


def _payload(kind: str, funcs: Sequence[str], supervisor, now: float) -> Any:
    available = {"supervisor": supervisor, "sup": supervisor, "now": now, "at": now}
    with contextlib.suppress(CliError):
        module = _import(_API_MODULE)
        target = _first(module, funcs)
        if target is not None:
            return _call(target, available)
    method = _method(supervisor, funcs)
    if method is not None:
        return _call(method, {"now": now, "at": now})
    raise CliError(
        f"nothing provides the {kind} payload: looked for {', '.join(funcs)} "
        f"in {_API_MODULE} and on the supervisor"
    )


def launcher_payload(supervisor, now: float) -> Any:
    """``GET /api/bots/`` as web/README.md specifies it."""
    return _payload("launcher", _STATE_FUNCS, supervisor, now)


def badge_payload(supervisor, now: float) -> Any:
    """``GET /api/bots/status``: ``{"attention": n, "state": "ok"}``."""
    return _payload("badge", _BADGE_FUNCS, supervisor, now)


def detail_payload(supervisor, bot_id: str, now: float) -> Any:
    """``GET /api/bots/<id>``: what a generated detail page reads.

    Bound by shape like the other two, but with the id in ``available`` so
    either ``bot_detail(supervisor, bot_id)`` in :mod:`jarvis_bots.api` or
    ``supervisor.bot_detail(bot_id)`` will do.
    """
    available = {
        "supervisor": supervisor,
        "sup": supervisor,
        "bot_id": bot_id,
        "id": bot_id,
        "now": now,
        "at": now,
    }
    with contextlib.suppress(CliError):
        module = _import(_API_MODULE)
        target = _first(module, _DETAIL_FUNCS)
        if target is not None:
            return _call(target, available)
    method = _method(supervisor, _DETAIL_FUNCS)
    if method is not None:
        return _call(method, {"bot_id": bot_id, "id": bot_id, "now": now, "at": now})
    raise CliError(
        f"nothing provides the detail payload: looked for "
        f"{', '.join(_DETAIL_FUNCS)} in {_API_MODULE} and on the supervisor"
    )


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------


def _relative(delta_s: float) -> str:
    """``next due`` as a human reads it, sign kept: a bot that is overdue is
    a fact worth seeing."""
    if delta_s != delta_s:  # NaN
        return "-"
    late = delta_s < 0
    s = int(abs(delta_s))
    if s < 60:
        text = f"{s}s"
    elif s < 3600:
        text = f"{s // 60}m {s % 60:02d}s"
    elif s < 86400:
        text = f"{s // 3600}h {(s % 3600) // 60:02d}m"
    else:
        text = f"{s // 86400}d {(s % 86400) // 3600:02d}h"
    return f"{text} ago" if late else f"in {text}"


def _next_due(supervisor, bot_id: str, now: float) -> str:
    """``Health.next_due_at``, out of wherever the supervisor keeps it.

    Health is the supervisor's bookkeeping (contracts.Health) and the
    launcher payload deliberately does not carry it, so this is a best
    effort: a dash when it is not exposed beats refusing to list the bots.
    """
    health = None
    getter = getattr(supervisor, "health", None)
    if callable(getter):
        with contextlib.suppress(Exception):
            health = getter(bot_id)
    else:
        for name in ("health", "healths", "_health"):
            table = getattr(supervisor, name, None)
            if isinstance(table, Mapping):
                health = table.get(bot_id)
                break
    due = getattr(health, "next_due_at", None)
    if not isinstance(due, (int, float)) or due <= 0:
        return "-"
    return _relative(float(due) - now)


def _bots_of(payload: Any) -> List[Mapping[str, Any]]:
    if isinstance(payload, Mapping):
        bots = payload.get("bots")
        if isinstance(bots, list):
            return [b for b in bots if isinstance(b, Mapping)]
    raise CliError("the launcher payload has no 'bots' list")


def _dump(value: Any) -> str:
    """JSON for a payload that may still hold enums or dataclasses.

    ``default=`` is the honest fallback: the payload is supposed to be
    JSON-able, so turning what is not into its string form keeps the command
    useful while leaving the offender visible in the output.
    """

    def coerce(obj: Any) -> Any:
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        for attr in ("ui", "value"):
            if hasattr(obj, attr) and not callable(getattr(obj, attr)):
                return getattr(obj, attr)
        return str(obj)

    return json.dumps(value, indent=2, default=coerce)


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]], out: TextIO) -> None:
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    out.write("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip() + "\n")
    for row in rows:
        out.write("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip() + "\n")


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_new_bot(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Render a bot, its page and its test.  Prints what it wrote, one path
    per line, so the output pipes into an editor."""
    paths = new_bot(
        bot_id=args.id,
        name=args.name,
        blurb=args.blurb or "",
        kind=args.kind,
        dest_dir=args.dest,
        force=args.force,
    )
    for path in paths:
        out.write(f"{path}\n")
    module = paths[0].stem
    out.write(
        f"\nNext: python3 -m pytest {paths[2]} -q\n"
        f"      then register {module}.build(clock=...) and point the page at it.\n"
    )
    return EXIT_OK


def cmd_list(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Registered bots with state and next due."""
    now = args.clock()
    supervisor = _build_supervisor(args)
    bots = _bots_of(launcher_payload(supervisor, now))
    if not bots:
        out.write("no bots registered (pass --bots MODULE[:ATTR])\n")
        return EXIT_OK
    rows = []
    for bot in bots:
        bot_id = str(bot.get("id", "?"))
        attention = int(bot.get("attention") or 0)
        rows.append(
            (
                bot_id,
                str(bot.get("state", "?")),
                str(attention) if attention else "-",
                _next_due(supervisor, bot_id, now),
                str(bot.get("name", "")),
            )
        )
    _table(("ID", "STATE", "ATTN", "NEXT DUE", "NAME"), rows, out)
    return EXIT_OK


def cmd_status(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """The badge payload."""
    now = args.clock()
    out.write(_dump(badge_payload(_build_supervisor(args), now)) + "\n")
    return EXIT_OK


def cmd_state(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """The launcher payload, so the page can be driven with no server."""
    now = args.clock()
    out.write(_dump(launcher_payload(_build_supervisor(args), now)) + "\n")
    return EXIT_OK


def cmd_detail(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """One bot's detail payload, so a generated page can be driven with no
    server: ``cli detail --id weather > weather.json``."""
    now = args.clock()
    out.write(_dump(detail_payload(_build_supervisor(args), args.id, now)) + "\n")
    return EXIT_OK


def _set_paused(supervisor, bot_id: str, paused: bool) -> None:
    setter = getattr(supervisor, "set_paused", None)
    if callable(setter):
        setter(bot_id, paused)
        return
    method = getattr(supervisor, "pause" if paused else "resume", None)
    if callable(method):
        method(bot_id)
        return
    raise CliError(
        "the supervisor exposes neither set_paused(bot_id, paused) nor "
        "pause(bot_id)/resume(bot_id)"
    )


def cmd_pause(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Paused means paused: the supervisor stops ticking it and it reports no
    attention (contracts, "Paused means paused")."""
    supervisor = _build_supervisor(args)
    _set_paused(supervisor, args.id, True)
    _save(supervisor, args)
    out.write(f"paused {args.id}\n")
    return EXIT_OK


def cmd_resume(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    supervisor = _build_supervisor(args)
    _set_paused(supervisor, args.id, False)
    _save(supervisor, args)
    out.write(f"resumed {args.id}\n")
    return EXIT_OK


def cmd_round(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """One supervisor pass, then its report (contracts.RoundReport)."""
    now = args.clock()
    supervisor = _build_supervisor(args)
    method = _method(supervisor, _ROUND_METHODS)
    if method is None:
        raise CliError("the supervisor exposes none of " + ", ".join(_ROUND_METHODS))
    report = _call(method, {"now": now, "at": now})
    _save(supervisor, args)
    out.write(_format_report(report) + "\n")
    return EXIT_OK


def _format_report(report: Any) -> str:
    if dataclasses.is_dataclass(report) and not isinstance(report, type):
        return "  ".join(f"{k}={_scalar(v)}" for k, v in dataclasses.asdict(report).items())
    if isinstance(report, Mapping):
        return "  ".join(f"{k}={_scalar(v)}" for k, v in report.items())
    return "ok" if report is None else str(report)


def _scalar(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (list, tuple)):
        return ",".join(str(v) for v in value) or "-"
    return str(value)


def cmd_gate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Run the package's gate: N rounds from one seed, optionally with a
    named defect injected, and say whether the framework held.

    The seed goes through :mod:`lucifer_gen.seed`, the only randomness the
    package may draw, so the same seed replays the same run.
    """
    from lucifer_gen.seed import SeedFields, format_seed, parse_seed

    try:
        seed = parse_seed(args.seed)
    except (TypeError, ValueError) as exc:
        raise CliError(f"--seed must be decimal or 0x hex: {exc}") from exc
    if args.rounds < 1:
        raise CliError("--rounds must be at least 1")

    module = _import(_VALIDATE_MODULE)
    target = _first(module, _GATE_FUNCS)
    if target is None:
        raise CliError(f"{_VALIDATE_MODULE} defines none of " + ", ".join(_GATE_FUNCS))
    result = _call(
        target,
        {
            "rounds": args.rounds,
            "n_rounds": args.rounds,
            "seed": seed,
            "fields": SeedFields.parse(seed),
            "defect": args.defect,
            "clock": args.clock,
        },
    )
    return _report_gate(result, seed, format_seed, out)


def _report_gate(result: Any, seed: int, format_seed, out: TextIO) -> int:
    """Turn whatever the gate returned into lines and an exit status.

    A gate may reasonably return an exit code, a bool, a report object or
    nothing; all four are read here, and only an explicit failure fails.
    """
    if isinstance(result, bool):
        ok, lines = result, ()
    elif isinstance(result, int):
        ok, lines = result == 0, ()
    elif result is None:
        ok, lines = True, ()
    else:
        ok = bool(getattr(result, "ok", True))
        lines = getattr(result, "lines", None) or getattr(result, "failures", None) or ()
        if isinstance(lines, str):
            lines = (lines,)
        if not lines and not hasattr(result, "ok"):
            lines = (str(result),)
    for line in lines:
        out.write(f"{line}\n")
    out.write(f"gate {'PASS' if ok else 'FAIL'}  seed={format_seed(seed)}\n")
    return EXIT_OK if ok else EXIT_FAIL


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    # The two options every supervisor command shares, attached as a parent
    # so they read the same before or after the command name.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--bots",
        default=None,
        metavar="MODULE[:ATTR]",
        help="where the bots to run come from (ATTR defaults to BOTS)",
    )
    common.add_argument(
        "--state",
        dest="state_path",
        default=None,
        metavar="PATH",
        help="JSON file to load state from and save it back to",
    )

    parser = argparse.ArgumentParser(
        prog="python3 -m jarvis_bots.cli",
        description="Add bots, and drive the bot supervisor.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    p_new = sub.add_parser("new-bot", help="generate a bot, its page and its test")
    p_new.add_argument("--id", required=True, help="slug, e.g. stock-watcher")
    p_new.add_argument("--name", required=True, help="display name for the card")
    p_new.add_argument("--blurb", default="", help="one line on what it watches")
    p_new.add_argument(
        "--kind", default="bot", choices=list(KNOWN_KINDS), help="launcher icon"
    )
    p_new.add_argument("--dest", default=".", help="directory to write into")
    p_new.add_argument("--force", action="store_true", help="overwrite existing files")
    p_new.set_defaults(func=cmd_new_bot)

    sub.add_parser(
        "list", parents=[common], help="registered bots with state and next due"
    ).set_defaults(func=cmd_list)
    sub.add_parser("status", parents=[common], help="badge_status JSON").set_defaults(
        func=cmd_status
    )
    sub.add_parser("state", parents=[common], help="launcher_state JSON").set_defaults(
        func=cmd_state
    )

    p_detail = sub.add_parser(
        "detail", parents=[common], help="one bot's detail JSON, for its page"
    )
    p_detail.add_argument("--id", required=True)
    p_detail.set_defaults(func=cmd_detail)

    p_pause = sub.add_parser("pause", parents=[common], help="pause a bot")
    p_pause.add_argument("--id", required=True)
    p_pause.set_defaults(func=cmd_pause)

    p_resume = sub.add_parser("resume", parents=[common], help="resume a bot")
    p_resume.add_argument("--id", required=True)
    p_resume.set_defaults(func=cmd_resume)

    sub.add_parser(
        "round", parents=[common], help="run one supervisor round"
    ).set_defaults(func=cmd_round)

    p_gate = sub.add_parser("gate", help="run the framework gate")
    p_gate.add_argument("--rounds", type=int, default=50, help="rounds to simulate")
    p_gate.add_argument("--seed", default="0x1", help="decimal or 0x hex seed")
    p_gate.add_argument("--defect", default=None, help="inject a named defect")
    p_gate.set_defaults(func=cmd_gate)

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
    clock: Optional[Callable[[], float]] = None,
) -> int:
    """Parse ``argv``, run the command, return the exit status.

    ``clock`` is injected here and nowhere deeper: this is the composition
    root, so it is the one function in the package that may default to
    ``time.time`` (contracts: "Time is injected everywhere").  A test passes
    its own and gets a deterministic run.
    """
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = build_parser()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE

    args.clock = clock if clock is not None else time.time
    try:
        return int(args.func(args, out, err))
    except BrokenPipeError:
        # `... | head` closed the pipe: the reader left, nothing failed.
        with contextlib.suppress(Exception):
            sys.stderr.close()
        return EXIT_OK
    except (CliError, ScaffoldError) as exc:
        err.write(f"error: {exc}\n")
        err.flush()
        return EXIT_FAIL
    except Exception as exc:  # noqa: BLE001 - one line, exit 1, is the contract
        err.write(f"error: {str(exc) or type(exc).__name__}\n")
        err.flush()
        return EXIT_FAIL


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
