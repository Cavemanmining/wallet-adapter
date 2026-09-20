"""Command line front end: ``python3 -m jarvis_alerts.cli --db PATH <command>``.

Design: :mod:`jarvis_alerts.contracts`, module docstring.  This is the shell
view of the three points -- the outbox is one sqlite file, the worker runs
here as a process, and every command names subscriptions by
(profile_id, device_id) and never shows a blob.  It composes the pieces the
way the app would: :class:`jarvis_alerts.api.AlertService` over
:class:`jarvis_alerts.outbox.Outbox` for the app-side commands, and
:class:`jarvis_alerts.worker.Worker` over the transports for ``worker``.

    publish    --profile P --kind K --title T --body B
               [--priority low|normal|high] [--dedupe KEY] [--data JSON]
               prints the alert id
    register   --profile P --device D --transport fake|webpush|fcm --blob-file PATH
               the blob is read from the file, never from the command line,
               so it cannot land in shell history
    unregister --profile P --device D
    worker     [--once] [--idle 1.0] [--batch 50] [--lease 30] [--seed S]
               lease due rows and push them until stopped (SIGINT/SIGTERM)
    stats      counts per row state, as JSON
    dead       [--limit N]   the newest dead-lettered rows, one per line,
               each with its dead_reason: an ``exhausted`` row (the retry
               budget ran out during an outage) comes back by itself after
               the cooldown while its alert is younger than a day; the
               others wait for ``requeue``
    requeue    --row ID [--row ID ...] | --all [--profile P [--device D]]
               put dead-lettered rows back in flight with a fresh budget
    gate       --profiles N --alerts M --seed S
               runs ``jarvis_alerts.validate.run_gate``; exit 1 on failure

``--db PATH`` is accepted before or after the command and defaults to
``alerts.sqlite3`` in the current directory (``gate`` does not use it).
Exit status: 0 on success, 1 on any error (one line on stderr, no
traceback), 2 for a usage error.  No usage error ever echoes a value: an
invalid choice, an unparsable number or an unrecognised token is named
by its option, never by what was typed, because the value might be a
blob pasted in the wrong place.

Transport wiring for ``worker``
-------------------------------
"fake" is always available (:class:`jarvis_alerts.transports.FakeTransport`,
which says ok to everything).  "webpush" and "fcm" are built only when the
app has installed a sender for them with :func:`jarvis_alerts.api.set_sender`
at import time; ``python3 -m jarvis_alerts.cli`` alone imports no app code,
so an app runs its worker through a wrapper that does::

    from jarvis_alerts import api, cli
    api.set_sender("webpush", my_webpush_sender)      # (endpoint, body, headers) -> status
    raise SystemExit(cli.main())                       # argv from sys.argv

Rows whose transport is not wired in this process are *not* due here: the
CLI prints one line per such transport and the rows wait for a process
that has it.  The worker itself dead-letters a row whose transport is
missing ("no transport"), which is right for a typo in a subscription but
wrong for a sender that just is not installed in this worker, so the CLI
puts :class:`ParkUnwired` between the outbox and the worker: it leases
with ``Outbox.lease(..., transports=<the wired names>)``, so a row for
another transport is never touched by this process -- not leased, not
held, not delayed -- and a process that has the transport takes it the
moment it is due.  The per-pass line reports such rows as ``parked``.
:attr:`TRANSPORT_FACTORIES` is the seam a test uses to script the fake.

Randomness and time
-------------------
The CLI is the composition root, so it is the one place that hands
``time.time`` to the outbox and the worker.  Backoff jitter is a
``lucifer_gen.seed.Stream`` labelled ``"alerts.backoff"`` from ``--seed``
(default 0): the same seed gives the same retry schedule, which is what an
operator replaying an incident wants.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, TextIO, Tuple

from lucifer_gen.seed import SeedFields, format_seed, parse_seed

from . import api
from .contracts import LEASE_S, Alert, OutboxRow, RowState, SendResult, Subscription, Transport
from .outbox import Outbox, OutboxError
from .worker import RunReport, Worker

__all__ = [
    "DEFAULT_DB",
    "EXIT_FAIL",
    "EXIT_OK",
    "EXIT_USAGE",
    "JITTER_LABEL",
    "NEEDS_SENDER",
    "TRANSPORT_FACTORIES",
    "TRANSPORT_NAMES",
    "CliError",
    "ParkUnwired",
    "build_parser",
    "build_transports",
    "format_report",
    "gate_verdict",
    "main",
    "read_blob_file",
    "run_worker",
]

DEFAULT_DB = "alerts.sqlite3"
JITTER_LABEL = "alerts.backoff"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

#: The transport names contracts.py lists for ``Subscription.transport``.
TRANSPORT_NAMES: Tuple[str, ...] = ("fake", "webpush", "fcm")
#: Those that talk to a real service and so need the app's sender.
NEEDS_SENDER = frozenset({"webpush", "fcm"})

Sender = Optional[api.Sender]
TransportFactory = Callable[[Sender], Transport]


class CliError(Exception):
    """Something the CLI refuses to do; printed as one line, exit 1."""


# ---------------------------------------------------------------------------
# Transport wiring
# ---------------------------------------------------------------------------


def _make_fake(sender: Sender) -> Transport:
    from .transports import FakeTransport  # lazy: only ``worker`` needs it
    return FakeTransport()


def _make_webpush(sender: Sender) -> Transport:
    from .transports import WebPushTransport
    return WebPushTransport(sender)


def _make_fcm(sender: Sender) -> Transport:
    from .transports import FCMTransport
    return FCMTransport(sender)


#: name -> factory(sender) -> Transport.  The seam for tests and for an app
#: that wants to substitute its own transport class under one of the names.
TRANSPORT_FACTORIES: Dict[str, TransportFactory] = {
    "fake": _make_fake,
    "webpush": _make_webpush,
    "fcm": _make_fcm,
}


def build_transports() -> Tuple[Dict[str, Transport], Dict[str, str]]:
    """Construct every transport this process can wire.

    Returns ``(transports, unwired)``: the transports by name, and for
    each name that could not be wired the one line to print about it.  A
    transport that needs a sender and has none installed is unwired; so
    is one whose factory could not import or find its class (the
    transports module missing or renamed).  Any other failure to construct
    a transport propagates: it is a bug, not a wiring gap.
    """
    transports: Dict[str, Transport] = {}
    unwired: Dict[str, str] = {}
    for name, factory in TRANSPORT_FACTORIES.items():
        sender = api.get_sender(name)
        if name in NEEDS_SENDER and sender is None:
            unwired[name] = (
                f"transport {name!r} is not wired in this process (no sender installed "
                f"via jarvis_alerts.api.set_sender); its rows are left pending"
            )
            continue
        try:
            transports[name] = factory(sender)
        except (ImportError, AttributeError) as exc:
            unwired[name] = (
                f"transport {name!r} is not wired in this process (its transport class is "
                f"unavailable: {type(exc).__name__}); its rows are left pending"
            )
    return transports, unwired


class ParkUnwired:
    """The worker's outbox port, minus rows whose transport is not wired.

    Wraps an :class:`Outbox` for :class:`jarvis_alerts.worker.Worker`.
    ``lease`` asks the real outbox only for rows whose subscription names
    a transport in ``wired`` (``Outbox.lease``'s ``transports`` filter), so
    a row for any other transport stays PENDING for a process that can
    send it; it is never leased here and so never delayed by a lease it
    cannot use.  The worker still sees a subscription that names nothing
    wired when one changes transport between lease and lookup, and
    applies its own "no transport" policy.

    :meth:`take_parked` returns how many PENDING rows behind a live
    subscription this process is leaving for an unwired transport (from
    ``Outbox.backlog_by_transport``), for the per-pass report, and hands
    ``notify`` one line the first time each unwired transport name is seen
    with rows, except names in ``announced`` (those the CLI already
    printed a line for at startup), so every unwired transport gets
    exactly one line.  The name is shown ``repr``-escaped and clipped,
    because it is text a client chose.  ``mark``, ``release``,
    ``revive_exhausted``, ``subscription``, ``alert`` and ``stats`` pass
    straight through (a revived row of an unwired transport is simply not
    leased here).
    """

    NAME_CLIP = 40

    def __init__(
        self,
        outbox: Outbox,
        wired: Set[str],
        notify: Callable[[str], None],
        announced: Optional[Set[str]] = None,
    ) -> None:
        self._outbox = outbox
        self._wired = set(wired)
        self._notify = notify
        self._seen: Set[str] = set(announced or ())
        self._lock = threading.Lock()

    def lease(self, now: float, limit: int, lease_s: float = LEASE_S) -> List[OutboxRow]:
        return self._outbox.lease(now, limit, lease_s, transports=self._wired)

    def take_parked(self) -> int:
        backlog = self._outbox.backlog_by_transport()
        parked = 0
        for name in sorted(backlog):
            if name in self._wired:
                continue
            parked += backlog[name]
            with self._lock:
                first_time = name not in self._seen
                self._seen.add(name)
            if first_time:
                shown = repr(name)
                if len(shown) > self.NAME_CLIP:
                    shown = shown[: self.NAME_CLIP] + "...'"
                self._notify(
                    f"transport {shown} is not wired in this process; its rows are left "
                    f"pending ({backlog[name]} now)"
                )
        return parked

    def mark(self, row_id: int, result: SendResult, now: float,
             jitter: Optional[float] = None) -> RowState:
        return self._outbox.mark(row_id, result, now, jitter)

    def release(self, row_id: int, reason: str = "") -> RowState:
        return self._outbox.release(row_id, reason)

    def revive_exhausted(self, now: float) -> int:
        return self._outbox.revive_exhausted(now)

    def subscription(self, profile_id: str, device_id: str) -> Optional[Subscription]:
        return self._outbox.subscription(profile_id, device_id)

    def alert(self, alert_id: str) -> Optional[Alert]:
        return self._outbox.alert(alert_id)

    def stats(self, dead_limit: int = 50) -> Dict[str, Any]:
        return self._outbox.stats(dead_limit=dead_limit)


# ---------------------------------------------------------------------------
# The worker loop
# ---------------------------------------------------------------------------


def format_report(report: RunReport, parked: int) -> str:
    """One line per pass.  ``expired`` (rows not sent because the batch
    outlived the lease) and ``revived`` (dead letters the outbox put back
    at the start of the pass) are appended only when they happened, so
    the common line keeps its shape."""
    line = (
        f"pass: leased={report.leased} delivered={report.delivered} retried={report.retried} "
        f"dead={report.dead} pruned={report.pruned} errors={report.errors} parked={parked}"
    )
    if report.expired:
        line += f" expired={report.expired}"
    if report.revived:
        line += f" revived={report.revived}"
    return line


def run_worker(
    worker: Worker,
    port: ParkUnwired,
    stop: threading.Event,
    idle_s: float,
    once: bool,
    out: TextIO,
) -> None:
    """Run passes until ``stop`` is set, or one pass when ``once``.

    Mirrors :meth:`Worker.run_forever` (idle on ``stop.wait``, never
    ``time.sleep``) but prints a line per pass that did something -- leased
    rows, or a change in how many rows are parked for another process --
    and always prints the one pass of ``--once`` so "nothing was due" is
    visible.  A store failure propagates to :func:`main`'s one-line error.
    """
    last_parked = 0
    while True:
        report = worker.run_once()
        parked = port.take_parked()
        if once or report.leased or parked != last_parked:
            out.write(format_report(report, parked) + "\n")
            out.flush()
        last_parked = parked
        if once or stop.is_set():
            return
        if report.leased == 0:
            stop.wait(idle_s)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def read_blob_file(path: str) -> str:
    """The blob, from a file.  Errors never echo the value.

    The one mistake this option exists to prevent is the blob itself -- or
    its endpoint, or an FCM token -- being typed on the command line, so
    the value is shown in an error only when it names a file that exists
    (then it is a path, and naming it helps); anything that cannot be
    opened is described, not quoted.  A value that starts like JSON text
    is refused outright.
    """
    if path.lstrip().startswith(("{", "[", "\ufeff{", "\ufeff[")):
        raise CliError("--blob-file takes a path; it looks like the blob itself was given "
                       "(not shown). Write the blob to a file and pass that")
    shown = path if os.path.exists(path) else "(value not shown)"
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = fh.read()
    except OSError as exc:
        raise CliError(f"cannot read blob file {shown}: {exc.strerror or type(exc).__name__}") from None
    except UnicodeDecodeError:
        raise CliError(f"blob file {shown} is not UTF-8 text") from None
    blob = blob.strip()
    if not blob:
        raise CliError(f"blob file {shown} is empty")
    return blob


def _open(args: argparse.Namespace, jitter: Optional[Callable[[], float]] = None) -> Outbox:
    if jitter is None:
        return Outbox(args.db, clock=time.time)
    return Outbox(args.db, clock=time.time, jitter=jitter)


def _service(outbox: Outbox) -> api.AlertService:
    return api.AlertService(outbox, clock=time.time)


def cmd_publish(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    data: Optional[Dict[str, Any]] = None
    if args.data is not None:
        try:
            data = json.loads(args.data)
        except ValueError:
            raise CliError("--data must be a JSON object") from None
        if not isinstance(data, dict):
            raise CliError("--data must be a JSON object")
    with _open(args) as outbox:
        alert_id = _service(outbox).publish(
            args.profile, args.kind, args.title, args.body,
            data=data, priority=args.priority, dedupe_key=args.dedupe,
        )
    out.write(alert_id + "\n")
    return EXIT_OK


def cmd_register(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    blob = read_blob_file(args.blob_file)
    with _open(args) as outbox:
        _service(outbox).register_device(args.profile, args.device, args.transport, blob)
    out.write(f"registered ({args.profile}, {args.device}) via {args.transport}\n")
    return EXIT_OK


def cmd_unregister(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    with _open(args) as outbox:
        was = _service(outbox).unregister_device(args.profile, args.device)
    if not was:
        # The ids are not echoed: an error line is the one place a value
        # pasted into the wrong option would otherwise be printed.
        raise CliError("that (profile, device) is not registered")
    out.write(f"unregistered ({args.profile}, {args.device})\n")
    return EXIT_OK


def cmd_stats(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    with _open(args) as outbox:
        stats = _service(outbox).stats(dead_limit=0)
        stats["subscriptions_pruned"] = len(outbox.pruned_subscriptions())
        stats["pending_by_transport"] = outbox.backlog_by_transport()
    stats.pop("dead_letters", None)  # ``dead`` lists them
    out.write(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    return EXIT_OK


def cmd_dead(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.limit < 0:
        raise CliError("--limit must not be negative")
    with _open(args) as outbox:
        rows = outbox.dead_letters(limit=args.limit)
    for row in rows:
        cause = "?" if row.dead_reason is None else row.dead_reason.value
        out.write(
            f"row={row.row_id} alert={row.alert_id} profile={row.profile_id} "
            f"device={row.device_id} attempts={row.attempts} reason={row.last_reason!r} "
            f"dead_reason={cause}\n"
        )
    return EXIT_OK


def cmd_requeue(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if bool(args.row) == bool(args.all):
        raise CliError("requeue takes --row ID (repeatable) or --all, not both and not neither")
    if args.device and not args.profile:
        raise CliError("--device needs --profile")
    if args.row and (args.profile or args.device):
        raise CliError("--profile/--device go with --all")
    with _open(args) as outbox:
        service = _service(outbox)
        if args.all:
            count = service.requeue_dead(args.profile, args.device)
        else:
            count = 0
            for row_id in args.row:
                try:
                    count += int(service.requeue(row_id))
                except OutboxError as exc:
                    raise CliError(str(exc)) from None
    out.write(f"requeued {count} row(s)\n")
    return EXIT_OK


def cmd_worker(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.idle < 0:
        raise CliError("--idle must not be negative")
    if args.batch < 1:
        raise CliError("--batch must be at least 1")
    if args.lease <= 0:
        raise CliError("--lease must be positive")

    transports, unwired = build_transports()
    for name in sorted(unwired):
        err.write(unwired[name] + "\n")
    jitter = SeedFields.parse(parse_seed(args.seed)).stream(JITTER_LABEL).random

    stop = threading.Event()
    with _open(args, jitter=jitter) as outbox:
        port = ParkUnwired(
            outbox, set(transports),
            notify=lambda line: (err.write(line + "\n"), err.flush()),
            announced=set(unwired),
        )
        worker = Worker(port, transports, clock=time.time, jitter=jitter,
                        batch=args.batch, lease_s=args.lease)
        with _stop_on_signals(stop), _package_warnings_to(err):
            run_worker(worker, port, stop, args.idle, args.once, out)
    return EXIT_OK


class _package_warnings_to:
    """Route this package's log warnings (the worker's "transport raised
    X" lines) to ``err`` while the block runs.  A handler scoped to the
    block, rather than ``logging.basicConfig``, so a process that calls
    :func:`main` more than once with different streams gets each right."""

    def __init__(self, err: TextIO) -> None:
        self._handler = logging.StreamHandler(err)
        self._handler.setLevel(logging.WARNING)
        self._handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        self._logger = logging.getLogger("jarvis_alerts")

    def __enter__(self) -> "_package_warnings_to":
        self._logger.addHandler(self._handler)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.flush()


class _stop_on_signals:
    """Set ``stop`` on SIGINT/SIGTERM while the block runs; restore after.
    Handlers can only be installed from the main thread, so elsewhere
    (a test driving ``main`` from a thread) this is a no-op."""

    def __init__(self, stop: threading.Event) -> None:
        self._stop = stop
        self._previous: Dict[int, Any] = {}

    def __enter__(self) -> "_stop_on_signals":
        if threading.current_thread() is not threading.main_thread():
            return self
        for signum in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous[signum] = signal.signal(signum, lambda *_: self._stop.set())
            except (ValueError, OSError):
                pass
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)


def gate_verdict(result: Any) -> Tuple[bool, List[str]]:
    """Read ``run_gate``'s result: ``(passed, lines to print)``.

    ``jarvis_alerts.validate`` is a sibling written separately, so the
    shape of its report is taken loosely: a bool, or anything with an
    ``ok`` or ``passed`` attribute or key (a callable one is called), plus
    an optional ``problems`` / ``failures`` / ``errors`` list of strings
    to print.  Anything else is refused rather than guessed.
    """
    if isinstance(result, bool):
        return result, []
    passed: Any = None
    for key in ("ok", "passed"):
        value = _field(result, key)
        if value is not None:
            passed = value() if callable(value) else value
            break
    if passed is None:
        raise CliError(f"gate returned {type(result).__name__} with no ok/passed verdict")
    lines: List[str] = []
    for key in ("problems", "failures", "errors"):
        value = _field(result, key)
        if isinstance(value, (list, tuple)):
            lines.extend(str(item) for item in value)
            break
    return bool(passed), lines


def _field(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def cmd_gate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    if args.profiles < 1 or args.alerts < 1:
        raise CliError("--profiles and --alerts must be at least 1")
    seed = parse_seed(args.seed)
    # ``importlib.import_module`` rather than ``from . import validate``: the
    # latter answers from the package's attribute once the module has been
    # imported anywhere in the process, so a test (or an app) that swaps or
    # removes ``sys.modules["jarvis_alerts.validate"]`` would not be seen.
    try:
        validate = importlib.import_module(".validate", __package__)
    except ImportError as exc:
        raise CliError(
            f"gate unavailable: jarvis_alerts.validate cannot be imported ({type(exc).__name__})"
        ) from None
    run_gate = getattr(validate, "run_gate", None)
    if run_gate is None:
        raise CliError("gate unavailable: jarvis_alerts.validate has no run_gate")
    # The keyword names are ``validate.run_gate``'s own (``n_profiles``,
    # ``n_alerts``, ``seed``); crash cadence and defect injection keep their
    # defaults here, ``python3 -m jarvis_alerts.validate`` exposes them.
    report = run_gate(n_profiles=args.profiles, n_alerts=args.alerts, seed=seed)
    passed, lines = gate_verdict(report)
    summary = getattr(report, "summary", None)
    if callable(summary):
        # The real GateReport: its summary carries the counts an operator
        # wants and the problems, so the verdict lines are not repeated.
        out.write(str(summary()) + "\n")
    else:
        for line in lines:
            out.write(line + "\n")
    out.write(
        f"gate {'passed' if passed else 'FAILED'}: profiles={args.profiles} "
        f"alerts={args.alerts} seed={format_seed(seed)}\n"
    )
    return EXIT_OK if passed else EXIT_FAIL


# ---------------------------------------------------------------------------
# Parser and entry point
# ---------------------------------------------------------------------------


class _ValueBlindParser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose errors never quote what was typed.

    argparse's own messages for a value outside ``choices`` ("invalid
    choice: 'x'") and for a value a ``type`` rejects ("invalid int value:
    'x'") echo the value.  A blob, endpoint or token pasted into the wrong
    slot would land on stderr, so both messages are replaced by ones that
    name the option and, for choices, the alternatives.  Subparsers are
    created with ``parser_class=type(self)``, so they inherit this.
    """

    def _check_value(self, action: argparse.Action, value: Any) -> None:
        if action.choices is not None and value not in action.choices:
            choices = ", ".join(map(repr, action.choices))
            raise argparse.ArgumentError(action, f"invalid choice (value not shown); choose from {choices}")

    def _get_value(self, action: argparse.Action, arg_string: str) -> Any:
        try:
            return super()._get_value(action, arg_string)
        except argparse.ArgumentError:
            name = getattr(action.type, "__name__", repr(action.type))
            raise argparse.ArgumentError(action, f"invalid {name} value (not shown)") from None


def _option_strings(parser: argparse.ArgumentParser) -> Set[str]:
    """Every option string the parser or any of its subparsers knows."""
    names: Set[str] = set()
    for action in parser._actions:
        names.update(action.option_strings)
        if isinstance(action, argparse._SubParsersAction):
            for child in action.choices.values():
                names.update(_option_strings(child))
    return names


def build_parser() -> argparse.ArgumentParser:
    # ``--db`` lives on the top-level parser (its default) and on every
    # subcommand (default SUPPRESS, so a value given after the command wins
    # and one given before survives).
    db_top = _ValueBlindParser(add_help=False)
    db_top.add_argument("--db", default=DEFAULT_DB, metavar="PATH",
                        help=f"sqlite file of the outbox (default: {DEFAULT_DB})")
    db_sub = _ValueBlindParser(add_help=False)
    db_sub.add_argument("--db", default=argparse.SUPPRESS, metavar="PATH", help=argparse.SUPPRESS)

    # ``allow_abbrev=False`` here and on every subparser (each is its own
    # parser): otherwise ``--blob TEXT`` would be taken as an abbreviation
    # of ``--blob-file`` and a blob typed on the command line would be
    # treated as a path and echoed back in the error.
    parser = _ValueBlindParser(
        prog="python3 -m jarvis_alerts.cli",
        description="Jarvis alert delivery: publish, register devices, run the worker.",
        parents=[db_top],
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p = sub.add_parser("publish", parents=[db_sub], allow_abbrev=False, help="store an alert and fan it out")
    p.add_argument("--profile", required=True)
    p.add_argument("--kind", required=True, help='e.g. "render_done"')
    p.add_argument("--title", required=True)
    p.add_argument("--body", required=True)
    p.add_argument("--priority", choices=["low", "normal", "high"], default="normal")
    p.add_argument("--dedupe", metavar="KEY", default=None,
                   help="repeats within the dedupe window collapse into one alert")
    p.add_argument("--data", metavar="JSON", default=None, help="extra payload, a JSON object")
    p.set_defaults(func=cmd_publish)

    p = sub.add_parser("register", parents=[db_sub], allow_abbrev=False, help="register a device's push subscription")
    p.add_argument("--profile", required=True)
    p.add_argument("--device", required=True)
    p.add_argument("--transport", required=True, choices=list(TRANSPORT_NAMES))
    p.add_argument("--blob-file", required=True, metavar="PATH",
                   help="file holding the subscription blob (never given on the command line)")
    p.set_defaults(func=cmd_register)

    p = sub.add_parser("unregister", parents=[db_sub], allow_abbrev=False, help="forget a device")
    p.add_argument("--profile", required=True)
    p.add_argument("--device", required=True)
    p.set_defaults(func=cmd_unregister)

    p = sub.add_parser("worker", parents=[db_sub], allow_abbrev=False, help="deliver due rows")
    p.add_argument("--once", action="store_true", help="one pass, then exit")
    p.add_argument("--idle", type=float, default=1.0, metavar="SECONDS",
                   help="wait after an empty pass (default 1.0)")
    p.add_argument("--batch", type=int, default=50, help="rows per pass (default 50)")
    p.add_argument("--lease", type=float, default=LEASE_S, metavar="SECONDS",
                   help=f"visibility timeout of a leased row (default {LEASE_S:g})")
    p.add_argument("--seed", default="0", help="seed of the backoff jitter stream (default 0)")
    p.set_defaults(func=cmd_worker)

    p = sub.add_parser("stats", parents=[db_sub], allow_abbrev=False, help="counts per row state, as JSON")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("dead", parents=[db_sub], allow_abbrev=False, help="list dead-lettered rows, newest first")
    p.add_argument("--limit", type=int, default=50)
    p.set_defaults(func=cmd_dead)

    p = sub.add_parser("requeue", parents=[db_sub], allow_abbrev=False,
                       help="put dead-lettered rows back in flight with a fresh attempt budget")
    p.add_argument("--row", type=int, action="append", metavar="ID", default=[],
                   help="a dead row id from ``dead``; repeatable")
    p.add_argument("--all", action="store_true", help="every dead row (or a profile's, or a device's)")
    p.add_argument("--profile", default=None, help="with --all: only this profile's rows")
    p.add_argument("--device", default=None, help="with --all --profile: only this device's rows")
    p.set_defaults(func=cmd_requeue)

    p = sub.add_parser("gate", parents=[db_sub], allow_abbrev=False, help="run the validation gate")
    p.add_argument("--profiles", type=int, required=True)
    p.add_argument("--alerts", type=int, required=True)
    p.add_argument("--seed", required=True, help="0x... or decimal")
    p.set_defaults(func=cmd_gate)

    return parser


def main(argv: Optional[Sequence[str]] = None, out: Optional[TextIO] = None,
         err: Optional[TextIO] = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), run the command, return the
    exit status.  Every error is one line on ``err``; no traceback."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = build_parser()
    try:
        # argparse prints to the process streams; route them to ours.
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args, unknown = parser.parse_known_args(argv)
    except SystemExit as exc:  # argparse already printed its message
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    if unknown:
        # argparse's own message would echo the values; a blob pasted on the
        # command line by mistake must not land in a terminal or a log.
        err.write(f"error: unrecognized arguments: {_describe_unknown(unknown, _option_strings(parser))}\n")
        return EXIT_USAGE
    try:
        return int(args.func(args, out, err))
    except Exception as exc:  # noqa: BLE001 - one line, exit 1, is the contract
        message = str(exc) or type(exc).__name__
        err.write(f"error: {message}\n")
        err.flush()
        return EXIT_FAIL


def _describe_unknown(tokens: Sequence[str], known: Set[str] = frozenset()) -> str:
    """Option names the CLI knows (given to the wrong command, say) are
    shown; every other token is counted, never shown, because a token that
    merely starts with ``-`` may be a base64url token or a pasted blob."""
    names = [t.split("=", 1)[0] for t in tokens if t.split("=", 1)[0] in known]
    values = len(tokens) - len(names)
    parts = names + ([f"({values} value{'s' if values != 1 else ''} not shown)"] if values else [])
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
