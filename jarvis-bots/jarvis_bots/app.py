"""The composition root: config in, a running supervisor out.

Everything else in :mod:`jarvis_bots` is deliberately unwired.  Bots take
injected probes, the framework takes injected bots, and no module reaches
a socket, a clock or a subprocess on its own.  That is the right shape for
testing and the wrong shape for running: something, once, has to say *what
this machine actually has*.  This is that something, and it is the only
file in the package that reads a config file or names a real service.

Why the page said "nothing inside"
----------------------------------
The launcher shipped before this did, so ``/api/bots/`` was served by a
registry with no bots in it.  That was correct -- the alternative was
inventing a GPU and a restock, and a dashboard that shows a plausible lie
is worse than one that shows an honest nothing -- but it is not useful.
The fix is not to relax that rule; it is to give the registry something
real to hold.

What it refuses to do
---------------------
* **Guess.**  A bot is built from config or not built at all.  The disk
  watch is not pointed at ``/`` because that was handy; the service watch
  invents no unit names.  A bot with no config is skipped and says so.
* **Hide a skip.**  :func:`build` returns the bots *and* a
  :class:`SkipNote` for every one it did not build, with the reason in
  words.  :func:`launcher_notes` puts those on the page, so "nothing
  inside" becomes "disk watch: no mountpoints configured" -- which a
  person can act on.
* **Fake the one thing it cannot supply.**  The Pokemon assistant needs a
  fetcher and a parser for a retailer's page, and this package ships
  neither: ``jarvis_poke.contracts`` keeps them injected precisely so that
  no retailer's scraping selectors live in this repository.  Config points
  at the app's own module; without one, the bot is skipped with that
  sentence as its reason.

Determinism and the clock
-------------------------
Exactly one wall clock enters the process, here, and is handed to every
bot and to the supervisor.  ``jarvis_poke.engine`` refuses collaborators
on a different clock, and a bot that can reach ``time.time()`` eventually
does, so the one call to :func:`time.time` in the whole package is the
default value of :func:`build`'s ``clock`` argument.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from jarvis_bots.contracts import Bot
from jarvis_bots.registry import BotRegistry
from jarvis_bots.supervisor import Supervisor

__all__ = [
    "AppError",
    "BuildResult",
    "SkipNote",
    "build",
    "build_supervisor",
    "launcher_notes",
    "load_config",
]


class AppError(ValueError):
    """A config this file will not pretend to understand."""


@dataclass(frozen=True)
class SkipNote:
    """One bot that was not built, and why -- in words for the page.

    ``fixable`` separates "you have not configured this yet" from "this
    machine cannot run it".  The page shows the first as a prompt and the
    second as a fact.
    """

    bot_id: str
    reason: str
    fixable: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.bot_id, "reason": self.reason, "fixable": self.fixable}


@dataclass(frozen=True)
class BuildResult:
    """What :func:`build` found: the bots, and an honest account of the rest."""

    bots: Tuple[Bot, ...]
    skipped: Tuple[SkipNote, ...]

    @property
    def registry(self) -> BotRegistry:
        return BotRegistry(self.bots)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "built": [bot.info.id for bot in self.bots],
            "skipped": [note.as_dict() for note in self.skipped],
        }


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------


def load_config(path: str) -> Dict[str, Any]:
    """Read a JSON config, refusing anything that is not an object.

    No defaults are filled in here.  A key that is absent means "do not
    build that bot", and a config that half-parses is an error rather than
    a machine that silently watches less than the owner thinks it does.
    """
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise AppError(f"{path}: the config must be a JSON object, got {type(obj).__name__}")
    bots = obj.get("bots", {})
    if not isinstance(bots, dict):
        raise AppError(f"{path}: 'bots' must be an object of id -> settings")
    return obj


def _section(config: Mapping[str, Any], bot_id: str) -> Optional[Mapping[str, Any]]:
    """The settings for one bot, or ``None`` when it is absent or off.

    ``"enabled": false`` and a missing section are the same answer to the
    question this function asks; they differ only in the skip note.
    """
    bots = config.get("bots") or {}
    block = bots.get(bot_id)
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise AppError(f"bots.{bot_id} must be an object, got {type(block).__name__}")
    if block.get("enabled") is False:
        return None
    return block


def _import_callable(spec: Any, what: str) -> Callable[..., Any]:
    """``"package.module:name"`` -> the callable.

    The app's own fetcher and parser live outside this package by design,
    so config has to be able to name them.  The form is explicit -- a
    colon, not a guess at where the module ends -- and the result is
    checked to be callable before anything depends on it.
    """
    if not isinstance(spec, str) or ":" not in spec:
        raise AppError(
            f"{what} must be \"package.module:name\"; got {spec!r}"
        )
    module_name, _, attribute = spec.partition(":")
    try:
        module = import_module(module_name)
    except ImportError as exc:
        raise AppError(f"{what}: cannot import {module_name!r} ({exc})") from None
    found = getattr(module, attribute, None)
    if not callable(found):
        raise AppError(f"{what}: {spec!r} is not callable")
    return found


# --------------------------------------------------------------------------
# the builders, one per bot
# --------------------------------------------------------------------------
#
# Each returns a bot or raises AppError with a sentence for the page.  They
# are separate functions rather than a table of lambdas so that a bot that
# grows a dependency changes in one place, and so a traceback names the
# bot that could not be built.


def _build_gpu(block: Mapping[str, Any], clock: Callable[[], float]) -> Bot:
    from jarvis_bots.bots import gpu_bot

    probe_spec = block.get("probe")
    probe = (
        _import_callable(probe_spec, "bots.gpu.probe") if probe_spec
        else gpu_bot.nvidia_smi_probe
    )
    kwargs: Dict[str, Any] = {}
    for key in ("temp_action_c", "temp_notice_c"):
        if key in block:
            kwargs[key] = float(block[key])
    return gpu_bot.build(clock=clock, probe=probe, **kwargs)


def _build_services(block: Mapping[str, Any], clock: Callable[[], float]) -> Bot:
    from jarvis_bots.bots import service_bot

    names = block.get("services")
    if not isinstance(names, (list, tuple)) or not names:
        raise AppError(
            "no services listed. Add the unit names you want watched, e.g. "
            "\"services\": [\"comfyui.service\", \"ollama.service\"]"
        )
    kwargs: Dict[str, Any] = {}
    for key, cast in (("restart_threshold", int), ("window_s", float),
                      ("memory_ceiling_bytes", int), ("min_uptime_s", float)):
        if key in block and block[key] is not None:
            kwargs[key] = cast(block[key])
    probe_spec = block.get("probe")
    if probe_spec:
        kwargs["probe"] = _import_callable(probe_spec, "bots.services.probe")
    return service_bot.build([str(name) for name in names], clock=clock, **kwargs)


def _build_disk(block: Mapping[str, Any], clock: Callable[[], float]) -> Bot:
    from jarvis_bots.bots import disk_bot

    mounts = block.get("mountpoints")
    probe_spec = block.get("probe")
    if not probe_spec and (not isinstance(mounts, (list, tuple)) or not mounts):
        raise AppError(
            "no mountpoints configured. A disk watch that picks its own "
            "filesystems watches the container's overlay and tells you "
            "nothing; list the ones you care about, e.g. "
            "\"mountpoints\": [\"/\", \"/scratch\"]"
        )
    kwargs: Dict[str, Any] = {}
    for key, cast in (("percent_threshold", float),
                      ("projected_days_threshold", float),
                      ("drop_bytes", int)):
        if key in block and block[key] is not None:
            kwargs[key] = cast(block[key])
    if probe_spec:
        kwargs["probe"] = _import_callable(probe_spec, "bots.disk.probe")
    if mounts:
        kwargs["mountpoints"] = [str(m) for m in mounts]
    return disk_bot.build(clock=clock, **kwargs)


def _build_health(block: Mapping[str, Any], clock: Callable[[], float]) -> Bot:
    from jarvis_bots.bots import health_bot

    rows = block.get("checks")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise AppError(
            "no checks configured. List the urls that being up actually "
            "means, e.g. the frontend, its api, and one built asset"
        )
    checks = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AppError(f"each check must be an object; got {type(row).__name__}")
        fields = {str(k): v for k, v in row.items()}
        try:
            checks.append(health_bot.Check(**fields))
        except TypeError as exc:
            raise AppError(f"check {fields.get('name')!r}: {exc}") from None
    kwargs: Dict[str, Any] = {}
    pairs = block.get("pairs")
    if isinstance(pairs, Mapping):
        kwargs["pairs"] = {str(k): str(v) for k, v in pairs.items()}
    probe_spec = block.get("probe")
    if probe_spec:
        kwargs["probe"] = _import_callable(probe_spec, "bots.health.probe")
    return health_bot.build(checks, clock, **kwargs)


def _build_poke(block: Mapping[str, Any], clock: Callable[[], float]) -> Bot:
    """The Pokemon assistant, with its drop windows if config names any.

    This is the one bot this package cannot finish on its own.  A fetcher
    and a parser for a retailer's page are the app's, by design: keeping
    them injected is what stops a retailer's scraping selectors from
    living in this repository at all.  Without both named in config there
    is nothing to build, and saying so is more use than a bot that watches
    a placeholder.
    """
    from jarvis_poke.catalog import Catalog
    from jarvis_poke.engine import DecisionEngine
    from jarvis_poke.prices import PriceHistory
    from jarvis_poke.rules import RuleSet
    from jarvis_poke.sources import PollScheduler, load_policies
    from jarvis_poke.store import PokeStore
    from jarvis_bots.bots import poke_bot

    fetcher_spec, parser_spec = block.get("fetcher"), block.get("parser")
    if not fetcher_spec or not parser_spec:
        raise AppError(
            "no fetcher and parser configured. This package ships no "
            "retailer's page parser on purpose; point \"fetcher\" and "
            "\"parser\" at your own, as \"your.module:name\""
        )
    fetcher = _import_callable(fetcher_spec, "bots.poke.fetcher")
    parser = _import_callable(parser_spec, "bots.poke.parser")

    # The catalog and the source policies ship with the package (the
    # placeholder retailers), so a path is optional; the watchlist and the
    # budget are the owner's and live in their store.
    catalog = Catalog.load(block.get("catalog"), block.get("sources"))
    policies = load_policies(block.get("sources"))

    db_path = block.get("db")
    if not db_path:
        raise AppError(
            "no \"db\" path. The watchlist, the budget and the price history "
            "live in one sqlite file; without it nothing is remembered "
            "between runs and every restart re-alerts"
        )
    store = PokeStore(str(db_path), clock)
    rules = RuleSet(store.load_rules(), budget=store.load_budget())
    history = PriceHistory()
    engine = DecisionEngine(catalog, rules, history, clock)
    scheduler = PollScheduler(catalog, policies, clock)

    snipe = _build_snipe(block, scheduler, clock)
    return poke_bot.PokeBot(
        catalog, rules, history, engine, scheduler, fetcher, parser, clock,
        snipe=snipe,
    )


def _build_snipe(block: Mapping[str, Any], scheduler: Any,
                 clock: Callable[[], float]) -> Any:
    """The drop windows, when config lists any.

    No windows means no snipe controller and a bot that behaves exactly as
    it did before they existed -- which is the point of the controller
    being optional.  The probes are wired here because this is the only
    place that knows where the alert registry and the rules live; an
    unwired probe reports "unknown", never "fine", so a half-wired
    composition root is visible on the arming report rather than silent.
    """
    from jarvis_poke.snipe import DropWindow, SnipeController, SnipePlan, daily_windows

    rows = block.get("windows")
    if not isinstance(rows, (list, tuple)) or not rows:
        return None
    windows: List[DropWindow] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise AppError(f"each window must be an object; got {type(row).__name__}")
        fields = {str(k): v for k, v in row.items()}
        try:
            if "first_day" in fields:
                name = fields.pop("name")
                source = fields.pop("source")
                windows.extend(daily_windows(name, source, **fields))
            else:
                windows.append(DropWindow(**fields))
        except TypeError as exc:
            raise AppError(f"window {fields.get('name')!r}: {exc}") from None
    return SnipeController(scheduler, SnipePlan.of(windows))


#: id -> builder.  The order is the order the launcher lists them in.
BUILDERS: Dict[str, Callable[[Mapping[str, Any], Callable[[], float]], Bot]] = {
    "poke": _build_poke,
    "gpu": _build_gpu,
    "services": _build_services,
    "disk": _build_disk,
    "health": _build_health,
}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build(config: Mapping[str, Any], *,
          clock: Optional[Callable[[], float]] = None) -> BuildResult:
    """Build every bot the config asks for; note every one it does not.

    One bot that cannot be built never stops the others: a machine with no
    GPU should still get its disk watched.  The failure is recorded as a
    :class:`SkipNote` and shown on the page.

    ``clock`` defaults to :func:`time.time`, the one wall clock in the
    package, and is shared by every bot and the supervisor so that
    ``jarvis_poke``'s same-clock check passes and a snapshot means the
    same thing to all of them.
    """
    if not isinstance(config, Mapping):
        raise AppError(f"config must be a mapping, got {type(config).__name__}")
    tick = clock if clock is not None else time.time
    bots: List[Bot] = []
    skipped: List[SkipNote] = []
    for bot_id, builder in BUILDERS.items():
        block = _section(config, bot_id)
        if block is None:
            skipped.append(SkipNote(bot_id, "not configured", fixable=True))
            continue
        try:
            bots.append(builder(block, tick))
        except AppError as exc:
            skipped.append(SkipNote(bot_id, str(exc), fixable=True))
        except Exception as exc:  # noqa: BLE001 - one bot never breaks the rest
            # The type name only: a config error's message can quote a
            # path or a url, and this string goes to a web page.
            skipped.append(
                SkipNote(bot_id, f"could not be built ({type(exc).__name__})",
                         fixable=False)
            )
    return BuildResult(tuple(bots), tuple(skipped))


def build_supervisor(config: Mapping[str, Any], *,
                     clock: Optional[Callable[[], float]] = None,
                     store: Any = None, alerts: Any = None) -> Tuple[Supervisor, BuildResult]:
    """:func:`build`, plus the supervisor that runs the result.

    ``store`` and ``alerts`` stay the caller's: persistence and delivery
    are the app's seams and this file has no business choosing a file path
    or a push transport.
    """
    result = build(config, clock=clock)
    tick = clock if clock is not None else time.time
    supervisor = Supervisor(result.registry, tick, store=store, alerts=alerts)
    return supervisor, result


def launcher_notes(result: BuildResult) -> List[Dict[str, Any]]:
    """The skips, JSON-ready, for the launcher's empty state.

    ``jarvis_bots/web/bots.html`` renders a card per bot; these are what
    it shows underneath when a bot is absent, so the page answers "why is
    there nothing here" instead of raising the question.
    """
    return [note.as_dict() for note in result.skipped]


# --------------------------------------------------------------------------
# python3 -m jarvis_bots.app <config.json>
# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Print what this config would build, and why it would skip the rest.

    Builds nothing lasting and runs no round: a dry run an owner can use
    to see the reasons before wiring the config into the app.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python3 -m jarvis_bots.app")
    parser.add_argument("config", help="path to the JSON config")
    args = parser.parse_args(argv)
    try:
        result = build(load_config(args.config))
    except (AppError, OSError, json.JSONDecodeError) as exc:
        print(f"config: {exc}")
        return 2
    print(json.dumps(result.as_dict(), indent=2))
    return 0 if result.bots else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
