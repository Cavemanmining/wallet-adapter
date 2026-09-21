"""Service watch -- a Jarvis bot that knows "running" from "failing fast".

Why this bot exists
-------------------
A ComfyUI unit on the owner's machine crash-looped about **15,000 times**
against a GPU that had vanished, and nobody noticed for days.  Every tool
that looked at it said ``active (running)``, because systemd had just
restarted it again, one second ago, for the fifteen-thousandth time.

A service that is "running" because something keeps restarting it is not
running; it is failing fast.  Telling those two apart is the whole point of
this module, and it is why the crash-loop rule here fires **while
``ActiveState`` is ``active``** rather than waiting for ``failed``.  A
watcher that only reports ``failed`` would have stayed silent through the
entire incident, which is precisely what happened.

What it watches, and how it is wired
------------------------------------
The bot never shells out.  It is handed a ``probe()`` that returns a list
of :class:`ServiceSample`, exactly as ``jarvis_bots/base.py`` has the
weather example handed a ``fetch``: "this package opens no sockets", and a
test hands in a stub.  :func:`systemctl_probe` is the real one the app
injects::

    from functools import partial
    from jarvis_bots.bots.service_bot import ServiceBot, systemctl_probe

    names = ["comfyui.service", "ollama.service"]
    bot = ServiceBot(names, clock=clock, probe=partial(systemctl_probe, names))

:func:`systemctl_probe` is split so the half that matters can be tested
without a machine: :func:`systemctl_command` builds the argv and
:func:`parse_systemctl_show` turns captured ``key=value`` output into a
sample.  Only :func:`_run_systemctl` touches ``subprocess``.

The rules, and the key each one owns
------------------------------------
``svc:crashloop:<name>``  **The headline.**  More than
    ``restart_threshold`` restarts inside ``window_s`` -- ACTION, whatever
    ``ActiveState`` says.  The text names the service, the count, the
    window and the last exit result.
``svc:failed:<name>``     ``ActiveState=failed`` -- ACTION.
``svc:stopped:<name>``    Was active, is now inactive, and no restart went
    with it: something stopped it and nothing is bringing it back --
    ACTION, and deliberately a *different* key from a crash, because "it
    is gone" and "it keeps dying" are different questions.
``svc:absent:<name>``     A watched unit the probe cannot find at all --
    ACTION.  A typo in a unit name and a deleted unit look the same from
    here, and both mean the owner is watching nothing.
``svc:oom:<name>``        ``Result=oom-kill`` seen at any point -- ACTION,
    and the text says "out-of-memory" in words, because an OOM kill is a
    different fix from a crash.
``svc:memory:<name>``     Over ``memory_ceiling_bytes`` -- NOTICE.  Growth
    is a warning, not a decision.
``svc:probe``             The probe itself failing three times in a row --
    ACTION.  A blind watcher is worth saying out loud.

Three deliberate non-rules, each of which would otherwise cost the owner
their trust in the badge:

* **an uptime under 60s is not on its own an alert.**  A deploy restarts
  things; one restart is normal life.  It still *counts* toward the
  crash-loop window, which is where a restart becomes evidence.
* **flapping ``active`` -> ``activating`` -> ``active`` is not its own
  rule.**  It is a crash loop and is reported as one, under one key, once.
  Two keys for one fault is two notifications for one decision.
* **a crash loop suppresses ``failed`` and ``stopped`` for that unit.**
  While a unit is looping, its momentary ``failed`` and ``inactive`` dips
  are the loop, not news of their own.  The crash-loop text carries the
  current state, so nothing is hidden.

Every rule is a *condition*, evaluated fresh each tick.  A rule that is
true and unopened raises ACTION; a rule that has gone false closes its key
the way the framework expects -- the same key, below ACTION, with
``resolved=True`` in the data -- which is verbatim what ``PokeBot._resolve``
and the scaffold template do, and what ``Supervisor.RESOLVED_FLAG`` reads.
So a service coming back healthy empties the badge on its own.

Failure is an event, not an exception
-------------------------------------
``tick`` never raises.  contracts.py allows it, but four failures in a row
quarantine a bot for half an hour, and ``systemctl`` timing out once is not
a reason to stop watching a machine for thirty minutes.  A raising probe is
an ERROR event; the third consecutive failure escalates to ACTION under
``svc:probe``, and a successful probe resolves it.  Only an exception's
*type name* is kept, never its message, which may quote a path or a
command line.

Time is injected, always: nothing in this module calls ``time.time()``.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterable as _IterableABC
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from jarvis_bots.base import BaseBot, Clock
from jarvis_bots.contracts import BotInfo, BotState, BotStatus, Event, Severity

__all__ = [
    "BOT_ID",
    "INFO",
    "SNAPSHOT_VERSION",
    "DEFAULT_RESTART_THRESHOLD",
    "DEFAULT_WINDOW_S",
    "DEFAULT_MIN_UPTIME_S",
    "DEFAULT_PROBE_TIMEOUT_S",
    "PROBE_FAILURES_BEFORE_ACTION",
    "PROBE_ATTENTION_KEY",
    "SHOW_PROPERTIES",
    "ACTIVE_STATES",
    "ServiceSample",
    "ServiceBot",
    "ServiceBotError",
    "systemctl_command",
    "parse_systemctl_show",
    "systemctl_probe",
    "crashloop_key",
    "failed_key",
    "stopped_key",
    "absent_key",
    "oom_key",
    "memory_key",
    "build",
]


#: The id the supervisor, the launcher and every event carry.
BOT_ID = "services"

#: Static identity, declared once (contracts.BotInfo).  ``radar`` is the
#: launcher glyph for a watcher (``jarvis_bots/web/README.md``); two
#: minutes is fast enough to catch a loop while it is still looping and
#: cheap enough that a handful of ``systemctl show`` calls cost nothing.
INFO = BotInfo(
    id=BOT_ID,
    name="Service watch",
    blurb=(
        "Watches your services and tells you when one is crash-looping, "
        "failed, stopped or out of memory -- including the ones systemd "
        "keeps restarting so fast they still look alive."
    ),
    kind="radar",
    interval_s=120.0,
    href="/bots/services",
)

#: More than this many restarts inside :data:`DEFAULT_WINDOW_S` is a crash
#: loop.  Strictly more: three restarts in ten minutes is a bad deploy
#: afternoon, four is a loop.
DEFAULT_RESTART_THRESHOLD = 3

#: The rolling window restarts are counted in.  Ten minutes is long enough
#: that a loop restarting every few seconds is unmistakable and short
#: enough that yesterday's maintenance has fallen out of it.
DEFAULT_WINDOW_S = 600.0

#: An uptime under this is *not* an alert on its own; see the module
#: docstring.  It is reported on the card and in the crash-loop event's
#: data, because "up for 4 seconds" is what makes the count believable.
DEFAULT_MIN_UPTIME_S = 60.0

#: How long :func:`_run_systemctl` waits for one ``systemctl show``.
DEFAULT_PROBE_TIMEOUT_S = 10.0

#: Consecutive probe failures before the bot stops merely logging and asks
#: the owner to look.  Three, so one timeout and one transient are quiet.
PROBE_FAILURES_BEFORE_ACTION = 3

#: The one key that is not per-service: the watcher itself is blind.
PROBE_ATTENTION_KEY = "svc:probe"

#: Bumped when :meth:`ServiceBot.snapshot` changes shape.
SNAPSHOT_VERSION = 1

#: ``ActiveState`` values that mean the unit is up as far as systemd is
#: concerned.  ``activating`` is included on purpose: a unit that spends
#: its life in ``activating`` is flapping, and flapping is caught by the
#: restart counter, not by calling it down.
ACTIVE_STATES = ("active", "activating")

#: The properties :func:`systemctl_command` asks for, in a fixed order so
#: two runs build byte-identical argv.
SHOW_PROPERTIES: Tuple[str, ...] = (
    "Id",
    "LoadState",
    "ActiveState",
    "SubState",
    "NRestarts",
    "ActiveEnterTimestamp",
    "MainPID",
    "MemoryCurrent",
    "ExecMainStatus",
    "Result",
)

#: What systemd prints for "no value": ``MemoryCurrent`` on a dead unit and
#: the 64-bit sentinel a cgroup-less unit reports.
_UNSET_TOKENS = frozenset({"", "[not set]", "-", "infinity", "18446744073709551615"})


class ServiceBotError(ValueError):
    """A wiring mistake: no services to watch, or a probe that is not
    callable.  Raised from ``__init__`` only -- the supervisor catches what
    a *tick* raises, but a bot built wrong should fail on the line that
    built it, which is the stance :mod:`jarvis_bots.registry` takes."""


# --------------------------------------------------------------------------
# what a probe returns
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceSample:
    """One reading of one unit: what ``systemctl show`` would say about it.

    Frozen, because a sample is an observation of a moment and nothing
    downstream has any business editing it.

    ``n_restarts`` is *cumulative* -- systemd's ``NRestarts``, which counts
    up for the life of the unit and is what makes a rolling window
    possible: the bot stores the differences it sees and ages them out.

    ``active_enter_timestamp`` is unix seconds or ``None``.  ``None`` is
    honest and common: a unit that has never started has no such moment,
    and a systemd too old for ``--timestamp=unix`` prints a date this
    module refuses to guess at.
    """

    name: str
    active_state: str
    sub_state: str = ""
    n_restarts: int = 0
    active_enter_timestamp: Optional[float] = None
    main_pid: Optional[int] = None
    memory_bytes: Optional[int] = None
    exit_code: Optional[int] = None
    result: str = "success"

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"a service sample needs a unit name; got {self.name!r}")
        if not isinstance(self.active_state, str) or not self.active_state:
            raise ValueError(
                f"{self.name}: active_state must be one of 'active', 'inactive', "
                f"'failed', 'activating', 'deactivating'; got {self.active_state!r}"
            )

    @property
    def is_up(self) -> bool:
        return self.active_state in ACTIVE_STATES

    def uptime_s(self, now: float) -> Optional[float]:
        """Seconds since the unit last entered ``active``, or ``None``.

        Never negative: a clock that has stepped backwards should read as
        "just started", not as a service that starts in the future.
        """
        if self.active_enter_timestamp is None or not self.is_up:
            return None
        return max(0.0, float(now) - float(self.active_enter_timestamp))


# --------------------------------------------------------------------------
# the real probe: argv, parser, and the one call that shells out
# --------------------------------------------------------------------------


def systemctl_command(name: str) -> List[str]:
    """The argv for one unit: ``systemctl show <name> --property=...``.

    One invocation per unit rather than one for all of them.  ``systemctl
    show a b`` separates units with a blank line and drops the ones it
    cannot find, so a single call cannot tell "not found" from "the
    separator moved in this systemd"; one call per unit makes an absent
    unit an unambiguous empty answer.

    ``--timestamp=unix`` is what makes ``ActiveEnterTimestamp`` a number
    (``@1700000000``) instead of a localised date.  An older systemd
    ignores nothing and simply prints the date; the parser then reports
    ``None`` rather than guessing at a month name in an unknown locale.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"a unit name must be a non-empty string; got {name!r}")
    return [
        "systemctl",
        "show",
        name,
        "--no-pager",
        "--timestamp=unix",
        "--property=" + ",".join(SHOW_PROPERTIES),
    ]


def parse_systemctl_show(name: str, text: str) -> Optional[ServiceSample]:
    """Turn captured ``key=value`` output into a :class:`ServiceSample`.

    Returns ``None`` when the output describes a unit that is not there --
    ``LoadState=not-found`` (systemd's answer for a unit it has never
    heard of), or no usable properties at all.  "Absent" has exactly one
    representation in this module, a name missing from the probe's list,
    so the bot has one rule for it and not two.

    The unit's own ``Id`` is ignored in favour of the requested ``name``:
    the bot looks its watched names up in the probe's answer, and a unit
    that is an alias (``comfyui`` -> ``comfyui.service``) would otherwise
    come back under a name the owner never wrote.

    Unknown keys are skipped, not rejected: systemd adds properties
    between releases, and a parser that dies on an unfamiliar line would
    make a machine upgrade look like an outage.
    """
    if not isinstance(text, str):
        raise TypeError(f"systemctl output must be text; got {type(text).__name__}")

    fields: Dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key:
            fields[key] = value.strip()

    if not fields:
        return None
    load_state = fields.get("LoadState", "").lower()
    if load_state in ("not-found", "bad-setting", "masked"):
        # masked is included on purpose: a masked unit cannot run, and
        # reporting it as "inactive" would let the stopped rule claim it
        # every tick with no way for the owner to act from the badge.
        return None
    active_state = fields.get("ActiveState", "").strip()
    if not active_state:
        return None

    return ServiceSample(
        name=name,
        active_state=active_state,
        sub_state=fields.get("SubState", ""),
        n_restarts=_as_int(fields.get("NRestarts"), 0) or 0,
        active_enter_timestamp=_as_unix(fields.get("ActiveEnterTimestamp")),
        main_pid=_as_positive_int(fields.get("MainPID")),
        memory_bytes=_as_positive_int(fields.get("MemoryCurrent"), allow_zero=True),
        exit_code=_as_int(fields.get("ExecMainStatus"), None),
        result=fields.get("Result", "") or "success",
    )


def _run_systemctl(argv: Sequence[str], timeout_s: float) -> str:
    """The only place in this package that starts a process.

    Returns stdout even when the exit status is non-zero: ``systemctl
    show`` reports an unknown unit on stderr with a non-zero status in
    some versions and with ``LoadState=not-found`` and status 0 in others,
    and both mean the same thing to :func:`parse_systemctl_show`.  A
    missing binary or a timeout *does* raise, because that is the watcher
    being blind rather than a unit being absent, and the bot has a rule
    for that.
    """
    completed = subprocess.run(  # noqa: S603 - fixed argv, no shell
        list(argv),
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    return completed.stdout or ""


def systemctl_probe(
    names: Iterable[str],
    *,
    run: Optional[Callable[[Sequence[str], float], str]] = None,
    timeout_s: float = DEFAULT_PROBE_TIMEOUT_S,
) -> List[ServiceSample]:
    """Read the named units, in order, and return the ones that exist.

    This is the probe the app injects (``partial(systemctl_probe,
    names)``); the bot never calls it directly and holds no reference to
    ``subprocess``.  ``run`` is a seam for tests and for a wiring that
    wants to reach a machine some other way -- over ssh, say -- without
    reimplementing the parser.

    A unit that does not exist is simply missing from the result, which is
    what ``svc:absent:<name>`` keys on.
    """
    runner = run if run is not None else _run_systemctl
    samples: List[ServiceSample] = []
    for name in names:
        output = runner(systemctl_command(name), float(timeout_s))
        sample = parse_systemctl_show(name, output)
        if sample is not None:
            samples.append(sample)
    return samples


# --------------------------------------------------------------------------
# keys: built by a function each, so the raise and the resolve cannot drift
# --------------------------------------------------------------------------


def crashloop_key(name: str) -> str:
    return f"svc:crashloop:{name}"


def failed_key(name: str) -> str:
    return f"svc:failed:{name}"


def stopped_key(name: str) -> str:
    return f"svc:stopped:{name}"


def absent_key(name: str) -> str:
    return f"svc:absent:{name}"


def oom_key(name: str) -> str:
    return f"svc:oom:{name}"


def memory_key(name: str) -> str:
    return f"svc:memory:{name}"


# --------------------------------------------------------------------------
# the bot
# --------------------------------------------------------------------------


class ServiceBot(BaseBot):
    """Watches a list of units and says which of them is not really running.

    Implements ``contracts.Bot``: ``info``, :meth:`tick` and :meth:`status`
    are the required three; :meth:`snapshot` and :meth:`restore` carry the
    rolling window across a restart, without which a process restart would
    forgive every loop in progress.
    """

    info = INFO

    def __init__(
        self,
        services: Sequence[str],
        *,
        clock: Clock,
        probe: Callable[[], Sequence[ServiceSample]],
        restart_threshold: int = DEFAULT_RESTART_THRESHOLD,
        window_s: float = DEFAULT_WINDOW_S,
        memory_ceiling_bytes: Optional[int] = None,
        min_uptime_s: float = DEFAULT_MIN_UPTIME_S,
    ) -> None:
        super().__init__(clock)

        watched = [str(n).strip() for n in (services or []) if str(n).strip()]
        if not watched:
            raise ServiceBotError(
                "ServiceBot needs at least one service name to watch: a bot "
                "watching nothing reports 'all healthy' for ever"
            )
        # Deduplicated, order kept: the card and every event list services
        # in the order the owner wrote them.
        seen: Dict[str, None] = {}
        for name in watched:
            seen.setdefault(name, None)
        self._services: Tuple[str, ...] = tuple(seen)

        if not callable(probe):
            raise ServiceBotError(
                f"ServiceBot needs an injected probe() -> list[ServiceSample]; "
                f"got {type(probe).__name__}. This bot does not shell out: wire "
                f"it with partial(systemctl_probe, names)"
            )
        self._probe = probe

        threshold = int(restart_threshold)
        if threshold < 1:
            raise ServiceBotError("restart_threshold must be at least 1")
        self._threshold = threshold

        window = float(window_s)
        if window <= 0:
            raise ServiceBotError("window_s must be positive")
        self._window_s = window

        if memory_ceiling_bytes is not None:
            ceiling = int(memory_ceiling_bytes)
            if ceiling <= 0:
                raise ServiceBotError("memory_ceiling_bytes must be positive or None")
            self._memory_ceiling: Optional[int] = ceiling
        else:
            # Off by default and on purpose: this bot cannot know what is a
            # lot of memory for a model server, and a ceiling guessed here
            # would be a NOTICE every tick on the machine that needed the
            # crash-loop rule most.
            self._memory_ceiling = None

        self._min_uptime_s = max(0.0, float(min_uptime_s))

        # --- state, all of it snapshotted -------------------------------
        #: name -> [[at, delta], ...] restarts seen inside the window.
        self._restarts: Dict[str, List[List[float]]] = {}
        #: name -> the last sample, as a JSON-able row.
        self._last_seen: Dict[str, Dict[str, Any]] = {}
        #: name -> has this unit ever been seen up?  Without it a unit that
        #: is inactive the first time the bot ever runs would be reported
        #: as "stopped", which it is not: it is switched off.
        self._seen_up: Dict[str, bool] = {}
        #: name -> an OOM kill has been seen and not yet lived down.
        self._oom: Dict[str, bool] = {}
        #: attention key -> when it was raised.
        self._open: Dict[str, float] = {}
        self._probe_failures = 0
        self._ticks = 0
        self._last_tick_at = 0.0
        self._last_error = ""

    # -- config, read-only ---------------------------------------------------

    @property
    def services(self) -> Tuple[str, ...]:
        return self._services

    @property
    def window_s(self) -> float:
        return self._window_s

    @property
    def restart_threshold(self) -> int:
        return self._threshold

    def open_attention_keys(self) -> List[str]:
        """The keys this bot believes are open, sorted.  The supervisor owns
        the badge; this is the bot's own record, exposed so a wiring can
        check the two agree."""
        return sorted(self._open)

    # -- the work ------------------------------------------------------------

    def tick(self, now: float) -> Sequence[Event]:
        """One pass over the watched units.

        Never raises.  contracts.py allows it, but a ``systemctl`` that
        timed out twice is not a reason to quarantine the watcher for half
        an hour -- and a quarantined watcher during a crash loop is this
        bot's own failure mode.
        """
        at = float(now)
        try:
            samples = self._probe()
            index = self._index(samples)
        except Exception as exc:  # noqa: BLE001 - deliberate; see the docstring
            return tuple(self._probe_failed(exc, at))

        events: List[Event] = list(self._probe_recovered(at))
        self._ticks += 1
        self._last_tick_at = at
        self._last_error = ""

        for name in self._services:
            events.extend(self._service_events(name, index.get(name), at))
        return tuple(events)

    def _index(self, samples: Any) -> Dict[str, ServiceSample]:
        """The probe's answer, checked and keyed by name.

        A probe returning something that is not a sequence of
        :class:`ServiceSample` is a wiring fault, and raising here puts it
        down the same path as a probe that threw: an ERROR event that
        names the type, never a half-read round that quietly reports every
        unit absent.
        """
        if isinstance(samples, (str, bytes)) or not isinstance(samples, _IterableABC):
            raise TypeError(
                f"probe() must return a sequence of ServiceSample; got "
                f"{type(samples).__name__}"
            )
        index: Dict[str, ServiceSample] = {}
        for sample in samples:
            if not isinstance(sample, ServiceSample):
                raise TypeError(
                    f"probe() returned a {type(sample).__name__}, not a "
                    f"ServiceSample"
                )
            index[sample.name] = sample
        return index

    # -- probe health ---------------------------------------------------------

    def _probe_failed(self, exc: BaseException, now: float) -> List[Event]:
        """A probe that raised.  An event, never an exception.

        Only the type name is kept: a ``CalledProcessError`` stringifies to
        the command line it ran, and an event is rendered on a page.
        """
        self._probe_failures += 1
        self._last_error = type(exc).__name__
        self._last_tick_at = now
        count = self._probe_failures
        if count < PROBE_FAILURES_BEFORE_ACTION:
            return [
                self.event(
                    Severity.ERROR,
                    f"Could not read service state ({self._last_error}); "
                    f"{_plural(count, 'failure')} in a row.",
                    href=INFO.href,
                    failures=count,
                    error=self._last_error,
                )
            ]
        # Third in a row: the watcher is blind, and a blind watcher during a
        # crash loop is how the incident this bot exists for happened twice.
        return self._raise_key(
            PROBE_ATTENTION_KEY,
            Severity.ACTION,
            f"Service watch is blind: {_plural(count, 'probe failure')} in a "
            f"row ({self._last_error}). Nothing on this machine is being "
            f"checked until it reads again.",
            now,
            failures=count,
            error=self._last_error,
        )

    def _probe_recovered(self, now: float) -> List[Event]:
        self._probe_failures = 0
        return self._clear_key(
            PROBE_ATTENTION_KEY,
            "Service watch can read systemd again.",
            now,
        )

    # -- one service ----------------------------------------------------------

    def _service_events(
        self, name: str, sample: Optional[ServiceSample], now: float
    ) -> List[Event]:
        if sample is None:
            return self._absent_events(name, now)

        events: List[Event] = self._clear_key(
            absent_key(name), f"{name} is back in systemd's list.", now
        )

        delta = self._record_restarts(name, sample, now)
        window_restarts = self._window_restarts(name, now)
        if sample.is_up:
            self._seen_up[name] = True

        oomed_now = sample.result == "oom-kill"
        if oomed_now:
            self._oom[name] = True

        crash = window_restarts > self._threshold
        # A loop is one fault: while it is open, this unit's momentary
        # 'failed' and 'inactive' readings are the loop itself, not news.
        failed = sample.active_state == "failed" and not crash
        stopped = (
            not crash
            and sample.active_state == "inactive"
            and bool(self._seen_up.get(name))
            and delta == 0
        )
        if sample.active_state == "active" and not oomed_now and not crash:
            # Healthy again: an old OOM stops counting once the unit has
            # been seen running clean, otherwise the key could never close.
            self._oom.pop(name, None)
        oom = bool(self._oom.get(name))
        over_memory = (
            self._memory_ceiling is not None
            and sample.memory_bytes is not None
            and sample.memory_bytes > self._memory_ceiling
        )

        events.extend(
            self._condition(
                crash,
                crashloop_key(name),
                Severity.ACTION,
                lambda: self._crashloop_text(
                    name, sample, window_restarts, sample.uptime_s(now)
                ),
                f"{name} has settled: no more than "
                f"{_plural(self._threshold, 'restart')} in the last "
                f"{_duration(self._window_s)}.",
                now,
                service=name,
                restarts=window_restarts,
                window_s=self._window_s,
                active_state=sample.active_state,
                sub_state=sample.sub_state,
                result=sample.result,
                exit_code=sample.exit_code,
                n_restarts=sample.n_restarts,
                uptime_s=sample.uptime_s(now),
            )
        )
        events.extend(
            self._condition(
                failed,
                failed_key(name),
                Severity.ACTION,
                lambda: (
                    f"{name} has failed ({sample.sub_state or 'failed'}"
                    f"{_result_phrase(sample)}). It is not running and "
                    f"systemd is not retrying."
                ),
                f"{name} is no longer failed.",
                now,
                service=name,
                active_state=sample.active_state,
                result=sample.result,
                exit_code=sample.exit_code,
            )
        )
        events.extend(
            self._condition(
                stopped,
                stopped_key(name),
                Severity.ACTION,
                lambda: (
                    f"{name} was running and is now stopped, with no restart "
                    f"behind it. Something took it down and nothing is "
                    f"bringing it back."
                ),
                f"{name} is running again.",
                now,
                service=name,
                active_state=sample.active_state,
                result=sample.result,
            )
        )
        events.extend(
            self._condition(
                oom,
                oom_key(name),
                Severity.ACTION,
                lambda: (
                    f"{name} was killed by the kernel out-of-memory killer "
                    f"(oom-kill). It ran the machine out of memory; more RAM, "
                    f"a smaller workload or a MemoryMax= is the fix, not a "
                    f"restart."
                ),
                f"{name} is running clean since the out-of-memory kill.",
                now,
                service=name,
                result=sample.result,
                memory_bytes=sample.memory_bytes,
            )
        )
        events.extend(
            self._condition(
                over_memory,
                memory_key(name),
                Severity.NOTICE,
                lambda: (
                    f"{name} is using {_bytes(sample.memory_bytes)}, over the "
                    f"{_bytes(self._memory_ceiling)} you set."
                ),
                f"{name} is back under {_bytes(self._memory_ceiling)}.",
                now,
                service=name,
                memory_bytes=sample.memory_bytes,
                ceiling_bytes=self._memory_ceiling,
            )
        )

        self._last_seen[name] = _row(sample)
        return events

    def _absent_events(self, name: str, now: float) -> List[Event]:
        """A watched unit the probe could not find.

        The unit's other keys are left exactly as they are.  "I cannot see
        it" is not evidence that it recovered, and closing a crash-loop
        request because the unit vanished would empty the badge at the
        worst possible moment.
        """
        self._last_seen.pop(name, None)
        return self._raise_key(
            absent_key(name),
            Severity.ACTION,
            f"{name} is not a unit systemd knows about. Either the name is "
            f"wrong or the unit is gone -- nothing is watching it either way.",
            now,
            service=name,
        )

    # -- the rolling window ----------------------------------------------------

    def _record_restarts(
        self, name: str, sample: ServiceSample, now: float
    ) -> int:
        """Fold this reading's ``NRestarts`` into the window.  Returns the
        number of restarts this tick saw.

        ``NRestarts`` counts up for the life of the unit, so the window is
        built from *differences*.  A counter that went **down** means the
        unit was restarted from scratch or the counter was reset
        (``systemctl reset-failed``, a ``daemon-reload``); that is taken as
        a new baseline and nothing is recorded, because inventing restarts
        out of a reset is how a watcher cries wolf.
        """
        history = self._restarts.setdefault(name, [])
        previous = self._last_seen.get(name)
        delta = 0
        if previous is not None:
            before = _as_int(previous.get("n_restarts"), 0) or 0
            if sample.n_restarts > before:
                delta = int(sample.n_restarts) - before
                history.append([float(now), float(delta)])
        self._prune(name, now)
        return delta

    def _prune(self, name: str, now: float) -> None:
        history = self._restarts.get(name)
        if not history:
            return
        edge = float(now) - self._window_s
        # An entry exactly ``window_s`` old is still inside the window; one
        # older than that is out. The edge has to be somewhere, and keeping
        # it inclusive means a bot ticking on the window boundary does not
        # lose the restart that made the count.
        self._restarts[name] = [row for row in history if row[0] >= edge]

    def _window_restarts(self, name: str, now: float) -> int:
        self._prune(name, now)
        return self._count_in_window(name, now)

    def _count_in_window(self, name: str, now: float) -> int:
        """The same count without pruning, for :meth:`status`.

        ``status`` is called on every page load and is not a tick; it
        reports what the bot knows and does not quietly edit it.
        """
        edge = float(now) - self._window_s
        return int(
            sum(row[1] for row in self._restarts.get(name, []) if row[0] >= edge)
        )

    def _crashloop_text(
        self, name: str, sample: ServiceSample, restarts: int, uptime_s: Optional[float]
    ) -> str:
        """The headline event's words.

        It names the service, the count, the window and the last exit
        result, and -- when the unit reads ``active`` -- says so, because
        that contradiction *is* the finding: the thing every other tool
        reports as fine has restarted N times this window.  A current
        uptime under ``min_uptime_s`` is quoted too: on its own it is no
        alert at all (a deploy restarts things), but next to a restart
        count it is the sentence that ends the argument.
        """
        text = (
            f"Crash loop: {name} restarted {_plural(restarts, 'time')} in the "
            f"last {_duration(self._window_s)}"
        )
        return text + self._crashloop_tail(sample, uptime_s)

    def _crashloop_tail(
        self, sample: ServiceSample, uptime_s: Optional[float]
    ) -> str:
        tail = f", last exit {_result_words(sample)}"
        if sample.active_state == "active":
            tail += (
                ". It reads 'active' right now because systemd just restarted "
                "it again -- it is not running, it is failing fast"
            )
        else:
            tail += f". It is {sample.active_state}"
            if sample.sub_state:
                tail += f" ({sample.sub_state})"
        if uptime_s is not None and uptime_s < self._min_uptime_s:
            tail += f", up {_duration(uptime_s)}"
        return tail + f". Cumulative restarts: {sample.n_restarts}."

    # -- opening and closing requests for a decision ---------------------------

    def _condition(
        self,
        holds: bool,
        key: str,
        severity: Severity,
        text: Callable[[], str],
        resolved_text: str,
        now: float,
        **data: Any,
    ) -> List[Event]:
        """One rule, as the framework sees it.

        True and not open -> raise it once.  True and already open ->
        nothing: contracts.py has the badge collapse repeats, and a bot
        that re-raised would also re-alert on every tick of a fault that
        has not changed.  False and open -> close it.
        """
        if holds:
            return self._raise_key(key, severity, text(), now, **data)
        return self._clear_key(key, resolved_text, now)

    def _raise_key(
        self, key: str, severity: Severity, text: str, now: float, **data: Any
    ) -> List[Event]:
        if key in self._open:
            return []
        self._open[key] = float(now)
        return [
            self.event(severity, text, attention_key=key, href=INFO.href, **data)
        ]

    def _clear_key(self, key: str, text: str, now: float) -> List[Event]:
        """Close one open request, in the shape the framework expects.

        The same key, below ACTION so ``Event.wants_attention`` is false,
        and ``resolved=True`` in the data -- which is what
        ``Supervisor.RESOLVED_FLAG`` reads and what ``PokeBot._resolve``
        and the scaffold template both write.  The badge empties itself.
        """
        if key not in self._open:
            return []
        self._open.pop(key, None)
        return [
            self.event(
                Severity.NOTICE,
                text,
                attention_key=key,
                href=INFO.href,
                resolved=True,
            )
        ]

    # -- what the launcher renders ---------------------------------------------

    def status(self) -> BotStatus:
        """The card (contracts.BotStatus).  Cheap, and never raises: the
        page calls it on every load.

        RUNNING, including before the first tick, because a service watch
        that has been registered *is* watching -- ``detail`` says when it
        last managed to read.  PAUSED and QUARANTINED are the supervisor's
        facts about this bot and are not invented here.
        """
        now = self.now()
        healthy = sum(1 for name in self._services if self._is_healthy(name))
        total = len(self._services)

        worst_name, worst_count = self._worst_offender(now)
        if worst_name is None:
            worst = "none"
        else:
            worst = (
                f"{worst_name} x{worst_count} in {_duration(self._window_s)}"
            )

        longest_name, longest_s = self._longest_uptime(now)
        if longest_name is None:
            longest = "none up"
        else:
            longest = f"{_duration(longest_s)} ({longest_name})"

        return BotStatus(
            state=BotState.RUNNING,
            stats=(
                self.stat("Healthy", f"{healthy} of {total}"),
                self.stat("Most restarts", worst),
                self.stat("Longest uptime", longest),
            ),
            detail=self._detail(now, healthy, total),
        )

    def _is_healthy(self, name: str) -> bool:
        row = self._last_seen.get(name)
        if row is None:
            return False
        if row.get("active_state") != "active":
            return False
        return not any(
            key in self._open
            for key in (
                crashloop_key(name),
                failed_key(name),
                stopped_key(name),
                absent_key(name),
                oom_key(name),
            )
        )

    def _worst_offender(self, now: float) -> Tuple[Optional[str], int]:
        worst_name: Optional[str] = None
        worst_count = 0
        for name in self._services:
            count = self._count_in_window(name, now)
            if count > worst_count:
                worst_name, worst_count = name, count
        return worst_name, worst_count

    def _longest_uptime(self, now: float) -> Tuple[Optional[str], float]:
        best_name: Optional[str] = None
        best_s = -1.0
        for name in self._services:
            row = self._last_seen.get(name)
            if not row or row.get("active_state") not in ACTIVE_STATES:
                continue
            entered = row.get("active_enter_timestamp")
            if entered is None:
                continue
            uptime = max(0.0, float(now) - float(entered))
            if uptime > best_s:
                best_name, best_s = name, uptime
        return best_name, max(0.0, best_s)

    def _detail(self, now: float, healthy: int, total: int) -> str:
        if self._probe_failures:
            return (
                f"Cannot read systemd: "
                f"{_plural(self._probe_failures, 'failure')} in a row"
                f"{f' ({self._last_error})' if self._last_error else ''}."
            )
        if not self._ticks:
            return f"{total} service{'' if total == 1 else 's'} watched, not read yet."
        open_actions = sum(1 for key in self._open if not key.startswith("svc:memory:"))
        age = _ago(max(0.0, now - self._last_tick_at))
        if open_actions:
            return f"{open_actions} waiting on you, read {age}."
        return f"{healthy} of {total} healthy, read {age}."

    # -- persistence -------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """JSON-able state: "State is the bot's, persistence is ours."

        The rolling window is the reason this method matters.  Restarting
        the process is exactly when a crash loop is most likely to be in
        progress -- both are symptoms of the same bad afternoon -- and a
        bot that forgot its window would forgive the loop and start
        counting again from zero.  The last seen states come with it, so
        the first tick after a restart compares against the last tick
        before it instead of treating every unit as new; and the open keys
        come with it so a request raised before the restart can still be
        *resolved* rather than sitting in the badge for ever.

        Configuration -- the watched names, the threshold, the window, the
        probe -- is deliberately absent: it is handed in at construction,
        and a snapshot that pinned it would quietly resurrect yesterday's
        config.
        """
        return {
            "version": SNAPSHOT_VERSION,
            "ticks": self._ticks,
            "last_tick_at": self._last_tick_at,
            "probe_failures": self._probe_failures,
            "last_error": self._last_error,
            "restarts": {
                name: [[float(at), float(n)] for at, n in rows]
                for name, rows in sorted(self._restarts.items())
                if rows
            },
            "last_seen": {
                name: dict(row) for name, row in sorted(self._last_seen.items())
            },
            "seen_up": sorted(name for name, up in self._seen_up.items() if up),
            "oom": sorted(name for name, hit in self._oom.items() if hit),
            "open": [
                {"key": key, "since": since}
                for key, since in sorted(self._open.items())
            ],
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Take the state back (contracts.Bot.restore).

        Tolerant field by field: a snapshot comes off disk, may have been
        written by an older build, and a bot that refuses to start is
        worse than a bot that has forgotten a restart.  A snapshot from a
        *newer* build is refused outright, because a half-read window is a
        crash loop nobody is counting.

        Rows for services no longer watched are dropped rather than kept:
        they would never be evaluated again, and an open key for a service
        nobody watches can never be resolved.
        """
        if not isinstance(snapshot, Mapping) or not snapshot:
            return
        version = snapshot.get("version", SNAPSHOT_VERSION)
        if isinstance(version, int) and version > SNAPSHOT_VERSION:
            raise ServiceBotError(
                f"service bot snapshot version {version!r} is newer than this "
                f"build understands (version {SNAPSHOT_VERSION})"
            )

        watched = set(self._services)

        self._ticks = _as_int(snapshot.get("ticks"), self._ticks) or 0
        self._last_tick_at = _as_float(snapshot.get("last_tick_at"), self._last_tick_at)
        self._probe_failures = max(
            0, _as_int(snapshot.get("probe_failures"), self._probe_failures) or 0
        )
        last_error = snapshot.get("last_error")
        self._last_error = last_error if isinstance(last_error, str) else ""

        restarts: Dict[str, List[List[float]]] = {}
        raw_restarts = snapshot.get("restarts")
        if isinstance(raw_restarts, Mapping):
            for name, rows in raw_restarts.items():
                if name not in watched or not isinstance(rows, (list, tuple)):
                    continue
                kept: List[List[float]] = []
                for row in rows:
                    if not isinstance(row, (list, tuple)) or len(row) != 2:
                        continue
                    at = _as_float(row[0], None)
                    count = _as_float(row[1], None)
                    if at is None or count is None or count <= 0:
                        continue
                    kept.append([at, count])
                if kept:
                    restarts[str(name)] = kept
        self._restarts = restarts

        last_seen: Dict[str, Dict[str, Any]] = {}
        raw_last = snapshot.get("last_seen")
        if isinstance(raw_last, Mapping):
            for name, row in raw_last.items():
                if name in watched and isinstance(row, Mapping):
                    last_seen[str(name)] = dict(row)
        self._last_seen = last_seen

        self._seen_up = {
            str(name): True
            for name in _as_str_list(snapshot.get("seen_up"))
            if name in watched
        }
        self._oom = {
            str(name): True
            for name in _as_str_list(snapshot.get("oom"))
            if name in watched
        }

        open_keys: Dict[str, float] = {}
        for row in snapshot.get("open") or []:
            if not isinstance(row, Mapping):
                continue
            key = row.get("key")
            if not isinstance(key, str) or not key.strip():
                continue
            if key != PROBE_ATTENTION_KEY and not any(
                key.endswith(f":{name}") for name in watched
            ):
                continue
            open_keys[key] = _as_float(row.get("since"), 0.0) or 0.0
        self._open = open_keys

    def on_pause(self) -> None:
        """contracts.py: "Paused means paused ... a badge asking you to act
        on something you switched off is a lie."

        The supervisor clears this bot's attention items on pause, so the
        bot forgets its own record of them too.  Keeping it would leave the
        bot believing those requests were still open and never re-raising
        them on resume -- an empty badge over a service that is still
        looping.  The restart window is *not* cleared: it is evidence about
        the machine, not about the badge.
        """
        self._open.clear()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _row(sample: ServiceSample) -> Dict[str, Any]:
    """The JSON-able part of a sample that the next tick needs."""
    return {
        "active_state": sample.active_state,
        "sub_state": sample.sub_state,
        "n_restarts": int(sample.n_restarts),
        "active_enter_timestamp": sample.active_enter_timestamp,
        "memory_bytes": sample.memory_bytes,
        "exit_code": sample.exit_code,
        "result": sample.result,
    }


def _result_words(sample: ServiceSample) -> str:
    if sample.result == "oom-kill":
        return "out-of-memory kill"
    if sample.result and sample.result != "success":
        if sample.exit_code is not None:
            return f"{sample.result} (status {sample.exit_code})"
        return sample.result
    if sample.exit_code:
        return f"status {sample.exit_code}"
    return "success"


def _result_phrase(sample: ServiceSample) -> str:
    words = _result_words(sample)
    return "" if words == "success" else f", {words}"


def _plural(n: Any, word: str) -> str:
    try:
        count = int(n)
    except (TypeError, ValueError):
        count = 0
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _duration(seconds: Any) -> str:
    """A span as the card and the alerts spell it.  Coarse on purpose."""
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        return "?"
    s = max(0.0, s)
    if s < 90:
        return f"{int(s)}s"
    if s < 5400:
        # "min" rather than "mins": the unit is an abbreviation, and
        # "10 min" is how a duration reads on a card.
        return f"{round(s / 60)} min"
    if s < 172800:
        return _plural(round(s / 3600), "hour")
    return _plural(round(s / 86400), "day")


def _ago(seconds: float) -> str:
    return "just now" if seconds < 60 else f"{_duration(seconds)} ago"


def _bytes(value: Any) -> str:
    """Bytes as a person reads them.  Binary units, which is what systemd
    and every memory tool on the machine use."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return "?"
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GiB"  # pragma: no cover - unreachable, kept honest


def _as_int(value: Any, fallback: Optional[int]) -> Optional[int]:
    if isinstance(value, bool):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: Any, fallback: Optional[float]) -> Optional[float]:
    if isinstance(value, bool):
        return fallback
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _as_positive_int(value: Any, allow_zero: bool = False) -> Optional[int]:
    """An int from systemd, or ``None`` for its several spellings of unset.

    ``MainPID=0`` means "no main process" and ``MemoryCurrent=[not set]``
    (or the 64-bit sentinel) means "no cgroup accounting".  Both are
    ``None`` here: a zero that the caller cannot tell apart from "I could
    not look" is how a watcher reports nonsense with confidence.
    """
    if isinstance(value, str) and value.strip().lower() in _UNSET_TOKENS:
        return None
    number = _as_int(value, None)
    if number is None:
        return None
    if number < 0:
        return None
    if number == 0 and not allow_zero:
        return None
    return number


def _as_unix(value: Any) -> Optional[float]:
    """``ActiveEnterTimestamp`` as unix seconds, or ``None``.

    ``--timestamp=unix`` prints ``@1700000000``.  A bare number is taken as
    seconds too.  Anything else -- a localised date from a systemd that
    does not know the flag, or the ``0`` a never-started unit reports -- is
    ``None``, because a guessed start time makes an uptime that is worse
    than no uptime.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in _UNSET_TOKENS:
        return None
    if text.startswith("@"):
        text = text[1:]
    seconds = _as_float(text, None)
    if seconds is None or seconds <= 0:
        return None
    return seconds


def _as_str_list(value: Any) -> List[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [item for item in value if isinstance(item, str) and item.strip()]


def build(
    services: Sequence[str],
    *,
    clock: Clock,
    probe: Optional[Callable[[], Sequence[ServiceSample]]] = None,
    restart_threshold: int = DEFAULT_RESTART_THRESHOLD,
    window_s: float = DEFAULT_WINDOW_S,
    memory_ceiling_bytes: Optional[int] = None,
    min_uptime_s: float = DEFAULT_MIN_UPTIME_S,
) -> "ServiceBot":
    """Construct the bot with everything injected.

    ``probe`` defaults to :func:`systemctl_probe` bound to ``services`` --
    the *composition root's* default, not the bot's: the bot still holds a
    plain callable and still cannot reach a process on its own.  A config
    can register ``{BOT_ID: lambda: build(names, clock=clock)}`` without
    knowing the class, and a test passes its own probe.
    """
    if probe is None:
        names = tuple(str(n) for n in services)

        def probe_systemctl() -> List[ServiceSample]:
            return systemctl_probe(names)

        probe = probe_systemctl
    return ServiceBot(
        services,
        clock=clock,
        probe=probe,
        restart_threshold=restart_threshold,
        window_s=window_s,
        memory_ceiling_bytes=memory_ceiling_bytes,
        min_uptime_s=min_uptime_s,
    )
