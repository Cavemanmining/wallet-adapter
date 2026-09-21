"""GPU watch -- a Jarvis bot that notices when a card stops being there.

Why this exists
---------------
A GPU vanished from the bus after a reboot on the owner's machine and
nobody noticed for days, while a service crash-looped against it roughly
fifteen thousand times.  Every other check in this module is a bonus; the
one that earns it its place on the launcher is :meth:`GpuBot._missing`,
which is why that is the loudest thing it does and why the fleet it
expects is remembered in the snapshot rather than re-derived from whatever
happens to answer today.  A watcher that re-derives "what should be here"
from "what is here" can never report an absence.

What it watches, and how loud each thing is
-------------------------------------------
``ACTION`` -- the badge, and a push:

* a card in the remembered fleet that is **missing**;
* a card **at or over its temperature limit** (the card's own slowdown
  temperature when the probe supplies one, otherwise
  :data:`DEFAULT_TEMP_ACTION_C`);
* a **stuck fan**: 0% while the card is working and hot, which is how a
  card dies;
* **ECC errors increasing** since the previous tick;
* **three probe failures in a row** -- one hiccup from ``nvidia-smi`` is
  not news, a driver that has stopped answering is.

``NOTICE`` -- a line in the feed, no badge:

* a **new** card.  Hardware being added is normal; it joins the fleet.
* a card **warm** but under its limit;
* **memory over 95%** for three consecutive ticks.  A full GPU is usually
  someone working, so this is never an ACTION;
* a **PCIe link below the best this bot has ever seen for that card**;
* a **thermal or power throttle** standing for two consecutive ticks.

Everything is *edge triggered*: a condition raises its key once when it
starts and reports it resolved once when it ends.  ``contracts.py`` has
the badge count distinct open keys "so one restock nagging across ten
ticks is one item of attention, not ten", and a bot that re-raises the
same fact every five minutes teaches the owner to ignore it.

PCIe, and the card that is *meant* to be slow
---------------------------------------------
The degradation check compares the current link against the best link
this bot has **ever seen for that uuid**, never against
``pcie_gen_max``.  This is not a stylistic preference: the owner's CMP
170HX is legitimately a Gen 2 x4 card whose Gen 3 support is fused off,
so it reports ``pcie_gen_current=2, pcie_gen_max=3`` for ever and a check
against the maximum would alert on it on every tick of its life, for ever,
about nothing.  A check against "the best this card has actually managed"
is silent for it and still catches the real failure, which is a card that
used to train at Gen 3 x16 and now trains at Gen 1 x4 because a riser
came loose.

Injection
---------
The probe is injected and the clock is injected.  :class:`GpuBot` never
shells out, never imports a GPU library, and never calls ``time.time()``:
it is handed a ``probe() -> list[GpuSample]`` and calls it.
:func:`nvidia_smi_probe` is the real one the app injects, and it lives
here so it can be tested against captured text -- but nothing in the bot
calls it, so the bot's whole behaviour is reachable from a stub.

A probe that raises never reaches the supervisor.  ``contracts.py`` allows
a tick to raise and handles it, but four raises in a row quarantine a bot
for half an hour, and ``nvidia-smi`` returning nothing once while the
driver reloads is ordinary.  It comes back as an event instead, and only a
*run* of failures is treated as a fault worth the owner's attention.

Wiring it up
------------
::

    from jarvis_bots.bots.gpu_bot import build, nvidia_smi_probe
    from jarvis_bots.registry import BotRegistry
    from jarvis_bots.supervisor import Supervisor

    registry = BotRegistry([build(clock=time.time, probe=nvidia_smi_probe)])
    supervisor = Supervisor(registry, time.time)
    supervisor.run_round()
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from jarvis_bots.base import BaseBot, Clock
from jarvis_bots.contracts import BotInfo, BotState, BotStatus, Event, Severity

__all__ = [
    "BOT_ID",
    "INFO",
    "SNAPSHOT_VERSION",
    "DEFAULT_TEMP_ACTION_C",
    "DEFAULT_TEMP_NOTICE_C",
    "FAN_STUCK_UTILIZATION_PCT",
    "FAN_STUCK_TEMPERATURE_C",
    "MEMORY_PRESSURE_FRACTION",
    "MEMORY_PRESSURE_TICKS",
    "THROTTLE_TICKS",
    "PROBE_FAILURES_FOR_ACTION",
    "NVIDIA_SMI_QUERY_FIELDS",
    "THROTTLE_REASON_BITS",
    "GpuSample",
    "GpuBot",
    "GpuBotError",
    "GpuProbeError",
    "build",
    "nvidia_smi_command",
    "nvidia_smi_probe",
    "parse_nvidia_smi_csv",
    "missing_key",
    "temperature_key",
    "fan_key",
    "ecc_key",
    "PROBE_KEY",
]


#: The id the supervisor, the launcher and every event carry.
BOT_ID = "gpu"

#: Static identity, declared once.  ``radar`` picks the launcher's radar
#: glyph (``jarvis_bots/web/README.md``); five minutes is frequent enough
#: that a card that drops off the bus is news the same hour, and rare
#: enough that running ``nvidia-smi`` costs nothing measurable.
INFO = BotInfo(
    id=BOT_ID,
    name="GPU watch",
    blurb=(
        "Watches the GPUs and says when one vanishes off the bus, cooks, "
        "stops spinning its fan, or starts collecting ECC errors."
    ),
    kind="radar",
    interval_s=300.0,
    href="/bots/gpu",
    can_pause=True,
)

#: Bumped when :meth:`GpuBot.snapshot` changes shape.
SNAPSHOT_VERSION = 1

#: At or above this, a card is an ACTION.  Used only when the probe does
#: not supply the card's own slowdown temperature, which is always the
#: better number because it is the one the card itself acts on.
DEFAULT_TEMP_ACTION_C = 85.0

#: At or above this and below the action threshold, a card is a NOTICE.
DEFAULT_TEMP_NOTICE_C = 80.0

#: A fan reading 0% is only a fault when the card is actually working and
#: actually hot: an idle card at 40C with its fan stopped is a card doing
#: what it is told.  Both are strict inequalities.
FAN_STUCK_UTILIZATION_PCT = 30.0
FAN_STUCK_TEMPERATURE_C = 60.0

#: Memory "full".  Strictly above, over this many consecutive ticks, and
#: even then only a NOTICE: a full GPU is usually someone working.
MEMORY_PRESSURE_FRACTION = 0.95
MEMORY_PRESSURE_TICKS = 3

#: A thermal or power throttle has to stand for this many consecutive
#: ticks before it is worth a line.  One tick of ``sw_power_cap`` during a
#: benchmark is the power limit doing its job.
THROTTLE_TICKS = 2

#: Probe failures in a row before the owner is asked to look.  Below this
#: each failure is an ERROR line in the feed and nothing more.
PROBE_FAILURES_FOR_ACTION = 3

#: Substrings that make a throttle reason thermal-or-power.  ``hw_slowdown``
#: is deliberately absent: it is set for both thermal and power events *and*
#: for an external power-brake signal, so on its own it does not say which
#: and is not worth waking anyone over.
_THROTTLE_WORDS = ("thermal", "power")


class GpuBotError(ValueError):
    """A wiring mistake: a probe that is not callable, a threshold that is
    not a pair of numbers.

    Raised from ``__init__`` and from :meth:`GpuBot.restore` only.  Errors
    from a *tick* are never this; they are turned into events.
    """


class GpuProbeError(RuntimeError):
    """The probe could not read the GPUs.

    Raised by :func:`nvidia_smi_probe` and by :func:`parse_nvidia_smi_csv`.
    A probe raising is expected and handled: see the module docstring.
    """


# --------------------------------------------------------------------------
# the sample
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuSample:
    """One card, as one probe saw it once.

    Frozen, because a sample is a reading and a reading does not change
    after the fact; every field but the identity three is ``Optional``
    because ``nvidia-smi`` answers ``[N/A]`` for anything the card does
    not expose -- a passively cooled Tesla P40 has no fan to report, a
    mining card may report no power limit, and a consumer card reports no
    ECC counters at all.  ``None`` means "did not say", which is never the
    same as zero and is never a fault.

    ``temperature_slowdown_c`` is the card's own slowdown point.  Standard
    ``--query-gpu`` has no field for it, so :func:`nvidia_smi_probe`
    leaves it ``None`` and the bot falls back to its configured threshold;
    a probe that reads ``nvidia-smi -q`` can fill it in and the bot will
    prefer it, because the temperature a card starts throttling itself at
    is a better answer than any number in this file.
    """

    index: int
    uuid: str
    name: str
    memory_total_mb: Optional[float] = None
    memory_used_mb: Optional[float] = None
    temperature_c: Optional[float] = None
    fan_percent: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    pcie_gen_current: Optional[int] = None
    pcie_gen_max: Optional[int] = None
    pcie_width_current: Optional[int] = None
    pcie_width_max: Optional[int] = None
    ecc_errors: Optional[int] = None
    utilization_pct: Optional[float] = None
    throttle_reasons: Tuple[str, ...] = ()
    temperature_slowdown_c: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.uuid, str) or not self.uuid.strip():
            raise GpuProbeError(
                f"a GPU sample needs a uuid: it is the only stable name a "
                f"card has across reboots, and the whole fleet is keyed on "
                f"it; got {self.uuid!r}"
            )
        object.__setattr__(self, "uuid", self.uuid.strip())
        object.__setattr__(self, "name", str(self.name).strip())
        object.__setattr__(self, "index", int(self.index))
        object.__setattr__(self, "throttle_reasons", tuple(self.throttle_reasons or ()))

    @property
    def label(self) -> str:
        """How the card is named to a person: ``Tesla P40 #1``."""
        return f"{self.name or 'GPU'} #{self.index}"

    @property
    def memory_fraction(self) -> Optional[float]:
        """Used over total, or ``None`` when either was not reported."""
        total, used = self.memory_total_mb, self.memory_used_mb
        if total is None or used is None or total <= 0:
            return None
        return float(used) / float(total)

    @property
    def link(self) -> Optional[Tuple[int, int]]:
        """``(gen, width)`` of the current link, or ``None``."""
        if self.pcie_gen_current is None or self.pcie_width_current is None:
            return None
        return (int(self.pcie_gen_current), int(self.pcie_width_current))

    def throttling_hot_or_capped(self) -> Tuple[str, ...]:
        """The active throttle reasons that are thermal or power."""
        return tuple(
            reason
            for reason in self.throttle_reasons
            if any(word in str(reason).lower() for word in _THROTTLE_WORDS)
        )


# --------------------------------------------------------------------------
# attention keys
# --------------------------------------------------------------------------

#: One key per card per condition, so two cards overheating are two items
#: in the badge and one card overheating across a night is one.  Built by
#: functions rather than f-strings at the call site, so the tick that
#: raises a key and the tick that resolves it cannot drift apart.


def missing_key(uuid: str) -> str:
    """The key the incident this bot was written for raises."""
    return f"gpu:missing:{uuid}"


def temperature_key(uuid: str) -> str:
    return f"gpu:temp:{uuid}"


def fan_key(uuid: str) -> str:
    return f"gpu:fan:{uuid}"


def ecc_key(uuid: str) -> str:
    return f"gpu:ecc:{uuid}"


#: Not per card: when the probe cannot run there is no card to blame.
PROBE_KEY = "gpu:probe"


# --------------------------------------------------------------------------
# the real probe: built here, injected by the app, never called by the bot
# --------------------------------------------------------------------------

#: The ``--query-gpu`` fields, in the order :func:`parse_nvidia_smi_csv`
#: reads them.  Every one of these is a real ``nvidia-smi`` field name;
#: the list and the parser must be changed together, which is why they sit
#: next to each other.
NVIDIA_SMI_QUERY_FIELDS: Tuple[str, ...] = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "temperature.gpu",
    "fan.speed",
    "power.draw",
    "power.limit",
    "pcie.link.gen.current",
    "pcie.link.gen.max",
    "pcie.link.width.current",
    "pcie.link.width.max",
    "ecc.errors.uncorrected.volatile.total",
    "utilization.gpu",
    "clocks_throttle_reasons.active",
)

#: NVML's ``clocksThrottleReasons`` bitmask, which is what
#: ``clocks_throttle_reasons.active`` returns -- a hex string such as
#: ``0x0000000000000004``.  Decoded to names here so that the rest of the
#: module can match on words and a test can read the expected output.
THROTTLE_REASON_BITS: Tuple[Tuple[int, str], ...] = (
    (0x0000000000000001, "gpu_idle"),
    (0x0000000000000002, "applications_clocks_setting"),
    (0x0000000000000004, "sw_power_cap"),
    (0x0000000000000008, "hw_slowdown"),
    (0x0000000000000010, "sync_boost"),
    (0x0000000000000020, "sw_thermal_slowdown"),
    (0x0000000000000040, "hw_thermal_slowdown"),
    (0x0000000000000080, "hw_power_brake_slowdown"),
    (0x0000000000000100, "display_clock_setting"),
)

#: Everything ``nvidia-smi`` says instead of a number when it has nothing
#: to say.  All of them mean the same thing to this module: ``None``.
_NOT_AVAILABLE = frozenset(
    {
        "",
        "n/a",
        "[n/a]",
        "not available",
        "[not available]",
        "not supported",
        "[not supported]",
        "unknown error",
        "[unknown error]",
        "insufficient permissions",
        "[insufficient permissions]",
        "error",
        "[error]",
    }
)

#: Units the parser tolerates on the end of a value.  ``--format`` says
#: ``nounits``, so none of these should appear; stripping them anyway
#: costs one line and means a caller who forgot ``nounits`` gets numbers
#: instead of a parse error.
_UNIT_SUFFIXES = ("mib", "mb", "gib", "gb", "w", "%", "c")


def nvidia_smi_command(binary: str = "nvidia-smi") -> Tuple[str, ...]:
    """The exact argv to run.  Separated from running it so a test can
    assert the command without a GPU, and so an app that runs it over ssh
    or in a container can borrow the argument list.

    ``noheader`` and ``nounits`` are not optional: the parser reads by
    position and expects bare numbers.
    """
    return (
        str(binary),
        "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY_FIELDS),
        "--format=csv,noheader,nounits",
    )


def _clean(cell: str) -> str:
    return " ".join(str(cell).split())


def _text(cell: str) -> Optional[str]:
    """A cell as text, or ``None`` for any of ``nvidia-smi``'s ways of
    saying it has no answer."""
    cleaned = _clean(cell)
    return None if cleaned.lower() in _NOT_AVAILABLE else cleaned


def _number(cell: str, field: str) -> Optional[float]:
    """A cell as a float, or ``None``.  Raises for a cell that is neither."""
    cleaned = _text(cell)
    if cleaned is None:
        return None
    lowered = cleaned.lower()
    for suffix in _UNIT_SUFFIXES:
        if lowered.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
            break
    try:
        return float(cleaned)
    except ValueError:
        raise GpuProbeError(
            f"nvidia-smi field {field!r} is not a number: {_clean(cell)!r}"
        ) from None


def _integer(cell: str, field: str) -> Optional[int]:
    value = _number(cell, field)
    return None if value is None else int(value)


def decode_throttle_reasons(cell: str) -> Tuple[str, ...]:
    """``0x0000000000000024`` -> ``('sw_power_cap', 'sw_thermal_slowdown')``.

    An unreadable mask is not a parse failure: the throttle reasons drive
    a NOTICE two ticks later, and refusing the whole probe -- which is how
    the bot learns a card is *missing* -- over a field this soft would be
    the wrong trade.  ``gpu_idle`` is kept rather than filtered, because
    the bot matches on words and a caller reading ``throttle_reasons``
    should see what the card said.
    """
    cleaned = _text(cell)
    if cleaned is None:
        return ()
    try:
        mask = int(cleaned, 16) if cleaned.lower().startswith("0x") else int(cleaned)
    except ValueError:
        # Some drivers spell the reasons out instead of masking them.
        words = tuple(
            part.strip().lower().replace(" ", "_")
            for part in cleaned.replace(";", ",").split(",")
            if part.strip()
        )
        return words
    if mask <= 0:
        return ()
    return tuple(name for bit, name in THROTTLE_REASON_BITS if mask & bit)


def parse_nvidia_smi_csv(text: str) -> List[GpuSample]:
    """Parse ``--format=csv,noheader,nounits`` output into samples.

    Deliberately strict about shape and lax about content.  A row with the
    wrong number of columns, or with no uuid, raises: those mean the query
    and this parser have drifted apart, and a probe that quietly returns
    three cards when four were asked for would be read by the bot as a
    card that has *vanished* -- exactly the false alarm that would get
    this bot switched off.  A row full of ``[N/A]`` readings is fine and
    normal, and becomes a sample full of ``None``.

    Blank lines are skipped, and so is a header row if someone runs the
    command without ``noheader``.
    """
    if not isinstance(text, str):
        raise GpuProbeError(f"nvidia-smi output must be text; got {type(text).__name__}")
    samples: List[GpuSample] = []
    width = len(NVIDIA_SMI_QUERY_FIELDS)
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        cells = [cell.strip() for cell in line.split(",")]
        if cells[0].lower() in {"index", "#index"}:
            continue  # a header the caller did not ask to be suppressed
        if len(cells) != width:
            raise GpuProbeError(
                f"nvidia-smi line {line_no} has {len(cells)} fields, expected "
                f"{width} ({', '.join(NVIDIA_SMI_QUERY_FIELDS)})"
            )
        index = _integer(cells[0], "index")
        uuid = _text(cells[1])
        name = _text(cells[2])
        if index is None or not uuid:
            raise GpuProbeError(
                f"nvidia-smi line {line_no} names no card (index={cells[0]!r}, "
                f"uuid={cells[1]!r}); a sample without a uuid cannot be told "
                f"apart from a card that is gone"
            )
        samples.append(
            GpuSample(
                index=index,
                uuid=uuid,
                name=name or "GPU",
                memory_total_mb=_number(cells[3], "memory.total"),
                memory_used_mb=_number(cells[4], "memory.used"),
                temperature_c=_number(cells[5], "temperature.gpu"),
                fan_percent=_number(cells[6], "fan.speed"),
                power_w=_number(cells[7], "power.draw"),
                power_limit_w=_number(cells[8], "power.limit"),
                pcie_gen_current=_integer(cells[9], "pcie.link.gen.current"),
                pcie_gen_max=_integer(cells[10], "pcie.link.gen.max"),
                pcie_width_current=_integer(cells[11], "pcie.link.width.current"),
                pcie_width_max=_integer(cells[12], "pcie.link.width.max"),
                ecc_errors=_integer(cells[13], "ecc.errors"),
                utilization_pct=_number(cells[14], "utilization.gpu"),
                throttle_reasons=decode_throttle_reasons(cells[15]),
            )
        )
    return samples


def _run_nvidia_smi(argv: Sequence[str], timeout_s: float) -> str:
    """The one place in this package that starts a process."""
    try:
        completed = subprocess.run(
            list(argv),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except FileNotFoundError:
        raise GpuProbeError(f"{argv[0]} is not installed") from None
    except subprocess.TimeoutExpired:
        raise GpuProbeError(f"{argv[0]} did not answer within {timeout_s:g}s") from None
    except OSError as exc:
        raise GpuProbeError(f"{argv[0]} could not be run: {type(exc).__name__}") from None
    if completed.returncode != 0:
        first = _clean((completed.stderr or "").splitlines()[0]) if completed.stderr else ""
        raise GpuProbeError(
            f"{argv[0]} exited {completed.returncode}" + (f": {first[:120]}" if first else "")
        )
    return completed.stdout or ""


def nvidia_smi_probe(
    *,
    binary: str = "nvidia-smi",
    timeout_s: float = 20.0,
    runner: Optional[Callable[[Sequence[str], float], str]] = None,
) -> List[GpuSample]:
    """Read the real GPUs.  This is what the app injects into
    :class:`GpuBot`; **the bot never calls it itself**.

    Keeping it in this module and out of the bot is what makes the bot
    testable from a stub and this function testable from captured text:
    :func:`parse_nvidia_smi_csv` does all the work that can be got wrong,
    and needs no GPU, no driver and no subprocess to exercise.

    ``runner`` is the seam for the process itself, so a test can drive the
    whole function -- command building included -- without running
    anything.  It is handed the argv and the timeout and returns stdout.

    Raises :class:`GpuProbeError` when the GPUs cannot be read.  That is
    the contract the bot is built around: a probe says "I could not look",
    it does not report an empty fleet.
    """
    argv = nvidia_smi_command(binary)
    run = runner if runner is not None else _run_nvidia_smi
    return parse_nvidia_smi_csv(run(argv, float(timeout_s)))


# --------------------------------------------------------------------------
# formatting helpers for the card
# --------------------------------------------------------------------------


def _c(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.0f}C"


def _gib(mb: Optional[float]) -> str:
    return "n/a" if mb is None else f"{float(mb) / 1024.0:.1f} GiB"


def _link_text(link: Optional[Tuple[int, int]]) -> str:
    return "unknown" if link is None else f"Gen {link[0]} x{link[1]}"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _as_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _as_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


# --------------------------------------------------------------------------
# the bot
# --------------------------------------------------------------------------


class GpuBot(BaseBot):
    """Watches a fleet of GPUs it remembers.

    Implements ``contracts.Bot``: ``info``, :meth:`tick` and
    :meth:`status`, plus :meth:`snapshot` / :meth:`restore` because this
    bot's whole point is state that outlives a restart -- a fleet it has
    forgotten is a fleet it cannot miss a member of.
    """

    info = INFO

    def __init__(
        self,
        *,
        clock: Clock,
        probe: Callable[[], Sequence[GpuSample]],
        temp_action_c: float = DEFAULT_TEMP_ACTION_C,
        temp_notice_c: float = DEFAULT_TEMP_NOTICE_C,
        info: Optional[BotInfo] = None,
    ) -> None:
        super().__init__(clock, info)
        if not callable(probe):
            raise GpuBotError(
                f"GpuBot needs an injected probe: a callable returning a list "
                f"of GpuSample (this module never shells out itself); got "
                f"{probe!r}"
            )
        action = float(temp_action_c)
        notice = float(temp_notice_c)
        if not notice < action:
            raise GpuBotError(
                f"the temperature notice threshold must be below the action "
                f"threshold; got notice={notice:g} action={action:g}"
            )
        self._probe = probe
        self._temp_action_c = action
        self._temp_notice_c = notice
        #: How far under the action threshold the notice band reaches.  Kept
        #: as a width so that a card supplying its own slowdown temperature
        #: gets a warning band of the same size around *its* number.
        self._temp_band_c = action - notice

        # --- persisted state -------------------------------------------------
        #: uuid -> {"name", "index"}: the fleet this bot expects to see.
        self._baseline: Dict[str, Dict[str, Any]] = {}
        #: False until the first successful probe, so that an empty first
        #: tick is a baseline of nothing rather than a fleet that vanished.
        self._have_baseline = False
        #: uuid -> (gen, width): the best link ever seen.  See the module
        #: docstring: never pcie_gen_max, or the 170HX alerts for ever.
        self._best_link: Dict[str, Tuple[int, int]] = {}
        #: uuid -> ECC count at the previous tick.
        self._ecc: Dict[str, int] = {}
        #: uuid -> consecutive ticks over the memory/throttle thresholds.
        self._mem_ticks: Dict[str, int] = {}
        self._throttle_ticks: Dict[str, int] = {}
        #: uuid -> "ok" | "notice" | "action": which temperature band the
        #: card was in last tick, so a band is announced on entry only.
        self._temp_band: Dict[str, str] = {}
        #: Consecutive probe failures.
        self._probe_failures = 0
        #: attention_key -> when it was raised.  The bot's own record of
        #: what it believes is open; the supervisor owns the badge.
        self._open: Dict[str, float] = {}
        self._ticks = 0
        self._last_tick_at = 0.0

        # --- not persisted: the last reading, for the card -------------------
        self._last: Tuple[GpuSample, ...] = ()
        self._last_error = ""

    # -- the work ------------------------------------------------------------

    def tick(self, now: float) -> Sequence[Event]:
        """One look at the fleet.

        Never raises.  ``contracts.py`` permits it and the supervisor
        handles it, but a quarantined GPU watch is a GPU watch that cannot
        tell you a card is gone, which is the one job it has.  Both failure
        paths return events instead: a probe that raised, and (belt and
        braces) a bug in this bot's own arithmetic.
        """
        at = float(now)
        try:
            samples = _checked_samples(self._probe())
        except Exception as exc:  # noqa: BLE001 - deliberate; see the docstring
            return self._probe_failed(at, exc)
        try:
            return self._evaluate(samples, at)
        except Exception as exc:  # noqa: BLE001 - a bug here must not quarantine
            self._last_error = type(exc).__name__
            return (
                self.event(
                    Severity.ERROR,
                    f"GPU watch could not make sense of its own reading "
                    f"({type(exc).__name__}).",
                    href=INFO.href,
                ),
            )

    def _evaluate(self, samples: Tuple[GpuSample, ...], now: float) -> Tuple[Event, ...]:
        self._ticks += 1
        self._last_tick_at = now
        self._last = samples
        self._last_error = ""
        events: List[Event] = []

        # A probe that answers again closes the probe fault first: the feed
        # should read "it is back" before anything it then found.
        events.extend(self._probe_recovered(now))

        seen = {sample.uuid: sample for sample in samples}

        if not self._have_baseline:
            # First tick ever: whatever is here *is* the fleet.  No events --
            # a bot cannot report a change on the first thing it ever saw.
            self._have_baseline = True
            for sample in samples:
                self._remember(sample)
                self._record_link(sample)
                self._ecc[sample.uuid] = int(sample.ecc_errors or 0)
        else:
            events.extend(self._missing(seen, now))
            events.extend(self._returned(seen, now))
            events.extend(self._new_cards(samples, now))

        for sample in sorted(samples, key=lambda s: (s.index, s.uuid)):
            self._remember(sample)
            events.extend(self._temperature(sample, now))
            events.extend(self._fan(sample, now))
            events.extend(self._ecc_errors(sample, now))
            events.extend(self._memory(sample, now))
            events.extend(self._pcie(sample, now))
            events.extend(self._throttle(sample, now))

        # A card that is gone keeps its counters frozen rather than having
        # them decay: when it comes back the question is whether it is still
        # collecting ECC errors, and the answer is the count from before it
        # left, not zero.
        return tuple(events)

    # -- the incident this bot exists for -------------------------------------

    def _missing(self, seen: Mapping[str, GpuSample], now: float) -> List[Event]:
        """A remembered card that is not in the probe's answer.

        The loudest thing this bot does.  The card is named by the name and
        index it had *when it was last seen*, because it is not here to name
        itself, and "GPU 2 is gone" is not something the owner can act on
        without knowing which card that was.
        """
        events: List[Event] = []
        for uuid in sorted(
            self._baseline, key=lambda u: (self._baseline[u].get("index", 0), u)
        ):
            if uuid in seen:
                continue
            key = missing_key(uuid)
            if key in self._open:
                continue  # already asked; one standing request, not one a tick
            remembered = self._baseline[uuid]
            label = f"{remembered.get('name') or 'GPU'} #{remembered.get('index')}"
            self._open[key] = now
            events.append(
                self.event(
                    Severity.ACTION,
                    f"GPU gone: {label} is no longer on the bus. It was here "
                    f"and nvidia-smi does not list it any more -- check the "
                    f"riser, the power and dmesg before anything else starts "
                    f"failing against it.",
                    attention_key=key,
                    href=INFO.href,
                    uuid=uuid,
                    gpu_name=remembered.get("name"),
                    gpu_index=remembered.get("index"),
                )
            )
        return events

    def _returned(self, seen: Mapping[str, GpuSample], now: float) -> List[Event]:
        """A card that was missing and is back: resolve its key.

        The framework's own close signal, exactly as ``poke_bot._resolve``
        and ``templates/bot.py.tmpl`` write it: the same key, below ACTION
        so ``Event.wants_attention`` is false and the request is not
        re-opened, and ``resolved=True`` in the data
        (``supervisor.RESOLVED_FLAG``).
        """
        events: List[Event] = []
        for key in sorted(self._open):
            if not key.startswith("gpu:missing:"):
                continue
            uuid = key[len("gpu:missing:") :]
            sample = seen.get(uuid)
            if sample is None:
                continue
            del self._open[key]
            events.append(
                self.event(
                    Severity.NOTICE,
                    f"GPU back: {sample.label} is on the bus again.",
                    attention_key=key,
                    href=INFO.href,
                    uuid=uuid,
                    resolved=True,
                )
            )
        return events

    def _new_cards(self, samples: Sequence[GpuSample], now: float) -> List[Event]:
        """A uuid nobody has seen before.  Hardware being added is normal,
        so this is a NOTICE and the card simply joins the fleet."""
        events: List[Event] = []
        for sample in sorted(samples, key=lambda s: (s.index, s.uuid)):
            if sample.uuid in self._baseline:
                continue
            events.append(
                self.event(
                    Severity.NOTICE,
                    f"New GPU: {sample.label} joined the fleet.",
                    href=INFO.href,
                    uuid=sample.uuid,
                    gpu_name=sample.name,
                    gpu_index=sample.index,
                )
            )
            self._remember(sample)
            self._record_link(sample)
            self._ecc[sample.uuid] = int(sample.ecc_errors or 0)
        return events

    # -- heat ------------------------------------------------------------------

    def _temperature_limits(self, sample: GpuSample) -> Tuple[float, float]:
        """``(action_at, notice_at)`` for this card.

        The card's own slowdown temperature wins when the probe supplies
        one: a P40 slows itself at a different point than a 5070, and a
        single number in this file is only ever a guess at both.
        """
        slowdown = sample.temperature_slowdown_c
        if slowdown is not None and float(slowdown) > 0.0:
            action_at = float(slowdown)
            return action_at, action_at - self._temp_band_c
        return self._temp_action_c, self._temp_notice_c

    def _temperature(self, sample: GpuSample, now: float) -> List[Event]:
        temp = sample.temperature_c
        if temp is None:
            return []
        action_at, notice_at = self._temperature_limits(sample)
        was = self._temp_band.get(sample.uuid, "ok")
        key = temperature_key(sample.uuid)
        events: List[Event] = []

        if temp >= action_at:
            band = "action"
            if was != "action":
                self._open[key] = now
                events.append(
                    self.event(
                        Severity.ACTION,
                        f"GPU hot: {sample.label} is at {_c(temp)}, at or over "
                        f"its {_c(action_at)} limit.",
                        attention_key=key,
                        href=INFO.href,
                        uuid=sample.uuid,
                        temperature_c=temp,
                        limit_c=action_at,
                    )
                )
        elif temp >= notice_at:
            band = "notice"
            if was == "action":
                events.append(self._cooled(sample, temp, key))
            elif was == "ok":
                events.append(
                    self.event(
                        Severity.NOTICE,
                        f"GPU warm: {sample.label} is at {_c(temp)} "
                        f"({_c(action_at)} is the line).",
                        href=INFO.href,
                        uuid=sample.uuid,
                        temperature_c=temp,
                    )
                )
        else:
            band = "ok"
            if was == "action":
                events.append(self._cooled(sample, temp, key))
            # notice -> ok is silent on purpose: a card cooling down is not
            # news, and "GPU warm" followed by "GPU no longer warm" every
            # evening is how a feed becomes wallpaper.
        self._temp_band[sample.uuid] = band
        return events

    def _cooled(self, sample: GpuSample, temp: float, key: str) -> Event:
        self._open.pop(key, None)
        return self.event(
            Severity.NOTICE,
            f"GPU cooled: {sample.label} is back down to {_c(temp)}.",
            attention_key=key,
            href=INFO.href,
            uuid=sample.uuid,
            temperature_c=temp,
            resolved=True,
        )

    def _fan(self, sample: GpuSample, now: float) -> List[Event]:
        """A fan at 0% on a card that is working and hot.

        A card with no fan to report (a passively cooled Tesla P40 in a
        server chassis) reports ``None``, not zero, and is never flagged:
        its cooling is the chassis's job and this bot cannot see it.
        """
        key = fan_key(sample.uuid)
        fan, util, temp = sample.fan_percent, sample.utilization_pct, sample.temperature_c
        stuck = (
            fan is not None
            and float(fan) == 0.0
            and util is not None
            and float(util) > FAN_STUCK_UTILIZATION_PCT
            and temp is not None
            and float(temp) > FAN_STUCK_TEMPERATURE_C
        )
        if stuck and key not in self._open:
            self._open[key] = now
            return [
                self.event(
                    Severity.ACTION,
                    f"Fan stuck: {sample.label} reports 0% fan at {_c(temp)} "
                    f"with {float(util):.0f}% load. A card that works hot with "
                    f"a stopped fan does not last -- stop the load on it.",
                    attention_key=key,
                    href=INFO.href,
                    uuid=sample.uuid,
                    temperature_c=temp,
                    utilization_pct=util,
                )
            ]
        if not stuck and key in self._open:
            del self._open[key]
            if fan is None:
                text = f"Fan fault cleared on {sample.label}: it reports no fan now."
            else:
                text = f"Fan spinning again: {sample.label} reports {float(fan):.0f}%."
            return [
                self.event(
                    Severity.NOTICE,
                    text,
                    attention_key=key,
                    href=INFO.href,
                    uuid=sample.uuid,
                    resolved=True,
                )
            ]
        return []

    def _ecc_errors(self, sample: GpuSample, now: float) -> List[Event]:
        """ECC counts that went up since the previous tick.

        Steady is silent: a card that has had two uncorrected errors since
        boot has had two, and saying so every five minutes says nothing.
        A count that goes *down* is a driver reload or a reboot having
        cleared the volatile counters, which closes the request rather
        than opening one -- the evidence it was raised on no longer exists.
        """
        count = sample.ecc_errors
        if count is None:
            return []
        count = int(count)
        previous = self._ecc.get(sample.uuid)
        self._ecc[sample.uuid] = count
        key = ecc_key(sample.uuid)
        if previous is None:
            return []
        if count > previous:
            gained = count - previous
            if key in self._open:
                return []
            self._open[key] = now
            return [
                self.event(
                    Severity.ACTION,
                    f"ECC errors on {sample.label}: {_plural(gained, 'new error')} "
                    f"since the last check ({count} total). Memory that is "
                    f"failing corrupts results long before it stops working.",
                    attention_key=key,
                    href=INFO.href,
                    uuid=sample.uuid,
                    ecc_errors=count,
                    ecc_gained=gained,
                )
            ]
        if count < previous and key in self._open:
            del self._open[key]
            return [
                self.event(
                    Severity.NOTICE,
                    f"ECC counters reset on {sample.label} (now {count}).",
                    attention_key=key,
                    href=INFO.href,
                    uuid=sample.uuid,
                    ecc_errors=count,
                    resolved=True,
                )
            ]
        return []

    def _memory(self, sample: GpuSample, now: float) -> List[Event]:
        """Memory over 95% for three ticks running.

        A NOTICE, never an ACTION, and that is the judgement the incident
        teaches: a full GPU is almost always someone working, and a bot
        that puts "your GPU is busy" in the badge is a bot the owner
        switches off before the day it has something real to say.
        """
        fraction = sample.memory_fraction
        if fraction is None:
            return []
        if fraction <= MEMORY_PRESSURE_FRACTION:
            self._mem_ticks[sample.uuid] = 0
            return []
        count = self._mem_ticks.get(sample.uuid, 0) + 1
        self._mem_ticks[sample.uuid] = count
        if count != MEMORY_PRESSURE_TICKS:
            return []  # once, on the third: not again on the fourth
        return [
            self.event(
                Severity.NOTICE,
                f"GPU memory full: {sample.label} has used "
                f"{_gib(sample.memory_used_mb)} of {_gib(sample.memory_total_mb)} "
                f"({fraction * 100:.0f}%) for {_plural(count, 'tick')}.",
                href=INFO.href,
                uuid=sample.uuid,
                memory_used_mb=sample.memory_used_mb,
                memory_total_mb=sample.memory_total_mb,
            )
        ]

    def _pcie(self, sample: GpuSample, now: float) -> List[Event]:
        """A link worse than the best this card has ever managed.

        The comparison is against ``self._best_link[uuid]`` and never
        against ``sample.pcie_gen_max``.  The owner's CMP 170HX is a Gen 2
        x4 card with Gen 3 fused off: it reports ``current=2, max=3`` on
        every tick of its life, and a check against the maximum would
        alert on it for ever, about nothing, until the bot was muted -- and
        a muted bot does not tell you a card has vanished either.  "Worse
        than this card has ever actually been" is silent for hardware that
        is doing its best and still catches the riser that came loose.
        """
        link = sample.link
        if link is None:
            return []
        best = self._best_link.get(sample.uuid)
        if best is None or link >= best:
            self._best_link[sample.uuid] = link if best is None else max(best, link)
            return []
        self._best_link[sample.uuid] = best  # a bad day does not lower the bar
        return [
            self.event(
                Severity.NOTICE,
                f"PCIe link degraded: {sample.label} is training at "
                f"{_link_text(link)}, below the {_link_text(best)} it has "
                f"reached before.",
                href=INFO.href,
                uuid=sample.uuid,
                link_gen=link[0],
                link_width=link[1],
                best_gen=best[0],
                best_width=best[1],
            )
        ]

    def _throttle(self, sample: GpuSample, now: float) -> List[Event]:
        """A thermal or power throttle standing for two ticks."""
        reasons = sample.throttling_hot_or_capped()
        if not reasons:
            self._throttle_ticks[sample.uuid] = 0
            return []
        count = self._throttle_ticks.get(sample.uuid, 0) + 1
        self._throttle_ticks[sample.uuid] = count
        if count != THROTTLE_TICKS:
            return []
        return [
            self.event(
                Severity.NOTICE,
                f"{sample.label} has been throttling for "
                f"{_plural(count, 'tick')}: {', '.join(reasons)}.",
                href=INFO.href,
                uuid=sample.uuid,
                throttle_reasons=list(reasons),
            )
        ]

    # -- the probe itself --------------------------------------------------

    def _probe_failed(self, now: float, exc: BaseException) -> Tuple[Event, ...]:
        """A probe that raised.

        Only the exception's *type name* is quoted, never its message: the
        same rule ``jarvis_poke.sources`` and ``poke_bot`` hold themselves
        to, and a probe's message may carry a path or a command line.

        One failure is a line in the feed; three in a row is a question for
        the owner, because at that point the bot has been blind for a
        quarter of an hour and cannot see a card leave.
        """
        self._probe_failures += 1
        self._last_tick_at = now
        self._last_error = type(exc).__name__
        if self._probe_failures < PROBE_FAILURES_FOR_ACTION:
            return (
                self.event(
                    Severity.ERROR,
                    f"GPU probe failed ({type(exc).__name__}); "
                    f"{_plural(self._probe_failures, 'failure')} in a row.",
                    href=INFO.href,
                    failures=self._probe_failures,
                ),
            )
        if PROBE_KEY in self._open:
            return ()  # already asked; the question has not changed
        self._open[PROBE_KEY] = now
        return (
            self.event(
                Severity.ACTION,
                f"GPU probe has failed {self._probe_failures} times in a row "
                f"({type(exc).__name__}). The watch is blind: it cannot tell "
                f"you a card has gone while it cannot read any of them.",
                attention_key=PROBE_KEY,
                href=INFO.href,
                failures=self._probe_failures,
            ),
        )

    def _probe_recovered(self, now: float) -> List[Event]:
        self._probe_failures = 0
        if PROBE_KEY not in self._open:
            return []
        del self._open[PROBE_KEY]
        return [
            self.event(
                Severity.NOTICE,
                "GPU probe is answering again.",
                attention_key=PROBE_KEY,
                href=INFO.href,
                resolved=True,
            )
        ]

    # -- bookkeeping -------------------------------------------------------

    def _remember(self, sample: GpuSample) -> None:
        """Keep the name and index a card had while it was here, so it can
        be named after it is gone."""
        self._baseline[sample.uuid] = {"name": sample.name, "index": sample.index}

    def _record_link(self, sample: GpuSample) -> None:
        link = sample.link
        if link is None:
            return
        best = self._best_link.get(sample.uuid)
        self._best_link[sample.uuid] = link if best is None else max(best, link)

    # -- what the launcher renders -----------------------------------------

    def status(self) -> BotStatus:
        """The card.  Cheap, and it does not raise.

        RUNNING once it has looked, *including while a card is missing*:
        the bot is fine, the hardware is not, and quarantining the
        messenger in the UI would hide the message.  The stats are where
        that shows -- "3 of 4 present" with the absent card named -- and
        the badge, which the supervisor owns, carries the actual request.
        """
        if not self._ticks:
            return self.idle_status("No probe yet.")

        present = {sample.uuid for sample in self._last}
        expected = set(self._baseline)
        missing = sorted(
            expected - present,
            key=lambda u: (self._baseline[u].get("index", 0), u),
        )
        cards = f"{len(present)} of {max(len(expected), len(present))} present"
        if missing:
            cards += f" -- {_plural(len(missing), 'MISSING card')}"

        hottest = max(
            (s for s in self._last if s.temperature_c is not None),
            key=lambda s: float(s.temperature_c or 0.0),
            default=None,
        )
        used = sum(float(s.memory_used_mb or 0.0) for s in self._last)
        total = sum(float(s.memory_total_mb or 0.0) for s in self._last)

        stats = [
            self.stat("Cards", cards),
            self.stat(
                "Hottest",
                "n/a" if hottest is None else f"{_c(hottest.temperature_c)} {hottest.label}",
            ),
            self.stat("Memory in use", f"{_gib(used)} of {_gib(total)}" if total else "n/a"),
        ]
        if missing:
            names = ", ".join(
                f"{self._baseline[u].get('name') or 'GPU'} "
                f"#{self._baseline[u].get('index')}"
                for u in missing
            )
            stats.insert(1, self.stat("Missing", names))
        return BotStatus(state=BotState.RUNNING, stats=tuple(stats), detail=self._detail())

    def _detail(self) -> str:
        if self._last_error and not self._last:
            return f"Last probe failed ({self._last_error})."
        missing = sorted(set(self._baseline) - {s.uuid for s in self._last})
        if missing:
            names = ", ".join(
                f"{self._baseline[u].get('name') or 'GPU'} "
                f"#{self._baseline[u].get('index')}"
                for u in missing
            )
            return f"{names} not on the bus."
        if self._last_error:
            return f"{_plural(len(self._last), 'card')} seen; last probe failed ({self._last_error})."
        return f"{_plural(len(self._last), 'card')} present and answering."

    # -- the owner's own questions -------------------------------------------

    def open_keys(self) -> List[str]:
        """The attention keys this bot believes are open, sorted.

        The supervisor owns the badge; this is the bot's record, and the
        two agree because the supervisor closes on the same resolution
        events this bot raises.  Exposed so a wiring can check that.
        """
        return sorted(self._open)

    def expected_fleet(self) -> Dict[str, Dict[str, Any]]:
        """A copy of the remembered fleet: uuid -> name and index."""
        return {uuid: dict(row) for uuid, row in sorted(self._baseline.items())}

    # -- the framework's optional half ---------------------------------------

    def on_pause(self) -> None:
        """The supervisor clears this bot's attention when it is paused, so
        the bot forgets its own record too.

        Keeping it would mean a card that is still missing on resume is
        never re-raised -- every condition here is edge triggered -- and
        the badge would stay empty over a dead GPU.  The temperature bands
        go with it for the same reason: they are the edge state for heat,
        and a card still at 90C on resume has to be announced again.
        """
        self._open.clear()
        self._temp_band.clear()

    def snapshot(self) -> Dict[str, Any]:
        """JSON-able state: "State is the bot's, persistence is ours."

        All five things that would be *wrong* to lose:

        * the **baseline fleet**, with each card's remembered name and
          index -- lose this and the bot re-derives "what should be here"
          from "what is here", which is precisely the blindness it exists
          to fix;
        * the **best link seen per uuid** -- lose it and the first tick
          after a restart re-baselines a degraded link as normal;
        * the **last ECC counts** -- lose them and a restart either
          re-alerts on errors already reported or misses the next one;
        * the **consecutive-condition counters** (memory, throttle, probe
          failures, and which temperature band each card was in), so a
          restart does not restart the count and so a standing condition
          is not announced twice;
        * the **open attention keys**, so a restart can still *resolve* a
          request it raised before it, instead of leaving a card in the
          badge for ever.

        Configuration -- the probe, the thresholds -- is deliberately
        absent: a snapshot that pinned it would quietly resurrect
        yesterday's settings.
        """
        return {
            "version": SNAPSHOT_VERSION,
            "ticks": self._ticks,
            "last_tick_at": self._last_tick_at,
            "have_baseline": self._have_baseline,
            "baseline": {
                uuid: {"name": row.get("name", ""), "index": row.get("index", 0)}
                for uuid, row in sorted(self._baseline.items())
            },
            "best_link": {
                uuid: [int(link[0]), int(link[1])]
                for uuid, link in sorted(self._best_link.items())
            },
            "ecc": {uuid: int(count) for uuid, count in sorted(self._ecc.items())},
            "memory_ticks": {
                uuid: int(n) for uuid, n in sorted(self._mem_ticks.items()) if n
            },
            "throttle_ticks": {
                uuid: int(n) for uuid, n in sorted(self._throttle_ticks.items()) if n
            },
            "temp_band": {
                uuid: band
                for uuid, band in sorted(self._temp_band.items())
                if band != "ok"
            },
            "probe_failures": int(self._probe_failures),
            "open": {key: float(at) for key, at in sorted(self._open.items())},
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Take back what :meth:`snapshot` returned.

        A snapshot from a *newer* build is refused rather than half-read:
        a half-read baseline is a fleet with a card silently dropped from
        it, and a card dropped from the baseline is a card whose
        disappearance is never reported -- the original incident,
        reintroduced by the persistence layer.  Anything merely missing or
        malformed inside a snapshot of a version this build understands is
        treated as absent, because a bot that will not start is worse than
        a bot that has forgotten a counter.
        """
        if not isinstance(snapshot, Mapping) or not snapshot:
            return
        version = snapshot.get("version", SNAPSHOT_VERSION)
        if not isinstance(version, int) or version > SNAPSHOT_VERSION:
            raise GpuBotError(
                f"gpu bot snapshot version {version!r} is newer than this build "
                f"understands (version {SNAPSHOT_VERSION})"
            )

        self._ticks = _as_int(snapshot.get("ticks"), self._ticks)
        self._last_tick_at = _as_float(snapshot.get("last_tick_at"), self._last_tick_at)
        self._probe_failures = max(0, _as_int(snapshot.get("probe_failures"), 0))

        baseline = snapshot.get("baseline")
        if isinstance(baseline, Mapping):
            self._baseline = {
                str(uuid): {
                    "name": str((row or {}).get("name", "")),
                    "index": _as_int((row or {}).get("index"), 0),
                }
                for uuid, row in baseline.items()
                if isinstance(uuid, str) and uuid and isinstance(row, Mapping)
            }
        # A snapshot that holds a fleet has a baseline whether or not the
        # flag survived; an older snapshot without the flag is still a
        # restart that must not re-baseline.
        self._have_baseline = bool(
            snapshot.get("have_baseline", bool(self._baseline))
        ) or bool(self._baseline)

        best = snapshot.get("best_link")
        if isinstance(best, Mapping):
            self._best_link = {}
            for uuid, pair in best.items():
                if not isinstance(uuid, str) or not isinstance(pair, (list, tuple)):
                    continue
                if len(pair) != 2:
                    continue
                self._best_link[uuid] = (_as_int(pair[0], 0), _as_int(pair[1], 0))

        ecc = snapshot.get("ecc")
        if isinstance(ecc, Mapping):
            self._ecc = {
                str(uuid): _as_int(count, 0)
                for uuid, count in ecc.items()
                if isinstance(uuid, str)
            }

        self._mem_ticks = _counter_map(snapshot.get("memory_ticks"))
        self._throttle_ticks = _counter_map(snapshot.get("throttle_ticks"))

        band = snapshot.get("temp_band")
        if isinstance(band, Mapping):
            self._temp_band = {
                str(uuid): str(value)
                for uuid, value in band.items()
                if isinstance(uuid, str) and value in ("ok", "notice", "action")
            }

        open_keys = snapshot.get("open")
        if isinstance(open_keys, Mapping):
            self._open = {
                str(key): _as_float(at, 0.0)
                for key, at in open_keys.items()
                if isinstance(key, str) and key.strip()
            }


def _counter_map(raw: Any) -> Dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    return {
        str(uuid): max(0, _as_int(count, 0))
        for uuid, count in raw.items()
        if isinstance(uuid, str)
    }


def _checked_samples(samples: Any) -> Tuple[GpuSample, ...]:
    """What the probe returned, checked before any of it is believed.

    A probe that returns something that is not a list of samples is a
    broken probe, and a broken probe must be read as "I could not look"
    and never as "there are no GPUs" -- the second empties the fleet and
    reports every card missing at once.  Raising here puts it on the probe
    failure path, which is where it belongs.
    """
    if isinstance(samples, GpuSample) or isinstance(samples, (str, bytes)):
        raise GpuProbeError(
            f"probe must return a sequence of GpuSample; got "
            f"{type(samples).__name__}"
        )
    try:
        listed = list(samples)
    except TypeError:
        raise GpuProbeError(
            f"probe must return a sequence of GpuSample; got "
            f"{type(samples).__name__}"
        ) from None
    out: List[GpuSample] = []
    seen: Dict[str, int] = {}
    for item in listed:
        if not isinstance(item, GpuSample):
            raise GpuProbeError(
                f"probe returned a {type(item).__name__} where a GpuSample "
                f"was expected"
            )
        if item.uuid in seen:
            raise GpuProbeError(f"probe returned uuid {item.uuid!r} twice")
        seen[item.uuid] = 1
        out.append(item)
    return tuple(out)


def build(
    *,
    clock: Clock,
    probe: Callable[[], Sequence[GpuSample]],
    temp_action_c: float = DEFAULT_TEMP_ACTION_C,
    temp_notice_c: float = DEFAULT_TEMP_NOTICE_C,
) -> GpuBot:
    """Construct the bot with everything injected.

    The one place to change when this bot grows a dependency, and what a
    config registers: ``{BOT_ID: lambda: build(clock=clock,
    probe=nvidia_smi_probe)}``.  ``clock`` and ``probe`` are both required
    and have no defaults -- a bot that *can* fall back to the wall clock
    eventually will, and a bot that defaults to shelling out is a bot that
    shells out in somebody's test suite.
    """
    return GpuBot(
        clock=clock,
        probe=probe,
        temp_action_c=temp_action_c,
        temp_notice_c=temp_notice_c,
    )
