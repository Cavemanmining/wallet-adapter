"""Disk watch -- a Jarvis bot that says *when* a disk fills, not just that
it is full.

Why this bot exists
-------------------
The owner has had to go and free up space by hand, which means the warning
arrived too late or not at all.  A disk filling is one of the few faults
that is completely predictable from its own trend, so the useful sentence
is not "you are at 90 percent" -- by then the render is already dying --
but "at the current rate this fills on Tuesday morning".  This machine
renders video and 3D frames, so a directory growing by tens of gigabytes
is *normal*; a bot that shouted at every large write would be muted within
a week.  Hence the shape of the rules below: the projection is the
headline, absolute usage is the backstop, and everything else is a line in
the feed.

The rules, and the key each one stands on
-----------------------------------------
Every standing request for a decision carries a stable
``attention_key`` (contracts.py: the badge "counts distinct open keys"),
so one fact nagging across a hundred ticks is one badge item, and two
different facts about one mountpoint are two:

==============================  ============================================
``disk:filling:<mountpoint>``   ACTION.  The trend says this fills inside
                                the projected-days threshold.  Carries the
                                rate in GB/day and the projected date.
``disk:full:<mountpoint>``      ACTION.  Usage is at or above the percent
                                threshold *now*.
``disk:inodes:<mountpoint>``    ACTION.  Inodes are at or above the same
                                percent threshold.  A different failure
                                with the same symptom ("no space left on
                                device" while ``df`` shows free space), and
                                the one people miss.
``disk:readonly:<mountpoint>``  ACTION.  The filesystem has been remounted
                                read-only, which usually means the disk is
                                failing.
``disk:probe-failed``           ACTION, and only after
                                :data:`ESCALATE_AFTER_FAILURES` consecutive
                                failures.  A watcher that cannot look is
                                worse than no watcher, because the launcher
                                still says "running".
==============================  ============================================

``disk:filling`` and ``disk:full`` can be open at the same time and are
never merged: one says "it is full now", the other says "it will be", and
the second is the one that is still actionable.

The sudden-drop rule is the deliberate exception: free space falling by
more than the configured amount in a single interval is a ``NOTICE`` --
something just wrote a lot -- with *no* attention key.  It is news, not a
standing question: there is nothing for the owner to decide and nothing
that could ever resolve it, and a key that never closes is how a badge
becomes wallpaper.  Its stable identifier travels in ``data["rule"]`` so a
feed can still group them.

Recovery closes a key the way the framework documents it
(:data:`jarvis_bots.supervisor.RESOLVED_FLAG`, and see
``jarvis_bots/templates/bot.py.tmpl`` and
:meth:`jarvis_bots.bots.poke_bot.PokeBot._resolve`): the *same key*
re-raised below ACTION -- so ``Event.wants_attention`` is false and the
request is not re-opened -- with ``resolved=True`` in ``data``.

How the projection is computed
------------------------------
Plain integer and float arithmetic, no numpy, no dependencies:

1. Each tick appends ``(now, free_bytes)`` to a bounded per-mountpoint
   history (:data:`MAX_SAMPLES_PER_MOUNT` points, :data:`HISTORY_WINDOW_S`
   of age; both pruned on every append, so the snapshot cannot grow
   forever).
2. Over the retained window we fit an ordinary least-squares line
   ``free = a + b * t`` -- :func:`linear_trend` -- computed mean-centred
   (``b = sum((t-t̄)(f-f̄)) / sum((t-t̄)²)``) because unix timestamps are
   around 1.7e9 and the uncentred sums lose most of their significant
   digits to cancellation.
3. ``b`` is bytes per second and is *negative* while the disk fills; the
   fill rate the owner reads is ``-b * 86400`` bytes per day.
4. Time to full is the **measured** free space divided by that fitted
   rate: the intercept is a fit, but how much room is left right now is
   something we actually observed, and it is the number the owner would
   check with ``df``.

Nothing is projected from less than :data:`MIN_SAMPLES` points or less
than :data:`MIN_SPAN_S` of history.  When there is not enough yet the bot
*says so* (:meth:`DiskBot.reasons`, and the card's detail line) rather
than extrapolating from two readings taken a minute apart.

A single huge delete -- the thing this machine does after every render --
must not turn into a nonsense negative rate or a projection into the past.
Two guards: free space *gaining* more than :data:`DEFAULT_DROP_BYTES`
worth in one interval discards the history before it (the trend that
existed described a different disk), and a fitted slope that is not
negative never projects at all.

Injection, and the boundary
---------------------------
``probe`` and ``clock`` are constructor arguments; this module opens no
sockets, calls no ``time.time()`` and **never shells out**.
:func:`statvfs_probe` (``os.statvfs``) and :func:`du_probe` (which only
*builds* a ``du`` command line and *parses* its output) are module-level
helpers for a composition root to use or inject -- running a subprocess is
the app's decision, made where a person can see it, not something a
background tick does on its own.

Nothing here deletes, moves or touches a single file.  The widest thing
this bot can do is return an :class:`~jarvis_bots.contracts.Event` with a
link.
"""

from __future__ import annotations

import datetime
import math
import os
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

from jarvis_bots.base import BaseBot
from jarvis_bots.contracts import BotInfo, BotState, BotStatus, Event, Severity

__all__ = [
    "BOT_ID",
    "INFO",
    "DEFAULT_PERCENT_THRESHOLD",
    "DEFAULT_PROJECTED_DAYS",
    "DEFAULT_DROP_BYTES",
    "MIN_SAMPLES",
    "MIN_SPAN_S",
    "MAX_SAMPLES_PER_MOUNT",
    "HISTORY_WINDOW_S",
    "ESCALATE_AFTER_FAILURES",
    "PROBE_ATTENTION_KEY",
    "SNAPSHOT_VERSION",
    "DiskBotError",
    "MountSample",
    "Projection",
    "DuPlan",
    "DiskBot",
    "build",
    "linear_trend",
    "statvfs_probe",
    "du_probe",
    "parse_du_output",
    "filling_key",
    "full_key",
    "inodes_key",
    "readonly_key",
    "human_bytes",
]

#: The id the supervisor, the launcher and every event carry.
BOT_ID = "disk"

#: Static identity (contracts.BotInfo), declared once.
INFO = BotInfo(
    id=BOT_ID,
    name="Disk watch",
    blurb="Says when a disk will fill, not just that it has.",
    kind="grid",
    interval_s=900.0,
    href="/bots/disk",
    can_pause=True,
)

#: Usage at or above this percentage is full enough to ask about.
DEFAULT_PERCENT_THRESHOLD = 90.0

#: A projection inside this many days is worth the owner's attention.
DEFAULT_PROJECTED_DAYS = 3.0

#: Free space falling by more than this in one interval is a NOTICE.  It is
#: also the gain that resets a trend: on a machine that writes 10 GB of
#: frames in a quarter of an hour, 10 GB back in a quarter of an hour is a
#: cleanup, and the trend before it described a different disk.
DEFAULT_DROP_BYTES = 10 * 1024 ** 3

#: No projection from fewer points than this, however tidy they look.
MIN_SAMPLES = 4

#: No projection from a window shorter than this.  Four samples a minute
#: apart during a render would "prove" the disk fills before lunch.
MIN_SPAN_S = 3600.0

#: Bounds on the retained history, per mountpoint, applied on every append
#: *and* on restore: "snapshot/restore persists the sample history ...
#: bounded so it cannot grow forever."  At the 900s default interval 240
#: points is two and a half days of trend, which is plenty to see three
#: days ahead.
MAX_SAMPLES_PER_MOUNT = 240
HISTORY_WINDOW_S = 3.0 * 86400.0

#: Consecutive probe failures before the bot stops merely logging and asks
#: for a decision.  One failed read is a hiccup; three in a row (three
#: quarters of an hour at the default interval) means nobody is watching
#: the disks any more, and that is worth the badge.
ESCALATE_AFTER_FAILURES = 3

#: The one key that is not per-mountpoint: the bot cannot see *anything*.
PROBE_ATTENTION_KEY = "disk:probe-failed"

#: A fill rate below this is not a trend, it is noise on a machine that
#: writes logs.  Reported as steady rather than "full in 41 years".
MIN_FILL_BYTES_PER_DAY = 1024 ** 2

#: Projections further out than this are not projections.
MAX_PROJECTION_DAYS = 3650.0

#: The projected-days boundary is inclusive, with an epsilon of about a
#: tenth of a second: whether a disk that fills in *exactly* three days
#: alerts must not depend on the last bit of a float division.
DAYS_EPSILON = 1e-6

#: Snapshot format.  Bump it when the shape changes and teach
#: :meth:`DiskBot.restore` the old shape.
SNAPSHOT_VERSION = 1

#: ``du -k`` reports fixed 1024-byte blocks.
DU_BLOCK_BYTES = 1024

_GIB = 1024 ** 3


class DiskBotError(Exception):
    """Misconfiguration of the bot itself -- a threshold that cannot mean
    anything, a probe that is not callable.  Raised at construction, never
    from a tick: a tick reports trouble as an event."""


# --------------------------------------------------------------------------
# What a probe hands over
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MountSample:
    """One filesystem, as one look at it.

    Frozen, because a sample is a measurement: something that happened at a
    moment, and nothing downstream has any business editing it.  Bytes are
    integers -- a disk is counted, not estimated -- and the three byte
    fields are kept separately rather than derived from one another
    because they genuinely do not add up: reserved blocks mean
    ``used + free < total`` on most Linux filesystems, and a bot that
    "helpfully" computed free as ``total - used`` would report several
    gigabytes of room that only root can use.
    """

    mountpoint: str
    device: str = ""
    filesystem: str = ""
    total_bytes: int = 0
    free_bytes: int = 0
    used_bytes: int = 0
    inodes_total: int = 0
    inodes_free: int = 0
    read_only: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.mountpoint, str) or not self.mountpoint.strip():
            raise ValueError(
                f"a sample needs a mountpoint: it is what every rule, key and "
                f"line of history is filed under; got {self.mountpoint!r}"
            )
        for field_name in (
            "total_bytes",
            "free_bytes",
            "used_bytes",
            "inodes_total",
            "inodes_free",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{field_name} must be an int of {self.mountpoint}; got "
                    f"{type(value).__name__}"
                )
            if value < 0:
                raise ValueError(f"{field_name} cannot be negative; got {value}")

    @property
    def percent_used(self) -> float:
        """Bytes used as a percentage of the total, or ``0.0`` for a
        filesystem with no size (a pseudo-filesystem, or one the probe
        could only half read)."""
        if self.total_bytes <= 0:
            return 0.0
        return self.used_bytes * 100.0 / self.total_bytes

    @property
    def inodes_used(self) -> int:
        return max(0, self.inodes_total - self.inodes_free)

    @property
    def percent_inodes_used(self) -> float:
        """Inodes used as a percentage.  ``0.0`` when the filesystem does
        not have a fixed inode table at all (btrfs, xfs with dynamic
        inodes, and APFS all report zero), which is honest: there is no
        exhaustion to warn about."""
        if self.inodes_total <= 0:
            return 0.0
        return self.inodes_used * 100.0 / self.inodes_total

    def at_or_over(self, percent: float) -> bool:
        """Usage at or above ``percent``, compared without floating point
        division so the boundary is exact.

        ``used / total * 100 >= 90`` decided a 90%-exactly disk on the last
        bit of a division; ``used * 100 >= 90 * total`` is the same
        question asked in whole numbers.
        """
        if self.total_bytes <= 0:
            return False
        return self.used_bytes * 100.0 >= float(percent) * self.total_bytes

    def inodes_at_or_over(self, percent: float) -> bool:
        if self.inodes_total <= 0:
            return False
        return self.inodes_used * 100.0 >= float(percent) * self.inodes_total


@dataclass(frozen=True)
class Projection:
    """What the trend says about one mountpoint, including "nothing yet".

    ``days_to_full`` is ``None`` whenever there is no honest projection to
    give -- not enough history, free space steady or growing, or a rate so
    slow the answer is decades -- and ``reason`` always says which, in the
    words the card and the event use.  A projection that reports ``None``
    with an explanation is the whole point of this dataclass: the failure
    mode being designed out is a bot that guesses.
    """

    mountpoint: str
    samples: int
    span_s: float
    free_bytes: int
    fill_bytes_per_day: float
    days_to_full: Optional[float]
    full_at: Optional[float]
    reason: str

    @property
    def trending(self) -> bool:
        return self.days_to_full is not None


# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------


def filling_key(mountpoint: str) -> str:
    """The key for "it will be full", which is the one still worth acting
    on."""
    return f"disk:filling:{mountpoint}"


def full_key(mountpoint: str) -> str:
    """The key for "it is full now".  Never merged with
    :func:`filling_key`: they are two different facts about one disk, and
    the owner does something different about each."""
    return f"disk:full:{mountpoint}"


def inodes_key(mountpoint: str) -> str:
    return f"disk:inodes:{mountpoint}"


def readonly_key(mountpoint: str) -> str:
    return f"disk:readonly:{mountpoint}"


def drop_rule(mountpoint: str) -> str:
    """The stable identifier of the sudden-drop NOTICE.

    Deliberately *not* an attention key -- see the module docstring -- but
    stable, and carried in the event's ``data`` so a feed can group the
    repeats of one mountpoint's writes.
    """
    return f"disk:drop:{mountpoint}"


# --------------------------------------------------------------------------
# The trend, in plain arithmetic
# --------------------------------------------------------------------------


def linear_trend(points: Sequence[Tuple[float, int]]) -> Optional[float]:
    """Ordinary least-squares slope of ``free_bytes`` against time, in
    bytes per second, or ``None`` when no line can be fitted.

    The method, stated so it can be checked by hand:

    * ``t̄`` and ``f̄`` are the means of the times and the free-byte
      readings;
    * ``b = Σ (tᵢ - t̄)(fᵢ - f̄) / Σ (tᵢ - t̄)²``.

    Mean-centring is not cosmetic.  Unix timestamps are ~1.7e9, so
    ``Σ tᵢ fᵢ`` in the textbook form is ~1e20 and the answer is the
    difference of two such numbers: in IEEE doubles that throws away
    almost every significant digit of the slope, and the projection
    wobbles by days between ticks.  Centred, every term is a small
    displacement and the sum is well conditioned.

    ``None`` when there are fewer than two points, or when every reading
    was taken at the same instant (``Σ (tᵢ - t̄)² == 0``) -- a vertical
    line has no slope, and dividing by that zero is how a watcher starts
    reporting infinities.
    """
    count = len(points)
    if count < 2:
        return None
    mean_t = math.fsum(float(t) for t, _f in points) / count
    mean_f = math.fsum(float(f) for _t, f in points) / count
    numerator = math.fsum(
        (float(t) - mean_t) * (float(f) - mean_f) for t, f in points
    )
    denominator = math.fsum((float(t) - mean_t) ** 2 for t, _f in points)
    if denominator <= 0.0:
        return None
    slope = numerator / denominator
    if not math.isfinite(slope):
        return None
    return slope


# --------------------------------------------------------------------------
# Probes: the two things that touch the machine, neither of them the bot
# --------------------------------------------------------------------------


def statvfs_probe(
    mountpoints: Iterable[str],
    *,
    statvfs: Callable[[str], Any] = os.statvfs,
    mounts_path: str = "/proc/self/mounts",
) -> List[MountSample]:
    """Read each mountpoint with ``os.statvfs`` and return the samples.

    This is the real probe a composition root injects.  It raises -- a
    missing mountpoint, a permission error -- rather than reporting a
    plausible zero, because a zero the caller cannot tell apart from "I
    could not look" is how a watcher goes quiet without anyone noticing.
    :meth:`DiskBot.tick` catches it and reports an ``ERROR`` event.

    Which numbers, and why those:

    ``free_bytes``   ``f_bavail``, the space available to *this* user, not
                     ``f_bfree``, which includes the 5% root reservation.
                     The owner's render cannot write into the reservation,
                     so counting it would promise room that does not exist.
    ``used_bytes``   ``f_blocks - f_bfree``, which is what ``df`` calls
                     Used.  It plus ``free_bytes`` is less than
                     ``total_bytes`` by exactly the reservation; that gap
                     is real and is not papered over.
    ``read_only``    the ``ST_RDONLY`` bit of ``f_flag``.  A filesystem the
                     kernel remounted read-only after an I/O error looks
                     perfectly healthy in every byte count.

    ``device`` and ``filesystem`` are not in ``statvfs`` at all; they are
    looked up in ``/proc/self/mounts`` when that file exists and left empty
    when it does not (macOS, a container without ``/proc``).  They are
    labels for the alert text, so an empty one costs nothing.
    """
    table = _mount_table(mounts_path)
    samples: List[MountSample] = []
    for mountpoint in mountpoints:
        path = str(mountpoint)
        stats = statvfs(path)
        frsize = int(getattr(stats, "f_frsize", 0) or getattr(stats, "f_bsize", 0) or 0)
        blocks = int(getattr(stats, "f_blocks", 0) or 0)
        bfree = int(getattr(stats, "f_bfree", 0) or 0)
        bavail = int(getattr(stats, "f_bavail", 0) or 0)
        files = int(getattr(stats, "f_files", 0) or 0)
        favail = int(getattr(stats, "f_favail", getattr(stats, "f_ffree", 0)) or 0)
        flag = int(getattr(stats, "f_flag", 0) or 0)
        readonly = bool(flag & getattr(os, "ST_RDONLY", 1))
        device, filesystem = table.get(_real(path), ("", ""))
        samples.append(
            MountSample(
                mountpoint=path,
                device=device,
                filesystem=filesystem,
                total_bytes=max(0, blocks * frsize),
                free_bytes=max(0, bavail * frsize),
                used_bytes=max(0, (blocks - bfree) * frsize),
                inodes_total=max(0, files),
                inodes_free=max(0, min(favail, files) if files else favail),
                read_only=readonly,
            )
        )
    return samples


def _real(path: str) -> str:
    try:
        return os.path.realpath(path)
    except OSError:  # pragma: no cover - realpath on a vanished path
        return path


def _mount_table(mounts_path: str) -> Dict[str, Tuple[str, str]]:
    """``{mountpoint: (device, fstype)}`` from ``/proc/self/mounts``.

    Best effort by design: every failure returns what has been read so
    far, because a label missing from an alert is a cosmetic loss and a
    disk watcher that will not start is not.  Later entries win, which is
    what the kernel means -- a mount over a mount shadows it.
    """
    table: Dict[str, Tuple[str, str]] = {}
    try:
        with open(mounts_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) < 3:
                    continue
                device, mountpoint, fstype = parts[0], parts[1], parts[2]
                table[_unescape_mount(mountpoint)] = (
                    _unescape_mount(device),
                    fstype,
                )
    except OSError:
        return table
    return table


def _unescape_mount(field: str) -> str:
    """``/proc`` escapes space, tab, newline and backslash as octal."""
    if "\\" not in field:
        return field
    out: List[str] = []
    index = 0
    while index < len(field):
        char = field[index]
        octal = field[index + 1 : index + 4]
        if char == "\\" and len(octal) == 3 and octal.isdigit():
            try:
                out.append(chr(int(octal, 8)))
                index += 4
                continue
            except ValueError:  # pragma: no cover - not octal after all
                pass
        out.append(char)
        index += 1
    return "".join(out)


@dataclass(frozen=True)
class DuPlan:
    """A ``du`` invocation the *app* may run, and the parser for what comes
    back.

    The bot never shells out; this is a plan for a subprocess, not a
    subprocess.  Keeping the command line and its parser in one object is
    the point: the flags below are exactly what make the output parseable,
    and a caller who copies the argv but writes their own parser (or the
    reverse) gets figures that are wrong by a factor of two without
    anything raising.
    """

    path: str
    depth: int
    argv: Tuple[str, ...]

    def parse(self, output: str) -> List[Tuple[str, int]]:
        """``du``'s stdout to ``(path, bytes)`` pairs."""
        return parse_du_output(output)


def du_probe(path: str, depth: int = 1) -> DuPlan:
    """Build the ``du`` command line for the largest-growers feature.

    Every flag earns its place:

    ``-k``   sizes in fixed 1024-byte blocks.  ``du``'s default unit is
             whatever ``BLOCKSIZE``/``DU_BLOCK_SIZE``/``POSIXLY_CORRECT``
             say it is, so without this the same command reports 512-byte
             blocks on one machine and 1024 on another and every figure is
             silently out by two.
    ``-x``   stay on one filesystem.  Without it, ``du`` on a mountpoint
             walks into everything mounted underneath and the "largest
             directory on this disk" is a directory on another disk.
    ``-d``   the depth limit, spelled the way both GNU coreutils and BSD
             ``du`` accept (GNU's ``--max-depth`` does not exist on macOS).
    ``--``   so a path that begins with a dash is a path.

    ``depth`` must be a non-negative integer: a negative depth is
    meaningless to ``du`` and a float would be passed through as
    ``"1.0"``, which it rejects.
    """
    text = str(path)
    if not text:
        raise ValueError("du needs a path to measure")
    if isinstance(depth, bool) or not isinstance(depth, int):
        raise TypeError(f"du depth must be an int; got {type(depth).__name__}")
    if depth < 0:
        raise ValueError(f"du depth cannot be negative; got {depth}")
    return DuPlan(
        path=text,
        depth=depth,
        argv=("du", "-k", "-x", "-d", str(depth), "--", text),
    )


def parse_du_output(output: str) -> List[Tuple[str, int]]:
    """``du -k`` output to ``(path, bytes)`` pairs, in the order given.

    One line is ``<blocks>\\t<path>``.  Lines that do not start with an
    integer are skipped rather than raising: ``du`` writes
    ``du: cannot read directory ...`` for every unreadable directory, and
    a caller that merged stderr into stdout (or a locale that translated
    those messages) must not lose the ninety directories it *could* read.
    A path containing a tab survives, because only the first tab is
    treated as the separator.

    Order is ``du``'s own -- children before their parent -- and is not
    sorted here: the caller decides whether it wants the biggest, the
    fastest-growing, or the tree.
    """
    if not isinstance(output, str):
        raise TypeError(f"du output must be text; got {type(output).__name__}")
    pairs: List[Tuple[str, int]] = []
    for raw in output.splitlines():
        line = raw.rstrip("\r")
        if not line.strip():
            continue
        head, sep, tail = line.partition("\t")
        if not sep:
            head, _, tail = line.strip().partition(" ")
            tail = tail.strip()
        head = head.strip()
        if not tail:
            continue
        try:
            blocks = int(head)
        except ValueError:
            continue
        if blocks < 0:
            continue
        pairs.append((tail, blocks * DU_BLOCK_BYTES))
    return pairs


# --------------------------------------------------------------------------
# Formatting: the bot knows how its own numbers should read
# --------------------------------------------------------------------------


def human_bytes(value: float) -> str:
    """Bytes as the owner's ``df -h`` spells them: binary multiples, one
    decimal, and the ``G`` suffix written out as GB."""
    number = float(value)
    for label, size in (
        ("TB", 1024 ** 4),
        ("GB", _GIB),
        ("MB", 1024 ** 2),
        ("kB", 1024),
    ):
        if abs(number) >= size:
            return f"{number / size:.1f} {label}"
    return f"{int(number)} B"


def _pct(value: float) -> str:
    return f"{value:.1f}%"


def _days(value: float) -> str:
    if value < 1.0:
        hours = value * 24.0
        return "under an hour" if hours < 1.0 else f"{hours:.0f} hours"
    return f"{value:.1f} days"


def _when(timestamp: float) -> str:
    """The projected date, in UTC.

    UTC and not local time on purpose: the bot is handed a unix timestamp
    by an injected clock and has no business reading the machine's
    timezone, which would also make the same sequence of ticks produce
    different text on two machines.
    """
    moment = datetime.datetime.fromtimestamp(float(timestamp), datetime.timezone.utc)
    return moment.strftime("%a %d %b %H:%M UTC")


def _count(value: int) -> str:
    return f"{int(value):,}"


# --------------------------------------------------------------------------
# The bot
# --------------------------------------------------------------------------


class DiskBot(BaseBot):
    """Watches mountpoints and says when they will fill.

    Implements ``contracts.Bot``: ``info``, :meth:`tick` and :meth:`status`
    are the required three; :meth:`snapshot`, :meth:`restore` and
    :meth:`on_pause` are overridden because this bot has state worth
    keeping and attention worth forgetting.
    """

    info = INFO

    def __init__(
        self,
        *,
        clock: Callable[[], float],
        probe: Callable[[], Sequence[MountSample]],
        mountpoints: Optional[Iterable[str]] = None,
        percent_threshold: float = DEFAULT_PERCENT_THRESHOLD,
        projected_days_threshold: float = DEFAULT_PROJECTED_DAYS,
        drop_bytes: int = DEFAULT_DROP_BYTES,
        max_samples: int = MAX_SAMPLES_PER_MOUNT,
        history_window_s: float = HISTORY_WINDOW_S,
    ) -> None:
        super().__init__(clock)

        if not callable(probe):
            raise DiskBotError(
                "probe must be callable() -> list[MountSample]; this bot reads "
                "no filesystem itself (see statvfs_probe), so a probe it "
                f"cannot call can never see anything. Got {type(probe).__name__}"
            )
        self._probe = probe

        #: The mountpoints to watch, or ``None`` for "whatever the probe
        #: reports".  A fixed list is the honest default for a machine with
        #: one scratch disk that matters; ``None`` suits a probe that has
        #: already decided.
        self._watched: Optional[Tuple[str, ...]] = (
            None if mountpoints is None else tuple(str(m) for m in mountpoints)
        )
        if self._watched is not None and not self._watched:
            raise DiskBotError(
                "mountpoints is an empty list: that is a bot that watches "
                "nothing and reports it is running. Pass None to watch "
                "everything the probe returns."
            )

        percent = float(percent_threshold)
        if not 0.0 < percent <= 100.0:
            raise DiskBotError(
                f"percent_threshold must be inside (0, 100]; got {percent_threshold!r}"
            )
        self.percent_threshold = percent

        days = float(projected_days_threshold)
        if not days > 0.0 or not math.isfinite(days):
            raise DiskBotError(
                f"projected_days_threshold must be a positive number of days; "
                f"got {projected_days_threshold!r}"
            )
        self.projected_days_threshold = days

        drop = int(drop_bytes)
        if drop <= 0:
            raise DiskBotError(
                f"drop_bytes must be a positive number of bytes; got {drop_bytes!r}"
            )
        self.drop_bytes = drop

        self.max_samples = max(MIN_SAMPLES, int(max_samples))
        window = float(history_window_s)
        if window < MIN_SPAN_S:
            raise DiskBotError(
                f"history_window_s must be at least MIN_SPAN_S ({MIN_SPAN_S}s), "
                f"or nothing could ever be projected; got {history_window_s!r}"
            )
        self.history_window_s = window

        # --- state; all of it persisted except the per-tick notes ---
        #: mountpoint -> [(unix seconds, free bytes)], oldest first.
        self._history: Dict[str, List[Tuple[float, int]]] = {}
        #: attention key -> when this bot raised it.
        self._open: Dict[str, float] = {}
        #: mountpoint -> the last sample, so a restarted card is not blank.
        self._last_seen: Dict[str, MountSample] = {}
        self._consecutive_failures = 0
        self._ticks = 0
        self._last_tick_at = 0.0
        self._last_fault = ""

        #: mountpoint -> the current projection, recomputed every tick.
        self._projections: Dict[str, Projection] = {}
        #: mountpoint -> a note about a trend reset, for this tick only.
        self._reset_notes: Dict[str, str] = {}

    # -- the work -----------------------------------------------------------

    def tick(self, now: float) -> Sequence[Event]:
        """One look at every watched filesystem.

        Returns promptly, never sleeps, and -- unlike the scaffold's
        example -- never raises.  contracts.py allows a tick to raise and
        the supervisor handles it, but a failing probe here is an ordinary
        event (an unmounted disk, a permission change), and four of those
        in a row would quarantine the bot for half an hour, leaving the
        owner with *no* disk watch at exactly the moment something is wrong
        with the disks.  So the probe's failure comes back as an ``ERROR``
        event, and three in a row escalate to a request for a decision.
        """
        at = float(now)
        self._reset_notes = {}
        try:
            samples = self._read(self._probe())
        except Exception as exc:  # noqa: BLE001 - deliberate; see the docstring
            return self._probe_failed(exc, at)

        events: List[Event] = list(self._probe_recovered(at))
        self._ticks += 1
        self._last_tick_at = at

        for mountpoint in sorted(samples):
            events.extend(self._examine(samples[mountpoint], at))
        events.extend(self._forget_missing(set(samples), at))
        return tuple(events)

    def _read(self, samples: Any) -> Dict[str, MountSample]:
        """Validate what the probe returned and filter it to the watched set.

        A probe that returns rubbish is a broken probe, and is reported the
        same way as one that raised: this method raises, :meth:`tick`
        catches.  The alternative -- skipping the rows it cannot read -- is
        a disk watch that silently watches nothing.
        """
        if isinstance(samples, MountSample) or isinstance(samples, (str, bytes)):
            raise DiskBotError(
                f"probe must return a sequence of MountSample; got "
                f"{type(samples).__name__}"
            )
        rows: Dict[str, MountSample] = {}
        for sample in samples:
            if not isinstance(sample, MountSample):
                raise DiskBotError(
                    f"probe returned {type(sample).__name__}, not a MountSample"
                )
            if self._watched is not None and sample.mountpoint not in self._watched:
                continue
            rows[sample.mountpoint] = sample
        return rows

    def _examine(self, sample: MountSample, now: float) -> List[Event]:
        """Every rule, for one filesystem, in a fixed order.

        The order is the order the owner would want to read them in if all
        of them fired at once: the disk is dying, the disk is full, the
        disk is out of inodes, the disk will be full, something just wrote
        a lot.
        """
        mountpoint = sample.mountpoint
        previous_free = self._previous_free(mountpoint)
        self._record(mountpoint, sample, now)
        self._last_seen[mountpoint] = sample

        projection = self._project(mountpoint, sample, now)
        self._projections[mountpoint] = projection

        events: List[Event] = []
        events.extend(self._readonly_rule(sample, now))
        events.extend(self._full_rule(sample, now))
        events.extend(self._inode_rule(sample, now))
        events.extend(self._filling_rule(sample, projection, now))
        events.extend(self._drop_rule(sample, previous_free))
        return events

    # -- history ------------------------------------------------------------

    def _previous_free(self, mountpoint: str) -> Optional[int]:
        history = self._history.get(mountpoint)
        return history[-1][1] if history else None

    def _record(self, mountpoint: str, sample: MountSample, now: float) -> None:
        """Append this reading, resetting the trend after a big delete.

        Two things worth saying out loud:

        * **the reset.** Free space *gaining* more than ``drop_bytes`` in
          one interval means someone (or a render cleanup) deleted a lot.
          The samples before that describe a disk that no longer exists,
          and fitting a line across the step is what produces "fills in
          -4 days".  They are discarded, and the bot says so in the reason
          until it has enough new history to speak again.
        * **the clock not moving.** A replayed round, or two ticks inside
          the same second, must not stack points at one timestamp: the
          newer reading replaces the older, so the fit keeps one reading
          per instant and ``Σ(t - t̄)²`` cannot collapse to zero.
        """
        history = self._history.setdefault(mountpoint, [])
        if history:
            last_t, last_free = history[-1]
            gained = sample.free_bytes - last_free
            if gained >= self.drop_bytes:
                self._reset_notes[mountpoint] = (
                    f"trend reset: {human_bytes(gained)} was freed since the "
                    f"last look"
                )
                history.clear()
            elif now <= last_t:
                history.pop()
        history.append((float(now), int(sample.free_bytes)))
        self._prune(history, float(now))

    def _prune(self, history: List[Tuple[float, int]], now: float) -> None:
        """Keep the history bounded by both age and count, always.

        Applied on every append and on every restore, because the snapshot
        is persisted: an unbounded list here is a state file that grows
        until the disk it is watching is full, which would be a funny way
        to fail.
        """
        cutoff = now - self.history_window_s
        if history and history[0][0] < cutoff:
            kept = [point for point in history if point[0] >= cutoff]
            # Never prune away the newest reading, whatever the clock did.
            history[:] = kept or history[-1:]
        if len(history) > self.max_samples:
            del history[: len(history) - self.max_samples]

    # -- the projection ------------------------------------------------------

    def _project(self, mountpoint: str, sample: MountSample, now: float) -> Projection:
        """Fit the retained history and say what it means, or why it does
        not mean anything yet."""
        history = self._history.get(mountpoint, [])
        samples = len(history)
        span = (history[-1][0] - history[0][0]) if samples >= 2 else 0.0
        prefix = self._reset_notes.get(mountpoint, "")

        def result(
            fill_per_day: float,
            days: Optional[float],
            full_at: Optional[float],
            reason: str,
        ) -> Projection:
            return Projection(
                mountpoint=mountpoint,
                samples=samples,
                span_s=span,
                free_bytes=sample.free_bytes,
                fill_bytes_per_day=fill_per_day,
                days_to_full=days,
                full_at=full_at,
                reason=f"{prefix}; {reason}" if prefix else reason,
            )

        if samples < MIN_SAMPLES:
            return result(
                0.0,
                None,
                None,
                f"not enough history yet: {samples} of the {MIN_SAMPLES} "
                f"readings a trend needs",
            )
        if span < MIN_SPAN_S:
            return result(
                0.0,
                None,
                None,
                f"not enough history yet: {_minutes(span)} of readings, and an "
                f"hour is the minimum before a trend means anything",
            )

        slope = linear_trend(history)
        if slope is None:
            return result(0.0, None, None, "no trend could be fitted to the readings")

        fill_per_day = -slope * 86400.0
        if fill_per_day < MIN_FILL_BYTES_PER_DAY:
            return result(
                fill_per_day,
                None,
                None,
                "free space is steady or growing",
            )

        days = sample.free_bytes / fill_per_day
        if not math.isfinite(days) or days > MAX_PROJECTION_DAYS:
            return result(
                fill_per_day,
                None,
                None,
                f"losing {human_bytes(fill_per_day)}/day, which is further out "
                f"than this bot will guess",
            )
        return result(
            fill_per_day,
            days,
            float(now) + days * 86400.0,
            f"losing {human_bytes(fill_per_day)}/day, full in {_days(days)}",
        )

    # -- the rules ------------------------------------------------------------

    def _filling_rule(
        self, sample: MountSample, projection: Projection, now: float
    ) -> List[Event]:
        """The headline: this fills before you would have noticed.

        Inclusive at the boundary, with :data:`DAYS_EPSILON`, so a disk
        projected to fill in exactly the threshold number of days alerts.
        """
        mountpoint = sample.mountpoint
        days = projection.days_to_full
        active = days is not None and days <= self.projected_days_threshold + DAYS_EPSILON
        text = ""
        if active and days is not None and projection.full_at is not None:
            text = (
                f"{mountpoint} fills in {_days(days)}: losing "
                f"{human_bytes(projection.fill_bytes_per_day)}/day with "
                f"{human_bytes(sample.free_bytes)} free, full around "
                f"{_when(projection.full_at)}"
            )
        return self._standing(
            key=filling_key(mountpoint),
            active=active,
            now=now,
            text=text,
            resolved_text=(
                f"{mountpoint} is no longer projected to fill: {projection.reason}"
            ),
            mountpoint=mountpoint,
            rate_bytes_per_day=projection.fill_bytes_per_day,
            days_to_full=days,
            full_at=projection.full_at,
            free_bytes=sample.free_bytes,
            samples=projection.samples,
            span_s=projection.span_s,
            reason=projection.reason,
        )

    def _full_rule(self, sample: MountSample, now: float) -> List[Event]:
        """The backstop, and a different question from the projection: this
        one is about right now, and it stays open while it is true."""
        mountpoint = sample.mountpoint
        active = sample.at_or_over(self.percent_threshold)
        return self._standing(
            key=full_key(mountpoint),
            active=active,
            now=now,
            text=(
                f"{mountpoint} is {_pct(sample.percent_used)} full: only "
                f"{human_bytes(sample.free_bytes)} free of "
                f"{human_bytes(sample.total_bytes)}"
            ),
            resolved_text=(
                f"{mountpoint} is back under {_pct(self.percent_threshold)}: "
                f"{_pct(sample.percent_used)} used, "
                f"{human_bytes(sample.free_bytes)} free"
            ),
            mountpoint=mountpoint,
            percent_used=sample.percent_used,
            free_bytes=sample.free_bytes,
            total_bytes=sample.total_bytes,
            threshold=self.percent_threshold,
        )

    def _inode_rule(self, sample: MountSample, now: float) -> List[Event]:
        """Inode exhaustion: the same symptom, a different failure.

        A filesystem out of inodes refuses every new file with "no space
        left on device" while ``df`` cheerfully reports half the disk
        free, and the owner spends an hour looking for space that was
        never the problem.  Millions of small render frames is exactly how
        it happens, so it gets its own key and its own sentence.
        """
        mountpoint = sample.mountpoint
        active = sample.inodes_at_or_over(self.percent_threshold)
        return self._standing(
            key=inodes_key(mountpoint),
            active=active,
            now=now,
            text=(
                f"{mountpoint} has used {_pct(sample.percent_inodes_used)} of its "
                f"inodes ({_count(sample.inodes_used)} of "
                f"{_count(sample.inodes_total)}): writes will fail with 'no space "
                f"left on device' while df still shows "
                f"{human_bytes(sample.free_bytes)} free"
            ),
            resolved_text=(
                f"{mountpoint} has inodes again: "
                f"{_pct(sample.percent_inodes_used)} used"
            ),
            mountpoint=mountpoint,
            percent_inodes_used=sample.percent_inodes_used,
            inodes_free=sample.inodes_free,
            inodes_total=sample.inodes_total,
            threshold=self.percent_threshold,
        )

    def _readonly_rule(self, sample: MountSample, now: float) -> List[Event]:
        """A filesystem the kernel remounted read-only after an I/O error.

        Every byte count still looks healthy, which is why this is a rule
        of its own rather than something inferred from the numbers.
        """
        mountpoint = sample.mountpoint
        where = f" ({sample.device})" if sample.device else ""
        return self._standing(
            key=readonly_key(mountpoint),
            active=bool(sample.read_only),
            now=now,
            text=(
                f"{mountpoint}{where} has gone read-only: nothing can write to "
                f"it, and a filesystem that remounts itself read-only is "
                f"usually a failing disk. Check the kernel log before writing "
                f"anything else to it"
            ),
            resolved_text=f"{mountpoint} is writable again",
            mountpoint=mountpoint,
            device=sample.device,
            filesystem=sample.filesystem,
        )

    def _drop_rule(
        self, sample: MountSample, previous_free: Optional[int]
    ) -> List[Event]:
        """Something just wrote a lot.

        A NOTICE and not a request for a decision: on a machine that
        renders video this is usually the machine doing its job, and the
        owner wants it in the feed next to the projection, not in the
        badge.  No attention key -- there is nothing standing to close --
        but a stable ``rule`` in the data.
        """
        if previous_free is None:
            return []
        lost = previous_free - sample.free_bytes
        if lost <= self.drop_bytes:
            return []
        return [
            self.event(
                Severity.NOTICE,
                f"{sample.mountpoint} lost {human_bytes(lost)} since the last "
                f"look: {human_bytes(sample.free_bytes)} free, "
                f"{_pct(sample.percent_used)} used",
                href=INFO.href,
                rule=drop_rule(sample.mountpoint),
                mountpoint=sample.mountpoint,
                lost_bytes=lost,
                free_bytes=sample.free_bytes,
            )
        ]

    # -- the attention lifecycle, in one place --------------------------------

    def _standing(
        self,
        *,
        key: str,
        active: bool,
        now: float,
        text: str,
        resolved_text: str,
        **data: Any,
    ) -> List[Event]:
        """Open a standing request, close it, or stay quiet.

        The three-line version of the convention the whole framework
        agrees on:

        * newly true -> one ``ACTION`` carrying the stable key.  The
          supervisor turns that into the badge item and, if policy says
          so, a push;
        * still true -> nothing.  contracts.py has the badge collapse
          repeats, and re-raising would re-alert every quarter of an hour;
        * no longer true -> the *same* key at ``NOTICE`` with
          ``resolved=True`` (``supervisor.RESOLVED_FLAG``), which is below
          ACTION so ``wants_attention`` is false and the request is closed
          rather than re-opened.
        """
        open_now = key in self._open
        if active and not open_now:
            self._open[key] = float(now)
            return [
                self.event(
                    Severity.ACTION,
                    text,
                    attention_key=key,
                    href=INFO.href,
                    **data,
                )
            ]
        if not active and open_now:
            del self._open[key]
            return [
                self.event(
                    Severity.NOTICE,
                    resolved_text,
                    attention_key=key,
                    href=INFO.href,
                    resolved=True,
                    **data,
                )
            ]
        return []

    def _forget_missing(self, seen: Iterable[str], now: float) -> List[Event]:
        """Close the keys of a mountpoint the probe no longer reports.

        An unmounted disk is not a full disk.  Leaving "``/data`` is 96%
        full" in the badge for a filesystem that is not there is a lie the
        owner can do nothing about, so the keys are resolved and the
        history is dropped; if it comes back, the next ticks say so again
        from fresh readings.
        """
        present = set(seen)
        events: List[Event] = []
        for key in sorted(self._open):
            mountpoint = _mountpoint_of(key)
            if mountpoint is None or mountpoint in present:
                continue
            del self._open[key]
            events.append(
                self.event(
                    Severity.NOTICE,
                    f"{mountpoint} is no longer reported by the disk probe; "
                    f"closing what was open about it",
                    attention_key=key,
                    href=INFO.href,
                    resolved=True,
                    mountpoint=mountpoint,
                )
            )
        for mountpoint in [m for m in self._history if m not in present]:
            self._history.pop(mountpoint, None)
            self._projections.pop(mountpoint, None)
            self._last_seen.pop(mountpoint, None)
        return events

    # -- the probe's own health -----------------------------------------------

    def _probe_failed(self, exc: BaseException, now: float) -> Tuple[Event, ...]:
        """Report a probe that raised, and escalate the third in a row.

        ``ERROR`` for the first failures: a line in the feed, and a push if
        the app's threshold is that low, but nothing standing.  At
        :data:`ESCALATE_AFTER_FAILURES` consecutive failures the bot has
        been blind for three quarters of an hour and that becomes an
        ``ACTION`` on :data:`PROBE_ATTENTION_KEY` -- once, not every tick,
        because the key is what keeps it in the badge.

        Only the exception's type name and message are kept; a traceback in
        an alert is not something anyone reads on a phone.
        """
        self._consecutive_failures += 1
        signature = type(exc).__name__
        self._last_fault = f"{signature}: {exc}".strip()
        failures = self._consecutive_failures

        if failures >= ESCALATE_AFTER_FAILURES:
            escalation = self._standing(
                key=PROBE_ATTENTION_KEY,
                active=True,
                now=now,
                text=(
                    f"Disk watch has failed to read the disks {failures} times "
                    f"in a row ({self._last_fault}). Nothing is watching your "
                    f"free space until this is fixed"
                ),
                resolved_text="Disk watch can read the disks again",
                failures=failures,
                fault=signature,
            )
            if escalation:
                return tuple(escalation)
            # Already escalated and still failing: stay quiet, the badge
            # is already carrying it.
            return ()
        return (
            self.event(
                Severity.ERROR,
                f"Disk watch could not read the disks ({self._last_fault}). "
                f"The next pass will look again",
                href=INFO.href,
                failures=failures,
                fault=signature,
            ),
        )

    def _probe_recovered(self, now: float) -> List[Event]:
        """A successful read after failures: reset the count, close the key."""
        if not self._consecutive_failures:
            return []
        self._consecutive_failures = 0
        self._last_fault = ""
        return self._standing(
            key=PROBE_ATTENTION_KEY,
            active=False,
            now=now,
            text="",
            resolved_text="Disk watch can read the disks again",
        )

    # -- what the launcher renders ---------------------------------------------

    def status(self) -> BotStatus:
        """The card.  Cheap, and it does not raise: the page calls it on
        every load.

        Three figures, which together answer "do I need to go and free up
        space this week":

        * **Tightest** -- the mountpoint with the least room, named, with
          its free space and percentage.  Not an average across disks: the
          one that is about to stop the render is the one that matters.
        * **Watching** -- how many filesystems were read, so a probe that
          quietly lost one is visible.
        * **Fills in** -- the *worst* projection, or ``stable`` when
          nothing is trending down, which is exactly what "no projection"
          should read as on a card.

        IDLE until the first successful read, RUNNING after.  PAUSED and
        QUARANTINED are the supervisor's facts about this bot, not the
        bot's, and ``Supervisor.launcher_state`` overrides the state it
        owns.
        """
        if not self._last_seen:
            return self.idle_status(self._detail())

        tightest = self.tightest()
        worst = self.worst_projection()
        if worst is not None and worst.days_to_full is not None:
            fills = f"{_days(worst.days_to_full)} ({worst.mountpoint})"
        else:
            fills = "stable"

        stats = [
            self.stat(
                "Tightest",
                f"{tightest.mountpoint} {human_bytes(tightest.free_bytes)} free "
                f"({_pct(tightest.percent_used)} used)"
                if tightest is not None
                else "nothing read yet",
            ),
            self.stat("Watching", _plural(len(self._last_seen), "filesystem")),
            self.stat("Fills in", fills),
        ]
        return BotStatus(
            state=BotState.RUNNING,
            stats=tuple(stats),
            detail=self._detail(),
        )

    def _detail(self) -> str:
        if self._consecutive_failures:
            return (
                f"Could not read the disks {_plural(self._consecutive_failures, 'time')} "
                f"in a row: {self._last_fault}"
            )
        if not self._last_seen:
            return "Nothing read yet."
        parts = [
            f"{mountpoint}: {projection.reason}"
            for mountpoint, projection in sorted(self._projections.items())
        ]
        return "; ".join(parts) if parts else "Nothing read yet."

    # -- what a page, a test or the app can ask --------------------------------

    def tightest(self) -> Optional[MountSample]:
        """The watched filesystem with the least free space, or ``None``.

        Least *free bytes*, not the highest percentage: 4% of a 40 TB array
        is more room than 30% of a 20 GB boot disk, and it is the bytes
        that run out.
        """
        if not self._last_seen:
            return None
        return min(
            self._last_seen.values(),
            key=lambda sample: (sample.free_bytes, sample.mountpoint),
        )

    def projections(self) -> Dict[str, Projection]:
        """This tick's projection per mountpoint, including the ones that
        are deliberately empty."""
        return dict(self._projections)

    def worst_projection(self) -> Optional[Projection]:
        """The soonest projected fill, or ``None`` when nothing is trending
        down."""
        trending = [p for p in self._projections.values() if p.days_to_full is not None]
        if not trending:
            return None
        return min(trending, key=lambda p: (p.days_to_full, p.mountpoint))

    def reasons(self) -> Dict[str, str]:
        """Why each mountpoint is or is not being projected, in words.

        This is the "say so rather than guessing" half of the projection
        rule, exposed so the detail page (and the tests) can read the same
        sentence the card shows.
        """
        return {
            mountpoint: projection.reason
            for mountpoint, projection in self._projections.items()
        }

    def open_attention_keys(self) -> List[str]:
        """The keys this bot believes are open, sorted.  The supervisor owns
        the badge; this is the bot's own record, and the two agree because
        the supervisor closes a key on the resolution this bot raises."""
        return sorted(self._open)

    def history(self, mountpoint: str) -> List[Tuple[float, int]]:
        """The retained ``(time, free bytes)`` readings for one mountpoint."""
        return list(self._history.get(mountpoint, ()))

    # -- the framework's optional half ------------------------------------------

    def on_pause(self) -> None:
        """contracts.py: "Paused means paused ... a badge asking you to act
        on something you switched off is a lie."

        The supervisor clears this bot's attention on pause.  If the bot
        kept its own record it would believe those requests were still
        open and would never re-raise them, so a disk that filled while the
        bot was paused would never be mentioned again.  The *history* is
        kept: it is measurement, not attention, and throwing it away would
        cost hours before a projection was possible after a resume.
        """
        self._open.clear()

    def snapshot(self) -> Dict[str, Any]:
        """JSON-able state (contracts.Bot.snapshot).

        The sample history is the interesting part: lose it and the bot
        cannot project anything for the first hour after every restart,
        which is the hour a machine is most likely to be restarted in.  It
        is stored as ``[[time, free bytes], ...]`` -- two numbers, already
        bounded by :meth:`_prune` -- so a state file cannot grow without
        limit.

        The thresholds, the mountpoints and the probe are *configuration*
        handed in at construction, not state, and are deliberately absent:
        a snapshot that pinned them would quietly resurrect yesterday's
        configuration after the owner changed it.
        """
        return {
            "version": SNAPSHOT_VERSION,
            "ticks": self._ticks,
            "last_tick_at": self._last_tick_at,
            "consecutive_failures": self._consecutive_failures,
            "last_fault": self._last_fault,
            "history": {
                mountpoint: [[float(t), int(free)] for t, free in points]
                for mountpoint, points in sorted(self._history.items())
            },
            "open": {key: float(at) for key, at in sorted(self._open.items())},
            "last_seen": {
                mountpoint: _sample_to_obj(sample)
                for mountpoint, sample in sorted(self._last_seen.items())
            },
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Take the state back (contracts.Bot.restore).

        Tolerant, like the scaffold's: a snapshot comes off disk, may have
        been written by an older build, and must never stop the bot
        starting -- a bot that will not start is worse than a bot that has
        forgotten its history.  A snapshot from a *newer* build is the one
        thing refused, because reading half of one would mean a trend
        fitted to points whose meaning has changed.

        Everything restored is re-bounded and re-filtered: the history is
        pruned again (the file may be older than the window, or from a
        build with a larger cap) and mountpoints no longer watched are
        dropped, so editing the config shrinks the state rather than
        leaving orphan histories in it forever.
        """
        if not isinstance(snapshot, Mapping) or not snapshot:
            return
        version = snapshot.get("version", SNAPSHOT_VERSION)
        if isinstance(version, int) and version > SNAPSHOT_VERSION:
            raise DiskBotError(
                f"disk bot snapshot version {version!r} is newer than this build "
                f"understands (version {SNAPSHOT_VERSION})"
            )

        self._ticks = _as_int(snapshot.get("ticks"), self._ticks)
        self._last_tick_at = _as_float(snapshot.get("last_tick_at"), self._last_tick_at)
        self._consecutive_failures = max(
            0, _as_int(snapshot.get("consecutive_failures"), 0)
        )
        fault = snapshot.get("last_fault")
        self._last_fault = fault if isinstance(fault, str) else ""

        history: Dict[str, List[Tuple[float, int]]] = {}
        raw_history = snapshot.get("history")
        if isinstance(raw_history, Mapping):
            for mountpoint, points in raw_history.items():
                name = str(mountpoint)
                if self._watched is not None and name not in self._watched:
                    continue
                kept: List[Tuple[float, int]] = []
                for point in points if isinstance(points, (list, tuple)) else ():
                    if not isinstance(point, (list, tuple)) or len(point) != 2:
                        continue
                    when = _as_float(point[0], float("nan"))
                    free = _as_int(point[1], -1)
                    if free < 0 or when != when:
                        continue
                    kept.append((when, free))
                kept.sort(key=lambda pair: pair[0])
                if kept:
                    self._prune(kept, kept[-1][0])
                    history[name] = kept
        self._history = history

        self._open = {}
        raw_open = snapshot.get("open")
        if isinstance(raw_open, Mapping):
            for key, at in raw_open.items():
                if isinstance(key, str) and key.strip():
                    self._open[key] = _as_float(at, 0.0)

        self._last_seen = {}
        raw_seen = snapshot.get("last_seen")
        if isinstance(raw_seen, Mapping):
            for mountpoint, row in raw_seen.items():
                name = str(mountpoint)
                if self._watched is not None and name not in self._watched:
                    continue
                sample = _sample_from_obj(name, row)
                if sample is not None:
                    self._last_seen[name] = sample


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _mountpoint_of(key: str) -> Optional[str]:
    """The mountpoint a per-mountpoint key names, or ``None``.

    ``disk:filling:/mnt/data`` -> ``/mnt/data``.  The probe-failure key has
    no mountpoint and returns ``None``, which is what keeps it out of
    :meth:`DiskBot._forget_missing`.
    """
    for prefix in ("disk:filling:", "disk:full:", "disk:inodes:", "disk:readonly:"):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return None


def _sample_to_obj(sample: MountSample) -> Dict[str, Any]:
    return {
        "mountpoint": sample.mountpoint,
        "device": sample.device,
        "filesystem": sample.filesystem,
        "total_bytes": sample.total_bytes,
        "free_bytes": sample.free_bytes,
        "used_bytes": sample.used_bytes,
        "inodes_total": sample.inodes_total,
        "inodes_free": sample.inodes_free,
        "read_only": sample.read_only,
    }


def _sample_from_obj(mountpoint: str, row: Any) -> Optional[MountSample]:
    if not isinstance(row, Mapping):
        return None
    try:
        return MountSample(
            mountpoint=mountpoint,
            device=str(row.get("device") or ""),
            filesystem=str(row.get("filesystem") or ""),
            total_bytes=max(0, _as_int(row.get("total_bytes"), 0)),
            free_bytes=max(0, _as_int(row.get("free_bytes"), 0)),
            used_bytes=max(0, _as_int(row.get("used_bytes"), 0)),
            inodes_total=max(0, _as_int(row.get("inodes_total"), 0)),
            inodes_free=max(0, _as_int(row.get("inodes_free"), 0)),
            read_only=bool(row.get("read_only")),
        )
    except (TypeError, ValueError):
        return None


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


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _minutes(seconds: float) -> str:
    return _plural(int(max(0.0, seconds) // 60), "minute")


def build(
    *,
    clock: Callable[[], float],
    probe: Optional[Callable[[], Sequence[MountSample]]] = None,
    mountpoints: Optional[Iterable[str]] = None,
    percent_threshold: float = DEFAULT_PERCENT_THRESHOLD,
    projected_days_threshold: float = DEFAULT_PROJECTED_DAYS,
    drop_bytes: int = DEFAULT_DROP_BYTES,
) -> DiskBot:
    """Construct the bot with everything injected.

    With no ``probe`` the mountpoints are read through :func:`statvfs_probe`
    -- the real thing, still a plain function this module calls rather than
    a hidden dependency -- and ``mountpoints`` is then required: a disk
    watcher that picks its own filesystems would watch the container's
    overlay and tell the owner nothing.

    ``clock`` is required and has no default: a bot that *can* fall back to
    the wall clock eventually will.
    """
    if probe is None:
        if mountpoints is None:
            raise DiskBotError(
                "build() needs either a probe or the mountpoints to watch: "
                "there is no sensible default set of filesystems"
            )
        watched = tuple(str(m) for m in mountpoints)

        def probe() -> List[MountSample]:  # noqa: F811 - the injected default
            return statvfs_probe(watched)

        mountpoints = watched
    return DiskBot(
        clock=clock,
        probe=probe,
        mountpoints=mountpoints,
        percent_threshold=percent_threshold,
        projected_days_threshold=projected_days_threshold,
        drop_bytes=drop_bytes,
    )
