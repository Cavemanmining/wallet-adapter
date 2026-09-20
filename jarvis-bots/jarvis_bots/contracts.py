"""The bot framework Jarvis builds on.

A "bot" here is a long-lived background helper that watches something and
tells the owner when to act. The Pokemon buying assistant is the first one;
the point of this module is that it is not special. Adding the second bot
should be a class and a page, not a refactor.

What a bot has to provide
-------------------------
Implement :class:`Bot`: an identity, a :meth:`Bot.tick` that does one unit of
work and returns events, and a :meth:`Bot.status` the launcher can render.
Everything else, scheduling, failure isolation, health, persistence, the
badge count, alert delivery, is the supervisor's job and is written once.

The rules the supervisor enforces for you
-----------------------------------------
* **A bot never blocks another.** Ticks are isolated; an exception is caught,
  recorded against that bot's health, and the round continues.
* **A sick bot backs off.** Consecutive failures widen the interval and then
  quarantine the bot, so a broken bot degrades instead of hammering.
* **Paused means paused.** A paused bot is not ticked and reports no
  attention, because a badge asking you to act on something you switched off
  is a lie.
* **Events are the only output.** A bot does not send alerts itself; it
  returns events and the supervisor decides what is worth a push. That keeps
  alert policy in one place.
* **State is the bot's, persistence is ours.** A bot hands over a JSON-able
  snapshot and gets it back on the next start.

Time is injected everywhere. Nothing here calls time.time().
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple


class BotState(enum.Enum):
    IDLE = "idle"              # registered, nothing to do yet
    RUNNING = "running"
    PAUSED = "paused"          # by the owner
    QUARANTINED = "error"      # by the supervisor, after repeated failures

    @property
    def ui(self) -> str:
        """The value the launcher page expects."""
        return self.value


class Severity(enum.IntEnum):
    DEBUG = 0
    INFO = 1
    NOTICE = 2      # worth a line in the feed
    ACTION = 3      # the owner should decide something: this drives the badge
    ERROR = 4


@dataclass(frozen=True)
class Event:
    """Something a bot noticed. The supervisor logs every event and alerts on
    the ones whose severity earns it."""

    bot_id: str
    at: float
    severity: Severity
    text: str
    #: Set when the event is a standing request for a decision. The badge
    #: counts distinct open keys, so one restock nagging across ten ticks is
    #: one item of attention, not ten.
    attention_key: Optional[str] = None
    #: Where tapping the alert should land, relative to the app.
    href: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the four fields the supervisor reads without a net.

        An ``Event`` is the one object a bot hands the framework, and it is
        exported and documented as such, so a bot builds them directly.
        Until this existed, nothing checked them: an event carrying a stale
        integer where a :class:`Severity` belongs blew up in
        :attr:`wants_attention`, and one whose ``attention_key`` was not a
        string blew up in the supervisor's ``str(key)`` -- both *outside*
        the try/except that isolates a tick, so one hand-built event stopped
        every bot in the fleet for ever.

        Validating here means the mistake surfaces where it is made, on the
        line that built the event, and (because a bot builds its events
        inside ``tick``) as an ordinary tick failure that the supervisor
        records and backs off from.  The supervisor re-checks anyway: an
        object can still reach it through ``dataclasses.replace`` of a
        mutated instance or ``object.__new__``, and the round is not a place
        to be trusting.

        ``severity`` given as a plain int is coerced, because that is a
        config file's spelling and not a mistake; everything else raises.
        """
        if not isinstance(self.bot_id, str) or not self.bot_id:
            raise ValueError(f"event bot_id must be a non-empty string; got {self.bot_id!r}")
        try:
            at = float(self.at)
        except (TypeError, ValueError):
            raise ValueError(f"event at must be unix seconds; got {self.at!r}") from None
        if at != at:  # NaN: every schedule comparison against it is false
            raise ValueError("event at must be a real number of unix seconds, not NaN")
        object.__setattr__(self, "at", at)
        severity = self.severity
        if not isinstance(severity, Severity):
            if isinstance(severity, bool) or not isinstance(severity, int):
                raise TypeError(
                    f"event severity must be a Severity; got "
                    f"{type(severity).__name__}"
                )
            try:
                severity = Severity(severity)
            except ValueError:
                raise ValueError(f"not a severity: {self.severity!r}") from None
            object.__setattr__(self, "severity", severity)
        if not isinstance(self.text, str):
            raise TypeError(f"event text must be a string; got {type(self.text).__name__}")
        if self.attention_key is not None and (
            not isinstance(self.attention_key, str) or not self.attention_key.strip()
        ):
            raise ValueError(
                f"attention_key must be a non-empty string or None; got "
                f"{self.attention_key!r} -- the badge keys on it, and a key that "
                f"is not a string collapses two unrelated requests into one item "
                f"or crashes the round"
            )
        if not isinstance(self.href, str):
            raise TypeError(f"event href must be a string; got {type(self.href).__name__}")
        if not isinstance(self.data, dict):
            raise TypeError(f"event data must be a dict; got {type(self.data).__name__}")

    @property
    def wants_attention(self) -> bool:
        return self.severity >= Severity.ACTION and self.attention_key is not None


@dataclass(frozen=True)
class Stat:
    """One figure on the bot's launcher card. Pre-formatted: the bot knows how
    its own numbers should read, and the page should not be doing money maths."""

    label: str
    value: str


@dataclass(frozen=True)
class BotStatus:
    """Everything the launcher and the badge need, and nothing else."""

    state: BotState
    stats: Tuple[Stat, ...] = ()
    detail: str = ""

    @staticmethod
    def idle(detail: str = "") -> "BotStatus":
        return BotStatus(BotState.IDLE, detail=detail)


@dataclass(frozen=True)
class BotInfo:
    """Static identity, declared once."""

    id: str
    name: str
    blurb: str
    #: Picks the launcher icon: cart, grid, radar, or anything else for the
    #: generic glyph.
    kind: str = "bot"
    #: Seconds between ticks when healthy. The supervisor may widen this when
    #: the bot is failing, never narrow it.
    interval_s: float = 300.0
    #: Route the launcher's Open button points at.
    href: str = ""
    can_pause: bool = True

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("_", "").replace("-", "").isalnum():
            raise ValueError(f"bot id must be a simple slug: {self.id!r}")
        if self.interval_s < 5.0:
            raise ValueError("interval_s floor is 5s")


class Bot(Protocol):
    """What Jarvis implements to add a helper.

    Only ``info``, ``tick`` and ``status`` are required; the rest have usable
    defaults in :class:`jarvis_bots.base.BaseBot`.
    """

    info: BotInfo

    def tick(self, now: float) -> Sequence[Event]:
        """Do one unit of work. Must return promptly and must not sleep.

        Raising is allowed and is handled: the supervisor records it against
        this bot's health and carries on with the others.
        """
        ...

    def status(self) -> BotStatus: ...

    def snapshot(self) -> Dict[str, Any]:
        """JSON-able state to persist. Return {} if the bot is stateless."""
        ...

    def restore(self, snapshot: Dict[str, Any]) -> None: ...

    def on_pause(self) -> None: ...

    def on_resume(self) -> None: ...


# --------------------------------------------------------------------------
# Supervisor bookkeeping
# --------------------------------------------------------------------------


@dataclass
class Health:
    """Per-bot running health. Owned by the supervisor, not the bot."""

    consecutive_failures: int = 0
    total_failures: int = 0
    total_ticks: int = 0
    last_tick_at: float = 0.0
    last_ok_at: float = 0.0
    last_error: str = ""
    next_due_at: float = 0.0
    quarantined_until: float = 0.0

    def healthy(self) -> bool:
        return self.consecutive_failures == 0


#: Consecutive tick failures before a bot is quarantined.
QUARANTINE_AFTER_FAILURES = 4
#: How long a quarantine lasts before one probe tick is allowed through.
QUARANTINE_S = 1800.0
#: Multiplier applied per consecutive failure, capped, so a failing bot slows
#: down instead of retrying at full rate.
BACKOFF_FACTOR = 2.0
BACKOFF_CAP_S = 3600.0
#: The highest exponent :func:`backoff_interval` will raise
#: :data:`BACKOFF_FACTOR` to.  ``2.0 ** 1024`` raises ``OverflowError``, and
#: nothing resets ``consecutive_failures`` while a bot is quarantined, so a
#: watcher pointed at a retired endpoint reaches that exponent after about
#: three weeks of probes -- inside the supervisor's failure handler, where
#: the overflow takes the whole round with it for ever.  The clamp costs
#: nothing: ``BACKOFF_FACTOR ** 32`` already exceeds :data:`BACKOFF_CAP_S`
#: for any base above a microsecond, so every widened interval this can
#: produce is the cap anyway.
BACKOFF_MAX_EXPONENT = 32
#: A tick that runs longer than this is reported as an anomaly; the framework
#: cannot kill it, but a bot hogging the round should be visible.
SLOW_TICK_S = 10.0

#: The most events one tick may return.  ``tick`` is typed as returning a
#: ``Sequence[Event]``, but a generator is also iterable, and a tick that
#: returns an endless one hangs the round in the very loop written to
#: validate it -- no exception is raised, so no try/except can help.  The
#: supervisor stops reading at this many and records the tick as failed.
#: A bot with more than a thousand things to say in one tick has a bug, and
#: the bug should back it off rather than fill the badge.
MAX_EVENTS_PER_TICK = 1000

#: The most distinct open requests for a decision one bot may hold.  The
#: badge "counts distinct open keys"; a bot that mints a new key every tick
#: would make that count grow for as long as the process runs, and a badge
#: reading 40000 is not a badge.  Keys past this are refused and reported
#: through ``Supervisor.last_overflow_error``, never silently swallowed.
MAX_ATTENTION_PER_BOT = 200


def backoff_interval(base_s: float, consecutive_failures: int) -> float:
    """Widen the interval for a failing bot. Never narrower than base.

    Two things this function must not do, because it is called from inside
    the supervisor's failure handler, where raising aborts the whole round:

    * **overflow.** The exponent is clamped to
      :data:`BACKOFF_MAX_EXPONENT`; without it a bot whose endpoint stays
      down long enough reaches ``2.0 ** 1024`` and every round after that
      dies in the code that exists to make a broken bot harmless.
    * **narrow.** The floor is applied *last*, so the promise in the first
      line holds for every base, including one above
      :data:`BACKOFF_CAP_S`.  Applying the cap last (as this did) returned
      3600 for a two-hour bot that had failed once, which is the hammering
      "a sick bot backs off" exists to prevent.
    """
    if consecutive_failures <= 0:
        return base_s
    exponent = min(int(consecutive_failures), BACKOFF_MAX_EXPONENT)
    widened = base_s * (BACKOFF_FACTOR ** exponent)
    return max(base_s, min(BACKOFF_CAP_S, widened))


@dataclass
class RoundReport:
    """What one supervisor pass did. Returned so the caller can log or test."""

    at: float = 0.0
    ticked: int = 0
    skipped: int = 0
    failed: int = 0
    quarantined: int = 0
    events: int = 0
    alerts: int = 0
    slow: Tuple[str, ...] = ()


@dataclass(frozen=True)
class AttentionItem:
    """An open request for a decision, keyed so repeats collapse."""

    bot_id: str
    key: str
    text: str
    href: str
    since: float


class RegistryError(Exception):
    pass


__all__ = [
    "BotState", "Severity", "Event", "Stat", "BotStatus", "BotInfo", "Bot",
    "Health", "QUARANTINE_AFTER_FAILURES", "QUARANTINE_S", "BACKOFF_FACTOR",
    "BACKOFF_CAP_S", "BACKOFF_MAX_EXPONENT", "MAX_EVENTS_PER_TICK",
    "MAX_ATTENTION_PER_BOT", "SLOW_TICK_S", "backoff_interval", "RoundReport",
    "AttentionItem", "RegistryError",
]
