"""Everything a bot does *not* have to write.

Contract: :mod:`jarvis_bots.contracts` -- ":class:`Bot`: ... Only ``info``,
``tick`` and ``status`` are required; the rest have usable defaults in
:class:`jarvis_bots.base.BaseBot`."  This module is those defaults, and the
two helpers that make writing ``tick`` and ``status`` a few lines each.

A second bot should be a class and a page, not a refactor, so the whole of
a new bot is::

    class WeatherBot(BaseBot):
        info = BotInfo(id="weather", name="Weather", blurb="Rain before you leave.",
                       kind="radar", interval_s=900.0, href="/bots/weather")

        def __init__(self, clock, fetch):
            super().__init__(clock)
            self._fetch = fetch          # injected: this package opens no sockets
            self._last = ""

        def tick(self, now):
            reading = self._fetch()
            self._last = reading
            if reading == "rain":
                return [self.event(Severity.ACTION, "Rain in an hour; take the coat.",
                                   attention_key="rain", href="/bots/weather")]
            return []

        def status(self):
            return BotStatus(BotState.RUNNING, (self.stat("Last reading", self._last),))

Everything else -- when it is ticked, what happens when it raises, whether
the owner gets a push, what the badge says -- is
:class:`jarvis_bots.supervisor.Supervisor`'s job, written once.

The rules this file holds up
----------------------------
* **"Events are the only output."**  :meth:`BaseBot.event` builds an
  :class:`~jarvis_bots.contracts.Event`; there is deliberately no way to
  send an alert from here.  A bot that concludes the owner should buy
  something raises an event carrying a link, and a person acts on it.
* **"Time is injected everywhere. Nothing here calls time.time()."**  The
  clock is a constructor argument and :meth:`BaseBot.event` stamps ``at``
  from it.
* **"State is the bot's, persistence is ours."**  :meth:`snapshot` returns
  ``{}`` and :meth:`restore` ignores it, which is the correct behaviour for
  a stateless bot; a bot with state overrides both.

Money: a :class:`~jarvis_bots.contracts.Stat` value is a *pre-formatted
string*.  :meth:`BaseBot.stat` never does arithmetic on it, so the "money is
integer cents, never a float" rule is kept where the cents are -- format
with the owning package's formatter (``jarvis_poke.contracts.fmt_cents``)
and hand the result over as text.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Union

from jarvis_bots.contracts import (
    Bot,
    BotInfo,
    BotState,
    BotStatus,
    Event,
    Severity,
    Stat,
)

__all__ = ["BaseBot", "Clock", "parse_severity"]

#: A callable returning unix seconds.  Injected, always: contracts.py's
#: "Time is injected everywhere."
Clock = Callable[[], float]

_SEVERITY_NAMES: Dict[str, Severity] = {s.name.lower(): s for s in Severity}


def parse_severity(value: Union[Severity, int, str]) -> Severity:
    """Accept a :class:`~jarvis_bots.contracts.Severity`, its int value, or
    its name ("action", case-insensitive).

    Config files spell severities as words; the supervisor's threshold and a
    bot's own events should not disagree about what "action" means.  Raises
    ``ValueError`` rather than guessing, because guessing low would mean a
    push that never arrives and guessing high a phone that buzzes at DEBUG.
    """
    if isinstance(value, Severity):
        return value
    if isinstance(value, bool):
        raise ValueError(f"not a severity: {value!r}")
    if isinstance(value, int):
        try:
            return Severity(value)
        except ValueError:
            raise ValueError(f"not a severity: {value!r}") from None
    if isinstance(value, str):
        try:
            return _SEVERITY_NAMES[value.strip().lower()]
        except KeyError:
            raise ValueError(f"not a severity: {value!r}") from None
    raise ValueError(f"not a severity: {value!r}")


class BaseBot:
    """The optional half of the :class:`~jarvis_bots.contracts.Bot` protocol.

    Subclass it and write ``info``, :meth:`tick` and :meth:`status`.

    ``clock``  a callable returning unix seconds.  Required: contracts.py
               says nothing here calls ``time.time()``, and a bot that reads
               the real clock cannot be tested against a fixed sequence of
               rounds.
    ``info``   normally a class attribute (it is static identity, "declared
               once").  The constructor also accepts it, for a bot whose
               identity comes from config -- one class, several instances.

    :meth:`tick` and :meth:`status` raise ``NotImplementedError`` here on
    purpose: they are the two things the protocol requires, so a subclass
    that forgets one fails loudly at the first round rather than quietly
    rendering an empty card.
    """

    #: Static identity.  Subclasses set this; see :class:`BotInfo`.
    info: BotInfo

    def __init__(self, clock: Clock, info: Optional[BotInfo] = None) -> None:
        if not callable(clock):
            raise ValueError(
                f"{type(self).__name__} needs an injected clock: a callable "
                f"returning unix seconds (contracts.py: nothing here calls "
                f"time.time()); got {clock!r}"
            )
        if info is not None:
            if not isinstance(info, BotInfo):
                raise ValueError(f"info must be a BotInfo; got {type(info).__name__}")
            self.info = info
        if not isinstance(getattr(self, "info", None), BotInfo):
            raise ValueError(
                f"{type(self).__name__} has no info: set a BotInfo as a class "
                f"attribute or pass info= to __init__"
            )
        self._clock = clock

    # -- identity -----------------------------------------------------------

    @property
    def id(self) -> str:
        """Shorthand for ``self.info.id``, which is also the key the
        supervisor's health, attention and persistence are filed under."""
        return self.info.id

    @property
    def clock(self) -> Clock:
        """The injected clock, for a subclass that needs to hand it on (to a
        store, a bridge, or a nested helper) rather than read it."""
        return self._clock

    def now(self) -> float:
        """The current time, from the injected clock.

        ``tick`` is handed ``now`` and should use that argument -- every bot
        in a round then agrees on when the round was.  This is for the places
        outside a tick (``status``, a restore) that still need a timestamp.
        """
        return float(self._clock())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} {self.info.id!r}>"

    # -- the required half, deliberately unimplemented -----------------------

    def tick(self, now: float) -> Sequence[Event]:
        """Do one unit of work and return what was noticed.

        contracts.py: "Must return promptly and must not sleep.  Raising is
        allowed and is handled."
        """
        raise NotImplementedError(f"{type(self).__name__} must implement tick(now)")

    def status(self) -> BotStatus:
        """What the launcher card shows.  Must not raise and must be cheap:
        the page calls it on every load."""
        raise NotImplementedError(f"{type(self).__name__} must implement status()")

    # -- the optional half: usable defaults ----------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """JSON-able state to persist.  ``{}`` -- the right answer for a
        stateless bot (contracts.py: "Return {} if the bot is stateless")."""
        return {}

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Take back what :meth:`snapshot` returned.  A no-op by default, to
        pair with the default ``{}``."""
        return None

    def on_pause(self) -> None:
        """Called once when the owner pauses this bot.  A no-op by default.

        contracts.py: "Paused means paused."  A bot that holds an open
        connection or a pending job drops it here; it will not be ticked
        again until :meth:`on_resume`.
        """
        return None

    def on_resume(self) -> None:
        """Called once when the owner resumes this bot.  A no-op by default."""
        return None

    # -- helpers, so a tick is a few lines ------------------------------------

    def event(
        self,
        severity: Union[Severity, int, str],
        text: str,
        attention_key: Optional[str] = None,
        href: str = "",
        **data: Any,
    ) -> Event:
        """Build one :class:`~jarvis_bots.contracts.Event`, stamped with this
        bot's id and the injected clock.

        ``attention_key`` makes the event a *standing request for a
        decision*: contracts.py has the badge count distinct open keys, "so
        one restock nagging across ten ticks is one item of attention, not
        ten".  Use the same key every time the same open question is still
        open, and a different key for a different question.  An event at
        ACTION or above without a key is a one-off notice: it can earn an
        alert but never sits in the badge, which is why the key is checked
        to be a non-empty string rather than accidentally ``""``.

        ``href`` is where tapping lands, relative to the app; keyword
        arguments become the event's ``data``.
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("an event needs text: it is what the owner reads")
        if attention_key is not None and (
            not isinstance(attention_key, str) or not attention_key.strip()
        ):
            raise ValueError(
                f"attention_key must be a non-empty string or None; got "
                f"{attention_key!r} -- an empty key would collapse two "
                f"unrelated requests for a decision into one badge item"
            )
        if not isinstance(href, str):
            raise ValueError(f"href must be a string; got {type(href).__name__}")
        return Event(
            bot_id=self.info.id,
            at=float(self._clock()),
            severity=parse_severity(severity),
            text=text,
            attention_key=attention_key,
            href=href,
            data=dict(data),
        )

    def stat(self, label: str, value: Any) -> Stat:
        """One figure for the launcher card.

        contracts.py: "Pre-formatted: the bot knows how its own numbers
        should read, and the page should not be doing money maths."  A
        non-string ``value`` is rendered with ``str``; money must arrive
        already formatted from integer cents, because this helper cannot
        know the currency and must never divide by 100 on a float.
        """
        if not isinstance(label, str) or not label.strip():
            raise ValueError("a stat needs a label")
        return Stat(label=label, value=value if isinstance(value, str) else str(value))

    def idle_status(self, detail: str = "") -> BotStatus:
        """``BotStatus.idle`` with this bot's stats left off -- the honest
        card for a bot that has been registered but has not yet run."""
        return BotStatus(BotState.IDLE, detail=detail)


def _protocol_check(bot: Bot) -> Bot:  # pragma: no cover - typing only
    """Static proof that :class:`BaseBot` satisfies the protocol."""
    return bot
