"""Which bots exist, in a fixed order.

Contract: :mod:`jarvis_bots.contracts` -- a bot is identified by
``info.id``, and everything the supervisor keeps (health, attention,
persistence, the launcher card) is filed under that id.  This module is the
one place that decides an id is real and unique.

Explicit registration only
--------------------------
Jarvis knows its own bots.  There is no entry-point scan, no import of
every module in a package, no plugin directory: you can read this file and
a wiring line and know exactly which bots will run.  Dynamic discovery buys
nothing here -- the bots ship with the app -- and costs the two things that
matter for a background scheduler: a startup that fails differently
depending on what is installed, and an order that changes between runs.

Order is registration order, and :meth:`BotRegistry.all` promises it.  The
supervisor ticks in that order, so "a fixed sequence of rounds is
deterministic" starts here; a set or a dict keyed by id sorted
incidentally would make the same config produce a different transcript
after an unrelated rename.

Errors are :class:`~jarvis_bots.contracts.RegistryError`, raised at wiring
time.  A malformed bot is a programming or config mistake, and the moment
to find out is the line that registers it, not the first round at 3am.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

from jarvis_bots.contracts import Bot, BotInfo, RegistryError

__all__ = ["BotRegistry", "check_bot"]


def check_bot(bot: Any) -> BotInfo:
    """Raise :class:`RegistryError` unless ``bot`` can actually be ticked.

    The protocol's required half (contracts.py: "Only ``info``, ``tick`` and
    ``status`` are required") plus the two invariants ``BotInfo`` states
    about itself, re-checked here because a hand-rolled object that never
    ran ``BotInfo.__post_init__`` can still turn up with an ``info``
    attribute.  Returns the validated :class:`BotInfo`.
    """
    info = getattr(bot, "info", None)
    if info is None:
        raise RegistryError(f"{bot!r} has no info: a bot declares a BotInfo")
    for attribute in ("id", "name", "interval_s"):
        if not hasattr(info, attribute):
            raise RegistryError(
                f"{bot!r}.info is not a BotInfo (no {attribute!r}): got "
                f"{type(info).__name__}"
            )
    bot_id = info.id
    if not isinstance(bot_id, str) or not bot_id:
        raise RegistryError(f"bot id must be a non-empty string; got {bot_id!r}")
    # The same rule BotInfo.__post_init__ applies, restated so an object that
    # skipped it cannot smuggle a "poke/../etc" through: ids end up in state
    # keys, attention keys and urls.
    if not bot_id.replace("_", "").replace("-", "").isalnum():
        raise RegistryError(f"bot id must be a simple slug: {bot_id!r}")
    if not isinstance(info.interval_s, (int, float)) or isinstance(info.interval_s, bool):
        raise RegistryError(
            f"{bot_id!r}: interval_s must be a number; got {info.interval_s!r}"
        )
    if float(info.interval_s) < 5.0:
        raise RegistryError(f"{bot_id!r}: interval_s floor is 5s; got {info.interval_s}")
    for method in ("tick", "status"):
        if not callable(getattr(bot, method, None)):
            raise RegistryError(f"{bot_id!r} has no {method}(): see contracts.Bot")
    return info


class BotRegistry:
    """The bots Jarvis will run, in the order they were registered.

    Construct empty and :meth:`register`, or hand the constructor an
    iterable of bots for a one-liner.  The registry holds no state beyond
    membership: health, pausing and attention belong to the supervisor, so
    a registry can be rebuilt from config without losing any of it.
    """

    def __init__(self, bots: Iterable[Bot] = ()) -> None:
        #: id -> bot, in registration order (dicts preserve insertion order,
        #: which is exactly the guarantee all() needs).
        self._bots: Dict[str, Bot] = {}
        for bot in bots:
            self.register(bot)

    # -- membership ----------------------------------------------------------

    def register(self, bot: Bot) -> Bot:
        """Add one bot.  Returns it, so wiring can be a single expression.

        Raises :class:`RegistryError` for a malformed bot (see
        :func:`check_bot`) and for a duplicate id.  A duplicate is never a
        harmless overwrite: two bots under one id would share one health
        record, one snapshot and one set of attention items, and the second
        would silently inherit the first's state.
        """
        info = check_bot(bot)
        if info.id in self._bots:
            raise RegistryError(
                f"duplicate bot id {info.id!r}: already registered as "
                f"{type(self._bots[info.id]).__name__}"
            )
        self._bots[info.id] = bot
        return bot

    def register_from(self, mapping: Mapping[str, Any]) -> List[Bot]:
        """Register a config-shaped ``{id: bot}`` mapping, all or nothing.

        A value may be the bot itself or a zero-argument factory returning
        one, which is what a config that builds bots lazily hands over; the
        two are told apart by whether the value carries an ``info``.

        The key must equal the bot's own ``info.id``.  A config that calls
        the weather bot "wether" and a bot that calls itself "weather" is a
        typo, and the failure it causes otherwise -- a pause button that
        pauses nothing -- is invisible until someone needs it.

        Nothing is registered if any entry is bad: a half-wired registry
        would start a round with some of the owner's bots missing and no
        indication which.
        """
        if not isinstance(mapping, Mapping):
            raise RegistryError(
                f"register_from takes a mapping of id -> bot; got "
                f"{type(mapping).__name__}"
            )
        staged: List[Bot] = []
        for key, value in mapping.items():
            bot = value
            if getattr(value, "info", None) is None and callable(value):
                try:
                    bot = value()
                except Exception as exc:  # a factory that cannot build its bot
                    raise RegistryError(f"{key!r}: factory raised {exc!r}") from exc
            info = check_bot(bot)
            if not isinstance(key, str) or key != info.id:
                raise RegistryError(
                    f"config key {key!r} does not match the bot's own id "
                    f"{info.id!r}; the id is what pause, health and state are "
                    f"filed under"
                )
            if info.id in self._bots or any(b.info.id == info.id for b in staged):
                raise RegistryError(f"duplicate bot id {info.id!r}")
            staged.append(bot)
        for bot in staged:
            self._bots[bot.info.id] = bot
        return staged

    def unregister(self, bot_id: str) -> Bot:
        """Remove a bot and return it.  Raises :class:`RegistryError` if it
        was never registered, so a typo in a teardown is not a silent no-op.

        The supervisor's health and attention for that id are the
        supervisor's to drop (:meth:`Supervisor.forget`); a registry that
        reached into them would make "rebuild the registry from config"
        destructive.
        """
        try:
            return self._bots.pop(bot_id)
        except KeyError:
            raise RegistryError(f"no bot registered as {bot_id!r}") from None

    # -- lookup --------------------------------------------------------------

    def get(self, bot_id: str) -> Bot:
        """The bot with this id.  Raises :class:`RegistryError` if unknown --
        an unknown id from a pause endpoint or a restored state file is a
        mistake worth naming, not a ``None`` to trip over later."""
        try:
            return self._bots[bot_id]
        except KeyError:
            raise RegistryError(
                f"no bot registered as {bot_id!r}; registered: "
                f"{', '.join(self._bots) or '(none)'}"
            ) from None

    def find(self, bot_id: str) -> Optional[Bot]:
        """The bot with this id, or ``None``.  For the callers that are
        legitimately asking a question rather than making a mistake."""
        return self._bots.get(bot_id)

    def all(self) -> List[Bot]:
        """Every bot, in registration order.  A fresh list: a caller
        iterating it may pause or unregister without mutating what it is
        walking."""
        return list(self._bots.values())

    def ids(self) -> List[str]:
        """Every id, in registration order."""
        return list(self._bots)

    def __contains__(self, bot_id: object) -> bool:
        return bot_id in self._bots

    def __iter__(self) -> Iterator[Bot]:
        return iter(self.all())

    def __len__(self) -> int:
        return len(self._bots)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<BotRegistry {len(self._bots)}: {', '.join(self._bots) or '(empty)'}>"
