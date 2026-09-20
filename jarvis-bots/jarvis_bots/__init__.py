"""Jarvis's background helpers: the framework, and the bots that run on it.

:mod:`jarvis_bots.contracts` opens by promising that "adding the second bot
should be a class and a page, not a refactor".  This file is the front door
that makes the promise cheap to take up: one import gives a composition root
everything it needs to wire a bot, and nothing it does not.

    from jarvis_bots import BaseBot, BotInfo, BotRegistry, Severity, Supervisor

What is re-exported, and what is not
------------------------------------
Exported: the vocabulary (:mod:`~jarvis_bots.contracts`), the base class
(:mod:`~jarvis_bots.base`), the registry, the supervisor and its store, and
:func:`~jarvis_bots.scaffold.new_bot`.  Those five modules are what an app
touches.

*Not* exported, and deliberately: :mod:`jarvis_bots.cli` and
:mod:`jarvis_bots.validate`.  The CLI is the composition root -- "the one
place allowed to hand ``time.time`` to anything" -- so importing this package
must not drag a wall clock into scope; and the gate builds a synthetic fleet
that nothing in production needs.  Both remain perfectly importable by their
own names::

    python3 -m jarvis_bots.cli new-bot ...
    from jarvis_bots.validate import run_gate

Nothing shadows a submodule
---------------------------
No name bound here is the name of a module or subpackage of this package --
``base``, ``bots``, ``cli``, ``contracts``, ``registry``, ``scaffold``,
``supervisor``, ``templates``, ``validate``, ``web``.  If it were, then after
``import jarvis_bots.registry`` the attribute ``jarvis_bots.registry`` would
be a class or a function on one import path and the module on another,
depending on what had been imported first, and ``from jarvis_bots import
registry`` would hand back different objects in different processes.
:data:`SUBMODULES` records the reserved names and the check at the bottom of
this file fails the import if a future edit collides with one, because an
import-order-dependent bug is the kind that only shows up in production.

The boundary
------------
This package schedules bots and surfaces what they find.  Nothing in it
buys, pays or transacts, and no bot can make it: the widest thing a tick can
do is return an :class:`Event` with an ``href``, which becomes a
notification a person taps.  The purchase happens on the seller's own page,
by hand.  See :mod:`jarvis_bots.supervisor`, "The boundary".

No module reachable from here opens a socket, reads the wall clock or draws
randomness: fetchers, senders and clocks are injected, and the only
permitted entropy is a :mod:`lucifer_gen.seed` stream, drawn in the gate.
"""

from __future__ import annotations

from jarvis_bots.base import BaseBot, Clock, parse_severity
from jarvis_bots.contracts import (
    BACKOFF_CAP_S,
    BACKOFF_FACTOR,
    QUARANTINE_AFTER_FAILURES,
    QUARANTINE_S,
    SLOW_TICK_S,
    AttentionItem,
    Bot,
    BotInfo,
    BotState,
    BotStatus,
    Event,
    Health,
    RegistryError,
    RoundReport,
    Severity,
    Stat,
    backoff_interval,
)
from jarvis_bots.registry import BotRegistry, check_bot
from jarvis_bots.scaffold import ScaffoldError, new_bot
from jarvis_bots.supervisor import (
    ALERT_KIND,
    DEFAULT_PROFILE_ID,
    EVENT_HISTORY,
    STATE_KEY,
    STATE_VERSION,
    JsonFileStore,
    Supervisor,
    SupervisorError,
)

#: Every module and subpackage of this package.  A name bound in this file
#: may not be one of these; see the module docstring and the check below.
SUBMODULES = frozenset(
    {
        "api",
        "base",
        "bots",
        "cli",
        "contracts",
        "registry",
        "scaffold",
        "supervisor",
        "templates",
        "validate",
        "web",
    }
)

__all__ = [
    # the vocabulary (contracts)
    "AttentionItem",
    "Bot",
    "BotInfo",
    "BotState",
    "BotStatus",
    "Event",
    "Health",
    "RegistryError",
    "RoundReport",
    "Severity",
    "Stat",
    "backoff_interval",
    "BACKOFF_CAP_S",
    "BACKOFF_FACTOR",
    "QUARANTINE_AFTER_FAILURES",
    "QUARANTINE_S",
    "SLOW_TICK_S",
    # writing a bot
    "BaseBot",
    "Clock",
    "parse_severity",
    # wiring it
    "BotRegistry",
    "check_bot",
    # running it
    "Supervisor",
    "SupervisorError",
    "JsonFileStore",
    "ALERT_KIND",
    "DEFAULT_PROFILE_ID",
    "EVENT_HISTORY",
    "STATE_KEY",
    "STATE_VERSION",
    # generating the next one
    "new_bot",
    "ScaffoldError",
    # this file
    "SUBMODULES",
]

_collisions = sorted(SUBMODULES.intersection(__all__))
if _collisions:  # pragma: no cover - a guard against a future edit
    raise ImportError(
        "jarvis_bots/__init__.py would shadow the submodule(s) "
        + ", ".join(_collisions)
        + ": after 'import jarvis_bots.<name>' the attribute would be the "
        "module on one import path and this object on another. Rename the "
        "export or reach it through its module."
    )
del _collisions
