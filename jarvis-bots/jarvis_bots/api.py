"""The four payloads the pages read, and nothing else.

``jarvis_bots/web/README.md`` documents three endpoints and
``jarvis_bots/templates/page.html.tmpl`` a fourth.  This module is the one
place that turns a :class:`~jarvis_bots.supervisor.Supervisor` into the
objects behind them, so an app can wire a route in a line and a second bot
author never has to guess a key name::

    from jarvis_bots import api

    GET  /api/bots/        -> api.launcher_state(supervisor)
    GET  /api/bots/status  -> api.badge_status(supervisor)
    GET  /api/bots/<id>    -> api.bot_detail(supervisor, bot_id)
    POST /api/bots/pause   -> api.set_paused(supervisor, request.json)

Why this exists as a module
---------------------------
``jarvis_bots/cli.py`` has named ``jarvis_bots.api`` since it was written
("``jarvis_bots.supervisor``, ``jarvis_bots.registry``, ``jarvis_bots.api``
and ``jarvis_bots.validate`` are written alongside this file") and binds to
it by shape, falling back to the supervisor's own methods when it is
absent.  The fallback worked, so the missing module was invisible -- and
so was the hole underneath it: the *detail* payload had no producer at
all, because the supervisor kept one event per bot and the generated page
renders a feed of twenty.  A scaffold whose docstring says "Nothing is a
stub" has to be able to feed what it writes.

No framework, no routing, no I/O
--------------------------------
Every function here is ``(supervisor, ...) -> a JSON-able dict``.  Nothing
imports a web framework, opens a socket or reads a clock: ``now`` is an
argument, defaulted from the supervisor's own injected clock, exactly as
everything else in this package takes time as an argument.  Serialising and
status codes are the app's -- the one thing this module does say about
them is which error means 404 (:class:`ApiError` carries ``status``).

The boundary
------------
Three reads and one switch.  Nothing here buys, pays or transacts, and the
pause endpoint is the only thing that changes any state at all.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from jarvis_bots.contracts import RegistryError
from jarvis_bots.supervisor import EVENT_HISTORY, Supervisor, SupervisorError

__all__ = [
    "ApiError",
    "badge_status",
    "bot_detail",
    "launcher_state",
    "set_paused",
]


class ApiError(Exception):
    """A request the app should turn into a status code, not a traceback.

    ``status`` is 404 for a bot id that is not registered and 400 for a
    body this module cannot read.  A 404 rather than an empty page matters:
    a detail page for a bot that does not exist should be a dead link, not
    a bot that looks like it has nothing to say.
    """

    def __init__(self, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.status = int(status)


def launcher_state(
    supervisor: Supervisor,
    now: Optional[float] = None,
    *,
    include_detail: bool = False,
) -> Dict[str, Any]:
    """``GET /api/bots/`` -- the object ``web/bots.html`` renders.

    ``include_detail`` adds the keys web/README.md does not document
    (``detail``, and ``severity``/``href`` on the last event); leave it off
    unless the page asking for it knows about them.
    """
    return supervisor.launcher_state(now, include_detail=include_detail)


def badge_status(supervisor: Supervisor) -> Dict[str, Any]:
    """``GET /api/bots/status`` -- ``{"attention": int, "state": str}``.

    Keep it cheap; ``<jarvis-bots-button>`` polls it.
    """
    return supervisor.badge_status()


def bot_detail(
    supervisor: Supervisor,
    bot_id: str,
    now: Optional[float] = None,
    *,
    limit: int = EVENT_HISTORY,
) -> Dict[str, Any]:
    """``GET /api/bots/<id>`` -- what a generated detail page reads.

    The shape ``jarvis_bots/templates/page.html.tmpl`` documents, verbatim::

        {"generated_at": int,
         "bot": {<the launcher card>, "detail": str,
                 "attention_items": [{"key","text","href","since"}],
                 "events": [{"at","severity","text","href"}]}}

    Newest event first, at most :data:`~jarvis_bots.supervisor.EVENT_HISTORY`
    of them.  Raises :class:`ApiError` with ``status=404`` for an unknown id.
    """
    try:
        return supervisor.bot_detail(bot_id, now, limit=limit)
    except RegistryError as exc:
        raise ApiError(str(exc), status=404) from exc


def set_paused(supervisor: Supervisor, body: Mapping[str, Any]) -> Dict[str, Any]:
    """``POST /api/bots/pause`` -- body ``{"bot_id": "poke", "paused": true}``.

    Returns the badge payload, so the page that pauses a bot can update the
    nav from the same response rather than racing the next poll.  Raises
    :class:`ApiError` for a body that is not that shape (400), an unknown
    bot (404), or a bot that declares ``can_pause=False`` (400) -- the
    launcher does not offer the control for one, but an endpoint is not
    only reached by the launcher.
    """
    if not isinstance(body, Mapping):
        raise ApiError(f"body must be an object; got {type(body).__name__}")
    bot_id = body.get("bot_id")
    if not isinstance(bot_id, str) or not bot_id:
        raise ApiError("body needs a bot_id")
    paused = body.get("paused")
    if not isinstance(paused, bool):
        raise ApiError("body needs paused: true or false")
    try:
        supervisor.set_paused(bot_id, paused)
    except RegistryError as exc:
        raise ApiError(str(exc), status=404) from exc
    except SupervisorError as exc:
        raise ApiError(str(exc), status=400) from exc
    return supervisor.badge_status()
