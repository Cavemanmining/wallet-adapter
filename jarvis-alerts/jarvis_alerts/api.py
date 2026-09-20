"""The app-facing surface: what Jarvis calls where it now emits to the socket.

Design: the module docstring of :mod:`jarvis_alerts.contracts`.  The symptom
is that alerts reach the owner only while the app is open, because the
backend emits to live connections and nothing else.  This module is the
*publish* side of the fix: every place the app currently does
``socket.emit("alert", ...)`` first hands the alert to :class:`AlertService`,
which writes it to the durable outbox (design point 1) before anyone tries
to send it.  The worker (:mod:`jarvis_alerts.worker`, run by
``python3 -m jarvis_alerts.cli worker``) does the sending through the push
transports (design point 2) and records every attempt (design point 3).

The three lines at an emit point::

    service = AlertService(Outbox("alerts.sqlite3", clock=time.time), clock=time.time)
    alert_id = service.publish(profile_id, "render_done", "Render finished", "crypt.png is ready")
    socket.emit("alert", {"id": alert_id, ...})   # keep the fast path; the worker does the delivery

The socket emit stays: a live connection is a fine fast path.  The client
collapses the socket copy and the push copy by ``alert_id`` (design point
3), which is why :meth:`AlertService.publish` returns it.

Devices come and go through :meth:`AlertService.register_device` and
:meth:`AlertService.unregister_device`, called from the endpoint the
service worker posts its ``PushSubscription`` to (the client half of the
design, ``jarvis_alerts/client/``).  :meth:`AlertService.backfill_device`
lets a device registered after an alert was published still receive it.

Senders
-------
Design point 2 says the transport is injected and this package never
holds authentication material.  The app installs, at import time, one
*sender* callable per real transport with :func:`set_sender`; the CLI's
``worker`` command builds ``WebPushTransport`` / ``FCMTransport`` from
:mod:`jarvis_alerts.transports` around them.  A transport with no sender
installed is simply not wired in that process and its rows wait (see
:mod:`jarvis_alerts.cli`).  Nothing here reads the environment or a file
for a key: a sender closes over whatever it needs on the app's side.

Privacy: the subscription blob is the owner's device data.  It goes into
the outbox and comes out only for a transport.  No message raised here
quotes it; subscriptions are named by (profile_id, device_id).

Determinism: the clock and the alert-id source are injected callables.
:func:`default_id_source` uses ``uuid4``; a test passes its own so alert
ids are reproducible (see :func:`alert_id_for`).
"""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Callable, Dict, List, Optional, Union

from .contracts import Alert, Priority, Subscription
from .outbox import Outbox

__all__ = [
    "ALERT_ID_HEX",
    "AlertService",
    "Clock",
    "IdSource",
    "Sender",
    "alert_id_for",
    "clear_senders",
    "default_id_source",
    "get_sender",
    "installed_senders",
    "parse_priority",
    "set_sender",
]

Clock = Callable[[], float]
IdSource = Callable[[], str]
#: A sender's exact signature is the transport's business
#: (``jarvis_alerts.transports.WebPushSender`` / ``FCMSender``).
Sender = Callable[..., Any]

#: Alert ids are the first 32 hex digits (128 bits) of a SHA-256: unique
#: for any realistic volume and short enough to read in ``cli dead`` output.
ALERT_ID_HEX = 32

_PRIORITY_NAMES: Dict[str, Priority] = {p.name.lower(): p for p in Priority}


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def default_id_source() -> str:
    """The production id source: a fresh ``uuid4``.  Injected by default so a
    test can replace it with a counter and get reproducible alert ids."""
    return uuid.uuid4().hex


def alert_id_for(profile_id: str, kind: str, key: str, created_at: float) -> str:
    """The alert id: a hash of profile, kind, ``key`` and creation time.

    ``key`` is the alert's ``dedupe_key`` when it has one, otherwise a value
    from the id source.  Hashing (rather than using the uuid directly) makes
    ids for keyed alerts a pure function of what the app said and when,
    so two publishers of the same keyed alert in the same instant agree on
    the id, and the outbox's dedupe window collapses them before the id
    could ever collide.
    """
    material = "\x1f".join((profile_id, kind, key, repr(float(created_at))))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:ALERT_ID_HEX]


def parse_priority(value: Union[Priority, int, str]) -> Priority:
    """Accept a :class:`Priority`, its int value, or its name ("high",
    case-insensitive).  Raises ``ValueError`` for anything else."""
    if isinstance(value, Priority):
        return value
    if isinstance(value, str):
        try:
            return _PRIORITY_NAMES[value.strip().lower()]
        except KeyError:
            raise ValueError(
                f"priority must be one of {', '.join(_PRIORITY_NAMES)}; got {value!r}"
            ) from None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"priority must be a Priority, an int or a name; got {type(value).__name__}")
    return Priority(value)


def _check_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _check_blob(profile_id: str, device_id: str, blob: Any) -> str:
    """The blob is opaque JSON text.  Its content is never part of the
    message: a bad blob is reported by (profile_id, device_id) only."""
    where = f"blob for ({profile_id}, {device_id})"
    if not isinstance(blob, str) or not blob.strip():
        raise ValueError(f"{where} must be non-empty JSON text")
    try:
        json.loads(blob)
    except ValueError:
        # ``from None``: the decoder's message carries a position, which is
        # harmless, but the chain would still be one more place to inspect.
        raise ValueError(f"{where} is not valid JSON") from None
    try:
        blob.encode("utf-8")
    except UnicodeEncodeError:
        # A JSON body may carry a lone surrogate ("\ud800"); sqlite would
        # refuse to bind it with an exception whose ``args`` are the whole
        # blob.  Refuse it here, by name only.
        raise ValueError(f"{where} is not valid UTF-8 text") from None
    return blob


# ---------------------------------------------------------------------------
# The service
# ---------------------------------------------------------------------------


class AlertService:
    """What the app calls.  A thin, stateless front on the outbox.

    ``outbox``     a :class:`jarvis_alerts.outbox.Outbox` (the durable store
                   of design point 1)
    ``clock``      returns unix seconds; stamps ``created_at`` on alerts and
                   subscriptions.  Give the outbox the same clock: its dedupe
                   window compares its own clock against these stamps.
    ``id_source``  returns a fresh unique string for alerts without a
                   dedupe key; :func:`default_id_source` in production.

    Safe to share between threads: it holds no state of its own and the
    outbox serialises its own access.
    """

    def __init__(
        self,
        outbox: Outbox,
        clock: Clock,
        id_source: IdSource = default_id_source,
    ) -> None:
        self._outbox = outbox
        self._clock = clock
        self._id_source = id_source

    @property
    def outbox(self) -> Outbox:
        return self._outbox

    # -- publishing (design point 1: write before send) ----------------------

    def publish(
        self,
        profile_id: str,
        kind: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        priority: Union[Priority, int, str] = Priority.NORMAL,
        dedupe_key: Optional[str] = None,
    ) -> str:
        """Store an alert for ``profile_id`` and fan it out; return its id.

        The id is :func:`alert_id_for` over the profile, the kind, the
        dedupe key (or a value from the id source when there is none) and
        ``clock()``.  ``data`` must be JSON-encodable; ``priority`` takes
        anything :func:`parse_priority` does.

        Dedupe (design point 3): when ``dedupe_key`` repeats one this profile
        published less than ``outbox.DEDUPE_WINDOW_S`` ago, the outbox
        collapses the new alert into the earlier one and stores nothing.
        The id is returned all the same, so an emit point never has to
        branch; ``outbox.alert(id)`` is then ``None``, unless the repeat
        came in the very same clock instant, when the two ids coincide and
        the lookup finds the alert that was kept.
        """
        _check_text("profile_id", profile_id)
        _check_text("kind", kind)
        if not isinstance(title, str) or not isinstance(body, str):
            raise ValueError("title and body must be strings")
        if dedupe_key is not None:
            _check_text("dedupe_key", dedupe_key)
        if data is not None and not isinstance(data, dict):
            raise ValueError("data must be a dict (a JSON object) or None")

        created_at = float(self._clock())
        key = dedupe_key if dedupe_key is not None else _check_text("id_source()", self._id_source())
        alert = Alert(
            id=alert_id_for(profile_id, kind, key, created_at),
            profile_id=profile_id,
            kind=kind,
            title=title,
            body=body,
            created_at=created_at,
            priority=parse_priority(priority),
            dedupe_key=dedupe_key,
            data=dict(data) if data else {},
        )
        self._outbox.publish(alert)
        return alert.id

    # -- the registry (design point 2) ----------------------------------------

    def register_device(
        self,
        profile_id: str,
        device_id: str,
        transport: str,
        blob: str,
        backfill_s: Optional[float] = None,
        *,
        supersede_same_blob: bool = False,
    ) -> int:
        """Upsert one device's subscription; return the rows backfilled.

        ``transport`` names the transport the worker should use ("webpush",
        "fcm", "fake"; any name is stored, and rows for a name no worker has
        wired simply wait).  ``blob`` is the opaque JSON the transport
        understands, checked to be JSON and UTF-8 and otherwise untouched.
        Re-registering a device clears its failure count and ``gone`` /
        ``pruned`` flags (see :meth:`Outbox.register`).

        ``backfill_s``: when given, the profile's alerts of the last that
        many seconds are queued for the device in the same transaction as
        the registration, so a publish racing the registration can never
        slip between the two (it either fans out to the new device or is
        picked up by the backfill).  ``None`` backfills nothing; call
        :meth:`backfill_device` later instead.  ``supersede_same_blob``
        forgets any other device id of the profile holding byte-identical
        blob text, so one browser cannot become two devices (see
        :meth:`Outbox.register`); the reference subscribe endpoint sets it.
        """
        _check_text("profile_id", profile_id)
        _check_text("device_id", device_id)
        _check_text("transport", transport)
        _check_blob(profile_id, device_id, blob)
        if backfill_s is not None and backfill_s < 0:
            raise ValueError("backfill_s must not be negative")
        now = float(self._clock())
        return self._outbox.register(
            Subscription(
                profile_id=profile_id,
                device_id=device_id,
                transport=transport,
                blob=blob,
                created_at=now,
            ),
            backfill_since=None if backfill_s is None else now - float(backfill_s),
            supersede_same_blob=supersede_same_blob,
        )

    def unregister_device(self, profile_id: str, device_id: str) -> bool:
        """Forget a device; returns whether it was registered.  Its pending
        rows are parked, not deleted (see :meth:`Outbox.unregister`)."""
        return self._outbox.unregister(profile_id, device_id)

    def backfill_device(self, profile_id: str, device_id: str, since_s: float = 3600.0) -> int:
        """Queue the profile's alerts of the last ``since_s`` seconds for a
        device that has no row for them.  Returns the rows created.

        Raises ``LookupError`` when (profile_id, device_id) has no live or
        pruned subscription, because backfilling nothing is more likely a
        bug than an intent (a pruned device's rows wait for its cooldown).
        The outbox backfills per profile, so any *other* such device of the
        profile that is missing a row for one of those alerts gets one too;
        in practice that only happens to a device that was registered
        without a backfill, and the count includes it.
        """
        if since_s < 0:
            raise ValueError("since_s must not be negative")
        sub = self._outbox.subscription(profile_id, device_id)
        if sub is None or (sub.gone and not sub.pruned):
            raise LookupError(f"no live subscription for ({profile_id}, {device_id})")
        return self._outbox.backfill(profile_id, since=float(self._clock()) - float(since_s))

    # -- dead letters -----------------------------------------------------------

    def requeue(self, row_id: int) -> bool:
        """Put one dead-lettered row back in flight with a fresh attempt
        budget; see :meth:`Outbox.requeue`.  Returns whether it was DEAD."""
        return self._outbox.requeue(int(row_id))

    def requeue_dead(self, profile_id: Optional[str] = None, device_id: Optional[str] = None) -> int:
        """Re-queue every dead letter, or a profile's, or one device's;
        returns how many.  See :meth:`Outbox.requeue_dead`."""
        return self._outbox.requeue_dead(profile_id, device_id)

    # -- inspection ------------------------------------------------------------

    def stats(self, dead_limit: int = 50) -> Dict[str, Any]:
        """Counts per row state plus the newest dead letters; see
        :meth:`Outbox.stats`.  JSON-serialisable and blob-free."""
        return self._outbox.stats(dead_limit=dead_limit)


# ---------------------------------------------------------------------------
# Senders: the one thing the app injects for the real transports
# ---------------------------------------------------------------------------

_SENDERS: Dict[str, Sender] = {}


def set_sender(name: str, sender: Sender) -> None:
    """Install the app's sender for transport ``name`` ("webpush", "fcm").

    Call it at import time, before the worker starts; the CLI's ``worker``
    command reads the registry once when it wires its transports.  The
    callable's signature is the transport's contract
    (``jarvis_alerts.transports.WebPushSender`` / ``FCMSender``).  Whatever
    credential it needs lives inside it, on the app's side.
    """
    _check_text("sender name", name)
    if not callable(sender):
        raise TypeError(f"sender for {name!r} must be callable")
    _SENDERS[name] = sender


def get_sender(name: str) -> Optional[Sender]:
    """The installed sender for ``name``, or ``None``."""
    return _SENDERS.get(name)


def installed_senders() -> List[str]:
    """Names with a sender installed, sorted."""
    return sorted(_SENDERS)


def clear_senders() -> None:
    """Forget every installed sender (tests)."""
    _SENDERS.clear()


# ---------------------------------------------------------------------------
# Smoke run: python3 -m jarvis_alerts.api
# ---------------------------------------------------------------------------


def _demo() -> Dict[str, Any]:
    """Publish, register late, backfill, on an in-memory outbox with a
    hand-driven clock and a counting id source; return the stats."""
    ticks = [1_700_000_000.0]
    counter = iter(range(1, 100))
    outbox = Outbox(":memory:", clock=lambda: ticks[0])
    service = AlertService(outbox, clock=lambda: ticks[0], id_source=lambda: f"id{next(counter)}")

    first = service.publish("owner", "render_done", "Render finished", "crypt.png is ready",
                            data={"file": "crypt.png"}, priority="high", dedupe_key="render:crypt")
    ticks[0] += 30.0
    again = service.publish("owner", "render_done", "Render finished", "crypt.png is ready",
                            dedupe_key="render:crypt")             # collapsed by the outbox
    assert outbox.alert(first) is not None and outbox.alert(again) is None
    ticks[0] += 30.0
    service.register_device("owner", "phone", "fake", '{"endpoint": "opaque"}')
    created = service.backfill_device("owner", "phone", since_s=3600)
    stats = service.stats()
    stats["backfilled"] = created
    stats["alert_id"] = first
    return stats


if __name__ == "__main__":
    print(json.dumps(_demo(), indent=2, sort_keys=True))
