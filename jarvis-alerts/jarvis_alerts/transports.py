"""Push transports: design point 2 of :mod:`jarvis_alerts.contracts`.

Contracts.py, module docstring, point 2: "Sending goes through a subscription
registry and a transport interface. The transport is injected; this package
never holds or prints the material a transport needs to authenticate, it only
stores each device's opaque subscription blob as the app's own data."

This module is the transport side of that sentence. Three classes satisfy the
:class:`~jarvis_alerts.contracts.Transport` protocol (``name`` plus
``send(subscription, alert) -> SendResult``):

    FakeTransport       scripted, records calls; for tests and the gate
    WebPushTransport    RFC 8030 Web Push, via an injected sender
    FCMTransport        Firebase Cloud Messaging HTTP v1, via an injected sender

What the real transports do and do not do
-----------------------------------------
They *shape the payload* from the :class:`~jarvis_alerts.contracts.Alert`,
*read the target* out of the subscription blob, hand both to the injected
``sender`` callable the app supplies, and *interpret the result* into a
:class:`~jarvis_alerts.contracts.SendResult` whose three flags drive the
retry policy at the bottom of contracts.py:

    ok          delivered to the push service; the row is DELIVERED
    retryable   transient (429, 5xx); the outbox backs off and tries again
    gone        the device is dead (404, 410, UNREGISTERED); prune it

Everything else is the sender's job: VAPID signing and payload encryption
for Web Push, OAuth for FCM, the HTTP call itself. Nothing in this module
holds, reads from the environment or embeds any authentication material,
which is how the sentence quoted above is kept true.

Privacy (contracts.py, ``Subscription.blob`` "never logged"): the blob is
parsed and the endpoint (with the subscription's ``keys``, for Web Push) or
token is passed to the sender, and that is all.
No log line, print, exception message or ``SendResult.reason`` produced here
carries the blob, the endpoint, the token or a response body. Reasons name a
status code and, for FCM, a fixed marker word. Subscriptions are identified
by (profile_id, device_id) only; :class:`FakeTransport` records exactly that.

Determinism: the transports hold no clock and draw no randomness, so they
take neither a clock nor a jitter callable; there is nothing for either to
drive. Simulated failures belong in a :class:`FakeTransport` script, which
may close over a ``lucifer_gen.seed.Stream`` (see the tests). The payload
encoding is canonical (sorted keys, fixed separators) so the same alert
always produces the same bytes.

Idempotency (design point 3): :func:`payload_for` always carries ``id`` so
the client can collapse duplicates by alert id, and FCM's data map carries
the same keys as strings.
"""

from __future__ import annotations

import inspect
import json
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .contracts import Alert, Priority, SendResult, Subscription

#: Web Push ``Urgency`` header value and ``TTL`` (seconds) per priority.
#: HIGH is something the owner is waiting on right now (a render finished);
#: LOW may sit at the push service for three days if the device is off.
WEBPUSH_URGENCY: Dict[Priority, str] = {
    Priority.HIGH: "high",
    Priority.NORMAL: "normal",
    Priority.LOW: "low",
}
WEBPUSH_TTL_S: Dict[Priority, int] = {
    Priority.HIGH: 3600,
    Priority.NORMAL: 86400,
    Priority.LOW: 259200,
}

#: FCM only knows two Android priorities; LOW collapses into "normal".
FCM_ANDROID_PRIORITY: Dict[Priority, str] = {
    Priority.HIGH: "high",
    Priority.NORMAL: "normal",
    Priority.LOW: "normal",
}

#: Substrings of an FCM response body that mean the token is dead.
FCM_GONE_MARKERS: Tuple[str, ...] = ("UNREGISTERED", "NOT_FOUND")

WEBPUSH_OK_STATUSES = frozenset({200, 201, 202})
WEBPUSH_GONE_STATUSES = frozenset({404, 410})

#: ``sender(endpoint, body, headers, keys) -> status``.  ``keys`` is the
#: subscription's own ``keys`` object ({"p256dh": ..., "auth": ...}) that
#: payload encryption needs; a sender written for the older three-argument
#: form is still called with three.  See :class:`WebPushTransport`.
WebPushSender = Callable[..., int]
FCMSender = Callable[[str, Dict[str, Any]], Tuple[int, str]]
FakeScript = Callable[[Subscription, Alert, int], SendResult]


# ---------------------------------------------------------------------------
# Payload shaping, shared by both real transports.
# ---------------------------------------------------------------------------


def payload_for(alert: Alert) -> Dict[str, Any]:
    """The notification payload for one alert, as a JSON-ready dict.

    Keys are exactly ``id``, ``kind``, ``title``, ``body``, ``data`` and
    ``priority``. ``id`` is what the client dedupes on (design point 3), so
    it is never omitted. ``priority`` is the lowercase name ("low",
    "normal", "high") rather than the IntEnum value so a service worker
    needs no knowledge of the enum's numbering. ``data`` is a copy, so
    mutating the result cannot reach the (frozen) alert's dict.

    Pure and deterministic: the same alert always gives an equal dict.
    Raises ``ValueError`` if ``alert.priority`` is not a :class:`Priority`.
    """
    return {
        "id": alert.id,
        "kind": alert.kind,
        "title": alert.title,
        "body": alert.body,
        "data": dict(alert.data),
        "priority": Priority(alert.priority).name.lower(),
    }


def encode_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON: sorted keys, no whitespace.

    Used for the Web Push body and for non-string FCM data values, so two
    encodings of the same alert are byte-identical whatever the insertion
    order of ``alert.data``. Raises ``TypeError``/``ValueError`` from
    ``json.dumps`` when the alert's ``data`` is not JSON-encodable.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def payload_bytes(alert: Alert) -> bytes:
    """:func:`payload_for` encoded with :func:`encode_json`."""
    return encode_json(payload_for(alert))


def webpush_headers(priority: Priority) -> Dict[str, str]:
    """``TTL`` and ``Urgency`` headers for a priority (RFC 8030 sections 5.2, 5.3).

    HIGH -> urgency "high", TTL 3600; NORMAL -> "normal", 86400;
    LOW -> "low", 259200. The sender adds the headers only it can produce
    (Authorization, Content-Encoding, Crypto-Key, ...).
    """
    priority = Priority(priority)
    return {"TTL": str(WEBPUSH_TTL_S[priority]), "Urgency": WEBPUSH_URGENCY[priority]}


def fcm_message(alert: Alert) -> Dict[str, Any]:
    """The FCM v1 ``message`` dict for an alert (without the ``token``).

    ``notification`` carries title and body for the platform to display.
    ``data`` is the same six keys as :func:`payload_for`, every value a
    string because FCM rejects anything else: strings pass through, the
    rest (``data``, if it were ever non-string) is canonical JSON the client
    parses back. Keeping the nested ``data`` as one JSON string, rather
    than flattening it into the map, means an alert's own data key called
    ``id`` cannot shadow the alert id the client dedupes on.

    ``android`` carries the priority and TTL that :func:`webpush_headers`
    carries for Web Push; without it FCM treats every message as normal
    priority and a HIGH alert would not wake a dozing device.
    """
    priority = Priority(alert.priority)
    data = {key: _fcm_string(value) for key, value in payload_for(alert).items()}
    return {
        "notification": {"title": alert.title, "body": alert.body},
        "data": data,
        "android": {
            "priority": FCM_ANDROID_PRIORITY[priority],
            "ttl": f"{WEBPUSH_TTL_S[priority]}s",
        },
    }


def _fcm_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    return encode_json(value).decode("utf-8")


# ---------------------------------------------------------------------------
# Result interpretation. Total functions over the status space: any status
# the tables do not name is a permanent failure that names the status.
# ---------------------------------------------------------------------------


def map_webpush_status(status: int) -> SendResult:
    """Interpret a push service HTTP status (RFC 8030 section 8).

    200/201/202 -> ok.  404/410 -> gone (the subscription expired or was
    unsubscribed; prune it).  429 and any 5xx -> retryable.  Anything else,
    which includes 400/401/403 (bad request, bad or missing VAPID, key
    mismatch), is permanent: a retry with the same bytes cannot help, and
    the reason names the status and nothing else.
    """
    if status in WEBPUSH_OK_STATUSES:
        return SendResult(ok=True)
    reason = f"webpush status {status}"
    if status in WEBPUSH_GONE_STATUSES:
        return SendResult(ok=False, gone=True, reason=reason)
    if status == 429 or 500 <= status <= 599:
        return SendResult(ok=False, retryable=True, reason=reason)
    return SendResult(ok=False, retryable=False, reason=reason)


def map_fcm_response(status: int, body: str) -> SendResult:
    """Interpret an FCM v1 response, in this order.

    200 -> ok.  A body containing ``UNREGISTERED`` or ``NOT_FOUND`` -> gone,
    whatever the status (FCM reports a dead token as 404 with the error
    code in the body; the code is the reliable part).  429 and any 5xx ->
    retryable (QUOTA_EXCEEDED, UNAVAILABLE, INTERNAL).  Anything else ->
    permanent.  The reason names the status and, for gone, the marker word
    that matched; the body itself is never copied into it.
    """
    if status == 200:
        return SendResult(ok=True)
    reason = f"fcm status {status}"
    for marker in FCM_GONE_MARKERS:
        if marker in body:
            return SendResult(ok=False, gone=True, reason=f"{reason} {marker}")
    if status == 429 or 500 <= status <= 599:
        return SendResult(ok=False, retryable=True, reason=reason)
    return SendResult(ok=False, retryable=False, reason=reason)


# ---------------------------------------------------------------------------
# Blob handling. The one place the blob is read; nothing about it escapes
# except the field the sender needs.
# ---------------------------------------------------------------------------


def _blob_field(blob: Any, key: str) -> Optional[str]:
    """``blob[key]`` if the blob is a JSON object with a non-empty string
    there, else ``None``. Never raises and never formats the blob."""
    if not isinstance(blob, str):
        return None
    try:
        parsed = json.loads(blob)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    value = parsed.get(key)
    if not isinstance(value, str) or not value:
        return None
    return value


def _blob_keys(blob: Any) -> Dict[str, str]:
    """The blob's ``keys`` object, string values only, or ``{}``.  Never
    raises and never formats the blob."""
    if not isinstance(blob, str):
        return {}
    try:
        parsed = json.loads(blob)
    except ValueError:
        return {}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("keys"), dict):
        return {}
    return {str(k): v for k, v in parsed["keys"].items() if isinstance(v, str)}


def _accepts_positional(callable_: Any, n: int) -> bool:
    """Can ``callable_`` be called with ``n`` positional arguments?  Read
    from its signature; a callable without one (a builtin, say) is taken
    to accept ``n`` only if ``n`` is the smallest number asked about."""
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return False
    try:
        signature.bind(*range(n))
    except TypeError:
        return False
    return True


def _unusable_blob(key: str) -> SendResult:
    """A blob without a usable target can never be delivered to: it is gone.

    Returning gone rather than raising means the outbox prunes the device
    instead of retrying MAX_ATTEMPTS times against a blob that will never
    parse. The reason names the missing key, never the blob.
    """
    return SendResult(ok=False, gone=True, reason=f"blob has no usable {key}")


#: An alert that cannot be shaped into a payload (``data`` not
#: JSON-encodable, or ``priority`` not a :class:`Priority`) is a publisher
#: bug. It is reported as permanent (not retryable) so the row dead-letters
#: at once with a reason that says why, instead of after MAX_ATTEMPTS
#: identical exceptions. The exception text is dropped: it could quote the
#: data.
_BAD_ALERT = SendResult(ok=False, retryable=False,
                        reason="alert not encodable: bad priority or non-JSON data")


# ---------------------------------------------------------------------------
# The transports.
# ---------------------------------------------------------------------------


class FakeTransport:
    """A scripted transport that records every call. For tests and the gate.

    ``script(subscription, alert, index)`` decides the result of each send;
    ``index`` is the 0-based position of that call in :attr:`calls`, i.e.
    how many sends came before it, so a script can fail the first N and
    then succeed, or close over a ``lucifer_gen.seed.Stream`` for
    reproducible simulated failures. The default script always says ok.

    :attr:`calls` holds ``(profile_id, device_id, alert_id)`` per send, in
    order, recorded *before* the script runs so a script that raises still
    leaves its trace. The blob is never kept. Safe to share between worker
    threads: recording is under a lock; the script runs outside it.

    ``name`` defaults to "fake", the third transport name contracts.py
    lists for ``Subscription.transport``; pass another to stand in for a
    real transport under its own name.
    """

    def __init__(self, script: Optional[FakeScript] = None, name: str = "fake") -> None:
        self.name = name
        self.script: FakeScript = script if script is not None else _always_ok
        self.calls: List[Tuple[str, str, str]] = []
        self._lock = threading.Lock()

    def send(self, subscription: Subscription, alert: Alert) -> SendResult:
        with self._lock:
            index = len(self.calls)
            self.calls.append((subscription.profile_id, subscription.device_id, alert.id))
        return self.script(subscription, alert, index)

    def calls_for(self, profile_id: str, device_id: str) -> int:
        """How many sends went to one device."""
        return sum(1 for p, d, _ in self.calls if (p, d) == (profile_id, device_id))


def _always_ok(subscription: Subscription, alert: Alert, index: int) -> SendResult:
    return SendResult(ok=True)


class WebPushTransport:
    """Web Push (RFC 8030) through an injected ``sender``.

    ``sender(endpoint, payload_bytes, headers, keys) -> int`` makes the
    signed, encrypted request and returns the HTTP status. VAPID signing
    and the ``aes128gcm`` encryption of the body are the sender's business;
    this class never sees a VAPID key. Encryption needs the subscription's
    own ``keys.p256dh`` and ``keys.auth``, which live in the blob next to
    the endpoint, so the transport hands the blob's ``keys`` object to the
    sender as the fourth argument: the outbox is the only copy of the
    subscription the app has to keep. A sender that takes only three
    arguments (``endpoint, body, headers``) is still supported and gets no
    keys; it must then be a sender that needs none. A sender that cannot
    reach the service should raise; the worker turns a raise into a
    retryable attempt.

    The blob is JSON with an ``endpoint`` key (the shape the browser's
    ``PushSubscription.toJSON()`` gives). A blob without one is reported
    as gone; ``keys`` is passed as-is (string values only, ``{}`` when
    absent) because whether it is needed is the sender's business.
    Headers come from :func:`webpush_headers`; the body from
    :func:`payload_bytes`; the status goes through
    :func:`map_webpush_status`.
    """

    name = "webpush"

    def __init__(self, sender: WebPushSender) -> None:
        self._sender = sender
        self._pass_keys = _accepts_positional(sender, 4)

    def send(self, subscription: Subscription, alert: Alert) -> SendResult:
        endpoint = _blob_field(subscription.blob, "endpoint")
        if endpoint is None:
            return _unusable_blob("endpoint")
        try:
            body = payload_bytes(alert)
        except (TypeError, ValueError):
            return _BAD_ALERT
        headers = webpush_headers(alert.priority)
        if self._pass_keys:
            status = self._sender(endpoint, body, headers, _blob_keys(subscription.blob))
        else:
            status = self._sender(endpoint, body, headers)
        if not isinstance(status, int):
            raise TypeError(f"webpush sender must return an int status, got {type(status).__name__}")
        return map_webpush_status(status)


class FCMTransport:
    """Firebase Cloud Messaging HTTP v1 through an injected ``sender``.

    ``sender(token, message) -> (status, body)`` posts ``{"message": {
    "token": token, **message}}`` with whatever OAuth credential the app
    holds and returns the HTTP status and response body text (``bytes`` is
    accepted and decoded). This class never sees the credential.

    The blob is JSON with a ``token`` key (the device's registration
    token). A blob without one is reported as gone. The message comes from
    :func:`fcm_message`; the response goes through :func:`map_fcm_response`.
    """

    name = "fcm"

    def __init__(self, sender: FCMSender) -> None:
        self._sender = sender

    def send(self, subscription: Subscription, alert: Alert) -> SendResult:
        token = _blob_field(subscription.blob, "token")
        if token is None:
            return _unusable_blob("token")
        try:
            message = fcm_message(alert)
        except (TypeError, ValueError):
            return _BAD_ALERT
        status, body = _unpack_fcm_response(self._sender(token, message))
        return map_fcm_response(status, body)


def _unpack_fcm_response(response: Any) -> Tuple[int, str]:
    """Validate the sender's ``(status, body)`` without echoing its content."""
    if not isinstance(response, Sequence) or isinstance(response, (str, bytes)) or len(response) != 2:
        raise TypeError(f"fcm sender must return (status, body), got {type(response).__name__}")
    status, body = response
    if not isinstance(status, int):
        raise TypeError(f"fcm sender status must be an int, got {type(status).__name__}")
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    elif body is None:
        body = ""
    elif not isinstance(body, str):
        raise TypeError(f"fcm sender body must be str or bytes, got {type(body).__name__}")
    return status, body


__all__ = [
    "FakeTransport", "WebPushTransport", "FCMTransport",
    "payload_for", "payload_bytes", "encode_json", "webpush_headers", "fcm_message",
    "map_webpush_status", "map_fcm_response",
    "WEBPUSH_URGENCY", "WEBPUSH_TTL_S", "FCM_ANDROID_PRIORITY", "FCM_GONE_MARKERS",
    "WebPushSender", "FCMSender", "FakeScript",
]


# ---------------------------------------------------------------------------
# python3 -m jarvis_alerts.transports: a self-check with stub senders.
# Prints statuses, header names and (profile_id, device_id); never a blob.
# ---------------------------------------------------------------------------


def _demo() -> int:
    alert = Alert("a1", "owner", "render_done", "Render finished", "crypt.png is ready", 0.0,
                  priority=Priority.HIGH, data={"file": "crypt.png"})
    sub_wp = Subscription("owner", "laptop", "webpush", '{"endpoint":"https://push.example/x"}', 0.0)
    sub_fcm = Subscription("owner", "phone", "fcm", '{"token":"t"}', 0.0)
    scripted = iter([201, 429, 410, 403])
    web = WebPushTransport(lambda endpoint, body, headers: next(scripted))
    fcm_scripted = iter([(200, ""), (503, "UNAVAILABLE"), (404, '{"error":{"status":"NOT_FOUND"}}')])
    fcm = FCMTransport(lambda token, message: next(fcm_scripted))
    print("payload:", payload_bytes(alert).decode("utf-8"))
    print("headers:", webpush_headers(alert.priority))
    for transport, sub, n in ((web, sub_wp, 4), (fcm, sub_fcm, 3)):
        for _ in range(n):
            r = transport.send(sub, alert)
            print(f"{transport.name} ({sub.profile_id}, {sub.device_id}) ok={r.ok} "
                  f"retryable={r.retryable} gone={r.gone} reason={r.reason!r}")
    bad = Subscription("owner", "tablet", "webpush", "not json", 0.0)
    print("malformed blob ->", web.send(bad, alert))
    fake = FakeTransport()
    fake.send(sub_wp, alert)
    print("fake calls:", fake.calls)
    return 0


if __name__ == "__main__":
    raise SystemExit(_demo())
