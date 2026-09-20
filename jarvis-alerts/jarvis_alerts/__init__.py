"""jarvis_alerts: durable, push-backed alert delivery for the Jarvis assistant.

The problem and the design are stated in :mod:`jarvis_alerts.contracts`:
alerts reached the owner only while the app was open, because the backend
emitted to live connections and nothing else.  Delivery needs three things,
and each has a module here:

    1. a durable outbox        :mod:`jarvis_alerts.outbox`     (``Outbox``)
    2. a push transport        :mod:`jarvis_alerts.transports` (``WebPushTransport``,
                               ``FCMTransport``, ``FakeTransport``) driven by
                               :mod:`jarvis_alerts.worker`     (``Worker``)
    3. idempotency             dedupe keys and recorded attempts, in the
                               outbox; the client collapses by alert id
                               (``jarvis_alerts/client/``)

    app side   :mod:`jarvis_alerts.api`      (``AlertService``, ``set_sender``)
    operator   :mod:`jarvis_alerts.cli`      (``python3 -m jarvis_alerts.cli``)
    gate       :mod:`jarvis_alerts.validate` (``run_gate``)

The three lines at an emit point::

    from jarvis_alerts import AlertService, Outbox
    service = AlertService(Outbox("alerts.sqlite3", clock=time.time), clock=time.time)
    alert_id = service.publish(profile_id, "render_done", "Render finished", "crypt.png is ready")

This module re-exports the public surface so callers name the package
rather than a file.  No re-exported name is spelled like a submodule
(``api``, ``cli``, ``contracts``, ``outbox``, ``transports``, ``validate``,
``worker``), so ``from jarvis_alerts import validate`` still gives the
module; the two ``main`` entry points (``cli.main``, ``validate.main``) are
deliberately not re-exported because they would collide.  ``cli`` is not
imported here: it is an entry point, reached as ``jarvis_alerts.cli``.

Nothing in this package holds, reads from the environment or embeds any
authentication material for a push service, and nothing logs, prints or
puts a subscription blob in an exception message.
"""

from __future__ import annotations

from .contracts import (
    ALERT_MAX_AGE_S,
    BACKOFF_BASE_S,
    BACKOFF_CAP_S,
    EXHAUSTED_RETRY_COOLDOWN_S,
    LEASE_S,
    MAX_ATTEMPTS,
    PRUNE_AFTER_FAILURES,
    PRUNE_COOLDOWN_S,
    Alert,
    DeadReason,
    DeliveryAttempt,
    OutboxRow,
    Priority,
    RowState,
    SendResult,
    Subscription,
    Transport,
    backoff_seconds,
    dead_reason_for,
)
from .outbox import (
    DEDUPE_WINDOW_S,
    DuplicateAlert,
    Outbox,
    OutboxError,
    SchemaError,
    UnknownRow,
)
from .transports import (
    FakeTransport,
    FCMTransport,
    WebPushTransport,
    fcm_message,
    map_fcm_response,
    map_webpush_status,
    payload_bytes,
    payload_for,
    webpush_headers,
)
from .worker import MemoryStore, OutboxPort, RunReport, SimClock, Worker, drain
from .api import (
    AlertService,
    alert_id_for,
    clear_senders,
    get_sender,
    installed_senders,
    parse_priority,
    set_sender,
)
from .validate import DEFECTS, GateReport, Problem, run_gate

__all__ = [
    # contracts
    "Priority", "RowState", "DeadReason", "Alert", "Subscription", "SendResult", "Transport",
    "OutboxRow", "DeliveryAttempt", "BACKOFF_BASE_S", "BACKOFF_CAP_S",
    "MAX_ATTEMPTS", "LEASE_S", "PRUNE_AFTER_FAILURES", "PRUNE_COOLDOWN_S",
    "EXHAUSTED_RETRY_COOLDOWN_S", "ALERT_MAX_AGE_S", "backoff_seconds", "dead_reason_for",
    # outbox
    "Outbox", "DEDUPE_WINDOW_S", "OutboxError", "SchemaError", "UnknownRow", "DuplicateAlert",
    # transports
    "FakeTransport", "WebPushTransport", "FCMTransport",
    "payload_for", "payload_bytes", "webpush_headers", "fcm_message",
    "map_webpush_status", "map_fcm_response",
    # worker
    "Worker", "RunReport", "SimClock", "MemoryStore", "OutboxPort", "drain",
    # api
    "AlertService", "alert_id_for", "parse_priority",
    "set_sender", "get_sender", "installed_senders", "clear_senders",
    # validate
    "run_gate", "GateReport", "Problem", "DEFECTS",
]
