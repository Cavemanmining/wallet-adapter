"""Shared types for Jarvis alert delivery.

The symptom this package exists to fix: alerts reach the owner only while the
app is open. That happens when the backend emits to live connections and
nothing else. A live connection is a fine fast path, but it is not delivery.
Delivery needs three things this package provides the server half of:

1. A durable outbox. An alert is written to storage before anyone tries to
   send it, so a closed app, a dropped socket or a crashed worker loses
   nothing. Rows are leased, not popped, so a worker that dies mid-send hands
   the row back after a visibility timeout.
2. A push transport. Sending goes through a subscription registry and a
   transport interface. The transport is injected; this package never holds
   or prints the material a transport needs to authenticate, it only stores
   each device's opaque subscription blob as the app's own data.
3. Idempotency. Retries are at-least-once, so every alert carries a dedupe
   key and every delivery attempt is recorded, and the client side is expected
   to collapse duplicates by alert id.

The other half, the service worker that receives the push and shows the
notification while the app is closed, is a client concern; reference files
live in jarvis_alerts/client/.

Retry policy
------------
The constants at the foot of this file are the whole policy.  A transient
failure (``SendResult.retryable``) on attempt ``n`` waits
``backoff_seconds(n, jitter)``; after ``MAX_ATTEMPTS`` transient failures
in one budget the row is DEAD with :class:`DeadReason` ``EXHAUSTED``.  That
budget spans about four minutes (``BACKOFF_BASE_S`` doubling up to
``BACKOFF_CAP_S``), which is shorter than an ordinary push-service outage,
so exhaustion is *not* a verdict on the alert: an EXHAUSTED row is put back
to PENDING with a fresh budget, automatically, ``EXHAUSTED_RETRY_COOLDOWN_S``
after it died (``Outbox.revive_exhausted``, which the worker calls at the
start of every pass), for as long as its alert is younger than
``ALERT_MAX_AGE_S``.  An alert older than that is never auto-requeued: its
row stays DEAD as EXHAUSTED, the dead-letter list keeps it, and only the
operator's ``requeue`` brings it back.  Every other dead reason --
``PERMANENT`` (the transport rejected the send for good), ``GONE`` (the
transport reported the subscription dead), ``NO_SUBSCRIPTION`` and
``NO_TRANSPORT`` (the worker had nowhere to send) -- is final for the
automatic path; ``requeue`` is the only way back for those.  A device that
fails ``PRUNE_AFTER_FAILURES`` times in a row is *pruned* for
``PRUNE_COOLDOWN_S``: its rows wait, untouched, and are probed again after.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol


class Priority(enum.IntEnum):
    LOW = 0
    NORMAL = 1
    HIGH = 2


class RowState(enum.Enum):
    PENDING = "pending"        # written, not yet attempted or due for retry
    LEASED = "leased"          # a worker holds it until lease_until
    DELIVERED = "delivered"
    DEAD = "dead"              # gave up: see DeadReason for whether that is final


class DeadReason(enum.Enum):
    """Why a row is DEAD (``OutboxRow.dead_reason``).  Recorded on every
    DEAD transition, cleared when the row leaves DEAD, ``None`` otherwise.
    Only ``EXHAUSTED`` is undone by the outbox itself (see the module
    docstring's retry policy); the rest wait for ``requeue``."""

    EXHAUSTED = "exhausted"                # MAX_ATTEMPTS transient failures in one budget
    PERMANENT = "permanent"                # a non-retryable, non-gone failure
    GONE = "gone"                          # the transport reported the subscription gone
    NO_SUBSCRIPTION = "no_subscription"    # the worker found no live subscription
    NO_TRANSPORT = "no_transport"          # the worker had no transport of that name

    @property
    def revivable(self) -> bool:
        """Whether ``Outbox.revive_exhausted`` may put such a row back."""
        return self is DeadReason.EXHAUSTED


@dataclass(frozen=True)
class Alert:
    """What the app wants the owner to know. Immutable once published."""

    id: str
    profile_id: str
    kind: str                  # e.g. "render_done", "gpu_missing", "portal_open"
    title: str
    body: str
    created_at: float          # unix seconds, supplied by the caller's clock
    priority: Priority = Priority.NORMAL
    dedupe_key: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Subscription:
    """One device's push endpoint, stored as the app's own opaque data."""

    profile_id: str
    device_id: str
    transport: str             # "webpush" | "fcm" | "fake"
    # ``repr=False``: the blob is "never logged", so it must not ride along in
    # ``repr(sub)``, ``str(sub)``, a log line's ``%r`` or an exception-locals
    # capture.  It is still data: ``asdict``, ``==`` and ``replace`` see it.
    blob: str = field(repr=False)   # opaque JSON the transport understands; never logged
    created_at: float
    failures: int = 0          # consecutive transient failures
    gone: bool = False         # the outbox will not send here: see ``pruned``
    #: ``gone`` was set by the outbox after PRUNE_AFTER_FAILURES consecutive
    #: transient failures, not by the transport.  Such a device is retried
    #: after PRUNE_COOLDOWN_S; a device the transport reported gone (410,
    #: unusable blob) has ``pruned=False`` and waits for a re-registration.
    pruned: bool = False


@dataclass(frozen=True)
class SendResult:
    ok: bool
    retryable: bool = False    # transient: back off and try again
    gone: bool = False         # permanent: the subscription is dead, prune it
    reason: str = ""
    #: Only for a failure the *worker* made up because it had nowhere to
    #: send (``NO_SUBSCRIPTION``, ``NO_TRANSPORT``); a transport leaves it
    #: ``None``.  The outbox derives every other :class:`DeadReason` from
    #: the flags above (:func:`dead_reason_for`) and ignores any other tag.
    dead_reason: Optional[DeadReason] = None


class Transport(Protocol):
    """Anything that can push one alert to one subscription."""

    name: str

    def send(self, subscription: Subscription, alert: Alert) -> SendResult: ...


@dataclass
class OutboxRow:
    """One (alert, subscription) pair on its way out."""

    row_id: int
    alert_id: str
    profile_id: str
    device_id: str
    state: RowState
    attempts: int
    next_due: float            # unix seconds; PENDING rows before this are not due
    lease_until: float         # meaningful only in LEASED
    last_reason: str = ""
    dead_reason: Optional[DeadReason] = None   # why it is DEAD; None in any other state
    dead_at: float = 0.0       # when it went DEAD; 0 in any other state
    attempts_base: int = 0     # ``attempts`` at the last re-queue: the budget counts from here


@dataclass(frozen=True)
class DeliveryAttempt:
    row_id: int
    attempt: int
    at: float
    ok: bool
    retryable: bool
    gone: bool
    reason: str


#: Retry policy. Attempt n waits base * 2**(n-1) seconds, capped, plus jitter.
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 300.0
MAX_ATTEMPTS = 8
LEASE_S = 30.0
#: Consecutive transient failures after which a subscription is treated as gone.
PRUNE_AFTER_FAILURES = 20
#: How long a subscription pruned that way stays gone before the outbox
#: probes it again (``Subscription.pruned``).  A transport-reported gone
#: never expires; only a re-registration clears it.
PRUNE_COOLDOWN_S = 300.0
#: A row DEAD as ``DeadReason.EXHAUSTED`` is put back to PENDING with a
#: fresh budget of MAX_ATTEMPTS, automatically, once this many seconds have
#: passed since it died (``Outbox.revive_exhausted``; the worker calls it
#: every pass).  Longer than the budget it follows, so a service that is
#: down is probed at most once per cooldown, not hammered.
EXHAUSTED_RETRY_COOLDOWN_S = 900.0
#: ... but never for an alert older than this (``Alert.created_at`` against
#: the clock at revival time).  Such a row stays DEAD as EXHAUSTED, the
#: dead-letter list keeps it, and only ``requeue`` brings it back.
ALERT_MAX_AGE_S = 86400.0


def backoff_seconds(attempt: int, jitter: float) -> float:
    """Delay before attempt ``attempt`` (1-based). ``jitter`` is in [0, 1)."""
    if attempt < 1:
        raise ValueError("attempt is 1-based")
    raw = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2 ** (attempt - 1)))
    return raw * (0.5 + 0.5 * jitter)


def dead_reason_for(result: SendResult, budget_spent: bool) -> Optional[DeadReason]:
    """The :class:`DeadReason` a failed ``result`` gives its row, or ``None``
    when the row does not die: ``GONE`` for a gone result; ``EXHAUSTED``
    for a retryable one when ``budget_spent`` (this is the MAX_ATTEMPTS-th
    transient failure of the budget), else ``None``; the result's own
    ``NO_SUBSCRIPTION`` / ``NO_TRANSPORT`` tag for a failure the worker
    made up; ``PERMANENT`` for any other non-retryable failure.  An ok
    result gives ``None``."""
    if result.ok:
        return None
    if result.gone:
        return DeadReason.GONE
    if result.retryable:
        return DeadReason.EXHAUSTED if budget_spent else None
    if result.dead_reason in (DeadReason.NO_SUBSCRIPTION, DeadReason.NO_TRANSPORT):
        return result.dead_reason
    return DeadReason.PERMANENT


__all__ = [
    "Priority", "RowState", "DeadReason", "Alert", "Subscription", "SendResult", "Transport",
    "OutboxRow", "DeliveryAttempt", "BACKOFF_BASE_S", "BACKOFF_CAP_S",
    "MAX_ATTEMPTS", "LEASE_S", "PRUNE_AFTER_FAILURES", "PRUNE_COOLDOWN_S",
    "EXHAUSTED_RETRY_COOLDOWN_S", "ALERT_MAX_AGE_S", "backoff_seconds", "dead_reason_for",
]
