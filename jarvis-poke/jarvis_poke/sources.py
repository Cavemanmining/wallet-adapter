"""The polling scheduler: what to look at next, and when it is allowed to.

Implements the "Fetching, kept polite by construction" section of
:mod:`jarvis_poke.contracts` -- :class:`~jarvis_poke.contracts.FetchPolicy`,
:class:`~jarvis_poke.contracts.FetchResult`,
:class:`~jarvis_poke.contracts.Fetcher` and
:class:`~jarvis_poke.contracts.Parser` -- and reads the ``policy`` blocks of
``jarvis_poke/data/sources.json`` whose ``skus`` blocks
:mod:`jarvis_poke.catalog` reads.

The package opens no sockets.  :meth:`PollScheduler.poll_once` takes a
:class:`~jarvis_poke.contracts.Fetcher` and a
:class:`~jarvis_poke.contracts.Parser` as arguments, exactly as
``jarvis_alerts`` takes an injected sender, so everything here is testable
without a network and no retailer's page structure lives in this package.

Politeness is code, not a paragraph
-----------------------------------
Every rule contracts.py states is enforced here, in the one place a caller
cannot route around -- :meth:`poll_once` re-checks all of them itself, so a
caller that skips :meth:`due` still cannot hammer a host:

* **Per-host minimum interval.** ``min_interval_s`` is a property of the
  *host*, not of one listing ("never poll a host faster than this").  So
  the gate is per source: after any attempt on a source, no SKU of that
  source is due again until the interval has passed.  :meth:`due`
  therefore returns *at most one SKU per source* -- the most overdue one
  -- because returning four listings of one shop at one instant is an
  invitation to poll them at one instant.  Listings of a host rotate:
  the least recently looked at is the most overdue, so it goes first.
  ``FetchPolicy`` itself refuses an interval under 30s, and
  :func:`load_policies` holds config files to a higher floor still.
* **robots.txt.** A source whose ``robots_allows`` is false is never
  returned by :meth:`due` and never fetched by :meth:`poll_once`.  It
  stays in the catalog so a person can still be handed its deep link.
  The app sets that flag from its own robots.txt check.
* **Conditional requests.** :meth:`conditional_headers` sends
  ``If-None-Match`` / ``If-Modified-Since`` as soon as a source has given
  us a validator, so a repeat poll costs the host a 304 and no body.  A
  304 is a *successful* attempt that yields no observation.
* **Widening backoff.** Consecutive errors double the interval
  (``min_interval_s * 2**(errors-1)``, capped at
  :data:`BACKOFF_MAX_DOUBLINGS` doublings) for the whole host, not just
  the listing that failed.
* **Pause.** At ``max_errors_before_pause`` consecutive errors the source
  is paused for ``pause_s``.  A ``FetchResult.retry_after_s`` pauses it
  for at least that long whatever else happened.  A pause applies to
  every SKU of the source and simply expires; it is held apart from the
  interval gate, so :meth:`PollScheduler.resume_source` can lift one by
  hand without also cancelling the interval the host is owed.  A page
  that fetches but will not parse counts as an error too, so an
  unreadable listing backs off instead of being polled at full rate.

The page reads :meth:`PollScheduler.pause_state` and
:meth:`PollScheduler.stats`; both are plain JSON types, including *why*
each source is or is not being polled.

Jitter, and why
---------------
A next-due time is ``now + delay + jitter``, where jitter is drawn from a
:class:`lucifer_gen.seed.Stream` and lies in
``[0, jitter_fraction * min_interval_s)`` -- never negative, so jitter can
only ever make a poll *later* than the policy allows, never earlier.
Without it, every SKU of a host that was first seen in the same pass would
stay in lockstep forever: they would all come due in the same second, and
the host would see a burst every interval instead of a trickle.  The draw
is deterministic: the stream is labelled
``poke.poll:<source>:<product_id>#<n>`` with ``n`` the number of draws
that SKU has already made, so a given seed always produces the same
schedule, a restored snapshot resumes the same sequence, and one SKU's
draws never shift another's.  (``SeedFields.stream`` routes by label
prefix and does not know ``poke``; per its own docstring an unrecognised
label falls back to the whole seed, which is what we want here -- this
schedule is not one of the map-generation stages.)

Overlapping runners
-------------------
The gate is not only per host, it is *atomic* per host.  ``poll_once``
re-reads the stored schedule inside a lock, claims the host's next slot
by pushing its gate out a full interval, saves that, and only then
fetches -- so a second ``poll --once`` that starts while the first is
waiting on the network is told "not due" instead of being handed the
same slot.  The lock is the scheduler's own within a process and the
store's ``lock()`` across processes (:class:`PollStore`); a store
without one gives in-process safety only, which is why
:class:`jarvis_poke.store.SqlitePollStore` has one.  This matters
because the CLI deliberately refuses a long-running poller and tells
you to invoke it repeatedly from outside: overlapping runs are the
expected shape, and N of them used to mean N times the agreed rate,
arriving together.

Determinism and state
---------------------
Nothing here calls ``time.time()`` or ``random``.  The clock is an
injected callable and every decision method takes ``now`` explicitly; the
only randomness is the seed stream above.  Scheduler state (last attempt,
validators, error counts, pauses) lives in :class:`SourceState` /
:class:`SkuState` and can be handed to an injected ``store`` -- any object
with ``load()`` and ``save(snapshot)``, see :class:`PollStore` -- so a
restart does not forget that a host asked to be left alone.
"""

from __future__ import annotations

import email.utils
import json
import math
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import (
    Any, Dict, Iterable, List, Mapping, Optional, Protocol, Set, Tuple, Union,
)

from jarvis_poke.catalog import SOURCES_PATH, Catalog
from jarvis_poke.contracts import (
    FetchPolicy,
    FetchResult,
    Fetcher,
    Observation,
    Parser,
    SourceSku,
)
from lucifer_gen.seed import SeedFields

__all__ = [
    "BACKOFF_MAX_DOUBLINGS",
    "DEFAULT_JITTER_FRACTION",
    "DEFAULT_SEED",
    "MIN_CONFIG_INTERVAL_S",
    "STATE_VERSION",
    "MemoryPollStore",
    "PolicyError",
    "PollScheduler",
    "PollStore",
    "SchedulerError",
    "SkuState",
    "SourceState",
    "SourcesError",
    "load_policies",
]

#: A consecutive-error backoff doubles at most this many times, so a long
#: outage settles at ``min_interval_s * 64`` rather than growing forever.
BACKOFF_MAX_DOUBLINGS = 6

#: Jitter spans this fraction of ``min_interval_s``, added to the next-due
#: time.  Ten percent is enough to break lockstep between the SKUs of one
#: host within a few rounds and small enough that the poll rate is still
#: recognisably the configured one.
DEFAULT_JITTER_FRACTION = 0.10

#: Seed used when the caller does not supply one.  Any fixed value gives a
#: reproducible schedule; this one is the golden-ratio constant.
DEFAULT_SEED = 0x9E3779B97F4A7C15

#: Floor :func:`load_policies` holds a *config file* to.  ``FetchPolicy``
#: enforces an absolute 30s floor; a file that names a host to poll should
#: be politer than the minimum the type allows, and the shipped file sits
#: between 300 and 900.  An app that means it can lower this per call.
MIN_CONFIG_INTERVAL_S = 300.0

#: Bumped only when a snapshot an older reader could misread changes.
STATE_VERSION = 1


class SourcesError(ValueError):
    """Base for everything this module refuses to do."""


class PolicyError(SourcesError):
    """A ``policy`` block in a sources file that cannot be trusted."""


class SchedulerError(SourcesError):
    """The scheduler was asked about a source it has no policy for, or was
    handed a store it cannot use.  Distinct from a fetch failing: this is a
    wiring mistake, not a bad day on the network."""


# --------------------------------------------------------------------------
# policies from the sources file
# --------------------------------------------------------------------------


def load_policies(
    path: Optional[Path] = None,
    *,
    min_interval_floor: float = MIN_CONFIG_INTERVAL_S,
) -> Dict[str, FetchPolicy]:
    """Read the ``policy`` block of every source in ``sources.json``.

    Returns ``{source_id: FetchPolicy}``.  Raises :class:`PolicyError`
    naming the source for anything malformed.  ``min_interval_floor``
    lowers or raises the config floor described at
    :data:`MIN_CONFIG_INTERVAL_S`; it can never go below the 30s floor
    ``FetchPolicy`` enforces itself.
    """
    path = Path(path) if path is not None else SOURCES_PATH
    try:
        with open(path, "r", encoding="utf-8") as handle:
            # ``json`` accepts bare NaN / Infinity by default.  A NaN
            # interval compares False against every floor, so a single
            # token in a config file used to turn the whole politeness
            # layer off without an error.  Refuse them at the door.
            obj = json.load(handle, parse_constant=_refuse_constant)
    except FileNotFoundError:
        raise PolicyError(f"no such sources file: {path}") from None
    except json.JSONDecodeError as exc:
        raise PolicyError(f"{path.name}: not valid JSON ({exc})") from None
    return policies_from_obj(obj, origin=path.name, min_interval_floor=min_interval_floor)


def policies_from_obj(
    obj: Any,
    *,
    origin: str = "sources",
    min_interval_floor: float = MIN_CONFIG_INTERVAL_S,
) -> Dict[str, FetchPolicy]:
    """:func:`load_policies` for an already-parsed object."""
    entries = obj.get("sources", []) if isinstance(obj, dict) else obj
    if not isinstance(entries, list):
        raise PolicyError(f"{origin}: 'sources' must be a list")
    policies: Dict[str, FetchPolicy] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise PolicyError(f"{origin}: source #{index} is not an object")
        source = entry.get("id")
        if not isinstance(source, str) or not source.strip():
            raise PolicyError(f"{origin}: source #{index} has no id")
        source = source.strip()
        if source in policies:
            raise PolicyError(f"{origin}: duplicate source id {source!r}")
        block = entry.get("policy", {})
        if not isinstance(block, dict):
            raise PolicyError(f"source {source!r}: 'policy' must be an object")
        policies[source] = _policy_from_obj(block, source, min_interval_floor)
    return policies


def _refuse_constant(token: str) -> float:
    raise PolicyError(
        f"sources file contains {token}: NaN and Infinity are not intervals, "
        f"and NaN compares False against every floor there is"
    )


def _number(value: Any, name: str, source: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"source {source!r}: {name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        # Every comparison against NaN is False, so a NaN sails through
        # ``< floor`` and through ``<= 0`` alike; an infinity is a gate
        # that never opens.  Neither is a number of seconds.
        raise PolicyError(
            f"source {source!r}: {name} must be a finite number of seconds, got {value!r}"
        )
    return number


def _policy_from_obj(block: Mapping[str, Any], source: str, floor: float) -> FetchPolicy:
    min_interval = _number(block.get("min_interval_s", FetchPolicy.min_interval_s),
                           "min_interval_s", source)
    if min_interval < floor:
        raise PolicyError(
            f"source {source!r}: min_interval_s {min_interval:g} is below the "
            f"configured floor of {floor:g}s -- a listing is not worth being "
            f"someone else's incident"
        )
    robots = block.get("robots_allows", True)
    if not isinstance(robots, bool):
        raise PolicyError(f"source {source!r}: robots_allows must be true or false, got {robots!r}")
    max_errors = block.get("max_errors_before_pause", FetchPolicy.max_errors_before_pause)
    if isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 1:
        raise PolicyError(
            f"source {source!r}: max_errors_before_pause must be an integer >= 1, got {max_errors!r}"
        )
    pause_s = _number(block.get("pause_s", FetchPolicy.pause_s), "pause_s", source)
    if pause_s <= 0:
        raise PolicyError(f"source {source!r}: pause_s must be positive, got {pause_s!r}")
    user_agent = block.get("user_agent", FetchPolicy.user_agent)
    if not isinstance(user_agent, str) or not user_agent.strip():
        raise PolicyError(f"source {source!r}: user_agent must be a non-empty string")
    try:
        return FetchPolicy(
            source=source,
            min_interval_s=min_interval,
            robots_allows=robots,
            max_errors_before_pause=max_errors,
            pause_s=pause_s,
            user_agent=user_agent.strip(),
        )
    except ValueError as exc:  # the 30s floor in FetchPolicy.__post_init__
        raise PolicyError(f"source {source!r}: {exc}") from None


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------


@dataclass
class SourceState:
    """Per-host bookkeeping: the rate gate, the pause, the error run."""

    source: str
    last_attempt_at: float = 0.0
    next_due_at: float = 0.0          # the host-level gate
    paused_until: float = 0.0
    pause_reason: str = ""
    #: True when the pause in force was demanded by the host itself (a
    #: ``Retry-After``), so ``resume_source`` will not lift it by hand.
    pause_by_host: bool = False
    consecutive_errors: int = 0
    attempts: int = 0
    ok: int = 0
    not_modified: int = 0
    errors: int = 0
    parse_errors: int = 0
    refusals: int = 0
    pauses: int = 0
    observations: int = 0
    last_status: int = 0
    last_reason: str = ""
    last_error_at: float = 0.0

    def paused(self, now: float) -> bool:
        return self.paused_until > now


@dataclass
class SkuState:
    """Per-listing bookkeeping: its own gate and its conditional validators."""

    source: str
    product_id: str
    last_attempt_at: float = 0.0
    next_due_at: float = 0.0
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    attempts: int = 0
    errors: int = 0
    consecutive_errors: int = 0
    not_modified: int = 0
    observations: int = 0
    last_status: int = 0
    last_outcome: str = ""
    #: How many jitter draws this SKU has made.  Part of the stream label,
    #: so the sequence survives a snapshot/restore unchanged.
    draws: int = 0

    @property
    def key(self) -> Tuple[str, str]:
        return (self.source, self.product_id)


class PollStore(Protocol):
    """Where scheduler state is kept between runs.

    Any object with these two methods will do; the snapshot is plain JSON
    types.  :class:`MemoryPollStore` is the in-process one used by the
    tests and by ``python3 -m jarvis_poke.sources``.
    """

    def load(self) -> Optional[Dict[str, Any]]: ...

    def save(self, snapshot: Dict[str, Any]) -> None: ...

    # Optional.  A store that can exclude *other processes* offers
    # ``lock()``, a re-entrant context manager held across
    # read-decide-claim and again across record-save.  Without it the
    # scheduler is safe within one process and no more, which is not
    # enough: the CLI refuses a long-running poller and tells you to
    # invoke ``poll --once`` from outside, so two runs overlapping is the
    # ordinary case, not the exotic one.
    # def lock(self) -> ContextManager[None]: ...


class MemoryPollStore:
    """A :class:`PollStore` that keeps the snapshot in memory."""

    def __init__(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.snapshot: Optional[Dict[str, Any]] = snapshot
        self.saves = 0

    def load(self) -> Optional[Dict[str, Any]]:
        return self.snapshot

    def save(self, snapshot: Dict[str, Any]) -> None:
        # Copy through JSON so a caller holding the store cannot mutate
        # live scheduler state, and so anything unserialisable is caught
        # here rather than by a real file store later.
        self.snapshot = json.loads(json.dumps(snapshot))
        self.saves += 1


# --------------------------------------------------------------------------
# the scheduler
# --------------------------------------------------------------------------


class PollScheduler:
    """Decides which listing to look at next, and refuses to look early.

    ``catalog``   a :class:`jarvis_poke.catalog.Catalog`; its SKUs are the
                  pollable listings.
    ``policies``  ``{source: FetchPolicy}`` (a sequence of policies is also
                  accepted).  A SKU whose source has no policy is never
                  polled -- unknown rules are not permission.
    ``clock``     callable returning unix seconds.  Used only where ``now``
                  is not passed in; every decision method takes ``now``.
    ``store``     optional :class:`PollStore`; loaded at construction and
                  saved after every state change.
    """

    def __init__(
        self,
        catalog: Catalog,
        policies: Union[Mapping[str, FetchPolicy], Iterable[FetchPolicy]],
        clock,
        store: Optional[PollStore] = None,
        *,
        seed: int = DEFAULT_SEED,
        jitter_fraction: float = DEFAULT_JITTER_FRACTION,
    ) -> None:
        if not callable(clock):
            raise SchedulerError("clock must be a callable returning unix seconds")
        if isinstance(jitter_fraction, bool) or not isinstance(jitter_fraction, (int, float)):
            raise SchedulerError(f"jitter_fraction must be a number, got {jitter_fraction!r}")
        if not math.isfinite(float(jitter_fraction)):
            # NaN passes ``< 0.0``, then ``now + delay + jitter`` is NaN,
            # then ``now < due_at`` is False for ever: the one parameter
            # with an explicit "never polls earlier" guard would have
            # removed the gate entirely.
            raise SchedulerError(
                "jitter_fraction must be a finite number: a NaN jitter makes every "
                "next-due time NaN, and a NaN due time is never in the future"
            )
        if jitter_fraction < 0.0:
            raise SchedulerError("jitter_fraction must not be negative: jitter never polls earlier")
        self.catalog = catalog
        self._clock = clock
        self._seed_fields = SeedFields.parse(int(seed))
        self.jitter_fraction = float(jitter_fraction)
        self._policies: Dict[str, FetchPolicy] = _normalise_policies(policies)
        self._sources: Dict[str, SourceState] = {}
        self._skus: Dict[Tuple[str, str], SkuState] = {}
        #: Guards read-decide-claim and record-save against other threads;
        #: the store's own ``lock()``, when it has one, guards them
        #: against other processes.
        self._lock = threading.RLock()
        self.store = _checked_store(store)
        if self.store is not None:
            snapshot = self.store.load()
            if snapshot:
                self.restore(snapshot)

    # -- policies ----------------------------------------------------------

    @property
    def policies(self) -> Dict[str, FetchPolicy]:
        return dict(self._policies)

    def policy(self, source: str) -> FetchPolicy:
        """The policy for a source, or :class:`SchedulerError`."""
        try:
            return self._policies[source]
        except KeyError:
            raise SchedulerError(
                f"no FetchPolicy for source {source!r}; it will never be polled"
            ) from None

    def set_policy(self, policy: FetchPolicy) -> None:
        """Install or replace one source's policy (the app does this after
        a robots.txt check)."""
        if not isinstance(policy, FetchPolicy):
            raise SchedulerError(f"not a FetchPolicy: {policy!r}")
        self._policies[policy.source] = policy

    # -- what is due -------------------------------------------------------

    def effective_due_at(self, sku: SourceSku) -> float:
        """The earliest time this SKU may be fetched.

        The later of its own next-due, its host's next-due (the per-host
        interval) and its host's pause.
        """
        sku_state = self._skus.get((sku.source, sku.product_id))
        source_state = self._sources.get(sku.source)
        due = sku_state.next_due_at if sku_state else 0.0
        if source_state is not None:
            due = max(due, source_state.next_due_at, source_state.paused_until)
        return due

    def can_poll(self, sku: SourceSku, now: float) -> Tuple[bool, str]:
        """``(allowed, reason)`` -- the single place the rules are applied.

        Both :meth:`due` and :meth:`poll_once` go through this, so there is
        no path that polls a source that robots.txt disallows, a paused
        source, or a host inside its interval.
        """
        policy = self._policies.get(sku.source)
        if policy is None:
            return False, f"no policy for source {sku.source!r}"
        if not policy.robots_allows:
            return False, f"robots.txt disallows {sku.source!r}"
        state = self._sources.get(sku.source)
        if state is not None and state.paused(now):
            remaining = state.paused_until - now
            reason = state.pause_reason or "paused"
            return False, f"{sku.source!r} paused for another {remaining:.0f}s ({reason})"
        due_at = self.effective_due_at(sku)
        if now < due_at:
            return False, f"not due for another {due_at - now:.0f}s"
        return True, "due"

    def overdue_by(self, sku: SourceSku, now: float) -> float:
        """How long this listing has been waiting for its own next look.

        Measured against the *listing's* next-due time, not the host gate.
        The host gate decides *whether* anything on that host may be
        polled; this decides *which* of its listings goes first, and it
        has to be the one waiting longest or a host with more listings
        than its interval allows would starve the tail of the list
        forever (the alphabetically first one would win every round).
        """
        state = self._skus.get((sku.source, sku.product_id))
        return now - (state.next_due_at if state else 0.0)

    def due(self, now: float) -> List[SourceSku]:
        """Pollable listings, most overdue first.

        Never returns a SKU whose source is paused, whose ``robots_allows``
        is false, whose source has no policy, or whose host interval has
        not elapsed.  At most one SKU per source (see the module
        docstring): the interval belongs to the host, so two of its
        listings cannot both be polled at this instant.  Which of a host's
        listings that is comes from :meth:`overdue_by`, so they rotate.
        """
        best: Dict[str, Tuple[float, SourceSku]] = {}
        for sku in self.catalog.skus():
            allowed, _ = self.can_poll(sku, now)
            if not allowed:
                continue
            overdue = self.overdue_by(sku, now)
            current = best.get(sku.source)
            # Longest wait wins; a tie goes to the listing looked at least
            # recently, then to the lower product id, so the order is the
            # same on every machine and every run.
            if current is None or self._rank(sku, overdue) < self._rank(current[1], current[0]):
                best[sku.source] = (overdue, sku)
        ordered = sorted(
            best.values(),
            key=lambda pair: (-pair[0], pair[1].source, pair[1].product_id),
        )
        return [sku for _, sku in ordered]

    def _rank(self, sku: SourceSku, overdue: float) -> Tuple[float, float, str]:
        state = self._skus.get((sku.source, sku.product_id))
        return (-overdue, state.last_attempt_at if state else 0.0, sku.product_id)

    def next_due_at(self, now: Optional[float] = None) -> Optional[float]:
        """When the next pollable listing comes due, or ``None`` if nothing
        will ever be (everything disallowed or unknown)."""
        soonest: Optional[float] = None
        for sku in self.catalog.skus():
            policy = self._policies.get(sku.source)
            if policy is None or not policy.robots_allows:
                continue
            at = self.effective_due_at(sku)
            if soonest is None or at < soonest:
                soonest = at
        return soonest

    # -- conditional requests ---------------------------------------------

    def conditional_headers(self, sku: SourceSku) -> Dict[str, str]:
        """Headers for one fetch: the policy's User-Agent, plus the
        conditional validators *once the source has given us one*.

        A first fetch therefore carries no ``If-None-Match`` /
        ``If-Modified-Since``; every later one does, which is what turns a
        repeat poll into a 304 with no body.
        """
        policy = self.policy(sku.source)
        headers = {"User-Agent": policy.user_agent}
        state = self._skus.get((sku.source, sku.product_id))
        if state is not None:
            if state.etag:
                headers["If-None-Match"] = state.etag
            if state.last_modified:
                headers["If-Modified-Since"] = state.last_modified
        return headers

    # -- recording ---------------------------------------------------------

    def record_attempt(self, sku: SourceSku, result: FetchResult, now: float) -> None:
        """Fold one fetch outcome into the schedule.

        Updates the last-attempt time and both gates, stores ETag /
        Last-Modified, counts consecutive errors, applies the widening
        backoff, pauses the source past ``max_errors_before_pause`` and
        honours ``retry_after_s`` (pausing at least that long).  A 304
        (``not_modified``) is a successful attempt that produces no
        observation.
        """
        self._apply(sku, result, now, count_attempt=True)

    def _apply(
        self,
        sku: SourceSku,
        result: FetchResult,
        now: float,
        *,
        count_attempt: bool,
        parse_error: bool = False,
        resume_errors_from: Optional[Tuple[int, int]] = None,
    ) -> None:
        policy = self.policy(sku.source)
        source = self._source_state(sku.source)
        state = self._sku_state(sku)

        if count_attempt:
            source.attempts += 1
            state.attempts += 1
        source.last_attempt_at = now
        state.last_attempt_at = now
        source.last_status = int(getattr(result, "status", 0) or 0)
        state.last_status = source.last_status

        if result.not_modified:
            # A 304: the host did the cheap thing we asked it to do. It
            # counts as a healthy attempt and yields no new observation.
            source.not_modified += 1
            state.not_modified += 1
            source.consecutive_errors = 0
            state.consecutive_errors = 0
            state.last_outcome = "not_modified"
            source.last_reason = result.reason or "not modified"
            # Servers commonly omit validators on a 304; keep the ones we
            # already have unless fresh ones came back.
            if result.etag:
                state.etag = result.etag
            if result.last_modified:
                state.last_modified = result.last_modified
            delay = policy.min_interval_s
        elif result.ok:
            source.ok += 1
            source.consecutive_errors = 0
            state.consecutive_errors = 0
            state.last_outcome = "ok"
            source.last_reason = result.reason or "ok"
            # A fresh 200 replaces the validators outright, *including*
            # clearing them: a body served without an ETag has none, and
            # sending a stale one back would be a lie about what we hold.
            state.etag = result.etag
            state.last_modified = result.last_modified
            delay = policy.min_interval_s
        else:
            source.errors += 1
            state.errors += 1
            if parse_error:
                source.parse_errors += 1
            if resume_errors_from is None:
                source.consecutive_errors += 1
                state.consecutive_errors += 1
            else:
                # A parse failure follows a fetch this method has already
                # recorded as a success, which cleared the error run. The
                # run it cleared is handed back here and continued, so a
                # page that fetches perfectly and never parses still backs
                # off and still pauses -- otherwise it would be polled at
                # full rate forever.
                source.consecutive_errors = resume_errors_from[0] + 1
                state.consecutive_errors = resume_errors_from[1] + 1
            source.last_error_at = now
            source.last_reason = result.reason or "error"
            state.last_outcome = "error"
            doublings = min(source.consecutive_errors - 1, BACKOFF_MAX_DOUBLINGS)
            delay = policy.min_interval_s * float(2 ** doublings)
            if source.consecutive_errors >= policy.max_errors_before_pause:
                self._pause(
                    source,
                    now + policy.pause_s,
                    f"{source.consecutive_errors} consecutive errors",
                )

        retry_after = getattr(result, "retry_after_s", None)
        if retry_after is not None:
            wait = _retry_after_seconds(retry_after, now)
            if wait is None:
                # A host asked to be left alone and we could not read how
                # long for.  Silently ignoring that -- which is what an
                # unparseable value, an HTTP-date or a NaN used to do --
                # is the one mistake this module exists to prevent, so
                # fall back to the policy's own pause and say why.
                self._pause(
                    source,
                    now + policy.pause_s,
                    "retry-after we could not read; pausing the policy's "
                    f"{policy.pause_s:.0f}s instead",
                )
            elif wait > 0.0:
                # "at least that long": never shorten an existing pause.
                self._pause(source, now + wait, f"retry-after {wait:.0f}s", host_asked=True)

        jitter = self._jitter(state, policy)
        next_at = now + delay + jitter
        state.next_due_at = next_at
        source.next_due_at = next_at
        self._save()

    def _pause(
        self, source: SourceState, until: float, reason: str, *, host_asked: bool = False
    ) -> None:
        """Extend a pause, never shorten one.

        The pause is kept separate from ``next_due_at`` (the interval
        gate) rather than folded into it: :meth:`effective_due_at` takes
        the later of the two anyway, and keeping them apart means
        :meth:`resume_source` can lift a pause without also cancelling the
        interval the host is owed since its last attempt.

        ``host_asked`` marks a pause the *host* demanded (a
        ``Retry-After``).  That is not ours to lift by hand --
        :meth:`resume_source` refuses it unless the caller says
        ``force=True`` -- because a 24-hour Retry-After we cancel after
        five minutes is the request storm the header was sent to prevent.
        """
        if not math.isfinite(float(until)):
            raise SchedulerError(
                f"cannot pause {source.source!r} until {until!r}: a non-finite pause "
                f"is never installed, which is no pause at all"
            )
        if until > source.paused_until:
            source.paused_until = until
            source.pause_reason = reason
            source.pause_by_host = bool(host_asked)
            source.pauses += 1
        elif host_asked and source.paused_until > 0.0:
            # A shorter Retry-After than the pause already in force still
            # says the host asked; keep the longer pause but remember who
            # wants it.
            source.pause_by_host = True

    def pause_source(self, source: str, until: float, reason: str = "paused by hand") -> None:
        """Pause a source until ``until`` (the page's "leave them alone"
        button, and how an app reacts to a notice outside a fetch)."""
        self.policy(source)
        self._pause(self._source_state(source), until, reason)
        self._save()

    def resume_source(self, source: str, *, force: bool = False) -> None:
        """Clear a pause early.  The interval gate still applies.

        A pause the *host* asked for is not cleared: ``Retry-After: 86400``
        means the host wants a day, and a resume five minutes later is
        exactly the burst the header was sent to stop.  Pass
        ``force=True`` to override it anyway -- an explicit, auditable
        decision by a person -- and a :class:`SchedulerError` names the
        header otherwise.
        """
        state = self._source_state(source)
        if state.pause_by_host and state.paused(self._clock()) and not force:
            raise SchedulerError(
                f"{source!r} is paused because the host asked "
                f"({state.pause_reason or 'retry-after'}); that is not ours to "
                f"lift. Pass force=True if you mean to ignore it."
            )
        state.paused_until = 0.0
        state.pause_reason = ""
        state.pause_by_host = False
        self._save()

    def retime_source(self, source: str, now: float) -> int:
        """Re-derive one source's next-due times from its *current* policy.

        ``set_policy`` replaces the interval, but every gate already on the
        books was computed at the last poll under the *old* one.  Tighten a
        policy from 300s to 30s at 11:59 and nothing happens until the 300s
        already ticking runs out -- the drop is over by then.  Relax it and
        the host keeps being polled at the old fast rate until each gate
        expires.  This makes the change take effect now, in both
        directions: each gate becomes ``last_attempt_at + min_interval_s``,
        which is what the gate would have been had the policy been in force
        at the last attempt.  Returns how many gates moved.

        What it deliberately does not touch:

        * **Pauses.**  ``paused_until`` is a separate field and
          :meth:`effective_due_at` takes the later of the two, so a host
          that asked for a day with ``Retry-After`` still gets its day.
          A window that tightens polling can never shorten a pause; that
          is the whole reason the pause was kept out of ``next_due_at``.
        * **The floor.**  The new gate is derived from the policy, and
          :class:`~jarvis_poke.contracts.FetchPolicy` refuses an interval
          under 30s at construction, so there is no arithmetic here that
          can produce a faster poll than the floor allows.
        * **A listing never polled.**  ``last_attempt_at == 0`` means we
          have never fetched it; its gate is already open and moving it to
          ``0 + interval`` would be a delay invented out of nothing.

        The jitter from the last poll is dropped, because it was a fraction
        of the old interval.  The next poll draws a fresh one.
        """
        policy = self.policy(source)
        if not math.isfinite(float(now)):
            raise SchedulerError(f"cannot retime {source!r} at a non-finite now: {now!r}")
        moved = 0
        with self._guard():
            state = self._sources.get(source)
            if state is not None and state.last_attempt_at > 0.0:
                target = state.last_attempt_at + policy.min_interval_s
                if target != state.next_due_at:
                    state.next_due_at = target
                    moved += 1
            for (src_name, _product), sku_state in self._skus.items():
                if src_name != source or sku_state.last_attempt_at <= 0.0:
                    continue
                target = sku_state.last_attempt_at + policy.min_interval_s
                if target != sku_state.next_due_at:
                    sku_state.next_due_at = target
                    moved += 1
            if moved:
                self._save()
        return moved

    # -- polling -----------------------------------------------------------

    @contextmanager
    def _guard(self):
        """Hold the schedule still, and start from what is actually stored.

        The gate used to be read once at construction, decided from that
        in-memory copy, and written back after the fetch, with nothing
        between the read and the write.  Two overlapping ``poll --once``
        runs -- a cron entry that overran, a cron entry plus the app --
        therefore each saw a host as due and each fetched it, giving N
        times the agreed rate delivered as one simultaneous burst.  This
        closes that: the state is re-read from the store inside the lock,
        so a decision is made against what every other runner has already
        written.
        """
        with self._lock:
            opener = getattr(self.store, "lock", None) if self.store is not None else None
            if callable(opener):
                with opener():
                    self._refresh()
                    yield
            else:
                self._refresh()
                yield

    def _refresh(self) -> None:
        """Re-read the snapshot, so this instance is not deciding from a
        copy another process has already moved on from."""
        if self.store is None:
            return
        snapshot = self.store.load()
        if snapshot:
            self.restore(snapshot)

    def _claim(self, sku: SourceSku, now: float) -> Tuple[bool, str, Dict[str, str]]:
        """Take this host's next slot, or refuse.

        Returns ``(allowed, reason, headers)``.  When allowed, the host's
        gate is pushed out by a full interval and *saved before the
        fetch*, so a second runner that looks while this one is waiting
        on the network is told "not due" rather than handed the same
        slot.  The fetch itself happens outside the lock -- holding a
        cross-process lock across a network round trip would serialise
        every runner behind the slowest host -- and the real next-due
        time is written when the attempt is recorded.
        """
        with self._guard():
            allowed, reason = self.can_poll(sku, now)
            if not allowed:
                if sku.source in self._policies:
                    self._source_state(sku.source).refusals += 1
                    self._save()
                return False, reason, {}
            headers = self.conditional_headers(sku)
            policy = self.policy(sku.source)
            source = self._source_state(sku.source)
            state = self._sku_state(sku)
            claimed = now + policy.min_interval_s
            source.next_due_at = max(source.next_due_at, claimed)
            state.next_due_at = max(state.next_due_at, claimed)
            source.last_attempt_at = now
            state.last_attempt_at = now
            self._save()
            return True, reason, headers

    def poll_once(
        self,
        sku: SourceSku,
        fetcher: Fetcher,
        parser: Parser,
        now: float,
    ) -> Optional[Observation]:
        """One polite look at one listing, or ``None``.

        Re-checks :meth:`can_poll` first and simply declines (counted as a
        refusal, not an error) if the listing is not pollable right now --
        the rules hold even for a caller that never asked :meth:`due`.
        Otherwise it builds the conditional headers, calls the *injected*
        fetcher, records the attempt, and on a fresh 200 hands the body to
        the *injected* parser.  It never opens a socket and never parses a
        page itself.

        A fetcher or parser that raises is recorded as an error for the
        source -- so it feeds the backoff and the pause like any other
        failure -- and ``None`` is returned; the exception does not
        propagate, because one broken listing must not stop a pass.  Only
        the exception's *type name* is kept, never its message, which may
        quote a URL or a page.
        """
        allowed, reason, headers = self._claim(sku, now)
        if not allowed:
            return None

        policy = self.policy(sku.source)
        try:
            result = fetcher(sku.url, headers, policy)
        except Exception as exc:  # noqa: BLE001 - deliberate: one listing must not kill a pass
            with self._guard():
                self._apply(
                    sku,
                    FetchResult(ok=False, reason=f"fetcher raised {type(exc).__name__}"),
                    now,
                    count_attempt=True,
                )
            return None

        if not isinstance(result, FetchResult):
            with self._guard():
                self._apply(
                    sku,
                    FetchResult(ok=False, reason=f"fetcher returned {type(result).__name__}"),
                    now,
                    count_attempt=True,
                )
            return None

        with self._guard():
            prior_errors = (
                self._source_state(sku.source).consecutive_errors,
                self._sku_state(sku).consecutive_errors,
            )
            self.record_attempt(sku, result, now)
        if result.not_modified or not result.ok:
            return None

        try:
            observation = parser(sku, result.body, now)
        except Exception as exc:  # noqa: BLE001 - same reason as above
            # The fetch was fine and is already recorded as such; this is a
            # second, parse-side failure at the same instant. It does not
            # count as another attempt, but it does count as an error, so a
            # page we cannot read backs off exactly like one we cannot
            # reach.
            with self._guard():
                self._apply(
                    sku,
                    FetchResult(ok=False, reason=f"parser raised {type(exc).__name__}"),
                    now,
                    count_attempt=False,
                    parse_error=True,
                    resume_errors_from=prior_errors,
                )
            return None

        if observation is None:
            return None
        if not isinstance(observation, Observation):
            with self._guard():
                self._apply(
                    sku,
                    FetchResult(ok=False, reason=f"parser returned {type(observation).__name__}"),
                    now,
                    count_attempt=False,
                    parse_error=True,
                    resume_errors_from=prior_errors,
                )
            return None

        with self._guard():
            state = self._sku_state(sku)
            state.observations += 1
            self._source_state(sku.source).observations += 1
            self._save()
        return observation

    # -- views for the page ------------------------------------------------

    def host_conflicts(self) -> Dict[str, List[str]]:
        """``{hostname: [source ids]}`` for any host two sources share.

        ``min_interval_s`` is a property of the *host* -- contracts.py,
        "never poll a host faster than this" -- but a
        :class:`~jarvis_poke.contracts.FetchPolicy`, a robots.txt result
        and the gate that enforces them are all keyed by *source id*.
        Two source ids whose listings point at one hostname therefore get
        one interval each and hit that host at twice the agreed rate,
        with nothing in the schedule looking wrong.

        This module cannot silently merge them -- each has its own
        robots.txt answer and its own policy, and deciding they are the
        same host is a judgement about someone else's infrastructure --
        but it can refuse to let the situation be invisible.  The CLI's
        ``sources`` command prints this, and an app should treat a
        non-empty answer as a configuration error.
        """
        hosts: Dict[str, Set[str]] = {}
        for sku in self.catalog.skus():
            if sku.source not in self._policies:
                continue
            url = str(getattr(sku, "url", "") or "")
            host = url.split("//", 1)[-1].split("/", 1)[0].split("@")[-1].lower()
            if not host:
                continue
            hosts.setdefault(host, set()).add(sku.source)
        return {
            host: sorted(sources)
            for host, sources in sorted(hosts.items())
            if len(sources) > 1
        }

    def pause_state(self, now: Optional[float] = None) -> Dict[str, Dict[str, Any]]:
        """Per-source: is it paused, until when, why, and may it be polled
        at all.  JSON-ready, for the page and the CLI."""
        at = self._clock() if now is None else now
        out: Dict[str, Dict[str, Any]] = {}
        for source in sorted(set(self._policies) | set(self._sources)):
            policy = self._policies.get(source)
            state = self._sources.get(source) or SourceState(source=source)
            paused = state.paused(at)
            out[source] = {
                "source": source,
                "paused": paused,
                "paused_until": state.paused_until,
                "seconds_remaining": max(0.0, state.paused_until - at) if paused else 0.0,
                "reason": state.pause_reason if paused else "",
                "by_host": bool(state.pause_by_host) if paused else False,
                "pauses": state.pauses,
                "consecutive_errors": state.consecutive_errors,
                "last_error_at": state.last_error_at,
                "last_reason": state.last_reason,
                "robots_allows": bool(policy.robots_allows) if policy else False,
                "min_interval_s": policy.min_interval_s if policy else None,
                "pollable": bool(policy and policy.robots_allows and not paused),
                "blocked_reason": (
                    "" if policy and policy.robots_allows and not paused
                    else ("no policy" if policy is None
                          else "robots.txt disallows" if not policy.robots_allows
                          else "paused")
                ),
            }
        return out

    def stats(self, now: Optional[float] = None) -> Dict[str, Any]:
        """Counters for the page: totals, per source and per listing.

        Everything is a JSON type; nothing here is a live object, so the
        page can serialise it straight out.
        """
        at = self._clock() if now is None else now
        sources: Dict[str, Dict[str, Any]] = {}
        for source in sorted(set(self._policies) | set(self._sources)):
            state = self._sources.get(source) or SourceState(source=source)
            policy = self._policies.get(source)
            row = asdict(state)
            row["paused"] = state.paused(at)
            row["min_interval_s"] = policy.min_interval_s if policy else None
            row["robots_allows"] = bool(policy.robots_allows) if policy else False
            row["skus"] = len(self.catalog.skus_from(source))
            sources[source] = row

        skus: List[Dict[str, Any]] = []
        due_now = 0
        for sku in self.catalog.skus():
            state = self._skus.get((sku.source, sku.product_id))
            allowed, reason = self.can_poll(sku, at)
            due_now += 1 if allowed else 0
            row = asdict(state) if state else asdict(SkuState(sku.source, sku.product_id))
            row["url"] = sku.url
            row["sku"] = sku.sku
            row["effective_due_at"] = self.effective_due_at(sku)
            row["due"] = allowed
            row["status"] = reason
            # Validators are metadata, not secrets, but only their presence
            # is interesting on a page.
            row["has_validator"] = bool(row.get("etag") or row.get("last_modified"))
            skus.append(row)

        totals = {
            "attempts": sum(s["attempts"] for s in sources.values()),
            "ok": sum(s["ok"] for s in sources.values()),
            "not_modified": sum(s["not_modified"] for s in sources.values()),
            "errors": sum(s["errors"] for s in sources.values()),
            "parse_errors": sum(s["parse_errors"] for s in sources.values()),
            "refusals": sum(s["refusals"] for s in sources.values()),
            "pauses": sum(s["pauses"] for s in sources.values()),
            "observations": sum(s["observations"] for s in sources.values()),
            "sources": len(sources),
            "skus": len(skus),
            "due_now": due_now,
            "paused_sources": sum(1 for s in sources.values() if s["paused"]),
            "disallowed_sources": sum(1 for s in sources.values() if not s["robots_allows"]),
        }
        return {
            "at": at,
            "totals": totals,
            "sources": sources,
            "skus": skus,
            "next_due_at": self.next_due_at(at),
            "jitter_fraction": self.jitter_fraction,
            "host_conflicts": self.host_conflicts(),
            "seed": self._seed_fields.raw,
        }

    # -- persistence -------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Everything a restart needs to stay as polite as it was."""
        return {
            "version": STATE_VERSION,
            "sources": {name: asdict(state) for name, state in sorted(self._sources.items())},
            "skus": [asdict(state) for _, state in sorted(self._skus.items())],
        }

    def restore(self, snapshot: Mapping[str, Any]) -> None:
        """Load a :meth:`snapshot`.  Unknown keys are ignored so an older
        file still opens; a newer ``version`` is refused rather than
        half-read."""
        version = snapshot.get("version", STATE_VERSION)
        if not isinstance(version, int) or version > STATE_VERSION:
            raise SchedulerError(
                f"poll state version {version!r} is newer than this build understands "
                f"(version {STATE_VERSION})"
            )
        self._sources = {}
        self._skus = {}
        for name, row in (snapshot.get("sources") or {}).items():
            self._sources[name] = _from_row(SourceState, dict(row, source=name))
        for row in snapshot.get("skus") or []:
            state = _from_row(SkuState, row)
            self._skus[(state.source, state.product_id)] = state

    def save(self) -> None:
        """Write the snapshot to the store, if there is one."""
        self._save()

    def _save(self) -> None:
        if self.store is not None:
            self.store.save(self.snapshot())

    # -- internals ---------------------------------------------------------

    def _source_state(self, source: str) -> SourceState:
        state = self._sources.get(source)
        if state is None:
            state = SourceState(source=source)
            self._sources[source] = state
        return state

    def _sku_state(self, sku: SourceSku) -> SkuState:
        key = (sku.source, sku.product_id)
        state = self._skus.get(key)
        if state is None:
            state = SkuState(source=sku.source, product_id=sku.product_id)
            self._skus[key] = state
        return state

    def _jitter(self, state: SkuState, policy: FetchPolicy) -> float:
        """A non-negative spread on the next-due time (module docstring).

        Labelled with the draw count, so the sequence is reproducible for a
        seed and survives snapshot/restore. Never negative: jitter may only
        delay a poll, never bring one forward inside the interval.
        """
        if self.jitter_fraction <= 0.0:
            return 0.0
        label = f"poke.poll:{state.source}:{state.product_id}#{state.draws}"
        state.draws += 1
        return self._seed_fields.stream(label).random() * self.jitter_fraction * policy.min_interval_s

    def __repr__(self) -> str:
        return (
            f"<PollScheduler {len(self.catalog.skus())} skus over "
            f"{len(self._policies)} sources>"
        )


def _retry_after_seconds(value: Any, now: float) -> Optional[float]:
    """``Retry-After`` as a number of seconds from ``now``, or ``None``.

    HTTP allows either delta-seconds or an HTTP-date, and a host that
    sends the date form means it exactly as much as one that sends the
    number.  ``None`` means "the host asked for something we could not
    read", which the caller turns into the policy's own pause rather
    than into silence.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        seconds = float(value)
        return seconds if math.isfinite(seconds) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            seconds = float(text)
        except ValueError:
            pass
        else:
            return seconds if math.isfinite(seconds) else None
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return None
        if when is None:
            return None
        try:
            return max(0.0, when.timestamp() - float(now))
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _normalise_policies(
    policies: Union[Mapping[str, FetchPolicy], Iterable[FetchPolicy]],
) -> Dict[str, FetchPolicy]:
    if isinstance(policies, Mapping):
        items = list(policies.values())
    else:
        items = list(policies)
    out: Dict[str, FetchPolicy] = {}
    for policy in items:
        if not isinstance(policy, FetchPolicy):
            raise SchedulerError(f"not a FetchPolicy: {policy!r}")
        out[policy.source] = policy
    return out


def _checked_store(store: Optional[PollStore]) -> Optional[PollStore]:
    if store is None:
        return None
    if not callable(getattr(store, "load", None)) or not callable(getattr(store, "save", None)):
        raise SchedulerError(
            "store must have load() and save(snapshot) methods "
            f"(see PollStore); got {type(store).__name__}"
        )
    return store


def _from_row(cls, row: Mapping[str, Any]):
    fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in row.items() if k in fields})


# --------------------------------------------------------------------------
# smoke run: a simulated day against a fake fetcher, no network
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    from jarvis_poke.contracts import Stock

    catalog = Catalog.load()
    policies = load_policies()
    clock = {"now": 1_700_000_000.0}
    scheduler = PollScheduler(catalog, policies, lambda: clock["now"], MemoryPollStore())

    # A deliberately dull fake host: every 9th look at a listing fails,
    # every other look with a validator is a 304. Deterministic -- no
    # ``hash()`` (salted per process) and no ``random``.
    seen: Dict[str, int] = {}

    def fetcher(url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
        n = seen[url] = seen.get(url, 0) + 1
        if n % 9 == 0:
            return FetchResult(ok=False, status=503, reason="upstream busy")
        if "If-None-Match" in headers and n % 2 == 0:
            return FetchResult(ok=True, status=304, not_modified=True)
        return FetchResult(ok=True, status=200, body="<fake/>", etag=f'W/"{n}"')

    def parser(sku: SourceSku, body: str, at: float) -> Observation:
        return Observation(
            product_id=sku.product_id, source=sku.source, sku=sku.sku, at=at,
            stock=Stock.IN_STOCK, price=4999, shipping=0, url=sku.url,
        )

    polls: Dict[str, List[float]] = {}
    for _ in range(1440):  # a day at one-minute ticks
        now = clock["now"]
        for sku in scheduler.due(now):
            scheduler.poll_once(sku, fetcher, parser, now)
            polls.setdefault(sku.source, []).append(now)
        clock["now"] = now + 60.0

    print(repr(scheduler))
    for source, times in sorted(polls.items()):
        gaps = [b - a for a, b in zip(times, times[1:])]
        floor = policies[source].min_interval_s
        ok = "ok" if not gaps or min(gaps) >= floor else "VIOLATION"
        print(f"  {source:<12} polls={len(times):<4} min gap={min(gaps) if gaps else 0:.0f}s "
              f"(floor {floor:.0f}s) {ok}")
    for source in sorted(scheduler.policies):
        if source not in polls:
            print(f"  {source:<12} never polled: "
                  f"{scheduler.pause_state(clock['now'])[source]['blocked_reason']}")
    totals = scheduler.stats(clock["now"])["totals"]
    print("  totals:", {k: totals[k] for k in
                        ("attempts", "ok", "not_modified", "errors", "observations", "pauses")})
