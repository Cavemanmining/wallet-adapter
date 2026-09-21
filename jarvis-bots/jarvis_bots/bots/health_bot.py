"""Application health and deploy verification, as a bot on the framework.

The app this watches is a local backend plus a Firebase-hosted frontend,
shipped by a versioned deploy script.  Two things have actually gone wrong
during the project, and neither of them is what a plain uptime monitor
looks at:

1. **A deploy that succeeds and ships nothing.**  The build silently left
   the new assets out, the deploy reported success, hosting kept serving
   the previous bundle, and everything was "up".  The only honest question
   is *is the version that is live the version that was deployed*, so
   :meth:`HealthBot.deployed` records what was just shipped and
   :data:`VERSION_DRIFT_KEY` is a first-class rule rather than a footnote
   on a status page.
2. **A frontend that is up and cannot reach its backend.**  The page loads,
   hosting answers 200 on every path, and the person sees an empty screen
   with a spinner.  Two green checks and one broken app.  The pairing is
   declared as configuration (``pairs``), not guessed, and produces its own
   :data:`ORPHANED_KEY` alert that says the page is up but cannot reach its
   data.

Everything else here follows from those two.  An 'asset' check exists
because Firebase hosting answers a missing ``.js`` with ``index.html`` and
a 200 (see :func:`_asset_fault`), so "status 200" is not evidence that the
asset shipped.  A critical check must fail twice before it is an ACTION,
because one blip in the middle of the night is not a decision anyone needs
to make.

What this module does not do
----------------------------
It opens no sockets.  ``probe(check) -> CheckResult`` is injected, exactly
as :class:`~jarvis_bots.bots.poke_bot.PokeBot` takes its fetcher, so the
whole rule engine is testable against a fixed sequence of rounds with no
network, no wall clock and no sleeping.  :func:`urllib_probe` is the real
implementation the *app* injects; it is the only thing in this file that
touches the network or reads a clock, and the bot never calls it itself.

Events, and the badge
---------------------
``jarvis_bots.contracts``: "Events are the only output", and the badge
"counts distinct open keys, so one restock nagging across ten ticks is one
item of attention, not ten".  Every rule below has a stable key, raises at
ACTION once while the fault stands, and closes the key on recovery by
re-raising it below ACTION with ``resolved=True`` -- the convention
``jarvis_bots/templates/bot.py.tmpl`` states ("a supervisor that closes
keys on the flag and one that closes them on the severity agree") and
:meth:`jarvis_bots.bots.poke_bot.PokeBot._resolve` uses.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

from jarvis_bots.base import BaseBot, Clock
from jarvis_bots.contracts import BotInfo, BotState, BotStatus, Event, Severity

__all__ = [
    "BOT_ID",
    "INFO",
    "CHECK_KINDS",
    "Check",
    "CheckResult",
    "HealthBot",
    "HealthBotError",
    "BODY_EXCERPT_CHARS",
    "MAX_EXCERPT_CHARS",
    "MAX_BODY_BYTES",
    "DEFAULT_ASSET_FLOOR_BYTES",
    "DEFAULT_LATENCY_MS",
    "CRITICAL_FAILS_FOR_ACTION",
    "NONCRITICAL_FAILS_FOR_ACTION",
    "DRIFT_TICKS_FOR_ACTION",
    "LATENCY_TICKS_FOR_NOTICE",
    "VERSION_DRIFT_KEY",
    "ORPHANED_KEY",
    "down_key",
    "asset_key",
    "latency_key",
    "build_request",
    "urllib_probe",
    "parse_version",
]

BOT_ID = "health"

INFO = BotInfo(
    id=BOT_ID,
    name="App health",
    blurb="Can the frontend actually get an answer, and is the live version the one you deployed.",
    kind="radar",
    interval_s=180.0,
    href="/bots/health",
)

#: The four things a check can be looking at.  ``kind`` is not decoration:
#: 'asset' turns on the hosting-fallback rule, 'version' feeds the drift
#: rule, and 'frontend'/'backend' are the two halves of a ``pairs`` entry.
CHECK_KINDS: Tuple[str, ...] = ("backend", "frontend", "asset", "version")

#: How much of a response body a probe keeps by default.  Enough to hold a
#: version string or a marker, not enough to put a page in the state file.
BODY_EXCERPT_CHARS = 200
#: A hard ceiling :class:`CheckResult` enforces whatever a probe hands it.
#: ``body_excerpt`` is carried in ``Event.data`` and persisted; a probe with
#: a generous ``excerpt_chars`` should not be able to push a megabyte
#: through the supervisor's state file.
MAX_EXCERPT_CHARS = 4096
#: The most :func:`urllib_probe` will read off a socket.  A bounded read,
#: because "the endpoint answered with a 4 GB stream" must not be the way
#: the health bot takes the app down.
MAX_BODY_BYTES = 64 * 1024

#: Below this many bytes, a 200 from an 'asset' check is not evidence that
#: the asset shipped.  See :func:`_asset_fault`.
DEFAULT_ASSET_FLOOR_BYTES = 512
#: Above this, a check is slow enough to be worth a line (not an alarm).
DEFAULT_LATENCY_MS = 2000.0

#: Consecutive failures before a *critical* check is a decision.  Two, so
#: one blip is not an alarm.
CRITICAL_FAILS_FOR_ACTION = 2
#: Consecutive failures before a *non-critical* check is a decision.  Below
#: this it is a NOTICE: a line in the feed, no badge item.
NONCRITICAL_FAILS_FOR_ACTION = 3
#: Consecutive ticks of a live version that is not the deployed one before
#: that is a decision.
DRIFT_TICKS_FOR_ACTION = 2
#: Consecutive slow ticks before latency is worth a line.
LATENCY_TICKS_FOR_NOTICE = 3

#: "The deploy said it worked and the old bundle is still live."
VERSION_DRIFT_KEY = "health:version-drift"
#: "The page is up and cannot reach its data."  One key: however many pairs
#: are broken, that is one condition of the app and one decision.
ORPHANED_KEY = "health:frontend-orphaned"


def down_key(name: str) -> str:
    """The stable key for "this check is not answering"."""
    return f"health:down:{name}"


def asset_key(name: str) -> str:
    """The stable key for "this asset answered 200 and is not the asset"."""
    return f"health:asset:{name}"


def latency_key(name: str) -> str:
    """The stable key for "this check is slow"."""
    return f"health:slow:{name}"


class HealthBotError(Exception):
    """A wiring mistake: a bad check, a bad pair, a probe that is not
    callable.  Raised at construction, never during a tick."""


# ---------------------------------------------------------------------------
# What a check is, and what came back
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Check:
    """One thing to ask, declared once.

    ``critical``  a failure here is the app being down for somebody, so it
                  earns an ACTION after :data:`CRITICAL_FAILS_FOR_ACTION`
                  consecutive failures.  A non-critical check is a NOTICE
                  until :data:`NONCRITICAL_FAILS_FOR_ACTION`.

    ``expect_contains``  the marker that proves the answer is the *right*
                  answer.  For an 'asset' check this is the thing that
                  separates "hosting served me the bundle" from "hosting
                  served me index.html with a 200"; for a 'version' check
                  it is optional, because the body is parsed for the live
                  version anyway.
    """

    name: str
    url: str
    kind: str
    expect_status: int = 200
    expect_contains: Optional[str] = None
    timeout_s: float = 10.0
    critical: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise HealthBotError("a check needs a name; it is the attention key")
        if not isinstance(self.url, str) or not self.url.strip():
            raise HealthBotError(f"check {self.name!r} needs a url")
        if self.kind not in CHECK_KINDS:
            raise HealthBotError(
                f"check {self.name!r} has kind {self.kind!r}; expected one of "
                f"{', '.join(CHECK_KINDS)}"
            )
        try:
            status = int(self.expect_status)
        except (TypeError, ValueError):
            raise HealthBotError(
                f"check {self.name!r} expect_status must be an int; got "
                f"{self.expect_status!r}"
            ) from None
        object.__setattr__(self, "expect_status", status)
        if self.expect_contains is not None and not isinstance(self.expect_contains, str):
            raise HealthBotError(
                f"check {self.name!r} expect_contains must be a string or None"
            )
        try:
            timeout = float(self.timeout_s)
        except (TypeError, ValueError):
            raise HealthBotError(
                f"check {self.name!r} timeout_s must be seconds; got {self.timeout_s!r}"
            ) from None
        if not timeout > 0.0:
            raise HealthBotError(f"check {self.name!r} timeout_s must be positive")
        object.__setattr__(self, "timeout_s", timeout)
        object.__setattr__(self, "critical", bool(self.critical))


@dataclass(frozen=True)
class CheckResult:
    """What one probe call came back with.

    ``ok``  the probe's own verdict: the status was the expected one and
            any ``expect_contains`` marker was there.  The bot does not
            re-derive it, because the probe saw the whole body and the bot
            only ever sees a bounded excerpt of it.

    ``body_bytes``  how many bytes of body the probe read, which is the one
            number the asset-presence rule turns on and the one thing
            ``body_excerpt`` cannot carry (a 300 byte fallback page and a
            300 kB bundle have the same first 200 characters).  ``None``
            means the probe did not say, and the excerpt's length is used.
    """

    name: str
    ok: bool
    status: Optional[int] = None
    elapsed_ms: float = 0.0
    body_excerpt: str = ""
    error: str = ""
    body_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "ok", bool(self.ok))
        if self.status is not None:
            object.__setattr__(self, "status", int(self.status))
        object.__setattr__(self, "elapsed_ms", float(self.elapsed_ms or 0.0))
        excerpt = self.body_excerpt if isinstance(self.body_excerpt, str) else ""
        # Bounded, always: this string is carried in Event.data and lands in
        # the supervisor's state file.
        object.__setattr__(self, "body_excerpt", excerpt[:MAX_EXCERPT_CHARS])
        object.__setattr__(self, "error", str(self.error or ""))
        if self.body_bytes is not None:
            object.__setattr__(self, "body_bytes", max(0, int(self.body_bytes)))

    @property
    def size(self) -> int:
        """The body's length in bytes, as well as it is known."""
        if self.body_bytes is not None:
            return self.body_bytes
        return len(self.body_excerpt.encode("utf-8", "replace"))


#: What the bot calls, once per configured check.
Probe = Callable[[Check], CheckResult]
#: How a 'version' check's body becomes a version string.
VersionParser = Callable[[str], Optional[str]]


# ---------------------------------------------------------------------------
# The real probe, which is the thing the app injects
# ---------------------------------------------------------------------------


_VERSION_KEYS = ("version", "build", "revision", "commit", "buildId", "build_id")
_VERSION_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+\-]{0,63}")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse to follow redirects, so a redirect is *reported*.

    A silently followed redirect is how "the frontend is up" stays true
    after hosting starts bouncing the app to a login wall or a parked
    domain: urllib follows it, the final hop answers 200, and the check is
    green.  Returning ``None`` here makes urllib raise the 3xx as an
    ``HTTPError``, which :func:`urllib_probe` turns into a failure naming
    the Location it wanted to send us to.
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def build_request(check: Check, *, user_agent: str = "jarvis-health/1") -> urllib.request.Request:
    """The GET :func:`urllib_probe` sends, built separately so it can be
    inspected without opening a socket.

    ``Cache-Control: no-cache`` is not politeness: a cached 200 from before
    the deploy is exactly the evidence this bot must never accept.
    """
    if not isinstance(check, Check):
        raise HealthBotError(f"build_request needs a Check; got {type(check).__name__}")
    scheme = check.url.split(":", 1)[0].lower()
    if scheme not in ("http", "https"):
        raise HealthBotError(
            f"check {check.name!r} url must be http or https; got {check.url!r}"
        )
    return urllib.request.Request(
        check.url,
        method="GET",
        headers={
            "User-Agent": user_agent,
            "Accept": "*/*",
            "Cache-Control": "no-cache",
            "Pragma": "no-cache",
        },
    )


def urllib_probe(
    check: Check,
    *,
    excerpt_chars: int = BODY_EXCERPT_CHARS,
    max_body_bytes: int = MAX_BODY_BYTES,
    opener: Optional[Any] = None,
) -> CheckResult:
    """Ask one check over HTTP, with the standard library and nothing else.

    This is the ``probe`` the *app* injects; the bot never calls it.  It is
    therefore the one function in this module that opens a socket and reads
    a clock (``time.monotonic``, for the elapsed measurement only --
    nothing in the bot's own logic reads a clock it was not given).

    Three things it will not do:

    * **hang** -- every call carries ``check.timeout_s``;
    * **follow a redirect quietly** -- see :class:`_NoRedirect`; a 3xx comes
      back as a failure naming the target;
    * **read without a bound** -- at most ``max_body_bytes`` off the
      socket, so a misbehaving endpoint cannot exhaust memory.

    It never raises: every failure is a ``CheckResult`` with ``ok=False``
    and an ``error``, because a probe that raises would be recorded as the
    check failing anyway and this way the reason survives.
    """
    started = time.monotonic()

    def elapsed() -> float:
        return (time.monotonic() - started) * 1000.0

    try:
        request = build_request(check)
    except Exception as exc:  # noqa: BLE001 - a bad url is a config fault, reported
        return CheckResult(
            name=check.name, ok=False, elapsed_ms=elapsed(), error=_why(exc)
        )

    open_url = opener.open if opener is not None else urllib.request.build_opener(
        _NoRedirect
    ).open

    try:
        with open_url(request, timeout=check.timeout_s) as response:
            raw = response.read(int(max_body_bytes))
            status = int(getattr(response, "status", None) or response.getcode() or 0)
    except urllib.error.HTTPError as exc:
        # A 3xx arrives here because redirects are refused, and so does any
        # 4xx/5xx.  Both are answers, so the status is reported.
        try:
            raw = exc.read(int(max_body_bytes))
        except Exception:  # noqa: BLE001 - the body is a nicety, the status is not
            raw = b""
        location = exc.headers.get("Location") if exc.headers else None
        error = f"HTTP {exc.code}"
        if location:
            error += f" redirect to {location} (not followed)"
        return CheckResult(
            name=check.name,
            ok=False,
            status=int(exc.code),
            elapsed_ms=elapsed(),
            body_excerpt=_decode(raw)[:excerpt_chars],
            error=error,
            body_bytes=len(raw),
        )
    except Exception as exc:  # noqa: BLE001 - timeout, DNS, refused, reset, TLS
        return CheckResult(
            name=check.name, ok=False, elapsed_ms=elapsed(), error=_why(exc)
        )

    body = _decode(raw)
    ok = status == check.expect_status
    error = "" if ok else f"expected status {check.expect_status}, got {status}"
    if ok and check.expect_contains and check.expect_contains not in body:
        ok = False
        error = f"body does not contain {check.expect_contains!r}"
    return CheckResult(
        name=check.name,
        ok=ok,
        status=status,
        elapsed_ms=elapsed(),
        body_excerpt=body[:excerpt_chars],
        error=error,
        body_bytes=len(raw),
    )


def _decode(raw: bytes) -> str:
    try:
        return raw.decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 - pragma: no cover
        return ""


def _why(exc: BaseException) -> str:
    try:
        text = f"{type(exc).__name__}: {exc}".strip()
    except BaseException:  # noqa: BLE001 - pragma: no cover
        text = type(exc).__name__
    return text[:200] or type(exc).__name__


def parse_version(body: str) -> Optional[str]:
    """Pull the live version out of a version endpoint's body.

    Handles the two shapes a deploy script actually writes: a JSON object
    with a version-ish key, and a file containing the bare string.  A body
    it cannot read returns ``None``, which the drift rule treats as "no
    evidence" rather than as drift -- claiming the wrong bundle is live
    because a parser got confused would be worse than staying quiet.
    """
    if not isinstance(body, str):
        return None
    text = body.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except Exception:  # noqa: BLE001 - not JSON, try the plain shapes
        parsed = None
    if isinstance(parsed, Mapping):
        for key in _VERSION_KEYS:
            value = parsed.get(key)
            if isinstance(value, (str, int, float)) and str(value).strip():
                return str(value).strip()
        return None
    if isinstance(parsed, (str, int, float)):
        return str(parsed).strip() or None
    first = text.splitlines()[0].strip()
    if 0 < len(first) <= 64:
        match = _VERSION_TOKEN.fullmatch(first)
        if match:
            return first
    return None


# ---------------------------------------------------------------------------
# The bot
# ---------------------------------------------------------------------------


@dataclass
class _CheckState:
    """Per-check running state.  JSON-able, because it is snapshotted."""

    fails: int = 0
    since: float = 0.0
    slow_ticks: int = 0


class HealthBot(BaseBot):
    """Is the app actually working, and is the live build the deployed one.

    ``checks``  the :class:`Check` list, in the order they are probed.
    ``probe``   ``probe(check) -> CheckResult``, injected.  Called once per
                check per tick.  It may raise; that is caught per check and
                recorded as that check failing, so a round completes even
                when every probe raises.
    ``clock``   injected (contracts.py: "Time is injected everywhere").
    ``pairs``   ``{frontend check name: backend check name}``.  Explicit
                configuration, never inferred from kinds: which backend a
                page needs is a fact about the app, and guessing it wrong
                would put the wrong words in an alert.
    ``expected_version``  what the last deploy shipped, if the app already
                knows at construction.  :meth:`deployed` is how it is set
                afterwards.

    Not thread safe, like the supervisor that drives it.
    """

    info = INFO

    def __init__(
        self,
        checks: Iterable[Check],
        probe: Probe,
        clock: Clock,
        *,
        info: Optional[BotInfo] = None,
        pairs: Optional[Mapping[str, str]] = None,
        expected_version: Optional[str] = None,
        asset_floor_bytes: int = DEFAULT_ASSET_FLOOR_BYTES,
        latency_ms: float = DEFAULT_LATENCY_MS,
        version_parser: VersionParser = parse_version,
    ) -> None:
        super().__init__(clock, info)

        self._checks: Tuple[Check, ...] = tuple(checks)
        if not self._checks:
            raise HealthBotError("a health bot with no checks watches nothing")
        seen: Dict[str, Check] = {}
        for check in self._checks:
            if not isinstance(check, Check):
                raise HealthBotError(
                    f"checks must be Check instances; got {type(check).__name__}"
                )
            if check.name in seen:
                raise HealthBotError(
                    f"two checks named {check.name!r}; names are attention keys "
                    f"and must be unique"
                )
            seen[check.name] = check
        self._by_name = seen

        if not callable(probe):
            raise HealthBotError(
                f"probe must be callable(check) -> CheckResult; got "
                f"{type(probe).__name__}.  This bot opens no sockets; inject "
                f"jarvis_bots.bots.health_bot.urllib_probe."
            )
        self._probe = probe

        if not callable(version_parser):
            raise HealthBotError("version_parser must be callable(body) -> str|None")
        self._version_parser = version_parser

        self._pairs: Dict[str, str] = {}
        for frontend, backend in dict(pairs or {}).items():
            if frontend not in seen:
                raise HealthBotError(f"pair names an unknown frontend check: {frontend!r}")
            if backend not in seen:
                raise HealthBotError(f"pair names an unknown backend check: {backend!r}")
            if frontend == backend:
                raise HealthBotError(f"pair {frontend!r} is paired with itself")
            self._pairs[frontend] = backend

        self._asset_floor = max(0, int(asset_floor_bytes))
        self._latency_ms = float(latency_ms)
        if not self._latency_ms > 0.0:
            raise HealthBotError("latency_ms must be positive")

        #: check name -> its running state.
        self._state: Dict[str, _CheckState] = {c.name: _CheckState() for c in self._checks}
        #: attention key -> when it was first raised.  The bot's own record;
        #: the supervisor owns the badge.
        self._open: Dict[str, float] = {}
        #: The version the last deploy said it shipped, and what is live.
        self._expected_version: Optional[str] = (
            str(expected_version) if expected_version else None
        )
        self._live_version: Optional[str] = None
        self._drift_ticks = 0
        self._deployed_at = 0.0

        self._ticks = 0
        self._last_tick_at = 0.0
        self._last_results: Dict[str, CheckResult] = {}

    # -- what the app calls after a deploy ----------------------------------

    def deployed(self, version: str, at: Optional[float] = None) -> None:
        """Record what the deploy script just shipped.

        This is the other half of the drift rule: without it the bot knows
        what is live and has nothing to compare it against, and "deploy
        succeeded but shipped nothing" is invisible.  The streak is reset,
        so the two ticks the rule waits for are two ticks *after this
        deploy*, not two ticks carried over from the last one.
        """
        if not isinstance(version, str) or not version.strip():
            raise HealthBotError("deployed() needs the version that was shipped")
        self._expected_version = version.strip()
        self._drift_ticks = 0
        self._deployed_at = float(self.now() if at is None else at)

    @property
    def expected_version(self) -> Optional[str]:
        return self._expected_version

    @property
    def live_version(self) -> Optional[str]:
        return self._live_version

    def open_attention_keys(self) -> List[str]:
        """The keys this bot believes are open, sorted.  The supervisor owns
        the badge; this is the bot's own record, and the two agree because
        the supervisor closes a key on the resolution this bot raises."""
        return sorted(self._open)

    # -- the work ------------------------------------------------------------

    def tick(self, now: float) -> Sequence[Event]:
        """One round of checks.

        contracts.py: "Do one unit of work.  Must return promptly and must
        not sleep."  Every probe call is isolated, so the round completes
        and every check is judged even if the probe raises on all of them.
        """
        now = float(now)
        self._ticks += 1
        self._last_tick_at = now

        results: Dict[str, CheckResult] = {}
        for check in self._checks:
            results[check.name] = self._run(check)
        self._last_results = results

        events: List[Event] = []
        # Order matters only in that the pairing rule has to know which
        # checks it is speaking for before the per-check rules fire, so the
        # more specific alert wins over the generic one.
        orphaned = self._orphaned_pairs(results)
        events.extend(self._pair_events(orphaned, results, now))
        spoken_for = {backend for _front, backend in orphaned}

        for check in self._checks:
            events.extend(self._check_events(check, results[check.name], now, spoken_for))

        events.extend(self._version_events(results, now))
        return tuple(events)

    def _run(self, check: Check) -> CheckResult:
        """One probe call, never allowed to escape.

        A probe that raises *is* the check failing -- a connection reset and
        a bug in the probe look the same from here, and neither is a reason
        to abandon the other checks in this round.
        """
        try:
            result = self._probe(check)
        except BaseException as exc:  # noqa: BLE001 - deliberate; see the docstring
            return CheckResult(name=check.name, ok=False, error=_why(exc))
        if not isinstance(result, CheckResult):
            return CheckResult(
                name=check.name,
                ok=False,
                error=f"probe returned {type(result).__name__}, not a CheckResult",
            )
        return result

    # -- rule: a check that is not answering ---------------------------------

    def _check_events(
        self,
        check: Check,
        result: CheckResult,
        now: float,
        spoken_for: Iterable[str],
    ) -> List[Event]:
        events: List[Event] = []
        state = self._state.setdefault(check.name, _CheckState())

        fault = _asset_fault(check, result, self._asset_floor)
        if fault:
            # The asset answered -- it is the answer that is wrong.  That is
            # its own rule, and the down rule is not also run for it.  If the
            # check had been unreachable, that question is closed: it is a
            # different fault now, and leaving the old key open would leave
            # two badge items standing for one asset.
            events.extend(
                self._resolve_if_open(
                    down_key(check.name), f"{check.name} is answering again", now
                )
            )
            events.extend(self._asset_events(check, result, fault, now))
            state.fails = 0
            state.since = 0.0
            events.extend(self._latency_events(check, result, now, state))
            return events
        events.extend(self._resolve_if_open(asset_key(check.name),
                                            f"{check.name} is serving real content again",
                                            now))

        if result.ok:
            state.fails = 0
            state.since = 0.0
            events.extend(
                self._resolve_if_open(
                    down_key(check.name),
                    f"{check.name} is answering again"
                    + (f" ({result.status})" if result.status else ""),
                    now,
                )
            )
        else:
            state.fails += 1
            if state.since <= 0.0:
                state.since = now
            events.extend(
                self._down_events(check, result, now, state, spoken_for)
            )

        events.extend(self._latency_events(check, result, now, state))
        return events

    def _down_events(
        self,
        check: Check,
        result: CheckResult,
        now: float,
        state: _CheckState,
        spoken_for: Iterable[str],
    ) -> List[Event]:
        key = down_key(check.name)
        threshold = (
            CRITICAL_FAILS_FOR_ACTION if check.critical else NONCRITICAL_FAILS_FOR_ACTION
        )
        why = result.error or (
            f"status {result.status}" if result.status is not None else "no answer"
        )

        if state.fails >= threshold:
            if check.name in set(spoken_for):
                # The pairing rule already said this, in the words that
                # matter ("the page is up and cannot reach its data").  Two
                # badge items for one fault is one item too many.
                return []
            if key in self._open:
                return []
            self._open[key] = now
            return [
                self.event(
                    Severity.ACTION,
                    f"{check.name} has been failing for {_since(state.since, now, state.fails)}"
                    f" -- {why}",
                    attention_key=key,
                    href=self.info.href,
                    check=check.name,
                    url=check.url,
                    kind=check.kind,
                    critical=check.critical,
                    consecutive_failures=state.fails,
                    failing_since=state.since,
                    status=result.status,
                    error=result.error,
                )
            ]

        if check.critical:
            # One blip is not an alarm and is not a line in the feed either:
            # a critical endpoint that misses a single poll at 3am is noise.
            return []

        # Non-critical, below the threshold: worth a line, not a decision.
        return [
            self.event(
                Severity.NOTICE,
                f"{check.name} failed ({state.fails} in a row) -- {why}",
                attention_key=key,
                href=self.info.href,
                check=check.name,
                url=check.url,
                kind=check.kind,
                consecutive_failures=state.fails,
                status=result.status,
                error=result.error,
            )
        ]

    # -- rule: an asset that answered 200 and is not the asset ---------------

    def _asset_events(
        self, check: Check, result: CheckResult, fault: str, now: float
    ) -> List[Event]:
        key = asset_key(check.name)
        if key in self._open:
            return []
        self._open[key] = now
        return [
            self.event(
                Severity.ACTION,
                f"{check.name} returned {result.status} but is not the asset: {fault}. "
                f"A deploy that built nothing looks exactly like this.",
                attention_key=key,
                href=self.info.href,
                check=check.name,
                url=check.url,
                kind=check.kind,
                status=result.status,
                body_bytes=result.size,
                floor_bytes=self._asset_floor,
                expect_contains=check.expect_contains,
            )
        ]

    # -- rule: the page is up and cannot reach its data ----------------------

    def _orphaned_pairs(
        self, results: Mapping[str, CheckResult]
    ) -> List[Tuple[str, str]]:
        """Pairs whose frontend passed and whose backend did not.

        Declared pairs only.  Inferring "this frontend probably talks to
        that backend" from the kinds would put a guess in an alert.
        """
        orphaned: List[Tuple[str, str]] = []
        for frontend, backend in sorted(self._pairs.items()):
            front = results.get(frontend)
            back = results.get(backend)
            if front is None or back is None:
                continue
            front_ok = front.ok and not _asset_fault(
                self._by_name[frontend], front, self._asset_floor
            )
            back_ok = back.ok and not _asset_fault(
                self._by_name[backend], back, self._asset_floor
            )
            if front_ok and not back_ok:
                orphaned.append((frontend, backend))
        return orphaned

    def _pair_events(
        self,
        orphaned: Sequence[Tuple[str, str]],
        results: Mapping[str, CheckResult],
        now: float,
    ) -> List[Event]:
        if not orphaned:
            return self._resolve_if_open(
                ORPHANED_KEY, "The frontend can reach its backend again", now
            )
        if ORPHANED_KEY in self._open:
            return []
        self._open[ORPHANED_KEY] = now
        pairs_text = ", ".join(f"{front} -> {back}" for front, back in orphaned)
        reasons = "; ".join(
            f"{back}: {results[back].error or f'status {results[back].status}'}"
            for _front, back in orphaned
        )
        return [
            self.event(
                Severity.ACTION,
                f"The page is up but cannot reach its data: {pairs_text}. "
                f"A monitor watching only the frontend would call this healthy "
                f"({reasons}).",
                attention_key=ORPHANED_KEY,
                href=self.info.href,
                pairs=[list(pair) for pair in orphaned],
                frontends=[front for front, _back in orphaned],
                backends=[back for _front, back in orphaned],
            )
        ]

    # -- rule: the deploy shipped nothing ------------------------------------

    def _version_events(
        self, results: Mapping[str, CheckResult], now: float
    ) -> List[Event]:
        live: Optional[str] = None
        for check in self._checks:
            if check.kind != "version":
                continue
            result = results.get(check.name)
            if result is None or not result.ok:
                continue
            parsed = self._safe_parse(result.body_excerpt)
            if parsed:
                live = parsed
                break
        if live is not None:
            self._live_version = live

        expected = self._expected_version
        if expected is None or live is None:
            # Nothing was declared deployed, or the version endpoint did not
            # answer this tick.  Neither is drift; the down rule covers the
            # endpoint being unreachable.
            return []

        if live == expected:
            self._drift_ticks = 0
            return self._resolve_if_open(
                VERSION_DRIFT_KEY, f"Live version is {live}, as deployed", now
            )

        self._drift_ticks += 1
        if self._drift_ticks < DRIFT_TICKS_FOR_ACTION or VERSION_DRIFT_KEY in self._open:
            return []
        self._open[VERSION_DRIFT_KEY] = now
        return [
            self.event(
                Severity.ACTION,
                f"Version drift: {live} is live, {expected} was deployed. "
                f"The deploy reported success and the old build is still being "
                f"served.",
                attention_key=VERSION_DRIFT_KEY,
                href=self.info.href,
                live_version=live,
                expected_version=expected,
                ticks_drifted=self._drift_ticks,
                deployed_at=self._deployed_at,
            )
        ]

    def _safe_parse(self, body: str) -> Optional[str]:
        try:
            parsed = self._version_parser(body)
        except Exception:  # noqa: BLE001 - an injected parser is not trusted with the round
            return None
        if parsed is None:
            return None
        text = str(parsed).strip()
        return text or None

    # -- rule: slow ----------------------------------------------------------

    def _latency_events(
        self, check: Check, result: CheckResult, now: float, state: _CheckState
    ) -> List[Event]:
        if result.elapsed_ms > self._latency_ms:
            state.slow_ticks += 1
        else:
            state.slow_ticks = 0
            return []
        if state.slow_ticks != LATENCY_TICKS_FOR_NOTICE:
            # Exactly at the threshold, so a persistently slow endpoint is
            # one line and not one line every three minutes.
            return []
        return [
            self.event(
                Severity.NOTICE,
                f"{check.name} has been slow for {state.slow_ticks} checks "
                f"({result.elapsed_ms:.0f} ms, over {self._latency_ms:.0f} ms)",
                attention_key=latency_key(check.name),
                href=self.info.href,
                check=check.name,
                elapsed_ms=result.elapsed_ms,
                threshold_ms=self._latency_ms,
                slow_ticks=state.slow_ticks,
            )
        ]

    # -- closing a key -------------------------------------------------------

    def _resolve_if_open(self, key: str, text: str, now: float) -> List[Event]:
        """Close one open request, in the shape the framework expects.

        NOTICE -- below ACTION, so ``Event.wants_attention`` is false and
        the supervisor cannot re-open the key -- carrying that same key and
        ``resolved=True``.  Verbatim the convention in
        ``jarvis_bots/templates/bot.py.tmpl`` and
        :meth:`jarvis_bots.bots.poke_bot.PokeBot._resolve`; the supervisor
        closes on the flag (``RESOLVED_FLAG``), so this empties the badge by
        itself.
        """
        if key not in self._open:
            return []
        since = self._open.pop(key)
        return [
            self.event(
                Severity.NOTICE,
                text + f" (open for {_duration(max(0.0, now - since))})",
                attention_key=key,
                href=self.info.href,
                resolved=True,
                open_since=since,
            )
        ]

    # -- the card ------------------------------------------------------------

    def status(self) -> BotStatus:
        """What the launcher renders.  Cheap, and it does not raise.

        RUNNING: PAUSED and QUARANTINED are the supervisor's facts about
        this bot, not the bot's, and ``Supervisor.launcher_state`` overrides
        the state it owns.
        """
        total = len(self._checks)
        passing = sum(1 for check in self._checks if self._is_passing(check))
        slowest_name, slowest_ms = self._slowest()
        stats = (
            self.stat("Checks", f"{passing} of {total} passing"),
            self.stat("Live version", self._live_version or "unknown"),
            self.stat(
                "Slowest",
                f"{slowest_name} {slowest_ms:.0f} ms" if slowest_name else "not looked yet",
            ),
        )
        return BotStatus(BotState.RUNNING, stats=stats, detail=self._detail(passing, total))

    def _is_passing(self, check: Check) -> bool:
        result = self._last_results.get(check.name)
        if result is None:
            return False
        return result.ok and not _asset_fault(check, result, self._asset_floor)

    def _slowest(self) -> Tuple[str, float]:
        best_name, best_ms = "", -1.0
        for check in self._checks:
            result = self._last_results.get(check.name)
            # Strictly greater, so a tie is broken by the configured order
            # and the card does not flicker between two equal checks.
            if result is not None and result.elapsed_ms > best_ms:
                best_name, best_ms = check.name, result.elapsed_ms
        return best_name, max(0.0, best_ms)

    def _detail(self, passing: int, total: int) -> str:
        if not self._ticks:
            return "not looked yet"
        if self._open:
            return f"{len(self._open)} waiting on you"
        if passing < total:
            return f"{total - passing} check(s) not passing"
        expected = self._expected_version
        if expected and self._live_version == expected:
            return f"all clear on {expected}"
        return f"all clear over {self._ticks} passes"

    # -- the framework's optional half ---------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """JSON-able state.  What has to survive a restart is what a restart
        would otherwise re-learn wrongly: the consecutive failure counts
        (without them a restart resets every streak to zero and a check that
        has been down for an hour is silent for two more ticks), the
        expected version (a deploy is not re-announced after a crash), and
        the open keys (so a fault already in the badge is not re-alerted,
        and a recovery still finds a key to close).  ``since`` rides along
        so the alert can still say how long it has been failing."""
        return {
            "version": 1,
            "ticks": self._ticks,
            "last_tick_at": self._last_tick_at,
            "expected_version": self._expected_version,
            "live_version": self._live_version,
            "drift_ticks": self._drift_ticks,
            "deployed_at": self._deployed_at,
            "open": dict(self._open),
            "checks": {
                name: {
                    "fails": state.fails,
                    "since": state.since,
                    "slow_ticks": state.slow_ticks,
                }
                for name, state in self._state.items()
            },
        }

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """Take back what :meth:`snapshot` returned.

        A snapshot from a newer build is refused rather than half-read; a
        missing key reads as absent, so an older snapshot still opens.  State
        for a check that is no longer configured is dropped on the floor,
        which is what "the config changed" should mean.
        """
        if not isinstance(snapshot, Mapping) or not snapshot:
            return
        version = snapshot.get("version", 1)
        try:
            version = int(version)
        except (TypeError, ValueError):
            return
        if version > 1:
            raise HealthBotError(
                f"health bot snapshot version {version} is newer than this build "
                f"understands; refusing to half-read it"
            )
        self._ticks = int(snapshot.get("ticks") or 0)
        self._last_tick_at = float(snapshot.get("last_tick_at") or 0.0)
        expected = snapshot.get("expected_version")
        self._expected_version = str(expected) if expected else None
        live = snapshot.get("live_version")
        self._live_version = str(live) if live else None
        self._drift_ticks = int(snapshot.get("drift_ticks") or 0)
        self._deployed_at = float(snapshot.get("deployed_at") or 0.0)

        self._open = {
            str(key): float(since)
            for key, since in dict(snapshot.get("open") or {}).items()
            if isinstance(key, str) and key.strip()
        }
        self._state = {c.name: _CheckState() for c in self._checks}
        for name, row in dict(snapshot.get("checks") or {}).items():
            if name not in self._state or not isinstance(row, Mapping):
                continue
            self._state[name] = _CheckState(
                fails=max(0, int(row.get("fails") or 0)),
                since=float(row.get("since") or 0.0),
                slow_ticks=max(0, int(row.get("slow_ticks") or 0)),
            )

    def on_pause(self) -> None:
        """contracts.py: "Paused means paused ... a badge asking you to act
        on something you switched off is a lie."

        The supervisor clears this bot's attention when it is paused.  If
        the bot kept its own record it would believe those requests were
        still open and would never re-raise them on resume, so a backend
        that stayed down would be silent for ever.  Forgetting them here is
        what makes resume honest: the next tick re-raises whatever is still
        broken.
        """
        self._open.clear()


# ---------------------------------------------------------------------------
# The rule that needs its own paragraph
# ---------------------------------------------------------------------------


def _asset_fault(check: Check, result: CheckResult, floor_bytes: int) -> str:
    """Why this 200 is not the asset, or "" if it is fine.

    THE HOSTING FALLBACK.  Firebase Hosting (and every other single-page-app
    host) is configured to rewrite unknown paths to ``index.html`` so client
    side routing works.  That rewrite does not know the difference between
    ``/some/route`` and ``/assets/main-9f2a1c.js``: when the build drops an
    asset, the request for it gets ``index.html`` back -- with **status
    200**.  A monitor that checks status codes therefore reports a perfectly
    healthy site that serves a blank page, which is precisely the "deploy
    succeeded but shipped nothing" failure this bot exists to catch.

    So an asset is judged on its *body*: a real bundle is bigger than
    ``floor_bytes`` (an SPA shell is a few hundred bytes of HTML) and
    contains the marker the check declared.  Only checks of kind 'asset' are
    judged this way, and only when the status was the expected one -- a 404
    is an ordinary failure and the down rule already covers it.
    """
    if check.kind != "asset":
        return ""
    if result.status != check.expect_status:
        return ""
    size = result.size
    if size < floor_bytes:
        return f"{size} bytes, under the {floor_bytes} byte floor (an index.html fallback?)"
    if check.expect_contains and check.expect_contains not in result.body_excerpt:
        return f"body does not contain {check.expect_contains!r}"
    return ""


def _since(started: float, now: float, fails: int) -> str:
    """How long this has been failing, in the words the alert uses."""
    if started > 0.0 and now > started:
        return f"{_duration(now - started)} ({fails} checks in a row)"
    return f"{fails} checks in a row"


def _duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 90.0:
        return f"{seconds:.0f}s"
    minutes = seconds / 60.0
    if minutes < 90.0:
        return f"{minutes:.0f} min"
    hours = minutes / 60.0
    if hours < 48.0:
        return f"{hours:.1f} h"
    return f"{hours / 24.0:.1f} days"


def build(
    checks: Iterable[Check],
    clock: Clock,
    *,
    probe: Probe = urllib_probe,
    **kwargs: Any,
) -> HealthBot:
    """The wiring the app uses: the real probe, injected rather than called.

    ``HealthBot(checks, urllib_probe, clock)`` says the same thing; this
    exists so the default lives in one place and a test can pass its own
    probe through the same door.
    """
    return HealthBot(checks, probe, clock, **kwargs)
