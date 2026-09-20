"""The decision engine: act, wait, or do nothing -- and why.

Implements the "Market reference and verdicts" section of
:mod:`jarvis_poke.contracts` -- :class:`~jarvis_poke.contracts.Action`,
:class:`~jarvis_poke.contracts.Verdict`,
:class:`~jarvis_poke.contracts.MarketRef` and
:class:`~jarvis_poke.contracts.WatchState` -- on top of the rules and the
ledger in :mod:`jarvis_poke.rules`.

Where it stops
--------------
contracts.py, "What this is not": the engine's terminal output is a
:class:`~jarvis_poke.contracts.Verdict` plus a URL a person taps.  Nothing
here adds anything to a cart, touches a payment method, or opens a socket:
:meth:`DecisionEngine.evaluate` is a pure function of the observations it
is handed, the rules, the ledger and an injected clock.

The order of the tests, and why it is that order
------------------------------------------------
Each step appends a reason whether or not it changes the outcome, so a
BUY can be audited as closely as a refusal and
:attr:`~jarvis_poke.contracts.Verdict.reasons` is never empty.

1. **No rule, or a disabled rule** -> SKIP.  The owner has not said what
   they would pay, so there is nothing to decide.
2. **Nothing purchasable** -> NO_STOCK.
3. **``allowed_sources``**, when the rule names any.
4. **The cheapest listing**, by landed price, or by the shelf price when
   ``include_shipping`` is false.
5. **The market reference.**  If it is not usable the discount test is
   dropped, the reasons say so out loud, and ``max_price`` carries the
   decision alone.  A reference is never invented: an unusable one leaves
   :attr:`Verdict.market` ``None``.
6. **The outlier gate, before any BUY.**  A price far under the market is
   far more often a mis-parse -- a per-pack price on a box listing, a
   dropped digit, a placeholder -- than a deal.  Refusing costs a bargain
   once in a while; not refusing spends real money on a parsing bug, so
   this sits ahead of every test that could say yes.
7. **``max_price``** as a hard ceiling, then **``min_discount_pct``**
   against the reference.
8. **Quantity**: ``rule.quantity`` clamped to the listing's
   ``per_customer_limit``.
9. **The budget**, through :meth:`jarvis_poke.rules.RuleSet.affordable`,
   which applies ``MAX_BUDGET_FRACTION_PER_VERDICT``.
10. **The cooldown**, from ``watch_state.last_alert_at``.

WATCH means "in stock, and the only thing wrong is the price" -- it is the
outcome a falling price can fix on its own.  SKIP means structurally
ineligible: disabled, outlier, no budget, inside a cooldown, wrong source.
The distinction is what lets the page show "waiting for $5 off" separately
from "this cannot fire at all".

Injected everything
-------------------
``clock`` is a callable returning unix seconds; there is no ``time.time()``
anywhere in this package, and no ``random`` either -- the engine draws no
random numbers at all, so a verdict is reproducible from its inputs.
``history`` supplies the market reference (any object with a
``market_ref(product_id, now)``-shaped method, a bare callable, or
``None``), and the outlier test comes from
:func:`jarvis_poke.prices.is_outlier` when that module is present, from
``history.is_outlier`` when it offers one, from an explicit
``outlier_check=`` argument, or -- failing all three -- from the
conservative built-in :func:`looks_mis_parsed` below.  The engine is
written against a sibling module that may not be there yet; what it will
not do is quietly stop checking.

Judgement calls
---------------
* **``min_discount_pct == 0`` means "no discount test"**, not "must be at
  or under the median".  Taken literally, a 0% discount requirement would
  refuse every price above the market median even when the owner's ceiling
  allows it, which is not what leaving a default alone means.
* **An unusable reference still allows a BUY under the ceiling**, per the
  step-5 rule above, *even when the rule asks for a discount*.  The
  ceiling is a number the owner wrote down; the discount is measured
  against a number nobody has.  The reasons say the test was dropped, and
  :func:`explain` puts that in the alert body.
* **Listings the rule does not allow give SKIP, not NO_STOCK**: the
  product is in stock, so "no stock" would be a lie, and it is not a price
  problem, so WATCH would be too.
* **Only the newest observation per (source, sku) is considered.**  A
  caller handing over a history rather than a snapshot should not have the
  engine act on last week's price.
* **A BUY reserves its spend** (``reserve_on_buy``), so two verdicts in one
  pass cannot each promise the same money.  The app releases the
  reservation when the owner ignores the link
  (:meth:`DecisionEngine.release_reservation`) and commits it when they do
  not (:meth:`DecisionEngine.commit_purchase`).
"""

from __future__ import annotations

import inspect
import math
from contextlib import contextmanager, nullcontext
from fractions import Fraction
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Protocol,
    Sequence,
    Tuple,
)

from jarvis_poke.contracts import (
    MARKET_WINDOW_S,
    Action,
    Cents,
    MarketRef,
    Observation,
    Rule,
    Verdict,
    WatchState,
    fmt_cents,
)
from jarvis_poke.rules import BudgetError, RuleSet

__all__ = [
    "DEFAULT_EXPLAIN_CHARS",
    "DISCOUNT_PRECISION",
    "OUTLIER_LOW_FRACTION",
    "DecisionEngine",
    "EngineError",
    "MarketHistory",
    "OutlierCheck",
    "discount_against",
    "discount_threshold",
    "explain",
    "looks_mis_parsed",
]


class EngineError(ValueError):
    """A mis-wired engine: no clock, an unusable history, a bad observation.

    Raised at construction wherever possible, so a monitoring loop fails
    when it is built rather than three hours later next to a price.
    """


class MarketHistory(Protocol):
    """What the engine needs of :mod:`jarvis_poke.prices`.

    One method, returning what this product has actually been selling for,
    or ``None`` when there is not enough to say.  The engine also accepts a
    plain callable with the same shape, and ``None`` for "no reference
    available", in which case every verdict says the discount test was
    dropped.
    """

    def market_ref(self, product_id: str, now: float) -> Optional[MarketRef]: ...


class OutlierCheck(Protocol):
    """``prices.is_outlier``: is this price implausibly low for this product?

    May return a bool, or a ``(bool, reason)`` pair when it can say what it
    suspects.
    """

    def __call__(self, price: Cents, ref: Optional[MarketRef]) -> Any: ...


#: A price under this fraction of the market median is treated as a
#: mis-parse by the built-in fallback gate.  Two fifths is well below any
#: real sale on a product with a working market reference and well above
#: the two classic parsing failures -- a per-pack price read off a box
#: listing, and a dropped leading digit.
OUTLIER_LOW_FRACTION = 0.4

#: ``min_discount_pct`` is honoured to six decimal places.  Rounding the
#: percentage to a fixed grid before converting it to a
#: :class:`~fractions.Fraction` keeps ``10.0`` meaning exactly a tenth,
#: rather than the binary double that is a hair above it and would move the
#: boundary by a cent.
DISCOUNT_PRECISION = 1_000_000

#: How much of the reasoning :func:`explain` puts in an alert body.
DEFAULT_EXPLAIN_CHARS = 260


# --------------------------------------------------------------------------
# the money arithmetic, kept in integers
# --------------------------------------------------------------------------


def discount_threshold(reference: Cents, min_discount_pct: float) -> Cents:
    """The highest landed price that still meets ``min_discount_pct``.

    Computed in exact rationals and rounded *against* the buyer: the
    required discount is rounded up to the next whole cent, so a price one
    cent above the threshold fails, and a price exactly at it passes.
    """
    if isinstance(reference, bool) or not isinstance(reference, int):
        raise EngineError(f"market reference must be integer cents, got {reference!r}")
    if reference <= 0:
        raise EngineError(f"market reference must be positive, got {reference}")
    pct = Fraction(round(float(min_discount_pct) * DISCOUNT_PRECISION), DISCOUNT_PRECISION)
    if pct <= 0:
        return reference
    off = math.ceil(Fraction(reference) * pct / 100)
    return reference - off


def discount_against(reference: Cents, landed: Cents) -> Optional[float]:
    """How far under the reference this price is, as a percentage.

    Named for its argument order, which is the reverse of
    :func:`jarvis_poke.prices.discount_pct` -- two functions called
    ``discount_pct`` taking ``(a, b)`` and ``(b, a)`` is a bug waiting to
    be written, so this one says which way round it goes.

    Negative when the price is *above* the market.  ``None`` when there is
    no usable reference to measure against -- the engine reports that as an
    absent number rather than a zero, because "no idea" and "no discount"
    are different things.
    """
    if not reference or reference <= 0:
        return None
    return round((reference - landed) * 100.0 / reference, 4)


def looks_mis_parsed(
    landed: Optional[Cents], ref: Optional[MarketRef]
) -> Tuple[bool, str]:
    """The built-in outlier gate, used when no ``prices`` module is wired in.

    Takes the *landed* price, because that is what a market reference is
    built from (:func:`jarvis_poke.prices.market_reference` samples
    ``price + shipping``) and comparing a shelf price against a landed
    median is how a cheap item with expensive postage reads as a bargain.

    Deliberately narrow: it only fires on a price that cannot be real
    (``None`` or not positive) or one far under a *usable* market
    reference.  It never invents a reference, and it never says "fine" in a
    way that lets a non-positive price through.
    """
    price = landed
    if price is None:
        return True, "the listing had no parseable price at all"
    if price <= 0:
        return True, (
            f"a price of {fmt_cents(price)} is not a price -- the page was "
            f"mis-parsed (a missing value, or a currency symbol read as a digit)"
        )
    if ref is None or not getattr(ref, "usable", False) or not ref.median:
        return False, ""
    floor = math.floor(
        Fraction(ref.median)
        * Fraction(round(OUTLIER_LOW_FRACTION * DISCOUNT_PRECISION), DISCOUNT_PRECISION)
    )
    if price < floor:
        return True, (
            f"{fmt_cents(price)} is under {fmt_cents(floor)}, "
            f"{int(OUTLIER_LOW_FRACTION * 100)}% of the {fmt_cents(ref.median)} market "
            f"median from {ref.samples} samples -- suspected mis-parse "
            f"(a per-pack price on a sealed-box listing, or a dropped digit)"
        )
    return False, ""


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------


_HEADLINES = {
    Action.BUY: "Buy",
    Action.WATCH: "Watching",
    Action.SKIP: "Skipping",
    Action.NO_STOCK: "No stock",
}


def explain(verdict: Verdict, max_chars: int = DEFAULT_EXPLAIN_CHARS) -> str:
    """Render a verdict's reasons as one short sentence for an alert body.

    The headline carries the numbers a person acts on -- how many, at what
    price, from where -- and the reasons follow in the order the engine
    reached them, truncated at a reason boundary so the body never ends
    mid-clause.
    """
    if not isinstance(verdict, Verdict):
        raise EngineError(f"not a Verdict: {verdict!r}")
    head = _HEADLINES.get(verdict.action, verdict.action.value)
    parts: List[str] = []
    if verdict.action is Action.BUY:
        parts.append(f"{verdict.quantity} x {verdict.product_id}")
    else:
        parts.append(verdict.product_id)
    cost = verdict.landed if verdict.landed is not None else verdict.price
    if cost is not None:
        at = f"at {fmt_cents(cost)}"
        if verdict.source:
            at += f" from {verdict.source}"
        parts.append(at)
    if verdict.discount_pct is not None and verdict.market:
        sign = "under" if verdict.discount_pct >= 0 else "over"
        parts.append(
            f"{abs(verdict.discount_pct):.1f}% {sign} the {fmt_cents(verdict.market)} market"
        )
    headline = f"{head}: " + ", ".join(parts)

    reasons = [r for r in verdict.reasons if r]
    if not reasons:  # pragma: no cover - the engine guarantees otherwise
        return headline + "."
    # Fill from the end: the last reason is the one that decided the
    # verdict, and an alert body that truncates the decisive clause to make
    # room for "the rule allows any source" is worse than useless.
    kept: List[str] = []
    length = len(headline) + 2
    for reason in reversed(reasons):
        extra = len(reason) + (2 if kept else 0)
        if kept and length + extra > max_chars:
            kept.insert(0, "...")
            break
        kept.insert(0, reason)
        length += extra
    return f"{headline}. " + "; ".join(kept) + "."


# --------------------------------------------------------------------------
# wiring up the injected pieces
# --------------------------------------------------------------------------


_MARKET_METHODS = ("market_ref", "market_reference", "reference", "market")


def _positional(func: Callable[..., Any]) -> Tuple[int, int, List[str]]:
    """``(accepted, required, names)`` positional parameters of a callable.

    ``inspect`` sees through a bound method, so ``self`` is already gone.
    A ``*args`` callable is treated as accepting as many as we want to
    give it.
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):  # pragma: no cover - C callables
        return 2, 0, []
    accepted = 0
    required = 0
    names: List[str] = []
    for param in signature.parameters.values():
        if param.kind is inspect.Parameter.VAR_POSITIONAL:
            return 99, required, names
        if param.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        accepted += 1
        names.append(param.name)
        if param.default is inspect.Parameter.empty:
            required += 1
    return accepted, required, names


def _prices_module() -> Any:
    """:mod:`jarvis_poke.prices` if it is installed beside us, else ``None``.

    Imported lazily and optionally: the market lane is a sibling module,
    and an engine wired up with its own history object must not need it to
    exist.
    """
    try:
        from jarvis_poke import prices as module  # noqa: WPS433 - optional sibling
    except ImportError:  # pragma: no cover - only when the lane is absent
        return None
    return module


def _checked_catalog(catalog: Any) -> Any:
    """Refuse an object that is plainly not a catalog, at construction.

    ``catalog`` is the *first* positional parameter and ``rules`` the
    second, so ``DecisionEngine(rule_set, clock=...)`` is an easy slip --
    and a silent one, because :class:`~jarvis_poke.rules.RuleSet` happens
    to support ``in``, which is all :meth:`DecisionEngine._known` needs.
    The engine would then hold an empty RuleSet and answer every product
    with SKIP "no rule for X": a monitor that has quietly stopped
    monitoring, which is the failure this package least wants to ship.

    Anything catalog-shaped passes -- a callable ``find``, ``product`` or
    ``sku`` -- and so does a bare container supporting ``in``, which
    ``_known`` documents as the fallback.  Only a RuleSet, or an object
    offering no way to ask about a product at all, is turned away.
    """
    if catalog is None:
        return None
    for name in ("find", "product", "sku"):
        if callable(getattr(catalog, name, None)):
            return catalog
    if callable(getattr(catalog, "get", None)) and callable(
        getattr(catalog, "affordable", None)
    ):
        raise EngineError(
            f"{type(catalog).__name__} is a rule set, not a catalog; "
            f"DecisionEngine takes (catalog, rules, history, clock) -- pass it "
            f"as rules=, or None for no catalog"
        )
    if hasattr(catalog, "__contains__"):
        return catalog
    raise EngineError(
        f"catalog {catalog!r} offers no find(), product(), sku() or 'in'; "
        f"pass None for 'no catalog'"
    )


def _resolve_market_ref(
    history: Any, window_s: float = MARKET_WINDOW_S
) -> Callable[[str, float], Optional[MarketRef]]:
    """Adapt whatever the app injected to ``(product_id, now) -> MarketRef?``.

    The market lane is a sibling module written in parallel, so the shape
    is discovered rather than assumed -- but only at construction, and a
    shape that cannot be called raises there and then instead of silently
    turning the discount test off at three in the morning.
    """
    if history is None:
        return lambda product_id, now: None
    func: Optional[Callable[..., Any]] = None
    for name in _MARKET_METHODS:
        candidate = getattr(history, name, None)
        if callable(candidate):
            func = candidate
            break
    prices = _prices_module()
    if func is None and prices is not None and callable(getattr(prices, "market_reference", None)):
        store = getattr(prices, "PriceHistory", None)
        if store is not None and isinstance(history, store):
            # The shipped market lane keeps the reference as a module-level
            # function over a plain store, so a bare PriceHistory is wired
            # up here rather than made to grow a method it does not have.
            def from_history(product_id: str, now: float) -> Optional[MarketRef]:
                return prices.market_reference(history, product_id, now, window_s)

            return from_history
    if func is None and callable(history):
        func = history
    if func is None:
        raise EngineError(
            f"history {history!r} offers none of "
            f"{', '.join(n + '()' for n in _MARKET_METHODS)} and is not callable; "
            f"pass None for 'no market reference'"
        )
    accepted, required, names = _positional(func)
    if required > 2 or accepted < 1:
        raise EngineError(
            f"history's {getattr(func, '__name__', func)!r} does not take "
            f"(product_id, now); wrap it in a callable that does"
        )
    if accepted >= 2:
        # A third, optional parameter named for a window is the trailing span
        # the reference is computed over -- ``PriceHistory.market_ref`` and
        # ``prices.market_reference`` both spell it ``window_s``.  Passing
        # ``market_window_s`` through matters: without this the method's own
        # default silently wins and an engine built with a narrower window
        # scores against a reference it did not ask for.  Anything whose third
        # parameter is named otherwise is left alone -- guessing at an unknown
        # argument is worse than not passing it.
        if accepted >= 3 and len(names) >= 3 and "window" in names[2].lower():
            return lambda product_id, now: func(product_id, now, window_s)
        return lambda product_id, now: func(product_id, now)
    return lambda product_id, now: func(product_id)


def _resolve_outlier_check(
    outlier_check: Any, history: Any
) -> Tuple[Optional[Callable[[Optional[Cents], Optional[MarketRef]], Any]], str]:
    """Find the outlier gate, and say where it came from.

    Order: an explicit argument, then ``history.is_outlier``, then
    :func:`jarvis_poke.prices.is_outlier` if that module exists yet.  The
    caller falls back to :func:`looks_mis_parsed` when all three are
    absent, which is a real gate, not a shrug.
    """
    func = outlier_check
    origin = "outlier_check argument"
    if func is None and history is not None:
        candidate = getattr(history, "is_outlier", None)
        if callable(candidate):
            func, origin = candidate, "history.is_outlier"
    if func is None:
        prices = _prices_module()
        candidate = getattr(prices, "is_outlier", None) if prices is not None else None
        if callable(candidate):
            func, origin = candidate, "prices.is_outlier"
    if func is None:
        return None, "built-in"
    if not callable(func):
        raise EngineError(f"outlier check {func!r} is not callable")

    accepted, required, names = _positional(func)
    if required > 2:
        raise EngineError(
            f"{origin} takes {required} required arguments; the engine can only "
            f"offer (price, market_ref) -- wrap it"
        )
    first = (names[0].lower() if names else "")
    ref_first = any(token in first for token in ("ref", "market", "median"))

    def call(price: Optional[Cents], ref: Optional[MarketRef]) -> Any:
        if accepted <= 1:
            return func(price)
        if ref_first:
            return func(ref, price)
        return func(price, ref)

    call.origin = origin  # type: ignore[attr-defined]
    return call, origin


def _as_outlier_answer(raw: Any) -> Tuple[bool, str]:
    """Normalise a bool, a ``(bool, reason)`` pair, or a result object."""
    if isinstance(raw, tuple) and len(raw) == 2:
        return bool(raw[0]), str(raw[1] or "")
    flag = getattr(raw, "is_outlier", None)
    if flag is not None and not isinstance(raw, (bool, int, float)):
        return bool(flag), str(getattr(raw, "reason", "") or "")
    return bool(raw), ""


# --------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------


class DecisionEngine:
    """Turns observations into verdicts, and remembers that it did.

    ``catalog`` is optional: when given, a product it does not know is
    SKIPped rather than acted on, which catches a rule whose product id has
    drifted.  ``rules`` is a :class:`~jarvis_poke.rules.RuleSet` (or
    anything with the same ``get``/``affordable`` surface).  ``clock`` is
    required -- there is no default, because a default would be
    ``time.time`` and contracts.py does not allow one.  ``watch_states`` is
    used *in place*, so a caller that keeps it in a store sees the engine's
    updates.
    """

    def __init__(
        self,
        catalog: Any = None,
        rules: Optional[RuleSet] = None,
        history: Any = None,
        clock: Optional[Callable[[], float]] = None,
        watch_states: Optional[MutableMapping[str, WatchState]] = None,
        *,
        outlier_check: Optional[OutlierCheck] = None,
        reserve_on_buy: bool = True,
        market_window_s: float = MARKET_WINDOW_S,
    ) -> None:
        if clock is None or not callable(clock):
            raise EngineError(
                "DecisionEngine needs an injected clock: a callable returning "
                "unix seconds (this package never calls time.time())"
            )
        self.catalog = _checked_catalog(catalog)
        self.rules = rules if rules is not None else RuleSet()
        needed = ["get", "affordable"]
        if reserve_on_buy:
            # record_verdict will reach for these on the first BUY; better to
            # find out now than three hours into a polling run.
            needed += ["reserve", "release", "remaining"]
        for method in needed:
            if not callable(getattr(self.rules, method, None)):
                raise EngineError(f"rules object has no {method}(): {self.rules!r}")
        self.history = history
        self._clock = clock
        self.market_window_s = float(market_window_s)
        self._market_ref = _resolve_market_ref(history, self.market_window_s)
        self._outlier, self.outlier_origin = _resolve_outlier_check(outlier_check, history)
        self.reserve_on_buy = bool(reserve_on_buy)
        # Reservations belong to the ledger, not to this object, whenever
        # the ledger can hold them: an app that builds an engine per run
        # against one long-lived RuleSet (cli.cmd_decide does exactly
        # that) would otherwise book the same listing again every run,
        # with no live engine left holding the money to give it back.
        # ``_local_reservations`` is the fallback for a duck-typed rules
        # object that has no keyed half.
        self._keyed_rules = all(
            callable(getattr(self.rules, name, None))
            for name in ("reserve_for", "release_for", "reservations")
        )
        self._local_reservations: Dict[str, Cents] = {}

        if watch_states is None:
            self.watch_states: MutableMapping[str, WatchState] = {}
        elif isinstance(watch_states, MutableMapping):
            self.watch_states = watch_states
        elif isinstance(watch_states, Mapping):
            self.watch_states = dict(watch_states)
        else:
            self.watch_states = {w.product_id: w for w in watch_states}
        for key, state in self.watch_states.items():
            if not isinstance(state, WatchState):
                raise EngineError(f"watch_states[{key!r}] is not a WatchState: {state!r}")

    # -- state -------------------------------------------------------------

    def now(self) -> float:
        """The injected clock, as a float."""
        return float(self._clock())

    def watch_state(self, product_id: str) -> WatchState:
        """The bookkeeping for one product, created empty on first ask."""
        pid = str(product_id)
        state = self.watch_states.get(pid)
        if state is None:
            state = WatchState(product_id=pid)
            self.watch_states[pid] = state
        return state

    def _ledger(self):
        """The ledger's lock, as a context manager.

        "Is this affordable?" and "reserve it" are one decision; held
        apart, a second evaluation slips between them, sees the money
        that is about to be promised, and is told it too fits the
        per-verdict cap.  :meth:`evaluate` therefore runs inside this.
        A rules object that offers no lock gets a no-op, which is exactly
        the old behaviour rather than a crash.
        """
        lock = getattr(self.rules, "ledger_lock", None)
        if lock is None or not hasattr(lock, "__enter__"):
            return nullcontext()
        return lock

    def reservations(self) -> Dict[str, Cents]:
        """Cents currently promised to an alerted BUY, by product id."""
        if self._keyed_rules:
            return dict(self.rules.reservations())
        return dict(self._local_reservations)

    def _held(self, product_id: str) -> Cents:
        if self._keyed_rules:
            getter = getattr(self.rules, "reserved_for", None)
            if callable(getter):
                return getter(product_id)
            return self.rules.reservations().get(str(product_id), 0)
        return self._local_reservations.get(str(product_id), 0)

    def release_reservation(self, product_id: str) -> Cents:
        """The owner did not use the deep link: give the money back."""
        if self._keyed_rules:
            return self.rules.release_for(str(product_id))
        amount = self._local_reservations.pop(str(product_id), 0)
        if amount:
            self.rules.release(amount)
        return amount

    def commit_purchase(self, product_id: str, amount: Optional[Cents] = None) -> Cents:
        """The owner bought it: turn the reservation into spend.

        ``amount`` defaults to what was reserved; pass the real total when
        the checkout page disagreed, which it will (tax, a coupon, a
        shipping band).
        """
        pid = str(product_id)
        if self._keyed_rules:
            committer = getattr(self.rules, "commit_for", None)
            if callable(committer):
                return committer(pid, amount)
        with self._ledger():
            reserved = self.release_reservation(pid)
            spend = reserved if amount is None else amount
            if spend:
                self.rules.commit(spend)
            return spend

    # -- deciding ----------------------------------------------------------

    def evaluate(
        self, product_id: str, observations: Sequence[Observation]
    ) -> Verdict:
        """Decide what to do about one product, and record that we decided.

        The observations are read, never written, and never re-ordered in
        place: the caller's sequence is untouched.  The steps are the ones
        listed in this module's docstring, in that order, each leaving a
        reason behind.

        The whole of it runs under the ledger's lock
        (:attr:`jarvis_poke.rules.RuleSet.ledger_lock`).  Step 9 reads
        what is left and the reservation is taken at the end; anything
        allowed to run in between would be told the same money is free
        twice, and two verdicts would each be shown "fits the cap" for
        the whole budget.  Evaluations therefore serialise against each
        other, which is what ``evaluate_all``'s ordering already assumes.
        """
        with self._ledger():
            return self._evaluate(product_id, observations)

    def _evaluate(
        self, product_id: str, observations: Sequence[Observation]
    ) -> Verdict:
        now = self.now()
        pid = str(product_id)
        reasons: List[str] = []

        snapshot = tuple(observations)
        for index, obs in enumerate(snapshot):
            if not isinstance(obs, Observation):
                raise EngineError(f"observations[{index}] is not an Observation: {obs!r}")

        mine = [obs for obs in snapshot if obs.product_id == pid]
        strays = len(snapshot) - len(mine)
        latest = _newest_per_listing(mine)
        in_stock = [obs for obs in latest if obs.purchasable]

        # 1. a rule, and an enabled one
        rule = self.rules.get(pid)
        if rule is None:
            reasons.append(
                f"no rule for {pid}: the owner has not said what they would pay"
            )
            return self._finish(pid, Action.SKIP, now, reasons, seen_in_stock=bool(in_stock))
        if not rule.enabled:
            reasons.append(f"the rule for {pid} is switched off")
            return self._finish(pid, Action.SKIP, now, reasons, seen_in_stock=bool(in_stock))
        reasons.append(_rule_summary(rule))
        if strays:
            reasons.append(f"ignored {strays} observation(s) for other products")

        if self.catalog is not None and not self._known(pid):
            reasons.append(
                f"{pid} is not in the catalog: the rule's product id is stale or "
                f"misspelt, and acting on it would be guessing"
            )
            return self._finish(pid, Action.SKIP, now, reasons, seen_in_stock=bool(in_stock))

        # 2. anything to buy at all
        if not in_stock:
            reasons.append(_stock_summary(latest))
            return self._finish(pid, Action.NO_STOCK, now, reasons, seen_in_stock=False)
        reasons.append(
            f"{len(in_stock)} of {len(latest)} listing(s) purchasable: "
            + ", ".join(
                f"{obs.source} {fmt_cents(_cost(obs, rule))}"
                for obs in sorted(in_stock, key=lambda o: (_cost(o, rule), o.source))
            )
        )

        # 3. allowed sources
        if rule.allowed_sources:
            allowed = [obs for obs in in_stock if obs.source in rule.allowed_sources]
            dropped = [obs.source for obs in in_stock if obs.source not in rule.allowed_sources]
            if dropped:
                reasons.append(
                    f"ignored {', '.join(sorted(set(dropped)))}: the rule allows only "
                    f"{', '.join(rule.allowed_sources)}"
                )
            if not allowed:
                reasons.append(
                    "in stock, but at no source the rule allows -- structurally "
                    "ineligible, not a price problem"
                )
                return self._finish(pid, Action.SKIP, now, reasons, seen_in_stock=True)
            in_stock = allowed
        else:
            reasons.append("the rule allows any source")

        # 4. the cheapest one.  ``cost`` is what the rule's ceiling and
        # discount are measured against -- the shelf price when the rule
        # says to ignore shipping.  ``spend`` is what the owner's card is
        # charged, which is the landed price whatever the rule says, and
        # it is the only number the ledger is allowed to see: a budget
        # that does not count postage is not a ceiling on the account.
        best = min(in_stock, key=lambda obs: (_cost(obs, rule), obs.source, obs.sku, -obs.at))
        cost = _cost(best, rule)
        spend = best.landed if best.landed is not None else cost
        basis = "landed" if rule.include_shipping else "shelf price, shipping excluded"
        reasons.append(
            f"cheapest is {best.source} at {fmt_cents(cost)} ({basis})"
            + (f", shipping {fmt_cents(best.shipping)}" if rule.include_shipping else "")
        )
        if not rule.include_shipping and spend != cost:
            reasons.append(
                f"the rule's ceiling ignores shipping, but the card is charged "
                f"{fmt_cents(spend)}: that is what the budget is asked for"
            )

        # 5. the market reference, or an honest absence
        ref = self._reference(pid, now)
        market: Optional[Cents] = None
        measured: Optional[float] = None
        if ref is not None and getattr(ref, "usable", False) and ref.median:
            market = ref.median
            # prices.market_reference samples *landed* prices, so the
            # discount has to be measured on the landed price too, or a
            # listing with heavy postage reads as a bargain it is not.
            measured = discount_against(market, spend)
            reasons.append(
                f"market reference {fmt_cents(market)} (median of {ref.samples} samples)"
            )
        else:
            reasons.append(
                "market reference unusable "
                f"({_why_unusable(ref)}): no discount test applied, "
                f"the {fmt_cents(rule.max_price)} ceiling decides alone"
            )

        # 6. the outlier gate, ahead of anything that could say buy
        outlier, why = self._is_outlier(best.landed, ref)
        if outlier:
            reasons.append(
                f"refusing as a suspected mis-parse: {why or 'the price is implausibly low'}"
            )
            return self._finish(
                pid, Action.SKIP, now, reasons, obs=best, market=market,
                discount=measured, seen_in_stock=True,
            )
        reasons.append("price is plausible for this product")

        # 7. the ceiling, then the discount
        if cost > rule.max_price:
            reasons.append(
                f"{fmt_cents(cost)} is {fmt_cents(cost - rule.max_price)} over the "
                f"{fmt_cents(rule.max_price)} ceiling"
            )
            return self._finish(
                pid, Action.WATCH, now, reasons, obs=best, market=market,
                discount=measured, seen_in_stock=True,
            )
        reasons.append(
            f"at or under the {fmt_cents(rule.max_price)} ceiling "
            f"(by {fmt_cents(rule.max_price - cost)})"
        )

        if market is not None and rule.min_discount_pct > 0:
            threshold = discount_threshold(market, rule.min_discount_pct)
            if spend > threshold:
                reasons.append(
                    f"{measured:.2f}% off is short of the {rule.min_discount_pct:g}% the "
                    f"rule asks for: that wants {fmt_cents(threshold)} or better"
                )
                return self._finish(
                    pid, Action.WATCH, now, reasons, obs=best, market=market,
                    discount=measured, seen_in_stock=True,
                )
            reasons.append(
                f"{measured:.2f}% off meets the {rule.min_discount_pct:g}% the rule asks "
                f"for (needs {fmt_cents(threshold)} or better)"
            )
        elif market is not None:
            reasons.append(
                f"the rule asks for no particular discount; this is {measured:.2f}% off"
            )

        # 8. how many
        quantity = rule.quantity
        limit = best.per_customer_limit
        if limit is not None:
            if limit <= 0:
                reasons.append(
                    f"{best.source} allows {limit} per customer: there is nothing to buy"
                )
                return self._finish(
                    pid, Action.SKIP, now, reasons, obs=best, market=market,
                    discount=measured, seen_in_stock=True,
                )
            if limit < quantity:
                reasons.append(
                    f"{best.source} allows {limit} per customer, the rule wanted "
                    f"{rule.quantity}: buying {limit}"
                )
                quantity = limit
            else:
                reasons.append(
                    f"quantity {quantity}, within {best.source}'s limit of {limit}"
                )
        else:
            reasons.append(f"quantity {quantity}, no per-customer limit seen")

        # 9. the budget.  A reservation this product's own earlier BUY is
        # holding is money this verdict supersedes rather than competes
        # with -- the owner did not act on that alert, and this one
        # replaces it -- so it is added back before the cap is applied.
        held = self._held(pid)
        money = self.rules.affordable(
            rule, spend, quantity, self.rules.remaining() + held
        )
        reasons.append(money.reason)
        if held:
            reasons.append(
                f"counting the {fmt_cents(held)} this product's last alert is still "
                f"holding as available: this verdict replaces it"
            )
        if not money:
            return self._finish(
                pid, Action.SKIP, now, reasons, obs=best, market=market,
                discount=measured, seen_in_stock=True,
            )

        # 10. the cooldown
        state = self.watch_state(pid)
        since = now - state.last_alert_at
        if state.last_alert_at > 0 and since < rule.cooldown_s:
            reasons.append(
                f"alerted {since:.0f}s ago, inside the {rule.cooldown_s:.0f}s cooldown: "
                f"{rule.cooldown_s - since:.0f}s to go"
            )
            return self._finish(
                pid, Action.SKIP, now, reasons, obs=best, market=market,
                discount=measured, seen_in_stock=True,
            )
        reasons.append(
            "outside the cooldown" if state.last_alert_at > 0 else "never alerted before"
        )

        return self._finish(
            pid, Action.BUY, now, reasons, obs=best, market=market, discount=measured,
            quantity=quantity, seen_in_stock=True,
        )

    def evaluate_all(
        self,
        observations: Sequence[Observation],
        product_ids: Optional[Iterable[str]] = None,
    ) -> List[Verdict]:
        """Evaluate several products, in sorted product-id order.

        Deliberately sequential and ordered: a BUY reserves its spend, so
        an earlier product can leave a later one unaffordable, and the
        alphabetical order makes which one wins reproducible rather than a
        function of dict insertion.  Without ``product_ids`` it covers
        every product with a rule plus every product observed, so a
        listing for something with no rule still gets a verdict saying so.
        """
        snapshot = tuple(observations)
        if product_ids is None:
            wanted = set(getattr(self.rules, "product_ids", list)())
            wanted.update(obs.product_id for obs in snapshot)
        else:
            wanted = {str(pid) for pid in product_ids}
        by_product: Dict[str, List[Observation]] = {pid: [] for pid in wanted}
        for obs in snapshot:
            if obs.product_id in by_product:
                by_product[obs.product_id].append(obs)
        return [self.evaluate(pid, by_product[pid]) for pid in sorted(by_product)]

    # -- recording ---------------------------------------------------------

    def record_verdict(self, verdict: Verdict, *, seen_in_stock: bool = False) -> WatchState:
        """Fold one verdict into the watch state, and reserve a BUY's money.

        :meth:`evaluate` calls this itself on every path, so a caller
        cannot forget it and end up alerting the same restock every thirty
        seconds.  It is idempotent in the sense that matters -- re-recording
        a BUY for a product replaces that product's reservation rather than
        stacking a second one -- but it is not a pure function: calling it
        by hand on a BUY *will* move the cooldown.
        """
        if not isinstance(verdict, Verdict):
            raise EngineError(f"not a Verdict: {verdict!r}")
        state = self.watch_state(verdict.product_id)
        state.last_action = verdict.action
        if seen_in_stock:
            state.last_seen_in_stock = verdict.at
        if verdict.action is Action.BUY:
            # The reservation goes first.  If the ledger refuses it there
            # is no alert, and an alert that was never produced must not
            # leave the product inside a cooldown -- that is a missed
            # restock caused by the bookkeeping for one that never
            # happened.
            if self.reserve_on_buy:
                self._reserve_for(verdict)
            state.last_alert_at = verdict.at
            state.alerts_sent += 1
        return state

    def _reserve_for(self, verdict: Verdict) -> None:
        """Hold the money this BUY will cost.

        The *landed* price, whatever ``include_shipping`` says: that flag
        is about which number the owner's ceiling is measured against,
        not about what their card is charged, and a ledger that leaves
        postage out is not the ceiling contracts.py promises.
        """
        cost = verdict.landed if verdict.landed is not None else verdict.price
        if cost is None or verdict.quantity < 1:
            return
        # Shipping is counted once per copy.  A retailer that charges one
        # postage for an order of three is over-reserved by two, which
        # errs towards holding the owner's money rather than spending it;
        # the package cannot know which model a given shop uses, and of
        # the two errors this is the one that cannot overdraw an account.
        amount = cost * verdict.quantity
        with self._ledger():
            if self._keyed_rules:
                self.rules.reserve_for(verdict.product_id, amount)
                return
            self.release_reservation(verdict.product_id)
            self.rules.reserve(amount)
            self._local_reservations[verdict.product_id] = amount

    def explain(self, verdict: Verdict, max_chars: int = DEFAULT_EXPLAIN_CHARS) -> str:
        """:func:`explain`, as a method, for callers holding only an engine."""
        return explain(verdict, max_chars)

    # -- internals ---------------------------------------------------------

    def _known(self, product_id: str) -> bool:
        finder = getattr(self.catalog, "find", None)
        if callable(finder):
            return finder(product_id) is not None
        try:
            return product_id in self.catalog
        except TypeError:  # pragma: no cover - an exotic catalog
            return True

    def _reference(self, product_id: str, now: float) -> Optional[MarketRef]:
        ref = self._market_ref(product_id, now)
        if ref is None:
            return None
        if not hasattr(ref, "usable") or not hasattr(ref, "median"):
            raise EngineError(
                f"history returned {ref!r} for {product_id!r}; expected a MarketRef "
                f"or None"
            )
        return ref

    def _is_outlier(
        self, landed: Optional[Cents], ref: Optional[MarketRef]
    ) -> Tuple[bool, str]:
        """Is this price implausibly low?  Asked of the wired-in gate.

        Three rules, in this order:

        * **A missing or non-positive price is always refused**, with or
          without a reference.  No gate gets to wave that through; it is a
          parse that failed, not a bargain.
        * **With a reference, the wired-in gate decides.**  Where
          :mod:`jarvis_poke.prices` is present that is its ``is_outlier``,
          and its threshold is deliberately generous -- a real 75%-off
          clearance exists -- so this module does not second-guess it with
          a stricter rule of its own.  Owning the question in one place is
          worth more than two opinions.
        * **Without a reference there is nothing to compare to**, so the
          built-in :func:`looks_mis_parsed` answers, which for a positive
          price with no usable reference means "no, carry on to the
          ceiling".

        A gate that raises counts as a refusal: "the outlier check
        crashed" is not a reason to spend money.
        """
        built_in, why = looks_mis_parsed(landed, ref)
        if landed is None or landed <= 0:
            return True, why
        if self._outlier is None or ref is None:
            return built_in, why
        try:
            flagged, reason = _as_outlier_answer(self._outlier(landed, ref))
        except Exception as exc:  # noqa: BLE001 - deliberately broad, see docstring
            return True, (
                f"the {self.outlier_origin} check raised {type(exc).__name__}: {exc} "
                f"-- refusing rather than guessing"
            )
        if flagged:
            return True, reason or (
                f"{self.outlier_origin} calls {fmt_cents(landed)} implausible against "
                f"a {fmt_cents(ref.median or 0)} median"
            )
        return False, ""

    def _finish(
        self,
        product_id: str,
        action: Action,
        now: float,
        reasons: List[str],
        *,
        obs: Optional[Observation] = None,
        market: Optional[Cents] = None,
        discount: Optional[float] = None,
        quantity: int = 0,
        seen_in_stock: bool = False,
    ) -> Verdict:
        """Build the verdict, guarantee its reasons, and record it.

        Every return path in :meth:`evaluate` goes through here, which is
        what makes "``reasons`` is never empty" and "the watch state is
        always updated" true by construction rather than by review.
        """
        if not reasons:  # pragma: no cover - no path leaves them empty
            reasons = [f"{action.value}: no reason recorded, which is itself a bug"]
        verdict = Verdict(
            product_id=product_id,
            action=action,
            at=now,
            source=obs.source if obs is not None else None,
            sku=obs.sku if obs is not None else None,
            price=obs.price if obs is not None else None,
            landed=obs.landed if obs is not None else None,
            market=market,
            discount_pct=discount,
            quantity=quantity if action is Action.BUY else 0,
            url=(obs.url if obs is not None else "") or self._catalog_url(product_id, obs),
            reasons=tuple(reasons),
        )
        try:
            self.record_verdict(verdict, seen_in_stock=seen_in_stock)
        except BudgetError as exc:
            if action is not Action.BUY:
                raise
            # The ledger moved under us.  Under the lock evaluate() holds
            # this should be unreachable, but a BudgetError escaping into
            # a polling loop -- after the cooldown had already been
            # stamped for an alert nobody sent -- is the one outcome
            # worth spending a branch on.
            return self._finish(
                product_id,
                Action.SKIP,
                now,
                list(reasons) + [f"the ledger refused to hold the money: {exc}"],
                obs=obs,
                market=market,
                discount=discount,
                seen_in_stock=seen_in_stock,
            )
        return verdict

    def _catalog_url(self, product_id: str, obs: Optional[Observation]) -> str:
        """A deep link even when the observation carried none."""
        if obs is None or self.catalog is None:
            return ""
        sku = getattr(self.catalog, "sku", None)
        if not callable(sku):
            return ""
        entry = sku(obs.source, product_id)
        return getattr(entry, "url", "") if entry is not None else ""

    def __repr__(self) -> str:
        return (
            f"<DecisionEngine {len(getattr(self.rules, 'rules', list)())} rules, "
            f"outlier gate {self.outlier_origin}, "
            f"{len(self.reservations())} reservation(s)>"
        )


# --------------------------------------------------------------------------
# small pure helpers
# --------------------------------------------------------------------------


def _cost(obs: Observation, rule: Rule) -> Cents:
    """What this listing costs under this rule: landed, or shelf price."""
    if rule.include_shipping:
        return obs.landed if obs.landed is not None else 0
    return obs.price if obs.price is not None else 0


def _newest_per_listing(observations: Iterable[Observation]) -> List[Observation]:
    """One observation per ``(source, sku)``: the most recent.

    Stable for equal timestamps -- the first one wins -- so a caller
    handing the same list twice gets the same verdict twice.
    """
    newest: Dict[Tuple[str, str], Observation] = {}
    for obs in observations:
        key = (obs.source, obs.sku)
        previous = newest.get(key)
        if previous is None or obs.at > previous.at:
            newest[key] = obs
    return [newest[key] for key in sorted(newest)]


def _rule_summary(rule: Rule) -> str:
    bits = [f"rule: up to {fmt_cents(rule.max_price)}"]
    bits.append("landed" if rule.include_shipping else "before shipping")
    if rule.quantity > 1:
        bits.append(f"x{rule.quantity}")
    if rule.min_discount_pct > 0:
        bits.append(f"at {rule.min_discount_pct:g}% off or better")
    if rule.allowed_sources:
        bits.append("from " + ", ".join(rule.allowed_sources))
    return " ".join(bits)


def _stock_summary(observations: Sequence[Observation]) -> str:
    if not observations:
        return "no listings observed at all"
    counts: Dict[str, int] = {}
    for obs in observations:
        label = obs.stock.value if obs.price is not None else f"{obs.stock.value}, no price"
        counts[label] = counts.get(label, 0) + 1
    seen = ", ".join(f"{label} x{n}" for label, n in sorted(counts.items()))
    return f"nothing purchasable across {len(observations)} listing(s): {seen}"


def _why_unusable(ref: Optional[MarketRef]) -> str:
    if ref is None:
        return "no history for this product yet"
    parts = []
    if getattr(ref, "median", None) is None:
        parts.append("no median")
    if getattr(ref, "samples", 0) < 3:
        parts.append(f"only {getattr(ref, 'samples', 0)} sample(s)")
    if getattr(ref, "stale", False):
        parts.append("the samples are stale")
    return ", ".join(parts) or "the history says it is not to be trusted"


if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    from jarvis_poke.contracts import Budget, Stock

    clock_at = [1_700_000_000.0]
    rules = RuleSet(
        [Rule(product_id="demo-etb", max_price=5999, quantity=2, min_discount_pct=10.0)],
        Budget(total=30000),
    )

    class _History:
        def market_ref(self, product_id: str, now: float) -> MarketRef:
            return MarketRef(product_id, samples=9, median=6999, p25=6499,
                             low=5899, window_s=30 * 86400.0)

    engine = DecisionEngine(None, rules, _History(), lambda: clock_at[0])
    looks = [
        Observation("demo-etb", "examplemart", "EM-1", clock_at[0], Stock.IN_STOCK, 5499,
                    shipping=0, url="https://examplemart.example.com/p/demo-etb"),
        Observation("demo-etb", "cardbarn", "CB-1", clock_at[0], Stock.IN_STOCK, 5799,
                    shipping=499, url="https://cardbarn.example.com/p/demo-etb"),
    ]
    first = engine.evaluate("demo-etb", looks)
    print(explain(first))
    second = engine.evaluate("demo-etb", looks)
    print(explain(second))
    clock_at[0] += 3600.0
    print(explain(engine.evaluate("demo-etb", looks)))
    print(repr(engine), rules.to_obj()["budget"]["display"])
