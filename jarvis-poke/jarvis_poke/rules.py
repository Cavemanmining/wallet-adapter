"""What the owner is willing to buy, and the money standing behind it.

Implements the "Rules: what the owner is willing to buy" section of
:mod:`jarvis_poke.contracts` -- :class:`~jarvis_poke.contracts.Rule` and
:class:`~jarvis_poke.contracts.Budget` -- together with
:data:`~jarvis_poke.contracts.MAX_BUDGET_FRACTION_PER_VERDICT`, the cap on
how much of what is left a single verdict may commit.

Three jobs
----------
* **Hold the rules.**  One rule per ``product_id``, because that is the key
  :meth:`jarvis_poke.engine.DecisionEngine.evaluate` looks a rule up by.
  :meth:`RuleSet.add` refuses a second rule for a product instead of letting
  two standing instructions fight over one listing, and the error says
  *which sources* the two overlap on so the owner can merge them.
* **Hold the budget.**  :meth:`RuleSet.reserve`, :meth:`RuleSet.release`,
  :meth:`RuleSet.commit` and :meth:`RuleSet.remaining` are the whole of the
  accounting.  A BUY verdict reserves; a purchase the owner actually made
  commits; a deep link the owner ignored releases.  ``remaining()`` is net
  of reservations, so two verdicts in one pass cannot each spend the same
  money.  :meth:`RuleSet.reserve_for`, :meth:`RuleSet.release_for` and
  :meth:`RuleSet.commit_for` do the same *keyed by product id*, which is
  what makes a second BUY for one product replace its reservation rather
  than stack a second one -- even when the two verdicts came from
  different engine objects, or different runs restored through
  :meth:`RuleSet.load_reservations`.

Thread safety
-------------
Every ledger mutation is made under one re-entrant lock, and
:attr:`RuleSet.ledger_lock` exposes it, because "is this affordable?"
followed by "reserve it" is a single decision.  A caller that spans the
two without holding the lock -- as the engine used to -- lets a second
evaluation see money that is about to be promised, and both are told
they fit the per-verdict cap.
* **Say what is affordable.**  :func:`affordable` applies
  ``MAX_BUDGET_FRACTION_PER_VERDICT``: a single verdict may commit at most
  that fraction of what is left, so one mis-parsed price cannot propose
  spending the lot.  It *refuses* rather than quietly buying fewer copies
  than the rule asked for -- see "Judgement calls" below.

Money is integer cents
----------------------
contracts.py, "Money is integer cents": nothing here is a float.  The one
fractional constant, ``MAX_BUDGET_FRACTION_PER_VERDICT``, is applied
through :class:`fractions.Fraction` and floored, so the per-verdict cap is
always a whole number of cents and the rounding always favours the owner
(the cap can only come out lower, never higher).  An ``int``-typed field
that arrives as a ``float`` -- or as a ``bool``, which is an ``int`` in
Python and would otherwise sail through -- is rejected rather than
coerced.

No clock, no randomness, no network
-----------------------------------
Nothing in this module reads a clock, draws a random number or opens a
socket.  :attr:`Budget.window_s` is carried but *not* interpreted here:
contracts.py gives a window length with no window start, so rolling the
window over is the caller's job -- do it by handing
:meth:`RuleSet.set_budget` a fresh :class:`~jarvis_poke.contracts.Budget`.

Judgement calls
---------------
* **Refuse, do not shrink.**  When ``quantity * landed`` is over the cap,
  :func:`affordable` says no.  It could instead propose fewer copies, but
  alerting "buy 1" when the owner wrote "buy 3" is a decision the owner did
  not make; a refusal with the numbers in it lets them raise the budget or
  lower the quantity themselves.
* **One rule per product.**  Two rules for one product with disjoint
  ``allowed_sources`` are arguably not in conflict, but the engine looks up
  exactly one rule per product id, so a second one would silently never
  fire.  Refusing at :meth:`RuleSet.add` is louder than that.
"""

from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from fractions import Fraction
from typing import Any, Dict, Iterable, Iterator, Iterable as _Iterable, List, Mapping, Optional, Tuple, Union

from jarvis_poke.contracts import (
    MAX_BUDGET_FRACTION_PER_VERDICT,
    Budget,
    Cents,
    Rule,
    fmt_cents,
)

__all__ = [
    "Affordability",
    "BudgetError",
    "RuleConflict",
    "RuleError",
    "RuleSet",
    "affordable",
    "budget_cap",
]


class RuleError(ValueError):
    """A rule, or a budget operation, that cannot be trusted.

    The message always names the product id or the amounts involved: a
    money error that does not say which money is a bug report nobody can
    act on.
    """


class RuleConflict(RuleError):
    """Two rules that overlap, or one rule that contradicts itself."""


class BudgetError(RuleError):
    """A reserve/release/commit that the ledger cannot honour."""


# --------------------------------------------------------------------------
# the per-verdict cap
# --------------------------------------------------------------------------


def budget_cap(remaining: Cents, fraction: float = MAX_BUDGET_FRACTION_PER_VERDICT) -> Cents:
    """Most one verdict may commit out of ``remaining`` cents.

    ``contracts.MAX_BUDGET_FRACTION_PER_VERDICT`` is a float, and this is
    the only place in the package that touches it.  It is converted with
    :class:`~fractions.Fraction` and floored, so the answer is an exact
    number of cents that is never above the true fraction.
    """
    remaining = _cents(remaining, "remaining")
    if remaining <= 0:
        return 0
    if not 0.0 < float(fraction) <= 1.0:
        raise BudgetError(
            f"budget fraction must be in (0, 1], got {fraction!r}"
        )
    return math.floor(Fraction(remaining) * Fraction(float(fraction)))


@dataclass(frozen=True)
class Affordability:
    """Why one verdict's spend is, or is not, allowed.

    Truthy exactly when :attr:`ok`, so ``if not rules.affordable(...)``
    reads the way a caller expects, while the numbers stay available for
    the verdict's reasons.
    """

    ok: bool
    quantity: int
    unit: Cents
    total: Cents
    cap: Cents
    remaining: Cents
    reason: str

    def __bool__(self) -> bool:
        return self.ok


def affordable(
    rule: Rule,
    landed: Cents,
    budget: Union[Budget, int],
    quantity: Optional[int] = None,
) -> Affordability:
    """Can this verdict commit ``quantity * landed`` right now?

    Applies :data:`~jarvis_poke.contracts.MAX_BUDGET_FRACTION_PER_VERDICT`:
    the spend must fit inside :func:`budget_cap` of what is left, which is
    half of it by default, so a single verdict can never empty the budget
    and a single mis-parsed price can never propose that it should.

    ``budget`` is a :class:`~jarvis_poke.contracts.Budget` or a plain
    ``int`` of remaining cents -- :class:`RuleSet` passes the latter,
    because its own ``remaining()`` is already net of reservations.
    ``quantity`` defaults to ``rule.quantity``; the engine passes the
    quantity it has already clamped to the listing's per-customer limit.
    """
    if not isinstance(rule, Rule):
        raise RuleError(f"not a Rule: {rule!r}")
    remaining = budget.remaining if isinstance(budget, Budget) else _cents(budget, "budget")
    quantity = rule.quantity if quantity is None else quantity
    if isinstance(quantity, bool) or not isinstance(quantity, int):
        raise RuleError(f"quantity must be an int, got {quantity!r}")
    unit = _cents(landed, "landed price")
    cap = budget_cap(remaining)

    if quantity < 1:
        return Affordability(
            False, quantity, unit, 0, cap, remaining,
            f"quantity is {quantity}: nothing to buy",
        )
    if unit <= 0:
        # The engine's outlier gate should have caught this already; the
        # helper refuses on its own account so that no caller can talk it
        # into a free purchase.
        return Affordability(
            False, quantity, unit, unit * quantity, cap, remaining,
            f"refusing a non-positive price of {fmt_cents(unit)}",
        )

    total = unit * quantity
    if remaining <= 0:
        return Affordability(
            False, quantity, unit, total, cap, remaining,
            f"budget exhausted: {fmt_cents(0)} remaining",
        )
    if total > cap:
        return Affordability(
            False, quantity, unit, total, cap, remaining,
            f"{quantity} x {fmt_cents(unit)} = {fmt_cents(total)} is over the "
            f"{fmt_cents(cap)} one verdict may commit "
            f"({int(MAX_BUDGET_FRACTION_PER_VERDICT * 100)}% of {fmt_cents(remaining)} left)",
        )
    return Affordability(
        True, quantity, unit, total, cap, remaining,
        f"{quantity} x {fmt_cents(unit)} = {fmt_cents(total)} fits the "
        f"{fmt_cents(cap)} single-verdict cap ({fmt_cents(remaining)} left)",
    )


# --------------------------------------------------------------------------
# the rule set
# --------------------------------------------------------------------------


class RuleSet:
    """The owner's rules, plus the ledger they spend from.

    Ordering is deterministic: rules come back sorted by ``product_id``.
    Nothing here reads a clock or draws a random number.
    """

    def __init__(
        self,
        rules: Iterable[Rule] = (),
        budget: Optional[Budget] = None,
    ) -> None:
        self._rules: Dict[str, Rule] = {}
        self._budget = _checked_budget(budget if budget is not None else Budget(total=0))
        self._reserved: Cents = 0
        #: Reservations that belong to a named thing -- in practice a
        #: product id.  Keyed so that one product's second BUY *replaces*
        #: its first reservation rather than stacking a second one, no
        #: matter which engine object (or which process, via the store)
        #: produced it.  Their total is included in ``_reserved``.
        self._holds: Dict[str, Cents] = {}
        #: Re-entrant, because ``reserve_for`` calls ``release``/``reserve``
        #: and the engine holds it across a whole ``evaluate``.
        self._lock = threading.RLock()
        for rule in rules:
            self.add(rule)

    # -- rules -------------------------------------------------------------

    def add(self, rule: Rule, *, replace_existing: bool = False) -> Rule:
        """Add one rule.

        Raises :class:`RuleConflict` when a rule for the product already
        exists, naming the sources the two overlap on, unless
        ``replace_existing`` says the caller means to overwrite it.
        """
        _check_rule(rule)
        existing = self._rules.get(rule.product_id)
        if existing is not None and not replace_existing:
            raise RuleConflict(_overlap_message(existing, rule))
        self._rules[rule.product_id] = rule
        return rule

    def remove(self, product_id: str) -> Rule:
        """Remove and return one rule.  Raises for an unknown product."""
        try:
            return self._rules.pop(str(product_id))
        except KeyError:
            raise RuleError(f"no rule for product {product_id!r}") from None

    def get(self, product_id: str) -> Optional[Rule]:
        """The rule for a product, or ``None``.

        ``None`` rather than an exception: "the owner has no rule for this"
        is an ordinary answer, and the engine turns it into a SKIP verdict
        with a reason rather than a traceback.
        """
        return self._rules.get(str(product_id))

    def rules(self) -> List[Rule]:
        """Every rule, enabled or not, sorted by product id."""
        return [self._rules[pid] for pid in sorted(self._rules)]

    def enabled_rules(self) -> List[Rule]:
        """Only the rules the owner has switched on, sorted by product id."""
        return [rule for rule in self.rules() if rule.enabled]

    def product_ids(self) -> List[str]:
        return sorted(self._rules)

    def set_enabled(self, product_id: str, enabled: bool) -> Rule:
        """Switch one rule on or off, keeping everything else about it."""
        rule = self.get(product_id)
        if rule is None:
            raise RuleError(f"no rule for product {product_id!r}")
        updated = replace(rule, enabled=bool(enabled))
        self._rules[updated.product_id] = updated
        return updated

    # -- budget ------------------------------------------------------------

    @property
    def budget(self) -> Budget:
        """The current budget.  Frozen; change it with :meth:`set_budget`."""
        return self._budget

    @property
    def reserved(self) -> Cents:
        """Cents promised to verdicts that have neither committed nor lapsed."""
        return self._reserved

    @property
    def ledger_lock(self) -> "threading.RLock":
        """The re-entrant lock every ledger mutation is made under.

        Public because "check what is affordable, then reserve it" is one
        decision, not two, and the caller that spans both -- in practice
        :meth:`jarvis_poke.engine.DecisionEngine.evaluate` -- has to hold
        the lock for the whole of it.  Every method here takes it too, so
        a caller that does not bother still cannot corrupt the ledger; it
        can only be told a stale ``remaining()``.
        """
        return self._lock

    @contextmanager
    def transaction(self):
        """``with rules.transaction():`` -- the ledger cannot move inside.

        Sugar over :attr:`ledger_lock` so a caller does not have to know
        it is an ``RLock``.
        """
        with self._lock:
            yield self

    def remaining(self) -> Cents:
        """Cents free to spend: total, less spent, less reserved.

        Never negative.  This, not ``budget.remaining``, is what
        :meth:`affordable` measures against, because money already
        promised to an un-acted-on BUY is not money a second BUY may
        promise again.
        """
        with self._lock:
            return max(0, self._budget.remaining - self._reserved)

    def set_budget(self, budget: Budget) -> Budget:
        """Replace the budget -- the way a caller rolls the window over.

        Reservations survive, because they refer to deep links already sent
        to a phone.  Raises when the new budget cannot cover what is
        already reserved, since that would be a ledger that does not add
        up.
        """
        checked = _checked_budget(budget)
        with self._lock:
            if checked.remaining < self._reserved:
                raise BudgetError(
                    f"new budget leaves {fmt_cents(checked.remaining)} but "
                    f"{fmt_cents(self._reserved)} is already reserved; release first"
                )
            previous, self._budget = self._budget, checked
            return previous

    def reserve(self, amount: Cents) -> Cents:
        """Set aside cents for a verdict that has just been alerted on.

        Returns what is left afterwards.  Unkeyed: prefer
        :meth:`reserve_for` when the money belongs to one product, so a
        second verdict for it replaces the reservation instead of adding
        to it.
        """
        amount = _positive(amount, "reserve")
        with self._lock:
            free = self.remaining()
            if amount > free:
                raise BudgetError(
                    f"cannot reserve {fmt_cents(amount)}: only {fmt_cents(free)} is free"
                )
            self._reserved += amount
            return self.remaining()

    def release(self, amount: Cents) -> Cents:
        """Give reserved cents back -- the owner did not use the deep link."""
        amount = _positive(amount, "release")
        with self._lock:
            if amount > self._reserved:
                raise BudgetError(
                    f"cannot release {fmt_cents(amount)}: only "
                    f"{fmt_cents(self._reserved)} is reserved"
                )
            self._reserved -= amount
            return self.remaining()

    def release_all(self) -> Cents:
        """Drop every reservation, keyed or not.  Returns the cents handed
        back."""
        with self._lock:
            released, self._reserved = self._reserved, 0
            self._holds.clear()
            return released

    def commit(self, amount: Cents) -> Cents:
        """Record cents actually spent.

        Reserved cents are consumed first, so the ordinary
        reserve-then-commit path does not double-count.  Committing more
        than the budget can cover raises rather than going overdrawn.
        """
        amount = _positive(amount, "commit")
        with self._lock:
            from_reserved = min(self._reserved, amount)
            rest = amount - from_reserved
            if rest > self.remaining():
                raise BudgetError(
                    f"cannot commit {fmt_cents(amount)}: {fmt_cents(from_reserved)} is "
                    f"reserved and only {fmt_cents(self.remaining())} more is free"
                )
            self._reserved -= from_reserved
            self._budget = replace(self._budget, spent=self._budget.spent + amount)
            return self.remaining()

    # -- reservations that belong to a product -----------------------------
    #
    # The keyed half of the ledger.  It lives here rather than on the
    # engine because an app that builds one engine per run (which is
    # exactly what ``cli.cmd_decide`` does) against one long-lived RuleSet
    # would otherwise book the same listing again on every run, with no
    # live object left holding the reservation to release it.

    def reservations(self) -> Dict[str, Cents]:
        """``{key: cents}`` for every keyed reservation."""
        with self._lock:
            return dict(self._holds)

    def reserved_for(self, key: str) -> Cents:
        """Cents this key is holding, or 0."""
        with self._lock:
            return self._holds.get(str(key), 0)

    def reserve_for(self, key: str, amount: Cents) -> Cents:
        """Reserve ``amount`` against ``key``, *replacing* its last
        reservation.

        Atomic: the old hold is released and the new one taken under one
        lock, so a failure leaves the key holding what it held before.
        """
        key = str(key)
        amount = _positive(amount, "reserve")
        with self._lock:
            previous = self._holds.pop(key, 0)
            if previous:
                self.release(previous)
            try:
                self.reserve(amount)
            except BudgetError:
                if previous:
                    self.reserve(previous)
                    self._holds[key] = previous
                raise
            self._holds[key] = amount
            return self.remaining()

    def release_for(self, key: str) -> Cents:
        """Drop one key's reservation.  Returns the cents handed back."""
        key = str(key)
        with self._lock:
            amount = self._holds.pop(key, 0)
            if amount:
                self.release(amount)
            return amount

    def commit_for(self, key: str, amount: Optional[Cents] = None) -> Cents:
        """The owner bought it: turn one key's reservation into spend.

        ``amount`` defaults to what was reserved; pass the real total when
        the checkout page disagreed, which it will (tax, a coupon, a
        shipping band).  Returns the cents committed.

        The money comes from *this key's* hold and then from what is
        free -- never from another product's reservation.  Releasing and
        then calling :meth:`commit` would do exactly that, because
        ``commit`` consumes reserved cents first and cannot tell whose
        they are; the later release of the reservation it ate then
        raises.
        """
        key = str(key)
        with self._lock:
            held = self._holds.get(key, 0)
            spend = held if amount is None else _cents(amount, "amount to commit")
            if spend < 0:
                raise BudgetError(f"cannot commit {spend} for {key!r}: negative")
            other = self._reserved - held
            if spend > self._budget.remaining - other:
                raise BudgetError(
                    f"cannot commit {fmt_cents(spend)} for {key!r}: "
                    f"{fmt_cents(other)} is reserved elsewhere and only "
                    f"{fmt_cents(max(0, self._budget.remaining - other))} is free"
                )
            self._holds.pop(key, None)
            self._reserved -= held
            if spend:
                self._budget = replace(
                    self._budget, spent=self._budget.spent + spend
                )
            return spend

    def load_reservations(self, holds: Mapping[str, Cents]) -> Cents:
        """Restore keyed reservations read back from a store.

        Replaces every hold this ledger has.  Raises without changing
        anything when the budget cannot cover them, because a ledger that
        does not add up is worse than one that has forgotten.
        """
        wanted = {str(k): _cents(v, f"reservation for {k!r}") for k, v in dict(holds).items()}
        for key, amount in wanted.items():
            if amount < 0:
                raise BudgetError(f"reservation for {key!r} must not be negative, got {amount}")
        total = sum(amount for amount in wanted.values() if amount > 0)
        with self._lock:
            unkeyed = self._reserved - sum(self._holds.values())
            if total + unkeyed > self._budget.remaining:
                raise BudgetError(
                    f"cannot restore {fmt_cents(total)} of reservations against a "
                    f"budget with {fmt_cents(self._budget.remaining)} left"
                )
            self._holds = {k: v for k, v in wanted.items() if v > 0}
            self._reserved = total + unkeyed
            return total

    def affordable(
        self,
        rule: Rule,
        landed: Cents,
        quantity: Optional[int] = None,
        budget: Union[Budget, int, None] = None,
    ) -> Affordability:
        """:func:`affordable`, measured against this ledger's
        :meth:`remaining` (so reservations count) unless a caller passes
        its own ``budget``."""
        if budget is None:
            budget = self.remaining()
        return affordable(rule, landed, budget, quantity)

    # -- review ------------------------------------------------------------

    def unsatisfiable(self) -> List[Tuple[str, str]]:
        """Rules that cannot fire as things stand, with the reason.

        A warning list, not an error: the budget moves, so a rule that is
        unsatisfiable this minute may be fine next month.  The page shows
        this so a rule that will never alert does not look like a rule that
        simply has not triggered yet.
        """
        out: List[Tuple[str, str]] = []
        for rule in self.rules():
            if not rule.enabled:
                out.append((rule.product_id, "rule is disabled"))
                continue
            verdict = self.affordable(rule, rule.max_price)
            if not verdict.ok:
                out.append((
                    rule.product_id,
                    f"even at its {fmt_cents(rule.max_price)} ceiling, {verdict.reason}",
                ))
        return out

    # -- conveniences ------------------------------------------------------

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self) -> Iterator[Rule]:
        return iter(self.rules())

    def __contains__(self, product_id: object) -> bool:
        return product_id in self._rules

    def __repr__(self) -> str:
        return (
            f"<RuleSet {len(self._rules)} rules "
            f"({len(self.enabled_rules())} enabled), "
            f"{fmt_cents(self.remaining())} free of {fmt_cents(self._budget.total)}>"
        )

    def to_obj(self) -> Dict[str, Any]:
        """A JSON-ready view for the page.  Money stays in cents; the
        formatted strings are alongside, not instead."""
        return {
            "budget": {
                "total": self._budget.total,
                "spent": self._budget.spent,
                "reserved": self._reserved,
                "reservations": self.reservations(),
                "remaining": self.remaining(),
                "per_verdict_cap": budget_cap(self.remaining()),
                "window_s": self._budget.window_s,
                "display": {
                    "total": fmt_cents(self._budget.total),
                    "spent": fmt_cents(self._budget.spent),
                    "reserved": fmt_cents(self._reserved),
                    "remaining": fmt_cents(self.remaining()),
                },
            },
            "rules": [
                {
                    "product_id": r.product_id,
                    "max_price": r.max_price,
                    "quantity": r.quantity,
                    "min_discount_pct": r.min_discount_pct,
                    "allowed_sources": list(r.allowed_sources),
                    "include_shipping": r.include_shipping,
                    "cooldown_s": r.cooldown_s,
                    "enabled": r.enabled,
                }
                for r in self.rules()
            ],
        }


# --------------------------------------------------------------------------
# validation helpers
# --------------------------------------------------------------------------


def _cents(value: Any, what: str) -> Cents:
    """An ``int`` number of cents.  ``bool`` and ``float`` are refused."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuleError(
            f"{what} must be an integer number of cents "
            f"(money is never a float in this package), got {value!r}"
        )
    return value


def _positive(value: Any, what: str) -> Cents:
    amount = _cents(value, f"amount to {what}")
    if amount <= 0:
        raise BudgetError(f"amount to {what} must be positive, got {amount}")
    return amount


def _checked_budget(budget: Budget) -> Budget:
    if not isinstance(budget, Budget):
        raise RuleError(f"not a Budget: {budget!r}")
    total = _cents(budget.total, "budget total")
    spent = _cents(budget.spent, "budget spent")
    if total < 0:
        raise RuleError(f"budget total must not be negative, got {total}")
    if spent < 0:
        raise RuleError(f"budget spent must not be negative, got {spent}")
    return budget


def _check_rule(rule: Rule) -> Rule:
    """Everything ``Rule.__post_init__`` does not check.

    contracts.py validates ``max_price``, ``quantity`` and
    ``min_discount_pct`` ranges on construction; it cannot check that
    ``max_price`` is an ``int`` rather than a ``float`` (``189.99 > 0`` is
    perfectly true), nor that the sources list makes sense.  Those are the
    ones that cost money, so they are checked here.
    """
    if not isinstance(rule, Rule):
        raise RuleError(f"not a Rule: {rule!r}")
    if not isinstance(rule.product_id, str) or not rule.product_id.strip():
        raise RuleError(f"rule product_id must be a non-empty string, got {rule.product_id!r}")
    where = f"rule for {rule.product_id!r}"
    _cents(rule.max_price, f"{where}: max_price")
    if isinstance(rule.quantity, bool) or not isinstance(rule.quantity, int):
        raise RuleError(f"{where}: quantity must be an int, got {rule.quantity!r}")
    if not isinstance(rule.allowed_sources, tuple):
        raise RuleError(
            f"{where}: allowed_sources must be a tuple, got "
            f"{type(rule.allowed_sources).__name__} -- a bare string would be "
            f"read one character at a time"
        )
    seen = set()
    for source in rule.allowed_sources:
        if not isinstance(source, str) or not source.strip():
            raise RuleError(f"{where}: allowed_sources holds {source!r}")
        if source in seen:
            raise RuleConflict(f"{where}: allowed_sources lists {source!r} twice")
        seen.add(source)
    cooldown = rule.cooldown_s
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)):
        raise RuleError(f"{where}: cooldown_s must be a number, got {cooldown!r}")
    if cooldown < 0 or cooldown != cooldown or cooldown == float("inf"):
        raise RuleError(f"{where}: cooldown_s must be a non-negative, finite number of seconds")
    return rule


def _overlap_message(existing: Rule, incoming: Rule) -> str:
    """Say precisely how two rules for one product collide."""
    lhs, rhs = set(existing.allowed_sources), set(incoming.allowed_sources)
    if not lhs or not rhs:
        scope = "both apply to every source"
    elif lhs & rhs:
        shared = ", ".join(sorted(lhs & rhs))
        scope = f"both apply to {shared}"
    else:
        scope = (
            "their sources do not overlap, but the engine looks up exactly one "
            "rule per product id, so the second would never fire"
        )
    return (
        f"a rule for {incoming.product_id!r} already exists "
        f"(ceiling {fmt_cents(existing.max_price)} vs {fmt_cents(incoming.max_price)}): "
        f"{scope}. Merge them, or pass replace_existing=True."
    )


if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    demo = RuleSet(
        [
            Rule(product_id="sv08-surging-sparks-etb", max_price=5999, quantity=2,
                 min_discount_pct=10.0, allowed_sources=("examplemart", "cardbarn")),
            Rule(product_id="sv08-surging-sparks-booster-box", max_price=17999,
                 enabled=False),
        ],
        Budget(total=40000),
    )
    print(repr(demo))
    for rule in demo.rules():
        check = demo.affordable(rule, rule.max_price)
        print(f"  {rule.product_id:<38} {'on ' if rule.enabled else 'off'} {check.reason}")
    demo.reserve(11998)
    print(f"  after reserving {fmt_cents(11998)}: {fmt_cents(demo.remaining())} free, "
          f"cap now {fmt_cents(budget_cap(demo.remaining()))}")
    demo.commit(11998)
    print(f"  after committing: {demo!r}")
    for pid, why in demo.unsatisfiable():
        print(f"  unsatisfiable: {pid}: {why}")
