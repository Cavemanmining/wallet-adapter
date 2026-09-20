"""Tests for the jarvis_poke rule set and decision engine.

Design: jarvis_poke/contracts.py -- "Rules: what the owner is willing to
buy" (Rule, Budget) and "Market reference and verdicts" (MarketRef,
Action, Verdict, WatchState), plus MAX_BUDGET_FRACTION_PER_VERDICT.

This is the module that decides how money gets spent, so most of the file
is written as boundaries rather than examples: exactly at the ceiling
buys and one cent over watches; exactly at the required discount buys and
one cent short watches; exactly at the single-verdict budget cap buys and
one cent over skips.  A test that only checks the middle of a range would
pass against an engine with every comparison off by one.

Everything is injected.  The clock is a ``SimClock``, the market
reference is a ``FixedHistory`` handing back a literal
:class:`MarketRef`, and the outlier gate is a local function, so nothing
here depends on a sibling lane's chosen thresholds -- with one deliberate
exception, ``test_wires_up_a_real_price_history``, which checks that the
engine actually fits the shipped :mod:`jarvis_poke.prices`.  Nothing
sleeps, nothing reaches the network, and what randomness the fuzz uses
comes from a ``lucifer_gen.seed`` stream so a failure reproduces exactly.
"""

from __future__ import annotations

import ast
import copy
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Runnable as `pytest tests/test_poke_engine.py` or
# `python3 tests/test_poke_engine.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_poke import engine as engine_module
from jarvis_poke.catalog import Catalog
from jarvis_poke.contracts import (
    MAX_BUDGET_FRACTION_PER_VERDICT,
    Action,
    Budget,
    MarketRef,
    Observation,
    Product,
    ProductKind,
    Rule,
    SourceSku,
    Stock,
    Verdict,
    WatchState,
)
from jarvis_poke.engine import (
    DecisionEngine,
    EngineError,
    discount_against,
    discount_threshold,
    explain,
    looks_mis_parsed,
)
from jarvis_poke.rules import (
    Affordability,
    BudgetError,
    RuleConflict,
    RuleError,
    RuleSet,
    affordable,
    budget_cap,
)
from lucifer_gen.seed import SeedFields

T0 = 1_700_000_000.0
PID = "sv08-surging-sparks-etb"
OTHER = "sv08-surging-sparks-booster-box"

#: Every boundary in the table is derived from this median, so the
#: arithmetic in the test ids is checkable by eye: 10% off $100.00 is
#: $90.00, and half of a $180.00 budget is $90.00.
MEDIAN = 10_000


# --------------------------------------------------------------------------
# doubles
# --------------------------------------------------------------------------


class SimClock:
    """An injected clock.  Nothing in the package may call time.time()."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class FixedHistory:
    """A market lane stand-in: one MarketRef per product, or none."""

    def __init__(self, ref: Optional[MarketRef] = None, **per_product: MarketRef) -> None:
        self.default = ref
        self.per_product = dict(per_product)
        self.calls: List[Tuple[str, float]] = []

    def market_ref(self, product_id: str, now: float) -> Optional[MarketRef]:
        self.calls.append((product_id, now))
        return self.per_product.get(product_id, self.default)


def market(median: Optional[int] = MEDIAN, samples: int = 9, stale: bool = False,
           product_id: str = PID) -> MarketRef:
    return MarketRef(
        product_id=product_id,
        samples=samples,
        median=median,
        p25=None if median is None else median - 500,
        low=None if median is None else median - 1500,
        window_s=30 * 86400.0,
        stale=stale,
    )


def quarter_gate(landed: Optional[int], ref: Optional[MarketRef]) -> bool:
    """The outlier gate the table runs with: under a quarter of the median.

    Local on purpose -- these tests pin the *engine's* behaviour around an
    outlier verdict, not the market lane's choice of threshold.
    """
    if landed is None or landed <= 0:
        return True
    if ref is None or ref.median is None:
        return False
    return landed * 4 < ref.median


def observation(
    source: str = "examplemart",
    price: Optional[int] = 9_000,
    *,
    product_id: str = PID,
    sku: Optional[str] = None,
    stock: Stock = Stock.IN_STOCK,
    shipping: int = 0,
    per_customer_limit: Optional[int] = None,
    at: float = T0,
    url: str = "",
) -> Observation:
    return Observation(
        product_id=product_id,
        source=source,
        sku=sku or f"{source[:2].upper()}-1",
        at=at,
        stock=stock,
        price=price,
        shipping=shipping,
        per_customer_limit=per_customer_limit,
        url=url or f"https://{source}.example.com/p/{product_id}",
    )


def rule(**kwargs: Any) -> Rule:
    params: Dict[str, Any] = {"product_id": PID, "max_price": 9_000}
    params.update(kwargs)
    return Rule(**params)


#: ``build(ref=None)`` means "this product has no market reference", so
#: the default has to be something else to say "the usual usable one".
UNSET = object()


def build(
    *,
    rules: Optional[RuleSet] = None,
    rule_: Optional[Rule] = None,
    budget: Optional[Budget] = None,
    ref: Any = UNSET,
    clock: Optional[SimClock] = None,
    catalog: Any = None,
    watch_states: Optional[Dict[str, WatchState]] = None,
    outlier_check: Optional[Callable[..., Any]] = quarter_gate,
    reserve_on_buy: bool = True,
) -> Tuple[DecisionEngine, SimClock, RuleSet]:
    clock = clock or SimClock()
    ref = market() if ref is UNSET else ref
    if rules is None:
        rules = RuleSet(
            [] if rule_ is None else [rule_],
            budget if budget is not None else Budget(total=1_000_000),
        )
    engine = DecisionEngine(
        catalog,
        rules,
        FixedHistory(ref),
        clock,
        watch_states,
        outlier_check=outlier_check,
        reserve_on_buy=reserve_on_buy,
    )
    return engine, clock, rules


def catalog_with(*product_ids: str) -> Catalog:
    cat = Catalog()
    for pid in product_ids:
        cat.add_product(Product(id=pid, name=pid.replace("-", " ").title(),
                                set_code="SV08", kind=ProductKind.ELITE_TRAINER_BOX))
        cat.add_sku(SourceSku(source="examplemart", product_id=pid, sku="EM-1",
                              url=f"https://examplemart.example.com/p/{pid}"))
    return cat


# --------------------------------------------------------------------------
# the table: every branch of the decision order, at its boundary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One trip through :meth:`DecisionEngine.evaluate`."""

    id: str
    observations: Tuple[Observation, ...]
    expected: Action
    reason_has: str
    rule: Optional[Rule] = None
    ref: Optional[MarketRef] = field(default_factory=market)
    budget: Budget = field(default_factory=lambda: Budget(total=1_000_000))
    quantity: int = 0
    source: Optional[str] = None
    catalog_ids: Tuple[str, ...] = ()
    watch: Optional[WatchState] = None


CASES: List[Case] = [
    # -- 1. a rule, and an enabled one ------------------------------------
    Case(
        id="no-rule-at-all",
        rule=None,
        observations=(observation(),),
        expected=Action.SKIP,
        reason_has="no rule for",
    ),
    Case(
        id="rule-disabled",
        rule=rule(enabled=False),
        observations=(observation(),),
        expected=Action.SKIP,
        reason_has="switched off",
    ),
    Case(
        id="product-not-in-catalog",
        rule=rule(),
        observations=(observation(),),
        expected=Action.SKIP,
        reason_has="not in the catalog",
        catalog_ids=(OTHER,),
    ),
    # -- 2. nothing purchasable -------------------------------------------
    Case(
        id="no-stock-everywhere",
        rule=rule(),
        observations=(
            observation(stock=Stock.OUT_OF_STOCK, price=None),
            observation(source="cardbarn", stock=Stock.OUT_OF_STOCK, price=None),
        ),
        expected=Action.NO_STOCK,
        reason_has="nothing purchasable",
    ),
    Case(
        id="in-stock-but-price-unparseable",
        rule=rule(),
        observations=(observation(price=None),),
        expected=Action.NO_STOCK,
        reason_has="no price",
    ),
    Case(
        id="preorder-is-not-stock",
        rule=rule(),
        observations=(observation(stock=Stock.PREORDER, price=8_000),),
        expected=Action.NO_STOCK,
        reason_has="preorder",
    ),
    Case(
        id="limited-counts-as-stock",
        rule=rule(),
        observations=(observation(stock=Stock.LIMITED),),
        expected=Action.BUY,
        reason_has="purchasable",
        quantity=1,
    ),
    Case(
        id="no-observations-at-all",
        rule=rule(),
        observations=(),
        expected=Action.NO_STOCK,
        reason_has="no listings observed",
    ),
    # -- 3. allowed_sources ------------------------------------------------
    Case(
        id="allowed-sources-filters-the-cheaper-one-out",
        rule=rule(allowed_sources=("examplemart",)),
        observations=(
            observation(source="cardbarn", price=5_000),
            observation(source="examplemart", price=9_000),
        ),
        expected=Action.BUY,
        reason_has="the rule allows only examplemart",
        quantity=1,
        source="examplemart",
    ),
    Case(
        id="allowed-sources-leaves-nothing",
        rule=rule(allowed_sources=("hobbyhub",)),
        observations=(observation(source="cardbarn", price=5_000),),
        expected=Action.SKIP,
        reason_has="at no source the rule allows",
    ),
    # -- 4/7. the ceiling, to the cent ------------------------------------
    Case(
        id="exactly-at-max-price-buys",
        rule=rule(max_price=9_000),
        observations=(observation(price=9_000),),
        expected=Action.BUY,
        reason_has="at or under the $90.00 ceiling",
        quantity=1,
    ),
    Case(
        id="one-cent-over-max-price-watches",
        rule=rule(max_price=9_000),
        observations=(observation(price=9_001),),
        expected=Action.WATCH,
        reason_has="$0.01 over the $90.00 ceiling",
    ),
    Case(
        id="shipping-pushes-it-over-the-ceiling",
        rule=rule(max_price=9_000),
        observations=(observation(price=8_999, shipping=2),),
        expected=Action.WATCH,
        reason_has="over the $90.00 ceiling",
    ),
    Case(
        id="shipping-ignored-when-rule-says-so",
        rule=rule(max_price=9_000, include_shipping=False),
        observations=(observation(price=9_000, shipping=2_000),),
        expected=Action.BUY,
        reason_has="shipping excluded",
        quantity=1,
    ),
    Case(
        id="cheapest-landed-wins-not-cheapest-shelf",
        rule=rule(max_price=9_000),
        observations=(
            observation(source="cardbarn", price=8_600, shipping=600),
            observation(source="examplemart", price=8_800, shipping=100),
        ),
        expected=Action.BUY,
        reason_has="cheapest is examplemart at $89.00",
        quantity=1,
        source="examplemart",
    ),
    Case(
        id="only-the-newest-look-at-a-listing-counts",
        rule=rule(max_price=9_000),
        observations=(
            observation(price=5_000, at=T0 - 86_400),
            observation(price=9_001, at=T0),
        ),
        expected=Action.WATCH,
        reason_has="over the $90.00 ceiling",
    ),
    # -- 5. the market reference, or an honest absence ---------------------
    Case(
        id="unusable-reference-still-buys-under-the-ceiling",
        rule=rule(max_price=9_000, min_discount_pct=25.0),
        observations=(observation(price=9_000),),
        ref=market(samples=2),
        expected=Action.BUY,
        reason_has="no discount test applied",
        quantity=1,
    ),
    Case(
        id="stale-reference-is-not-a-reference",
        rule=rule(max_price=9_000, min_discount_pct=25.0),
        observations=(observation(price=9_000),),
        ref=market(stale=True),
        expected=Action.BUY,
        reason_has="the samples are stale",
        quantity=1,
    ),
    Case(
        id="absent-reference-is-not-invented",
        rule=rule(max_price=9_000, min_discount_pct=25.0),
        observations=(observation(price=9_000),),
        ref=None,
        expected=Action.BUY,
        reason_has="no history for this product yet",
        quantity=1,
    ),
    # -- 6. the outlier gate, ahead of any BUY -----------------------------
    Case(
        id="outlier-refused-though-far-under-the-ceiling",
        rule=rule(max_price=9_000),
        observations=(observation(price=2_000),),
        expected=Action.SKIP,
        reason_has="suspected mis-parse",
    ),
    Case(
        id="a-quarter-of-the-median-is-not-an-outlier",
        rule=rule(max_price=9_000),
        observations=(observation(price=2_500),),
        expected=Action.BUY,
        reason_has="plausible",
        quantity=1,
    ),
    Case(
        id="a-price-of-nothing-is-always-a-mis-parse",
        rule=rule(max_price=9_000),
        observations=(observation(price=0, stock=Stock.IN_STOCK),),
        expected=Action.SKIP,
        reason_has="is not a price",
    ),
    # -- 7. the discount, to the cent --------------------------------------
    Case(
        id="exactly-at-min-discount-buys",
        rule=rule(max_price=MEDIAN, min_discount_pct=10.0),
        observations=(observation(price=9_000),),
        expected=Action.BUY,
        reason_has="meets the 10% the rule asks for",
        quantity=1,
    ),
    Case(
        id="one-cent-short-of-min-discount-watches",
        rule=rule(max_price=MEDIAN, min_discount_pct=10.0),
        observations=(observation(price=9_001),),
        expected=Action.WATCH,
        reason_has="short of the 10%",
    ),
    Case(
        id="fractional-min-discount-holds-to-the-cent",
        rule=rule(max_price=MEDIAN, min_discount_pct=12.5),
        observations=(observation(price=8_750),),
        expected=Action.BUY,
        reason_has="meets the 12.5%",
        quantity=1,
    ),
    Case(
        id="zero-min-discount-means-no-discount-test",
        rule=rule(max_price=MEDIAN, min_discount_pct=0.0),
        observations=(observation(price=9_999),),
        expected=Action.BUY,
        reason_has="no particular discount",
        quantity=1,
    ),
    Case(
        id="above-market-but-under-the-ceiling-still-buys",
        rule=rule(max_price=12_000),
        observations=(observation(price=11_000),),
        expected=Action.BUY,
        reason_has="-10.00% off",
        quantity=1,
    ),
    # -- 8. quantity --------------------------------------------------------
    Case(
        id="per-customer-limit-clamps-the-quantity",
        rule=rule(max_price=9_000, quantity=3),
        observations=(observation(price=9_000, per_customer_limit=1),),
        expected=Action.BUY,
        reason_has="allows 1 per customer, the rule wanted 3",
        quantity=1,
    ),
    Case(
        id="per-customer-limit-above-the-rule-changes-nothing",
        rule=rule(max_price=9_000, quantity=2),
        observations=(observation(price=9_000, per_customer_limit=5),),
        expected=Action.BUY,
        reason_has="within examplemart's limit of 5",
        quantity=2,
    ),
    Case(
        id="per-customer-limit-of-zero-skips",
        rule=rule(max_price=9_000),
        observations=(observation(price=9_000, per_customer_limit=0),),
        expected=Action.SKIP,
        reason_has="nothing to buy",
    ),
    # -- 9. the budget ------------------------------------------------------
    Case(
        id="exactly-at-the-single-verdict-cap-buys",
        rule=rule(max_price=9_000),
        observations=(observation(price=9_000),),
        budget=Budget(total=18_000),
        expected=Action.BUY,
        reason_has="fits the $90.00 single-verdict cap",
        quantity=1,
    ),
    Case(
        id="one-cent-over-the-cap-skips",
        rule=rule(max_price=9_000),
        observations=(observation(price=9_000),),
        budget=Budget(total=17_998),
        expected=Action.SKIP,
        reason_has="is over the $89.99 one verdict may commit",
    ),
    Case(
        id="quantity-multiplies-against-the-cap",
        rule=rule(max_price=9_000, quantity=2),
        observations=(observation(price=9_000),),
        budget=Budget(total=18_000),
        expected=Action.SKIP,
        reason_has="2 x $90.00 = $180.00 is over",
    ),
    Case(
        id="budget-already-spent-skips",
        rule=rule(max_price=9_000),
        observations=(observation(price=9_000),),
        budget=Budget(total=18_000, spent=18_000),
        expected=Action.SKIP,
        reason_has="budget exhausted",
    ),
    # -- 10. the cooldown ---------------------------------------------------
    Case(
        id="inside-the-cooldown-skips",
        rule=rule(max_price=9_000, cooldown_s=3_600.0),
        observations=(observation(price=9_000),),
        watch=WatchState(product_id=PID, last_alert_at=T0 - 3_599.0),
        expected=Action.SKIP,
        reason_has="inside the 3600s cooldown",
    ),
    Case(
        id="exactly-at-the-cooldown-boundary-buys",
        rule=rule(max_price=9_000, cooldown_s=3_600.0),
        observations=(observation(price=9_000),),
        watch=WatchState(product_id=PID, last_alert_at=T0 - 3_600.0),
        expected=Action.BUY,
        reason_has="outside the cooldown",
        quantity=1,
    ),
]


def run(case: Case) -> Tuple[Verdict, DecisionEngine, RuleSet]:
    watch = {case.watch.product_id: replace(case.watch)} if case.watch else None
    engine, _clock, rules = build(
        rule_=case.rule,
        budget=case.budget,
        ref=case.ref,
        catalog=catalog_with(*case.catalog_ids) if case.catalog_ids else None,
        watch_states=watch,
    )
    return engine.evaluate(PID, list(case.observations)), engine, rules


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_decision_table(case: Case) -> None:
    verdict, _engine, _rules = run(case)
    joined = " | ".join(verdict.reasons)
    assert verdict.action is case.expected, f"{case.id}: {joined}"
    assert case.reason_has in joined, f"{case.id}: {joined}"
    assert verdict.quantity == case.quantity, f"{case.id}: {joined}"
    if case.source is not None:
        assert verdict.source == case.source


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_every_verdict_carries_reasons(case: Case) -> None:
    """contracts.py: ``reasons`` always explains the outcome, BUY included."""
    verdict, _engine, _rules = run(case)
    assert verdict.reasons, case.id
    assert all(isinstance(r, str) and r.strip() for r in verdict.reasons), case.id
    assert isinstance(verdict.reasons, tuple)


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_buy_never_outruns_the_budget(case: Case) -> None:
    """No BUY may commit more than the per-verdict cap allowed it."""
    cap_before = budget_cap(case.budget.remaining)
    verdict, _engine, _rules = run(case)
    if verdict.action is not Action.BUY:
        return
    cost = verdict.landed if verdict.landed is not None else verdict.price
    assert cost is not None
    assert verdict.quantity >= 1
    assert verdict.quantity * cost <= cap_before, case.id
    assert verdict.quantity * cost <= case.budget.remaining, case.id


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_evaluate_does_not_mutate_its_observations(case: Case) -> None:
    given = list(case.observations)
    before = copy.deepcopy(given)
    identities = [id(obs) for obs in given]
    run(replace(case, observations=tuple(given)))
    engine, _clock, _rules = build(rule_=case.rule, budget=case.budget, ref=case.ref)
    engine.evaluate(PID, given)
    assert given == before
    assert [id(obs) for obs in given] == identities


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_same_inputs_give_the_same_verdict(case: Case) -> None:
    """Determinism: no clock reading, no random draw, no dict-order luck."""
    first, _e1, _r1 = run(case)
    second, _e2, _r2 = run(case)
    assert first == second, case.id
    assert explain(first) == explain(second)


# --------------------------------------------------------------------------
# the cooldown, over time rather than in a table
# --------------------------------------------------------------------------


def test_cooldown_blocks_then_expires() -> None:
    engine, clock, _rules = build(
        rule_=rule(max_price=9_000, cooldown_s=3_600.0), ref=market()
    )
    looks = [observation(price=9_000)]

    first = engine.evaluate(PID, looks)
    assert first.action is Action.BUY
    state = engine.watch_state(PID)
    assert state.last_alert_at == T0
    assert state.alerts_sent == 1

    clock.tick(3_599.0)
    blocked = engine.evaluate(PID, [observation(price=9_000, at=clock.now)])
    assert blocked.action is Action.SKIP
    assert "cooldown" in " ".join(blocked.reasons)
    assert engine.watch_state(PID).alerts_sent == 1, "a blocked alert is not an alert"
    assert engine.watch_state(PID).last_alert_at == T0

    clock.tick(1.0)
    again = engine.evaluate(PID, [observation(price=9_000, at=clock.now)])
    assert again.action is Action.BUY
    assert engine.watch_state(PID).alerts_sent == 2
    assert engine.watch_state(PID).last_alert_at == clock.now


def test_a_second_buy_supersedes_its_own_reservation() -> None:
    """A product's own un-acted-on reservation must not price it out.

    Without this the second alert for a restock is refused by the money
    the first alert is still holding, which reads as "budget gone" when
    nothing has been spent at all.
    """
    engine, clock, rules = build(
        rule_=rule(max_price=9_000, cooldown_s=60.0), budget=Budget(total=20_000)
    )
    first = engine.evaluate(PID, [observation(price=9_000)])
    assert first.action is Action.BUY
    assert rules.reserved == 9_000
    assert engine.reservations() == {PID: 9_000}

    clock.tick(61.0)
    second = engine.evaluate(PID, [observation(price=9_000, at=clock.now)])
    assert second.action is Action.BUY, " | ".join(second.reasons)
    assert rules.reserved == 9_000, "the reservation is replaced, not stacked"


def test_watch_state_records_every_outcome() -> None:
    engine, clock, _rules = build(rule_=rule(max_price=9_000))
    engine.evaluate(PID, [observation(price=9_500)])
    state = engine.watch_state(PID)
    assert state.last_action is Action.WATCH
    assert state.alerts_sent == 0
    assert state.last_seen_in_stock == T0

    clock.tick(10.0)
    engine.evaluate(PID, [observation(price=None, stock=Stock.OUT_OF_STOCK, at=clock.now)])
    state = engine.watch_state(PID)
    assert state.last_action is Action.NO_STOCK
    assert state.last_seen_in_stock == T0, "out of stock does not count as seen in stock"


def test_the_caller_s_watch_states_are_updated_in_place() -> None:
    """A caller keeping states in a store must see the engine's writes."""
    states: Dict[str, WatchState] = {}
    engine, _clock, _rules = build(rule_=rule(max_price=9_000), watch_states=states)
    engine.evaluate(PID, [observation(price=9_000)])
    assert states[PID].alerts_sent == 1
    assert states[PID].last_alert_at == T0


# --------------------------------------------------------------------------
# evaluate_all
# --------------------------------------------------------------------------


def test_evaluate_all_covers_rules_and_observations_in_order() -> None:
    rules = RuleSet(
        [rule(product_id=PID, max_price=9_000), rule(product_id=OTHER, max_price=9_000)],
        Budget(total=1_000_000),
    )
    engine, _clock, _rules = build(rules=rules)
    looks = [
        observation(product_id=OTHER, price=9_000),
        observation(product_id=PID, price=9_000),
        observation(product_id="unruled-product", price=100),
    ]
    verdicts = engine.evaluate_all(looks)
    assert [v.product_id for v in verdicts] == sorted(
        [PID, OTHER, "unruled-product"]
    )
    by_id = {v.product_id: v for v in verdicts}
    assert by_id[PID].action is Action.BUY
    assert by_id[OTHER].action is Action.BUY
    assert by_id["unruled-product"].action is Action.SKIP
    assert all(v.reasons for v in verdicts)


def test_evaluate_all_spends_the_budget_once() -> None:
    """Two products, one budget: the second sees what the first reserved."""
    rules = RuleSet(
        [rule(product_id=PID, max_price=9_000), rule(product_id=OTHER, max_price=9_000)],
        Budget(total=20_000),
    )
    engine, _clock, _rules = build(rules=rules)
    verdicts = engine.evaluate_all([
        observation(product_id=PID, price=9_000),
        observation(product_id=OTHER, price=9_000),
    ])
    actions = {v.product_id: v.action for v in verdicts}
    assert actions[OTHER] is Action.BUY, "alphabetically first, so it goes first"
    assert actions[PID] is Action.SKIP
    assert rules.reserved == 9_000
    total = sum(
        v.quantity * (v.landed or 0) for v in verdicts if v.action is Action.BUY
    )
    assert total <= rules.budget.total


def test_evaluate_all_is_deterministic() -> None:
    def once() -> List[Verdict]:
        rules = RuleSet(
            [rule(product_id=PID, max_price=9_000),
             rule(product_id=OTHER, max_price=9_000)],
            Budget(total=20_000),
        )
        engine, _clock, _rules = build(rules=rules)
        return engine.evaluate_all([
            observation(product_id=OTHER, price=9_000),
            observation(product_id=PID, price=9_000),
        ])

    assert once() == once()


# --------------------------------------------------------------------------
# reservations, commits and releases
# --------------------------------------------------------------------------


def test_commit_and_release_move_the_ledger() -> None:
    engine, _clock, rules = build(rule_=rule(max_price=9_000), budget=Budget(total=40_000))
    verdict = engine.evaluate(PID, [observation(price=9_000)])
    assert verdict.action is Action.BUY
    assert rules.remaining() == 31_000

    engine.commit_purchase(PID, 9_250)  # the checkout page added tax
    assert rules.reserved == 0
    assert rules.budget.spent == 9_250
    assert rules.remaining() == 30_750
    assert engine.reservations() == {}

    engine2, _c2, rules2 = build(rule_=rule(max_price=9_000), budget=Budget(total=40_000))
    engine2.evaluate(PID, [observation(price=9_000)])
    assert engine2.release_reservation(PID) == 9_000
    assert rules2.remaining() == 40_000
    assert rules2.budget.spent == 0
    assert engine2.release_reservation(PID) == 0, "releasing twice is not a refund"

    engine3, _c3, rules3 = build(rule_=rule(max_price=9_000), budget=Budget(total=40_000))
    engine3.evaluate(PID, [observation(price=9_000)])
    assert engine3.commit_purchase(PID) == 9_000, "commits what was reserved"
    assert rules3.budget.spent == 9_000 and rules3.reserved == 0


def test_reserve_on_buy_can_be_switched_off() -> None:
    engine, _clock, rules = build(
        rule_=rule(max_price=9_000), budget=Budget(total=40_000), reserve_on_buy=False
    )
    assert engine.evaluate(PID, [observation(price=9_000)]).action is Action.BUY
    assert rules.reserved == 0
    assert engine.reservations() == {}


# --------------------------------------------------------------------------
# explain()
# --------------------------------------------------------------------------


def test_explain_is_one_short_sentence_ending_in_the_decisive_reason() -> None:
    engine, _clock, _rules = build(rule_=rule(max_price=9_000, quantity=2),
                                   budget=Budget(total=100_000))
    verdict = engine.evaluate(PID, [observation(price=9_000)])
    body = explain(verdict)
    assert body.startswith("Buy: 2 x ")
    assert "$90.00 from examplemart" in body
    assert "10.0% under the $100.00 market" in body
    assert body.endswith("never alerted before.")
    assert len(body) <= 300
    assert "\n" not in body


def test_explain_says_when_the_market_reference_was_dropped() -> None:
    engine, _clock, _rules = build(
        rule_=rule(max_price=9_000, min_discount_pct=20.0), ref=market(samples=1)
    )
    verdict = engine.evaluate(PID, [observation(price=9_000)])
    assert verdict.action is Action.BUY
    assert verdict.market is None, "an unusable reference is never reported as one"
    assert verdict.discount_pct is None
    assert "no discount test applied" in explain(verdict, max_chars=10_000)


def test_explain_carries_the_refusal() -> None:
    engine, _clock, _rules = build(rule_=rule(max_price=9_000))
    verdict = engine.evaluate(PID, [observation(price=2_000)])
    body = explain(verdict)
    assert body.startswith("Skipping: ")
    assert "mis-parse" in body


def test_explain_rejects_things_that_are_not_verdicts() -> None:
    with pytest.raises(EngineError):
        explain("buy it")  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the injected pieces
# --------------------------------------------------------------------------


def test_a_rules_object_without_a_ledger_fails_at_construction() -> None:
    class RulesOnly:
        def get(self, product_id: str) -> None:
            return None

        def affordable(self, *args: Any, **kwargs: Any) -> None:
            return None

    with pytest.raises(EngineError) as excinfo:
        DecisionEngine(None, RulesOnly(), None, SimClock())
    assert "reserve" in str(excinfo.value)
    # ... unless it is never going to be asked to reserve anything.
    DecisionEngine(None, RulesOnly(), None, SimClock(), reserve_on_buy=False)


def test_a_clock_is_required() -> None:
    with pytest.raises(EngineError) as excinfo:
        DecisionEngine(None, RuleSet(), None, None)
    assert "clock" in str(excinfo.value)


def test_history_shapes_the_engine_accepts() -> None:
    ref = market()

    class TwoArg:
        def market_ref(self, product_id: str, now: float) -> MarketRef:
            return ref

    class OneArg:
        def reference(self, product_id: str) -> MarketRef:
            return ref

    for history in (TwoArg(), OneArg(), lambda pid, now: ref):
        engine = DecisionEngine(None, RuleSet([rule(max_price=9_000)], Budget(total=1_000_000)),
                                history, SimClock(), outlier_check=quarter_gate)
        verdict = engine.evaluate(PID, [observation(price=9_000)])
        assert verdict.market == MEDIAN


def test_a_history_that_cannot_be_called_fails_at_construction() -> None:
    with pytest.raises(EngineError) as excinfo:
        DecisionEngine(None, RuleSet(), object(), SimClock())
    assert "market_ref" in str(excinfo.value)


def test_a_history_returning_junk_is_loud() -> None:
    engine = DecisionEngine(None, RuleSet([rule(max_price=9_000)], Budget(total=100_000)),
                            lambda pid, now: "about a hundred dollars", SimClock())
    with pytest.raises(EngineError):
        engine.evaluate(PID, [observation(price=9_000)])


def test_wires_up_a_real_price_history() -> None:
    """The engine must fit the shipped market lane, not a lookalike.

    Skipped rather than failed when that lane is not present: the two are
    written in parallel and this file has to stand on its own.
    """
    prices = pytest.importorskip("jarvis_poke.prices")
    history = prices.PriceHistory()
    for day in range(1, 7):
        history.append(observation(source="cardbarn", price=MEDIAN, at=T0 - 86_400 * day))
    engine = DecisionEngine(
        None,
        RuleSet([rule(max_price=9_000, min_discount_pct=10.0)], Budget(total=1_000_000)),
        history,
        SimClock(),
    )
    verdict = engine.evaluate(PID, [observation(price=9_000)])
    assert verdict.action is Action.BUY, " | ".join(verdict.reasons)
    assert verdict.market == MEDIAN
    # ... and the lane's own outlier gate is the one that answers.
    assert engine.outlier_origin == "prices.is_outlier"
    refused = engine.evaluate(OTHER, [observation(product_id=OTHER, price=1)])
    assert refused.action is Action.SKIP


def test_an_injected_outlier_gate_is_consulted_with_the_landed_price() -> None:
    seen: List[Tuple[Optional[int], Optional[MarketRef]]] = []

    def gate(landed: Optional[int], ref: Optional[MarketRef]) -> Tuple[bool, str]:
        seen.append((landed, ref))
        return landed == 9_500, "the shipping line was read as the price"

    engine, _clock, _rules = build(rule_=rule(max_price=10_000), outlier_check=gate)
    verdict = engine.evaluate(PID, [observation(price=9_000, shipping=500)])
    assert seen[0][0] == 9_500, "the gate sees price + shipping, as the median is"
    assert verdict.action is Action.SKIP
    assert "shipping line was read as the price" in " ".join(verdict.reasons)


def test_an_outlier_gate_that_raises_refuses_the_buy() -> None:
    def gate(landed: Optional[int], ref: Optional[MarketRef]) -> bool:
        raise RuntimeError("history table is locked")

    engine, _clock, _rules = build(rule_=rule(max_price=9_000), outlier_check=gate)
    verdict = engine.evaluate(PID, [observation(price=9_000)])
    assert verdict.action is Action.SKIP
    assert "refusing rather than guessing" in " ".join(verdict.reasons)


def test_a_gate_taking_the_reference_first_is_called_the_right_way_round() -> None:
    def gate(market_ref: Optional[MarketRef], landed: Optional[int]) -> bool:
        return isinstance(market_ref, MarketRef) and landed == 9_000

    engine, _clock, _rules = build(rule_=rule(max_price=9_000), outlier_check=gate)
    assert engine.evaluate(PID, [observation(price=9_000)]).action is Action.SKIP


def test_the_built_in_gate_stands_in_when_no_lane_is_present(monkeypatch) -> None:
    monkeypatch.setattr(engine_module, "_prices_module", lambda: None)
    engine = DecisionEngine(None, RuleSet([rule(max_price=9_000)], Budget(total=100_000)),
                            FixedHistory(market()), SimClock())
    assert engine.outlier_origin == "built-in"
    # 39% of the median: under the built-in floor, over the quarter the
    # shipped lane uses, so this pins the fallback and not the sibling.
    verdict = engine.evaluate(PID, [observation(price=3_900)])
    assert verdict.action is Action.SKIP
    assert "suspected mis-parse" in " ".join(verdict.reasons)


@pytest.mark.parametrize(
    "landed,ref,flagged",
    [
        pytest.param(None, market(), True, id="no-price-at-all"),
        pytest.param(0, market(), True, id="free-is-not-a-price"),
        pytest.param(-500, market(), True, id="negative-is-not-a-price"),
        pytest.param(0, None, True, id="free-without-a-reference-either"),
        pytest.param(3_999, market(), True, id="just-under-the-floor"),
        pytest.param(4_000, market(), False, id="exactly-at-the-floor"),
        pytest.param(100, market(samples=2), False, id="no-usable-reference-no-opinion"),
        pytest.param(100, None, False, id="no-reference-no-opinion"),
    ],
)
def test_looks_mis_parsed_boundaries(landed, ref, flagged) -> None:
    got, why = looks_mis_parsed(landed, ref)
    assert got is flagged
    assert bool(why) is flagged


def test_observations_for_other_products_are_ignored_not_acted_on() -> None:
    engine, _clock, _rules = build(rule_=rule(max_price=9_000))
    verdict = engine.evaluate(PID, [
        observation(product_id=OTHER, price=100),
        observation(product_id=PID, price=9_000),
    ])
    assert verdict.action is Action.BUY
    assert verdict.product_id == PID
    assert "ignored 1 observation(s) for other products" in " ".join(verdict.reasons)


def test_a_non_observation_is_refused_rather_than_guessed_at() -> None:
    engine, _clock, _rules = build(rule_=rule(max_price=9_000))
    with pytest.raises(EngineError):
        engine.evaluate(PID, [{"price": 9_000}])  # type: ignore[list-item]


def test_the_deep_link_comes_from_the_listing_then_the_catalog() -> None:
    cat = catalog_with(PID)
    engine, _clock, _rules = build(rule_=rule(max_price=9_000), catalog=cat)
    verdict = engine.evaluate(PID, [observation(price=9_000, url="")])
    assert verdict.url == f"https://examplemart.example.com/p/{PID}"


# --------------------------------------------------------------------------
# the money arithmetic
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reference,pct,threshold",
    [
        pytest.param(10_000, 10.0, 9_000, id="ten-percent-of-a-hundred"),
        pytest.param(10_000, 12.5, 8_750, id="twelve-and-a-half"),
        pytest.param(10_000, 0.1, 9_990, id="a-tenth-of-a-percent"),
        pytest.param(9_999, 10.0, 8_999, id="rounds-against-the-buyer"),
        pytest.param(10_000, 0.0, 10_000, id="zero-percent-is-the-reference"),
        pytest.param(3, 33.0, 2, id="pennies"),
    ],
)
def test_discount_threshold_is_exact(reference: int, pct: float, threshold: int) -> None:
    assert discount_threshold(reference, pct) == threshold
    assert isinstance(discount_threshold(reference, pct), int)


def test_discount_threshold_refuses_a_float_reference() -> None:
    with pytest.raises(EngineError):
        discount_threshold(99.99, 10.0)  # type: ignore[arg-type]
    with pytest.raises(EngineError):
        discount_threshold(0, 10.0)


def test_discount_against_reads_reference_first() -> None:
    assert discount_against(10_000, 9_000) == 10.0
    assert discount_against(10_000, 11_000) == -10.0
    assert discount_against(0, 9_000) is None


@pytest.mark.parametrize(
    "remaining,cap",
    [
        pytest.param(0, 0, id="nothing-left"),
        pytest.param(1, 0, id="one-cent-rounds-down"),
        pytest.param(3, 1, id="odd-cents-round-down"),
        pytest.param(18_000, 9_000, id="half-of-a-hundred-and-eighty"),
    ],
)
def test_budget_cap_floors_in_the_owners_favour(remaining: int, cap: int) -> None:
    assert budget_cap(remaining) == cap
    assert isinstance(budget_cap(remaining), int)
    assert budget_cap(remaining) <= remaining * MAX_BUDGET_FRACTION_PER_VERDICT


def test_affordable_reports_the_numbers_it_refused_on() -> None:
    verdict = affordable(rule(max_price=9_000, quantity=2), 9_000, Budget(total=18_000))
    assert not verdict
    assert verdict.total == 18_000
    assert verdict.cap == 9_000
    assert "$180.00 is over the $90.00" in verdict.reason

    ok = affordable(rule(max_price=9_000), 9_000, Budget(total=18_000))
    assert ok and isinstance(ok, Affordability)
    assert ok.total == 9_000


def test_affordable_refuses_a_free_lunch() -> None:
    assert not affordable(rule(), 0, Budget(total=100_000))
    assert not affordable(rule(), -1, Budget(total=100_000))
    assert not affordable(rule(), 5_000, Budget(total=100_000), quantity=0)


def test_affordable_wants_integer_cents() -> None:
    with pytest.raises(RuleError):
        affordable(rule(), 89.99, Budget(total=100_000))  # type: ignore[arg-type]
    with pytest.raises(RuleError):
        affordable(rule(), True, Budget(total=100_000))  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# the rule set
# --------------------------------------------------------------------------


def test_rule_set_add_get_remove_and_enabled() -> None:
    rules = RuleSet(budget=Budget(total=50_000))
    rules.add(rule(product_id=PID, max_price=9_000))
    rules.add(rule(product_id=OTHER, max_price=17_000, enabled=False))
    assert [r.product_id for r in rules.rules()] == sorted([PID, OTHER])
    assert [r.product_id for r in rules.enabled_rules()] == [PID]
    assert rules.get(PID) is not None
    assert rules.get("nothing-like-it") is None
    assert PID in rules and len(rules) == 2

    rules.set_enabled(OTHER, True)
    assert len(rules.enabled_rules()) == 2
    removed = rules.remove(OTHER)
    assert removed.product_id == OTHER
    with pytest.raises(RuleError):
        rules.remove(OTHER)


def test_a_second_rule_for_one_product_is_refused_with_the_overlap_named() -> None:
    rules = RuleSet([rule(product_id=PID, max_price=9_000,
                          allowed_sources=("examplemart", "cardbarn"))])
    with pytest.raises(RuleConflict) as excinfo:
        rules.add(rule(product_id=PID, max_price=12_000, allowed_sources=("cardbarn",)))
    message = str(excinfo.value)
    assert "cardbarn" in message and "$90.00" in message and "$120.00" in message

    with pytest.raises(RuleConflict) as disjoint:
        rules.add(rule(product_id=PID, max_price=12_000, allowed_sources=("hobbyhub",)))
    assert "one rule per product id" in str(disjoint.value)

    rules.add(rule(product_id=PID, max_price=12_000), replace_existing=True)
    assert rules.get(PID).max_price == 12_000


@pytest.mark.parametrize(
    "bad,error",
    [
        pytest.param(dict(max_price=89.99), RuleError, id="float-ceiling"),
        pytest.param(dict(max_price=True), RuleError, id="bool-ceiling"),
        pytest.param(dict(product_id="  "), RuleError, id="blank-product-id"),
        pytest.param(dict(allowed_sources="examplemart"), RuleError, id="string-not-tuple"),
        pytest.param(dict(allowed_sources=("a", "a")), RuleConflict, id="duplicate-source"),
        pytest.param(dict(allowed_sources=("",)), RuleError, id="empty-source"),
        pytest.param(dict(cooldown_s=-1.0), RuleError, id="negative-cooldown"),
        pytest.param(dict(cooldown_s=float("inf")), RuleError, id="endless-cooldown"),
    ],
)
def test_rule_set_refuses_rules_contracts_cannot_check(bad: Dict[str, Any], error) -> None:
    with pytest.raises(error):
        RuleSet().add(rule(**bad))


def test_budget_reserve_release_commit_and_remaining() -> None:
    rules = RuleSet(budget=Budget(total=10_000))
    assert rules.remaining() == 10_000

    rules.reserve(4_000)
    assert rules.reserved == 4_000
    assert rules.remaining() == 6_000
    assert rules.budget.spent == 0

    rules.release(1_000)
    assert rules.remaining() == 7_000

    rules.commit(3_000)
    assert rules.reserved == 0
    assert rules.budget.spent == 3_000
    assert rules.remaining() == 7_000

    rules.commit(7_000)
    assert rules.remaining() == 0
    assert rules.budget.spent == 10_000


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda r: r.reserve(0), id="reserve-nothing"),
        pytest.param(lambda r: r.reserve(-5), id="reserve-negative"),
        pytest.param(lambda r: r.reserve(10_001), id="reserve-more-than-there-is"),
        pytest.param(lambda r: r.release(1), id="release-what-was-not-reserved"),
        pytest.param(lambda r: r.commit(10_001), id="commit-past-the-total"),
    ],
)
def test_the_ledger_refuses_what_it_cannot_honour(call) -> None:
    with pytest.raises(BudgetError):
        call(RuleSet(budget=Budget(total=10_000)))


def test_reservations_survive_a_new_budget_window() -> None:
    rules = RuleSet(budget=Budget(total=10_000))
    rules.reserve(4_000)
    rules.set_budget(Budget(total=20_000))
    assert rules.reserved == 4_000
    assert rules.remaining() == 16_000
    with pytest.raises(BudgetError):
        rules.set_budget(Budget(total=1_000))


def test_unsatisfiable_names_rules_that_can_never_fire() -> None:
    rules = RuleSet(
        [rule(product_id=PID, max_price=9_000, quantity=4),
         rule(product_id=OTHER, max_price=1_000, enabled=False)],
        Budget(total=20_000),
    )
    report = dict(rules.unsatisfiable())
    assert "is over the $100.00 one verdict may commit" in report[PID]
    assert report[OTHER] == "rule is disabled"


def test_to_obj_is_json_ready_and_keeps_money_in_cents() -> None:
    import json

    rules = RuleSet([rule(product_id=PID, max_price=9_000)], Budget(total=20_000))
    rules.reserve(5_000)
    obj = rules.to_obj()
    assert json.loads(json.dumps(obj)) == obj
    assert obj["budget"]["remaining"] == 15_000
    assert obj["budget"]["per_verdict_cap"] == 7_500
    assert obj["budget"]["display"]["remaining"] == "$150.00"
    assert obj["rules"][0]["max_price"] == 9_000


# --------------------------------------------------------------------------
# a seeded sweep: the invariants must hold for prices nobody chose by hand
# --------------------------------------------------------------------------


#: Any fixed value gives a reproducible sweep; this is the same
#: golden-ratio constant jarvis_poke.sources uses as its default seed.
SWEEP_SEED = 0x9E3779B97F4A7C15


def _sweep(seed: int, rounds: int = 400) -> List[Tuple[str, Verdict, int, int]]:
    """Run ``rounds`` random-ish scenarios; return what to assert about.

    Randomness comes from a ``lucifer_gen.seed`` stream, as everything
    else in this repo does, so a failure here is reproducible from the
    seed printed in the assertion.
    """
    stream = SeedFields.parse(seed).stream("poke.engine.sweep")
    out: List[Tuple[str, Verdict, int, int]] = []
    for index in range(rounds):
        median = stream.randint(1_000, 30_000)
        ceiling = stream.randint(500, 40_000)
        quantity = stream.randint(1, 4)
        discount = stream.choice([0.0, 5.0, 10.0, 12.5, 33.3])
        total = stream.choice([0, 5_000, 50_000, 500_000])
        cooldown = stream.choice([0.0, 3_600.0])
        shipping = stream.choice([0, 199, 1_500])
        limit = stream.choice([None, 0, 1, 2, 9])
        samples = stream.choice([0, 2, 3, 30])
        price = stream.randint(1, 40_000)
        the_rule = Rule(product_id=PID, max_price=ceiling, quantity=quantity,
                        min_discount_pct=discount, cooldown_s=cooldown)
        rules = RuleSet([the_rule], Budget(total=total))
        engine, _clock, _rules = build(
            rules=rules,
            ref=market(median=median, samples=samples),
        )
        looks = [observation(price=price, shipping=shipping, per_customer_limit=limit)]
        before = copy.deepcopy(looks)
        headroom = rules.remaining()
        verdict = engine.evaluate(PID, looks)
        assert looks == before, f"round {index}: evaluate mutated its input"
        out.append((f"seed=0x{seed:016X} round={index}", verdict, headroom, median))
    return out


def test_the_sweep_holds_every_invariant() -> None:
    for where, verdict, headroom, median in _sweep(SWEEP_SEED):
        assert verdict.reasons, where
        assert verdict.product_id == PID, where
        if verdict.action is not Action.BUY:
            assert verdict.quantity == 0, where
            continue
        cost = verdict.landed
        assert cost is not None and cost > 0, where
        assert 1 <= verdict.quantity, where
        assert verdict.quantity * cost <= budget_cap(headroom), where
        assert verdict.quantity * cost <= headroom, where
        assert cost * 4 >= median or verdict.market is None, where
        if verdict.market is not None and verdict.discount_pct is not None:
            assert verdict.discount_pct == discount_against(verdict.market, cost), where


def test_the_sweep_is_reproducible() -> None:
    first = [(w, v) for w, v, _h, _m in _sweep(SWEEP_SEED, rounds=60)]
    second = [(w, v) for w, v, _h, _m in _sweep(SWEEP_SEED, rounds=60)]
    assert first == second
    other = [v for _w, v, _h, _m in _sweep(SWEEP_SEED ^ 0xFFFF, rounds=60)]
    assert [v for _w, v in first] != other, "a different seed should explore elsewhere"


# --------------------------------------------------------------------------
# the standing constraints, checked against the source itself
# --------------------------------------------------------------------------


def test_this_lane_opens_no_sockets_and_reads_no_clock() -> None:
    """contracts.py: no network here, an injected clock, seeded randomness.

    Scoped to the two files this suite owns; a package-wide sweep belongs
    in the validation gate, where it will not trip over a sibling still
    being written.
    """
    banned_modules = ("urllib", "socket", "http", "requests", "httpx", "ssl",
                      "ftplib", "telnetlib", "asyncio", "subprocess", "random")
    banned_calls = {("time", "time"), ("time", "monotonic"), ("random", "random"),
                    ("random", "randint"), ("random", "choice"), ("random", "uniform")}

    for name in ("rules.py", "engine.py"):
        tree = ast.parse((ROOT / "jarvis_poke" / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in banned_modules, f"{name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                assert root not in banned_modules, f"{name} imports from {node.module}"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                owner = node.func.value
                if isinstance(owner, ast.Name):
                    assert (owner.id, node.func.attr) not in banned_calls, (
                        f"{name} calls {owner.id}.{node.func.attr}(): the clock is "
                        f"injected and randomness comes from a lucifer_gen seed stream"
                    )


def test_this_lane_automates_no_checkout() -> None:
    """contracts.py, "What this is not": the engine stops at a deep link.

    A word-level check is crude, but it is the kind of crude that catches
    the commit where somebody adds ``def add_to_cart``.
    """
    forbidden = ("cart", "checkout", "captcha", "proxy", "payment", "card_number",
                 "credential", "cookie")
    for name in ("rules.py", "engine.py"):
        tree = ast.parse((ROOT / "jarvis_poke" / name).read_text(encoding="utf-8"))
        names: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.append(node.name)
            elif isinstance(node, ast.Name):
                names.append(node.id)
            elif isinstance(node, ast.Attribute):
                names.append(node.attr)
            elif isinstance(node, ast.arg):
                names.append(node.arg)
        for identifier in names:
            lowered = identifier.lower()
            for word in forbidden:
                assert word not in lowered, (
                    f"{name} defines or calls {identifier!r}: this lane monitors and "
                    f"decides, and stops at a deep link a person taps"
                )


# --------------------------------------------------------------------------
# the ledger across engine objects, threads and runs
#
# Everything below is a regression: each one reproduced before the fix and
# each one is about money leaving the account, so they are written as
# properties of the ledger rather than as examples.
# --------------------------------------------------------------------------


def test_a_second_engine_over_one_ledger_replaces_the_reservation() -> None:
    """One listing, three runs, one hold -- not three.

    ``cli.cmd_decide`` builds an engine per run against the store's
    RuleSet.  When the per-product reservations lived on the engine, each
    run booked the money again and no live object was left to release it:
    one $200 listing could hold $600 of an $800 budget for ever.
    """
    shared = RuleSet([rule(max_price=20_000)], Budget(total=80_000))
    for _ in range(3):
        engine, _clock, _rules = build(rules=shared)
        verdict = engine.evaluate(PID, [observation(price=20_000)])
        assert verdict.action is Action.BUY
        # a fresh engine each round, exactly as a fresh process would be
        engine.watch_state(PID).last_alert_at = 0.0
    assert shared.reserved == 20_000
    assert shared.reservations() == {PID: 20_000}
    assert shared.remaining() == 60_000


def test_reservations_are_visible_to_every_engine_on_the_ledger() -> None:
    first_rules = RuleSet([rule(max_price=20_000)], Budget(total=80_000))
    one, _c1, _r = build(rules=first_rules)
    one.evaluate(PID, [observation(price=20_000)])
    two, _c2, _r2 = build(rules=first_rules)
    assert two.reservations() == {PID: 20_000}
    assert two.release_reservation(PID) == 20_000
    assert first_rules.reserved == 0


def test_a_second_evaluation_cannot_slip_between_the_check_and_the_reserve() -> None:
    """The TOCTOU, forced open at a public seam.

    Step 9 reads ``remaining()``; the reservation is taken at the end of
    ``_finish``.  ``_finish`` asks the catalog for a deep link when the
    observation carries none, so a catalog that takes its time holds the
    window open -- which is the whole point: with the ledger's lock held
    across ``evaluate`` the second caller waits, and without it the
    second caller is told the first one's money is still free.
    """
    import threading
    import time

    inside = threading.Event()

    class SlowCatalog:
        def find(self, product_id: str) -> object:
            return object()

        def sku(self, source: str, product_id: str) -> None:
            inside.set()
            time.sleep(0.4)
            return None

    rules = RuleSet(
        [Rule(product_id=pid, max_price=20_000, cooldown_s=0.0) for pid in ("a", "b")],
        Budget(total=60_000),
    )
    clock = SimClock()
    # No URL on the observation, so ``_finish`` has to ask the catalog
    # for the deep link -- the seam this test leans on.  ``observation``
    # fills one in, so these are built by hand.
    listing = {
        pid: [
            Observation(product_id=pid, source="examplemart", sku="EX-1", at=clock.now,
                        stock=Stock.IN_STOCK, price=20_000, shipping=0, url="")
        ]
        for pid in ("a", "b")
    }
    out: Dict[str, Verdict] = {}

    def first() -> None:
        engine = DecisionEngine(SlowCatalog(), rules, FixedHistory(None), clock,
                                outlier_check=quarter_gate)
        out["a"] = engine.evaluate("a", listing["a"])

    thread = threading.Thread(target=first)
    thread.start()
    assert inside.wait(5.0), "the slow catalog was never reached"
    second = DecisionEngine(SlowCatalog(), rules, FixedHistory(None), clock,
                            outlier_check=quarter_gate)
    out["b"] = second.evaluate("b", listing["b"])
    thread.join()

    assert out["a"].action is Action.BUY
    # $600 left, half of it is the most one verdict may commit, so the
    # second $200 has to be measured against $400 -- not against the $600
    # the first verdict had not yet promised away.
    assert "($400.00 left)" in " ".join(out["b"].reasons), out["b"].reasons
    assert rules.reserved == 40_000


def test_concurrent_evaluations_cannot_promise_the_same_money() -> None:
    """Two products, a budget only one of them fits, eight threads.

    ``evaluate`` read ``remaining()`` at step 9 and reserved at the end
    with nothing in between, so two threads were each told their spend
    "fits the cap" out of the whole budget and the ledger ended up
    reserving all of it.
    """
    import threading

    pids = [f"p{i}" for i in range(8)]
    rules = RuleSet(
        [Rule(product_id=pid, max_price=20_000, cooldown_s=0.0) for pid in pids],
        Budget(total=60_000),
    )
    clock = SimClock()
    start = threading.Barrier(len(pids))
    verdicts: Dict[str, Verdict] = {}
    errors: List[BaseException] = []

    def run(pid: str) -> None:
        engine = DecisionEngine(None, rules, FixedHistory(None), clock,
                                outlier_check=quarter_gate)
        start.wait()
        try:
            verdicts[pid] = engine.evaluate(pid, [observation(price=20_000,
                                                              product_id=pid)])
        except BaseException as exc:  # noqa: BLE001 - the point of the test
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(pid,)) for pid in pids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, f"evaluate raised into the caller: {errors[0]!r}"

    # The answer has to be the one a sequential pass gives: each verdict
    # measured against what the ones before it had already promised.
    sequential = RuleSet(
        [Rule(product_id=pid, max_price=20_000, cooldown_s=0.0) for pid in pids],
        Budget(total=60_000),
    )
    lone = DecisionEngine(None, sequential, FixedHistory(None), clock,
                          outlier_check=quarter_gate)
    for pid in pids:
        lone.evaluate(pid, [observation(price=20_000, product_id=pid)])

    buys = [v for v in verdicts.values() if v.action is Action.BUY]
    assert len(buys) == len(sequential.reservations()), (
        f"{len(buys)} concurrent BUYs against "
        f"{len(sequential.reservations())} sequential ones"
    )
    assert rules.reserved == sequential.reserved
    assert rules.reserved <= rules.budget.total


def test_a_refused_reservation_does_not_start_a_cooldown() -> None:
    """A BUY the ledger cannot fund is a SKIP, not an alert nobody sent.

    ``record_verdict`` used to stamp ``last_alert_at`` and bump
    ``alerts_sent`` before reserving, so a BudgetError left the product
    inside a cooldown for an alert that never existed -- and escaped
    ``evaluate`` into the caller's loop.
    """

    class MeanLedger(RuleSet):
        def reserve_for(self, key: str, amount: int) -> int:
            raise BudgetError("the ledger is having none of it")

    rules = MeanLedger([rule(max_price=9_000, cooldown_s=3_600.0)],
                       Budget(total=1_000_000))
    engine, _clock, _r = build(rules=rules)
    verdict = engine.evaluate(PID, [observation(price=9_000)])
    assert verdict.action is Action.SKIP
    assert any("the ledger refused to hold the money" in r for r in verdict.reasons)
    state = engine.watch_state(PID)
    assert state.last_alert_at == 0.0
    assert state.alerts_sent == 0


def test_the_ledger_is_charged_the_landed_price_even_when_the_rule_is_not() -> None:
    """``include_shipping=False`` is about the ceiling, not about the card.

    The reservation used to follow the rule and leave postage out, so a
    $110 shelf price with $50 postage held $110 against a budget the
    owner would be charged $160 from.
    """
    rules = RuleSet(
        [rule(max_price=13_000, include_shipping=False)],
        Budget(total=100_000),
    )
    engine, _clock, _r = build(rules=rules, ref=None)
    verdict = engine.evaluate(PID, [observation(price=11_000, shipping=5_000)])
    assert verdict.action is Action.BUY
    assert rules.reserved == 16_000, "the ledger must hold what the card is charged"
    assert any("the card is charged $160.00" in r for r in verdict.reasons)


def test_the_discount_is_measured_on_the_landed_price() -> None:
    """prices.market_reference samples landed prices, so the comparison
    has to be landed too: a shelf price against a landed median makes
    heavy postage read as a bargain."""
    rules = RuleSet(
        [rule(max_price=13_000, include_shipping=False, min_discount_pct=10.0)],
        Budget(total=100_000),
    )
    engine, _clock, _r = build(rules=rules, ref=market(median=13_000))
    verdict = engine.evaluate(PID, [observation(price=11_000, shipping=5_000)])
    assert verdict.action is Action.WATCH
    assert verdict.discount_pct is not None and verdict.discount_pct < 0


def test_commit_for_moves_spent_and_clears_the_hold() -> None:
    rules = RuleSet([rule(max_price=20_000)], Budget(total=80_000))
    engine, _clock, _r = build(rules=rules)
    engine.evaluate(PID, [observation(price=20_000)])
    assert rules.reserved == 20_000 and rules.budget.spent == 0
    assert engine.commit_purchase(PID) == 20_000
    assert rules.budget.spent == 20_000
    assert rules.reserved == 0
    assert rules.remaining() == 60_000


def test_load_reservations_restores_a_ledger_and_refuses_an_impossible_one() -> None:
    rules = RuleSet([rule(max_price=20_000)], Budget(total=50_000))
    assert rules.load_reservations({PID: 20_000}) == 20_000
    assert rules.remaining() == 30_000
    with pytest.raises(BudgetError):
        rules.load_reservations({PID: 20_000, "other": 40_000})
    assert rules.reservations() == {PID: 20_000}, "a refused restore changes nothing"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
