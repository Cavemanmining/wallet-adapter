"""The Pokemon gate: run a month of watching, and hold it to contracts.py.

Design: :mod:`jarvis_poke.contracts`.  Its module docstring makes four
promises, and every one of them is a thing that can quietly stop being
true as the package grows, so this module drives the real pieces --
:class:`jarvis_poke.catalog.Catalog`,
:class:`jarvis_poke.sources.PollScheduler`,
:class:`jarvis_poke.prices.PriceHistory`,
:class:`jarvis_poke.engine.DecisionEngine`,
:class:`jarvis_poke.rules.RuleSet`,
:class:`jarvis_poke.alerts_bridge.AlertBridge` and
:class:`jarvis_poke.store.PokeStore` -- through a scripted month and then
asserts the promises against what actually happened:

1. **"It does not check out."**  The gate's terminal output is a
   :class:`~jarvis_poke.contracts.Verdict` and a deep link.  Nothing here
   carts, pays or solves anything; the only "purchase" is the owner
   deciding, in the scripted way below, to commit money the ledger had
   already set aside.
2. **"Politeness is a design constraint."**  The *fetcher's own call log*
   is the evidence, not the scheduler's opinion of itself: no host is
   ever called twice inside its ``min_interval_s``, the robots-disallowed
   source is never called at all, a repeat call for a page we hold a
   validator for carries ``If-None-Match``, and the source that breaks
   backs off and pauses.
3. **"The package makes no network calls."**  The fetcher and the parser
   are injected (:class:`_ScriptedHost`, :func:`parse_listing`); the
   bodies are a placeholder JSON shape invented here, not any retailer's
   page structure, and every retailer in the run is one of the shipped
   placeholders (examplemart, cardbarn, hobbyhub, bigboxco) on
   ``example.com``.
4. **"Money is integer cents."**  Every amount in the scenario and in
   every check is an ``int``; the only float money would have to pass
   through is ``MAX_BUDGET_FRACTION_PER_VERDICT``, and the gate does
   *not* read it -- it carries its own exact
   :data:`GATE_BUDGET_FRACTION` and compares the two
   (:meth:`_Gate._check_contract`).  Applying the package's constant
   through the package's own helper, which is what this used to do, made
   ``verdict_over_cap`` a tautology: both sides of the comparison moved
   together, so doubling the cap was invisible to the check whose whole
   job is that cap.

The scripted month
------------------
One seed fixes everything (``lucifer_gen.seed.SeedFields``; there is no
``random`` and no ``time.time`` anywhere in this module -- the clock is a
cell the gate advances by hand).  The script contains, by construction:

* a price walk per product, shared across sources so the market reference
  means something, with each source quoting its own factor and shipping;
* **sellouts and restocks** -- a listing that goes out of stock and comes
  back ``LIMITED``, which is what a restock looks like from outside;
* a **genuine price crash** on one product, deep enough to clear a rule
  (55% of typical) but nowhere near the outlier floor, so it *must*
  produce a BUY;
* a **mis-parse that looks like a 95% discount**: one listing quotes a
  per-pack price on a box page for half a day.  The parser reads it
  faithfully, which is the point -- the price is real, the *meaning* is
  wrong -- and the engine must refuse it;
* a **listing that states a per-customer limit**, held in stock for the
  whole run, carrying a rule that asks for one more copy than it will
  sell -- so the engine's clamp is executed rather than merely asserted
  about;
* a **source outage**: one host answers 503 for a stretch, so the
  widening backoff and the pause are exercised for real;
* a **robots-disallowed source**, which must never be fetched once;
* a **304-heavy source**, whose page changes every few days and which
  therefore answers ``If-None-Match`` with a 304 nearly every time.

What the gate then asserts
--------------------------
Each is a :class:`Problem` with a stable ``kind``:

``poll_too_fast``         two calls to one host inside its interval
``robots_ignored``        a call to a source robots.txt disallows
``conditional_missing``   a repeat call with no validator on a page that
                          gave us one
``buy_over_ceiling``      a BUY above ``Rule.max_price``
``buy_under_discount``    a BUY short of ``min_discount_pct`` while the
                          market reference was usable
``buy_bad_quantity``      a BUY over ``Rule.quantity`` or over the
                          listing's ``per_customer_limit``
``buy_on_misparse``       a BUY on the mis-parsed bargain
``budget_exceeded``       committed more than the budget window allowed
``verdict_over_cap``      one verdict committing more than
                          ``MAX_BUDGET_FRACTION_PER_VERDICT`` of what was
                          left
``cooldown_broken``       two BUY alerts for one product inside its
                          cooldown
``reasons_empty``         a verdict with nothing in ``reasons``
``alert_count``           alerts published and BUYs issued disagree
``alert_not_buy``         an alert for something that was not a BUY
``round_trip``            sqlite gave back something different
``boundary_strict``       a boundary probe was refused something the
                          contract says must be accepted (the engine
                          being too *tight*, the mirror of the kinds
                          above)
``scenario_thin``         the run did not actually contain a hazard the
                          gate claims to test, so a pass would be vacuous
``crashed``               the run raised; the exception type is recorded

Shown to fail before it is trusted
----------------------------------
``inject_defect`` perturbs the run in one of the seven ways in
:data:`DEFECTS`, each of which a correct monitor never does, and the
tests demand that every one is reported.  ``python3 -m
jarvis_poke.validate --show-defects`` exits 1 if any is missed.

    poll_too_fast    a caller fetches a host again a second later,
                     routing around the scheduler
    ignore_robots    the robots-disallowed source is fetched once
    ignore_ceiling   a listing over the owner's ceiling is alerted as a BUY
    trust_outlier    the mis-parsed 95%-off bargain is alerted as a BUY
    ignore_budget    the ledger answers yes to every spend (the
                     per-verdict cap and the budget both stop applying)
    ignore_cooldown  the engine forgets when it last alerted
    alert_on_watch   a WATCH is pushed to the phone

Entry point
-----------
:func:`run_gate` returns a :class:`GateReport` -- ``counts`` for an
operator, ``problems`` in detection order, ``ok``.  ``python3 -m
jarvis_poke.validate`` prints one; ``jarvis_poke.cli gate`` runs the same
call.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from fractions import Fraction
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from lucifer_gen.seed import SeedFields, format_seed, parse_seed

from jarvis_poke.alerts_bridge import AlertBridge, dedupe_key_for
from jarvis_poke.catalog import Catalog
from jarvis_poke.contracts import (
    MARKET_WINDOW_S,
    MAX_BUDGET_FRACTION_PER_VERDICT,
    Action,
    Budget,
    Cents,
    FetchPolicy,
    FetchResult,
    Observation,
    Product,
    MarketRef,
    ProductKind,
    Rule,
    SourceSku,
    Stock,
    Verdict,
    fmt_cents,
)
from jarvis_poke.engine import DecisionEngine, discount_threshold
from jarvis_poke import prices as _prices
from jarvis_poke.prices import PriceHistory
from jarvis_poke.rules import Affordability, RuleSet, budget_cap
from jarvis_poke.sources import PollScheduler, load_policies
from jarvis_poke.store import PokeStore, first_difference

__all__ = [
    "DAY_S",
    "DEFAULT_SEED",
    "DEFECTS",
    "FRESH_S",
    "GATE_EPOCH",
    "MIN_DAYS",
    "MIN_PRODUCTS",
    "SLOTS_PER_DAY",
    "TICK_S",
    "GateError",
    "GateReport",
    "Problem",
    "main",
    "parse_listing",
    "run_gate",
]

# --------------------------------------------------------------------------
# Constants of the scripted month
# --------------------------------------------------------------------------

#: Where the gate's clock starts.  A fixed number, because a gate that
#: starts "now" is a gate whose failures cannot be reproduced.
GATE_EPOCH = 1_700_000_000.0

DAY_S = 86400.0

#: The driver's tick.  Equal to the fastest shipped ``min_interval_s``, so
#: the scheduler is asked as often as the politest source could possibly
#: be polled and never more often than that.
TICK_S = 300.0

#: The script's resolution: six four-hour slots a day.  A listing's price
#: and stock are constant inside a slot, which is what makes an ETag
#: meaningful -- the page really has not changed.
SLOTS_PER_DAY = 6
SLOT_S = DAY_S / SLOTS_PER_DAY

#: A listing not seen for this long is not offered to the engine.  The app
#: does the same: "it was in stock yesterday" is not a price you can buy
#: at, and during a deep backoff a host's last reading goes stale.
FRESH_S = 24 * 3600.0

#: How long a BUY's reservation is held before the gate decides the owner
#: ignored the deep link and hands the money back.
RESERVATION_TTL_S = 2 * 3600.0

#: Smallest run that still contains every hazard the gate claims to test.
#: Five, not four: four products are already spoken for by the four fixed
#: rules (dearest, crashed, mis-parsed, disabled), and the fifth carries
#: the rule whose quantity a listing's per-customer limit has to cut.
#: Without it the clamp in ``DecisionEngine`` step 8 is never executed and
#: the gate's assertion about it is made over dead code.
MIN_PRODUCTS = 5
MIN_DAYS = 3

#: Seed used when a caller does not give one (the golden-ratio constant,
#: as in :mod:`jarvis_poke.sources`).
DEFAULT_SEED = 0x9E3779B97F4A7C15

#: The four shipped placeholders, and what each one does in the script.
OUTAGE_SOURCE = "hobbyhub"       # answers 503 for a stretch
MISPARSE_SOURCE = "hobbyhub"     # quotes a per-pack price on a box page
STICKY_SOURCE = "cardbarn"       # changes its page rarely: mostly 304s
DISALLOWED_SOURCE = "bigboxco"   # robots.txt says no; never fetched

#: Each source's price as a percentage of the day's market level, the
#: shipping it charges, and the per-customer limit its page states.
SOURCE_FACTOR = {"examplemart": 100, "cardbarn": 97, "hobbyhub": 104, "bigboxco": 99}
SOURCE_SHIPPING = {"examplemart": 0, "cardbarn": 499, "hobbyhub": 0, "bigboxco": 799}
SOURCE_LIMIT: Dict[str, Optional[int]] = {
    "examplemart": None, "cardbarn": 2, "hobbyhub": 1, "bigboxco": None,
}

#: The 304-heavy source re-prices this rarely, in days.
STICKY_CHANGE_DAYS = 3

#: The crash: the market level for one product falls to this percentage of
#: its typical price.  Well above :data:`jarvis_poke.prices.
#: OUTLIER_MIN_PCT_OF_MEDIAN`, so it is a bargain and not a mis-parse.
CRASH_PCT = 55

#: The mis-parse: a box page quoting the price of one pack.
MISPARSE_DIVISOR = 20            # 5% of the real price: a 95% "discount"
MISPARSE_SLOTS = 3               # half a day of it

#: The owner acts on about one alert in twelve.  The rest lapse, and the
#: reservation goes back to the ledger after :data:`RESERVATION_TTL_S`.
OWNER_ACTS_P = 1.0 / 12.0

#: The gate's **own** copy of the per-verdict cap, written out as an exact
#: fraction rather than read from :data:`contracts.MAX_BUDGET_FRACTION_PER_VERDICT`.
#: ``verdict_over_cap`` used to re-derive the cap with ``rules.budget_cap``,
#: which defaults to that very constant -- so both sides of the comparison
#: moved together and changing the constant to 1.0 was invisible to the
#: check whose entire job is that constant.  This is the number
#: contracts.py documents; if the package's constant stops agreeing with
#: it, :meth:`_Gate._check_contract` says so rather than quietly adopting
#: the new value.
GATE_BUDGET_FRACTION = Fraction(1, 2)

#: The gate's own copy of :data:`jarvis_poke.prices.OUTLIER_MIN_PCT_OF_MEDIAN`,
#: for the same reason.  The scripted mis-parse sits at 5% of the median,
#: four times below this boundary, so the boundary itself is only tested by
#: :meth:`_Gate._check_boundaries`.
GATE_OUTLIER_MIN_PCT = 25

#: What the gate can break on purpose.  Every one is reported, or the gate
#: is not worth running; ``--show-defects`` proves it.
DEFECTS: Tuple[str, ...] = (
    "poll_too_fast",
    "ignore_robots",
    "ignore_ceiling",
    "trust_outlier",
    "ignore_budget",
    "ignore_cooldown",
    "alert_on_watch",
)

#: Float comparisons on times and percentages are exact in practice (the
#: clock is a sum of constants), but a gate that fails on the last bit of
#: a float is a gate nobody trusts.
EPS = 1e-6


def _gate_cap(remaining: Cents) -> Cents:
    """The gate's own per-verdict cap: floor of :data:`GATE_BUDGET_FRACTION`.

    Computed here, from the gate's own fraction, so that
    ``verdict_over_cap`` is comparing the engine's behaviour against the
    documented contract rather than against the same constant the engine
    just used.
    """
    if remaining <= 0:
        return 0
    return int(Fraction(int(remaining)) * GATE_BUDGET_FRACTION)


class GateError(ValueError):
    """The gate was asked for a run it cannot build: too few products or
    days to contain the hazards it claims to test, or a defect name it
    does not know."""


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Problem:
    """One failed assertion.

    ``kind`` is stable and is what the tests match on; ``detail`` is for a
    person; the rest says where.  Money in a detail is formatted with
    :func:`~jarvis_poke.contracts.fmt_cents`, never as a float.
    """

    kind: str
    detail: str
    product_id: str = ""
    source: str = ""
    at: float = 0.0

    def __str__(self) -> str:
        where = []
        if self.product_id:
            where.append(f"product={self.product_id}")
        if self.source:
            where.append(f"source={self.source}")
        if self.at:
            where.append(f"at=+{self.at - GATE_EPOCH:.0f}s")
        tail = f" [{' '.join(where)}]" if where else ""
        return f"{self.kind}: {self.detail}{tail}"


@dataclass
class GateReport:
    """What :func:`run_gate` found.

    ``counts`` is JSON-friendly and every value is an ``int``; it is also
    the evidence that the scenario contained what it claims to (polls,
    304s, pauses, restocks, mis-parses, BUYs).  ``problems`` is in
    detection order, so ``first_failure`` is the most upstream one.
    """

    n_products: int
    n_days: int
    seed: int
    inject_defect: Optional[str]
    counts: Dict[str, int] = field(default_factory=dict)
    problems: List[Problem] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def first_failure(self) -> Optional[Problem]:
        return self.problems[0] if self.problems else None

    @property
    def repro(self) -> str:
        return (
            f"run_gate(n_products={self.n_products}, n_days={self.n_days}, "
            f"seed={format_seed(self.seed)}, inject_defect={self.inject_defect!r})"
        )

    def kinds(self) -> List[str]:
        """Distinct problem kinds, in first-seen order."""
        seen: List[str] = []
        for problem in self.problems:
            if problem.kind not in seen:
                seen.append(problem.kind)
        return seen

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_products": self.n_products,
            "n_days": self.n_days,
            "seed": format_seed(self.seed),
            "inject_defect": self.inject_defect,
            "ok": self.ok,
            "counts": dict(self.counts),
            "problems": [dataclasses.asdict(p) for p in self.problems],
        }

    def summary(self, max_problems: int = 6) -> str:
        lines = [
            f"gate: products={self.n_products} days={self.n_days} "
            f"seed={format_seed(self.seed)} defect={self.inject_defect!r}",
            "  counts: " + " ".join(f"{k}={v}" for k, v in self.counts.items()),
        ]
        if self.ok:
            lines.append("  result: OK")
        else:
            lines.append(
                f"  result: FAIL ({len(self.problems)} problem(s); "
                f"kinds: {', '.join(self.kinds())})"
            )
            for problem in self.problems[:max_problems]:
                lines.append(f"    - {problem}")
            if len(self.problems) > max_problems:
                lines.append(f"    ... {len(self.problems) - max_problems} more")
            lines.append(f"  repro: {self.repro}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The scripted month
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Slot:
    """What one listing's page says during one four-hour slot."""

    price: Optional[Cents]           # None when the page shows no price
    shipping: Cents
    stock: Stock
    limit: Optional[int]
    note: str
    misparse: bool
    version: int                     # the slot this content first appeared in

    @property
    def content(self) -> Tuple[Any, ...]:
        return (self.price, self.shipping, self.stock.value, self.limit, self.note)


@dataclass(frozen=True)
class _Call:
    """One call the injected fetcher received: the ground truth of what
    this run did to somebody else's host."""

    source: str
    product_id: str
    at: float
    conditional: bool
    outcome: str


@dataclass
class _Script:
    """The whole month, decided before the first tick."""

    slots: Dict[Tuple[str, str], List[_Slot]]
    typical: Dict[str, Cents]
    crash_product: str
    crash_slots: Tuple[int, ...]
    misparse_key: Tuple[str, str]
    misparse_slots: Tuple[int, ...]
    #: ``(source, product_id)`` of the listing whose stated per-customer
    #: limit has to cut a rule's quantity.  It is held in stock for the
    #: whole run, for the same reason the mis-parsed listing is: a code
    #: path the run never reaches is a code path the checks cannot test.
    limit_key: Tuple[str, str]
    outage_slots: Tuple[int, ...]
    n_slots: int
    sellouts: int
    restocks: int

    def at(self, source: str, product_id: str, when: float) -> _Slot:
        rows = self.slots[(source, product_id)]
        index = int((when - GATE_EPOCH) // SLOT_S)
        return rows[min(max(index, 0), len(rows) - 1)]

    def slot_index(self, when: float) -> int:
        return min(max(int((when - GATE_EPOCH) // SLOT_S), 0), self.n_slots - 1)


def _kind_for(index: int) -> ProductKind:
    kinds = list(ProductKind)
    return kinds[index % len(kinds)]


def _gate_catalog(n_products: int, fields: SeedFields) -> Catalog:
    """A catalog subset: the shipped products first, then placeholders.

    The shipped catalog is the real fixture and is used as far as it goes
    (:mod:`jarvis_poke.catalog` loads it and validates it on the way in).
    A run larger than it asks for is topped up with obvious placeholders
    -- ``gate-placeholder-07`` on ``example.com`` -- because the point of
    a forty-product run is the scheduler's rotation and the ledger's
    arithmetic, neither of which cares what is in the box.

    Every product is listed by examplemart and cardbarn, every other one
    by hobbyhub and every third by bigboxco, so all four shipped sources
    -- including the one robots.txt disallows -- are in every run.  A
    shipped listing's own SKU and URL are kept where the shipped sources
    file has one.
    """
    shipped = Catalog.load()
    products: List[Product] = list(shipped.products())[:n_products]
    for index in range(len(products), n_products):
        number = index + 1
        products.append(
            Product(
                id=f"gate-placeholder-{number:03d}",
                name=f"Placeholder Sealed Product {number:03d} (gate fixture)",
                set_code=f"GT{number:02d}",
                kind=_kind_for(index),
                msrp=2999 + (index % 7) * 1500,
                released=None,
                upc=None,
            )
        )

    skus: List[SourceSku] = []
    for index, product in enumerate(products):
        wanted = ["examplemart", "cardbarn"]
        if index % 2 == 0:
            wanted.append("hobbyhub")
        if index % 3 == 0:
            wanted.append(DISALLOWED_SOURCE)
        for source in wanted:
            existing = shipped.sku(source, product.id)
            if existing is not None:
                skus.append(existing)
                continue
            skus.append(
                SourceSku(
                    source=source,
                    product_id=product.id,
                    sku=f"{source[:2].upper()}-{product.id.upper()}",
                    url=f"https://{source}.example.com/p/{product.id}",
                )
            )
    labels = {source: shipped.source_label(source) for source in SOURCE_FACTOR}
    return Catalog(products, skus, source_labels=labels, origin="gate")


def _typical_cents(product: Product, stream) -> Cents:
    """What this product actually changes hands for, as opposed to MSRP.

    contracts.py: "MSRP is fiction for anything in demand", so the
    scripted market sits anywhere between a clearance discount and a
    healthy premium on it.
    """
    base = product.msrp if product.msrp else 4999
    return max(500, base * stream.randint(85, 165) // 100)


def _build_script(catalog: Catalog, n_days: int, fields: SeedFields) -> _Script:
    """Decide the whole month up front: prices, stock, and the hazards."""
    n_slots = n_days * SLOTS_PER_DAY
    products = catalog.products()
    pids = [p.id for p in products]

    typical = {
        p.id: _typical_cents(p, fields.stream(f"poke.gate.typical:{p.id}"))
        for p in products
    }

    # The crash never lands on the dearest product: that one carries the
    # deliberately-unaffordable rule (:meth:`_Gate._build_rules`), so a
    # crash on it would be refused by the budget and the gate would lose
    # the one BUY it insists on seeing.
    dearest = max(sorted(typical), key=lambda pid: typical[pid])
    crash_product = fields.stream("poke.gate.crash").choice(
        [pid for pid in pids if pid != dearest] or pids
    )
    crash_day = n_days // 2
    crash_days = [crash_day] + ([crash_day + 1] if crash_day + 1 < n_days else [])
    crash_slots = tuple(
        slot for slot in range(n_slots) if slot // SLOTS_PER_DAY in crash_days
    )

    # The market level per product per day: one walk, shared by every
    # source, so the median of the observations is a real market and not
    # four unrelated numbers.
    daily: Dict[str, List[Cents]] = {}
    for pid in pids:
        stream = fields.stream(f"poke.gate.walk:{pid}")
        level = typical[pid]
        step = max(25, typical[pid] // 40)
        row: List[Cents] = []
        for _day in range(n_days):
            level = min(
                max(level + stream.randint(-step, step), typical[pid] * 75 // 100),
                typical[pid] * 130 // 100,
            )
            row.append(level)
        for day in crash_days if pid == crash_product else ():
            row[day] = typical[pid] * (CRASH_PCT + (day - crash_day) * 3) // 100
        daily[pid] = row

    # The mis-parse goes on a product the mis-parsing source actually
    # lists, and never on the crashed one: the gate has to be able to tell
    # "refused the mis-parse" from "missed the crash".
    misparse_candidates = [
        sku.product_id
        for sku in catalog.skus_from(MISPARSE_SOURCE)
        if sku.product_id not in (crash_product, dearest)
    ] or [sku.product_id for sku in catalog.skus_from(MISPARSE_SOURCE)]
    if not misparse_candidates:  # pragma: no cover - every run lists this source
        raise GateError(f"no {MISPARSE_SOURCE} listing to hang the mis-parse on")
    misparse_pid = fields.stream("poke.gate.misparse").choice(sorted(misparse_candidates))
    misparse_key = (MISPARSE_SOURCE, misparse_pid)
    misparse_start = (2 * n_days // 3) * SLOTS_PER_DAY
    misparse_slots = tuple(
        slot for slot in range(misparse_start, misparse_start + MISPARSE_SLOTS)
        if slot < n_slots
    ) or (n_slots - 1,)

    # The per-customer limit.  The listing that states the *smallest*
    # limit -- so the clamped quantity is the one the tight gate budget
    # can actually afford -- on the cheapest product not already carrying
    # a fixed rule.  Scripted rather than left to chance because it was
    # left to chance before: across 48 runs at 24 seeds, 1102 BUYs, the
    # limit bound exactly zero times, and the clamp in
    # ``DecisionEngine`` step 8 was never executed at any size or seed.
    limit_candidates = [
        (limit, typical[sku.product_id], sku.product_id, source)
        for source in sorted(SOURCE_LIMIT)
        for limit in (SOURCE_LIMIT[source],)
        if limit is not None and limit >= 1 and source != DISALLOWED_SOURCE
        for sku in catalog.skus_from(source)
        if sku.product_id not in (crash_product, dearest, misparse_pid)
    ]
    if limit_candidates:
        _, _, limit_pid, limit_source = min(limit_candidates)
        limit_key = (limit_source, limit_pid)
    else:  # pragma: no cover - only a catalog without the shipped sources
        limit_key = ("", "")

    # The outage: a stretch of 503s long enough to widen the backoff past
    # the source's ``max_errors_before_pause`` and pause it.
    outage_start_day = max(1, n_days // 3)
    outage_days = range(outage_start_day, min(n_days, outage_start_day + max(1, n_days // 10)))
    outage_slots = tuple(
        slot for slot in range(n_slots) if slot // SLOTS_PER_DAY in outage_days
    )

    slots: Dict[Tuple[str, str], List[_Slot]] = {}
    sellouts = 0
    restocks = 0
    for index, sku in enumerate(catalog.skus()):
        pid = sku.product_id
        source = sku.source
        stream = fields.stream(f"poke.gate.listing:{source}:{pid}")
        cycle = stream.randint(4, 9)
        offset = stream.randint(0, cycle - 1)
        out_len = stream.randint(1, 3)
        wobble = max(10, typical[pid] // 80)
        rows: List[_Slot] = []
        previous_out = False
        for slot in range(n_slots):
            day = slot // SLOTS_PER_DAY
            if source == STICKY_SOURCE:
                # One price per change-window, so the page -- and its ETag
                # -- stay put and a repeat poll costs the host a 304.
                level = daily[pid][(day // STICKY_CHANGE_DAYS) * STICKY_CHANGE_DAYS]
                nudge = 0
            else:
                level = daily[pid][day]
                nudge = stream.randint(-wobble, wobble)
            price = max(100, level * SOURCE_FACTOR[source] // 100 + nudge)

            out = ((day + offset) % cycle) < out_len
            if index == 0 and source == "examplemart":
                # One listing is pinned to a known sellout on day 1, so a
                # sellout and a restock are in every run however short.
                out = day == 1
            if pid == crash_product and slot in crash_slots:
                out = False
            if pid == misparse_pid and slot <= misparse_slots[-1]:
                # Every listing of the mis-parsed product stays in stock
                # until the per-pack price appears.  A market reference
                # needs samples, and a product nobody has seen in stock
                # has none -- in which case refusing the mis-parse would
                # be luck rather than judgement, and the gate would be
                # testing nothing.  This is also the realistic case: the
                # mis-parse that costs money is the one on a product you
                # have been watching for weeks.
                out = False
            if (source, pid) == limit_key:
                # Always on the shelf: the rule over it asks for one more
                # copy than the page will sell, so every look at it is a
                # look at the clamp.
                out = False

            note = ""
            misparse = False
            if out:
                stock = Stock.OUT_OF_STOCK
                shown: Optional[Cents] = None
                sellouts += 1 if not previous_out else 0
            else:
                stock = Stock.LIMITED if previous_out else Stock.IN_STOCK
                restocks += 1 if previous_out else 0
                shown = price
                if (source, pid) == misparse_key and slot in misparse_slots:
                    # A box page quoting the price of a single pack. The
                    # parser will read it correctly; the number means
                    # something else.
                    shown = max(1, price // MISPARSE_DIVISOR)
                    note = "placeholder page quoted a per-pack price"
                    misparse = True
            previous_out = out

            candidate = _Slot(
                price=shown,
                shipping=SOURCE_SHIPPING[source] if shown is not None else 0,
                stock=stock,
                limit=SOURCE_LIMIT[source],
                note=note,
                misparse=misparse,
                version=slot,
            )
            if rows and rows[-1].content == candidate.content:
                candidate = replace(candidate, version=rows[-1].version)
            rows.append(candidate)
        slots[(source, pid)] = rows

    return _Script(
        slots=slots,
        typical=typical,
        crash_product=crash_product,
        crash_slots=crash_slots,
        misparse_key=misparse_key,
        misparse_slots=misparse_slots,
        limit_key=limit_key,
        outage_slots=outage_slots,
        n_slots=n_slots,
        sellouts=sellouts,
        restocks=restocks,
    )


# --------------------------------------------------------------------------
# The injected fetcher and parser
# --------------------------------------------------------------------------


class _ScriptedHost:
    """The injected :class:`~jarvis_poke.contracts.Fetcher`.

    It stands in for four placeholder retailers and keeps the *call log*
    the politeness checks are made against -- every call, whoever made it,
    including one that routed around the scheduler.  That is the whole
    reason the log lives out here rather than in
    :meth:`PollScheduler.stats`: a scheduler cannot be its own witness.

    It serves a small JSON body of this module's own invention.  No
    retailer's markup, no selectors, nothing fetched: contracts.py, "The
    package makes no network calls".
    """

    def __init__(self, script: _Script, catalog: Catalog, clock) -> None:
        self.script = script
        self.clock = clock
        self._by_url = {sku.url: sku for sku in catalog.skus()}
        self.calls: List[_Call] = []
        self.served: Dict[str, str] = {}      # url -> the last ETag we gave out
        self.unknown_urls = 0
        self.missing_conditional: List[_Call] = []

    def __call__(
        self, url: str, headers: Dict[str, str], policy: FetchPolicy
    ) -> FetchResult:
        at = float(self.clock())
        sku = self._by_url.get(url)
        if sku is None:  # pragma: no cover - the catalog is built from these urls
            self.unknown_urls += 1
            return FetchResult(ok=False, status=404, reason="unknown listing")

        sent = headers.get("If-None-Match")
        call = _Call(
            source=sku.source,
            product_id=sku.product_id,
            at=at,
            conditional=bool(sent),
            outcome="",
        )
        if url in self.served and not sent:
            # We hold a validator for this page and asked for the whole
            # body again: contracts.py's conditional-request rule, broken.
            self.missing_conditional.append(call)

        slot_index = self.script.slot_index(at)
        if sku.source == OUTAGE_SOURCE and slot_index in self.script.outage_slots:
            self.calls.append(replace(call, outcome="error"))
            return FetchResult(ok=False, status=503, reason="upstream busy")

        slot = self.script.at(sku.source, sku.product_id, at)
        etag = f'W/"{sku.source}-{sku.product_id}-{slot.version}"'
        self.served[url] = etag
        if sent == etag:
            self.calls.append(replace(call, outcome="not_modified"))
            return FetchResult(ok=True, status=304, not_modified=True, etag=etag)

        body = json.dumps(
            {
                "sku": sku.sku,
                "stock": slot.stock.value,
                "price_cents": slot.price,
                "shipping_cents": slot.shipping,
                "limit": slot.limit,
                "note": slot.note,
            },
            sort_keys=True,
        )
        self.calls.append(replace(call, outcome="ok"))
        return FetchResult(ok=True, status=200, body=body, etag=etag)

    # -- views ------------------------------------------------------------

    def by_source(self) -> Dict[str, List[_Call]]:
        out: Dict[str, List[_Call]] = {}
        for call in self.calls:
            out.setdefault(call.source, []).append(call)
        return out


def parse_listing(sku: SourceSku, body: str, at: float) -> Observation:
    """The injected :class:`~jarvis_poke.contracts.Parser`.

    It reads the placeholder JSON :class:`_ScriptedHost` serves and
    believes it, which is deliberate: the mis-parse in the script is a
    page that states a *different price than it means*, and a parser that
    second-guessed the number would hide exactly the failure the engine's
    outlier gate exists to catch.  Judging a price is the engine's job,
    not the parser's.
    """
    obj = json.loads(body)
    price = obj.get("price_cents")
    if price is not None:
        price = int(price)
    limit = obj.get("limit")
    return Observation(
        product_id=sku.product_id,
        source=sku.source,
        sku=sku.sku,
        at=at,
        stock=Stock(obj["stock"]),
        price=price,
        shipping=int(obj.get("shipping_cents") or 0),
        per_customer_limit=None if limit is None else int(limit),
        url=sku.url,
        note=str(obj.get("note") or ""),
    )


# --------------------------------------------------------------------------
# The defects that are not a one-line poke at the driver
# --------------------------------------------------------------------------


class _LooseLedger(RuleSet):
    """The ``ignore_budget`` defect: a ledger that says yes to everything.

    It answers every :meth:`affordable` with ok, and neither
    :meth:`reserve` nor :meth:`commit` refuses to go overdrawn -- which is
    what a ledger that has stopped applying
    ``MAX_BUDGET_FRACTION_PER_VERDICT`` looks like from the engine's side.
    It reaches into :class:`RuleSet`'s own fields because it is standing
    in for that class's arithmetic; nothing outside this module does that.
    """

    def affordable(self, rule, landed, quantity=None, budget=None) -> Affordability:
        count = rule.quantity if quantity is None else int(quantity)
        unit = int(landed)
        left = self.remaining()
        return Affordability(
            True, count, unit, unit * count, left, left,
            "defect ignore_budget: the ledger waved this through",
        )

    def reserve(self, amount: Cents) -> Cents:
        self._reserved += int(amount)
        return self.remaining()

    def release(self, amount: Cents) -> Cents:
        self._reserved = max(0, self._reserved - int(amount))
        return self.remaining()

    def commit(self, amount: Cents) -> Cents:
        amount = int(amount)
        self._reserved = max(0, self._reserved - min(self._reserved, amount))
        self._budget = replace(self._budget, spent=self._budget.spent + amount)
        return self.remaining()


class _RecordingAlertService:
    """Stands in for :class:`jarvis_alerts.api.AlertService`.

    It records what it was handed and nothing more -- the gate is about
    jarvis_poke's decisions, and jarvis_alerts has its own gate for
    delivery.  Ids are a counter, so a run is reproducible.
    """

    def __init__(self) -> None:
        self.sent: List[Dict[str, Any]] = []

    def publish(
        self,
        profile_id: str,
        kind: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        priority: Any = None,
        dedupe_key: Optional[str] = None,
    ) -> str:
        alert_id = f"gate-alert-{len(self.sent) + 1:05d}"
        self.sent.append(
            {
                "id": alert_id,
                "profile_id": profile_id,
                "kind": kind,
                "title": title,
                "body": body,
                "data": dict(data or {}),
                "priority": getattr(priority, "name", str(priority)),
                "dedupe_key": dedupe_key,
            }
        )
        return alert_id


# --------------------------------------------------------------------------
# One decision, with everything the checks need about the moment it was made
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Decision:
    verdict: Verdict
    rule: Optional[Rule]
    best: Optional[Observation]
    market_usable: bool
    remaining_before: Cents
    cap_before: Cents
    fabricated: bool                 # the gate forged this one, for a defect


# --------------------------------------------------------------------------
# The gate itself
# --------------------------------------------------------------------------


class _Gate:
    """Builds the month, runs it, and then argues with the result."""

    def __init__(
        self, n_products: int, n_days: int, seed: int, inject_defect: Optional[str]
    ) -> None:
        if n_products < MIN_PRODUCTS:
            raise GateError(f"n_products must be at least {MIN_PRODUCTS}, got {n_products}")
        if n_days < MIN_DAYS:
            raise GateError(f"n_days must be at least {MIN_DAYS}, got {n_days}")
        if inject_defect is not None and inject_defect not in DEFECTS:
            raise GateError(
                f"unknown defect {inject_defect!r}; known: {', '.join(DEFECTS)}"
            )

        self.n_products = int(n_products)
        self.n_days = int(n_days)
        self.seed = int(seed)
        self.defect = inject_defect
        self.fields = SeedFields.parse(self.seed)

        self._now = GATE_EPOCH
        self.clock = lambda: self._now

        self.catalog = _gate_catalog(self.n_products, self.fields)
        self.policies = {
            source: policy
            for source, policy in load_policies().items()
            if source in set(self.catalog.sources())
        }
        self.script = _build_script(self.catalog, self.n_days, self.fields)

        self.host = _ScriptedHost(self.script, self.catalog, self.clock)
        self.scheduler = PollScheduler(
            self.catalog, self.policies, self.clock, seed=self.seed
        )
        self.history = PriceHistory()

        ledger_class = _LooseLedger if self.defect == "ignore_budget" else RuleSet
        self.window_days = max(1, self.n_days // 4)
        self.budget_total = self._budget_total()
        self.rules: RuleSet = ledger_class(
            self._build_rules(),
            Budget(total=self.budget_total, window_s=self.window_days * DAY_S),
        )
        # ``reserve_on_buy=True``: the *engine* takes the reservation, and
        # the gate only plays the owner -- buy it, or let the deep link
        # lapse -- through :meth:`DecisionEngine.commit_purchase` and
        # :meth:`DecisionEngine.release_reservation`.  This used to be
        # False, with the gate driving ``RuleSet.reserve`` and
        # ``RuleSet.commit`` itself, which proved a budget lifecycle that
        # existed only inside the gate: the one ``cli.py`` actually uses
        # was never executed here, and a reservation that left postage
        # out was a green run.
        self.engine = DecisionEngine(
            self.catalog,
            self.rules,
            self.history,
            self.clock,
            reserve_on_buy=True,
            market_window_s=MARKET_WINDOW_S,
        )
        self.service = _RecordingAlertService()
        self.bridge = AlertBridge(self.service, self.clock)

        self.decisions: List[_Decision] = []
        self.problems: List[Problem] = []
        self.misparse_keys: Set[Tuple[str, str, str, float]] = set()
        #: product id -> when this product's reservation lapses.  The
        #: cents live on the ledger, where the shipped product keeps
        #: them; only "when does the owner give up on it" is the gate's.
        self.lapses: Dict[str, float] = {}
        self.window_committed: List[Cents] = [0]
        self.committed_total: Cents = 0
        self.commits = 0
        self.releases = 0
        self.cap_refusals = 0
        self.misparse_refusals = 0
        self.budget_rolls = 0
        self.observations = 0
        self.disallowed_offered = 0
        #: How many non-BUY verdicts were handed to the bridge (which must
        #: publish none of them), and how many BUYs had their quantity cut
        #: by a listing's stated per-customer limit.
        self.bridge_non_buys = 0
        self.limit_clamps = 0
        self._ledger_complained = False
        self._injected: Set[str] = set()
        self._owner_rolls = 0

    # -- building ----------------------------------------------------------

    def _budget_total(self) -> Cents:
        """Half again the dearest thing on the list.

        Deliberately tight.  ``MAX_BUDGET_FRACTION_PER_VERDICT`` is half of
        what is left, so a fresh window covers one ordinary box and refuses
        the dearest ones outright -- which means a healthy month exercises
        the refusal as well as the approval, and ``ignore_budget`` has
        something to break.  :meth:`_check_scenario` insists that at least
        one verdict really was refused by the cap, so this staying true is
        not a matter of hoping.
        """
        dearest = max(self.script.typical.values())
        return int(dearest * 3 // 2)

    def _build_rules(self) -> List[Rule]:
        """One standing instruction per product, drawn from the seed.

        Ceilings sit near the market rather than far below it, so the
        month contains BUYs as well as waiting.  Four rules are fixed
        rather than drawn, because four outcomes have to be in every run
        or the checks above have nothing to look at:

        * the **dearest** product asks for two of them at 30% over market,
          which no fresh window can fit inside
          ``MAX_BUDGET_FRACTION_PER_VERDICT`` -- so the cap really refuses
          something, and ``ignore_budget`` has something to let through;
        * the **crashed** product asks for one, at 92%, 10% off, from any
          source: the crash clears all three, so a genuine bargain must
          produce a BUY and cannot be explained away by the ledger;
        * the **mis-parsed** product asks for one at 30% over market with
          no discount required, from any source, so that when the
          per-pack price appears the *only* thing between it and an alert
          is the outlier gate.  A ceiling or a discount refusing it first
          would make "no BUY on the mis-parse" true for the wrong reason;
        * one other rule is **switched off**, because a disabled rule is a
          path the engine has to take.

        Cooldowns are six hours -- comfortably longer than the bridge's
        own five-minute dedupe window, so the two suppressions cannot be
        mistaken for one another.
        """
        products = self.catalog.products()
        typicals = self.script.typical
        dearest = max(sorted(typicals), key=lambda pid: typicals[pid])
        crashed = self.script.crash_product
        misparsed = self.script.misparse_key[1]
        limit_pid_for_disable = self.script.limit_key[1]
        disabled = next(
            (
                p.id for p in products
                if p.id not in (dearest, crashed, misparsed, limit_pid_for_disable)
            ),
            "",
        )
        limit_source, limit_pid = self.script.limit_key
        limited = (
            (limit_pid, SOURCE_LIMIT.get(limit_source) or 1, limit_source)
            if limit_pid else ("", 0, "")
        )

        rules: List[Rule] = []
        for product in products:
            stream = self.fields.stream(f"poke.gate.rule:{product.id}")
            typical = typicals[product.id]
            ceiling_pct = stream.choice((80, 86, 92, 98))
            quantity = stream.choice((1, 1, 2))
            discount = stream.choice((0.0, 5.0, 10.0))
            restricted = stream.chance(0.2)
            if product.id == dearest:
                ceiling_pct, quantity, discount, restricted = 130, 2, 0.0, False
            elif product.id == crashed:
                ceiling_pct, quantity, discount, restricted = 92, 1, 10.0, False
            elif product.id == misparsed:
                ceiling_pct, quantity, discount, restricted = 130, 1, 0.0, False
            elif product.id == limited[0]:
                # Asks for more copies than the one source it allows will
                # sell to one customer.  Without it the clamp in
                # ``DecisionEngine`` step 8 was never executed in any run
                # at any seed: the gate's tight budget refused nearly
                # every quantity-2 rule, and the two sources that state a
                # limit are the dear ones, so they were rarely the
                # cheapest.  An engine that ignored "limit 1 per
                # customer" and told the owner to buy two -- a cancelled
                # order, or an account flagged for limit-busting -- was a
                # green run.
                ceiling_pct, discount, restricted = 130, 0.0, False
                quantity = limited[1] + 1
            sources: Tuple[str, ...] = ()
            if product.id == limited[0]:
                sources = (limited[2],)
            elif restricted:
                sources = ("examplemart", "cardbarn")
            rules.append(
                Rule(
                    product_id=product.id,
                    max_price=max(200, typical * ceiling_pct // 100),
                    quantity=quantity,
                    min_discount_pct=discount,
                    allowed_sources=sources,
                    include_shipping=True,
                    cooldown_s=6 * 3600.0,
                    enabled=product.id != disabled,
                )
            )
        return rules

    # -- running -----------------------------------------------------------

    def run(self) -> GateReport:
        try:
            self._tick_through()
        except Exception as exc:  # noqa: BLE001 - a crash is a finding, not a traceback
            self.problems.append(
                Problem(
                    kind="crashed",
                    detail=f"the run raised {type(exc).__name__}: {exc}",
                    at=self._now,
                )
            )
        self._check_politeness()
        self._check_buys()
        self._check_budget()
        self._check_cooldowns()
        self._check_reasons()
        self._check_alerts()
        self._check_round_trip()
        self._check_contract()
        self._check_boundaries()
        self._check_scenario()
        return GateReport(
            n_products=self.n_products,
            n_days=self.n_days,
            seed=self.seed,
            inject_defect=self.defect,
            counts=self._counts(),
            problems=self.problems,
        )

    def _tick_through(self) -> None:
        ticks = int(self.n_days * DAY_S / TICK_S)
        window_s = self.window_days * DAY_S
        next_roll = GATE_EPOCH + window_s
        for tick in range(ticks):
            now = GATE_EPOCH + tick * TICK_S
            self._now = now

            if now >= next_roll:
                self._roll_budget()
                next_roll += window_s

            self._expire_reservations(now)

            touched: Set[str] = set()
            for sku in self.scheduler.due(now):
                if not self.policies[sku.source].robots_allows:
                    self.disallowed_offered += 1
                observation = self.scheduler.poll_once(
                    sku, self.host, parse_listing, now
                )
                if observation is None:
                    continue
                self.observations += 1
                self.history.append(observation)
                if self.script.at(sku.source, sku.product_id, now).misparse:
                    self.misparse_keys.add(
                        (
                            observation.product_id,
                            observation.source,
                            observation.sku,
                            observation.at,
                        )
                    )
                touched.add(observation.product_id)
                self._maybe_poll_too_fast(sku, now)

            self._maybe_ignore_robots(now)

            for product_id in sorted(touched):
                self._decide(product_id, now)

    def _roll_budget(self) -> None:
        """A new window: contracts.py gives ``Budget`` a window length and
        no start, and rules.py says rolling it over is the caller's job.

        A fresh window that cannot cover what is already reserved is
        refused by ``RuleSet.set_budget``, and rightly -- it is a ledger
        that does not add up.  That is a finding, not a crash, because it
        is precisely what ``ignore_budget`` produces.
        """
        try:
            self.rules.set_budget(
                Budget(total=self.budget_total, window_s=self.window_days * DAY_S)
            )
        except ValueError as exc:
            self._fail(
                "budget_exceeded",
                f"a fresh {fmt_cents(self.budget_total)} window could not cover the "
                f"outstanding reservations: {exc}",
                at=self._now,
            )
            return
        self.window_committed.append(0)
        self.budget_rolls += 1

    def _expire_reservations(self, now: float) -> None:
        """The owner never opened the deep link: hand the money back."""
        for product_id, expires in sorted(self.lapses.items()):
            if now >= expires:
                if self.engine.release_reservation(product_id):
                    self.releases += 1
                del self.lapses[product_id]

    def _decide(self, product_id: str, now: float) -> None:
        candidates = self.history.for_product(product_id, since=now - FRESH_S)
        # This product's own un-acted-on alert is money this verdict
        # supersedes rather than competes with; the engine adds it back
        # before applying the cap, so the gate measures against the same
        # number -- computed here from the gate's own fraction.
        held = self.rules.reserved_for(product_id)
        remaining_before = self.rules.remaining() + held
        cap_before = _gate_cap(remaining_before)

        verdict = self.engine.evaluate(product_id, candidates)
        rule = self.rules.get(product_id)
        best = self._observation_for(verdict, candidates)
        reference = self.history.market_ref(product_id, now)
        decision = _Decision(
            verdict=verdict,
            rule=rule,
            best=best,
            market_usable=bool(reference.usable),
            remaining_before=remaining_before,
            cap_before=cap_before,
            fabricated=False,
        )
        self.decisions.append(decision)

        self._check_ledger_adds_up(product_id, now)
        self._note_misparse_refusal(decision, candidates)
        self._note_limit_clamp(decision)
        if verdict.action is Action.BUY:
            self._on_buy(decision, now)
        else:
            self._offer_to_bridge(decision)
            self._note_cap_refusal(decision)
            self._maybe_forge_buy(decision, candidates, now)
            self._maybe_alert_on_watch(decision)

        if self.defect == "ignore_cooldown" and verdict.action is Action.BUY:
            # An engine that forgets it just told its owner to go and spend.
            self.engine.watch_state(product_id).last_alert_at = 0.0

    def _observation_for(
        self, verdict: Verdict, candidates: Sequence[Observation]
    ) -> Optional[Observation]:
        if verdict.source is None or verdict.sku is None:
            return None
        matches = [
            obs for obs in candidates
            if obs.source == verdict.source and obs.sku == verdict.sku
        ]
        return max(matches, key=lambda obs: obs.at) if matches else None

    def _note_misparse_refusal(
        self, decision: _Decision, candidates: Sequence[Observation]
    ) -> None:
        """Count the times the per-pack price was the best offer and lost.

        "No BUY on the mis-parse" is only worth asserting if the engine
        was actually offered it as the cheapest thing on the list and
        turned it down.  A month where it was never the best offer --
        because the product was never seen in stock, say -- would satisfy
        the check by accident, so :meth:`_check_scenario` insists this is
        not zero.
        """
        if decision.verdict.product_id != self.script.misparse_key[1]:
            return
        purchasable = [obs for obs in _newest(candidates) if obs.purchasable]
        if not purchasable:
            return
        cheapest = min(purchasable, key=lambda obs: (obs.landed or 0, obs.source))
        key = (cheapest.product_id, cheapest.source, cheapest.sku, cheapest.at)
        if key in self.misparse_keys and decision.verdict.action is not Action.BUY:
            self.misparse_refusals += 1

    def _check_ledger_adds_up(self, product_id: str, now: float) -> None:
        """Total reserved equals the sum of the named reservations.

        Cheap, and it catches the drift a stacking reservation causes:
        a second BUY for one product that adds to its hold instead of
        replacing it leaves cents reserved that no key owns, which no
        release can ever hand back.
        """
        if self._ledger_complained:
            return
        named = sum(self.rules.reservations().values())
        if self.rules.reserved != named:
            self._ledger_complained = True
            self._fail(
                "budget_exceeded",
                f"the ledger reserves {fmt_cents(self.rules.reserved)} but only "
                f"{fmt_cents(named)} of it belongs to a product: the difference "
                f"is money nothing can release",
                product_id=product_id,
                at=now,
            )

    def _note_limit_clamp(self, decision: _Decision) -> None:
        """Count the BUYs a listing's per-customer limit actually cut.

        Coverage, not a check: :meth:`_check_buys` asserts no BUY is ever
        over a stated limit, and :meth:`_check_scenario` refuses a run in
        which the clamp never bound, because otherwise that assertion is
        made about a code path nothing executed.
        """
        verdict = decision.verdict
        rule, best = decision.rule, decision.best
        if verdict.action is not Action.BUY or rule is None or best is None:
            return
        limit = best.per_customer_limit
        if limit is None or limit >= rule.quantity:
            return
        if verdict.quantity == limit:
            self.limit_clamps += 1

    def _note_cap_refusal(self, decision: _Decision) -> None:
        """Count the verdicts the per-verdict cap was the reason for.

        A listing under the owner's ceiling that did not become a BUY and
        would not have fitted the cap is the budget doing its job.  The
        count is coverage, not a check: :meth:`_check_scenario` refuses a
        month in which the cap never once bound, because ``ignore_budget``
        would then have nothing to break.
        """
        rule = decision.rule
        best = decision.best
        if rule is None or not rule.enabled or best is None or not best.purchasable:
            return
        cost = best.landed if rule.include_shipping else best.price
        if cost is None or cost > rule.max_price:
            return
        if cost * rule.quantity > decision.cap_before:
            self.cap_refusals += 1

    def _on_buy(self, decision: _Decision, now: float) -> None:
        """Alert, reserve, then let the owner act on about one alert in twelve.

        Acting is a *commit* of money the ledger already reserved -- the
        owner opened the deep link and bought it by hand.  Ignoring it
        lets the reservation lapse after :data:`RESERVATION_TTL_S`.
        Nothing here checks anything out; contracts.py, "What this is not".
        """
        verdict = decision.verdict
        product = self.catalog.product(verdict.product_id)
        self.bridge.publish_verdict(verdict, product)
        if decision.fabricated:
            return

        # The engine took the reservation on the way out of evaluate();
        # what is left for the gate is the owner's half of the exchange.
        amount = self.rules.reserved_for(verdict.product_id)
        if amount <= 0:
            return

        self._owner_rolls += 1
        # The owner buys the first thing each window brings them and then
        # about one alert in twelve after that.  The first is not a draw:
        # a month in which nobody ever acted would never exercise
        # ``RuleSet.commit`` at all, and "the budget was never spent" is
        # a poor way to pass a budget check.
        acts = self.window_committed[-1] == 0 or self.fields.stream(
            f"poke.gate.owner#{self._owner_rolls}"
        ).chance(OWNER_ACTS_P)
        if acts:
            spent = self.engine.commit_purchase(verdict.product_id)
            self.lapses.pop(verdict.product_id, None)
            self.commits += 1
            self.committed_total += spent
            self.window_committed[-1] += spent
        else:
            self.lapses[verdict.product_id] = now + RESERVATION_TTL_S

    # -- defect injection --------------------------------------------------

    def _offer_to_bridge(self, decision: _Decision) -> None:
        """Hand a non-BUY to the bridge, which must refuse to publish it.

        The gate used to call ``bridge.publish_verdict`` only for a BUY,
        so ``AlertBridge``'s "BUY only" guard -- the thing standing
        between the owner and three hundred HIGH-priority buzzes a week
        for "still out of stock" -- was never once reached with anything
        else.  A bridge that published on WATCH passed clean.  It is
        offered here, and publishing anything is the finding.

        ``alert_on_watch`` reaches the same failure from the other side,
        through the service; this reaches it through the filter that is
        supposed to stop it.
        """
        verdict = decision.verdict
        product = self.catalog.product(verdict.product_id)
        if product is None:  # pragma: no cover - every verdict names a catalog product
            return
        before = self.bridge.published
        alert_id = self.bridge.publish_verdict(verdict, product)
        self.bridge_non_buys += 1
        if alert_id is not None or self.bridge.published != before:
            self._fail(
                "alert_on_watch",
                f"the bridge published a {verdict.action.value} verdict; only a BUY "
                f"may ever reach the phone",
                product_id=verdict.product_id,
                source=verdict.source or "",
                at=verdict.at,
            )

    def _maybe_poll_too_fast(self, sku: SourceSku, now: float) -> None:
        if self.defect != "poll_too_fast" or "poll_too_fast" in self._injected:
            return
        if sku.source != "examplemart":
            return
        self._injected.add("poll_too_fast")
        headers = self.scheduler.conditional_headers(sku)
        self._now = now + 1.0
        try:
            self.host(sku.url, headers, self.policies[sku.source])
        finally:
            self._now = now

    def _maybe_ignore_robots(self, now: float) -> None:
        if self.defect != "ignore_robots" or "ignore_robots" in self._injected:
            return
        listings = self.catalog.skus_from(DISALLOWED_SOURCE)
        if not listings:  # pragma: no cover - every run lists it
            return
        self._injected.add("ignore_robots")
        self.host(listings[0].url, {}, self.policies[DISALLOWED_SOURCE])

    def _maybe_forge_buy(
        self, decision: _Decision, candidates: Sequence[Observation], now: float
    ) -> None:
        """``ignore_ceiling`` and ``trust_outlier``: an engine that says
        BUY where the real one says wait.

        The forged verdict is pushed through the same alert path as a real
        one, so the alert checks stay quiet and the check that names the
        defect is the one that fires.
        """
        if self.defect not in ("ignore_ceiling", "trust_outlier"):
            return
        if self.defect in self._injected:
            return
        rule = decision.rule
        if rule is None or not rule.enabled:
            return
        purchasable = [obs for obs in _newest(candidates) if obs.purchasable]
        if not purchasable:
            return
        cheapest = min(purchasable, key=lambda obs: (obs.landed or 0, obs.source))
        landed = cheapest.landed or 0

        if self.defect == "ignore_ceiling":
            if landed <= rule.max_price:
                return
            reason = (
                f"defect ignore_ceiling: {fmt_cents(landed)} is over the "
                f"{fmt_cents(rule.max_price)} ceiling and was alerted anyway"
            )
        else:
            key = (cheapest.product_id, cheapest.source, cheapest.sku, cheapest.at)
            if key not in self.misparse_keys:
                return
            reason = "defect trust_outlier: the mis-parsed bargain was alerted"

        self._injected.add(self.defect)
        forged = Verdict(
            product_id=decision.verdict.product_id,
            action=Action.BUY,
            at=now,
            source=cheapest.source,
            sku=cheapest.sku,
            price=cheapest.price,
            landed=cheapest.landed,
            market=decision.verdict.market,
            discount_pct=decision.verdict.discount_pct,
            quantity=1,
            url=cheapest.url,
            reasons=tuple(decision.verdict.reasons) + (reason,),
        )
        forged_decision = replace(
            decision, verdict=forged, best=cheapest, fabricated=True
        )
        self.decisions.append(forged_decision)
        self._on_buy(forged_decision, now)

    def _maybe_alert_on_watch(self, decision: _Decision) -> None:
        """``alert_on_watch``: a monitor that buzzes for a price it is
        still waiting on.  It has to go round the bridge, which refuses
        to alert for anything but a BUY."""
        if self.defect != "alert_on_watch" or "alert_on_watch" in self._injected:
            return
        if decision.verdict.action is not Action.WATCH:
            return
        self._injected.add("alert_on_watch")
        self.service.publish(
            self.bridge.profile_id,
            self.bridge.kind,
            f"Watching {decision.verdict.product_id}",
            "defect alert_on_watch: a WATCH was pushed to the phone",
            data={"product_id": decision.verdict.product_id},
            dedupe_key=f"poke:watch|{decision.verdict.product_id}",
        )

    # -- boundary probes ---------------------------------------------------

    def _probe(
        self,
        *,
        landed: Cents,
        max_price: Cents,
        quantity: int = 1,
        min_discount: float = 0.0,
        median: Optional[Cents] = None,
        budget_total: Optional[Cents] = None,
        limit: Optional[int] = None,
        shipping: Cents = 0,
        include_shipping: bool = True,
        reserve: bool = False,
        ledger_out: Optional[List[RuleSet]] = None,
    ) -> Verdict:
        """One controlled evaluation, on a ledger and a clock of its own.

        Nothing here touches the run's rules, budget, engine or watch
        states: it is a fresh :class:`DecisionEngine` over a fresh
        :class:`RuleSet` with ``reserve_on_buy=False``, so a probe can
        neither spend the run's money nor move its cooldowns.
        """
        pid = "gate-boundary-probe"
        if budget_total is None:
            budget_total = landed * max(1, quantity) * 4 + 100_000
        rules = RuleSet(
            [
                Rule(
                    product_id=pid,
                    max_price=max_price,
                    quantity=quantity,
                    min_discount_pct=min_discount,
                    include_shipping=include_shipping,
                    cooldown_s=0.0,
                )
            ],
            Budget(total=budget_total),
        )
        if ledger_out is not None:
            ledger_out.append(rules)
        reference = (
            None if median is None
            else MarketRef(pid, samples=9, median=median, p25=median, low=median,
                           window_s=MARKET_WINDOW_S)
        )

        def market_ref(product_id: str, now: float) -> Optional[MarketRef]:
            return reference

        engine = DecisionEngine(
            None, rules, market_ref, lambda: GATE_EPOCH, reserve_on_buy=reserve
        )
        observation = Observation(
            product_id=pid,
            source="examplemart",
            sku="PROBE-1",
            at=GATE_EPOCH,
            stock=Stock.IN_STOCK,
            price=landed - shipping,
            shipping=shipping,
            per_customer_limit=limit,
            url="https://examplemart.example.com/p/probe",
        )
        return engine.evaluate(pid, [observation])

    def _expect(
        self, what: str, kind: str, verdict: Verdict, action: Action, quantity: int = -1
    ) -> None:
        if verdict.action is not action:
            self._fail(
                kind,
                f"boundary probe: {what} -- expected {action.value}, got "
                f"{verdict.action.value} ({'; '.join(verdict.reasons[-2:])})",
            )
        elif quantity >= 0 and verdict.quantity != quantity:
            self._fail(
                kind,
                f"boundary probe: {what} -- expected quantity {quantity}, got "
                f"{verdict.quantity}",
            )

    def _check_contract(self) -> None:
        """The constants contracts.py documents still say what it says.

        The gate carries its own copies (:data:`GATE_BUDGET_FRACTION`,
        :data:`GATE_OUTLIER_MIN_PCT`).  ``verdict_over_cap`` used to
        re-derive the cap from the package's constant with the package's
        own helper, so both sides moved together and doubling the
        constant -- the one thing standing between a mis-parsed price and
        the whole budget -- was invisible.  Comparing the two here is
        what makes that visible.
        """
        if Fraction(MAX_BUDGET_FRACTION_PER_VERDICT) != GATE_BUDGET_FRACTION:
            self._fail(
                "verdict_over_cap",
                f"contracts.MAX_BUDGET_FRACTION_PER_VERDICT is "
                f"{MAX_BUDGET_FRACTION_PER_VERDICT!r}, but the documented "
                f"per-verdict cap is {GATE_BUDGET_FRACTION}",
            )
        for remaining in (1, 3, 999, 10_000, 10_001, 123_457):
            if budget_cap(remaining) != _gate_cap(remaining):
                self._fail(
                    "verdict_over_cap",
                    f"rules.budget_cap({remaining}) is {budget_cap(remaining)}, but "
                    f"{GATE_BUDGET_FRACTION} of it is {_gate_cap(remaining)}",
                )
        if _prices.OUTLIER_MIN_PCT_OF_MEDIAN != GATE_OUTLIER_MIN_PCT:
            self._fail(
                "buy_on_misparse",
                f"prices.OUTLIER_MIN_PCT_OF_MEDIAN is "
                f"{_prices.OUTLIER_MIN_PCT_OF_MEDIAN}, not the documented "
                f"{GATE_OUTLIER_MIN_PCT}",
            )

    def _check_boundaries(self) -> None:
        """Probe every money comparison one cent either side of its edge.

        The scripted month is not a boundary test and cannot be made into
        one: the tightest BUY in a default run clears its ceiling by two
        cents at one seed and by nine at another, and clears its discount
        threshold by six dollars, so whether a one-cent slip in
        ``engine.py`` is noticed is a coin flip on the seed.  The
        mis-parse sits at 5% of the median against a 25% floor -- four
        times below the boundary it is supposed to be testing.  These
        probes are exact and seed-independent: each one is the last cent
        that must be accepted and the first that must be refused.

        contracts.py, ``to_cents``: "a float budget is how you end up
        buying something for a cent more than the cap."
        """
        # 1. the ceiling, with no market reference so nothing else can decide
        ceiling = 10_000
        self._expect(
            f"landed exactly at the {fmt_cents(ceiling)} ceiling",
            "boundary_strict",
            self._probe(landed=ceiling, max_price=ceiling),
            Action.BUY,
        )
        self._expect(
            f"landed one cent over the {fmt_cents(ceiling)} ceiling",
            "buy_over_ceiling",
            self._probe(landed=ceiling + 1, max_price=ceiling),
            Action.WATCH,
        )

        # 2. the discount threshold against a usable reference
        median = 20_000
        threshold = discount_threshold(median, 10.0)
        self._expect(
            f"landed exactly at the {fmt_cents(threshold)} discount threshold",
            "boundary_strict",
            self._probe(landed=threshold, max_price=median * 2,
                        min_discount=10.0, median=median),
            Action.BUY,
        )
        self._expect(
            "landed one cent short of the discount the rule asks for",
            "buy_under_discount",
            self._probe(landed=threshold + 1, max_price=median * 2,
                        min_discount=10.0, median=median),
            Action.WATCH,
        )

        # 3. the outlier floor, from the gate's own percentage
        floor = median * GATE_OUTLIER_MIN_PCT // 100
        self._expect(
            f"landed exactly at the {GATE_OUTLIER_MIN_PCT}% outlier floor",
            "boundary_strict",
            self._probe(landed=floor, max_price=median * 2, median=median),
            Action.BUY,
        )
        self._expect(
            f"landed one cent under the {GATE_OUTLIER_MIN_PCT}% outlier floor",
            "buy_on_misparse",
            self._probe(landed=floor - 1, max_price=median * 2, median=median),
            Action.SKIP,
        )

        # 4. the per-verdict cap, measured with the gate's own fraction
        budget = 20_000
        cap = _gate_cap(budget)
        self._expect(
            f"a spend exactly at the {fmt_cents(cap)} per-verdict cap",
            "boundary_strict",
            self._probe(landed=cap, max_price=budget * 2, budget_total=budget),
            Action.BUY,
        )
        self._expect(
            "a spend one cent over the per-verdict cap",
            "verdict_over_cap",
            self._probe(landed=cap + 1, max_price=budget * 2, budget_total=budget),
            Action.SKIP,
        )

        # 5. the per-customer limit
        self._expect(
            "a rule for 3 against a page that will sell 2",
            "buy_bad_quantity",
            self._probe(landed=1_000, max_price=10_000, quantity=3, limit=2),
            Action.BUY,
            quantity=2,
        )
        self._expect(
            "a rule for 3 against a page that will sell 0",
            "buy_bad_quantity",
            self._probe(landed=1_000, max_price=10_000, quantity=3, limit=0),
            Action.SKIP,
        )
        self._expect(
            "a rule for 3 against a page that states no limit",
            "boundary_strict",
            self._probe(landed=1_000, max_price=10_000, quantity=3),
            Action.BUY,
            quantity=3,
        )

        # 6. the ledger is charged what the card is charged.
        #    ``include_shipping`` says which number the owner's *ceiling*
        #    is measured against; it has never said anything about what
        #    the shop takes off them, and a budget that leaves postage
        #    out is not the ceiling contracts.py promises.
        ledgers: List[RuleSet] = []
        verdict = self._probe(
            landed=16_000, shipping=5_000, max_price=13_000,
            include_shipping=False, quantity=2, reserve=True, ledger_out=ledgers,
        )
        self._expect(
            "a $110 shelf price with $50 postage, ceiling on the shelf price",
            "boundary_strict",
            verdict,
            Action.BUY,
            quantity=2,
        )
        if ledgers and verdict.action is Action.BUY:
            held = ledgers[0].reserved
            if held != 32_000:
                self._fail(
                    "verdict_over_cap",
                    f"boundary probe: the ledger held {fmt_cents(held)} for a "
                    f"purchase the card is charged {fmt_cents(32_000)} for",
                )
        # and the *cap* has to be applied to that same number: a budget
        # whose cap is $300 must refuse a $320 charge even though the
        # shelf total the ceiling looks at is only $220.
        self._expect(
            "a $320 charge against a $300 per-verdict cap, ceiling on the "
            "$220 shelf total",
            "verdict_over_cap",
            self._probe(
                landed=16_000, shipping=5_000, max_price=13_000,
                include_shipping=False, quantity=2, budget_total=60_000,
            ),
            Action.SKIP,
        )

        # 7. one product, two verdicts: the second replaces the first
        #    reservation rather than stacking a second one -- and it does
        #    so across engine objects, because the reservation belongs to
        #    the ledger and an app builds an engine per run.
        shared = RuleSet(
            [Rule(product_id="gate-hold-probe", max_price=20_000, cooldown_s=0.0)],
            Budget(total=200_000),
        )
        listing = Observation(
            product_id="gate-hold-probe", source="examplemart", sku="PROBE-2",
            at=GATE_EPOCH, stock=Stock.IN_STOCK, price=20_000, shipping=0,
            url="https://examplemart.example.com/p/hold",
        )
        for _ in range(3):
            fresh = DecisionEngine(
                None, shared, lambda pid, at: None, lambda: GATE_EPOCH,
                reserve_on_buy=True,
            )
            fresh.evaluate("gate-hold-probe", [listing])
        if shared.reserved != 20_000:
            self._fail(
                "verdict_over_cap",
                f"boundary probe: one {fmt_cents(20_000)} listing alerted three "
                f"times holds {fmt_cents(shared.reserved)}; a verdict replaces "
                f"its product's last reservation, it does not add to it",
            )
        if sum(shared.reservations().values()) != shared.reserved:
            self._fail(
                "budget_exceeded",
                "boundary probe: the ledger reserves more than its named "
                "reservations add up to, so some of it can never be released",
            )

    # -- checks ------------------------------------------------------------

    def _fail(self, kind: str, detail: str, **where: Any) -> None:
        self.problems.append(Problem(kind=kind, detail=detail, **where))

    def _check_politeness(self) -> None:
        """contracts.py, "Politeness is a design constraint".

        Measured on the fetcher's call log: what the run actually did to
        somebody else's host, not what the scheduler believes it did.
        """
        for source, calls in sorted(self.host.by_source().items()):
            policy = self.policies.get(source)
            if policy is None:  # pragma: no cover - urls come from the catalog
                self._fail("robots_ignored", "a source with no policy was fetched",
                           source=source)
                continue
            if not policy.robots_allows:
                self._fail(
                    "robots_ignored",
                    f"robots.txt disallows {source} and it was fetched "
                    f"{len(calls)} time(s); the first was the whole failure",
                    source=source,
                    at=calls[0].at,
                )
                continue
            times = sorted(call.at for call in calls)
            for earlier, later in zip(times, times[1:]):
                gap = later - earlier
                if gap < policy.min_interval_s - EPS:
                    self._fail(
                        "poll_too_fast",
                        f"{gap:.0f}s between calls, under the "
                        f"{policy.min_interval_s:.0f}s minimum interval",
                        source=source,
                        at=later,
                    )
                    break
        if self.disallowed_offered:
            self._fail(
                "robots_ignored",
                f"the scheduler offered a disallowed listing {self.disallowed_offered} "
                f"time(s) as due",
                source=DISALLOWED_SOURCE,
            )
        for call in self.host.missing_conditional[:1]:
            self._fail(
                "conditional_missing",
                "a repeat fetch carried no If-None-Match for a page that had "
                "already given us an ETag",
                product_id=call.product_id,
                source=call.source,
                at=call.at,
            )

    def _check_buys(self) -> None:
        """Every BUY satisfied its rule at the moment it was issued.

        contracts.py: BUY "meets every rule".  Re-derived here from the
        rule and the observation the verdict names, never from the
        engine's own summary of itself.
        """
        for decision in self.decisions:
            verdict = decision.verdict
            if verdict.action is not Action.BUY:
                continue
            rule = decision.rule
            where = dict(
                product_id=verdict.product_id, source=verdict.source or "", at=verdict.at
            )
            if rule is None or not rule.enabled:
                self._fail("buy_over_ceiling", "a BUY with no enabled rule behind it", **where)
                continue
            cost = verdict.landed if rule.include_shipping else verdict.price
            if cost is None:
                self._fail("buy_over_ceiling", "a BUY with no price", **where)
                continue
            if cost > rule.max_price:
                self._fail(
                    "buy_over_ceiling",
                    f"{fmt_cents(cost)} is over the {fmt_cents(rule.max_price)} ceiling",
                    **where,
                )
            if (
                verdict.market is not None
                and rule.min_discount_pct > 0
                and decision.market_usable
            ):
                threshold = discount_threshold(verdict.market, rule.min_discount_pct)
                if cost > threshold:
                    self._fail(
                        "buy_under_discount",
                        f"{fmt_cents(cost)} against a {fmt_cents(verdict.market)} market "
                        f"is short of the {rule.min_discount_pct:g}% the rule asks "
                        f"(needs {fmt_cents(threshold)})",
                        **where,
                    )
            if verdict.quantity < 1 or verdict.quantity > rule.quantity:
                self._fail(
                    "buy_bad_quantity",
                    f"quantity {verdict.quantity} against a rule asking for "
                    f"{rule.quantity}",
                    **where,
                )
            best = decision.best
            if best is not None:
                if best.per_customer_limit is not None and verdict.quantity > best.per_customer_limit:
                    self._fail(
                        "buy_bad_quantity",
                        f"quantity {verdict.quantity} over {best.source}'s stated "
                        f"limit of {best.per_customer_limit}",
                        **where,
                    )
                if not best.purchasable:
                    self._fail(
                        "buy_over_ceiling",
                        "a BUY on a listing that was not purchasable",
                        **where,
                    )
                key = (best.product_id, best.source, best.sku, best.at)
                if key in self.misparse_keys:
                    self._fail(
                        "buy_on_misparse",
                        f"a BUY on the mis-parsed listing at {fmt_cents(best.landed or 0)}, "
                        f"which is a per-pack price on a sealed-box page",
                        **where,
                    )

    def _check_budget(self) -> None:
        """The ledger held: no window overspent, no verdict over the cap."""
        for index, spent in enumerate(self.window_committed):
            if spent > self.budget_total:
                self._fail(
                    "budget_exceeded",
                    f"window {index} committed {fmt_cents(spent)} against a "
                    f"{fmt_cents(self.budget_total)} budget",
                )
        for decision in self.decisions:
            verdict = decision.verdict
            if verdict.action is not Action.BUY or decision.rule is None:
                continue
            cost = verdict.landed if decision.rule.include_shipping else verdict.price
            if cost is None:
                continue
            total = cost * max(1, verdict.quantity)
            if total > decision.cap_before:
                self._fail(
                    "verdict_over_cap",
                    f"{verdict.quantity} x {fmt_cents(cost)} = {fmt_cents(total)} is over "
                    f"the {fmt_cents(decision.cap_before)} one verdict may commit out of "
                    f"{fmt_cents(decision.remaining_before)}",
                    product_id=verdict.product_id,
                    source=verdict.source or "",
                    at=verdict.at,
                )

    def _check_cooldowns(self) -> None:
        """No product was alerted on twice inside its own cooldown."""
        by_product: Dict[str, List[float]] = {}
        for decision in self.decisions:
            if decision.verdict.action is Action.BUY:
                by_product.setdefault(decision.verdict.product_id, []).append(
                    decision.verdict.at
                )
        for product_id, times in sorted(by_product.items()):
            rule = self.rules.get(product_id)
            if rule is None:
                continue
            ordered = sorted(times)
            for earlier, later in zip(ordered, ordered[1:]):
                if later - earlier < rule.cooldown_s - EPS:
                    self._fail(
                        "cooldown_broken",
                        f"two BUY alerts {later - earlier:.0f}s apart, inside the "
                        f"{rule.cooldown_s:.0f}s cooldown",
                        product_id=product_id,
                        at=later,
                    )
                    break

    def _check_reasons(self) -> None:
        """contracts.py: ``reasons`` "always explains the outcome"."""
        for decision in self.decisions:
            if not decision.verdict.reasons:
                # One is enough: a month's worth of "and this one too"
                # would bury every other finding in the report.
                self._fail(
                    "reasons_empty",
                    f"a {decision.verdict.action.value} verdict with no reasons "
                    f"(the first of possibly many)",
                    product_id=decision.verdict.product_id,
                    at=decision.verdict.at,
                )
                return

    def _check_alerts(self) -> None:
        """One alert per BUY, and none for anything else.

        contracts.py gives BUY alone ``should_alert``; a monitor that
        buzzes for "still out of stock" gets muted, after which it cannot
        do its one job.
        """
        buys = [d.verdict for d in self.decisions if d.verdict.action is Action.BUY]
        expected: Dict[str, int] = {}
        for verdict in buys:
            expected[dedupe_key_for(verdict)] = expected.get(dedupe_key_for(verdict), 0) + 1
        seen: Dict[str, int] = {}
        for alert in self.service.sent:
            key = alert["dedupe_key"] or ""
            seen[key] = seen.get(key, 0) + 1
        for key, count in sorted(seen.items()):
            if key not in expected:
                self._fail(
                    "alert_not_buy",
                    f"{count} alert(s) published under {key!r}, which no BUY verdict "
                    f"asked for",
                )
        if len(self.service.sent) != len(buys):
            self._fail(
                "alert_count",
                f"{len(self.service.sent)} alert(s) published for {len(buys)} BUY "
                f"verdict(s) (bridge suppressed {self.bridge.suppressed})",
            )
        for key, count in sorted(expected.items()):
            if seen.get(key, 0) != count:
                self._fail(
                    "alert_count",
                    f"{seen.get(key, 0)} alert(s) under {key!r} for {count} BUY(s)",
                )
                break

    def _check_round_trip(self) -> None:
        """Everything the month produced, through sqlite and back."""
        watch_states = {
            pid: state for pid, state in sorted(self.engine.watch_states.items())
        }
        verdicts = [d.verdict for d in self.decisions]
        snapshot = self.scheduler.snapshot()
        try:
            with PokeStore(":memory:", self.clock) as store:
                store.save_catalog(self.catalog)
                store.save_rule_set(self.rules)
                store.save_watch_states(watch_states)
                store.save_observations(self._all_observations())
                store.append_verdicts(verdicts)
                store.save_poll_snapshot(snapshot)

                checks = (
                    ("products", self.catalog.products(), store.load_products()),
                    ("skus", self.catalog.skus(), store.load_source_skus()),
                    ("rules", self.rules.rules(), store.load_rules()),
                    ("budget", self.rules.budget, store.load_budget()),
                    ("watch_states", watch_states, store.load_watch_states()),
                    (
                        "observations",
                        self._all_observations(),
                        sorted(
                            store.load_observations(),
                            key=lambda o: (o.product_id, o.at, o.source, o.sku),
                        ),
                    ),
                    ("verdicts", verdicts, store.verdicts()),
                    ("poll_state", snapshot, store.load_poll_snapshot()),
                )
        except Exception as exc:  # noqa: BLE001 - a store that raises is a finding
            self._fail("round_trip", f"the store raised {type(exc).__name__}: {exc}")
            return
        for what, before, after in checks:
            difference = first_difference(before, after)
            if difference is not None:
                self._fail("round_trip", f"{what}: {difference}")

    def _all_observations(self) -> List[Observation]:
        rows: List[Observation] = []
        for product_id in self.history.products():
            rows.extend(self.history.for_product(product_id))
        return sorted(rows, key=lambda o: (o.product_id, o.at, o.source, o.sku))

    def _check_scenario(self) -> None:
        """The run really did contain the hazards this gate claims to test.

        Without this a future change that quietly stops the outage, the
        mis-parse or the crash from happening would turn every check above
        into a test of an empty month, and the gate would pass by having
        nothing to look at.
        """
        counts = self._counts()
        wanted = {
            "polls": "no listing was ever fetched",
            "observations": "no observation was ever parsed",
            "not_modified": "no conditional request was answered with a 304",
            "errors": "the scripted outage produced no failed fetch",
            "pauses": "the scripted outage never paused a source",
            "sellouts": "no listing ever went out of stock",
            "restocks": "no listing ever came back in stock",
            "misparse_observations": "the mis-parsed bargain was never observed",
            "misparse_refusals": (
                "the mis-parsed bargain was never the best offer on the table, so "
                "refusing it was never actually tested"
            ),
            "buys": "no BUY was ever issued, so the BUY checks tested nothing",
            "crash_buys": (
                "the scripted price crash produced no BUY, so the engine was never "
                "seen to take a bargain it should take"
            ),
            "cap_refusals": (
                "the per-verdict budget cap never once refused a listing under the "
                "owner's ceiling, so the budget checks tested nothing"
            ),
            "limit_clamps": (
                "no BUY ever had its quantity cut by a listing's stated "
                "per-customer limit, so the clamp -- and the assertion about it "
                "-- was never executed"
            ),
            "bridge_non_buys": (
                "the alert bridge was never offered a non-BUY verdict, so its "
                "'BUY only' filter was never tested and a bridge that buzzed for "
                "WATCH would pass"
            ),
        }
        for key, complaint in wanted.items():
            if counts.get(key, 0) < 1:
                self._fail("scenario_thin", complaint)
        if counts["disallowed_polls"] == 0 and counts["disallowed_listings"] == 0:
            self._fail(
                "scenario_thin",
                "the run had no robots-disallowed source, so 'never polled' was free",
            )

    # -- counters ----------------------------------------------------------

    def _crash_buys(self) -> int:
        """BUYs on the crashed product while it was actually crashed.

        The scripted crash is the one bargain the gate insists the engine
        take: it is under the ceiling, far over the discount the rule asks
        for, in stock at every source, and nowhere near the outlier floor.
        A month in which it produced nothing is a month whose BUY checks
        were never exercised at all.
        """
        return sum(
            1
            for decision in self.decisions
            if decision.verdict.action is Action.BUY
            and decision.verdict.product_id == self.script.crash_product
            and self.script.slot_index(decision.verdict.at) in self.script.crash_slots
        )

    def _counts(self) -> Dict[str, int]:
        stats = self.scheduler.stats(self._now)["totals"]
        by_source = self.host.by_source()
        actions = {action: 0 for action in Action}
        for decision in self.decisions:
            actions[decision.verdict.action] += 1
        return {
            "products": len(self.catalog.products()),
            "listings": len(self.catalog.skus()),
            "days": self.n_days,
            "ticks": int(self.n_days * DAY_S / TICK_S),
            "polls": len(self.host.calls),
            "observations": self.observations,
            "not_modified": int(stats["not_modified"]),
            "errors": int(stats["errors"]),
            "pauses": int(stats["pauses"]),
            "refusals": int(stats["refusals"]),
            "disallowed_listings": len(self.catalog.skus_from(DISALLOWED_SOURCE)),
            "disallowed_polls": len(by_source.get(DISALLOWED_SOURCE, ())),
            "sellouts": self.script.sellouts,
            "restocks": self.script.restocks,
            "misparse_observations": len(self.misparse_keys),
            "misparse_refusals": self.misparse_refusals,
            "crash_buys": self._crash_buys(),
            "verdicts": len(self.decisions),
            "buys": actions[Action.BUY],
            "watches": actions[Action.WATCH],
            "skips": actions[Action.SKIP],
            "no_stock": actions[Action.NO_STOCK],
            "alerts": len(self.service.sent),
            "suppressed": self.bridge.suppressed,
            "commits": self.commits,
            "releases": self.releases,
            "cap_refusals": self.cap_refusals,
            "limit_clamps": self.limit_clamps,
            "bridge_non_buys": self.bridge_non_buys,
            "reserved_cents": self.rules.reserved,
            "committed_cents": self.committed_total,
            "budget_total_cents": self.budget_total,
            "budget_rolls": self.budget_rolls,
            "problems": len(self.problems),
        }


def _newest(observations: Iterable[Observation]) -> List[Observation]:
    """The newest observation per (source, sku), as the engine sees them."""
    best: Dict[Tuple[str, str], Observation] = {}
    for obs in observations:
        key = (obs.source, obs.sku)
        current = best.get(key)
        if current is None or obs.at > current.at:
            best[key] = obs
    return [best[key] for key in sorted(best)]


# --------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------


def run_gate(
    n_products: int = 12,
    n_days: int = 7,
    seed: int = DEFAULT_SEED,
    inject_defect: Optional[str] = None,
) -> GateReport:
    """Run the scripted month and report what it found.

    ``n_products`` products (the shipped catalog first, then placeholders)
    over ``n_days`` days, everything drawn from ``seed``.
    ``inject_defect`` is one of :data:`DEFECTS` and must be reported: the
    gate is shown to fail before it is trusted.

    Raises :class:`GateError` for a run too small to contain the hazards
    it claims to test, or for a defect name it does not know.
    """
    return _Gate(n_products, n_days, seed, inject_defect).run()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m jarvis_poke.validate",
        description="Run a scripted month of watching and hold it to contracts.py.",
    )
    parser.add_argument("--products", type=int, default=12)
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--seed", default=format_seed(DEFAULT_SEED))
    parser.add_argument("--defect", choices=list(DEFECTS), default=None)
    parser.add_argument("--json", action="store_true", help="the report as JSON")
    parser.add_argument(
        "--show-defects",
        action="store_true",
        help="run every defect and exit 1 if any goes unreported",
    )
    args = parser.parse_args(argv)

    try:
        seed = parse_seed(args.seed)
    except Exception:  # noqa: BLE001 - one line, never echo the value
        print("error: --seed must be an integer, decimal or 0x hex", file=sys.stderr)
        return 2

    if args.show_defects:
        failed = 0
        for defect in DEFECTS:
            report = run_gate(args.products, args.days, seed, defect)
            caught = not report.ok
            failed += 0 if caught else 1
            kinds = ", ".join(report.kinds()) if report.problems else "nothing"
            print(f"{'caught ' if caught else 'MISSED '} {defect:<16} {kinds}")
        healthy = run_gate(args.products, args.days, seed)
        print(healthy.summary())
        if not healthy.ok:
            failed += 1
        return 0 if failed == 0 else 1

    report = run_gate(args.products, args.days, seed, args.defect)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
