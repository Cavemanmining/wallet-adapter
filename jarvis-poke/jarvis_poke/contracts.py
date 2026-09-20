"""Shared types for the Jarvis Pokemon buying assistant.

What this is
------------
A watchlist, a stock and price monitor, a deal scorer, and a decision engine
that tells the owner *when to act and why*, delivering that through
``jarvis_alerts`` so it reaches a phone with the app closed.

What this is not
----------------
It does not check out. There is no cart automation, no CAPTCHA handling, no
proxy rotation, no account cycling. The engine's terminal output is a
:class:`Verdict` plus a deep link; a person completes the purchase. That
keeps the tool inside every retailer's terms of service, and it costs almost
nothing in practice: the slow part of catching a restock is *finding out*,
which is what this automates.

Politeness is a design constraint, not an afterthought
------------------------------------------------------
Sources declare a minimum poll interval and are rate limited per host.
:class:`FetchPolicy` carries robots.txt permission, a conditional-request
validator, and a backoff that widens on errors. A source that has been told
to slow down, or that robots.txt disallows, is not polled. The package makes
no network calls itself: the app injects a fetcher, exactly as the alert
transports take an injected sender.

Money is integer cents
----------------------
Every amount in this package is an ``int`` number of cents in
:data:`CURRENCY`. Floats are never used for money; a float budget is how you
end up buying something for a cent more than the cap.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

CURRENCY = "USD"

Cents = int


def to_cents(text: str) -> Cents:
    """Parse "$189.99", "189.99", "1,299.00" into integer cents.

    Raises ValueError rather than guessing, because a mis-parsed price is a
    mis-spent budget.
    """
    cleaned = text.strip().replace("$", "").replace(",", "").replace(" ", "")
    if not cleaned:
        raise ValueError("empty price")
    neg = cleaned.startswith("-")
    if neg:
        cleaned = cleaned[1:]
    if not all(c.isdigit() or c == "." for c in cleaned):
        raise ValueError(f"not a price: {text!r}")
    if cleaned.count(".") > 1:
        raise ValueError(f"not a price: {text!r}")
    if "." in cleaned:
        whole, frac = cleaned.split(".")
        frac = (frac + "00")[:2]
    else:
        whole, frac = cleaned, "00"
    value = int(whole or "0") * 100 + int(frac)
    return -value if neg else value


def fmt_cents(cents: Cents) -> str:
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}${cents // 100:,}.{cents % 100:02d}"


# --------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------


class ProductKind(enum.Enum):
    """Sealed product shapes. Singles are deliberately out of scope: their
    condition and grading make automated comparison unreliable."""

    BOOSTER_PACK = "booster_pack"
    BOOSTER_BOX = "booster_box"
    ELITE_TRAINER_BOX = "elite_trainer_box"
    COLLECTION_BOX = "collection_box"
    TIN = "tin"
    BUNDLE = "bundle"
    PREMIUM_COLLECTION = "premium_collection"


@dataclass(frozen=True)
class Product:
    """One canonical product, independent of who sells it."""

    id: str                    # stable slug, e.g. "sv08-surging-sparks-etb"
    name: str
    set_code: str              # e.g. "SV08"
    kind: ProductKind
    msrp: Optional[Cents] = None
    released: Optional[str] = None   # ISO date; None if unannounced
    upc: Optional[str] = None


@dataclass(frozen=True)
class SourceSku:
    """How one retailer refers to a product."""

    source: str                # source id, e.g. "examplemart"
    product_id: str
    sku: str
    url: str                   # the page a person would open to buy it


# --------------------------------------------------------------------------
# Observations
# --------------------------------------------------------------------------


class Stock(enum.Enum):
    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    PREORDER = "preorder"
    LIMITED = "limited"        # in stock but the page says few remain
    UNKNOWN = "unknown"        # fetch failed or the page could not be parsed


@dataclass(frozen=True)
class Observation:
    """One look at one listing at one moment. Immutable; the history is the
    sequence of these."""

    product_id: str
    source: str
    sku: str
    at: float                  # unix seconds
    stock: Stock
    price: Optional[Cents]     # None when out of stock or unparseable
    shipping: Cents = 0
    per_customer_limit: Optional[int] = None
    url: str = ""
    note: str = ""

    @property
    def landed(self) -> Optional[Cents]:
        """Price the owner actually pays, shipping included."""
        return None if self.price is None else self.price + self.shipping

    @property
    def purchasable(self) -> bool:
        return self.stock in (Stock.IN_STOCK, Stock.LIMITED) and self.price is not None


# --------------------------------------------------------------------------
# Fetching, kept polite by construction
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FetchPolicy:
    """What a source is allowed to do. The scheduler honours all of it."""

    source: str
    min_interval_s: float = 300.0     # never poll a host faster than this
    robots_allows: bool = True        # set from the app's robots.txt check
    max_errors_before_pause: int = 5
    pause_s: float = 3600.0
    user_agent: str = "JarvisPokeWatch/1 (+personal use; contact in app settings)"

    def __post_init__(self) -> None:
        # ADDITIVE, and loudly: NaN is refused here and in ``pause_s``
        # below.  Every comparison against NaN is False, so a NaN interval
        # sailed through ``< 30.0``, made ``effective_due_at`` NaN, made
        # ``now < due_at`` False, and left the source permanently due --
        # one fetch per tick, no error, no warning.  A number that cannot
        # be compared cannot be a floor, so it is not a number we accept.
        # Nothing that was valid before is refused now.
        if not isinstance(self.min_interval_s, (int, float)) or isinstance(
            self.min_interval_s, bool
        ):
            raise ValueError("min_interval_s must be a number of seconds")
        if not math.isfinite(float(self.min_interval_s)):
            raise ValueError(
                "min_interval_s must be a finite number of seconds: a NaN or "
                "infinite interval compares False against every floor, which is "
                "an ungated poll rather than a polite one"
            )
        if self.min_interval_s < 30.0:
            # A floor, not a preference. Nothing here needs sub-30s polling,
            # and faster is how a hobby tool becomes someone's incident.
            raise ValueError("min_interval_s floor is 30s")
        if not isinstance(self.pause_s, (int, float)) or isinstance(self.pause_s, bool):
            raise ValueError("pause_s must be a number of seconds")
        if not math.isfinite(float(self.pause_s)) or self.pause_s <= 0.0:
            raise ValueError(
                "pause_s must be a finite, positive number of seconds: a NaN "
                "pause is a pause that is never installed, which removes the "
                "stop-after-repeated-errors rail with no signal"
            )


@dataclass(frozen=True)
class FetchResult:
    """What an injected fetcher hands back. The package never opens a socket."""

    ok: bool
    status: int = 0
    body: str = ""
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    not_modified: bool = False      # a 304; reuse the previous observation
    retry_after_s: Optional[float] = None
    reason: str = ""


class Fetcher(Protocol):
    """Supplied by the app. Given a url and conditional headers, return a
    FetchResult. Implementations are expected to respect ``policy``."""

    def __call__(
        self, url: str, headers: Dict[str, str], policy: FetchPolicy
    ) -> FetchResult: ...


class Parser(Protocol):
    """Turns a fetched body into an Observation. One per source, supplied by
    the app so this package carries no retailer-specific scraping."""

    def __call__(self, sku: SourceSku, body: str, at: float) -> Observation: ...


# --------------------------------------------------------------------------
# Rules: what the owner is willing to buy
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    """The owner's standing instruction for one product.

    ``max_price`` is a hard ceiling on the landed price. ``min_discount_pct``
    is measured against the market reference, not MSRP, because MSRP is
    fiction for anything in demand.
    """

    product_id: str
    max_price: Cents
    quantity: int = 1
    min_discount_pct: float = 0.0
    allowed_sources: Tuple[str, ...] = ()    # empty means any known source
    include_shipping: bool = True
    cooldown_s: float = 3600.0               # do not re-alert inside this
    enabled: bool = True

    def __post_init__(self) -> None:
        if self.max_price <= 0:
            raise ValueError("max_price must be positive")
        if self.quantity < 1:
            raise ValueError("quantity must be at least 1")
        if not 0.0 <= self.min_discount_pct < 100.0:
            raise ValueError("min_discount_pct must be in [0, 100)")


@dataclass(frozen=True)
class Budget:
    """A ceiling across everything, so a good week cannot empty an account."""

    total: Cents
    spent: Cents = 0
    window_s: float = 30 * 86400.0

    @property
    def remaining(self) -> Cents:
        return max(0, self.total - self.spent)


# --------------------------------------------------------------------------
# Market reference and verdicts
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MarketRef:
    """What this product has actually been selling for lately."""

    product_id: str
    samples: int
    median: Optional[Cents]
    p25: Optional[Cents]
    low: Optional[Cents]
    window_s: float
    stale: bool = False        # too few or too old to be trusted

    @property
    def usable(self) -> bool:
        return self.median is not None and self.samples >= 3 and not self.stale


class Action(enum.Enum):
    BUY = "buy"            # meets every rule; alert loudly with a deep link
    WATCH = "watch"        # in stock but fails a rule; no alert unless asked
    SKIP = "skip"          # rule disabled, budget gone, cooldown, or unusable
    NO_STOCK = "no_stock"


@dataclass(frozen=True)
class Verdict:
    """The engine's answer, with the reasons spelled out.

    ``reasons`` always explains the outcome, including for BUY, so the alert
    can say why and the log can be audited later.
    """

    product_id: str
    action: Action
    at: float
    source: Optional[str] = None
    sku: Optional[str] = None
    price: Optional[Cents] = None
    landed: Optional[Cents] = None
    market: Optional[Cents] = None
    discount_pct: Optional[float] = None
    quantity: int = 0
    url: str = ""
    reasons: Tuple[str, ...] = ()

    @property
    def should_alert(self) -> bool:
        return self.action is Action.BUY


@dataclass
class WatchState:
    """Mutable per-product bookkeeping the engine needs between runs."""

    product_id: str
    last_alert_at: float = 0.0
    last_action: Optional[Action] = None
    alerts_sent: int = 0
    last_seen_in_stock: float = 0.0


#: Cap on how much of the remaining budget one verdict may commit, so a single
#: mis-parsed price cannot propose spending everything.
MAX_BUDGET_FRACTION_PER_VERDICT = 0.5

#: A market reference older than this is stale regardless of sample count.
MARKET_STALE_AFTER_S = 14 * 86400.0

#: Default trailing window for computing a market reference.
MARKET_WINDOW_S = 30 * 86400.0


__all__ = [
    "CURRENCY", "Cents", "to_cents", "fmt_cents", "ProductKind", "Product",
    "SourceSku", "Stock", "Observation", "FetchPolicy", "FetchResult",
    "Fetcher", "Parser", "Rule", "Budget", "MarketRef", "Action", "Verdict",
    "WatchState", "MAX_BUDGET_FRACTION_PER_VERDICT", "MARKET_STALE_AFTER_S",
    "MARKET_WINDOW_S",
]
