"""Tests for jarvis_poke price history and the market reference.

Design: jarvis_poke/contracts.py -- the "Observations" section (Observation,
its ``landed`` and ``purchasable`` properties, Stock) and "Market reference
and verdicts" (MarketRef, MARKET_WINDOW_S, MARKET_STALE_AFTER_S).

Two things this file is really about.

**Integer money.**  contracts.py says floats are never used for money, "a
float budget is how you end up buying something for a cent more than the
cap".  So the statistics are checked against medians and percentiles worked
out by hand on small integer samples -- including the even-length case, where
the rule is the lower of the two middles rather than their average -- and a
separate test walks every Cents value that comes back out and asserts it is
exactly ``int``.  ``True`` is an ``int`` to ``isinstance``; these use
``type(x) is int`` so a stray bool cannot pass either.

**The edges, because the edges are where money is lost.**  The window is
half-open and tested one float on each side of both boundaries; staleness is
tested at the exact age threshold and at the exact sample count; the outlier
threshold is tested at the cent that decides it.  An off-by-one in any of
these is not a wrong pixel, it is a wrong purchase.

Nothing here touches a network or a clock: ``now`` is a number every time, and
what randomness the fuzz test needs comes from a ``lucifer_gen.seed`` stream,
so a failure reproduces exactly.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import List, Optional

# Runnable as `pytest tests/test_poke_prices.py` or
# `python3 tests/test_poke_prices.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_poke.contracts import (
    MARKET_STALE_AFTER_S,
    MARKET_WINDOW_S,
    Cents,
    MarketRef,
    Observation,
    Stock,
)
from jarvis_poke.prices import (
    HISTORY_VERSION,
    MIN_SAMPLES_FOR_REFERENCE,
    OUTLIER_MIN_PCT_OF_MEDIAN,
    MemoryHistoryStore,
    PriceHistory,
    PricesError,
    discount_pct,
    is_outlier,
    market_reference,
    price_trend,
)
from lucifer_gen.seed import SeedFields

DAY = 86400.0
NOW = 1_700_000_000.0
PID = "sv08-surging-sparks-etb"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def obs(
    at: float,
    price: Optional[Cents],
    *,
    product_id: str = PID,
    source: str = "examplemart",
    sku: str = "EM-1",
    stock: Stock = Stock.IN_STOCK,
    shipping: Cents = 0,
) -> Observation:
    return Observation(
        product_id=product_id, source=source, sku=sku, at=at,
        stock=stock, price=price, shipping=shipping,
        url=f"https://{source}.example.com/p/{sku}",
    )


def history_of(*observations: Observation) -> PriceHistory:
    history = PriceHistory()
    for observation in observations:
        history.append(observation)
    return history


def priced(prices: List[Cents], *, start: float = NOW - 10 * DAY, step: float = 3600.0):
    """One in-stock observation per price, an hour apart, newest last."""
    return [obs(start + i * step, p) for i, p in enumerate(prices)]


# --------------------------------------------------------------------------
# median and percentile, hand-computed
# --------------------------------------------------------------------------


def test_single_sample_is_its_own_median_p25_and_low():
    ref = market_reference(history_of(*priced([4999])), PID, NOW)
    assert (ref.samples, ref.median, ref.p25, ref.low) == (1, 4999, 4999, 4999)
    # One sample is never enough to price against, whatever it says.
    assert ref.stale is True and ref.usable is False


def test_odd_length_median_is_the_middle_sample():
    # sorted: 3999 4499 4999 5499 5999 -> middle index 2
    ref = market_reference(history_of(*priced([5999, 3999, 4999, 5499, 4499])), PID, NOW)
    assert ref.samples == 5
    assert ref.median == 4999
    # p25: index (5-1)//4 = 1 -> 4499
    assert ref.p25 == 4499
    assert ref.low == 3999


def test_even_length_median_takes_the_lower_middle_not_the_average():
    # sorted: 4000 4500 5500 6000.  Middles are 4500 and 5500; averaging
    # would invent 5000, a price no listing ever asked.  The rule is the
    # lower middle, and it biases the ceiling test the safe way.
    ref = market_reference(history_of(*priced([6000, 4000, 5500, 4500])), PID, NOW)
    assert ref.samples == 4
    assert ref.median == 4500
    assert ref.median != (4500 + 5500) // 2
    # p25: index (4-1)//4 = 0 -> the lowest sample
    assert ref.p25 == 4000
    assert ref.low == 4000


def test_even_length_median_with_an_odd_cent_gap_stays_a_real_sample():
    # sorted: 1001 1002.  An average would be 1001.5 -- a half cent, which
    # contracts.py's integer-cents rule has no way to represent.
    ref = market_reference(history_of(*priced([1002, 1001])), PID, NOW)
    assert ref.median == 1001
    assert type(ref.median) is int


@pytest.mark.parametrize(
    "prices, median, p25",
    [
        ([10, 20, 30], 20, 10),                       # n=3: idx 1, idx 0
        ([10, 20, 30, 40, 50, 60], 30, 20),           # n=6: idx 2, idx 1
        ([10, 20, 30, 40, 50, 60, 70], 40, 20),       # n=7: idx 3, idx 1
        ([10, 20, 30, 40, 50, 60, 70, 80], 40, 20),   # n=8: idx 3, idx 1
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 5, 3),          # n=9: idx 4, idx 2
        ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], 6, 3),  # n=12: idx 5, idx 2
    ],
)
def test_quantiles_against_hand_computed_indices(prices, median, p25):
    ref = market_reference(history_of(*priced(prices)), PID, NOW)
    assert (ref.median, ref.p25, ref.low) == (median, p25, min(prices))


def test_p25_is_never_above_the_median_and_low_is_never_above_p25():
    for n in range(1, 40):
        ref = market_reference(history_of(*priced([100 * i for i in range(1, n + 1)])), PID, NOW)
        assert ref.low <= ref.p25 <= ref.median


def test_reference_uses_landed_price_so_shipping_counts():
    # Same list price, wildly different postage. A shipping-blind reference
    # would call all three 5000 and rate the $15-postage listing a bargain.
    history = history_of(
        obs(NOW - 3 * DAY, 5000, shipping=0, sku="A"),
        obs(NOW - 2 * DAY, 5000, shipping=500, sku="B"),
        obs(NOW - 1 * DAY, 5000, shipping=1500, sku="C"),
    )
    ref = market_reference(history, PID, NOW)
    assert (ref.low, ref.median) == (5000, 5500)


# --------------------------------------------------------------------------
# what does not count as a sample
# --------------------------------------------------------------------------


def test_out_of_stock_and_unpriced_observations_are_excluded():
    history = history_of(
        obs(NOW - 5 * DAY, 4000, sku="A"),
        obs(NOW - 4 * DAY, None, sku="B", stock=Stock.OUT_OF_STOCK),
        obs(NOW - 3 * DAY, 9999, sku="C", stock=Stock.OUT_OF_STOCK),  # stale page price
        obs(NOW - 2 * DAY, None, sku="D", stock=Stock.UNKNOWN),       # parse failed
        obs(NOW - 2 * DAY, 8888, sku="E", stock=Stock.PREORDER),      # not buyable now
        obs(NOW - 1 * DAY, 4200, sku="F", stock=Stock.LIMITED),       # "few left" counts
        obs(NOW - 1 * DAY, None, sku="G", stock=Stock.IN_STOCK),      # price unparsed
        obs(NOW - 0.5 * DAY, 4400, sku="H"),
    )
    ref = market_reference(history, PID, NOW)
    assert ref.samples == 3
    assert (ref.low, ref.median) == (4000, 4200)


def test_a_product_with_no_observations_is_an_empty_stale_reference():
    ref = market_reference(PriceHistory(), "never-seen", NOW)
    assert ref.samples == 0
    assert ref.median is None and ref.p25 is None and ref.low is None
    assert ref.stale is True and ref.usable is False
    assert ref.product_id == "never-seen"


def test_other_products_do_not_leak_into_a_reference():
    history = history_of(
        obs(NOW - DAY, 4000),
        obs(NOW - DAY, 1, product_id="some-other-etb"),
        obs(NOW - DAY, 2, product_id="some-other-etb"),
    )
    ref = market_reference(history, PID, NOW)
    assert (ref.samples, ref.low) == (1, 4000)


# --------------------------------------------------------------------------
# the window, at the edge
# --------------------------------------------------------------------------


WINDOW = 1000.0
WINDOW_NOW = 10_000.0
WINDOW_CUTOFF = WINDOW_NOW - WINDOW      # 9000.0


@pytest.mark.parametrize(
    "at, counted, why",
    [
        (WINDOW_CUTOFF, False, "exactly window_s old: aged out"),
        (WINDOW_CUTOFF + 0.001, True, "a hair inside the old edge"),
        (WINDOW_NOW - 0.001, True, "a hair before now"),
        (WINDOW_NOW, True, "stamped exactly now: the closed end"),
        (WINDOW_NOW + 0.001, False, "the future: a source's clock is not ours"),
    ],
)
def test_window_is_half_open_at_the_old_end_and_closed_at_now(at, counted, why):
    ref = market_reference(history_of(obs(at, 4999)), PID, WINDOW_NOW, window_s=WINDOW)
    assert ref.samples == (1 if counted else 0), why
    assert ref.median == (4999 if counted else None), why


def test_window_edges_together():
    history = history_of(
        obs(WINDOW_CUTOFF, 111, sku="A"),          # out
        obs(WINDOW_CUTOFF + 0.001, 222, sku="B"),  # in
        obs(WINDOW_NOW, 333, sku="C"),             # in
        obs(WINDOW_NOW + 0.001, 444, sku="D"),     # out
    )
    ref = market_reference(history, PID, WINDOW_NOW, window_s=WINDOW)
    assert ref.samples == 2
    assert ref.low == 222          # 111 aged out, or it would be the low
    assert ref.median == 222       # n=2, the lower of the two middles
    # 444 was excluded for being in the future, not for being out of range:
    # advance ``now`` past it and it counts, while 222 ages out behind it.
    later = market_reference(history, PID, WINDOW_NOW + 10.0, window_s=WINDOW)
    assert later.samples == 2
    assert (later.low, later.median) == (333, 333)


def test_the_observation_that_triggers_the_evaluation_is_in_its_own_window():
    # The engine evaluates at the moment it observes.  A window closed at the
    # old end and open at now would throw that sample away.
    latest = obs(NOW, 4999)
    ref = market_reference(history_of(latest), PID, NOW)
    assert ref.samples == 1 and ref.median == 4999


def test_a_shorter_window_excludes_older_samples():
    history = history_of(*[obs(NOW - d * DAY, 1000 + d) for d in range(1, 10)])
    assert market_reference(history, PID, NOW, window_s=30 * DAY).samples == 9
    assert market_reference(history, PID, NOW, window_s=5 * DAY).samples == 4
    assert market_reference(history, PID, NOW, window_s=5 * DAY).low == 1001


@pytest.mark.parametrize("bad", [0.0, -1.0, -MARKET_WINDOW_S])
def test_a_non_positive_window_is_refused_rather_than_silently_empty(bad):
    with pytest.raises(PricesError):
        market_reference(history_of(*priced([100])), PID, NOW, window_s=bad)


# --------------------------------------------------------------------------
# staleness
# --------------------------------------------------------------------------


def test_stale_when_the_newest_sample_is_older_than_the_threshold():
    base = NOW - MARKET_STALE_AFTER_S - DAY
    history = history_of(*[obs(base - i * 3600.0, 4000 + i) for i in range(5)])
    ref = market_reference(history, PID, NOW)
    assert ref.samples == 5
    assert ref.median is not None          # the numbers are still reported
    assert ref.stale is True and ref.usable is False


def test_staleness_age_boundary_is_exact():
    # newest exactly MARKET_STALE_AFTER_S old -> not yet stale; one second
    # more -> stale. Plenty of samples either way, so age is the only cause.
    def build(newest_age: float) -> MarketRef:
        newest = NOW - newest_age
        return market_reference(
            history_of(*[obs(newest - i * 3600.0, 4000 + i) for i in range(4)]), PID, NOW)

    assert build(MARKET_STALE_AFTER_S).stale is False
    assert build(MARKET_STALE_AFTER_S + 1.0).stale is True


def test_staleness_sample_count_boundary_is_exact():
    def build(n: int) -> MarketRef:
        return market_reference(history_of(*priced([4000 + i for i in range(n)])), PID, NOW)

    assert MIN_SAMPLES_FOR_REFERENCE == 3
    assert build(2).stale is True and build(2).usable is False
    assert build(3).stale is False and build(3).usable is True


def test_a_fresh_thick_reference_is_usable():
    ref = market_reference(history_of(*priced([4000, 4500, 5000, 5500])), PID, NOW)
    assert ref.stale is False and ref.usable is True
    assert ref.window_s == MARKET_WINDOW_S


def test_only_in_window_samples_decide_freshness():
    # Old and plentiful inside the window, nothing recent: the count alone
    # must not make it look fresh.
    old = NOW - MARKET_STALE_AFTER_S - 2 * DAY
    history = history_of(*[obs(old - i * 3600.0, 4000) for i in range(10)])
    assert market_reference(history, PID, NOW).stale is True


# --------------------------------------------------------------------------
# discount_pct
# --------------------------------------------------------------------------


def test_discount_is_positive_below_the_reference_and_negative_above():
    assert discount_pct(9000, 10000) == 10.0
    assert discount_pct(11000, 10000) == -10.0
    assert discount_pct(10000, 10000) == 0.0


def test_discount_rounds_to_one_decimal_half_away_from_zero():
    # 1819/6118 = 29.732...%  -> 29.7
    assert discount_pct(4299, 6118) == 29.7
    # exactly 12.25% -> 12.3 (half away from zero, not half to even)
    assert discount_pct(8775, 10000) == 12.3
    # exactly -12.25% -> -12.3
    assert discount_pct(11225, 10000) == -12.3
    # 1/3 off -> 33.3
    assert discount_pct(2000, 3000) == 33.3
    assert round(discount_pct(2000, 3000), 10) == discount_pct(2000, 3000)


def test_discount_result_has_at_most_one_decimal_place():
    for reference in (999, 1234, 5999, 10_000, 33_333):
        for landed in range(1, reference, max(1, reference // 37)):
            value = discount_pct(landed, reference)
            assert value == round(value, 1)


@pytest.mark.parametrize("reference", [0, None, -1, -5000])
def test_discount_guards_division_by_zero_and_junk_references(reference):
    assert discount_pct(4999, reference) == 0.0


def test_discount_with_no_landed_price_claims_nothing():
    assert discount_pct(None, 10000) == 0.0
    assert discount_pct(None, None) == 0.0


def test_discount_of_a_free_listing_is_a_hundred_percent():
    # Not clamped: the caller pairs this with is_outlier, which is what
    # actually refuses it.
    assert discount_pct(0, 10000) == 100.0


# --------------------------------------------------------------------------
# is_outlier
# --------------------------------------------------------------------------


def ref_with_median(median: Cents) -> MarketRef:
    return MarketRef(product_id=PID, samples=9, median=median, p25=median,
                     low=median, window_s=MARKET_WINDOW_S, stale=False)


def test_outlier_boundary_is_exactly_a_quarter_of_the_median():
    assert OUTLIER_MIN_PCT_OF_MEDIAN == 25
    ref = ref_with_median(10_000)
    assert is_outlier(2500, ref) is False   # exactly 25%: a real clearance
    assert is_outlier(2499, ref) is True    # one cent under: a mis-parse
    assert is_outlier(2501, ref) is False


def test_outlier_boundary_when_a_quarter_is_not_a_whole_cent():
    # median 10001 -> threshold 2500.25; integer comparison keeps it exact.
    ref = ref_with_median(10_001)
    assert is_outlier(2500, ref) is True
    assert is_outlier(2501, ref) is False


def test_an_ordinary_deal_is_not_an_outlier():
    ref = ref_with_median(6000)
    assert is_outlier(4200, ref) is False    # 30% off
    assert is_outlier(6000, ref) is False
    assert is_outlier(9000, ref) is False    # a premium, not an outlier


def test_a_free_or_negative_or_missing_price_is_always_an_outlier():
    ref = ref_with_median(6000)
    assert is_outlier(0, ref) is True
    assert is_outlier(-100, ref) is True
    assert is_outlier(None, ref) is True
    # ...and that holds with no reference at all: free is a failed parse.
    empty = MarketRef(product_id=PID, samples=0, median=None, p25=None,
                      low=None, window_s=MARKET_WINDOW_S, stale=True)
    assert is_outlier(0, empty) is True
    assert is_outlier(None, empty) is True


def test_without_a_median_nothing_positive_can_be_called_an_outlier():
    empty = MarketRef(product_id=PID, samples=0, median=None, p25=None,
                      low=None, window_s=MARKET_WINDOW_S, stale=True)
    assert is_outlier(1, empty) is False
    assert empty.usable is False    # the check the caller must make instead
    assert is_outlier(1, ref_with_median(0)) is False


def test_outlier_accepts_no_reference_at_all():
    # jarvis_poke.engine's OutlierCheck protocol is (price, Optional[MarketRef]).
    assert is_outlier(1000, None) is False      # nothing to judge against
    assert is_outlier(0, None) is True          # free is a failed parse regardless
    assert is_outlier(None, None) is True


def test_outlier_refuses_something_that_is_neither_a_marketref_nor_none():
    with pytest.raises(PricesError):
        is_outlier(1000, 4000)      # type: ignore[arg-type]


def test_price_history_satisfies_the_engines_market_history_protocol():
    history = history_of(*priced([4000, 4500, 5000, 5500]))
    assert history.market_ref(PID, NOW) == market_reference(history, PID, NOW)
    assert history.market_ref(PID, NOW, window_s=DAY) == \
        market_reference(history, PID, NOW, window_s=DAY)
    # Never None, even for a product never seen: the empty ref says why.
    empty = history.market_ref("never-seen", NOW)
    assert isinstance(empty, MarketRef) and empty.usable is False


def test_the_mis_parse_that_would_clear_every_rule():
    # A "from $4.99 a pack" price scraped off a $59.99 box page: cheapest
    # ever seen, biggest discount ever seen, and completely fictional.
    history = history_of(*priced([5999, 6199, 5899, 6099]))
    history.append(obs(NOW - 60.0, 499, sku="CB-9", source="cardbarn"))
    ref = market_reference(history, PID, NOW)
    assert ref.low == 499
    assert discount_pct(499, ref.median) > 90.0
    assert is_outlier(ref.low, ref) is True


# --------------------------------------------------------------------------
# price_trend
# --------------------------------------------------------------------------


def test_trend_returns_one_slot_per_bucket_oldest_first():
    now = 6000.0
    history = history_of(
        obs(500.0, 100), obs(1500.0, 200), obs(2500.0, 300),
        obs(3500.0, 400), obs(4500.0, 500), obs(5500.0, 600),
    )
    trend = price_trend(history, PID, now, buckets=6, window_s=6000.0)
    assert [value for _, value in trend] == [100, 200, 300, 400, 500, 600]
    times = [t for t, _ in trend]
    assert times == sorted(times)                  # oldest first
    assert times[-1] == pytest.approx(now)         # the last bucket ends at now


def test_trend_keeps_empty_buckets_as_none_and_never_drops_them():
    now = 6000.0
    # Two clusters months apart in miniature: bucket 0 and bucket 5 only.
    history = history_of(
        obs(400.0, 100), obs(600.0, 200),
        obs(5400.0, 900), obs(5600.0, 1100),
    )
    trend = price_trend(history, PID, now, buckets=6, window_s=6000.0)
    assert len(trend) == 6
    assert [value for _, value in trend] == [100, None, None, None, None, 900]
    # The honest x-axis: the gap occupies four of six slots.
    assert sum(1 for _, v in trend if v is None) == 4


def test_trend_of_a_product_with_nothing_in_the_window_is_all_none():
    trend = price_trend(PriceHistory(), "never-seen", NOW, buckets=4)
    assert len(trend) == 4
    assert all(value is None for _, value in trend)
    times = [t for t, _ in trend]
    assert times == sorted(times)


def test_trend_bucket_edges_are_half_open_the_same_way_as_the_window():
    now = 6000.0
    history = history_of(
        obs(0.0, 1),            # exactly at the window's old edge: excluded
        obs(1000.0, 10),        # exactly a bucket edge: closes bucket 0
        obs(1000.001, 20),      # a hair later: opens bucket 1
        obs(6000.0, 60),        # exactly now: closes the last bucket
    )
    trend = price_trend(history, PID, now, buckets=6, window_s=6000.0)
    assert [value for _, value in trend] == [10, 20, None, None, None, 60]


def test_trend_bucket_value_is_the_same_integer_median_rule():
    now = 2000.0
    history = history_of(
        obs(100.0, 4000), obs(200.0, 4500), obs(300.0, 5500), obs(400.0, 6000),
    )
    trend = price_trend(history, PID, now, buckets=2, window_s=2000.0)
    assert [value for _, value in trend] == [4500, None]   # lower middle, not 5000


def test_trend_excludes_out_of_stock_and_unpriced_like_the_reference():
    now = 2000.0
    history = history_of(
        obs(100.0, 4000),
        obs(200.0, 1, stock=Stock.OUT_OF_STOCK),
        obs(300.0, None, stock=Stock.UNKNOWN),
        obs(1500.0, 9, stock=Stock.PREORDER),
    )
    trend = price_trend(history, PID, now, buckets=2, window_s=2000.0)
    assert [value for _, value in trend] == [4000, None]


def test_trend_counts_landed_price():
    now = 2000.0
    history = history_of(obs(500.0, 4000, shipping=999))
    assert price_trend(history, PID, now, buckets=1, window_s=2000.0)[0][1] == 4999


def test_trend_with_one_bucket_matches_the_reference_median():
    history = history_of(*priced([5999, 4999, 6499, 5499, 5299]))
    ref = market_reference(history, PID, NOW)
    only = price_trend(history, PID, NOW, buckets=1)[0][1]
    assert only == ref.median


def test_buckets_tile_the_window_with_no_gap_and_no_overlap():
    # 59 observations 100s apart across a 6000s window in 6 buckets of 1000s:
    # 10 land in every bucket but the last, which holds the remaining 9.
    now = 6000.0
    prices = list(range(1, 60))
    history = history_of(*[obs(float(i) * 100.0, p) for i, p in enumerate(prices, start=1)])
    assert market_reference(history, PID, now, window_s=6000.0).samples == len(prices)

    trend = price_trend(history, PID, now, buckets=6, window_s=6000.0)
    # Bucket i holds prices (10i+1 .. 10i+10); its median is the lower middle,
    # index (10-1)//2 = 4, i.e. 10i+5.  The last bucket holds 51..59, odd, so
    # its median is the true middle, 55.
    assert [value for _, value in trend] == [5, 15, 25, 35, 45, 55]


@pytest.mark.parametrize("bad", [0, -1, -6])
def test_a_nonsensical_bucket_count_is_refused(bad):
    with pytest.raises(PricesError):
        price_trend(history_of(*priced([100])), PID, NOW, buckets=bad)


def test_trend_refuses_a_non_positive_window():
    with pytest.raises(PricesError):
        price_trend(history_of(*priced([100])), PID, NOW, window_s=0.0)


# --------------------------------------------------------------------------
# PriceHistory itself
# --------------------------------------------------------------------------


def test_for_product_is_sorted_oldest_first_however_it_was_appended():
    history = history_of(obs(300.0, 3), obs(100.0, 1), obs(200.0, 2))
    assert [o.at for o in history.for_product(PID)] == [100.0, 200.0, 300.0]


def test_for_product_since_is_an_inclusive_lower_bound():
    history = history_of(obs(100.0, 1), obs(200.0, 2), obs(300.0, 3))
    assert [o.at for o in history.for_product(PID, since=200.0)] == [200.0, 300.0]
    assert [o.at for o in history.for_product(PID, since=200.001)] == [300.0]
    assert history.for_product(PID, since=301.0) == []


def test_for_product_returns_a_copy_the_caller_cannot_corrupt():
    history = history_of(obs(100.0, 1), obs(200.0, 2))
    rows = history.for_product(PID)
    rows.append(obs(300.0, 3))
    rows.clear()
    assert len(history.for_product(PID)) == 2


def test_for_product_of_an_unknown_product_is_empty_not_an_error():
    assert PriceHistory().for_product("never-seen") == []


def test_latest_is_the_newest_and_can_be_filtered_by_source():
    history = history_of(
        obs(100.0, 1, source="examplemart", sku="EM-1"),
        obs(300.0, 3, source="cardbarn", sku="CB-9"),
        obs(200.0, 2, source="examplemart", sku="EM-1"),
    )
    assert history.latest(PID).at == 300.0
    assert history.latest(PID, source="examplemart").at == 200.0
    assert history.latest(PID, source="cardbarn").at == 300.0
    assert history.latest(PID, source="nowhere") is None
    assert history.latest("never-seen") is None


def test_latest_includes_out_of_stock_because_that_is_news_too():
    history = history_of(
        obs(100.0, 4000),
        obs(200.0, None, stock=Stock.OUT_OF_STOCK),
    )
    assert history.latest(PID).stock is Stock.OUT_OF_STOCK


def test_prune_drops_older_and_keeps_the_cutoff_itself():
    history = history_of(obs(100.0, 1), obs(200.0, 2), obs(300.0, 3))
    assert history.prune(200.0) == 1
    assert [o.at for o in history.for_product(PID)] == [200.0, 300.0]
    # The cutoff is the same boundary for_product(since=) uses, so pruning at
    # t never removes a row a since=t read would have returned.
    assert history.prune(200.0) == 0


def test_prune_forgets_a_product_entirely_when_nothing_is_left():
    history = history_of(obs(100.0, 1), obs(100.0, 2, product_id="other"))
    assert history.prune(500.0) == 2
    assert history.products() == []
    assert len(history) == 0


def test_re_appending_one_moment_replaces_rather_than_double_counting():
    history = PriceHistory()
    history.append(obs(100.0, 4000))
    history.append(obs(100.0, 4000))            # replayed snapshot
    assert len(history) == 1
    history.append(obs(100.0, 4200))            # re-parsed body, same instant
    assert len(history) == 1
    assert history.for_product(PID)[0].price == 4200
    # A different listing at the same instant is a different observation.
    history.append(obs(100.0, 4300, source="cardbarn", sku="CB-9"))
    assert len(history) == 2


def test_append_refuses_anything_that_is_not_an_observation():
    with pytest.raises(PricesError):
        PriceHistory().append({"product_id": PID, "price": 4999})  # type: ignore[arg-type]


def test_a_store_round_trip_preserves_every_field():
    store = MemoryHistoryStore()
    original = PriceHistory(store)
    original.append(Observation(
        product_id=PID, source="cardbarn", sku="CB-9", at=1234.5,
        stock=Stock.LIMITED, price=4999, shipping=599, per_customer_limit=2,
        url="https://cardbarn.example.com/i/CB-9", note="few left",
    ))
    original.append(obs(2345.6, None, stock=Stock.OUT_OF_STOCK))

    restored = PriceHistory(store)
    assert restored.for_product(PID) == original.for_product(PID)
    assert market_reference(restored, PID, 3000.0, window_s=3000.0) == \
        market_reference(original, PID, 3000.0, window_s=3000.0)


def test_extend_saves_once_not_once_per_row():
    store = MemoryHistoryStore()
    history = PriceHistory(store)
    history.extend(priced([100, 200, 300, 400]))
    assert store.saves == 1
    assert len(history) == 4


def test_a_snapshot_from_a_future_version_is_refused_not_guessed_at():
    with pytest.raises(PricesError):
        PriceHistory(MemoryHistoryStore({"version": HISTORY_VERSION + 1, "observations": []}))


def test_a_snapshot_holding_float_money_is_refused():
    # A float here means someone stored dollars; contracts.py says money is
    # integer cents, and 49.99 dollars silently becomes 49 cents otherwise.
    bad = {"version": HISTORY_VERSION, "observations": [{
        "product_id": PID, "source": "examplemart", "sku": "EM-1", "at": 100.0,
        "stock": "in_stock", "price": 49.99, "shipping": 0,
    }]}
    with pytest.raises(PricesError):
        PriceHistory(MemoryHistoryStore(bad))


def test_a_corrupt_snapshot_row_is_refused():
    for row in (
        {"product_id": PID, "source": "e", "sku": "s", "at": 1.0, "stock": "nonsense"},
        {"product_id": PID, "source": "e", "sku": "s", "stock": "in_stock"},
        "not a row",
    ):
        with pytest.raises(PricesError):
            PriceHistory(MemoryHistoryStore({"version": HISTORY_VERSION, "observations": [row]}))


def test_a_store_without_load_and_save_is_refused_at_construction():
    with pytest.raises(PricesError):
        PriceHistory(object())      # type: ignore[arg-type]


def test_market_reference_needs_a_price_history():
    with pytest.raises(PricesError):
        market_reference([obs(100.0, 1)], PID, NOW)   # type: ignore[arg-type]
    with pytest.raises(PricesError):
        price_trend([obs(100.0, 1)], PID, NOW)        # type: ignore[arg-type]


# --------------------------------------------------------------------------
# integer money, everywhere, no exceptions
# --------------------------------------------------------------------------


def seeded_history(seed: int, n: int = 120) -> PriceHistory:
    """A pseudo-random history from a lucifer_gen stream -- reproducible, so
    a failure here can be replayed exactly."""
    stream = SeedFields.parse(seed).stream(f"poke.prices:{PID}")
    history = PriceHistory()
    rows = []
    for i in range(n):
        at = NOW - MARKET_WINDOW_S + stream.randint(1, int(MARKET_WINDOW_S))
        roll = stream.randint(0, 9)
        if roll == 0:
            rows.append(obs(at, None, stock=Stock.OUT_OF_STOCK, sku=f"S{i}"))
        elif roll == 1:
            rows.append(obs(at, None, stock=Stock.UNKNOWN, sku=f"S{i}"))
        else:
            rows.append(obs(at, stream.randint(1999, 9999),
                            shipping=stream.choice([0, 0, 499, 999]), sku=f"S{i}"))
    history.extend(rows)
    return history


def test_every_cents_value_returned_is_exactly_an_int():
    history = seeded_history(0x5EED_0000_0000_0001)
    ref = market_reference(history, PID, NOW)
    values = [ref.median, ref.p25, ref.low]
    values += [v for _, v in price_trend(history, PID, NOW, buckets=8)]
    assert any(v is not None for v in values)
    for value in values:
        if value is None:
            continue
        # `isinstance(True, int)` is True, so check the exact type: a bool or
        # a float that slipped through would pass a looser assertion.
        assert type(value) is int, f"{value!r} is {type(value).__name__}, not int"
        assert not isinstance(value, bool)


def test_every_statistic_is_a_price_some_listing_actually_asked():
    history = seeded_history(0x5EED_0000_0000_0002)
    landed = {o.landed for o in history.for_product(PID) if o.purchasable}
    ref = market_reference(history, PID, NOW)
    assert {ref.median, ref.p25, ref.low} <= landed
    for _, value in price_trend(history, PID, NOW, buckets=8):
        assert value is None or value in landed


def test_trend_bucket_timestamps_are_evenly_spaced_and_end_at_now():
    trend = price_trend(seeded_history(0x5EED_3), PID, NOW, buckets=7, window_s=7000.0)
    times = [t for t, _ in trend]
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(g == pytest.approx(1000.0) for g in gaps)
    assert times[-1] == pytest.approx(NOW)
    assert times[0] == pytest.approx(NOW - 6000.0)


# --------------------------------------------------------------------------
# determinism
# --------------------------------------------------------------------------


def test_the_same_history_and_now_always_give_the_same_answer():
    for seed in (0x5EED_A, 0x5EED_B, 0x5EED_C):
        a, b = seeded_history(seed), seeded_history(seed)
        assert market_reference(a, PID, NOW) == market_reference(b, PID, NOW)
        assert price_trend(a, PID, NOW, buckets=9) == price_trend(b, PID, NOW, buckets=9)
    # ...and re-asking one history twice cannot drift.
    once = seeded_history(0x5EED_A)
    assert market_reference(once, PID, NOW) == market_reference(once, PID, NOW)


def test_insertion_order_does_not_change_any_statistic():
    rows = list(seeded_history(0x5EED_D).for_product(PID))
    forward, backward = PriceHistory(), PriceHistory()
    forward.extend(rows)
    backward.extend(reversed(rows))
    assert market_reference(forward, PID, NOW) == market_reference(backward, PID, NOW)
    assert price_trend(forward, PID, NOW) == price_trend(backward, PID, NOW)
    assert forward.snapshot() == backward.snapshot()


def test_different_seeds_really_do_give_different_histories():
    # Otherwise the determinism tests above would pass trivially.
    a = market_reference(seeded_history(0x5EED_A), PID, NOW)
    b = market_reference(seeded_history(0x5EED_B), PID, NOW)
    assert (a.median, a.p25, a.low) != (b.median, b.p25, b.low)


# --------------------------------------------------------------------------
# the boundary contracts.py draws, asserted against the source
# --------------------------------------------------------------------------


def test_prices_module_opens_no_socket_and_reads_no_clock():
    """contracts.py: "The package makes no network calls itself", and the
    house rule that the clock is injected and randomness comes from a
    lucifer_gen seed stream."""
    banned_modules = {
        "urllib", "http", "socket", "ssl", "requests", "httpx", "asyncio",
        "subprocess", "random", "time", "datetime", "secrets",
        # statistics.median averages the two middles and returns a float --
        # exactly the thing this module exists to avoid.
        "statistics",
    }
    banned_calls = {("time", "time"), ("time", "monotonic"), ("random", "random"),
                    ("random", "randint"), ("random", "choice"), ("random", "shuffle")}
    source = (ROOT / "jarvis_poke" / "prices.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in banned_modules, \
                    f"prices.py imports {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in banned_modules, \
                f"prices.py imports from {node.module}"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = node.func.value
            if isinstance(owner, ast.Name):
                assert (owner.id, node.func.attr) not in banned_calls, \
                    f"prices.py calls {owner.id}.{node.func.attr}()"


def test_prices_module_never_divides_money():
    """No ``/`` may touch a cents value: true division makes a float, and
    contracts.py forbids floats for money.  The three divisions this module
    does perform are on percentages, times and the final tenths-to-float
    conversion, all named here so a new one has to be argued for."""
    source = (ROOT / "jarvis_poke" / "prices.py").read_text(encoding="utf-8")
    allowed = {
        "tenths / 10.0",             # integer tenths -> the promised float
        "window_s / buckets",        # seconds, not money
        "(o.at - low) / span",       # seconds, not money
        "(NOW - end) / DAY",         # the smoke run printing an age in days
    }
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            found.add(ast.unparse(node))
    assert found == allowed, f"unexpected true division in prices.py: {found - allowed}"


def test_no_real_retailer_names_appear():
    """The scope boundary: shipped data and examples use obvious placeholders
    on example.com, so nothing here reads as a scraping recipe."""
    source = (ROOT / "jarvis_poke" / "prices.py").read_text(encoding="utf-8").lower()
    for name in ("amazon", "walmart", "target.com", "costco", "bestbuy",
                 "gamestop", "ebay", "pokemoncenter"):
        assert name not in source
    for marker in ("https://", "http://"):
        start = 0
        while (start := source.find(marker, start)) != -1:
            tail = source[start:start + 80]
            assert "example.com" in tail, f"non-placeholder url in prices.py: {tail!r}"
            start += 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
