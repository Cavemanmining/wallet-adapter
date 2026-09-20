"""Price history and the market reference that deal scoring is measured against.

Implements the "Observations" and "Market reference and verdicts" sections of
:mod:`jarvis_poke.contracts`: it stores
:class:`~jarvis_poke.contracts.Observation` records and turns them into a
:class:`~jarvis_poke.contracts.MarketRef`, honouring
:data:`~jarvis_poke.contracts.MARKET_WINDOW_S` and
:data:`~jarvis_poke.contracts.MARKET_STALE_AFTER_S`.  It exists because
``Rule.min_discount_pct`` is, in contracts.py's words, "measured against the
market reference, not MSRP, because MSRP is fiction for anything in demand" --
so something has to say what the market actually is.

This module monitors and describes.  It buys nothing, opens no socket and
knows no retailer: it only ever sees ``Observation`` objects that an injected
:class:`~jarvis_poke.contracts.Parser` produced elsewhere.

Money is integer cents, end to end
----------------------------------
Every statistic here is computed with integer arithmetic on
``Observation.landed`` (contracts.py: "Price the owner actually pays, shipping
included").  No ``float`` ever touches a price, not even transiently: the
median of an even-length sample is the *lower* of the two middles rather than
their average, and the 25th percentile is a real observed sample rather than
an interpolation between two.  Two reasons, in order of importance:

1. The reference is used as a **ceiling test** -- ``Rule.max_price`` and
   ``min_discount_pct`` are compared against it.  An averaged median is a
   price nobody ever charged; deciding to spend real money against a number
   that never existed is exactly the class of mistake contracts.py's "Money
   is integer cents" section is about.  Biasing low makes "this is cheap"
   harder to claim, which is the safe direction for a tool that tells
   someone to go and spend.
2. ``(a + b) // 2`` on cents silently drops a half-cent and ``(a + b) / 2``
   introduces a float.  Picking an actual sample avoids inventing either.

The same "pick a real sample, round down" rule gives every quantile here:
:func:`_quantile_index` is ``(n - 1) * num // den``, so the median is
``(n - 1) // 2`` and p25 is ``(n - 1) // 4``.

Time, and the shape of the window
---------------------------------
Nothing in this module reads a clock.  ``now`` is a parameter of every
time-dependent function, and :class:`PriceHistory` has no notion of "current"
at all -- so a caller passes its injected clock in, and a test passes a
number.  There is no randomness here of any kind: the same history and the
same ``now`` always give byte-identical results.

The trailing window is **half-open at the old end and closed at ``now``**::

    (now - window_s, now]

A sample exactly ``window_s`` old has aged out; a sample stamped exactly
``now`` is in.  The alternative convention, ``[now - window_s, now)``, would
drop the observation that *triggered* the evaluation -- the engine calls
:func:`market_reference` with ``now`` set to the moment it just observed --
and an off-by-one that silently discards the newest price is the worst one to
have here.  Observations stamped after ``now`` are excluded too, which quietly
protects against a source whose page states a timestamp in the future.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from jarvis_poke.contracts import (
    MARKET_STALE_AFTER_S,
    MARKET_WINDOW_S,
    Cents,
    MarketRef,
    Observation,
    Stock,
    fmt_cents,
)

__all__ = [
    "HISTORY_VERSION",
    "OUTLIER_MIN_PCT_OF_MEDIAN",
    "MIN_SAMPLES_FOR_REFERENCE",
    "HistoryStore",
    "MemoryHistoryStore",
    "PricesError",
    "PriceHistory",
    "market_reference",
    "discount_pct",
    "is_outlier",
    "price_trend",
]

#: Snapshot format written to an injected store.  Bumped if the row shape
#: changes; an unknown version is refused rather than guessed at.
HISTORY_VERSION = 1

#: A landed price under this percentage of the median is treated as a
#: mis-parse or a scam listing, not as a deal.  See :func:`is_outlier`.
OUTLIER_MIN_PCT_OF_MEDIAN = 25

#: ``MarketRef.usable`` requires at least this many samples (contracts.py:
#: ``samples >= 3``).  Named here so :func:`market_reference` marks a
#: thin reference stale by the same number the property tests against.
MIN_SAMPLES_FOR_REFERENCE = 3


class PricesError(ValueError):
    """Something this module refuses to do: a corrupt snapshot, a
    non-positive window, a nonsensical bucket count.  A wiring or data
    mistake, never "the network had a bad day"."""


# --------------------------------------------------------------------------
# integer statistics
# --------------------------------------------------------------------------


def _quantile_index(n: int, num: int, den: int) -> int:
    """Index into a sorted list of ``n`` samples for quantile ``num/den``.

    ``(n - 1) * num // den``.  Pure integer, always a real index, and always
    biased to the lower of two candidates -- the module docstring explains why
    that bias is the safe one for a ceiling test.  ``num/den = 1/2`` gives the
    median (``(n - 1) // 2``: the middle for odd ``n``, the lower of the two
    middles for even ``n``); ``1/4`` gives p25.
    """
    if n < 1:
        raise PricesError("no samples to take a quantile of")
    return (n - 1) * num // den


def _median_cents(values: Sequence[Cents]) -> Optional[Cents]:
    """Median of already-collected landed prices, or None for no samples.

    Returns one of the input values unchanged, so the result is an ``int``
    number of cents that some listing actually asked for.
    """
    if not values:
        return None
    ordered = sorted(values)
    return ordered[_quantile_index(len(ordered), 1, 2)]


def _round_div(numerator: int, denominator: int) -> int:
    """Integer division rounded half away from zero.

    Used to reach one decimal place of a percentage without ever building a
    float from cents: the caller scales by 1000 and divides the result by 10.
    ``round()`` on a float would be half-to-even *and* would inherit binary
    representation error (``0.1 + 0.2``), which is not something to drag into
    a number that decides whether to spend money.
    """
    if denominator == 0:
        raise PricesError("round division by zero")
    negative = (numerator < 0) != (denominator < 0)
    num, den = abs(numerator), abs(denominator)
    magnitude = (num * 2 + den) // (2 * den)
    return -magnitude if negative else magnitude


# --------------------------------------------------------------------------
# the history
# --------------------------------------------------------------------------


class HistoryStore(Protocol):
    """Where the observation history is kept between runs.

    Same two-method shape as ``jarvis_poke.sources.PollStore``, and the
    snapshot is plain JSON types for the same reason: the app decides what a
    file, a row in a database or nothing at all means.
    """

    def load(self) -> Optional[Dict[str, Any]]: ...

    def save(self, snapshot: Dict[str, Any]) -> None: ...


class MemoryHistoryStore:
    """A :class:`HistoryStore` that keeps the snapshot in memory.

    Deliberately does *not* copy through ``json`` on save: a history can hold
    tens of thousands of rows and this store is used in the smoke run and in
    tests, where a serialise-round-trip per append turns a linear test into a
    quadratic one.  :meth:`PriceHistory.snapshot` already builds fresh plain
    dicts, so a caller still cannot reach live state through it.
    """

    def __init__(self, snapshot: Optional[Dict[str, Any]] = None) -> None:
        self.snapshot: Optional[Dict[str, Any]] = snapshot
        self.saves = 0

    def load(self) -> Optional[Dict[str, Any]]:
        return self.snapshot

    def save(self, snapshot: Dict[str, Any]) -> None:
        self.snapshot = snapshot
        self.saves += 1


class PriceHistory:
    """Every observation this tool has made, indexed by product.

    contracts.py calls an :class:`~jarvis_poke.contracts.Observation` "one look
    at one listing at one moment", immutable, and says "the history is the
    sequence of these".  This is that sequence, plus the two lookups the rest
    of the package needs and a prune so it does not grow without bound.

    ``store`` is optional and injected -- any object with ``load()`` and
    ``save(snapshot)``.  When present it is loaded at construction and written
    after every mutation, so a restart does not forget what a product has been
    selling for.  Saving rewrites the whole snapshot, which is linear in the
    history; that is fine at the scale this tool works at (a handful of
    listings polled no faster than every few minutes) and keeps the store
    contract to two methods.

    There is no clock here.  :meth:`prune` takes the cutoff, and every
    statistic in this module takes ``now``.
    """

    def __init__(self, store: Optional[HistoryStore] = None) -> None:
        self._store = _checked_store(store)
        # product_id -> observations, kept per product because every read is
        # per product.  Insertion order is not meaningful: every read sorts
        # by (at, source, sku), a total order, so nothing downstream can
        # depend on the order a caller happened to append in.
        self._by_product: Dict[str, List[Observation]] = {}
        if self._store is not None:
            snapshot = self._store.load()
            if snapshot is not None:
                self._restore(snapshot)

    # -- mutation ---------------------------------------------------------

    def append(self, observation: Observation) -> None:
        """Record one observation.

        Re-appending the same listing at the same instant -- same
        ``(source, sku, at)`` -- replaces the earlier record in place rather
        than adding a second one.  contracts.py says an Observation is one
        look at one moment, so two rows for one moment is always a bug (a
        replayed snapshot, a re-parsed body); keeping both would double-count
        that price in the median and quietly move the reference.  The later
        record wins: a re-parse is the better information, and refusing it
        would mean a fixed parser could never correct a bad row.
        """
        if not isinstance(observation, Observation):
            raise PricesError(f"not an Observation: {observation!r}")
        rows = self._by_product.setdefault(observation.product_id, [])
        key = (observation.source, observation.sku, observation.at)
        for i, existing in enumerate(rows):
            if (existing.source, existing.sku, existing.at) == key:
                rows[i] = observation
                break
        else:
            rows.append(observation)
        self._persist()

    def extend(self, observations) -> None:
        """Append many, persisting once at the end rather than per row."""
        store, self._store = self._store, None
        try:
            for observation in observations:
                self.append(observation)
        finally:
            self._store = store
        self._persist()

    def prune(self, before: float) -> int:
        """Drop every observation older than ``before``; return how many went.

        The cutoff is exclusive of itself: an observation stamped exactly
        ``before`` is kept, matching :meth:`for_product`'s ``since`` so that
        ``prune(t)`` never removes a row ``for_product(product_id, since=t)``
        would have returned.
        """
        removed = 0
        for product_id, rows in list(self._by_product.items()):
            kept = [o for o in rows if o.at >= before]
            removed += len(rows) - len(kept)
            if kept:
                self._by_product[product_id] = kept
            else:
                del self._by_product[product_id]
        if removed:
            self._persist()
        return removed

    # -- reads ------------------------------------------------------------

    def for_product(
        self, product_id: str, since: Optional[float] = None
    ) -> List[Observation]:
        """Every observation of one product, oldest first.

        ``since`` is an inclusive lower bound -- "everything from this moment
        onwards" -- because a caller naming a timestamp means to include it.
        That is *not* the convention :func:`market_reference` uses for its
        trailing window, which measures an age and therefore drops a sample
        exactly ``window_s`` old; both are stated where they apply, and
        :func:`market_reference` does its own filtering rather than routing
        through ``since`` so the two can never be confused for one another.

        The returned list is a fresh list, so a caller cannot mutate the
        history by holding on to it.  Observations sharing a timestamp are
        ordered by source then sku -- a total order, since contracts.py makes
        ``(source, sku, at)`` the identity of a look at a listing -- so the
        result does not depend on the order things were appended in, and a
        history restored from a snapshot reads back identically.
        """
        rows = self._by_product.get(product_id)
        if not rows:
            return []
        if since is not None:
            rows = [o for o in rows if o.at >= since]
        return sorted(rows, key=lambda o: (o.at, o.source, o.sku))

    def latest(
        self, product_id: str, source: Optional[str] = None
    ) -> Optional[Observation]:
        """The newest observation of a product, optionally from one source.

        Newest by ``at``; a tie is broken by source then sku, the same total
        order :meth:`for_product` uses, so the answer does not depend on the
        order rows were appended in.  Returns None when nothing matches --
        including when the product is unknown, which is not an error: a
        watchlist entry is allowed to have never been seen.
        """
        rows = self._by_product.get(product_id)
        if not rows:
            return None
        if source is not None:
            rows = [o for o in rows if o.source == source]
        if not rows:
            return None
        return max(rows, key=lambda o: (o.at, o.source, o.sku))

    def market_ref(
        self, product_id: str, now: float, window_s: float = MARKET_WINDOW_S
    ) -> MarketRef:
        """:func:`market_reference` as a method, so a bare ``PriceHistory``
        satisfies ``jarvis_poke.engine``'s ``MarketHistory`` protocol
        (``market_ref(product_id, now)``) with no adapter in between.

        It always returns a :class:`~jarvis_poke.contracts.MarketRef`, never
        None: a reference with no samples is still an answer, and it carries
        ``stale`` and ``usable`` so the caller can see *why* there is nothing
        to price against instead of guessing at a None.
        """
        return market_reference(self, product_id, now, window_s=window_s)

    def products(self) -> List[str]:
        """Product ids with at least one observation, sorted for determinism."""
        return sorted(self._by_product)

    def __len__(self) -> int:
        return sum(len(rows) for rows in self._by_product.values())

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (f"PriceHistory(products={len(self._by_product)}, "
                f"observations={len(self)})")

    # -- persistence ------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """A plain-JSON snapshot: version plus rows, oldest first per product.

        Cents stay ints; ``Stock`` becomes its ``value`` string.  Rows are
        emitted in a canonical order -- product id, then ``at``, then source
        and sku, which is a total order because those four are the identity
        of a row -- so two histories holding the same observations produce
        byte-identical snapshots however they were built, and re-saving an
        unchanged history is an empty diff.
        """
        rows: List[Dict[str, Any]] = []
        for product_id in sorted(self._by_product):
            ordered = sorted(self._by_product[product_id],
                             key=lambda o: (o.at, o.source, o.sku))
            for o in ordered:
                rows.append({
                    "product_id": o.product_id,
                    "source": o.source,
                    "sku": o.sku,
                    "at": o.at,
                    "stock": o.stock.value,
                    "price": o.price,
                    "shipping": o.shipping,
                    "per_customer_limit": o.per_customer_limit,
                    "url": o.url,
                    "note": o.note,
                })
        return {"version": HISTORY_VERSION, "observations": rows}

    def _persist(self) -> None:
        if self._store is not None:
            self._store.save(self.snapshot())

    def _restore(self, snapshot: Mapping[str, Any]) -> None:
        if not isinstance(snapshot, Mapping):
            raise PricesError(f"snapshot must be a mapping, got {type(snapshot).__name__}")
        version = snapshot.get("version")
        if version != HISTORY_VERSION:
            # Refused, not guessed at: a row shape we do not know could put a
            # wrong price into a median, and a wrong median spends money.
            raise PricesError(
                f"history snapshot version {version!r}, expected {HISTORY_VERSION}")
        rows = snapshot.get("observations", [])
        if not isinstance(rows, (list, tuple)):
            raise PricesError("snapshot 'observations' must be a list")
        for row in rows:
            self.append(_observation_from_row(row))


def _checked_store(store: Optional[HistoryStore]) -> Optional[HistoryStore]:
    if store is None:
        return None
    if not callable(getattr(store, "load", None)) or not callable(getattr(store, "save", None)):
        raise PricesError(
            "store must have load() and save(snapshot) methods "
            f"(see HistoryStore); got {type(store).__name__}")
    return store


def _observation_from_row(row: Mapping[str, Any]) -> Observation:
    if not isinstance(row, Mapping):
        raise PricesError(f"observation row must be a mapping, got {row!r}")
    try:
        stock = Stock(row["stock"])
    except (KeyError, ValueError) as exc:
        raise PricesError(f"bad stock in snapshot row: {row!r}") from exc
    for money in ("price", "shipping"):
        value = row.get(money)
        if isinstance(value, bool) or (value is not None and not isinstance(value, int)):
            # A float here means someone stored dollars. contracts.py:
            # "Floats are never used for money".
            raise PricesError(f"{money} must be integer cents, got {value!r}")
    try:
        return Observation(
            product_id=str(row["product_id"]),
            source=str(row["source"]),
            sku=str(row["sku"]),
            at=float(row["at"]),
            stock=stock,
            price=row.get("price"),
            shipping=row.get("shipping", 0) or 0,
            per_customer_limit=row.get("per_customer_limit"),
            url=str(row.get("url", "")),
            note=str(row.get("note", "")),
        )
    except KeyError as exc:
        raise PricesError(f"snapshot row missing {exc.args[0]!r}: {row!r}") from exc


# --------------------------------------------------------------------------
# the market reference
# --------------------------------------------------------------------------


def market_reference(
    history: PriceHistory,
    product_id: str,
    now: float,
    window_s: float = MARKET_WINDOW_S,
) -> MarketRef:
    """What this product has actually been selling for lately.

    Builds the :class:`~jarvis_poke.contracts.MarketRef` that
    ``Rule.min_discount_pct`` is measured against.  Only purchasable
    observations inside ``(now - window_s, now]`` count, and each contributes
    its **landed** price -- ``price + shipping``, contracts.py's "price the
    owner actually pays".  Comparing a shipping-inclusive candidate against a
    shipping-exclusive reference is how a $5 item with $15 postage reads as a
    bargain.

    Out-of-stock and unparseable rows are excluded by
    ``Observation.purchasable`` -- contracts.py's own definition, in stock or
    limited with a price that parsed -- rather than being treated as zero or
    carried forward: "what it sold for while it was unavailable" is not a
    price, and a row whose parse failed is the one thing least safe to
    average in.

    ``median``, ``p25`` and ``low`` are all real observed samples, computed by
    integer indexing into the sorted list (see the module docstring on why an
    even-length median takes the lower of the two middles rather than their
    average).  ``low`` is the cheapest landed price in the window, which is a
    floor sighting, not a target: it is frequently the one listing that turned
    out to be a mis-parse, which is what :func:`is_outlier` is for.

    ``stale`` is set when the newest sample is more than
    :data:`~jarvis_poke.contracts.MARKET_STALE_AFTER_S` old, or when there are
    fewer than :data:`MIN_SAMPLES_FOR_REFERENCE` samples.  Both feed
    ``MarketRef.usable``; a stale reference is still returned, with its numbers
    filled in, so the page can show what little is known and say it is thin
    rather than showing nothing.
    """
    if not isinstance(history, PriceHistory):
        raise PricesError(f"history must be a PriceHistory, got {type(history).__name__}")
    if window_s <= 0:
        raise PricesError(f"window_s must be positive, got {window_s!r}")

    cutoff = now - window_s
    # One pass, one filter: (when, landed) for every row that counts.  The
    # ``landed is not None`` arm is redundant with ``purchasable`` -- which
    # already requires a price -- and is kept so a future change to either
    # property cannot put a None into the arithmetic.
    in_window = [
        (o.at, o.landed)
        for o in history.for_product(product_id)
        if cutoff < o.at <= now and o.purchasable and o.landed is not None
    ]
    samples = len(in_window)

    if samples == 0:
        return MarketRef(
            product_id=product_id, samples=0, median=None, p25=None, low=None,
            window_s=window_s, stale=True,
        )

    ordered = sorted(landed for _, landed in in_window)
    median = ordered[_quantile_index(samples, 1, 2)]
    p25 = ordered[_quantile_index(samples, 1, 4)]
    low = ordered[0]

    newest = max(at for at, _ in in_window)
    stale = (now - newest) > MARKET_STALE_AFTER_S or samples < MIN_SAMPLES_FOR_REFERENCE

    return MarketRef(
        product_id=product_id, samples=samples, median=median, p25=p25, low=low,
        window_s=window_s, stale=stale,
    )


def discount_pct(landed: Optional[Cents], reference: Optional[Cents]) -> float:
    """How far below ``reference`` a landed price is, in percent, one decimal.

    Positive means cheaper than the reference, which is the direction
    ``Rule.min_discount_pct`` is written in ("at least 10% off"); negative
    means the listing is asking a premium, which is worth showing rather than
    clamping to zero, because "12% over market" is the sentence that stops
    someone buying.

    Computed from integer cents throughout: the percentage is scaled by 1000
    and rounded half-away-from-zero by :func:`_round_div` before a single
    division by ten produces the float the signature promises.  Nothing is
    ever averaged or accumulated in floating point.

    Returns ``0.0`` -- "no discount is being claimed" -- when there is nothing
    to measure against: a missing landed price, a missing reference, or a
    reference of zero or less.  That is the guard against dividing by zero,
    and it is the conservative answer: 0.0 fails every ``min_discount_pct``
    above zero.  A caller must still check ``MarketRef.usable`` before
    believing a *non*-zero result; this function cannot tell a thin reference
    from a good one.
    """
    if landed is None or reference is None or reference <= 0:
        return 0.0
    tenths = _round_div((reference - landed) * 1000, reference)
    return tenths / 10.0


def is_outlier(landed: Optional[Cents], ref: Optional[MarketRef]) -> bool:
    """True when a price is too good to be true, so the engine must not act.

    A landed price under :data:`OUTLIER_MIN_PCT_OF_MEDIAN` percent of the
    median is almost never a deal.  In practice it is a mis-parse -- a monthly
    instalment, a "from $4.99" per-pack price on a box page, a stripped
    thousands separator -- or a listing that will never ship.  Treating it as
    the best deal ever seen is the single most expensive mistake this tool
    could make, because it is exactly the case that clears every rule at once:
    under the ceiling, over the discount threshold, loudest possible alert.

    **The engine must refuse to act on a True here.**  Not downgrade the
    score, not alert with a warning: no ``Action.BUY``.  The right outcome is
    ``Action.WATCH`` or ``Action.SKIP`` with the reason recorded in
    ``Verdict.reasons``, so a person can look at the listing and decide,
    which is the whole division of labour contracts.py sets out -- this tool
    decides *when to look*, a person decides whether to spend.

    The test is ``landed * 100 < median * 25``, integer both sides, so a price
    exactly at a quarter of the median is *not* an outlier; the boundary
    belongs to the deal, because a genuine 75%-off clearance does happen and
    the strict comparison is what lets it through.

    A landed price of zero or less is always an outlier: free is not a price,
    it is a parse that failed, and that holds with no reference at all.  A
    ``None`` landed price counts too -- there is nothing there to buy.

    With no reference, or one that has no median, nothing else can be judged
    and the answer is False -- ``ref`` is ``Optional`` for exactly that case,
    which is the shape ``jarvis_poke.engine``'s ``OutlierCheck`` protocol
    calls this with.  False there means "this gate has nothing to say", not
    "this price is fine": ``MarketRef.usable`` is the check that covers a
    missing or thin reference, and it is the caller's to make.  Anything that
    is neither a ``MarketRef`` nor ``None`` is a wiring mistake and raises.
    """
    if landed is None or landed <= 0:
        return True
    if ref is None:
        return False
    if not isinstance(ref, MarketRef):
        raise PricesError(f"ref must be a MarketRef or None, got {type(ref).__name__}")
    if ref.median is None or ref.median <= 0:
        return False
    return landed * 100 < ref.median * OUTLIER_MIN_PCT_OF_MEDIAN


def price_trend(
    history: PriceHistory,
    product_id: str,
    now: float,
    buckets: int = 6,
    window_s: float = MARKET_WINDOW_S,
) -> List[Tuple[float, Optional[Cents]]]:
    """Bucketed medians across the window, oldest first, for the page sparkline.

    Returns exactly ``buckets`` pairs of ``(bucket_end_time, median_or_None)``.
    The window is split into equal spans, each half-open in the same direction
    as :func:`market_reference`'s -- ``(start, end]`` -- so the last bucket
    ends exactly at ``now`` and the buckets tile the window without overlap or
    gap.  Each pair is stamped with its bucket's **end**, which is a moment
    inside that bucket, so the newest point sits exactly at ``now`` rather
    than one span short of it.

    **Empty buckets carry None and are never dropped.**  A sparkline drawn
    from only the buckets that have data silently rescales its x-axis: a
    product seen twice in January and twice in June draws as four evenly
    spaced points, which reads as a steady trickle of stock instead of two
    brief windows months apart -- and "how often does this come back in
    stock" is the question the chart is there to answer.  Rendering decides
    what a gap looks like; this function's job is to report that there is one.

    Each bucket's value is the median of the landed prices of the purchasable
    observations in it, by the same integer rule
    :func:`market_reference` uses, so a bucket median is always a price some
    listing actually asked for.
    """
    if not isinstance(history, PriceHistory):
        raise PricesError(f"history must be a PriceHistory, got {type(history).__name__}")
    if window_s <= 0:
        raise PricesError(f"window_s must be positive, got {window_s!r}")
    if buckets < 1:
        raise PricesError(f"buckets must be at least 1, got {buckets!r}")

    low = now - window_s
    span = window_s / buckets
    collected: List[List[Cents]] = [[] for _ in range(buckets)]

    for o in history.for_product(product_id):
        if not (low < o.at <= now) or not o.purchasable:
            continue
        landed = o.landed
        if landed is None:  # unreachable while purchasable implies a price
            continue
        # (start, end] per bucket: ceil puts an exact boundary in the bucket
        # it closes, not the one it opens.  Clamped because float spans do
        # not land exactly on their edges.
        index = math.ceil((o.at - low) / span) - 1
        if index < 0:
            index = 0
        elif index >= buckets:
            index = buckets - 1
        collected[index].append(landed)

    return [
        (low + span * (i + 1), _median_cents(values))
        for i, values in enumerate(collected)
    ]


# --------------------------------------------------------------------------
# smoke run: a made-up history, no network, no clock, no randomness
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover - a smoke run, not a CLI
    DAY = 86400.0
    NOW = 1_700_000_000.0
    PRODUCT = "sv08-surging-sparks-etb"

    store = MemoryHistoryStore()
    history = PriceHistory(store)

    # A fortnight of looks at two placeholder shops (contracts.py's example
    # source ids), drifting down, with a gap and a mis-parse in it.
    rows: List[Observation] = []
    for day in range(14):
        at = NOW - (13 - day) * DAY
        if 4 <= day <= 8:
            continue  # nothing in stock anywhere that week
        rows.append(Observation(
            product_id=PRODUCT, source="examplemart", sku="EM-1", at=at,
            stock=Stock.IN_STOCK, price=5999 - day * 40, shipping=599,
            url="https://examplemart.example.com/p/EM-1",
        ))
        rows.append(Observation(
            product_id=PRODUCT, source="cardbarn", sku="CB-9", at=at + 3600.0,
            stock=Stock.OUT_OF_STOCK if day % 3 else Stock.LIMITED,
            price=None if day % 3 else 5749 - day * 35, shipping=0,
            url="https://cardbarn.example.com/i/CB-9",
        ))
    rows.append(Observation(  # a "from $4.99 a pack" mis-parse on a box page
        product_id=PRODUCT, source="cardbarn", sku="CB-9", at=NOW - 60.0,
        stock=Stock.IN_STOCK, price=499, shipping=0,
    ))
    history.extend(rows)

    ref = market_reference(history, PRODUCT, NOW)
    print(repr(history), f"store saves={store.saves}")
    print(f"  samples={ref.samples} median={fmt_cents(ref.median)} "
          f"p25={fmt_cents(ref.p25)} low={fmt_cents(ref.low)} "
          f"stale={ref.stale} usable={ref.usable}")

    for candidate in (4299, 5499, 6999, 499):
        flagged = is_outlier(candidate, ref)
        print(f"  {fmt_cents(candidate):>10}  "
              f"{discount_pct(candidate, ref.median):+6.1f}% vs median  "
              f"{'OUTLIER -- engine must not act' if flagged else 'plausible'}")

    print("  trend (oldest first, 6 buckets over 30d):")
    for end, value in price_trend(history, PRODUCT, NOW):
        age = (NOW - end) / DAY
        print(f"    -{age:5.1f}d  {fmt_cents(value) if value is not None else '(no data)'}")

    stat_types = {type(v) for v in (ref.median, ref.p25, ref.low)}
    stat_types |= {type(v) for _, v in price_trend(history, PRODUCT, NOW) if v is not None}
    print(f"  every cents value is an int: {stat_types == {int}}")
