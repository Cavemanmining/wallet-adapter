"""Command line front end: ``python3 -m jarvis_poke.cli --db PATH <command>``.

Design: :mod:`jarvis_poke.contracts`.  This is the shell view of the whole
package -- the watchlist is one sqlite file
(:class:`jarvis_poke.store.PokeStore`), polling goes through
:class:`jarvis_poke.sources.PollScheduler` so it stays polite, and the
terminal output of ``decide`` is a :class:`~jarvis_poke.contracts.Verdict`
and a link.  **Nothing here buys anything.**  There is no checkout
command and there will not be one: contracts.py, "What this is not".

    init         seed the store from the bundled catalog and sources
    watch        --product ID --max-price 189.99 [--qty N] [--min-discount PCT]
                 [--sources a,b] [--cooldown SECONDS]   (dollars in, cents stored)
    unwatch      --product ID
    budget       --total 500.00 [--window-days N]
    bought       --product ID [--amount 189.99]   record a purchase
    release      --product ID | --all             give a held alert's money back
    list         the watchlist with best price, market and last verdict
    poll         --once   one polling round
    decide       evaluate now and print verdicts with reasons
    sources      per-source state, next due and pauses
    pause        --source ID [--off] [--for SECONDS]
    serve-state  the page's state JSON on stdout
    gate         --products N --days D --seed S [--defect NAME]

The money the tool holds, and who moves it
-----------------------------------------
A BUY is an alert plus a deep link, and the cents behind it are
*reserved* the moment the alert is produced -- kept in the same file as
the rules, so they survive the process that made them.  Every run of
``decide`` is a fresh process and the link is still on somebody's phone;
a ledger that forgot on every invocation would leave the owner's monthly
ceiling decorative, with only the per-verdict cap doing any work.

Because this tool does not check out, it cannot know what the owner did
about a link.  So they say: ``bought`` turns a reservation into spend
(it is the only thing in the package that moves ``Budget.spent``), and
``release`` hands it back.  ``decide`` prints both invitations under
every BUY, and ``budget`` reports all three figures -- spent, held, free.

``--db PATH`` is accepted before or after the command and defaults to
``poke.sqlite3`` in the current directory (``gate`` does not use it).
Exit status: 0 on success, 1 on any error -- one line on stderr, never a
traceback -- and 2 for a usage error.  **No message ever contains a
URL's query string**: a listing URL can carry an affiliate tag or a
session token, and an error is the one place a URL gets copied into a
chat window.  :func:`scrub` cuts every URL at its ``?``.

Where the prices come from
--------------------------
This package opens no sockets, so ``poll`` needs an injected
:class:`~jarvis_poke.contracts.Fetcher` and
:class:`~jarvis_poke.contracts.Parser`, exactly as ``jarvis_alerts``
needs an injected sender.  An app installs real ones at import time and
then hands over::

    from jarvis_poke import cli
    cli.install_fetcher(my_fetcher)      # (url, headers, policy) -> FetchResult
    cli.install_parser(my_parser)        # (sku, body, at) -> Observation
    raise SystemExit(cli.main())

With nothing installed, ``poll`` falls back to :func:`stub_fetcher` --
**a stub, not a source**.  It invents a deterministic price walk from
``--seed`` and never touches a network.  Invented numbers in a real
watchlist would be worse than no numbers at all, so every stub
observation is stored with ``note=`` :data:`STUB_NOTE`, the store is
flagged once with the ``stub_data`` meta key, and from then on ``list``,
``decide`` and ``serve-state`` all say so out loud -- ``serve-state``
sets the page's own ``demo`` flag, which is the banner the page already
shows for made-up data.

Time and randomness
-------------------
The CLI is the composition root, so it is the one place allowed to hand
``time.time`` to anything; every module it composes takes that clock as
an argument and none of them reads a clock of its own (contracts.py).
The stub's price walk is a ``lucifer_gen.seed.Stream`` from ``--seed``,
so the same seed gives the same sample data.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, TextIO, Tuple

from lucifer_gen.seed import SeedFields, parse_seed

from jarvis_poke.catalog import Catalog, CatalogError
from jarvis_poke.contracts import (
    MARKET_WINDOW_S,
    Action,
    Budget,
    Cents,
    FetchPolicy,
    FetchResult,
    Observation,
    Product,
    Rule,
    SourceSku,
    Stock,
    Verdict,
    fmt_cents,
    to_cents,
)
from jarvis_poke.engine import DecisionEngine
from jarvis_poke.prices import PriceHistory, price_trend
from jarvis_poke.rules import RuleError, RuleSet, budget_cap
from jarvis_poke.sources import PollScheduler, load_policies
from jarvis_poke.store import PokeStore

__all__ = [
    "DEFAULT_DB",
    "EXIT_FAIL",
    "EXIT_OK",
    "EXIT_USAGE",
    "FRESH_S",
    "RECENT_VERDICTS",
    "STUB_DATA_KEY",
    "TREND_BUCKETS",
    "STUB_NOTE",
    "STUB_ROUND_KEY",
    "CliError",
    "build_parser",
    "install_fetcher",
    "install_parser",
    "main",
    "page_state",
    "scrub",
    "stub_fetcher",
    "stub_parser",
]

DEFAULT_DB = "poke.sqlite3"

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

#: A listing not seen for this long is not offered to the engine as a
#: current price.  The same window the gate uses.
FRESH_S = 24 * 3600.0

#: How many verdicts the page's "recent decisions" strip carries.
RECENT_VERDICTS = 20

#: Buckets in the page's sparkline (``state.watch[].trend``).
#: ``jarvis_poke/web/README.md``: "Roughly 20-30 buckets reads well".
#: :func:`jarvis_poke.prices.price_trend` defaults to 6, which is a chart
#: the page was never designed against -- its embedded demo dataset carries
#: 28 -- so the number the page wants is named here rather than inherited
#: from whatever the market lane happens to default to.
TREND_BUCKETS = 28

#: Stamped on every observation the stub invented, so a price that was
#: never fetched can always be told from one that was.
STUB_NOTE = "stub: invented sample price, nothing was fetched"

#: ``meta`` keys: that this store has stub data in it at all, and which
#: round the stub's walk is up to.
STUB_DATA_KEY = "stub_data"
STUB_ROUND_KEY = "stub_round"

#: Default seed for the stub's sample walk.
DEFAULT_SEED = 0


class CliError(ValueError):
    """Something the CLI refuses to do.  Printed as one line, exit 1."""


# --------------------------------------------------------------------------
# Never print a query string
# --------------------------------------------------------------------------


def scrub(text: str) -> str:
    """Cut every URL in ``text`` at its query string.

    A product URL is shown to the owner all day long, but its query can
    carry an affiliate tag, a session id or a one-time token, and an
    error message is the thing people paste into a chat window.  The path
    is kept -- it is what identifies the listing -- and everything from
    ``?`` onwards becomes ``?...``.

    Done with a pattern rather than by splitting on spaces because the
    URL is usually quoted, bracketed or followed by a full stop by the
    time it reaches a message, and a check that only recognised a bare
    token would let exactly those cases through.
    """
    return _URL_QUERY.sub(r"\1?...", str(text))


#: A URL up to its ``?``, then its query.  Quotes and whitespace end the
#: query, so the punctuation around a quoted URL survives untouched.
_URL_QUERY = re.compile(r"(https?://[^\s?'\"<>]*)\?[^\s'\"<>]*", re.IGNORECASE)


# --------------------------------------------------------------------------
# The injected fetcher and parser, and the stub that stands in for them
# --------------------------------------------------------------------------

_FETCHER: Optional[Callable[..., FetchResult]] = None
_PARSER: Optional[Callable[..., Observation]] = None


def install_fetcher(fetcher: Optional[Callable[..., FetchResult]]) -> None:
    """Install the app's fetcher (``None`` restores the stub)."""
    global _FETCHER
    if fetcher is not None and not callable(fetcher):
        raise CliError("fetcher must be callable: (url, headers, policy) -> FetchResult")
    _FETCHER = fetcher


def install_parser(parser: Optional[Callable[..., Observation]]) -> None:
    """Install the app's parser (``None`` restores the stub)."""
    global _PARSER
    if parser is not None and not callable(parser):
        raise CliError("parser must be callable: (sku, body, at) -> Observation")
    _PARSER = parser


def _stub_price(
    fields: SeedFields, url: str, round_no: int
) -> Tuple[Optional[Cents], Cents, str, Optional[int]]:
    """A sample price for one listing in one round.

    The level comes from the *last path segment* of the URL and the
    premium from its *host*, so every source quotes the same product
    within a few percent of every other -- which is what a real market
    looks like, and what keeps the sample data from tripping the
    engine's own mis-parse gate on every single round.
    """
    path = url.split("?", 1)[0].split("#", 1)[0]
    slug = path.rstrip("/").rsplit("/", 1)[-1] or path
    host = path.split("//", 1)[-1].split("/", 1)[0]
    level = fields.stream(f"poke.stub.level:{slug}").randint(1500, 18000)
    premium = 95 + fields.stream(f"poke.stub.host:{host}").randint(0, 12)
    walk = fields.stream(f"poke.stub.round:{slug}:{host}#{round_no}")
    price = max(100, level * premium // 100 + walk.randint(-level // 25, level // 25))
    if walk.chance(0.2):
        return None, 0, Stock.OUT_OF_STOCK.value, None
    stock = Stock.LIMITED.value if walk.chance(0.15) else Stock.IN_STOCK.value
    shipping = 0 if price >= 5000 else 599
    limit = 2 if walk.chance(0.25) else None
    return price, shipping, stock, limit


def stub_fetcher(
    url: str, headers: Dict[str, str], policy: FetchPolicy, *, seed: int = DEFAULT_SEED,
    round_no: int = 0,
) -> FetchResult:
    """A stand-in for a real fetcher.  It invents; it does not fetch.

    No socket is opened here and none can be: the body is built from a
    seeded stream.  It answers ``If-None-Match`` with a 304 while the
    round has not moved on, so the conditional-request path is exercised
    rather than stubbed out, and every body it serves is marked so the
    parser can stamp :data:`STUB_NOTE` on the observation.
    """
    fields = SeedFields.parse(seed)
    etag = f'W/"stub-{round_no}"'
    if headers.get("If-None-Match") == etag:
        return FetchResult(ok=True, status=304, not_modified=True, etag=etag)
    price, shipping, stock, limit = _stub_price(fields, url, round_no)
    body = json.dumps(
        {
            "stub": True,
            "stock": stock,
            "price_cents": price,
            "shipping_cents": shipping,
            "limit": limit,
        },
        sort_keys=True,
    )
    return FetchResult(ok=True, status=200, body=body, etag=etag)


def stub_parser(sku: SourceSku, body: str, at: float) -> Observation:
    """Read a :func:`stub_fetcher` body.  Marks the observation as stub."""
    obj = json.loads(body)
    price = obj.get("price_cents")
    limit = obj.get("limit")
    return Observation(
        product_id=sku.product_id,
        source=sku.source,
        sku=sku.sku,
        at=at,
        stock=Stock(obj.get("stock", "unknown")),
        price=None if price is None else int(price),
        shipping=int(obj.get("shipping_cents") or 0),
        per_customer_limit=None if limit is None else int(limit),
        url=sku.url,
        note=STUB_NOTE if obj.get("stub") else "",
    )


# --------------------------------------------------------------------------
# Loading the pieces out of the store
# --------------------------------------------------------------------------


def _clock() -> float:
    """The one place this package is allowed to read a wall clock."""
    return time.time()


def _open(args: argparse.Namespace) -> PokeStore:
    return PokeStore(args.db, _clock)


def _catalog(store: PokeStore) -> Catalog:
    products = store.load_products()
    if not products:
        raise CliError("this store has no catalog yet; run 'init' first")
    return Catalog(products, store.load_source_skus(), origin=store.path)


def _rule_set(store: PokeStore) -> RuleSet:
    return store.load_rule_set()


def _history(store: PokeStore, now: float, window_s: float = MARKET_WINDOW_S) -> PriceHistory:
    history = PriceHistory()
    history.extend(store.load_observations(since=now - window_s))
    return history


def _scheduler(store: PokeStore, catalog: Catalog) -> PollScheduler:
    """A scheduler wired to the store, so a pause survives the process.

    Its policies come from the shipped ``sources.json``; an app that has
    run its own robots.txt check installs the result with
    :meth:`PollScheduler.set_policy` before polling.
    """
    return PollScheduler(catalog, load_policies(), _clock, store.poll_store())


def _product_map(catalog: Catalog) -> Dict[str, Product]:
    return {product.id: product for product in catalog.products()}


def _latest_by_source(
    observations: Sequence[Observation],
) -> List[Observation]:
    """The newest observation per (source, sku), newest-first by price."""
    best: Dict[Tuple[str, str], Observation] = {}
    for obs in observations:
        key = (obs.source, obs.sku)
        current = best.get(key)
        if current is None or obs.at > current.at:
            best[key] = obs
    return [best[key] for key in sorted(best)]


def _fresh(observations: Sequence[Observation], now: float) -> List[Observation]:
    return [obs for obs in observations if obs.at >= now - FRESH_S]


def _cheapest(observations: Sequence[Observation]) -> Optional[Observation]:
    purchasable = [obs for obs in observations if obs.purchasable]
    if not purchasable:
        return None
    return min(purchasable, key=lambda obs: (obs.landed or 0, obs.source, obs.sku))


def _stub_warning(store: PokeStore) -> str:
    if store.meta(STUB_DATA_KEY) != "1":
        return ""
    return (
        "note: this store holds stub observations -- sample prices this tool "
        "invented because no fetcher is installed. They are not real prices."
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_init(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Seed the store from the bundled catalog and sources file."""
    try:
        catalog = Catalog.load()
    except CatalogError as exc:
        raise CliError(f"bundled catalog: {exc}") from None
    policies = load_policies()
    with _open(args) as store:
        products, skus = store.save_catalog(catalog)
        if store.load_budget() is None:
            store.save_budget(Budget(total=0))
    out.write(
        f"initialised {args.db}: {products} products, {skus} listings, "
        f"{len(policies)} sources\n"
    )
    for source in sorted(policies):
        policy = policies[source]
        allowed = "robots allows" if policy.robots_allows else "robots DISALLOWS, never polled"
        out.write(
            f"  {source:<12} every {policy.min_interval_s:.0f}s at most, {allowed}\n"
        )
    out.write(
        "  no fetcher is installed, so nothing here has been fetched; every "
        "retailer above is a placeholder\n"
    )
    return EXIT_OK


def cmd_watch(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Add or replace one standing instruction."""
    max_price = _dollars(args.max_price, "--max-price")
    if args.qty < 1:
        raise CliError("--qty must be at least 1")
    if not 0.0 <= args.min_discount < 100.0:
        raise CliError("--min-discount must be at least 0 and under 100")
    sources = tuple(s.strip() for s in (args.sources or "").split(",") if s.strip())
    with _open(args) as store:
        catalog = _catalog(store)
        product = catalog.find(args.product)
        if product is None:
            raise CliError(
                f"no product {args.product!r} in this store; 'list' shows what is here"
            )
        known = set(catalog.sources())
        for source in sources:
            if source not in known:
                raise CliError(f"unknown source {source!r}; known: {', '.join(sorted(known))}")
        rules = _rule_set(store)
        rule = Rule(
            product_id=product.id,
            max_price=max_price,
            quantity=args.qty,
            min_discount_pct=float(args.min_discount),
            allowed_sources=sources,
            cooldown_s=float(args.cooldown),
        )
        rules.add(rule, replace_existing=True)
        store.save_rule_set(rules)
    out.write(
        f"watching {product.id}: up to {fmt_cents(max_price)} landed, "
        f"{args.qty} of them"
        + (f", {args.min_discount:g}% off or better" if args.min_discount else "")
        + (f", only {', '.join(sources)}" if sources else "")
        + "\n"
    )
    out.write("  this tool will tell you when to buy it. It will not buy it.\n")
    return EXIT_OK


def cmd_unwatch(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    with _open(args) as store:
        rules = _rule_set(store)
        if rules.get(args.product) is None:
            raise CliError(f"not watching {args.product!r}")
        rules.remove(args.product)
        store.save_rule_set(rules)
    out.write(f"no longer watching {args.product}\n")
    return EXIT_OK


def cmd_budget(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    total = _dollars(args.total, "--total")
    with _open(args) as store:
        rules = _rule_set(store)
        current = rules.budget
        rules.set_budget(
            Budget(
                total=total,
                spent=current.spent,
                window_s=float(args.window_days) * 86400.0,
            )
        )
        store.save_rule_set(rules)
        remaining = rules.remaining()
        spent = rules.budget.spent
        reserved = rules.reserved
    out.write(
        f"budget {fmt_cents(total)} per {args.window_days} days; "
        f"{fmt_cents(spent)} spent, {fmt_cents(reserved)} held for alerts you have "
        f"not answered, {fmt_cents(remaining)} left, at most "
        f"{fmt_cents(budget_cap(remaining))} on any one alert\n"
    )
    return EXIT_OK


def cmd_bought(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Record a purchase against the budget.

    The one verb that moves ``Budget.spent``.  Nothing else can: this
    tool does not check out, so it cannot know the owner acted on a deep
    link unless they say so, and a ceiling nobody ever charges against is
    not a ceiling.  ``--amount`` overrides what was reserved, which the
    checkout page will disagree with (tax, a coupon, a shipping band).
    """
    with _open(args) as store:
        rules = _rule_set(store)
        if rules.get(args.product) is None and args.product not in rules.reservations():
            raise CliError(f"not watching {args.product!r} and nothing is held for it")
        held = rules.reserved_for(args.product)
        amount = held if args.amount is None else _dollars(args.amount, "--amount")
        if amount <= 0:
            raise CliError(
                f"nothing is reserved for {args.product!r}; pass --amount to record "
                f"a purchase anyway"
            )
        try:
            spent = rules.commit_for(args.product, amount)
        except RuleError as exc:
            raise CliError(str(exc)) from None
        store.save_rule_set(rules)
        remaining = rules.remaining()
        budget = rules.budget
    out.write(
        f"recorded {fmt_cents(spent)} spent on {args.product}; "
        f"{fmt_cents(budget.spent)} of {fmt_cents(budget.total)} used, "
        f"{fmt_cents(remaining)} left, at most {fmt_cents(budget_cap(remaining))} "
        f"on any one alert\n"
    )
    return EXIT_OK


def cmd_release(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Hand back money held for a BUY alert the owner ignored."""
    with _open(args) as store:
        rules = _rule_set(store)
        if args.all:
            freed = rules.release_all()
        else:
            if not args.product:
                raise CliError("release takes --product ID, or --all")
            freed = rules.release_for(args.product)
            if not freed:
                raise CliError(f"nothing is held for {args.product!r}")
        store.save_rule_set(rules)
        remaining = rules.remaining()
    out.write(
        f"released {fmt_cents(freed)}; {fmt_cents(remaining)} free, at most "
        f"{fmt_cents(budget_cap(remaining))} on any one alert\n"
    )
    return EXIT_OK


def cmd_list(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """The watchlist: best price, market, last verdict."""
    with _open(args) as store:
        now = store.now()
        catalog = _catalog(store)
        products = _product_map(catalog)
        rules = _rule_set(store)
        history = _history(store, now)
        warning = _stub_warning(store)
        rows: List[str] = []
        for rule in rules.rules():
            product = products.get(rule.product_id)
            name = product.name if product else f"{rule.product_id} (not in the catalog)"
            latest = _fresh(_latest_by_source(history.for_product(rule.product_id)), now)
            best = _cheapest(latest)
            market = history.market_ref(rule.product_id, now)
            verdict = store.last_verdict(rule.product_id)
            rows.append(
                "  {name:<44} {best:>12}  {market:>12}  {verdict}".format(
                    name=_short(name, 44),
                    best=(
                        f"{fmt_cents(best.landed or 0)} {best.source}"
                        if best is not None else "no price"
                    ),
                    market=(
                        f"med {fmt_cents(market.median)}" if market.usable and market.median
                        else "market thin"
                    ),
                    verdict=(
                        f"{verdict.action.value} ({len(verdict.reasons)} reasons)"
                        if verdict is not None else "never decided"
                    ),
                )
            )
        budget = rules.budget
    if not rows:
        out.write("nothing is being watched; add one with 'watch --product ID --max-price ...'\n")
        return EXIT_OK
    out.write(
        f"watchlist: {len(rows)} product(s), budget {fmt_cents(budget.total)} "
        f"({fmt_cents(rules.remaining())} left)\n"
    )
    for row in rows:
        out.write(row + "\n")
    if warning:
        out.write(warning + "\n")
    return EXIT_OK


def cmd_poll(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """One polling round, as polite as the scheduler insists.

    Every listing the scheduler says is due is fetched once.  Nothing
    else is: a source robots.txt disallows, one inside its interval and
    one that is paused are all simply not offered, and the reason is
    printed.
    """
    if not args.once:
        raise CliError(
            "poll takes --once: a long-running poller belongs in the app, which "
            "owns the fetcher and the schedule"
        )
    seed = _seed(args.seed)
    with _open(args) as store:
        catalog = _catalog(store)
        scheduler = _scheduler(store, catalog)
        now = store.now()
        using_stub = _FETCHER is None or _PARSER is None
        round_no = int(store.meta(STUB_ROUND_KEY) or 0) + 1
        if using_stub:
            store.set_meta(STUB_ROUND_KEY, str(round_no))
            fetcher: Callable[..., FetchResult] = (
                lambda url, headers, policy: stub_fetcher(
                    url, headers, policy, seed=seed, round_no=round_no
                )
            )
            parser: Callable[..., Observation] = stub_parser
        else:
            fetcher, parser = _FETCHER, _PARSER

        due = scheduler.due(now)
        blocked = [
            (sku, scheduler.can_poll(sku, now)[1])
            for sku in catalog.skus()
            if not scheduler.can_poll(sku, now)[0]
        ]
        written = 0
        observations: List[Observation] = []
        for sku in due:
            observation = scheduler.poll_once(sku, fetcher, parser, now)
            if observation is not None:
                observations.append(observation)
        if observations:
            written = store.save_observations(observations)
            if using_stub:
                store.set_meta(STUB_DATA_KEY, "1")

    out.write(
        f"polled {len(due)} listing(s), stored {written} observation(s); "
        f"{len(blocked)} listing(s) were not due\n"
    )
    for observation in observations:
        out.write(
            f"  {observation.source:<12} {observation.product_id:<40} "
            f"{observation.stock.value:<13} "
            f"{fmt_cents(observation.landed) if observation.landed is not None else '-':>10}\n"
        )
    for source in sorted({sku.source for sku, _ in blocked}):
        reason = next(reason for sku, reason in blocked if sku.source == source)
        out.write(f"  {source:<12} skipped: {scrub(reason)}\n")
    if using_stub:
        out.write(
            "  no fetcher is installed, so these are STUB prices this tool invented "
            "(round " + str(round_no) + "); nothing was fetched from anywhere\n"
        )
    return EXIT_OK


def cmd_decide(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Evaluate every watched product now and say why.

    A BUY is a verdict and a link; publishing it to a phone is
    :class:`jarvis_poke.alerts_bridge.AlertBridge`'s job, which the app
    wires to its own ``jarvis_alerts`` service.  This command decides and
    records; it does not notify and it does not buy.
    """
    with _open(args) as store:
        now = store.now()
        catalog = _catalog(store)
        products = _product_map(catalog)
        rules = _rule_set(store)
        history = _history(store, now)
        # Reservations are on, so two BUYs in one pass cannot each
        # promise the same money -- and they are *kept*, by the RuleSet
        # and then by the store, so a BUY alerted on Monday is still
        # holding its money on Tuesday.  Every run of this command is a
        # fresh process; if the ledger did not survive it, the owner's
        # monthly ceiling would reset to untouched on every invocation
        # and only the per-verdict cap would ever say no.
        engine = DecisionEngine(
            catalog, rules, history, _clock, store.load_watch_states()
        )
        verdicts: List[Verdict] = []
        for rule in rules.rules():
            candidates = _fresh(history.for_product(rule.product_id), now)
            verdicts.append(engine.evaluate(rule.product_id, candidates))
        store.append_verdicts(verdicts)
        store.save_watch_states(engine.watch_states)
        store.save_rule_set(rules)
        held = rules.reservations()
        warning = _stub_warning(store)

    if not verdicts:
        out.write("nothing is being watched, so there is nothing to decide\n")
        return EXIT_OK
    counts: Dict[str, int] = {}
    for verdict in verdicts:
        counts[verdict.action.value] = counts.get(verdict.action.value, 0) + 1
    out.write(
        "decided " + str(len(verdicts)) + ": "
        + " ".join(f"{action}={counts[action]}" for action in sorted(counts))
        + "\n"
    )
    for verdict in verdicts:
        product = products.get(verdict.product_id)
        out.write(
            f"\n{verdict.action.value.upper():<8} {_short(product.name if product else verdict.product_id, 56)}"
            + (f"  {fmt_cents(verdict.landed)}" if verdict.landed is not None else "")
            + (f"  x{verdict.quantity}" if verdict.quantity else "")
            + "\n"
        )
        for reason in verdict.reasons:
            out.write(f"    - {scrub(reason)}\n")
        if verdict.action is Action.BUY and verdict.url:
            out.write(f"    open: {verdict.url}\n")
            out.write("    (you buy it there, by hand; this tool stops here)\n")
            out.write(
                f"    then tell the budget: 'bought --product {verdict.product_id}' "
                f"if you did, 'release --product {verdict.product_id}' if you did not\n"
            )
    if held:
        out.write(
            "\nheld against the budget until you say otherwise: "
            + ", ".join(
                f"{pid} {fmt_cents(amount)}" for pid, amount in sorted(held.items())
            )
            + f"  ({fmt_cents(rules.remaining())} still free)\n"
        )
    if warning:
        out.write("\n" + warning + "\n")
    return EXIT_OK


def cmd_sources(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Per source: may we poll it, when next, and why not."""
    with _open(args) as store:
        catalog = _catalog(store)
        scheduler = _scheduler(store, catalog)
        now = store.now()
        pauses = scheduler.pause_state(now)
        stats = scheduler.stats(now)
        conflicts = scheduler.host_conflicts()
    out.write(
        f"{'source':<12} {'state':<12} {'next due':>10} {'listings':>9} "
        f"{'attempts':>9} {'304s':>6} {'errors':>7} {'pauses':>7}  reason\n"
    )
    for source in sorted(stats["sources"]):
        row = stats["sources"][source]
        pause = pauses.get(source, {})
        due_in = _next_due_in(stats, source, now, _source_state(row, pause))
        out.write(
            f"{source:<12} {_source_state(row, pause):<12} {due_in:>10} "
            f"{row['skus']:>9} {row['attempts']:>9} {row['not_modified']:>6} "
            f"{row['errors']:>7} {row['pauses']:>7}  "
            f"{scrub(pause.get('blocked_reason') or row.get('last_reason') or '')}\n"
        )
    for host, sources in conflicts.items():
        # The interval belongs to the host, but the policy and the
        # robots.txt answer are keyed by source id, so two ids on one
        # host get one interval each and that host sees double the
        # agreed rate with nothing in the table looking wrong.
        out.write(
            f"  WARNING {', '.join(sources)} all point at {host}: that host is "
            f"polled once per source, so it sees {len(sources)}x the interval "
            f"each of them promises. Merge them into one source id.\n"
        )
    return EXIT_OK


def cmd_pause(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Stop polling a source by hand, or let it start again."""
    with _open(args) as store:
        catalog = _catalog(store)
        scheduler = _scheduler(store, catalog)
        now = store.now()
        try:
            policy = scheduler.policy(args.source)
        except Exception as exc:  # noqa: BLE001 - reported as one line
            raise CliError(str(exc)) from None
        if args.off:
            scheduler.resume_source(args.source)
            out.write(
                f"{args.source} may be polled again, no sooner than its "
                f"{policy.min_interval_s:.0f}s interval allows\n"
            )
            return EXIT_OK
        seconds = float(args.duration if args.duration is not None else policy.pause_s)
        if seconds <= 0:
            raise CliError("--for must be positive")
        scheduler.pause_source(args.source, now + seconds, "paused by hand")
    out.write(f"{args.source} paused for {seconds:.0f}s; nothing there will be fetched\n")
    return EXIT_OK


def cmd_serve_state(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """The page's state JSON on stdout, so the page needs no server.

    Two ways to drive ``jarvis_poke/web/index.html`` with it: paste it
    into the page's ``jarvis-demo-state`` script tag, or hand it over
    live with ``window.JARVIS_POKE_API = {getState: () => STATE}`` before
    the page's own script runs.  Either way nothing is served and nothing
    is fetched.
    """
    with _open(args) as store:
        state = page_state(store)
    out.write(json.dumps(state, indent=None if args.compact else 2, sort_keys=True) + "\n")
    return EXIT_OK


def cmd_gate(args: argparse.Namespace, out: TextIO, err: TextIO) -> int:
    """Run :func:`jarvis_poke.validate.run_gate`.

    Without ``--defect`` this passes when the gate found nothing.  With
    one, it passes when the gate *did* find something: an injected defect
    that goes unreported is the failure, and that is the question worth
    asking of a gate.
    """
    try:
        validate = importlib.import_module(".validate", __package__)
    except ImportError as exc:
        raise CliError(
            f"gate unavailable: jarvis_poke.validate cannot be imported "
            f"({type(exc).__name__})"
        ) from None
    seed = _seed(args.seed)
    report = validate.run_gate(
        n_products=args.products,
        n_days=args.days,
        seed=seed,
        inject_defect=args.defect,
    )
    out.write(scrub(report.summary()) + "\n")
    if args.defect is None:
        passed = report.ok
        out.write(f"gate {'passed' if passed else 'FAILED'}\n")
    else:
        passed = not report.ok
        out.write(
            f"defect {args.defect!r} was {'caught' if passed else 'MISSED'} "
            f"({len(report.problems)} problem(s))\n"
        )
    return EXIT_OK if passed else EXIT_FAIL


# --------------------------------------------------------------------------
# The page's state
# --------------------------------------------------------------------------


def page_state(store: PokeStore, now: Optional[float] = None) -> Dict[str, Any]:
    """The JSON ``jarvis_poke/web/index.html`` reads.

    Money is ``*_cents`` integers throughout, exactly as it is stored;
    the page formats it.  ``demo`` is true when any of the numbers came
    from :func:`stub_fetcher`, which is the flag the page's own
    made-up-data banner reads.  Everything is a plain JSON type, so the
    file this prints can be served, pasted or committed as a fixture.
    """
    at = store.now() if now is None else now
    catalog = _catalog(store)
    products = _product_map(catalog)
    rules = _rule_set(store)
    history = _history(store, at)
    scheduler = _scheduler(store, catalog)
    pauses = scheduler.pause_state(at)
    stats = scheduler.stats(at)

    watch: List[Dict[str, Any]] = []
    for rule in rules.rules():
        product = products.get(rule.product_id)
        rows = history.for_product(rule.product_id)
        offers = _fresh(_latest_by_source(rows), at)
        best = _cheapest(offers)
        market = history.market_ref(rule.product_id, at)
        verdict = store.last_verdict(rule.product_id)
        watch.append(
            {
                "product": _product_json(rule.product_id, product),
                "rule": _rule_json(rule),
                "best": _offer_json(best),
                "offers": [_offer_json(o) for o in offers],
                "market": {
                    "median_cents": market.median,
                    "p25_cents": market.p25,
                    "low_cents": market.low,
                    "samples": market.samples,
                    "stale": bool(market.stale),
                    "window_days": int(round(market.window_s / 86400.0)),
                },
                "verdict": _verdict_json(verdict),
                # Cents this product's own un-answered BUY alert is
                # holding.  Per row rather than one map on the budget,
                # so the page's shape does not depend on which products
                # happen to be reserved.
                "held_cents": rules.reserved_for(rule.product_id),
                "trend": [
                    [int(when), value]
                    for when, value in price_trend(
                        history, rule.product_id, at, buckets=TREND_BUCKETS
                    )
                ],
            }
        )

    sources: List[Dict[str, Any]] = []
    for source in sorted(stats["sources"]):
        row = stats["sources"][source]
        pause = pauses.get(source, {})
        sources.append(
            {
                "id": source,
                "name": catalog.source_label(source),
                "state": _source_state(row, pause),
                "next_due_at": int(row.get("next_due_at") or 0),
                "errors": int(row.get("errors") or 0),
                "paused_until": int(row.get("paused_until") or 0),
                "min_interval_s": row.get("min_interval_s"),
                "robots_allows": bool(row.get("robots_allows")),
                "last_reason": scrub(str(row.get("last_reason") or "")),
            }
        )

    recent: List[Dict[str, Any]] = []
    for verdict in store.verdicts(limit=RECENT_VERDICTS, newest_first=True):
        entry = _verdict_json(verdict)
        product = products.get(verdict.product_id)
        entry["product_name"] = product.name if product else verdict.product_id
        entry["set_code"] = product.set_code if product else ""
        recent.append(entry)

    budget = rules.budget
    return {
        "generated_at": int(at),
        "demo": store.meta(STUB_DATA_KEY) == "1",
        "budget": {
            "total_cents": budget.total,
            "spent_cents": budget.spent,
            # Money promised to a BUY alert the owner has not answered.
            # The page needs it or its meter and its "left of" disagree:
            # remaining() is net of reservations, spent is not.
            "reserved_cents": rules.reserved,
            "remaining_cents": rules.remaining(),
            "window_days": int(round(budget.window_s / 86400.0)),
        },
        "watch": watch,
        "sources": sources,
        "recent": recent,
    }


def _product_json(product_id: str, product: Optional[Product]) -> Dict[str, Any]:
    if product is None:
        return {
            "id": product_id,
            "name": product_id,
            "set_code": "",
            "kind": "",
            "msrp_cents": None,
        }
    return {
        "id": product.id,
        "name": product.name,
        "set_code": product.set_code,
        "kind": product.kind.value,
        "msrp_cents": product.msrp,
    }


def _rule_json(rule: Rule) -> Dict[str, Any]:
    return {
        "product_id": rule.product_id,
        "max_price_cents": rule.max_price,
        "quantity": rule.quantity,
        "min_discount_pct": rule.min_discount_pct,
        "enabled": rule.enabled,
        "cooldown_s": rule.cooldown_s,
        "include_shipping": rule.include_shipping,
        "allowed_sources": list(rule.allowed_sources),
    }


def _offer_json(observation: Optional[Observation]) -> Optional[Dict[str, Any]]:
    if observation is None:
        return None
    return {
        "source": observation.source,
        "sku": observation.sku,
        "price_cents": observation.price,
        "shipping_cents": observation.shipping,
        "landed_cents": observation.landed,
        "stock": observation.stock.value,
        "url": observation.url,
        "at": int(observation.at),
    }


def _verdict_json(verdict: Optional[Verdict]) -> Optional[Dict[str, Any]]:
    if verdict is None:
        return None
    return {
        "product_id": verdict.product_id,
        "action": verdict.action.value,
        "at": int(verdict.at),
        "source": verdict.source,
        "sku": verdict.sku,
        "price_cents": verdict.price,
        "landed_cents": verdict.landed,
        "market_cents": verdict.market,
        "discount_pct": verdict.discount_pct,
        "quantity": verdict.quantity,
        "url": verdict.url,
        "reasons": [scrub(reason) for reason in verdict.reasons],
    }


def _source_state(row: Dict[str, Any], pause: Dict[str, Any]) -> str:
    """One of the five words the page knows: ok, backoff, paused,
    disallowed, error."""
    if not row.get("robots_allows"):
        return "disallowed"
    if row.get("paused") or pause.get("paused"):
        return "paused"
    errors = int(row.get("consecutive_errors") or 0)
    if errors >= 3:
        return "error"
    if errors > 0:
        return "backoff"
    return "ok"


def _next_due_in(stats: Dict[str, Any], source: str, now: float, state: str) -> str:
    """When this source's soonest listing may be looked at, in words.

    A source robots.txt disallows has no next time at all, and saying
    "now" for one would be the wrong answer to the only question that
    matters about it.
    """
    if state == "disallowed":
        return "never"
    due = [row["effective_due_at"] for row in stats["skus"] if row["source"] == source]
    if not due:
        return "-"
    soonest = min(due)
    return "now" if soonest <= now else f"{soonest - now:.0f}s"


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def _dollars(text: str, option: str) -> Cents:
    """Dollars on the command line, cents in the store.

    contracts.py: money is integer cents, and :func:`to_cents` is the one
    parser for it.  A value that will not parse is refused by the name of
    its option, never by echoing what was typed.
    """
    try:
        cents = to_cents(str(text))
    except ValueError:
        raise CliError(f"{option} must be an amount like 189.99") from None
    if cents <= 0:
        raise CliError(f"{option} must be positive")
    return cents


def _seed(value: Any) -> int:
    try:
        return parse_seed(value)
    except Exception:  # noqa: BLE001 - never echo the value
        raise CliError("--seed must be an integer, decimal or 0x hex") from None


def _short(text: str, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


class _ValueBlindParser(argparse.ArgumentParser):
    """An argparse parser whose errors name the option, not the value.

    A listing URL with a query string can arrive as an argument; argparse
    would put it straight into its error message and from there into a
    log.  :func:`scrub` catches the rest.
    """

    def error(self, message: str) -> None:  # type: ignore[override]
        self.exit(EXIT_USAGE, f"usage error: {scrub(message)}\n")


# --------------------------------------------------------------------------
# Parser and entry point
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    db_top = _ValueBlindParser(add_help=False)
    db_top.add_argument(
        "--db", default=DEFAULT_DB, metavar="PATH",
        help=f"sqlite file of the watchlist (default: {DEFAULT_DB})",
    )
    db_sub = _ValueBlindParser(add_help=False)
    db_sub.add_argument("--db", default=argparse.SUPPRESS, metavar="PATH",
                        help=argparse.SUPPRESS)

    parser = _ValueBlindParser(
        prog="python3 -m jarvis_poke.cli",
        description=(
            "Watch sealed Pokemon product, decide when to act, and hand you a "
            "link. It never buys anything."
        ),
        parents=[db_top],
        allow_abbrev=False,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    sub.required = True

    p = sub.add_parser("init", parents=[db_sub], allow_abbrev=False,
                       help="seed the store from the bundled catalog and sources")
    p.set_defaults(func=cmd_init)

    p = sub.add_parser("watch", parents=[db_sub], allow_abbrev=False,
                       help="add or replace a rule")
    p.add_argument("--product", required=True, metavar="ID")
    p.add_argument("--max-price", required=True, metavar="DOLLARS",
                   help="hard ceiling on the landed price, e.g. 189.99")
    p.add_argument("--qty", type=int, default=1, metavar="N")
    p.add_argument("--min-discount", type=float, default=0.0, metavar="PCT",
                   help="against the market reference, not MSRP")
    p.add_argument("--sources", default="", metavar="A,B",
                   help="only these sources (default: any known source)")
    p.add_argument("--cooldown", type=float, default=3600.0, metavar="SECONDS",
                   help="do not alert again inside this (default 3600)")
    p.set_defaults(func=cmd_watch)

    p = sub.add_parser("unwatch", parents=[db_sub], allow_abbrev=False,
                       help="drop a rule")
    p.add_argument("--product", required=True, metavar="ID")
    p.set_defaults(func=cmd_unwatch)

    p = sub.add_parser("budget", parents=[db_sub], allow_abbrev=False,
                       help="set the spending ceiling")
    p.add_argument("--total", required=True, metavar="DOLLARS")
    p.add_argument("--window-days", type=float, default=30.0, metavar="N")
    p.set_defaults(func=cmd_budget)

    p = sub.add_parser("bought", parents=[db_sub], allow_abbrev=False,
                       help="record a purchase: the only thing that moves 'spent'")
    p.add_argument("--product", required=True, metavar="ID")
    p.add_argument("--amount", default=None, metavar="DOLLARS",
                   help="what you actually paid (default: what was held for it)")
    p.set_defaults(func=cmd_bought)

    p = sub.add_parser("release", parents=[db_sub], allow_abbrev=False,
                       help="hand back money held for an alert you ignored")
    p.add_argument("--product", default="", metavar="ID")
    p.add_argument("--all", action="store_true", help="release every hold")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("list", parents=[db_sub], allow_abbrev=False,
                       help="the watchlist, with prices and the last verdict")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("poll", parents=[db_sub], allow_abbrev=False,
                       help="one polling round")
    p.add_argument("--once", action="store_true",
                   help="one round, then exit (the only mode this CLI offers)")
    p.add_argument("--seed", default="0", help="seed of the stub's sample walk")
    p.set_defaults(func=cmd_poll)

    p = sub.add_parser("decide", parents=[db_sub], allow_abbrev=False,
                       help="evaluate now and print verdicts with reasons")
    p.set_defaults(func=cmd_decide)

    p = sub.add_parser("sources", parents=[db_sub], allow_abbrev=False,
                       help="per-source state, next due and pauses")
    p.set_defaults(func=cmd_sources)

    p = sub.add_parser("pause", parents=[db_sub], allow_abbrev=False,
                       help="stop polling a source, or let it start again")
    p.add_argument("--source", required=True, metavar="ID")
    p.add_argument("--off", action="store_true", help="lift the pause instead")
    p.add_argument("--for", dest="duration", type=float, default=None,
                   metavar="SECONDS", help="how long (default: the source's own pause)")
    p.set_defaults(func=cmd_pause)

    p = sub.add_parser("serve-state", parents=[db_sub], allow_abbrev=False,
                       help="print the page's state JSON")
    p.add_argument("--compact", action="store_true", help="one line, no indent")
    p.set_defaults(func=cmd_serve_state)

    p = sub.add_parser("gate", parents=[db_sub], allow_abbrev=False,
                       help="run the validation gate")
    p.add_argument("--products", type=int, required=True)
    p.add_argument("--days", type=int, required=True)
    p.add_argument("--seed", required=True, help="0x... or decimal")
    p.add_argument("--defect", default=None, metavar="NAME",
                   help="inject a defect; the gate must catch it")
    p.set_defaults(func=cmd_gate)

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    out: Optional[TextIO] = None,
    err: Optional[TextIO] = None,
) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``), run the command, return
    the exit status.  Every error is one line on ``err``; no traceback,
    and no URL query string."""
    out = sys.stdout if out is None else out
    err = sys.stderr if err is None else err
    parser = build_parser()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            args, unknown = parser.parse_known_args(argv)
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else EXIT_USAGE
    if unknown:
        err.write(f"usage error: unrecognized arguments: {_describe_unknown(unknown)}\n")
        return EXIT_USAGE
    try:
        return int(args.func(args, out, err))
    except BrokenPipeError:
        # `... | head` closed the pipe.  The reader went away; nothing
        # failed, so say nothing and leave quietly.
        with contextlib.suppress(Exception):
            sys.stderr.close()
        return EXIT_OK
    except Exception as exc:  # noqa: BLE001 - one line, exit 1, is the contract
        message = str(exc) or type(exc).__name__
        err.write(f"error: {scrub(message)}\n")
        err.flush()
        return EXIT_FAIL


def _describe_unknown(tokens: Sequence[str]) -> str:
    """Name the options, count the values.

    A bare value may be a URL with a token in its query; it is counted,
    never shown.
    """
    names = [t.split("=", 1)[0] for t in tokens if t.startswith("-")]
    values = len(tokens) - len(names)
    parts = names + ([f"({values} value{'s' if values != 1 else ''} not shown)"] if values else [])
    return " ".join(parts)


if __name__ == "__main__":
    sys.exit(main())
