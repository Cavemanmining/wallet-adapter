"""Tests for the jarvis_poke catalog and the polite polling scheduler.

Design: jarvis_poke/contracts.py -- the "Products" section (Product,
SourceSku, ProductKind) and "Fetching, kept polite by construction"
(FetchPolicy, FetchResult, injected Fetcher and Parser).

The point of most of this file is the politeness contract, so the tests
are written as things the scheduler must *refuse* to do: poll inside a
host's ``min_interval_s``, poll a source robots.txt disallows, poll a
paused source, ignore a ``Retry-After``, or let one broken listing throw
its way out of a pass.  The last test simulates 500 ticks and asserts the
per-host floor held every single time.

Nothing here touches the network: the fetcher and the parser are ordinary
local callables, which is the whole reason contracts.py makes them
injected.  Nothing sleeps: the clock is a ``SimClock`` and every method
takes ``now``.  What randomness the simulation needs comes from a
``lucifer_gen.seed`` stream, so a failure here reproduces exactly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Runnable as `pytest tests/test_poke_sources.py` or
# `python3 tests/test_poke_sources.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_poke.catalog import (
    CATALOG_PATH,
    SOURCES_PATH,
    Catalog,
    CatalogError,
)
from jarvis_poke.contracts import (
    FetchPolicy,
    FetchResult,
    Observation,
    Product,
    ProductKind,
    SourceSku,
    Stock,
)
from jarvis_poke.sources import (
    BACKOFF_MAX_DOUBLINGS,
    DEFAULT_SEED,
    MemoryPollStore,
    PolicyError,
    PollScheduler,
    SchedulerError,
    load_policies,
    policies_from_obj,
)
from lucifer_gen.seed import SeedFields

T0 = 1_700_000_000.0


# --------------------------------------------------------------------------
# fixtures and doubles
# --------------------------------------------------------------------------


class SimClock:
    """An injected clock. Nothing in the package may call time.time()."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> float:
        self.now += seconds
        return self.now


class RecordingFetcher:
    """A scripted Fetcher. Records (url, headers, policy.source, at)."""

    def __init__(self, script: Optional[Dict[str, List[Any]]] = None,
                 default: Optional[FetchResult] = None,
                 clock: Optional[SimClock] = None) -> None:
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default or FetchResult(ok=True, status=200, body="body")
        self.calls: List[Tuple[str, Dict[str, str], str, float]] = []
        self.clock = clock

    def __call__(self, url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
        at = self.clock.now if self.clock else 0.0
        self.calls.append((url, dict(headers), policy.source, at))
        queue = self.script.get(url)
        outcome = queue.pop(0) if queue else self.default
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    @property
    def urls(self) -> List[str]:
        return [call[0] for call in self.calls]


def simple_parser(sku: SourceSku, body: str, at: float) -> Observation:
    return Observation(
        product_id=sku.product_id,
        source=sku.source,
        sku=sku.sku,
        at=at,
        stock=Stock.IN_STOCK,
        price=4999,
        shipping=500,
        url=sku.url,
    )


def boom_parser(sku: SourceSku, body: str, at: float) -> Observation:
    raise ValueError("secret-looking parser detail that must not be logged")


PRODUCTS = [
    Product("alpha-etb", "Alpha Elite Trainer Box", "AA01", ProductKind.ELITE_TRAINER_BOX, 4999,
            "2024-01-01"),
    Product("beta-box", "Beta Booster Box", "BB02", ProductKind.BOOSTER_BOX, 16164, "2024-06-01"),
    Product("gamma-tin", "Gamma Mini Tin", "CC03", ProductKind.TIN, 849, None),
]


def make_catalog() -> Catalog:
    """Two polite sources, one that robots.txt disallows."""
    skus = []
    for source, host in (("alpha", "alphamart"), ("bravo", "bravobarn"), ("nogo", "nogoco")):
        for product in PRODUCTS:
            skus.append(SourceSku(
                source=source,
                product_id=product.id,
                sku=f"{source.upper()}-{product.id}",
                url=f"https://{host}.example.com/p/{product.id}",
            ))
    return Catalog(PRODUCTS, skus)


def make_policies(**overrides: Any) -> Dict[str, FetchPolicy]:
    policies = {
        "alpha": FetchPolicy("alpha", min_interval_s=300.0, max_errors_before_pause=3,
                             pause_s=1800.0),
        "bravo": FetchPolicy("bravo", min_interval_s=600.0, max_errors_before_pause=5,
                             pause_s=3600.0),
        "nogo": FetchPolicy("nogo", min_interval_s=900.0, robots_allows=False),
    }
    policies.update(overrides)
    return policies


def make_scheduler(clock: Optional[SimClock] = None, **kwargs: Any) -> Tuple[PollScheduler, SimClock]:
    clock = clock or SimClock()
    catalog = kwargs.pop("catalog", None) or make_catalog()
    policies = kwargs.pop("policies", None) or make_policies()
    return PollScheduler(catalog, policies, clock, **kwargs), clock


def sku_of(catalog: Catalog, source: str, product_id: str) -> SourceSku:
    found = catalog.sku(source, product_id)
    assert found is not None
    return found


# ==========================================================================
# catalog: validation
# ==========================================================================


def _catalog_obj(**over: Any) -> Dict[str, Any]:
    product = {"id": "alpha-etb", "name": "Alpha ETB", "set_code": "AA01",
               "kind": "elite_trainer_box", "msrp": 4999, "released": "2024-01-01"}
    product.update(over)
    return {"products": [product]}


def _sources_obj(**over: Any) -> Dict[str, Any]:
    sku = {"product_id": "alpha-etb", "sku": "AM-1",
           "url": "https://alphamart.example.com/p/alpha-etb"}
    sku.update(over)
    return {"sources": [{"id": "alpha", "name": "AlphaMart", "policy": {}, "skus": [sku]}]}


MALFORMED_CATALOGS = [
    pytest.param({"products": [_catalog_obj()["products"][0],
                               dict(_catalog_obj()["products"][0], name="Twin")]},
                 "duplicate product id", id="duplicate-product-id"),
    pytest.param(_catalog_obj(msrp=0), "msrp must be positive", id="msrp-zero"),
    pytest.param(_catalog_obj(msrp=-100), "msrp must be positive", id="msrp-negative"),
    pytest.param(_catalog_obj(msrp=49.99), "integer number of cents", id="msrp-float"),
    pytest.param(_catalog_obj(kind="single_card"), "unknown kind", id="unknown-kind"),
    pytest.param(_catalog_obj(kind=None), "unknown kind", id="missing-kind"),
    pytest.param(_catalog_obj(id=""), "id must be a non-empty string", id="empty-id"),
    pytest.param(_catalog_obj(name=""), "name must be a non-empty string", id="empty-name"),
    pytest.param(_catalog_obj(set_code=None), "set_code must be a non-empty string",
                 id="missing-set-code"),
    pytest.param(_catalog_obj(released="2024-13-01"), "ISO YYYY-MM-DD", id="bad-release-month"),
    pytest.param(_catalog_obj(released="March 2024"), "ISO YYYY-MM-DD", id="bad-release-text"),
    pytest.param(_catalog_obj(upc=820650855658), "upc must be a string", id="numeric-upc"),
    pytest.param({"products": ["alpha-etb"]}, "not an object", id="product-not-an-object"),
    pytest.param({"products": {"alpha": {}}}, "must be a list", id="products-not-a-list"),
]


@pytest.mark.parametrize("obj,message", MALFORMED_CATALOGS)
def test_catalog_rejects_malformed_products(obj: Any, message: str) -> None:
    with pytest.raises(CatalogError) as excinfo:
        Catalog.from_obj(obj, {"sources": []}, origin="unit")
    assert message in str(excinfo.value)


MALFORMED_SOURCES = [
    pytest.param({"sources": [{"id": "alpha", "skus": [
        {"product_id": "ghost-product", "sku": "AM-9",
         "url": "https://alphamart.example.com/p/ghost"}]}]},
        "unknown product", id="sku-points-nowhere"),
    pytest.param(_sources_obj(url="/p/alpha-etb"), "absolute http(s) url", id="relative-url"),
    pytest.param(_sources_obj(url="alphamart.example.com/p/alpha-etb"), "absolute http(s) url",
                 id="scheme-less-url"),
    pytest.param(_sources_obj(url="ftp://alphamart.example.com/p/a"), "absolute http(s) url",
                 id="wrong-scheme"),
    pytest.param(_sources_obj(url="https:///p/alpha-etb"), "absolute http(s) url",
                 id="url-without-host"),
    pytest.param(_sources_obj(url=None), "absolute http(s) url", id="missing-url"),
    pytest.param(_sources_obj(sku=""), "sku must be a non-empty string", id="empty-sku"),
    pytest.param(_sources_obj(product_id=""), "product_id must be a non-empty string",
                 id="empty-product-id"),
    pytest.param({"sources": [{"id": "alpha", "skus": [
        {"product_id": "alpha-etb", "sku": "AM-1", "url": "https://a.example.com/1"},
        {"product_id": "alpha-etb", "sku": "AM-2", "url": "https://a.example.com/2"}]}]},
        "already lists product", id="same-source-twice"),
    pytest.param({"sources": [{"id": "", "skus": []}]}, "source id must be a non-empty string",
                 id="empty-source-id"),
    pytest.param({"sources": [{"id": "alpha", "skus": {"a": 1}}]}, "'skus' must be a list",
                 id="skus-not-a-list"),
    pytest.param({"sources": ["alpha"]}, "is not an object", id="source-not-an-object"),
]


@pytest.mark.parametrize("obj,message", MALFORMED_SOURCES)
def test_catalog_rejects_malformed_skus(obj: Any, message: str) -> None:
    with pytest.raises(CatalogError) as excinfo:
        Catalog.from_obj(_catalog_obj(), obj, origin="unit")
    assert message in str(excinfo.value)


def test_catalog_error_names_the_offending_entry() -> None:
    with pytest.raises(CatalogError) as excinfo:
        Catalog.from_obj(_catalog_obj(msrp=-1), {"sources": []})
    assert "alpha-etb" in str(excinfo.value)

    with pytest.raises(CatalogError) as excinfo:
        Catalog.from_obj(_catalog_obj(), _sources_obj(url="nope"))
    text = str(excinfo.value)
    assert "alpha" in text and "alpha-etb" in text


def test_catalog_accepts_missing_optional_fields() -> None:
    catalog = Catalog.from_obj(
        {"products": [{"id": "p", "name": "P", "set_code": "S", "kind": "tin"}]},
        {"sources": []},
    )
    product = catalog.product("p")
    assert product.msrp is None and product.released is None and product.upc is None


def test_catalog_load_reports_a_missing_or_broken_file(tmp_path: Path) -> None:
    with pytest.raises(CatalogError) as excinfo:
        Catalog.load(tmp_path / "nope.json", SOURCES_PATH)
    assert "no such catalog file" in str(excinfo.value)

    broken = tmp_path / "catalog.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(CatalogError) as excinfo:
        Catalog.load(broken, SOURCES_PATH)
    assert "not valid JSON" in str(excinfo.value)


# ==========================================================================
# catalog: reading, searching, runtime edits
# ==========================================================================


def test_catalog_lookup_and_ordering() -> None:
    catalog = make_catalog()
    assert [p.id for p in catalog.products()] == ["alpha-etb", "beta-box", "gamma-tin"]
    assert catalog.product("beta-box").kind is ProductKind.BOOSTER_BOX
    assert catalog.find("nope") is None
    assert "alpha-etb" in catalog and len(catalog) == 3
    assert catalog.sources() == ["alpha", "bravo", "nogo"]
    assert [s.source for s in catalog.skus_for("alpha-etb")] == ["alpha", "bravo", "nogo"]
    assert catalog.sku("alpha", "beta-box").sku == "ALPHA-beta-box"
    assert catalog.sku("alpha", "ghost") is None
    assert [(s.source, s.product_id) for s in catalog.skus()] == sorted(
        (s.source, s.product_id) for s in catalog.skus())


def test_catalog_product_raises_for_unknown_id() -> None:
    catalog = make_catalog()
    with pytest.raises(CatalogError) as excinfo:
        catalog.product("no-such-thing")
    assert "no-such-thing" in str(excinfo.value)
    with pytest.raises(CatalogError):
        catalog.skus_for("no-such-thing")


def test_search_matches_name_set_code_and_kind_case_insensitively() -> None:
    catalog = make_catalog()
    assert [p.id for p in catalog.search("alpha")] == ["alpha-etb"]
    assert [p.id for p in catalog.search("ALPHA")] == ["alpha-etb"]
    assert [p.id for p in catalog.search("bb02")] == ["beta-box"]        # set code
    assert [p.id for p in catalog.search("TIN")] == ["gamma-tin"]        # kind
    assert [p.id for p in catalog.search("booster_box")] == ["beta-box"]
    assert [p.id for p in catalog.search("elite trainer")] == ["alpha-etb"]
    assert catalog.search("nothing at all") == []
    assert len(catalog.search("")) == 3
    assert len(catalog.search("   ")) == 3


def test_search_requires_every_token_to_match() -> None:
    catalog = make_catalog()
    assert [p.id for p in catalog.search("beta bb02")] == ["beta-box"]   # name + set code
    assert [p.id for p in catalog.search("beta booster_box")] == ["beta-box"]
    assert catalog.search("beta tin") == []


def test_runtime_add_and_remove() -> None:
    catalog = make_catalog()
    product = Product("delta-bundle", "Delta Booster Bundle", "DD04", ProductKind.BUNDLE, 2694)
    catalog.add_product(product)
    assert catalog.product("delta-bundle") is product

    with pytest.raises(CatalogError):
        catalog.add_product(product)  # duplicate id

    sku = SourceSku("alpha", "delta-bundle", "ALPHA-delta", "https://alphamart.example.com/p/d")
    catalog.add_sku(sku)
    assert catalog.sku("alpha", "delta-bundle") is sku
    with pytest.raises(CatalogError):
        catalog.add_sku(sku)  # same source lists it twice
    with pytest.raises(CatalogError):
        catalog.add_sku(SourceSku("alpha", "ghost", "X", "https://a.example.com/x"))
    with pytest.raises(CatalogError):
        catalog.add_sku(SourceSku("alpha", "beta-box", "X", "javascript:alert(1)"))

    catalog.remove_sku("alpha", "delta-bundle")
    assert catalog.sku("alpha", "delta-bundle") is None
    with pytest.raises(CatalogError):
        catalog.remove_sku("alpha", "delta-bundle")

    # Removing a product takes its listings with it: no SKU may ever point
    # at a product that is gone.
    catalog.add_sku(sku)
    catalog.remove_product("delta-bundle")
    assert catalog.find("delta-bundle") is None
    assert catalog.sku("alpha", "delta-bundle") is None
    assert all(s.product_id != "delta-bundle" for s in catalog.skus())


# ==========================================================================
# shipped data
# ==========================================================================

REAL_RETAILERS = ("amazon", "walmart", "target", "costco", "gamestop", "bestbuy",
                  "best-buy", "ebay", "tcgplayer", "pokemoncenter", "sams club",
                  "barnesandnoble", "meijer", "kroger")
SCRAPING_WORDS = ("selector", "xpath", "css_path", "queryselector", "regex_price")


def test_shipped_catalog_loads_and_is_plausible() -> None:
    catalog = Catalog.load()
    products = catalog.products()
    assert len(products) >= 12
    assert {p.kind for p in products} == set(ProductKind), "every ProductKind should be shown"
    assert len({p.set_code for p in products}) >= 4
    for product in products:
        assert product.msrp is None or (isinstance(product.msrp, int) and product.msrp > 0)
        assert product.name and product.set_code
    assert catalog.sources() == ["bigboxco", "cardbarn", "examplemart", "hobbyhub"]
    assert catalog.skus(), "the shipped sources should list something"


def test_shipped_sources_are_placeholders_with_example_com_urls() -> None:
    raw = SOURCES_PATH.read_text(encoding="utf-8").lower()
    for name in REAL_RETAILERS:
        assert name not in raw, f"shipped data must not name a real retailer ({name})"
    for word in SCRAPING_WORDS:
        assert word not in raw, f"shipped data must carry no scraping detail ({word})"
    for sku in Catalog.load().skus():
        host = sku.url.split("://", 1)[1].split("/", 1)[0]
        assert host.endswith("example.com"), sku.url
        assert sku.url.startswith("https://"), sku.url


def test_shipped_policies_are_polite() -> None:
    policies = load_policies()
    assert sorted(policies) == ["bigboxco", "cardbarn", "examplemart", "hobbyhub"]
    for source, policy in policies.items():
        assert 300.0 <= policy.min_interval_s <= 900.0, source
        assert policy.max_errors_before_pause >= 1
        assert policy.pause_s > 0
        assert "JarvisPokeWatch" in policy.user_agent
    # One placeholder stands for a host whose robots.txt says no, so the
    # shipped data exercises the rule rather than only describing it.
    assert policies["bigboxco"].robots_allows is False


def test_shipped_catalog_json_is_the_shape_the_loader_documents() -> None:
    obj = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    assert isinstance(obj["products"], list)
    assert all(isinstance(p["msrp"], int) for p in obj["products"] if p.get("msrp") is not None)


def test_policy_file_validation() -> None:
    with pytest.raises(PolicyError) as excinfo:
        policies_from_obj({"sources": [{"id": "x", "policy": {"min_interval_s": 5}}]})
    assert "below the configured floor" in str(excinfo.value)

    # The config floor is a preference; the 30s floor in FetchPolicy is not.
    with pytest.raises(PolicyError) as excinfo:
        policies_from_obj({"sources": [{"id": "x", "policy": {"min_interval_s": 5}}]},
                          min_interval_floor=1.0)
    assert "30s" in str(excinfo.value)

    for block, message in (
        ({"robots_allows": "yes"}, "robots_allows must be true or false"),
        ({"max_errors_before_pause": 0}, "max_errors_before_pause"),
        ({"pause_s": 0}, "pause_s must be positive"),
        ({"user_agent": ""}, "user_agent"),
    ):
        with pytest.raises(PolicyError) as excinfo:
            policies_from_obj({"sources": [{"id": "x", "policy": dict({"min_interval_s": 300},
                                                                     **block)}]})
        assert message in str(excinfo.value)

    with pytest.raises(PolicyError):
        policies_from_obj({"sources": [{"id": "x", "policy": {"min_interval_s": 300}},
                                       {"id": "x", "policy": {"min_interval_s": 300}}]})


def test_shipped_catalog_and_policies_agree() -> None:
    catalog = Catalog.load()
    policies = load_policies()
    assert set(catalog.sources()) <= set(policies), "every listed source needs a policy"


# ==========================================================================
# scheduler: what is due
# ==========================================================================


def test_due_returns_at_most_one_sku_per_source() -> None:
    scheduler, clock = make_scheduler()
    due = scheduler.due(clock.now)
    assert [sku.source for sku in due] == ["alpha", "bravo"]
    # The interval belongs to the host, so two listings of one shop can
    # never both be due at one instant.
    assert len({sku.source for sku in due}) == len(due)


def test_due_never_returns_a_source_robots_disallows() -> None:
    scheduler, clock = make_scheduler()
    for _ in range(50):
        assert all(sku.source != "nogo" for sku in scheduler.due(clock.now))
        clock.tick(1000.0)
    allowed, reason = scheduler.can_poll(sku_of(scheduler.catalog, "nogo", "alpha-etb"), clock.now)
    assert not allowed and "robots" in reason


def test_poll_once_refuses_a_source_robots_disallows() -> None:
    scheduler, clock = make_scheduler()
    fetcher = RecordingFetcher(clock=clock)
    sku = sku_of(scheduler.catalog, "nogo", "alpha-etb")
    assert scheduler.poll_once(sku, fetcher, simple_parser, clock.now) is None
    assert fetcher.calls == [], "a disallowed source must never be fetched"


def test_due_never_returns_a_sku_before_its_interval() -> None:
    scheduler, clock = make_scheduler()
    fetcher = RecordingFetcher(clock=clock)
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    scheduler.poll_once(sku, fetcher, simple_parser, clock.now)

    # The whole host waits, not just the listing that was polled.
    for step in range(1, 6):
        clock.tick(50.0)  # 50..250s after the attempt
        assert all(s.source != "alpha" for s in scheduler.due(clock.now)), step
    clock.tick(300.0)  # well past 300s + jitter
    assert any(s.source == "alpha" for s in scheduler.due(clock.now))


def test_poll_once_enforces_the_interval_even_without_due() -> None:
    scheduler, clock = make_scheduler()
    fetcher = RecordingFetcher(clock=clock)
    catalog = scheduler.catalog
    first = sku_of(catalog, "alpha", "alpha-etb")
    second = sku_of(catalog, "alpha", "beta-box")

    assert scheduler.poll_once(first, fetcher, simple_parser, clock.now) is not None
    # Same instant, different listing of the same host: refused.
    assert scheduler.poll_once(second, fetcher, simple_parser, clock.now) is None
    # Same listing again: refused.
    assert scheduler.poll_once(first, fetcher, simple_parser, clock.now) is None
    assert len(fetcher.calls) == 1
    assert scheduler.stats(clock.now)["sources"]["alpha"]["refusals"] == 2

    clock.tick(400.0)
    assert scheduler.poll_once(second, fetcher, simple_parser, clock.now) is not None
    assert len(fetcher.calls) == 2


def test_due_orders_by_how_overdue_and_rotates_listings() -> None:
    scheduler, clock = make_scheduler()
    fetcher = RecordingFetcher(clock=clock)
    seen: List[str] = []
    for _ in range(9):
        for sku in scheduler.due(clock.now):
            if sku.source == "alpha":
                seen.append(sku.product_id)
                scheduler.poll_once(sku, fetcher, simple_parser, clock.now)
        clock.tick(340.0)   # past 300s + the widest 10% jitter
    # Three listings share one host and one 300s interval, so each gets a
    # third of the looks -- in rotation, because the listing waiting
    # longest is the one that goes. Without that, the host's interval
    # would be spent on whichever listing sorts first, forever.
    assert len(seen) == 9
    assert sorted(seen) == sorted(["alpha-etb", "beta-box", "gamma-tin"] * 3)
    assert len(set(seen[:3])) == 3, f"first round should touch each listing once: {seen[:3]}"
    assert seen[:3] == seen[3:6] == seen[6:9], "and then repeat the same rotation"


def test_unknown_source_is_never_polled_and_says_so() -> None:
    catalog = make_catalog()
    scheduler = PollScheduler(catalog, {"alpha": make_policies()["alpha"]}, SimClock())
    assert {sku.source for sku in scheduler.due(T0)} == {"alpha"}
    allowed, reason = scheduler.can_poll(sku_of(catalog, "bravo", "beta-box"), T0)
    assert not allowed and "no policy" in reason
    with pytest.raises(SchedulerError):
        scheduler.policy("bravo")
    with pytest.raises(SchedulerError):
        scheduler.conditional_headers(sku_of(catalog, "bravo", "beta-box"))


# ==========================================================================
# scheduler: conditional requests
# ==========================================================================


def test_conditional_headers_appear_only_once_validators_are_known() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")

    first = scheduler.conditional_headers(sku)
    assert set(first) == {"User-Agent"}
    assert first["User-Agent"] == scheduler.policy("alpha").user_agent

    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x", etag='W/"v1"'),
                             clock.now)
    second = scheduler.conditional_headers(sku)
    assert second["If-None-Match"] == 'W/"v1"'
    assert "If-Modified-Since" not in second

    clock.tick(400.0)
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x", etag='W/"v2"',
                                              last_modified="Wed, 01 Jan 2025 00:00:00 GMT"),
                             clock.now)
    third = scheduler.conditional_headers(sku)
    assert third["If-None-Match"] == 'W/"v2"'
    assert third["If-Modified-Since"] == "Wed, 01 Jan 2025 00:00:00 GMT"

    # Another listing on the same host has its own validators.
    other = scheduler.conditional_headers(sku_of(scheduler.catalog, "alpha", "beta-box"))
    assert set(other) == {"User-Agent"}


def test_a_fresh_200_without_validators_clears_them() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, etag='W/"v1"'), clock.now)
    clock.tick(400.0)
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200), clock.now)
    assert set(scheduler.conditional_headers(sku)) == {"User-Agent"}


def test_headers_are_passed_to_the_fetcher() -> None:
    scheduler, clock = make_scheduler()
    fetcher = RecordingFetcher(clock=clock)
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    fetcher.default = FetchResult(ok=True, status=200, body="x", etag='W/"v1"')
    scheduler.poll_once(sku, fetcher, simple_parser, clock.now)
    clock.tick(400.0)
    scheduler.poll_once(sku, fetcher, simple_parser, clock.now)
    assert "If-None-Match" not in fetcher.calls[0][1]
    assert fetcher.calls[1][1]["If-None-Match"] == 'W/"v1"'
    assert fetcher.calls[1][1]["User-Agent"] == scheduler.policy("alpha").user_agent


def test_304_counts_as_a_successful_attempt_and_yields_no_observation() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    fetcher = RecordingFetcher(clock=clock)
    fetcher.default = FetchResult(ok=True, status=200, body="x", etag='W/"v1"')
    assert scheduler.poll_once(sku, fetcher, simple_parser, clock.now) is not None

    clock.tick(400.0)
    fetcher.default = FetchResult(ok=True, status=304, not_modified=True)
    before = scheduler.stats(clock.now)["sources"]["alpha"]
    observation = scheduler.poll_once(sku, fetcher, simple_parser, clock.now)
    after = scheduler.stats(clock.now)["sources"]["alpha"]

    assert observation is None, "a 304 produces no new observation"
    assert after["attempts"] == before["attempts"] + 1
    assert after["not_modified"] == before["not_modified"] + 1
    assert after["errors"] == before["errors"]
    assert after["consecutive_errors"] == 0
    assert after["last_attempt_at"] == clock.now
    # Timing moved on exactly as a 200 would have moved it.
    assert scheduler.effective_due_at(sku) >= clock.now + 300.0
    # The validator survives a 304 that does not resend one.
    assert scheduler.conditional_headers(sku)["If-None-Match"] == 'W/"v1"'


# ==========================================================================
# scheduler: errors, pauses and retry-after
# ==========================================================================


def test_consecutive_errors_widen_the_backoff() -> None:
    scheduler, clock = make_scheduler(
        policies=make_policies(alpha=FetchPolicy("alpha", min_interval_s=300.0,
                                                 max_errors_before_pause=99)))
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    interval = 300.0
    for n in range(1, 6):
        scheduler.record_attempt(sku, FetchResult(ok=False, status=500, reason="boom"), clock.now)
        expected = interval * (2 ** min(n - 1, BACKOFF_MAX_DOUBLINGS))
        gap = scheduler.effective_due_at(sku) - clock.now
        assert expected <= gap < expected * 1.11, (n, gap, expected)
        clock.tick(gap + 1.0)
    # And a success clears the run.
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), clock.now)
    assert scheduler.effective_due_at(sku) - clock.now < interval * 1.11


def test_repeated_errors_pause_the_source_and_the_pause_expires() -> None:
    scheduler, clock = make_scheduler()          # alpha: 3 errors -> 1800s pause
    catalog = scheduler.catalog
    sku = sku_of(catalog, "alpha", "alpha-etb")
    policy = scheduler.policy("alpha")

    for n in range(policy.max_errors_before_pause):
        if n:
            clock.tick(10_000.0)   # long enough that only a pause can hold it back
        assert scheduler.pause_state(clock.now)["alpha"]["paused"] is (
            n >= policy.max_errors_before_pause)
        scheduler.record_attempt(sku, FetchResult(ok=False, status=503, reason="busy"), clock.now)

    state = scheduler.pause_state(clock.now)["alpha"]
    assert state["paused"] is True
    assert "consecutive errors" in state["reason"]
    assert state["seconds_remaining"] == pytest.approx(policy.pause_s)

    # Every listing of the source is held, not only the one that failed.
    assert all(s.source != "alpha" for s in scheduler.due(clock.now))
    fetcher = RecordingFetcher(clock=clock)
    assert scheduler.poll_once(sku_of(catalog, "alpha", "beta-box"), fetcher, simple_parser,
                               clock.now) is None
    assert fetcher.calls == []

    # ... and it expires by itself.
    clock.tick(policy.pause_s + 1.0)
    assert scheduler.pause_state(clock.now)["alpha"]["paused"] is False
    assert any(s.source == "alpha" for s in scheduler.due(clock.now))
    assert scheduler.poll_once(sku, RecordingFetcher(clock=clock), simple_parser,
                               clock.now) is not None


def test_errors_on_one_source_do_not_touch_another() -> None:
    scheduler, clock = make_scheduler()
    alpha = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    for _ in range(5):
        scheduler.record_attempt(alpha, FetchResult(ok=False, reason="boom"), clock.now)
    assert scheduler.pause_state(clock.now)["alpha"]["paused"] is True
    assert scheduler.pause_state(clock.now)["bravo"]["paused"] is False
    assert any(s.source == "bravo" for s in scheduler.due(clock.now))


def test_retry_after_is_honoured_for_at_least_that_long() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    scheduler.record_attempt(
        sku, FetchResult(ok=False, status=429, retry_after_s=5400.0, reason="slow down"),
        clock.now)

    assert scheduler.pause_state(clock.now)["alpha"]["paused"] is True
    clock.tick(5399.0)
    assert all(s.source != "alpha" for s in scheduler.due(clock.now))
    clock.tick(2.0)
    assert scheduler.pause_state(clock.now)["alpha"]["paused"] is False


def test_retry_after_on_a_successful_response_is_still_honoured() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x", retry_after_s=3600.0),
                             clock.now)
    assert scheduler.pause_state(clock.now)["alpha"]["paused"] is True
    assert scheduler.effective_due_at(sku) >= clock.now + 3600.0


def test_retry_after_never_shortens_a_longer_pause() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    scheduler.pause_source("alpha", clock.now + 7200.0, "by hand")
    scheduler.record_attempt(sku, FetchResult(ok=False, status=429, retry_after_s=60.0), clock.now)
    assert scheduler.pause_state(clock.now)["alpha"]["seconds_remaining"] == pytest.approx(7200.0)


def test_pause_and_resume_by_hand() -> None:
    scheduler, clock = make_scheduler()
    scheduler.pause_source("bravo", clock.now + 900.0)
    assert all(s.source != "bravo" for s in scheduler.due(clock.now))
    scheduler.resume_source("bravo")
    assert any(s.source == "bravo" for s in scheduler.due(clock.now))
    with pytest.raises(SchedulerError):
        scheduler.pause_source("ghost", clock.now + 10.0)


# ==========================================================================
# scheduler: a fetcher or a parser that blows up
# ==========================================================================


def test_a_raising_fetcher_is_recorded_as_an_error_and_does_not_propagate() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    fetcher = RecordingFetcher(script={sku.url: [RuntimeError("network on fire")]}, clock=clock)

    assert scheduler.poll_once(sku, fetcher, simple_parser, clock.now) is None
    row = scheduler.stats(clock.now)["sources"]["alpha"]
    assert row["errors"] == 1 and row["attempts"] == 1 and row["consecutive_errors"] == 1
    assert row["last_reason"] == "fetcher raised RuntimeError"
    assert "network on fire" not in row["last_reason"], "only the type name is kept"
    # It feeds the backoff like any other failure.
    assert scheduler.effective_due_at(sku) >= clock.now + 300.0


def test_a_raising_parser_is_recorded_as_an_error_and_does_not_propagate() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    fetcher = RecordingFetcher(clock=clock)

    assert scheduler.poll_once(sku, fetcher, boom_parser, clock.now) is None
    row = scheduler.stats(clock.now)["sources"]["alpha"]
    assert row["errors"] == 1 and row["parse_errors"] == 1
    assert row["attempts"] == 1, "the fetch was one attempt; the parse is not a second one"
    assert row["last_reason"] == "parser raised ValueError"
    assert "secret-looking" not in row["last_reason"]
    assert row["observations"] == 0


def test_a_parser_returning_nothing_is_not_an_error() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    assert scheduler.poll_once(sku, RecordingFetcher(clock=clock),
                               lambda s, b, at: None, clock.now) is None
    assert scheduler.stats(clock.now)["sources"]["alpha"]["errors"] == 0


def test_a_parser_returning_the_wrong_type_is_an_error() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    assert scheduler.poll_once(sku, RecordingFetcher(clock=clock),
                               lambda s, b, at: {"price": 1999}, clock.now) is None
    row = scheduler.stats(clock.now)["sources"]["alpha"]
    assert row["errors"] == 1 and row["parse_errors"] == 1


def test_a_fetcher_returning_the_wrong_type_is_an_error() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    assert scheduler.poll_once(sku, lambda url, headers, policy: "<html>", simple_parser,
                               clock.now) is None
    assert scheduler.stats(clock.now)["sources"]["alpha"]["errors"] == 1


def test_repeated_parse_failures_eventually_pause_the_source() -> None:
    """A page that fetches perfectly and never parses must still back off.

    The successful fetch clears the source's error run a moment before
    the parser fails; if the parse failure did not pick that run back up,
    an unparseable listing would be polled at full rate forever.
    """
    scheduler, clock = make_scheduler()          # alpha: 3 errors -> pause
    catalog = scheduler.catalog
    products = ("alpha-etb", "beta-box", "gamma-tin")
    for n, product in enumerate(products):
        if n:
            clock.tick(20_000.0)   # past the widening backoff
        assert scheduler.pause_state(clock.now)["alpha"]["paused"] is False
        observation = scheduler.poll_once(sku_of(catalog, "alpha", product),
                                          RecordingFetcher(clock=clock), boom_parser, clock.now)
        assert observation is None
        row = scheduler.stats(clock.now)["sources"]["alpha"]
        assert row["parse_errors"] == n + 1
        assert row["consecutive_errors"] == n + 1, "a successful fetch must not clear a parse run"
    assert scheduler.pause_state(clock.now)["alpha"]["paused"] is True


def test_a_successful_poll_returns_the_parsers_observation() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    observation = scheduler.poll_once(sku, RecordingFetcher(clock=clock), simple_parser, clock.now)
    assert isinstance(observation, Observation)
    assert observation.product_id == "alpha-etb" and observation.source == "alpha"
    assert observation.landed == 4999 + 500        # contracts.Observation.landed
    assert observation.purchasable is True
    assert scheduler.stats(clock.now)["totals"]["observations"] == 1


# ==========================================================================
# scheduler: jitter determinism and state
# ==========================================================================


def _due_sequence(seed: int, steps: int = 8) -> List[float]:
    scheduler, clock = make_scheduler(seed=seed)
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    out: List[float] = []
    for _ in range(steps):
        scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), clock.now)
        out.append(round(scheduler.effective_due_at(sku) - clock.now, 9))
        clock.tick(1000.0)
    return out


def test_jitter_is_deterministic_for_a_fixed_seed() -> None:
    first = _due_sequence(0xC0FFEE)
    assert first == _due_sequence(0xC0FFEE)
    assert first != _due_sequence(0xBEEF), "a different seed should schedule differently"
    assert len(set(first)) > 1, "jitter should actually vary between draws"


def test_jitter_only_ever_delays_a_poll() -> None:
    interval = 300.0
    for seed in (0, 1, DEFAULT_SEED, 2 ** 63):
        for gap in _due_sequence(seed, steps=12):
            assert interval <= gap < interval * 1.10 + 1e-9, (seed, gap)


def test_jitter_differs_between_listings_of_one_host() -> None:
    scheduler, clock = make_scheduler()
    gaps = []
    for product in ("alpha-etb", "beta-box", "gamma-tin"):
        sku = sku_of(scheduler.catalog, "alpha", product)
        scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), clock.now)
        state = scheduler.stats(clock.now)["skus"]
        gaps.append([row["next_due_at"] for row in state
                     if row["source"] == "alpha" and row["product_id"] == product][0] - clock.now)
    assert len(set(gaps)) == 3, "listings of one host must not stay in lockstep"


def test_zero_jitter_is_allowed_and_exact() -> None:
    scheduler, clock = make_scheduler(jitter_fraction=0.0)
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), clock.now)
    assert scheduler.effective_due_at(sku) == clock.now + 300.0
    with pytest.raises(SchedulerError):
        make_scheduler(jitter_fraction=-0.1)


def test_a_bad_clock_or_store_is_refused_at_construction() -> None:
    with pytest.raises(SchedulerError):
        PollScheduler(make_catalog(), make_policies(), clock="not callable")
    with pytest.raises(SchedulerError):
        PollScheduler(make_catalog(), make_policies(), SimClock(), store=object())


def test_state_survives_a_restart_through_the_store() -> None:
    store = MemoryPollStore()
    catalog = make_catalog()
    clock = SimClock()
    first = PollScheduler(catalog, make_policies(), clock, store)
    sku = sku_of(catalog, "alpha", "alpha-etb")
    first.poll_once(sku, RecordingFetcher(
        default=FetchResult(ok=True, status=200, body="x", etag='W/"v9"'), clock=clock),
        simple_parser, clock.now)
    first.record_attempt(sku_of(catalog, "bravo", "beta-box"),
                         FetchResult(ok=False, status=500, reason="boom"), clock.now)
    assert store.saves > 0

    # A restart must not forget that a host was just looked at.
    second = PollScheduler(catalog, make_policies(), clock, store)
    assert all(s.source != "alpha" for s in second.due(clock.now))
    assert second.conditional_headers(sku)["If-None-Match"] == 'W/"v9"'
    assert second.stats(clock.now)["sources"]["bravo"]["consecutive_errors"] == 1
    assert second.effective_due_at(sku) == first.effective_due_at(sku)

    # And the jitter sequence continues where it left off rather than
    # restarting: the draw counter is part of the snapshot.
    clock.tick(5000.0)
    first_gap = _next_gap(first, sku, clock.now)
    second_gap = _next_gap(second, sku, clock.now)
    assert first_gap == second_gap


def _next_gap(scheduler: PollScheduler, sku: SourceSku, now: float) -> float:
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, body="x"), now)
    return scheduler.effective_due_at(sku) - now


def test_snapshot_round_trips_and_refuses_a_future_version() -> None:
    scheduler, clock = make_scheduler()
    sku = sku_of(scheduler.catalog, "alpha", "alpha-etb")
    scheduler.record_attempt(sku, FetchResult(ok=True, status=200, etag='W/"v1"'), clock.now)
    snapshot = json.loads(json.dumps(scheduler.snapshot()))

    other, _ = make_scheduler()
    other.restore(snapshot)
    assert other.snapshot() == scheduler.snapshot()

    with pytest.raises(SchedulerError):
        other.restore(dict(snapshot, version=999))


def test_pause_state_and_stats_are_json_ready_and_explain_themselves() -> None:
    scheduler, clock = make_scheduler()
    scheduler.record_attempt(sku_of(scheduler.catalog, "alpha", "alpha-etb"),
                             FetchResult(ok=True, status=200, body="x"), clock.now)
    pause_state = scheduler.pause_state(clock.now)
    stats = scheduler.stats(clock.now)
    json.dumps(pause_state)     # must not raise
    json.dumps(stats)

    assert pause_state["nogo"]["pollable"] is False
    assert pause_state["nogo"]["blocked_reason"] == "robots.txt disallows"
    assert pause_state["alpha"]["pollable"] is True
    assert stats["totals"]["attempts"] == 1
    assert stats["totals"]["skus"] == len(scheduler.catalog.skus())
    assert stats["totals"]["disallowed_sources"] == 1
    assert stats["next_due_at"] is not None
    rows = {(row["source"], row["product_id"]): row for row in stats["skus"]}
    assert rows[("alpha", "alpha-etb")]["due"] is False
    assert rows[("nogo", "alpha-etb")]["status"].startswith("robots")


def test_pause_state_uses_the_injected_clock_when_now_is_omitted() -> None:
    scheduler, clock = make_scheduler()
    scheduler.pause_source("alpha", clock.now + 100.0)
    assert scheduler.pause_state()["alpha"]["paused"] is True
    clock.tick(101.0)
    assert scheduler.pause_state()["alpha"]["paused"] is False


# ==========================================================================
# the property test: 500 ticks, no host ever polled too fast
# ==========================================================================


def test_no_host_is_ever_polled_faster_than_its_min_interval() -> None:
    """500 simulated ticks against a flaky fake host.

    Outcomes come from a seeded ``lucifer_gen`` stream, so a failure here
    is reproducible; nothing sleeps and nothing reaches the network. The
    assertion is the one promise contracts.py makes to the retailer: two
    fetches of one host are never closer together than that host's
    ``min_interval_s``.
    """
    catalog = make_catalog()
    policies = make_policies()
    clock = SimClock()
    scheduler = PollScheduler(catalog, policies, clock, MemoryPollStore(), seed=0xD15EA5E)
    rolls = SeedFields.parse(0xD15EA5E).stream("poke.test.sim")

    fetched_at: Dict[str, List[float]] = {}
    per_sku: Dict[Tuple[str, str], List[float]] = {}

    def fetcher(url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
        fetched_at.setdefault(policy.source, []).append(clock.now)
        roll = rolls.randint(1, 100)
        if roll <= 10:
            raise RuntimeError("flaky host")
        if roll <= 25:
            return FetchResult(ok=False, status=503, reason="busy")
        if roll <= 30:
            return FetchResult(ok=False, status=429, retry_after_s=1200.0, reason="slow down")
        if roll <= 55 and "If-None-Match" in headers:
            return FetchResult(ok=True, status=304, not_modified=True)
        return FetchResult(ok=True, status=200, body="x", etag=f'W/"{roll}"')

    def parser(sku: SourceSku, body: str, at: float) -> Observation:
        if rolls.randint(1, 100) <= 5:
            raise ValueError("unparseable page")
        return simple_parser(sku, body, at)

    polls = 0
    for _ in range(500):
        now = clock.now
        due = scheduler.due(now)
        assert len({sku.source for sku in due}) == len(due), "two listings of one host at once"
        for sku in due:
            allowed, reason = scheduler.can_poll(sku, now)
            assert allowed, reason
            scheduler.poll_once(sku, fetcher, parser, now)
            per_sku.setdefault((sku.source, sku.product_id), []).append(now)
            polls += 1
        clock.tick(45.0)

    assert polls > 50, "the simulation should have done real work"
    assert "nogo" not in fetched_at, "robots.txt disallowed this host for the whole run"

    for source, times in fetched_at.items():
        floor = policies[source].min_interval_s
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert all(gap >= floor for gap in gaps), (
            f"{source}: polled after {min(gaps):.0f}s, floor is {floor:.0f}s")

    for (source, product_id), times in per_sku.items():
        floor = policies[source].min_interval_s
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert all(gap >= floor for gap in gaps), (source, product_id, min(gaps))

    # Everything polled was pollable, and the counters add up.
    totals = scheduler.stats(clock.now)["totals"]
    assert totals["attempts"] == sum(len(t) for t in fetched_at.values())
    assert totals["ok"] + totals["not_modified"] + totals["errors"] >= totals["attempts"]


def test_the_simulation_is_reproducible() -> None:
    def run() -> Dict[str, Any]:
        catalog = make_catalog()
        clock = SimClock()
        scheduler = PollScheduler(catalog, make_policies(), clock, seed=0x5EED)
        rolls = SeedFields.parse(0x5EED).stream("poke.test.replay")

        def fetcher(url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
            return (FetchResult(ok=False, status=500, reason="boom")
                    if rolls.randint(1, 4) == 1
                    else FetchResult(ok=True, status=200, body="x", etag='W/"e"'))

        order: List[Tuple[float, str, str]] = []
        for _ in range(200):
            for sku in scheduler.due(clock.now):
                order.append((clock.now, sku.source, sku.product_id))
                scheduler.poll_once(sku, fetcher, simple_parser, clock.now)
            clock.tick(60.0)
        return {"order": order, "stats": scheduler.stats(clock.now)["totals"]}

    assert run() == run()


# --------------------------------------------------------------------------
# the package keeps its hands off the network
# --------------------------------------------------------------------------


def test_the_modules_this_file_covers_contain_no_network_code() -> None:
    """contracts.py: "The package makes no network calls itself."

    An AST walk, not a grep, so a docstring that *explains* why urllib is
    absent does not read as a urllib import. Scoped to the two modules
    under test; a package-wide sweep belongs in the validation gate,
    where it will not trip over a sibling still being written.
    """
    import ast

    banned_modules = ("urllib", "socket", "http", "requests", "httpx", "ssl", "ftplib",
                      "telnetlib", "asyncio", "subprocess")
    banned_calls = {("time", "time"), ("time", "monotonic"), ("random", "random"),
                    ("random", "randint"), ("random", "choice"), ("random", "uniform")}

    for name in ("catalog.py", "sources.py"):
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
                        f"{name} calls {owner.id}.{node.func.attr}(): the clock is injected "
                        f"and randomness comes from a lucifer_gen seed stream")


# ==========================================================================
# the gate under numbers that compare False, and under overlapping runners
#
# Each of these reproduced before the fix, and each one turns the whole
# politeness layer off with no error and no warning, which is the worst
# shape a politeness bug can have.
# ==========================================================================


NAN = float("nan")


def test_a_nan_interval_is_refused_rather_than_removing_the_gate() -> None:
    """Every comparison against NaN is False.

    ``min_interval_s = NaN`` used to pass ``< 30.0`` and ``< floor``,
    make ``effective_due_at`` NaN, and make ``now < due_at`` False for
    ever -- one fetch per tick against a documented 300s floor.
    """
    with pytest.raises(ValueError):
        FetchPolicy("alpha", min_interval_s=NAN)
    with pytest.raises(ValueError):
        FetchPolicy("alpha", min_interval_s=float("inf"))


def test_a_nan_pause_is_refused_rather_than_removing_the_stop() -> None:
    with pytest.raises(ValueError):
        FetchPolicy("alpha", pause_s=NAN)


def test_a_nan_jitter_fraction_is_refused() -> None:
    with pytest.raises(SchedulerError):
        PollScheduler(make_catalog(), make_policies(), SimClock(), jitter_fraction=NAN)
    # the negative guard this one used to walk straight past still holds
    with pytest.raises(SchedulerError):
        PollScheduler(make_catalog(), make_policies(), SimClock(), jitter_fraction=-0.1)


def test_a_config_file_may_not_smuggle_nan_past_the_floor(tmp_path: Path) -> None:
    """``json.load`` accepts bare ``NaN``, so a file was enough."""
    path = tmp_path / "sources.json"
    path.write_text(
        '{"sources": [{"id": "s1", "policy": {"min_interval_s": NaN}, "skus": []}]}',
        encoding="utf-8",
    )
    with pytest.raises(PolicyError) as caught:
        load_policies(path)
    assert "NaN" in str(caught.value)

    path.write_text(
        '{"sources": [{"id": "s1", "policy": {"pause_s": NaN}, "skus": []}]}',
        encoding="utf-8",
    )
    with pytest.raises(PolicyError):
        load_policies(path)


def test_an_http_date_retry_after_is_honoured() -> None:
    """HTTP allows a date as well as a number of seconds, and a host that
    sends the date form means it just as much."""
    import email.utils

    scheduler, clock = make_scheduler()
    catalog = make_catalog()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    when = email.utils.formatdate(clock.now + 86_400.0, usegmt=True)
    scheduler.record_attempt(
        sku, FetchResult(ok=False, status=429, retry_after_s=when), clock.now
    )
    allowed, reason = scheduler.can_poll(sku, clock.now + 3_600.0)
    assert not allowed and "paused" in reason
    assert scheduler.can_poll(sku, clock.now + 86_401.0)[0]


def test_an_unreadable_retry_after_pauses_rather_than_being_dropped() -> None:
    scheduler, clock = make_scheduler()
    catalog = make_catalog()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    scheduler.record_attempt(
        sku, FetchResult(ok=False, status=429, retry_after_s="whenever"), clock.now
    )
    state = scheduler.pause_state(clock.now + 1.0)["alpha"]
    assert state["paused"], "a host asked to be left alone and we could not read how long"
    assert "could not read" in state["reason"]


def test_resume_source_will_not_lift_a_pause_the_host_asked_for() -> None:
    """``Retry-After: 86400`` means a day.  Coming back in five minutes
    is the burst the header was sent to stop."""
    scheduler, clock = make_scheduler()
    catalog = make_catalog()
    sku = sku_of(catalog, "alpha", "alpha-etb")
    scheduler.record_attempt(
        sku, FetchResult(ok=False, status=429, retry_after_s=86_400.0), clock.now
    )
    with pytest.raises(SchedulerError) as caught:
        scheduler.resume_source("alpha")
    assert "the host asked" in str(caught.value)
    assert not scheduler.can_poll(sku, clock.now + 3_600.0)[0]

    # a person may still overrule it, but has to say so
    scheduler.resume_source("alpha", force=True)
    assert scheduler.can_poll(sku, clock.now + 3_600.0)[0]

    # a pause set by hand is still ours to lift
    scheduler.pause_source("alpha", clock.now + 10_000.0)
    scheduler.resume_source("alpha")
    assert scheduler.can_poll(sku, clock.now + 3_600.0)[0]


def test_two_overlapping_pollers_over_one_store_fetch_a_host_once() -> None:
    """The gate is read-decide-fetch-write, and nothing used to hold it
    still in between: N overlapping ``poll --once`` runs each saw the
    host as due and each fetched it, N times the agreed rate arriving as
    one burst."""
    store = MemoryPollStore()
    catalog = make_catalog()
    clock = SimClock()
    fetcher = RecordingFetcher(clock=clock)
    runners = [
        PollScheduler(catalog, make_policies(), clock, store) for _ in range(6)
    ]
    sku = sku_of(catalog, "alpha", "alpha-etb")
    for runner in runners:
        runner.poll_once(sku, fetcher, simple_parser, clock.now)
    assert len(fetcher.calls) == 1, (
        f"{len(fetcher.calls)} fetches of one host in one instant"
    )
    # and the refusals were counted, not swallowed
    assert store.snapshot["sources"]["alpha"]["refusals"] == 5


def test_a_slot_is_claimed_before_the_fetch_not_after_it() -> None:
    """The claim has to be saved before the network call, or a second
    runner that looks while the first is waiting is handed the same
    slot."""
    store = MemoryPollStore()
    catalog = make_catalog()
    clock = SimClock()
    one = PollScheduler(catalog, make_policies(), clock, store)
    two = PollScheduler(catalog, make_policies(), clock, store)
    sku = sku_of(catalog, "alpha", "alpha-etb")
    seen: List[bool] = []

    interloper = RecordingFetcher(clock=clock)

    def slow_fetcher(url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
        # Mid-flight.  The other runner re-reads the stored schedule, so
        # the claim has to be *saved* before this call, not after it --
        # ``poll_once`` swallows a fetcher's exception, so the proof is
        # that the second fetcher was never called at all.
        seen.append(two.poll_once(sku, interloper, simple_parser, clock.now) is None)
        return FetchResult(ok=True, status=200, body="body")

    one.poll_once(sku, slow_fetcher, simple_parser, clock.now)
    assert interloper.calls == [], (
        "the host was fetched twice: the claim was not saved before the first call"
    )
    assert seen == [True]


def test_two_source_ids_on_one_host_are_named_rather_than_hidden() -> None:
    """``min_interval_s`` belongs to the host; the policy belongs to the
    source id.  Two ids on one host means that host is polled once per
    id, at twice the rate either of them promises, with nothing in the
    schedule looking wrong."""
    skus = [
        SourceSku(source=source, product_id=product.id,
                  sku=f"{source}-{product.id}",
                  url=f"https://onehost.example.com/p/{product.id}")
        for source in ("shopa", "shopb")
        for product in PRODUCTS
    ]
    catalog = Catalog(PRODUCTS, skus)
    policies = {
        "shopa": FetchPolicy("shopa", min_interval_s=300.0),
        "shopb": FetchPolicy("shopb", min_interval_s=300.0),
    }
    scheduler = PollScheduler(catalog, policies, SimClock())
    assert scheduler.host_conflicts() == {"onehost.example.com": ["shopa", "shopb"]}
    assert scheduler.stats(0.0)["host_conflicts"]

    # the shipped configuration has none
    from jarvis_poke.catalog import Catalog as ShippedCatalog

    shipped = PollScheduler(ShippedCatalog.load(), load_policies(), SimClock())
    assert shipped.host_conflicts() == {}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
