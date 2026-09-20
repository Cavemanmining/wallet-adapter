"""Tests for jarvis_poke's durable state and its bridge to jarvis_alerts.

Design: jarvis_poke/contracts.py -- "Money is integer cents" (every write
here is asserted to refuse a float), the politeness constraints whose state
``source_state`` / ``sku_state`` carry across a restart, and
:class:`~jarvis_poke.contracts.Verdict`, whose BUY is the only action
:mod:`jarvis_poke.alerts_bridge` is allowed to interrupt someone about.

The shape of this file follows what the two modules promise:

* what goes in comes out identical (``round_trip_equal``, which reports the
  first differing field rather than dumping two objects side by side);
* a float price never reaches a money column, because sqlite would take the
  truncation silently and the owner would find out at a checkout;
* the verdict log cannot be rewritten -- the tests aim ``UPDATE`` and
  ``DELETE`` at it directly and expect sqlite to refuse;
* a save that dies halfway leaves the previous state, proved by breaking
  the per-table insert seam on purpose;
* the bridge alerts for BUY and for nothing else, once per landed price,
  and never carries a credential.

Nothing here touches the network, sleeps, or reads a real clock: the store
and the bridge both take an injected clock, and the alert service is a
fake that records what it was handed.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Runnable as `pytest tests/test_poke_store.py` or
# `python3 tests/test_poke_store.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_alerts.contracts import Priority

from jarvis_poke.alerts_bridge import (
    ALERT_KIND,
    DATA_KEYS,
    AlertBridge,
    BridgeError,
    dedupe_key_for,
)
from jarvis_poke.contracts import (
    Action,
    Budget,
    Observation,
    Product,
    ProductKind,
    Rule,
    SourceSku,
    Stock,
    Verdict,
    WatchState,
)
from jarvis_poke.store import (
    DB_FILE_MODE,
    SCHEMA_VERSION,
    PokeStore,
    SchemaError,
    StoreError,
    assert_round_trip_equal,
    first_difference,
    round_trip_equal,
)

T0 = 1_700_000_000.0
DAY = 86400.0


class SimClock:
    """A clock the test moves by hand.  Nothing in the package may read a
    real one (contracts.py: the clock is injected)."""

    def __init__(self, now: float = T0) -> None:
        self.now = float(now)

    def __call__(self) -> float:
        return self.now

    def tick(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


# --------------------------------------------------------------------------
# Fixtures: placeholder retailers only, example.com urls only
# --------------------------------------------------------------------------

PRODUCTS = [
    Product(
        id="sv08-surging-sparks-etb",
        name="Placeholder Elite Trainer Box",
        set_code="SV08",
        kind=ProductKind.ELITE_TRAINER_BOX,
        msrp=4999,
        released="2024-11-08",
        upc="0123456789012",
    ),
    Product(
        id="sv035-151-booster-bundle",
        name="Placeholder Booster Bundle",
        set_code="SV03.5",
        kind=ProductKind.BUNDLE,
        msrp=None,
        released=None,
        upc=None,
    ),
]

SKUS = [
    SourceSku(
        source="examplemart",
        product_id="sv08-surging-sparks-etb",
        sku="EM-SV08-ETB",
        url="https://examplemart.example.com/p/sv08-surging-sparks-etb",
    ),
    SourceSku(
        source="cardbarn",
        product_id="sv08-surging-sparks-etb",
        sku="CB-SV08-ETB",
        url="https://cardbarn.example.com/item/sv08-etb",
    ),
    SourceSku(
        source="examplemart",
        product_id="sv035-151-booster-bundle",
        sku="EM-SV035-BUNDLE",
        url="https://examplemart.example.com/p/sv035-151-booster-bundle",
    ),
]


def observation(
    at: float,
    *,
    product_id: str = "sv08-surging-sparks-etb",
    source: str = "examplemart",
    sku: str = "EM-SV08-ETB",
    price: Optional[int] = 4499,
    stock: Stock = Stock.IN_STOCK,
    shipping: int = 599,
    limit: Optional[int] = 2,
    note: str = "",
) -> Observation:
    return Observation(
        product_id=product_id,
        source=source,
        sku=sku,
        at=at,
        stock=stock,
        price=price,
        shipping=shipping,
        per_customer_limit=limit,
        url=f"https://{source}.example.com/p/{product_id}",
        note=note,
    )


def buy_verdict(
    *,
    at: float = T0,
    landed: Optional[int] = 5098,
    price: Optional[int] = 4499,
    action: Action = Action.BUY,
    source: str = "examplemart",
    market: Optional[int] = 6200,
) -> Verdict:
    return Verdict(
        product_id="sv08-surging-sparks-etb",
        action=action,
        at=at,
        source=source,
        sku="EM-SV08-ETB",
        price=price,
        landed=landed,
        market=market,
        discount_pct=17.8,
        quantity=2,
        url="https://examplemart.example.com/p/sv08-surging-sparks-etb",
        reasons=(
            "in stock at examplemart",
            "landed $50.98 is at or under the $54.99 cap",
            "17.8% under the $62.00 market reference",
            "within the remaining budget",
        ),
    )


@pytest.fixture()
def clock() -> SimClock:
    return SimClock()


@pytest.fixture()
def store(tmp_path: Path, clock: SimClock):
    with PokeStore(tmp_path / "poke.sqlite3", clock=clock) as opened:
        yield opened


@pytest.fixture()
def stocked(store: PokeStore) -> PokeStore:
    """A store with the catalog in it, which the listing table needs."""
    store.save_catalog(PRODUCTS, SKUS)
    return store


# --------------------------------------------------------------------------
# The file itself
# --------------------------------------------------------------------------


def test_database_file_is_private(tmp_path: Path, clock: SimClock) -> None:
    """0600, like the alert outbox: a watchlist is a list of what its owner
    is about to spend money on."""
    path = tmp_path / "poke.sqlite3"
    with PokeStore(path, clock=clock):
        pass
    assert os.stat(path).st_mode & 0o777 == DB_FILE_MODE
    assert DB_FILE_MODE == 0o600


def test_wal_and_foreign_keys_and_version(store: PokeStore) -> None:
    assert store.schema_version() == SCHEMA_VERSION
    (mode,) = store._conn.execute("PRAGMA journal_mode").fetchone()
    assert str(mode).lower() == "wal"
    (fk,) = store._conn.execute("PRAGMA foreign_keys").fetchone()
    assert int(fk) == 1


def test_foreign_key_is_enforced_not_just_enabled(store: PokeStore) -> None:
    """A listing for a product nobody stored is a poll that can never
    produce a usable observation, so the write is refused."""
    store.save_products(PRODUCTS)
    orphan = SourceSku(
        source="examplemart",
        product_id="not-in-the-catalog",
        sku="EM-GHOST",
        url="https://examplemart.example.com/p/ghost",
    )
    with pytest.raises(StoreError):
        store.save_source_skus(list(SKUS) + [orphan])
    assert len(store.load_source_skus()) == 0


def test_a_second_open_of_the_same_file_reads_the_same_state(
    tmp_path: Path, clock: SimClock
) -> None:
    path = tmp_path / "poke.sqlite3"
    with PokeStore(path, clock=clock) as first:
        first.save_catalog(PRODUCTS, SKUS)
        first.append_verdict(buy_verdict())
    with PokeStore(path, clock=clock) as second:
        assert_round_trip_equal(sorted(PRODUCTS, key=lambda p: p.id), second.load_products())
        assert len(second.verdicts()) == 1


def test_a_foreign_schema_version_is_refused(tmp_path: Path, clock: SimClock) -> None:
    path = tmp_path / "poke.sqlite3"
    with PokeStore(path, clock=clock):
        pass
    conn = sqlite3.connect(path)
    conn.execute("UPDATE schema_version SET version = 99 WHERE id = 1")
    conn.commit()
    conn.close()
    with pytest.raises(SchemaError):
        PokeStore(path, clock=clock)


def test_store_refuses_to_be_built_without_a_clock(tmp_path: Path) -> None:
    with pytest.raises(StoreError):
        PokeStore(tmp_path / "poke.sqlite3", clock=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------
# Round trips: everything the store holds
# --------------------------------------------------------------------------


def test_products_round_trip(store: PokeStore) -> None:
    store.save_products(PRODUCTS)
    back = store.load_products()
    assert_round_trip_equal(sorted(PRODUCTS, key=lambda p: p.id), back)
    by_id = {p.id: p for p in back}
    etb = by_id["sv08-surging-sparks-etb"]
    assert etb.msrp == 4999 and isinstance(etb.msrp, int)
    bundle = by_id["sv035-151-booster-bundle"]
    assert bundle.msrp is None and bundle.released is None and bundle.upc is None


def test_source_skus_round_trip(stocked: PokeStore) -> None:
    back = stocked.load_source_skus()
    assert_round_trip_equal(sorted(SKUS, key=lambda s: (s.source, s.product_id)), back)
    assert [s.source for s in stocked.load_source_skus("examplemart")] == [
        "examplemart",
        "examplemart",
    ]


def test_observations_round_trip_including_the_unparseable_one(stocked: PokeStore) -> None:
    """An out-of-stock look has no price; contracts.py says ``None`` there,
    and ``None`` is what has to come back -- not 0, which would poison the
    market reference."""
    written = [
        observation(T0 - 2 * DAY),
        observation(T0 - DAY, price=None, stock=Stock.OUT_OF_STOCK, limit=None),
        observation(T0, stock=Stock.LIMITED, note="only a few left"),
    ]
    assert stocked.save_observations(written) == 3
    back = stocked.load_observations("sv08-surging-sparks-etb")
    assert_round_trip_equal(written, back)
    assert back[1].price is None and back[1].landed is None
    assert back[0].landed == 4499 + 599


def test_rules_round_trip_including_one_for_an_unknown_product(store: PokeStore) -> None:
    """The engine SKIPs a rule whose product the catalog does not know, with
    a reason.  That only works if such a rule can be stored at all, so
    ``rules`` deliberately has no foreign key."""
    rules = [
        Rule(
            product_id="sv08-surging-sparks-etb",
            max_price=5499,
            quantity=2,
            min_discount_pct=10.0,
            allowed_sources=("examplemart", "cardbarn"),
            include_shipping=True,
            cooldown_s=7200.0,
            enabled=True,
        ),
        Rule(product_id="a-product-that-drifted", max_price=1, enabled=False),
    ]
    store.save_rules(rules)
    back = store.load_rules()
    assert_round_trip_equal(sorted(rules, key=lambda r: r.product_id), back)
    drifted, kept = back
    assert drifted.product_id == "a-product-that-drifted" and drifted.allowed_sources == ()
    assert kept.allowed_sources == ("examplemart", "cardbarn") and kept.enabled is True


def test_budget_with_spend_round_trips(store: PokeStore) -> None:
    budget = Budget(total=250_000, spent=137_450, window_s=14 * DAY)
    store.save_budget(budget)
    back = store.load_budget()
    assert_round_trip_equal(budget, back)
    assert back.remaining == 112_550
    assert isinstance(back.total, int) and isinstance(back.spent, int)


def test_rule_set_saves_rules_and_budget_as_one_unit(store: PokeStore) -> None:
    from jarvis_poke.rules import RuleSet

    rule_set = RuleSet(
        [Rule(product_id="sv08-surging-sparks-etb", max_price=5499)],
        Budget(total=100_000, spent=25_000),
    )
    store.save_rule_set(rule_set)
    back = store.load_rule_set()
    assert_round_trip_equal(rule_set.rules(), back.rules())
    assert_round_trip_equal(rule_set.budget, back.budget)
    assert back.budget.remaining == 75_000


def test_watch_states_round_trip(store: PokeStore) -> None:
    states = {
        "sv08-surging-sparks-etb": WatchState(
            product_id="sv08-surging-sparks-etb",
            last_alert_at=T0 - 900.0,
            last_action=Action.BUY,
            alerts_sent=3,
            last_seen_in_stock=T0 - 60.0,
        ),
        "sv035-151-booster-bundle": WatchState(product_id="sv035-151-booster-bundle"),
    }
    store.save_watch_states(states)
    assert_round_trip_equal(states, store.load_watch_states())
    assert store.load_watch_states()["sv035-151-booster-bundle"].last_action is None


def test_verdict_with_reasons_round_trips(store: PokeStore) -> None:
    verdict = buy_verdict()
    store.append_verdict(verdict)
    (back,) = store.verdicts()
    assert_round_trip_equal(verdict, back)
    assert back.reasons == verdict.reasons and isinstance(back.reasons, tuple)
    assert back.should_alert is True


def test_a_verdict_with_nothing_set_round_trips(store: PokeStore) -> None:
    """NO_STOCK carries no price, no source and no sku.  Those have to come
    back as ``None``, not as empty strings."""
    verdict = Verdict(
        product_id="sv035-151-booster-bundle",
        action=Action.NO_STOCK,
        at=T0,
        reasons=("nothing in stock at any allowed source",),
    )
    store.append_verdict(verdict)
    assert_round_trip_equal(verdict, store.verdicts()[0])


def test_poll_state_round_trips_with_its_validators(store: PokeStore) -> None:
    """The conditional validators are the politeness that survives a
    restart: lose the ETag and every page is downloaded again."""
    snapshot = {
        "version": 1,
        "sources": {
            "examplemart": {
                "source": "examplemart",
                "last_attempt_at": T0 - 120.0,
                "next_due_at": T0 + 180.0,
                "paused_until": T0 + 3600.0,
                "pause_reason": "5 errors in a row",
                "consecutive_errors": 5,
                "attempts": 41,
                "ok": 33,
                "not_modified": 3,
                "errors": 5,
                "parse_errors": 0,
                "refusals": 2,
                "pauses": 1,
                "observations": 33,
                "last_status": 503,
                "last_reason": "upstream busy",
                "last_error_at": T0 - 120.0,
            }
        },
        "skus": [
            {
                "source": "examplemart",
                "product_id": "sv08-surging-sparks-etb",
                "last_attempt_at": T0 - 120.0,
                "next_due_at": T0 + 180.0,
                "etag": 'W/"7-abc"',
                "last_modified": "Wed, 21 Oct 2024 07:28:00 GMT",
                "attempts": 21,
                "errors": 1,
                "consecutive_errors": 0,
                "not_modified": 9,
                "observations": 11,
                "last_status": 304,
                "last_outcome": "not_modified",
                "draws": 21,
            }
        ],
    }
    store.save_poll_snapshot(snapshot)
    back = store.load_poll_snapshot()
    assert_round_trip_equal(snapshot, back)
    assert back["skus"][0]["etag"] == 'W/"7-abc"'
    assert back["sources"]["examplemart"]["paused_until"] == T0 + 3600.0


def test_no_poll_state_reads_as_none_not_as_empty(store: PokeStore) -> None:
    """``PollScheduler`` restores only a truthy snapshot; "nothing yet" has
    to be ``None`` or a fresh file looks like a scheduler with no hosts."""
    assert store.load_poll_snapshot() is None


def test_poll_state_survives_a_field_this_schema_does_not_know(store: PokeStore) -> None:
    """A field a future ``jarvis_poke.sources`` grows rides in ``extra``
    rather than being dropped on the floor."""
    snapshot = {
        "version": 1,
        "sources": {"examplemart": {"source": "examplemart", "a_new_counter": 7}},
        "skus": [],
    }
    store.save_poll_snapshot(snapshot)
    back = store.load_poll_snapshot()
    assert back["sources"]["examplemart"]["a_new_counter"] == 7
    assert back["sources"]["examplemart"]["paused_until"] == 0.0


def test_typed_source_and_sku_state_round_trip(store: PokeStore) -> None:
    from jarvis_poke.sources import SkuState, SourceState

    sources = {
        "cardbarn": SourceState(
            source="cardbarn",
            last_attempt_at=T0,
            paused_until=T0 + 60.0,
            pause_reason="told to slow down",
            consecutive_errors=2,
            errors=2,
        )
    }
    skus = {
        ("cardbarn", "sv08-surging-sparks-etb"): SkuState(
            source="cardbarn",
            product_id="sv08-surging-sparks-etb",
            etag='W/"42"',
            last_modified=None,
            draws=4,
        )
    }
    store.save_source_states(sources)
    store.save_sku_states(skus)
    assert_round_trip_equal(sources, store.load_source_states())
    assert_round_trip_equal(skus, store.load_sku_states())


def test_the_scheduler_can_use_the_store_as_its_poll_store(
    tmp_path: Path, clock: SimClock
) -> None:
    """The integration that matters: a restart stays as polite as it was.

    A real ``PollScheduler`` polls once through an injected fetcher, saves
    into this file, and a second scheduler built on the same file sends the
    validator back as ``If-None-Match`` and refuses to poll inside the
    host's interval.
    """
    from jarvis_poke.catalog import Catalog
    from jarvis_poke.contracts import FetchPolicy, FetchResult
    from jarvis_poke.sources import PollScheduler

    catalog = Catalog(PRODUCTS, SKUS)
    policies = [FetchPolicy(source=name, min_interval_s=300.0) for name in ("examplemart", "cardbarn")]
    sku = catalog.sku("examplemart", "sv08-surging-sparks-etb")
    seen: List[Dict[str, str]] = []

    def fetcher(url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
        seen.append(dict(headers))
        return FetchResult(ok=True, status=200, body="<fake/>", etag='W/"77"')

    def parser(sku_arg: SourceSku, body: str, at: float) -> Observation:
        return observation(at, source=sku_arg.source, sku=sku_arg.sku, price=4499)

    with PokeStore(tmp_path / "poke.sqlite3", clock=clock) as store:
        first = PollScheduler(catalog, policies, clock, store.poll_store())
        assert first.poll_once(sku, fetcher, parser, clock.now) is not None

        clock.tick(60.0)  # well inside the 300s floor
        second = PollScheduler(catalog, policies, clock, store.poll_store())
        allowed, reason = second.can_poll(sku, clock.now)
        assert allowed is False, "a restart must not reset the per-host rate gate"
        assert second.conditional_headers(sku)["If-None-Match"] == 'W/"77"'

        clock.tick(600.0)
        assert second.poll_once(sku, fetcher, parser, clock.now) is not None
    assert seen[0] == {k: v for k, v in seen[0].items() if k == "User-Agent"}
    assert seen[1]["If-None-Match"] == 'W/"77"'


def test_first_difference_names_the_field(store: PokeStore) -> None:
    a = buy_verdict()
    b = buy_verdict(landed=5099)
    assert round_trip_equal(a, a) is True
    assert round_trip_equal(a, b) is False
    assert first_difference(a, b) == "value.landed: 5098 != 5099"
    reworded = dataclasses.replace(
        a, reasons=a.reasons[:2] + ("something else",) + a.reasons[3:]
    )
    assert first_difference(a, reworded).startswith("value.reasons[2]: ")


def test_round_trip_comparison_does_not_confuse_an_int_with_a_float() -> None:
    """The one place this comparison is stricter than its cousin in
    lucifer_descent: money is integer cents, so 4999 and 4999.0 differ."""
    assert round_trip_equal(Budget(total=4999), Budget(total=4999.0)) is False
    assert "int != float" in first_difference(Budget(total=4999), Budget(total=4999.0))


# --------------------------------------------------------------------------
# Money is integer cents, enforced at the write
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(
            lambda s: s.save_products(
                [Product(id="p", name="n", set_code="S", kind=ProductKind.TIN, msrp=1999.0)]
            ),
            id="product.msrp",
        ),
        pytest.param(
            lambda s: s.save_observations([observation(T0, price=4499.0)]),
            id="observation.price",
        ),
        pytest.param(
            lambda s: s.save_observations([observation(T0, shipping=5.99)]),
            id="observation.shipping",
        ),
        pytest.param(
            lambda s: s.save_rules([Rule(product_id="p", max_price=54.99)]),
            id="rule.max_price",
        ),
        pytest.param(lambda s: s.save_budget(Budget(total=250.0)), id="budget.total"),
        pytest.param(
            lambda s: s.save_budget(Budget(total=25000, spent=137.45)), id="budget.spent"
        ),
        pytest.param(
            lambda s: s.append_verdict(buy_verdict(landed=50.98)), id="verdict.landed"
        ),
        pytest.param(
            lambda s: s.append_verdict(buy_verdict(landed=None, price=44.99)),
            id="verdict.price",
        ),
    ],
)
def test_float_money_is_refused_on_the_way_in(store: PokeStore, make) -> None:
    """contracts.py: "a float budget is how you end up buying something for
    a cent more than the cap".  The column is INTEGER, so sqlite would
    round the float away in silence; this refuses it instead."""
    with pytest.raises(StoreError) as caught:
        make(store)
    assert "integer cents" in str(caught.value)


def test_a_float_that_looks_harmless_is_still_refused(store: PokeStore) -> None:
    """4999.0 is not 4999: it has been through a float, so it may already be
    4998.999999999999, and there is no way to tell from here."""
    with pytest.raises(StoreError):
        store.save_budget(Budget(total=4999.0))
    assert store.load_budget() is None


def test_nothing_float_valued_is_in_a_money_column_after_a_normal_save(
    stocked: PokeStore,
) -> None:
    stocked.save_observations([observation(T0)])
    stocked.save_rules([Rule(product_id="p", max_price=5499)])
    stocked.save_budget(Budget(total=100_000, spent=1))
    stocked.append_verdict(buy_verdict())
    money = {
        "products": ("msrp",),
        "observations": ("price", "shipping"),
        "rules": ("max_price",),
        "budget": ("total", "spent"),
        "verdicts": ("price", "landed", "market"),
    }
    for table, columns in money.items():
        for column in columns:
            for (value,) in stocked._conn.execute(
                f"SELECT {column} FROM {table} WHERE {column} IS NOT NULL"
            ):
                assert isinstance(value, int), f"{table}.{column} came back {value!r}"


# --------------------------------------------------------------------------
# The verdict log: append-only and ordered
# --------------------------------------------------------------------------


def test_the_verdict_log_is_ordered_and_queryable_by_product_and_time(
    store: PokeStore,
) -> None:
    for offset, action in enumerate(
        [Action.NO_STOCK, Action.WATCH, Action.BUY, Action.SKIP]
    ):
        store.append_verdict(buy_verdict(at=T0 + offset * 600.0, action=action))
    store.append_verdict(
        Verdict(product_id="sv035-151-booster-bundle", action=Action.WATCH, at=T0 + 60.0)
    )

    everything = store.verdicts()
    assert [v.at for v in everything] == sorted(v.at for v in everything)

    mine = store.verdicts("sv08-surging-sparks-etb")
    assert [v.action for v in mine] == [Action.NO_STOCK, Action.WATCH, Action.BUY, Action.SKIP]

    window = store.verdicts("sv08-surging-sparks-etb", since=T0 + 600.0, until=T0 + 1800.0)
    assert [v.action for v in window] == [Action.WATCH, Action.BUY]

    assert store.last_verdict("sv08-surging-sparks-etb").action is Action.SKIP
    assert [v.action for v in store.verdicts(actions=[Action.BUY])] == [Action.BUY]


def test_two_verdicts_in_the_same_instant_keep_their_order(store: PokeStore) -> None:
    """A pass over two sources stamps one clock reading; insertion order is
    the tie-break, so the log still reads as what happened."""
    for reason in ("first", "second", "third"):
        store.append_verdict(
            Verdict(product_id="p", action=Action.WATCH, at=T0, reasons=(reason,))
        )
    assert [v.reasons[0] for v in store.verdicts()] == ["first", "second", "third"]
    assert [v.reasons[0] for v in store.verdicts(newest_first=True)] == [
        "third",
        "second",
        "first",
    ]


def test_the_verdict_log_cannot_be_rewritten(store: PokeStore) -> None:
    """Append-only in sqlite, not in a docstring: this is the audit trail of
    a tool that tells its owner to spend money."""
    store.append_verdict(buy_verdict())
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("UPDATE verdicts SET price = 1 WHERE verdict_id = 1")
    with pytest.raises(sqlite3.DatabaseError):
        store._conn.execute("DELETE FROM verdicts")
    (back,) = store.verdicts()
    assert_round_trip_equal(buy_verdict(), back)


def test_the_store_offers_no_way_to_change_a_verdict(store: PokeStore) -> None:
    for forbidden in ("update_verdict", "delete_verdict", "prune_verdicts", "save_verdicts"):
        assert not hasattr(store, forbidden), f"PokeStore should not offer {forbidden}"


def test_appending_a_pass_is_all_or_nothing(store: PokeStore, monkeypatch) -> None:
    store.append_verdict(buy_verdict(at=T0))
    good = [buy_verdict(at=T0 + 1.0), buy_verdict(at=T0 + 2.0)]
    real = store._insert_verdict
    calls = {"n": 0}

    def flaky(cur, verdict):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the disk went away")
        return real(cur, verdict)

    monkeypatch.setattr(store, "_insert_verdict", flaky)
    with pytest.raises(RuntimeError):
        store.append_verdicts(good)
    assert len(store.verdicts()) == 1, "half a pass must not survive"


# --------------------------------------------------------------------------
# Pruning
# --------------------------------------------------------------------------


def test_pruning_keeps_the_window(stocked: PokeStore, clock: SimClock) -> None:
    for age in range(10):
        stocked.save_observations([observation(T0 - age * DAY, price=4400 + age)])
    assert len(stocked.load_observations()) == 10

    pruned = stocked.prune_observations(window_s=3 * DAY)
    kept = stocked.load_observations()
    assert pruned == 6
    assert [o.at for o in kept] == [T0 - 3 * DAY, T0 - 2 * DAY, T0 - DAY, T0]
    assert min(o.at for o in kept) >= clock.now - 3 * DAY


def test_pruning_spares_the_newest_look_at_every_listing(stocked: PokeStore) -> None:
    """That row is the current state of the listing; the scheduler reuses it
    on a 304.  Dropping it would make a quiet listing look unknown and pull
    a fresh fetch out of a host that had said nothing changed."""
    stocked.save_observations(
        [
            observation(T0 - 40 * DAY, source="cardbarn", sku="CB-SV08-ETB"),
            observation(T0 - 39 * DAY, source="cardbarn", sku="CB-SV08-ETB"),
            observation(T0 - DAY),
        ]
    )
    pruned = stocked.prune_observations(window_s=7 * DAY)
    assert pruned == 1
    survivors = {(o.source, o.at) for o in stocked.load_observations()}
    assert ("cardbarn", T0 - 39 * DAY) in survivors
    assert ("cardbarn", T0 - 40 * DAY) not in survivors

    hard = stocked.prune_observations(window_s=7 * DAY, keep_latest_per_listing=False)
    assert hard == 1
    assert [o.source for o in stocked.load_observations()] == ["examplemart"]


def test_pruning_takes_an_absolute_cutoff_too(stocked: PokeStore) -> None:
    for age in (0, 1, 2):
        stocked.save_observations([observation(T0 - age * DAY, price=4400 + age)])
    assert stocked.prune_observations(before=T0 - DAY, keep_latest_per_listing=False) == 1
    assert len(stocked.load_observations()) == 2  # the cutoff itself is kept
    with pytest.raises(StoreError):
        stocked.prune_observations()
    with pytest.raises(StoreError):
        stocked.prune_observations(before=T0, window_s=DAY)


def test_the_same_look_stored_twice_is_one_row(stocked: PokeStore) -> None:
    """A restart replaying its last poll must not double-count a price into
    the market reference."""
    look = observation(T0)
    assert stocked.save_observations([look, look]) == 1
    assert stocked.save_observation(look) == 0
    assert len(stocked.load_observations()) == 1


# --------------------------------------------------------------------------
# Atomicity: an interrupted save leaves the prior state
# --------------------------------------------------------------------------


def test_an_interrupted_product_save_leaves_the_previous_catalog(
    store: PokeStore, monkeypatch
) -> None:
    store.save_catalog(PRODUCTS, SKUS)
    before_products = store.load_products()
    before_skus = store.load_source_skus()

    real = store._insert_product
    calls = {"n": 0}

    def flaky(cur, product):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("the disk went away mid-save")
        return real(cur, product)

    monkeypatch.setattr(store, "_insert_product", flaky)
    replacement = [
        Product(id="brand-new", name="New", set_code="SV09", kind=ProductKind.TIN),
        Product(id="also-new", name="Also", set_code="SV09", kind=ProductKind.TIN),
    ]
    with pytest.raises(RuntimeError):
        store.save_products(replacement)

    assert_round_trip_equal(before_products, store.load_products())
    assert_round_trip_equal(before_skus, store.load_source_skus())


def test_an_interrupted_rule_set_save_leaves_rules_and_budget_together(
    store: PokeStore, monkeypatch
) -> None:
    """Rules and the ledger they spend from are one unit: a save that
    dropped half would let the monitor commit against the wrong ceiling."""
    from jarvis_poke.rules import RuleSet

    store.save_rule_set(
        RuleSet([Rule(product_id="p1", max_price=5000)], Budget(total=100_000, spent=1_000))
    )
    before_rules = store.load_rules()
    before_budget = store.load_budget()

    monkeypatch.setattr(
        store, "_insert_budget", lambda cur, budget: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        store.save_rule_set(
            RuleSet(
                [Rule(product_id="p2", max_price=9999)], Budget(total=1, spent=0)
            )
        )
    assert_round_trip_equal(before_rules, store.load_rules())
    assert_round_trip_equal(before_budget, store.load_budget())


def test_an_interrupted_poll_state_save_leaves_the_pause_in_place(
    store: PokeStore, monkeypatch
) -> None:
    """The worst possible half-write: a restart that forgot a pause and
    went back to hammering a host that had already told us to stop."""
    paused = {
        "version": 1,
        "sources": {
            "examplemart": {
                "source": "examplemart",
                "paused_until": T0 + 3600.0,
                "pause_reason": "5 errors in a row",
                "consecutive_errors": 5,
            }
        },
        "skus": [
            {"source": "examplemart", "product_id": "sv08-surging-sparks-etb", "etag": 'W/"9"'}
        ],
    }
    store.save_poll_snapshot(paused)
    before = store.load_poll_snapshot()

    monkeypatch.setattr(
        store, "_insert_sku_state", lambda cur, row: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    with pytest.raises(RuntimeError):
        store.save_poll_snapshot(
            {
                "version": 1,
                "sources": {"examplemart": {"source": "examplemart", "paused_until": 0.0}},
                "skus": [
                    {"source": "examplemart", "product_id": "sv08-surging-sparks-etb",
                     "etag": None}
                ],
            }
        )
    assert_round_trip_equal(before, store.load_poll_snapshot())
    assert store.load_poll_snapshot()["sources"]["examplemart"]["paused_until"] == T0 + 3600.0


def test_an_interrupted_watch_state_save_keeps_the_old_states(
    store: PokeStore, monkeypatch
) -> None:
    store.save_watch_states([WatchState(product_id="p1", alerts_sent=2, last_alert_at=T0)])
    before = store.load_watch_states()
    monkeypatch.setattr(
        store,
        "_insert_watch_state",
        lambda cur, state: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    with pytest.raises(RuntimeError):
        store.save_watch_states([WatchState(product_id="p2")])
    assert_round_trip_equal(before, store.load_watch_states())


# --------------------------------------------------------------------------
# The bridge to jarvis_alerts
# --------------------------------------------------------------------------

#: A stand-in for the one thing jarvis_alerts holds and never puts in a
#: message: a device's push subscription.  No published alert may contain
#: any of it.
SECRET_BLOB = '{"endpoint": "https://push.example.com/AAAA", "keys": {"auth": "s3cr3t-auth"}}'
SECRET_TOKEN = "s3cr3t-auth"


class FakeAlertService:
    """The surface of :class:`jarvis_alerts.api.AlertService` the bridge is
    allowed to use, plus a device registry it must never read."""

    def __init__(self) -> None:
        self.published: List[Dict[str, Any]] = []
        # Deliberately reachable: the bridge must not go looking.
        self.devices = {("owner", "phone"): SECRET_BLOB}
        self.api_key = "sk-live-not-a-real-key"

    def publish(
        self,
        profile_id: str,
        kind: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        priority: Any = Priority.NORMAL,
        dedupe_key: Optional[str] = None,
    ) -> str:
        self.published.append(
            {
                "profile_id": profile_id,
                "kind": kind,
                "title": title,
                "body": body,
                "data": dict(data or {}),
                "priority": priority,
                "dedupe_key": dedupe_key,
            }
        )
        return f"alert{len(self.published)}"


@pytest.fixture()
def service() -> FakeAlertService:
    return FakeAlertService()


@pytest.fixture()
def bridge(service: FakeAlertService, clock: SimClock) -> AlertBridge:
    return AlertBridge(service, clock)


def test_a_buy_publishes_exactly_one_alert(
    bridge: AlertBridge, service: FakeAlertService
) -> None:
    verdict = buy_verdict()
    alert_id = bridge.publish_verdict(verdict, PRODUCTS[0])

    assert alert_id == "alert1"
    assert len(service.published) == 1
    sent = service.published[0]
    assert sent["profile_id"] == "owner"
    assert sent["kind"] == ALERT_KIND
    assert sent["priority"] is Priority.HIGH
    assert PRODUCTS[0].name in sent["title"] and "$50.98" in sent["title"]
    assert "2 x" in sent["title"], "the owner is being told to buy two of them"


def test_the_body_is_the_engine_explanation_plus_the_source(
    bridge: AlertBridge, service: FakeAlertService
) -> None:
    from jarvis_poke.engine import explain

    verdict = buy_verdict()
    bridge.publish_verdict(verdict, PRODUCTS[0])
    body = service.published[0]["body"]
    assert body.startswith(explain(verdict))
    assert "examplemart" in body
    assert verdict.reasons[-1] in body, "the reason that decided it must survive"


def test_the_data_deep_links_to_the_product_page(
    bridge: AlertBridge, service: FakeAlertService
) -> None:
    verdict = buy_verdict()
    bridge.publish_verdict(verdict, PRODUCTS[0])
    data = service.published[0]["data"]

    assert set(data) == set(DATA_KEYS)
    assert data == {
        "product_id": "sv08-surging-sparks-etb",
        "source": "examplemart",
        "sku": "EM-SV08-ETB",
        "url": "https://examplemart.example.com/p/sv08-surging-sparks-etb",
        "price_cents": 5098,
        "quantity": 2,
        "market_cents": 6200,
    }
    assert isinstance(data["price_cents"], int), "money is integer cents, here too"
    assert data["url"] == verdict.url


@pytest.mark.parametrize("action", [Action.WATCH, Action.SKIP, Action.NO_STOCK])
def test_only_buy_is_ever_published(
    bridge: AlertBridge, service: FakeAlertService, action: Action
) -> None:
    """contracts.py gives BUY alone ``should_alert``.  A monitor that buzzes
    for "still out of stock" gets muted, and then it cannot do its job."""
    assert bridge.publish_verdict(buy_verdict(action=action), PRODUCTS[0]) is None
    assert service.published == []
    assert bridge.published == 0


def test_the_same_listing_at_the_same_price_alerts_once(
    bridge: AlertBridge, service: FakeAlertService, clock: SimClock
) -> None:
    """A restock flaps: in stock, out, in again.  That is one piece of news."""
    first = bridge.publish_verdict(buy_verdict(), PRODUCTS[0])
    for _ in range(4):
        clock.tick(30.0)
        again = bridge.publish_verdict(buy_verdict(at=clock.now), PRODUCTS[0])
        assert again == first, "a caller should never have to branch on a repeat"
    assert len(service.published) == 1
    assert bridge.suppressed == 4


def test_a_price_drop_alerts_again(
    bridge: AlertBridge, service: FakeAlertService, clock: SimClock
) -> None:
    """The first alert said $50.98 and the owner passed; $40.98 is a
    different decision."""
    bridge.publish_verdict(buy_verdict(), PRODUCTS[0])
    clock.tick(30.0)
    second = bridge.publish_verdict(buy_verdict(at=clock.now, landed=4098), PRODUCTS[0])

    assert second == "alert2"
    assert len(service.published) == 2
    keys = [p["dedupe_key"] for p in service.published]
    assert keys == [
        "poke:buy|sv08-surging-sparks-etb|examplemart|5098",
        "poke:buy|sv08-surging-sparks-etb|examplemart|4098",
    ]
    assert keys[0] != keys[1]


def test_the_same_price_from_another_source_is_its_own_alert(
    bridge: AlertBridge, service: FakeAlertService
) -> None:
    """Two retailers at the cap in one minute are two purchases with two
    different deep links."""
    bridge.publish_verdict(buy_verdict(), PRODUCTS[0])
    bridge.publish_verdict(buy_verdict(source="cardbarn"), PRODUCTS[0])
    assert len(service.published) == 2


def test_the_window_expires(
    bridge: AlertBridge, service: FakeAlertService, clock: SimClock
) -> None:
    bridge.publish_verdict(buy_verdict(), PRODUCTS[0])
    clock.tick(bridge.dedupe_window_s + 1.0)
    bridge.publish_verdict(buy_verdict(at=clock.now), PRODUCTS[0])
    assert len(service.published) == 2
    assert len(bridge._recent) == 1, "the memory of old keys is pruned"


def test_the_bridge_window_matches_the_outbox_window() -> None:
    from jarvis_alerts.outbox import DEDUPE_WINDOW_S as OUTBOX_WINDOW

    from jarvis_poke.alerts_bridge import DEDUPE_WINDOW_S as BRIDGE_WINDOW

    assert BRIDGE_WINDOW == OUTBOX_WINDOW


def test_no_credential_and_no_subscription_blob_ever_reaches_an_alert(
    bridge: AlertBridge, service: FakeAlertService, clock: SimClock
) -> None:
    """The bridge never reads the device registry, so nothing from it can
    appear in a message -- checked against the whole published alert, not
    just its data."""
    for landed in (5098, 4998, 4898):
        clock.tick(60.0)
        bridge.publish_verdict(buy_verdict(at=clock.now, landed=landed), PRODUCTS[0])
    assert len(service.published) == 3

    for sent in service.published:
        blob = json.dumps(sent, default=str)
        assert SECRET_TOKEN not in blob
        assert SECRET_BLOB not in blob
        assert "endpoint" not in blob
        assert service.api_key not in blob
        for key, value in sent["data"].items():
            assert value is None or isinstance(value, (str, int)), key
            assert not isinstance(value, dict), f"data[{key!r}] must not be a blob"


def test_a_verdict_paired_with_the_wrong_product_is_refused(bridge: AlertBridge) -> None:
    """An alert naming the wrong box is worse than no alert."""
    with pytest.raises(BridgeError):
        bridge.publish_verdict(buy_verdict(), PRODUCTS[1])


def test_a_buy_with_no_price_is_refused(bridge: AlertBridge) -> None:
    with pytest.raises(BridgeError):
        bridge.publish_verdict(buy_verdict(landed=None, price=None), PRODUCTS[0])


def test_a_float_price_never_becomes_an_alert(bridge: AlertBridge) -> None:
    with pytest.raises(BridgeError):
        bridge.publish_verdict(buy_verdict(landed=50.98), PRODUCTS[0])


def test_the_dedupe_key_is_the_landed_price_not_the_shelf_price() -> None:
    """Keying on the shelf price would let a dollar of shipping pass in
    silence, and the rule's cap is on the landed price."""
    dearer_shipping = buy_verdict(price=4499, landed=5598)
    assert dedupe_key_for(buy_verdict()).endswith("|5098")
    assert dedupe_key_for(dearer_shipping).endswith("|5598")


def test_the_bridge_needs_a_service_and_a_clock(service: FakeAlertService) -> None:
    with pytest.raises(BridgeError):
        AlertBridge(object(), SimClock())
    with pytest.raises(BridgeError):
        AlertBridge(service, None)  # type: ignore[arg-type]


def test_the_bridge_survives_a_verdict_straight_out_of_the_log(
    store: PokeStore, bridge: AlertBridge, service: FakeAlertService
) -> None:
    """The two halves of this assignment meet here: a verdict written to the
    log, read back, and published, is the same alert as the original."""
    original = buy_verdict()
    store.append_verdict(original)
    (from_log,) = store.verdicts()
    assert_round_trip_equal(original, from_log)

    bridge.publish_verdict(from_log, PRODUCTS[0])
    assert len(service.published) == 1
    assert service.published[0]["dedupe_key"] == dedupe_key_for(original)


# --------------------------------------------------------------------------
# The boundary, checked in the source rather than trusted
# --------------------------------------------------------------------------


def test_neither_module_can_reach_the_network_or_a_real_clock() -> None:
    """contracts.py: "The package makes no network calls itself: the app
    injects a fetcher".  And the clock is injected everywhere, so a
    ``time.time()`` or a ``random.random()`` in here would make a failure
    irreproducible."""
    banned_modules = {
        "urllib", "http", "socket", "requests", "httpx", "ssl", "asyncio",
        "subprocess", "random", "secrets",
    }
    banned_calls = {("time", "time"), ("time", "monotonic"), ("random", "random")}
    for name in ("store.py", "alerts_bridge.py"):
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


def test_the_bridge_carries_no_checkout_and_no_retailer_knowledge() -> None:
    """contracts.py: "It does not check out."  The bridge is the last stop
    before a person's phone, and it stays a notification."""
    source = (ROOT / "jarvis_poke" / "alerts_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Names, not prose: the module docstring says the words "checkout" and
    # "CAPTCHA" precisely to say it does none of it.
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name.lower())
        elif isinstance(node, ast.Name):
            names.add(node.id.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
    for word in ("cart", "checkout", "captcha", "proxy", "cvv", "payment", "solve"):
        offenders = sorted(n for n in names if word in n)
        assert not offenders, f"alerts_bridge.py defines or calls {offenders}"
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
    assert not any(m.startswith("jarvis_poke.sources") for m in imported), (
        "the bridge has no business knowing how pages are fetched"
    )


def test_shipped_data_names_no_real_retailer() -> None:
    """Placeholders only, example.com only -- the scope boundary, checked
    against the files this package actually ships."""
    for name in ("catalog.json", "sources.json"):
        text = (ROOT / "jarvis_poke" / "data" / name).read_text(encoding="utf-8")
        for host in ("http://", "https://"):
            for chunk in text.split(host)[1:]:
                domain = chunk.split("/")[0].split('"')[0]
                assert domain.endswith("example.com") or domain.endswith("example.org"), (
                    f"{name} points at {domain!r}, which is not a placeholder"
                )


def test_reservations_survive_a_save_and_load(store: PokeStore) -> None:
    """A reservation is money promised to an alert on someone's phone.

    It used to be dropped on every save, on the theory that "after a
    restart nobody is holding a deep link open" -- which is wrong for a
    tool whose ``decide`` is a fresh process every run.  The ledger reset
    to nothing-promised on every invocation.
    """
    from jarvis_poke.rules import RuleSet

    rules = RuleSet(
        [Rule(product_id="p1", max_price=20_000), Rule(product_id="p2", max_price=20_000)],
        Budget(total=100_000),
    )
    rules.reserve_for("p1", 12_000)
    store.save_rule_set(rules)

    back = store.load_rule_set()
    assert back.reservations() == {"p1": 12_000}
    assert back.reserved == 12_000
    assert back.remaining() == 88_000

    # the owner bought it: spend replaces the hold, and that persists too
    back.commit_for("p1")
    store.save_rule_set(back)
    final = store.load_rule_set()
    assert final.reservations() == {}
    assert final.budget.spent == 12_000
    assert final.remaining() == 88_000


def test_reservations_can_be_written_and_read_on_their_own(store: PokeStore) -> None:
    assert store.load_reservations() == {}
    assert store.save_reservations({"p1": 500, "p2": 0, "p3": 700}) == 2
    assert store.load_reservations() == {"p1": 500, "p3": 700}
    assert store.save_reservations({}) == 0
    assert store.load_reservations() == {}


def test_the_poll_store_offers_a_lock_other_processes_respect(
    tmp_path: Path, clock: SimClock
) -> None:
    """The poll gate is read-decide-fetch-write; without a cross-process
    lock, N overlapping ``poll --once`` runs each fetch."""
    with PokeStore(tmp_path / "poke.sqlite3", clock=clock) as opened:
        poll_store = opened.poll_store()
        assert callable(getattr(poll_store, "lock", None))
        with poll_store.lock():
            # re-entrant: the scheduler saves inside the locked region
            with poll_store.lock():
                poll_store.save({"version": 1, "sources": {}, "skus": []})
        assert poll_store.lock_path() == str(tmp_path / "poke.sqlite3") + ".pollock"

    memory = PokeStore(":memory:", clock=clock)
    assert memory.poll_store().lock_path() is None
    with memory.poll_store().lock():
        pass
    memory.close()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
