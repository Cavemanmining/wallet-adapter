"""Tests for the Pokemon assistant on the bot framework.

The bot is the framework's first non-trivial user, so these tests are
really two claims at once: that ``jarvis_poke`` reaches the launcher
correctly, and that ``jarvis_bots`` needed no poke-shaped hole to let it.

What is checked, and which line of which contract it comes from:

* "the badge counts distinct open keys" -- a BUY opens exactly one, the
  same BUY next tick opens no second one, and a *price change* opens a new
  one and closes the old, because the landed price is in the key;
* "Events are the only output" -- every path returns events, including
  every failure path, and nothing ever reaches the supervisor as an
  exception: a source that errors is a line in the feed, and the bot's
  health is untouched afterwards;
* "State is the bot's, persistence is ours" -- the snapshot round trips
  through JSON and brings the engine's watch states (the cooldowns) back
  with it;
* "Sources declare a minimum poll interval" -- the bot polls exactly the
  listings :meth:`PollScheduler.due` offered it, and nothing else;
* ``jarvis_poke.contracts``, "What this is not" -- there is no checkout
  surface anywhere in the module, checked against the source rather than
  against a promise.

Nothing here sleeps, opens a socket or reads a real clock: the fetcher and
parser are stubs, the clock is a fake that only moves when a test moves it,
and the one test that wants variety draws it from ``lucifer_gen.seed``,
which is the only permitted randomness.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_gen.seed import SeedFields

from jarvis_bots.bots.poke_bot import (
    DEFAULT_OBSERVATION_TTL_S,
    INFO,
    SNAPSHOT_VERSION,
    PokeBot,
    PokeBotError,
    attention_key_for,
    close_resolved,
)
from jarvis_bots.contracts import BotState, Event, Severity
from jarvis_bots.registry import BotRegistry, check_bot
from jarvis_bots.supervisor import Supervisor
from jarvis_poke.catalog import Catalog
from jarvis_poke.contracts import (
    Budget,
    FetchPolicy,
    FetchResult,
    Observation,
    Product,
    ProductKind,
    Rule,
    SourceSku,
    Stock,
    fmt_cents,
)
from jarvis_poke.engine import DecisionEngine
from jarvis_poke.prices import PriceHistory
from jarvis_poke.rules import RuleSet
from jarvis_poke.sources import PollScheduler

MODULE = ROOT / "jarvis_bots" / "bots" / "poke_bot.py"

T0 = 1_700_000_000.0
ETB = "sv08-surging-sparks-etb"
BOX = "sv08-surging-sparks-booster-box"
SHOP_A = "shopa"
SHOP_B = "shopb"
URL_A = "https://example.com/shopa/surging-sparks-etb"
URL_B = "https://example.com/shopb/surging-sparks-booster-box"


# --------------------------------------------------------------------------
# the stubs: a clock that only a test moves, and a shop that sells nothing
# --------------------------------------------------------------------------


class Clock:
    """An injected clock.  contracts.py: nothing calls ``time.time()``."""

    def __init__(self, at: float = T0) -> None:
        self.at = float(at)

    def __call__(self) -> float:
        return self.at

    def advance(self, seconds: float) -> float:
        self.at += float(seconds)
        return self.at


class Shop:
    """A stub retailer: an injected fetcher and parser, and a log of both.

    ``listing[(source, product_id)]`` is what the parser will say -- a
    ``(stock, price, shipping)`` triple, or an exception instance to raise.
    ``fetch_fault[url]`` makes the fetcher raise or return a failure.
    """

    def __init__(self) -> None:
        self.listing: Dict[Tuple[str, str], Any] = {}
        self.fetch_fault: Dict[str, Any] = {}
        self.fetched: List[str] = []
        self.parsed: List[Tuple[str, str]] = []

    # the injected Fetcher
    def fetch(self, url: str, headers: Dict[str, str], policy: FetchPolicy) -> FetchResult:
        self.fetched.append(url)
        fault = self.fetch_fault.get(url)
        if isinstance(fault, BaseException):
            raise fault
        if isinstance(fault, FetchResult):
            return fault
        return FetchResult(ok=True, status=200, body="<page/>")

    # the injected Parser
    def parse(self, sku: SourceSku, body: str, at: float) -> Observation:
        self.parsed.append((sku.source, sku.product_id))
        answer = self.listing.get((sku.source, sku.product_id))
        if isinstance(answer, BaseException):
            raise answer
        stock, price, shipping = answer or (Stock.OUT_OF_STOCK, None, 0)
        return Observation(
            product_id=sku.product_id,
            source=sku.source,
            sku=sku.sku,
            at=at,
            stock=stock,
            price=price,
            shipping=shipping,
            url=sku.url,
        )

    def sell(self, source: str, product_id: str, price: int, shipping: int = 0) -> None:
        self.listing[(source, product_id)] = (Stock.IN_STOCK, price, shipping)

    def sold_out(self, source: str, product_id: str) -> None:
        self.listing[(source, product_id)] = (Stock.OUT_OF_STOCK, None, 0)


class Rig:
    """Everything wired together the way an app would wire it."""

    def __init__(
        self,
        *,
        rules: Sequence[Rule],
        budget: Budget = Budget(total=1_000_00),
        observation_ttl_s: float = DEFAULT_OBSERVATION_TTL_S,
    ) -> None:
        self.clock = Clock()
        self.shop = Shop()
        self.catalog = Catalog(
            products=[
                Product(ETB, "Surging Sparks ETB", "SV08", ProductKind.ELITE_TRAINER_BOX),
                Product(BOX, "Surging Sparks Booster Box", "SV08", ProductKind.BOOSTER_BOX),
            ],
            skus=[
                SourceSku(SHOP_A, ETB, "A-1", URL_A),
                SourceSku(SHOP_B, BOX, "B-1", URL_B),
            ],
            source_labels={SHOP_A: "Shop A", SHOP_B: "Shop B"},
        )
        self.rules = RuleSet(list(rules), budget=budget)
        self.history = PriceHistory()
        self.engine = DecisionEngine(self.catalog, self.rules, self.history, self.clock)
        self.scheduler = PollScheduler(
            self.catalog,
            {
                SHOP_A: FetchPolicy(SHOP_A, min_interval_s=300.0),
                SHOP_B: FetchPolicy(SHOP_B, min_interval_s=300.0),
            },
            self.clock,
            jitter_fraction=0.0,  # the jitter is the scheduler's own test
        )
        self.bot = PokeBot(
            self.catalog,
            self.rules,
            self.history,
            self.engine,
            self.scheduler,
            self.shop.fetch,
            self.shop.parse,
            self.clock,
            observation_ttl_s=observation_ttl_s,
        )
        self.supervisor = Supervisor(BotRegistry([self.bot]), self.clock)

        # Record what each tick returned, so a test driving whole rounds can
        # still look at the events the supervisor saw.
        self.seen: List[Event] = []
        inner = self.bot.tick

        def recording(now: float) -> Sequence[Event]:
            events = tuple(inner(now))
            self.seen = list(events)
            return events

        self.bot.tick = recording  # type: ignore[method-assign]

    def tick(self, advance: float = 600.0) -> Tuple[Event, ...]:
        """One tick, straight at the bot, after moving the clock."""
        self.clock.advance(advance)
        return tuple(self.bot.tick(self.clock.at))

    def round(self, advance: float = 600.0):
        """One supervisor round, then close whatever the bot resolved.

        ``close_resolved`` is the app-side half the framework is missing;
        see the module docstring of ``poke_bot``.
        """
        self.clock.advance(advance)
        report = self.supervisor.run_round(self.clock.at)
        close_resolved(self.supervisor, self.bot.drain_resolved())
        return report


def rule(product_id: str, max_price: int, **kwargs: Any) -> Rule:
    kwargs.setdefault("cooldown_s", 3600.0)
    return Rule(product_id=product_id, max_price=max_price, **kwargs)


def actions(events: Sequence[Event]) -> List[Event]:
    return [e for e in events if e.severity is Severity.ACTION]


def keys(events: Sequence[Event]) -> List[Optional[str]]:
    return [e.attention_key for e in events]


# --------------------------------------------------------------------------
# identity: the launcher's contract
# --------------------------------------------------------------------------


def test_info_is_what_the_launcher_and_the_registry_expect() -> None:
    """``jarvis_bots/web/README.md``: ``kind`` picks the icon, ``href`` is
    what Open points at.  ``check_bot`` is the registry's own gate."""
    assert (INFO.id, INFO.kind, INFO.interval_s, INFO.href) == (
        "poke", "cart", 300.0, "/bots/poke"
    )
    assert INFO.name == "Pokemon buying assistant"
    assert len(INFO.blurb) > 20 and INFO.blurb[0].isupper()
    rig = Rig(rules=[rule(ETB, 6000)])
    assert check_bot(rig.bot) is INFO
    assert rig.bot.id == "poke"


def test_the_card_renders_through_the_supervisor() -> None:
    """The bot is only real if ``launcher_state`` can render it."""
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.round()
    card = rig.supervisor.launcher_state(rig.clock.at)["bots"][0]
    assert card["id"] == "poke"
    assert card["kind"] == "cart"
    assert card["state"] == "running"
    assert card["href"] == "/bots/poke"
    assert card["attention"] == 1
    assert [s["label"] for s in card["stats"]] == [
        "Watching", "Budget left", "Best discount"
    ]
    assert card["last_event"]["text"].startswith("Buy: Surging Sparks ETB")


# --------------------------------------------------------------------------
# polling: only what the scheduler offers
# --------------------------------------------------------------------------


def test_the_bot_only_polls_what_the_scheduler_offers() -> None:
    """``jarvis_poke.contracts``: "A source that has been told to slow
    down, or that robots.txt disallows, is not polled."  The bot does not
    get a say -- it fetches exactly the listings ``due`` named."""
    rig = Rig(rules=[rule(ETB, 6000), rule(BOX, 20_000)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.shop.sell(SHOP_B, BOX, 15_000)

    rig.clock.advance(600.0)
    offered = [sku.url for sku in rig.scheduler.due(rig.clock.at)]
    assert sorted(offered) == sorted([URL_A, URL_B])
    rig.bot.tick(rig.clock.at)
    assert sorted(rig.shop.fetched) == sorted(offered)

    # A second tick one second later: nothing is due inside the 300s host
    # interval, so nothing is fetched and nothing is parsed.
    before = len(rig.shop.fetched)
    rig.clock.advance(1.0)
    assert rig.scheduler.due(rig.clock.at) == []
    events = rig.bot.tick(rig.clock.at)
    assert len(rig.shop.fetched) == before
    assert events == ()


def test_a_source_robots_disallows_is_never_fetched() -> None:
    """The listing exists and is in stock; the policy says no."""
    rig = Rig(rules=[rule(ETB, 6000), rule(BOX, 20_000)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.shop.sell(SHOP_B, BOX, 15_000)
    rig.scheduler.set_policy(
        FetchPolicy(SHOP_A, min_interval_s=300.0, robots_allows=False)
    )
    rig.tick()
    assert rig.shop.fetched == [URL_B]
    assert rig.bot.open_attention_keys() == [f"buy|{BOX}|{SHOP_B}|15000"]


# --------------------------------------------------------------------------
# attention: the badge
# --------------------------------------------------------------------------


def test_nothing_in_stock_produces_no_attention() -> None:
    """contracts.py: an ACTION event "drives the badge".  Out of stock is
    not a decision, so it earns a line in the feed and nothing more."""
    rig = Rig(rules=[rule(ETB, 6000), rule(BOX, 20_000)])
    rig.shop.sold_out(SHOP_A, ETB)
    rig.shop.sold_out(SHOP_B, BOX)

    report = rig.round()
    assert report.failed == 0 and report.ticked == 1
    assert rig.supervisor.attention_count() == 0
    assert rig.bot.open_attention_keys() == []

    events = rig.supervisor.last_event("poke")
    assert events is not None and events.text.startswith("Out of stock: ")
    assert rig.supervisor.badge_status() == {"attention": 0, "state": "ok"}

    # And it does not keep saying so: the second look is not news.
    assert rig.tick() == ()


def test_a_buy_raises_exactly_one_action_event_with_the_listing_href() -> None:
    """The one event that asks for a decision.  ``href`` is the seller's own
    page, because "a person completes the purchase"."""
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.shop.sell(SHOP_A, ETB, 4799, shipping=200)

    events = rig.tick()
    buys = actions(events)
    assert len(buys) == 1
    (buy,) = buys
    assert buy.bot_id == "poke"
    assert buy.severity is Severity.ACTION
    assert buy.href == URL_A
    assert buy.wants_attention
    assert buy.attention_key == f"buy|{ETB}|{SHOP_A}|4999"
    # names the product, the price, and (here) the absence of a market
    assert "Surging Sparks ETB" in buy.text
    assert "$49.99" in buy.text
    assert "Shop A" in buy.text
    # money stays integer cents on the wire
    assert buy.data["price_cents"] == 4999
    assert isinstance(buy.data["price_cents"], int)
    assert buy.data["url"] == URL_A


def test_the_buy_text_names_the_discount_when_there_is_a_market() -> None:
    """contracts.py measures a discount "against the market reference, not
    MSRP, because MSRP is fiction for anything in demand"."""
    rig = Rig(rules=[rule(ETB, 6000)])
    for offset in (300.0, 200.0, 100.0):
        rig.history.append(
            Observation(ETB, SHOP_A, "A-1", T0 - offset, Stock.IN_STOCK, 6000, url=URL_A)
        )
    rig.shop.sell(SHOP_A, ETB, 4800)

    (buy,) = actions(rig.tick())
    assert "20.0% off" in buy.text
    assert "$60.00 market price" in buy.text
    assert buy.data["market_cents"] == 6000
    assert buy.data["discount_pct"] == pytest.approx(20.0)


def test_the_same_buy_next_tick_does_not_duplicate_the_attention() -> None:
    """contracts.py: "one restock nagging across ten ticks is one item of
    attention, not ten".  With no cooldown the engine says BUY again; the
    bot must not raise the key again."""
    rig = Rig(rules=[rule(ETB, 6000, cooldown_s=0.0)])
    rig.shop.sell(SHOP_A, ETB, 5000)

    first = rig.round()
    assert first.alerts == 0  # no alert service wired; attention is the point
    assert rig.supervisor.attention_count() == 1
    opened = rig.supervisor.attention_items()[0]

    second = rig.tick()
    assert actions(second) == []
    assert second == ()
    assert rig.bot.open_attention_keys() == [opened.key]

    rig.round()
    assert rig.supervisor.attention_count() == 1
    assert rig.supervisor.attention_items()[0].since == opened.since


def test_a_buy_inside_the_cooldown_keeps_the_same_request_open() -> None:
    """A SKIP taken on the cooldown still names the same listing at the same
    landed price.  The offer has not changed, so the request has not been
    answered and must not be closed."""
    rig = Rig(rules=[rule(ETB, 6000, cooldown_s=3600.0)])
    rig.shop.sell(SHOP_A, ETB, 5000)

    rig.round()
    key = rig.bot.open_attention_keys()[0]
    assert rig.supervisor.attention_count() == 1

    for _ in range(3):  # still inside the 3600s cooldown
        assert rig.round().failed == 0
    assert rig.bot.open_attention_keys() == [key]
    assert rig.supervisor.attention_count() == 1


def test_a_price_change_is_a_new_item_and_clears_the_old() -> None:
    """The landed price is in the key on purpose: "the first alert said
    $54.99 and the owner passed; $44.99 is a different decision"."""
    rig = Rig(rules=[rule(ETB, 6000, cooldown_s=0.0)])
    rig.shop.sell(SHOP_A, ETB, 5499)

    rig.round()
    first_key = f"buy|{ETB}|{SHOP_A}|5499"
    assert rig.bot.open_attention_keys() == [first_key]
    assert [i.key for i in rig.supervisor.attention_items()] == [first_key]

    rig.shop.sell(SHOP_A, ETB, 4499)
    rig.round()
    second_key = f"buy|{ETB}|{SHOP_A}|4499"
    assert keys(rig.seen) == [first_key, second_key]
    resolution, raised = rig.seen
    assert resolution.severity is Severity.NOTICE
    assert resolution.wants_attention is False
    assert resolution.data["resolved"] is True
    assert "$44.99" in resolution.text
    assert raised.severity is Severity.ACTION
    assert "$44.99" in raised.text

    # one item of attention, and it is the live offer
    assert [i.key for i in rig.supervisor.attention_items()] == [second_key]
    assert rig.supervisor.badge_status()["attention"] == 1
    assert rig.bot.open_attention_keys() == [second_key]


def test_attention_clears_when_a_product_stops_being_a_buy() -> None:
    """The listing sells out: the question is answered by the world."""
    rig = Rig(rules=[rule(ETB, 6000, cooldown_s=0.0)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.round()
    assert rig.supervisor.attention_count() == 1

    rig.shop.sold_out(SHOP_A, ETB)
    rig.round()
    assert rig.supervisor.attention_count() == 0
    assert rig.bot.open_attention_keys() == []
    assert rig.supervisor.last_event("poke").text.startswith("No longer a buy: ")


def test_a_flapping_listing_at_one_price_comes_back_at_the_same_key() -> None:
    """In stock, out, in again is what a restock looks like from outside.
    The key is unchanged, so it is the same question -- which is what lets
    ``jarvis_alerts``' dedupe window collapse it to one push."""
    rig = Rig(rules=[rule(ETB, 6000, cooldown_s=0.0)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.round()
    (before,) = rig.bot.open_attention_keys()

    rig.shop.sold_out(SHOP_A, ETB)
    rig.round()
    assert rig.bot.open_attention_keys() == []

    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.round()
    assert rig.bot.open_attention_keys() == [before]
    assert rig.supervisor.attention_count() == 1


def test_a_watch_is_at_most_a_notice_and_only_when_it_is_news() -> None:
    """contracts.py has NOTICE be "worth a line in the feed"; WATCH is in
    stock but over the ceiling, which is a line, not a decision."""
    rig = Rig(rules=[rule(ETB, 4000)])
    rig.shop.sell(SHOP_A, ETB, 5000)

    (note,) = rig.tick()
    assert note.severity is Severity.NOTICE
    assert note.attention_key is None
    assert "Watching Surging Sparks ETB" in note.text
    assert "$50.00" in note.text
    assert "$40.00 ceiling" in note.text

    assert rig.tick() == ()          # same price, not news
    rig.shop.sell(SHOP_A, ETB, 4900)
    (again,) = rig.tick()            # a new price is news
    assert "$49.00" in again.text
    assert rig.supervisor.attention_count() == 0


def test_pausing_forgets_the_open_request_so_resuming_re_raises_it() -> None:
    """contracts.py: "a badge asking you to act on something you switched
    off is a lie".  The supervisor clears the item; the bot must clear its
    own record or it would never raise the key again."""
    rig = Rig(rules=[rule(ETB, 6000, cooldown_s=0.0)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.round()
    assert rig.supervisor.attention_count() == 1

    rig.supervisor.pause("poke")
    assert rig.bot.open_attention_keys() == []
    assert rig.supervisor.attention_count() == 0
    report = rig.round()
    assert report.skipped == 1 and report.ticked == 0

    rig.supervisor.resume("poke")
    rig.round()
    assert rig.supervisor.attention_count() == 1


# --------------------------------------------------------------------------
# failure: an event, never an exception
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fault, expected",
    [
        ("fetch-raises", "the fetcher raised RuntimeError"),
        ("fetch-fails", "HTTP 503 (upstream busy)"),
        ("parse-raises", "the parser raised ValueError"),
    ],
)
def test_a_source_error_is_reported_as_an_event_and_never_propagates(
    fault: str, expected: str
) -> None:
    """The lane's rule: "the supervisor should not have to quarantine a bot
    over one bad poll"."""
    rig = Rig(rules=[rule(ETB, 6000)])
    if fault == "fetch-raises":
        rig.shop.fetch_fault[URL_A] = RuntimeError("connection reset by https://x/y")
    elif fault == "fetch-fails":
        rig.shop.fetch_fault[URL_A] = FetchResult(
            ok=False, status=503, reason="upstream busy"
        )
    else:
        rig.shop.sell(SHOP_A, ETB, 5000)
        rig.shop.listing[(SHOP_A, ETB)] = ValueError("no price on the page")

    events = rig.tick()           # the call itself must not raise
    (note,) = [e for e in events if "Could not read" in e.text]
    assert note.severity is Severity.NOTICE      # not ERROR: no push for a 503
    assert expected in note.text
    assert note.attention_key is None
    assert note.data["source"] == SHOP_A

    report = rig.round()
    assert report.failed == 0
    assert rig.supervisor.health("poke").consecutive_failures == 0
    assert not rig.supervisor.is_quarantined("poke")
    assert rig.bot.status().state is BotState.RUNNING


def test_an_error_message_never_carries_a_url_into_the_feed() -> None:
    """``jarvis_poke.sources`` keeps "only the exception's type name, never
    its message, which may quote a URL or a page".  So does this."""
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.shop.fetch_fault[URL_A] = RuntimeError("GET https://example.com/secret?token=abc")
    rig.shop.fetch_fault[URL_B] = FetchResult(
        ok=False, status=500, reason="see https://example.com/status for details"
    )
    events = rig.tick()
    for event in events:
        assert "://" not in event.text
        assert "token" not in event.text
    assert any("HTTP 500" in e.text for e in events)


def test_an_unexpected_internal_fault_becomes_an_event_once() -> None:
    """Even a bug in the bot comes back as an event: contracts.py allows a
    tick to raise, but a broken poll should not cost half an hour of
    quarantine.  ERROR the first time, NOTICE for the same fault again, so
    a bot that is broken says so and then stops shouting."""

    class Exploding:
        def due(self, now: float) -> List[SourceSku]:
            raise KeyError("no such source")

        def poll_once(self, *a: Any, **k: Any) -> None:  # pragma: no cover
            raise AssertionError("never reached")

    rig = Rig(rules=[rule(ETB, 6000)])
    rig.bot._scheduler = Exploding()

    (first,) = rig.tick()
    assert first.severity is Severity.ERROR
    assert "KeyError" in first.text
    assert "no such source" not in first.text
    (second,) = rig.tick()
    assert second.severity is Severity.NOTICE

    assert rig.round().failed == 0
    assert rig.supervisor.health("poke").consecutive_failures == 0


def test_a_paused_source_is_reported_once_and_its_return_too() -> None:
    """A source the scheduler has given up on is the one failure the owner
    cannot otherwise see: the bot keeps ticking and simply finds nothing."""
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.scheduler.pause_source(SHOP_A, rig.clock.at + 4000.0, "too many errors")

    def about_the_source(events: Sequence[Event]) -> List[Event]:
        return [e for e in events if "source" in e.data and "product_id" not in e.data]

    (paused,) = about_the_source(rig.tick())
    assert paused.severity is Severity.ERROR
    assert "too many errors" in paused.text
    assert paused.attention_key is None
    assert about_the_source(rig.tick()) == []

    rig.scheduler.resume_source(SHOP_A)
    (back,) = about_the_source(rig.tick())
    assert back.severity is Severity.NOTICE
    assert "again" in back.text


# --------------------------------------------------------------------------
# the card
# --------------------------------------------------------------------------


def test_status_stats_are_correct() -> None:
    """contracts.py: a Stat is "pre-formatted: the bot knows how its own
    numbers should read, and the page should not be doing money maths"."""
    rig = Rig(
        rules=[rule(ETB, 6000), rule(BOX, 20_000), rule("unlisted", 1000, enabled=False)],
        budget=Budget(total=50_000),
    )
    idle = rig.bot.status()
    assert idle.state is BotState.IDLE          # nothing looked at yet
    assert idle.detail == "not looked yet"
    # ...but the figures it already knows are true, so it states them
    assert {s.label: s.value for s in idle.stats}["Watching"] == "2 products"

    for offset in (300.0, 200.0, 100.0):
        rig.history.append(
            Observation(ETB, SHOP_A, "A-1", T0 - offset, Stock.IN_STOCK, 6000, url=URL_A)
        )
    rig.shop.sell(SHOP_A, ETB, 4800)
    rig.shop.sold_out(SHOP_B, BOX)
    rig.tick()

    status = rig.bot.status()
    assert status.state is BotState.RUNNING
    labels = {s.label: s.value for s in status.stats}
    assert labels["Watching"] == "2 products"          # the disabled rule is not watched
    assert labels["Budget left"] == fmt_cents(rig.rules.remaining())
    assert labels["Budget left"].startswith("$")
    assert labels["Best discount"] == "20.0% off"
    assert rig.bot.best_discount() == pytest.approx(20.0)

    # A discount on something out of stock is not a discount on anything.
    rig.shop.sold_out(SHOP_A, ETB)
    rig.tick()
    assert rig.bot.best_discount() is None
    assert {s.label: s.value for s in rig.bot.status().stats}["Best discount"] == "none yet"


def test_one_watched_product_reads_in_the_singular() -> None:
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.tick()
    assert {s.label: s.value for s in rig.bot.status().stats}["Watching"] == "1 product"


def test_status_does_not_take_the_page_down_when_the_rules_do() -> None:
    """contracts.py: ``status`` "must not raise and must be cheap: the page
    calls it on every load"."""

    class Broken:
        def get(self, product_id: str) -> None: ...
        def enabled_rules(self) -> List[Rule]:
            raise RuntimeError("ledger is locked")
        def remaining(self) -> int: ...

    rig = Rig(rules=[rule(ETB, 6000)])
    rig.bot._rules = Broken()
    status = rig.bot.status()
    assert status.stats == ()
    assert "RuntimeError" in status.detail


# --------------------------------------------------------------------------
# persistence
# --------------------------------------------------------------------------


def test_snapshot_restore_preserves_watch_state() -> None:
    """contracts.py: "State is the bot's, persistence is ours."  What must
    survive is the cooldown: lose it and a restart re-alerts every open
    deal."""
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.round()

    before = rig.engine.watch_state(ETB)
    assert before.alerts_sent == 1 and before.last_alert_at > 0

    snapshot = rig.bot.snapshot()
    assert snapshot["version"] == SNAPSHOT_VERSION
    assert json.loads(json.dumps(snapshot)) == snapshot   # JSON-able, as promised

    fresh = Rig(rules=[rule(ETB, 6000)])
    fresh.clock.at = rig.clock.at
    fresh.bot.restore(json.loads(json.dumps(snapshot)))

    after = fresh.engine.watch_state(ETB)
    assert after.alerts_sent == before.alerts_sent
    assert after.last_alert_at == before.last_alert_at
    assert after.last_action is before.last_action
    assert after.last_seen_in_stock == before.last_seen_in_stock
    assert fresh.bot.open_attention_keys() == rig.bot.open_attention_keys()
    assert fresh.bot.status().state is BotState.RUNNING
    assert fresh.bot.snapshot()["watch_states"] == snapshot["watch_states"]

    # The restored cooldown is real: the same listing does not re-alert.
    fresh.shop.sell(SHOP_A, ETB, 5000)
    assert actions(fresh.tick()) == []


def test_the_snapshot_carries_the_schedulers_politeness() -> None:
    """``PollScheduler.snapshot`` is "everything a restart needs to stay as
    polite as it was"; a restart that forgets it polls immediately."""
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.shop.sell(SHOP_A, ETB, 5000)
    rig.tick()

    fresh = Rig(rules=[rule(ETB, 6000)])
    fresh.clock.at = rig.clock.at
    assert fresh.scheduler.due(fresh.clock.at)          # a fresh one would poll now
    fresh.bot.restore(json.loads(json.dumps(rig.bot.snapshot())))
    assert fresh.scheduler.due(fresh.clock.at) == []    # the restored one waits


def test_a_snapshot_from_a_newer_build_is_refused_not_half_read() -> None:
    rig = Rig(rules=[rule(ETB, 6000)])
    with pytest.raises(PokeBotError):
        rig.bot.restore({"version": SNAPSHOT_VERSION + 1})
    rig.bot.restore({})          # absent state is not an error
    rig.bot.restore({"version": SNAPSHOT_VERSION})


def test_restore_updates_the_engines_mapping_in_place() -> None:
    """``DecisionEngine`` documents ``watch_states`` as "used in place, so a
    caller that keeps it in a store sees the engine's updates"; replacing
    the object would leave the engine writing to the old one."""
    rig = Rig(rules=[rule(ETB, 6000)])
    held = rig.engine.watch_states
    rig.bot.restore(
        {
            "version": SNAPSHOT_VERSION,
            "watch_states": {ETB: {"last_alert_at": 42.0, "last_action": "buy",
                                   "alerts_sent": 3, "last_seen_in_stock": 41.0}},
        }
    )
    assert rig.engine.watch_states is held
    assert held[ETB].alerts_sent == 3
    assert held[ETB].last_action.value == "buy"


# --------------------------------------------------------------------------
# wiring, determinism, and the boundary
# --------------------------------------------------------------------------


def test_a_missing_or_mis_wired_collaborator_fails_at_construction() -> None:
    """The registry's stance: "A malformed bot is a programming or config
    mistake, and the moment to find out is the line that registers it"."""
    rig = Rig(rules=[rule(ETB, 6000)])
    args = (rig.catalog, rig.rules, rig.history, rig.engine, rig.scheduler)
    with pytest.raises(PokeBotError, match="fetcher must be callable"):
        PokeBot(*args, "not a fetcher", rig.shop.parse, rig.clock)
    with pytest.raises(PokeBotError, match="parser must be callable"):
        PokeBot(*args, rig.shop.fetch, None, rig.clock)
    with pytest.raises(PokeBotError, match=r"scheduler has no due\(\), poll_once\(\)"):
        PokeBot(rig.catalog, rig.rules, rig.history, rig.engine, object(),
                rig.shop.fetch, rig.shop.parse, rig.clock)
    with pytest.raises(ValueError, match="needs an injected clock"):
        PokeBot(*args, rig.shop.fetch, rig.shop.parse, 1_700_000_000.0)


def test_an_engine_on_a_different_clock_is_refused() -> None:
    """``DecisionEngine.evaluate`` takes no ``now`` and stamps verdicts from
    its own clock; two clocks put the cooldown and the feed on different
    timelines."""
    rig = Rig(rules=[rule(ETB, 6000)])
    elsewhere = DecisionEngine(rig.catalog, rig.rules, rig.history, Clock(T0 + 90_000.0))
    with pytest.raises(PokeBotError, match="clock"):
        PokeBot(rig.catalog, rig.rules, rig.history, elsewhere, rig.scheduler,
                rig.shop.fetch, rig.shop.parse, rig.clock)


def test_attention_key_for_is_total_and_integer() -> None:
    rig = Rig(rules=[rule(ETB, 6000)])
    rig.shop.sell(SHOP_A, ETB, 4799, shipping=200)
    rig.tick()
    verdict = rig.engine.evaluate(ETB, rig.history.for_product(ETB))
    assert attention_key_for(verdict).endswith("|4999")
    assert "4999.0" not in attention_key_for(verdict)
    rig.shop.sold_out(SHOP_A, ETB)
    rig.tick()
    assert attention_key_for(
        rig.engine.evaluate(ETB, rig.history.for_product(ETB))
    ) is None
    with pytest.raises(PokeBotError):
        attention_key_for("buy|x")


def test_the_same_scenario_twice_produces_the_same_events() -> None:
    """Determinism: the clock is injected, no randomness is drawn here, and
    the scheduler's jitter comes from ``lucifer_gen.seed``.  The variety in
    this test comes from there too, rather than from ``random``."""
    stream = SeedFields.parse(0xC0FFEE).stream("route.poke-bot")
    prices = [4000 + stream.randint(0, 2000) for _ in range(6)]

    def transcript() -> List[Tuple[int, str, Optional[str]]]:
        rig = Rig(rules=[rule(ETB, 6000, cooldown_s=0.0)])
        out: List[Tuple[int, str, Optional[str]]] = []
        for price in prices:
            rig.shop.sell(SHOP_A, ETB, price)
            for event in rig.tick():
                out.append((int(event.severity), event.text, event.attention_key))
        return out

    first = transcript()
    assert first == transcript()
    assert first  # the scenario actually did something


def test_nothing_here_buys_anything() -> None:
    """``jarvis_poke.contracts``, "What this is not": "It does not check
    out."  Checked against the source, not against a promise: no name in
    this module carts, pays or commits a purchase, and the engine's
    spend-committing methods are never called."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name.lower())
        elif isinstance(node, ast.Attribute):
            names.add(node.attr.lower())
        elif isinstance(node, ast.Name):
            names.add(node.id.lower())
    for word in ("checkout", "payment", "purchase", "captcha", "cvv", "card_number"):
        assert not [n for n in names if word in n], f"{word} has no business here"
    assert "commit" not in names and "commit_purchase" not in names


def test_nothing_here_reads_a_clock_or_opens_a_socket() -> None:
    """contracts.py: "Time is injected everywhere.  Nothing here calls
    time.time()", and no package in this tree makes a network call."""
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            roots.add((node.module or "").split(".")[0])
    assert not roots & {"time", "datetime", "random", "urllib", "http",
                        "socket", "requests", "ssl"}, sorted(roots)

    # and no attribute call that would reach one of them by another route
    attributes = {
        f"{node.value.id}.{node.attr}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
    }
    assert not [a for a in attributes if a.split(".")[0] in
                {"time", "random", "urllib", "socket", "os"}], sorted(attributes)


def test_the_bots_package_imports_nothing_by_itself() -> None:
    """``jarvis_bots.registry``: "explicit registration only ... no import
    of every module in a package"."""
    init = (ROOT / "jarvis_bots" / "bots" / "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(init)
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))
                and getattr(n, "module", "") != "__future__"]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
