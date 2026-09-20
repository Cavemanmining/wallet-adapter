"""Tests for the bot framework: base, registry, supervisor.

Design: jarvis_bots/contracts.py.  Every rule it states about the
supervisor is a test here, because the framework's whole value is that a
new bot inherits them without writing any of them:

* "A bot never blocks another" -- a bot that raises every tick is
  quarantined after exactly ``QUARANTINE_AFTER_FAILURES`` while its
  neighbours tick on with their health untouched;
* "A sick bot backs off" -- the interval widens, is capped, and is never
  narrower than the base one (which is where contracts.py's own
  ``backoff_interval`` is shown to be wrong, and the workaround checked);
* an expired quarantine admits exactly one probe and re-quarantines on
  failure;
* "Paused means paused" -- not ticked, no attention, no alert, ever;
* "The badge counts distinct open keys" -- repeats collapse, clears clear,
  and ``badge_status`` is checked for every combination of states;
* "Events are the only output" -- alerts happen in one place, deduped on
  the attention key so one open request does not re-alert every round;
* "State is the bot's, persistence is ours" -- snapshot/restore round trips
  through JSON, health included.

``launcher_state`` is validated key by key against the *actual* JSON in
jarvis_bots/web/README.md, parsed out of the file rather than copied into
this one: a mismatch there is silent in the page, so the contract is read
from the document that defines it.

Nothing here sleeps, opens a socket, or reads a real clock: the clock is a
fake that only moves when a test moves it, and the one test that needs
variety draws it from ``lucifer_gen.seed``, the only permitted randomness.
"""

from __future__ import annotations

import ast
import dataclasses
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from lucifer_gen.seed import SeedFields

from jarvis_bots.base import BaseBot, parse_severity
from jarvis_bots.contracts import (
    BACKOFF_CAP_S,
    QUARANTINE_AFTER_FAILURES,
    QUARANTINE_S,
    SLOW_TICK_S,
    BotInfo,
    BotState,
    BotStatus,
    Event,
    Health,
    RegistryError,
    Severity,
    backoff_interval,
)
from jarvis_bots.registry import BotRegistry, check_bot
from jarvis_bots.supervisor import (
    JsonFileStore,
    Supervisor,
    SupervisorError,
    _widened_interval,
)

T0 = 1_758_340_000.0
INTERVAL = 300.0
WEB_README = ROOT / "jarvis_bots" / "web" / "README.md"
OWNED = ("base.py", "registry.py", "supervisor.py")


# --------------------------------------------------------------------------
# Fakes: a clock that only moves when a test moves it, a bot, an alert service
# --------------------------------------------------------------------------


class FakeClock:
    """The injected clock: contracts.py has time injected everywhere."""

    def __init__(self, t: float = T0) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t


class DemoBot(BaseBot):
    """A bot written the way the framework intends: info, tick, status.

    Everything else -- snapshot/restore and the pause hooks -- is overridden
    only so the tests can see that the supervisor calls them.
    """

    def __init__(
        self,
        clock: FakeClock,
        bot_id: str,
        *,
        interval_s: float = INTERVAL,
        kind: str = "bot",
        can_pause: bool = True,
    ) -> None:
        super().__init__(
            clock,
            BotInfo(
                id=bot_id,
                name=f"{bot_id.title()} watcher",
                blurb=f"Watches {bot_id}.",
                kind=kind,
                interval_s=interval_s,
                href=f"/bots/{bot_id}",
                can_pause=can_pause,
            ),
        )
        self.ticks = 0
        self.fail = False
        self.tick_cost = 0.0
        self.paused_calls = 0
        self.resumed_calls = 0
        self.restored: Optional[Dict[str, Any]] = None
        self.script = None  # callable(bot, now) -> sequence of Events

    def tick(self, now: float) -> Sequence[Event]:
        self.ticks += 1
        if self.tick_cost:
            self.clock.advance(self.tick_cost)
        if self.fail:
            raise RuntimeError(f"{self.info.id} is broken")
        if self.script is not None:
            return self.script(self, now)
        return []

    def status(self) -> BotStatus:
        return BotStatus(
            BotState.RUNNING, (self.stat("Ticks", self.ticks),), detail=f"{self.ticks} ticks"
        )

    def snapshot(self) -> Dict[str, Any]:
        return {"ticks": self.ticks}

    def restore(self, snapshot: Dict[str, Any]) -> None:
        self.ticks = int(snapshot.get("ticks", 0))
        self.restored = dict(snapshot)

    def on_pause(self) -> None:
        self.paused_calls += 1

    def on_resume(self) -> None:
        self.resumed_calls += 1


def asks(key: str, text: str = "decide something", severity: Severity = Severity.ACTION):
    """A tick script that raises one standing request for a decision."""

    def script(bot: DemoBot, now: float) -> Sequence[Event]:
        return [bot.event(severity, text, attention_key=key, href=bot.info.href)]

    return script


class FakeAlerts:
    """``jarvis_alerts.api.AlertService.publish``'s signature and nothing
    else, which is all the supervisor is allowed to need."""

    def __init__(self, fail: bool = False) -> None:
        self.published: List[Dict[str, Any]] = []
        self.fail = fail

    def publish(
        self,
        profile_id: str,
        kind: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        priority: Any = "normal",
        dedupe_key: Optional[str] = None,
    ) -> str:
        if self.fail:
            raise RuntimeError("push service unreachable")
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


def build(*ids: str, alerts: Any = None, store: Any = None, clock: Optional[FakeClock] = None):
    clock = clock or FakeClock()
    bots = [DemoBot(clock, bot_id) for bot_id in ids]
    supervisor = Supervisor(BotRegistry(bots), clock, store=store, alerts=alerts)
    return clock, bots, supervisor


def drive_to_quarantine(clock: FakeClock, supervisor: Supervisor, bot: DemoBot) -> None:
    """Fail the bot until the supervisor quarantines it."""
    bot.fail = True
    for _ in range(QUARANTINE_AFTER_FAILURES):
        clock.advance(BACKOFF_CAP_S + 1.0)
        supervisor.run_round(clock())
    assert supervisor.is_quarantined(bot.info.id)


# --------------------------------------------------------------------------
# BaseBot: the optional half of the protocol
# --------------------------------------------------------------------------


def test_base_bot_supplies_every_optional_part() -> None:
    """contracts.py: "the rest have usable defaults in BaseBot"."""

    class Minimal(BaseBot):
        info = BotInfo(id="minimal", name="Minimal", blurb="Nothing at all.")

        def tick(self, now):
            return []

        def status(self):
            return BotStatus.idle()

    bot = Minimal(FakeClock())
    assert bot.snapshot() == {}
    assert bot.restore({"anything": 1}) is None
    assert bot.on_pause() is None and bot.on_resume() is None
    assert BotRegistry([bot]).all() == [bot]


def test_base_bot_event_stamps_id_and_the_injected_clock() -> None:
    clock = FakeClock()
    bot = DemoBot(clock, "poke")
    clock.advance(42.0)
    event = bot.event("action", "Restocked", attention_key="restock", href="/x", sku="abc")
    assert (event.bot_id, event.at, event.severity) == ("poke", T0 + 42.0, Severity.ACTION)
    assert event.data == {"sku": "abc"} and event.href == "/x"
    assert event.wants_attention  # ACTION + a key is what the badge counts


def test_base_bot_rejects_the_mistakes_that_would_corrupt_the_badge() -> None:
    bot = DemoBot(FakeClock(), "poke")
    with pytest.raises(ValueError):
        bot.event(Severity.ACTION, "text", attention_key="")  # would collapse two requests
    with pytest.raises(ValueError):
        bot.event(Severity.ACTION, "   ")
    with pytest.raises(ValueError):
        DemoBot(object(), "poke")  # a clock that is not callable

    class NoInfo(BaseBot):
        pass

    with pytest.raises(ValueError):
        NoInfo(FakeClock())


def test_base_bot_requires_the_two_methods_the_protocol_requires() -> None:
    class Half(BaseBot):
        info = BotInfo(id="half", name="Half", blurb="Forgot to finish.")

    bot = Half(FakeClock())
    with pytest.raises(NotImplementedError):
        bot.tick(T0)
    with pytest.raises(NotImplementedError):
        bot.status()


def test_stat_is_preformatted_text() -> None:
    """contracts.py: "the page should not be doing money maths"."""
    bot = DemoBot(FakeClock(), "poke")
    assert bot.stat("Watching", 12).value == "12"
    assert bot.stat("Best price", "$47.99").value == "$47.99"  # cents formatted upstream
    with pytest.raises(ValueError):
        bot.stat("", 1)


def test_parse_severity_accepts_what_config_writes() -> None:
    assert parse_severity("ACTION") is Severity.ACTION
    assert parse_severity(4) is Severity.ERROR
    assert parse_severity(Severity.INFO) is Severity.INFO
    for bad in ("nope", 99, True, None):
        with pytest.raises(ValueError):
            parse_severity(bad)


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_keeps_registration_order_and_rejects_duplicates() -> None:
    clock = FakeClock()
    registry = BotRegistry()
    for bot_id in ("zeta", "alpha", "mid"):
        registry.register(DemoBot(clock, bot_id))
    assert registry.ids() == ["zeta", "alpha", "mid"]  # not sorted: registration order
    with pytest.raises(RegistryError):
        registry.register(DemoBot(clock, "alpha"))
    assert registry.ids() == ["zeta", "alpha", "mid"]
    assert registry.get("mid") is registry.find("mid")
    assert "alpha" in registry and len(registry) == 3
    removed = registry.unregister("alpha")
    assert removed.info.id == "alpha" and registry.ids() == ["zeta", "mid"]
    with pytest.raises(RegistryError):
        registry.unregister("alpha")
    with pytest.raises(RegistryError):
        registry.get("alpha")
    assert registry.find("alpha") is None


def test_registry_rejects_malformed_bots() -> None:
    clock = FakeClock()

    class NoTick:
        info = BotInfo(id="notick", name="No tick", blurb="")

        def status(self):
            return BotStatus.idle()

    class BadId:
        """Never ran BotInfo.__post_init__, so the id was never checked."""

        class _Info:
            id = "poke/../etc"
            name = "Sneaky"
            blurb = ""
            kind = "bot"
            interval_s = 300.0
            href = ""
            can_pause = True

        info = _Info()

        def tick(self, now):
            return []

        def status(self):
            return BotStatus.idle()

    class FastInfo(BadId):
        class _Info(BadId._Info):
            id = "fast"
            interval_s = 1.0

        info = _Info()

    registry = BotRegistry()
    for bad in (object(), NoTick(), BadId(), FastInfo()):
        with pytest.raises(RegistryError):
            registry.register(bad)
    assert registry.ids() == []
    with pytest.raises(RegistryError):
        check_bot(None)
    # BotInfo itself refuses the same ids, which is why the above had to
    # hand-roll one to get past it.
    with pytest.raises(ValueError):
        BotInfo(id="not a slug", name="x", blurb="")
    with pytest.raises(ValueError):
        BotInfo(id="fast", name="x", blurb="", interval_s=1.0)


def test_register_from_is_config_shaped_and_all_or_nothing() -> None:
    clock = FakeClock()
    registry = BotRegistry()
    registry.register_from(
        {
            "poke": DemoBot(clock, "poke"),
            "weather": lambda: DemoBot(clock, "weather"),  # a factory, for lazy config
        }
    )
    assert registry.ids() == ["poke", "weather"]

    with pytest.raises(RegistryError):  # key does not match the bot's own id
        registry.register_from({"wether": DemoBot(clock, "weather2")})
    with pytest.raises(RegistryError):  # one bad entry registers none of them
        registry.register_from({"good": DemoBot(clock, "good"), "bad": object()})
    assert registry.ids() == ["poke", "weather"]
    with pytest.raises(RegistryError):
        registry.register_from([DemoBot(clock, "list")])


# --------------------------------------------------------------------------
# "A bot never blocks another" and "A sick bot backs off"
# --------------------------------------------------------------------------


def test_a_failing_bot_is_quarantined_after_exactly_the_threshold() -> None:
    """contracts.py: ticks are isolated, and repeated failures quarantine.

    The neighbours are checked field by field: a failing bot must not show
    up in anyone else's health, tick count or attention.
    """
    clock, (good_a, bad, good_b), supervisor = build("gooda", "bad", "goodb")
    bad.fail = True
    good_a.script = asks("a-open", "A wants a decision")

    for attempt in range(1, QUARANTINE_AFTER_FAILURES + 1):
        clock.advance(BACKOFF_CAP_S + 1.0)  # past every widened interval
        report = supervisor.run_round(clock())
        health = supervisor.health("bad")
        assert (report.ticked, report.failed) == (2, 1)
        assert health.consecutive_failures == attempt
        assert health.total_failures == attempt and health.total_ticks == attempt
        assert "RuntimeError: bad is broken" == health.last_error
        expected_quarantine = attempt >= QUARANTINE_AFTER_FAILURES
        assert supervisor.is_quarantined("bad") is expected_quarantine
        assert report.quarantined == (1 if expected_quarantine else 0)
        # ... while the neighbours are untouched, every round.
        for neighbour in (good_a, good_b):
            neighbour_health = supervisor.health(neighbour.info.id)
            assert neighbour.ticks == attempt
            assert neighbour_health.consecutive_failures == 0
            assert neighbour_health.total_failures == 0
            assert neighbour_health.last_error == ""
            assert neighbour_health.healthy()
            assert neighbour_health.last_ok_at == report.at

    assert bad.ticks == QUARANTINE_AFTER_FAILURES
    assert supervisor.badge_status() == {"attention": 1, "state": "error"}

    # The quarantined bot is skipped; the neighbours keep going.
    clock.advance(INTERVAL + 1.0)
    report = supervisor.run_round(clock())
    assert (report.ticked, report.skipped, report.failed) == (2, 1, 0)
    assert bad.ticks == QUARANTINE_AFTER_FAILURES
    assert good_a.ticks == QUARANTINE_AFTER_FAILURES + 1


def test_backoff_widens_is_capped_and_never_goes_below_base() -> None:
    """contracts.py: "Never narrower than base" -- and where it is not."""
    widths = [backoff_interval(INTERVAL, n) for n in range(5)]
    assert widths == [300.0, 600.0, 1200.0, 2400.0, 3600.0]  # widens
    assert backoff_interval(INTERVAL, 50) == BACKOFF_CAP_S  # capped
    assert backoff_interval(INTERVAL, 0) == INTERVAL
    assert backoff_interval(INTERVAL, -3) == INTERVAL  # never narrower than base

    # This used to be the defect: the cap was applied after the floor, so a
    # two-hour bot that failed once was retried twice as often as when it
    # was healthy.  contracts.backoff_interval now applies the floor last
    # and keeps its own first line for every base.
    assert backoff_interval(2 * BACKOFF_CAP_S, 1) == 2 * BACKOFF_CAP_S
    assert backoff_interval(2 * BACKOFF_CAP_S, 9) == 2 * BACKOFF_CAP_S

    # And it never overflows: consecutive_failures is an exponent, the
    # counter climbs one per probe for as long as a bot stays broken, and
    # this is called from inside the failure handler, where raising takes
    # the whole round with it.
    for failures in (64, 1024, 10**6):
        assert backoff_interval(INTERVAL, failures) == BACKOFF_CAP_S

    # The supervisor's floor is what actually schedules, so the promise holds.
    for base in (5.0, INTERVAL, BACKOFF_CAP_S, 2 * BACKOFF_CAP_S):
        previous = 0.0
        for failures in range(6):
            widened = _widened_interval(base, failures)
            assert widened >= base
            assert widened <= max(base, BACKOFF_CAP_S)
            assert widened >= previous  # monotonic: it only ever slows down
            previous = widened


def test_observed_intervals_widen_across_real_rounds() -> None:
    clock, (bot,), supervisor = build("bad")
    bot.fail = True
    seen = []
    for _ in range(3):
        clock.advance(BACKOFF_CAP_S + 1.0)
        supervisor.run_round(clock())
        seen.append(supervisor.health("bad").next_due_at - clock())
    assert seen == [600.0, 1200.0, 2400.0]

    # A round before it is due does not tick it, and does not count a failure.
    before = dataclasses.asdict(supervisor.health("bad"))
    clock.advance(10.0)
    report = supervisor.run_round(clock())
    assert (report.ticked, report.skipped, report.failed) == (0, 1, 0)
    assert dataclasses.asdict(supervisor.health("bad")) == before


def test_an_expired_quarantine_admits_exactly_one_probe() -> None:
    clock, (bot, neighbour), supervisor = build("bad", "good")
    drive_to_quarantine(clock, supervisor, bot)
    at_quarantine = supervisor.health("bad").quarantined_until
    assert at_quarantine == clock() + QUARANTINE_S
    ticks_at_quarantine = bot.ticks

    # The probe is held back by whichever is longer, the quarantine or the
    # back-off this failure earned.  On a 300s bot four failures in a row
    # widen the interval to the 3600s cap, which is past QUARANTINE_S, and
    # the wait is the wider of the two: reaching quarantine must not make a
    # failing bot due *sooner* than the failure before it did.
    held_for = max(QUARANTINE_S, _widened_interval(INTERVAL, QUARANTINE_AFTER_FAILURES))
    assert supervisor.health("bad").next_due_at == clock() + held_for
    assert held_for > QUARANTINE_S

    # Still inside the wait: skipped however many rounds are run.
    for _ in range(3):
        clock.advance(held_for / 4.0)
        report = supervisor.run_round(clock())
        assert bot.ticks == ticks_at_quarantine
        assert report.skipped >= 1

    # Expired: exactly one probe goes through, and it fails, so the bot is
    # quarantined again rather than retried.
    clock.advance(held_for)
    report = supervisor.run_round(clock())
    assert bot.ticks == ticks_at_quarantine + 1
    assert report.failed == 1 and report.quarantined == 1
    assert supervisor.health("bad").quarantined_until == clock() + QUARANTINE_S

    # No second probe in the same quarantine.
    clock.advance(1.0)
    supervisor.run_round(clock())
    assert bot.ticks == ticks_at_quarantine + 1

    # A probe that succeeds clears the quarantine and the error.
    bot.fail = False
    clock.advance(held_for + 1.0)
    report = supervisor.run_round(clock())
    health = supervisor.health("bad")
    assert bot.ticks == ticks_at_quarantine + 2
    assert (health.consecutive_failures, health.quarantined_until) == (0, 0.0)
    assert health.last_error == "" and health.healthy()
    assert health.next_due_at == clock() + INTERVAL
    assert supervisor.badge_status()["state"] == "ok"
    assert neighbour.ticks > 0  # untouched throughout


def test_a_bot_that_returns_rubbish_fails_alone() -> None:
    """An event stamped with someone else's id would file attention, a card
    entry and a push under that bot; it is a failure of the bot that did it."""
    clock, (liar, honest), supervisor = build("liar", "honest")
    honest.script = asks("honest-open")
    liar.script = lambda bot, now: [
        Event(bot_id="honest", at=now, severity=Severity.ACTION, text="not mine",
              attention_key="stolen")
    ]
    report = supervisor.run_round(clock())
    assert report.failed == 1 and report.ticked == 1
    assert "honest" in supervisor.health("liar").last_error
    assert [item.key for item in supervisor.attention_items()] == ["honest-open"]

    liar.script = lambda bot, now: "not a sequence"
    clock.advance(BACKOFF_CAP_S + 1)  # past the back-off its first failure earned
    assert supervisor.run_round(clock()).failed == 1
    assert supervisor.health("honest").healthy()


# --------------------------------------------------------------------------
# "Paused means paused"
# --------------------------------------------------------------------------


def test_paused_bots_are_not_ticked_hold_no_attention_and_never_alert() -> None:
    alerts = FakeAlerts()
    clock, (bot, other), supervisor = build("poke", "other", alerts=alerts)
    bot.script = asks("restock", "Surging Sparks ETB is in stock")

    supervisor.run_round(clock())
    assert supervisor.attention_count() == 1 and len(alerts.published) == 1
    ticks_before = bot.ticks

    supervisor.pause("poke")
    assert bot.paused_calls == 1
    assert supervisor.is_paused("poke")
    assert supervisor.attention_items() == []  # a badge for something switched off is a lie
    assert supervisor.badge_status() == {"attention": 0, "state": "warn"}

    for _ in range(4):
        clock.advance(INTERVAL * 2)
        report = supervisor.run_round(clock())
        assert report.skipped >= 1
    assert bot.ticks == ticks_before          # not ticked
    assert len(alerts.published) == 1         # never alerted
    assert supervisor.attention_count() == 0  # no attention
    assert other.ticks >= 4                   # and the others carried on

    card = {c["id"]: c for c in supervisor.launcher_state(clock())["bots"]}["poke"]
    assert card["state"] == "paused" and card["attention"] == 0
    # History stays on the card -- it is how the owner decides to resume --
    # but it is no longer an open request and never a push.
    assert card["last_event"]["text"] == "Surging Sparks ETB is in stock"

    supervisor.resume("poke")
    assert bot.resumed_calls == 1
    clock.advance(1.0)
    supervisor.run_round(clock())
    assert bot.ticks == ticks_before + 1
    # The question is asked again after a resume, so it is news again.
    assert supervisor.attention_count() == 1 and len(alerts.published) == 2


def test_pause_is_idempotent_and_respects_can_pause() -> None:
    clock = FakeClock()
    pinned = DemoBot(clock, "pinned", can_pause=False)
    normal = DemoBot(clock, "normal")
    supervisor = Supervisor(BotRegistry([pinned, normal]), clock)
    with pytest.raises(SupervisorError):
        supervisor.pause("pinned")
    with pytest.raises(RegistryError):
        supervisor.pause("ghost")
    supervisor.pause("normal")
    supervisor.pause("normal")
    assert normal.paused_calls == 1
    supervisor.resume("normal")
    supervisor.resume("normal")
    assert normal.resumed_calls == 1
    supervisor.set_paused("normal", True)
    assert supervisor.paused_ids() == ["normal"]
    supervisor.set_paused("normal", False)
    assert supervisor.paused_ids() == []


# --------------------------------------------------------------------------
# Attention and alerts
# --------------------------------------------------------------------------


def test_attention_keys_collapse_repeats_and_clear() -> None:
    """contracts.py: "one restock nagging across ten ticks is one item of
    attention, not ten"."""
    alerts = FakeAlerts()
    clock, (bot,), supervisor = build("poke", alerts=alerts)
    bot.script = asks("restock", "in stock at $47.99")
    supervisor.run_round(clock())
    first = supervisor.attention_items()[0]
    assert first.since == clock() and first.text == "in stock at $47.99"

    for round_number in range(9):
        clock.advance(INTERVAL + 1)
        bot.script = asks("restock", f"still in stock, look {round_number}")
        supervisor.run_round(clock())
    items = supervisor.attention_items()
    assert len(items) == 1                       # ten ticks, one item
    assert items[0].since == first.since         # open since it was first asked
    assert items[0].text == "still in stock, look 8"  # but the text is current
    assert len(alerts.published) == 1            # and one push, not ten

    # A different question is a different item.
    clock.advance(INTERVAL + 1)
    bot.script = asks("preorder", "preorder opened")
    supervisor.run_round(clock())
    assert {item.key for item in supervisor.attention_items()} == {"restock", "preorder"}
    assert supervisor.badge_status() == {"attention": 2, "state": "ok"}
    assert supervisor.bots_wanting_attention() == ["poke"]
    assert len(alerts.published) == 2

    assert supervisor.clear_attention("poke", "restock") is True
    assert supervisor.clear_attention("poke", "restock") is False
    assert supervisor.attention_count() == 1
    assert supervisor.clear_attention("nobody", "nothing") is False

    # Cleared means the next occurrence is news again.
    clock.advance(INTERVAL + 1)
    bot.script = asks("restock", "back in stock")
    supervisor.run_round(clock())
    assert len(alerts.published) == 3
    assert supervisor.clear_bot_attention("poke") == 2
    assert supervisor.attention_items() == []


def test_alerts_respect_the_threshold_and_carry_a_link_not_a_cart() -> None:
    alerts = FakeAlerts()
    clock, (bot,), supervisor = build("poke", alerts=alerts)
    bot.script = asks("chatter", "just so you know", severity=Severity.NOTICE)
    supervisor.run_round(clock())
    assert alerts.published == []  # below the ACTION default
    assert supervisor.attention_count() == 0  # NOTICE never earns a badge item

    loud = Supervisor(supervisor.registry, clock, alerts=alerts, alert_min_severity="notice")
    clock.advance(INTERVAL + 1)
    loud.run_round(clock())
    published = alerts.published[-1]
    assert published["kind"] == "bot_event" and published["priority"] == "normal"
    assert published["title"] == bot.info.name and published["body"] == "just so you know"
    assert published["dedupe_key"] == "bot:poke|chatter"
    assert published["data"]["url"] == "/bots/poke"  # a link a person taps
    assert published["data"]["bot_id"] == "poke"
    assert json.dumps(published["data"])  # flat and serialisable


def test_a_broken_alert_service_does_not_break_the_round() -> None:
    alerts = FakeAlerts(fail=True)
    clock, (bot, other), supervisor = build("poke", "other", alerts=alerts)
    bot.script = asks("restock")
    report = supervisor.run_round(clock())
    assert report.ticked == 2 and report.failed == 0 and report.alerts == 0
    assert supervisor.health("poke").healthy()
    assert "RuntimeError" in supervisor.last_alert_error
    assert supervisor.attention_count() == 1  # the request is still open


def test_a_plain_callable_works_as_the_alert_sink() -> None:
    seen: List[Event] = []
    clock, (bot,), supervisor = build("poke", alerts=seen.append)
    bot.script = asks("restock")
    supervisor.run_round(clock())
    assert [event.attention_key for event in seen] == ["restock"]
    with pytest.raises(SupervisorError):
        Supervisor(BotRegistry(), clock, alerts=object())
    with pytest.raises(SupervisorError):
        Supervisor(BotRegistry(), object())
    with pytest.raises(SupervisorError):
        Supervisor(BotRegistry(), clock, store=object())


# --------------------------------------------------------------------------
# The badge and the launcher, against web/README.md
# --------------------------------------------------------------------------


def readme_json_blocks() -> List[Any]:
    text = WEB_README.read_text(encoding="utf-8")
    return [json.loads(block) for block in re.findall(r"```json\n(.*?)```", text, re.S)]


def readme_sample(*keys: str) -> Dict[str, Any]:
    for block in readme_json_blocks():
        if isinstance(block, dict) and set(keys) <= set(block):
            return block
    raise AssertionError(f"web/README.md has no JSON sample with {keys}")


def json_kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "null"


def test_badge_status_matches_the_documented_endpoint() -> None:
    sample = readme_sample("attention", "state")
    clock, (bot,), supervisor = build("poke")
    status = supervisor.badge_status()
    assert set(status) == set(sample)
    for key in sample:
        assert json_kind(status[key]) == json_kind(sample[key]), key
    assert isinstance(status["attention"], int) and not isinstance(status["attention"], bool)
    assert status["state"] in ("ok", "warn", "error")


def test_badge_status_for_every_combination_of_states() -> None:
    """error when any bot is quarantined, warn when any is paused and none
    quarantined, else ok -- for each of those, with and without attention."""
    for quarantined in (False, True):
        for paused in (False, True):
            for attention in (0, 2):
                clock, (a, b), supervisor = build("aaa", "bbb")
                if quarantined:
                    drive_to_quarantine(clock, supervisor, a)
                if paused:
                    supervisor.pause("bbb")
                for index in range(attention):
                    supervisor._open_attention(
                        Event(bot_id="aaa", at=clock(), severity=Severity.ACTION,
                              text=f"open {index}", attention_key=f"k{index}")
                    )
                expected_state = "error" if quarantined else ("warn" if paused else "ok")
                assert supervisor.badge_status() == {
                    "attention": attention,
                    "state": expected_state,
                }, (quarantined, paused, attention)

    # A paused bot's items never count, even if it somehow holds some.
    clock, (a, b), supervisor = build("aaa", "bbb")
    supervisor._open_attention(
        Event(bot_id="bbb", at=clock(), severity=Severity.ACTION, text="x", attention_key="k")
    )
    assert supervisor.badge_status() == {"attention": 1, "state": "ok"}
    supervisor._paused["bbb"] = True  # as a restored state file might
    assert supervisor.badge_status() == {"attention": 0, "state": "warn"}

    assert Supervisor(BotRegistry(), FakeClock()).badge_status() == {
        "attention": 0,
        "state": "ok",
    }


def test_launcher_state_matches_the_readme_contract_key_by_key() -> None:
    sample = readme_sample("generated_at", "bots")
    sample_card = sample["bots"][0]
    alerts = FakeAlerts()
    clock, (poke, other), supervisor = build("poke", "other", alerts=alerts)
    poke.script = asks("restock", "Buy: Surging Sparks ETB at $47.99")
    supervisor.run_round(clock())
    clock.advance(11.5)

    state = supervisor.launcher_state(clock())
    assert set(state) == set(sample), "top-level keys must match the README exactly"
    for key in sample:
        assert json_kind(state[key]) == json_kind(sample[key]), key
    assert state["generated_at"] == int(clock())
    assert isinstance(state["generated_at"], int)  # whole seconds, as documented

    card = state["bots"][0]
    assert set(card) == set(sample_card), "card keys must match the README exactly"
    for key, example in sample_card.items():
        assert json_kind(card[key]) == json_kind(example), key
    assert card == {
        "id": "poke",
        "name": "Poke watcher",
        "blurb": "Watches poke.",
        "kind": "bot",
        "state": "running",
        "attention": 1,
        "href": "/bots/poke",
        "can_pause": True,
        "stats": [{"label": "Ticks", "value": "1"}],
        "last_event": {"at": int(T0), "text": "Buy: Surging Sparks ETB at $47.99"},
    }
    assert set(card["stats"][0]) == set(sample_card["stats"][0])
    assert set(card["last_event"]) == set(sample_card["last_event"])
    assert card["state"] in {state.ui for state in BotState}
    assert json.loads(json.dumps(state)) == state  # it is what an endpoint returns

    # Every documented state reaches the page.
    supervisor.pause("other")
    drive_to_quarantine(clock, supervisor, poke)
    cards = {c["id"]: c for c in supervisor.launcher_state(clock())["bots"]}
    assert cards["other"]["state"] == "paused"
    assert cards["poke"]["state"] == "error"
    assert [c["id"] for c in supervisor.launcher_state(clock())["bots"]] == ["poke", "other"]


def test_launcher_state_survives_a_bot_whose_status_raises() -> None:
    clock, (bad, good), supervisor = build("bad", "good")
    bad.status = lambda: 1 / 0
    cards = {c["id"]: c for c in supervisor.launcher_state(clock())["bots"]}
    assert cards["bad"]["state"] == "error" and cards["bad"]["stats"] == []
    assert cards["good"]["state"] == "running"
    detailed = supervisor.launcher_state(clock(), include_detail=True)["bots"][0]
    assert "ZeroDivisionError" in detailed["detail"]
    assert set(detailed) - set(cards["bad"]) == {"detail"}


def test_include_detail_is_the_only_way_to_get_undocumented_keys() -> None:
    clock, (bot,), supervisor = build("poke")
    bot.script = asks("restock", "look")
    supervisor.run_round(clock())
    plain = supervisor.launcher_state(clock())["bots"][0]
    rich = supervisor.launcher_state(clock(), include_detail=True)["bots"][0]
    assert set(rich) - set(plain) == {"detail"}
    assert set(rich["last_event"]) - set(plain["last_event"]) == {"severity", "href"}


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def test_snapshot_restore_round_trips_including_health(tmp_path) -> None:
    store = JsonFileStore(str(tmp_path / "bots.json"))
    alerts = FakeAlerts()
    clock, (poke, weather, broken), supervisor = build(
        "poke", "weather", "broken", alerts=alerts, store=store
    )
    poke.script = asks("restock", "in stock")
    broken.fail = True
    for _ in range(2):
        clock.advance(BACKOFF_CAP_S + 1)
        supervisor.run_round(clock())
    supervisor.pause("weather")
    before_state = supervisor.save_state()
    assert json.loads(json.dumps(before_state)) == before_state  # JSON-able

    # A fresh process: new bot objects, new supervisor, same store.
    clock2 = FakeClock(clock())
    bots2 = [DemoBot(clock2, bot_id) for bot_id in ("poke", "weather", "broken")]
    supervisor2 = Supervisor(BotRegistry(bots2), clock2, store=store, alerts=alerts)
    assert supervisor2.load_state() is True

    for bot_id in ("poke", "weather", "broken"):
        assert supervisor2.health(bot_id) == supervisor.health(bot_id)
    assert bots2[0].ticks == poke.ticks and bots2[0].restored == {"ticks": poke.ticks}
    assert supervisor2.is_paused("weather") and not supervisor2.is_paused("poke")
    assert bots2[1].paused_calls == 0  # a restart is not the moment the owner paused
    assert supervisor2.attention_items() == supervisor.attention_items()
    assert supervisor2.badge_status() == supervisor.badge_status()
    assert supervisor2.launcher_state(clock2()) == supervisor.launcher_state(clock())
    assert supervisor2.save_state()["bots"] == before_state["bots"]

    # The alert memory came back: an open request does not re-push on restart.
    published_before = len(alerts.published)
    clock2.advance(INTERVAL + 1)
    bots2[0].script = asks("restock", "still in stock")
    supervisor2.run_round(clock2())
    assert len(alerts.published) == published_before


def test_load_state_keeps_state_for_bots_that_are_not_registered_now() -> None:
    store: Dict[str, Any] = {}
    clock, (poke, weather), supervisor = build("poke", "weather", store=store)
    supervisor.run_round(clock())
    supervisor.save_state()

    clock2 = FakeClock(clock())
    only_poke = Supervisor(BotRegistry([DemoBot(clock2, "poke")]), clock2, store=store)
    only_poke.load_state()
    assert only_poke.registry.ids() == ["poke"]
    assert "weather" in only_poke.save_state()["bots"]  # kept, not lost

    # And it comes back when the bot returns.
    clock3 = FakeClock(clock2())
    both = [DemoBot(clock3, "poke"), DemoBot(clock3, "weather")]
    returning = Supervisor(BotRegistry(both), clock3, store=store)
    returning.load_state()
    assert both[1].ticks == weather.ticks


def test_pause_persists_through_the_store() -> None:
    store: Dict[str, Any] = {}
    clock, (bot,), supervisor = build("poke", store=store)
    supervisor.pause("poke")
    assert store["jarvis_bots"]["bots"]["poke"]["paused"] is True
    supervisor.resume("poke")
    assert store["jarvis_bots"]["bots"]["poke"]["paused"] is False


def test_load_state_refuses_what_it_cannot_read() -> None:
    clock, (bot,), supervisor = build("poke")
    with pytest.raises(SupervisorError):
        supervisor.load_state()  # no store, no state
    assert supervisor.load_state({}) is False
    with pytest.raises(SupervisorError):
        supervisor.load_state({"version": STATE_VERSION_TOO_NEW, "bots": {}})
    with pytest.raises(SupervisorError):
        supervisor.load_state({"bots": {"poke": "not an object"}})
    with pytest.raises(SupervisorError):
        supervisor.load_state({"bots": []})


STATE_VERSION_TOO_NEW = 99


def test_a_bot_whose_snapshot_raises_does_not_stop_the_save() -> None:
    clock, (bad, good), supervisor = build("bad", "good")
    bad.snapshot = lambda: 1 / 0
    state = supervisor.save_state()
    assert state["bots"]["bad"]["snapshot"] == {}
    assert state["bots"]["good"]["snapshot"] == {"ticks": 0}
    assert "ZeroDivisionError" in supervisor.last_state_error


def test_json_file_store_is_atomic_and_private(tmp_path) -> None:
    import os
    import stat

    path = tmp_path / "nested" / "bots.json"
    store = JsonFileStore(str(path))
    assert store.load() == {}  # nothing yet
    store.save({"version": 1, "bots": {}})
    assert store.load() == {"version": 1, "bots": {}}
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    assert not [p for p in path.parent.iterdir() if p.name.startswith(".bots-")]
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(SupervisorError):
        store.load()


def test_forget_drops_every_trace() -> None:
    clock, (bot,), supervisor = build("poke")
    bot.script = asks("restock")
    supervisor.run_round(clock())
    supervisor.pause("poke")
    supervisor.registry.unregister("poke")
    supervisor.forget("poke")
    assert supervisor.badge_status() == {"attention": 0, "state": "ok"}
    assert supervisor.save_state()["bots"] == {}


# --------------------------------------------------------------------------
# Timing and determinism
# --------------------------------------------------------------------------


def test_a_slow_tick_is_reported() -> None:
    clock, (slow, quick), supervisor = build("slow", "quick")
    slow.tick_cost = SLOW_TICK_S + 1.0
    report = supervisor.run_round(clock())
    assert report.slow == ("slow",)
    assert report.ticked == 2  # reported, not killed: the framework cannot kill it

    slow.tick_cost = 0.0
    clock.advance(INTERVAL * 10)
    assert supervisor.run_round(clock()).slow == ()

    # A tick that is slow *and* fails is still reported as slow.
    slow.tick_cost = SLOW_TICK_S + 1.0
    slow.fail = True
    clock.advance(INTERVAL * 10)
    report = supervisor.run_round(clock())
    assert report.slow == ("slow",) and report.failed == 1


def run_sequence(seed: int, rounds: int = 12) -> List[str]:
    """One fixed sequence of rounds, driven by a seed stream.

    Randomness comes from ``lucifer_gen.seed`` and nowhere else, and the
    clock is the fake: the transcript is therefore a pure function of the
    seed, which is what "a fixed sequence of rounds is deterministic" means.
    """
    clock = FakeClock()
    bots = [DemoBot(clock, f"bot{index}") for index in range(3)]
    alerts = FakeAlerts()
    supervisor = Supervisor(BotRegistry(bots), clock, alerts=alerts)
    stream = SeedFields.parse(seed).stream("bots:determinism")
    transcript: List[str] = []
    for _ in range(rounds):
        clock.advance(stream.choice((60.0, 400.0, 4000.0)))
        for bot in bots:
            bot.fail = stream.chance(0.4)
            bot.script = asks(f"k{stream.randint(0, 1)}") if stream.chance(0.5) else None
        report = supervisor.run_round(clock())
        if stream.chance(0.1):
            target = stream.choice([b.info.id for b in bots])
            if supervisor.is_paused(target):
                supervisor.resume(target)
            else:
                supervisor.pause(target)
        transcript.append(
            json.dumps(
                {
                    "report": dataclasses.astuple(report),
                    "badge": supervisor.badge_status(),
                    "launcher": supervisor.launcher_state(clock()),
                    "attention": [dataclasses.astuple(i) for i in supervisor.attention_items()],
                    "alerts": alerts.published,
                },
                sort_keys=True,
            )
        )
    transcript.append(json.dumps(supervisor.save_state(), sort_keys=True))
    return transcript


def test_a_fixed_sequence_of_rounds_is_deterministic() -> None:
    assert run_sequence(0xC0FFEE) == run_sequence(0xC0FFEE)
    assert run_sequence(0xC0FFEE) != run_sequence(0xBADBEE)  # the seed really drives it


# --------------------------------------------------------------------------
# What this package is not allowed to do
# --------------------------------------------------------------------------


def owned_sources() -> Dict[str, str]:
    return {
        name: (ROOT / "jarvis_bots" / name).read_text(encoding="utf-8") for name in OWNED
    }


def test_no_module_reads_a_real_clock_or_draws_its_own_randomness() -> None:
    """contracts.py: "Time is injected everywhere. Nothing here calls
    time.time()." -- and randomness may only come from lucifer_gen.seed."""
    for name, source in owned_sources().items():
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
        assert "random" not in imported, f"{name} draws its own randomness"
        assert "time" not in imported, f"{name} imports time"
        assert "datetime" not in imported, f"{name} imports datetime"
        for network in ("socket", "urllib", "http", "requests", "ssl"):
            assert network not in imported, f"{name} imports {network}"
        # Checked on the tree, not the text: the docstrings quote the rule.
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            base = node.func.value
            assert not (
                isinstance(base, ast.Name) and base.id in ("time", "datetime", "random")
            ), f"{name} calls {base.id}.{node.func.attr}()"


def test_nothing_in_the_framework_can_buy_anything() -> None:
    """SCOPE: this framework schedules bots and surfaces what they find.  A
    bot that concludes the owner should buy something raises an event with a
    link, and a person acts on it."""
    names = set()
    for name, source in owned_sources().items():
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name.lower())
            elif isinstance(node, ast.Attribute):
                names.add(node.attr.lower())
    for word in ("cart", "checkout", "payment", "purchase", "captcha", "cvv", "order"):
        assert not [n for n in names if word in n], f"{word} has no business here"


def test_health_is_the_supervisors_and_the_snapshot_is_the_bots() -> None:
    """contracts.py: "Owned by the supervisor, not the bot" / "State is the
    bot's, persistence is ours"."""
    clock, (bot,), supervisor = build("poke")
    supervisor.run_round(clock())
    assert not hasattr(bot, "health")
    assert supervisor.health("poke").total_ticks == 1
    assert "health" not in bot.snapshot()
    assert set(dataclasses.asdict(Health())) <= set(
        supervisor.save_state()["bots"]["poke"]["health"]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))


# --------------------------------------------------------------------------
# Isolation, for real: everything a bot's own code can reach
# --------------------------------------------------------------------------


def returns(*events):
    def script(bot, now):
        return list(events)

    return script


def test_a_hand_built_event_with_a_bad_severity_fails_only_its_own_bot() -> None:
    """``_apply_events`` used to run outside the try, so an event carrying
    junk aborted the round *and* recorded nothing: the bot never backed
    off, the badge read ok, and the next round died at the same place."""
    clock, (bad, victim), supervisor = build("bad", "victim")

    poisoned = Event(bot_id="bad", at=clock(), severity=Severity.ACTION, text="hi")
    object.__setattr__(poisoned, "severity", object())  # past __post_init__
    bad.script = returns(poisoned)

    report = supervisor.run_round(clock())
    assert victim.ticks == 1, "one bot's bad event must not cost another its round"
    assert report.failed == 1 and report.ticked == 1
    health = supervisor.health("bad")
    assert health.consecutive_failures == 1 and health.last_error
    assert supervisor.attention_count() == 0


def test_an_unprintable_attention_key_fails_only_its_own_bot() -> None:
    class Rude:
        def __str__(self) -> str:
            raise RuntimeError("boom")

    clock, (bad, victim), supervisor = build("bad", "victim")
    poisoned = Event(
        bot_id="bad", at=clock(), severity=Severity.ACTION, text="hi", attention_key="k"
    )
    object.__setattr__(poisoned, "attention_key", Rude())
    bad.script = returns(poisoned)

    supervisor.run_round(clock())
    assert victim.ticks == 1
    assert supervisor.health("bad").consecutive_failures == 1
    assert supervisor.attention_count() == 0


def test_an_event_validates_itself_where_it_is_built() -> None:
    """The mistake should surface on the line that made it."""
    with pytest.raises(TypeError):
        Event(bot_id="b", at=T0, severity=object(), text="x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Event(bot_id="b", at=T0, severity=Severity.ACTION, text="x", attention_key=5)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Event(bot_id="b", at=float("nan"), severity=Severity.INFO, text="x")
    with pytest.raises(TypeError):
        Event(bot_id="b", at=T0, severity=Severity.INFO, text=b"bytes")  # type: ignore[arg-type]
    # An int severity is a config file's spelling, not a mistake.
    assert Event(bot_id="b", at=T0, severity=3, text="x").severity is Severity.ACTION  # type: ignore[arg-type]


def test_an_endless_tick_return_does_not_hang_the_round() -> None:
    """A generator is Iterable. The validation loop had no length cap, so
    the round died in the code written to protect it -- and no try/except
    could help, because nothing raised."""
    from jarvis_bots.contracts import MAX_EVENTS_PER_TICK

    clock, (greedy, victim), supervisor = build("greedy", "victim")

    def endless(bot, now):
        def gen():
            while True:
                yield bot.event(Severity.INFO, "more")

        return gen()

    greedy.script = endless
    report = supervisor.run_round(clock())
    assert victim.ticks == 1
    assert report.failed == 1
    assert "MAX" in supervisor.health("greedy").last_error or str(
        MAX_EVENTS_PER_TICK
    ) in supervisor.health("greedy").last_error
    assert supervisor.attention_count() == 0


def test_a_systemexit_from_a_tick_is_the_bots_failure_not_the_fleets() -> None:
    """contracts.py: "Raising is allowed and is handled", unqualified.
    ``sys.exit`` inside a library is a SystemExit, and catching only
    Exception took the whole fleet down with health untouched."""

    class Exiter(DemoBot):
        def tick(self, now: float):
            self.ticks += 1
            raise SystemExit(1)

    clock = FakeClock()
    exiter = Exiter(clock, "exiter")
    victim = DemoBot(clock, "victim")
    supervisor = Supervisor(BotRegistry([exiter, victim]), clock)

    report = supervisor.run_round(clock())
    assert victim.ticks == 1
    assert report.failed == 1
    health = supervisor.health("exiter")
    assert health.consecutive_failures == 1
    assert "SystemExit" in health.last_error


def test_a_keyboardinterrupt_is_recorded_and_then_re_raised() -> None:
    """The failure is the bot's; Ctrl-C is the operator's."""

    class Interrupted(DemoBot):
        def tick(self, now: float):
            self.ticks += 1
            raise KeyboardInterrupt

    clock = FakeClock()
    bot = Interrupted(clock, "interrupted")
    supervisor = Supervisor(BotRegistry([bot]), clock)
    with pytest.raises(KeyboardInterrupt):
        supervisor.run_round(clock())
    assert supervisor.health("interrupted").consecutive_failures == 1


def test_an_exception_whose_str_raises_does_not_escape_the_handler() -> None:
    class Nasty(Exception):
        def __str__(self) -> str:
            raise RuntimeError("even the message is broken")

    clock, (bad, victim), supervisor = build("bad", "victim")

    def blow_up(bot, now):
        raise Nasty()

    bad.script = blow_up
    supervisor.run_round(clock())
    assert victim.ticks == 1
    assert supervisor.health("bad").consecutive_failures == 1
    assert supervisor.health("bad").last_error


def test_backoff_never_overflows_however_long_a_bot_stays_broken() -> None:
    """The counter is an exponent and nothing resets it while quarantined,
    so a watcher pointed at a retired endpoint used to reach 2.0**1024 --
    inside the failure handler, which took every round with it, for ever."""
    clock, (bad,), supervisor = build("bad")
    bad.fail = True
    supervisor.health("bad").consecutive_failures = 5000
    clock.advance(BACKOFF_CAP_S * 10)
    report = supervisor.run_round(clock())
    assert report.failed == 1
    assert supervisor.health("bad").next_due_at > clock()


def test_a_restored_failure_count_is_bounded() -> None:
    from jarvis_bots.supervisor import MAX_FAILURES_PERSISTED

    clock, (bad,), supervisor = build("bad", store={})
    supervisor.load_state(
        {"version": 1, "bots": {"bad": {"health": {"consecutive_failures": 10**9}}}}
    )
    assert supervisor.health("bad").consecutive_failures == MAX_FAILURES_PERSISTED


def test_a_bot_registered_after_load_state_keeps_its_quarantine() -> None:
    """A lazy registry, a feature flag, a config resolved after startup.
    The bot used to come up healthy with its back-off reset, and the next
    save overwrote the state that had been kept aside for it."""
    clock = FakeClock()
    store: Dict[str, Any] = {}
    bot = DemoBot(clock, "broken")
    first = Supervisor(BotRegistry([bot]), clock, store=store)
    bot.fail = True
    drive_to_quarantine(clock, first, bot)
    first.save_state()
    assert first.is_quarantined("broken")

    registry = BotRegistry()
    second = Supervisor(registry, clock, store=store)
    second.load_state()
    later = DemoBot(clock, "broken")
    registry.register(later)

    assert second.is_quarantined("broken")
    assert second.health("broken").consecutive_failures == QUARANTINE_AFTER_FAILURES
    assert later.ticks == bot.ticks  # the snapshot came with it
    assert second.badge_status()["state"] == "error"
    assert second.save_state()["bots"]["broken"]["snapshot"]["ticks"] == bot.ticks


def test_one_failed_snapshot_does_not_erase_what_the_bot_knew() -> None:
    """The fallback looked in _orphans, which is empty for a registered
    bot -- so a locked cache file replaced the watchlist with {}."""
    clock, (keeper,), supervisor = build("keeper", store={})
    keeper.ticks = 7
    supervisor.save_state()

    def boom() -> Dict[str, Any]:
        raise RuntimeError("cache file locked")

    keeper.snapshot = boom  # type: ignore[assignment]
    state = supervisor.save_state()
    assert state["bots"]["keeper"]["snapshot"] == {"ticks": 7}
    assert "cache file locked" in supervisor.last_state_error


def test_a_bot_cannot_grow_the_badge_without_bound() -> None:
    from jarvis_bots.contracts import MAX_ATTENTION_PER_BOT

    clock, (noisy,), supervisor = build("noisy", alerts=FakeAlerts())

    def many(bot, now):
        return [
            bot.event(Severity.ACTION, f"q{i}", attention_key=f"k{i}")
            for i in range(MAX_ATTENTION_PER_BOT + 50)
        ]

    noisy.script = many
    supervisor.run_round(clock())
    assert supervisor.attention_count() == MAX_ATTENTION_PER_BOT
    assert supervisor.last_overflow_error
    # A key the badge will not carry must not buzz the phone either.
    assert len(supervisor._alerts.published) == MAX_ATTENTION_PER_BOT


# --------------------------------------------------------------------------
# The alert memory belongs to the open question
# --------------------------------------------------------------------------


def test_a_resolution_does_not_poison_the_memory_of_its_own_key() -> None:
    """The badge goes to 1 and the phone stays silent. ``clear_attention``
    dropped the memory and ``_maybe_alert``, publishing the resolution in
    the same iteration, put it straight back."""
    from jarvis_bots.supervisor import RESOLVED_FLAG

    alerts = FakeAlerts()
    clock = FakeClock()
    bot = DemoBot(clock, "shop")
    supervisor = Supervisor(
        BotRegistry([bot]), clock, alerts=alerts, alert_min_severity=Severity.NOTICE
    )

    bot.script = asks("restock", "Buy: it is back")
    supervisor.run_round(clock())
    assert [row["body"] for row in alerts.published] == ["Buy: it is back"]

    bot.script = returns(
        Event(
            bot_id="shop",
            at=clock(),
            severity=Severity.NOTICE,
            text="no longer a buy",
            attention_key="restock",
            data={RESOLVED_FLAG: True},
        )
    )
    clock.advance(INTERVAL)
    supervisor.run_round(clock())
    assert supervisor.attention_count() == 0

    bot.script = asks("restock", "Buy: it is back")
    clock.advance(INTERVAL)
    supervisor.run_round(clock())
    assert supervisor.attention_count() == 1
    assert [row["body"] for row in alerts.published].count("Buy: it is back") == 2


def test_a_sub_action_event_reusing_a_key_does_not_suppress_the_real_push() -> None:
    """Under a lowered threshold any keyed event used to file the key as
    "already pushed", permanently suppressing the ACTION that followed."""
    alerts = FakeAlerts()
    clock = FakeClock()
    bot = DemoBot(clock, "shop")
    supervisor = Supervisor(
        BotRegistry([bot]), clock, alerts=alerts, alert_min_severity=Severity.NOTICE
    )

    bot.script = asks("restock", "just watching", severity=Severity.NOTICE)
    supervisor.run_round(clock())
    assert supervisor.attention_count() == 0  # NOTICE never opens a request

    bot.script = asks("restock", "Buy: it is back")
    clock.advance(INTERVAL)
    supervisor.run_round(clock())
    assert supervisor.attention_count() == 1
    assert "Buy: it is back" in [row["body"] for row in alerts.published]


# --------------------------------------------------------------------------
# The feed the detail page reads
# --------------------------------------------------------------------------


def test_the_supervisor_keeps_a_bounded_feed_per_bot() -> None:
    from jarvis_bots.supervisor import EVENT_HISTORY

    clock, (bot,), supervisor = build("chatty", store={})
    counter = {"n": 0}

    def chatter(b, now):
        counter["n"] += 1
        return [b.event(Severity.INFO, f"line {counter['n']}")]

    bot.script = chatter
    for _ in range(EVENT_HISTORY + 12):
        supervisor.run_round(clock())
        clock.advance(INTERVAL)

    feed = supervisor.events("chatty")
    assert len(feed) == EVENT_HISTORY
    assert feed[0].text == f"line {counter['n']}", "newest first"
    assert supervisor.events("chatty", limit=3) == feed[:3]

    # and it survives a restart, because the page reads it after one
    state = supervisor.save_state()
    clone = DemoBot(clock, "chatty")
    fresh = Supervisor(BotRegistry([clone]), clock)
    fresh.load_state(state)
    assert [e.text for e in fresh.events("chatty")] == [e.text for e in feed]


def test_bot_detail_is_the_shape_the_generated_page_reads() -> None:
    clock, (bot,), supervisor = build("weather")
    bot.script = asks("rain", "Rain in an hour")
    supervisor.run_round(clock())

    detail = supervisor.bot_detail("weather", clock())
    assert sorted(detail) == ["bot", "generated_at"]
    card = detail["bot"]
    for key in ("id", "name", "state", "stats", "detail", "attention_items", "events"):
        assert key in card
    assert card["attention_items"] == [
        {"key": "rain", "text": "Rain in an hour", "href": "/bots/weather",
         "since": int(clock())}
    ]
    assert card["events"][0]["severity"] == "action"
    assert json.loads(json.dumps(detail)) == detail

    with pytest.raises(RegistryError):
        supervisor.bot_detail("nope")


# --------------------------------------------------------------------------
# Identity is static, and the floor is a floor
# --------------------------------------------------------------------------


def test_a_bot_cannot_change_its_identity_inside_a_tick() -> None:
    """``run_round`` captured the id before the tick and re-read info
    after it, so a bot that rebuilt its own info filed its event, its
    card, its push and its dedupe key under another bot's id -- and set
    its own next tick to whatever interval it liked."""

    class Fake:
        id, name, blurb, kind, href, can_pause = "victim", "Victim", "", "bot", "/v", True
        interval_s = 0.001

    alerts = FakeAlerts()
    clock = FakeClock()
    rude = DemoBot(clock, "rude")
    victim = DemoBot(clock, "victim")

    def swap(bot, now):
        event = bot.event(Severity.ACTION, "buy now", attention_key="k", href="/rude")
        bot.info = Fake()  # type: ignore[assignment]
        return [event]

    rude.script = swap
    supervisor = Supervisor(BotRegistry([rude, victim]), clock, alerts=alerts)
    supervisor.run_round(clock())

    assert alerts.published == [], "nothing may be pushed in another bot's name"
    assert supervisor.attention_count() == 0
    assert supervisor.health("rude").consecutive_failures == 1
    assert supervisor.health("rude").next_due_at - clock() >= 5.0
    cards = {c["id"]: c for c in supervisor.launcher_state(clock())["bots"]}
    assert sorted(cards) == ["rude", "victim"], "cards key on the registered id"
    assert cards["rude"]["state"] == "error"
    assert victim.ticks == 1


def test_the_five_second_interval_floor_holds_after_registration_too() -> None:
    from jarvis_bots.supervisor import _interval_of

    clock, (bot,), supervisor = build("bot")
    assert _interval_of(bot.info) == INTERVAL
    supervisor.run_round(clock())
    object.__setattr__(bot.info, "interval_s", 0.0)
    clock.advance(INTERVAL)
    supervisor.run_round(clock())
    assert supervisor.health("bot").next_due_at - clock() >= 5.0

    object.__setattr__(bot.info, "interval_s", float("nan"))
    clock.advance(INTERVAL)
    supervisor.run_round(clock())
    due = supervisor.health("bot").next_due_at
    assert due == due and due - clock() >= 5.0  # not NaN, and not immediate


def test_a_paused_bots_attention_does_not_come_back_out_of_the_state_file() -> None:
    """``pause()`` clears the items and early-returns for a bot that is
    already paused, so a file carrying both a pause and that bot's items
    handed them straight back."""
    clock, (bot,), supervisor = build("nagger")
    supervisor.load_state(
        {
            "version": 1,
            "bots": {"nagger": {"paused": True}},
            "attention": [
                {"bot_id": "nagger", "key": "k", "text": "t", "href": "", "since": T0}
            ],
            "alerted": [["nagger", "k", T0]],
        }
    )
    assert supervisor.is_paused("nagger")
    assert supervisor.attention_count() == 0
    assert supervisor.save_state()["attention"] == []
    supervisor.resume("nagger")
    assert supervisor.attention_count() == 0
