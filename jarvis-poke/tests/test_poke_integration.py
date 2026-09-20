"""Cross-module integration: the seams six lanes wrote independently.

Every other ``tests/test_poke_*.py`` file exercises one module against its
own fakes.  That is exactly the shape of test that cannot catch a lane
disagreeing with its neighbour about an argument's name, its position, or
what a field means -- so this file only ever wires the *real* modules
together, and the real ``jarvis_alerts`` behind them.

What it pins, and why each one is here rather than in a lane's own file:

* **The package surface.**  ``jarvis_poke/__init__.py`` must not re-export a
  name that is also a submodule.  ``jarvis_gen`` shipped that bug: a
  re-exported function named like a module makes ``from pkg import module``
  answer with whichever the import order happened to leave in
  ``sys.modules``.  The check is mechanical so a future export cannot
  reintroduce it.
* **The page's contract.**  ``jarvis_poke/web/index.html`` carries an
  embedded demo dataset and ``cli.page_state`` produces the live one.  Two
  authors, one JSON shape, no shared code: they drift silently unless
  something walks both.
* **The engine's discovered wiring.**  ``engine._resolve_market_ref`` and
  ``_resolve_outlier_check`` sniff the market lane's shape at construction.
  Sniffing is what lets the lanes land in any order; it is also what fails
  quietly when the shape moves, so the resolution is asserted against the
  module it resolves to.
* **The alert path.**  ``alerts_bridge`` calls ``AlertService.publish``
  positionally.  Its own tests pass a fake, and a fake agrees with whatever
  it was written against.
"""

from __future__ import annotations

import ast
import html
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import jarvis_poke
from jarvis_poke import cli, engine, prices, validate
from jarvis_poke.alerts_bridge import (
    AlertBridge,
    alert_data,
    dedupe_key_for,
    landed_cents,
)
from jarvis_poke.catalog import load_catalog
from jarvis_poke.contracts import (
    Action,
    Budget,
    Observation,
    Rule,
    Stock,
    Verdict,
)
from jarvis_poke.engine import DecisionEngine, EngineError
from jarvis_poke.prices import PriceHistory
from jarvis_poke.rules import RuleSet
from jarvis_poke.store import PokeStore, first_difference, round_trip_equal

#: Every module under ``jarvis_poke/``, plus the two package directories.
SUBMODULES = (
    "alerts_bridge",
    "catalog",
    "cli",
    "contracts",
    "engine",
    "prices",
    "rules",
    "sources",
    "store",
    "validate",
)
PACKAGE_DIRS = ("data", "web")

T0 = 1_700_000_000.0


def clock_at(value):
    """An injected clock over a one-element list, so a test can move time."""
    return lambda: value[0]


# --------------------------------------------------------------------------
# The package surface
# --------------------------------------------------------------------------


class PackageSurface(unittest.TestCase):
    def test_no_export_shadows_a_submodule_name(self):
        """The jarvis_gen bug: a re-exported name that is also a module.

        ``from jarvis_poke import catalog`` must be the module every time,
        not the module on a cold import and a function on a warm one.
        """
        reserved = set(SUBMODULES) | set(PACKAGE_DIRS)
        for name in jarvis_poke.__all__:
            if name in reserved:
                value = getattr(jarvis_poke, name)
                self.assertTrue(
                    inspect.ismodule(value),
                    f"jarvis_poke.{name} is {value!r}, not the {name} submodule",
                )

    def test_every_submodule_is_reachable_as_an_attribute(self):
        for name in SUBMODULES:
            self.assertTrue(inspect.ismodule(getattr(jarvis_poke, name)))

    def test_every_name_in_all_resolves(self):
        missing = [n for n in jarvis_poke.__all__ if not hasattr(jarvis_poke, n)]
        self.assertEqual([], missing)

    def test_all_has_no_duplicates(self):
        names = list(jarvis_poke.__all__)
        self.assertEqual(sorted(set(names)), sorted(names))

    def test_a_cold_interpreter_imports_each_submodule_as_a_module(self):
        """The shadowing bug only shows on a cold import, so buy one."""
        script = (
            "import sys; sys.path.insert(0, %r)\n"
            "import inspect\n"
            "from jarvis_poke import %s as m\n"
            "assert inspect.ismodule(m), m\n"
            "print(m.__name__)\n"
        )
        for name in SUBMODULES:
            with self.subTest(submodule=name):
                proc = subprocess.run(
                    [sys.executable, "-c", script % (ROOT, name)],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    0, proc.returncode, f"{name}: {proc.stderr.strip()}"
                )
                self.assertEqual(f"jarvis_poke.{name}", proc.stdout.strip())

    def test_importing_the_package_does_not_drag_in_the_heavy_lanes(self):
        """``import jarvis_poke`` stays light; cli/validate arrive on demand.

        ``alerts_bridge`` needs ``jarvis_alerts`` installed, and the other
        two pull in the whole package.  None of that should be the price of
        importing a dataclass.
        """
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                f"import sys; sys.path.insert(0, {ROOT!r})\n"
                "import jarvis_poke, sys\n"
                "lazy = [m for m in ('jarvis_poke.cli', 'jarvis_poke.validate',"
                " 'jarvis_poke.alerts_bridge') if m in sys.modules]\n"
                "print(','.join(lazy))\n",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, proc.returncode, proc.stderr)
        self.assertEqual("", proc.stdout.strip())

    def test_the_ambiguous_names_are_not_flattened(self):
        """Two modules define each; a package-level spelling would be a guess."""
        for name in ("main", "DEFAULT_SEED", "FRESH_S"):
            self.assertNotIn(name, jarvis_poke.__all__)


# --------------------------------------------------------------------------
# engine <-> prices
# --------------------------------------------------------------------------


def _history_with_a_recent_cliff():
    """Cheap for the last two days, dear for the twenty before it.

    A 5-day reference and a 30-day one disagree about this product, which
    is what makes it able to tell which window was actually used.
    """
    history = PriceHistory()
    for i in range(10):
        history.append(
            Observation(
                "p", "examplemart", "EX-1", T0 - (i + 1) * 2 * 86400,
                Stock.IN_STOCK, 1000 + i * 1000,
            )
        )
    return history


class EnginePricesSeam(unittest.TestCase):
    def test_market_window_s_reaches_the_market_lane(self):
        """``PriceHistory`` grew a ``market_ref`` with its own default window.

        The engine finds that method first and used to call it as
        ``(product_id, now)``, so ``market_window_s=`` was silently dropped
        and the engine scored against a 30-day reference it had not asked
        for -- here a $50.00 median instead of $10.00, which is the
        difference between a BUY and a refusal.
        """
        history = _history_with_a_recent_cliff()
        narrow = engine._resolve_market_ref(history, 5 * 86400.0)("p", T0)
        wide = engine._resolve_market_ref(history, 30 * 86400.0)("p", T0)
        self.assertEqual(5 * 86400.0, narrow.window_s)
        self.assertEqual(30 * 86400.0, wide.window_s)
        self.assertEqual(2, narrow.samples)
        self.assertEqual(10, wide.samples)
        direct = prices.market_reference(history, "p", T0, window_s=5 * 86400.0)
        self.assertEqual(direct, narrow)

    def test_the_engine_uses_the_window_it_was_built_with(self):
        history = _history_with_a_recent_cliff()
        for window, expected in ((5 * 86400.0, 2), (30 * 86400.0, 10)):
            with self.subTest(window=window):
                eng = DecisionEngine(
                    None, RuleSet(), history, lambda: T0, market_window_s=window
                )
                self.assertEqual(expected, eng._reference("p", T0).samples)

    def test_the_outlier_gate_resolves_to_the_market_lane(self):
        gate, origin = engine._resolve_outlier_check(None, None)
        self.assertEqual("prices.is_outlier", origin)
        ref = prices.market_reference(_history_with_a_recent_cliff(), "p", T0)
        self.assertTrue(gate(499, ref), "a 95% 'discount' is a mis-parse")
        self.assertFalse(gate(5500, ref))

    def test_a_rule_set_passed_as_the_catalog_is_refused_at_construction(self):
        """``catalog`` is the first parameter and ``rules`` the second.

        A ``RuleSet`` supports ``in``, which is all ``_known`` needs, so the
        slip used to build an engine with *no rules* that answered every
        product with SKIP "no rule for X" -- a monitor that has quietly
        stopped monitoring.
        """
        with self.assertRaises(EngineError) as caught:
            DecisionEngine(RuleSet(budget=Budget(total=1000)), clock=lambda: T0)
        self.assertIn("rule set, not a catalog", str(caught.exception))

    def test_a_catalog_shaped_object_and_none_are_both_accepted(self):
        self.assertIsNotNone(DecisionEngine(load_catalog(), RuleSet(), None, lambda: T0))
        self.assertIsNotNone(DecisionEngine(None, RuleSet(), None, lambda: T0))
        self.assertIsNotNone(DecisionEngine({"p"}, RuleSet(), None, lambda: T0))

    def test_discount_runs_the_opposite_way_round_in_the_two_modules(self):
        """A standing trap, pinned so a refactor cannot quietly align them.

        ``engine.discount_against(reference, landed)`` and
        ``prices.discount_pct(landed, reference)`` take the same two numbers
        in opposite orders and agree on the answer.
        """
        self.assertAlmostEqual(
            engine.discount_against(10000, 9000),
            prices.discount_pct(9000, 10000),
            places=6,
        )


# --------------------------------------------------------------------------
# engine -> alerts_bridge -> jarvis_alerts
# --------------------------------------------------------------------------


class AlertPath(unittest.TestCase):
    def test_the_bridge_calls_the_real_publish_signature(self):
        """The bridge's own tests use a fake, which agrees with itself."""
        from jarvis_alerts.api import AlertService

        parameters = list(inspect.signature(AlertService.publish).parameters)
        self.assertEqual(
            ["self", "profile_id", "kind", "title", "body", "data", "priority",
             "dedupe_key"],
            parameters,
        )

    def test_a_buy_reaches_a_device_through_the_real_outbox(self):
        from jarvis_alerts import contracts as alerts_contracts
        from jarvis_alerts.api import AlertService
        from jarvis_alerts.outbox import Outbox
        from jarvis_alerts.transports import FakeTransport
        from jarvis_alerts.worker import Worker

        now = [T0]
        clock = clock_at(now)
        catalog = load_catalog()
        product = catalog.products()[0]

        history = PriceHistory()
        for i in range(12):
            history.append(
                Observation(
                    product.id, "examplemart", "EX-1", T0 - (i + 1) * 3600,
                    Stock.IN_STOCK, 15000 + i * 50,
                    url="https://examplemart.example.com/p/x",
                )
            )
        best = Observation(
            product.id, "cardbarn", "CA-1", T0 - 60, Stock.IN_STOCK, 13000,
            url="https://cardbarn.example.com/p/x",
        )
        history.append(best)

        rules = RuleSet(budget=Budget(total=60000))
        rules.add(Rule(product_id=product.id, max_price=14000, min_discount_pct=5.0))
        verdict = DecisionEngine(catalog, rules, history, clock).evaluate(
            product.id, [best]
        )
        self.assertIs(Action.BUY, verdict.action, verdict.reasons)
        self.assertTrue(verdict.reasons, "contracts.py: reasons explain every outcome")

        path = os.path.join(tempfile.mkdtemp(), "alerts.sqlite3")
        outbox = Outbox(path, clock)
        outbox.migrate()
        outbox.register(
            alerts_contracts.Subscription(
                "owner", "phone", "fake", '{"endpoint":"POKE"}', clock()
            )
        )
        bridge = AlertBridge(AlertService(outbox, clock), clock)
        alert_id = bridge.publish_verdict(verdict, product)
        self.assertTrue(alert_id)

        seen = []

        def record(subscription, alert, index):
            seen.append(alert)
            return alerts_contracts.SendResult(ok=True)

        Worker(outbox, {"fake": FakeTransport(record)}, clock, lambda: 0.0).run_once()
        self.assertEqual(1, outbox.stats().get("delivered"))
        self.assertEqual(1, len(seen))
        self.assertEqual("poke_buy", seen[0].kind)
        self.assertIn(".example.com/", (seen[0].data or {}).get("url", ""))
        json.dumps(seen[0].data)  # AlertService requires JSON-encodable data

    def test_only_buy_is_published(self):
        published = []

        class Service:
            def publish(self, profile_id, kind, title, body, data=None,
                        priority=None, dedupe_key=None):
                published.append(dedupe_key)
                return "id-%d" % len(published)

        product = load_catalog().products()[0]
        bridge = AlertBridge(Service(), lambda: T0)
        for action in (Action.WATCH, Action.SKIP, Action.NO_STOCK):
            verdict = Verdict(
                product_id=product.id, action=action, at=T0, reasons=("x",)
            )
            self.assertIsNone(bridge.publish_verdict(verdict, product))
        self.assertEqual([], published)

    def test_the_bridge_reads_landed_the_way_contracts_defines_it(self):
        product = load_catalog().products()[0]
        verdict = Verdict(
            product_id=product.id, action=Action.BUY, at=T0, source="cardbarn",
            sku="CA-1", price=13000, landed=13499, market=15250, quantity=1,
            url="https://cardbarn.example.com/p/x", reasons=("ok",),
        )
        self.assertEqual(13499, landed_cents(verdict))
        data = alert_data(verdict)
        for key, value in data.items():
            if key.endswith("_cents"):
                self.assertIsInstance(value, int, key)
        self.assertIn(product.id, dedupe_key_for(verdict))


# --------------------------------------------------------------------------
# engine -> store
# --------------------------------------------------------------------------


class StoreSeam(unittest.TestCase):
    def test_a_real_verdict_survives_the_store(self):
        now = [T0]
        catalog = load_catalog()
        product = catalog.products()[0]
        history = PriceHistory()
        observations = [
            Observation(
                product.id, "examplemart", "EX-1", T0 - (i + 1) * 3600,
                Stock.IN_STOCK, 15000 + i * 50,
                url="https://examplemart.example.com/p/x",
            )
            for i in range(6)
        ]
        history.extend(observations)
        rules = RuleSet(budget=Budget(total=60000))
        rules.add(Rule(product_id=product.id, max_price=20000))
        verdict = DecisionEngine(
            catalog, rules, history, clock_at(now)
        ).evaluate(product.id, observations[:1])

        store = PokeStore(
            os.path.join(tempfile.mkdtemp(), "poke.sqlite3"), clock_at(now)
        )
        store.append_verdict(verdict)
        self.assertIsNone(first_difference(verdict, store.last_verdict(product.id)))
        store.save_observations(observations)
        self.assertTrue(
            round_trip_equal(
                sorted(observations, key=lambda o: (o.at, o.source)),
                sorted(store.load_observations(), key=lambda o: (o.at, o.source)),
            )
        )
        store.save_rule_set(rules)
        self.assertEqual(rules.budget, store.load_rule_set().budget)


# --------------------------------------------------------------------------
# cli.page_state <-> web/index.html
# --------------------------------------------------------------------------


def embedded_demo_state():
    """The dataset ``jarvis_poke/web/index.html`` falls back to."""
    path = os.path.join(ROOT, "jarvis_poke", "web", "index.html")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    match = re.search(
        r'<script type="application/json" id="jarvis-demo-state">(.*?)</script>',
        source,
        re.S,
    )
    assert match, "the page's embedded demo dataset is gone"
    return json.loads(html.unescape(match.group(1)))


def shape_of(node, path="$", acc=None):
    """``{json path: set of keys}`` -- the shape, ignoring every value."""
    if acc is None:
        acc = {}
    if isinstance(node, dict):
        acc.setdefault(path, set()).update(node.keys())
        for key, value in node.items():
            shape_of(value, f"{path}.{key}", acc)
    elif isinstance(node, list):
        acc.setdefault(path, set()).add("<list>")
        for value in node:
            shape_of(value, f"{path}[]", acc)
    return acc


def seeded_store_state():
    """A real store, stub-polled, through the real CLI -- then its state JSON."""
    directory = tempfile.mkdtemp()
    database = os.path.join(directory, "poke.sqlite3")
    devnull = open(os.devnull, "w")
    try:
        def run(*argv):
            code = cli.main(["--db", database, *argv], out=devnull, err=devnull)
            assert code == 0, (argv, code)

        run("init")
        run("budget", "--total", "600")
        catalog = load_catalog()
        for product in catalog.products()[:4]:
            run("watch", "--product", product.id, "--max-price", "500")
        for _ in range(40):
            run("poll", "--once")
        run("decide")
        store = PokeStore(database, cli._clock)
        return cli.page_state(store)
    finally:
        devnull.close()


class PageContract(unittest.TestCase):
    """The page and the API are two files by two authors sharing one shape."""

    @classmethod
    def setUpClass(cls):
        cls.demo = embedded_demo_state()
        cls.live = seeded_store_state()

    def test_the_live_state_and_the_demo_dataset_have_the_same_shape(self):
        live, demo = shape_of(self.live), shape_of(self.demo)
        problems = []
        for path in sorted(set(live) | set(demo)):
            in_live, in_demo = live.get(path), demo.get(path)
            if in_live is None:
                problems.append(f"{path}: only the page's demo data has it")
            elif in_demo is None:
                problems.append(f"{path}: only the live API has it")
            else:
                for key in sorted(in_demo - in_live):
                    problems.append(f"{path}.{key}: the page reads it, the API omits it")
                for key in sorted(in_live - in_demo):
                    problems.append(f"{path}.{key}: the API sends it, the demo lacks it")
        self.assertEqual([], problems)

    def test_the_live_state_covers_the_documented_contract(self):
        """``jarvis_poke/web/README.md``'s ``GET state`` block, transcribed."""
        required = {
            "$": {"generated_at", "budget", "watch", "sources", "recent"},
            "$.budget": {
                "total_cents", "spent_cents", "remaining_cents", "window_days",
            },
            "$.watch[]": {
                "product", "rule", "best", "offers", "market", "verdict", "trend",
            },
            "$.watch[].product": {"id", "name", "set_code", "kind", "msrp_cents"},
            "$.watch[].rule": {
                "max_price_cents", "quantity", "min_discount_pct", "enabled",
                "cooldown_s", "include_shipping",
            },
            "$.watch[].best": {
                "source", "sku", "price_cents", "shipping_cents", "landed_cents",
                "stock", "url", "at",
            },
            "$.watch[].offers[]": {
                "source", "sku", "price_cents", "shipping_cents", "landed_cents",
                "stock", "url", "at",
            },
            "$.watch[].market": {
                "median_cents", "p25_cents", "low_cents", "samples", "stale",
            },
            "$.watch[].verdict": {
                "action", "reasons", "discount_pct", "quantity", "at", "source",
                "sku", "price_cents", "landed_cents", "market_cents", "url",
            },
            "$.sources[]": {
                "id", "state", "next_due_at", "errors", "paused_until", "name",
                "min_interval_s", "robots_allows", "last_reason",
            },
            "$.recent[]": {"action", "reasons", "at", "product_name", "set_code"},
        }
        shape = shape_of(self.live)
        for path, keys in required.items():
            with self.subTest(path=path):
                self.assertIn(path, shape, f"{path} is absent from the live state")
                self.assertEqual(set(), keys - shape[path])

    def test_the_sparkline_gets_the_number_of_buckets_the_page_wants(self):
        """``price_trend`` defaults to 6; the page was built against 28.

        A shape check cannot see this -- both sides are a list of pairs --
        so the count is asserted on its own.
        """
        demo_counts = {len(row["trend"]) for row in self.demo["watch"]}
        live_counts = {len(row["trend"]) for row in self.live["watch"]}
        self.assertEqual({cli.TREND_BUCKETS}, demo_counts)
        self.assertEqual({cli.TREND_BUCKETS}, live_counts)

    def test_a_trend_bucket_is_a_timestamp_and_cents_or_an_honest_null(self):
        """README: "A bucket with no reading must be null, not 0"."""
        for row in self.live["watch"]:
            for bucket in row["trend"]:
                self.assertIsInstance(bucket, list)
                self.assertEqual(2, len(bucket))
                self.assertIsInstance(bucket[0], int)
                self.assertTrue(bucket[1] is None or isinstance(bucket[1], int))

    def test_every_cents_field_in_the_live_state_is_an_integer(self):
        """contracts.py, "Money is integer cents"."""
        floats = []

        def walk(node, path="$"):
            if isinstance(node, dict):
                for key, value in node.items():
                    if (
                        key.endswith("_cents")
                        and value is not None
                        and not isinstance(value, int)
                    ):
                        floats.append(f"{path}.{key}={value!r}")
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")

        walk(self.live)
        self.assertEqual([], floats)

    def test_the_state_is_json_and_nothing_but_json(self):
        json.loads(json.dumps(self.live))

    def test_every_shipped_url_is_a_placeholder(self):
        """No real retailer in anything this package serves."""
        urls = re.findall(r'https?://[^"\s]+', json.dumps(self.live))
        self.assertTrue(urls)
        for url in urls:
            self.assertRegex(url, r"^https?://[a-z0-9.-]*example\.com(/|$)", url)

    def test_the_source_state_words_are_the_five_the_page_knows(self):
        known = {"ok", "backoff", "paused", "disallowed", "error"}
        for row in self.live["sources"]:
            self.assertIn(row["state"], known)
        for row in self.demo["sources"]:
            self.assertIn(row["state"], known)

    def test_a_verdict_always_carries_its_reasons(self):
        for row in self.live["watch"]:
            verdict = row["verdict"]
            if verdict is not None:
                self.assertTrue(verdict["reasons"], row["product"]["id"])
        for entry in self.live["recent"]:
            self.assertTrue(entry["reasons"])


# --------------------------------------------------------------------------
# The boundary, checked over the whole shipped package
# --------------------------------------------------------------------------


class Boundary(unittest.TestCase):
    """contracts.py: "It does not check out" and "makes no network calls"."""

    def _sources(self):
        directory = os.path.join(ROOT, "jarvis_poke")
        for name in sorted(os.listdir(directory)):
            if name.endswith(".py"):
                with open(os.path.join(directory, name), encoding="utf-8") as handle:
                    yield name, handle.read()

    @staticmethod
    def _code_only(source):
        """Source with docstrings and comments removed.

        The modules state the boundary in prose -- cli.py's own docstring
        says "there is no checkout command and there will not be one" --
        so a scan that reads the prose finds the package's promises and
        calls them violations.  Only code is evidence.
        """
        without_docstrings = re.sub(r'\"\"\".*?\"\"\"', "", source, flags=re.S)
        return re.sub(r"#.*", "", without_docstrings)

    def test_no_module_opens_a_socket(self):
        forbidden = ("urllib", "http.client", "socket", "requests", "httpx")
        for name, source in self._sources():
            for module in forbidden:
                pattern = rf"^\s*(?:import|from)\s+{re.escape(module)}\b"
                self.assertIsNone(
                    re.search(pattern, source, re.M),
                    f"{name} imports {module}; fetching is injected",
                )

    def test_no_module_reads_the_wall_clock_or_draws_its_own_randomness(self):
        """contracts.py: the clock is injected, and randomness is seeded.

        Parsed, not grepped.  Three modules carry the sentence "this
        package never calls time.time()" inside an error message, so a
        textual scan finds the package explaining the rule and reports it
        as a breach of it.  Only a real call counts, and ``ast`` is the
        only thing that can tell the two apart.

        ``cli.py`` is the one exception: it is the entry point, so it is
        where the wall clock is *read* in order to be injected into
        everything else (``cli._clock``, "the one place this package is
        allowed to read a wall clock").
        """
        banned_calls = {
            ("time", "time"),
            ("time", "monotonic"),
            ("random", "random"),
            ("random", "randint"),
            ("random", "choice"),
            ("random", "uniform"),
            ("random", "shuffle"),
            ("random", "seed"),
        }
        for name, source in self._sources():
            tree = ast.parse(source, filename=name)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if not isinstance(func, ast.Attribute):
                    continue
                if not isinstance(func.value, ast.Name):
                    continue
                call = (func.value.id, func.attr)
                if call == ("time", "time") and name == "cli.py":
                    continue
                self.assertNotIn(
                    call,
                    banned_calls,
                    f"{name}:{node.lineno} calls {call[0]}.{call[1]}(); "
                    f"the clock is injected and randomness comes from "
                    f"lucifer_gen.seed",
                )

    def test_the_package_carries_no_checkout_machinery(self):
        banned = (
            "add to cart", "add_to_cart", "checkout", "place_order", "captcha",
            "proxy_rotat", "autobuy", "auto_buy",
        )
        for name, source in self._sources():
            lowered = self._code_only(source).lower()
            for token in banned:
                self.assertNotIn(token, lowered, f"{name} implements {token!r}")

    def test_every_shipped_source_url_is_a_placeholder(self):
        path = os.path.join(ROOT, "jarvis_poke", "data", "sources.json")
        with open(path, encoding="utf-8") as handle:
            blob = handle.read()
        for url in re.findall(r'https?://[^"\s]+', blob):
            self.assertRegex(url, r"^https?://[a-z0-9.-]*example\.com(/|$)", url)

    def test_the_polling_floor_holds_for_every_shipped_source(self):
        """contracts.FetchPolicy: 30s is a floor, not a preference."""
        from jarvis_poke.sources import load_policies

        policies = load_policies()
        self.assertTrue(policies)
        for source, policy in policies.items():
            self.assertGreaterEqual(policy.min_interval_s, 30.0, source)


# --------------------------------------------------------------------------
# cli <-> validate
# --------------------------------------------------------------------------


class GateSeam(unittest.TestCase):
    def test_the_cli_calls_run_gate_by_the_names_it_declares(self):
        parameters = inspect.signature(validate.run_gate).parameters
        for name in ("n_products", "n_days", "seed", "inject_defect"):
            self.assertIn(name, parameters)

    def test_a_healthy_gate_passes_and_every_defect_is_caught(self):
        report = validate.run_gate(n_products=6, n_days=4, seed=1)
        self.assertTrue(report.ok, report.summary())
        for defect in validate.DEFECTS:
            with self.subTest(defect=defect):
                broken = validate.run_gate(
                    n_products=6, n_days=4, seed=1, inject_defect=defect
                )
                self.assertFalse(broken.ok, f"{defect} went unreported")

    def test_the_gate_is_reproducible_from_its_seed(self):
        first = validate.run_gate(n_products=6, n_days=4, seed=0xA11CE).to_dict()
        second = validate.run_gate(n_products=6, n_days=4, seed=0xA11CE).to_dict()
        self.assertEqual(first, second)

    def test_the_gate_never_fetches_a_source_robots_txt_disallows(self):
        counts = validate.run_gate(n_products=6, n_days=4, seed=1).to_dict()["counts"]
        self.assertGreater(counts["disallowed_listings"], 0, "nothing was disallowed")
        self.assertEqual(0, counts["disallowed_polls"])


class BudgetLifecycle(unittest.TestCase):
    """The ledger through the shipped CLI, over several runs.

    Before the fix nothing in the shipped product ever moved
    ``Budget.spent``: ``decide`` never saved the rule set,
    ``save_rule_set`` dropped reservations on purpose, and no verb
    recorded a purchase.  Five BUY alerts at $120 against a $400 budget
    left the ledger reading "$0.00 spent, $400.00 remaining", so the
    owner's monthly ceiling was decorative and the per-verdict cap --
    which resets on every invocation -- was the only live control.
    """

    def _cli(self, database):
        devnull = open(os.devnull, "w")
        self.addCleanup(devnull.close)

        def run(*argv):
            code = cli.main(["--db", database, *argv], out=devnull, err=devnull)
            assert code == 0, (argv, code)

        return run

    def _seeded(self):
        directory = tempfile.mkdtemp()
        database = os.path.join(directory, "poke.sqlite3")
        run = self._cli(database)
        run("init")
        run("budget", "--total", "600")
        catalog = load_catalog()
        for product in catalog.products()[:4]:
            run("watch", "--product", product.id, "--max-price", "500")
        for _ in range(40):
            run("poll", "--once")
        return database, run

    def test_a_reservation_outlives_the_process_that_made_it(self):
        database, run = self._seeded()
        run("decide")
        with PokeStore(database, cli._clock) as store:
            held = store.load_reservations()
            first = store.load_rule_set()
        self.assertTrue(held, "a BUY alert is on a phone and nothing is holding its money")
        self.assertEqual(sum(held.values()), first.reserved)
        self.assertLess(first.remaining(), first.budget.total)

        # a second run, a second process: the hold is still there, and
        # this product's own alert is not booked twice
        run("decide")
        with PokeStore(database, cli._clock) as store:
            again = store.load_reservations()
            second = store.load_rule_set()
        self.assertEqual(set(held), set(again) & set(held))
        self.assertEqual(second.reserved, sum(again.values()))
        self.assertLessEqual(second.reserved, second.budget.total)

    def test_bought_is_the_one_verb_that_moves_spent(self):
        database, run = self._seeded()
        run("decide")
        with PokeStore(database, cli._clock) as store:
            held = store.load_reservations()
            before = store.load_rule_set()
        self.assertEqual(before.budget.spent, 0)
        product_id, amount = sorted(held.items())[0]

        run("bought", "--product", product_id)
        with PokeStore(database, cli._clock) as store:
            after = store.load_rule_set()
        self.assertEqual(after.budget.spent, amount)
        self.assertNotIn(product_id, after.reservations())
        self.assertEqual(after.remaining(), before.remaining())

    def test_release_hands_money_back(self):
        database, run = self._seeded()
        run("decide")
        with PokeStore(database, cli._clock) as store:
            before = store.load_rule_set()
        self.assertGreater(before.reserved, 0)

        run("release", "--all")
        with PokeStore(database, cli._clock) as store:
            after = store.load_rule_set()
        self.assertEqual(after.reserved, 0)
        self.assertEqual(after.budget.spent, 0)
        self.assertEqual(after.remaining(), after.budget.total)

    def test_the_page_state_says_what_is_held(self):
        database, run = self._seeded()
        run("decide")
        with PokeStore(database, cli._clock) as store:
            state = cli.page_state(store)
        budget = state["budget"]
        self.assertIn("reserved_cents", budget)
        self.assertEqual(
            budget["remaining_cents"],
            max(0, budget["total_cents"] - budget["spent_cents"] - budget["reserved_cents"]),
        )
        self.assertEqual(
            budget["reserved_cents"],
            sum(row["held_cents"] for row in state["watch"]),
        )


if __name__ == "__main__":
    unittest.main()
