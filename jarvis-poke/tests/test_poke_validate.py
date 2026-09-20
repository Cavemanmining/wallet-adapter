"""Tests for the Pokemon gate and the command line, jarvis_poke.

Design: jarvis_poke/contracts.py.  :mod:`jarvis_poke.validate` drives a
scripted month through the real catalog, scheduler, price history, engine,
ledger, alert bridge and sqlite store, and then argues with the result;
:mod:`jarvis_poke.cli` is the same pieces wired up for a shell.  These
tests hold both to what contracts.py promises:

* a healthy month reports **zero** problems, and really did contain every
  hazard the gate claims to test -- restocks, a crash that must be taken,
  a mis-parse that must not be, an outage that pauses a host, a
  robots-disallowed source that is never touched, and a budget cap that
  actually refused something;
* **every** injected defect is caught, with the kind that names it: the
  gate is shown to fail before it is trusted;
* the same seed gives the same report twice, and a different seed does
  not;
* forty products over thirty days finishes well inside thirty seconds;
* the CLI round trip ``init -> watch -> budget -> poll --once -> decide
  -> serve-state`` works on a temporary database and prints JSON the
  shipped page can read -- checked against the page's *own* sample state,
  not against a copy of it here;
* an error is one line and exit 1, never a traceback, and never carries a
  URL's query string;
* neither module opens a socket, reads a wall clock outside the CLI's
  composition root, or draws a random number outside ``lucifer_gen``.

Nothing here sleeps or touches a network.  The one test that reads the
wall clock is the timing test, which measures the gate from outside.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Runnable as `pytest tests/test_poke_validate.py` or
# `python3 tests/test_poke_validate.py` from anywhere.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from jarvis_poke import cli, validate
from jarvis_poke.contracts import Action, Stock
from dataclasses import replace
from fractions import Fraction

from jarvis_poke import prices
from jarvis_poke.contracts import MAX_BUDGET_FRACTION_PER_VERDICT, Action
from jarvis_poke.validate import DEFECTS, GateError, GateReport, Problem, run_gate

SEED = 0x5EED_A1E7_0000_0002
SMALL = dict(n_products=6, n_days=4, seed=SEED)

#: Which problem kind each defect must produce.  A defect that trips some
#: *other* check is not good enough: the gate has to name what went wrong.
EXPECTED_KIND = {
    "poll_too_fast": "poll_too_fast",
    "ignore_robots": "robots_ignored",
    "ignore_ceiling": "buy_over_ceiling",
    "trust_outlier": "buy_on_misparse",
    "ignore_budget": "verdict_over_cap",
    "ignore_cooldown": "cooldown_broken",
    "alert_on_watch": "alert_not_buy",
}


@pytest.fixture(scope="module")
def healthy() -> GateReport:
    return run_gate(**SMALL)


# ---------------------------------------------------------------------------
# A healthy month
# ---------------------------------------------------------------------------


def test_healthy_run_has_no_problems(healthy: GateReport) -> None:
    assert healthy.problems == [], healthy.summary()
    assert healthy.ok is True
    assert healthy.first_failure is None
    assert healthy.kinds() == []


def test_healthy_run_contains_every_hazard(healthy: GateReport) -> None:
    """A pass is only worth something if the month was not empty."""
    counts = healthy.counts
    for key in (
        "polls",              # somebody was fetched
        "observations",       # and parsed
        "not_modified",       # the 304-heavy source did its thing
        "errors",             # the outage happened
        "pauses",             # and was severe enough to pause a host
        "sellouts",
        "restocks",
        "misparse_observations",
        "misparse_refusals",  # and it was the best offer, and lost
        "crash_buys",         # the genuine bargain was taken
        "buys",
        "cap_refusals",       # the budget cap refused something
        "commits",            # and the owner bought something
    ):
        assert counts[key] > 0, f"{key} was zero:\n{healthy.summary()}"
    assert counts["disallowed_listings"] > 0
    assert counts["disallowed_polls"] == 0


def test_market_reference_is_real_money_not_floats(healthy: GateReport) -> None:
    """contracts.py: money is integer cents, everywhere, always."""
    for key, value in healthy.counts.items():
        assert isinstance(value, int), f"{key} is {type(value).__name__}"
    assert healthy.counts["committed_cents"] >= 0


def test_report_is_json_and_carries_a_repro(healthy: GateReport) -> None:
    payload = json.dumps(healthy.to_dict(), sort_keys=True)
    parsed = json.loads(payload)
    assert parsed["ok"] is True
    assert parsed["n_products"] == SMALL["n_products"]
    assert parsed["inject_defect"] is None
    assert "run_gate(" in healthy.repro and "inject_defect=None" in healthy.repro


def test_no_verdict_is_ever_reasonless() -> None:
    """The one promise that holds for every action, not just BUY."""
    gate = validate._Gate(**SMALL, inject_defect=None)
    gate.run()
    assert gate.decisions, "the month decided nothing"
    for decision in gate.decisions:
        assert decision.verdict.reasons, decision.verdict


def test_only_buys_were_alerted() -> None:
    gate = validate._Gate(**SMALL, inject_defect=None)
    gate.run()
    buys = [d.verdict for d in gate.decisions if d.verdict.action is Action.BUY]
    assert len(gate.service.sent) == len(buys) > 0
    assert gate.bridge.suppressed == 0
    for alert in gate.service.sent:
        assert alert["kind"] == "poke_buy"
        assert alert["dedupe_key"].startswith("poke:buy|")
        # The deep link a person taps, and nothing that could be a secret.
        assert set(alert["data"]) == {
            "product_id", "source", "sku", "url", "price_cents", "quantity",
            "market_cents",
        }


def test_the_mis_parse_is_observed_and_refused() -> None:
    """A 95% discount is a per-pack price on a box page, not a bargain."""
    gate = validate._Gate(**SMALL, inject_defect=None)
    gate.run()
    assert gate.misparse_keys, "the mis-parse never made it into the history"
    misparsed = {(p, s, k) for p, s, k, _ in gate.misparse_keys}
    for decision in gate.decisions:
        if decision.verdict.action is Action.BUY and decision.best is not None:
            best = decision.best
            assert (best.product_id, best.source, best.sku) not in misparsed or (
                best.product_id, best.source, best.sku, best.at
            ) not in gate.misparse_keys


def test_the_robots_disallowed_source_is_never_fetched() -> None:
    gate = validate._Gate(**SMALL, inject_defect=None)
    gate.run()
    fetched = {call.source for call in gate.host.calls}
    assert validate.DISALLOWED_SOURCE not in fetched
    assert gate.catalog.skus_from(validate.DISALLOWED_SOURCE), "nothing to disallow"


def test_no_host_is_polled_faster_than_its_interval() -> None:
    gate = validate._Gate(**SMALL, inject_defect=None)
    gate.run()
    for source, calls in gate.host.by_source().items():
        floor = gate.policies[source].min_interval_s
        times = sorted(call.at for call in calls)
        gaps = [b - a for a, b in zip(times, times[1:])]
        assert not gaps or min(gaps) >= floor, f"{source}: {min(gaps)}s < {floor}s"


# ---------------------------------------------------------------------------
# Shown to fail before it is trusted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("defect", DEFECTS)
def test_every_defect_is_caught(defect: str) -> None:
    report = run_gate(**SMALL, inject_defect=defect)
    assert not report.ok, f"{defect} went unreported:\n{report.summary()}"
    assert EXPECTED_KIND[defect] in report.kinds(), report.summary()
    assert report.first_failure is not None
    assert isinstance(report.first_failure, Problem)
    assert defect in report.repro


def test_defect_list_is_the_one_the_task_named() -> None:
    assert set(DEFECTS) == set(EXPECTED_KIND)
    assert set(DEFECTS) == {
        "poll_too_fast", "ignore_robots", "ignore_ceiling", "trust_outlier",
        "ignore_budget", "ignore_cooldown", "alert_on_watch",
    }


def test_show_defects_entry_point(capsys: Any) -> None:
    assert validate.main(["--products", "6", "--days", "4", "--show-defects"]) == 0
    out = capsys.readouterr().out
    assert out.count("caught ") == len(DEFECTS)
    assert "MISSED" not in out


def test_json_entry_point(capsys: Any) -> None:
    assert validate.main(["--products", "5", "--days", "3", "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["ok"] is True and parsed["n_products"] == 5


def test_a_run_too_small_to_test_anything_is_refused() -> None:
    with pytest.raises(GateError):
        run_gate(n_products=3, n_days=5, seed=SEED)
    with pytest.raises(GateError):
        run_gate(n_products=8, n_days=2, seed=SEED)
    with pytest.raises(GateError):
        run_gate(n_products=8, n_days=5, seed=SEED, inject_defect="make_money")


# ---------------------------------------------------------------------------
# Determinism and cost
# ---------------------------------------------------------------------------


def test_the_same_seed_gives_the_same_report() -> None:
    first = run_gate(n_products=9, n_days=5, seed=0xA11CE)
    second = run_gate(n_products=9, n_days=5, seed=0xA11CE)
    assert first.to_dict() == second.to_dict()
    assert first.ok and second.ok


def test_a_different_seed_gives_a_different_month() -> None:
    first = run_gate(n_products=9, n_days=5, seed=0xA11CE)
    other = run_gate(n_products=9, n_days=5, seed=0xB0B)
    assert first.counts != other.counts
    assert first.ok and other.ok


def test_forty_products_over_thirty_days_is_quick() -> None:
    started = time.monotonic()
    report = run_gate(n_products=40, n_days=30, seed=SEED)
    elapsed = time.monotonic() - started
    assert report.ok, report.summary()
    assert report.counts["products"] == 40 and report.counts["days"] == 30
    assert report.counts["observations"] > 1000
    assert elapsed < 30.0, f"took {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> str:
    return str(tmp_path / "poke.sqlite3")


def run_cli(*argv: str) -> Dict[str, Any]:
    """Run one command, capturing both streams; never raises."""
    import io

    out, err = io.StringIO(), io.StringIO()
    code = cli.main(list(argv), out=out, err=err)
    return {"code": code, "out": out.getvalue(), "err": err.getvalue()}


#: The product examplemart lists first, which is therefore the one a
#: single polling round actually reaches -- one listing per host per
#: round is the whole point of the scheduler.
FIRST_PRODUCT = "sv035-151-booster-bundle"


def test_cli_round_trip(db: str) -> None:
    """init -> watch -> budget -> poll --once -> decide -> serve-state."""
    result = run_cli("--db", db, "init")
    assert result["code"] == 0, result["err"]
    assert "products" in result["out"] and "robots DISALLOWS" in result["out"]

    result = run_cli(
        "--db", db, "watch", "--product", FIRST_PRODUCT,
        "--max-price", "300.00", "--qty", "2", "--min-discount", "5",
    )
    assert result["code"] == 0, result["err"]
    assert "$300.00" in result["out"]
    assert "will not buy" in result["out"]

    result = run_cli("--db", db, "budget", "--total", "500.00")
    assert result["code"] == 0, result["err"]
    assert "$500.00" in result["out"] and "$250.00" in result["out"]

    result = run_cli("--db", db, "poll", "--once")
    assert result["code"] == 0, result["err"]
    assert "STUB prices" in result["out"], "stub data must announce itself"
    assert "robots.txt disallows 'bigboxco'" in result["out"]

    result = run_cli("--db", db, "list")
    assert result["code"] == 0, result["err"]
    assert FIRST_PRODUCT[:12] in result["out"] or "151 Booster" in result["out"]
    assert "not real prices" in result["out"]

    result = run_cli("--db", db, "decide")
    assert result["code"] == 0, result["err"]
    assert "decided 1" in result["out"]

    result = run_cli("--db", db, "sources")
    assert result["code"] == 0, result["err"]
    assert "disallowed" in result["out"] and "never" in result["out"]

    result = run_cli("--db", db, "serve-state")
    assert result["code"] == 0, result["err"]
    state = json.loads(result["out"])
    assert state["demo"] is True, "stub data must set the page's demo flag"
    assert state["budget"]["total_cents"] == 50000
    assert len(state["watch"]) == 1
    entry = state["watch"][0]
    assert entry["product"]["id"] == FIRST_PRODUCT
    assert entry["rule"]["max_price_cents"] == 30000
    assert entry["rule"]["quantity"] == 2
    assert entry["verdict"]["action"] in {a.value for a in Action}
    assert entry["verdict"]["reasons"]
    assert entry["offers"], "a polled listing should show as an offer"
    for offer in entry["offers"]:
        assert offer["stock"] in {s.value for s in Stock}
        assert isinstance(offer["price_cents"], (int, type(None)))
    assert state["recent"] and state["recent"][0]["product_id"] == FIRST_PRODUCT
    assert {s["id"] for s in state["sources"]} == {
        "examplemart", "cardbarn", "hobbyhub", "bigboxco",
    }
    disallowed = next(s for s in state["sources"] if s["id"] == "bigboxco")
    assert disallowed["state"] == "disallowed" and disallowed["robots_allows"] is False


def test_cli_state_matches_the_page_contract(db: str) -> None:
    """Every key the shipped page's own sample state carries.

    The contract is the page, not a copy of it in this file: the sample
    state embedded in ``jarvis_poke/web/index.html`` is what the page is
    written against, so anything it carries must be in what
    ``serve-state`` prints.
    """
    page = ROOT / "jarvis_poke" / "web" / "index.html"
    if not page.exists():  # pragma: no cover - the page is a sibling's file
        pytest.skip("the page has not been written yet")
    match = re.search(
        r'<script type="application/json" id="jarvis-demo-state">(.*?)</script>',
        page.read_text(encoding="utf-8"),
        re.S,
    )
    if match is None:  # pragma: no cover - the page may carry no sample
        pytest.skip("the page ships no sample state to compare against")
    sample = json.loads(match.group(1))

    run_cli("--db", db, "init")
    run_cli("--db", db, "watch", "--product", FIRST_PRODUCT, "--max-price", "300.00")
    run_cli("--db", db, "budget", "--total", "500.00")
    run_cli("--db", db, "poll", "--once")
    run_cli("--db", db, "decide")
    state = json.loads(run_cli("--db", db, "serve-state")["out"])

    def missing(expected: Any, actual: Any, path: str) -> List[str]:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return [f"{path}: expected an object, got {type(actual).__name__}"]
            gaps: List[str] = []
            for key, value in expected.items():
                if key not in actual:
                    gaps.append(f"{path}.{key} is missing")
                else:
                    gaps.extend(missing(value, actual[key], f"{path}.{key}"))
            return gaps
        if isinstance(expected, list) and expected and isinstance(actual, list):
            return missing(expected[0], actual[0], f"{path}[0]") if actual else []
        return []

    gaps = missing(sample, state, "state")
    assert not gaps, "the page reads keys serve-state does not print: " + "; ".join(gaps)


def test_cli_errors_are_one_line_and_exit_one(db: str) -> None:
    for argv in (
        ("--db", db, "list"),                                    # no catalog yet
        ("--db", db, "decide"),
    ):
        result = run_cli(*argv)
        assert result["code"] == cli.EXIT_FAIL
        assert result["err"].count("\n") == 1
        assert "Traceback" not in result["err"]
        assert result["err"].startswith("error: ")

    run_cli("--db", db, "init")
    for argv, fragment in (
        (("--db", db, "watch", "--product", "nope", "--max-price", "1.00"), "nope"),
        (("--db", db, "watch", "--product", FIRST_PRODUCT, "--max-price", "free"), "--max-price"),
        (("--db", db, "unwatch", "--product", "nope"), "not watching"),
        (("--db", db, "pause", "--source", "nowhere"), "nowhere"),
        (("--db", db, "poll"), "--once"),
    ):
        result = run_cli(*argv)
        assert result["code"] == cli.EXIT_FAIL, argv
        assert result["err"].count("\n") == 1 and fragment in result["err"]


def test_a_usage_error_exits_two(db: str) -> None:
    assert run_cli("--db", db, "not-a-command")["code"] == cli.EXIT_USAGE
    assert run_cli("--db", db)["code"] == cli.EXIT_USAGE


def test_no_error_ever_prints_a_query_string(db: str) -> None:
    """A listing URL can carry an affiliate tag or a session token."""
    secret = "https://shop.example.com/p/thing?token=SECRET&aff=me"
    run_cli("--db", db, "init")
    for result in (
        run_cli("--db", db, "watch", "--product", secret, "--max-price", "1.00"),
        run_cli("--db", db, "watch", "--product", "x", "--max-price", secret),
        run_cli("--db", db, "unwatch", "--product", secret),
        run_cli("--db", db, "pause", "--source", secret),
        run_cli("--db", db, "gate", "--products", "5", "--days", "3", "--seed", secret),
    ):
        assert "token=SECRET" not in result["err"] + result["out"]
        assert "aff=me" not in result["err"] + result["out"]
    assert cli.scrub(secret) == "https://shop.example.com/p/thing?..."
    assert cli.scrub("no url here") == "no url here"


def test_cli_gate_command(db: str) -> None:
    passed = run_cli("--db", db, "gate", "--products", "5", "--days", "3", "--seed", "7")
    assert passed["code"] == 0, passed["err"]
    assert "gate passed" in passed["out"]

    caught = run_cli(
        "--db", db, "gate", "--products", "5", "--days", "3", "--seed", "7",
        "--defect", "ignore_robots",
    )
    assert caught["code"] == 0, caught["err"]
    assert "was caught" in caught["out"]
    assert "robots_ignored" in caught["out"]


def test_the_stub_says_it_is_a_stub(db: str) -> None:
    """Invented prices must never pass for fetched ones."""
    run_cli("--db", db, "init")
    run_cli("--db", db, "watch", "--product", FIRST_PRODUCT, "--max-price", "300.00")
    run_cli("--db", db, "poll", "--once")
    from jarvis_poke.store import PokeStore

    with PokeStore(db, lambda: 0.0) as store:
        observations = store.load_observations()
        assert observations
        for observation in observations:
            assert observation.note == cli.STUB_NOTE
        assert store.meta(cli.STUB_DATA_KEY) == "1"


def test_an_installed_fetcher_replaces_the_stub(db: str) -> None:
    """The app's fetcher and parser are what a real run uses."""
    from jarvis_poke.contracts import FetchResult, Observation

    seen: List[str] = []

    def fetcher(url: str, headers: Dict[str, str], policy: Any) -> FetchResult:
        seen.append(url)
        assert "User-Agent" in headers
        return FetchResult(ok=True, status=200, body="49.99", etag='W/"1"')

    def parser(sku: Any, body: str, at: float) -> Observation:
        from jarvis_poke.contracts import to_cents

        return Observation(
            product_id=sku.product_id, source=sku.source, sku=sku.sku, at=at,
            stock=Stock.IN_STOCK, price=to_cents(body), url=sku.url,
        )

    run_cli("--db", db, "init")
    cli.install_fetcher(fetcher)
    cli.install_parser(parser)
    try:
        result = run_cli("--db", db, "poll", "--once")
    finally:
        cli.install_fetcher(None)
        cli.install_parser(None)
    assert result["code"] == 0, result["err"]
    assert seen and all("bigboxco" not in url for url in seen)
    assert "STUB prices" not in result["out"]
    assert "$49.99" in result["out"]
    state = json.loads(run_cli("--db", db, "serve-state")["out"])
    assert state["demo"] is False


# ---------------------------------------------------------------------------
# The boundary contracts.py draws
# ---------------------------------------------------------------------------


OWNED = ("validate.py", "cli.py")


def _module_source(name: str) -> str:
    return (ROOT / "jarvis_poke" / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("name", OWNED)
def test_no_module_opens_a_socket(name: str) -> None:
    """contracts.py: "The package makes no network calls"."""
    tree = ast.parse(_module_source(name))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert not imported & {
        "urllib", "http", "socket", "ssl", "requests", "httpx", "ftplib",
    }, f"{name} imports something that can open a connection"


@pytest.mark.parametrize("name", OWNED)
def test_randomness_comes_only_from_the_seed(name: str) -> None:
    source = _module_source(name)
    assert "import random" not in source
    assert "random." not in source.replace("_rng.random", "")
    assert "lucifer_gen.seed" in source


def test_only_the_cli_reads_a_wall_clock() -> None:
    """The CLI is the composition root; the gate is not."""
    assert "time.time" not in _module_source("validate.py").replace(
        "``time.time``", ""
    )
    cli_source = _module_source("cli.py")
    assert cli_source.count("time.time()") == 1, "one clock, in one place"


@pytest.mark.parametrize("name", OWNED)
def test_nothing_here_checks_out(name: str) -> None:
    """contracts.py, "What this is not": no cart, no payment, no evasion."""
    source = _module_source(name).lower()
    for word in (
        "add_to_cart", "addtocart", "checkout(", "payment", "credit_card",
        "captcha", "proxy_rotat", "user_agent_rotat", "bypass",
    ):
        assert word not in source, f"{name} mentions {word!r}"


def test_the_gate_ships_no_real_retailer() -> None:
    """Placeholders on example.com, and nothing else."""
    gate = validate._Gate(**SMALL, inject_defect=None)
    for sku in gate.catalog.skus():
        assert sku.url.startswith("https://") and ".example.com/" in sku.url
        assert sku.source in {"examplemart", "cardbarn", "hobbyhub", "bigboxco"}


# ---------------------------------------------------------------------------
# The blind spots the gate used to have
# ---------------------------------------------------------------------------


def test_the_bridge_is_offered_every_non_buy_verdict(healthy: GateReport) -> None:
    """``AlertBridge``'s "BUY only" guard has to be reached with a
    non-BUY or it is untested, and a bridge that buzzed for WATCH passed
    clean at every size and seed."""
    assert healthy.counts["bridge_non_buys"] > 0
    assert healthy.counts["bridge_non_buys"] >= healthy.counts["watches"]
    # and nothing was published for any of them
    assert healthy.counts["alerts"] == healthy.counts["buys"]


def test_a_bridge_that_published_on_watch_is_reported() -> None:
    """The guard is the thing under test, so break the guard."""
    from jarvis_poke import alerts_bridge, validate as validate_module

    real = alerts_bridge.AlertBridge.publish_verdict

    def leaky(self, verdict, product):  # noqa: ANN001 - a stand-in, not an API
        original = verdict
        if verdict.action is not Action.BUY:
            original = replace(verdict, action=Action.BUY, quantity=1)
        return real(self, original, product)

    alerts_bridge.AlertBridge.publish_verdict = leaky
    try:
        report = run_gate(**SMALL)
    finally:
        alerts_bridge.AlertBridge.publish_verdict = real
    assert not report.ok
    assert "alert_on_watch" in report.kinds()


def test_the_per_customer_limit_actually_cuts_a_buy(healthy: GateReport) -> None:
    """``buy_bad_quantity`` was dead code: across 48 runs and 24 seeds
    the limit bound zero times, so the clamp was never executed."""
    assert healthy.counts["limit_clamps"] > 0


def test_the_gate_carries_its_own_copy_of_the_cap() -> None:
    """``verdict_over_cap`` re-derived the cap with the same function and
    the same constant the engine used, so both sides moved together."""
    from jarvis_poke.validate import GATE_BUDGET_FRACTION, GATE_OUTLIER_MIN_PCT, _gate_cap

    assert Fraction(MAX_BUDGET_FRACTION_PER_VERDICT) == GATE_BUDGET_FRACTION
    assert prices.OUTLIER_MIN_PCT_OF_MEDIAN == GATE_OUTLIER_MIN_PCT
    for remaining in (0, 1, 3, 9_999, 10_000, 10_001):
        assert _gate_cap(remaining) == max(0, remaining) // 2


def test_a_doubled_cap_constant_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    from jarvis_poke import contracts, rules as rules_module, validate as validate_module

    monkeypatch.setattr(contracts, "MAX_BUDGET_FRACTION_PER_VERDICT", 1.0)
    monkeypatch.setattr(rules_module, "MAX_BUDGET_FRACTION_PER_VERDICT", 1.0)
    monkeypatch.setattr(validate_module, "MAX_BUDGET_FRACTION_PER_VERDICT", 1.0)
    monkeypatch.setattr(
        rules_module.budget_cap, "__defaults__", (1.0,)
    )
    report = run_gate(**SMALL)
    assert not report.ok
    assert "verdict_over_cap" in report.kinds()


def test_the_boundary_probes_pin_every_money_comparison(healthy: GateReport) -> None:
    """The scripted month clears its ceiling by two cents at one seed and
    nine at another, and puts the mis-parse four times below the outlier
    floor: whether a one-cent slip was caught was a coin flip.  The
    probes are exact."""
    from jarvis_poke.validate import _Gate

    for name in ("_check_boundaries", "_check_contract"):
        assert callable(getattr(_Gate, name))
    assert healthy.ok


def test_a_loosened_outlier_floor_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """The scripted mis-parse sits at 5% of the median against a 25%
    floor -- four times below the boundary it is meant to be testing --
    so the floor could be loosened four-fold and the gate stayed green.
    The probe sits one cent either side of it.
    """
    monkeypatch.setattr(prices, "OUTLIER_MIN_PCT_OF_MEDIAN", 6)
    report = run_gate(**SMALL)
    assert not report.ok
    assert "buy_on_misparse" in report.kinds()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
