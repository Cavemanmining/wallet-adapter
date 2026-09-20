"""jarvis_poke -- the Pokemon card buying assistant for the Jarvis app.

What this package is, and the line it does not cross
----------------------------------------------------
It watches sealed product, prices it against what the thing has actually
been selling for, and tells the owner *when to act and why*.  It does not
buy.  Per :mod:`jarvis_poke.contracts` ("What this is not") the terminal
output of :class:`~jarvis_poke.engine.DecisionEngine` is a
:class:`~jarvis_poke.contracts.Verdict` plus a deep link a person taps;
there is no cart automation, no payment handling, no CAPTCHA solving, no
proxy rotation and no anti-bot evasion anywhere in it.  The package opens
no sockets either: fetching is an injected
:class:`~jarvis_poke.contracts.Fetcher` and parsing an injected
:class:`~jarvis_poke.contracts.Parser`, so no retailer's selectors ship
here and the shipped data names placeholders on ``example.com``.

The layers, bottom up
---------------------
``contracts``      the types, and the money-is-integer-cents rule.
``catalog``        products and per-retailer SKUs, loaded from ``data/``.
``sources``        :class:`~jarvis_poke.sources.PollScheduler` -- the
                   politeness layer: per-host minimum interval (floor 30s),
                   robots.txt permission, ETag/Last-Modified conditional
                   requests, widening backoff, pause after repeated errors.
``prices``         price history, the market reference, the outlier gate.
``rules``          the owner's standing instructions and the budget --
                   including the reservations a BUY holds, keyed by
                   product and kept under one lock.
``engine``         observations in, :class:`Verdict` out, with reasons.
``store``          sqlite persistence for all of the above.
``alerts_bridge``  a BUY handed to ``jarvis_alerts`` so it reaches a phone.
``validate``       the gate: a scripted month held to ``contracts.py``.
``cli``            the thirteen commands (``bought`` and ``release`` are
                   the two that move the ledger), and the page's state JSON.
``web/``           ``index.html``, the page, and its API contract.

Importing
---------
The seven core modules are imported with the package, and their public
names are re-exported here, so ``from jarvis_poke import DecisionEngine``
works.  ``alerts_bridge``, ``validate`` and ``cli`` are resolved lazily on
first attribute access (PEP 562): ``alerts_bridge`` needs ``jarvis_alerts``
installed and the other two pull in the whole package, and none of that
should be the price of ``import jarvis_poke``.  ``from jarvis_poke import
validate`` works either way.

**No re-exported name is also a submodule name.**  Re-exporting a function
called, say, ``catalog`` beside the ``catalog`` module makes
``from jarvis_poke import catalog`` hand back the function on a warm import
and the module on a cold one -- the same bug ``jarvis_gen`` shipped.  The
check is mechanical and lives in ``tests/test_poke_integration.py``, which
fails if any future export collides with a module name.

Three names are deliberately *not* flattened, because two modules define
each and the package-level spelling would be a coin toss:
``main`` (``cli.main`` returns an exit code, ``validate.main`` runs the
gate), ``DEFAULT_SEED`` (``sources`` and ``validate``) and ``FRESH_S``
(``cli`` and ``validate``).  Reach those through their module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from jarvis_poke import (
    catalog,
    contracts,
    engine,
    prices,
    rules,
    sources,
    store,
)
from jarvis_poke.catalog import (
    CATALOG_PATH,
    DATA_DIR,
    SOURCES_PATH,
    Catalog,
    CatalogError,
    load_catalog,
)
from jarvis_poke.contracts import (
    CURRENCY,
    MARKET_STALE_AFTER_S,
    MARKET_WINDOW_S,
    MAX_BUDGET_FRACTION_PER_VERDICT,
    Action,
    Budget,
    Cents,
    FetchPolicy,
    FetchResult,
    Fetcher,
    MarketRef,
    Observation,
    Parser,
    Product,
    ProductKind,
    Rule,
    SourceSku,
    Stock,
    Verdict,
    WatchState,
    fmt_cents,
    to_cents,
)
from jarvis_poke.engine import (
    DEFAULT_EXPLAIN_CHARS,
    DISCOUNT_PRECISION,
    OUTLIER_LOW_FRACTION,
    DecisionEngine,
    EngineError,
    MarketHistory,
    OutlierCheck,
    discount_against,
    discount_threshold,
    explain,
    looks_mis_parsed,
)
from jarvis_poke.prices import (
    HISTORY_VERSION,
    MIN_SAMPLES_FOR_REFERENCE,
    OUTLIER_MIN_PCT_OF_MEDIAN,
    HistoryStore,
    MemoryHistoryStore,
    PriceHistory,
    PricesError,
    discount_pct,
    is_outlier,
    market_reference,
    price_trend,
)
from jarvis_poke.rules import (
    Affordability,
    BudgetError,
    RuleConflict,
    RuleError,
    RuleSet,
    affordable,
    budget_cap,
)
from jarvis_poke.sources import (
    BACKOFF_MAX_DOUBLINGS,
    DEFAULT_JITTER_FRACTION,
    MIN_CONFIG_INTERVAL_S,
    STATE_VERSION,
    MemoryPollStore,
    PolicyError,
    PollScheduler,
    PollStore,
    SchedulerError,
    SkuState,
    SourcesError,
    SourceState,
    load_policies,
)
from jarvis_poke.store import (
    BUSY_TIMEOUT_S,
    DB_FILE_MODE,
    POLL_STATE_VERSION_KEY,
    SCHEMA_VERSION,
    PokeStore,
    SchemaError,
    SqlitePollStore,
    StoreError,
    assert_round_trip_equal,
    first_difference,
    round_trip_equal,
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    from jarvis_poke import alerts_bridge, cli, validate

__version__ = "1.0.0"

#: Submodules resolved on first attribute access, and the public names each
#: one lends to the package.  ``alerts_bridge`` needs ``jarvis_alerts``;
#: ``validate`` and ``cli`` pull in everything.
_LAZY_MODULES = {
    "alerts_bridge": (
        "ALERT_KIND",
        "DEDUPE_WINDOW_S",
        "DEFAULT_PROFILE_ID",
        "AlertBridge",
        "BridgeError",
        "alert_data",
        "dedupe_key_for",
        "landed_cents",
    ),
    "validate": (
        "DEFECTS",
        "GATE_EPOCH",
        "GateError",
        "GateReport",
        "Problem",
        "parse_listing",
        "run_gate",
    ),
    "cli": (
        "DEFAULT_DB",
        "EXIT_FAIL",
        "EXIT_OK",
        "EXIT_USAGE",
        "TREND_BUCKETS",
        "CliError",
        "build_parser",
        "install_fetcher",
        "install_parser",
        "page_state",
        "scrub",
        "stub_fetcher",
        "stub_parser",
    ),
}

#: ``{exported name: module that defines it}`` for the lazy modules.
_LAZY_NAMES = {
    name: module for module, names in _LAZY_MODULES.items() for name in names
}

_EAGER_MODULES = (
    "catalog",
    "contracts",
    "engine",
    "prices",
    "rules",
    "sources",
    "store",
)


def __getattr__(name: str) -> Any:
    """Resolve the lazy submodules and the names they export (PEP 562)."""
    import importlib

    if name in _LAZY_MODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    owner = _LAZY_NAMES.get(name)
    if owner is not None:
        module = importlib.import_module(f"{__name__}.{owner}")
        globals().setdefault(owner, module)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))


__all__ = [
    # submodules
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
    # contracts
    "CURRENCY",
    "Cents",
    "to_cents",
    "fmt_cents",
    "ProductKind",
    "Product",
    "SourceSku",
    "Stock",
    "Observation",
    "FetchPolicy",
    "FetchResult",
    "Fetcher",
    "Parser",
    "Rule",
    "Budget",
    "MarketRef",
    "Action",
    "Verdict",
    "WatchState",
    "MAX_BUDGET_FRACTION_PER_VERDICT",
    "MARKET_STALE_AFTER_S",
    "MARKET_WINDOW_S",
    # catalog
    "CATALOG_PATH",
    "DATA_DIR",
    "SOURCES_PATH",
    "Catalog",
    "CatalogError",
    "load_catalog",
    # sources
    "BACKOFF_MAX_DOUBLINGS",
    "DEFAULT_JITTER_FRACTION",
    "MIN_CONFIG_INTERVAL_S",
    "STATE_VERSION",
    "MemoryPollStore",
    "PolicyError",
    "PollScheduler",
    "PollStore",
    "SchedulerError",
    "SkuState",
    "SourceState",
    "SourcesError",
    "load_policies",
    # prices
    "HISTORY_VERSION",
    "MIN_SAMPLES_FOR_REFERENCE",
    "OUTLIER_MIN_PCT_OF_MEDIAN",
    "HistoryStore",
    "MemoryHistoryStore",
    "PriceHistory",
    "PricesError",
    "discount_pct",
    "is_outlier",
    "market_reference",
    "price_trend",
    # rules
    "Affordability",
    "BudgetError",
    "RuleConflict",
    "RuleError",
    "RuleSet",
    "affordable",
    "budget_cap",
    # engine
    "DEFAULT_EXPLAIN_CHARS",
    "DISCOUNT_PRECISION",
    "OUTLIER_LOW_FRACTION",
    "DecisionEngine",
    "EngineError",
    "MarketHistory",
    "OutlierCheck",
    "discount_against",
    "discount_threshold",
    "explain",
    "looks_mis_parsed",
    # store
    "BUSY_TIMEOUT_S",
    "DB_FILE_MODE",
    "POLL_STATE_VERSION_KEY",
    "SCHEMA_VERSION",
    "PokeStore",
    "SchemaError",
    "SqlitePollStore",
    "StoreError",
    "assert_round_trip_equal",
    "first_difference",
    "round_trip_equal",
    # alerts_bridge (lazy)
    "ALERT_KIND",
    "DEDUPE_WINDOW_S",
    "DEFAULT_PROFILE_ID",
    "AlertBridge",
    "BridgeError",
    "alert_data",
    "dedupe_key_for",
    "landed_cents",
    # validate (lazy)
    "DEFECTS",
    "GATE_EPOCH",
    "GateError",
    "GateReport",
    "Problem",
    "parse_listing",
    "run_gate",
    # cli (lazy)
    "DEFAULT_DB",
    "EXIT_FAIL",
    "EXIT_OK",
    "EXIT_USAGE",
    "TREND_BUCKETS",
    "CliError",
    "build_parser",
    "install_fetcher",
    "install_parser",
    "page_state",
    "scrub",
    "stub_fetcher",
    "stub_parser",
    "__version__",
]
