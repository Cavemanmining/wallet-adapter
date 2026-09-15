"""The Descent: Lucifer's endgame node web.

Spec: docs/WORLD_BIBLE.md section 03, and the reconnect rule in section 07.

    >>> from lucifer_descent import generate_web, DescentEngine, mint_sigil
    >>> web = generate_web(0x5EED)
    >>> state = DescentEngine.new_profile("p1", web)
    >>> state.reachable_ids()[:3]
    [1, 2, 3]

Layout of the package:

    :mod:`lucifer_descent.contracts`  the shared types and the transition table
    :mod:`lucifer_descent.web`        one planar web per profile seed
    :mod:`lucifer_descent.engine`     the rules: opening portals, clears, deaths
    :mod:`lucifer_descent.sigils`     minting, the sustain drop rule, pre-roll
    :mod:`lucifer_descent.store`      persistence (memory and sqlite)
    :mod:`lucifer_descent.render`     the table view as a PNG
    :mod:`lucifer_descent.validate`   the gate, and ``simulate``
    :mod:`lucifer_descent.cli`        ``python3 -m lucifer_descent.cli``

Every submodule is imported here except ``cli`` (importing the package should
not pull in argparse), so ``lucifer_descent.web`` is always the module.  No
function re-exported below shares a name with a submodule -- ``generate_web``
not ``web``, ``render_web`` not ``render``, ``run_suite`` not ``validate`` --
so ``from lucifer_descent import web`` keeps giving the module.  That is the
shadowing bug ``lucifer_gen`` hit with ``route``/``translate``/``tileize``.
"""

from __future__ import annotations

__version__ = "0.1.0"

# The submodules, so ``lucifer_descent.<name>`` is always the module object.
from . import (  # noqa: F401
    contracts,
    engine,
    render,
    sigils,
    store,
    validate,
    web,
)

# -- contracts -------------------------------------------------------------
from .contracts import (
    ELITE_CLEAR_FRACTION,
    FRAGMENTS_TO_UNLOCK,
    MAX_TIER,
    SIGIL_CONSUMING_EVENTS,
    STATE_COLOUR,
    TRANSITIONS,
    Event,
    IllegalTransition,
    Instance,
    LedgerEntry,
    Mechanic,
    NodeState,
    Pinnacle,
    PortalOpened,
    ProfileState,
    Sigil,
    Web,
    WebEdge,
    WebNode,
    next_state,
)

# -- web -------------------------------------------------------------------
from .web import (
    adjacency_of,
    bfs_tiers,
    find_crossing,
    generate_web,
    is_planar_layout,
    matches_seed,
    regenerated,
    rings_of,
    segments_cross,
)

# -- engine ----------------------------------------------------------------
from .engine import (
    CorruptInstance,
    DescentEngine,
    DescentError,
    DuplicateSigil,
    InstanceAlreadyActive,
    LedgerMismatch,
    MalformedWeb,
    MapProbe,
    NoActiveInstance,
    NodeNotOpenable,
    PinnacleLocked,
    PortalCandidate,
    SigilTooWeak,
    UnknownNode,
    UnknownSigil,
    WrongProfile,
    default_map_probe,
)

# -- sigils ----------------------------------------------------------------
from .sigils import (
    MIN_TIER,
    clamp_tier,
    is_genuine,
    mint_sigil,
    node_is_openable,
    prewarm_candidates,
    roll_drops,
    sigil_id,
    stash_summary,
    tier_from_id,
    unmint,
)

# -- store -----------------------------------------------------------------
from .store import (
    SCHEMA_VERSION,
    DescentStore,
    MemoryStore,
    SchemaError,
    SqliteStore,
    StaleState,
    assert_round_trip_equal,
    first_difference,
    revision_of,
    round_trip_equal,
    validate_state,
)

# -- render ----------------------------------------------------------------
from .render import draw_web, render_web, render_web_bytes

# -- validate (the gate) ---------------------------------------------------
from .validate import (
    Problem,
    SuiteReport,
    check_replay,
    check_state,
    check_web,
    run_suite,
    simulate,
)

__all__ = [
    "__version__",
    # submodules (``cli`` is importable but not loaded here)
    "contracts", "engine", "render", "sigils", "store", "validate", "web",
    # contracts
    "ELITE_CLEAR_FRACTION", "FRAGMENTS_TO_UNLOCK", "MAX_TIER",
    "SIGIL_CONSUMING_EVENTS", "STATE_COLOUR", "TRANSITIONS", "Event",
    "IllegalTransition", "Instance", "LedgerEntry", "Mechanic", "NodeState",
    "Pinnacle", "PortalOpened", "ProfileState", "Sigil", "Web", "WebEdge",
    "WebNode", "next_state",
    # web
    "adjacency_of", "bfs_tiers", "find_crossing", "generate_web",
    "is_planar_layout", "matches_seed", "regenerated", "rings_of", "segments_cross",
    # engine
    "CorruptInstance", "DescentEngine", "DescentError", "DuplicateSigil", "InstanceAlreadyActive",
    "LedgerMismatch", "MalformedWeb", "MapProbe", "NoActiveInstance",
    "NodeNotOpenable", "PinnacleLocked", "PortalCandidate", "SigilTooWeak",
    "UnknownNode", "UnknownSigil", "WrongProfile", "default_map_probe",
    # sigils
    "MIN_TIER", "clamp_tier", "is_genuine", "mint_sigil", "node_is_openable",
    "prewarm_candidates", "roll_drops", "sigil_id", "stash_summary",
    "tier_from_id", "unmint",
    # store
    "SCHEMA_VERSION", "DescentStore", "MemoryStore", "SchemaError",
    "SqliteStore", "StaleState", "assert_round_trip_equal", "first_difference",
    "revision_of", "round_trip_equal", "validate_state",
    # render
    "draw_web", "render_web", "render_web_bytes",
    # validate
    "Problem", "SuiteReport", "check_replay", "check_state", "check_web",
    "run_suite", "simulate",
]

# Guard against the lucifer_gen shadowing bug: every submodule name must still
# resolve to a module after all the re-exports above.
import types as _types

for _name in ("contracts", "engine", "render", "sigils", "store", "validate", "web"):
    if not isinstance(globals()[_name], _types.ModuleType):
        raise ImportError(f"lucifer_descent.{_name} was shadowed by a re-export")
del _types, _name
