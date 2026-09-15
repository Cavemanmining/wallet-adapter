"""Shared types for the Descent, Lucifer's endgame node web.

Spec: docs/WORLD_BIBLE.md section 03, and the reconnect rule in section 07.

This module is the keystone for Phase 4: every other module builds against
these types, and the state-transition table lives here and nowhere else so it
cannot drift between the engine that applies it and the storage that records
it.

Terminology
-----------
A *profile* owns exactly one *web*, one passive-point ledger, one Sigil
stash, and a fragment ledger for each Pinnacle. A web is a planar graph of
*nodes* joined by *edges* that are never removed. A node's *tier* is its
graph distance from the origin node, clamped to 15. A *Sigil* is a consumable
key item; inserting one at the Descent table opens a portal into a reachable
node and creates an *instance*. If the character dies inside, the Sigil is
gone and the instance is destroyed.

Judgement calls the spec does not settle, recorded here so they are visible:

* A Sigil may open a node only if its tier is at least the node's tier. The
  Sigil's tier, not the node's, feeds the spawn density multiplier, so a
  high Sigil on a low node is harder and richer.
* Leaving an active instance without completing it (``ABANDON``) is treated
  like death: the Sigil is consumed and the node becomes ``FAILED``.
* The 60-second reconnect window from section 07 closes the instance and
  consumes the Sigil; that arrives here as ``TIMEOUT`` and is also ``FAILED``.
* The Pinnacle glyph on a tier-15 node is separate from a node *being* a
  Pinnacle arena. Fragment sources carry ``glyph``; arenas carry ``pinnacle``.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------


class NodeState(enum.Enum):
    LOCKED = "locked"
    REACHABLE = "reachable"
    ACTIVE = "active"
    CLEARED = "cleared"
    FAILED = "failed"


#: The colour a node is *shown* as. Derived from state only, never stored.
STATE_COLOUR: Dict[NodeState, str] = {
    NodeState.LOCKED: "grey",
    NodeState.REACHABLE: "blue",
    NodeState.ACTIVE: "amber",
    NodeState.CLEARED: "green",
    NodeState.FAILED: "red",
}


class Mechanic(enum.Enum):
    BREACH = "breach"
    RITUAL = "ritual"
    DIG = "dig"
    SHRINE = "shrine"


class Pinnacle(enum.Enum):
    ARBITER = "arbiter"    # the Arbiter of Cinders
    MONOLITH = "monolith"  # the Blind Monolith


#: Fragments needed from glyph-bearing cleared tier-15 nodes to unlock an arena.
FRAGMENTS_TO_UNLOCK = 3

MAX_TIER = 15

#: Fraction of elite packs that must die to clear a node with no boss.
ELITE_CLEAR_FRACTION = 0.80


class Event(enum.Enum):
    """Something that happened to a node or its live instance."""

    NEIGHBOUR_CLEARED = "neighbour_cleared"
    OPEN = "open"            # a Sigil was inserted
    BOSS_KILLED = "boss_killed"
    ELITES_MET = "elites_met"  # the 80 percent threshold was crossed
    DIED = "died"
    ABANDON = "abandon"
    TIMEOUT = "timeout"


# --------------------------------------------------------------------------
# The transition table. This is the single source of truth.
# --------------------------------------------------------------------------

#: (state, event) -> next state. Absent pairs are illegal and must raise.
TRANSITIONS: Dict[Tuple[NodeState, Event], NodeState] = {
    (NodeState.LOCKED, Event.NEIGHBOUR_CLEARED): NodeState.REACHABLE,
    (NodeState.REACHABLE, Event.OPEN): NodeState.ACTIVE,
    (NodeState.FAILED, Event.OPEN): NodeState.ACTIVE,
    (NodeState.ACTIVE, Event.BOSS_KILLED): NodeState.CLEARED,
    (NodeState.ACTIVE, Event.ELITES_MET): NodeState.CLEARED,
    (NodeState.ACTIVE, Event.DIED): NodeState.FAILED,
    (NodeState.ACTIVE, Event.ABANDON): NodeState.FAILED,
    (NodeState.ACTIVE, Event.TIMEOUT): NodeState.FAILED,
}

#: Events on which the live instance is destroyed and its Sigil is consumed.
SIGIL_CONSUMING_EVENTS: FrozenSet[Event] = frozenset(
    {Event.BOSS_KILLED, Event.ELITES_MET, Event.DIED, Event.ABANDON, Event.TIMEOUT}
)


class IllegalTransition(Exception):
    """Raised when an event is applied to a node whose state does not allow it."""


def next_state(state: NodeState, event: Event) -> NodeState:
    try:
        return TRANSITIONS[(state, event)]
    except KeyError:
        raise IllegalTransition(f"{event.value} is not allowed while {state.value}") from None


# --------------------------------------------------------------------------
# Static web structure (generated once per profile, never mutated)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class WebNode:
    id: int
    tier: int
    ring_index: int            # position around its ring, for drawing
    template: str              # generator template id, e.g. "crypt"
    mechanic: Optional[Mechanic] = None
    pinnacle: Optional[Pinnacle] = None   # this node IS an arena
    glyph: Optional[Pinnacle] = None      # this node yields a fragment when cleared
    x: float = 0.0             # layout position for the table view
    y: float = 0.0


@dataclass(frozen=True)
class WebEdge:
    a: int
    b: int

    def other(self, node_id: int) -> int:
        if node_id == self.a:
            return self.b
        if node_id == self.b:
            return self.a
        raise ValueError(f"edge {self.a}-{self.b} does not touch {node_id}")


@dataclass(frozen=True)
class Web:
    profile_seed: int
    origin_id: int
    nodes: Tuple[WebNode, ...]
    edges: Tuple[WebEdge, ...]
    version: int = 1

    def node(self, node_id: int) -> WebNode:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)

    def neighbours(self, node_id: int) -> List[int]:
        out = [e.other(node_id) for e in self.edges if e.a == node_id or e.b == node_id]
        return sorted(out)

    def nodes_at_tier(self, tier: int) -> List[WebNode]:
        return [n for n in self.nodes if n.tier == tier]


# --------------------------------------------------------------------------
# Items and ledgers (mutable per profile)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Sigil:
    """A consumable key. ``seed`` becomes the map seed when it is used."""

    id: str
    tier: int
    seed: int

    def can_open(self, node: WebNode) -> bool:
        return self.tier >= node.tier

    def __post_init__(self) -> None:
        if not 1 <= self.tier <= MAX_TIER:
            raise ValueError(f"sigil tier out of range: {self.tier}")


@dataclass
class Instance:
    """A live portal into one node, funded by one Sigil."""

    node_id: int
    sigil: Sigil
    map_seed: int
    has_boss: bool
    elite_total: int
    elite_killed: int = 0
    opened_tick: int = 0

    def elite_fraction(self) -> float:
        if self.elite_total <= 0:
            return 1.0
        return self.elite_killed / self.elite_total


@dataclass
class ProfileState:
    """Everything mutable about one profile's Descent."""

    profile_id: str
    web: Web
    states: Dict[int, NodeState]
    stash: Dict[str, Sigil] = field(default_factory=dict)
    passive_points: int = 0
    fragments: Dict[Pinnacle, int] = field(default_factory=dict)
    unlocked_pinnacles: FrozenSet[Pinnacle] = frozenset()
    instance: Optional[Instance] = None
    history: List["LedgerEntry"] = field(default_factory=list)

    def state_of(self, node_id: int) -> NodeState:
        return self.states[node_id]

    def reachable_ids(self) -> List[int]:
        return sorted(i for i, s in self.states.items() if s is NodeState.REACHABLE)

    def cleared_ids(self) -> List[int]:
        return sorted(i for i, s in self.states.items() if s is NodeState.CLEARED)


@dataclass(frozen=True)
class LedgerEntry:
    """One applied event, kept so a profile can be audited or replayed.

    An ``OPEN`` entry also records the two facts the map probe answered when
    the portal opened -- ``has_boss`` and ``elite_total`` -- so a replay can
    check that the node was later cleared the way that map allows (a boss
    kill on a boss map, the elite threshold on a bossless one) and an audit
    can compare the live instance with what the ledger says it was opened
    on.  Both are ``None`` on every other kind of entry.
    """

    seq: int
    node_id: int
    event: Event
    before: NodeState
    after: NodeState
    sigil_id: Optional[str] = None
    tick: int = 0
    has_boss: Optional[bool] = None
    elite_total: Optional[int] = None


# --------------------------------------------------------------------------
# Portal-opening result, what the table returns to the game
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PortalOpened:
    node_id: int
    template: str
    map_seed: int
    sigil_tier: int
    node_tier: int
    mechanic: Optional[Mechanic]
    pinnacle: Optional[Pinnacle]


__all__ = [
    "NodeState", "STATE_COLOUR", "Mechanic", "Pinnacle", "FRAGMENTS_TO_UNLOCK",
    "MAX_TIER", "ELITE_CLEAR_FRACTION", "Event", "TRANSITIONS",
    "SIGIL_CONSUMING_EVENTS", "IllegalTransition", "next_state", "WebNode",
    "WebEdge", "Web", "Sigil", "Instance", "ProfileState", "LedgerEntry",
    "PortalOpened",
]
