"""The rules engine that drives one profile's Descent.

Spec: docs/WORLD_BIBLE.md section 03, and the reconnect rule in section 07.

Everything mutable about a profile lives in a :class:`ProfileState`; this
module owns the *rules* for changing it.  The engine holds one state and
answers the Descent table's questions: may this Sigil open that node, what
happened when the boss died, which portals could be pre-rolled.

Three properties are load-bearing and every method is written to keep them:

* **One choke point.**  Every state change goes through :meth:`DescentEngine._apply`,
  which consults :func:`lucifer_descent.contracts.next_state` (the only legal
  set of moves), appends a :class:`LedgerEntry`, and performs the consequences
  of entering ``CLEARED`` or ``FAILED``.  Nothing else writes ``states``.
* **The ledger replays, and replay re-derives.**  The ledger is append-only
  and sequential, and :meth:`DescentEngine.replay` rebuilds the identical
  ``states`` dict, passive points, fragments and unlocked Pinnacles from it
  over a fresh profile.  Replay does not merely check each entry against the
  transition table: it re-derives everything the live engine derives -- which
  neighbours a clear makes reachable and in what order, that an arena was
  unlocked when it was opened, that one instance ran at a time, that each
  Sigil funded one open and was consumed by the event that closed that same
  instance, and that a node was cleared the way its map allows.  A ledger
  that says otherwise is refused with :class:`LedgerMismatch`.
* **Determinism.**  The engine draws no randomness at all.  The only seed a
  portal ever uses is the Sigil's own ``seed`` (spec: "the Sigil's own item
  seed becomes the map seed"), and every iteration over a set or dict whose
  order could reach the ledger is sorted first.

Judgement calls the spec and the contracts left open, recorded here:

* The origin starts ``CLEARED`` by genesis, not by an event: the transition
  table has no ``LOCKED -> CLEARED`` move, so the origin's initial state is
  assigned directly by :meth:`DescentEngine.new_profile` and only the
  resulting ``NEIGHBOUR_CLEARED`` propagation to ring 1 is recorded in the
  ledger.  The origin was never cleared *by the player*, so it grants no
  passive point or fragment.
* The Pinnacle-unlocked check for an arena node sits between the node-state
  check and the Sigil checks: it is a property of the node, so it is reported
  before anything about the key that was offered.
* The map probe runs *before* the Sigil leaves the stash, so a generator
  failure cannot destroy an item.  Once the probe has answered, the open is
  committed in one go.
* A bossless map with zero elite packs is cleared the moment it opens: "80
  percent of elite packs dead" is satisfied by zero of zero, which is what
  :meth:`Instance.elite_fraction` reports for an empty map.  The ledger then
  shows ``OPEN`` followed at once by ``ELITES_MET`` at the same tick, and no
  elite kill is ever reported on such a map because there is no instance to
  report it to.
* A Sigil id is spent once per profile.  :meth:`DescentEngine.add_sigil`
  refuses an id that is in the stash, is funding the live instance, or has
  ever funded an open recorded in the ledger.
* The live instance is checked against the ledger before any report is
  applied to it: the instance's node must be ``ACTIVE``, and the last ledger
  entry naming that node must be the ``OPEN`` that created the instance,
  with the same Sigil id, tick and probe facts.  A persisted instance that
  disagrees with the ledger is refused with :class:`CorruptInstance` rather
  than played on.
* A game clock that runs backwards is not the engine's concern: it stamps
  whatever ``tick_source()`` says.  The gate reports non-monotone ticks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Set

from lucifer_descent.contracts import (
    ELITE_CLEAR_FRACTION,
    FRAGMENTS_TO_UNLOCK,
    SIGIL_CONSUMING_EVENTS,
    STATE_COLOUR,
    Event,
    IllegalTransition,
    Instance,
    LedgerEntry,
    NodeState,
    PortalOpened,
    ProfileState,
    Sigil,
    Web,
    WebNode,
    next_state,
)

__all__ = [
    "MapProbe",
    "default_map_probe",
    "PortalCandidate",
    "DescentError",
    "WrongProfile",
    "UnknownNode",
    "NodeNotOpenable",
    "PinnacleLocked",
    "UnknownSigil",
    "SigilTooWeak",
    "InstanceAlreadyActive",
    "NoActiveInstance",
    "CorruptInstance",
    "DuplicateSigil",
    "MalformedWeb",
    "LedgerMismatch",
    "DescentEngine",
]


# --------------------------------------------------------------------------
# The map probe: what the engine needs to know about a map to judge completion
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MapProbe:
    """The two facts about a generated map that decide when a node is cleared.

    Spec: "Completion: boss dead, or 80 percent of elite packs dead when the
    map has no boss."  ``elite_total`` counts *packs* flagged elite, since the
    rule is written in packs.  Both facts are recorded on the ``OPEN`` ledger
    entry so a replay can hold the clear to them.
    """

    has_boss: bool
    elite_total: int

    def __post_init__(self) -> None:
        if not isinstance(self.has_boss, bool):
            raise ValueError(f"has_boss must be a bool, got {type(self.has_boss).__name__}")
        if isinstance(self.elite_total, bool) or not isinstance(self.elite_total, int):
            raise ValueError(f"elite_total must be an int, got {type(self.elite_total).__name__}")
        if self.elite_total < 0:
            raise ValueError(f"elite_total must not be negative, got {self.elite_total}")

    def clears_on_open(self) -> bool:
        """A bossless map with no elite packs is complete the moment it opens."""
        return not self.has_boss and self.elite_total == 0


def default_map_probe(template: str, map_seed: int, sigil_tier: int) -> MapProbe:
    """Probe a map by generating it with the real pipeline.

    The generator is imported lazily so an engine fed a stub probe (every test
    here) never loads the tile databases.  ``sigil_tier`` is passed as the
    generator's ``tier`` because the Sigil's tier, not the node's, feeds spawn
    density (the judgement call recorded in ``contracts``).

    ``has_boss`` is true when any stamped set piece id contains ``"boss"`` --
    the rule this module was asked to implement -- *or* when the routed layout
    has a node in the ``BOSS`` role.  The second test is there because the
    shipped ``ashen_ramparts`` template stamps its boss as
    ``ramparts_warlord_v1``: by the substring alone it would count as bossless
    and clear on 80 percent of its elites.
    """
    from lucifer_gen.contracts import Role
    from lucifer_gen.pipeline import generate

    gmap = generate(template, None, None, map_seed, tier=sigil_tier)
    by_piece = any("boss" in piece.id for piece in gmap.set_pieces)
    by_role = gmap.routed.node_of_role(Role.BOSS) is not None
    elite_total = sum(1 for pack in gmap.spawns if pack.elite)
    return MapProbe(has_boss=by_piece or by_role, elite_total=elite_total)


@dataclass(frozen=True)
class PortalCandidate:
    """One (openable node, usable Sigil, map seed) triple for pre-rolling.

    Spec: "The 170HX pre-rolls layouts for reachable nodes so portals open
    instantly."  The template and Sigil tier ride along because they are the
    other two inputs the generator needs, so a pre-roller can call
    ``generate(template, None, None, map_seed, tier=sigil_tier)`` directly.
    """

    node_id: int
    sigil_id: str
    map_seed: int
    template: str
    sigil_tier: int


# --------------------------------------------------------------------------
# Exceptions: one class per rule so callers can tell them apart
# --------------------------------------------------------------------------


class DescentError(Exception):
    """Base class for every rule the engine refuses to break."""


class WrongProfile(DescentError):
    """The profile id offered is not the profile this engine drives.

    Spec: "identity is checked on every portal open."  The engine checks it on
    every call that names a profile, not only on open.
    """


class UnknownNode(DescentError):
    """No node with that id exists in this profile's web."""


class NodeNotOpenable(DescentError):
    """The node is not ``REACHABLE`` or ``FAILED``, the only states ``OPEN`` may leave."""


class PinnacleLocked(DescentError):
    """The node is a Pinnacle arena whose three fragments have not been collected."""


class UnknownSigil(DescentError):
    """No Sigil with that id is in this profile's stash."""


class SigilTooWeak(DescentError):
    """``sigil.tier < node.tier``; a Sigil opens a node only of its tier or lower."""


class InstanceAlreadyActive(DescentError):
    """A profile runs one instance at a time; finish or fail it first."""


class NoActiveInstance(DescentError):
    """A kill, death, abandon or timeout was reported with no live instance."""


class CorruptInstance(DescentError):
    """The live instance does not agree with the ledger's record of its open.

    The instance is persisted beside the ledger, not derived from it, so the
    two can be made to disagree by editing a save.  The engine refuses to
    apply a report to an instance whose node is not ``ACTIVE``, whose Sigil,
    tick or probe facts differ from the ``OPEN`` entry that created it, or
    whose kill count is impossible.
    """


class DuplicateSigil(DescentError):
    """That Sigil id is already in the stash, funding the live instance, or
    was spent on an earlier open; ids are unique per profile, for ever."""


class MalformedWeb(DescentError):
    """The web's structure is inconsistent: unknown origin, dangling edge, duplicate id."""


class LedgerMismatch(DescentError):
    """A ledger being replayed does not describe a legal history of this web."""


# --------------------------------------------------------------------------
# The engine
# --------------------------------------------------------------------------


def _unavailable_probe(template: str, map_seed: int, sigil_tier: int) -> MapProbe:
    """Probe used by the engines :meth:`DescentEngine.new_profile` and
    :meth:`DescentEngine.replay` build internally; they never open a portal."""
    raise RuntimeError("this engine only builds or replays a profile; it cannot open portals")


class DescentEngine:
    """Applies the section 03 rules to one :class:`ProfileState`.

    ``map_probe(template, map_seed, sigil_tier)`` says whether the map about to
    be opened has a boss and how many elite packs it holds.
    ``tick_source()`` is the game clock; its value is stamped on the instance
    and on every ledger entry, and the engine never reads any other clock.

    The web is read once, at construction: the node table and the adjacency
    the engine propagates over are both snapshots of ``state.web`` taken
    here, so the rules can never see two different webs mid-run even if the
    state's ``web`` attribute is reassigned underneath them.
    """

    def __init__(
        self,
        state: ProfileState,
        map_probe: Callable[[str, int, int], MapProbe],
        tick_source: Callable[[], int],
    ) -> None:
        self._validate_web(state.web)
        self.state = state
        self._map_probe = map_probe
        self._tick_source = tick_source
        self._nodes: Dict[int, WebNode] = {n.id: n for n in state.web.nodes}
        self._adjacency: Dict[int, List[int]] = {n.id: [] for n in state.web.nodes}
        for edge in state.web.edges:
            self._adjacency[edge.a].append(edge.b)
            self._adjacency[edge.b].append(edge.a)
        for neighbours in self._adjacency.values():
            neighbours.sort()
        #: Every Sigil id the ledger has ever recorded: spent, for good.
        self._spent: Set[str] = {e.sigil_id for e in state.history if e.sigil_id is not None}
        # Replay bookkeeping; only :meth:`replay` drives these.
        self._pending: List[int] = []            # neighbours the last clear must still reach
        self._pending_tick: Optional[int] = None  # the tick that clear carried
        self._pending_autoclear: Optional[int] = None  # node an empty map must clear next
        self._replay_live: Optional[LedgerEntry] = None  # the OPEN of the live instance

    # ------------------------------------------------------------------
    # Construction and replay
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_web(web: Web) -> None:
        """Refuse a web whose structure the rules cannot run on.

        The engine trusts the web generator for the *design* rules -- tier
        equals graph distance, glyphs and arenas only at tier 15 -- and only
        checks what would otherwise surface as a ``KeyError`` mid-clear.
        Whether a stored web is the one its seed generates is the gate's and
        the shell's question (:func:`lucifer_descent.web.matches_seed`).
        """
        ids = [n.id for n in web.nodes]
        if len(ids) != len(set(ids)):
            raise MalformedWeb("duplicate node ids in web")
        known = set(ids)
        if web.origin_id not in known:
            raise MalformedWeb(f"origin {web.origin_id} is not a node of the web")
        for edge in web.edges:
            if edge.a not in known or edge.b not in known:
                raise MalformedWeb(f"edge {edge.a}-{edge.b} touches an unknown node")
            if edge.a == edge.b:
                raise MalformedWeb(f"edge {edge.a}-{edge.b} is a self loop")

    @classmethod
    def _genesis(cls, profile_id: str, web: Web) -> ProfileState:
        """The state before any event: all ``LOCKED``, the origin ``CLEARED``.

        Spec: "A node is reachable only if it shares an edge with a cleared
        node."  Something must be cleared before anything is reachable, and
        the origin is that something.  There is no transition into
        ``CLEARED`` from ``LOCKED``, so this is an assignment, not an event,
        and it is the only write to ``states`` outside :meth:`_apply`.
        """
        cls._validate_web(web)
        states = {n.id: NodeState.LOCKED for n in web.nodes}
        states[web.origin_id] = NodeState.CLEARED
        return ProfileState(profile_id=profile_id, web=web, states=states)

    @classmethod
    def new_profile(cls, profile_id: str, web: Web, tick: int = 0) -> ProfileState:
        """A fresh profile: origin cleared, ring 1 reachable, everything else locked.

        Ring 1 becomes ``REACHABLE`` through the ordinary
        ``NEIGHBOUR_CLEARED`` propagation, so those moves are the first
        entries in the ledger and stamped with ``tick``.
        """
        state = cls._genesis(profile_id, web)
        engine = cls(state, _unavailable_probe, lambda: tick)
        engine._propagate(web.origin_id, tick)
        return state

    @classmethod
    def replay(
        cls, profile_id: str, web: Web, history: Sequence[LedgerEntry]
    ) -> ProfileState:
        """Rebuild a profile's states and earned rewards from its ledger.

        Replay starts from genesis -- the origin cleared, nothing recorded --
        with the origin's propagation to ring 1 *expected* rather than
        applied, so the ring-1 entries are checked like every later one:
        each entry must be the next ``seq``, must start from the state the
        rebuilt profile is in, must land where the transition table says,
        and must be the entry the live engine would have written at that
        point.  In particular:

        * after any clear (the genesis one included) the next entries must
          be ``NEIGHBOUR_CLEARED`` for exactly that node's ``LOCKED``
          neighbours, in ascending id order, all carrying the clear's tick;
          a ``NEIGHBOUR_CLEARED`` anywhere else is refused;
        * an ``OPEN`` must name its Sigil and record the probe's facts, must
          not open a locked arena, must not start a second instance, and
          must not reuse a Sigil id an earlier ``OPEN`` spent;
        * a Sigil-consuming event must close the instance that is live, with
          the Sigil that opened it, and a boss kill needs a boss while the
          elite threshold needs the map to have none;
        * an ``OPEN`` on a bossless map with no packs must be followed at
          once by ``ELITES_MET``.

        The genesis tick is whatever the first entry carries, so a profile
        created at any tick replays.  A ledger cut short mid-propagation
        rebuilds the state *at* that cut -- a cleared node with locked
        neighbours -- which the gate's neighbour rule then reports; replay
        never invents the entries a truncated ledger lacks.

        The ledger records what happened to nodes, not the stash or the live
        instance.  A replayed profile therefore has an empty stash and no
        instance even if its last entry is ``OPEN``; storage that needs those
        back must persist them alongside the ledger.
        """
        state = cls._genesis(profile_id, web)
        engine = cls(state, _unavailable_probe, lambda: 0)
        engine._expect_propagation(web.origin_id, tick=None)
        for entry in history:
            engine._replay_entry(entry)
        return state

    def _replay_entry(self, entry: LedgerEntry) -> None:
        """Apply one recorded entry after checking it is the entry the live
        engine would have written next; see :meth:`replay` for the rules."""
        seq = entry.seq
        if seq != len(self.state.history) + 1:
            raise LedgerMismatch(
                f"entry {seq} arrived when {len(self.state.history) + 1} was expected"
            )
        if entry.node_id not in self._nodes:
            raise LedgerMismatch(f"entry {seq} names unknown node {entry.node_id}")
        node = self._nodes[entry.node_id]

        # Whatever the last clear left to propagate comes first, and nothing
        # else may be a propagation.
        if self._pending:
            want = self._pending[0]
            if entry.event is not Event.NEIGHBOUR_CLEARED or entry.node_id != want:
                raise LedgerMismatch(
                    f"entry {seq}: expected neighbour_cleared on node {want} to finish "
                    f"the last clear's propagation, got {entry.event.value} on node {entry.node_id}"
                )
            if self._pending_tick is None:
                self._pending_tick = entry.tick  # the genesis batch sets its own tick
            elif entry.tick != self._pending_tick:
                raise LedgerMismatch(
                    f"entry {seq}: neighbour_cleared carries tick {entry.tick}, "
                    f"the clear it propagates was at tick {self._pending_tick}"
                )
            self._pending.pop(0)
        elif entry.event is Event.NEIGHBOUR_CLEARED:
            raise LedgerMismatch(
                f"entry {seq}: neighbour_cleared on node {entry.node_id} with no clear to propagate"
            )
        elif self._pending_autoclear is not None:
            if entry.event is not Event.ELITES_MET or entry.node_id != self._pending_autoclear:
                raise LedgerMismatch(
                    f"entry {seq}: node {self._pending_autoclear} opened on a bossless map with "
                    f"no elite packs and must clear at once; got {entry.event.value} on node {entry.node_id}"
                )
            self._pending_autoclear = None

        before = self.state.states[entry.node_id]
        if before is not entry.before:
            raise LedgerMismatch(
                f"entry {seq}: node {entry.node_id} is {before.value}, "
                f"ledger says {entry.before.value}"
            )
        try:
            after = next_state(before, entry.event)
        except IllegalTransition as exc:
            raise LedgerMismatch(f"entry {seq}: {exc}") from None
        if after is not entry.after:
            raise LedgerMismatch(
                f"entry {seq}: {entry.event.value} leads to {after.value}, "
                f"ledger says {entry.after.value}"
            )

        if entry.event is Event.OPEN:
            self._replay_check_open(entry, node)
        elif entry.event in SIGIL_CONSUMING_EVENTS:
            self._replay_check_consume(entry)

        self._commit(entry)
        if entry.event is Event.OPEN:
            self._replay_live = entry
            if entry.has_boss is False and entry.elite_total == 0:
                self._pending_autoclear = entry.node_id
        elif entry.event in SIGIL_CONSUMING_EVENTS:
            self._replay_live = None
        if after is NodeState.CLEARED:
            self._grant_clear_rewards(node)
            self._expect_propagation(entry.node_id, entry.tick)

    def _replay_check_open(self, entry: LedgerEntry, node: WebNode) -> None:
        """The rules :meth:`open_portal` applies that the ledger can witness."""
        seq = entry.seq
        if entry.sigil_id is None:
            raise LedgerMismatch(f"entry {seq}: open records no sigil id")
        if entry.has_boss is None or entry.elite_total is None:
            raise LedgerMismatch(f"entry {seq}: open records no map facts (has_boss, elite_total)")
        if entry.elite_total < 0:
            raise LedgerMismatch(f"entry {seq}: open records {entry.elite_total} elite packs")
        if node.pinnacle is not None and node.pinnacle not in self.state.unlocked_pinnacles:
            raise LedgerMismatch(
                f"entry {seq}: node {node.id} is the {node.pinnacle.value} arena, opened while locked"
            )
        active = [nid for nid in sorted(self.state.states) if self.state.states[nid] is NodeState.ACTIVE]
        if active:
            raise LedgerMismatch(
                f"entry {seq}: node {node.id} opened while node {active[0]} was already active"
            )
        if entry.sigil_id in self._spent:
            raise LedgerMismatch(
                f"entry {seq}: sigil {entry.sigil_id!r} was already spent by an earlier open"
            )

    def _replay_check_consume(self, entry: LedgerEntry) -> None:
        """A consuming event closes the live instance with the Sigil that
        opened it, and only in the way that instance's map allows."""
        seq = entry.seq
        live = self._replay_live
        if live is None or live.node_id != entry.node_id:
            raise LedgerMismatch(
                f"entry {seq}: {entry.event.value} on node {entry.node_id} but no open is live there"
            )
        if entry.sigil_id != live.sigil_id:
            raise LedgerMismatch(
                f"entry {seq}: {entry.event.value} consumes sigil {entry.sigil_id!r}, "
                f"the open at entry {live.seq} was funded by {live.sigil_id!r}"
            )
        if entry.event is Event.BOSS_KILLED and not live.has_boss:
            raise LedgerMismatch(
                f"entry {seq}: boss_killed on node {entry.node_id}, whose map has no boss"
            )
        if entry.event is Event.ELITES_MET and live.has_boss:
            raise LedgerMismatch(
                f"entry {seq}: elites_met on node {entry.node_id}, whose map has a boss"
            )

    def _expect_propagation(self, cleared_id: int, tick: Optional[int]) -> None:
        """Note what the live engine would propagate after this clear."""
        self._pending = self._propagation_targets(cleared_id)
        self._pending_tick = tick

    # ------------------------------------------------------------------
    # Read-only queries
    # ------------------------------------------------------------------

    def colour_of(self, node_id: int) -> str:
        """The colour a node is shown as; derived from its state, never stored.

        Spec: "locked grey, reachable blue, active amber, cleared green,
        failed red. Colour is derived from state, never stored."
        """
        return STATE_COLOUR[self.state.states[self._node(node_id).id]]

    def spent_sigil_ids(self) -> Set[str]:
        """Every Sigil id the ledger records; none of them can enter the stash again."""
        return set(self._spent)

    def portal_candidates(self, profile_id: str) -> List[PortalCandidate]:
        """Every (openable node, usable Sigil, map seed) triple, in a fixed order.

        This is what the 170HX pre-rolls.  It applies the same node and Sigil
        rules as :meth:`open_portal` -- state, Pinnacle lock, tier -- but
        deliberately ignores the one-instance rule, since pre-rolling while a
        run is in progress is the whole point.  Sorted by node id then Sigil
        id so the output is a stable function of the state.
        """
        self._require_profile(profile_id)
        out: List[PortalCandidate] = []
        for node_id in sorted(self.state.states):
            node = self._nodes[node_id]
            if not self._node_openable(node):
                continue
            for sigil_id in sorted(self.state.stash):
                sigil = self.state.stash[sigil_id]
                if sigil.can_open(node):
                    out.append(
                        PortalCandidate(
                            node_id=node.id,
                            sigil_id=sigil.id,
                            map_seed=sigil.seed,
                            template=node.template,
                            sigil_tier=sigil.tier,
                        )
                    )
        return out

    # ------------------------------------------------------------------
    # The stash
    # ------------------------------------------------------------------

    def add_sigil(self, profile_id: str, sigil: Sigil) -> None:
        """Put a Sigil in this profile's stash.

        Ids are unique within a profile for its whole life: an id already in
        the stash, funding the live instance, or recorded by the ledger as
        spent is refused with :class:`DuplicateSigil`.  A consumable key that
        could re-enter the stash under its old id would be spendable twice.
        """
        self._require_profile(profile_id)
        if sigil.id in self.state.stash:
            raise DuplicateSigil(f"sigil {sigil.id!r} is already in the stash")
        live = self.state.instance
        if live is not None and live.sigil.id == sigil.id:
            raise DuplicateSigil(f"sigil {sigil.id!r} is funding the live instance")
        if sigil.id in self._spent:
            raise DuplicateSigil(f"sigil {sigil.id!r} was already spent; ids are never reused")
        self.state.stash[sigil.id] = sigil

    # ------------------------------------------------------------------
    # Opening a portal
    # ------------------------------------------------------------------

    def open_portal(self, profile_id: str, node_id: int, sigil_id: str) -> PortalOpened:
        """Insert a Sigil at the table and open a portal into ``node_id``.

        Spec: "Inserting one opens a portal into a reachable node; the Sigil's
        own item seed becomes the map seed", the fixed judgement call that
        ``sigil.tier >= node.tier``, and "identity is checked on every portal
        open."  The checks run in this order, each with its own exception:

        1. wrong profile            -> :class:`WrongProfile`
        2. unknown node             -> :class:`UnknownNode`
        3. node not REACHABLE/FAILED -> :class:`NodeNotOpenable`
        4. arena still locked        -> :class:`PinnacleLocked`
        5. unknown sigil            -> :class:`UnknownSigil`
        6. sigil.tier < node.tier   -> :class:`SigilTooWeak`
        7. an instance is active    -> :class:`InstanceAlreadyActive`

        Only once all seven pass is anything changed: the Sigil leaves the
        stash, the instance is created with the Sigil's seed, ``OPEN`` is
        applied through the transition table and recorded in the ledger with
        the probe's two facts.  If the probe says the map has no boss and no
        elite packs, the node is cleared in the same call (see the module
        docstring) and the returned :class:`PortalOpened` describes a portal
        that has already closed.
        """
        self._require_profile(profile_id)
        node = self._node(node_id)
        state = self.state.states[node.id]
        if state not in (NodeState.REACHABLE, NodeState.FAILED):
            raise NodeNotOpenable(f"node {node.id} is {state.value}, not reachable or failed")
        if node.pinnacle is not None and node.pinnacle not in self.state.unlocked_pinnacles:
            raise PinnacleLocked(
                f"node {node.id} is the {node.pinnacle.value} arena; "
                f"{FRAGMENTS_TO_UNLOCK} fragments are needed to unlock it"
            )
        try:
            sigil = self.state.stash[sigil_id]
        except KeyError:
            raise UnknownSigil(f"no sigil {sigil_id!r} in the stash") from None
        if not sigil.can_open(node):
            raise SigilTooWeak(
                f"sigil {sigil.id!r} is tier {sigil.tier}, node {node.id} is tier {node.tier}"
            )
        if self.state.instance is not None or any(
            s is NodeState.ACTIVE for s in self.state.states.values()
        ):
            raise InstanceAlreadyActive("an instance is already active; finish or fail it first")

        # Probe first: a generator failure must not eat the Sigil.
        probe = self._map_probe(node.template, sigil.seed, sigil.tier)
        tick = self._tick_source()

        del self.state.stash[sigil.id]
        self.state.instance = Instance(
            node_id=node.id,
            sigil=sigil,
            map_seed=sigil.seed,
            has_boss=probe.has_boss,
            elite_total=probe.elite_total,
            opened_tick=tick,
        )
        self._apply(
            node.id, Event.OPEN, sigil_id=sigil.id, tick=tick,
            has_boss=probe.has_boss, elite_total=probe.elite_total,
        )
        if probe.clears_on_open():
            self._apply(node.id, Event.ELITES_MET, sigil_id=sigil.id, tick=tick)
        return PortalOpened(
            node_id=node.id,
            template=node.template,
            map_seed=sigil.seed,
            sigil_tier=sigil.tier,
            node_tier=node.tier,
            mechanic=node.mechanic,
            pinnacle=node.pinnacle,
        )

    # ------------------------------------------------------------------
    # Reports from inside a live instance
    # ------------------------------------------------------------------

    def report_elite_kill(self, profile_id: str) -> Optional[NodeState]:
        """An elite pack died.  Returns the node's new state if that cleared it.

        Spec: "80 percent of elite packs dead when the map has no boss."  A
        map with a boss is never cleared this way, however many elites fall.
        """
        instance = self._live_instance(profile_id)
        instance.elite_killed += 1
        if self._threshold_met(instance):
            return self._apply(instance.node_id, Event.ELITES_MET, sigil_id=instance.sigil.id)
        return None

    def report_boss_kill(self, profile_id: str) -> NodeState:
        """The boss died.  Spec: "Completion: boss dead"."""
        return self._end_instance(profile_id, Event.BOSS_KILLED)

    def report_death(self, profile_id: str) -> NodeState:
        """The character died.  Spec: "Death consumes the Sigil and destroys
        the instance; the node returns to reachable, not cleared" -- which is
        what ``FAILED`` is: openable again, at the price of another Sigil."""
        return self._end_instance(profile_id, Event.DIED)

    def report_abandon(self, profile_id: str) -> NodeState:
        """The player left the instance.  Spec: "Abandoning ... behave[s] like death"."""
        return self._end_instance(profile_id, Event.ABANDON)

    def report_timeout(self, profile_id: str) -> NodeState:
        """The 60 s reconnect window closed (section 07); behaves like death."""
        return self._end_instance(profile_id, Event.TIMEOUT)

    def _end_instance(self, profile_id: str, event: Event) -> NodeState:
        instance = self._live_instance(profile_id)
        return self._apply(instance.node_id, event, sigil_id=instance.sigil.id)

    @staticmethod
    def _threshold_met(instance: Instance) -> bool:
        """The bossless completion rule, in one place."""
        return not instance.has_boss and instance.elite_fraction() >= ELITE_CLEAR_FRACTION

    def _live_instance(self, profile_id: str) -> Instance:
        """The live instance, checked against the ledger before it is played on.

        The instance is stored beside the ledger, so the two are compared
        here on every report: its node is ``ACTIVE``; the last entry naming
        that node is the ``OPEN`` that created it, with the same Sigil id,
        tick and probe facts; its map seed is its Sigil's seed; its kill
        count is not negative; and, on a bossless map, the threshold has not
        already been met (the engine clears synchronously, so a live
        instance past its threshold cannot come from play).
        """
        self._require_profile(profile_id)
        instance = self.state.instance
        if instance is None:
            raise NoActiveInstance("no instance is active")
        node_id = instance.node_id
        if node_id not in self._nodes:
            raise CorruptInstance(f"instance names node {node_id}, which is not in the web")
        if self.state.states[node_id] is not NodeState.ACTIVE:
            raise CorruptInstance(
                f"instance is live on node {node_id} but the node is {self.state.states[node_id].value}"
            )
        opened = next((e for e in reversed(self.state.history) if e.node_id == node_id), None)
        if opened is None or opened.event is not Event.OPEN:
            raise CorruptInstance(f"the ledger has no open of node {node_id} as its last event there")
        if opened.sigil_id != instance.sigil.id:
            raise CorruptInstance(
                f"instance is funded by sigil {instance.sigil.id!r}, "
                f"the ledger's open (entry {opened.seq}) by {opened.sigil_id!r}"
            )
        if opened.tick != instance.opened_tick:
            raise CorruptInstance(
                f"instance was opened at tick {instance.opened_tick}, the ledger says {opened.tick}"
            )
        if opened.has_boss is None or opened.elite_total is None:
            raise CorruptInstance(f"the ledger's open (entry {opened.seq}) records no map facts")
        if opened.has_boss != instance.has_boss or opened.elite_total != instance.elite_total:
            raise CorruptInstance(
                f"instance says boss {instance.has_boss}, {instance.elite_total} elite packs; "
                f"the ledger's open (entry {opened.seq}) says boss {opened.has_boss}, "
                f"{opened.elite_total} elite packs"
            )
        if instance.map_seed != instance.sigil.seed:
            raise CorruptInstance("instance map seed is not its sigil's seed")
        if instance.elite_killed < 0:
            raise CorruptInstance(f"instance records {instance.elite_killed} elite kills")
        if self._threshold_met(instance):
            raise CorruptInstance(
                f"instance on a bossless map has {instance.elite_killed}/{instance.elite_total} "
                "elite kills, past the threshold, yet the node is not cleared"
            )
        return instance

    # ------------------------------------------------------------------
    # The single choke point for state changes
    # ------------------------------------------------------------------

    def _apply(
        self,
        node_id: int,
        event: Event,
        *,
        sigil_id: Optional[str] = None,
        tick: Optional[int] = None,
        has_boss: Optional[bool] = None,
        elite_total: Optional[int] = None,
    ) -> NodeState:
        """Apply ``event`` to ``node_id`` and do what entering the new state means.

        The move itself comes from :func:`contracts.next_state`, so an illegal
        pair raises :class:`IllegalTransition` before anything is written.
        Then, in order:

        * the state is written and a ledger entry appended (``seq`` is
          ``len(history) + 1``, so the ledger is dense and 1-based);
        * on any Sigil-consuming event the instance is destroyed -- the Sigil
          left the stash when the portal opened, so it is now spent;
        * on entering ``CLEARED`` the node's rewards are granted and
          ``NEIGHBOUR_CLEARED`` is propagated to each ``LOCKED`` neighbour,
          and only to those: a neighbour that is already reachable, active,
          cleared or failed is never touched (spec: "A node is reachable only
          if it shares an edge with a cleared node"; edges are never removed,
          so reachability, once earned, is permanent).

        ``tick`` defaults to the clock; propagation passes the clearing entry's
        tick down so one clear is one moment in the ledger.  ``has_boss`` and
        ``elite_total`` are the probe's facts and travel only on ``OPEN``.
        """
        before = self.state.states[node_id]
        after = next_state(before, event)
        if tick is None:
            tick = self._tick_source()
        entry = LedgerEntry(
            seq=len(self.state.history) + 1,
            node_id=node_id,
            event=event,
            before=before,
            after=after,
            sigil_id=sigil_id,
            tick=tick,
            has_boss=has_boss,
            elite_total=elite_total,
        )
        self._commit(entry)
        if event in SIGIL_CONSUMING_EVENTS:
            self.state.instance = None
        if after is NodeState.CLEARED:
            self._grant_clear_rewards(self._nodes[node_id])
            self._propagate(node_id, tick)
        return after

    def _commit(self, entry: LedgerEntry) -> None:
        """Write the state and append the entry; the only writer of both."""
        self.state.states[entry.node_id] = entry.after
        self.state.history.append(entry)
        if entry.sigil_id is not None:
            self._spent.add(entry.sigil_id)

    def _propagation_targets(self, cleared_id: int) -> List[int]:
        """The ``LOCKED`` neighbours a clear of ``cleared_id`` makes reachable, ascending."""
        return [
            neighbour_id
            for neighbour_id in self._adjacency[cleared_id]
            if self.state.states[neighbour_id] is NodeState.LOCKED
        ]

    def _propagate(self, cleared_id: int, tick: int) -> None:
        """Make each ``LOCKED`` neighbour of a cleared node ``REACHABLE``.

        The adjacency is the construction-time snapshot and its lists are
        sorted, so the order of the resulting ledger entries is a function of
        the web alone.
        """
        for neighbour_id in self._propagation_targets(cleared_id):
            self._apply(neighbour_id, Event.NEIGHBOUR_CLEARED, tick=tick)

    def _grant_clear_rewards(self, node: WebNode) -> None:
        """What clearing a node earns, granted exactly once because ``CLEARED``
        has no exit in the transition table.

        Spec: "Clearing a mechanic node grants one Descent passive point" and
        "Each arena is unlocked by collecting fragments from three cleared
        tier-15 nodes bearing that Pinnacle's glyph."
        """
        if node.mechanic is not None:
            self.state.passive_points += 1
        if node.glyph is not None:
            count = self.state.fragments.get(node.glyph, 0) + 1
            self.state.fragments[node.glyph] = count
            if count >= FRAGMENTS_TO_UNLOCK:
                self.state.unlocked_pinnacles = self.state.unlocked_pinnacles | {node.glyph}

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def _require_profile(self, profile_id: str) -> None:
        if profile_id != self.state.profile_id:
            raise WrongProfile(
                f"profile {profile_id!r} offered to the table of {self.state.profile_id!r}"
            )

    def _node(self, node_id: int) -> WebNode:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise UnknownNode(f"no node {node_id} in this web") from None

    def _node_openable(self, node: WebNode) -> bool:
        """The node-side half of :meth:`open_portal`'s checks, as a predicate."""
        if self.state.states[node.id] not in (NodeState.REACHABLE, NodeState.FAILED):
            return False
        if node.pinnacle is not None and node.pinnacle not in self.state.unlocked_pinnacles:
            return False
        return True
