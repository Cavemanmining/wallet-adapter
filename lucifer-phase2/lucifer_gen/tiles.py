"""The greybox tile database and the tile matcher used by stage 4.

Spec: docs/WORLD_BIBLE.md stage 4, "Tileization".

Stage 4 walks the grid, works out from a cell's neighbours what that cell has
to present on each of its four sides, and asks this module for a tile that
presents it.  The answer is a :class:`~lucifer_gen.contracts.Placement`: a tile
id plus the rotation and flip that make it fit.

What "fits" means
-----------------
``find`` takes ``required``, four entries in N, E, S, W order.  ``None`` means
that side is unconstrained.  A :class:`SideSpec` means *the tile must present
this*, judged by :func:`side_satisfies`: the signature must match exactly, and
for an open side the tile's connection slots must cover every slot asked for.

Note the direction of that rule.  ``required`` describes **this cell's own
side**, not the neighbour's.  Stage 4 usually knows the neighbour first, so
:func:`facing` converts a neighbour's side into the requirement it imposes
here, using the complement rules in ``contracts``.  Doing the complement once,
at the caller, keeps the matcher a plain equality test and keeps the two
readings of "required" from being silently confused.

Coverage
--------
The database is total over the requests each class can actually receive, which
is the property stage 4 leans on.  What that means differs by class, and the
difference is not a weakening -- it is the honest statement:

* ``OUTDOOR`` and ``BOTH`` must serve **every** combination of the five edge
  signatures across four sides, with any slot subset (:data:`FULL_ALPHABET`).
  The terrain family carries one tile per rotation/flip orbit of that
  alphabet: 120 tiles that between them reach all 625 combinations once the
  matcher is allowed to turn and mirror them.
* ``DUNGEON`` must serve the OPEN/WALL alphabet (:data:`DUNGEON_ALPHABET`)
  and nothing else, because ``tileize.surface_of`` degrades ``CLIFF`` and
  ``WATER`` cells to ``VOID`` for a dungeon plan -- a dungeon is never asked
  for an escarpment or a shoreline.  The dungeon family lists all sixteen
  OPEN/WALL combinations explicitly, as the spec asks, rather than leaning on
  rotation.

That split is why the terrain family is ``"class": "outdoor"`` rather than
``"both"``.  Labelling it ``"both"`` did make ``prove_coverage(DUNGEON)`` pass
over the *full* alphabet -- but it paid for the claim with wrong geometry: the
terrain tiles outnumber the dungeon tiles, so roughly one matched cell in
seven of every underground map was built out of an outdoor mesh, and both
terrain hero tiles ("open ground around a leaning monolith", "shoreline with a
half-drowned idol") were eligible underground while ``dungeon_end_n_hero``
went unused.  Nothing downstream could see it: the validator judges cells by
kind and signature, and the renderer colours by ``CellKind``, so a dungeon
made of hillside shipped through a green gate.  The stronger-sounding claim
and the wrong meshes were the same fact.

Every open side in the database exposes all three connection slots, so a
request naming any subset of slots is satisfiable too.
:func:`prove_coverage` enumerates a request space and checks it;
:func:`prove_class_coverage` picks the space that class must cover.

Randomness comes only from a :class:`~lucifer_gen.seed.Stream`; ``find`` draws
exactly twice per call (a weighted pick of tile, then a uniform pick among that
tile's fitting orientations), so a caller can reason about stream position.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from .contracts import (
    ALL_SLOTS,
    NO_SLOTS,
    SIDE_NAMES,
    SIDES,
    SIG_COMPLEMENT,
    EdgeSig,
    Placement,
    SideSpec,
    Tile,
    TileClass,
    _reverse_slots,
)
from .seed import SeedFields, Stream

#: Where the checked-in greybox database lives.
DEFAULT_TILE_DATA = Path(__file__).resolve().parent / "data" / "tiles_greybox.json"

#: Spec stage 4: "a hero variant appears at most once per 20 cells".
HERO_PERIOD = 20


class NoFittingTile(LookupError):
    """Raised when no tile in the database can satisfy a request.

    Given the coverage guarantee this should only ever mean the request itself
    is malformed, so the message spells the request out.
    """


# --------------------------------------------------------------------------
# Matching rules
# --------------------------------------------------------------------------


def side_satisfies(have: SideSpec, want: SideSpec) -> bool:
    """True when a tile side ``have`` meets the requirement ``want``.

    Signatures must be equal, not complementary: ``want`` already describes
    this tile's own side.  For an open side the tile must offer at least the
    slots asked for, so a request is only ever narrowed by asking for more.
    ``want.slots == 0`` therefore reads as "open, slots don't matter".

    Note which of the two roles a ``SideSpec`` is playing.  As a *request*,
    ``OPEN`` with no slots is the wildcard just described.  As a side a tile
    actually *presents* it is a contradiction -- an open edge nothing can ever
    join -- which is why :meth:`TileDatabase` refuses to load one and why
    :func:`facing` refuses to turn one into a requirement.
    """
    if have.sig is not want.sig:
        return False
    if want.sig is EdgeSig.OPEN:
        return (have.slots & want.slots) == want.slots
    return True


def facing(neighbour_side: SideSpec) -> SideSpec:
    """The requirement a neighbour's side imposes on the side facing it.

    The signature flips to its complement (a rise meets a fall) and the slot
    mask is read back to front, because the two sides run along the shared
    edge in opposite directions.  Feed the result straight into ``find``: any
    placement that satisfies the result is guaranteed to pass
    ``contracts.sides_compatible`` against ``neighbour_side``.

    That guarantee is the reason for the one rejection below.  An open
    neighbour side with no connection slots offers nothing to join, so *no*
    tile can sit beside it -- but the naive translation of it is
    ``SideSpec(OPEN, 0)``, which ``side_satisfies`` reads as the wildcard
    "open, slots don't matter" and every open tile meets.  ``find`` would
    hand back a placement, and the seam would then fail
    ``sides_compatible``: the requirement would have promised something it
    cannot deliver.  There is no ``SideSpec`` that means "unsatisfiable", so
    say so instead of lying quietly.  (The shipped database can never produce
    such a side; ``TileDatabase`` rejects one at load.)
    """
    if neighbour_side.sig is EdgeSig.OPEN and not neighbour_side.slots:
        raise ValueError(
            "an OPEN side offering no connection slot can never be joined, so "
            "it imposes no requirement a tile could satisfy"
        )
    sig = SIG_COMPLEMENT[neighbour_side.sig]
    if sig is EdgeSig.OPEN:
        return SideSpec(sig, _reverse_slots(neighbour_side.slots))
    return SideSpec(sig, NO_SLOTS)


def open_side(slots: int = ALL_SLOTS) -> SideSpec:
    """Shorthand for an open side with the given connection slots."""
    return SideSpec(EdgeSig.OPEN, slots)


def wall_side() -> SideSpec:
    """Shorthand for a closed side."""
    return SideSpec(EdgeSig.WALL, NO_SLOTS)


def describe_request(required: Sequence[Optional[SideSpec]]) -> str:
    """Render a request the way the error messages and tests want to read it."""
    parts = []
    for side, want in zip(SIDES, required):
        if want is None:
            parts.append(f"{SIDE_NAMES[side]}=*")
        elif want.sig is EdgeSig.OPEN:
            parts.append(f"{SIDE_NAMES[side]}=OPEN:{want.slots:03b}")
        else:
            parts.append(f"{SIDE_NAMES[side]}={want.sig.name}")
    return " ".join(parts)


# --------------------------------------------------------------------------
# Orientations
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Orientation:
    """One tile turned and mirrored a particular way, with its sides resolved."""

    tile: Tile
    rot: int
    flip: bool
    sides: Tuple[SideSpec, SideSpec, SideSpec, SideSpec]

    def placement(self) -> Placement:
        return Placement(self.tile.id, self.rot, self.flip)

    def fits(self, required: Sequence[Optional[SideSpec]]) -> bool:
        return all(
            want is None or side_satisfies(self.sides[side], want)
            for side, want in zip(SIDES, required)
        )


# --------------------------------------------------------------------------
# The database
# --------------------------------------------------------------------------

_SIG_BY_NAME = {sig.name: sig for sig in EdgeSig}
_CLASS_BY_NAME = {cls.value: cls for cls in TileClass}


def _parse_side(raw) -> SideSpec:
    """Read one side from JSON.

    A bare string is the common case and means "all slots if open, none
    otherwise"; an object may name ``slots`` explicitly.
    """
    if isinstance(raw, str):
        sig = _SIG_BY_NAME[raw.strip().upper()]
        return SideSpec(sig, ALL_SLOTS if sig is EdgeSig.OPEN else NO_SLOTS)
    sig = _SIG_BY_NAME[str(raw["sig"]).strip().upper()]
    default = ALL_SLOTS if sig is EdgeSig.OPEN else NO_SLOTS
    return SideSpec(sig, int(raw.get("slots", default)))


def _parse_tile(raw: dict) -> Tuple[Tile, Dict[str, object]]:
    sides = [_parse_side(s) for s in raw["sides"]]
    if len(sides) != 4:
        raise ValueError(f"tile {raw.get('id')!r} needs exactly four sides")
    tile = Tile(
        id=int(raw["id"]),
        name=str(raw["name"]),
        tile_class=_CLASS_BY_NAME[str(raw.get("class", "both"))],
        sides=(sides[0], sides[1], sides[2], sides[3]),
        walkable=bool(raw.get("walkable", True)),
        weight=int(raw.get("weight", 1)),
        hero=bool(raw.get("hero", False)),
        mesh=str(raw.get("mesh", "greybox/plain")),
    )
    known = {"id", "name", "class", "sides", "walkable", "weight", "hero", "mesh"}
    extra = {k: v for k, v in raw.items() if k not in known}
    return tile, extra


class TileDatabase:
    """The tiles stage 4 may place, indexed so matching is cheap.

    Every tile is expanded into its eight orientations up front, and each
    orientation is filed under the side spec it shows on each side.  A request
    is then a handful of set intersections, and the answer is cached, which
    matters because a 48x48 grid asks the same few hundred questions over and
    over.
    """

    def __init__(
        self,
        tiles: Iterable[Tile],
        *,
        name: str = "greybox",
        revision: int = 1,
        cell_m: float = 4.0,
        filler_tile_id: Optional[int] = None,
        hero_period: int = HERO_PERIOD,
        extras: Optional[Dict[int, Dict[str, object]]] = None,
    ) -> None:
        self.tiles: Tuple[Tile, ...] = tuple(sorted(tiles, key=lambda t: t.id))
        if not self.tiles:
            raise ValueError("a tile database needs at least one tile")
        self.name = name
        self.revision = int(revision)
        self.cell_m = float(cell_m)
        self.hero_period = int(hero_period)
        self.extras: Dict[int, Dict[str, object]] = dict(extras or {})

        self._by_id: Dict[int, Tile] = {}
        for tile in self.tiles:
            if tile.id in self._by_id:
                raise ValueError(f"duplicate tile id {tile.id}")
            if not 0 <= tile.id <= 0xFF:
                raise ValueError(f"tile id {tile.id} does not fit in one byte")
            for side, spec in zip(SIDES, tile.sides):
                if spec.sig is EdgeSig.OPEN and not spec.slots:
                    # Nothing could ever abut it: sides_compatible needs a
                    # shared slot between two open sides.  Catch it here
                    # rather than at the seam it would silently break.
                    raise ValueError(
                        f"tile {tile.name!r} side {SIDE_NAMES[side]} is OPEN "
                        "but offers no connection slot, so no tile could ever "
                        "sit beside it"
                    )
            self._by_id[tile.id] = tile

        if filler_tile_id is None:
            filler_tile_id = self._infer_filler()
        if filler_tile_id not in self._by_id:
            raise ValueError(f"filler tile {filler_tile_id} is not in the database")
        self.filler_tile_id = int(filler_tile_id)

        self._build_index()

    # -- construction ------------------------------------------------------

    def _infer_filler(self) -> int:
        """Fall back to the first unwalkable, fully walled tile."""
        for tile in self.tiles:
            if not tile.walkable and all(s.sig is EdgeSig.WALL for s in tile.sides):
                return tile.id
        raise ValueError("no filler tile: need one unwalkable tile walled on every side")

    def _build_index(self) -> None:
        orientations: List[Orientation] = []
        for tile in self.tiles:
            for flip in (False, True):
                for rot in (0, 1, 2, 3):
                    sides = tile.transformed(rot, flip)
                    orientations.append(Orientation(tile, rot, flip, tuple(sides)))
        self._orientations: Tuple[Orientation, ...] = tuple(orientations)

        everything = frozenset(range(len(self._orientations)))
        self._class_members: Dict[TileClass, FrozenSet[int]] = {}
        for cls in TileClass:
            if cls is TileClass.BOTH:
                members = everything
            else:
                members = frozenset(
                    i
                    for i, o in enumerate(self._orientations)
                    if o.tile.tile_class in (cls, TileClass.BOTH)
                )
            self._class_members[cls] = members
        self._non_hero: FrozenSet[int] = frozenset(
            i for i, o in enumerate(self._orientations) if not o.tile.hero
        )

        # Which distinct side specs actually occur, per side. There are only a
        # handful, so resolving a request against them is trivial work.
        self._side_specs: List[Tuple[SideSpec, ...]] = []
        for side in SIDES:
            seen = {o.sides[side] for o in self._orientations}
            self._side_specs.append(
                tuple(sorted(seen, key=lambda s: (int(s.sig), s.slots)))
            )
        self._side_buckets: List[Dict[SideSpec, FrozenSet[int]]] = []
        for side in SIDES:
            bucket: Dict[SideSpec, List[int]] = {}
            for i, o in enumerate(self._orientations):
                bucket.setdefault(o.sides[side], []).append(i)
            self._side_buckets.append({k: frozenset(v) for k, v in bucket.items()})

        self._satisfy_cache: Dict[Tuple[int, SideSpec], FrozenSet[int]] = {}
        self._group_cache: Dict[
            Tuple[Tuple[Optional[SideSpec], ...], TileClass, bool],
            Tuple[Tuple[Tile, Tuple[Orientation, ...]], ...],
        ] = {}

    @staticmethod
    def load(path: Optional[Path] = None) -> "TileDatabase":
        """Load a database from JSON; defaults to the checked-in greybox set."""
        path = Path(path) if path is not None else DEFAULT_TILE_DATA
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        parsed = [_parse_tile(raw) for raw in doc["tiles"]]
        return TileDatabase(
            [t for t, _ in parsed],
            name=str(doc.get("id", "greybox")),
            revision=int(doc.get("version", 1)),
            cell_m=float(doc.get("cell_m", 4.0)),
            filler_tile_id=doc.get("filler_tile_id"),
            hero_period=int(doc.get("hero_period", HERO_PERIOD)),
            extras={t.id: extra for t, extra in parsed if extra},
        )

    # -- identity ----------------------------------------------------------

    @property
    def version(self) -> str:
        """The reference stage 6 hashes into ``layout_hash``, e.g. ``greybox@1``."""
        return f"{self.name}@{self.revision}"

    #: ``GraphTemplate`` spells the same idea ``ref``; accept either name.
    @property
    def ref(self) -> str:
        return self.version

    def content_digest(self) -> str:
        """A digest of the tile data itself, for spotting silent edits.

        Diagnostic only -- the stable identity of this database is
        :attr:`version`, and that is what ``layout_hash`` folds in.  blake2b
        always, never the optional blake3 wheel: a digest two machines compute
        differently for identical data cannot spot anything.
        """
        parts = [self.version]
        for tile in self.tiles:
            sides = ",".join(f"{s.sig.name}:{s.slots}" for s in tile.sides)
            parts.append(
                f"{tile.id}|{tile.name}|{tile.tile_class.value}|{sides}|"
                f"{int(tile.walkable)}|{tile.weight}|{int(tile.hero)}|{tile.mesh}"
            )
        data = "\n".join(parts).encode("utf-8")
        return "blake2b:" + hashlib.blake2b(data, digest_size=32).hexdigest()

    # -- lookup ------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.tiles)

    def by_id(self, tile_id: int) -> Tile:
        return self._by_id[tile_id]

    @property
    def filler(self) -> Tile:
        """The impassable filler tile stage 4 drops on untouched cells."""
        return self._by_id[self.filler_tile_id]

    def filler_placement(self) -> Placement:
        """Filler is symmetric, so it needs no rotation and no flip."""
        return Placement(self.filler_tile_id, 0, False)

    def meta(self, tile_id: int) -> Dict[str, object]:
        """Extra JSON fields for a tile, such as the filler's collision data."""
        return dict(self.extras.get(tile_id, {}))

    def heroes(self) -> Tuple[Tile, ...]:
        return tuple(t for t in self.tiles if t.hero)

    # -- matching ----------------------------------------------------------

    def _satisfying(self, side: int, want: SideSpec) -> FrozenSet[int]:
        key = (side, want)
        hit = self._satisfy_cache.get(key)
        if hit is None:
            buckets = self._side_buckets[side]
            matched: FrozenSet[int] = frozenset()
            for spec in self._side_specs[side]:
                if side_satisfies(spec, want):
                    matched |= buckets[spec]
            self._satisfy_cache[key] = matched
            hit = matched
        return hit

    @staticmethod
    def _normalise(required: Sequence[Optional[SideSpec]]) -> Tuple[Optional[SideSpec], ...]:
        want = tuple(required)
        if len(want) != 4:
            raise ValueError(
                f"a request needs four sides in N, E, S, W order, got {len(want)}"
            )
        for entry in want:
            if entry is not None and not isinstance(entry, SideSpec):
                raise TypeError(f"required entries must be SideSpec or None, got {entry!r}")
        return want

    def grouped_candidates(
        self,
        required: Sequence[Optional[SideSpec]],
        tile_class: TileClass,
        allow_hero: bool = False,
    ) -> Tuple[Tuple[Tile, Tuple[Orientation, ...]], ...]:
        """Fitting orientations, grouped by tile and ordered by tile id.

        Grouping is what keeps the weighting honest: a tile that happens to be
        symmetric fits in fewer distinct orientations than an asymmetric one,
        and weighting per orientation would quietly punish it for that.
        """
        want = self._normalise(required)
        key = (want, tile_class, bool(allow_hero))
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached

        pools: List[FrozenSet[int]] = [self._class_members[tile_class]]
        if not allow_hero:
            pools.append(self._non_hero)
        for side, entry in zip(SIDES, want):
            if entry is not None:
                pools.append(self._satisfying(side, entry))
        # Smallest first: the cheapest way to collapse the search early.
        pools.sort(key=len)
        pool = pools[0]
        for other in pools[1:]:
            pool &= other
            if not pool:
                break

        grouped: Dict[int, List[Orientation]] = {}
        for index in sorted(pool):
            orientation = self._orientations[index]
            grouped.setdefault(orientation.tile.id, []).append(orientation)
        result = tuple(
            (self._by_id[tile_id], tuple(orientations))
            for tile_id, orientations in sorted(grouped.items())
        )
        self._group_cache[key] = result
        return result

    def candidates(
        self,
        required: Sequence[Optional[SideSpec]],
        tile_class: TileClass,
        allow_hero: bool = False,
    ) -> List[Orientation]:
        """Every fitting orientation, flattened. Mostly useful to tests."""
        out: List[Orientation] = []
        for _tile, orientations in self.grouped_candidates(required, tile_class, allow_hero):
            out.extend(orientations)
        return out

    def find_orientation(
        self,
        required: Sequence[Optional[SideSpec]],
        tile_class: TileClass,
        stream: Stream,
        allow_hero: bool = False,
    ) -> Orientation:
        """Pick a fitting orientation: weighted over tiles, uniform over turns.

        Exactly two draws come off ``stream``, whatever the request, so a
        caller that needs to reason about stream position can.
        """
        groups = self.grouped_candidates(required, tile_class, allow_hero)
        if not groups:
            raise NoFittingTile(
                f"no {tile_class.value} tile fits {describe_request(required)}"
                f" (hero variants {'allowed' if allow_hero else 'excluded'})"
            )
        tiles = [tile for tile, _ in groups]
        weights = [max(0, tile.weight) for tile in tiles]
        chosen = stream.weighted_choice(tiles, weights)
        options = next(orients for tile, orients in groups if tile.id == chosen.id)
        return stream.choice(options)

    def find(
        self,
        required: Sequence[Optional[SideSpec]],
        tile_class: TileClass,
        stream: Stream,
        allow_hero: bool = False,
    ) -> Placement:
        """Stage 4's entry point: a tile plus the rotation and flip that fit."""
        return self.find_orientation(required, tile_class, stream, allow_hero).placement()

    def sides_of(self, placement: Placement) -> Tuple[SideSpec, ...]:
        """The four sides a placement actually presents, after transforming."""
        return self._by_id[placement.tile_id].transformed(placement.rot, placement.flip)

    def placement_fits(
        self, placement: Placement, required: Sequence[Optional[SideSpec]]
    ) -> bool:
        """Check a placement against a request; the tests lean on this."""
        sides = self.sides_of(placement)
        return all(
            want is None or side_satisfies(sides[side], want)
            for side, want in zip(SIDES, self._normalise(required))
        )

    def is_walkable(self, placement: Placement) -> bool:
        return self._by_id[placement.tile_id].walkable


# --------------------------------------------------------------------------
# The hero budget (stage 4, "at most once per 20 cells")
# --------------------------------------------------------------------------


class HeroBudget:
    """Rations hero variants across a tileization pass.

    Two rules, both enforced, because "at most once per 20 cells" can be read
    either as a density or as a spacing and neither reading should be
    violated: the running count of hero cells never exceeds cells/period, and
    two hero cells are never fewer than ``period`` cells apart.  Feed
    :meth:`allows_hero` to ``find``'s ``allow_hero`` and call :meth:`record`
    once per emitted cell.
    """

    __slots__ = ("period", "cells", "heroes", "_last_hero_index")

    def __init__(self, period: int = HERO_PERIOD) -> None:
        if period < 1:
            raise ValueError("hero period must be at least one cell")
        self.period = int(period)
        self.cells = 0
        self.heroes = 0
        self._last_hero_index: Optional[int] = None

    def allows_hero(self) -> bool:
        """May the cell about to be emitted be a hero variant?"""
        index = self.cells  # zero-based index of the cell being decided
        if self._last_hero_index is not None:
            if index - self._last_hero_index < self.period:
                return False
        return (self.heroes + 1) * self.period <= index + 1

    def record(self, hero: bool) -> None:
        """Note one emitted cell and whether it turned out to be a hero.

        Recording a hero the budget did not allow is permitted -- stage 5 set
        pieces overwrite tiles without asking -- but it will delay the next
        one, so the rate stays honest over the whole pass.
        """
        if hero:
            self.heroes += 1
            self._last_hero_index = self.cells
        self.cells += 1

    def record_placement(self, db: TileDatabase, placement: Placement) -> None:
        """Convenience for callers holding a ``Placement`` rather than a flag."""
        self.record(db.by_id(placement.tile_id).hero)

    def reset(self) -> None:
        self.cells = 0
        self.heroes = 0
        self._last_hero_index = None

    @property
    def within_budget(self) -> bool:
        """The invariant callers assert: never more than cells/period heroes."""
        return self.heroes * self.period <= self.cells

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"HeroBudget(period={self.period}, cells={self.cells}, "
            f"heroes={self.heroes})"
        )


# --------------------------------------------------------------------------
# Coverage proof
# --------------------------------------------------------------------------

#: The five signatures as requests, each with every slot a tile could offer.
FULL_ALPHABET: Tuple[Optional[SideSpec], ...] = (None,) + tuple(
    SideSpec(sig, ALL_SLOTS if sig is EdgeSig.OPEN else NO_SLOTS) for sig in EdgeSig
)

#: What a dungeon alone can be asked for, per the spec's stage 4 notes.  It is
#: also everything a dungeon *can* be asked for: ``tileize.surface_of`` turns
#: a dungeon plan's CLIFF and WATER cells into VOID, and void is walled.
DUNGEON_ALPHABET: Tuple[Optional[SideSpec], ...] = (
    None,
    SideSpec(EdgeSig.OPEN, ALL_SLOTS),
    SideSpec(EdgeSig.WALL, NO_SLOTS),
)

#: The request space each class must be total over.  See the module docstring:
#: a dungeon is never asked for an escarpment, so requiring it to answer one
#: buys nothing and costs outdoor meshes underground.
COVERAGE_ALPHABET: Dict[TileClass, Tuple[Optional[SideSpec], ...]] = {
    TileClass.DUNGEON: DUNGEON_ALPHABET,
    TileClass.OUTDOOR: FULL_ALPHABET,
    TileClass.BOTH: FULL_ALPHABET,
}


def prove_coverage(
    db: TileDatabase,
    tile_class: TileClass,
    *,
    seed: int = 0x0123456789ABCDEF,
    alphabet: Sequence[Optional[SideSpec]] = FULL_ALPHABET,
    slot_sweep: bool = True,
    allow_hero: bool = False,
) -> int:
    """Enumerate every request the matcher can be asked for and prove it fits.

    Two sweeps.  The first crosses ``alphabet`` over all four sides, so every
    combination of signatures -- and every way of leaving sides unconstrained
    -- is tried.  The second opens all four sides and walks every connection
    slot mask, which is the only other axis a request varies on.  Each answer
    is re-checked by transforming the chosen tile and comparing, so this
    proves the matcher as well as the data.

    Returns the number of requests proved; raises :class:`NoFittingTile` on the
    first request the database cannot serve, and ``AssertionError`` if a
    returned placement does not actually fit.
    """
    stream = SeedFields.parse(seed).stream("tile-coverage")
    proved = 0

    for combo in itertools.product(alphabet, repeat=4):
        placement = db.find(combo, tile_class, stream, allow_hero)
        if not db.placement_fits(placement, combo):
            raise AssertionError(
                f"{db.by_id(placement.tile_id).name} rot={placement.rot} "
                f"flip={placement.flip} does not fit {describe_request(combo)}"
            )
        proved += 1

    if slot_sweep:
        for masks in itertools.product(range(ALL_SLOTS + 1), repeat=4):
            combo = tuple(SideSpec(EdgeSig.OPEN, mask) for mask in masks)
            placement = db.find(combo, tile_class, stream, allow_hero)
            if not db.placement_fits(placement, combo):
                raise AssertionError(
                    f"{db.by_id(placement.tile_id).name} rot={placement.rot} "
                    f"flip={placement.flip} does not fit {describe_request(combo)}"
                )
            proved += 1

    return proved


def prove_class_coverage(
    db: TileDatabase, tile_class: TileClass, **kwargs
) -> int:
    """Prove the totality property ``tile_class`` actually has to hold.

    Thin wrapper over :func:`prove_coverage` that picks the request space out
    of :data:`COVERAGE_ALPHABET` instead of assuming every class must answer
    every signature.  This is the call a test or a gate should make: it states
    the property in terms of what stage 4 can ask for, so the claim cannot be
    propped up by labelling outdoor tiles as dungeon ones.
    """
    kwargs.setdefault("alphabet", COVERAGE_ALPHABET[tile_class])
    return prove_coverage(db, tile_class, **kwargs)


__all__ = [
    "COVERAGE_ALPHABET",
    "DEFAULT_TILE_DATA",
    "DUNGEON_ALPHABET",
    "FULL_ALPHABET",
    "HERO_PERIOD",
    "HeroBudget",
    "NoFittingTile",
    "Orientation",
    "TileDatabase",
    "describe_request",
    "facing",
    "open_side",
    "prove_class_coverage",
    "prove_coverage",
    "side_satisfies",
    "wall_side",
]
