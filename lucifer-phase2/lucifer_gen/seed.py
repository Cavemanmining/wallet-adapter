"""Seed field extraction and per-stage deterministic randomness.

Spec: docs/WORLD_BIBLE.md section 02, "Seed variation".

A map seed is one 64-bit value whose bits are assigned to stages so that
changing a field never disturbs the others:

    bits  0-1   rotation of the macro shape (0, 90, 180, 270)
    bit   2     mirror across the shape's primary axis
    bit   3     swap entrance and exit, where the shape allows it
    bits  4-7   which optional set piece is present (up to 16)
    bits  8-31  node jitter and edge routing        (stage 2)
    bits 32-63  tile selection, filler, spawns      (stages 4 and 6)

Each stage draws from its own stream, built by mixing its field value with a
stage label. Two stages therefore never share a sequence, and re-running one
stage cannot shift another's draws.

Which field a label draws from
------------------------------
:meth:`SeedFields.stream` routes by label prefix, and the prefixes are the
contract:

    ``setpiece*``            the set-piece field, bits 4-7
    ``route*``, ``place*``   the routing field, bits 8-31
    ``tile*``, ``spawn*``, ``fill*``   the tile field, bits 32-63

The ``setpiece`` prefix carries the line above about bits 4-7 -- an optional
set piece is present exactly when stage 2 keeps the optional edge that reaches
it, so that coin flip is what the field has to fund.  It used to be labelled
``route.optional:...`` and therefore drawn from bits 8-31, which made the
documented mapping false in both directions: bits 4-7 decided nothing that
survived into a map (their one consumer's output was discarded), while the
routing field silently owned optional set-piece presence.  A designer asking
for "the same layout with and without the side room" got a full reroute of
every corridor instead.  Each optional edge still has its own labelled stream,
so flipping one branch on or off leaves every other edge's draws untouched.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")

MASK64 = (1 << 64) - 1

ROTATION_BITS = (0, 2)
MIRROR_BIT = 2
SWAP_BIT = 3
SET_PIECE_BITS = (4, 4)
ROUTING_BITS = (8, 24)
TILE_BITS = (32, 32)


def _field(seed: int, offset: int, width: int) -> int:
    return (seed >> offset) & ((1 << width) - 1)


@dataclass(frozen=True)
class SeedFields:
    """The decoded fields of a map seed."""

    raw: int
    rotation: int
    mirror: bool
    swap_ends: bool
    set_piece_choice: int
    routing: int
    tiles: int

    @staticmethod
    def parse(seed: int) -> "SeedFields":
        seed &= MASK64
        return SeedFields(
            raw=seed,
            rotation=_field(seed, *ROTATION_BITS),
            mirror=bool((seed >> MIRROR_BIT) & 1),
            swap_ends=bool((seed >> SWAP_BIT) & 1),
            set_piece_choice=_field(seed, *SET_PIECE_BITS),
            routing=_field(seed, *ROUTING_BITS),
            tiles=_field(seed, *TILE_BITS),
        )

    def stream(self, label: str) -> "Stream":
        """A named random stream for one stage.

        ``label`` picks which field feeds the stream, so a change to routing
        bits cannot perturb tile selection or vice versa.  The prefixes are
        listed in the module docstring; anything unrecognised falls back to
        the whole seed, which is deliberate -- an unlabelled draw is coupled
        to everything rather than quietly sharing another stage's field.
        """
        if label.startswith("setpiece"):
            field_value = self.set_piece_choice
        elif label.startswith("route") or label.startswith("place"):
            field_value = self.routing
        elif label.startswith("tile") or label.startswith("spawn") or label.startswith("fill"):
            field_value = self.tiles
        else:
            field_value = self.raw
        return Stream(field_value, label)


class Stream:
    """A reproducible random stream seeded from one field plus a label."""

    __slots__ = ("_rng", "label")

    def __init__(self, field_value: int, label: str) -> None:
        self.label = label
        # Only the field and the label feed the stream. Mixing in the whole
        # seed would couple the stages back together and break the guarantee
        # that editing one field leaves the others' output identical.
        digest = hashlib.blake2b(
            f"{field_value}:{label}".encode("utf-8"), digest_size=32
        ).digest()
        self._rng = random.Random(int.from_bytes(digest, "big"))

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive on both ends."""
        return self._rng.randint(lo, hi)

    def random(self) -> float:
        return self._rng.random()

    def chance(self, p: float) -> bool:
        return self._rng.random() < p

    def choice(self, items: Sequence[T]) -> T:
        if not items:
            raise ValueError(f"stream {self.label!r} asked to choose from nothing")
        return items[self._rng.randrange(len(items))]

    def weighted_choice(self, items: Sequence[T], weights: Sequence[int]) -> T:
        if not items:
            raise ValueError(f"stream {self.label!r} asked to choose from nothing")
        total = sum(weights)
        if total <= 0:
            return self.choice(items)
        roll = self._rng.randrange(total)
        upto = 0
        for item, w in zip(items, weights):
            upto += w
            if roll < upto:
                return item
        return items[-1]

    def shuffled(self, items: Iterable[T]) -> List[T]:
        out = list(items)
        self._rng.shuffle(out)
        return out


def parse_seed(value) -> int:
    """Accept an int, or a string in hex (0x...) or decimal form."""
    if isinstance(value, int):
        return value & MASK64
    text = str(value).strip().lower()
    if text.startswith("0x"):
        return int(text, 16) & MASK64
    return int(text) & MASK64


def format_seed(seed: int) -> str:
    return f"0x{seed & MASK64:016X}"


__all__ = ["SeedFields", "Stream", "parse_seed", "format_seed", "MASK64"]
