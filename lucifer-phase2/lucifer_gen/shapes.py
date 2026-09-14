"""The six macro shapes and their anchor algebra.

Spec: docs/WORLD_BIBLE.md stage 2, "Shuffle shape and route edges".

A macro shape is a silhouette drawn in normalised ``[0, 1] x [0, 1]`` space.
It carries named anchor points; a graph template pins each of its nodes to one
of them (``"shape.bend"``, ``"shape.pocket"``, ...), and stage 2 resolves the
anchor to a grid cell for the seed at hand.

Exactness
---------
The seed transform -- rotation by 0/90/180/270 degrees clockwise, a mirror
across the vertical axis, and an entrance/exit swap -- is applied in normalised
space *before* scaling to the grid, so that a transform composed with its own
inverse is the identity down to the last bit.  To make "down to the last bit"
literally true the anchors are :class:`fractions.Fraction`, not floats:
``1 - (1 - Fraction(14, 100))`` is exactly ``Fraction(14, 100)`` while the same
round trip through binary floating point is not.

Order of operations
-------------------
Mirror first, then rotate.  This matches :meth:`contracts.Tile.transformed`,
which flips a tile before rotating it, so a tile and the shape it sits in agree
on what "mirrored and rotated" means.

Frames
------
Anchor names ("start", "spoke_n", ...) are labels in the shape's *own* frame,
before any transform.  After a 90 degree rotation the point named ``spoke_n``
is of course no longer to the north; the name still identifies the same part of
the silhouette, which is what a template author cares about.

Coordinates follow :mod:`contracts`: x runs east, y runs south, origin at the
top-left corner of the grid.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, Mapping, NamedTuple, Tuple

from .contracts import SWAPPABLE_SHAPES, Cell, Shape

__all__ = [
    "Point",
    "Axis",
    "MacroShape",
    "SHAPES",
    "ANCHOR_PREFIX",
    "REQUIRED_ANCHORS",
    "get_shape",
    "normalise_anchor_name",
    "has_anchor",
    "transform_point",
    "transform_axis",
    "point_to_cell",
    "anchor_point",
    "anchor_cell",
    "shape_axis",
]

#: Templates spell anchors as ``"shape.bend"``; this prefix is optional.
ANCHOR_PREFIX = "shape."

#: Every shape must define at least these, per the spec.
REQUIRED_ANCHORS = ("start", "bend", "pocket", "end", "centre")


class Point(NamedTuple):
    """A point in normalised shape space, held as exact rationals."""

    x: Fraction
    y: Fraction

    def as_floats(self) -> Tuple[float, float]:
        """For display and for callers that only need an approximation."""
        return (float(self.x), float(self.y))


def _pt(x_hundredths: int, y_hundredths: int) -> Point:
    """Build a point from hundredths, so ``_pt(14, 86)`` is (0.14, 0.86)."""
    return Point(Fraction(x_hundredths, 100), Fraction(y_hundredths, 100))


# --------------------------------------------------------------------------
# Primary axis
# --------------------------------------------------------------------------


class Axis(enum.Enum):
    """The direction a shape's long features run.

    Outdoor tilesets align ridges, walls and cliff lines to this axis, so it
    has to follow the shape through the seed transform.  An axis is undirected:
    the stored vector is the canonical representative of the pair
    ``{v, -v}`` (first non-zero component positive).
    """

    HORIZONTAL = (1, 0)
    VERTICAL = (0, 1)
    DIAGONAL_SE = (1, 1)  # north-west to south-east
    DIAGONAL_NE = (1, -1)  # south-west to north-east

    @property
    def vector(self) -> Tuple[int, int]:
        return self.value


def _canonical_axis(dx: int, dy: int) -> Axis:
    if dx < 0 or (dx == 0 and dy < 0):
        dx, dy = -dx, -dy
    return Axis((dx, dy))


def transform_axis(axis: Axis, rotation: int = 0, mirror: bool = False) -> Axis:
    """Carry an axis through the seed transform (mirror first, then rotate)."""
    dx, dy = axis.vector
    if mirror:
        dx = -dx
    for _ in range(rotation & 3):
        # A quarter turn clockwise in screen coordinates (y south).
        dx, dy = -dy, dx
    return _canonical_axis(dx, dy)


# --------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MacroShape:
    """One macro shape: its anchors, in its own frame, plus its primary axis."""

    shape: Shape
    anchors: Mapping[str, Point]
    primary_axis: Axis
    spokes: Tuple[str, ...] = ()

    def anchor(self, name: str) -> Point:
        """Look up an anchor, accepting the ``shape.`` prefix templates use."""
        key = normalise_anchor_name(name)
        try:
            return self.anchors[key]
        except KeyError:
            known = ", ".join(sorted(self.anchors))
            raise KeyError(
                f"shape {self.shape.value} has no anchor {name!r}; known: {known}"
            ) from None

    # The five anchors the spec requires of every shape, as attributes so that
    # ``shape.start`` reads the way the spec writes it.
    @property
    def start(self) -> Point:
        return self.anchors["start"]

    @property
    def bend(self) -> Point:
        return self.anchors["bend"]

    @property
    def pocket(self) -> Point:
        return self.anchors["pocket"]

    @property
    def end(self) -> Point:
        return self.anchors["end"]

    @property
    def centre(self) -> Point:
        return self.anchors["centre"]

    @property
    def swappable(self) -> bool:
        """True when seed bit 3 may trade this shape's entrance and exit."""
        return self.shape in SWAPPABLE_SHAPES

    def anchor_names(self) -> Tuple[str, ...]:
        """Sorted, so callers never iterate a mapping in insertion order."""
        return tuple(sorted(self.anchors))


# "U": two long legs joined by a bar along the south, mouth facing north.
_U = MacroShape(
    shape=Shape.U,
    primary_axis=Axis.VERTICAL,  # the legs are the long runs
    anchors={
        "start": _pt(14, 14),  # north tip of the west leg
        "bend": _pt(14, 86),  # south-west corner
        "bar": _pt(50, 86),  # middle of the southern bar
        "bend2": _pt(86, 86),  # south-east corner
        "pocket": _pt(50, 65),  # chamber hanging into the mouth
        "end": _pt(86, 14),  # north tip of the east leg
        "centre": _pt(50, 50),
    },
)

# "C": a back along the west with two arms reaching east, mouth facing east.
_C = MacroShape(
    shape=Shape.C,
    primary_axis=Axis.VERTICAL,  # the back is the long run
    anchors={
        "start": _pt(86, 20),  # east tip of the north arm
        "bend": _pt(20, 20),  # north-west corner
        "back": _pt(20, 50),  # middle of the western back
        "bend2": _pt(20, 80),  # south-west corner
        "pocket": _pt(65, 50),  # chamber in the mouth
        "end": _pt(86, 80),  # east tip of the south arm
        "centre": _pt(50, 50),
    },
)

# "I": one straight run from north to south with a slight kink.
_I = MacroShape(
    shape=Shape.I,
    primary_axis=Axis.VERTICAL,
    anchors={
        "start": _pt(50, 10),
        "bend": _pt(50, 35),
        "bend2": _pt(50, 65),
        "pocket": _pt(28, 50),  # chamber off the western flank
        "end": _pt(50, 90),
        "centre": _pt(50, 50),
    },
)

# "Diamond": four vertices, four diagonal runs.
_DIAMOND = MacroShape(
    shape=Shape.DIAMOND,
    # Every run of a diamond is diagonal; the two diagonals are equivalent
    # under the shape's own symmetry, so either one may stand for the axis.
    primary_axis=Axis.DIAGONAL_SE,
    anchors={
        "start": _pt(50, 6),  # north vertex
        "bend": _pt(94, 50),  # east vertex
        "pocket": _pt(6, 50),  # west vertex
        "end": _pt(50, 94),  # south vertex
        "centre": _pt(50, 50),
    },
)

# "Spiral": a clockwise inward coil starting at the north-west.
_SPIRAL = MacroShape(
    shape=Shape.SPIRAL,
    primary_axis=Axis.HORIZONTAL,  # the outermost run is the long one
    anchors={
        "start": _pt(10, 10),
        "bend": _pt(90, 10),
        "turn2": _pt(90, 90),
        "turn3": _pt(10, 90),
        "turn4": _pt(10, 35),
        "turn5": _pt(65, 35),
        "turn6": _pt(65, 65),
        "pocket": _pt(50, 20),  # trapped between the outer and second coil
        "end": _pt(40, 65),  # the innermost chamber
        "centre": _pt(50, 50),
    },
)

# "Hub": a central chamber with eight spokes.
_HUB = MacroShape(
    shape=Shape.HUB,
    # A hub has no long feature; it is four-fold symmetric, so the axis is
    # nominal and simply names the east-west spoke pair.
    primary_axis=Axis.HORIZONTAL,
    spokes=(
        "spoke_n",
        "spoke_ne",
        "spoke_e",
        "spoke_se",
        "spoke_s",
        "spoke_sw",
        "spoke_w",
        "spoke_nw",
    ),
    anchors={
        "centre": _pt(50, 50),
        "spoke_n": _pt(50, 10),
        "spoke_ne": _pt(78, 22),
        "spoke_e": _pt(90, 50),
        "spoke_se": _pt(78, 78),
        "spoke_s": _pt(50, 90),
        "spoke_sw": _pt(22, 78),
        "spoke_w": _pt(10, 50),
        "spoke_nw": _pt(22, 22),
        # The five names every shape owes the spec, aliased onto spokes.
        "start": _pt(50, 10),  # == spoke_n
        "bend": _pt(90, 50),  # == spoke_e
        "pocket": _pt(10, 50),  # == spoke_w
        "end": _pt(50, 90),  # == spoke_s
    },
)

#: Every macro shape, by its :class:`contracts.Shape`.
SHAPES: Dict[Shape, MacroShape] = {
    s.shape: s for s in (_U, _C, _I, _DIAMOND, _SPIRAL, _HUB)
}

for _shape_enum in Shape:
    if _shape_enum not in SHAPES:  # pragma: no cover - guards a typo above
        raise RuntimeError(f"no macro shape defined for {_shape_enum}")
for _macro in SHAPES.values():
    _missing = [a for a in REQUIRED_ANCHORS if a not in _macro.anchors]
    if _missing:  # pragma: no cover - guards a typo above
        raise RuntimeError(f"{_macro.shape} is missing anchors {_missing}")
del _shape_enum, _macro, _missing


def get_shape(shape: Shape) -> MacroShape:
    """The :class:`MacroShape` for a :class:`contracts.Shape`."""
    try:
        return SHAPES[shape]
    except KeyError:  # pragma: no cover - Shape is a closed enum
        raise KeyError(f"unknown shape {shape!r}") from None


def normalise_anchor_name(name: str) -> str:
    """Strip the optional ``shape.`` prefix and fold case."""
    key = str(name).strip().lower()
    if key.startswith(ANCHOR_PREFIX):
        key = key[len(ANCHOR_PREFIX):]
    return key


def has_anchor(shape: Shape, name: str) -> bool:
    """True when ``shape`` defines ``name`` (with or without the prefix)."""
    return normalise_anchor_name(name) in get_shape(shape).anchors


# --------------------------------------------------------------------------
# The seed transform
# --------------------------------------------------------------------------


def transform_point(p: Point, rotation: int = 0, mirror: bool = False) -> Point:
    """Mirror across the vertical axis, then rotate clockwise.

    ``rotation`` is in quarter turns (seed bits 0-1); ``mirror`` is seed bit 2.
    Both act about the centre of the unit square, in exact rational arithmetic,
    so ``transform_point(transform_point(p, 2), 2) == p`` exactly.
    """
    x, y = p.x, p.y
    if mirror:
        x = 1 - x
    for _ in range(rotation & 3):
        # One quarter turn clockwise about (1/2, 1/2): (x, y) -> (1 - y, x).
        x, y = 1 - y, x
    return Point(x, y)


def _swapped_name(shape: Shape, name: str, swap_ends: bool) -> str:
    """Apply seed bit 3, which trades ``start`` and ``end`` on U, C and I.

    Other shapes would become a different shape if their ends traded places,
    so the bit is ignored for them (see ``contracts.SWAPPABLE_SHAPES``).
    """
    if not swap_ends or shape not in SWAPPABLE_SHAPES:
        return name
    if name == "start":
        return "end"
    if name == "end":
        return "start"
    return name


def point_to_cell(p: Point, grid: int, margin: int = 1) -> Cell:
    """Scale a normalised point to a grid cell, keeping ``margin`` cells clear.

    The unit square maps onto ``0..grid-1`` inclusive and rounds half up, in
    exact arithmetic.  The result is then clamped so that nothing may be placed
    within ``margin`` cells of the border: stage 2 must leave room for the
    one-cell margin every routed path keeps.
    """
    if grid <= 2 * margin:
        raise ValueError(f"grid {grid} is too small for a {margin}-cell margin")
    span = grid - 1
    half = Fraction(1, 2)
    x = int(p.x * span + half)  # int() of a Fraction truncates; values are >= 0
    y = int(p.y * span + half)
    lo, hi = margin, grid - 1 - margin
    return (min(max(x, lo), hi), min(max(y, lo), hi))


def anchor_point(
    shape: Shape,
    name: str,
    *,
    rotation: int = 0,
    mirror: bool = False,
    swap_ends: bool = False,
) -> Point:
    """Resolve an anchor to a transformed point in normalised space."""
    macro = get_shape(shape)
    key = _swapped_name(shape, normalise_anchor_name(name), swap_ends)
    return transform_point(macro.anchor(key), rotation, mirror)


def anchor_cell(
    shape: Shape,
    name: str,
    grid: int,
    *,
    rotation: int = 0,
    mirror: bool = False,
    swap_ends: bool = False,
    margin: int = 1,
) -> Cell:
    """Resolve an anchor to a grid cell for one seed's transform.

    This is the function stage 2 calls: rotation (seed bits 0-1), mirror (bit
    2) and entrance/exit swap (bit 3) are applied in normalised space, and only
    the final point is scaled to the grid.
    """
    return point_to_cell(
        anchor_point(
            shape, name, rotation=rotation, mirror=mirror, swap_ends=swap_ends
        ),
        grid,
        margin,
    )


def shape_axis(shape: Shape, rotation: int = 0, mirror: bool = False) -> Axis:
    """The shape's primary axis after the seed transform.

    Outdoor tilesets align their long features to this.
    """
    return transform_axis(get_shape(shape).primary_axis, rotation, mirror)
