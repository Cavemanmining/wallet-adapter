"""The table view: one PNG of a profile's Descent web.

Spec: docs/WORLD_BIBLE.md section 03 -- "States and colours: locked grey,
reachable blue, active amber, cleared green, failed red. Colour is derived
from state, never stored."  This module is the one place that turns a colour
*name* from :data:`lucifer_descent.contracts.STATE_COLOUR` into RGB; it reads
the name off the node's state on every render and stores nothing.

What is drawn, back to front
----------------------------
* faint concentric ring guides, one per tier that has nodes, at the mean
  radius of that tier's nodes (spec: "tier 15 nodes ring the outer edge");
* every edge as a one-pixel line (spec: "edges are never removed", so the
  picture never has to show a missing one);
* every node as a filled disc in its state colour, the origin larger;
* a Pinnacle *arena* gets a thick ring outline and a *glyph* node a hollow
  diamond outline, each in that Pinnacle's colour (the contracts keep arena
  and glyph apart, so the picture does too);
* a mechanic node gets a small dark mark inside the disc, one shape per
  mechanic, because "a node may carry one mechanic ... decided at web
  generation" and a player plans routes around them;
* the profile id, passive point count and profile seed stamped top-left in
  the generator's 3x5 block font.

The PNG goes through :func:`lucifer_gen.png.write_png`; the font and the
pixel buffer come from :mod:`lucifer_gen.render`, which exposes both.

Determinism
-----------
The image is a pure function of the :class:`ProfileState` and ``size``.  No
random stream is opened -- rendering makes no choice a seed could vary -- and
every iteration is over tuples or sorted ids, so two renders of one state are
byte-identical, which :func:`render_web_bytes` lets a caller check without
touching the disk.

Every colour used is distinct from every other (checked at import), so a test
can prove "a node in state S exists" by finding S's RGB in the pixels and
prove the converse by not finding it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from lucifer_descent.contracts import (
    STATE_COLOUR,
    Mechanic,
    NodeState,
    Pinnacle,
    ProfileState,
    WebNode,
)
from lucifer_gen.png import encode_png, write_png
from lucifer_gen.render import Canvas, GLYPH_H, fit_text
from lucifer_gen.seed import format_seed

__all__ = [
    "RGB",
    "STATE_RGB",
    "PALETTE",
    "PINNACLE_RGB",
    "DEFAULT_SIZE",
    "state_rgb",
    "WebCanvas",
    "Geometry",
    "geometry_for",
    "draw_web",
    "render_web_bytes",
    "render_web",
]

RGB = Tuple[int, int, int]

DEFAULT_SIZE = 900

# --------------------------------------------------------------------------
# Colours
# --------------------------------------------------------------------------

#: RGB for each colour *name* in :data:`STATE_COLOUR`.  Keyed by the name, not
#: the state, so this table cannot disagree with the contracts about which
#: state is which colour: the name is looked up there first.
STATE_RGB: Dict[str, RGB] = {
    "grey": (118, 120, 128),
    "blue": (64, 132, 240),
    "amber": (240, 168, 40),
    "green": (72, 200, 96),
    "red": (224, 56, 56),
}

#: Everything that is not a node fill.
PALETTE: Dict[str, RGB] = {
    "background": (18, 18, 22),
    "ring_guide": (38, 40, 48),
    "edge": (74, 78, 92),
    "mark": (12, 12, 14),        # mechanic mark, inside the node disc
    "text": (226, 230, 238),
    "arbiter": (255, 96, 32),    # the Arbiter of Cinders: ember
    "monolith": (120, 220, 255), # the Blind Monolith: pale stone-light
}

#: Outline colour per Pinnacle, for both its arena and its glyph nodes.
PINNACLE_RGB: Dict[Pinnacle, RGB] = {
    Pinnacle.ARBITER: PALETTE["arbiter"],
    Pinnacle.MONOLITH: PALETTE["monolith"],
}


def _check_palette_distinct() -> None:
    """Every colour must be unique, or a presence test on pixels would lie."""
    seen: Dict[RGB, str] = {}
    for name, rgb in list(STATE_RGB.items()) + list(PALETTE.items()):
        if rgb in seen:
            raise AssertionError(f"colour {rgb} is used by both {seen[rgb]!r} and {name!r}")
        seen[rgb] = name
    for name in STATE_COLOUR.values():
        if name not in STATE_RGB:
            raise AssertionError(f"contracts colour {name!r} has no RGB in STATE_RGB")


_check_palette_distinct()


def state_rgb(state: NodeState) -> RGB:
    """The fill colour for a node in ``state``, via the contracts' colour name.

    Spec: "Colour is derived from state, never stored."
    """
    return STATE_RGB[STATE_COLOUR[state]]


# --------------------------------------------------------------------------
# Canvas: the generator's buffer plus the two primitives a graph needs
# --------------------------------------------------------------------------


class WebCanvas(Canvas):
    """:class:`lucifer_gen.render.Canvas` with lines and circles.

    Everything is integer arithmetic (Bresenham for lines, the midpoint
    algorithm for thin circles, an inequality scan for thick ones), so the
    pixels never depend on floating-point rounding.
    """

    def line(self, x0: int, y0: int, x1: int, y1: int, colour: RGB) -> None:
        """A one-pixel Bresenham line from ``(x0, y0)`` to ``(x1, y1)`` inclusive."""
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            self.set_pixel(x0, y0, colour)
            if x0 == x1 and y0 == y1:
                return
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(self, cx: int, cy: int, radius: int, colour: RGB) -> None:
        """A one-pixel midpoint circle; ``radius`` 0 is a single pixel."""
        r = max(0, radius)
        x, y, err = r, 0, 1 - r
        while x >= y:
            for px, py in (
                (cx + x, cy + y), (cx + y, cy + x), (cx - y, cy + x), (cx - x, cy + y),
                (cx - x, cy - y), (cx - y, cy - x), (cx + y, cy - x), (cx + x, cy - y),
            ):
                self.set_pixel(px, py, colour)
            y += 1
            if err < 0:
                err += 2 * y + 1
            else:
                x -= 1
                err += 2 * (y - x) + 1

    def annulus(self, cx: int, cy: int, radius: int, thickness: int, colour: RGB) -> None:
        """A filled ring ``thickness`` pixels wide whose outer radius is ``radius``.

        Scanned by inequality rather than stacked midpoint circles, which can
        leave one-pixel gaps on the diagonals between consecutive radii.
        """
        r_out = max(0, radius)
        r_in = max(0, r_out - max(1, thickness))
        lo, hi = r_in * r_in, r_out * r_out
        for dy in range(-r_out, r_out + 1):
            for dx in range(-r_out, r_out + 1):
                d = dx * dx + dy * dy
                if lo < d <= hi:
                    self.set_pixel(cx + dx, cy + dy, colour)


# --------------------------------------------------------------------------
# Geometry: where the web lands on the canvas
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Pixel placement for one render.

    ``scale`` converts layout units (:attr:`WebNode.x`/``y``) to pixels about
    the centre ``(cx, cy)``; ``ring_radius`` is the guide radius per tier.
    """

    size: int
    cx: int
    cy: int
    scale: float
    node_radius: int
    origin_radius: int
    outline_gap: int
    outline_thickness: int
    text_scale: int
    margin: int
    ring_radius: Dict[int, int]

    def pixel(self, node: WebNode) -> Tuple[int, int]:
        return (
            self.cx + int(round(node.x * self.scale)),
            self.cy + int(round(node.y * self.scale)),
        )

    def radius_of(self, node: WebNode, origin_id: int) -> int:
        return self.origin_radius if node.id == origin_id else self.node_radius


def geometry_for(state: ProfileState, size: int = DEFAULT_SIZE) -> Geometry:
    """Fit the web into a ``size`` x ``size`` image with a margin all round.

    The outermost node plus its outline must stay inside the margin, so the
    scale is set from the largest layout radius, not from the tier count:
    a hand-built web with odd coordinates still fits.
    """
    size = int(size)
    if size < 64:
        raise ValueError(f"size must be at least 64 pixels, got {size}")
    web = state.web
    node_radius = max(2, size // 150)
    origin_radius = max(node_radius + 2, (node_radius * 9) // 5)
    outline_gap = max(1, size // 450)
    outline_thickness = max(1, size // 450)
    text_scale = max(1, size // 300)
    margin = max(12, size // 16)

    reach = node_radius + outline_gap + outline_thickness + 2
    furthest = max((math.hypot(n.x, n.y) for n in web.nodes), default=0.0)
    usable = size / 2.0 - margin - reach
    scale = usable / furthest if furthest > 0 else 1.0

    # Ring guides: the mean radius of each tier's nodes, tiers with nodes only.
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for n in web.nodes:
        if n.tier <= 0:
            continue
        sums[n.tier] = sums.get(n.tier, 0.0) + math.hypot(n.x, n.y)
        counts[n.tier] = counts.get(n.tier, 0) + 1
    ring_radius = {
        tier: int(round(sums[tier] / counts[tier] * scale)) for tier in sorted(sums)
    }

    return Geometry(
        size=size,
        cx=size // 2,
        cy=size // 2,
        scale=scale,
        node_radius=node_radius,
        origin_radius=origin_radius,
        outline_gap=outline_gap,
        outline_thickness=outline_thickness,
        text_scale=text_scale,
        margin=margin,
        ring_radius=ring_radius,
    )


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------


def _draw_rings(canvas: WebCanvas, geo: Geometry) -> None:
    """Spec: tiers are rings of graph distance; a faint guide per tier."""
    for tier in sorted(geo.ring_radius):
        canvas.circle(geo.cx, geo.cy, geo.ring_radius[tier], PALETTE["ring_guide"])


def _draw_edges(canvas: WebCanvas, state: ProfileState, geo: Geometry) -> None:
    nodes = {n.id: n for n in state.web.nodes}
    for edge in state.web.edges:
        ax, ay = geo.pixel(nodes[edge.a])
        bx, by = geo.pixel(nodes[edge.b])
        canvas.line(ax, ay, bx, by, PALETTE["edge"])


def _draw_mechanic_mark(canvas: WebCanvas, cx: int, cy: int, radius: int, mechanic: Mechanic) -> None:
    """One small dark shape per mechanic, inside the node disc.

    breach: a cross; ritual: a hollow diamond; dig: a square; shrine: a dot.
    The shapes are told apart by eye; the colour is the same for all four so
    a pixel test for "any mechanic" needs only :data:`PALETTE`\\ ``["mark"]``.
    """
    m = max(1, radius // 2)
    colour = PALETTE["mark"]
    if mechanic is Mechanic.BREACH:
        canvas.cross(cx, cy, m, colour, thickness=1)
    elif mechanic is Mechanic.RITUAL:
        canvas.diamond(cx, cy, m, colour, hollow=True)
    elif mechanic is Mechanic.DIG:
        half = max(1, m // 2)
        canvas.fill_rect(cx - half, cy - half, 2 * half + 1, 2 * half + 1, colour)
    else:  # SHRINE, and any mechanic added later still gets a visible mark
        canvas.disc(cx, cy, max(1, m // 2), colour)


def _draw_node(canvas: WebCanvas, state: ProfileState, geo: Geometry, node: WebNode) -> None:
    """Disc in the state colour, then outlines, then the mechanic mark on top."""
    try:
        node_state = state.states[node.id]
    except KeyError:
        raise ValueError(f"profile state has no entry for node {node.id}") from None
    cx, cy = geo.pixel(node)
    r = geo.radius_of(node, state.web.origin_id)
    canvas.disc(cx, cy, r, state_rgb(node_state))

    outer = r + geo.outline_gap + geo.outline_thickness
    if node.pinnacle is not None:
        # Spec: an arena is the Pinnacle fight itself; a thick ring says so.
        canvas.annulus(cx, cy, outer, geo.outline_thickness, PINNACLE_RGB[node.pinnacle])
    if node.glyph is not None:
        # Spec: "fragments from three cleared tier-15 nodes bearing that
        # Pinnacle's glyph"; a hollow diamond, unlike the arena's ring.
        colour = PINNACLE_RGB[node.glyph]
        # Diamonds are drawn corner-to-corner, so give the radius a little
        # more than the ring's to keep the shape clear of the disc.
        for k in range(geo.outline_thickness):
            canvas.diamond(cx, cy, outer + 1 + k, colour, hollow=True)
    if node.mechanic is not None:
        _draw_mechanic_mark(canvas, cx, cy, r, node.mechanic)


def _draw_nodes(canvas: WebCanvas, state: ProfileState, geo: Geometry) -> None:
    """Nodes in id order, the origin last so nothing paints over it."""
    origin_id = state.web.origin_id
    ordered = sorted(state.web.nodes, key=lambda n: (n.id == origin_id, n.id))
    for node in ordered:
        _draw_node(canvas, state, geo, node)


def _stamp(canvas: WebCanvas, state: ProfileState, geo: Geometry) -> None:
    """Profile id, passive points and profile seed, top-left, in the block font.

    Upper-cased because the font has one case; truncated to the image width
    because a long profile id must not change the image size.
    """
    t = geo.text_scale
    x = geo.margin // 2
    y = geo.margin // 2
    available = canvas.width - 2 * x
    lines = (
        f"PROFILE {state.profile_id}",
        f"POINTS {state.passive_points}",
        f"SEED {format_seed(state.web.profile_seed)}",
    )
    for index, line in enumerate(lines):
        canvas.text(
            x,
            y + index * (GLYPH_H * t + 2 * t),
            fit_text(line.upper(), available, t),
            PALETTE["text"],
            t,
        )


def draw_web(state: ProfileState, size: int = DEFAULT_SIZE) -> WebCanvas:
    """Draw the table view into a fresh canvas and return it.

    The one function both writers share, so the file on disk and the bytes
    in memory can never differ.
    """
    geo = geometry_for(state, size)
    canvas = WebCanvas(geo.size, geo.size, PALETTE["background"])
    _draw_rings(canvas, geo)
    _draw_edges(canvas, state, geo)
    _draw_nodes(canvas, state, geo)
    _stamp(canvas, state, geo)
    return canvas


def render_web_bytes(state: ProfileState, size: int = DEFAULT_SIZE) -> bytes:
    """The PNG bytes of the table view, without touching the filesystem."""
    canvas = draw_web(state, size)
    return encode_png(canvas.width, canvas.height, canvas.pixels)


def render_web(state: ProfileState, path, size: int = DEFAULT_SIZE) -> int:
    """Write the table view of ``state`` to ``path`` as a PNG.

    Returns the number of bytes written, as :func:`lucifer_gen.png.write_png`
    does.  Byte-deterministic: the same state and size give the same file.
    """
    canvas = draw_web(state, size)
    return write_png(path, canvas.width, canvas.height, canvas.pixels)
