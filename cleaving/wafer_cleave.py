"""Wafer cleaving / die layout planner.

Plans how many dies of one or more sizes can be cleaved out of a *partial*
wafer (a wafer that has already been cleaved along one or more straight lines),
and draws a diagram of the result.

Everything is in millimetres, in wafer coordinates with the wafer centre at
(0, 0).

Model
-----
Cleaving only produces straight, full-length cuts along crystal planes, so the
layout has to be guillotine-cuttable.  This planner uses the two-stage scheme
you would actually use at the bench:

  1. Cleave *strips* parallel to the existing straight edge, marching inward.
     Each strip is one die wide.
  2. Cleave each strip crosswise into dies.

Because step 2 happens after the strip has been separated, different strips may
be diced with different die lengths -- which is what lets several die sizes
share one piece of wafer.

Die types are allocated in the order given: the first type takes strips off the
reference edge until its ``max_count`` is reached, the used band is cleaved
free, and the remainder is handed to the next type -- which may run its own
strips along the *other* axis.  A type with ``max_count=None`` takes everything
that is left.  Pin a type's orientation with ``DieType.strip_axis='x'`` (or the
trailing ``:x`` in a die spec) to see what you get by cleaving that way first.

``align='grid'`` (the default) puts every die of a type on one shared lattice
so the cross cuts line up from strip to strip.  ``align='center'`` centres each
strip's dies in that strip's own span instead, which fits marginally more dies
but leaves the rows staggered, since the usable span shrinks as you move out
along the wafer curve.

Usage
-----
Interactive::

    python wafer_cleave.py

Command line::

    python wafer_cleave.py --diameter 50.8 --cleave 5:larger \
        --die "A:10.5x9.5:8" --die "B:5x5" --out layout.png

As a library::

    from wafer_cleave import Wafer, Cut, DieType, plan_layout, plot_layout
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Sequence

# ---------------------------------------------------------------------------
# geometry primitives
# ---------------------------------------------------------------------------

_EPS = 1e-9

# A half-plane constraint is (axis, sense, position):
#   ('x', '<=', 5.0)  ->  keep x <= 5.0
HalfPlane = tuple


@dataclass
class Wafer:
    """A round wafer, optionally with a primary flat and an edge exclusion."""

    diameter: float = 50.8          # 2" wafer
    flat_length: float = 0.0        # chord length of the primary flat, 0 = none
    flat_side: str = "bottom"       # bottom | top | left | right
    edge_exclusion: float = 0.0     # unusable rim width

    @property
    def radius(self) -> float:
        return self.diameter / 2.0

    @property
    def usable_radius(self) -> float:
        return self.radius - self.edge_exclusion

    def flat_offset(self, usable: bool = True) -> float | None:
        """Distance from wafer centre to the flat chord (None if no flat)."""
        if self.flat_length <= 0:
            return None
        d = math.sqrt(max(self.radius ** 2 - (self.flat_length / 2.0) ** 2, 0.0))
        return d - self.edge_exclusion if usable else d

    def flat_half_plane(self, usable: bool = True) -> HalfPlane | None:
        d = self.flat_offset(usable=usable)
        if d is None:
            return None
        return {
            "bottom": ("y", ">=", -d),
            "top": ("y", "<=", d),
            "left": ("x", ">=", -d),
            "right": ("x", "<=", d),
        }[self.flat_side]


@dataclass
class Cut:
    """A cleave already made in the wafer.

    ``keep='below'`` retains the material with coordinate <= ``position``;
    ``keep='above'`` retains coordinate >= ``position``.
    """

    axis: str            # 'x' or 'y'
    position: float
    keep: str = "below"  # 'below' or 'above'

    def half_plane(self) -> HalfPlane:
        return (self.axis, "<=" if self.keep == "below" else ">=", self.position)


@dataclass
class DieType:
    name: str
    width: float                 # extent along x as entered
    height: float                # extent along y as entered
    max_count: int | None = None
    allow_rotate: bool = False
    color: str | None = None
    strip_axis: str | None = None   # 'x' / 'y' to pin this type, None = let it choose


@dataclass
class PlacedDie:
    type_name: str
    x: float      # lower-left corner
    y: float
    w: float
    h: float


@dataclass
class Strip:
    type_name: str
    axis: str             # axis the strips march along
    lo: float             # strip extent along the strip axis
    hi: float
    die_pitch: float      # die size along the perpendicular axis
    starts: list          # perpendicular coordinate of each die's low edge
    n: int
    ref: float            # edge this type was measured from
    direction: int        # +1 / -1, direction of march away from `ref`


@dataclass
class Layout:
    wafer: Wafer
    cuts: list
    die_types: list
    strip_axis: str
    align: str
    dies: list
    strips: list
    counts: dict
    dims: list            # (w, h) in x/y terms actually used, per die type
    axes: dict            # die type name -> strip axis actually used
    half_planes: list = field(default_factory=list)

    @property
    def total_dies(self) -> int:
        return len(self.dies)

    @property
    def die_area(self) -> float:
        return sum(d.w * d.h for d in self.dies)


# ---------------------------------------------------------------------------
# region helpers
# ---------------------------------------------------------------------------

def build_half_planes(wafer: Wafer, cuts: Sequence[Cut]) -> list:
    hp = []
    flat = wafer.flat_half_plane(usable=True)
    if flat is not None:
        hp.append(flat)
    hp.extend(c.half_plane() for c in cuts)
    return hp


def axis_bounds(wafer: Wafer, half_planes: Sequence[HalfPlane], axis: str):
    """Bounding interval of the retained region along ``axis``."""
    lo, hi = -wafer.usable_radius, wafer.usable_radius
    for ax, sense, pos in half_planes:
        if ax != axis:
            continue
        if sense == "<=":
            hi = min(hi, pos)
        else:
            lo = max(lo, pos)
    return lo, hi


def perp_span(wafer: Wafer, half_planes: Sequence[HalfPlane],
              axis: str, lo: float, hi: float):
    """Usable span perpendicular to ``axis`` across the whole strip [lo, hi].

    A die fills the full width of its strip, so the strip is limited by its
    narrowest point -- the end farthest from the wafer centre.
    """
    Ru = wafer.usable_radius
    far = max(abs(lo), abs(hi))
    if far >= Ru:
        return None
    half = math.sqrt(Ru * Ru - far * far)

    perp = "y" if axis == "x" else "x"
    plo, phi = -half, half
    for ax, sense, pos in half_planes:
        if ax == perp:
            if sense == "<=":
                phi = min(phi, pos)
            else:
                plo = max(plo, pos)
        else:  # constraint along the strip axis: strip must lie fully inside
            if sense == "<=" and hi > pos + _EPS:
                return None
            if sense == ">=" and lo < pos - _EPS:
                return None
    if phi - plo <= _EPS:
        return None
    return plo, phi


def clip_polygon(poly, axis, sense, pos):
    """Sutherland-Hodgman clip of a polygon against an axis-aligned half-plane."""
    idx = 0 if axis == "x" else 1
    if sense == "<=":
        inside = lambda p: p[idx] <= pos + 1e-12
    else:
        inside = lambda p: p[idx] >= pos - 1e-12

    out = []
    n = len(poly)
    for i in range(n):
        cur, prv = poly[i], poly[i - 1]
        ci, pi = inside(cur), inside(prv)
        if ci:
            if not pi:
                out.append(_intersect(prv, cur, idx, pos))
            out.append(cur)
        elif pi:
            out.append(_intersect(prv, cur, idx, pos))
    return out


def _intersect(a, b, idx, pos):
    denom = b[idx] - a[idx]
    t = 0.0 if abs(denom) < 1e-15 else (pos - a[idx]) / denom
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def region_polygon(wafer: Wafer, half_planes: Sequence[HalfPlane],
                   radius: float, n: int = 1440):
    poly = [
        (radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    for ax, sense, pos in half_planes:
        poly = clip_polygon(poly, ax, sense, pos)
        if not poly:
            break
    return poly


# ---------------------------------------------------------------------------
# planner
# ---------------------------------------------------------------------------

ALIGNMENTS = ("grid", "center", "low", "high")


def edge_candidates(wafer: Wafer, region: Sequence[HalfPlane], axis: str):
    """Places to start cleaving strips, as (edge, direction) pairs.

    A straight edge -- an old cleave, the flat, or the boundary left behind by
    a previous die type -- is the reference you would actually measure from at
    the bench, so those come first.  Falling off the end of the list means the
    region is bounded only by the wafer curve.
    """
    lo, hi = axis_bounds(wafer, region, axis)
    Ru = wafer.usable_radius
    cands = []
    if hi < Ru - 1e-9:
        cands.append((hi, -1))
    if lo > -Ru + 1e-9:
        cands.append((lo, +1))
    return cands or [(lo, +1)]


def _strip_spans(wafer, region, axis, start, direction, w, max_strips=2000):
    """Walk strips of width ``w`` away from ``start``, until they leave region.

    Returns ``(lo, hi, span)`` per strip, where ``span`` is the usable extent
    perpendicular to ``axis`` (None if the strip holds nothing).
    """
    lo_b, hi_b = axis_bounds(wafer, region, axis)
    spans = []
    pos = start
    while len(spans) < max_strips:
        a, b = (pos, pos + w) if direction > 0 else (pos - w, pos)
        if a < lo_b - _EPS or b > hi_b + _EPS:
            break
        spans.append((a, b, perp_span(wafer, region, axis, a, b)))
        pos = b if direction > 0 else a
    return spans


def _place_in_spans(spans, h, kerf, align, origin, max_count):
    """Cut each strip crosswise into dies of pitch ``h``.

    ``align='grid'`` puts every die of this type on one shared lattice through
    ``origin``, so the cross cuts line up from strip to strip.  The other modes
    place each strip independently, which fits marginally more dies but leaves
    the rows staggered.
    """
    pitch = h + kerf
    placed = []
    total = 0
    for a, b, span in spans:
        if max_count is not None and total >= max_count:
            break
        starts = []
        if span is not None:
            plo, phi = span
            if align == "grid":
                k = math.ceil((plo - origin) / pitch - 1e-9)
                while origin + k * pitch + h <= phi + 1e-9:
                    starts.append(origin + k * pitch)
                    k += 1
            else:
                n = int(math.floor((phi - plo + kerf) / pitch + 1e-9))
                if n > 0:
                    used = n * h + (n - 1) * kerf
                    if align == "low":
                        p0 = plo
                    elif align == "high":
                        p0 = phi - used
                    else:
                        p0 = plo + (phi - plo - used) / 2.0
                    starts = [p0 + i * pitch for i in range(n)]
        if max_count is not None:
            starts = starts[:max_count - total]
        total += len(starts)
        placed.append((a, b, starts))

    while placed and not placed[-1][2]:
        placed.pop()          # trailing empty strips are never cleaved
    return placed, total


def _plan_type(wafer, region, t, strip_axis, kerf, phase_search, phase_step, align):
    """Best strip plan for one die type on the region it inherits."""
    if t.strip_axis:
        axes = [t.strip_axis]
    elif strip_axis == "both":
        axes = ["x", "y"]
    else:
        axes = [strip_axis]

    best_total = 0
    ties = []
    for axis in axes:
        base = (t.width, t.height) if axis == "x" else (t.height, t.width)
        dims = [base] + ([(base[1], base[0])] if t.allow_rotate else [])
        for w, h in dims:
            offsets = ([i * phase_step for i in range(max(1, int(round(w / phase_step))))]
                       if phase_search else [0.0])
            for edge, direction in edge_candidates(wafer, region, axis):
                for off in offsets:
                    spans = _strip_spans(wafer, region, axis, edge + direction * off,
                                         direction, w)
                    real = [s for _, _, s in spans if s is not None]
                    if not real:
                        continue
                    if align == "grid":
                        pitch = h + kerf
                        base_p = min(s[0] for s in real)
                        n_org = max(1, int(round(pitch / phase_step))) if phase_search else 1
                        origins = [base_p + i * phase_step for i in range(n_org)]
                    else:
                        origins = [0.0]
                    for origin in origins:
                        placed, total = _place_in_spans(spans, h, kerf, align,
                                                        origin, t.max_count)
                        if total == 0 or total < best_total:
                            continue
                        end = placed[-1][1] if direction > 0 else placed[-1][0]
                        if total > best_total:
                            best_total, ties = total, []
                        ties.append((axis, w, h, edge, direction, end, off, placed))
    if not ties:
        return None

    # Among the plans that yield the same number of dies, keep the one that
    # leaves the largest usable area behind -- a band consumed along x and one
    # consumed along y are not comparable by width alone.
    leftover = {}

    def remaining(axis, direction, end):
        key = (axis, direction, end)
        if key not in leftover:
            rest = list(region) + [(axis, ">=" if direction > 0 else "<=", end)]
            leftover[key] = _polygon_area(
                region_polygon(wafer, rest, wafer.usable_radius, n=360))
        return leftover[key]

    axis, w, h, edge, direction, end, off, placed = max(
        ties, key=lambda c: (remaining(c[0], c[4], c[5]), -c[6]))
    return (best_total, axis, w, h, edge, direction, end, placed)


def plan_layout(wafer: Wafer,
                cuts: Sequence[Cut],
                die_types: Sequence[DieType],
                strip_axis: str = "x",
                kerf: float = 0.0,
                phase_search: bool = True,
                phase_step: float = 0.25,
                align: str = "grid") -> Layout:
    """Plan a cleaving layout.

    Die types are handled in order.  Each one claims a band of strips off a
    straight edge; the leftover is cleaved free and handed to the next type,
    which is free to run its strips along the other axis.  Pin a type's
    orientation with ``DieType.strip_axis``, or all of them with ``strip_axis``
    (``'x'``, ``'y'``, or ``'both'`` to let each type choose).
    """
    die_types = list(die_types)
    if not die_types:
        raise ValueError("at least one die type is required")
    for t in die_types:
        if t.width <= 0 or t.height <= 0:
            raise ValueError(f"die type {t.name!r} has a non-positive dimension")
        if t.strip_axis not in (None, "x", "y"):
            raise ValueError(f"die type {t.name!r} has a bad strip_axis "
                             f"{t.strip_axis!r}; use 'x', 'y' or None")
    if align not in ALIGNMENTS:
        raise ValueError(f"align must be one of {ALIGNMENTS}, got {align!r}")
    if strip_axis not in ("x", "y", "both"):
        raise ValueError(f"strip_axis must be 'x', 'y' or 'both', got {strip_axis!r}")

    half_planes = build_half_planes(wafer, cuts)
    region = list(half_planes)
    dies, strips, counts, dims, axes = [], [], {}, [], {}

    for t in die_types:
        res = _plan_type(wafer, region, t, strip_axis, kerf,
                         phase_search, phase_step, align)
        if res is None:
            counts[t.name] = 0
            dims.append((t.width, t.height))
            axes[t.name] = None
            continue

        total, axis, w, h, edge, direction, end, placed = res
        for a, b, starts in placed:
            for q in starts:
                if axis == "x":
                    dies.append(PlacedDie(t.name, a, q, w, h))
                else:
                    dies.append(PlacedDie(t.name, q, a, h, w))
            strips.append(Strip(t.name, axis, a, b, h, starts, len(starts),
                                edge, direction))
        counts[t.name] = total
        dims.append((w, h) if axis == "x" else (h, w))
        axes[t.name] = axis
        # cleave the used band away; the rest goes to the next die type
        region = region + [(axis, ">=" if direction > 0 else "<=", end)]

    return Layout(wafer=wafer, cuts=list(cuts), die_types=die_types,
                  strip_axis=strip_axis, align=align, dies=dies, strips=strips,
                  counts=counts, dims=dims, axes=axes, half_planes=half_planes)


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def report(layout: Layout) -> str:
    w = layout.wafer
    lines = []
    add = lines.append

    add("=" * 62)
    add("WAFER CLEAVING PLAN")
    add("=" * 62)
    add(f"Wafer            : {w.diameter:.2f} mm dia "
        f"({w.diameter / 25.4:.2f}\"), R = {w.radius:.2f} mm")
    if w.flat_length > 0:
        add(f"Primary flat     : {w.flat_length:.2f} mm chord on the {w.flat_side}")
    if w.edge_exclusion > 0:
        add(f"Edge exclusion   : {w.edge_exclusion:.2f} mm "
            f"(usable R = {w.usable_radius:.2f} mm)")
    for c in layout.cuts:
        side = "<=" if c.keep == "below" else ">="
        add(f"Existing cleave  : {c.axis} = {c.position:+.2f} mm, "
            f"keeping {c.axis} {side} {c.position:+.2f}")
    add("")

    add(f"Cross-cut alignment: {layout.align}")
    add("")

    add("Die yield:")
    total_area = 0.0
    for t in layout.die_types:
        dw, dh = next((d.w, d.h) for d in layout.dies if d.type_name == t.name) \
            if any(d.type_name == t.name for d in layout.dies) else (t.width, t.height)
        n = layout.counts[t.name]
        total_area += n * dw * dh
        rot = "  (rotated)" if (dw, dh) != (t.width, t.height) else ""
        ax = layout.axes.get(t.name)
        pin = " (pinned)" if t.strip_axis else ""
        along = f"strips along {ax}{pin}" if ax else "-"
        add(f"  {t.name:<12} {dw:5.2f} x {dh:5.2f} mm   ->  {n:3d} "
            f"{'die ' if n == 1 else 'dies'}   {along}{rot}")
    add(f"  {'TOTAL':<12} {'':<16}   ->  {layout.total_dies:3d} dies")

    poly = region_polygon(w, layout.half_planes, w.usable_radius)
    area = _polygon_area(poly)
    if area > 0:
        add(f"  usable area {area:.1f} mm^2, "
            f"die area {total_area:.1f} mm^2 ({100 * total_area / area:.1f} %)")
    add("")

    add("Cleave recipe, in order:")
    if not layout.strips:
        add("  nothing fits")
    for t in layout.die_types:
        mine = [s for s in layout.strips if s.type_name == t.name]
        if not mine:
            continue
        ref, axis = mine[0].ref, mine[0].axis
        perp = "y" if axis == "x" else "x"
        way = "+" if mine[0].direction > 0 else "-"
        add("")
        add(f"  [{t.name}]  {len(mine)} "
            f"{'strip' if len(mine) == 1 else 'strips'} along {axis}, from the "
            f"edge at {axis} = {ref:+.2f} mm, working {way}{axis}")
        for i, s in enumerate(mine, 1):
            near = s.lo if s.direction > 0 else s.hi
            far = s.hi if s.direction > 0 else s.lo
            add(f"    strip {i:2d}  {axis} = {s.lo:+7.2f} .. {s.hi:+7.2f}"
                f"   (cumulative {abs(near - ref):6.2f} -> {abs(far - ref):6.2f} mm)"
                f"   {s.n} {'die' if s.n == 1 else 'dies'}")
            if s.n:
                cuts_perp = ", ".join(f"{q:+.2f}" for q in s.starts) + \
                            f", {s.starts[-1] + s.die_pitch:+.2f}"
                add(f"              {perp} cuts: {cuts_perp}")
        end = mine[-1].hi if mine[0].direction > 0 else mine[-1].lo
        add(f"    -> band ends at {axis} = {end:+.2f} mm "
            f"({abs(end - ref):.2f} mm of material used)")
    add("=" * 62)
    return "\n".join(lines)


def _polygon_area(poly) -> float:
    if len(poly) < 3:
        return 0.0
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i - 1]
        x1, y1 = poly[i]
        a += x0 * y1 - x1 * y0
    return abs(a) / 2.0


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------

_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
            "#8172B3", "#937860", "#DA8BC3", "#8C8C8C"]


def plot_layout(layout: Layout, ax=None, out: str | None = None,
                show_labels: bool = False, title: str | None = None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon as MplPolygon

    w = layout.wafer
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    # full wafer outline (with flat), for reference
    flat_full = w.flat_half_plane(usable=False)
    outline = region_polygon(w, [flat_full] if flat_full else [], w.radius)
    ax.add_patch(MplPolygon(outline, closed=True, facecolor="#F2F2F2",
                            edgecolor="#9A9A9A", lw=1.2, ls="--", zorder=0))

    # retained piece
    piece = region_polygon(w, layout.half_planes, w.usable_radius)
    if piece:
        ax.add_patch(MplPolygon(piece, closed=True, facecolor="#DCE6F1",
                                edgecolor="#2F4B7C", lw=1.6, zorder=1))

    colors = {}
    for i, t in enumerate(layout.die_types):
        colors[t.name] = t.color or _PALETTE[i % len(_PALETTE)]

    for d in layout.dies:
        ax.add_patch(Rectangle((d.x, d.y), d.w, d.h,
                               facecolor=colors[d.type_name], edgecolor="white",
                               lw=0.8, alpha=0.9, zorder=3))
    if show_labels:
        for i, d in enumerate(layout.dies, 1):
            ax.text(d.x + d.w / 2, d.y + d.h / 2, str(i), ha="center",
                    va="center", fontsize=6, color="white", zorder=4)

    # existing cleave lines
    R = w.radius * 1.05
    for c in layout.cuts:
        if c.axis == "x":
            ax.plot([c.position, c.position], [-R, R], color="#C0392B",
                    lw=1.6, ls="--", zorder=5)
            ax.annotate(f"cleave x = {c.position:+.1f}", (c.position, R),
                        textcoords="offset points", xytext=(4, -12),
                        color="#C0392B", fontsize=9)
        else:
            ax.plot([-R, R], [c.position, c.position], color="#C0392B",
                    lw=1.6, ls="--", zorder=5)
            ax.annotate(f"cleave y = {c.position:+.1f}", (-R, c.position),
                        textcoords="offset points", xytext=(4, 4),
                        color="#C0392B", fontsize=9)

    # centre lines
    ax.axhline(0, color="#B0B0B0", lw=0.6, zorder=2)
    ax.axvline(0, color="#B0B0B0", lw=0.6, zorder=2)

    handles = []
    from matplotlib.patches import Patch
    for t in layout.die_types:
        dims = next(((d.w, d.h) for d in layout.dies if d.type_name == t.name),
                    (t.width, t.height))
        handles.append(Patch(facecolor=colors[t.name],
                             label=f"{t.name}: {dims[0]:g} x {dims[1]:g} mm  "
                                   f"(n = {layout.counts[t.name]})"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
              frameon=False, fontsize=10)

    lim = w.radius * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    if title is None:
        used = ", ".join(f"{n}: {a}" for n, a in layout.axes.items() if a)
        title = (f"{w.diameter / 25.4:g}\" wafer piece — {layout.total_dies} dies"
                 + (f"  (strips along {used})" if used else ""))
    ax.set_title(title)
    ax.grid(alpha=0.15, lw=0.5)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig, ax


# ---------------------------------------------------------------------------
# parsing / CLI
# ---------------------------------------------------------------------------

def parse_die(spec: str) -> DieType:
    """``name:WxH[:count|inf][:rot][:x|:y]``  e.g. ``A:10.5x9.5:8:rot:x``

    A trailing ``x`` or ``y`` pins the axis this type's strips run along.
    """
    parts = [p.strip() for p in spec.split(":")]
    if len(parts) < 2:
        raise ValueError(f"bad die spec {spec!r}; expected name:WxH[:count][:rot][:x|y]")
    name = parts[0]
    dims = parts[1].lower().replace("mm", "").split("x")
    if len(dims) != 2:
        raise ValueError(f"bad die size {parts[1]!r}; expected WxH")
    w, h = float(dims[0]), float(dims[1])
    count, rot, axis = None, False, None
    for extra in parts[2:]:
        e = extra.lower()
        if e in ("rot", "rotate", "r"):
            rot = True
        elif e in ("x", "y"):
            axis = e
        elif e in ("", "inf", "all", "rest", "max", "none"):
            count = None
        else:
            count = int(e)
    return DieType(name=name, width=w, height=h, max_count=count,
                   allow_rotate=rot, strip_axis=axis)


def parse_cut(spec: str) -> Cut:
    """``axis,position,keep``  e.g. ``x,5,below``  (keep: below|above|larger|smaller)"""
    parts = [p.strip().lower() for p in spec.replace(":", ",").split(",")]
    axis = parts[0]
    if axis not in ("x", "y"):
        raise ValueError(f"cut axis must be x or y, got {axis!r}")
    pos = float(parts[1])
    keep = parts[2] if len(parts) > 2 else "larger"
    return Cut(axis=axis, position=pos, keep=_resolve_keep(keep, pos))


def _resolve_keep(keep: str, pos: float) -> str:
    keep = keep.lower()
    if keep in ("below", "under", "left", "bottom", "-", "minus"):
        return "below"
    if keep in ("above", "over", "right", "top", "+", "plus"):
        return "above"
    if keep in ("larger", "large", "big", "bigger", "major"):
        return "below" if pos >= 0 else "above"
    if keep in ("smaller", "small", "minor"):
        return "above" if pos >= 0 else "below"
    raise ValueError(f"unrecognised keep side {keep!r}")


def _ask(prompt, default=None, cast=str):
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw:
            if default is None:
                continue
            return default if not isinstance(default, str) else cast(default)
        try:
            return cast(raw)
        except (ValueError, TypeError) as exc:
            print(f"  -> {exc}")


def interactive() -> tuple:
    print("\nWafer cleaving planner (all dimensions in mm)\n" + "-" * 44)
    dia = _ask("Wafer diameter (50.8 = 2\", 76.2 = 3\")", 50.8, float)
    flat = _ask("Primary flat chord length (0 for none)", 0.0, float)
    side = _ask("Flat side (bottom/top/left/right)", "bottom") if flat > 0 else "bottom"
    ee = _ask("Edge exclusion", 0.0, float)
    wafer = Wafer(diameter=dia, flat_length=flat, flat_side=side, edge_exclusion=ee)

    cuts = []
    print("\nExisting cleaves (blank line when done).")
    print("  format: axis,position[,keep]   e.g.  x,5,larger")
    while True:
        raw = input(f"  cleave {len(cuts) + 1}: ").strip()
        if not raw:
            break
        try:
            cuts.append(parse_cut(raw))
        except ValueError as exc:
            print(f"  -> {exc}")

    dies = []
    print("\nDie types, in priority order (blank line when done).")
    print("  format: name:WxH[:count][:rot][:x|:y]   e.g.  A:10.5x9.5:8:x")
    print("  omit the count to let that type use everything that is left")
    print("  trailing :x or :y pins the axis that type's strips run along")
    while True:
        raw = input(f"  die {len(dies) + 1}: ").strip()
        if not raw:
            if dies:
                break
            print("  -> need at least one die type")
            continue
        try:
            dies.append(parse_die(raw))
        except ValueError as exc:
            print(f"  -> {exc}")

    axis = _ask("\nCleave strips along which axis (x/y/both)", "both")
    align = _ask("Cross-cut alignment (grid/center/low/high)", "grid")
    kerf = _ask("Kerf / cleave allowance", 0.0, float)
    return wafer, cuts, dies, axis, align, kerf


def best_layout(wafer, cuts, dies, axis="both", kerf=0.0, **kw) -> Layout:
    """Convenience wrapper: ``axis='both'`` lets each die type pick its own."""
    return plan_layout(wafer, cuts, dies, strip_axis=axis, kerf=kerf, **kw)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--diameter", type=float, default=50.8,
                   help="wafer diameter in mm (default 50.8 = 2 inch)")
    p.add_argument("--flat", type=float, default=0.0,
                   help="primary flat chord length in mm")
    p.add_argument("--flat-side", default="bottom",
                   choices=["bottom", "top", "left", "right"])
    p.add_argument("--edge-exclusion", type=float, default=0.0)
    p.add_argument("--cleave", action="append", default=[], metavar="AXIS,POS[,KEEP]",
                   help="existing cleave, e.g. x,5,larger (repeatable)")
    p.add_argument("--die", action="append", default=[], metavar="NAME:WxH[:N][:rot]",
                   help="die type in priority order (repeatable)")
    p.add_argument("--axis", default="both", choices=["x", "y", "both"],
                   help="axis the strips run along; 'both' lets each type choose "
                        "(override per type with a trailing :x or :y on --die)")
    p.add_argument("--align", default="grid", choices=list(ALIGNMENTS),
                   help="grid = one shared lattice so cross cuts line up "
                        "between strips; center/low/high place each strip alone")
    p.add_argument("--kerf", type=float, default=0.0)
    p.add_argument("--no-phase-search", action="store_true",
                   help="start strips exactly at the reference edge")
    p.add_argument("--labels", action="store_true", help="number each die")
    p.add_argument("--out", default=None, help="save the diagram to this file")
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args(argv)

    if args.die:
        wafer = Wafer(diameter=args.diameter, flat_length=args.flat,
                      flat_side=args.flat_side, edge_exclusion=args.edge_exclusion)
        cuts = [parse_cut(s) for s in args.cleave]
        dies = [parse_die(s) for s in args.die]
        axis, align, kerf = args.axis, args.align, args.kerf
    else:
        wafer, cuts, dies, axis, align, kerf = interactive()

    layout = best_layout(wafer, cuts, dies, axis=axis, kerf=kerf, align=align,
                         phase_search=not args.no_phase_search)
    print()
    print(report(layout))

    plot_layout(layout, out=args.out, show_labels=args.labels)
    if args.out:
        print(f"\ndiagram written to {args.out}")
    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()
    return layout


if __name__ == "__main__":
    main()
