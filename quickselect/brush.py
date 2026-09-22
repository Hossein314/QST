"""Brush stamping.

Photoshop paints a stroke as a sequence of overlapping *stamps* spaced a fixed
fraction of the brush diameter apart, each stamp a radial falloff controlled by
"hardness".  Reproducing that matters for behaviour, not just looks: the set of
stamped pixels is exactly the set of hard foreground seeds handed to the graph
cut, so spacing determines how often the local solve runs during a drag.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import BrushConfig
from .imagedata import ROI

# Cached stamp kernels, keyed by (rounded radius, rounded hardness).
_KERNEL_CACHE: dict = {}
_CACHE_LIMIT = 256


def stamp_kernel(radius: float, hardness: float) -> np.ndarray:
    """Radial falloff kernel, float32 in [0, 1], odd-sized and centred.

    ``hardness == 1`` gives an anti-aliased hard disc; lower values ramp the
    alpha down from the ``hardness`` fraction of the radius out to the rim,
    using a smoothstep so the profile matches Photoshop's soft brushes closely.
    """
    radius = max(0.5, float(radius))
    hardness = float(np.clip(hardness, 0.0, 1.0))
    key = (round(radius, 2), round(hardness, 3))
    cached = _KERNEL_CACHE.get(key)
    if cached is not None:
        return cached

    size = int(np.ceil(radius)) * 2 + 1
    c = size // 2
    yy, xx = np.mgrid[0:size, 0:size]
    d = np.hypot(yy - c, xx - c).astype(np.float32)

    inner = radius * hardness
    if radius - inner < 1e-3:
        # Hard brush: one pixel of anti-aliasing at the rim.
        alpha = np.clip(radius - d + 0.5, 0.0, 1.0)
    else:
        t = np.clip((d - inner) / (radius - inner), 0.0, 1.0)
        alpha = 1.0 - (t * t * (3.0 - 2.0 * t))  # smoothstep
        alpha *= np.clip(radius - d + 0.5, 0.0, 1.0)

    alpha = alpha.astype(np.float32)
    if len(_KERNEL_CACHE) > _CACHE_LIMIT:
        _KERNEL_CACHE.clear()
    _KERNEL_CACHE[key] = alpha
    return alpha


def interpolate_points(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    spacing: float,
    carry: float = 0.0,
) -> Tuple[List[Tuple[float, float]], float]:
    """Stamp centres along the segment ``p0 -> p1``.

    ``carry`` is the leftover distance from the previous segment, so stamp
    spacing stays uniform across a whole stroke rather than restarting at each
    mouse-move event.  Returns the new leftover.
    """
    spacing = max(0.5, float(spacing))
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    dist = float(np.hypot(dx, dy))
    pts: List[Tuple[float, float]] = []
    if dist < 1e-6:
        return pts, carry
    t = spacing - carry
    while t <= dist:
        f = t / dist
        pts.append((x0 + dx * f, y0 + dy * f))
        t += spacing
    leftover = (carry + dist) % spacing
    return pts, leftover


class StrokeRasterizer:
    """Accumulates stamps into a coverage buffer for one resolution level."""

    def __init__(self, height: int, width: int) -> None:
        self.height = height
        self.width = width
        self.coverage = np.zeros((height, width), dtype=np.float32)
        self._roi: Optional[ROI] = None       # everything stamped this stroke
        self._pending: Optional[ROI] = None   # stamped since the last take()

    def reset(self) -> None:
        if self._roi is not None:
            self._roi.slice(self.coverage)[:] = 0.0
        else:
            self.coverage[:] = 0.0
        self._roi = None
        self._pending = None

    @property
    def roi(self) -> Optional[ROI]:
        return self._roi

    @property
    def pending_roi(self) -> Optional[ROI]:
        return self._pending

    def take_pending(self) -> Optional[ROI]:
        """Return the region stamped since the previous call, and clear it."""
        r, self._pending = self._pending, None
        return r

    def stamp(self, cx: float, cy: float, radius: float, hardness: float) -> None:
        k = stamp_kernel(radius, hardness)
        ks = k.shape[0]
        half = ks // 2
        # Round to integer pixel centres; sub-pixel jitter is invisible here and
        # integer placement keeps the kernel cache small.
        x = int(round(cx)) - half
        y = int(round(cy)) - half

        sx0, sy0 = max(0, -x), max(0, -y)
        dx0, dy0 = max(0, x), max(0, y)
        dx1, dy1 = min(self.width, x + ks), min(self.height, y + ks)
        if dx1 <= dx0 or dy1 <= dy0:
            return
        sub = k[sy0 : sy0 + (dy1 - dy0), sx0 : sx0 + (dx1 - dx0)]
        dst = self.coverage[dy0:dy1, dx0:dx1]
        np.maximum(dst, sub, out=dst)

        r = ROI(dy0, dy1, dx0, dx1)
        self._roi = r if self._roi is None else self._roi.union(r)
        self._pending = r if self._pending is None else self._pending.union(r)

    def stamp_many(
        self, points: Iterable[Tuple[float, float]], radius: float, hardness: float
    ) -> None:
        for cx, cy in points:
            self.stamp(cx, cy, radius, hardness)

    def seed_mask(self, threshold: float = 0.5) -> np.ndarray:
        """Binary seed mask for the pixels this stroke covered."""
        return self.coverage >= threshold


class BrushStroke:
    """Tracks one press-drag-release gesture at a given resolution."""

    def __init__(self, brush: BrushConfig, scale: float, height: int, width: int):
        self.brush = brush
        self.scale = scale
        self.radius = max(0.5, brush.radius * scale)
        self.spacing = max(1.0, self.radius * 2.0 * brush.spacing)
        self.raster = StrokeRasterizer(height, width)
        self._last: Optional[Tuple[float, float]] = None
        self._carry = 0.0

    def begin(self, x: float, y: float) -> Optional[ROI]:
        self.raster.reset()
        self._last = (x, y)
        self._carry = 0.0
        self.raster.stamp(x, y, self.radius, self.brush.hardness)
        return self.raster.pending_roi

    def extend(self, x: float, y: float) -> Optional[ROI]:
        """Stamp along to ``(x, y)``; returns the newly covered region, if any."""
        if self._last is None:
            return self.begin(x, y)
        pts, self._carry = interpolate_points(
            self._last, (x, y), self.spacing, self._carry
        )
        self._last = (x, y)
        if not pts:
            return None
        self.raster.stamp_many(pts, self.radius, self.brush.hardness)
        return self.raster.pending_roi

    def take(self) -> Tuple[np.ndarray, Optional[ROI]]:
        """Consume the region stamped since the last call.

        The coverage buffer itself is cumulative for the whole stroke -- seeding
        an already-seeded pixel is idempotent -- but the ROI covers only the new
        stamps, which is what keeps the solve local.
        """
        roi = self.raster.take_pending()
        return self.raster.seed_mask(), roi

    @property
    def full_mask(self) -> np.ndarray:
        return self.raster.seed_mask()


def points_from_polyline(
    pts: Sequence[Tuple[float, float]], spacing: float
) -> List[Tuple[float, float]]:
    """Convenience for scripted / headless strokes."""
    if not pts:
        return []
    out: List[Tuple[float, float]] = [tuple(pts[0])]
    carry = 0.0
    for a, b in zip(pts, pts[1:]):
        seg, carry = interpolate_points(a, b, spacing, carry)
        out.extend(seg)
    return out
