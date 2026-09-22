"""Image container, layer stack and cached per-scale features.

The engine never touches raw RGB directly: it asks an :class:`ImagePyramid`
for a *feature image* (Lab by default) plus a set of pre-computed neighbour
("n-link") weights.  Pre-computing those weights once per image is what makes
the per-stroke local solve cheap enough to run inside a frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import EngineConfig

# Neighbour offsets.  Each is used once with ``symmetric=True``, which adds the
# reverse edge too -- so these four cover the full 8-neighbourhood.
OFFSETS_4: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0))
OFFSETS_8: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))


def offsets_for(neighborhood: int) -> Tuple[Tuple[int, int], ...]:
    return OFFSETS_8 if neighborhood == 8 else OFFSETS_4


def structure_for(dy: int, dx: int) -> np.ndarray:
    """3x3 structure matrix selecting the single neighbour at ``(dy, dx)``."""
    st = np.zeros((3, 3), dtype=np.float64)
    st[1 + dy, 1 + dx] = 1.0
    return st


# --------------------------------------------------------------------------- #
# Colour conversion
# --------------------------------------------------------------------------- #
def to_feature_space(rgb: np.ndarray, space: str) -> np.ndarray:
    """Convert uint8 RGB to a float32 feature image.

    All spaces are scaled to roughly comparable magnitudes (0..255-ish) so the
    same ``beta`` estimation and GMM covariance floors work everywhere.
    """
    space = space.lower()
    if space == "rgb":
        return rgb.astype(np.float32)
    if space == "lab":
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        # OpenCV 8-bit Lab is already 0..255 per channel.
        return lab
    if space == "hsv":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        # Hue is 0..179; stretch so it carries comparable weight.
        hsv[..., 0] *= 255.0 / 179.0
        return hsv
    raise ValueError(f"unknown color space {space!r}")


# --------------------------------------------------------------------------- #
# One resolution level
# --------------------------------------------------------------------------- #
class ScaleLevel:
    """Cached data for a single resolution of the image."""

    def __init__(
        self,
        rgb: np.ndarray,
        cfg: EngineConfig,
        scale: float,
        build_weights: bool = True,
    ) -> None:
        self.rgb = rgb
        self.cfg = cfg
        self.scale = scale  # full-res -> this level
        self.height, self.width = rgb.shape[:2]
        self.feat = to_feature_space(rgb, cfg.color_space)
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        self._gradient: Optional[np.ndarray] = None
        self.offsets = offsets_for(cfg.neighborhood)
        self.beta: float = 0.0
        self.weights: Dict[Tuple[int, int], np.ndarray] = {}
        if build_weights:
            self._build_weights()

    # -- neighbour weights ------------------------------------------------- #
    def _squared_diff(self, dy: int, dx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel ||I(p) - I(p+offset)||^2 and a validity mask."""
        h, w = self.height, self.width
        sq = np.zeros((h, w), dtype=np.float32)
        valid = np.zeros((h, w), dtype=bool)
        ys0, ys1 = max(0, -dy), min(h, h - dy)
        xs0, xs1 = max(0, -dx), min(w, w - dx)
        if ys0 >= ys1 or xs0 >= xs1:
            return sq, valid
        a = self.feat[ys0:ys1, xs0:xs1]
        b = self.feat[ys0 + dy : ys1 + dy, xs0 + dx : xs1 + dx]
        d = a - b
        sq[ys0:ys1, xs0:xs1] = np.einsum("ijk,ijk->ij", d, d)
        valid[ys0:ys1, xs0:xs1] = True
        return sq, valid

    def _build_weights(self) -> None:
        """w(i,j) = gamma * exp(-beta ||I_i - I_j||^2) / dist(i,j).

        ``beta`` is the adaptive normaliser from Boykov & Jolly / GrabCut:
        ``beta = 1 / (2 * E[||I_i - I_j||^2])``.  It makes the contrast term
        scale-free, so the same gamma works on flat and on busy images.
        """
        sqs: Dict[Tuple[int, int], np.ndarray] = {}
        valids: Dict[Tuple[int, int], np.ndarray] = {}
        total = 0.0
        count = 0
        for off in self.offsets:
            sq, valid = self._squared_diff(*off)
            sqs[off] = sq
            valids[off] = valid
            total += float(sq[valid].sum())
            count += int(valid.sum())
        mean_sq = total / max(count, 1)
        self.beta = 1.0 / (2.0 * mean_sq) if mean_sq > 1e-8 else 0.0

        gamma = self.cfg.gamma_smooth
        for off in self.offsets:
            dist = float(np.hypot(off[0], off[1]))
            w = (gamma / dist) * np.exp(-self.beta * sqs[off], dtype=np.float32)
            w[~valids[off]] = 0.0
            self.weights[off] = np.ascontiguousarray(w, dtype=np.float64)

    # -- lazily-computed extras -------------------------------------------- #
    @property
    def gradient(self) -> np.ndarray:
        """Normalised Sobel gradient magnitude, used by Auto-Enhance."""
        if self._gradient is None:
            blur = cv2.GaussianBlur(self.gray, (0, 0), 1.0)
            gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            m = float(mag.max())
            self._gradient = mag / m if m > 1e-6 else mag
        return self._gradient

    def weight_slice(self, off: Tuple[int, int], roi: "ROI") -> np.ndarray:
        """Weights for ``off`` cropped to ``roi``.

        Edges that would leave the ROI are zeroed: the local solve must not
        reference nodes it does not own.
        """
        w = self.weights[off][roi.y0 : roi.y1, roi.x0 : roi.x1].copy()
        dy, dx = off
        # An edge leaves node (y, x) for (y + dy, x + dx); zero it whenever the
        # destination falls outside the cropped array.
        if dy > 0:
            w[-dy:, :] = 0.0
        elif dy < 0:
            w[:-dy, :] = 0.0
        if dx > 0:
            w[:, -dx:] = 0.0
        elif dx < 0:
            w[:, :-dx] = 0.0
        return w


# --------------------------------------------------------------------------- #
# Layer stack
# --------------------------------------------------------------------------- #
@dataclass
class Layer:
    name: str
    rgb: np.ndarray  # uint8 HxWx3
    alpha: Optional[np.ndarray] = None  # uint8 HxW, None = opaque
    visible: bool = True


class LayerStack:
    """A minimal layer model, enough to implement "Sample All Layers".

    Layers are composited bottom-to-top with simple source-over alpha.  A real
    editor would support blend modes; the selection engine only needs pixels.
    """

    def __init__(self, layers: Sequence[Layer]) -> None:
        if not layers:
            raise ValueError("LayerStack needs at least one layer")
        h, w = layers[0].rgb.shape[:2]
        for lyr in layers:
            if lyr.rgb.shape[:2] != (h, w):
                raise ValueError("all layers must share the same size")
        self.layers: List[Layer] = list(layers)
        self.height, self.width = h, w
        self.active_index = len(self.layers) - 1

    @classmethod
    def from_image(cls, rgb: np.ndarray, name: str = "Background") -> "LayerStack":
        return cls([Layer(name=name, rgb=rgb)])

    @property
    def active(self) -> Layer:
        return self.layers[self.active_index]

    def composite(self) -> np.ndarray:
        out = np.zeros((self.height, self.width, 3), dtype=np.float32)
        filled = False
        for lyr in self.layers:
            if not lyr.visible:
                continue
            src = lyr.rgb.astype(np.float32)
            if lyr.alpha is None:
                out = src
                filled = True
            else:
                a = (lyr.alpha.astype(np.float32) / 255.0)[..., None]
                out = src * a + out * (1.0 - a)
                filled = True
        if not filled:
            out = self.layers[0].rgb.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def sample_source(self, sample_all_layers: bool) -> np.ndarray:
        """Pixels the selection should be computed from."""
        if sample_all_layers or len(self.layers) == 1:
            return self.composite()
        return self.active.rgb


# --------------------------------------------------------------------------- #
# ROI helper
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ROI:
    y0: int
    y1: int
    x0: int
    x1: int

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.y1 - self.y0, self.x1 - self.x0)

    @property
    def area(self) -> int:
        return (self.y1 - self.y0) * (self.x1 - self.x0)

    def slice(self, arr: np.ndarray) -> np.ndarray:
        return arr[self.y0 : self.y1, self.x0 : self.x1]

    def clipped(self, h: int, w: int) -> "ROI":
        return ROI(
            max(0, self.y0), min(h, self.y1), max(0, self.x0), min(w, self.x1)
        )

    def expanded(self, pad: int, h: int, w: int) -> "ROI":
        return ROI(self.y0 - pad, self.y1 + pad, self.x0 - pad, self.x1 + pad).clipped(
            h, w
        )

    @staticmethod
    def from_mask(mask: np.ndarray) -> Optional["ROI"]:
        ys = np.flatnonzero(mask.any(axis=1))
        if ys.size == 0:
            return None
        xs = np.flatnonzero(mask.any(axis=0))
        return ROI(int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1)

    def union(self, other: "ROI") -> "ROI":
        return ROI(
            min(self.y0, other.y0),
            max(self.y1, other.y1),
            min(self.x0, other.x0),
            max(self.x1, other.x1),
        )


# --------------------------------------------------------------------------- #
# Pyramid
# --------------------------------------------------------------------------- #
class ImagePyramid:
    """Full-resolution image plus the down-scaled level used interactively."""

    def __init__(self, rgb_full: np.ndarray, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.full_rgb = np.ascontiguousarray(rgb_full)
        self.full_h, self.full_w = rgb_full.shape[:2]

        scale = self._fit_scale(cfg.interactive_max_dim)
        if scale < 1.0:
            small = cv2.resize(
                rgb_full,
                (max(1, int(round(self.full_w * scale))),
                 max(1, int(round(self.full_h * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            small = rgb_full
            scale = 1.0
        self.interactive = ScaleLevel(small, cfg, scale)

        self._coarse: Optional[ScaleLevel] = None
        self._refine: Optional[ScaleLevel] = None
        self._full_guide: Optional[np.ndarray] = None

    def _fit_scale(self, max_dim: int) -> float:
        if max_dim <= 0:
            return 1.0
        longest = max(self.full_h, self.full_w)
        return min(1.0, max_dim / float(longest))

    @property
    def coarse(self) -> ScaleLevel:
        """Half-resolution level used for large local solves.

        When a brush step has to search a big region -- the adaptive expansion
        that happens on mouse-down inside a large object -- doing it at half
        resolution cuts the node count by four.  The boundary it produces is
        half a pixel coarser, which nothing downstream can see: the on-screen
        preview is up-sampled anyway and mouse-release re-cuts the boundary at
        full resolution.
        """
        if self._coarse is None:
            lvl = self.interactive
            ch = max(1, lvl.height // 2)
            cw = max(1, lvl.width // 2)
            small = cv2.resize(lvl.rgb, (cw, ch), interpolation=cv2.INTER_AREA)
            self._coarse = ScaleLevel(small, self.cfg, lvl.scale * 0.5)
        return self._coarse

    @property
    def coarse_ratio(self) -> float:
        """Interactive-scale pixels -> coarse-scale pixels."""
        return self.coarse.width / float(self.interactive.width)

    @property
    def refine(self) -> ScaleLevel:
        """Higher-resolution level, built on demand (mouse-release path)."""
        if self._refine is None:
            scale = self._fit_scale(self.cfg.refine_max_dim)
            if scale >= 1.0:
                lvl_rgb, scale = self.full_rgb, 1.0
            else:
                lvl_rgb = cv2.resize(
                    self.full_rgb,
                    (max(1, int(round(self.full_w * scale))),
                     max(1, int(round(self.full_h * scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            self._refine = ScaleLevel(lvl_rgb, self.cfg, scale)
        return self._refine

    @property
    def full_guide(self) -> np.ndarray:
        """Float32 0..1 RGB guide image for the guided filter."""
        if self._full_guide is None:
            self._full_guide = self.full_rgb.astype(np.float32) / 255.0
        return self._full_guide

    # -- coordinate helpers ------------------------------------------------- #
    def to_interactive(self, x: float, y: float) -> Tuple[float, float]:
        s = self.interactive.scale
        return x * s, y * s

    def interactive_radius(self, full_radius: float) -> float:
        return max(0.5, full_radius * self.interactive.scale)

    def upscale_mask(self, mask: np.ndarray, smooth: bool = True) -> np.ndarray:
        """Interactive-scale binary mask -> full-resolution float alpha 0..1."""
        src = mask.astype(np.float32)
        if src.shape[:2] == (self.full_h, self.full_w):
            return src
        interp = cv2.INTER_LINEAR if smooth else cv2.INTER_NEAREST
        return cv2.resize(src, (self.full_w, self.full_h), interpolation=interp)

    def downscale_mask(self, mask: np.ndarray) -> np.ndarray:
        """Full-resolution mask -> interactive-scale bool mask."""
        lvl = self.interactive
        if mask.shape[:2] == (lvl.height, lvl.width):
            return mask.astype(bool)
        small = cv2.resize(
            mask.astype(np.float32),
            (lvl.width, lvl.height),
            interpolation=cv2.INTER_AREA,
        )
        return small > 0.5
