"""The Quick Selection engine.

Behavioural model
-----------------
Photoshop's Quick Selection Tool feels the way it does because of three
decisions, all of which are reproduced here:

1. **The solve is local.**  Painting never re-segments the whole image; it
   optimises a bounded region around the brush.  This is the core idea of
   *Paint Selection* (Liu, Sun & Shum, SIGGRAPH 2009).  It is what makes the
   tool both fast and predictable -- a stroke in the top-left corner cannot
   make the bottom-right corner change its mind.

2. **Background is inferred, not asked for.**  The user only paints what they
   want.  Everything beyond a reach radius around the stroke and the current
   selection is treated as hard background for that step, which both bounds the
   flood and supplies negative colour samples.  Alt-painting adds explicit
   background seeds on top of that.

3. **Growth is monotone within a gesture.**  In Add mode the local result is
   unioned into the selection.  The selection never flickers backwards while
   you drag, which is the single biggest difference between a tool that feels
   solid and one that feels twitchy.

The per-step pipeline is:

    stamp brush  ->  update trimap  ->  pick ROI  ->  infer background band
                 ->  fit FG/BG colour models  ->  build graph  ->  min-cut
                 ->  clean up  ->  merge into the selection

and on mouse release a narrow-band refinement runs at full resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .brush import BrushStroke
from .colormodel import build_model
from .config import (
    BACKGROUND,
    FOREGROUND,
    UNKNOWN,
    BrushConfig,
    EngineConfig,
    RefineEdgeConfig,
    ToolState,
)
from .graphcut import (
    clean_components,
    estimate_beta,
    fill_small_holes,
    solve_band_min_cut,
    solve_min_cut,
)
from .imagedata import ROI, ImagePyramid, Layer, LayerStack, to_feature_space
from .refine import apply_refine_edge, guided_filter


class SelectionMode(Enum):
    NEW = "new"
    ADD = "add"
    SUBTRACT = "subtract"
    INTERSECT = "intersect"


@dataclass
class SelectionUpdate:
    """What changed, so the UI can repaint the smallest possible area."""

    mask: np.ndarray                 # interactive-scale bool selection
    roi: Optional[ROI] = None        # interactive-scale region touched
    final: bool = False              # True once the release refinement ran
    alpha: Optional[np.ndarray] = None  # full-res float32 0..1, only when final


def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


#: Above this radius a structuring-element morphology becomes the bottleneck
#: (a 91x91 ellipse kernel costs milliseconds per call), so we switch to a
#: distance transform, which is O(N) regardless of radius.
_DT_THRESHOLD = 6


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask
    if radius <= _DT_THRESHOLD:
        return cv2.dilate(mask.astype(np.uint8), _disk(radius)) > 0
    if mask.all():
        return mask
    # distanceTransform measures distance to the nearest *zero*, so invert.
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return dist <= float(radius)


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any():
        return mask
    if radius <= _DT_THRESHOLD:
        return cv2.erode(mask.astype(np.uint8), _disk(radius)) > 0
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3)
    return dist > float(radius)


def _resize_mask(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resample a boolean mask, keeping anything with majority coverage."""
    if mask.shape[:2] == tuple(shape):
        return mask.copy()
    small = cv2.resize(
        mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_AREA
    )
    return small > 0


def _resize_labels(labels: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """Resample a trimap by label coverage.

    Nearest-neighbour would be wrong: a one-pixel brush line downscaled by two
    lands on a quarter of a target pixel and nearest sampling drops it most of
    the time, so a seed the user painted would simply vanish from the coarse
    solve.  Each label is resampled by area coverage instead, any coverage
    counts, and the stronger coverage wins a tie.
    """
    if labels.shape[:2] == tuple(shape):
        return labels.copy()
    fg = (labels == FOREGROUND).astype(np.float32)
    bg = (labels == BACKGROUND).astype(np.float32)
    size = (shape[1], shape[0])
    fg = (cv2.resize(fg, size, interpolation=cv2.INTER_AREA)
          if fg.any() else np.zeros(shape, np.float32))
    bg = (cv2.resize(bg, size, interpolation=cv2.INTER_AREA)
          if bg.any() else np.zeros(shape, np.float32))
    eps = 1e-3
    out = np.full(shape, UNKNOWN, dtype=np.uint8)
    out[(fg > eps) & (fg >= bg)] = FOREGROUND
    out[(bg > eps) & (bg > fg)] = BACKGROUND
    return out


# --------------------------------------------------------------------------- #
class QuickSelectEngine:
    """Stateful selection engine driven by brush gestures."""

    def __init__(
        self,
        layers: LayerStack,
        cfg: Optional[EngineConfig] = None,
        tool: Optional[ToolState] = None,
    ) -> None:
        self.cfg = cfg or EngineConfig()
        self.tool = tool or ToolState()
        self.layers = layers
        self._source_rgb: Optional[np.ndarray] = None
        self._full_feat: Optional[np.ndarray] = None
        self._full_beta: Optional[float] = None
        self.pyramid: ImagePyramid
        self._rebuild_source()

        lvl = self.pyramid.interactive
        self.trimap = np.full((lvl.height, lvl.width), UNKNOWN, dtype=np.uint8)
        self.selection = np.zeros((lvl.height, lvl.width), dtype=bool)
        self.full_alpha = np.zeros(
            (self.pyramid.full_h, self.pyramid.full_w), dtype=np.float32
        )

        self._stroke: Optional[BrushStroke] = None
        self._stroke_mode: SelectionMode = SelectionMode.ADD
        self._pre_stroke_state: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._dirty_full = False

    # ------------------------------------------------------------------ #
    # Source image handling
    # ------------------------------------------------------------------ #
    def _rebuild_source(self) -> None:
        rgb = self.layers.sample_source(self.tool.sample_all_layers)
        if (
            self._source_rgb is not None
            and rgb.shape == self._source_rgb.shape
            and np.array_equal(rgb, self._source_rgb)
        ):
            return  # identical pixels: keep the cached pyramid and weights
        self._source_rgb = rgb
        self.pyramid = ImagePyramid(rgb, self.cfg)
        self._full_feat = None
        self._full_beta = None

    def refresh_source(self, sample_all_layers: Optional[bool] = None) -> None:
        """Re-read pixels from the layer stack, keeping the current selection.

        Called when a layer is added, hidden or made active, and when the
        "Sample All Layers" option changes.  The selection survives because the
        image dimensions do not change; only the pixels the models see do.
        """
        if sample_all_layers is not None:
            self.tool.sample_all_layers = sample_all_layers
        old_sel = self.selection.copy()
        old_tri = self.trimap.copy()
        old_alpha = self.full_alpha
        self._rebuild_source()
        lvl = self.pyramid.interactive
        if old_sel.shape != (lvl.height, lvl.width):
            self.trimap = np.full((lvl.height, lvl.width), UNKNOWN, dtype=np.uint8)
            self.selection = np.zeros((lvl.height, lvl.width), dtype=bool)
            self.full_alpha = np.zeros(
                (self.pyramid.full_h, self.pyramid.full_w), dtype=np.float32
            )
        else:
            self.selection, self.trimap, self.full_alpha = old_sel, old_tri, old_alpha

    # Backwards-compatible alias.
    def set_sample_all_layers(self, value: bool) -> None:
        if value == self.tool.sample_all_layers:
            return
        self.refresh_source(value)

    @property
    def full_feat(self) -> np.ndarray:
        if self._full_feat is None:
            self._full_feat = to_feature_space(
                self.pyramid.full_rgb, self.cfg.color_space
            )
        return self._full_feat

    @property
    def full_beta(self) -> float:
        if self._full_beta is None:
            self._full_beta = estimate_beta(self.full_feat)
        return self._full_beta

    # ------------------------------------------------------------------ #
    # Gesture API -- coordinates are in full-resolution image space
    # ------------------------------------------------------------------ #
    def begin_stroke(
        self, x: float, y: float, mode: SelectionMode = SelectionMode.ADD
    ) -> SelectionUpdate:
        lvl = self.pyramid.interactive
        # Captured *before* anything changes, so undo returns the exact state
        # the canvas had at mouse-down -- one undo step per gesture.
        self._pre_stroke_state = self.snapshot()
        self._stroke_mode = mode
        if mode == SelectionMode.NEW:
            self.selection[:] = False
            self.trimap[:] = UNKNOWN
        self._stroke = BrushStroke(
            self.tool.brush, lvl.scale, lvl.height, lvl.width
        )
        sx, sy = self.pyramid.to_interactive(x, y)
        self._stroke.begin(sx, sy)
        return self._step()

    def continue_stroke(self, x: float, y: float) -> Optional[SelectionUpdate]:
        if self._stroke is None:
            return None
        sx, sy = self.pyramid.to_interactive(x, y)
        if self._stroke.extend(sx, sy) is None:
            return None  # brush has not travelled a full spacing step yet
        return self._step()

    def end_stroke(self, refine: bool = True) -> SelectionUpdate:
        self._stroke = None
        if refine:
            self.refine_now()
        return SelectionUpdate(
            mask=self.selection, final=True, alpha=self.full_alpha
        )

    def _step(self) -> SelectionUpdate:
        assert self._stroke is not None
        stroke_mask, pending = self._stroke.take()
        if pending is None:
            return SelectionUpdate(mask=self.selection)
        roi = self._local_step(stroke_mask, pending, self._stroke_mode)
        self._dirty_full = True
        return SelectionUpdate(mask=self.selection, roi=roi)

    # ------------------------------------------------------------------ #
    # The local optimisation
    # ------------------------------------------------------------------ #
    def _local_step(
        self, stroke_mask: np.ndarray, pending: ROI, mode: SelectionMode
    ) -> ROI:
        """One brush step: seed, infer background, cut, merge.

        The region solved is chosen *adaptively*.  We start with a reach of a
        few brush radii; if the resulting foreground presses against the edge
        of that region, the region clearly does not contain the whole object,
        so we grow it and solve again (Paint Selection, section 4.1).  Painting
        inside a large flat object therefore leaps out to its real edges, while
        painting on a small one converges on the first try and costs nothing
        extra.
        """
        cfg = self.cfg
        lvl = self.pyramid.interactive
        h, w = lvl.height, lvl.width

        subtracting = mode == SelectionMode.SUBTRACT
        seed_label = BACKGROUND if subtracting else FOREGROUND

        # 1. Record the new brush pixels in the trimap (hard constraints).
        tri_pending = pending.slice(self.trimap)
        tri_pending[pending.slice(stroke_mask)] = seed_label

        brush_r = self.pyramid.interactive_radius(self.tool.brush.radius)
        base_reach = cfg.reach_radius(brush_r)
        max_area = cfg.max_solve_pixels

        roi = pending.expanded(base_reach + cfg.roi_padding, h, w)
        result: Optional[np.ndarray] = None
        allowed: Optional[np.ndarray] = None
        cached_fg = None

        # Subtracting never expands.  Growing the region while removing would
        # let one careless stroke eat far more of the selection than the brush
        # covered, which is the opposite of what "subtract" should feel like.
        attempts = 1 if subtracting else max(1, cfg.max_expand_steps)
        for attempt in range(attempts):
            reach = int(round(base_reach * (cfg.expand_factor ** attempt)))
            next_roi = pending.expanded(reach + cfg.roi_padding, h, w)
            if attempt > 0 and next_roi.area > max_area:
                break  # keep the interactive path inside its time budget
            roi = next_roi

            outcome = self._solve_local_region(
                roi, stroke_mask, reach, subtracting, fg_model=cached_fg
            )
            if outcome is None:
                return roi
            result, allowed, pressure, cached_fg = outcome
            if pressure < cfg.expand_pressure:
                break  # the object fits inside this region; stop growing
            if roi.area >= max_area:
                break

        if result is None or allowed is None:
            return roi

        sel_sub = roi.slice(self.selection)
        if subtracting:
            sel_sub &= ~result
        else:
            sel_sub |= result
        return roi

    def _solve_local_region(
        self,
        roi: ROI,
        stroke_mask: np.ndarray,
        reach: int,
        subtracting: bool,
        fg_model=None,
    ) -> Optional[Tuple[np.ndarray, np.ndarray, float, object]]:
        """Run one min-cut inside ``roi``.

        Returns ``(result, allowed, pressure, fg_model)`` where *pressure* is
        the fraction of the *newly reachable* frontier the result touches --
        the signal used to decide whether the region needs to grow.  The fitted
        foreground model is handed back so repeated expansion attempts within
        one brush step do not refit it.
        """
        cfg = self.cfg
        sel_sub_fine = roi.slice(self.selection)

        # Large regions are solved at half resolution; see ImagePyramid.coarse.
        coarse = roi.area > cfg.coarse_solve_threshold
        if coarse:
            lvl = self.pyramid.coarse
            r = self.pyramid.coarse_ratio
            croi = ROI(
                int(roi.y0 * r), max(int(roi.y0 * r) + 2, int(round(roi.y1 * r))),
                int(roi.x0 * r), max(int(roi.x0 * r) + 2, int(round(roi.x1 * r))),
            ).clipped(lvl.height, lvl.width)
            shape_c = croi.shape
            stroke_sub = _resize_mask(roi.slice(stroke_mask), shape_c)
            sel_sub = _resize_mask(sel_sub_fine, shape_c)
            tri_sub = _resize_labels(roi.slice(self.trimap), shape_c)
            feat_sub = croi.slice(lvl.feat)
            solve_roi = croi
            reach = max(2, int(round(reach * r)))
        else:
            lvl = self.pyramid.interactive
            croi = roi
            stroke_sub = roi.slice(stroke_mask)
            sel_sub = sel_sub_fine
            tri_sub = roi.slice(self.trimap)
            feat_sub = roi.slice(lvl.feat)
            solve_roi = roi

        reach_mask = _dilate(stroke_sub, reach)

        if subtracting:
            # Removing: the flood may only eat into the existing selection, and
            # never into pixels the user explicitly painted as foreground.
            allowed = reach_mask & sel_sub
            pos_seed = (tri_sub == BACKGROUND) & reach_mask
            hard_neg = (tri_sub == FOREGROUND) | ~_dilate(allowed, 2)
            # "Keep" samples come from the part of the selection the brush is
            # not touching -- that is what the user is preserving.
            neg_sample_mask = (sel_sub & ~reach_mask) | (tri_sub == FOREGROUND)
        else:
            # Adding: the flood may occupy the reach region or anything already
            # selected.  Note the reach boundary is *not* a hard background
            # wall: min-cut prefers short boundaries, so a wall of background
            # right next to the brush would make the cheapest cut the one that
            # hugs the bristles, and the selection would never grow.  Instead
            # the result is clipped to the reach afterwards.
            allowed = reach_mask | sel_sub
            pos_seed = (tri_sub == FOREGROUND) & allowed
            hard_neg = tri_sub == BACKGROUND
            neg_sample_mask = None

        # Background band: a ring just outside the allowed region.  Sampling a
        # ring rather than the whole exterior keeps the negative model local --
        # that is what stops a red shirt being rejected because there happens
        # to be a red car elsewhere in the frame.
        band_w = max(4, int(round(reach * cfg.bg_band_factor)))
        outer = _dilate(allowed, band_w)
        ring = outer & ~allowed

        # Everything past the ring is pinned to background.  Two reasons: it
        # bounds the cut, and it gives max-flow a wide sink to drain into --
        # with only a one-pixel frame the augmenting paths run the width of the
        # region and the solve costs several times more.  The wall sits a full
        # band beyond the region the result is clipped to, far enough that
        # min-cut's preference for short boundaries cannot pull the selection
        # back onto the bristles.
        hard_neg = hard_neg | (~outer & ~pos_seed)
        border = np.zeros(solve_roi.shape, dtype=bool)
        border[0, :] = border[-1, :] = True
        border[:, 0] = border[:, -1] = True
        hard_neg = hard_neg | (border & ~pos_seed)
        pos_seed = pos_seed & ~(tri_sub == (FOREGROUND if subtracting else BACKGROUND))

        if not pos_seed.any():
            return None
        if neg_sample_mask is None:
            neg_sample_mask = ring | (tri_sub == BACKGROUND)
        if not neg_sample_mask.any():
            neg_sample_mask = ~allowed
        if not neg_sample_mask.any():
            grown = _resize_mask(allowed, roi.shape) if coarse else allowed.copy()
            return grown, grown, 0.0, fg_model

        # Colour models.
        if fg_model is None:
            pos_sample_mask = pos_seed
            if not subtracting and sel_sub.any():
                # Include the interior of what is already selected so the model
                # stays stable as the stroke crosses a textured object.
                pos_sample_mask = pos_seed | _erode(sel_sub, 2)
            fg_model = build_model(feat_sub[pos_sample_mask], cfg, seed=1)

        # How trustworthy is the *inferred* background?  When the brush is deep
        # inside a large object the ring is still object, so a model fitted to
        # it would be indistinguishable from the foreground model and the data
        # term would carry no information.  Detect that by scoring the negative
        # samples under the foreground model: if they look like foreground, drop
        # the background model and fall back to a plain likelihood threshold,
        # which behaves like an edge-stopped flood fill.
        neg_px = feat_sub[neg_sample_mask]
        neg_under_fg = float(np.mean(fg_model.negative_log_likelihood(neg_px)))
        trust_bg = neg_under_fg >= cfg.bg_trust_threshold

        unknown = ~(pos_seed | hard_neg)
        shape = solve_roi.shape
        fg_nll = np.zeros(shape, dtype=np.float32)
        bg_nll = np.full(shape, cfg.fg_bias, dtype=np.float32)
        if unknown.any():
            px = feat_sub[unknown]
            fg_nll[unknown] = fg_model.negative_log_likelihood(px)
            if trust_bg:
                bg_model = build_model(neg_px, cfg, seed=2)
                bg_nll[unknown] = np.maximum(
                    bg_model.negative_log_likelihood(px), cfg.bg_nll_floor
                )

        result = solve_min_cut(lvl, solve_roi, pos_seed, hard_neg, fg_nll, bg_nll, cfg)

        # Measure how hard the result pushes on the frontier, *excluding* parts
        # of the frontier that are already selected.  Without that exclusion a
        # finished selection keeps reporting pressure at its own boundary and
        # every subsequent brush step would pay for a pointless expansion.
        frontier = allowed & ~_erode(allowed, 2) & ~sel_sub
        front_n = int(frontier.sum())
        pressure = float((result & frontier).sum()) / front_n if front_n else 0.0

        result &= allowed
        min_area = cfg.min_component_area
        if coarse:
            min_area = max(4, min_area // 4)
        result = clean_components(result, keep_seed=pos_seed, min_area=min_area)
        result = fill_small_holes(result, max_area=min_area * 4)
        result |= pos_seed

        if coarse:
            result = _resize_mask(result, roi.shape)
            allowed = _resize_mask(allowed, roi.shape)
            result &= allowed
        return result, allowed, pressure, fg_model

    # ------------------------------------------------------------------ #
    # Release-time refinement
    # ------------------------------------------------------------------ #
    def refine_now(self) -> np.ndarray:
        """Produce the full-resolution alpha for the current selection."""
        mask_full = self._upsample_selection()
        if mask_full.any():
            mask_full = self._refine_boundary(
                mask_full,
                band=self._upsample_band(),
                edge_gain=1.0,
            )
            if self.tool.auto_enhance:
                mask_full = self._refine_boundary(
                    mask_full,
                    band=self.cfg.auto_enhance_band,
                    edge_gain=self.cfg.auto_enhance_edge_gain,
                    use_models=False,
                )
        alpha = self._to_alpha(mask_full)
        self.full_alpha = alpha
        # Keep the interactive mask consistent with what the user now sees.
        self.selection = self.pyramid.downscale_mask(alpha > 0.5)
        self._dirty_full = False
        return alpha

    def _upsample_band(self) -> int:
        """Band half-width that covers the up-sampling error, in full-res px."""
        s = self.pyramid.interactive.scale
        return int(np.clip(round(2.5 / max(s, 1e-3)), 4, 48))

    def _upsample_selection(self) -> np.ndarray:
        up = self.pyramid.upscale_mask(self.selection, smooth=True)
        return up > 0.5

    def _refine_boundary(
        self,
        mask_full: np.ndarray,
        band: int,
        edge_gain: float = 1.0,
        use_models: bool = True,
    ) -> np.ndarray:
        """Re-cut a narrow band around the boundary at full resolution.

        The interior and the far exterior are pinned, so this can only move the
        boundary -- it can never delete a region the user painted.  That
        guarantee is what makes it safe to run automatically on mouse release.
        """
        cfg = self.cfg
        inner = _erode(mask_full, band)
        outer = _dilate(mask_full, band)
        band_mask = outer & ~inner
        if not band_mask.any():
            return mask_full

        region = ROI.from_mask(outer)
        if region is None:
            return mask_full
        region = region.expanded(2, self.pyramid.full_h, self.pyramid.full_w)

        feat = region.slice(self.full_feat)
        band_c = region.slice(band_mask)
        inner_c = region.slice(inner)
        outer_c = region.slice(outer)
        fixed_fg = inner_c
        fixed_bg = ~outer_c
        # Explicit user strokes, lifted to full resolution, always win.
        stroke_fg = region.slice(
            self.pyramid.upscale_mask(self.trimap == FOREGROUND, smooth=False) > 0.5
        )
        stroke_bg = region.slice(
            self.pyramid.upscale_mask(self.trimap == BACKGROUND, smooth=False) > 0.5
        )
        # Explicit brush marks outrank the derived inner/outer bands, in both
        # directions.  The earlier ordering let ``inner`` (an eroded copy of
        # the current mask) win over an Alt-painted pixel, so a background mark
        # that happened to sit inside the selection was silently flipped back
        # to foreground on the next refinement -- the marks and the mask were
        # two sources of truth and the mask won.  Masking each derived band by
        # the *opposite* explicit set removes that: a mark can only be undone
        # by painting over it.
        fixed_fg = (fixed_fg | stroke_fg) & ~stroke_bg
        fixed_bg = (fixed_bg | stroke_bg) & ~stroke_fg
        band_c = band_c & ~fixed_fg & ~fixed_bg
        if not band_c.any():
            return mask_full

        fg_nll = bg_nll = None
        if use_models:
            fg_src = _erode(inner_c, 1)
            bg_src = fixed_bg & _dilate(outer_c, band)
            if fg_src.any() and bg_src.any():
                fg_model = build_model(feat[fg_src], cfg, seed=3)
                bg_model = build_model(feat[bg_src], cfg, seed=4)
                fg_nll = np.zeros(band_c.shape, dtype=np.float32)
                bg_nll = np.zeros(band_c.shape, dtype=np.float32)
                px = feat[band_c]
                fg_nll[band_c] = fg_model.negative_log_likelihood(px)
                bg_nll[band_c] = bg_model.negative_log_likelihood(px)

        solved = solve_band_min_cut(
            feat,
            band_c,
            fixed_fg,
            fixed_bg,
            fg_nll,
            bg_nll,
            self.full_beta,
            cfg,
            edge_gain=edge_gain,
        )
        out = mask_full.copy()
        region.slice(out)[:] = solved
        return out

    def _to_alpha(self, mask_full: np.ndarray) -> np.ndarray:
        """Binary full-res mask -> soft alpha, then apply Refine Edge."""
        alpha = mask_full.astype(np.float32)
        refine = self.tool.refine
        if refine.edge_aware and mask_full.any():
            alpha = guided_filter(
                self.pyramid.full_guide,
                alpha,
                radius=self.cfg.guided_radius,
                eps=self.cfg.guided_eps,
            )
            alpha = np.clip((alpha - 0.35) / 0.3, 0.0, 1.0)
        alpha = apply_refine_edge(alpha, refine)

        # Re-assert the brush marks last.  The guided filter and the Refine
        # Edge chain both soften the boundary deliberately, which lets alpha
        # bleed a pixel or two across a small Alt-painted hole and flip it back
        # to foreground.  Fractional coverage is right for pixels the algorithm
        # chose; it is not right for pixels the user marked, so those are
        # stamped back to hard 1 and 0 here.
        if self.trimap.any():
            fg = self.pyramid.upscale_mask(self.trimap == FOREGROUND, smooth=False)
            bg = self.pyramid.upscale_mask(self.trimap == BACKGROUND, smooth=False)
            alpha[fg > 0.5] = 1.0
            alpha[bg > 0.5] = 0.0
        return alpha

    # ------------------------------------------------------------------ #
    # Direct manipulation helpers (used by the UI and by scripts)
    # ------------------------------------------------------------------ #
    def paint_polyline(
        self,
        points: Sequence[Tuple[float, float]],
        mode: SelectionMode = SelectionMode.ADD,
        refine: bool = True,
    ) -> SelectionUpdate:
        """Headless equivalent of a mouse drag; handy for tests and scripts."""
        if not points:
            return SelectionUpdate(mask=self.selection)
        self.begin_stroke(points[0][0], points[0][1], mode)
        for x, y in points[1:]:
            self.continue_stroke(x, y)
        return self.end_stroke(refine=refine)

    def clear(self) -> None:
        self.selection[:] = False
        self.trimap[:] = UNKNOWN
        self.full_alpha[:] = 0.0
        self._dirty_full = False

    def invert(self) -> None:
        self.selection = ~self.selection
        self.trimap[:] = UNKNOWN
        self.full_alpha = 1.0 - self.full_alpha

    def selection_alpha(self) -> np.ndarray:
        """Best available full-resolution alpha (refines lazily if stale)."""
        if self._dirty_full:
            self.refine_now()
        return self.full_alpha

    def preview_alpha(self) -> np.ndarray:
        """Cheap full-res alpha for display mid-drag (no graph cut)."""
        return self.pyramid.upscale_mask(self.selection, smooth=True)

    # -- state snapshots for undo/redo ------------------------------------ #
    def snapshot(self) -> dict:
        return {
            "selection": np.packbits(self.selection),
            "trimap": self.trimap.copy(),
            "shape": self.selection.shape,
            "alpha": (self.full_alpha * 255.0).astype(np.uint8),
            "dirty": self._dirty_full,
        }

    def restore(self, state: dict) -> None:
        shape = state["shape"]
        n = shape[0] * shape[1]
        self.selection = (
            np.unpackbits(state["selection"])[:n].astype(bool).reshape(shape)
        )
        self.trimap = state["trimap"].copy()
        self.full_alpha = state["alpha"].astype(np.float32) / 255.0
        self._dirty_full = bool(state["dirty"])

    def pre_stroke_snapshot(self) -> Optional[dict]:
        """Snapshot captured at mouse-down, for a one-stroke undo step."""
        return self._pre_stroke_state
