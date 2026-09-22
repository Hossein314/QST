"""Segmentation as a pure function of (image, constraints, params).

    mask = segment(image, constraints, params)

That signature is the contract. The mask is *derived*: nothing here writes to
the constraint matrix, and two calls with the same inputs give the same output.
Constraints are applied as infinite-capacity terminal edges inside the solve
and stamped onto the result again afterwards, so a mark always binds even if
the solve never looked at its neighbourhood.

Making that fast enough to run on every mouse-move takes four caches, all of
which are keyed so that a stale one can never change the answer:

``ImageCache``
    Per-image work: the downscaled working image, its Lab features, the
    contrast normaliser ``beta``, and the neighbour ("n-link") edge weights.
    These depend only on the image and the graph parameters, so they are
    computed once and reused by every stroke for the life of the image.

``ColorModels``
    Foreground/background colour models built from the constraint pixels. The
    histogram variant is genuinely incremental -- raw bin counts accumulate as
    you paint, so adding a stroke costs a ``bincount`` over the new pixels
    only. Rebuilding the per-pixel cost arrays is *throttled*: it happens when
    the sample set has grown by a set fraction, not on every stamp. That is
    what makes the next cache useful.

``_NllCache``
    The per-pixel data costs. Recomputed only when the models are rebuilt.

``WarmGraph``
    The max-flow graph itself. When the graph structure is unchanged, only the
    terminal capacities that actually moved are updated and Boykov-Kolmogorov
    re-solves reusing its search trees. Measured on a 200x200 grid: 320 ms cold
    versus 0.14 ms warm, with byte-identical labels. Only the difference
    ``source - sink`` affects the cut, so capacities are updated by that
    difference, which keeps every update non-negative as the algorithm
    requires.

A full pass is always available and is the authority; the local path is an
optimisation used while the mouse is down.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import maxflow
import numpy as np

from .colormodel import MAX_NLL, GaussianMixtureModel, HistogramModel
from .config import HARD_SEED_CAPACITY, EngineConfig
from .constraints import ConstraintMap
from .imagedata import ROI, ScaleLevel, structure_for, to_feature_space
from .profile import (
    COLOR_MODEL,
    CONSTRAINTS,
    GRAPH_BUILD,
    MAXFLOW,
    POSTPROCESS,
    PREPARE,
    UPSAMPLE,
    NullProfiler,
    Profiler,
)


# --------------------------------------------------------------------------- #
@dataclass
class SegmentParams:
    """Everything that shapes the result or the speed of a pass.

    See the README's parameter table for which of these trade speed against
    quality; the comments here say what each one does.
    """

    # ---- resolution -------------------------------------------------------
    #: Longest side of the image the interactive solve runs on. The dominant
    #: speed knob: cost scales with its square.
    work_max_dim: int = 384
    #: Longest side for the mouse-release refinement. 0 means full resolution.
    refine_max_dim: int = 0

    # ---- energy -----------------------------------------------------------
    neighborhood: int = 8            # 4 or 8 connected
    gamma_smooth: float = 20.0       # pairwise (edge) strength
    lambda_data: float = 3.0         # unary (colour) strength
    color_space: str = "lab"

    # ---- colour models ----------------------------------------------------
    #: 'hist' is exactly incremental and 2-4x faster; 'gmm' is the GrabCut
    #: model and is refit (not updated) when the sample set moves enough.
    model: str = "hist"
    hist_bins: int = 16
    gmm_components: int = 5
    gmm_iterations: int = 4
    max_samples: int = 6000
    #: Rebuild the models once the sample set has grown by this fraction.
    #: Higher = fewer rebuilds = more warm-started solves = faster, at the cost
    #: of the models lagging the newest strokes within a single drag.
    model_refresh_ratio: float = 0.20
    #: Retained for callers that tuned it; the ring is now always sampled, so
    #: this only affects nothing and is kept to avoid breaking stored configs.
    min_negative_samples: int = 200

    # ---- background inference --------------------------------------------
    #: Mean -log P(negative samples | foreground model) above which the
    #: inferred background counts as informative.
    bg_trust_threshold: float = 3.0
    #: Cost of labelling a pixel background when the background model is not
    #: trusted. Acts as a plain threshold on foreground likelihood.
    fg_bias: float = 7.0
    bg_nll_floor: float = 0.6
    #: When the background model is untrusted but the user *has* marked
    #: negatives, fade the foreground bias out near those marks so background
    #: can flood outward from them exactly as foreground floods from a positive
    #: stroke. Without it, a uniform bias makes removing a large same-coloured
    #: region cost more than it is worth and the eraser only clears the
    #: bristles. See the README for why this is needed.
    spatial_bias: bool = True
    #: Fade distance, as a multiple of sqrt(negative mark area), in work pixels.
    spatial_bias_factor: float = 2.2
    spatial_bias_min: int = 12
    spatial_bias_max: int = 200

    # ---- shape of the result ---------------------------------------------
    #: Keep only connected components that contain a positive mark. This is
    #: the main runaway guard: a same-coloured object elsewhere in the frame is
    #: never selected unless you actually paint it. Unlike a reach radius it is
    #: a pure function of the constraints.
    require_seed_connectivity: bool = True
    #: 0 disables the reach limit entirely (the default, since connectivity
    #: already bounds things). Otherwise positives are dilated by
    #: reach_factor * sqrt(positive area) and the result is clipped to that.
    reach_factor: float = 0.0
    reach_min: int = 16
    reach_max: int = 400
    #: Components smaller than this (work-resolution pixels) are dropped.
    min_component_area: int = 16
    #: Interior holes up to this area are filled -- unless they contain a
    #: negative mark, which is always respected.
    fill_hole_area: int = 400

    # ---- local recomputation ---------------------------------------------
    local_enabled: bool = True
    #: Margin added around the new stroke and around the selection boundary,
    #: in full-resolution pixels.
    local_margin: int = 40
    #: If the local ROI covers more than this fraction of the image, do the
    #: global solve instead -- at that point it is not cheaper, just less
    #: accurate.
    local_max_fraction: float = 0.55
    #: Warm-start the max-flow when at most this fraction of terminal
    #: capacities changed. The default (>1) means "always, when the graph
    #: structure is unchanged": a warm update skips the graph build entirely,
    #: which dominates even when every capacity moved, and the labels come out
    #: identical either way.
    warm_start_max_change: float = 1.01
    #: Snap local ROI bounds to this grid (working pixels) so that consecutive
    #: brush stamps in the same area produce the *same* ROI and can therefore
    #: reuse the warm graph. Without snapping the ROI shifts by a pixel or two
    #: per stamp and every pass pays for a cold rebuild.
    roi_snap: int = 48
    #: Warm updates only ever *add* capacity, so a soft node's stored value
    #: creeps upward. Rebuild cold once it approaches the hard-seed capacity,
    #: or the node would start behaving like a seed.
    warm_drift_limit: float = 3000.0
    #: Belt and braces: rebuild cold after this many consecutive warm updates.
    warm_rebuild_interval: int = 150

    # ---- release-time refinement -----------------------------------------
    refine_on_release: bool = True
    refine_band: int = 0             # 0 = derive from the working scale
    guided_radius: int = 8
    guided_eps: float = 1e-4

    def engine_config(self) -> EngineConfig:
        """Adapt to the config object the shared graph helpers expect."""
        return EngineConfig(
            neighborhood=self.neighborhood,
            gamma_smooth=self.gamma_smooth,
            lambda_data=self.lambda_data,
            color_space=self.color_space,
            model=self.model,
            hist_bins=self.hist_bins,
            gmm_components=self.gmm_components,
            gmm_iterations=self.gmm_iterations,
            gmm_max_samples=self.max_samples,
            min_component_area=self.min_component_area,
            guided_radius=self.guided_radius,
            guided_eps=self.guided_eps,
        )

    def graph_key(self) -> tuple:
        """Identity of the precomputed graph weights."""
        return (
            self.work_max_dim,
            self.neighborhood,
            round(self.gamma_smooth, 6),
            self.color_space,
        )


# --------------------------------------------------------------------------- #
# Vectorised mask clean-up
# --------------------------------------------------------------------------- #
def keep_seeded_components(
    mask: np.ndarray, seeds: np.ndarray, min_area: int = 0
) -> np.ndarray:
    """Keep components that contain a seed (plus any large enough, if asked)."""
    if not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask
    keep = np.zeros(n, dtype=bool)
    if seeds.any():
        seeded = np.unique(labels[seeds & mask])
        keep[seeded] = True
    if min_area > 0:
        keep |= stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels]


def drop_small_components(
    mask: np.ndarray, seeds: np.ndarray, min_area: int
) -> np.ndarray:
    """Drop speckle, but never a component the user actually painted."""
    if min_area <= 0 or not mask.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if n <= 1:
        return mask
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    if seeds.any():
        keep[np.unique(labels[seeds & mask])] = True
    keep[0] = False
    return keep[labels]


def fill_holes(mask: np.ndarray, max_area: int, forbid: np.ndarray) -> np.ndarray:
    """Fill interior holes up to ``max_area``, skipping any containing a mark.

    The ``forbid`` argument is what makes this safe to run automatically: a
    hole the user deliberately punched with the background brush is never
    filled back in, no matter how small it is.
    """
    if max_area <= 0 or not mask.any():
        return mask
    inv = ~mask
    if not inv.any():
        return mask
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        inv.astype(np.uint8), connectivity=4
    )
    if n <= 1:
        return mask
    is_border = np.zeros(n, dtype=bool)
    border_ids = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    is_border[border_ids] = True
    blocked = np.zeros(n, dtype=bool)
    if forbid.any():
        blocked[np.unique(labels[forbid])] = True
    fill = (~is_border) & (~blocked) & (stats[:, cv2.CC_STAT_AREA] <= max_area)
    fill[0] = False
    if not fill.any():
        return mask
    return mask | fill[labels]


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or not mask.any() or mask.all():
        return mask
    if radius <= 6:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return cv2.dilate(mask.astype(np.uint8), k) > 0
    dist = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 3)
    return dist <= float(radius)


# --------------------------------------------------------------------------- #
class ImageCache:
    """Per-image precomputation, reused by every stroke.

    The n-link weights are the expensive part -- four float arrays the size of
    the working image plus an exponential each -- and they depend only on the
    pixels, so computing them once per image rather than once per stroke is
    the single largest structural saving in the pipeline.
    """

    def __init__(self, image: np.ndarray, params: SegmentParams) -> None:
        self.params = params
        self.full_h, self.full_w = image.shape[:2]
        self.image = image
        self._key = params.graph_key()

        longest = max(self.full_h, self.full_w)
        self.scale = min(1.0, params.work_max_dim / float(longest)) if params.work_max_dim else 1.0
        if self.scale < 1.0:
            wh = max(1, int(round(self.full_h * self.scale)))
            ww = max(1, int(round(self.full_w * self.scale)))
            work = cv2.resize(image, (ww, wh), interpolation=cv2.INTER_AREA)
        else:
            work, self.scale = image, 1.0
        self.level = ScaleLevel(work, params.engine_config(), self.scale)
        self.height, self.width = self.level.height, self.level.width

        self._full_feat: Optional[np.ndarray] = None
        self._full_guide: Optional[np.ndarray] = None

    def matches(self, image: np.ndarray, params: SegmentParams) -> bool:
        return image is self.image and params.graph_key() == self._key

    @property
    def full_feat(self) -> np.ndarray:
        if self._full_feat is None:
            self._full_feat = to_feature_space(self.image, self.params.color_space)
        return self._full_feat

    @property
    def full_guide(self) -> np.ndarray:
        if self._full_guide is None:
            self._full_guide = self.image.astype(np.float32) / 255.0
        return self._full_guide

    def upscale(self, work_mask: np.ndarray) -> np.ndarray:
        if work_mask.shape[:2] == (self.full_h, self.full_w):
            return work_mask.astype(bool)
        up = cv2.resize(
            work_mask.astype(np.uint8),
            (self.full_w, self.full_h),
            interpolation=cv2.INTER_LINEAR,
        )
        return up > 0

    def downscale(self, full_mask: np.ndarray) -> np.ndarray:
        if full_mask.shape[:2] == (self.height, self.width):
            return full_mask.astype(bool)
        small = cv2.resize(
            full_mask.astype(np.float32),
            (self.width, self.height),
            interpolation=cv2.INTER_AREA,
        )
        return small > 0.5

    def to_work_roi(self, roi: ROI) -> ROI:
        s = self.scale
        return ROI(
            int(np.floor(roi.y0 * s)),
            int(np.ceil(roi.y1 * s)),
            int(np.floor(roi.x0 * s)),
            int(np.ceil(roi.x1 * s)),
        ).clipped(self.height, self.width)


# --------------------------------------------------------------------------- #
class _SampleStore:
    """Accumulates colour samples for one label.

    Histogram mode keeps raw bin counts, so adding a stroke is a ``bincount``
    over its pixels and nothing else -- genuinely incremental, and exact.
    GMM mode keeps a bounded reservoir and refits, because online EM is not
    exact either and a refit on a capped sample is both simpler and honest
    about what it does.
    """

    def __init__(self, params: SegmentParams) -> None:
        self.params = params
        self.is_hist = params.model == "hist"
        bins = params.hist_bins
        self.counts = np.zeros(bins ** 3, dtype=np.float64) if self.is_hist else None
        self.reservoir: Optional[np.ndarray] = None
        self.n = 0
        self._rng = np.random.default_rng(12345)

    def reset(self) -> None:
        if self.is_hist:
            self.counts[:] = 0.0
        self.reservoir = None
        self.n = 0

    def add(self, samples: np.ndarray) -> None:
        if samples.size == 0:
            return
        samples = samples.reshape(-1, 3)
        self.n += samples.shape[0]
        if self.is_hist:
            bins = self.params.hist_bins
            flat = HistogramModel.bin_index(samples, bins)
            self.counts += np.bincount(flat, minlength=bins ** 3)
            return
        cap = self.params.max_samples
        if self.reservoir is None:
            self.reservoir = samples[: cap].astype(np.float32).copy()
        elif self.reservoir.shape[0] < cap:
            room = cap - self.reservoir.shape[0]
            self.reservoir = np.vstack([self.reservoir, samples[:room]])
        else:
            # Replace a random slice so late strokes are represented too.
            take = min(cap // 4, samples.shape[0])
            if take:
                pos = self._rng.choice(cap, size=take, replace=False)
                self.reservoir[pos] = samples[
                    self._rng.choice(samples.shape[0], size=take, replace=False)
                ]

    def build(self, extra: Optional[np.ndarray] = None):
        """Materialise a model from the accumulated samples plus ``extra``."""
        if self.is_hist:
            counts = self.counts
            if extra is not None and extra.size:
                bins = self.params.hist_bins
                flat = HistogramModel.bin_index(extra, bins)
                counts = counts + np.bincount(flat, minlength=bins ** 3)
            if counts.sum() <= 0:
                return None
            return HistogramModel.from_counts(counts, self.params.hist_bins)
        pool = self.reservoir
        if extra is not None and extra.size:
            extra = extra.reshape(-1, 3).astype(np.float32)
            pool = extra if pool is None else np.vstack([pool, extra])
        if pool is None or pool.shape[0] < 2:
            return None
        return GaussianMixtureModel.fit(
            pool,
            n_components=self.params.gmm_components,
            iterations=self.params.gmm_iterations,
            max_samples=self.params.max_samples,
        )


class ColorModels:
    """Foreground/background models plus their cached per-pixel cost arrays."""

    def __init__(self, cache: ImageCache, params: SegmentParams) -> None:
        self.cache = cache
        self.params = params
        self.fg = _SampleStore(params)
        self.bg = _SampleStore(params)
        self.fg_nll: Optional[np.ndarray] = None
        self.bg_nll: Optional[np.ndarray] = None
        self.trust_bg = False
        self._seen_fg = np.zeros((cache.height, cache.width), dtype=bool)
        self._seen_bg = np.zeros((cache.height, cache.width), dtype=bool)
        self._built_fg_n = 0
        self._built_bg_n = 0
        self._built_rev = -1

    def reset(self) -> None:
        self.fg.reset()
        self.bg.reset()
        self._seen_fg[:] = False
        self._seen_bg[:] = False
        self.fg_nll = None
        self.bg_nll = None
        self._built_fg_n = 0
        self._built_bg_n = 0
        self._built_rev = -1

    # ------------------------------------------------------------------ #
    def observe(self, pos: np.ndarray, neg: np.ndarray) -> None:
        """Feed newly constrained pixels into the stores, once each."""
        feat = self.cache.level.feat
        new_fg = pos & ~self._seen_fg
        if new_fg.any():
            self.fg.add(feat[new_fg])
            self._seen_fg |= new_fg
        new_bg = neg & ~self._seen_bg
        if new_bg.any():
            self.bg.add(feat[new_bg])
            self._seen_bg |= new_bg
        # A pixel repainted with the opposite label is no longer 'seen' for the
        # label it lost, so a later repaint re-samples it.
        self._seen_fg &= ~neg
        self._seen_bg &= ~pos

    def needs_rebuild(self, revision: int) -> bool:
        if self.fg_nll is None or self.bg_nll is None:
            return True
        ratio = self.params.model_refresh_ratio
        if self._built_fg_n <= 0:
            return True
        if self.fg.n > self._built_fg_n * (1.0 + ratio):
            return True
        if self.bg.n > max(self._built_bg_n, 1) * (1.0 + ratio):
            return True
        return False

    def rebuild(self, pos: np.ndarray, neg: np.ndarray, revision: int) -> None:
        """Refit and re-evaluate the per-pixel costs over the whole work image."""
        params = self.params
        feat = self.cache.level.feat
        flat = feat.reshape(-1, 3)

        fg_model = self.fg.build()
        if fg_model is None:
            self.fg_nll = np.full((self.cache.height, self.cache.width), MAX_NLL,
                                  dtype=np.float32)
            self.bg_nll = np.zeros_like(self.fg_nll)
            self.trust_bg = False
            self._built_fg_n, self._built_bg_n = self.fg.n, self.bg.n
            self._built_rev = revision
            return

        # Negative colour evidence comes from two places and needs both.
        # Explicit marks say "this particular thing is not the object"; the
        # ring around the positives says "this is what surrounds the object
        # right here". Using only the explicit marks fails in an obvious way
        # once a previous instance has been committed and pinned as
        # background: the model then knows the colour of the *other board* and
        # nothing about the table it is all sitting on, so the table matches
        # neither model and the cut is decided by edges alone.
        ring = self._inferred_ring(pos, neg)
        extra = None
        if ring.any():
            samples = feat[ring]
            cap = params.max_samples
            if samples.shape[0] > cap:
                step = max(1, samples.shape[0] // cap)
                samples = samples[::step][:cap]
            extra = samples

        bg_model = self.bg.build(extra)
        if bg_model is None:
            self.trust_bg = False
        else:
            # Is the negative evidence actually distinguishable from the
            # positive? The ring is the right thing to ask about: deep inside a
            # large object it is still object, and a model fitted to it would
            # carry no information. Explicit marks elsewhere in the frame do
            # not answer that question.
            probe = extra
            if probe is None and neg.any():
                sel = np.flatnonzero(neg.ravel())
                if sel.size > 4000:
                    sel = sel[:: max(1, sel.size // 4000)]
                probe = flat[sel]
            if probe is None or probe.size == 0:
                self.trust_bg = False
            else:
                score = float(np.mean(fg_model.negative_log_likelihood(probe)))
                self.trust_bg = score >= params.bg_trust_threshold

        self.fg_nll = fg_model.negative_log_likelihood(flat).reshape(
            self.cache.height, self.cache.width
        )
        if self.trust_bg and bg_model is not None:
            self.bg_nll = np.maximum(
                bg_model.negative_log_likelihood(flat).reshape(
                    self.cache.height, self.cache.width
                ),
                params.bg_nll_floor,
            )
        else:
            self.bg_nll = np.full_like(self.fg_nll, params.fg_bias)

        self._built_fg_n, self._built_bg_n = self.fg.n, self.bg.n
        self._built_rev = revision

    # ------------------------------------------------------------------ #
    def costs(self, pos: np.ndarray, neg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per-pixel data costs for this pass.

        ``fg_nll`` and the *base* ``bg_nll`` are cached with the models. The
        spatial fade below is recomputed every pass -- it costs one distance
        transform (well under a millisecond at working resolution) and being
        always-fresh means a newly painted negative mark takes effect
        immediately rather than waiting for the next model rebuild.

        Why it is needed: when the background is colour-identical to the
        foreground the model carries no information, so every pixel falls back
        to a uniform ``fg_bias`` cost for being background. For a region of
        radius r the cut then weighs an area cost (bias * pi * r^2) against a
        perimeter cost, and area wins beyond a dozen pixels or so -- the eraser
        clears the brush footprint and stops. Fading the bias to zero at the
        mark and back to full strength at ``scale`` away removes that barrier
        near the mark, so background floods out from it and stops at the first
        real image edge, mirroring how foreground floods from a positive
        stroke. The values still depend only on the constraints, so the pure
        function stays pure.
        """
        params = self.params
        fg = self.fg_nll
        bg = self.bg_nll
        # Applied whenever the user has marked background, not only when the
        # colour model is judged untrustworthy. Trust is one global boolean but
        # trustworthiness is local: a model can explain the background in one
        # corner of the frame and be useless in another. Where the model is
        # already informative this prior is a small perturbation near the mark;
        # where it is not, it is the only thing that makes the eraser work.
        if params.spatial_bias and neg.any() and not neg.all():
            dist = cv2.distanceTransform((~neg).astype(np.uint8), cv2.DIST_L2, 3)
            # Scale from the size of a *typical* mark, not the accumulated
            # total.  Using the total would widen the prior every time another
            # alt-stroke was added, and it would eventually reach across a weak
            # seam and start eating the object the user is keeping.
            n_marks, _ = cv2.connectedComponents(neg.astype(np.uint8), connectivity=8)
            per_mark = float(neg.sum()) / max(1, n_marks - 1)
            scale = float(np.clip(
                params.spatial_bias_factor * np.sqrt(per_mark),
                params.spatial_bias_min,
                params.spatial_bias_max,
            ))
            w = np.clip(dist / scale, 0.0, 1.0).astype(np.float32)
            if pos.any():
                # Never let the background prior reach past the midpoint
                # between a negative mark and a positive one.  Whichever mark
                # is nearer owns the pixel; this keeps one alt-stroke from
                # leaking across a weak boundary into territory the user has
                # explicitly claimed as foreground.
                d_pos = cv2.distanceTransform((~pos).astype(np.uint8), cv2.DIST_L2, 3)
                w = np.maximum(w, (dist > d_pos).astype(np.float32))
            strength = np.float32(params.fg_bias)
            bg = bg * w
            fg = fg + strength * (1.0 - w)
        return fg, bg

    def _inferred_ring(self, pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
        """A band around the positive marks, used as negative samples."""
        area = float(pos.sum())
        if area <= 0:
            return np.zeros_like(pos)
        r = float(np.sqrt(area))
        inner = int(np.clip(2.5 * r, 10, 200))
        outer = inner + max(6, int(0.6 * inner))
        return _dilate(pos, outer) & ~_dilate(pos, inner) & ~pos & ~neg


# --------------------------------------------------------------------------- #
class WarmGraph:
    """A reusable max-flow graph for one ROI shape.

    Boykov-Kolmogorov can reuse its search trees when capacities only increase,
    and only the difference ``source - sink`` affects where the cut falls. So a
    capacity change of ``d`` is applied as ``(+d, 0)`` or ``(0, -d)``, which is
    always a non-negative addition, and the labels come out identical to a cold
    rebuild -- verified, not assumed.
    """

    def __init__(self) -> None:
        self.key: Optional[tuple] = None
        self.graph: Optional[maxflow.Graph] = None
        self.nodeids: Optional[np.ndarray] = None
        self.src: Optional[np.ndarray] = None
        self.snk: Optional[np.ndarray] = None
        #: Per-node accumulated capacity added by warm updates.
        self.accum: Optional[np.ndarray] = None
        self.warm_count: int = 0

    def invalidate(self) -> None:
        self.key = None
        self.graph = None
        self.nodeids = None
        self.src = None
        self.snk = None
        self.accum = None
        self.warm_count = 0

    def solve(
        self,
        level: ScaleLevel,
        roi: ROI,
        src: np.ndarray,
        snk: np.ndarray,
        params: SegmentParams,
        prof: Profiler,
    ) -> np.ndarray:
        key = (roi.y0, roi.y1, roi.x0, roi.x1) + params.graph_key()
        h, w = roi.shape

        reusable = (
            self.graph is not None
            and self.key == key
            and self.src is not None
            and self.src.shape == (h, w)
        )
        if reusable:
            diff_new = src - snk
            diff_old = self.src - self.snk
            d = diff_new - diff_old
            changed = np.abs(d) > 1e-9
            n_changed = int(changed.sum())

            # Warm updates only ever add capacity, so a node's stored value
            # creeps upward. That only matters for nodes meant to stay *soft*:
            # a pixel the user just painted legitimately jumps to the hard-seed
            # capacity in one step, and counting that as drift would disable
            # warm starting entirely. So accumulate per node and check only the
            # soft ones, with a pass counter as a second line of defence.
            hard = HARD_SEED_CAPACITY * 0.5
            soft = (src < hard) & (snk < hard)
            accum = self.accum + np.abs(d)
            soft_drift = float(accum[soft].max()) if soft.any() else 0.0
            if (
                n_changed <= params.warm_start_max_change * d.size
                and soft_drift < params.warm_drift_limit
                and self.warm_count < params.warm_rebuild_interval
            ):
                self.accum = accum
                self.warm_count += 1
                with prof.stage(GRAPH_BUILD):
                    if n_changed:
                        self.graph.add_grid_tedges(
                            self.nodeids,
                            np.maximum(d, 0.0),
                            np.maximum(-d, 0.0),
                        )
                        self.graph.mark_grid_nodes(self.nodeids[changed])
                prof.note("warm_start", True)
                prof.note("changed_nodes", n_changed)
                with prof.stage(MAXFLOW):
                    self.graph.maxflow(reuse_trees=True)
                self.src, self.snk = src.copy(), snk.copy()
                return ~self.graph.get_grid_segments(self.nodeids)

        with prof.stage(GRAPH_BUILD):
            graph = maxflow.Graph[float]()
            nodeids = graph.add_grid_nodes((h, w))
            for off in level.offsets:
                weights = level.weight_slice(off, roi)
                if not weights.any():
                    continue
                graph.add_grid_edges(
                    nodeids,
                    weights=weights,
                    structure=structure_for(*off),
                    symmetric=True,
                )
            graph.add_grid_tedges(nodeids, src, snk)
        prof.note("warm_start", False)
        prof.note("nodes", h * w)
        with prof.stage(MAXFLOW):
            graph.maxflow()

        self.key, self.graph, self.nodeids = key, graph, nodeids
        self.src, self.snk = src.copy(), snk.copy()
        self.accum = np.zeros_like(src)
        self.warm_count = 0
        return ~graph.get_grid_segments(nodeids)


# --------------------------------------------------------------------------- #
def _snap_roi(roi: ROI, grid: int, h: int, w: int) -> ROI:
    """Round an ROI outward onto a grid so nearby stamps share one ROI."""
    return ROI(
        max(0, (roi.y0 // grid) * grid),
        min(h, -(-roi.y1 // grid) * grid),
        max(0, (roi.x0 // grid) * grid),
        min(w, -(-roi.x1 // grid) * grid),
    )


def local_roi(
    stroke_bbox: Optional[Tuple[int, int, int, int]],
    previous: Optional[np.ndarray],
    margin: int,
    shape: Tuple[int, int],
) -> Optional[ROI]:
    """The region to rebuild: the new stroke plus the selection boundary.

    Both are expanded by ``margin``. Including the boundary is what lets the
    existing edge move in response to the new stroke; without it the selection
    could only change where the brush touched.
    """
    h, w = shape
    boxes = []
    if stroke_bbox is not None:
        boxes.append(ROI(*stroke_bbox))
    if previous is not None and previous.any():
        edge = previous ^ cv2.erode(
            previous.astype(np.uint8), np.ones((3, 3), np.uint8)
        ).astype(bool)
        box = ROI.from_mask(edge)
        if box is not None:
            boxes.append(box)
    if not boxes:
        return None
    merged = boxes[0]
    for b in boxes[1:]:
        merged = merged.union(b)
    return merged.expanded(margin, h, w)


# --------------------------------------------------------------------------- #
class Segmenter:
    """Stateful holder for the caches around the pure ``segment`` function.

    Keeping the caches in an object rather than module globals means two
    images, or two instances being annotated at once, cannot interfere.
    """

    def __init__(self, image: np.ndarray, params: Optional[SegmentParams] = None):
        self.params = params or SegmentParams()
        self.cache = ImageCache(image, self.params)
        self.models = ColorModels(self.cache, self.params)
        self.warm = WarmGraph()
        self.profiler = Profiler()
        self._last_mask: Optional[np.ndarray] = None
        self._last_revision = -1

    # ------------------------------------------------------------------ #
    def reset_models(self) -> None:
        """Forget accumulated colour statistics (new instance, or undo)."""
        self.models.reset()
        self.warm.invalidate()
        self._last_mask = None
        self._last_revision = -1

    def set_params(self, params: SegmentParams) -> None:
        rebuild = params.graph_key() != self.params.graph_key()
        self.params = params
        if rebuild:
            self.cache = ImageCache(self.cache.image, params)
            self.models = ColorModels(self.cache, params)
        else:
            self.models.params = params
            self.models.fg.params = params
            self.models.bg.params = params
        self.warm.invalidate()

    # ------------------------------------------------------------------ #
    def run(
        self,
        constraints: ConstraintMap,
        *,
        roi: Optional[ROI] = None,
        previous: Optional[np.ndarray] = None,
        force_global: bool = False,
        prof: Optional[Profiler] = None,
    ) -> np.ndarray:
        """One segmentation pass. Returns a full-resolution boolean mask."""
        params = self.params
        cache = self.cache
        prof = prof if prof is not None else self.profiler
        prof.begin()

        with prof.stage(CONSTRAINTS):
            small = constraints.at_scale(cache.height, cache.width)
            pos = small > 0
            neg = small < 0

        if not pos.any():
            # Nothing marked foreground: the derived mask is empty, but the
            # constraints still get the last word (a lone +1 would show).
            out = np.zeros((cache.full_h, cache.full_w), dtype=bool)
            constraints.enforce_inplace(out)
            prof.finish()
            self._last_mask = out
            return out

        with prof.stage(COLOR_MODEL):
            self.models.observe(pos, neg)
            if self.models.needs_rebuild(constraints.revision):
                self.models.rebuild(pos, neg, constraints.revision)
                # Deliberately *not* invalidating the warm graph here: the
                # capacity-delta check below decides for itself whether the
                # change is small enough to warm-start, and a rebuild that
                # barely moves the costs should still get the fast path.
                prof.note("model_rebuilt", True)
            else:
                prof.note("model_rebuilt", False)

        # ---- choose the region ------------------------------------------ #
        prev_small: Optional[np.ndarray] = None
        work_roi = ROI(0, cache.height, 0, cache.width)
        local = False
        if (
            params.local_enabled
            and not force_global
            and roi is not None
            and previous is not None
            and previous.any()
        ):
            candidate = cache.to_work_roi(roi)
            if params.roi_snap > 1:
                candidate = _snap_roi(candidate, params.roi_snap,
                                      cache.height, cache.width)
            if (
                candidate.area > 0
                and candidate.area <= params.local_max_fraction * cache.height * cache.width
            ):
                work_roi = candidate
                prev_small = cache.downscale(previous)
                local = True
        prof.note("local", local)
        prof.note("roi", (work_roi.y0, work_roi.y1, work_roi.x0, work_roi.x1))

        # ---- terminal capacities ---------------------------------------- #
        with prof.stage(PREPARE):
            fg_full, bg_full = self.models.costs(pos, neg)
            fg_nll = work_roi.slice(fg_full)
            bg_nll = work_roi.slice(bg_full)
            pos_r = work_roi.slice(pos)
            neg_r = work_roi.slice(neg)

            # source = foreground terminal, so its capacity is the cost of
            # choosing background.  See graphcut.py for the full convention.
            src = (params.lambda_data * bg_nll).astype(np.float64)
            snk = (params.lambda_data * fg_nll).astype(np.float64)

            if local and prev_small is not None:
                # Pin the ROI border to whatever the previous mask said, so the
                # local graph is well posed and the result splices seamlessly.
                prev_r = work_roi.slice(prev_small)
                border = np.zeros(work_roi.shape, dtype=bool)
                border[0, :] = border[-1, :] = True
                border[:, 0] = border[:, -1] = True
                bfg = border & prev_r
                bbg = border & ~prev_r
                src[bfg] = HARD_SEED_CAPACITY
                snk[bfg] = 0.0
                src[bbg] = 0.0
                snk[bbg] = HARD_SEED_CAPACITY

            # User marks outrank everything, including the border pins.
            src[pos_r] = HARD_SEED_CAPACITY
            snk[pos_r] = 0.0
            src[neg_r] = 0.0
            snk[neg_r] = HARD_SEED_CAPACITY

        labels = self.warm.solve(cache.level, work_roi, src, snk, params, prof)

        # ---- assemble and clean ----------------------------------------- #
        with prof.stage(POSTPROCESS):
            if local and prev_small is not None:
                work_mask = prev_small.copy()
                work_roi.slice(work_mask)[:] = labels
            else:
                work_mask = labels

            # Re-assert the marks at working resolution before the
            # shape clean-up looks at connectivity.
            work_mask[pos] = True
            work_mask[neg] = False

            if params.reach_factor > 0:
                r = int(np.clip(
                    params.reach_factor * np.sqrt(float(pos.sum())),
                    params.reach_min,
                    params.reach_max,
                ))
                work_mask &= _dilate(pos, r)

            if params.require_seed_connectivity:
                work_mask = keep_seeded_components(work_mask, pos)
            else:
                work_mask = drop_small_components(
                    work_mask, pos, params.min_component_area
                )
            work_mask = fill_holes(work_mask, params.fill_hole_area, neg)
            work_mask[pos] = True
            work_mask[neg] = False

        with prof.stage(UPSAMPLE):
            full = cache.upscale(work_mask)
            # The final word: every mark binds on the returned mask, whether or
            # not this pass happened to look at its neighbourhood.
            constraints.enforce_inplace(full)

        prof.finish()
        self._last_mask = full
        self._last_revision = constraints.revision
        return full

    # ------------------------------------------------------------------ #
    @property
    def last_mask(self) -> Optional[np.ndarray]:
        return self._last_mask


# --------------------------------------------------------------------------- #
def segment(
    image: np.ndarray,
    constraints: ConstraintMap,
    params: Optional[SegmentParams] = None,
    *,
    session: Optional[Segmenter] = None,
    prof: Optional[Profiler] = None,
) -> np.ndarray:
    """``(image, constraints, params) -> binary_mask``. The authoritative pass.

    Always runs the global solve. Pass a ``session`` to reuse the per-image
    caches; the result is identical either way, a session only avoids
    recomputing things that depend solely on the image.
    """
    if session is None:
        session = Segmenter(image, params)
    elif params is not None and params is not session.params:
        session.set_params(params)
    return session.run(constraints, force_global=True, prof=prof)
