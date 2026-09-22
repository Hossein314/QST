"""Graph construction and min-cut solving.

The energy minimised is the standard interactive-segmentation functional
(Boykov & Jolly 2001; Rother et al. 2004; Li et al. "Lazy Snapping" 2004):

    E(a) = lambda * SUM_p  D_p(a_p)  +  SUM_(p,q) in N  w(p,q) * [a_p != a_q]

with

    D_p(FG) = -log P(I_p | foreground model)
    D_p(BG) = -log P(I_p | background model)
    w(p,q)  = gamma * exp(-beta * ||I_p - I_q||^2) / dist(p, q)

Because the pairwise term is submodular (it is zero when the labels agree and
positive otherwise) the global minimum is found exactly by a single s-t min-cut,
solved here with the Boykov-Kolmogorov algorithm via PyMaxflow.

Terminal convention used throughout: **source = foreground, sink = background**.
Cutting the source edge of a pixel therefore assigns it to the background, so
the source capacity must be the *cost of labelling it background*, D_p(BG).
PyMaxflow's ``get_grid_segments`` returns ``True`` for the sink side, hence
``foreground = ~segments``.
"""

from __future__ import annotations

from typing import Optional

import cv2
import maxflow
import numpy as np

from .config import HARD_SEED_CAPACITY, EngineConfig
from .imagedata import ROI, ScaleLevel, structure_for


def solve_min_cut(
    level: ScaleLevel,
    roi: ROI,
    fg_seed: np.ndarray,
    bg_seed: np.ndarray,
    fg_nll: Optional[np.ndarray],
    bg_nll: Optional[np.ndarray],
    cfg: EngineConfig,
    edge_gain: float = 1.0,
) -> np.ndarray:
    """Solve the binary labelling inside ``roi``.

    Parameters
    ----------
    level:
        Resolution level supplying the pre-computed n-link weights.
    roi:
        Region of the image to solve.  Nodes outside it simply do not exist,
        which is what makes the incremental update cheap; callers are
        responsible for placing hard seeds on the ROI border so the cut is
        well posed.
    fg_seed, bg_seed:
        Boolean arrays shaped like ``roi``.  Hard constraints.
    fg_nll, bg_nll:
        ``-log P(I | model)`` arrays shaped like ``roi`` (or ``None`` to drop
        the data term entirely, giving a pure geodesic/edge-based cut).
    edge_gain:
        Multiplier on the pairwise term.  Auto-Enhance raises this so the
        boundary snaps harder onto image gradients.

    Returns
    -------
    Boolean array shaped like ``roi``; ``True`` where the pixel is foreground.
    """
    h, w = roi.shape
    if h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=bool)

    has_fg = bool(fg_seed.any())
    has_bg = bool(bg_seed.any())
    if not has_fg:
        # Nothing to grow from.
        return np.zeros((h, w), dtype=bool)
    if not has_bg:
        # No opposing terminal: the cut would be empty and everything becomes
        # foreground.  Short-circuit rather than paying for max-flow.
        return np.ones((h, w), dtype=bool)

    graph = maxflow.Graph[float]()
    nodeids = graph.add_grid_nodes((h, w))

    # ---- pairwise (n-link) terms ----------------------------------------- #
    for off in level.offsets:
        weights = level.weight_slice(off, roi)
        if edge_gain != 1.0:
            weights = weights * edge_gain
        if not weights.any():
            continue
        graph.add_grid_edges(
            nodeids,
            weights=weights,
            structure=structure_for(*off),
            symmetric=True,
        )

    # ---- unary (t-link) terms -------------------------------------------- #
    lam = cfg.lambda_data
    if fg_nll is None or bg_nll is None:
        source_cap = np.zeros((h, w), dtype=np.float64)
        sink_cap = np.zeros((h, w), dtype=np.float64)
    else:
        # source = FG terminal, so its capacity is the cost of choosing BG.
        source_cap = (lam * bg_nll).astype(np.float64)
        sink_cap = (lam * fg_nll).astype(np.float64)

    source_cap[fg_seed] = HARD_SEED_CAPACITY
    sink_cap[fg_seed] = 0.0
    source_cap[bg_seed] = 0.0
    sink_cap[bg_seed] = HARD_SEED_CAPACITY

    graph.add_grid_tedges(nodeids, source_cap, sink_cap)
    graph.maxflow()
    # True == sink == background, so invert.
    return ~graph.get_grid_segments(nodeids)


def solve_band_min_cut(
    feat: np.ndarray,
    band: np.ndarray,
    fixed_fg: np.ndarray,
    fixed_bg: np.ndarray,
    fg_nll: Optional[np.ndarray],
    bg_nll: Optional[np.ndarray],
    beta: float,
    cfg: EngineConfig,
    edge_gain: float = 1.0,
) -> np.ndarray:
    """Min-cut restricted to a narrow band around an existing boundary.

    Full-resolution refinement on a 1080p image would need millions of nodes if
    it were run over the whole frame.  Only the pixels near the current boundary
    can actually change label, so we build a graph over just those.

    The reduction is exact rather than approximate: an edge from a band pixel
    ``p`` to a *fixed* neighbour ``q`` cannot be cut on the ``q`` side, so it
    collapses into a terminal edge on ``p`` of the same capacity -- to the
    source if ``q`` is fixed foreground, to the sink if it is fixed background.

    Parameters
    ----------
    feat:
        Feature image (H, W, 3) for the region being refined.
    band, fixed_fg, fixed_bg:
        Disjoint boolean masks covering the region.
    fg_nll, bg_nll:
        ``-log P`` arrays, shaped like the region (only band entries are read).
    beta:
        Contrast normaliser, see :func:`estimate_beta`.

    Returns
    -------
    Boolean foreground mask for the whole region (fixed parts included).
    """
    h, w = band.shape
    out = fixed_fg.copy()
    ys, xs = np.nonzero(band)
    n = ys.size
    if n == 0:
        return out

    index = np.full((h, w), -1, dtype=np.int64)
    index[ys, xs] = np.arange(n)

    gamma = cfg.gamma_smooth * edge_gain
    flat_feat = feat.reshape(-1, feat.shape[2]).astype(np.float32)
    p_flat = ys * w + xs
    p_vals = flat_feat[p_flat]

    src_extra = np.zeros(n, dtype=np.float64)
    sink_extra = np.zeros(n, dtype=np.float64)

    graph = maxflow.Graph[float]()
    nodes = graph.add_nodes(n)

    all_offsets = (
        ((0, 1), (0, -1), (1, 0), (-1, 0))
        + (((1, 1), (1, -1), (-1, 1), (-1, -1)) if cfg.neighborhood == 8 else ())
    )
    pair_offsets = (
        ((0, 1), (1, 0), (1, 1), (1, -1))
        if cfg.neighborhood == 8
        else ((0, 1), (1, 0))
    )

    for dy, dx in all_offsets:
        ny, nx = ys + dy, xs + dx
        inside = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
        if not inside.any():
            continue
        sel = np.flatnonzero(inside)
        qy, qx = ny[sel], nx[sel]
        d = p_vals[sel] - flat_feat[qy * w + qx]
        sq = np.einsum("ij,ij->i", d, d)
        weight = (gamma / float(np.hypot(dy, dx))) * np.exp(-beta * sq)

        q_fg = fixed_fg[qy, qx]
        q_bg = fixed_bg[qy, qx]
        # bincount is far quicker than np.add.at for this scatter-add.
        if q_fg.any():
            src_extra += np.bincount(sel[q_fg], weights=weight[q_fg], minlength=n)
        if q_bg.any():
            sink_extra += np.bincount(sel[q_bg], weights=weight[q_bg], minlength=n)

        if (dy, dx) in pair_offsets:
            both = index[qy, qx] >= 0
            if both.any():
                i = np.ascontiguousarray(sel[both])
                j = np.ascontiguousarray(index[qy[both], qx[both]])
                cap = np.ascontiguousarray(weight[both])
                graph.add_edges(i, j, cap, cap)

    lam = cfg.lambda_data
    if fg_nll is not None and bg_nll is not None:
        src_extra += lam * bg_nll[ys, xs].astype(np.float64)
        sink_extra += lam * fg_nll[ys, xs].astype(np.float64)

    graph.add_grid_tedges(nodes, src_extra, sink_extra)
    graph.maxflow()
    labels = ~graph.get_grid_segments(nodes)  # True == foreground
    out[ys, xs] = labels
    return out


def estimate_beta(feat: np.ndarray, max_samples: int = 200_000,
                  seed: int = 0) -> float:
    """``beta = 1 / (2 E[||I_p - I_q||^2])`` from a random sample of pairs.

    Sampling keeps this O(1) in image size, which matters because the
    full-resolution refinement needs a beta but must never touch every pixel.
    """
    h, w = feat.shape[:2]
    if h < 2 or w < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    n = min(max_samples, h * w)
    ys = rng.integers(0, h - 1, size=n)
    xs = rng.integers(0, w - 1, size=n)
    f = feat.astype(np.float32)
    d1 = f[ys, xs] - f[ys, xs + 1]
    d2 = f[ys, xs] - f[ys + 1, xs]
    mean_sq = float(
        (np.einsum("ij,ij->i", d1, d1).sum() + np.einsum("ij,ij->i", d2, d2).sum())
        / (2.0 * n)
    )
    return 1.0 / (2.0 * mean_sq) if mean_sq > 1e-8 else 0.0


def clean_components(
    mask: np.ndarray,
    keep_seed: Optional[np.ndarray] = None,
    min_area: int = 24,
) -> np.ndarray:
    """Drop speckle: components smaller than ``min_area`` that hold no seed.

    The graph cut can leave isolated islands where the colour model happens to
    match; Photoshop's tool never produces those, so we prune them.  Any
    component containing a user brush stroke is always kept, however small.
    """
    if min_area <= 0 or not mask.any():
        return mask
    m = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return mask
    keep = np.zeros(n, dtype=bool)
    keep[0] = False
    areas = stats[:, cv2.CC_STAT_AREA]
    keep |= areas >= min_area
    if keep_seed is not None and keep_seed.any():
        seeded = np.unique(labels[keep_seed])
        keep[seeded[seeded > 0]] = True
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Close interior holes smaller than ``max_area``.

    Specular highlights inside an object otherwise punch holes that Photoshop's
    tool would have swallowed.
    """
    if max_area <= 0 or not mask.any():
        return mask
    inv = (~mask).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    if n <= 1:
        return mask
    out = mask.copy()
    border = set(np.unique(labels[0, :])) | set(np.unique(labels[-1, :]))
    border |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    for i in range(1, n):
        if i in border:
            continue
        if stats[i, cv2.CC_STAT_AREA] <= max_area:
            out[labels == i] = True
    return out
