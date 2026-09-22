"""Edge-aware boundary refinement and the "Refine Edge" operations.

Two jobs live here:

* Turning a hard binary mask into a soft alpha that follows the image's edges,
  via the **guided filter** (He, Sun & Tang, ECCV 2010).  The guided filter is
  the O(N) approximation to the closed-form matting Laplacian of Levin et al.
  (2008): both assume the matte is locally a linear function of the colour, but
  the guided filter solves it with box filters instead of a sparse linear
  system, which is what makes it usable at 1080p in real time.

* The Refine Edge sliders -- feather, smooth, contract/expand -- implemented on
  the *alpha*, not the binary mask, so fractional coverage survives.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from .config import RefineEdgeConfig


# --------------------------------------------------------------------------- #
def guided_filter(
    guide: np.ndarray,
    src: np.ndarray,
    radius: int = 8,
    eps: float = 1e-4,
) -> np.ndarray:
    """Colour-guided filter of ``src`` using ``guide``.

    ``guide`` is float32 HxWx3 in 0..1, ``src`` float32 HxW.  Returns float32
    HxW.  The 3x3 per-window covariance inverse is computed in closed form,
    which is both faster and more numerically predictable than calling
    ``np.linalg.inv`` per pixel.
    """
    if src.ndim != 2:
        raise ValueError("src must be 2-D")
    r = max(1, int(radius))
    ksize = (2 * r + 1, 2 * r + 1)

    def box(a: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(a, -1, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    g = guide.astype(np.float32)
    p = src.astype(np.float32)
    ir, ig, ib = g[..., 0], g[..., 1], g[..., 2]

    mean_i = box(g)
    mean_p = box(p)

    # Covariance of (I, p) per window.
    cov_ip = np.stack(
        [box(ir * p) - mean_i[..., 0] * mean_p,
         box(ig * p) - mean_i[..., 1] * mean_p,
         box(ib * p) - mean_i[..., 2] * mean_p],
        axis=-1,
    )

    # Variance of I per window (symmetric 3x3).
    var_rr = box(ir * ir) - mean_i[..., 0] * mean_i[..., 0] + eps
    var_rg = box(ir * ig) - mean_i[..., 0] * mean_i[..., 1]
    var_rb = box(ir * ib) - mean_i[..., 0] * mean_i[..., 2]
    var_gg = box(ig * ig) - mean_i[..., 1] * mean_i[..., 1] + eps
    var_gb = box(ig * ib) - mean_i[..., 1] * mean_i[..., 2]
    var_bb = box(ib * ib) - mean_i[..., 2] * mean_i[..., 2] + eps

    # Closed-form inverse of the symmetric 3x3 via cofactors.
    inv_rr = var_gg * var_bb - var_gb * var_gb
    inv_rg = var_gb * var_rb - var_rg * var_bb
    inv_rb = var_rg * var_gb - var_gg * var_rb
    inv_gg = var_rr * var_bb - var_rb * var_rb
    inv_gb = var_rb * var_rg - var_rr * var_gb
    inv_bb = var_rr * var_gg - var_rg * var_rg

    det = var_rr * inv_rr + var_rg * inv_rg + var_rb * inv_rb
    det = np.where(np.abs(det) < 1e-12, 1e-12, det)

    c0, c1, c2 = cov_ip[..., 0], cov_ip[..., 1], cov_ip[..., 2]
    a_r = (inv_rr * c0 + inv_rg * c1 + inv_rb * c2) / det
    a_g = (inv_rg * c0 + inv_gg * c1 + inv_gb * c2) / det
    a_b = (inv_rb * c0 + inv_gb * c1 + inv_bb * c2) / det
    b = mean_p - (
        a_r * mean_i[..., 0] + a_g * mean_i[..., 1] + a_b * mean_i[..., 2]
    )

    out = (
        box(a_r) * ir + box(a_g) * ig + box(a_b) * ib + box(b)
    )
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------- #
def shift_edge(alpha: np.ndarray, pixels: int) -> np.ndarray:
    """Contract (negative) or expand (positive) the 0.5 level set.

    Implemented as a signed-distance offset so the shift is isotropic and
    sub-pixel accurate, unlike repeated morphological erosion.
    """
    if pixels == 0:
        return alpha
    solid = (alpha >= 0.5).astype(np.uint8)
    if not solid.any() or solid.all():
        return alpha
    d_out = cv2.distanceTransform(1 - solid, cv2.DIST_L2, 5)
    d_in = cv2.distanceTransform(solid, cv2.DIST_L2, 5)
    sdf = d_out - d_in  # negative inside
    shifted = sdf + float(pixels)
    # Map the new signed distance back to a 1-pixel-wide soft edge.
    out = np.clip(0.5 - shifted, 0.0, 1.0)
    return out.astype(np.float32)


def smooth_alpha(alpha: np.ndarray, radius: int) -> np.ndarray:
    """Round off corners without moving the boundary, Photoshop's "Smooth"."""
    if radius <= 0:
        return alpha
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )
    solid = (alpha >= 0.5).astype(np.uint8)
    opened = cv2.morphologyEx(solid, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
    # Blend so partial coverage in the original is not thrown away entirely.
    hard = closed.astype(np.float32)
    return np.clip(0.5 * hard + 0.5 * cv2.GaussianBlur(hard, (0, 0), radius * 0.5), 0, 1)


def feather_alpha(alpha: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return alpha
    return np.clip(cv2.GaussianBlur(alpha, (0, 0), float(radius)), 0.0, 1.0)


def apply_refine_edge(alpha: np.ndarray, cfg: RefineEdgeConfig) -> np.ndarray:
    """Apply the Refine Edge chain in Photoshop's order."""
    out = alpha.astype(np.float32)
    if cfg.smooth > 0:
        out = smooth_alpha(out, cfg.smooth)
    if cfg.shift_edge != 0:
        out = shift_edge(out, cfg.shift_edge)
    if cfg.feather > 0:
        out = feather_alpha(out, cfg.feather)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
def snap_to_edges(
    alpha: np.ndarray,
    gradient: np.ndarray,
    search: int = 3,
    strength: float = 1.0,
) -> np.ndarray:
    """Cheap Auto-Enhance fallback: pull the 0.5 level set onto gradient ridges.

    The engine's preferred Auto-Enhance is a narrow-band graph cut, which is
    better behaved.  This purely local variant exists for very large images
    where even the band solve is too slow, and as a final polish pass.
    """
    if strength <= 0:
        return alpha
    k = 2 * max(1, int(search)) + 1
    ridge = cv2.dilate(gradient, np.ones((k, k), np.uint8))
    on_ridge = gradient >= ridge - 1e-6
    boundary = (alpha > 0.15) & (alpha < 0.85)
    pull = (boundary & on_ridge).astype(np.float32)
    sharpened = np.clip((alpha - 0.5) * (1.0 + 2.0 * strength) + 0.5, 0.0, 1.0)
    return (alpha * (1.0 - pull) + sharpened * pull).astype(np.float32)


# --------------------------------------------------------------------------- #
def mask_to_contours(mask: np.ndarray, epsilon: float = 0.0) -> list:
    """Outline polygons for marching-ants display."""
    m = (mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if epsilon > 0:
            c = cv2.approxPolyDP(c, epsilon, True)
        if len(c) >= 2:
            out.append(c.reshape(-1, 2))
    return out
