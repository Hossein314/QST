"""Visual mask comparison.

Deliberately *not* a scoring framework.  There are no metrics here -- the tool
is a deterministic algorithm with nothing to train, so it is evaluated by
looking at it.  What this module provides is one function that makes the
looking easy: given your result and a reference mask, produce an image where
the disagreement is obvious at a glance.

Colour key (warm = yours, cool = theirs, so it survives colour-blind viewing):

=====================  ===========================
region                 colour
=====================  ===========================
both masks agree (in)  light neutral
both agree (out)       the image, dimmed
only in *predicted*    warm orange
only in *reference*    cool blue
=====================  ===========================
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

from .io_utils import PathLike, load_image

# Warm / cool pair, chosen to stay distinguishable under deuteranopia.
COLOR_PRED_ONLY = (232, 131, 58)    # orange  -- in your mask, not the reference
COLOR_REF_ONLY = (58, 123, 213)     # blue    -- in the reference, not yours
COLOR_BOTH = (236, 238, 241)        # neutral -- both agree it is inside


def load_mask(source: Union[PathLike, np.ndarray], threshold: int = 127) -> np.ndarray:
    """Read a mask from a path or array and return a boolean array.

    Accepts grayscale or colour PNGs, and RGBA files where the selection lives
    in the alpha channel (which is what Photoshop exports if you save a cut-out
    rather than a mask).
    """
    if isinstance(source, np.ndarray):
        arr = source
    else:
        data = np.fromfile(str(source), dtype=np.uint8)
        if data.size == 0:
            raise FileNotFoundError(f"empty or missing mask: {source}")
        arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError(f"could not decode mask: {source}")

    if arr.dtype == bool:
        return arr
    if arr.ndim == 3:
        arr = arr[..., 3] if arr.shape[2] == 4 else cv2.cvtColor(
            arr, cv2.COLOR_BGR2GRAY
        )
    if arr.dtype != np.uint8:
        arr = np.clip(arr.astype(np.float32) * (255.0 if arr.max() <= 1.0 else 1.0),
                      0, 255).astype(np.uint8)
    return arr > threshold


def save_mask(path: PathLike, mask: np.ndarray) -> None:
    """Write a boolean or 0..1 float mask as an 8-bit PNG."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if mask.dtype == bool:
        out = mask.astype(np.uint8) * 255
    elif mask.dtype in (np.float32, np.float64):
        out = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    else:
        out = mask.astype(np.uint8)
    ok, buf = cv2.imencode(p.suffix or ".png", out)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(p))


def _match_size(mask: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    if mask.shape[:2] == shape:
        return mask
    return (
        cv2.resize(
            mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
        > 0
    )


def mask_diff(
    predicted: Union[PathLike, np.ndarray],
    reference: Union[PathLike, np.ndarray],
    image: Optional[Union[PathLike, np.ndarray]] = None,
    opacity: float = 0.75,
    outline: bool = True,
) -> np.ndarray:
    """Return an RGB image highlighting where two binary masks disagree.

    Parameters
    ----------
    predicted, reference:
        Boolean arrays, or paths to mask PNGs.  If their sizes differ, the
        reference is resampled (nearest) onto the predicted mask's grid.
    image:
        Optional source photo to use as the backdrop, so you can see *what*
        the disagreement is sitting on.  Without it the backdrop is dark grey.
    opacity:
        How strongly the colour overlay covers the backdrop, 0..1.
    outline:
        Draw a thin contour around each mask as well as filling the difference,
        which makes single-pixel-wide disagreements visible.

    Returns
    -------
    RGB uint8 image, same size as ``predicted``.
    """
    pred = load_mask(predicted)
    ref = _match_size(load_mask(reference), pred.shape[:2])

    if image is None:
        base = np.full((*pred.shape[:2], 3), 28, dtype=np.uint8)
    else:
        img = image if isinstance(image, np.ndarray) else load_image(image)
        if img.shape[:2] != pred.shape[:2]:
            img = cv2.resize(img, (pred.shape[1], pred.shape[0]))
        base = (img.astype(np.float32) * 0.45).astype(np.uint8)

    overlay = base.astype(np.float32).copy()
    both = pred & ref
    pred_only = pred & ~ref
    ref_only = ref & ~pred

    for region, color, strength in (
        (both, COLOR_BOTH, opacity * 0.45),
        (pred_only, COLOR_PRED_ONLY, opacity),
        (ref_only, COLOR_REF_ONLY, opacity),
    ):
        if region.any():
            overlay[region] = (
                overlay[region] * (1.0 - strength) + np.array(color, np.float32) * strength
            )

    out = np.clip(overlay, 0, 255).astype(np.uint8)

    if outline:
        for mask, color in ((ref, COLOR_REF_ONLY), (pred, COLOR_PRED_ONLY)):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(out, contours, -1, color, 1, cv2.LINE_AA)
    return out


def side_by_side(
    predicted: Union[PathLike, np.ndarray],
    reference: Union[PathLike, np.ndarray],
    image: Optional[Union[PathLike, np.ndarray]] = None,
    gap: int = 12,
) -> np.ndarray:
    """Yours on the left, the reference on the right, as one image."""
    pred = load_mask(predicted)
    ref = _match_size(load_mask(reference), pred.shape[:2])
    img = None
    if image is not None:
        img = image if isinstance(image, np.ndarray) else load_image(image)
        if img.shape[:2] != pred.shape[:2]:
            img = cv2.resize(img, (pred.shape[1], pred.shape[0]))

    def render(mask: np.ndarray, color) -> np.ndarray:
        if img is None:
            panel = np.full((*mask.shape, 3), 28, dtype=np.uint8)
        else:
            panel = (img.astype(np.float32) * 0.45).astype(np.uint8)
        p = panel.astype(np.float32)
        p[mask] = p[mask] * 0.25 + np.array(color, np.float32) * 0.75
        panel = np.clip(p, 0, 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(panel, contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        return panel

    left = render(pred, COLOR_PRED_ONLY)
    right = render(ref, COLOR_REF_ONLY)
    sep = np.full((left.shape[0], max(0, gap), 3), 18, dtype=np.uint8)
    return np.hstack([left, sep, right])


def blend(
    predicted: Union[PathLike, np.ndarray],
    reference: Union[PathLike, np.ndarray],
    t: float,
    image: Optional[Union[PathLike, np.ndarray]] = None,
) -> np.ndarray:
    """Cross-fade between the two masks; ``t=0`` is yours, ``t=1`` the reference.

    Used by the interactive viewer's slider.  Blending the *overlays* rather
    than the masks keeps both boundaries readable at intermediate positions.
    """
    pred = load_mask(predicted)
    ref = _match_size(load_mask(reference), pred.shape[:2])
    t = float(np.clip(t, 0.0, 1.0))

    if image is None:
        base = np.full((*pred.shape[:2], 3), 28, dtype=np.uint8)
    else:
        img = image if isinstance(image, np.ndarray) else load_image(image)
        if img.shape[:2] != pred.shape[:2]:
            img = cv2.resize(img, (pred.shape[1], pred.shape[0]))
        base = (img.astype(np.float32) * 0.45).astype(np.uint8)

    out = base.astype(np.float32)
    a = pred.astype(np.float32) * (1.0 - t)
    b = ref.astype(np.float32) * t
    out += a[..., None] * (np.array(COLOR_PRED_ONLY, np.float32) - out) * 0.75
    out += b[..., None] * (np.array(COLOR_REF_ONLY, np.float32) - out) * 0.75
    return np.clip(out, 0, 255).astype(np.uint8)
