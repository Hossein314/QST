"""Image loading / saving helpers.

Everything inside the package works in **RGB uint8**; OpenCV's native order is
BGR, so conversions are funnelled through here rather than sprinkled around.
``cv2.imdecode`` on raw bytes is used instead of ``cv2.imread`` so non-ASCII
paths (common on Windows) work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

PathLike = Union[str, Path]

IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm",
}


def load_image(path: PathLike) -> np.ndarray:
    """Load an image as RGB uint8 (alpha, if present, is dropped)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"empty or missing image: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_rgba(path: PathLike) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load an image as (RGB uint8, alpha uint8 or None)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB), None
    if img.shape[2] == 4:
        return cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB), img[..., 3].copy()
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), None


def save_image(path: PathLike, rgb: np.ndarray) -> None:
    """Write RGB uint8 (or RGBA) to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGRA)
    elif rgb.ndim == 3:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        bgr = rgb
    ok, buf = cv2.imencode(p.suffix or ".png", bgr)
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(p))


def cutout_rgba(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Compose an RGBA cut-out from an image and a 0..1 alpha."""
    a = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
    return np.dstack([rgb, a])
