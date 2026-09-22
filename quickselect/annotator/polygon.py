"""Binary mask -> COCO polygons.

The dataset stores polygons, not raster masks, so this is where a selection
stops being pixels and becomes an annotation. Three decisions matter:

**Which contours.** ``RETR_EXTERNAL`` by default: outer boundaries only, holes
filled. Standard COCO polygon segmentation has no hole semantics -- a hole
polygon sitting in the same list is rendered as *additional filled area* by
pycocotools, which silently corrupts the mask. ``RETR_CCOMP`` is available for
pipelines that handle holes themselves, and is not the default for that reason.

**How much to simplify.** ``approxPolyDP`` with epsilon proportional to the
contour's own perimeter, so a big board and a small one are simplified by the
same *relative* amount. 0.2% of the perimeter typically turns a few thousand
boundary points into a few dozen while staying visually identical.

**What counts as area.** The shoelace area of the simplified polygons, not the
pixel count of the mask. The polygon is what ends up in the file, so the area
should describe the polygon; reporting the pixel count would disagree with
anything that re-rasterises the annotation.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

#: A polygon is a flat [x1, y1, x2, y2, ...] list in absolute pixel coordinates.
Polygon = List[float]

RETRIEVAL_MODES = {
    "external": cv2.RETR_EXTERNAL,
    "ccomp": cv2.RETR_CCOMP,
}


def mask_to_polygons(
    mask: np.ndarray,
    epsilon_ratio: float = 0.002,
    min_area: float = 64.0,
    mode: str = "external",
    max_points: int = 0,
) -> List[Polygon]:
    """Extract simplified polygons from a binary mask.

    Parameters
    ----------
    mask:
        Boolean or 0/255 array.
    epsilon_ratio:
        ``approxPolyDP`` epsilon as a fraction of each contour's perimeter.
        Larger simplifies harder. 0 disables simplification.
    min_area:
        Contours with a smaller polygon area are dropped, which removes the
        one- and two-pixel specks that would otherwise become degenerate
        annotations.
    mode:
        ``'external'`` or ``'ccomp'``.
    max_points:
        If set, epsilon is increased until the polygon fits in this many
        points. Useful when a downstream format caps polygon length.

    Returns
    -------
    A list of polygons, each a flat ``[x1, y1, x2, y2, ...]`` list. Empty if
    nothing survived.
    """
    m = (mask > 0).astype(np.uint8)
    if not m.any():
        return []
    retrieval = RETRIEVAL_MODES.get(mode, cv2.RETR_EXTERNAL)
    contours, _ = cv2.findContours(m, retrieval, cv2.CHAIN_APPROX_SIMPLE)

    polygons: List[Polygon] = []
    for contour in contours:
        if len(contour) < 3:
            continue
        simplified = _simplify(contour, epsilon_ratio, max_points)
        if len(simplified) < 3:
            continue
        pts = simplified.reshape(-1, 2).astype(np.float64)
        if abs(_shoelace(pts)) < min_area:
            continue
        polygons.append([float(v) for v in pts.reshape(-1)])
    return polygons


def _simplify(contour: np.ndarray, epsilon_ratio: float, max_points: int) -> np.ndarray:
    if epsilon_ratio <= 0 and max_points <= 0:
        return contour
    perimeter = cv2.arcLength(contour, True)
    eps = max(epsilon_ratio * perimeter, 1e-6) if epsilon_ratio > 0 else 1e-6
    out = cv2.approxPolyDP(contour, eps, True)
    if max_points > 0:
        # Ratchet epsilon up until the point budget is met. Doubling converges
        # in a handful of steps even for a very convoluted outline.
        guard = 0
        while len(out) > max_points and guard < 24:
            eps *= 1.6
            out = cv2.approxPolyDP(contour, eps, True)
            guard += 1
    return out


def _shoelace(points: np.ndarray) -> float:
    """Signed polygon area."""
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygons_area(polygons: Sequence[Polygon]) -> float:
    """Total absolute area of the polygons."""
    total = 0.0
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] >= 3:
            total += abs(_shoelace(pts))
    return total


def polygons_bbox(polygons: Sequence[Polygon]) -> List[float]:
    """COCO ``[x, y, width, height]`` covering every polygon."""
    xs: List[float] = []
    ys: List[float] = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.size:
            xs.extend(pts[:, 0].tolist())
            ys.extend(pts[:, 1].tolist())
    if not xs:
        return [0.0, 0.0, 0.0, 0.0]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return [x0, y0, x1 - x0, y1 - y0]


def polygons_to_mask(
    polygons: Sequence[Polygon], height: int, width: int
) -> np.ndarray:
    """Rasterise polygons back to a boolean mask.

    Used when reopening a folder: committed annotations are stored as polygons,
    and the app needs pixels again to draw them and to seed new instances.
    """
    canvas = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        pts = np.asarray(poly, dtype=np.float64).reshape(-1, 2)
        if pts.shape[0] >= 3:
            cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas > 0


def annotation_from_mask(
    mask: np.ndarray,
    image_id: int,
    annotation_id: int,
    category_id: int = 4,
    epsilon_ratio: float = 0.002,
    min_area: float = 64.0,
    mode: str = "external",
) -> Optional[Dict]:
    """Build one COCO ``annotations`` entry, or ``None`` if the mask is empty."""
    polygons = mask_to_polygons(
        mask, epsilon_ratio=epsilon_ratio, min_area=min_area, mode=mode
    )
    if not polygons:
        return None
    return {
        "id": int(annotation_id),
        "image_id": int(image_id),
        "category_id": int(category_id),
        "segmentation": polygons,
        "area": float(polygons_area(polygons)),
        "bbox": [float(v) for v in polygons_bbox(polygons)],
        "iscrowd": 0,
    }


def describe(polygons: Sequence[Polygon]) -> str:
    """Short human summary, for the status bar."""
    if not polygons:
        return "no polygon"
    points = sum(len(p) // 2 for p in polygons)
    if len(polygons) == 1:
        return f"1 polygon, {points} pts"
    return f"{len(polygons)} polygons, {points} pts"
