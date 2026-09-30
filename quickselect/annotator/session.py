"""Per-image annotation state.

Holds everything about the image currently on screen: its constraint matrix,
the segmenter's caches, the instance being drawn, and the instances already
committed. The UI owns one of these at a time and throws it away on navigation.

The division of responsibility is deliberate. This class knows nothing about
Qt; the widget layer knows nothing about graph cuts. That means the whole
annotation workflow -- paint, commit, undo, export -- can be exercised in a
plain script, which is how it was tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..brush import BrushStroke
from ..config import BrushConfig
from ..constraints import NEGATIVE, POSITIVE, UNLABELED, ConstraintMap
from ..imagedata import ROI
from ..profile import Profiler
from ..segmenter import SegmentParams, Segmenter, local_roi
from .polygon import (
    annotation_from_mask,
    mask_to_polygons,
    polygons_bbox,
    polygons_to_mask,
)


@dataclass
class Instance:
    """One committed object: the polygons plus the pixels they came from.

    Instances loaded from a COCO file keep their original entry in ``raw``.
    As long as nobody edits their geometry they are written back verbatim, so
    ids, unusual categories and any extra keys the file carried survive a
    round trip through the tool.
    """

    mask: np.ndarray
    polygons: List[List[float]]
    annotation_id: Optional[int] = None
    category_id: Optional[int] = None
    source: str = "user"  # 'user' for this session, 'loaded' from the JSON
    raw: Optional[Dict] = None
    dirty: bool = False   # geometry changed since it was loaded
    _bounds: Optional[Tuple[int, int, int, int]] = field(
        default=None, repr=False, compare=False
    )

    @property
    def area(self) -> int:
        return int(self.mask.sum())

    def bounds(self) -> Tuple[int, int, int, int]:
        """Pixel bounding box as ``(x0, y0, x1, y1)``, x1/y1 exclusive.

        Derived from the polygons when there are any -- cheaper than scanning
        the mask -- and cached, because hit-testing asks for it on every mouse
        move.
        """
        if self._bounds is None:
            self._bounds = self._compute_bounds()
        return self._bounds

    def _compute_bounds(self) -> Tuple[int, int, int, int]:
        height, width = self.mask.shape[:2]
        if self.polygons:
            x, y, w, h = polygons_bbox(self.polygons)
            x0, y0 = int(np.floor(x)), int(np.floor(y))
            x1, y1 = int(np.ceil(x + w)) + 1, int(np.ceil(y + h)) + 1
        else:
            rows = np.any(self.mask, axis=1)
            cols = np.any(self.mask, axis=0)
            if not rows.any():
                return (0, 0, 0, 0)
            y0, y1 = int(np.argmax(rows)), height - int(np.argmax(rows[::-1]))
            x0, x1 = int(np.argmax(cols)), width - int(np.argmax(cols[::-1]))
        x0 = max(0, min(x0, width))
        y0 = max(0, min(y0, height))
        x1 = max(x0, min(x1, width))
        y1 = max(y0, min(y1, height))
        return (x0, y0, x1, y1)

    def contains(self, x: float, y: float) -> bool:
        """Is this image-space point inside the instance?"""
        xi, yi = int(x), int(y)
        x0, y0, x1, y1 = self.bounds()
        if not (x0 <= xi < x1 and y0 <= yi < y1):
            return False
        return bool(self.mask[yi, xi])

    def invalidate(self) -> None:
        """Drop cached geometry after the mask or polygons changed."""
        self._bounds = None
        self.dirty = True


class AnnotationSession:
    """Segmentation state for one image, plus its committed instances."""

    def __init__(
        self,
        image: np.ndarray,
        params: Optional[SegmentParams] = None,
        brush: Optional[BrushConfig] = None,
    ) -> None:
        self.image = image
        self.height, self.width = image.shape[:2]
        self.params = params or SegmentParams()
        self.brush = brush or BrushConfig(diameter=48.0)

        self.constraints = ConstraintMap(self.height, self.width)
        self.segmenter = Segmenter(image, self.params)
        self.profiler = Profiler()

        self.mask = np.zeros((self.height, self.width), dtype=bool)
        self.instances: List[Instance] = []

        self._stroke: Optional[BrushStroke] = None
        self._stroke_label = POSITIVE
        self._dirty = False

        # Options the UI sets directly.
        self.polygon_epsilon = 0.0015
        self.polygon_min_area = 64.0
        self.polygon_mode = "external"
        self.seed_committed_as_background = True
        # The class new instances are committed with. ``None`` defers to the
        # default passed to :meth:`to_annotations`.
        self.category_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Brush gestures. Coordinates are full-resolution image pixels.
    # ------------------------------------------------------------------ #
    def begin_stroke(self, x: float, y: float, label: int = POSITIVE) -> bool:
        self._stroke_label = label
        self._stroke = BrushStroke(self.brush, 1.0, self.height, self.width)
        self._stroke.begin(x, y)
        return self._consume_stamps()

    def continue_stroke(self, x: float, y: float) -> bool:
        if self._stroke is None:
            return False
        if self._stroke.extend(x, y) is None:
            return False  # brush has not travelled a full spacing step yet
        return self._consume_stamps()

    def end_stroke(self) -> bool:
        """Close the gesture: one undo step, and the authoritative solve."""
        if self._stroke is None:
            return False
        self._stroke = None
        self.constraints.commit()
        return self.recompute(force_global=True)

    def _consume_stamps(self) -> bool:
        """Write the new stamps into the constraints and re-solve locally."""
        assert self._stroke is not None
        stamp_mask, pending = self._stroke.take()
        if pending is None:
            return False
        bbox = (pending.y0, pending.y1, pending.x0, pending.x1)
        if not self.constraints.apply(stamp_mask, self._stroke_label, bbox=bbox):
            return False
        roi = local_roi(bbox, self.mask, self.params.local_margin,
                        (self.height, self.width))
        return self.recompute(roi=roi)

    # ------------------------------------------------------------------ #
    def recompute(
        self, roi: Optional[ROI] = None, force_global: bool = False
    ) -> bool:
        """Re-derive the mask from the constraints."""
        previous = self.mask if roi is not None and not force_global else None
        self.mask = self.segmenter.run(
            self.constraints,
            roi=roi,
            previous=previous,
            force_global=force_global,
            prof=self.profiler,
        )
        self._dirty = True
        return True

    # ------------------------------------------------------------------ #
    # Constraint-level operations. Undo/redo works on constraints, never on
    # the mask, so the mask is simply recomputed afterwards.
    # ------------------------------------------------------------------ #
    def undo(self) -> bool:
        if not self.constraints.undo():
            return False
        self._resync_models()
        self.recompute(force_global=True)
        return True

    def redo(self) -> bool:
        if not self.constraints.redo():
            return False
        self._resync_models()
        self.recompute(force_global=True)
        return True

    def clear_constraints(self) -> bool:
        """Start the current instance over. Distinct from undo."""
        changed = self.constraints.clear_all()
        self._reseed_committed()
        self._resync_models()
        self.recompute(force_global=True)
        return changed

    def _resync_models(self) -> None:
        """Colour statistics accumulate, so removing marks means starting over.

        An incremental model cannot un-see a sample. After an undo or a clear
        the accumulated histogram no longer matches the constraints, so it is
        discarded and rebuilt from whatever marks remain.
        """
        self.segmenter.reset_models()

    # ------------------------------------------------------------------ #
    # Instances
    # ------------------------------------------------------------------ #
    def can_commit(self) -> bool:
        return bool(self.mask.any())

    def commit_instance(self) -> Optional[Instance]:
        """Freeze the current mask as an instance and reset for the next one."""
        if not self.mask.any():
            return None
        polygons = mask_to_polygons(
            self.mask,
            epsilon_ratio=self.polygon_epsilon,
            min_area=self.polygon_min_area,
            mode=self.polygon_mode,
        )
        if not polygons:
            return None
        instance = Instance(
            mask=self.mask.copy(),
            polygons=polygons,
            category_id=self.category_id,
        )
        self.instances.append(instance)

        # Clear the marks for the next instance, then pin everything already
        # committed as background so the next flood cannot re-grab a board that
        # is already annotated.
        self.constraints.clear_all()
        self.mask = np.zeros((self.height, self.width), dtype=bool)
        self._reseed_committed()
        self._resync_models()
        return instance

    def delete_last_instance(self) -> bool:
        if not self.instances:
            return False
        return self.remove_instance(len(self.instances) - 1) is not None

    def remove_instance(self, index: int) -> Optional[Instance]:
        """Drop one instance by position and return it.

        The caller keeps the returned object, which is what makes undo a
        matter of handing the same instance back to :meth:`insert_instance`.
        """
        if not 0 <= index < len(self.instances):
            return None
        instance = self.instances.pop(index)
        self._reset_after_instance_change()
        return instance

    def insert_instance(self, index: int, instance: Instance) -> None:
        """Put an instance back at a given position (the undo of removal)."""
        index = max(0, min(int(index), len(self.instances)))
        self.instances.insert(index, instance)
        self._reset_after_instance_change()

    def _reset_after_instance_change(self) -> None:
        """Rebuild the derived state after the instance list changed.

        The constraint matrix carries a background seed for every committed
        instance and the colour models carry its samples, and neither records
        which instance it came from. Removing one therefore means starting the
        in-progress instance over, exactly as deleting the last one always
        has.
        """
        self.constraints.clear_all()
        self.mask = np.zeros((self.height, self.width), dtype=bool)
        self._reseed_committed()
        self._resync_models()

    def _reseed_committed(self) -> None:
        """Mark committed instance pixels as background for the next instance."""
        if not self.instances or not self.seed_committed_as_background:
            return
        union = np.zeros((self.height, self.width), dtype=bool)
        for inst in self.instances:
            union |= inst.mask
        if union.any():
            self.constraints.seed_region(union, NEGATIVE)
            self.constraints.commit()

    def committed_mask(self) -> np.ndarray:
        union = np.zeros((self.height, self.width), dtype=bool)
        for inst in self.instances:
            union |= inst.mask
        return union

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def load_instances(self, entries: List[Dict]) -> None:
        """Restore committed instances from COCO entries."""
        self.instances = []
        for entry in entries:
            polygons = entry.get("segmentation") or []
            if not isinstance(polygons, list) or not polygons:
                continue
            if not isinstance(polygons[0], (list, tuple)):
                continue  # RLE, which this tool does not produce or edit
            mask = polygons_to_mask(polygons, self.height, self.width)
            annotation_id = entry.get("id")
            self.instances.append(
                Instance(
                    mask=mask,
                    polygons=[list(map(float, p)) for p in polygons],
                    annotation_id=None if annotation_id is None else int(annotation_id),
                    category_id=(
                        int(entry["category_id"]) if "category_id" in entry else None
                    ),
                    source="loaded",
                    raw=dict(entry),
                )
            )
        self._reseed_committed()

    def to_annotations(self, image_id: int, next_id_fn, category_id: int = 4
                       ) -> List[Dict]:
        """Serialise every committed instance as COCO entries.

        An instance loaded from the file and never edited is written back as
        it came in, so its category and any keys this tool does not understand
        survive the round trip. Only new or modified instances have their
        geometry, area and bbox regenerated from the mask.
        """
        out: List[Dict] = []
        for inst in self.instances:
            ann_id = inst.annotation_id
            if ann_id is None:
                ann_id = next_id_fn()
                inst.annotation_id = ann_id
            if inst.raw is not None and not inst.dirty:
                entry = dict(inst.raw)
                entry["id"] = int(ann_id)
                entry["image_id"] = int(image_id)
                # A reclassified instance keeps its original polygons: only
                # the label changed, so re-deriving the geometry would be a
                # pointless (and lossy) rewrite.
                if inst.category_id is not None:
                    entry["category_id"] = int(inst.category_id)
                out.append(entry)
                continue
            entry = annotation_from_mask(
                inst.mask,
                image_id=image_id,
                annotation_id=ann_id,
                category_id=(
                    category_id if inst.category_id is None else inst.category_id
                ),
                epsilon_ratio=self.polygon_epsilon,
                min_area=self.polygon_min_area,
                mode=self.polygon_mode,
            )
            if entry is not None:
                out.append(entry)
                inst.dirty = False
        return out

    # ------------------------------------------------------------------ #
    @property
    def has_work(self) -> bool:
        """Is there anything worth saving for this image?"""
        return bool(self.instances)

    def constraint_overlays(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.constraints.positive(), self.constraints.negative()

    def timings(self) -> str:
        return self.profiler.summary()
