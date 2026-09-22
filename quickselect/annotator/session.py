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
from .polygon import annotation_from_mask, mask_to_polygons, polygons_to_mask


@dataclass
class Instance:
    """One committed object: the polygons plus the pixels they came from."""

    mask: np.ndarray
    polygons: List[List[float]]
    annotation_id: Optional[int] = None
    source: str = "user"  # 'user' for this session, 'loaded' from the JSON

    @property
    def area(self) -> int:
        return int(self.mask.sum())


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
        instance = Instance(mask=self.mask.copy(), polygons=polygons)
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
        self.instances.pop()
        self.constraints.clear_all()
        self.mask = np.zeros((self.height, self.width), dtype=bool)
        self._reseed_committed()
        self._resync_models()
        return True

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
            self.instances.append(
                Instance(
                    mask=mask,
                    polygons=[list(map(float, p)) for p in polygons],
                    annotation_id=entry.get("id"),
                    source="loaded",
                )
            )
        self._reseed_committed()

    def to_annotations(self, image_id: int, next_id_fn, category_id: int = 4
                       ) -> List[Dict]:
        """Serialise every committed instance as COCO entries."""
        out: List[Dict] = []
        for inst in self.instances:
            ann_id = inst.annotation_id
            if ann_id is None:
                ann_id = next_id_fn()
                inst.annotation_id = ann_id
            entry = annotation_from_mask(
                inst.mask,
                image_id=image_id,
                annotation_id=ann_id,
                category_id=category_id,
                epsilon_ratio=self.polygon_epsilon,
                min_area=self.polygon_min_area,
                mode=self.polygon_mode,
            )
            if entry is not None:
                out.append(entry)
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
