"""The constraint matrix: the single source of truth for a selection.

Why this module exists
----------------------
The earlier design stored the *selection* as primary state and kept brush marks
in a secondary trimap. That loses information. A pixel you deliberately marked
as background could be re-labelled foreground by a later pass, because the
selection and the marks were two separate things that could disagree, and
whichever ran last won.

Here there is exactly one authoritative array::

    +1  the user painted this as foreground
    -1  the user painted this as background
     0  unlabelled -- the algorithm decides

It is created once per image and is **never** written by the segmenter. Only
explicit user actions change it: painting, erasing, or an explicit clear. Every
segmentation pass reads it and honours ``+1`` / ``-1`` as infinite-capacity
terminal edges, so a mark made twenty strokes ago still binds. The binary mask
is a derived artifact, recomputed from ``(image, constraints)``.

The practical consequence: painting background in one corner and foreground in
the other cannot cause the corner to lose its background status, because
nothing in the pipeline has the authority to overwrite it except your brush.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

# Constraint values.  Signed so `constraints > 0` / `< 0` read naturally and a
# label can be flipped with a negation.
UNLABELED = 0
POSITIVE = 1   # definite foreground
NEGATIVE = -1  # definite background


@dataclass(frozen=True)
class Patch:
    """A rectangular before/after diff of the constraint matrix.

    Undo history stores these rather than whole-matrix snapshots: a brush stamp
    touches a few thousand pixels, so a step costs a couple of kilobytes
    instead of the two megabytes a 1920x1080 int8 copy would need. The diff is
    exact, so undo is lossless.
    """

    y0: int
    y1: int
    x0: int
    x1: int
    before: np.ndarray
    after: np.ndarray

    @property
    def pixels(self) -> int:
        return int(self.before.size)


class ConstraintHistory:
    """Undo/redo over constraint patches.

    Operates on the constraint matrix, never on the derived mask -- undoing a
    stroke restores exactly the marks that existed before it, and the mask is
    recomputed from those.
    """

    def __init__(self, limit: int = 200) -> None:
        self.limit = max(1, limit)
        self._undo: List[List[Patch]] = []
        self._redo: List[List[Patch]] = []

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, patches: List[Patch]) -> None:
        if not patches:
            return
        self._undo.append(patches)
        if len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def pop_undo(self) -> Optional[List[Patch]]:
        if not self._undo:
            return None
        patches = self._undo.pop()
        self._redo.append(patches)
        return patches

    def pop_redo(self) -> Optional[List[Patch]]:
        if not self._redo:
            return None
        patches = self._redo.pop()
        self._undo.append(patches)
        return patches

    def __len__(self) -> int:
        return len(self._undo)


class ConstraintMap:
    """Persistent per-pixel user marks for one image.

    Stored at **full image resolution**. Downscaled copies for the working
    resolution are derived and cached, never authoritative -- so switching the
    working resolution mid-session cannot corrupt what the user marked.
    """

    def __init__(self, height: int, width: int, history_limit: int = 200) -> None:
        self.height = height
        self.width = width
        self.matrix = np.zeros((height, width), dtype=np.int8)
        self.history = ConstraintHistory(history_limit)
        self._pending: List[Patch] = []
        self._revision = 0
        # Cached downscaled view plus the full-resolution rectangles that have
        # changed since it was last refreshed.
        self._scaled: Optional[np.ndarray] = None
        self._scaled_shape: Optional[Tuple[int, int]] = None
        self._dirty: List[Tuple[int, int, int, int]] = []

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    @property
    def revision(self) -> int:
        """Bumped on every change; lets caches tell if they are stale."""
        return self._revision

    def positive(self) -> np.ndarray:
        return self.matrix > 0

    def negative(self) -> np.ndarray:
        return self.matrix < 0

    def counts(self) -> Tuple[int, int]:
        return int((self.matrix > 0).sum()), int((self.matrix < 0).sum())

    def is_empty(self) -> bool:
        return not self.matrix.any()

    def bbox(self, label: Optional[int] = None) -> Optional[Tuple[int, int, int, int]]:
        """``(y0, y1, x0, x1)`` of the marked pixels, or ``None`` if none."""
        if label is None:
            mask = self.matrix != 0
        elif label > 0:
            mask = self.matrix > 0
        else:
            mask = self.matrix < 0
        ys = np.flatnonzero(mask.any(axis=1))
        if ys.size == 0:
            return None
        xs = np.flatnonzero(mask.any(axis=0))
        return int(ys[0]), int(ys[-1]) + 1, int(xs[0]), int(xs[-1]) + 1

    # ------------------------------------------------------------------ #
    # Mutation -- the only code allowed to change the matrix
    # ------------------------------------------------------------------ #
    def _record(self, y0: int, y1: int, x0: int, x1: int,
                before: np.ndarray, after: np.ndarray) -> None:
        if np.array_equal(before, after):
            return
        self._pending.append(Patch(y0, y1, x0, x1, before, after))
        self._revision += 1
        # Remember *where* it changed so the downscaled view can be patched
        # rather than rebuilt.  Rebuilding means two full-resolution resizes,
        # which at 1920x1080 costs more than the graph cut itself.
        self._dirty.append((y0, y1, x0, x1))

    def apply(
        self,
        stamp: np.ndarray,
        label: int,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Write ``label`` wherever ``stamp`` is set.

        ``stamp`` is a full-resolution boolean array (or a crop, when ``bbox``
        is given). Painting the opposite label over an existing mark overrides
        it -- that is the *only* way a mark changes, and it is deliberate: the
        user explicitly said "no, the other thing" about those pixels.

        Returns ``True`` if anything actually changed.
        """
        if bbox is None:
            y0, y1, x0, x1 = 0, self.height, 0, self.width
            sub_stamp = stamp
        else:
            y0, y1, x0, x1 = bbox
            y0, x0 = max(0, y0), max(0, x0)
            y1, x1 = min(self.height, y1), min(self.width, x1)
            if y1 <= y0 or x1 <= x0:
                return False
            sub_stamp = stamp if stamp.shape == (y1 - y0, x1 - x0) else stamp[y0:y1, x0:x1]

        region = self.matrix[y0:y1, x0:x1]
        before = region.copy()
        region[sub_stamp] = np.int8(label)
        self._record(y0, y1, x0, x1, before, region.copy())
        return not np.array_equal(before, region)

    def erase(
        self,
        stamp: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> bool:
        """Return marked pixels to unlabelled, freeing them for the algorithm."""
        return self.apply(stamp, UNLABELED, bbox)

    def seed_region(self, mask: np.ndarray, label: int) -> bool:
        """Mark a whole region at once (used to pin committed instances).

        Only writes where the pixel is currently unlabelled, so a bulk seed can
        never silently override something the user painted by hand.
        """
        target = mask & (self.matrix == UNLABELED)
        if not target.any():
            return False
        before = self.matrix.copy()
        self.matrix[target] = np.int8(label)
        self._record(0, self.height, 0, self.width, before, self.matrix.copy())
        return True

    def commit(self) -> None:
        """Close the current edit group so undo treats it as one step.

        Called on mouse release: a drag is many stamps but one undo step.
        """
        if self._pending:
            self.history.push(self._pending)
            self._pending = []

    def abandon_pending(self) -> None:
        self._pending = []

    # ------------------------------------------------------------------ #
    def clear_all(self) -> bool:
        """Drop every constraint. Distinct from undo: this is not reversible
        by one undo step, it is a deliberate 'start this instance over'."""
        if not self.matrix.any():
            return False
        before = self.matrix.copy()
        self.matrix[:] = UNLABELED
        self._record(0, self.height, 0, self.width, before, self.matrix.copy())
        self.commit()
        return True

    def clear_label(self, label: int) -> bool:
        """Drop only the positive, or only the negative, marks."""
        target = (self.matrix > 0) if label > 0 else (self.matrix < 0)
        if not target.any():
            return False
        before = self.matrix.copy()
        self.matrix[target] = UNLABELED
        self._record(0, self.height, 0, self.width, before, self.matrix.copy())
        self.commit()
        return True

    # ------------------------------------------------------------------ #
    # Undo / redo
    # ------------------------------------------------------------------ #
    def undo(self) -> bool:
        patches = self.history.pop_undo()
        if patches is None:
            return False
        for p in reversed(patches):
            self.matrix[p.y0 : p.y1, p.x0 : p.x1] = p.before
            self._dirty.append((p.y0, p.y1, p.x0, p.x1))
        self._revision += 1
        return True

    def redo(self) -> bool:
        patches = self.history.pop_redo()
        if patches is None:
            return False
        for p in patches:
            self.matrix[p.y0 : p.y1, p.x0 : p.x1] = p.after
            self._dirty.append((p.y0, p.y1, p.x0, p.x1))
        self._revision += 1
        return True

    @property
    def can_undo(self) -> bool:
        return self.history.can_undo

    @property
    def can_redo(self) -> bool:
        return self.history.can_redo

    # ------------------------------------------------------------------ #
    # Derived views
    # ------------------------------------------------------------------ #
    def at_scale(self, height: int, width: int) -> np.ndarray:
        """Downscaled copy for the working resolution, as int8 +1/-1/0.

        Nearest-neighbour would be wrong here. A one-pixel brush line
        downscaled 4x lands on a quarter of a target pixel, and nearest
        sampling drops it roughly three times out of four -- so a constraint
        the user painted would simply vanish from the solve. Instead each label
        is resampled by *area coverage* and any non-trivial coverage counts,
        with the stronger coverage winning where positive and negative land in
        the same target pixel.
        """
        if self._scaled is not None and self._scaled_shape == (height, width):
            if self._dirty:
                self._refresh_dirty(height, width)
            return self._scaled

        self._scaled = self._resample(
            0, self.height, 0, self.width, height, width
        )
        self._scaled_shape = (height, width)
        self._dirty = []
        return self._scaled

    def _resample(
        self, y0: int, y1: int, x0: int, x1: int, out_h: int, out_w: int
    ) -> np.ndarray:
        """Resample a full-resolution slab to ``(out_h, out_w)`` int8 labels."""
        region = self.matrix[y0:y1, x0:x1]
        if (y1 - y0, x1 - x0) == (out_h, out_w):
            return region.copy()
        pos = (region > 0).astype(np.float32)
        neg = (region < 0).astype(np.float32)
        pos = (cv2.resize(pos, (out_w, out_h), interpolation=cv2.INTER_AREA)
               if pos.any() else np.zeros((out_h, out_w), np.float32))
        neg = (cv2.resize(neg, (out_w, out_h), interpolation=cv2.INTER_AREA)
               if neg.any() else np.zeros((out_h, out_w), np.float32))
        eps = 1e-3
        out = np.zeros((out_h, out_w), dtype=np.int8)
        out[(pos > eps) & (pos >= neg)] = POSITIVE
        out[(neg > eps) & (neg > pos)] = NEGATIVE
        return out

    def _refresh_dirty(self, height: int, width: int) -> None:
        """Patch only the rectangles that changed into the cached small copy.

        Each dirty rectangle is padded by a couple of source pixels before
        resampling, because area-averaging near a crop edge differs slightly
        from the same average taken over the whole image; the padding is then
        discarded so only interior values are written.
        """
        sy = height / float(self.height)
        sx = width / float(self.width)
        pad = int(np.ceil(1.0 / max(min(sy, sx), 1e-6))) + 2
        total = 0
        for (y0, y1, x0, x1) in self._dirty:
            total += (y1 - y0) * (x1 - x0)
        # Past roughly a third of the image, one clean full resample is cheaper
        # than many padded partial ones.
        if total > 0.33 * self.height * self.width:
            self._scaled = self._resample(
                0, self.height, 0, self.width, height, width
            )
            self._dirty = []
            return

        for (y0, y1, x0, x1) in self._dirty:
            py0, py1 = max(0, y0 - pad), min(self.height, y1 + pad)
            px0, px1 = max(0, x0 - pad), min(self.width, x1 + pad)
            # Snap the source window to whole destination pixels so the crop
            # maps onto an integral block of the cached array.
            dy0, dy1 = int(np.floor(py0 * sy)), int(np.ceil(py1 * sy))
            dx0, dx1 = int(np.floor(px0 * sx)), int(np.ceil(px1 * sx))
            dy0, dx0 = max(0, dy0), max(0, dx0)
            dy1, dx1 = min(height, dy1), min(width, dx1)
            if dy1 <= dy0 or dx1 <= dx0:
                continue
            sub = self._resample(
                int(dy0 / sy), min(self.height, int(np.ceil(dy1 / sy))),
                int(dx0 / sx), min(self.width, int(np.ceil(dx1 / sx))),
                dy1 - dy0, dx1 - dx0,
            )
            self._scaled[dy0:dy1, dx0:dx1] = sub
        self._dirty = []

    def enforce(self, mask: np.ndarray) -> np.ndarray:
        """Stamp the constraints onto a derived mask.

        The last thing every segmentation pass does. Even if a local solve
        never looked at a distant marked pixel, this guarantees the returned
        mask agrees with every mark on the image -- which is the whole point of
        the module.
        """
        if mask.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"mask is {mask.shape[:2]}, constraints are "
                f"{(self.height, self.width)}"
            )
        out = mask if mask.dtype == bool else mask.astype(bool)
        out = out.copy()
        out[self.matrix > 0] = True
        out[self.matrix < 0] = False
        return out

    def enforce_inplace(self, mask: np.ndarray) -> np.ndarray:
        mask[self.matrix > 0] = True
        mask[self.matrix < 0] = False
        return mask

    # ------------------------------------------------------------------ #
    def snapshot(self) -> np.ndarray:
        """A copy of the matrix, for callers that need to stash it."""
        return self.matrix.copy()

    def restore(self, matrix: np.ndarray) -> None:
        """Replace the matrix wholesale (used when switching images)."""
        if matrix.shape != (self.height, self.width):
            raise ValueError("shape mismatch")
        before = self.matrix.copy()
        self.matrix[:] = matrix.astype(np.int8)
        self._record(0, self.height, 0, self.width, before, self.matrix.copy())
        self.commit()


def stamp_bbox(
    stamp_roi: Tuple[int, int, int, int], margin: int, height: int, width: int
) -> Tuple[int, int, int, int]:
    """Expand a stamp bounding box by ``margin``, clipped to the image."""
    y0, y1, x0, x1 = stamp_roi
    return (
        max(0, y0 - margin),
        min(height, y1 + margin),
        max(0, x0 - margin),
        min(width, x1 + margin),
    )
