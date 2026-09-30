"""How existing instances look in Edit Mode.

Three requirements pull against each other here. Every instance has to be
*distinguishable* from its neighbours, the one under the pointer has to stand
out, and the image underneath has to stay readable -- you are editing
annotations because you need to see what is wrong with them, and a solid wash
of colour hides exactly that.

So the fill stays very light and the *outline* does the work: a thin, fully
opaque border, thickening by state. That reads at any zoom, costs nothing to
draw, and leaves the pixels visible.

**Colour means class.** Every instance of the same category gets the same
hue, so a glance at the image tells you how it is labelled, and a mislabelled
object stands out as the one wrong colour in a group. Instances are still told
apart by their own borders, and each one carries a small chip with its class
id, which is what you change with the number keys.

Everything is composited into one RGBA layer plus a flat list of outline
paths, rather than one canvas overlay per instance: a hundred instances would
otherwise mean a hundred full-resolution RGBA buffers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QColor, QPainterPath

from ..ui.canvas import contour_path
from .session import Instance

#: One hue per class, indexed by ``category_id``. Picked to stay apart from
#: each other and from the tool's own overlay colours (blue = instance in
#: progress, green/red = constraint marks), and to read against the dark
#: canvas without being fluorescent.
PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (232, 160, 58),    # amber
    (86, 196, 214),    # cyan
    (198, 130, 226),   # violet
    (124, 205, 124),   # sage
    (236, 129, 129),   # salmon
    (150, 176, 240),   # periwinkle
    (214, 198, 96),    # ochre
    (108, 198, 176),   # teal
    (226, 146, 190),   # pink
    (168, 190, 110),   # olive
    (240, 176, 120),   # apricot
    (140, 164, 206),   # slate blue
)


@dataclass(frozen=True)
class _Tier:
    """The look of one interaction state."""

    fill_alpha: int
    line_width: float
    line_alpha: int


NORMAL = _Tier(fill_alpha=34, line_width=1.2, line_alpha=200)
HOVER = _Tier(fill_alpha=78, line_width=1.8, line_alpha=235)
SELECTED = _Tier(fill_alpha=124, line_width=2.6, line_alpha=255)

#: Outline drawn under the selected instance's own, so the emphasis survives a
#: light background as well as a dark one.
SELECTED_HALO = QColor(255, 255, 255, 170)

#: ``(path, colour, width)`` -- what :meth:`Canvas.set_paths` consumes.
PathEntry = Tuple[QPainterPath, QColor, float]

#: ``(point, text, colour)`` -- what :meth:`Canvas.set_labels` consumes.
LabelEntry = Tuple[QPointF, str, QColor]


def class_color(category_id: Optional[int]) -> QColor:
    """The colour that stands for one class. Stable across images and runs."""
    index = 0 if category_id is None else int(category_id) % len(PALETTE)
    r, g, b = PALETTE[index]
    return QColor(r, g, b)


def class_label(category_id: Optional[int]) -> str:
    return "?" if category_id is None else str(int(category_id))


def build_instance_layers(
    instances: Sequence[Instance],
    height: int,
    width: int,
    hover: Optional[Instance] = None,
    selected: Optional[Instance] = None,
    default_category: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], List[PathEntry], List[LabelEntry]]:
    """Composite every instance into one RGBA layer, its outlines and chips.

    Returns ``(rgba, paths, labels)``. ``rgba`` is ``None`` when there is
    nothing to draw. Points are in image coordinates; the canvas maps them to
    the view.
    """
    if not instances or height <= 0 or width <= 0:
        return None, [], []

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    paths: List[PathEntry] = []
    emphasis: List[PathEntry] = []
    labels: List[LabelEntry] = []

    # Draw order is by state, not by list position: where two instances
    # overlap, the hovered or selected one has to win both the fill and the
    # outline, whichever of them happens to come first in the list.
    ordered = sorted(
        enumerate(instances),
        key=lambda pair: _priority(pair[1], hover, selected),
    )

    for _index, inst in ordered:
        category = inst.category_id if inst.category_id is not None else default_category
        color = class_color(category)
        tier = _tier(inst, hover, selected)
        _fill(rgba, inst, color, tier.fill_alpha)
        labels.append((_anchor(inst), class_label(category), color))

        path = contour_path(inst.mask)
        if path is None:
            continue
        line = QColor(color)
        line.setAlpha(tier.line_alpha)
        entry = (path, line, tier.line_width)
        if tier is NORMAL:
            paths.append(entry)
            continue
        if tier is SELECTED:
            emphasis.append((path, SELECTED_HALO, tier.line_width + 2.0))
        emphasis.append(entry)

    return rgba, paths + emphasis, labels


def _tier(
    inst: Instance, hover: Optional[Instance], selected: Optional[Instance]
) -> _Tier:
    if inst is selected:
        return SELECTED
    if inst is hover:
        return HOVER
    return NORMAL


def _priority(
    inst: Instance, hover: Optional[Instance], selected: Optional[Instance]
) -> int:
    return {NORMAL: 0, HOVER: 1, SELECTED: 2}[_tier(inst, hover, selected)]


def _anchor(inst: Instance) -> QPointF:
    """Where the class chip sits: the centre of the instance's bounding box."""
    x0, y0, x1, y1 = inst.bounds()
    return QPointF((x0 + x1) / 2.0, (y0 + y1) / 2.0)


def _fill(rgba: np.ndarray, inst: Instance, color: QColor, alpha: int) -> None:
    """Paint one instance into the shared buffer, inside its bbox only."""
    x0, y0, x1, y1 = inst.bounds()
    if x1 <= x0 or y1 <= y0:
        return
    window = rgba[y0:y1, x0:x1]
    mask = inst.mask[y0:y1, x0:x1]
    if not mask.any():
        return
    window[mask] = (color.red(), color.green(), color.blue(), alpha)
