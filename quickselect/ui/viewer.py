"""Side-by-side mask comparison viewer with a blend slider.

Left panel: your mask.  Right panel: the reference mask.  The slider under them
cross-fades a third, superimposed view, which is the quickest way to see a
boundary that has drifted by a few pixels -- a difference that side-by-side
panels hide but a fade makes jump out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from ..diff import blend, load_mask, mask_diff, side_by_side
from ..io_utils import load_image
from .canvas import numpy_to_qimage


class _ImageLabel(QLabel):
    def __init__(self, title: str = "") -> None:
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(200, 200)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._array: Optional[np.ndarray] = None
        self.setStyleSheet("background:#1b1d20;")

    def set_array(self, rgb: np.ndarray) -> None:
        self._array = rgb
        self._rescale()

    def resizeEvent(self, event):  # noqa: N802
        self._rescale()
        super().resizeEvent(event)

    def _rescale(self) -> None:
        if self._array is None:
            return
        from PySide6.QtGui import QPixmap

        pix = QPixmap.fromImage(numpy_to_qimage(self._array))
        self.setPixmap(
            pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


class CompareWindow(QMainWindow):
    """Compare predicted masks against reference masks, one pair at a time."""

    def __init__(
        self,
        pairs: List[Tuple[str, Path, Path, Optional[Path]]],
    ) -> None:
        super().__init__()
        self.setWindowTitle("Mask Comparison")
        self.resize(1400, 820)
        self.pairs = pairs
        self._cache: dict = {}

        central = QWidget()
        root = QVBoxLayout(central)

        top = QHBoxLayout()
        top.addWidget(QLabel("Pair:"))
        self.picker = QComboBox()
        self.picker.addItems([p[0] for p in pairs])
        self.picker.currentIndexChanged.connect(self._reload)
        top.addWidget(self.picker, 1)
        root.addLayout(top)

        panels = QHBoxLayout()
        self.left = _ImageLabel()
        self.middle = _ImageLabel()
        self.right = _ImageLabel()
        for widget, caption in (
            (self.left, "Yours"),
            (self.middle, "Blend / difference"),
            (self.right, "Reference"),
        ):
            column = QVBoxLayout()
            label = QLabel(caption)
            label.setAlignment(Qt.AlignCenter)
            column.addWidget(label)
            column.addWidget(widget, 1)
            panels.addLayout(column, 1)
        root.addLayout(panels, 1)

        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Yours"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self._update_middle)
        slider_row.addWidget(self.slider, 1)
        slider_row.addWidget(QLabel("Reference"))
        root.addLayout(slider_row)

        hint = QLabel(
            "Middle panel: drag the slider to cross-fade.  "
            "Park it at either end to see one mask alone, or press D for the "
            "colour-coded difference (orange = only yours, blue = only the reference)."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9aa0a6;")
        root.addWidget(hint)

        self.setCentralWidget(central)
        self._show_diff = False
        if pairs:
            self._reload(0)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_D:
            self._show_diff = not self._show_diff
            self._update_middle()
        elif event.key() == Qt.Key_Right:
            self.picker.setCurrentIndex(
                min(self.picker.currentIndex() + 1, self.picker.count() - 1)
            )
        elif event.key() == Qt.Key_Left:
            self.picker.setCurrentIndex(max(self.picker.currentIndex() - 1, 0))
        else:
            super().keyPressEvent(event)

    def _current(self):
        idx = self.picker.currentIndex()
        if idx < 0 or idx >= len(self.pairs):
            return None
        name, pred_path, ref_path, img_path = self.pairs[idx]
        if name not in self._cache:
            image = load_image(img_path) if img_path else None
            self._cache[name] = (
                load_mask(pred_path),
                load_mask(ref_path),
                image,
            )
        return self._cache[name]

    def _reload(self, _index: int = 0) -> None:
        data = self._current()
        if data is None:
            return
        pred, ref, image = data
        panels = side_by_side(pred, ref, image)
        half = panels.shape[1] // 2
        self.left.set_array(panels[:, :half])
        self.right.set_array(panels[:, half:])
        self._update_middle()

    def _update_middle(self) -> None:
        data = self._current()
        if data is None:
            return
        pred, ref, image = data
        if self._show_diff:
            self.middle.set_array(mask_diff(pred, ref, image))
        else:
            self.middle.set_array(blend(pred, ref, self.slider.value() / 100.0, image))


def collect_pairs(
    pred_dir: Path, ref_dir: Path, image_dir: Optional[Path]
) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    """Pair mask files by stem, ignoring extension differences."""
    refs = {p.stem: p for p in sorted(ref_dir.glob("*")) if p.is_file()}
    images = {}
    if image_dir and image_dir.is_dir():
        images = {p.stem: p for p in sorted(image_dir.glob("*")) if p.is_file()}

    pairs = []
    for pred in sorted(pred_dir.glob("*")):
        if not pred.is_file():
            continue
        stem = pred.stem
        ref = refs.get(stem) or refs.get(stem.replace("_mask", ""))
        if ref is None:
            continue
        img = images.get(stem) or images.get(stem.replace("_mask", ""))
        pairs.append((stem, pred, ref, img))
    return pairs


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quickselect-compare",
        description="Side-by-side mask viewer with a blend slider.",
    )
    parser.add_argument("predicted", type=Path, help="folder of your masks")
    parser.add_argument("reference", type=Path, help="folder of reference masks")
    parser.add_argument(
        "--images", type=Path, default=None,
        help="optional folder of source photos, used as the backdrop",
    )
    args = parser.parse_args(argv)

    pairs = collect_pairs(args.predicted, args.reference, args.images)
    if not pairs:
        print("No matching mask pairs found (files are paired by filename stem).")
        return 1

    app = QApplication(sys.argv[:1])
    from .app import apply_dark_palette

    apply_dark_palette(app)
    window = CompareWindow(pairs)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
