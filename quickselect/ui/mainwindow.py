"""Main application window.

Layout mirrors Photoshop closely enough to be familiar: an options bar across
the top for the active tool, the canvas in the middle, a layers dock on the
right, a status bar at the bottom.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PySide6.QtCore import QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QSpinBox,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from ..config import EngineConfig, ToolState
from ..diff import save_mask
from ..engine import QuickSelectEngine, SelectionMode
from ..history import History
from ..imagedata import Layer, LayerStack
from ..io_utils import IMAGE_SUFFIXES, cutout_rgba, load_image, load_image_rgba, save_image
from .canvas import STYLE_ANTS, STYLE_BOTH, STYLE_CUTOUT, STYLE_OVERLAY, Canvas
from .worker import EngineWorker


class MainWindow(QMainWindow):
    def __init__(self, image_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Quick Select")
        self.resize(1360, 900)

        self.cfg = EngineConfig()
        self.tool = ToolState()
        self.history = History(limit=self.cfg.max_history)

        self.canvas = Canvas(self)
        self.setCentralWidget(self.canvas)

        self.engine: Optional[QuickSelectEngine] = None
        self.worker: Optional[EngineWorker] = None
        self.thread: Optional[QThread] = None
        self._image_path: Optional[Path] = None
        self._alpha: Optional[np.ndarray] = None

        self._build_actions()
        self._build_options_bar()
        self._build_layers_dock()
        self.statusBar().showMessage("Open an image to begin  (Ctrl+O)")

        self.canvas.strokeBegan.connect(self._on_stroke_began)
        self.canvas.strokeMoved.connect(self._on_stroke_moved)
        self.canvas.strokeEnded.connect(self._on_stroke_ended)
        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        self.canvas.cursorMoved.connect(self._on_cursor_moved)

        if image_path:
            self.open_path(Path(image_path))

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._act(file_menu, "&Open Image...", "Ctrl+O", self.open_image)
        self._act(file_menu, "Place Image as &Layer...", "Ctrl+Shift+O", self.add_layer)
        file_menu.addSeparator()
        self._act(file_menu, "Save &Mask...", "Ctrl+S", self.save_mask_as)
        self._act(file_menu, "Save &Cut-out (RGBA)...", "Ctrl+Shift+S", self.save_cutout)
        file_menu.addSeparator()
        self._act(file_menu, "&Quit", "Ctrl+Q", self.close)

        edit_menu = self.menuBar().addMenu("&Edit")
        self.undo_action = self._act(edit_menu, "&Undo", "Ctrl+Z", self.undo)
        self.redo_action = self._act(edit_menu, "&Redo", "Ctrl+Shift+Z", self.redo)
        edit_menu.addSeparator()
        self._act(edit_menu, "&Deselect", "Ctrl+D", self.deselect)
        self._act(edit_menu, "&Invert Selection", "Ctrl+Shift+I", self.invert)
        self._act(edit_menu, "&Refine Now", "Ctrl+R", self.refine_now)

        view_menu = self.menuBar().addMenu("&View")
        self._act(view_menu, "Zoom &In", "Ctrl+=", lambda: self._zoom_by(1.25))
        self._act(view_menu, "Zoom &Out", "Ctrl+-", lambda: self._zoom_by(0.8))
        self._act(view_menu, "&Fit on Screen", "Ctrl+0", self.canvas.fit_and_hold)
        self._act(view_menu, "&100%", "Ctrl+1", self.canvas.zoom_actual)

        # Brush size, Photoshop's [ and ].
        self._act(self, "Brush smaller", "[", lambda: self._nudge_brush(1 / 1.25))
        self._act(self, "Brush larger", "]", lambda: self._nudge_brush(1.25))
        self._act(self, "Softer brush", "{", lambda: self._nudge_hardness(-0.1))
        self._act(self, "Harder brush", "}", lambda: self._nudge_hardness(+0.1))

    def _act(self, parent, text: str, shortcut: str, slot) -> QAction:
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(QKeySequence(shortcut))
            action.setShortcutContext(Qt.ApplicationShortcut)
        action.triggered.connect(slot)
        if isinstance(parent, QMainWindow):
            self.addAction(action)
        else:
            parent.addAction(action)
        return action

    def _build_options_bar(self) -> None:
        bar = QToolBar("Options", self)
        bar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, bar)

        # --- selection mode ---
        bar.addWidget(QLabel("  Mode "))
        self.mode_box = QComboBox()
        self.mode_box.addItems(["New selection", "Add to selection", "Subtract from selection"])
        self.mode_box.setCurrentIndex(1)
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        self.canvas.mode = SelectionMode.ADD
        bar.addWidget(self.mode_box)

        # --- brush ---
        bar.addSeparator()
        bar.addWidget(QLabel("  Size "))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(1, 600)
        self.size_slider.setValue(int(self.tool.brush.diameter))
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        bar.addWidget(self.size_slider)
        self.size_label = QLabel(f"{int(self.tool.brush.diameter)} px ")
        self.size_label.setFixedWidth(52)
        bar.addWidget(self.size_label)

        bar.addWidget(QLabel(" Hardness "))
        self.hardness_slider = QSlider(Qt.Horizontal)
        self.hardness_slider.setRange(0, 100)
        self.hardness_slider.setValue(int(self.tool.brush.hardness * 100))
        self.hardness_slider.setFixedWidth(90)
        self.hardness_slider.valueChanged.connect(self._on_hardness_changed)
        bar.addWidget(self.hardness_slider)

        bar.addWidget(QLabel(" Spacing "))
        self.spacing_slider = QSlider(Qt.Horizontal)
        self.spacing_slider.setRange(5, 100)
        self.spacing_slider.setValue(int(self.tool.brush.spacing * 100))
        self.spacing_slider.setFixedWidth(80)
        self.spacing_slider.valueChanged.connect(self._on_spacing_changed)
        bar.addWidget(self.spacing_slider)

        # --- options ---
        bar.addSeparator()
        self.auto_enhance = QCheckBox("Auto-Enhance")
        self.auto_enhance.setToolTip(
            "Re-cut the boundary at full resolution with the edge term boosted,\n"
            "so it snaps onto image gradients.  Costs ~100 ms on mouse release."
        )
        self.auto_enhance.toggled.connect(self._on_auto_enhance)
        bar.addWidget(self.auto_enhance)

        self.sample_all = QCheckBox("Sample All Layers")
        self.sample_all.setChecked(True)
        self.sample_all.toggled.connect(self._on_sample_all)
        bar.addWidget(self.sample_all)

        bar.addSeparator()
        bar.addWidget(QLabel(" Show "))
        self.style_box = QComboBox()
        self.style_box.addItems(["Ants + overlay", "Marching ants", "Overlay", "Cut-out"])
        self.style_box.currentIndexChanged.connect(self._on_style_changed)
        bar.addWidget(self.style_box)

        # --- refine edge (second row) ---
        refine_bar = QToolBar("Refine Edge", self)
        refine_bar.setMovable(False)
        self.addToolBarBreak(Qt.TopToolBarArea)
        self.addToolBar(Qt.TopToolBarArea, refine_bar)

        refine_bar.addWidget(QLabel("  Refine Edge:   Feather "))
        self.feather_spin = QDoubleSpinBox()
        self.feather_spin.setRange(0.0, 50.0)
        self.feather_spin.setSingleStep(0.5)
        self.feather_spin.setSuffix(" px")
        self.feather_spin.valueChanged.connect(self._on_refine_changed)
        refine_bar.addWidget(self.feather_spin)

        refine_bar.addWidget(QLabel(" Smooth "))
        self.smooth_spin = QSpinBox()
        self.smooth_spin.setRange(0, 30)
        self.smooth_spin.valueChanged.connect(self._on_refine_changed)
        refine_bar.addWidget(self.smooth_spin)

        refine_bar.addWidget(QLabel(" Contract / Expand "))
        self.shift_spin = QSpinBox()
        self.shift_spin.setRange(-50, 50)
        self.shift_spin.setSuffix(" px")
        self.shift_spin.valueChanged.connect(self._on_refine_changed)
        refine_bar.addWidget(self.shift_spin)

        self.edge_aware = QCheckBox("Edge-aware (guided filter)")
        self.edge_aware.setChecked(True)
        self.edge_aware.toggled.connect(self._on_refine_changed)
        refine_bar.addWidget(self.edge_aware)

        refine_bar.addSeparator()
        apply_btn = QPushButton("Apply")
        apply_btn.clicked.connect(self.refine_now)
        refine_bar.addWidget(apply_btn)

    def _build_layers_dock(self) -> None:
        dock = QDockWidget("Layers", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea | Qt.LeftDockWidgetArea)
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 6, 6, 6)

        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self._on_layer_selected)
        self.layer_list.itemChanged.connect(self._on_layer_visibility)
        layout.addWidget(self.layer_list)

        row = QHBoxLayout()
        add = QPushButton("Place...")
        add.clicked.connect(self.add_layer)
        row.addWidget(add)
        layout.addLayout(row)

        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    # ------------------------------------------------------------------ #
    # Document
    # ------------------------------------------------------------------ #
    def open_image(self) -> None:
        patterns = " ".join(f"*{s}" for s in sorted(IMAGE_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Image", "", f"Images ({patterns});;All files (*)"
        )
        if path:
            self.open_path(Path(path))

    def open_path(self, path: Path) -> None:
        try:
            rgb = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Open failed", str(exc))
            return
        self._image_path = path
        self._teardown_worker()
        stack = LayerStack.from_image(rgb, name=path.name)
        self.engine = QuickSelectEngine(stack, self.cfg, self.tool)
        self._start_worker()
        self.canvas.set_image(rgb)
        self.canvas.set_selection(None)
        self.history.clear()
        self._alpha = None
        self._refresh_layers()
        self.setWindowTitle(f"Quick Select - {path.name}")
        self.statusBar().showMessage(
            f"{rgb.shape[1]} x {rgb.shape[0]}   "
            f"interactive solve at {self.engine.pyramid.interactive.width} x "
            f"{self.engine.pyramid.interactive.height}"
        )

    def add_layer(self) -> None:
        if self.engine is None:
            QMessageBox.information(self, "No document", "Open an image first.")
            return
        patterns = " ".join(f"*{s}" for s in sorted(IMAGE_SUFFIXES))
        path, _ = QFileDialog.getOpenFileName(
            self, "Place Image as Layer", "", f"Images ({patterns});;All files (*)"
        )
        if not path:
            return
        try:
            rgb, alpha = load_image_rgba(path)
        except Exception as exc:
            QMessageBox.critical(self, "Place failed", str(exc))
            return
        stack = self.engine.layers
        import cv2

        if rgb.shape[:2] != (stack.height, stack.width):
            rgb = cv2.resize(rgb, (stack.width, stack.height))
            if alpha is not None:
                alpha = cv2.resize(alpha, (stack.width, stack.height))
        stack.layers.append(Layer(name=Path(path).name, rgb=rgb, alpha=alpha))
        stack.active_index = len(stack.layers) - 1
        self._reload_source()

    def _reload_source(self) -> None:
        if self.engine is None:
            return
        self.engine.refresh_source(self.sample_all.isChecked())
        self.canvas.set_image(self.engine.layers.composite())
        self.canvas.set_selection(self.engine.full_alpha)
        self._refresh_layers()

    def _refresh_layers(self) -> None:
        self.layer_list.blockSignals(True)
        self.layer_list.clear()
        if self.engine is not None:
            for i, lyr in enumerate(reversed(self.engine.layers.layers)):
                item = QListWidgetItem(lyr.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if lyr.visible else Qt.Unchecked)
                item.setData(Qt.UserRole, len(self.engine.layers.layers) - 1 - i)
                self.layer_list.addItem(item)
            active = self.engine.layers.active_index
            for row in range(self.layer_list.count()):
                if self.layer_list.item(row).data(Qt.UserRole) == active:
                    self.layer_list.setCurrentRow(row)
                    break
        self.layer_list.blockSignals(False)

    def _on_layer_selected(self, row: int) -> None:
        if self.engine is None or row < 0:
            return
        item = self.layer_list.item(row)
        if item is not None:
            self.engine.layers.active_index = int(item.data(Qt.UserRole))
            if not self.sample_all.isChecked():
                self._reload_source()

    def _on_layer_visibility(self, item: QListWidgetItem) -> None:
        if self.engine is None:
            return
        idx = int(item.data(Qt.UserRole))
        self.engine.layers.layers[idx].visible = item.checkState() == Qt.Checked
        self._reload_source()

    # ------------------------------------------------------------------ #
    # Worker plumbing
    # ------------------------------------------------------------------ #
    def _start_worker(self) -> None:
        assert self.engine is not None
        self.thread = QThread(self)
        self.worker = EngineWorker(self.engine)
        self.worker.moveToThread(self.thread)
        self.worker.maskReady.connect(self._on_mask_ready)
        self.worker.finalReady.connect(self._on_final_ready)
        self.worker.busyChanged.connect(self._on_busy)
        self.worker.errorRaised.connect(self._on_worker_error)
        self.thread.start()

    def _teardown_worker(self) -> None:
        if self.thread is not None:
            self.thread.quit()
            self.thread.wait(3000)
            self.thread = None
            self.worker = None

    @Slot(object)
    def _on_mask_ready(self, mask) -> None:
        self.canvas.set_selection(mask)

    @Slot(object)
    def _on_final_ready(self, alpha) -> None:
        self._alpha = alpha
        self.canvas.set_selection(alpha)
        if alpha is not None:
            count = int((alpha > 0.5).sum())
            self.statusBar().showMessage(f"{count:,} pixels selected")

    @Slot(bool)
    def _on_busy(self, busy: bool) -> None:
        if busy:
            self.statusBar().showMessage("Working...")

    @Slot(str)
    def _on_worker_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Engine error: {message}", 8000)

    # ------------------------------------------------------------------ #
    # Gestures
    # ------------------------------------------------------------------ #
    @Slot(float, float, object)
    def _on_stroke_began(self, x: float, y: float, mode) -> None:
        if self.worker is None or self.engine is None:
            return
        # Snapshot for undo happens on the GUI side so the step is recorded
        # even if the engine is still chewing on the previous gesture.
        self.history.push(self.engine.snapshot())
        self.worker.begin(x, y, mode)

    @Slot(float, float)
    def _on_stroke_moved(self, x: float, y: float) -> None:
        if self.worker is not None:
            self.worker.move(x, y)

    @Slot()
    def _on_stroke_ended(self) -> None:
        if self.worker is not None:
            self.worker.end()
        # After the first stroke of a "New selection" gesture, behave like
        # Photoshop and switch to Add so the next stroke extends it.
        if self.mode_box.currentIndex() == 0:
            self.mode_box.setCurrentIndex(1)

    # ------------------------------------------------------------------ #
    # Option handlers
    # ------------------------------------------------------------------ #
    def _on_mode_changed(self, index: int) -> None:
        self.canvas.mode = [
            SelectionMode.NEW, SelectionMode.ADD, SelectionMode.SUBTRACT
        ][index]

    def _on_size_changed(self, value: int) -> None:
        self.tool.brush.diameter = float(value)
        self.canvas.brush_diameter = float(value)
        self.size_label.setText(f"{value} px ")
        self.canvas.update()

    def _nudge_brush(self, factor: float) -> None:
        new = int(round(np.clip(self.tool.brush.diameter * factor, 1, 600)))
        if new == int(self.tool.brush.diameter):
            new += 1 if factor > 1 else -1
        self.size_slider.setValue(int(np.clip(new, 1, 600)))

    def _on_hardness_changed(self, value: int) -> None:
        self.tool.brush.hardness = value / 100.0
        self.canvas.brush_hardness = self.tool.brush.hardness
        self.canvas.update()

    def _nudge_hardness(self, delta: float) -> None:
        self.hardness_slider.setValue(
            int(np.clip((self.tool.brush.hardness + delta) * 100, 0, 100))
        )

    def _on_spacing_changed(self, value: int) -> None:
        self.tool.brush.spacing = value / 100.0

    def _on_auto_enhance(self, checked: bool) -> None:
        self.tool.auto_enhance = checked

    def _on_sample_all(self, checked: bool) -> None:
        if self.engine is None:
            return
        self.engine.refresh_source(checked)
        # The canvas always shows the composite; "Sample All Layers" only
        # changes which pixels the *selection* is computed from.
        self.canvas.set_image(self.engine.layers.composite())

    def _on_style_changed(self, index: int) -> None:
        self.canvas.style = [STYLE_BOTH, STYLE_ANTS, STYLE_OVERLAY, STYLE_CUTOUT][index]
        self.canvas.update()

    def _on_refine_changed(self, *_) -> None:
        self.tool.refine.feather = float(self.feather_spin.value())
        self.tool.refine.smooth = int(self.smooth_spin.value())
        self.tool.refine.shift_edge = int(self.shift_spin.value())
        self.tool.refine.edge_aware = self.edge_aware.isChecked()

    def _on_zoom_changed(self, zoom: float) -> None:
        self.statusBar().showMessage(f"Zoom {zoom * 100:.0f}%", 1500)

    def _on_cursor_moved(self, x: float, y: float) -> None:
        pass

    def _zoom_by(self, factor: float) -> None:
        self.canvas.set_zoom(self.canvas.zoom * factor)

    # ------------------------------------------------------------------ #
    # Edit operations
    # ------------------------------------------------------------------ #
    def undo(self) -> None:
        if self.engine is None or self.worker is None:
            return
        state = self.history.undo(self.engine.snapshot())
        if state is None:
            self.statusBar().showMessage("Nothing to undo", 1500)
            return
        self.engine.restore(state)
        self.canvas.set_selection(self.engine.full_alpha)
        self._alpha = self.engine.full_alpha

    def redo(self) -> None:
        if self.engine is None:
            return
        state = self.history.redo(self.engine.snapshot())
        if state is None:
            self.statusBar().showMessage("Nothing to redo", 1500)
            return
        self.engine.restore(state)
        self.canvas.set_selection(self.engine.full_alpha)
        self._alpha = self.engine.full_alpha

    def deselect(self) -> None:
        if self.engine is None:
            return
        self.history.push(self.engine.snapshot())
        self.engine.clear()
        self._alpha = None
        self.canvas.set_selection(None)

    def invert(self) -> None:
        if self.engine is None:
            return
        self.history.push(self.engine.snapshot())
        self.engine.invert()
        self._alpha = self.engine.full_alpha
        self.canvas.set_selection(self.engine.full_alpha)

    def refine_now(self) -> None:
        if self.engine is None or self.worker is None:
            return
        from PySide6.QtCore import QMetaObject

        QMetaObject.invokeMethod(self.worker, "refine", Qt.QueuedConnection)

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #
    def _current_alpha(self) -> Optional[np.ndarray]:
        if self.engine is None:
            return None
        return self.engine.selection_alpha()

    def save_mask_as(self) -> None:
        alpha = self._current_alpha()
        if alpha is None or not alpha.any():
            QMessageBox.information(self, "Nothing to save", "The selection is empty.")
            return
        default = ""
        if self._image_path:
            default = str(self._image_path.with_name(self._image_path.stem + "_mask.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Save Mask", default, "PNG (*.png)")
        if path:
            save_mask(path, alpha)
            self.statusBar().showMessage(f"Saved {path}", 4000)

    def save_cutout(self) -> None:
        alpha = self._current_alpha()
        if alpha is None or not alpha.any() or self.engine is None:
            QMessageBox.information(self, "Nothing to save", "The selection is empty.")
            return
        default = ""
        if self._image_path:
            default = str(self._image_path.with_name(self._image_path.stem + "_cutout.png"))
        path, _ = QFileDialog.getSaveFileName(self, "Save Cut-out", default, "PNG (*.png)")
        if path:
            save_image(path, cutout_rgba(self.engine.pyramid.full_rgb, alpha))
            self.statusBar().showMessage(f"Saved {path}", 4000)

    def closeEvent(self, event) -> None:  # noqa: N802
        self._teardown_worker()
        super().closeEvent(event)
