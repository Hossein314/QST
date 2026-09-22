"""Annotation tool for electronic boards.

    python annotate.py /path/to/images

Opens a folder, walks it with the arrow keys, and writes one COCO
``annotations.json`` beside the images. Each object you paint and commit
becomes one instance with ``category_id`` 4.

The window is deliberately plain: a big canvas, one row of tool options, one
row of workflow buttons, and a status bar that tells you where you are and how
long the last segmentation took. Everything else is a keyboard shortcut,
because annotation is a repetitive job and reaching for the mouse costs time.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PySide6.QtCore import QEvent, QMetaObject, QThread, Qt, Slot
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QToolBar,
    QWidget,
)

from ..config import BrushConfig
from ..constraints import NEGATIVE, POSITIVE
from ..engine import SelectionMode
from ..io_utils import load_image
from ..segmenter import SegmentParams
from ..ui.canvas import INTERACT_PAINT, INTERACT_PICK, SELECTION, Canvas
from .dataset_io import (
    DEFAULT_CATEGORY_ID,
    DEFAULT_CATEGORY_NAME,
    DEFAULT_FILENAME,
    CocoDataset,
    scan_folder,
)
from .editing import AppMode, InstanceEditor
from .instance_view import build_instance_layers
from .session import AnnotationSession

# Overlay layers, drawn in this order.
LAYER_COMMITTED = "committed"
LAYER_CURRENT = SELECTION          # reuses the canvas' ants-enabled layer
LAYER_POS = "constraint_pos"
LAYER_NEG = "constraint_neg"
# Edit Mode draws every existing instance into one layer, with its outlines in
# a matching path layer.
LAYER_INSTANCES = "edit_instances"
PATHS_INSTANCES = "edit_outlines"

#: Layers that belong to the annotation workflow and have no place in Edit Mode.
ANNOTATE_LAYERS = (LAYER_COMMITTED, LAYER_CURRENT, LAYER_POS, LAYER_NEG)

COLOR_COMMITTED = QColor(232, 160, 58, 95)
COLOR_CURRENT = QColor(64, 132, 255, 110)
COLOR_POS = QColor(80, 220, 120, 215)
COLOR_NEG = QColor(240, 90, 90, 215)


class AnnotatorWindow(QMainWindow):
    def __init__(self, folder: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Board Annotator")
        self.resize(1500, 950)

        self.params = SegmentParams()
        self.brush = BrushConfig(diameter=48.0, hardness=0.9, spacing=0.22)
        self.category_id = DEFAULT_CATEGORY_ID
        self.category_name = DEFAULT_CATEGORY_NAME

        self.folder: Optional[Path] = None
        self.paths: List[Path] = []
        self.index = -1
        self.dataset: Optional[CocoDataset] = None
        self.session: Optional[AnnotationSession] = None
        self.app_mode = AppMode.ANNOTATE
        self.editor = InstanceEditor()
        self._image: Optional[np.ndarray] = None
        self._last_timings = ""
        self._busy = False

        self.canvas = Canvas(self)
        self.canvas.brush_diameter = self.brush.diameter
        self.canvas.brush_hardness = self.brush.hardness
        self.canvas.mode = SelectionMode.ADD
        self.setCentralWidget(self.canvas)

        self._build_actions()
        self._build_toolbars()
        self._build_status()
        self._start_worker()

        self.canvas.strokeBegan.connect(self._on_stroke_began)
        self.canvas.strokeMoved.connect(self._on_stroke_moved)
        self.canvas.strokeEnded.connect(self._on_stroke_ended)
        self.canvas.cursorMoved.connect(self._on_cursor_moved)
        self.canvas.picked.connect(self._on_picked)

        # Tab is focus navigation everywhere in Qt, so it never reaches a
        # QAction shortcut. Filtering it at the application level is the only
        # way to own the key -- and swallowing it means it does nothing else.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        if folder:
            self.open_folder(Path(folder))
        else:
            self.statusBar().showMessage("Open an image folder to begin  (Ctrl+O)")

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _act(self, text: str, shortcut, slot, menu=None) -> QAction:
        action = QAction(text, self)
        if shortcut:
            keys = shortcut if isinstance(shortcut, (list, tuple)) else [shortcut]
            action.setShortcuts([QKeySequence(k) for k in keys])
            action.setShortcutContext(Qt.ApplicationShortcut)
        action.triggered.connect(slot)
        (menu or self).addAction(action)
        return action

    def _build_actions(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        self._act("&Open Folder...", "Ctrl+O", self.choose_folder, file_menu)
        self._act("&Save All", "Ctrl+S", self.save_all, file_menu)
        file_menu.addSeparator()
        self._act("&Quit", "Ctrl+Q", self.close, file_menu)

        nav = self.menuBar().addMenu("&Navigate")
        self._act("&Previous Image", ["Left", "PgUp"], self.previous_image, nav)
        self._act("&Next Image", ["Right", "PgDown"], self.next_image, nav)
        self._act("&First Image", "Home", lambda: self.goto(0), nav)
        self._act("&Last Image", "End", lambda: self.goto(len(self.paths) - 1), nav)

        inst = self.menuBar().addMenu("&Instance")
        self._act("&Commit Instance", ["Return", "Enter"], self.commit_instance, inst)
        self._act("&Delete Last Instance", "Ctrl+Backspace", self.delete_last, inst)
        self._act("Clear Current C&onstraints", "Ctrl+K", self.clear_constraints, inst)
        inst.addSeparator()
        self._act("&Undo", "Ctrl+Z", self.undo, inst)
        self._act("&Redo", ["Ctrl+Shift+Z", "Ctrl+Y"], self.redo, inst)

        # Tab is handled by the event filter, not as a shortcut, so the menu
        # entry only advertises it.
        mode = self.menuBar().addMenu("&Mode")
        self._act("&Toggle Annotate / Edit Mode  (Tab)", None, self.toggle_mode, mode)
        mode.addSeparator()
        self._act(
            "&Delete Selected Instance", "Delete", self.delete_selected, mode
        )
        self._act("Des&elect", "Escape", self.deselect, mode)

        view = self.menuBar().addMenu("&View")
        self._act("Zoom &In", "Ctrl+=", lambda: self._zoom_by(1.25), view)
        self._act("Zoom &Out", "Ctrl+-", lambda: self._zoom_by(0.8), view)
        self._act("&Fit on Screen", "Ctrl+0", self.canvas.fit_and_hold, view)
        self._act("&100%", "Ctrl+1", self.canvas.zoom_actual, view)
        view.addSeparator()
        self.show_constraints_action = QAction("Show &Constraint Marks", self)
        self.show_constraints_action.setCheckable(True)
        self.show_constraints_action.setChecked(True)
        self.show_constraints_action.setShortcut(QKeySequence("C"))
        self.show_constraints_action.setShortcutContext(Qt.ApplicationShortcut)
        self.show_constraints_action.toggled.connect(self._refresh_overlays)
        view.addAction(self.show_constraints_action)
        self.addAction(self.show_constraints_action)

        self.show_committed_action = QAction("Show C&ommitted Instances", self)
        self.show_committed_action.setCheckable(True)
        self.show_committed_action.setChecked(True)
        self.show_committed_action.setShortcut(QKeySequence("V"))
        self.show_committed_action.setShortcutContext(Qt.ApplicationShortcut)
        self.show_committed_action.toggled.connect(self._refresh_overlays)
        view.addAction(self.show_committed_action)
        self.addAction(self.show_committed_action)

        # Brush size, Photoshop's [ and ].
        self._act("Brush smaller", "[", lambda: self._nudge_brush(1 / 1.25))
        self._act("Brush larger", "]", lambda: self._nudge_brush(1.25))

    def _build_toolbars(self) -> None:
        bar = QToolBar("Tool", self)
        bar.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, bar)

        bar.addWidget(QLabel("  Mode "))
        self.mode_box = QComboBox()
        self.mode_box.addItems(
            ["Add to instance", "Subtract (background)", "New instance"]
        )
        self.mode_box.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(self.mode_box)

        bar.addSeparator()
        bar.addWidget(QLabel("  Brush "))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(2, 400)
        self.size_slider.setValue(int(self.brush.diameter))
        self.size_slider.setFixedWidth(150)
        self.size_slider.valueChanged.connect(self._on_size_changed)
        bar.addWidget(self.size_slider)
        self.size_label = QLabel(f"{int(self.brush.diameter)} px ")
        self.size_label.setFixedWidth(52)
        bar.addWidget(self.size_label)

        bar.addWidget(QLabel(" Hardness "))
        self.hardness_slider = QSlider(Qt.Horizontal)
        self.hardness_slider.setRange(0, 100)
        self.hardness_slider.setValue(int(self.brush.hardness * 100))
        self.hardness_slider.setFixedWidth(80)
        self.hardness_slider.valueChanged.connect(self._on_hardness_changed)
        bar.addWidget(self.hardness_slider)

        bar.addWidget(QLabel(" Spacing "))
        self.spacing_slider = QSlider(Qt.Horizontal)
        self.spacing_slider.setRange(5, 100)
        self.spacing_slider.setValue(int(self.brush.spacing * 100))
        self.spacing_slider.setFixedWidth(70)
        self.spacing_slider.valueChanged.connect(self._on_spacing_changed)
        bar.addWidget(self.spacing_slider)

        bar.addSeparator()
        bar.addWidget(QLabel(" Polygon ε "))
        self.eps_spin = QDoubleSpinBox()
        self.eps_spin.setDecimals(4)
        self.eps_spin.setRange(0.0, 0.05)
        self.eps_spin.setSingleStep(0.0005)
        self.eps_spin.setValue(0.0015)
        self.eps_spin.setToolTip(
            "approxPolyDP epsilon as a fraction of each contour's perimeter.\n"
            "Larger = fewer points, coarser outline."
        )
        self.eps_spin.valueChanged.connect(self._on_polygon_opts)
        bar.addWidget(self.eps_spin)

        self.holes_box = QCheckBox("Keep holes")
        self.holes_box.setToolTip(
            "Use RETR_CCOMP so interior holes become separate polygons.\n"
            "Standard COCO has no hole semantics -- pycocotools renders them as\n"
            "additional filled area. Only enable if your loader handles it."
        )
        self.holes_box.toggled.connect(self._on_polygon_opts)
        bar.addWidget(self.holes_box)

        # ---- workflow row ----
        self.addToolBarBreak(Qt.TopToolBarArea)
        flow = QToolBar("Workflow", self)
        flow.setMovable(False)
        self.addToolBar(Qt.TopToolBarArea, flow)

        def button(text: str, slot, tip: str = "") -> QPushButton:
            b = QPushButton(text)
            b.clicked.connect(slot)
            if tip:
                b.setToolTip(tip)
            flow.addWidget(b)
            return b

        button("◀ Prev", self.previous_image, "Left arrow")
        button("Next ▶", self.next_image, "Right arrow")
        flow.addSeparator()
        self.commit_button = button(
            "Commit instance", self.commit_instance, "Enter"
        )
        button("Delete last", self.delete_last, "Ctrl+Backspace")
        button("Clear constraints", self.clear_constraints, "Ctrl+K")
        flow.addSeparator()
        button("Undo", self.undo, "Ctrl+Z")
        button("Redo", self.redo, "Ctrl+Shift+Z")
        flow.addSeparator()
        button("Save all", self.save_all, "Ctrl+S")

    def _build_status(self) -> None:
        # The mode indicator lives in the status bar rather than in a banner of
        # its own: always visible, never in the way of the image.
        self.mode_label = QLabel()
        font = self.mode_label.font()
        font.setBold(True)
        self.mode_label.setFont(font)
        self.image_label = QLabel("no folder")
        self.instance_label = QLabel("")
        self.class_label = QLabel(f"class {self.category_id}: {self.category_name}")
        self.selection_label = QLabel("")
        self.timing_label = QLabel("")
        for w in (self.mode_label, self.image_label, self.instance_label,
                  self.class_label, self.selection_label):
            self.statusBar().addWidget(w)
        self.statusBar().addPermanentWidget(self.timing_label)
        self._update_mode_label()

    def _update_mode_label(self) -> None:
        self.mode_label.setText(f"  [{self.app_mode.label}]  ")
        self.mode_label.setToolTip("Tab switches between Annotate and Edit Mode")

    def _start_worker(self) -> None:
        from .worker import SessionWorker

        self.thread = QThread(self)
        self.worker = SessionWorker()
        self.worker.moveToThread(self.thread)
        self.worker.maskReady.connect(self._on_mask_ready)
        self.worker.strokeFinished.connect(self._on_stroke_finished)
        self.worker.errorRaised.connect(self._on_error)
        self.worker.busyChanged.connect(self._on_busy)
        self.thread.start()

    # ------------------------------------------------------------------ #
    # Folder and navigation
    # ------------------------------------------------------------------ #
    def choose_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Open Image Folder")
        if path:
            self.open_folder(Path(path))

    def open_folder(self, folder: Path) -> None:
        paths = scan_folder(folder)
        if not paths:
            QMessageBox.warning(
                self, "No images", f"No image files found in {folder}"
            )
            return
        self.folder = folder
        self.paths = paths
        self.dataset = CocoDataset.load(
            folder / DEFAULT_FILENAME,
            category_id=self.category_id,
            category_name=self.category_name,
        )
        self.index = -1
        self.setWindowTitle(f"Board Annotator - {folder}")
        self.goto(0)
        self.statusBar().showMessage(
            f"{len(paths)} image(s); existing annotations: {self.dataset.summary()}",
            6000,
        )

    def goto(self, index: int) -> None:
        if not self.paths:
            return
        index = int(np.clip(index, 0, len(self.paths) - 1))
        if index == self.index:
            return
        self._flush_current()
        self.index = index
        self._load_current()

    def next_image(self) -> None:
        if self.index < len(self.paths) - 1:
            self.goto(self.index + 1)
        else:
            self.statusBar().showMessage("Last image", 1500)

    def previous_image(self) -> None:
        if self.index > 0:
            self.goto(self.index - 1)
        else:
            self.statusBar().showMessage("First image", 1500)

    def _load_current(self) -> None:
        path = self.paths[self.index]
        try:
            image = load_image(path)
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", f"{path.name}: {exc}")
            return
        self._image = image
        session = AnnotationSession(image, self.params, self.brush)
        session.polygon_epsilon = float(self.eps_spin.value())
        session.polygon_mode = "ccomp" if self.holes_box.isChecked() else "external"

        assert self.dataset is not None
        record = self.dataset.register_image(path.name, image.shape[1], image.shape[0])
        existing = self.dataset.get(path.name)
        if existing:
            session.load_instances(existing)

        session.category_id = self.category_id
        self.session = session
        self.editor.set_session(session)
        self.worker.set_session(session)
        self.canvas.set_image(image)
        self._refresh_overlays()
        self._update_status()

    def _flush_current(self) -> None:
        """Save the outgoing image's annotations before leaving it."""
        if self.session is None or self.dataset is None or self.index < 0:
            return
        path = self.paths[self.index]
        record = self.dataset.register_image(
            path.name, self.session.width, self.session.height
        )
        entries = self.session.to_annotations(
            record.id, self.dataset.next_annotation_id, self.category_id
        )
        had = self.dataset.count(path.name)
        if entries or had:
            self.dataset.set_annotations(path.name, entries)
            self._save_dataset(quiet=True)

    def _save_dataset(self, quiet: bool = False) -> None:
        if self.dataset is None:
            return
        try:
            target = self.dataset.save()
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        if not quiet:
            self.statusBar().showMessage(
                f"Saved {self.dataset.summary()} to {target.name}", 4000
            )

    def save_all(self) -> None:
        self._flush_current()
        self._save_dataset(quiet=False)

    # ------------------------------------------------------------------ #
    # Modes
    # ------------------------------------------------------------------ #
    def eventFilter(self, obj, event) -> bool:  # noqa: N802
        """Own the Tab key for this window, whatever has focus."""
        if (
            event.type() == QEvent.KeyPress
            and event.key() in (Qt.Key_Tab, Qt.Key_Backtab)
            and self.isActiveWindow()
        ):
            self.toggle_mode()
            return True
        return super().eventFilter(obj, event)

    def toggle_mode(self) -> None:
        self.set_mode(self.app_mode.toggled())

    def set_mode(self, mode: AppMode) -> None:
        if mode is self.app_mode:
            return
        if mode is AppMode.EDIT and self.session is None:
            self.statusBar().showMessage(
                "Edit Mode needs an open image  (Ctrl+O)", 2500
            )
            return
        self.app_mode = mode
        if mode is AppMode.ANNOTATE:
            # Leaving Edit Mode drops the selection: it refers to instances
            # that the brush is about to start seeding as background again.
            self.editor.clear_selection()
            self.editor.clear_hover()
        self.canvas.interaction = (
            INTERACT_PICK if mode is AppMode.EDIT else INTERACT_PAINT
        )
        self._update_mode_label()
        self._refresh_overlays()
        self._update_status()
        self.statusBar().showMessage(
            "Edit Mode -- click an instance to select, Delete removes it"
            if mode is AppMode.EDIT
            else "Annotate Mode -- paint to select, Enter commits",
            2500,
        )

    @property
    def editing(self) -> bool:
        return self.app_mode is AppMode.EDIT

    # ------------------------------------------------------------------ #
    # Edit Mode interaction
    # ------------------------------------------------------------------ #
    @Slot(float, float)
    def _on_cursor_moved(self, x: float, y: float) -> None:
        if not self.editing or self.session is None:
            return
        if self.editor.set_hover_at(x, y):
            self._refresh_overlays()

    @Slot(float, float)
    def _on_picked(self, x: float, y: float) -> None:
        if not self.editing or self.session is None:
            return
        if self.editor.select_at(x, y):
            self._refresh_overlays()
            self._update_status()

    def deselect(self) -> None:
        if self.editing and self.editor.clear_selection():
            self._refresh_overlays()
            self._update_status()

    def delete_selected(self) -> None:
        """Delete key. Deliberately inert outside Edit Mode."""
        if not self.editing:
            self.statusBar().showMessage(
                "Delete only removes instances in Edit Mode  (Tab)", 2500
            )
            return
        if self.editor.selected is None:
            self.statusBar().showMessage("No instance selected", 2000)
            return
        self._apply_edit(self.editor.delete_selected, "Nothing to delete")

    def _apply_edit(self, operation, empty_message: str) -> None:
        """Run one edit, then redraw and write the dataset back out.

        Edits mutate the session from the GUI thread. That is safe because Edit
        Mode cannot start a stroke, so the worker is idle here -- and if a pass
        started before the mode switch is still running, the edit is refused
        rather than racing it.
        """
        if self.session is None:
            return
        if self._busy:
            self.statusBar().showMessage("Segmentation still running -- retry", 2000)
            return
        if operation() is None:
            self.statusBar().showMessage(empty_message, 2000)
            return
        self._refresh_overlays()
        self._update_status()
        self._flush_current()   # persists to annotations.json, atomically

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def _label_for(self, mode: SelectionMode) -> int:
        return NEGATIVE if mode == SelectionMode.SUBTRACT else POSITIVE

    @Slot(float, float, object)
    def _on_stroke_began(self, x: float, y: float, mode) -> None:
        if self.session is None:
            return
        if mode == SelectionMode.NEW and self.session.mask.any():
            # "New instance" means: throw away the marks for the instance in
            # progress and start again, not commit it silently.
            QMetaObject.invokeMethod(self.worker, "clear_constraints",
                                     Qt.QueuedConnection)
        self.worker.begin(x, y, self._label_for(mode))

    @Slot(float, float)
    def _on_stroke_moved(self, x: float, y: float) -> None:
        if self.session is not None:
            self.worker.move(x, y)

    @Slot()
    def _on_stroke_ended(self) -> None:
        if self.session is not None:
            self.worker.end()
        if self.mode_box.currentIndex() == 2:
            self.mode_box.setCurrentIndex(0)

    @Slot(object, str)
    def _on_mask_ready(self, mask, timings: str) -> None:
        self._last_timings = timings
        self.canvas.set_overlay(LAYER_CURRENT, mask, COLOR_CURRENT, ants=True)
        self._update_constraint_overlays()
        self._update_status()

    @Slot(object, str)
    def _on_stroke_finished(self, mask, timings: str) -> None:
        self._last_timings = timings
        self._refresh_overlays()
        self._update_status()

    @Slot(bool)
    def _on_busy(self, busy: bool) -> None:
        self._busy = busy

    @Slot(str)
    def _on_error(self, message: str) -> None:
        self.statusBar().showMessage(f"Engine error: {message}", 8000)

    # ------------------------------------------------------------------ #
    # Instances
    # ------------------------------------------------------------------ #
    def commit_instance(self) -> None:
        if self.session is None:
            return
        if not self.session.can_commit():
            self.statusBar().showMessage(
                "Nothing to commit -- paint an object first", 2500
            )
            return
        QMetaObject.invokeMethod(self.worker, "commit", Qt.QueuedConnection)

    def delete_last(self) -> None:
        if self.session is None or not self.session.instances:
            self.statusBar().showMessage("No committed instances", 2000)
            return
        QMetaObject.invokeMethod(self.worker, "delete_last", Qt.QueuedConnection)

    def clear_constraints(self) -> None:
        if self.session is not None:
            QMetaObject.invokeMethod(self.worker, "clear_constraints",
                                     Qt.QueuedConnection)

    def undo(self) -> None:
        """Undo the last action *of the current mode*.

        The two modes have separate histories because they edit different
        things: constraints in Annotate Mode, instances in Edit Mode. Mixing
        them into one stack would mean Ctrl+Z sometimes redrawing a brush mark
        and sometimes resurrecting an annotation, with no way to tell which.
        """
        if self.editing:
            self._apply_edit(self.editor.undo, "Nothing to undo in Edit Mode")
            return
        if self.session is not None:
            QMetaObject.invokeMethod(self.worker, "undo", Qt.QueuedConnection)

    def redo(self) -> None:
        if self.editing:
            self._apply_edit(self.editor.redo, "Nothing to redo in Edit Mode")
            return
        if self.session is not None:
            QMetaObject.invokeMethod(self.worker, "redo", Qt.QueuedConnection)

    # ------------------------------------------------------------------ #
    # Display
    # ------------------------------------------------------------------ #
    def _refresh_overlays(self) -> None:
        if self.session is None:
            return
        if self.editing:
            self._show_instance_overlays()
        else:
            self._show_annotation_overlays()

    def _show_instance_overlays(self) -> None:
        """Edit Mode: existing instances, each in its own colour."""
        assert self.session is not None
        for layer in ANNOTATE_LAYERS:
            self.canvas.remove_overlay(layer)
        rgba, paths = build_instance_layers(
            self.session.instances,
            self.session.height,
            self.session.width,
            hover=self.editor.hover,
            selected=self.editor.selected,
        )
        self.canvas.set_overlay_rgba(LAYER_INSTANCES, rgba)
        self.canvas.set_paths(PATHS_INSTANCES, paths)

    def _show_annotation_overlays(self) -> None:
        assert self.session is not None
        self.canvas.remove_overlay(LAYER_INSTANCES)
        self.canvas.remove_paths(PATHS_INSTANCES)
        if self.show_committed_action.isChecked():
            committed = self.session.committed_mask()
            self.canvas.set_overlay(
                LAYER_COMMITTED, committed if committed.any() else None,
                COLOR_COMMITTED,
            )
        else:
            self.canvas.remove_overlay(LAYER_COMMITTED)
        self.canvas.set_overlay(
            LAYER_CURRENT,
            self.session.mask if self.session.mask.any() else None,
            COLOR_CURRENT,
            ants=True,
        )
        self._update_constraint_overlays()

    def _update_constraint_overlays(self) -> None:
        if self.session is None:
            return
        if not self.show_constraints_action.isChecked():
            self.canvas.remove_overlay(LAYER_POS)
            self.canvas.remove_overlay(LAYER_NEG)
            return
        pos, neg = self.session.constraint_overlays()
        # Committed instances are seeded as background wholesale; drawing that
        # as a red wash would bury the image. Only show marks the user painted.
        committed = self.session.committed_mask()
        if committed.any():
            neg = neg & ~committed
        self.canvas.set_overlay(LAYER_POS, pos if pos.any() else None, COLOR_POS)
        self.canvas.set_overlay(LAYER_NEG, neg if neg.any() else None, COLOR_NEG)

    def _update_status(self) -> None:
        if not self.paths:
            return
        name = self.paths[self.index].name if self.index >= 0 else "-"
        self.image_label.setText(
            f"  Image {self.index + 1}/{len(self.paths)}: {name}   "
        )
        if self.session is not None:
            n = len(self.session.instances)
            pending = " (+1 in progress)" if self.session.mask.any() else ""
            self.instance_label.setText(f"Instances: {n}{pending}   ")
        self.class_label.setText(
            f"class {self.category_id}: {self.category_name}   "
        )
        self.selection_label.setText(self._selection_summary())
        if self._last_timings:
            self.timing_label.setText(f"  last pass: {self._last_timings}  ")

    def _selection_summary(self) -> str:
        """What the selected instance is, for the status bar."""
        if not self.editing:
            return ""
        inst = self.editor.selected
        if inst is None:
            return "nothing selected   "
        category = self.category_id if inst.category_id is None else inst.category_id
        name = (
            self.dataset.category_name_for(category)
            if self.dataset is not None
            else self.category_name
        )
        ident = "unsaved" if inst.annotation_id is None else f"#{inst.annotation_id}"
        return f"selected {ident}  class {category}: {name}   "

    # ------------------------------------------------------------------ #
    # Options
    # ------------------------------------------------------------------ #
    def _on_mode_changed(self, index: int) -> None:
        self.canvas.mode = [
            SelectionMode.ADD, SelectionMode.SUBTRACT, SelectionMode.NEW
        ][index]

    def _on_size_changed(self, value: int) -> None:
        self.brush.diameter = float(value)
        self.canvas.brush_diameter = float(value)
        self.size_label.setText(f"{value} px ")
        self.canvas.update()

    def _nudge_brush(self, factor: float) -> None:
        new = int(round(np.clip(self.brush.diameter * factor, 2, 400)))
        if new == int(self.brush.diameter):
            new += 1 if factor > 1 else -1
        self.size_slider.setValue(int(np.clip(new, 2, 400)))

    def _on_hardness_changed(self, value: int) -> None:
        self.brush.hardness = value / 100.0
        self.canvas.brush_hardness = self.brush.hardness
        self.canvas.update()

    def _on_spacing_changed(self, value: int) -> None:
        self.brush.spacing = value / 100.0

    def _on_polygon_opts(self, *_) -> None:
        if self.session is not None:
            self.session.polygon_epsilon = float(self.eps_spin.value())
            self.session.polygon_mode = (
                "ccomp" if self.holes_box.isChecked() else "external"
            )

    def _zoom_by(self, factor: float) -> None:
        self.canvas.set_zoom(self.canvas.zoom * factor)

    # ------------------------------------------------------------------ #
    def closeEvent(self, event) -> None:  # noqa: N802
        self._flush_current()
        self._save_dataset(quiet=True)
        self.worker.set_session(None)
        self.thread.quit()
        self.thread.wait(4000)
        super().closeEvent(event)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="annotate",
        description="COCO instance-segmentation annotation for electronic boards.",
    )
    parser.add_argument("folder", nargs="?", help="folder of images to annotate")
    parser.add_argument(
        "--category-id", type=int, default=DEFAULT_CATEGORY_ID,
        help=f"COCO category id (default {DEFAULT_CATEGORY_ID})",
    )
    parser.add_argument(
        "--category-name", default=DEFAULT_CATEGORY_NAME,
        help=f"COCO category name (default {DEFAULT_CATEGORY_NAME})",
    )
    parser.add_argument(
        "--work-dim", type=int, default=None,
        help="interactive solve resolution, longest side (default 384)",
    )
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Board Annotator")
    from ..ui.app import apply_dark_palette

    apply_dark_palette(app)

    window = AnnotatorWindow()
    window.category_id = args.category_id
    window.category_name = args.category_name
    if args.work_dim:
        window.params.work_max_dim = int(args.work_dim)
    window.show()
    if args.folder:
        window.open_folder(Path(args.folder))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
