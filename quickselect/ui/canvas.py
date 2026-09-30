"""The painting canvas: zoom, pan, brush cursor, layered selection preview.

Rendering strategy
------------------
The canvas never waits on the segmenter. It holds the image as a ``QPixmap``,
a small stack of named RGBA overlay ``QImage``s, and the brush cursor, and
composites them every frame. Scaling is left to ``QPainter``, so a 60 fps
repaint costs almost nothing however far behind the engine is.

Overlays are named and drawn in insertion order, which is what lets the
annotation tool show committed instances, the instance being edited, and the
raw constraint marks at the same time in different colours. Marching ants are
drawn from contours extracted once per update and animated by advancing the
pen's dash offset, so no new geometry is built per frame. Named *path* layers
sit on top for outlines that are static rather than animated, which is how the
annotator draws the border of every existing instance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QTransform,
)
from PySide6.QtWidgets import QWidget

from ..engine import SelectionMode

OVERLAY_COLOR = QColor(64, 132, 255, 110)
ANTS_LIGHT = QColor(255, 255, 255)
ANTS_DARK = QColor(20, 20, 20)

#: Display styles for the primary selection overlay.
STYLE_ANTS = "ants"
STYLE_OVERLAY = "overlay"
STYLE_BOTH = "both"
STYLE_CUTOUT = "cutout"

SELECTION = "selection"

#: What the left mouse button does. ``INTERACT_PAINT`` is the brush; under
#: ``INTERACT_PICK`` the canvas emits :attr:`Canvas.picked` instead and no
#: stroke signal is ever produced, which is what keeps the annotator's Edit
#: Mode clicks away from the segmenter.
INTERACT_PAINT = "paint"
INTERACT_PICK = "pick"


def contour_path(
    binary: np.ndarray, sx: float = 1.0, sy: float = 1.0
) -> Optional[QPainterPath]:
    """Outline of a binary mask as a painter path, in image coordinates.

    ``sx``/``sy`` scale a mask held at a coarser resolution than the image up
    to image space.
    """
    contours, _ = cv2.findContours(
        binary.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    path = QPainterPath()
    drawn = False
    for contour in contours:
        simplified = cv2.approxPolyDP(contour, 0.8, True)
        if len(simplified) < 2:
            continue
        pts = simplified.reshape(-1, 2).astype(np.float32) * (sx, sy)
        path.moveTo(float(pts[0][0]), float(pts[0][1]))
        for x, y in pts[1:]:
            path.lineTo(float(x), float(y))
        path.closeSubpath()
        drawn = True
    return path if drawn else None


def _readable_on(color: QColor) -> QColor:
    """Black or white text, whichever the background can carry."""
    luma = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
    return QColor(20, 20, 20) if luma > 140 else QColor(255, 255, 255)


def numpy_to_qimage(rgb: np.ndarray) -> QImage:
    """RGB/RGBA/grayscale uint8 -> QImage, copied so it owns its buffer."""
    arr = np.ascontiguousarray(rgb)
    h, w = arr.shape[:2]
    if arr.ndim == 3 and arr.shape[2] == 3:
        return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()
    if arr.ndim == 3 and arr.shape[2] == 4:
        return QImage(arr.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
    return QImage(arr.data, w, h, w, QImage.Format_Grayscale8).copy()


@dataclass
class _Overlay:
    image: QImage
    color: QColor
    ants: bool = False
    path: Optional[QPainterPath] = None
    visible: bool = True


class Canvas(QWidget):
    """Image view with brush painting and layered mask preview."""

    strokeBegan = Signal(float, float, object)   # x, y, SelectionMode
    strokeMoved = Signal(float, float)
    strokeEnded = Signal()
    zoomChanged = Signal(float)
    cursorMoved = Signal(float, float)
    picked = Signal(float, float)                # left click under INTERACT_PICK

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setCursor(Qt.BlankCursor)
        self.setAutoFillBackground(False)

        self._pixmap: Optional[QPixmap] = None
        self._image_size: Tuple[int, int] = (0, 0)  # (w, h) full resolution
        self._overlays: Dict[str, _Overlay] = {}
        self._order: List[str] = []
        self._paths: Dict[str, List[Tuple[QPainterPath, QColor, float]]] = {}
        self._path_order: List[str] = []
        self._labels: Dict[str, List[Tuple[QPointF, str, QColor]]] = {}
        self._label_order: List[str] = []

        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)  # image-space point at widget origin
        # Until the user zooms or pans, the view keeps re-fitting as the window
        # resizes. Without this the initial fit is computed before Qt has laid
        # the widget out, and the image opens at a nonsense zoom.
        self._user_adjusted = False

        self.brush_diameter = 40.0
        self.brush_hardness = 0.85
        self.style = STYLE_BOTH
        self.mode = SelectionMode.NEW
        self._interaction = INTERACT_PAINT

        self._painting = False
        self._panning = False
        self._space_held = False
        self._last_pan: Optional[QPointF] = None
        self._cursor_pos = QPointF(-1e6, -1e6)

        self._ants_offset = 0.0
        self._ants_timer = QTimer(self)
        self._ants_timer.setInterval(80)
        self._ants_timer.timeout.connect(self._advance_ants)
        self._ants_timer.start()

    # ------------------------------------------------------------------ #
    # Content
    # ------------------------------------------------------------------ #
    def set_image(self, rgb: np.ndarray) -> None:
        self._pixmap = QPixmap.fromImage(numpy_to_qimage(rgb))
        self._image_size = (rgb.shape[1], rgb.shape[0])
        self.clear_overlays()
        self._user_adjusted = False
        self.fit_to_window()

    def clear_overlays(self) -> None:
        self._overlays.clear()
        self._order.clear()
        self._paths.clear()
        self._path_order.clear()
        self._labels.clear()
        self._label_order.clear()
        self.update()

    def set_overlay(
        self,
        name: str,
        mask: Optional[np.ndarray],
        color: QColor,
        ants: bool = False,
        opacity: Optional[int] = None,
    ) -> None:
        """Add or replace a named overlay.

        ``mask`` may be boolean or a 0..1 float alpha, at any resolution -- it
        is scaled to the image at draw time, so a coarse mid-drag preview costs
        less to build than a full-resolution one.
        """
        if mask is None or not getattr(mask, "size", 0):
            self.remove_overlay(name)
            return

        alpha = (
            mask.astype(np.float32)
            if mask.dtype == bool
            else np.clip(mask.astype(np.float32), 0.0, 1.0)
        )
        h, w = alpha.shape[:2]
        a_max = color.alpha() if opacity is None else int(opacity)
        rgba = np.empty((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = color.red()
        rgba[..., 1] = color.green()
        rgba[..., 2] = color.blue()
        rgba[..., 3] = (alpha * a_max).astype(np.uint8)

        path = None
        if ants:
            sx = self._image_size[0] / float(w) if w else 1.0
            sy = self._image_size[1] / float(h) if h else 1.0
            path = contour_path(alpha >= 0.5, sx, sy)

        if name not in self._overlays:
            self._order.append(name)
        self._overlays[name] = _Overlay(
            image=numpy_to_qimage(rgba), color=color, ants=ants, path=path
        )
        self.update()

    def set_overlay_rgba(self, name: str, rgba: Optional[np.ndarray]) -> None:
        """Add or replace a named overlay from a pre-composited RGBA array.

        The mask-plus-colour form of :meth:`set_overlay` cannot express a layer
        that holds several colours at once, which is what the annotator needs
        to draw every existing instance in its own colour without one canvas
        overlay per instance.
        """
        if rgba is None or not getattr(rgba, "size", 0):
            self.remove_overlay(name)
            return
        if name not in self._overlays:
            self._order.append(name)
        self._overlays[name] = _Overlay(
            image=numpy_to_qimage(rgba), color=QColor(0, 0, 0, 0)
        )
        self.update()

    def set_paths(
        self,
        name: str,
        entries: Optional[List[Tuple[QPainterPath, QColor, float]]],
    ) -> None:
        """Add or replace a named set of outlines, in image coordinates.

        Each entry is ``(path, colour, width)`` and is stroked with a cosmetic
        pen, so the line keeps its width at every zoom level. Unlike marching
        ants these are static -- they carry state, not attention.
        """
        if not entries:
            self.remove_paths(name)
            return
        if name not in self._paths:
            self._path_order.append(name)
        self._paths[name] = list(entries)
        self.update()

    def remove_paths(self, name: str) -> None:
        if name in self._paths:
            del self._paths[name]
            self._path_order.remove(name)
            self.update()

    def set_labels(
        self,
        name: str,
        entries: Optional[List[Tuple[QPointF, str, QColor]]],
    ) -> None:
        """Add or replace a named set of small text chips.

        Each entry is ``(point, text, colour)`` with the point in image
        coordinates. The chip itself is drawn in screen space at a fixed size,
        so a label stays readable when zoomed out and does not swell into a
        billboard when zoomed in.
        """
        if not entries:
            self.remove_labels(name)
            return
        if name not in self._labels:
            self._label_order.append(name)
        self._labels[name] = list(entries)
        self.update()

    def remove_labels(self, name: str) -> None:
        if name in self._labels:
            del self._labels[name]
            self._label_order.remove(name)
            self.update()

    def remove_overlay(self, name: str) -> None:
        if name in self._overlays:
            del self._overlays[name]
            self._order.remove(name)
            self.update()

    def set_overlay_visible(self, name: str, visible: bool) -> None:
        ov = self._overlays.get(name)
        if ov is not None and ov.visible != visible:
            ov.visible = visible
            self.update()

    def has_overlay(self, name: str) -> bool:
        return name in self._overlays

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    @property
    def interaction(self) -> str:
        return self._interaction

    @interaction.setter
    def interaction(self, value: str) -> None:
        if value == self._interaction:
            return
        self._interaction = value
        self._painting = False
        # The brush ring is a lie when the button does not paint, so the
        # pointer goes back to being a pointer.
        self.setCursor(
            Qt.BlankCursor if value == INTERACT_PAINT else Qt.ArrowCursor
        )
        self.update()

    @property
    def painting_enabled(self) -> bool:
        return self._interaction == INTERACT_PAINT

    # -- compatibility with the single-selection app -------------------- #
    def set_selection(self, mask: Optional[np.ndarray]) -> None:
        self.set_overlay(SELECTION, mask, OVERLAY_COLOR, ants=True)

    # ------------------------------------------------------------------ #
    # View transform
    # ------------------------------------------------------------------ #
    def _transform(self) -> QTransform:
        t = QTransform()
        t.translate(-self._offset.x() * self._zoom, -self._offset.y() * self._zoom)
        t.scale(self._zoom, self._zoom)
        return t

    def widget_to_image(self, p: QPointF) -> QPointF:
        return QPointF(
            p.x() / self._zoom + self._offset.x(),
            p.y() / self._zoom + self._offset.y(),
        )

    def image_to_widget(self, p: QPointF) -> QPointF:
        return QPointF(
            (p.x() - self._offset.x()) * self._zoom,
            (p.y() - self._offset.y()) * self._zoom,
        )

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_zoom(self, zoom: float, anchor: Optional[QPointF] = None) -> None:
        zoom = float(np.clip(zoom, 0.02, 32.0))
        if abs(zoom - self._zoom) < 1e-9:
            return
        if anchor is None:
            anchor = QPointF(self.width() / 2.0, self.height() / 2.0)
        before = self.widget_to_image(anchor)
        self._zoom = zoom
        after = self.widget_to_image(anchor)
        self._offset += before - after
        self._user_adjusted = True
        self._clamp_offset()
        self.zoomChanged.emit(self._zoom)
        self.update()

    def fit_to_window(self) -> None:
        w, h = self._image_size
        if not w or not h or self.width() < 2 or self.height() < 2:
            return
        self._zoom = min(self.width() / w, self.height() / h) * 0.98
        self._offset = QPointF(
            (w - self.width() / self._zoom) / 2.0,
            (h - self.height() / self._zoom) / 2.0,
        )
        self.zoomChanged.emit(self._zoom)
        self.update()

    def fit_and_hold(self) -> None:
        """Fit, and stop auto-fitting on subsequent resizes (the Ctrl+0 path)."""
        self.fit_to_window()
        self._user_adjusted = True

    def zoom_actual(self) -> None:
        self.set_zoom(1.0)

    def _clamp_offset(self) -> None:
        w, h = self._image_size
        if not w:
            return
        vw = self.width() / self._zoom
        vh = self.height() / self._zoom
        self._offset.setX(float(np.clip(self._offset.x(), -vw * 0.5, w - vw * 0.5)))
        self._offset.setY(float(np.clip(self._offset.y(), -vh * 0.5, h - vh * 0.5)))

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def _advance_ants(self) -> None:
        if any(o.ants and o.visible for o in self._overlays.values()):
            self._ants_offset = (self._ants_offset + 1.0) % 8.0
            self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(38, 40, 44))
        if self._pixmap is None:
            painter.end()
            return

        painter.setRenderHint(QPainter.SmoothPixmapTransform, self._zoom < 1.0)
        transform = self._transform()
        painter.setTransform(transform)

        w, h = self._image_size
        target = QRectF(0, 0, w, h)
        painter.drawPixmap(target, self._pixmap, QRectF(self._pixmap.rect()))

        cutout = self.style == STYLE_CUTOUT
        for name in self._order:
            ov = self._overlays[name]
            if not ov.visible:
                continue
            if name == SELECTION:
                if cutout:
                    painter.save()
                    painter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
                    painter.drawImage(target, ov.image)
                    painter.restore()
                    continue
                if self.style == STYLE_ANTS:
                    continue
            painter.drawImage(target, ov.image)

        painter.setTransform(QTransform())
        if not cutout:
            for name in self._order:
                ov = self._overlays[name]
                if not ov.visible or not ov.ants or ov.path is None:
                    continue
                if name == SELECTION and self.style == STYLE_OVERLAY:
                    continue
                self._draw_ants(painter, transform.map(ov.path))

        self._draw_paths(painter, transform)
        self._draw_labels(painter, transform)
        self._draw_brush_cursor(painter)
        painter.end()

    def _draw_paths(self, painter: QPainter, transform: QTransform) -> None:
        for name in self._path_order:
            for path, color, width in self._paths[name]:
                pen = QPen(color, width)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawPath(transform.map(path))

    def _draw_labels(self, painter: QPainter, transform: QTransform) -> None:
        if not self._label_order:
            return
        font = QFont(self.font())
        font.setPointSizeF(max(7.5, font.pointSizeF()))
        font.setBold(True)
        painter.setFont(font)
        metrics = QFontMetrics(font)
        for name in self._label_order:
            for point, text, color in self._labels[name]:
                at = transform.map(point)
                if not self.rect().adjusted(-40, -20, 40, 20).contains(at.toPoint()):
                    continue  # off screen; skip the work
                w = metrics.horizontalAdvance(text) + 10
                h = metrics.height() + 2
                chip = QRectF(at.x() - w / 2.0, at.y() - h / 2.0, w, h)
                back = QColor(color)
                back.setAlpha(225)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(back))
                painter.drawRoundedRect(chip, 3.0, 3.0)
                painter.setBrush(Qt.NoBrush)
                painter.setPen(QPen(_readable_on(back), 1.0))
                painter.drawText(chip, Qt.AlignCenter, text)

    def _draw_ants(self, painter: QPainter, path: QPainterPath) -> None:
        # A dark under-stroke keeps the ants visible on light images.
        pen = QPen(ANTS_DARK, 1.2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPath(path)
        pen = QPen(ANTS_LIGHT, 1.2)
        pen.setCosmetic(True)
        pen.setDashPattern([4, 4])
        pen.setDashOffset(self._ants_offset)
        painter.setPen(pen)
        painter.drawPath(path)

    def _draw_brush_cursor(self, painter: QPainter) -> None:
        if not self.painting_enabled:
            return  # the window manager's arrow is the cursor in pick mode
        if not self.underMouse() and not self._painting:
            return
        r = max(1.0, self.brush_diameter * 0.5 * self._zoom)
        c = self._cursor_pos
        painter.setBrush(Qt.NoBrush)
        if self._space_held or self._panning:
            painter.setPen(QPen(QColor(255, 255, 255, 200), 1.2))
            painter.drawLine(c.x() - 7, c.y(), c.x() + 7, c.y())
            painter.drawLine(c.x(), c.y() - 7, c.x(), c.y() + 7)
            return
        painter.setPen(QPen(QColor(0, 0, 0, 170), 2.0))
        painter.drawEllipse(c, r, r)
        painter.setPen(QPen(QColor(255, 255, 255, 230), 1.0))
        painter.drawEllipse(c, r, r)
        # Inner ring shows where the falloff starts (hardness).
        if self.brush_hardness < 0.98 and r > 4:
            inner = r * self.brush_hardness
            painter.setPen(QPen(QColor(255, 255, 255, 90), 1.0, Qt.DotLine))
            painter.drawEllipse(c, inner, inner)
        mode = self._effective_mode()
        if mode == SelectionMode.SUBTRACT:
            painter.setPen(QPen(QColor(255, 120, 120, 240), 1.6))
            painter.drawLine(c.x() - 4, c.y(), c.x() + 4, c.y())
        elif mode == SelectionMode.ADD:
            painter.setPen(QPen(QColor(150, 240, 150, 240), 1.6))
            painter.drawLine(c.x() - 4, c.y(), c.x() + 4, c.y())
            painter.drawLine(c.x(), c.y() - 4, c.x(), c.y() + 4)

    # ------------------------------------------------------------------ #
    # Input
    # ------------------------------------------------------------------ #
    def _effective_mode(self, mods=None) -> SelectionMode:
        # Prefer the modifiers carried by the event that started the stroke;
        # fall back to live keyboard state for cursor drawing between events.
        if mods is None:
            mods = self._modifiers()
        if mods & Qt.AltModifier:
            return SelectionMode.SUBTRACT
        if mods & Qt.ShiftModifier:
            return SelectionMode.ADD
        return self.mode

    @staticmethod
    def _modifiers():
        from PySide6.QtWidgets import QApplication

        return QApplication.keyboardModifiers()

    def resizeEvent(self, event) -> None:  # noqa: N802
        if not self._user_adjusted:
            self.fit_to_window()
        else:
            self._clamp_offset()
        super().resizeEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        delta = event.angleDelta().y()
        if delta:
            self.set_zoom(self._zoom * (1.0015 ** delta), QPointF(event.position()))

    def mousePressEvent(self, event) -> None:  # noqa: N802
        pos = QPointF(event.position())
        self._cursor_pos = pos
        if event.button() == Qt.MiddleButton or self._space_held:
            self._panning = True
            self._last_pan = pos
            self.update()
            return
        if event.button() != Qt.LeftButton or self._pixmap is None:
            return
        img = self.widget_to_image(pos)
        if not self.painting_enabled:
            self.picked.emit(img.x(), img.y())
            self.update()
            return
        self._painting = True
        self.strokeBegan.emit(
            img.x(), img.y(), self._effective_mode(event.modifiers())
        )
        self.update()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = QPointF(event.position())
        self._cursor_pos = pos
        img = self.widget_to_image(pos)
        self.cursorMoved.emit(img.x(), img.y())

        if self._panning and self._last_pan is not None:
            d = pos - self._last_pan
            self._offset -= QPointF(d.x() / self._zoom, d.y() / self._zoom)
            self._user_adjusted = True
            self._last_pan = pos
            self._clamp_offset()
            self.update()
            return
        if self._painting:
            self.strokeMoved.emit(img.x(), img.y())
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._panning and event.button() in (Qt.MiddleButton, Qt.LeftButton):
            self._panning = False
            self._last_pan = None
            self.update()
            return
        if self._painting and event.button() == Qt.LeftButton:
            self._painting = False
            self.strokeEnded.emit()
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._cursor_pos = QPointF(-1e6, -1e6)
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            self.update()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            self._panning = False
            self.update()
            return
        super().keyReleaseEvent(event)
