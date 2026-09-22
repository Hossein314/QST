"""Segmentation worker thread for the annotation tool.

Same shape as the selection tool's worker: a coalescing command queue so the
canvas never blocks. If the mouse produced ten move events while a pass was
running, all ten are fed to the session in order -- stamp spacing stays
correct -- and one repaint is emitted.

Everything that touches the session runs here, including commit and undo, so
there is exactly one thread mutating the constraint matrix. That is what makes
it safe for the GUI thread to read ``session.mask`` when a signal says it is
ready.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PySide6.QtCore import QMetaObject, QMutex, QMutexLocker, QObject, Qt, Signal, Slot

from ..constraints import NEGATIVE, POSITIVE
from .session import AnnotationSession

BEGIN = "begin"
MOVE = "move"
END = "end"


class SessionWorker(QObject):
    """Drives an :class:`AnnotationSession` off the GUI thread."""

    maskReady = Signal(object, str)      # mask, timing summary
    strokeFinished = Signal(object, str)
    errorRaised = Signal(str)
    busyChanged = Signal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.session: Optional[AnnotationSession] = None
        self._mutex = QMutex()
        self._queue: List[tuple] = []
        self._scheduled = False

    # -- called from the GUI thread ---------------------------------------- #
    def set_session(self, session: Optional[AnnotationSession]) -> None:
        with QMutexLocker(self._mutex):
            self._queue.clear()
            self._scheduled = False
            self.session = session

    def post(self, *command) -> None:
        with QMutexLocker(self._mutex):
            self._queue.append(command)
            already = self._scheduled
            self._scheduled = True
        if not already:
            QMetaObject.invokeMethod(self, "_drain", Qt.QueuedConnection)

    def begin(self, x: float, y: float, label: int) -> None:
        self.post(BEGIN, x, y, label)

    def move(self, x: float, y: float) -> None:
        self.post(MOVE, x, y)

    def end(self) -> None:
        self.post(END)

    # -- runs on the worker thread ----------------------------------------- #
    @Slot()
    def _drain(self) -> None:
        self.busyChanged.emit(True)
        try:
            while True:
                with QMutexLocker(self._mutex):
                    if not self._queue or self.session is None:
                        self._queue.clear()
                        self._scheduled = False
                        break
                    batch = self._queue[:]
                    self._queue.clear()
                    session = self.session
                self._run_batch(session, batch)
        except Exception as exc:  # never let one bad pass kill the thread
            with QMutexLocker(self._mutex):
                self._queue.clear()
                self._scheduled = False
            self.errorRaised.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.busyChanged.emit(False)

    def _run_batch(self, session: AnnotationSession, batch: List[tuple]) -> None:
        touched = False
        finished = False
        for cmd in batch:
            op = cmd[0]
            if op == BEGIN:
                touched |= session.begin_stroke(cmd[1], cmd[2], cmd[3])
            elif op == MOVE:
                touched |= session.continue_stroke(cmd[1], cmd[2])
            elif op == END:
                if touched:
                    self.maskReady.emit(session.mask, session.timings())
                    touched = False
                session.end_stroke()
                finished = True
        if touched:
            self.maskReady.emit(session.mask, session.timings())
        if finished:
            self.strokeFinished.emit(session.mask, session.timings())

    # -- operations invoked between gestures -------------------------------- #
    def _apply(self, fn) -> None:
        if self.session is None:
            return
        try:
            self.busyChanged.emit(True)
            fn(self.session)
            self.strokeFinished.emit(self.session.mask, self.session.timings())
        except Exception as exc:
            self.errorRaised.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.busyChanged.emit(False)

    @Slot()
    def undo(self) -> None:
        self._apply(lambda s: s.undo())

    @Slot()
    def redo(self) -> None:
        self._apply(lambda s: s.redo())

    @Slot()
    def clear_constraints(self) -> None:
        self._apply(lambda s: s.clear_constraints())

    @Slot()
    def commit(self) -> None:
        self._apply(lambda s: s.commit_instance())

    @Slot()
    def delete_last(self) -> None:
        self._apply(lambda s: s.delete_last_instance())

    @Slot()
    def recompute(self) -> None:
        self._apply(lambda s: s.recompute(force_global=True))
