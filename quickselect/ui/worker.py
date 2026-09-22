"""Engine worker thread.

The engine takes tens of milliseconds per brush step.  Running it on the GUI
thread would drop frames and make the brush cursor lag behind the mouse, which
is exactly the feel we are trying to avoid.  So the engine lives on its own
thread and the canvas talks to it through a small command queue.

The queue is *coalescing*: if the mouse produced ten move events while the
engine was busy with the previous step, the worker drains all ten, feeds them
to the engine in order (so stamp spacing stays correct) and emits **one**
repaint signal.  That decouples selection update rate from frame rate -- the
canvas keeps running at 60 fps showing the brush cursor and the last known
selection, while the selection itself refreshes as fast as the engine manages.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QMutex, QMutexLocker, QObject, Qt, QMetaObject, Signal, Slot

from ..engine import QuickSelectEngine, SelectionMode

# Command opcodes kept as plain tuples: cheap to queue, trivial to coalesce.
BEGIN = "begin"
MOVE = "move"
END = "end"


class EngineWorker(QObject):
    """Owns the engine and executes brush gestures off the GUI thread."""

    maskReady = Signal(object)        # interactive-scale bool mask
    finalReady = Signal(object)       # full-resolution float32 alpha
    busyChanged = Signal(bool)
    errorRaised = Signal(str)

    def __init__(self, engine: QuickSelectEngine) -> None:
        super().__init__()
        self.engine = engine
        self._mutex = QMutex()
        self._queue: List[tuple] = []
        self._scheduled = False

    # -- called from the GUI thread ---------------------------------------- #
    def post(self, *command) -> None:
        with QMutexLocker(self._mutex):
            self._queue.append(command)
            already = self._scheduled
            self._scheduled = True
        if not already:
            QMetaObject.invokeMethod(self, "_drain", Qt.QueuedConnection)

    def begin(self, x: float, y: float, mode: SelectionMode) -> None:
        self.post(BEGIN, x, y, mode)

    def move(self, x: float, y: float) -> None:
        self.post(MOVE, x, y)

    def end(self) -> None:
        self.post(END)

    def pending(self) -> bool:
        with QMutexLocker(self._mutex):
            return bool(self._queue)

    # -- runs on the worker thread ----------------------------------------- #
    @Slot()
    def _drain(self) -> None:
        self.busyChanged.emit(True)
        try:
            while True:
                with QMutexLocker(self._mutex):
                    if not self._queue:
                        self._scheduled = False
                        break
                    batch = self._queue[:]
                    self._queue.clear()
                self._run_batch(batch)
        except Exception as exc:  # never let a worker exception kill the thread
            with QMutexLocker(self._mutex):
                self._queue.clear()
                self._scheduled = False
            self.errorRaised.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.busyChanged.emit(False)

    def _run_batch(self, batch: List[tuple]) -> None:
        eng = self.engine
        touched = False
        finished = False
        for cmd in batch:
            op = cmd[0]
            if op == BEGIN:
                eng.begin_stroke(cmd[1], cmd[2], cmd[3])
                touched = True
            elif op == MOVE:
                if eng.continue_stroke(cmd[1], cmd[2]) is not None:
                    touched = True
            elif op == END:
                # Emit the coarse mask first so the overlay snaps immediately,
                # then pay for the full-resolution refinement.
                if touched:
                    self.maskReady.emit(eng.selection.copy())
                    touched = False
                eng.end_stroke(refine=True)
                finished = True
        if touched:
            self.maskReady.emit(eng.selection.copy())
        if finished:
            self.maskReady.emit(eng.selection.copy())
            self.finalReady.emit(eng.full_alpha)

    # -- synchronous operations, invoked between gestures ------------------- #
    @Slot()
    def refine(self) -> None:
        try:
            self.busyChanged.emit(True)
            self.engine.refine_now()
            self.maskReady.emit(self.engine.selection.copy())
            self.finalReady.emit(self.engine.full_alpha)
        except Exception as exc:
            self.errorRaised.emit(f"{type(exc).__name__}: {exc}")
        finally:
            self.busyChanged.emit(False)

    @Slot(object)
    def restore(self, state: dict) -> None:
        self.engine.restore(state)
        self.maskReady.emit(self.engine.selection.copy())
        self.finalReady.emit(self.engine.full_alpha)

    @Slot()
    def clear(self) -> None:
        self.engine.clear()
        self.maskReady.emit(self.engine.selection.copy())
        self.finalReady.emit(self.engine.full_alpha)

    @Slot()
    def invert(self) -> None:
        self.engine.invert()
        self.maskReady.emit(self.engine.selection.copy())
        self.finalReady.emit(self.engine.full_alpha)
