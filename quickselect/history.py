"""Undo / redo for selection state.

Snapshots are the engine's packed state dicts (bit-packed masks), so a step for
a 1080p image costs roughly 300 kB rather than 2 MB.
"""

from __future__ import annotations

from typing import Callable, List, Optional


class History:
    def __init__(self, limit: int = 60) -> None:
        self.limit = max(1, limit)
        self._undo: List[dict] = []
        self._redo: List[dict] = []

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, state: dict) -> None:
        """Record a state to return to.  Invalidates the redo branch."""
        self._undo.append(state)
        if len(self._undo) > self.limit:
            self._undo.pop(0)
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self, current: dict) -> Optional[dict]:
        if not self._undo:
            return None
        self._redo.append(current)
        return self._undo.pop()

    def redo(self, current: dict) -> Optional[dict]:
        if not self._redo:
            return None
        self._undo.append(current)
        return self._redo.pop()

    def __len__(self) -> int:
        return len(self._undo)
