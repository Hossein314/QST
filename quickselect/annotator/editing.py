"""Edit Mode: which instance the pointer is over, which one is selected, and
an undo stack for the edits made to them.

The annotation tool has two ways of interacting with an image and they want
different event routing, so the mode is an explicit piece of state rather than
a set of flags scattered through the window:

* ``AppMode.ANNOTATE`` -- the original workflow. The brush paints constraints,
  ``Enter`` commits, and nothing here is consulted at all.
* ``AppMode.EDIT`` -- the instances already on the image become objects you can
  point at. No stroke can start, so no edit-mode click can reach the segmenter.

Like :mod:`session`, this module knows nothing about Qt. Hit-testing and the
edit history are plain Python over :class:`~.session.Instance` objects, which
means the whole selection/delete/undo cycle can be driven from a script.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .session import AnnotationSession, Instance


class AppMode(Enum):
    """What the pointer does on the canvas."""

    ANNOTATE = "annotate"
    EDIT = "edit"

    @property
    def label(self) -> str:
        return "ANNOTATE" if self is AppMode.ANNOTATE else "EDIT"

    def toggled(self) -> "AppMode":
        return AppMode.EDIT if self is AppMode.ANNOTATE else AppMode.ANNOTATE


@dataclass
class DeleteRecord:
    """One removed instance and where it sat, which is all undo needs."""

    index: int
    instance: Instance


class InstanceEditor:
    """Selection state and edit history for the instances of one image.

    The hover and selection are held as the :class:`Instance` objects
    themselves rather than as list positions, so they stay meaningful while
    the list around them changes -- and an instance restored by undo is the
    same object that was deleted, with its id and polygons intact.
    """

    #: How many edits are kept. Deep enough for a working session, bounded so
    #: a long day of annotating cannot pin every deleted mask in memory.
    history_limit = 64

    def __init__(self, session: Optional[AnnotationSession] = None) -> None:
        self.session: Optional[AnnotationSession] = session
        self.hover: Optional[Instance] = None
        self.selected: Optional[Instance] = None
        self._undo: List[DeleteRecord] = []
        self._redo: List[DeleteRecord] = []

    def set_session(self, session: Optional[AnnotationSession]) -> None:
        """Point at a different image. History does not cross images."""
        self.session = session
        self.hover = None
        self.selected = None
        self._undo.clear()
        self._redo.clear()

    # ------------------------------------------------------------------ #
    # Hit testing
    # ------------------------------------------------------------------ #
    @property
    def instances(self) -> List[Instance]:
        return self.session.instances if self.session is not None else []

    def hit_test(self, x: float, y: float) -> Optional[Instance]:
        """The instance under an image-space point, or ``None``.

        Where instances overlap the smallest one wins. A small object sitting
        on a large one would otherwise be unreachable, and the large one can
        still be picked anywhere the small one is not.
        """
        best: Optional[Instance] = None
        best_area = 0
        for inst in self.instances:
            if not inst.contains(x, y):
                continue
            area = inst.area
            if best is None or area < best_area:
                best, best_area = inst, area
        return best

    def set_hover_at(self, x: float, y: float) -> bool:
        """Update the hovered instance. True if it changed."""
        return self._set_hover(self.hit_test(x, y))

    def clear_hover(self) -> bool:
        return self._set_hover(None)

    def _set_hover(self, inst: Optional[Instance]) -> bool:
        if inst is self.hover:
            return False
        self.hover = inst
        return True

    # ------------------------------------------------------------------ #
    # Selection
    # ------------------------------------------------------------------ #
    def select_at(self, x: float, y: float) -> bool:
        """Select whatever is under the point; clicking empty space clears."""
        return self.select(self.hit_test(x, y))

    def select(self, inst: Optional[Instance]) -> bool:
        if inst is self.selected:
            return False
        self.selected = inst
        return True

    def clear_selection(self) -> bool:
        return self.select(None)

    def index_of(self, instance: Optional[Instance]) -> Optional[int]:
        """Position of an instance in the session's list, by identity."""
        if instance is None:
            return None
        for i, inst in enumerate(self.instances):
            if inst is instance:
                return i
        return None

    def selected_index(self) -> Optional[int]:
        return self.index_of(self.selected)

    # ------------------------------------------------------------------ #
    # Edits
    # ------------------------------------------------------------------ #
    def delete_selected(self) -> Optional[DeleteRecord]:
        """Remove the selected instance. Returns the record, or ``None``."""
        if self.session is None:
            return None
        index = self.selected_index()
        if index is None:
            return None
        instance = self.session.remove_instance(index)
        if instance is None:
            return None
        record = DeleteRecord(index=index, instance=instance)
        self._push(record)
        self.selected = None
        if self.hover is instance:
            self.hover = None
        return record

    def _push(self, record: DeleteRecord) -> None:
        self._undo.append(record)
        del self._undo[: max(0, len(self._undo) - self.history_limit)]
        self._redo.clear()

    @property
    def can_undo(self) -> bool:
        return bool(self._undo) and self.session is not None

    @property
    def can_redo(self) -> bool:
        return bool(self._redo) and self.session is not None

    def undo(self) -> Optional[DeleteRecord]:
        """Put the most recently deleted instance back where it was."""
        if not self.can_undo:
            return None
        record = self._undo.pop()
        assert self.session is not None
        self.session.insert_instance(record.index, record.instance)
        self._redo.append(record)
        self.selected = record.instance
        return record

    def redo(self) -> Optional[DeleteRecord]:
        if not self.can_redo:
            return None
        record = self._redo.pop()
        assert self.session is not None
        index = self.index_of(record.instance)
        if index is None:
            # The instance is no longer in the list; the redo is meaningless.
            return None
        self.session.remove_instance(index)
        record.index = index
        self._undo.append(record)
        if self.selected is record.instance:
            self.selected = None
        if self.hover is record.instance:
            self.hover = None
        return record
