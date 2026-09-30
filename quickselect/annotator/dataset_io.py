"""COCO instance-segmentation dataset: scan a folder, load and save the JSON.

One ``annotations.json`` per image folder, in standard COCO layout::

    {
      "images":      [{"id", "file_name", "width", "height"}, ...],
      "annotations": [{"id", "image_id", "category_id", "segmentation",
                       "area", "bbox", "iscrowd"}, ...],
      "categories":  [{"id": 4, "name": "electronic_board", ...}]
    }

Two properties matter for an annotation tool that saves constantly:

*Saves are atomic.* The file is written to a temporary sibling and then
``os.replace``d, so a crash or a pulled USB stick during a save leaves the
previous good file intact rather than a half-written one. Losing an afternoon
of annotation to a truncated JSON is a real failure mode, not a hypothetical.

*Image identity is the file name.* IDs are assigned on first sight and then
kept, so reopening a folder restores exactly the annotations you left, and
adding new images to the folder does not renumber the old ones.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..io_utils import IMAGE_SUFFIXES

DEFAULT_CATEGORY_ID = 4
DEFAULT_CATEGORY_NAME = "electronic_board"
DEFAULT_FILENAME = "annotations.json"


def scan_folder(folder: Path, recursive: bool = False) -> List[Path]:
    """Image files in ``folder``, sorted by name.

    Sorting is by the path as the user sees it, so left/right navigation
    matches what a file browser shows.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []
    globber = folder.rglob("*") if recursive else folder.glob("*")
    files = [
        p for p in globber
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(files, key=lambda p: str(p).lower())


@dataclass
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int

    def to_coco(self) -> Dict:
        return {
            "id": self.id,
            "file_name": self.file_name,
            "width": self.width,
            "height": self.height,
        }


class CocoDataset:
    """In-memory COCO annotations for one folder."""

    def __init__(
        self,
        path: Path,
        category_id: int = DEFAULT_CATEGORY_ID,
        category_name: str = DEFAULT_CATEGORY_NAME,
    ) -> None:
        self.path = Path(path)
        self.category_id = category_id
        self.category_name = category_name
        self.images: Dict[str, ImageRecord] = {}       # file_name -> record
        self.annotations: Dict[int, List[Dict]] = {}   # image_id -> entries
        self._next_image_id = 1
        self._next_ann_id = 1
        self._extra_categories: List[Dict] = []
        self.dirty = False

    # ------------------------------------------------------------------ #
    @classmethod
    def load(
        cls,
        path: Path,
        category_id: int = DEFAULT_CATEGORY_ID,
        category_name: str = DEFAULT_CATEGORY_NAME,
    ) -> "CocoDataset":
        """Load, or return an empty dataset if the file is absent."""
        ds = cls(path, category_id, category_name)
        p = Path(path)
        if not p.is_file():
            return ds
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        for entry in data.get("images", []):
            name = entry.get("file_name")
            if not name:
                continue
            record = ImageRecord(
                id=int(entry.get("id", ds._next_image_id)),
                file_name=str(name),
                width=int(entry.get("width", 0)),
                height=int(entry.get("height", 0)),
            )
            ds.images[record.file_name] = record
            ds._next_image_id = max(ds._next_image_id, record.id + 1)

        for entry in data.get("annotations", []):
            image_id = int(entry.get("image_id", -1))
            if image_id < 0:
                continue
            ds.annotations.setdefault(image_id, []).append(entry)
            ds._next_ann_id = max(ds._next_ann_id, int(entry.get("id", 0)) + 1)

        # Preserve any categories the file already had beyond ours, so opening
        # and re-saving a multi-class dataset does not quietly drop classes.
        for cat in data.get("categories", []):
            if int(cat.get("id", -1)) != category_id:
                ds._extra_categories.append(cat)
        return ds

    def save(self, path: Optional[Path] = None) -> Path:
        """Write the JSON atomically. Returns the path written."""
        target = Path(path) if path is not None else self.path
        target.parent.mkdir(parents=True, exist_ok=True)

        images = sorted(self.images.values(), key=lambda r: r.id)
        annotations: List[Dict] = []
        for record in images:
            annotations.extend(self.annotations.get(record.id, []))

        payload = {
            "images": [r.to_coco() for r in images],
            "annotations": annotations,
            "categories": self._categories(),
        }

        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent), prefix=target.name, suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=1)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        self.dirty = False
        return target

    def _categories(self) -> List[Dict]:
        cats = [
            {
                "id": self.category_id,
                "name": self.category_name,
                "supercategory": "object",
            }
        ]
        cats.extend(self._extra_categories)
        return sorted(cats, key=lambda c: int(c.get("id", 0)))

    def ensure_category(self, category_id: int, name: Optional[str] = None) -> None:
        """Declare a class in ``categories`` if the file does not have it yet.

        Assigning an instance a class the file has never seen would otherwise
        produce annotations pointing at a category that does not exist, which
        most COCO loaders treat as a hard error.
        """
        category_id = int(category_id)
        if category_id == int(self.category_id):
            return
        for cat in self._extra_categories:
            if int(cat.get("id", -1)) == category_id:
                return
        self._extra_categories.append({
            "id": category_id,
            "name": name or f"class_{category_id}",
            "supercategory": "object",
        })
        self.dirty = True

    def category_name_for(self, category_id: int) -> str:
        """Human name for a class id, including classes we only pass through."""
        if int(category_id) == int(self.category_id):
            return self.category_name
        for cat in self._extra_categories:
            if int(cat.get("id", -1)) == int(category_id):
                return str(cat.get("name", category_id))
        return str(category_id)

    # ------------------------------------------------------------------ #
    def register_image(self, file_name: str, width: int, height: int) -> ImageRecord:
        """Get (or create) the record for an image, keeping its existing id."""
        record = self.images.get(file_name)
        if record is None:
            record = ImageRecord(
                id=self._next_image_id, file_name=file_name,
                width=int(width), height=int(height),
            )
            self._next_image_id += 1
            self.images[file_name] = record
            self.dirty = True
        elif (record.width, record.height) != (width, height):
            # The file changed on disk since it was annotated. Record the new
            # size; existing polygons are in the old pixel space and would need
            # rescaling, so leave them and let the user see the mismatch.
            record.width, record.height = int(width), int(height)
            self.dirty = True
        return record

    def next_annotation_id(self) -> int:
        value = self._next_ann_id
        self._next_ann_id += 1
        self.dirty = True
        return value

    def get(self, file_name: str) -> List[Dict]:
        record = self.images.get(file_name)
        if record is None:
            return []
        return self.annotations.get(record.id, [])

    def set_annotations(self, file_name: str, entries: Sequence[Dict]) -> None:
        """Replace every annotation for one image."""
        record = self.images.get(file_name)
        if record is None:
            raise KeyError(f"{file_name} is not registered")
        if entries:
            self.annotations[record.id] = list(entries)
        else:
            self.annotations.pop(record.id, None)
        self.dirty = True

    def count(self, file_name: str) -> int:
        return len(self.get(file_name))

    def total_annotations(self) -> int:
        return sum(len(v) for v in self.annotations.values())

    def annotated_images(self) -> int:
        return sum(1 for v in self.annotations.values() if v)

    def summary(self) -> str:
        return (
            f"{self.total_annotations()} instance(s) across "
            f"{self.annotated_images()} image(s)"
        )
