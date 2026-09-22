"""Instance-segmentation annotation tool built on the quick-selection engine.

    python annotate.py /path/to/images

Produces one COCO ``annotations.json`` per image folder. See the package
README for the output format.
"""

from .dataset_io import CocoDataset, ImageRecord, scan_folder
from .polygon import (
    annotation_from_mask,
    mask_to_polygons,
    polygons_area,
    polygons_bbox,
    polygons_to_mask,
)

__all__ = [
    "CocoDataset",
    "ImageRecord",
    "annotation_from_mask",
    "mask_to_polygons",
    "polygons_area",
    "polygons_bbox",
    "polygons_to_mask",
    "scan_folder",
]
