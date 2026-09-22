"""quickselect -- an open re-implementation of a Quick-Selection-style tool.

Paint over an object; the selection floods out to similar pixels and stops at
image edges.  The algorithm is built from published work only:

* Boykov & Jolly, *Interactive Graph Cuts* (ICCV 2001)
* Li, Sun, Tang & Shum, *Lazy Snapping* (SIGGRAPH 2004)
* Rother, Kolmogorov & Blake, *GrabCut* (SIGGRAPH 2004)
* Liu, Sun & Shum, *Paint Selection* (SIGGRAPH 2009)
* Boykov & Kolmogorov, *An Experimental Comparison of Min-Cut/Max-Flow
  Algorithms* (PAMI 2004)
* He, Sun & Tang, *Guided Image Filtering* (ECCV 2010)
* Levin, Lischinski & Weiss, *A Closed-Form Solution to Natural Image Matting*
  (PAMI 2008) -- approximated by the guided filter, see ``refine.py``

No Adobe code, assets or APIs are involved.

Quick start::

    from quickselect import QuickSelectEngine, LayerStack, load_image
    engine = QuickSelectEngine(LayerStack.from_image(load_image("cat.jpg")))
    engine.paint_polyline([(320, 240), (340, 250), (360, 260)])
    alpha = engine.selection_alpha()
"""

from .brush import BrushStroke, StrokeRasterizer, stamp_kernel
from .colormodel import GaussianMixtureModel, HistogramModel, build_model
from .config import (
    BACKGROUND,
    FOREGROUND,
    UNKNOWN,
    BrushConfig,
    EngineConfig,
    RefineEdgeConfig,
    ToolState,
)
from .diff import mask_diff, load_mask, save_mask
from .engine import QuickSelectEngine, SelectionMode, SelectionUpdate
from .history import History
from .imagedata import ImagePyramid, Layer, LayerStack, ROI
from .io_utils import load_image, save_image
from .refine import apply_refine_edge, guided_filter, mask_to_contours

__version__ = "1.0.0"

__all__ = [
    "BACKGROUND",
    "FOREGROUND",
    "UNKNOWN",
    "BrushConfig",
    "BrushStroke",
    "EngineConfig",
    "GaussianMixtureModel",
    "History",
    "HistogramModel",
    "ImagePyramid",
    "Layer",
    "LayerStack",
    "QuickSelectEngine",
    "ROI",
    "RefineEdgeConfig",
    "SelectionMode",
    "SelectionUpdate",
    "StrokeRasterizer",
    "ToolState",
    "apply_refine_edge",
    "build_model",
    "guided_filter",
    "load_image",
    "load_mask",
    "mask_diff",
    "mask_to_contours",
    "save_image",
    "save_mask",
    "stamp_kernel",
    "__version__",
]
