"""Qt (PySide6) user interface for the quick-selection engine.

Importing this package pulls in PySide6.  The engine itself does not, so it can
be used headlessly without a GUI toolkit installed.
"""

from .app import main
from .canvas import Canvas
from .mainwindow import MainWindow
from .worker import EngineWorker

__all__ = ["Canvas", "EngineWorker", "MainWindow", "main"]
