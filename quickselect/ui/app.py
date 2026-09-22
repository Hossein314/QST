"""Application entry point."""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

from .mainwindow import MainWindow


def apply_dark_palette(app: QApplication) -> None:
    """A neutral dark theme -- a bright chrome biases how you judge a mask."""
    app.setStyle("Fusion")
    p = QPalette()
    base = QColor(43, 45, 49)
    alt = QColor(53, 56, 61)
    text = QColor(226, 228, 232)
    p.setColor(QPalette.Window, base)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, QColor(35, 37, 40))
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, alt)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.ToolTipBase, QColor(28, 30, 33))
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Highlight, QColor(64, 132, 255))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor(130, 133, 138))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(130, 133, 138))
    app.setPalette(p)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quickselect",
        description="Interactive quick-selection tool (graph-cut based).",
    )
    parser.add_argument("image", nargs="?", help="image to open on start-up")
    args = parser.parse_args(argv)

    app = QApplication(sys.argv[:1])
    app.setApplicationName("Quick Select")
    apply_dark_palette(app)

    window = MainWindow(args.image)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
