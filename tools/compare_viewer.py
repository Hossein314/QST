#!/usr/bin/env python3
"""Side-by-side mask viewer with a blend slider.

    python tools/compare_viewer.py predictions/ ground_truth/ --images samples/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quickselect.ui.viewer import main

if __name__ == "__main__":
    raise SystemExit(main())
