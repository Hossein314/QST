#!/usr/bin/env python3
"""Launch the Quick Select application.

    python run.py                 # start empty, open an image with Ctrl+O
    python run.py photo.jpg       # open an image straight away
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quickselect.ui.app import main

if __name__ == "__main__":
    raise SystemExit(main())
