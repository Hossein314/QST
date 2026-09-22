#!/usr/bin/env python3
"""Launch the board annotation tool.

    python annotate.py                    # pick a folder with Ctrl+O
    python annotate.py /path/to/images    # open a folder straight away
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from quickselect.annotator.app import main

if __name__ == "__main__":
    raise SystemExit(main())
