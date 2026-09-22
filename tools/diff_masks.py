#!/usr/bin/env python3
"""Batch visual diff between your masks and reference masks.

    python tools/diff_masks.py predictions/ ground_truth/ diffs/ --images samples/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quickselect.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
