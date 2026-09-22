"""Batch mask-diff command line.

Takes a folder of your masks and a folder of reference masks, pairs them by
filename stem, and writes one visual diff image per pair.  No scores are
computed -- the output is meant to be looked at.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .diff import mask_diff, save_mask
from .io_utils import save_image


def _index(folder: Path) -> dict:
    return {p.stem: p for p in sorted(folder.glob("*")) if p.is_file()}


def pair_folders(
    pred_dir: Path, ref_dir: Path, image_dir: Optional[Path] = None
) -> Tuple[List[Tuple[str, Path, Path, Optional[Path]]], List[str]]:
    """Match files by stem.  Returns ``(pairs, unmatched_stems)``.

    A trailing ``_mask`` on either side is tolerated, since that is what both
    this tool and a Photoshop export tend to add.
    """
    preds = _index(pred_dir)
    refs = _index(ref_dir)
    images = _index(image_dir) if image_dir and image_dir.is_dir() else {}

    def lookup(table: dict, stem: str) -> Optional[Path]:
        for candidate in (stem, stem.replace("_mask", ""), f"{stem}_mask"):
            if candidate in table:
                return table[candidate]
        return None

    pairs, missing = [], []
    for stem, pred in preds.items():
        ref = lookup(refs, stem)
        if ref is None:
            missing.append(stem)
            continue
        pairs.append((stem, pred, ref, lookup(images, stem)))
    return pairs, missing


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quickselect-diff",
        description=(
            "Write a visual diff image for each predicted/reference mask pair. "
            "Orange = only in your mask, blue = only in the reference."
        ),
    )
    parser.add_argument("predicted", type=Path, help="folder of your masks")
    parser.add_argument("reference", type=Path, help="folder of reference masks")
    parser.add_argument("output", type=Path, help="folder to write diff images into")
    parser.add_argument(
        "--images", type=Path, default=None,
        help="optional folder of source photos to use as the backdrop",
    )
    parser.add_argument(
        "--opacity", type=float, default=0.75,
        help="overlay opacity, 0..1 (default 0.75)",
    )
    parser.add_argument(
        "--no-outline", action="store_true",
        help="skip the contour outlines",
    )
    args = parser.parse_args(argv)

    for folder, label in ((args.predicted, "predicted"), (args.reference, "reference")):
        if not folder.is_dir():
            print(f"error: {label} folder does not exist: {folder}", file=sys.stderr)
            return 2

    pairs, missing = pair_folders(args.predicted, args.reference, args.images)
    if not pairs:
        print("No matching pairs. Files are paired by filename stem.", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    for stem, pred, ref, image in pairs:
        diff = mask_diff(
            pred, ref, image,
            opacity=args.opacity,
            outline=not args.no_outline,
        )
        out_path = args.output / f"{stem}_diff.png"
        save_image(out_path, diff)
        print(f"{stem}: {out_path}")

    if missing:
        print(
            f"\n{len(missing)} mask(s) had no reference and were skipped: "
            + ", ".join(sorted(missing)[:10])
            + (" ..." if len(missing) > 10 else ""),
            file=sys.stderr,
        )
    print(f"\nWrote {len(pairs)} diff image(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
