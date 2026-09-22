#!/usr/bin/env python3
"""Report where segmentation time goes, stage by stage.

    python tools/profile_segmentation.py photo.jpg
    python tools/profile_segmentation.py photo.jpg --work-dim 512 --model gmm
    python tools/profile_segmentation.py photo.jpg --compare

Simulates a drag across the image and prints the per-stage breakdown, so a
slow machine or an awkward image can be diagnosed without guessing. ``--compare``
sweeps the settings that matter for speed and prints one row each.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quickselect.constraints import POSITIVE, ConstraintMap
from quickselect.io_utils import load_image
from quickselect.profile import STAGE_ORDER, Profiler
from quickselect.segmenter import SegmentParams, Segmenter, local_roi


def _disc(h: int, w: int, cx: int, cy: int, r: int):
    m = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool), (max(0, cy - r), min(h, cy + r + 1),
                            max(0, cx - r), min(w, cx + r + 1))


def simulate(
    image: np.ndarray,
    params: SegmentParams,
    steps: int = 30,
    brush: int = 28,
) -> tuple:
    """Paint a sweep across the middle of the image and time every pass."""
    h, w = image.shape[:2]
    constraints = ConstraintMap(h, w)
    session = Segmenter(image, params)
    prof = Profiler()
    previous = None
    times: List[float] = []
    totals: dict = {}
    warm = 0

    for i in range(steps):
        x = int(w * 0.25 + (w * 0.5) * i / max(1, steps - 1))
        y = int(h * 0.5 + h * 0.12 * np.sin(i / 3.0))
        stamp, bbox = _disc(h, w, x, y, brush)
        constraints.apply(stamp, POSITIVE, bbox=bbox)
        roi = local_roi(bbox, previous, params.local_margin, (h, w))
        t = time.perf_counter()
        previous = session.run(constraints, roi=roi, previous=previous, prof=prof)
        times.append((time.perf_counter() - t) * 1000.0)
        for k, v in prof.stages.items():
            totals[k] = totals.get(k, 0.0) + v
        warm += 1 if prof.notes.get("warm_start") else 0
    return times, totals, warm, session


def report(label: str, times: List[float], totals: dict, warm: int, n: int,
           session: Segmenter) -> None:
    print(f"\n{label}")
    print(f"  working resolution : {session.cache.width} x {session.cache.height}"
          f"  (scale {session.cache.scale:.3f})")
    print(f"  median {np.median(times):7.1f} ms   p95 {np.percentile(times, 95):7.1f} ms"
          f"   max {max(times):7.1f} ms   first {times[0]:7.1f} ms")
    print(f"  warm-started {warm}/{n} passes")
    print("  mean per stage:")
    ordered = [k for k in STAGE_ORDER if k in totals]
    ordered += [k for k in totals if k not in STAGE_ORDER and k != "total"]
    total_mean = totals.get("total", sum(totals.values())) / n
    for k in ordered:
        mean = totals[k] / n
        share = 100.0 * mean / total_mean if total_mean else 0.0
        bar = "#" * int(round(share / 3))
        print(f"    {k:<13} {mean:7.2f} ms  {share:5.1f}%  {bar}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("image", type=Path)
    parser.add_argument("--work-dim", type=int, default=None)
    parser.add_argument("--model", choices=("hist", "gmm"), default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--brush", type=int, default=28)
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--no-warm", action="store_true")
    parser.add_argument(
        "--compare", action="store_true",
        help="sweep the speed-relevant settings instead of one run",
    )
    args = parser.parse_args(argv)

    if not args.image.is_file():
        print(f"error: no such image: {args.image}", file=sys.stderr)
        return 2
    image = load_image(args.image)
    print(f"{args.image.name}: {image.shape[1]} x {image.shape[0]}")

    if args.compare:
        rows = [
            ("default (384, hist, local+warm)", SegmentParams()),
            ("work_max_dim=320", SegmentParams(work_max_dim=320)),
            ("work_max_dim=512", SegmentParams(work_max_dim=512)),
            ("gmm colour model", SegmentParams(model="gmm")),
            ("4-neighbourhood", SegmentParams(neighborhood=4)),
            ("no warm start", SegmentParams(warm_start_max_change=-1.0)),
            ("no local ROI", SegmentParams(local_enabled=False)),
        ]
        print(f"\n{'setting':<34}{'median':>9}{'p95':>9}{'max':>9}{'warm':>8}")
        for label, params in rows:
            times, totals, warm, _ = simulate(image, params, args.steps, args.brush)
            print(f"{label:<34}{np.median(times):8.1f} {np.percentile(times,95):8.1f} "
                  f"{max(times):8.1f} {warm:5d}/{len(times)}")
        return 0

    params = SegmentParams()
    if args.work_dim:
        params.work_max_dim = args.work_dim
    if args.model:
        params.model = args.model
    if args.no_local:
        params.local_enabled = False
    if args.no_warm:
        params.warm_start_max_change = -1.0

    times, totals, warm, session = simulate(image, params, args.steps, args.brush)
    report("drag simulation", times, totals, warm, len(times), session)
    print("\n  (first pass includes one-off per-image setup: the working image,"
          "\n   its Lab features and the neighbour edge weights.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
