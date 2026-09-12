#!/usr/bin/env python3
"""Branchseed: find every artery that leaves the supplied abdominal aorta.

    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json

Writes one JSON file holding, for each daughter found, the centre of its
opening, a seed 5 mm along it, its initial direction and its local lumen
radius, all in the physical millimetre frame SimpleITK reports.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pipeline import process_case, to_prediction


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="CT volume (.nii or .nii.gz)")
    ap.add_argument("--aorta-mask", required=True, dest="aorta_mask",
                    help="binary mask of the parent aortic lumen")
    ap.add_argument("--output", required=True, help="where to write the prediction JSON")
    ap.add_argument("--case-id", default=None,
                    help="case identifier to record (default: the image's folder name)")
    ap.add_argument("--quiet", action="store_true", help="suppress the progress line")
    args = ap.parse_args(argv)

    case_id = args.case_id or os.path.basename(os.path.dirname(os.path.abspath(args.image))) \
        or os.path.splitext(os.path.basename(args.image))[0]

    started = time.time()
    try:
        record = process_case(args.image, args.aorta_mask, case_id, want_maps=False)
        prediction = to_prediction(record)
    except Exception as exc:                      # never fail the whole run on one case
        print(f"[branchseed] {case_id}: {exc}", file=sys.stderr)
        prediction = {"case_id": case_id, "parent": {"instance_id": "aorta"}, "daughters": []}

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(prediction, fh, indent=2)

    if not args.quiet:
        n = len(prediction["daughters"])
        print(f"[branchseed] {case_id}: {n} daughter{'' if n == 1 else 's'} "
              f"in {time.time() - started:.1f}s -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
