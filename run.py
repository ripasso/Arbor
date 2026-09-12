"""
CLI entry point.

Owner: Integration owner (wires all modules together once each is ready).

Usage (per spec):
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
"""

from __future__ import annotations

import argparse

import config
import io_utils
import roi
import region_growing
import vesselness
import components
import ostium
import tracing
import measurements
import output as output_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Branchseed: detect direct aortic daughter branches.")
    parser.add_argument("--image", required=True, help="Path to the CT volume (.nii/.nii.gz)")
    parser.add_argument("--aorta-mask", required=True, help="Path to the binary aorta mask (.nii/.nii.gz)")
    parser.add_argument("--output", required=True, help="Path to write the output prediction JSON")
    parser.add_argument("--case-id", default=None, help="Case identifier (defaults to inferring from --image)")
    return parser.parse_args()


def run_pipeline(image_path: str, mask_path: str, case_id: str) -> dict:
    """
    Full Stage A -> H pipeline for one case. Returns the prediction dict
    (matches output.build_json's schema).

    NOTE: this is the integration point -- fill in once each stage's
    module is implemented. Left unimplemented deliberately so each
    person's module can be unit-tested independently first.
    """
    raise NotImplementedError


def main() -> None:
    args = parse_args()
    case_id = args.case_id or args.image.split("/")[-1].split(".")[0]
    prediction = run_pipeline(args.image, args.aorta_mask, case_id)
    output_module.write_json(prediction, args.output)


if __name__ == "__main__":
    main()
