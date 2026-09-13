"""
CLI entry point.

Owner: Integration owner (wires all modules together once each is ready).

Usage (per spec):
    python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
"""

from __future__ import annotations

import argparse

import numpy as np

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


def run_pipeline(image_path: str, mask_path: str, case_id: str, label_map_path: str | None = None) -> dict:
    """
    Full Stage A -> H pipeline for one case. Returns the prediction dict
    (matches output.build_json's schema) and writes label map if path is provided.
    """
    case = io_utils.load_case(image_path, mask_path, case_id)
    image, mask = case.image, case.mask
    spacing = io_utils.get_spacing_mm(image)

    # Stage B/C: ROI -> search shell -> candidate vessel voxels
    roi_image, roi_mask, _ = roi.crop_aorta_roi(image, mask, margin_mm=20.0)
    shell = region_growing.extract_wall_search_shell(
        roi_mask, shell_radius_mm=config.SEARCH_SHELL_MARGIN_MM
    )
    candidate_mask = vesselness.detect_candidate_vessels(roi_image, shell, roi_mask)

    # Stage D: connected components touching the aorta wall
    comps = components.label_components(candidate_mask)
    touching = components.filter_touching_aorta(comps, roi_mask)

    # Stage E: ostium candidates -> crop-face rejection + dedup
    ostia = []
    for comp in touching:
        contact = ostium.find_contact_voxels(comp, roi_mask)
        if len(contact) == 0:
            continue
        ostia.append(
            ostium.OstiumCandidate(
                component=comp,
                contact_voxel_indices=contact,
                centroid_index=ostium.find_contact_centroid(contact),
                is_cropped_face=ostium.is_cropped_face(
                    contact,
                    roi_mask,
                    config.CROP_FACE_EDGE_SLICES,
                    config.PLANARITY_FLATNESS_RATIO,
                ),
            )
        )
    ostia = ostium.deduplicate_ostia(ostia)
    ostia = ostium.merge_nearby_ostia(ostia, roi_image, merge_dist_mm=4.0)

    # Diagnostic prints to track components, wall zones, and deduplication
    print("raw components:", len(comps))
    print("touching aorta:", len(touching))
    for c in touching:
        contact = ostium.find_contact_voxels(c, roi_mask)
        zones = ostium._contact_zones(contact)
        print(f"  component {c.label}: {len(contact)} contact voxels, {len(zones)} wall zones")
    print("ostia after dedup:", len(ostia))

    # Stage F/G: centerline -> eligibility -> proximal segment -> measurements
    daughters = []
    for cand in ostia:
        if cand.component.voxel_indices.size == 0:
            continue
            
        centerline = tracing.extract_centerline(cand.component, roi_mask)
        if centerline is None or len(centerline) < 2:
            continue
            
        # MANDATORY SPEC GUARD: Enforce that the branch tracks outward ≥ 5.0mm from the wall
        if not tracing.check_eligibility(
            centerline, spacing, config.ELIGIBILITY_MIN_TRACE_MM
        ):
            continue
            
        bif = tracing.find_first_bifurcation(cand.component, centerline)
        segment = tracing.trim_to_proximal_segment(
            centerline, spacing, config.PROXIMAL_MAX_TRACE_MM, bif
        )
        
        if segment is None or len(segment) == 0:
            continue

        path_idx = np.vstack(
            [np.asarray(cand.centroid_index, dtype=float), segment.astype(float)]
        )
        path_mm = np.array([io_utils.voxel_to_mm(roi_image, p) for p in path_idx])
        
        try:
            ostium_mm = io_utils.voxel_to_mm(roi_image, cand.centroid_index)
            seed_mm = measurements.get_seed_point(
                path_mm, config.ELIGIBILITY_MIN_TRACE_MM
            )
            direction = measurements.get_direction(path_mm)
            seed_idx = roi_image.TransformPhysicalPointToIndex(seed_mm)
            radius = measurements.get_radius_mm(cand.component, seed_idx, spacing)
        except (ValueError, IndexError):
            continue

        daughters.append(
            output_module.Daughter(
                instance_id="",
                parent_instance_id="aorta",
                ostium_xyz_mm=ostium_mm,
                seed_xyz_mm=seed_mm,
                radius_mm=radius,
                direction_xyz=direction,
            )
        )

    # Sort cleanly by vertical anatomical depth along the Z axis
    daughters.sort(key=lambda d: d.ostium_xyz_mm[2])
    for i, d in enumerate(daughters, start=1):
        d.instance_id = f"branch_{i:03d}"

    prediction = output_module.build_json(case_id, daughters)

    if label_map_path:
        output_module.write_label_map(image, mask, daughters, label_map_path)

    return prediction


def main() -> None:
    args = parse_args()
    case_id = args.case_id or args.image.split("/")[-1].split(".")[0]
    label_path = args.output[:-5] + "_labels.nii.gz" if args.output.endswith(".json") else args.output + "_labels.nii.gz"
    prediction = run_pipeline(args.image, args.aorta_mask, case_id, label_map_path=label_path)
    output_module.write_json(prediction, args.output)


if __name__ == "__main__":
    main()