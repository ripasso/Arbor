"""
Stage H — Output assembly.

Owner: Person 1

Responsibilities:
- Assemble the required JSON schema for one case.
- Write it to disk.
- Write label maps for ITK-SNAP visualization.

Schema-only module -- doesn't depend on any of the vision/algorithm
modules, so it can be built and unit-tested independently and early.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
import numpy as np
import SimpleITK as sitk


@dataclass
class Daughter:
    instance_id: str                                  # e.g. "branch_001"
    parent_instance_id: str                            # always "aorta"
    ostium_xyz_mm: tuple[float, float, float]
    seed_xyz_mm: tuple[float, float, float]
    radius_mm: float
    direction_xyz: tuple[float, float, float]          # unit vector


def build_json(case_id: str, daughters: list[Daughter]) -> dict:
    """
    Assemble the required output dict:
    {
      "case_id": ...,
      "parent": {"instance_id": "aorta"},
      "daughters": [ {...}, ... ]
    }
    An empty `daughters` list is valid (no eligible branches found).
    """
    return {
        "case_id": case_id,
        "parent": {"instance_id": "aorta"},
        "daughters": [
            {
                "instance_id": d.instance_id,
                "parent_instance_id": d.parent_instance_id,
                "ostium_xyz_mm": [float(v) for v in d.ostium_xyz_mm],
                "seed_xyz_mm": [float(v) for v in d.seed_xyz_mm],
                "radius_mm": float(d.radius_mm),
                "direction_xyz": [float(v) for v in d.direction_xyz],
            }
            for d in daughters
        ],
    }


def write_json(data: dict, output_path: str) -> None:
    """Write `data` as pretty-printed JSON to `output_path`."""
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)


def write_label_map(image: sitk.Image, aorta_mask: sitk.Image, daughters: list, output_path: str) -> None:
    """
    Builds a label map in the ORIGINAL image's grid (not the cropped ROI) so
    it overlays directly on orig*.nii in ITK-SNAP: 1 = aorta, 2..N+1 = each
    daughter's ostium marker + proximal direction segment.
    """
    arr = np.zeros(sitk.GetArrayFromImage(image).shape, dtype=np.uint8)  # (Z, Y, X)
    aorta_arr = sitk.GetArrayFromImage(aorta_mask) > 0
    arr[aorta_arr] = 1

    size = image.GetSize()
    spacing = np.asarray(image.GetSpacing())
    step_mm = float(min(spacing)) * 0.5  # sub-voxel step so the line has no gaps

    for i, d in enumerate(daughters, start=2):
        # Dynamically support both dictionary formats and Daughter objects
        is_dict = isinstance(d, dict)
        direction = np.asarray(d["direction_xyz"] if is_dict else d.direction_xyz, dtype=float)
        direction /= (np.linalg.norm(direction) + 1e-9)
        ostium = np.asarray(d["ostium_xyz_mm"] if is_dict else d.ostium_xyz_mm, dtype=float)

        n_steps = int(np.ceil(10.0 / step_mm))  # burn the same 10mm proximal length as the spec
        for s in range(n_steps + 1):
            pt = tuple(ostium + direction * step_mm * s)
            idx = image.TransformPhysicalPointToIndex(pt)
            if all(0 <= idx[k] < size[k] for k in range(3)):
                arr[idx[2], idx[1], idx[0]] = i

        # 2mm marker sphere at the ostium itself so it's visible without zooming in
        ostium_idx = np.asarray(image.TransformPhysicalPointToIndex(tuple(ostium)))
        r_vox = max(1, int(round(2.0 / min(spacing))))
        for dz in range(-r_vox, r_vox + 1):
            for dy in range(-r_vox, r_vox + 1):
                for dx in range(-r_vox, r_vox + 1):
                    if dx*dx + dy*dy + dz*dz <= r_vox*r_vox:
                        p = ostium_idx + np.array([dx, dy, dz])
                        if all(0 <= p[k] < size[k] for k in range(3)):
                            arr[p[2], p[1], p[0]] = i

    out = sitk.GetImageFromArray(arr)
    out.CopyInformation(image)
    sitk.WriteImage(out, output_path)