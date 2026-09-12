"""
Stage C (primary detector) — Region growing from the aorta mask.

Owner: Person 2

Per the math doc's critical evaluation (toralis-branchseed-math.md, section 11):
connectivity to the aorta is a stronger, simpler signal than tube-shape
scoring, and it's directly aligned with the spec's own definition of a
daughter ("lumen connects directly to the parent aorta"). This should be
the PRIMARY detector; vesselness.py is a SECONDARY filter used to reject
false positives (calcified plaque, bone, IVC) from this module's output.
"""

from __future__ import annotations

import SimpleITK as sitk


def calibrate_hu_range(image: sitk.Image, mask: sitk.Image) -> tuple[float, float]:
    """
    Compute intensity (HU) statistics INSIDE the given aorta mask to derive
    a per-case bright-blood threshold range. This adapts to per-patient
    contrast timing instead of relying on one hardcoded HU threshold.

    Returns (hu_low, hu_high).
    """
    raise NotImplementedError


def grow_from_mask(
    image: sitk.Image,
    mask: sitk.Image,
    roi_mask: sitk.Image,
    hu_low: float,
    hu_high: float,
) -> sitk.Image:
    """
    Region-grow outward from the aorta mask surface through voxels whose
    intensity falls within [hu_low, hu_high], restricted to `roi_mask`
    (the search shell). Returns a binary candidate_mask of grown voxels
    that are NOT part of the original aorta mask.
    """
    raise NotImplementedError
