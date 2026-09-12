"""
Stage G — Compute required outputs per accepted branch.

Owner: Person 4

Responsibilities:
- Seed point: 5mm along the proximal centerline from the ostium.
- Radius: distance-transform value at the seed point (distance to nearest
  non-vessel voxel), converted to mm.
- Direction: unit vector following the proximal path (PCA or normalized
  ostium->seed difference).

These three feed directly into the "Daughter-instance quality" (15%)
rubric category -- see toralis-branchseed-math.md section 12.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk

from components import Component


def get_seed_point(
    centerline_mm: np.ndarray,
    seed_arc_length_mm: float = 5.0,
) -> tuple[float, float, float]:
    """
    Walk along `centerline_mm` (already converted to physical mm, ordered
    from ostium outward) and return the point at exactly
    `seed_arc_length_mm` arc length (interpolating between the two
    nearest centerline points if needed).
    """
    raise NotImplementedError


def get_radius_mm(
    component: Component,
    seed_point_index: tuple[int, int, int],
    spacing_mm: tuple[float, float, float],
) -> float:
    """
    Compute the Euclidean distance transform of `component`'s voxel mask
    and return its value at `seed_point_index`, scaled to mm using
    `spacing_mm` -- this approximates the local vessel radius.
    """
    raise NotImplementedError


def get_direction(proximal_segment_mm: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate the unit direction vector of the proximal segment (already
    in physical mm), pointing FROM the ostium INTO the daughter vessel.

    Preferred: PCA on the point cloud, taking the first principal
    component, sign-corrected to point away from the aorta.
    Simpler fallback: normalize(seed_point - ostium_point).
    """
    raise NotImplementedError
