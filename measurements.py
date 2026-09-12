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
from scipy.ndimage import distance_transform_edt

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
    pts = np.asarray(centerline_mm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("centerline_mm must be an (N, 3) array with N >= 2")

    seg_lengths = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    if not 0.0 <= seed_arc_length_mm <= arc[-1]:
        raise ValueError(
            f"centerline arc length {arc[-1]:.3f}mm cannot reach seed at {seed_arc_length_mm}mm"
        )

    i = min(int(np.searchsorted(arc, seed_arc_length_mm, side="right")) - 1, len(pts) - 2)
    t = 0.0 if seg_lengths[i] == 0.0 else (seed_arc_length_mm - arc[i]) / seg_lengths[i]
    seed = pts[i] + t * (pts[i + 1] - pts[i])
    return tuple(float(x) for x in seed)


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
    idx = np.asarray(component.voxel_indices, dtype=int)
    if idx.ndim != 2 or idx.shape[1] != 3 or len(idx) == 0:
        raise ValueError("component.voxel_indices must be an (N, 3) array with N >= 1")

    # Crop to the component's bounding box with a 1-voxel background border
    # so the EDT sees the mask boundary on every side.
    mins = idx.min(axis=0)
    local = idx - mins + 1
    mask = np.zeros(idx.max(axis=0) - mins + 3, dtype=bool)
    mask[tuple(local.T)] = True

    edt_mm = distance_transform_edt(mask, sampling=spacing_mm)

    seed_local = np.round(np.asarray(seed_point_index, dtype=float) - mins + 1).astype(int)
    if np.any(seed_local < 0) or np.any(seed_local >= mask.shape) or not mask[tuple(seed_local)]:
        # Seed landed outside the mask -- use the nearest component voxel.
        seed_local = local[np.argmin(((idx - np.asarray(seed_point_index)) ** 2).sum(axis=1))]
    return float(edt_mm[tuple(seed_local)])


def get_direction(proximal_segment_mm: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate the unit direction vector of the proximal segment (already
    in physical mm), pointing FROM the ostium INTO the daughter vessel.

    Preferred: PCA on the point cloud, taking the first principal
    component, sign-corrected to point away from the aorta.
    Simpler fallback: normalize(seed_point - ostium_point).
    """
    pts = np.asarray(proximal_segment_mm, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < 2:
        raise ValueError("proximal_segment_mm must be an (N, 3) array with N >= 2")

    outward = pts[-1] - pts[0]
    centered = pts - pts.mean(axis=0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered)
    if eigenvalues[-1] <= 0.0:
        raise ValueError("proximal segment has no spatial extent; direction is undefined")

    direction = eigenvectors[:, -1]
    if np.dot(direction, outward) < 0.0:
        direction = -direction
    direction /= np.linalg.norm(direction)
    return tuple(float(x) for x in direction)
