"""
Stage F — Eligibility check & proximal tracing.

Owner: Person 4

Responsibilities:
- Extract a 1D centerline (skeleton) for a candidate component.
- Check the lumen is continuously traceable for >= 5mm beyond the aortic
  wall (eligibility rule) -- discard candidates that fail this.
- Trim the centerline to the proximal segment: up to 10mm beyond the
  ostium, or the first downstream bifurcation, whichever comes first.

Can be developed against a synthetic/stubbed centerline (e.g. a straight
line of points) before ostium.py's real output exists -- see the
parallelization note in the program outline doc.
"""

from __future__ import annotations

from collections import deque

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from skimage.morphology import skeletonize

from components import Component

_NEIGHBOR_OFFSETS = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
]


def _component_grid(component: Component) -> tuple[np.ndarray, np.ndarray]:
    """Build a padded local boolean grid of the component's voxels.

    Returns (grid, offset) where index coords = grid coords + offset.
    """
    ijk = np.asarray(component.voxel_indices)
    mins = ijk.min(axis=0)
    local = ijk - mins + 1  # 1-voxel padding so thinning works at edges
    grid = np.zeros(local.max(axis=0) + 2, dtype=bool)
    grid[tuple(local.T)] = True
    return grid, mins - 1


def _skeleton_index_set(component: Component) -> set[tuple[int, int, int]]:
    grid, offset = _component_grid(component)
    skel = skeletonize(grid, method="lee")
    return {tuple(p) for p in np.argwhere(skel) + offset}


def _skeleton_neighbors(p: tuple[int, int, int], skel: set) -> list[tuple[int, int, int]]:
    return [
        (p[0] + dx, p[1] + dy, p[2] + dz)
        for dx, dy, dz in _NEIGHBOR_OFFSETS
        if (p[0] + dx, p[1] + dy, p[2] + dz) in skel
    ]


def extract_centerline(component: Component, aorta_mask: sitk.Image | None = None) -> np.ndarray:
    """
    Skeletonize `component`'s voxel mask and order the resulting skeleton
    voxels into a single path (ordered array of shape (N, 3)) starting
    from the end closest to the aorta wall.

    If `aorta_mask` is given, the start endpoint is the skeleton end
    nearest the mask (in physical mm). Otherwise it falls back to the
    endpoint nearest the component's voxel centroid.
    """
    skel = _skeleton_index_set(component)
    if not skel:
        return np.empty((0, 3), dtype=int)

    adjacency = {p: _skeleton_neighbors(p, skel) for p in skel}
    endpoints = [p for p, ns in adjacency.items() if len(ns) <= 1]
    if not endpoints:
        endpoints = list(skel)  # closed loop: start anywhere

    if aorta_mask is not None:
        mask_arr = sitk.GetArrayFromImage(aorta_mask) > 0
        dist = ndimage.distance_transform_edt(
            ~mask_arr, sampling=aorta_mask.GetSpacing()
        )
        start = min(endpoints, key=lambda p: dist[p[2], p[1], p[0]])
    else:
        centroid = np.asarray(component.voxel_indices).mean(axis=0)
        start = min(endpoints, key=lambda p: np.linalg.norm(np.asarray(p) - centroid))

    # BFS tree from `start`; the main path is the shortest path to the
    # farthest skeleton voxel (the longest branch defines the trunk).
    parent = {start: None}
    dist = {start: 0}
    farthest = start
    queue = deque([start])
    while queue:
        p = queue.popleft()
        for nb in adjacency[p]:
            if nb not in parent:
                parent[nb] = p
                dist[nb] = dist[p] + 1
                if dist[nb] > dist[farthest]:
                    farthest = nb
                queue.append(nb)

    path = []
    node = farthest
    while node is not None:
        path.append(node)
        node = parent[node]
    return np.asarray(path[::-1], dtype=int)


def arc_length_mm(centerline: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """
    Given an ordered centerline (voxel indices), return the cumulative
    physical arc length in mm at each point (index 0 = 0.0).
    """
    steps = np.diff(centerline.astype(float), axis=0) * np.asarray(spacing_mm)
    seg_lengths = np.linalg.norm(steps, axis=1)
    return np.concatenate([[0.0], np.cumsum(seg_lengths)])


def check_eligibility(
    centerline: np.ndarray,
    spacing_mm: tuple[float, float, float],
    min_trace_mm: float,
) -> bool:
    """
    Return True if the centerline's total arc length reaches at least
    `min_trace_mm` (the 5mm rule) before fading out or ending.
    """
    if len(centerline) < 2:
        return False
    return float(arc_length_mm(centerline, spacing_mm)[-1]) >= min_trace_mm


def find_first_bifurcation(component: Component, centerline: np.ndarray) -> int | None:
    """
    Return the index (along `centerline`) of the first skeleton voxel with
    >=3 skeleton neighbours (a bifurcation point), or None if the segment
    has no bifurcation within the traced region.
    """
    skel = _skeleton_index_set(component)
    for idx in range(1, len(centerline)):
        p = tuple(centerline[idx])
        prev_pt = tuple(centerline[idx - 1])
        next_pt = tuple(centerline[idx + 1]) if idx + 1 < len(centerline) else None
        extra = [
            nb for nb in _skeleton_neighbors(p, skel)
            if nb != prev_pt and nb != next_pt
        ]
        if extra:
            return idx
    return None


def trim_to_proximal_segment(
    centerline: np.ndarray,
    spacing_mm: tuple[float, float, float],
    max_trace_mm: float,
    bifurcation_index: int | None,
) -> np.ndarray:
    """
    Trim `centerline` to the proximal segment: stop at `max_trace_mm`
    (the 10mm rule) or at `bifurcation_index`, whichever comes first.
    """
    if len(centerline) == 0:
        return centerline
    arc = arc_length_mm(centerline, spacing_mm)
    cut = int(np.searchsorted(arc, max_trace_mm, side="right"))
    if bifurcation_index is not None:
        cut = min(cut, bifurcation_index)
    return centerline[: max(cut, 1)]
