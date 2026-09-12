"""
Stage E — Ostium identification & de-duplication.

Owner: Person 3

Responsibilities:
- For each surviving component, find the contact centroid where it meets
  the aorta wall -- this is the candidate ostium.
- Reject contacts on the flat cropped top/bottom faces of the aorta volume
  (not real anatomy).
- Handle the spec's tricky merge/split rules:
    * two genuinely separate nearby ostia -> keep as two instances
    * one common trunk that forks shortly after leaving the aorta -> one
      instance
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

from components import Component


@dataclass
class OstiumCandidate:
    """A candidate ostium derived from one connected component."""
    component: Component
    contact_voxel_indices: np.ndarray   # voxels of `component` adjacent to aorta_mask
    centroid_index: tuple[float, float, float]
    is_cropped_face: bool


def find_contact_voxels(component: Component, aorta_mask: sitk.Image) -> np.ndarray:
    """
    Return the subset of `component`'s voxels that are adjacent to
    `aorta_mask` (the actual contact zone at the aortic wall).
    """
    mask_arr = sitk.GetArrayFromImage(aorta_mask) > 0
    dilated = ndimage.binary_dilation(mask_arr, structure=np.ones((3, 3, 3), dtype=int))
    zyx = component.voxel_indices[:, ::-1]
    is_contact = dilated[zyx[:, 0], zyx[:, 1], zyx[:, 2]]
    return component.voxel_indices[is_contact]


def find_contact_centroid(contact_voxel_indices: np.ndarray) -> tuple[float, float, float]:
    """
    Average the contact zone's voxel indices to get a sub-voxel-precision
    centroid -- this is what gets converted to mm and reported as
    ostium_xyz_mm (drives the 25%-weighted ostium localisation score).
    """
    return tuple(contact_voxel_indices.astype(float).mean(axis=0))


def is_cropped_face(
    contact_voxel_indices: np.ndarray,
    aorta_mask: sitk.Image,
    edge_slices: int,
    flatness_ratio: float,
) -> bool:
    """
    Reject contact zones sitting in the first/last `edge_slices` of the
    mask's z-extent AND whose local geometry is planar (PCA-based
    flatness test using `flatness_ratio`) -- these are the aorta's cut
    ends, not real branch origins.
    """
    mask_arr = sitk.GetArrayFromImage(aorta_mask) > 0
    occupied_ks = np.nonzero(mask_arr.any(axis=(1, 2)))[0]
    k_min, k_max = occupied_ks.min(), occupied_ks.max()

    contact_ks = contact_voxel_indices[:, 2]
    at_edge = (
        contact_ks.min() <= k_min + edge_slices
        or contact_ks.max() >= k_max - edge_slices
    )
    if not at_edge:
        return False

    spacing = np.asarray(aorta_mask.GetSpacing())
    pts = contact_voxel_indices.astype(float) * spacing
    pts = pts - pts.mean(axis=0)
    eigvals = np.linalg.eigvalsh(pts.T @ pts / len(pts))
    return bool(eigvals[0] < flatness_ratio * eigvals[-1])


def deduplicate_ostia(candidates: list[OstiumCandidate]) -> list[OstiumCandidate]:
    """
    Apply the spec's merge/split rules:
    - Distinct connected components at the wall stay distinct even if
      physically close together.
    - A component that touches the wall at one contact zone and only
      splits downstream is ONE instance, not multiple.
    Returns the final deduplicated, crop-face-filtered candidate list.
    """
    kept = []
    seen_labels = set()
    for cand in candidates:
        if cand.is_cropped_face:
            continue
        if cand.component.label in seen_labels:
            continue
        seen_labels.add(cand.component.label)

        zones = _contact_zones(cand.contact_voxel_indices)
        if len(zones) > 1:
            largest = max(zones, key=len)
            cand = OstiumCandidate(
                component=cand.component,
                contact_voxel_indices=largest,
                centroid_index=find_contact_centroid(largest),
                is_cropped_face=False,
            )
        kept.append(cand)
    return kept


def _contact_zones(contact_voxel_indices: np.ndarray) -> list[np.ndarray]:
    """Split a contact patch into its 26-connected sub-zones."""
    if len(contact_voxel_indices) == 0:
        return []
    local = contact_voxel_indices - contact_voxel_indices.min(axis=0)
    grid = np.zeros(local.max(axis=0) + 1, dtype=bool)
    grid[tuple(local.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3), dtype=int))
    per_voxel = labels[tuple(local.T)]
    return [contact_voxel_indices[per_voxel == i] for i in range(1, n + 1)]
