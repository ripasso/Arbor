"""
Stage D — Candidate branch detection (connected components).

Owner: Person 3

Responsibilities:
- Group candidate vessel voxels (output of region_growing + vesselness
  filtering) into connected components (26-connectivity in 3D).
- Determine which components actually touch the aorta mask's outer
  surface -- only those are candidate direct daughters. Components that
  float disconnected from the aorta are discarded.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import SimpleITK as sitk
from scipy import ndimage


@dataclass
class Component:
    """One connected component of candidate vessel voxels."""
    label: int
    voxel_indices: np.ndarray   # shape (N, 3), each row (i, j, k)
    touches_aorta: bool


def label_components(candidate_mask: sitk.Image, min_voxels: int = 15) -> list[Component]:
    """
    Labels 26-connected components, filtering out small speckles (<15 voxels) 
    and bulky non-tubular blobs using PCA elongation and width thresholds.
    """
    arr = sitk.GetArrayFromImage(candidate_mask) > 0
    labels, n = ndimage.label(arr, structure=np.ones((3, 3, 3), dtype=int))
    
    # Read the spacing vector dynamically from the SimpleITK header metadata
    spacing = np.array(candidate_mask.GetSpacing())
    
    valid_components = []
    for lab in range(1, n + 1):
        zyx = np.argwhere(labels == lab)
        if len(zyx) < min_voxels:
            continue
            
        ijk = np.ascontiguousarray(zyx[:, ::-1])
        pts_mm = ijk * spacing
        pts_c = pts_mm - pts_mm.mean(axis=0)
        
        # Compute covariance and eigenvalues for shape analysis
        cov = pts_c.T @ pts_c / max(1, len(pts_c))
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        elong = eigvals[0] / (eigvals[1] + 1e-6)
        width_mm = 2 * np.sqrt(max(eigvals[1], 0))
        
        # Enforce strict tubular vessel geometry (kill kidneys, IVC, and muscle blobs)
        if elong < 3.0 or width_mm > 4.5:
            continue
            
        valid_components.append(
            Component(label=lab, voxel_indices=ijk, touches_aorta=False)
        )
        
    return valid_components


def filter_touching_aorta(components: list[Component], aorta_mask: sitk.Image) -> list[Component]:
    """
    For each component, test whether any of its voxels are adjacent
    (26-connectivity) to a voxel in `aorta_mask`. Sets/filters by touches_aorta.
    """
    mask_arr = sitk.GetArrayFromImage(aorta_mask) > 0
    dilated = ndimage.binary_dilation(mask_arr, structure=np.ones((3, 3, 3), dtype=int))
    for comp in components:
        zyx = comp.voxel_indices[:, ::-1]
        comp.touches_aorta = bool(dilated[zyx[:, 0], zyx[:, 1], zyx[:, 2]].any())
    return [comp for comp in components if comp.touches_aorta]
