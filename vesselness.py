"""
Stage C (secondary filter) — Vesselness / tube-likeness enhancement.

Owner: Person 2

Used to FILTER region_growing.py's candidate mask, not as the primary
detector (see toralis-branchseed-math.md, section 11): computes a Frangi-
style score per voxel based on Hessian eigenvalues, so calcified plaque,
bone, and vein voxels that survived intensity-based region growing can be
suppressed for not being tube-shaped.
"""

from __future__ import annotations

import numpy as np
import SimpleITK as sitk


def compute_hessian_eigenvalues(
    image: sitk.Image,
    sigma_mm: float,
) -> np.ndarray:
    """
    Smooth `image` at scale `sigma_mm` and compute the 3x3 Hessian matrix
    at every voxel, returning its sorted eigenvalues (by |value|) as an
    array of shape (Z, Y, X, 3).
    """
    raise NotImplementedError


def frangi_score(eigenvalues: np.ndarray, alpha: float, beta: float, c: float) -> np.ndarray:
    """
    Combine sorted Hessian eigenvalues (lambda1, lambda2, lambda3) into a
    single 0-to-1 "tube-likeness" score per voxel using the Frangi formula.
    """
    raise NotImplementedError


def compute_vesselness(
    image: sitk.Image,
    roi_mask: sitk.Image,
    scales_mm: list[float],
) -> np.ndarray:
    """
    Multi-scale vesselness: compute frangi_score at each scale in
    `scales_mm`, restricted to voxels inside `roi_mask`, and take the
    per-voxel maximum across scales. Returns a vesselness map aligned with
    `image`'s array shape.
    """
    raise NotImplementedError


def filter_candidates_by_vesselness(
    candidate_mask: sitk.Image,
    vesselness_map: np.ndarray,
    min_score: float,
) -> sitk.Image:
    """
    Suppress voxels in `candidate_mask` whose vesselness score is below
    `min_score` -- used to strip non-tube-shaped false positives (plaque,
    bone, vein) from region_growing.py's output.
    """
    raise NotImplementedError
