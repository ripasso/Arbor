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
    raise NotImplementedError


def find_contact_centroid(contact_voxel_indices: np.ndarray) -> tuple[float, float, float]:
    """
    Average the contact zone's voxel indices to get a sub-voxel-precision
    centroid -- this is what gets converted to mm and reported as
    ostium_xyz_mm (drives the 25%-weighted ostium localisation score).
    """
    raise NotImplementedError


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
    raise NotImplementedError


def deduplicate_ostia(candidates: list[OstiumCandidate]) -> list[OstiumCandidate]:
    """
    Apply the spec's merge/split rules:
    - Distinct connected components at the wall stay distinct even if
      physically close together.
    - A component that touches the wall at one contact zone and only
      splits downstream is ONE instance, not multiple.
    Returns the final deduplicated, crop-face-filtered candidate list.
    """
    raise NotImplementedError
