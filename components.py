"""
Stage D — Candidate branch detection (connected components).

Owner: Person 3

Responsibilities:
- Group candidate vessel voxels (output of region_growing + vesselness
  filtering) into connected components (26-connectivity in 3D).
- Determine which components actually touch the aorta mask's outer
  surface -- only those are candidate direct daughters. Components that
  float disconnected from the aorta are discarded.

Can be developed against a synthetic/stubbed candidate_mask (e.g. a small
hand-made binary blob) before region_growing.py / vesselness.py are ready --
that's the parallelization trick described in the program outline doc.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import SimpleITK as sitk


@dataclass
class Component:
    """One connected component of candidate vessel voxels."""
    label: int
    voxel_indices: np.ndarray   # shape (N, 3), each row (i, j, k)
    touches_aorta: bool


def label_components(candidate_mask: sitk.Image) -> list[Component]:
    """
    Label `candidate_mask` into 26-connected components. Returns one
    Component per connected blob (touches_aorta not yet determined here).
    """
    raise NotImplementedError


def filter_touching_aorta(components: list[Component], aorta_mask: sitk.Image) -> list[Component]:
    """
    For each component, test whether any of its voxels are adjacent
    (26-connectivity) to a voxel in `aorta_mask`. Sets/filters by
    touches_aorta. Components that don't touch are dropped (not a direct
    daughter -- e.g. a branch-of-a-branch, or unrelated bright tissue).
    """
    raise NotImplementedError
