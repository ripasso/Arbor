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

import numpy as np
import SimpleITK as sitk

from components import Component


def extract_centerline(component: Component) -> np.ndarray:
    """
    Skeletonize `component`'s voxel mask and order the resulting skeleton
    voxels into a single path (ordered array of shape (N, 3)) starting
    from the end closest to the aorta wall.
    """
    raise NotImplementedError


def arc_length_mm(centerline: np.ndarray, spacing_mm: tuple[float, float, float]) -> np.ndarray:
    """
    Given an ordered centerline (voxel indices), return the cumulative
    physical arc length in mm at each point (index 0 = 0.0).
    """
    raise NotImplementedError


def check_eligibility(
    centerline: np.ndarray,
    spacing_mm: tuple[float, float, float],
    min_trace_mm: float,
) -> bool:
    """
    Return True if the centerline's total arc length reaches at least
    `min_trace_mm` (the 5mm rule) before fading out or ending.
    """
    raise NotImplementedError


def find_first_bifurcation(component: Component, centerline: np.ndarray) -> int | None:
    """
    Return the index (along `centerline`) of the first skeleton voxel with
    >=3 skeleton neighbours (a bifurcation point), or None if the segment
    has no bifurcation within the traced region.
    """
    raise NotImplementedError


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
    raise NotImplementedError
