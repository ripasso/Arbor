"""
Stage B — Define the search region (search shell).

Owner: Person 2

Responsibilities:
- Dilate the given aorta mask by a small physical margin to build a thin
  "search shell" around the aortic wall -- the only place branches can
  plausibly originate.
- Keep this fast: restrict all downstream expensive computation (vesselness,
  region growing) to this shell + a bit of outward margin, not the full CT
  volume, to help hit the ~60s/case runtime budget.
"""

from __future__ import annotations

import SimpleITK as sitk


def build_search_shell(mask: sitk.Image, margin_mm: float) -> sitk.Image:
    """
    Dilate `mask` by `margin_mm` (converted to voxels using the image's own
    spacing) and subtract the original mask, returning a binary shell image
    covering just the region right around the aorta's outer surface.
    """
    raise NotImplementedError


def crop_to_roi(image: sitk.Image, shell_mask: sitk.Image, pad_mm: float = 0.0) -> sitk.Image:
    """
    Crop `image` to the bounding box of `shell_mask` (with optional extra
    padding in mm), to shrink the volume that later stages operate on.
    """
    raise NotImplementedError
