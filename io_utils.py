"""
Stage A — Robust loading.

Owner: Person 1

Responsibilities:
- Load CT image + aorta mask, auto-detecting gzip-disguised files regardless
  of the file extension (confirmed real issue: subjects 016-025 in the
  provided dataset are gzip data saved with a plain .nii extension).
- Verify image and mask share the same grid (size/spacing/origin/direction).
- Provide the voxel-index -> physical-mm conversion used by every other
  module (never report raw voxel indices in final output).
"""

from __future__ import annotations

from dataclasses import dataclass

import SimpleITK as sitk


@dataclass
class Case:
    """A loaded image/mask pair, ready for the pipeline."""
    image: sitk.Image
    mask: sitk.Image
    case_id: str


def is_gzip(file_path: str) -> bool:
    """
    Sniff the first two bytes of a file to detect gzip compression,
    regardless of its extension (magic bytes: 0x1f 0x8b).
    """
    raise NotImplementedError


def load_image_any_extension(file_path: str) -> sitk.Image:
    """
    Load a .nii / .nii.gz / gzip-disguised-as-.nii file via SimpleITK,
    decompressing to a temp file with a recognized extension if needed.
    """
    raise NotImplementedError


def load_case(image_path: str, mask_path: str, case_id: str) -> Case:
    """
    Load an image + mask pair and verify they share the same physical grid
    (size, spacing, origin, direction). Raise a clear error if they don't.
    """
    raise NotImplementedError


def voxel_to_mm(image: sitk.Image, index: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Convert a voxel index (i, j, k) to physical mm coordinates (x, y, z)
    using the image's own spacing/origin/direction.

    Thin wrapper over image.TransformIndexToPhysicalPoint(index).
    """
    raise NotImplementedError


def get_spacing_mm(image: sitk.Image) -> tuple[float, float, float]:
    """Return the per-axis voxel spacing in mm for this image."""
    raise NotImplementedError
