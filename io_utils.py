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

import gzip
import os
import tempfile
from dataclasses import dataclass

import numpy as np
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
    with open(file_path, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def load_image_any_extension(file_path: str) -> sitk.Image:
    """
    Load a .nii / .nii.gz / gzip-disguised-as-.nii file via SimpleITK,
    decompressing to a temp file with a recognized extension if needed.
    """
    if not is_gzip(file_path):
        return sitk.ReadImage(file_path)

    # Gzip data (possibly disguised with a plain .nii extension): SimpleITK
    # needs a real .nii.gz suffix, so decompress to a temp file.
    with gzip.open(file_path, "rb") as f:
        data = f.read()
    fd, tmp_path = tempfile.mkstemp(suffix=".nii")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return sitk.ReadImage(tmp_path)
    finally:
        os.remove(tmp_path)


def load_case(image_path: str, mask_path: str, case_id: str) -> Case:
    """
    Load an image + mask pair and verify they share the same physical grid
    (size, spacing, origin, direction). Raise a clear error if they don't.
    """
    image = load_image_any_extension(image_path)
    mask = load_image_any_extension(mask_path)

    if image.GetSize() != mask.GetSize():
        raise ValueError(f"size mismatch: image {image.GetSize()} vs mask {mask.GetSize()}")
    for name, a, b in (
        ("spacing", image.GetSpacing(), mask.GetSpacing()),
        ("origin", image.GetOrigin(), mask.GetOrigin()),
        ("direction", image.GetDirection(), mask.GetDirection()),
    ):
        if not np.allclose(a, b, atol=1e-4):
            raise ValueError(f"{name} mismatch: image {a} vs mask {b}")

    return Case(image=image, mask=mask, case_id=case_id)


def voxel_to_mm(image: sitk.Image, index: tuple[int, int, int]) -> tuple[float, float, float]:
    """
    Convert a voxel index (i, j, k) to physical mm coordinates (x, y, z)
    using the image's own spacing/origin/direction.

    Thin wrapper over image.TransformIndexToPhysicalPoint(index).
    """
    # Continuous version also accepts fractional (sub-voxel) indices,
    # e.g. contact-patch centroids.
    return tuple(image.TransformContinuousIndexToPhysicalPoint([float(v) for v in index]))


def get_spacing_mm(image: sitk.Image) -> tuple[float, float, float]:
    """Return the per-axis voxel spacing in mm for this image."""
    return tuple(image.GetSpacing())
