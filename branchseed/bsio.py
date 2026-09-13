"""Robust NIfTI reading for the Branchseed challenge.

The dev set ships some volumes gzip-compressed but still named ``.nii``, so the
loader sniffs the magic bytes instead of trusting the extension. Physical points
are reported in the LPS frame that SimpleITK returns from
``TransformIndexToPhysicalPoint``, which is what the challenge asks for.
"""

from __future__ import annotations

import gzip
import io as _io
from dataclasses import dataclass

import nibabel as nib
import numpy as np

# NIfTI affines are RAS; SimpleITK hands back LPS.
_RAS_TO_LPS = np.diag([-1.0, -1.0, 1.0])


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as fh:
        head = fh.read(2)
    if head == b"\x1f\x8b":
        with gzip.open(path, "rb") as fh:
            return fh.read()
    with open(path, "rb") as fh:
        return fh.read()


def load_nifti(path: str):
    """Load a single-file NIfTI whatever its extension or compression."""
    raw = _read_bytes(path)
    holder = nib.FileHolder(fileobj=_io.BytesIO(raw))
    file_map = {"header": holder, "image": holder}
    try:
        return nib.Nifti1Image.from_file_map(file_map)
    except Exception:
        return nib.Nifti2Image.from_file_map(file_map)


@dataclass
class Volume:
    """A CT volume plus the geometry needed to report physical millimetres."""

    data: np.ndarray
    affine: np.ndarray          # voxel index -> RAS mm
    spacing: np.ndarray         # mm per voxel along each array axis

    @property
    def shape(self):
        return self.data.shape

    def index_to_physical(self, idx) -> np.ndarray:
        """Voxel index (i, j, k), possibly fractional, to LPS millimetres."""
        idx = np.asarray(idx, dtype=float)
        single = idx.ndim == 1
        pts = np.atleast_2d(idx)
        homogeneous = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
        ras = homogeneous @ self.affine.T
        lps = ras[:, :3] @ _RAS_TO_LPS.T
        return lps[0] if single else lps

    def physical_to_index(self, point) -> np.ndarray:
        """LPS millimetres back to a fractional voxel index (i, j, k)."""
        point = np.asarray(point, dtype=float)
        single = point.ndim == 1
        pts = np.atleast_2d(point)
        ras = pts @ np.linalg.inv(_RAS_TO_LPS).T
        homogeneous = np.concatenate([ras, np.ones((len(ras), 1))], axis=1)
        idx = homogeneous @ np.linalg.inv(self.affine).T
        return idx[0, :3] if single else idx[:, :3]

    def direction_to_physical(self, vec) -> np.ndarray:
        """Rotate a voxel-space direction into LPS mm space and normalise it."""
        vec = np.asarray(vec, dtype=float)
        single = vec.ndim == 1
        vecs = np.atleast_2d(vec)
        ras = vecs @ self.affine[:3, :3].T
        lps = ras @ _RAS_TO_LPS.T
        norm = np.linalg.norm(lps, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        out = lps / norm
        return out[0] if single else out


def load_case(image_path: str, mask_path: str):
    """Return (ct Volume, boolean aorta mask) resampled to nothing, just read."""
    img = load_nifti(image_path)
    msk = load_nifti(mask_path)

    ct = np.asanyarray(img.dataobj).astype(np.float32)
    aorta = np.asanyarray(msk.dataobj) > 0

    if aorta.shape != ct.shape:
        raise ValueError(f"mask shape {aorta.shape} does not match image {ct.shape}")

    affine = np.asarray(img.affine, dtype=float)
    spacing = np.abs(np.asarray(img.header.get_zooms()[:3], dtype=float))
    spacing[spacing == 0] = 1.0

    return Volume(ct, affine, spacing), aorta
