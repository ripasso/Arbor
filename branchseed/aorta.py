"""Aortic geometry: mask cleanup, centerline, rotation-minimising frames, unwrap.

Everything here works in *scaled voxel space* (voxel index multiplied by the
voxel size in mm), so Euclidean distances are millimetres even when the volume
is anisotropic. Conversion to the reported LPS frame happens in ``bsio``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.interpolate import splev, splprep
from skimage.graph import MCP_Geometric


# --------------------------------------------------------------------------- #
# mask cleanup
# --------------------------------------------------------------------------- #

def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Keep the largest connected component and close interior holes."""
    lbl, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 3))
    if n > 1:
        sizes = ndi.sum(mask, lbl, index=np.arange(1, n + 1))
        mask = lbl == (int(np.argmax(sizes)) + 1)
    return ndi.binary_fill_holes(mask)


def roi_bounds(mask: np.ndarray, spacing: np.ndarray, margin_mm: float = 28.0):
    """Bounding box around the mask, padded by a physical margin."""
    idx = np.array(np.nonzero(mask))
    lo = idx.min(axis=1)
    hi = idx.max(axis=1) + 1
    pad = np.ceil(margin_mm / spacing).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.array(mask.shape))
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi)), lo


# --------------------------------------------------------------------------- #
# centerline
# --------------------------------------------------------------------------- #

def _farthest_point(mask: np.ndarray, spacing: np.ndarray, seed) -> tuple:
    costs = np.where(mask, 1.0, np.inf)
    mcp = MCP_Geometric(costs, sampling=tuple(float(s) for s in spacing))
    dist, _ = mcp.find_costs([tuple(int(v) for v in seed)])
    dist = np.where(mask, dist, -np.inf)
    return np.unravel_index(int(np.argmax(dist)), mask.shape)


def centerline(mask: np.ndarray, spacing: np.ndarray, step_mm: float = 1.0):
    """Geodesic centerline through the lumen, resampled every ``step_mm``.

    Endpoints come from a double farthest-point sweep, which finds the two ends
    of the tube without assuming the aorta runs along any particular axis. The
    path itself is traced through a cost field that strongly prefers the lumen
    centre, so it does not hug the wall on curves.
    """
    edt = ndi.distance_transform_edt(mask, sampling=spacing)
    seed = np.unravel_index(int(np.argmax(edt)), mask.shape)

    a = _farthest_point(mask, spacing, seed)
    b = _farthest_point(mask, spacing, a)

    centred = edt / max(edt.max(), 1e-6)
    costs = np.where(mask, 1.0 / (centred + 0.08) ** 2, np.inf)
    mcp = MCP_Geometric(costs, sampling=tuple(float(s) for s in spacing))
    mcp.find_costs([tuple(int(v) for v in a)])
    path = np.array(mcp.traceback(tuple(int(v) for v in b)), dtype=float)

    if len(path) < 4:
        raise RuntimeError("aorta mask too small to trace a centerline")

    # orient superior -> inferior where the affine allows it; otherwise keep
    # the traced order. The caller only needs a consistent direction.
    pts_mm = path * spacing

    # smooth, then resample at a uniform arc-length step
    seg = np.linalg.norm(np.diff(pts_mm, axis=0), axis=1)
    total = float(seg.sum())
    smooth = max(total * 0.6, 1.0)
    k = 3 if len(path) > 3 else 1
    try:
        tck, _ = splprep(pts_mm.T, s=smooth, k=k)
        u = np.linspace(0, 1, max(int(total / step_mm) * 4, 16))
        dense = np.array(splev(u, tck)).T
    except Exception:
        dense = pts_mm

    seg = np.linalg.norm(np.diff(dense, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(int(arc[-1] / step_mm) + 1, 4)
    targets = np.linspace(0, arc[-1], n)
    resampled = np.stack([np.interp(targets, arc, dense[:, d]) for d in range(3)], axis=1)
    return resampled, targets  # mm-space points, arc length in mm


def anatomical_frames(points_mm: np.ndarray, anterior_mm: np.ndarray,
                      left_mm: np.ndarray):
    """Frame anchored to the patient, not transported along the curve.

    At every station the reference direction is the patient's anterior
    direction projected into the cross-sectional plane, so angle 0 is 12
    o'clock for every station and every patient. That is what lets the
    unwrapped maps be laid on top of each other across a cohort; a transported
    frame would drift and make the same clock position mean different things in
    different scans.
    """
    tangents = np.gradient(points_mm, axis=0)
    if len(points_mm) > 7:
        tangents = ndi.gaussian_filter1d(tangents, sigma=2.0, axis=0, mode="nearest")
    tangents /= np.maximum(np.linalg.norm(tangents, axis=1, keepdims=True), 1e-9)

    u = anterior_mm[None, :] - (tangents @ anterior_mm)[:, None] * tangents
    bad = np.linalg.norm(u, axis=1) < 1e-3
    if bad.any():
        fallback = left_mm[None, :] - (tangents @ left_mm)[:, None] * tangents
        u[bad] = fallback[bad]
    u /= np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)

    v = np.cross(tangents, u)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    flip = (v @ left_mm) < 0
    v[flip] = -v[flip]
    return tangents, u, v


# --------------------------------------------------------------------------- #
# surface unwrapping
# --------------------------------------------------------------------------- #

@dataclass
class Unwrap:
    arc_mm: np.ndarray        # (S,)   arc length along the centerline
    angles_deg: np.ndarray    # (A,)   0 = anterior, increasing clockwise
    wall_mm: np.ndarray       # (S, A) lumen radius at the wall
    shell_hu: np.ndarray      # (S, A) brightest CT value just outside the wall
    shell_depth_mm: np.ndarray  # (S, A) how far that bright tissue extends
    points_mm: np.ndarray     # (S, 3) centerline in scaled-voxel mm
    tangents: np.ndarray
    u: np.ndarray
    v: np.ndarray


def unwrap_surface(ct: np.ndarray, mask: np.ndarray, spacing: np.ndarray,
                   points_mm: np.ndarray, arc_mm: np.ndarray,
                   tangents, u, v,
                   n_angles: int = 180, max_radius_mm: float = 30.0,
                   shell_mm: float = 9.0, hu_low: float = 100.0) -> Unwrap:
    """Flatten the aortic wall into an arc-length by clock-angle image."""
    angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)
    cos_a, sin_a = np.cos(angles), np.sin(angles)

    # ray directions (S, A, 3) in mm space
    dirs = (u[:, None, :] * cos_a[None, :, None] +
            v[:, None, :] * sin_a[None, :, None])

    # --- find the wall by walking outward through the mask -----------------
    r_step = 0.4
    radii = np.arange(0.0, max_radius_mm + r_step, r_step)
    samples_mm = points_mm[:, None, None, :] + dirs[:, :, None, :] * radii[None, None, :, None]
    idx = samples_mm / spacing
    coords = np.moveaxis(idx, -1, 0)

    inside = ndi.map_coordinates(mask.astype(np.float32), coords, order=1,
                                 mode="constant", cval=0.0) > 0.5
    # radius = first sample that is outside, walking out from the centre
    outside_run = np.cumsum(~inside, axis=-1)
    wall_idx = np.argmax(outside_run > 0, axis=-1)
    wall_idx[outside_run[..., -1] == 0] = len(radii) - 1
    wall = radii[wall_idx]

    # --- sample the CT in a shell just beyond the wall ---------------------
    depths = np.arange(0.6, shell_mm + 0.6, 0.6)
    shell_pts = (points_mm[:, None, None, :] +
                 dirs[:, :, None, :] * (wall[:, :, None, None] + depths[None, None, :, None]))
    shell_idx = np.moveaxis(shell_pts / spacing, -1, 0)
    hu = ndi.map_coordinates(ct, shell_idx, order=1, mode="constant", cval=-1000.0)

    bright = hu >= hu_low
    run = np.cumprod(bright, axis=-1)          # contiguous brightness from the wall out
    depth = run.sum(axis=-1) * 0.6
    shell_hu = np.where(bright.any(axis=-1), hu.max(axis=-1), hu[..., 0])

    return Unwrap(arc_mm=arc_mm,
                  angles_deg=np.degrees(angles),
                  wall_mm=wall,
                  shell_hu=shell_hu.astype(np.float32),
                  shell_depth_mm=depth.astype(np.float32),
                  points_mm=points_mm,
                  tangents=tangents, u=u, v=v)


def project_to_centerline(points_mm: np.ndarray, targets_mm: np.ndarray,
                          tangents, u, v, arc_mm: np.ndarray):
    """Map arbitrary mm-space points to (arc length, clock angle, radius)."""
    d = targets_mm[:, None, :] - points_mm[None, :, :]
    along = np.einsum("tsd,sd->ts", d, tangents)
    perp = d - along[:, :, None] * tangents[None, :, :]
    radial = np.linalg.norm(perp, axis=2)
    # nearest centerline station, penalising points far along the tangent
    cost = radial + np.abs(along) * 0.35
    station = np.argmin(cost, axis=1)

    rows = np.arange(len(targets_mm))
    pu = np.einsum("td,td->t", perp[rows, station], u[station])
    pv = np.einsum("td,td->t", perp[rows, station], v[station])
    angle = np.degrees(np.arctan2(pv, pu)) % 360.0
    return arc_mm[station], angle, radial[rows, station], station
