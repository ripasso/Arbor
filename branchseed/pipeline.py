"""End-to-end case processing: geometry, detection, and the record the
visualisations are built from."""

from __future__ import annotations

import time

import numpy as np
from scipy import ndimage as ndi

import aorta as ao
import branches as br
from bsio import Volume, load_case

# Filters. The challenge fixes the 5 mm rule and leaves the minimum origin size
# to the final dataset, so the size floors below are conservative defaults.
MIN_REACH_MM = 5.0
MIN_SEED_RADIUS_MM = 0.65
MIN_OSTIUM_RADIUS_MM = 0.85
CAP_ALIGNMENT = 0.62
CAP_END_MM = 6.0


def _ramp(value, low, high):
    return float(np.clip((value - low) / max(high - low, 1e-6), 0.0, 1.0))


def _confidence(cand, stats):
    """A blunt 0-1 score used to shade the visualisations, not to filter.

    It rewards a branch that is bright like the aorta, leaves the wall cleanly,
    can be followed a decent distance, and has an opening in proportion to the
    lumen it feeds.
    """
    span = max(stats["lumen_median"] - stats["hu_low"], 1.0)
    contrast = _ramp((cand["path_hu_median"] - stats["hu_low"]) / span, 0.0, 0.55)
    reach = _ramp(cand["reach_mm"], 5.0, 16.0)
    clearance = _ramp(cand["seed_wall_clearance_mm"], 0.8, 3.0)
    proportion = 1.0 - _ramp(cand["footprint_ratio"], 2.2, 4.2)
    size = _ramp(cand["radius_mm"], 0.6, 1.6)
    score = (0.30 * contrast + 0.22 * reach + 0.18 * clearance
             + 0.18 * proportion + 0.12 * size)
    return round(float(np.clip(score, 0.0, 1.0)), 3)


def _anatomical_axes(affine: np.ndarray, spacing: np.ndarray):
    """Anterior and patient-left directions expressed in scaled-voxel mm."""
    rot = np.asarray(affine)[:3, :3]
    inv = np.linalg.inv(rot)
    anterior = inv @ np.array([0.0, 1.0, 0.0])   # RAS +y
    left = inv @ np.array([-1.0, 0.0, 0.0])      # RAS -x
    superior = inv @ np.array([0.0, 0.0, 1.0])   # RAS +z
    out = []
    for vec in (anterior, left, superior):
        v = vec * spacing
        n = np.linalg.norm(v)
        out.append(v / n if n > 1e-9 else v)
    return out


def process_case(image_path: str, mask_path: str, case_id: str,
                 want_maps: bool = True):
    t0 = time.time()
    vol, aorta_full = load_case(image_path, mask_path)
    aorta_full = ao.clean_mask(aorta_full)
    if aorta_full.sum() < 50:
        raise RuntimeError("aorta mask is empty after cleanup")

    sl, lo = ao.roi_bounds(aorta_full, vol.spacing, margin_mm=31.0)
    ct = np.ascontiguousarray(vol.data[sl])
    aorta = np.ascontiguousarray(aorta_full[sl])
    spacing = vol.spacing
    offset_vox = lo.astype(float)

    anterior, left, superior = _anatomical_axes(vol.affine, spacing)

    points_mm, arc_mm = ao.centerline(aorta, spacing, step_mm=1.0)

    # run the centerline superior -> inferior so arc length reads like a ruler
    if float(np.dot(points_mm[-1] - points_mm[0], superior)) > 0:
        points_mm = points_mm[::-1].copy()
        arc_mm = (arc_mm[-1] - arc_mm[::-1]).copy()

    tangents, u, v = ao.anatomical_frames(points_mm, anterior, left)

    found, stats = br.detect(ct, aorta, spacing, points_mm, arc_mm, tangents, u, v)

    calcium_ceiling = stats.get("lumen_p98", stats["lumen_median"]) + 110.0

    daughters, rejected, extras = [], [], []
    for cand in found:
        reason = None
        if cand["reach_mm"] < MIN_REACH_MM:
            reason = "shorter than 5 mm beyond the wall"
        elif cand["straight_mm"] < 2.6 or cand["seed_wall_clearance_mm"] < 1.0:
            reason = "creeps along the wall instead of leaving it"
        elif cand["radius_mm"] < MIN_SEED_RADIUS_MM:
            reason = "lumen too small at the seed"
        elif cand["ostium_radius_mm"] < MIN_OSTIUM_RADIUS_MM:
            reason = "origin below the minimum size"
        elif cand["footprint_ratio"] > 4.2:
            reason = "wall-hugging sheet, not an opening"
        elif cand["path_hu_median"] > calcium_ceiling:
            reason = "too bright for contrast, reads as calcium or bone"
        elif (cand["axial_alignment"] > CAP_ALIGNMENT
              and cand["distance_to_end_mm"] < CAP_END_MM):
            reason = "flat cropped end of the supplied segment"

        is_terminal = (
            reason is None
            and cand["ostium_radius_mm"] > 0.50 * max(cand["parent_radius_mm"], 1e-6)
            and cand["radius_mm"] > 0.42 * max(cand["parent_radius_mm"], 1e-6)
            and cand["distance_to_end_mm"] < 22.0
            and cand["axial_alignment"] > 0.45
        )

        if reason is not None:
            cand["reject_reason"] = reason
            rejected.append(cand)
        elif is_terminal:
            extras.append(cand)
        else:
            daughters.append(cand)

    # order by position along the aorta, superior first
    daughters.sort(key=lambda c: c["arc_mm"])
    extras.sort(key=lambda c: c["arc_mm"])

    for c in daughters + extras:
        c["confidence"] = _confidence(c, stats)

    unwrap = None
    if want_maps:
        unwrap = ao.unwrap_surface(ct, aorta, spacing, points_mm, arc_mm,
                                   tangents, u, v,
                                   hu_low=stats["hu_low"])

    aorta_edt = ndi.distance_transform_edt(aorta, sampling=spacing)
    radius_profile = ndi.map_coordinates(
        aorta_edt, np.moveaxis(points_mm / spacing, -1, 0)[:, :, None],
        order=1, mode="nearest")[:, 0]
    # a smoothed centerline can stray a whisker outside a tortuous lumen, which
    # would otherwise read as a zero-calibre aorta
    radius_profile = ndi.maximum_filter1d(radius_profile, size=5, mode="nearest")

    def to_lps(point_mm):
        return vol.index_to_physical(point_mm / spacing + offset_vox)

    def dir_to_lps(vec_mm):
        return vol.direction_to_physical(vec_mm / spacing)

    for cand in daughters + extras + rejected:
        cand["ostium_xyz_mm"] = to_lps(cand["ostium_mm"])
        cand["seed_xyz_mm"] = to_lps(cand["seed_mm"])
        cand["direction_xyz"] = dir_to_lps(cand["direction_mm"])

    elapsed = time.time() - t0
    return dict(
        case_id=case_id,
        volume=vol,
        aorta=aorta,
        ct=ct,
        spacing=spacing,
        offset_vox=offset_vox,
        slices=sl,
        points_mm=points_mm,
        arc_mm=arc_mm,
        tangents=tangents,
        u=u, v=v,
        radius_profile=radius_profile,
        centerline_lps=np.array([to_lps(p) for p in points_mm]),
        daughters=daughters,
        extras=extras,
        rejected=rejected,
        unwrap=unwrap,
        stats=stats,
        seconds=elapsed,
        superior_mm=superior,
    )


def to_prediction(record) -> dict:
    """The machine-readable output the challenge asks for."""
    out = {
        "case_id": record["case_id"],
        "parent": {"instance_id": "aorta"},
        "daughters": [],
    }
    for i, c in enumerate(record["daughters"], start=1):
        out["daughters"].append({
            "instance_id": f"branch_{i:03d}",
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": [round(float(x), 2) for x in c["ostium_xyz_mm"]],
            "seed_xyz_mm": [round(float(x), 2) for x in c["seed_xyz_mm"]],
            "radius_mm": round(float(c["radius_mm"]), 2),
            "direction_xyz": [round(float(x), 4) for x in c["direction_xyz"]],
        })
    return out
