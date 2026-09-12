"""Daughter-branch detection.

Fully classical, CPU-only, and built around one idea: grow *outward*.

1. Work out what contrast looks like in this particular scan from the aortic
   lumen itself. The dev set runs from roughly 85 HU to 580 HU in the aorta, so
   a fixed threshold is useless.
2. Label every patch where bright tissue touches the aortic wall. One patch is
   one ostium, which is exactly the rule the challenge states: two origins count
   separately only when they are separate at the wall, and a common trunk counts
   once however soon it divides.
3. Propagate those labels outward, shell by shell, in order of distance from the
   aortic surface. Because a voxel can only ever inherit a label from something
   closer to the aorta than itself, two branches can never fuse into one blob and
   a leak into neighbouring tissue stays attached to the origin it came from
   instead of swallowing the whole abdomen. That last property is what makes a
   permissive threshold safe.
4. Walk each label outward, shell by shell, stopping at the first bifurcation or
   at 10 mm, and read the seed, direction and radius off that proximal path.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

CONN18 = ndi.generate_binary_structure(3, 2)
CONN26 = ndi.generate_binary_structure(3, 3)
CONN6 = ndi.generate_binary_structure(3, 1)


# --------------------------------------------------------------------------- #
# contrast model
# --------------------------------------------------------------------------- #

def contrast_window(ct: np.ndarray, aorta: np.ndarray, spacing: np.ndarray):
    """Lower and upper HU bounds taken from the aortic lumen in this scan."""
    interior = ndi.binary_erosion(aorta, CONN6, iterations=2)
    if interior.sum() < 40:
        interior = aorta
    lumen = ct[interior]
    med = float(np.median(lumen))
    p95 = float(np.percentile(lumen, 95))
    p98 = float(np.percentile(lumen, 98))

    near = ndi.binary_dilation(aorta, CONN6, iterations=max(int(6 / spacing.min()), 3))
    surround = ct[near & ~aorta]
    bg = float(np.median(surround)) if surround.size else 0.0

    high = max(p95 * 1.35 + 150.0, 700.0)
    floor = max(bg + 50.0, 80.0)
    low = _wall_flood_threshold(ct, aorta, spacing, med, high, floor)
    return dict(hu_low=float(low), hu_high=float(high), lumen_median=med,
                lumen_p98=p98, background=bg)


def _wall_flood_threshold(ct, aorta, spacing, lumen_med, hu_high, floor,
                          max_patch_frac=0.12, min_patch_mm2=2.2):
    """Choose the threshold that resolves the most separate wall openings.

    Too high and faint branches never touch the wall at all; too low and the
    adjacent liver, bowel and muscle join in until the footprints fuse into one
    sheet wrapped round the aorta, which yields nothing either. Between those
    two failure modes there is a plateau where the number of distinct,
    plausibly sized openings peaks, and that is the operating point taken here.
    Any opening covering more than about an eighth of the wall is treated as
    flooding rather than as an origin.
    """
    outside = ndi.distance_transform_edt(~aorta, sampling=spacing)
    shell = (~aorta) & (outside <= float(np.min(spacing)) * 1.8)
    total = int(shell.sum())
    if total == 0:
        return max(0.55 * lumen_med, floor)

    face = float(np.sort(spacing)[0] * np.sort(spacing)[1])
    ladder = np.arange(0.95, 0.38, -0.03) * lumen_med
    ladder = ladder[ladder >= floor]
    if ladder.size == 0:
        return floor

    best_t, best_score = float(ladder[0]), -1
    for t in ladder:
        mask = shell & (ct >= t) & (ct <= hu_high)
        if not mask.any():
            continue
        lbl, n = ndi.label(mask, structure=CONN26)
        if n == 0:
            continue
        sizes = np.bincount(lbl.ravel())[1:]
        if (sizes.max() / total) > max_patch_frac:
            break
        score = int(((sizes * face) >= min_patch_mm2).sum())
        if score >= best_score:      # ties resolve toward the lower threshold
            best_score, best_t = score, float(t)
    return best_t


# --------------------------------------------------------------------------- #
# outward labelled growth
# --------------------------------------------------------------------------- #

def outward_growth(ct, aorta, spacing, window, reach_mm=20.0,
                   min_footprint_mm2=2.2):
    """Label bright tissue outside the aorta, rooted at its wall footprints."""
    outside = ndi.distance_transform_edt(~aorta, sampling=spacing)
    band = (~aorta) & (outside <= reach_mm)
    candidate = band & (ct >= window["hu_low"]) & (ct <= window["hu_high"])

    contact = candidate & ndi.binary_dilation(aorta, CONN18)
    seeds, n = ndi.label(contact, structure=CONN26)
    if n == 0:
        return np.zeros_like(seeds), outside, []

    face = float(np.sort(spacing)[0] * np.sort(spacing)[1])
    sizes = np.bincount(seeds.ravel())
    keep = np.array([0] + [i if sizes[i] * face >= min_footprint_mm2 else 0
                           for i in range(1, n + 1)])
    seeds = keep[seeds]

    labels = seeds.astype(np.int32)
    step = float(max(0.5, np.min(spacing) * 0.9))
    r = float(np.min(spacing))
    while r <= reach_mm:
        target = candidate & (labels == 0) & (outside <= r + step)
        if target.any():
            for _ in range(2):
                if not target.any():
                    break
                spread = ndi.maximum_filter(labels, size=3, mode="nearest")
                fill = target & (spread > 0)
                if not fill.any():
                    break
                labels = np.where(fill, spread, labels)
                target = target & ~fill
        r += step

    present = [int(i) for i in np.unique(labels) if i > 0]
    return labels, outside, present


# --------------------------------------------------------------------------- #
# proximal tracing
# --------------------------------------------------------------------------- #

def trace_outward(component, outside, spacing, start_mask, max_path_mm=10.0):
    """Follow one branch outward, shell by shell, accumulating path length."""
    step = float(max(0.5, np.min(spacing) * 0.9))
    idx = np.array(np.nonzero(start_mask), dtype=float)
    here = idx.mean(axis=1) * spacing
    r0 = float(outside[start_mask].mean())

    path = [(0.0, here)]
    prev = start_mask
    travelled = 0.0
    bifurcation = None

    r = r0 + step
    while travelled < max_path_mm + 1e-6:
        shell = component & (outside >= r - step * 0.5) & (outside < r + step * 0.5)
        if not shell.any():
            break
        lbl, n = ndi.label(shell, structure=CONN26)
        touching = [int(t) for t in np.unique(lbl[ndi.binary_dilation(prev, CONN26) & shell])
                    if t > 0]
        if not touching:
            break
        sizes = {t: int((lbl == t).sum()) for t in touching}
        biggest = max(sizes, key=sizes.get)
        if len(touching) > 1:
            second = sorted(sizes.values())[-2]
            if second >= max(3, 0.45 * sizes[biggest]):
                bifurcation = travelled
                break
        blob = lbl == biggest
        pts = np.array(np.nonzero(blob), dtype=float).mean(axis=1) * spacing
        travelled += float(np.linalg.norm(pts - path[-1][1]))
        path.append((travelled, pts))
        prev = blob
        r += step

    return path, bifurcation, r - r0


# --------------------------------------------------------------------------- #
# radius
# --------------------------------------------------------------------------- #

def measure_radius(ct, vessel, seed_mm, direction_mm, spacing, hu_low,
                   half_extent_mm=7.0, step_mm=0.25):
    """Cross-sectional radius at the seed, measured sub-voxel.

    A plane is cut perpendicular to the branch and the lumen is outlined at the
    half maximum between the local background and the vessel's own peak. That is
    the usual way to size a small vessel and it avoids the half-voxel staircase
    a distance transform gives on a 1.5 mm scan.
    """
    d = direction_mm / max(np.linalg.norm(direction_mm), 1e-9)
    helper = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(d, helper)
    e1 /= max(np.linalg.norm(e1), 1e-9)
    e2 = np.cross(d, e1)

    n = int(half_extent_mm / step_mm)
    grid = (np.arange(-n, n + 1)) * step_mm
    gx, gy = np.meshgrid(grid, grid, indexing="ij")
    pts = seed_mm[None, None, :] + gx[..., None] * e1 + gy[..., None] * e2
    coords = np.moveaxis(pts / spacing, -1, 0)

    hu = ndi.map_coordinates(ct, coords, order=1, mode="constant", cval=-1000.0)
    inside = ndi.map_coordinates(vessel.astype(np.float32), coords, order=1,
                                 mode="constant", cval=0.0)

    centre = (n, n)
    peak = float(np.median(hu[n - 2:n + 3, n - 2:n + 3]))
    ring = np.hypot(gx, gy) > half_extent_mm * 0.78
    background = float(np.median(hu[ring])) if ring.any() else 0.0
    level = max(0.5 * (peak + background), hu_low * 0.75)

    binary = (hu >= level) & (inside > 0.25)
    if not binary[centre]:
        binary = inside > 0.5
        if not binary[centre]:
            return None
    lbl, _ = ndi.label(binary)
    blob = lbl == lbl[centre]
    area = float(blob.sum()) * step_mm * step_mm
    dist = ndi.distance_transform_edt(blob, sampling=(step_mm, step_mm))
    return float(min(np.sqrt(area / np.pi), max(float(dist[centre]) * 1.6, 0.4)))


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #

def detect(ct, aorta, spacing, points_mm, arc_mm, tangents, u, v, merge_mm=2.6):
    window = contrast_window(ct, aorta, spacing)
    labels, outside, present = outward_growth(ct, aorta, spacing, window)
    if not present:
        return [], window

    vessel = (labels > 0) | aorta
    aorta_edt = ndi.distance_transform_edt(aorta, sampling=spacing)
    contact = (labels > 0) & ndi.binary_dilation(aorta, CONN18)

    # merge footprints that sit almost on top of each other
    centroids = {}
    for k in present:
        patch = contact & (labels == k)
        if not patch.any():
            patch = labels == k
        centroids[k] = np.array(np.nonzero(patch), dtype=float).mean(axis=1) * spacing
    groups = {}
    for k in sorted(present, key=lambda k: -int((labels == k).sum())):
        target = k
        for g in groups:
            if np.linalg.norm(centroids[k] - centroids[g]) < merge_mm:
                target = g
                break
        groups.setdefault(target, []).append(k)

    results = []
    for root, members in groups.items():
        component = np.isin(labels, members)
        patch = component & ndi.binary_dilation(aorta, CONN18)
        if not patch.any():
            continue

        path, bifurcation, radial_reach = trace_outward(component, outside, spacing, patch)
        if len(path) < 2:
            continue

        dists = np.array([d for d, _ in path])
        pts = np.array([p for _, p in path])
        ostium_mm = pts[0]

        ostium_idx = np.array(np.nonzero(patch), dtype=float)
        area_mm2 = float(ostium_idx.shape[1] * np.sort(spacing)[0] * np.sort(spacing)[1])
        ostium_radius = float(np.sqrt(area_mm2 / np.pi))

        seed_target = min(5.0, float(dists[-1]))
        seed_mm = np.array([np.interp(seed_target, dists, pts[:, k]) for k in range(3)])

        near = pts[dists <= 6.5] if (dists <= 6.5).any() else pts[:2]
        centred = near - near.mean(axis=0)
        if len(near) >= 3:
            _, _, vt = np.linalg.svd(centred, full_matrices=False)
            direction = vt[0]
        else:
            direction = near[-1] - near[0]
        if np.dot(direction, seed_mm - ostium_mm) < 0:
            direction = -direction
        norm = np.linalg.norm(direction)
        direction = direction / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])

        radius = measure_radius(ct, vessel, seed_mm, direction, spacing, window["hu_low"])
        if radius is None:
            seed_vox = np.clip(np.round(seed_mm / spacing).astype(int), 0,
                               np.array(ct.shape) - 1)
            radius = float(ndi.distance_transform_edt(component, sampling=spacing)[tuple(seed_vox)])

        path_hu = ndi.map_coordinates(ct, np.moveaxis(pts / spacing, -1, 0),
                                      order=1, mode="nearest")
        clearance = float(ndi.map_coordinates(outside, (seed_mm / spacing)[:, None],
                                              order=1, mode="nearest")[0])

        s_mm, angle_deg, _, station = _project_one(ostium_mm, points_mm, tangents,
                                                   u, v, arc_mm)
        station_vox = np.clip(np.round(points_mm[station] / spacing).astype(int), 0,
                              np.array(ct.shape) - 1)
        parent_radius = float(aorta_edt[tuple(station_vox)])

        normal = _outward_normal(aorta_edt, ostium_mm, spacing)
        axial_alignment = abs(float(np.dot(normal, tangents[station])))
        near_end = min(s_mm - arc_mm[0], arc_mm[-1] - s_mm)

        results.append(dict(
            ostium_mm=ostium_mm,
            seed_mm=seed_mm,
            direction_mm=direction,
            radius_mm=float(radius),
            ostium_radius_mm=ostium_radius,
            ostium_area_mm2=area_mm2,
            reach_mm=float(dists[-1]),
            radial_reach_mm=float(radial_reach),
            traced_mm=float(dists[-1]),
            bifurcation_mm=bifurcation,
            arc_mm=float(s_mm),
            clock_deg=float(angle_deg),
            parent_radius_mm=parent_radius,
            axial_alignment=axial_alignment,
            distance_to_end_mm=float(near_end),
            seed_hu=float(path_hu[-1]),
            station=int(station),
            n_voxels=int(component.sum()),
            path_hu_median=float(np.median(path_hu)),
            path_hu_max=float(np.max(path_hu)),
            seed_wall_clearance_mm=clearance,
            straight_mm=float(np.linalg.norm(seed_mm - ostium_mm)),
            footprint_ratio=float(ostium_radius / max(radius, 0.35)),
            path_mm=pts,
        ))

    return results, window


def _project_one(point_mm, points_mm, tangents, u, v, arc_mm):
    d = point_mm[None, :] - points_mm
    along = np.einsum("sd,sd->s", d, tangents)
    perp = d - along[:, None] * tangents
    radial = np.linalg.norm(perp, axis=1)
    station = int(np.argmin(radial + np.abs(along) * 0.35))
    pu = float(perp[station] @ u[station])
    pv = float(perp[station] @ v[station])
    angle = np.degrees(np.arctan2(pv, pu)) % 360.0
    return float(arc_mm[station]), angle, float(radial[station]), station


def _outward_normal(aorta_edt, point_mm, spacing):
    idx = np.clip(np.round(point_mm / spacing).astype(int), 1,
                  np.array(aorta_edt.shape) - 2)
    g = np.array([
        aorta_edt[idx[0] + 1, idx[1], idx[2]] - aorta_edt[idx[0] - 1, idx[1], idx[2]],
        aorta_edt[idx[0], idx[1] + 1, idx[2]] - aorta_edt[idx[0], idx[1] - 1, idx[2]],
        aorta_edt[idx[0], idx[1], idx[2] + 1] - aorta_edt[idx[0], idx[1], idx[2] - 1],
    ]) / (2.0 * spacing)
    n = -g
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-9 else np.array([0.0, 0.0, 1.0])
