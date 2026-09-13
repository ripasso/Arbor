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

    p25 = float(np.percentile(lumen, 25))
    p75 = float(np.percentile(lumen, 75))
    bg90 = float(np.percentile(surround, 90)) if surround.size else bg

    high = max(p95 * 1.35 + 150.0, 700.0)
    floor = max(bg + 50.0, 80.0)
    low = _wall_flood_threshold(ct, aorta, spacing, med, high, floor)

    # Is this actually an angiogram? The whole method rests on a daughter lumen
    # being brighter than the tissue it runs through, which is only true when
    # the parent is opacified. The test is the lower quartile of the aortic
    # lumen against an absolute floor: iodinated arterial blood sits far above
    # 150 HU, unopacified blood sits near 40. Across the development set this
    # reads 194 HU or higher on twenty-three cases and 50 and 18 on the two
    # that are not contrast studies at all, so the two groups do not overlap.
    # Comparing against the local background instead fails here, because the
    # vertebral body and other opacified vessels sit in that background and
    # drag it up on perfectly good scans.
    return dict(hu_low=float(low), hu_high=float(high), lumen_median=med,
                lumen_p25=p25, lumen_p75=p75, lumen_p98=p98,
                background=bg, background_p90=bg90,
                opacification_hu=float(p25),
                contrast_margin=float(p25 - bg90),
                contrast_ok=bool(p25 >= 150.0))


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

def opening_centre(patch, band, spacing):
    """The point furthest from the rim of a wall footprint, not its centroid.

    Tahoces et al. (Med Biol Eng Comput 2020) take the contact point as the
    voxel that maximises distance to the edge of the contact area. That matters
    because an opening cut obliquely through the wall is often crescent shaped,
    and the centroid of a crescent can sit on its rim or outside it altogether,
    which puts the reported ostium off the lumen.

    The distance has to be measured *along the wall*. The contact patch is a
    shell barely one voxel thick, so an ordinary distance transform of it is
    half a voxel everywhere and its maximum is wherever ties happen to break.
    Distance to the nearest wall voxel that is not part of the patch gives the
    intended quantity.
    """
    sl = ndi.find_objects(patch.astype(np.uint8))[0]
    pad = tuple(slice(max(s.start - 3, 0), min(s.stop + 3, patch.shape[i]))
                for i, s in enumerate(sl))
    local = patch[pad]
    rim = band[pad] & ~local
    if not rim.any():
        idx = np.array(np.nonzero(local), dtype=float).mean(axis=1)
    else:
        depth = ndi.distance_transform_edt(~rim, sampling=spacing)
        depth = np.where(local, depth, -1.0)
        idx = np.array(np.unravel_index(int(np.argmax(depth)), local.shape), dtype=float)
    idx += np.array([s.start for s in pad], dtype=float)
    return idx * spacing


def peel_dead_ends(labels, outside, spacing, present):
    """Drop the parts of a label that lead nowhere.

    Danilov et al. (Computation 2016) clean vessel masks near the aortic border
    by walking distance layers inward and deleting any voxel with no neighbour
    one layer further out. Applied per label, with each label's own farthest
    layer as the starting point, this trims the blobs that cling to the wall
    beside a real branch without shortening the branch itself. The point is not
    the count, which the reach test already handles, but the region the radius
    and direction are then measured from.
    """
    step = float(max(0.5, np.min(spacing) * 0.9))
    cleaned = labels.copy()
    for k in present:
        blob = labels == k
        if not blob.any():
            continue
        far = float(outside[blob].max())
        keep = blob & (outside >= far - step)
        if not keep.any():
            continue
        d = far - step
        while d > 0:
            layer = blob & (outside >= d - step) & (outside < d)
            if layer.any():
                attached = layer & ndi.binary_dilation(keep, CONN26)
                keep |= attached
            d -= step
        # anything touching the wall is kept, otherwise the branch loses its root
        keep |= blob & (outside <= step * 1.2)
        cleaned[blob & ~keep] = 0
    return cleaned


def bounding_box(mask, pad=2):
    """Index box around a mask, with a little room to dilate into.

    ``np.any`` along each pair of axes rather than ``find_objects``, which wants
    an integer copy of the whole volume and is the more expensive of the two by
    a wide margin when this is called once per label.
    """
    box = []
    for axis in range(mask.ndim):
        others = tuple(a for a in range(mask.ndim) if a != axis)
        hit = np.any(mask, axis=others)
        idx = np.nonzero(hit)[0]
        if not idx.size:
            return None
        box.append(slice(max(0, int(idx[0]) - pad),
                         min(mask.shape[axis], int(idx[-1]) + 1 + pad)))
    return tuple(box)


def trace_outward(component, outside, spacing, start_mask, start_point_mm,
                  max_path_mm=10.0):
    """Follow one branch outward, shell by shell, accumulating path length."""
    step = float(max(0.5, np.min(spacing) * 0.9))
    here = np.asarray(start_point_mm, dtype=float)

    # This walks a structure of a few hundred voxels, so every dilation and
    # relabel below should cost what that structure costs, not what the volume
    # costs. Cropping to the label's own box is worth about two thirds of the
    # detector's runtime.
    box = bounding_box(component | start_mask)
    if box is None:
        return [(0.0, here)], None, 0.0
    origin_mm = np.array([b.start for b in box], dtype=float) * spacing
    component = component[box]
    outside = outside[box]
    start_mask = start_mask[box]
    if not start_mask.any():
        return [(0.0, here)], None, 0.0
    # Start from the near edge of the footprint rather than its average depth.
    # A patch cut obliquely through the wall has voxels at a range of distances,
    # and starting at the mean can put the first shell past tissue that is
    # actually there.
    r0 = float(np.percentile(outside[start_mask], 25))

    path = [(0.0, here)]
    prev = start_mask
    travelled = 0.0
    bifurcation = None

    # A shell is about one voxel thick, so on a 1.5 mm scan a vessel running
    # obliquely can miss one entirely, and one dilation of the previous shell
    # does not always reach across the gap. Treating that as the end of the
    # vessel was deleting whole branches: a 500 voxel structure would come back
    # with a path of one point and be dropped without ever reaching a filter.
    # Two or three empty shells are tolerated, with the reach widened to match,
    # before the trace is called finished.
    max_gap = 3
    misses = 0

    r = r0 + step
    while travelled < max_path_mm + 1e-6:
        shell = component & (outside >= r - step * 0.5) & (outside < r + step * 0.5)
        if not shell.any():
            if misses >= max_gap:
                break
            misses += 1
            r += step
            continue
        lbl, n = ndi.label(shell, structure=CONN26)
        reach = ndi.binary_dilation(prev, CONN26, iterations=1 + misses)
        touching = [int(t) for t in np.unique(lbl[reach & shell]) if t > 0]
        if not touching:
            if misses >= max_gap:
                break
            misses += 1
            r += step
            continue
        misses = 0
        sizes = {t: int((lbl == t).sum()) for t in touching}
        biggest = max(sizes, key=sizes.get)
        if len(touching) > 1:
            second = sorted(sizes.values())[-2]
            if second >= max(3, 0.45 * sizes[biggest]):
                bifurcation = travelled
                break
        blob = lbl == biggest
        pts = np.array(np.nonzero(blob), dtype=float).mean(axis=1) * spacing + origin_mm
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
    radial = np.hypot(gx, gy)

    def anchor(mask):
        """The pixel to call the middle of the lumen.

        The seed comes off a path traced on a 1.5 mm grid, so it lands a
        fraction of a voxel off centre as a matter of course. Insisting on the
        exact centre pixel throws away a perfectly good cross-section whenever
        it does, so anything lit within about a millimetre of the seed counts,
        nearest first.
        """
        if mask[centre]:
            return centre
        reach = int(round(1.0 / step_mm))
        lo_i, hi_i = max(0, n - reach), min(mask.shape[0], n + reach + 1)
        lo_j, hi_j = max(0, n - reach), min(mask.shape[1], n + reach + 1)
        window = mask[lo_i:hi_i, lo_j:hi_j]
        if not window.any():
            return None
        ii, jj = np.nonzero(window)
        off = np.hypot(ii + lo_i - n, jj + lo_j - n)
        k = int(np.argmin(off))
        return (int(ii[k] + lo_i), int(jj[k] + lo_j))

    peak = float(np.median(hu[n - 2:n + 3, n - 2:n + 3]))
    ring = radial > half_extent_mm * 0.78
    background = float(np.median(hu[ring])) if ring.any() else 0.0
    level = max(0.5 * (peak + background), hu_low * 0.75)
    # Reading the peak and the background at a scale set by a first pass was
    # tried here and moved the median error by 0.04 mm, which is a fortieth of
    # a voxel, while letting several leaks back past the proportion filter. On
    # a 1.5 mm grid a 2.5 mm artery is under two voxels across and the limit is
    # the sampling, not the choice of level.

    binary = (hu >= level) & (inside > 0.25)
    at = anchor(binary)
    if at is None:
        binary = inside > 0.5
        at = anchor(binary)
        if at is None:
            return None
    lbl, _ = ndi.label(binary)
    blob = lbl == lbl[at]
    area = float(blob.sum()) * step_mm * step_mm
    dist = ndi.distance_transform_edt(blob, sampling=(step_mm, step_mm))
    # The area-equivalent radius is the measurement; the inscribed radius is
    # only there to catch a cross-section that has leaked sideways into the
    # parent or a neighbour, where the area balloons but the widest circle that
    # fits inside does not. Anchoring that guard on the seed pixel instead of
    # the widest point of the lumen made an off-centre seed read as a collapsed
    # vessel, which was quietly deleting real branches at the size floor.
    inscribed = float(dist.max())
    return float(min(np.sqrt(area / np.pi), max(inscribed * 1.35, 0.4)))


def split_arms(component, outside, spacing, wall_band, min_footprint_mm2=2.2,
               min_arm_mm3=12.0, probe_lo_mm=2.5, probe_hi_mm=9.0,
               probe_step_mm=0.75, max_arms=4):
    """Separate a footprint that is serving more than one daughter.

    "One footprint, one ostium" is the right rule and it is the one the brief
    sets out, but it only holds if the footprints are actually separate at the
    threshold the scan forced on us. A permissive threshold is what lets a faint
    branch reach the wall at all, and the same permissiveness runs the contact
    patches of two neighbouring arteries together into one. The count then
    collapses: one patch, one ostium, and the second artery is gone.

    So the number of instances is not settled at the wall. It is settled a few
    millimetres out, where two vessels that shared a patch have visibly parted
    and a single vessel has not. This walks a shell outward through the labelled
    tissue, takes the distance at which it separates into the most arms that are
    each big enough to be an artery, then propagates those arms back inward to
    the wall so each one ends up owning its own piece of the footprint.

    Returns a list of boolean masks. A single-element list means no split.
    """
    voxel_mm3 = float(np.prod(spacing))
    face = float(np.sort(spacing)[0] * np.sort(spacing)[1])

    # Everything below is local to this one label, and most labels are a few
    # hundred voxels in a volume of tens of millions, so work in the label's own
    # bounding box and paste the result back at the end. Without this the probe
    # loop relabels the whole volume once per label per probe distance.
    box = bounding_box(component)
    if box is None:
        return [component]
    full_shape = component.shape
    component = component[box]
    outside = outside[box]
    wall_band = wall_band[box]

    def restore(mask):
        out = np.zeros(full_shape, dtype=bool)
        out[box] = mask
        return out

    best_arms, best_count = None, 1
    probe = probe_lo_mm
    while probe <= probe_hi_mm:
        out_part = component & (outside >= probe)
        if out_part.any():
            lbl, n = ndi.label(out_part, structure=CONN26)
            if n >= 2:
                sizes = np.bincount(lbl.ravel())
                keep = []
                for i in range(1, n + 1):
                    if sizes[i] * voxel_mm3 < min_arm_mm3:
                        continue
                    # a stub that stops as soon as it appears is a bulge on one
                    # vessel, not a second vessel
                    arm = lbl == i
                    if float(outside[arm].max() - probe) < 3.0:
                        continue
                    keep.append(i)
                if len(keep) > best_count:
                    best_count = len(keep)
                    best_arms = (lbl, keep[:max_arms], probe)
        probe += probe_step_mm

    if best_arms is None:
        return [restore(component)]

    lbl, keep, probe = best_arms
    seeded = np.zeros(component.shape, dtype=np.int32)
    for j, i in enumerate(keep, start=1):
        seeded[lbl == i] = j

    # Mirror of the outward growth that produced the label in the first place:
    # claim unassigned tissue in order of *decreasing* distance from the aorta,
    # so each arm walks back down its own path to the wall rather than across.
    step = float(max(0.5, np.min(spacing) * 0.9))
    r = probe
    while r > -step:
        target = component & (seeded == 0) & (outside >= r - step)
        for _ in range(2):
            if not target.any():
                break
            spread = ndi.maximum_filter(seeded, size=3, mode="nearest")
            fill = target & (spread > 0)
            if not fill.any():
                break
            seeded = np.where(fill, spread, seeded)
            target = target & ~fill
        r -= step

    arms = []
    for j in range(1, len(keep) + 1):
        arm = seeded == j
        if not arm.any():
            continue
        if float((arm & wall_band).sum()) * face < min_footprint_mm2:
            continue
        arms.append(arm)

    # anything the arms failed to claim stays with the largest of them, so no
    # tissue is silently dropped
    if len(arms) < 2:
        return [restore(component)]
    leftover = component & ~np.any(arms, axis=0)
    if leftover.any():
        biggest = int(np.argmax([a.sum() for a in arms]))
        arms[biggest] = arms[biggest] | leftover
    return [restore(a) for a in arms]


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
    band = ndi.binary_dilation(aorta, CONN18) & ~aorta
    contact = (labels > 0) & band

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

    dilated_aorta = ndi.binary_dilation(aorta, CONN18)

    parts = []
    for root, members in groups.items():
        whole = np.isin(labels, members)
        if not (whole & dilated_aorta).any():
            continue
        parts.extend(split_arms(whole, outside, spacing, band))

    results = []
    for component in parts:
        patch = component & dilated_aorta
        if not patch.any():
            continue

        ostium_mm = opening_centre(patch, band, spacing)
        path, bifurcation, radial_reach = trace_outward(component, outside, spacing,
                                                        patch, ostium_mm)
        if len(path) < 2:
            continue

        dists = np.array([d for d, _ in path])
        pts = np.array([p for _, p in path])

        ostium_idx = np.array(np.nonzero(patch), dtype=float)
        area_mm2 = float(ostium_idx.shape[1] * np.sort(spacing)[0] * np.sort(spacing)[1])
        ostium_radius = float(np.sqrt(area_mm2 / np.pi))

        seed_target = min(5.0, float(dists[-1]))
        seed_mm = np.array([np.interp(seed_target, dists, pts[:, k]) for k in range(3)])

        def fit_direction(window_mm):
            near = pts[dists <= window_mm] if (dists <= window_mm).any() else pts[:2]
            centred = near - near.mean(axis=0)
            if len(near) >= 3:
                _, _, vt = np.linalg.svd(centred, full_matrices=False)
                d = vt[0]
            else:
                d = near[-1] - near[0]
            if np.dot(d, seed_mm - ostium_mm) < 0:
                d = -d
            n = np.linalg.norm(d)
            return d / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])

        # Riffaud et al. (Med Biol Eng Comput 2022) fit the direction a branch
        # leaves at over three times its own radius. A lumbar artery a
        # millimetre across and a renal four times that should not share a
        # fitting window, so the radius is measured once on a provisional fit
        # and the direction is then fitted again at the right scale.
        direction = fit_direction(6.5)
        radius = measure_radius(ct, vessel, seed_mm, direction, spacing, window["hu_low"])
        if radius is None:
            # Fall back to the distance transform, but read it at the nearest
            # voxel that is actually part of the branch. Reading it at the
            # rounded seed returns zero whenever the seed rounds just outside
            # the component, which is a measurement failure reported as a
            # collapsed lumen.
            edt = ndi.distance_transform_edt(component, sampling=spacing)
            seed_vox = np.clip(np.round(seed_mm / spacing).astype(int), 0,
                               np.array(ct.shape) - 1)
            if not component[tuple(seed_vox)]:
                idx = np.array(np.nonzero(component), dtype=float)
                if idx.size:
                    off = (idx - seed_vox[:, None]) * spacing[:, None]
                    seed_vox = idx[:, int(np.argmin((off ** 2).sum(axis=0)))].astype(int)
            radius = float(edt[tuple(seed_vox)])
        else:
            scaled = float(np.clip(3.0 * radius, 3.0, 9.0))
            direction = fit_direction(scaled)
            refined = measure_radius(ct, vessel, seed_mm, direction, spacing, window["hu_low"])
            if refined is not None:
                radius = refined

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
