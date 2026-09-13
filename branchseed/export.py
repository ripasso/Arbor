"""Turn a processed case into the compact data and images the atlas is drawn from.

Three artefacts per case:

* ``unwrap_<case>.png`` - the aortic wall flattened into arc length by clock
  angle, where every branch origin shows up as a blob.
* ``ct_<case>.jpg`` / ``mask_<case>.png`` - a sprite sheet of patient-aligned
  axial slices through the aorta, plus a transparent overlay of the supplied
  mask so it can be toggled.
* an entry in ``atlas.json`` holding the geometry, the branches and the
  branch-free landing zones.

Slices are resampled onto a patient-aligned grid (left / anterior) rather than
taken straight off the array axes, so every case is shown the same way up and
overlay positions are plain millimetres.
"""

from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

SLICE_PX = 176
SLICE_FOV_MM = 62.0
MAX_SLICES = 48
SPRITE_COLS = 8


# --------------------------------------------------------------------------- #
# unwrapped surface image
# --------------------------------------------------------------------------- #

# white where nothing leaves the wall, ink where a daughter does
_MAP_STOPS = np.array([
    [255, 255, 255],
    [214, 219, 226],
    [156, 165, 178],
    [96, 106, 122],
    [42, 50, 64],
    [8, 11, 18],
], dtype=float)


def _colormap(norm: np.ndarray) -> np.ndarray:
    x = np.clip(norm, 0.0, 1.0) * (len(_MAP_STOPS) - 1)
    lo = np.floor(x).astype(int)
    hi = np.minimum(lo + 1, len(_MAP_STOPS) - 1)
    f = (x - lo)[..., None]
    return (_MAP_STOPS[lo] * (1 - f) + _MAP_STOPS[hi] * f).astype(np.uint8)


def unwrap_image(unwrap, out_path: str, scale: int = 3):
    """Render the flattened wall, anterior in the middle of the frame."""
    depth = unwrap.shell_depth_mm
    roll = depth.shape[1] // 2
    depth = np.roll(depth, roll, axis=1)          # put 0 deg (anterior) centre
    depth = ndi.gaussian_filter(depth, sigma=(0.8, 1.2), mode=("nearest", "wrap"))
    # a millimetre or so of brightness beyond the wall is noise and partial
    # volume rather than a vessel, so hold it at the floor of the ramp
    norm = np.clip((depth - 1.0) / 4.3, 0.0, 1.0)
    norm = np.power(norm, 0.7)
    rgb = _colormap(norm)
    img = Image.fromarray(rgb, mode="RGB")
    img = img.resize((depth.shape[1] * scale, depth.shape[0] * scale), Image.BICUBIC)
    img.save(out_path, optimize=True)
    return dict(width=img.width, height=img.height,
                arc_min=float(unwrap.arc_mm[0]), arc_max=float(unwrap.arc_mm[-1]),
                rows=int(depth.shape[0]), cols=int(depth.shape[1]))


# --------------------------------------------------------------------------- #
# patient-aligned slice sprites
# --------------------------------------------------------------------------- #

def slice_sprites(record, ct_path: str, mask_path: str):
    ct = record["ct"]
    aorta = record["aorta"]
    spacing = record["spacing"]
    points = record["points_mm"]
    arc = record["arc_mm"]

    # patient-aligned in-plane axes, expressed in scaled-voxel mm
    from pipeline import _anatomical_axes
    anterior, left, superior = _anatomical_axes(record["volume"].affine, spacing)

    z_along = points @ superior
    order = np.argsort(z_along)
    z_sorted = z_along[order]

    n = min(MAX_SLICES, max(len(points) // 2, 6))
    z_targets = np.linspace(z_sorted[0], z_sorted[-1], n)

    pitch = SLICE_FOV_MM / SLICE_PX
    g = (np.arange(SLICE_PX) - (SLICE_PX - 1) / 2.0) * pitch
    gx, gy = np.meshgrid(g, g, indexing="xy")     # x -> left, y -> anterior

    tiles_ct, tiles_mask, centres = [], [], []
    for zt in z_targets:
        # centre the tile on the centerline at this level
        centre = np.stack([np.interp(zt, z_sorted, points[order][:, k]) for k in range(3)])
        pts = (centre[None, None, :]
               + gx[..., None] * left
               + gy[..., None] * anterior)
        # force the sample onto the requested axial level
        drift = (pts @ superior) - zt
        pts = pts - drift[..., None] * superior
        coords = np.moveaxis(pts / spacing, -1, 0)

        hu = ndi.map_coordinates(ct, coords, order=1, mode="constant", cval=-1000.0)
        msk = ndi.map_coordinates(aorta.astype(np.float32), coords, order=1,
                                  mode="constant", cval=0.0)
        tiles_ct.append(hu)
        tiles_mask.append(msk)
        centres.append(centre)

    window_lo, window_hi = -120.0, 420.0
    lumen = record["stats"]["lumen_median"]
    window_hi = max(window_hi, lumen * 1.15)

    rows = int(np.ceil(len(tiles_ct) / SPRITE_COLS))
    sheet = np.zeros((rows * SLICE_PX, SPRITE_COLS * SLICE_PX), dtype=np.uint8)
    overlay = np.zeros((rows * SLICE_PX, SPRITE_COLS * SLICE_PX, 4), dtype=np.uint8)

    for i, (hu, msk) in enumerate(zip(tiles_ct, tiles_mask)):
        r, c = divmod(i, SPRITE_COLS)
        y0, x0 = r * SLICE_PX, c * SLICE_PX
        norm = np.clip((hu - window_lo) / (window_hi - window_lo), 0, 1)
        # images are sampled with y = anterior, so flip rows to draw anterior up
        sheet[y0:y0 + SLICE_PX, x0:x0 + SLICE_PX] = (norm[::-1] * 255).astype(np.uint8)

        binary = msk[::-1] > 0.5
        edge = binary ^ ndi.binary_erosion(binary)
        tile = np.zeros((SLICE_PX, SLICE_PX, 4), dtype=np.uint8)
        tile[binary] = (92, 146, 230, 34)
        tile[edge] = (46, 104, 200, 225)
        overlay[y0:y0 + SLICE_PX, x0:x0 + SLICE_PX] = tile

    Image.fromarray(sheet, mode="L").convert("RGB").save(ct_path, quality=72,
                                                         optimize=True)
    Image.fromarray(overlay, mode="RGBA").save(mask_path, optimize=True)

    centres = np.array(centres)
    return dict(
        tile_px=SLICE_PX, cols=SPRITE_COLS, rows=rows, count=len(tiles_ct),
        pitch_mm=pitch, fov_mm=SLICE_FOV_MM,
        z_mm=[round(float(z), 2) for z in z_targets],
        centre_left=[round(float(c @ left), 2) for c in centres],
        centre_anterior=[round(float(c @ anterior), 2) for c in centres],
    )


# --------------------------------------------------------------------------- #
# landing zones
# --------------------------------------------------------------------------- #

def landing_zones(record, min_length_mm: float = 8.0):
    """Stretches of aorta with no branch origin, with the local diameter."""
    arc = record["arc_mm"]
    radius = record["radius_profile"]
    blocked = []
    for c in record["daughters"] + record["extras"]:
        half = max(c["ostium_radius_mm"], 1.5)
        blocked.append((c["arc_mm"] - half, c["arc_mm"] + half))
    blocked.sort()

    zones = []
    cursor = float(arc[0])
    for lo, hi in blocked + [(float(arc[-1]), float(arc[-1]))]:
        if lo - cursor >= min_length_mm:
            m = (arc >= cursor) & (arc <= lo)
            if m.any():
                zones.append(dict(
                    start_mm=round(cursor, 1),
                    end_mm=round(float(lo), 1),
                    length_mm=round(float(lo - cursor), 1),
                    min_diameter_mm=round(float(np.percentile(radius[m], 8) * 2), 1),
                    mean_diameter_mm=round(float(radius[m].mean() * 2), 1),
                ))
        cursor = max(cursor, float(hi))
    return zones


# --------------------------------------------------------------------------- #
# assembly
# --------------------------------------------------------------------------- #

def export_case(record, out_dir: str):
    case = record["case_id"]
    os.makedirs(out_dir, exist_ok=True)

    unwrap_meta = unwrap_image(record["unwrap"], os.path.join(out_dir, f"unwrap_{case}.png"))
    side_meta = side_sprites(record, dict(
        coronal_ct=os.path.join(out_dir, f"cor_{case}.jpg"),
        coronal_mask=os.path.join(out_dir, f"cormask_{case}.png"),
        sagittal_ct=os.path.join(out_dir, f"sag_{case}.jpg"),
        sagittal_mask=os.path.join(out_dir, f"sagmask_{case}.png")))
    evid_meta = evidence_sprite(record, os.path.join(out_dir, f"evid_{case}.jpg"))
    sprite_meta = slice_sprites(record,
                                os.path.join(out_dir, f"ct_{case}.jpg"),
                                os.path.join(out_dir, f"mask_{case}.png"))

    from pipeline import _anatomical_axes
    anterior, left, superior = _anatomical_axes(record["volume"].affine, record["spacing"])

    def branch_entry(c, kind, index=None):
        return dict(
            id=(f"branch_{index:03d}" if index else kind),
            kind=kind,
            arc_mm=round(float(c["arc_mm"]), 1),
            clock_deg=round(float(c["clock_deg"]), 1),
            radius_mm=round(float(c["radius_mm"]), 2),
            ostium_radius_mm=round(float(c["ostium_radius_mm"]), 2),
            reach_mm=round(float(c["reach_mm"]), 1),
            traced_mm=round(float(c["traced_mm"]), 1),
            bifurcation_mm=(round(float(c["bifurcation_mm"]), 1)
                            if c["bifurcation_mm"] else None),
            parent_radius_mm=round(float(c["parent_radius_mm"]), 2),
            confidence=c.get("confidence", 0.0),
            hu=round(float(c["path_hu_median"]), 0),
            ostium_xyz_mm=[round(float(x), 2) for x in c["ostium_xyz_mm"]],
            seed_xyz_mm=[round(float(x), 2) for x in c["seed_xyz_mm"]],
            direction_xyz=[round(float(x), 4) for x in c["direction_xyz"]],
            slice_z=round(float(c["ostium_mm"] @ superior), 2),
            slice_left=round(float(c["ostium_mm"] @ left), 2),
            slice_anterior=round(float(c["ostium_mm"] @ anterior), 2),
            seed_z=round(float(c["seed_mm"] @ superior), 2),
            seed_left=round(float(c["seed_mm"] @ left), 2),
            seed_anterior=round(float(c["seed_mm"] @ anterior), 2),
            seed_hu=round(float(c["seed_hu"]), 0),
            wall_clearance_mm=round(float(c["seed_wall_clearance_mm"]), 2),
            dir_left=round(float(c["direction_mm"] @ left), 3),
            dir_anterior=round(float(c["direction_mm"] @ anterior), 3),
            dir_superior=round(float(c["direction_mm"] @ superior), 3),
        )

    step = max(1, len(record["arc_mm"]) // 140)
    profile = [dict(s=round(float(a), 1), d=round(float(r * 2), 2))
               for a, r in zip(record["arc_mm"][::step], record["radius_profile"][::step])]

    from collections import Counter
    rejected = Counter(c["reject_reason"] for c in record["rejected"])

    return dict(
        case_id=case,
        seconds=round(float(record["seconds"]), 2),
        spacing_mm=[round(float(s), 3) for s in record["spacing"]],
        voxels=[int(v) for v in record["volume"].shape],
        aorta_length_mm=round(float(record["arc_mm"][-1]), 1),
        lumen_hu=round(float(record["stats"]["lumen_median"]), 0),
        hu_low=round(float(record["stats"]["hu_low"]), 0),
        opacification_hu=round(float(record["stats"].get("opacification_hu", 0.0)), 0),
        contrast_margin=round(float(record["stats"].get("contrast_margin", 0.0)), 0),
        contrast_ok=bool(record["stats"].get("contrast_ok", True)),
        peak_rss_gb=record.get("peak_rss_gb"),
        mean_diameter_mm=round(float(record["radius_profile"].mean() * 2), 1),
        min_diameter_mm=round(float(np.percentile(record["radius_profile"], 5) * 2), 1),
        max_diameter_mm=round(float(record["radius_profile"].max() * 2), 1),
        profile=profile,
        daughters=[branch_entry(c, "daughter", i)
                   for i, c in enumerate(record["daughters"], start=1)],
        extras=[branch_entry(c, "terminal") for c in record["extras"]],
        rejected=[dict(reason=k, count=int(v)) for k, v in rejected.most_common()],
        landing_zones=landing_zones(record),
        unwrap=unwrap_meta,
        slices=sprite_meta,
        side=side_meta,
        evidence=evid_meta,
    )


def write_atlas(entries, out_dir: str):
    payload = dict(
        generated_by="Branchseed pipeline",
        cases=entries,
    )
    with open(os.path.join(out_dir, "atlas.json"), "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return os.path.join(out_dir, "atlas.json")


# --------------------------------------------------------------------------- #
# coronal and sagittal stacks, and per-branch evidence
# --------------------------------------------------------------------------- #

SIDE_SLICES = 12
SIDE_FOV_MM = 96.0
SIDE_MAX_PX = 560
SIDE_SLAB_MM = 6.0
EVID_PX = 132
EVID_PITCH = 0.40
EVID_SLAB_MM = 5.0
EVID_COLS = 6


def _slab_max(ct, base_pts, normal, spacing, slab_mm, steps=5):
    """Maximum-intensity projection through a slab, which is what makes a
    vessel read as a continuous tube instead of a string of dots."""
    offs = np.linspace(-slab_mm / 2.0, slab_mm / 2.0, steps)
    best = None
    for o in offs:
        coords = np.moveaxis((base_pts + o * normal) / spacing, -1, 0)
        s = ndi.map_coordinates(ct, coords, order=1, mode="constant", cval=-1000.0)
        best = s if best is None else np.maximum(best, s)
    return best


def side_sprites(record, paths):
    """Coronal and sagittal stacks through the aorta, slab-projected.

    The axial view shows a branch as a bright dot that appears for a slice or
    two. Cut along the body instead and the same branch is a spur running away
    from the aorta over a centimetre, which is far harder to mistake for noise.
    """
    ct, aorta, spacing = record["ct"], record["aorta"], record["spacing"]
    from pipeline import _anatomical_axes
    anterior, left, superior = _anatomical_axes(record["volume"].affine, spacing)
    pts = record["points_mm"]

    u = pts @ left
    a = pts @ anterior
    z = pts @ superior
    z0, z1 = float(z.min()) - 6.0, float(z.max()) + 6.0
    length = max(z1 - z0, 20.0)

    pitch = max(0.35, length / SIDE_MAX_PX)
    h = int(min(round(length / pitch), SIDE_MAX_PX))
    w = int(min(round(SIDE_FOV_MM / pitch), 320))

    zs = z1 - (np.arange(h) + 0.5) * pitch                 # row 0 is superior
    gs = (np.arange(w) + 0.5 - w / 2.0) * pitch            # column offset

    window_lo = -120.0
    window_hi = max(420.0, record["stats"]["lumen_median"] * 1.15)

    meta = {}
    for kind, axis, other, centre_axis in (
            ("coronal", anterior, left, float(np.median(a))),
            ("sagittal", left, anterior, float(np.median(u)))):
        base = centre_axis + np.linspace(-18.0, 18.0, SIDE_SLICES)
        in_plane = float(np.median(u)) if kind == "coronal" else float(np.median(a))

        rows = int(np.ceil(SIDE_SLICES / 4))
        sheet = np.zeros((rows * h, 4 * w), dtype=np.uint8)
        overlay = np.zeros((rows * h, 4 * w, 4), dtype=np.uint8)

        for i, off in enumerate(base):
            grid = (zs[:, None, None] * superior[None, None, :]
                    + (in_plane + gs)[None, :, None] * other[None, None, :]
                    + off * axis[None, None, :])
            hu = _slab_max(ct, grid, axis, spacing, SIDE_SLAB_MM)
            coords = np.moveaxis(grid / spacing, -1, 0)
            msk = ndi.map_coordinates(aorta.astype(np.float32), coords, order=1,
                                      mode="constant", cval=0.0) > 0.5

            r, c = divmod(i, 4)
            y0, x0 = r * h, c * w
            norm = np.clip((hu - window_lo) / (window_hi - window_lo), 0, 1)
            sheet[y0:y0 + h, x0:x0 + w] = (norm * 255).astype(np.uint8)
            edge = msk ^ ndi.binary_erosion(msk)
            tile = np.zeros((h, w, 4), dtype=np.uint8)
            tile[msk] = (92, 146, 230, 30)
            tile[edge] = (46, 104, 200, 210)
            overlay[y0:y0 + h, x0:x0 + w] = tile

        Image.fromarray(sheet, mode="L").convert("RGB").save(
            paths[kind + "_ct"], quality=72, optimize=True)
        Image.fromarray(overlay, mode="RGBA").save(paths[kind + "_mask"], optimize=True)

        meta[kind] = dict(
            tile_w=w, tile_h=h, cols=4, rows=rows, count=SIDE_SLICES,
            pitch_mm=round(pitch, 4),
            z_top=round(z1, 2), in_plane=round(in_plane, 2),
            offsets=[round(float(o), 2) for o in base],
        )
    return meta


def evidence_sprite(record, out_path):
    """One thumbnail per daughter, cut in the plane that best shows it.

    A branch is most convincing in the plane that contains both its own
    direction and the aortic axis, slab-projected so the whole proximal run is
    in one picture. Every detection gets the same treatment, so the contact
    sheet is a fair look rather than a selection of the good ones.
    """
    ct, spacing = record["ct"], record["spacing"]
    ds = record["daughters"]
    if not ds:
        return None
    window_lo = -120.0
    window_hi = max(420.0, record["stats"]["lumen_median"] * 1.15)

    n = len(ds)
    rows = int(np.ceil(n / EVID_COLS))
    sheet = np.zeros((rows * EVID_PX, EVID_COLS * EVID_PX), dtype=np.uint8)
    g = (np.arange(EVID_PX) + 0.5 - EVID_PX / 2.0) * EVID_PITCH

    for i, c in enumerate(ds):
        d = np.asarray(c["direction_mm"], dtype=float)
        d = d / max(np.linalg.norm(d), 1e-9)
        t = np.asarray(record["tangents"][int(c["station"])], dtype=float)
        e2 = t - np.dot(t, d) * d
        if np.linalg.norm(e2) < 1e-6:
            e2 = np.cross(d, [1.0, 0.0, 0.0])
        e2 /= max(np.linalg.norm(e2), 1e-9)
        nrm = np.cross(d, e2)

        centre = np.asarray(c["ostium_mm"], dtype=float) + d * 6.0
        grid = centre[None, None, :] + g[None, :, None] * d + g[::-1, None, None] * e2
        hu = _slab_max(ct, grid, nrm, spacing, EVID_SLAB_MM)

        r, col = divmod(i, EVID_COLS)
        norm = np.clip((hu - window_lo) / (window_hi - window_lo), 0, 1)
        sheet[r * EVID_PX:(r + 1) * EVID_PX,
              col * EVID_PX:(col + 1) * EVID_PX] = (norm * 255).astype(np.uint8)

    Image.fromarray(sheet, mode="L").convert("RGB").save(out_path, quality=74,
                                                         optimize=True)
    return dict(tile_px=EVID_PX, cols=EVID_COLS, rows=rows, count=n,
                pitch_mm=EVID_PITCH, slab_mm=EVID_SLAB_MM,
                centre_offset_mm=6.0)
