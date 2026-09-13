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

_MAP_STOPS = np.array([
    [20, 11, 27],
    [61, 18, 51],
    [122, 27, 61],
    [184, 65, 47],
    [224, 132, 44],
    [246, 217, 160],
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
    norm = np.clip((depth - 1.3) / 6.0, 0.0, 1.0)
    norm = np.power(norm, 0.8)
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
        tile[binary] = (64, 196, 255, 40)
        tile[edge] = (110, 226, 255, 235)
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
            radius_low_confidence=bool(c.get("radius_low_confidence", False)),
            hu=round(float(c["path_hu_median"]), 0),
            ostium_xyz_mm=[round(float(x), 2) for x in c["ostium_xyz_mm"]],
            seed_xyz_mm=[round(float(x), 2) for x in c["seed_xyz_mm"]],
            direction_xyz=[round(float(x), 4) for x in c["direction_xyz"]],
            slice_z=round(float(c["ostium_mm"] @ superior), 2),
            slice_left=round(float(c["ostium_mm"] @ left), 2),
            slice_anterior=round(float(c["ostium_mm"] @ anterior), 2),
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
    )


def write_atlas(entries, out_dir: str):
    payload = dict(
        generated_by="Branchseed pipeline",
        cases=entries,
    )
    with open(os.path.join(out_dir, "atlas.json"), "w") as fh:
        json.dump(payload, fh, separators=(",", ":"))
    return os.path.join(out_dir, "atlas.json")
