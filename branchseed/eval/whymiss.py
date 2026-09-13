"""Why did the detector never see this reference daughter?

A miss with no nearby candidate means the failure happened before the filters,
so this walks back through the stages at the reference ostium: is the reference
path bright enough to clear the threshold this scan chose, does that bright
tissue actually touch the supplied wall, and did a footprint label form there.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

import branches as br
import diag
import sweep

TARGETS = {20: ["branch_003", "branch_004"], 22: ["branch_004"],
           23: ["branch_002", "branch_003"], 19: ["branch_003"]}


def look(num, wanted):
    rec = sweep.records()[num]
    ann = diag.ref_case(num)
    vol = rec["volume"]
    spacing = rec["spacing"]
    offset = rec["offset_vox"]

    import glob
    import bsio
    image = glob.glob(f"/home/claude/branchseed/data/subject{num:03d}/orig*.nii")[0]
    mask = glob.glob(f"/home/claude/branchseed/data/subject{num:03d}/mask*.nii")[0]
    v, aorta_full = bsio.load_case(image, mask)
    import aorta as ao
    aorta_full = ao.clean_mask(aorta_full)
    sl, lo = ao.roi_bounds(aorta_full, v.spacing, margin_mm=31.0)
    ct = np.ascontiguousarray(v.data[sl])
    aorta = np.ascontiguousarray(aorta_full[sl])

    window = br.contrast_window(ct, aorta, spacing)
    labels, outside, present = br.outward_growth(ct, aorta, spacing, window)
    band = ndi.binary_dilation(aorta, br.CONN18) & ~aorta
    contact = (labels > 0) & band

    print(f"\n=== case {num}  hu_low={window['hu_low']:.0f} hu_high={window['hu_high']:.0f} "
          f"lumen_med={window['lumen_median']:.0f} labels={len(present)}")

    for d in ann["daughters"]:
        if d["instance_id"] not in wanted:
            continue
        line = np.asarray(d["centerline_xyz_mm"], float)
        idx = vol.physical_to_index(line) - lo[None, :]
        vox = np.rint(idx).astype(int)
        ok = np.all((vox >= 0) & (vox < np.array(ct.shape)[None, :]), axis=1)
        vox = vox[ok]
        if not len(vox):
            print(f"  {d['instance_id']}: reference path falls outside the cropped ROI")
            continue
        hu = ct[tuple(vox.T)]
        lab = labels[tuple(vox.T)]
        inside_parent = aorta[tuple(vox.T)]
        in_band = band[tuple(vox.T)]
        above = hu >= window["hu_low"]
        print(f"  {d['instance_id']}  dia={d['origin_diameter_estimate_mm']} "
              f"conf={d['confidence']}")
        print(f"    path HU  min={hu.min():.0f} med={np.median(hu):.0f} max={hu.max():.0f}"
              f"   >= hu_low on {above.sum()}/{len(hu)} points")
        print(f"    in parent mask: {int(inside_parent.sum())}/{len(hu)}   "
              f"touches wall band: {int(in_band.sum())}")
        got = sorted(set(int(x) for x in lab if x > 0))
        print(f"    growth labels along the path: {got if got else 'NONE'}")
        if got:
            for g in got:
                mine = labels == g
                foot = mine & band
                _, nfoot = ndi.label(foot, structure=br.CONN26)
                print(f"      label {g}: {int(mine.sum())} voxels, "
                      f"{int(foot.sum())} on the wall in {nfoot} patch(es)")
                if foot.any():
                    idx = np.array(np.nonzero(foot), float).mean(axis=1)
                    here = vol.index_to_physical(idx + lo)
                    ro = np.asarray(d["ostium_xyz_mm"], float)
                    print(f"        footprint centre is {np.linalg.norm(here - ro):.1f} mm "
                          f"from this reference ostium")


if __name__ == "__main__":
    for num, wanted in sorted(TARGETS.items()):
        look(num, wanted)
