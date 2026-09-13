"""Centroid vs rim-distance maximum for the reported ostium centre.

No reference annotations exist, so the test is intrinsic: does the reported
point actually lie inside the contact patch it claims to be the centre of, and
is it sitting in contrast rather than in wall or fat?
"""
import glob, sys, numpy as np
from scipy import ndimage as ndi
from bsio import load_case
import aorta as ao, branches as br
from pipeline import _anatomical_axes

tot = dict(n=0, cen_out=0, dt_out=0, cen_hu=[], dt_hu=[], shift=[])
for s in sys.argv[1:]:
    d = "data/" + s
    vol, aorta = load_case(glob.glob(d+"/orig*.nii")[0], glob.glob(d+"/mask*.nii")[0])
    aorta = ao.clean_mask(aorta)
    sl, lo = ao.roi_bounds(aorta, vol.spacing, 31.0)
    ct = np.ascontiguousarray(vol.data[sl]); am = np.ascontiguousarray(aorta[sl])
    sp = vol.spacing
    win = br.contrast_window(ct, am, sp)
    labels, outside, present = br.outward_growth(ct, am, sp, win)
    if not present: continue
    band = ndi.binary_dilation(am, br.CONN18) & ~am
    for k in present:
        patch = (labels == k) & band
        if patch.sum() < 3: continue
        idx = np.array(np.nonzero(patch), dtype=float)
        cen_vox = idx.mean(axis=1)
        cen_nn = tuple(np.clip(np.round(cen_vox).astype(int), 0, np.array(ct.shape)-1))
        dt_mm = br.opening_centre(patch, band, sp)
        dt_nn = tuple(np.clip(np.round(dt_mm / sp).astype(int), 0, np.array(ct.shape)-1))
        tot["n"] += 1
        if not patch[cen_nn]: tot["cen_out"] += 1
        if not patch[dt_nn]:  tot["dt_out"] += 1
        tot["cen_hu"].append(float(ct[cen_nn]))
        tot["dt_hu"].append(float(ct[dt_nn]))
        tot["shift"].append(float(np.linalg.norm(cen_vox*sp - dt_mm)))
n = max(tot["n"], 1)
print(f"patches examined            {tot['n']}")
print(f"centroid falls off patch    {tot['cen_out']}  ({100*tot['cen_out']/n:.1f}%)")
print(f"rim-max falls off patch     {tot['dt_out']}  ({100*tot['dt_out']/n:.1f}%)")
print(f"median HU at centroid       {np.median(tot['cen_hu']):.0f}")
print(f"median HU at rim-max        {np.median(tot['dt_hu']):.0f}")
print(f"median shift between them   {np.median(tot['shift']):.2f} mm   (p90 {np.percentile(tot['shift'],90):.2f} mm)")
