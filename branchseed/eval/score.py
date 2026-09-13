"""Score a filter setting against the draft reference set.

The image processing is run once per case and cached; only the classification
step is repeated, so a setting can be scored in milliseconds. Every number here
is measured against annotations their own authors describe as draft and not
exhaustive, on five cases, so a difference of one instance moves the third
decimal place. Read it as a direction, not as a grade.
"""

from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np
from scipy.optimize import linear_sum_assignment

import diag
import pipeline as pl

CASES = [19, 20, 21, 22, 23]
CACHE = "/home/claude/branchseed/out/candcache"
TOL_MM = 6.0


def cached(force=False):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for num in CASES:
        path = f"{CACHE}/c{num}.pkl"
        if os.path.exists(path) and not force:
            with open(path, "rb") as fh:
                out[num] = pickle.load(fh)
                continue
        case = f"subject{num:03d}"
        image = glob.glob(f"/home/claude/branchseed/data/{case}/orig*.nii")[0]
        mask = glob.glob(f"/home/claude/branchseed/data/{case}/mask*.nii")[0]
        rec = pl.process_case(image, mask, case, want_maps=False)
        slim = dict(stats=rec["stats"], candidates=rec["candidates"],
                    aorta_length_mm=rec["aorta_length_mm"])
        with open(path, "wb") as fh:
            pickle.dump(slim, fh)
        out[num] = slim
    return out


def _dist_to_line(point, line):
    a, b = line[:-1], line[1:]
    ab = b - a
    den = np.einsum("ij,ij->i", ab, ab)
    den[den < 1e-9] = 1e-9
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / den, 0.0, 1.0)
    return float(np.min(np.linalg.norm(a + t[:, None] * ab - point, axis=1)))


def score(params=None, data=None, cases=None, detail=False):
    data = data or cached()
    cases = cases or CASES
    tp = fp = fn = 0
    od, so, ang, rerr = [], [], [], []
    per = {}
    for num in cases:
        rec = data[num]
        kept, _, _ = pl.classify(rec["candidates"], rec["stats"],
                                 rec["aorta_length_mm"], params=params)
        ann = diag.ref_case(num)
        refs = ann["daughters"]
        cost = np.full((len(refs), len(kept)), 1e6)
        for i, r in enumerate(refs):
            ro = np.asarray(r["ostium_xyz_mm"], float)
            for j, c in enumerate(kept):
                cost[i, j] = np.linalg.norm(ro - np.asarray(c["ostium_xyz_mm"], float))
        hits = 0
        if len(refs) and len(kept):
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] > TOL_MM:
                    continue
                hits += 1
                r, c = refs[i], kept[j]
                od.append(cost[i, j])
                so.append(_dist_to_line(np.asarray(c["seed_xyz_mm"], float),
                                        np.asarray(r["centerline_xyz_mm"], float)))
                cos = float(np.clip(np.dot(np.asarray(c["direction_xyz"], float),
                                           np.asarray(r["direction_xyz"], float)), -1, 1))
                ang.append(float(np.degrees(np.arccos(abs(cos)))))
                rerr.append(c["radius_mm"] - r["origin_diameter_estimate_mm"] / 2.0)
        tp += hits
        fp += len(kept) - hits
        fn += len(refs) - hits
        per[num] = (hits, len(kept) - hits, len(refs) - hits)

    prec = tp / max(tp + fp, 1)
    rec_ = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec_ / max(prec + rec_, 1e-9)
    out = dict(tp=tp, fp=fp, fn=fn, precision=round(prec, 3),
               recall=round(rec_, 3), f1=round(f1, 3),
               ostium_med=round(float(np.median(od)), 2) if od else None,
               ostium_mean=round(float(np.mean(od)), 2) if od else None,
               seed_med=round(float(np.median(so)), 2) if so else None,
               angle_med=round(float(np.median(ang)), 1) if ang else None,
               radius_med_abs=round(float(np.median(np.abs(rerr))), 2) if rerr else None)
    if detail:
        out["per_case"] = per
    return out


def line(tag, s):
    return (f"{tag:34} F1={s['f1']:.3f}  P={s['precision']:.3f} R={s['recall']:.3f}"
            f"  tp={s['tp']:2d} fp={s['fp']:2d} fn={s['fn']:2d}"
            f"  ost={s['ostium_med']} seed={s['seed_med']} ang={s['angle_med']}"
            f" dr={s['radius_med_abs']}")


if __name__ == "__main__":
    data = cached(force="--force" in os.sys.argv)
    base = score(data=data, detail=True)
    print(line("defaults", base))
    print("  per case (tp,fp,fn):", base["per_case"])

    grid = {
        "min_seed_radius_mm": [0.4, 0.65, 0.85, 1.0, 1.2],
        "min_ostium_radius_mm": [0.85, 1.0, 1.2, 1.5],
        "min_footprint_ratio": [0.0, 0.4, 0.5, 0.6, 0.7],
        "max_footprint_ratio": [2.5, 3.2, 4.2, 6.0],
        "min_reach_mm": [5.0, 6.0, 7.0, 8.0],
        "cap_end_mm": [3.0, 6.0, 9.0],
        "min_clearance_mm": [1.0, 2.0, 2.5, 3.0],
    }
    print("\none knob at a time, everything else at its default:")
    for key, values in grid.items():
        for v in values:
            s = score({key: v}, data=data)
            mark = "  <-" if abs(v - pl.DEFAULTS[key]) < 1e-9 else ""
            print(line(f"  {key}={v}", s) + mark)
