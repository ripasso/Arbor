"""Predicted radius against the reference origin diameter, for matched pairs.

The two are not quite the same quantity: the reference estimates the lumen at
the opening, we measure it at the seed five millimetres out, and a real artery
narrows on the way. So ours should sit at or a little below half the reference
diameter. Anything above it is a measurement that has run into the wall.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.optimize import linear_sum_assignment

import diag
import sweep

CASES = [19, 20, 21, 22, 23]


def pairs(recs=None):
    recs = recs or sweep.records()
    out = []
    for num in CASES:
        rec = recs[num]
        ann = diag.ref_case(num)
        refs = ann["daughters"]
        kept = rec["daughters"]
        if not refs or not kept:
            continue
        cost = np.zeros((len(refs), len(kept)))
        for i, r in enumerate(refs):
            ro = np.asarray(r["ostium_xyz_mm"], float)
            for j, c in enumerate(kept):
                cost[i, j] = np.linalg.norm(ro - np.asarray(c["ostium_xyz_mm"], float))
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] > 6.0:
                continue
            r, c = refs[i], kept[j]
            out.append(dict(case=num, ref=r["instance_id"],
                            ref_r=r["origin_diameter_estimate_mm"] / 2.0,
                            ref_meas=r.get("radius_mm"),
                            pred_r=round(c["radius_mm"], 2),
                            ostium_r=round(c["ostium_radius_mm"], 2),
                            d=round(cost[i, j], 2)))
    return out


if __name__ == "__main__":
    p = pairs()
    print(f"{'case':6}{'ref':13}{'ref_r':>7}{'pred_r':>8}{'ratio':>7}{'ostium_r':>9}")
    for r in p:
        print(f"{r['case']:<6}{r['ref']:13}{r['ref_r']:>7.2f}{r['pred_r']:>8.2f}"
              f"{r['pred_r']/r['ref_r']:>7.2f}{r['ostium_r']:>9.2f}")
    ratio = np.array([r["pred_r"] / r["ref_r"] for r in p])
    err = np.array([r["pred_r"] - r["ref_r"] for r in p])
    print(f"\nn={len(p)}  ratio median {np.median(ratio):.2f}  mean {ratio.mean():.2f}"
          f"  over 1.3x: {(ratio > 1.3).sum()}/{len(p)}")
    print(f"absolute error median {np.median(np.abs(err)):.2f} mm  "
          f"mean signed {err.mean():+.2f} mm")
