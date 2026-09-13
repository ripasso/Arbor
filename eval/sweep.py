"""Dump every candidate on the reference cases with the measurements a filter
could use, labelled by whether a reference daughter claims it.

Nothing here changes the pipeline. It exists so that any threshold that does
change can be chosen against a distribution instead of a hunch.
"""

from __future__ import annotations

import glob
import json
import os
import pickle

import numpy as np

import diag
import pipeline as pl

CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "diagcache")
DATA = os.environ.get("BRANCHSEED_DATA", "/Users/rastinabbaspour/Downloads/TORALIS CHALLENGE ")
CASES = [19, 20, 21, 22, 23]


def records(force=False):
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for num in CASES:
        path = f"{CACHE}/rec{num}.pkl"
        if os.path.exists(path) and not force:
            with open(path, "rb") as fh:
                out[num] = pickle.load(fh)
            continue
        case = f"subject{num:03d}"
        image = glob.glob(f"{DATA}/{case}/orig*.nii")[0]
        mask = glob.glob(f"{DATA}/{case}/mask*.nii")[0]
        rec = pl.process_case(image, mask, case, want_maps=False)
        slim = dict(case_id=rec["case_id"], stats=rec["stats"],
                    daughters=rec["daughters"], extras=rec["extras"],
                    rejected=rec["rejected"], volume=rec["volume"],
                    spacing=rec["spacing"], offset_vox=rec["offset_vox"])
        with open(path, "wb") as fh:
            pickle.dump(slim, fh)
        out[num] = slim
    return out


def table(force=False):
    recs = records(force)
    rows = []
    for num, rec in recs.items():
        ann = diag.ref_case(num)
        refs = [np.asarray(d["ostium_xyz_mm"], float) for d in ann["daughters"]]
        lumen = rec["stats"]["lumen_median"]
        for kind, group in (("kept", rec["daughters"]), ("extra", rec["extras"]),
                            ("refused", rec["rejected"])):
            for c in group:
                o = np.asarray(c["ostium_xyz_mm"], float)
                near = min((float(np.linalg.norm(o - r)) for r in refs), default=1e9)
                rows.append(dict(
                    case=num, kind=kind, reason=c.get("reject_reason"),
                    match=near <= 6.0, near_mm=round(near, 2),
                    hu=round(c["path_hu_median"], 0),
                    hu_ratio=round(c["path_hu_median"] / max(lumen, 1.0), 3),
                    lumen=round(lumen, 0),
                    radius=round(c["radius_mm"], 2),
                    ostium_radius=round(c["ostium_radius_mm"], 2),
                    reach=round(c["reach_mm"], 2),
                    straight=round(c["straight_mm"], 2),
                    clearance=round(c["seed_wall_clearance_mm"], 2),
                    fp_ratio=round(c["footprint_ratio"], 2),
                    align=round(c["axial_alignment"], 2),
                    end_mm=round(c["distance_to_end_mm"], 1),
                    parent_r=round(c["parent_radius_mm"], 2),
                ))
    return rows


def split(rows, key, kinds=("kept", "extra")):
    sel = [r for r in rows if r["kind"] in kinds]
    hit = sorted(r[key] for r in sel if r["match"])
    bad = sorted(r[key] for r in sel if not r["match"])
    def q(v, p):
        return round(float(np.percentile(v, p)), 3) if v else None
    return dict(key=key, n_hit=len(hit), n_bad=len(bad),
                hit_min=round(min(hit), 3) if hit else None,
                hit_p10=q(hit, 10), hit_med=q(hit, 50),
                bad_med=q(bad, 50), bad_p75=q(bad, 75),
                bad_max=round(max(bad), 3) if bad else None)


if __name__ == "__main__":
    rows = table(force="--force" in os.sys.argv)
    with open(f"{CACHE}/rows.json", "w") as fh:
        json.dump(rows, fh, indent=1, default=float)
    keys = ["hu_ratio", "hu", "radius", "ostium_radius", "reach", "straight",
            "clearance", "fp_ratio", "align", "end_mm"]
    print(f"{'key':15}{'nHit':>5}{'nBad':>5} | {'hitMin':>8}{'hitP10':>8}{'hitMed':>8}"
          f" | {'badMed':>8}{'badP75':>8}{'badMax':>8}")
    for k in keys:
        s = split(rows, k)
        print(f"{k:15}{s['n_hit']:>5}{s['n_bad']:>5} | {str(s['hit_min']):>8}"
              f"{str(s['hit_p10']):>8}{str(s['hit_med']):>8} | {str(s['bad_med']):>8}"
              f"{str(s['bad_p75']):>8}{str(s['bad_max']):>8}")
    print("\nlumen median per case:",
          {r['case']: r['lumen'] for r in rows})
