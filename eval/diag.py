"""Per-daughter diagnosis against the draft reference set.

For every reference daughter this reports the nearest candidate the detector
produced, whether that candidate survived the filters, and if not, which filter
removed it. That is the difference between "the detector never saw it" and "the
detector saw it and we threw it away", and the two call for opposite fixes.

It also reports every accepted candidate that no reference daughter claims,
with the measurements that would have to change to refuse it.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import sys

import nibabel as nib
import numpy as np
from scipy.optimize import linear_sum_assignment

import pipeline as pl

REFS = os.environ.get("BRANCHSEED_REFS", "/Users/rastinabbaspour/Downloads/EVAL_SET")
DATA = os.environ.get("BRANCHSEED_DATA", "/Users/rastinabbaspour/Downloads/TORALIS CHALLENGE ")
CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "diagcache")
CASES = [19, 20, 21, 22, 23]
TOL_MM = 6.0


def ref_case(num):
    with open(f"{REFS}/case_{num}/annotations.json") as fh:
        return json.load(fh)


def ref_labels(num):
    raw = gzip.decompress(open(f"{REFS}/case_{num}/daughters{num}_draft.nii.gz", "rb").read())
    img = nib.Nifti1Image.from_bytes(raw)
    return np.asarray(img.dataobj)


def summarise(cand):
    return dict(
        reach_mm=round(cand["reach_mm"], 2),
        straight_mm=round(cand["straight_mm"], 2),
        clearance_mm=round(cand["seed_wall_clearance_mm"], 2),
        radius_mm=round(cand["radius_mm"], 2),
        ostium_radius_mm=round(cand["ostium_radius_mm"], 2),
        footprint_ratio=round(cand["footprint_ratio"], 2),
        path_hu=round(cand["path_hu_median"], 0),
        axial_alignment=round(cand["axial_alignment"], 2),
        dist_to_end_mm=round(cand["distance_to_end_mm"], 1),
        arc_mm=round(cand["arc_mm"], 1),
    )


def run(num, record=None):
    case = f"subject{num:03d}"
    if record is None:
        image = glob.glob(f"{DATA}/{case}/orig*.nii")[0]
        mask = glob.glob(f"{DATA}/{case}/mask*.nii")[0]
        record = pl.process_case(image, mask, case, want_maps=False)

    ann = ref_case(num)
    labels = ref_labels(num)
    vol = record["volume"]
    spacing = record["spacing"]
    offset = record["offset_vox"]

    refs = []
    for d in ann["daughters"]:
        line = np.asarray(d["centerline_xyz_mm"], dtype=float)
        refs.append(dict(
            id=d["instance_id"],
            label=int(d["label_value"]),
            ostium=np.asarray(d["ostium_xyz_mm"], dtype=float),
            seed=np.asarray(d["seed_xyz_mm"], dtype=float),
            direction=np.asarray(d["direction_xyz"], dtype=float),
            radius=d.get("radius_mm"),
            diameter=d.get("origin_diameter_estimate_mm"),
            confidence=d.get("confidence"),
            line=line,
        ))

    pool = []
    for kind, group in (("kept", record["daughters"]), ("extra", record["extras"]),
                        ("refused", record["rejected"])):
        for c in group:
            pool.append(dict(kind=kind, cand=c,
                             reason=c.get("reject_reason"),
                             ostium=np.asarray(c["ostium_xyz_mm"], dtype=float),
                             seed=np.asarray(c["seed_xyz_mm"], dtype=float),
                             direction=np.asarray(c["direction_xyz"], dtype=float)))

    # voxel label under each candidate seed and ostium
    for p in pool:
        for key in ("seed", "ostium"):
            idx = np.rint(vol.physical_to_index(p[key])).astype(int)
            inside = all(0 <= idx[a] < labels.shape[a] for a in range(3))
            p[key + "_label"] = int(labels[tuple(idx)]) if inside else -1

    def dist_to_line(point, line):
        a = line[:-1]
        b = line[1:]
        ab = b - a
        denom = np.einsum("ij,ij->i", ab, ab)
        denom[denom < 1e-9] = 1e-9
        t = np.clip(np.einsum("ij,ij->i", point - a, ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        return float(np.min(np.linalg.norm(proj - point, axis=1)))

    # optimal assignment on ostium distance, kept candidates only
    kept = [p for p in pool if p["kind"] == "kept"]
    pairs = {}
    if refs and kept:
        cost = np.zeros((len(refs), len(kept)))
        for i, r in enumerate(refs):
            for j, p in enumerate(kept):
                cost[i, j] = np.linalg.norm(r["ostium"] - p["ostium"])
        ri, ci = linear_sum_assignment(cost)
        for i, j in zip(ri, ci):
            if cost[i, j] <= TOL_MM:
                pairs[i] = (j, cost[i, j])

    out = dict(case=case, n_ref=len(refs), n_kept=len(kept),
               n_refused=len(record["rejected"]), n_extra=len(record["extras"]),
               contrast=record["stats"].get("contrast_ok"), rows=[], fps=[])

    for i, r in enumerate(refs):
        row = dict(ref=r["id"], diameter=r["diameter"], confidence=r["confidence"],
                   ref_radius=r["radius"])
        if i in pairs:
            j, d = pairs[i]
            p = kept[j]
            row["status"] = "hit"
            row["ostium_mm"] = round(d, 2)
            row["seed_offset_mm"] = round(dist_to_line(p["seed"], r["line"]), 2)
            cos = float(np.clip(np.dot(p["direction"], r["direction"]), -1, 1))
            row["angle_deg"] = round(float(np.degrees(np.arccos(abs(cos)))), 1)
            row["seed_label"] = p["seed_label"]
            row["pred_radius"] = round(p["cand"]["radius_mm"], 2)
            row["stats"] = summarise(p["cand"])
        else:
            # nearest candidate of any kind
            best, bestd = None, 1e9
            for p in pool:
                d = np.linalg.norm(r["ostium"] - p["ostium"])
                if d < bestd:
                    best, bestd = p, d
            row["status"] = "miss"
            row["nearest_mm"] = round(bestd, 2)
            row["nearest_kind"] = best["kind"] if best else None
            row["nearest_reason"] = best["reason"] if best else None
            row["nearest_stats"] = summarise(best["cand"]) if best else None
        out["rows"].append(row)

    claimed = {j for j, _ in pairs.values()}
    for j, p in enumerate(kept):
        if j in claimed:
            continue
        near = min((np.linalg.norm(r["ostium"] - p["ostium"]) for r in refs), default=1e9)
        out["fps"].append(dict(nearest_ref_mm=round(float(near), 2),
                               seed_label=p["seed_label"],
                               ostium_label=p["ostium_label"],
                               stats=summarise(p["cand"])))
    return out, record


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    report = []
    for num in CASES:
        out, _ = run(num)
        report.append(out)
        tp = sum(1 for r in out["rows"] if r["status"] == "hit")
        print(f"\n=== {out['case']}  ref={out['n_ref']} kept={out['n_kept']} "
              f"hit={tp} fp={len(out['fps'])} refused={out['n_refused']}")
        for r in out["rows"]:
            if r["status"] == "hit":
                print(f"  HIT  {r['ref']} d={r['ostium_mm']}mm seed_off={r['seed_offset_mm']}mm "
                      f"ang={r['angle_deg']}deg label@seed={r['seed_label']} "
                      f"r_pred={r['pred_radius']} r_ref={r['ref_radius']} dia_ref={r['diameter']}")
            else:
                print(f"  MISS {r['ref']} dia_ref={r['diameter']} conf={r['confidence']}")
                print(f"       nearest {r['nearest_kind']} at {r['nearest_mm']}mm"
                      f"  reason: {r['nearest_reason']}")
                print(f"       {r['nearest_stats']}")
        for f in out["fps"]:
            print(f"  FP   nearest_ref={f['nearest_ref_mm']}mm seed_label={f['seed_label']} "
                  f"{f['stats']}")
    with open(f"{CACHE}/diag.json", "w") as fh:
        json.dump(report, fh, indent=1, default=float)
