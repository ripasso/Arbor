"""Write the measured scores to out/scores.json for the page to read.

The page's rule is that every number on it is either computed from the outputs
or cited. This keeps the scoring numbers on the right side of that line: they
are produced here from the predictions and the reference, not typed into the
HTML.
"""

from __future__ import annotations

import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

import diag
import score as sc

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def build():
    data = sc.cached()
    overall = sc.score(data=data, detail=True)

    per = []
    for num in sc.CASES:
        rec = data[num]
        kept, _, _ = pl_classify(rec)
        ann = diag.ref_case(num)
        refs = ann["daughters"]
        hits, dists = 0, []
        if refs and kept:
            cost = np.zeros((len(refs), len(kept)))
            for i, r in enumerate(refs):
                ro = np.asarray(r["ostium_xyz_mm"], float)
                for j, c in enumerate(kept):
                    cost[i, j] = np.linalg.norm(ro - np.asarray(c["ostium_xyz_mm"], float))
            ri, ci = linear_sum_assignment(cost)
            for i, j in zip(ri, ci):
                if cost[i, j] <= sc.TOL_MM:
                    hits += 1
                    dists.append(float(cost[i, j]))
        per.append(dict(case_id=f"subject{num:03d}", reference=len(refs),
                        predicted=len(kept), hit=hits,
                        false_positive=len(kept) - hits,
                        missed=len(refs) - hits,
                        median_ostium_mm=round(float(np.median(dists)), 2) if dists else None))

    loo = []
    for held in sc.CASES:
        rest = [c for c in sc.CASES if c != held]
        s = sc.score(data=data, cases=rest)
        loo.append(dict(without=f"subject{held:03d}", f1=s["f1"],
                        precision=s["precision"], recall=s["recall"]))

    payload = dict(
        reference=dict(
            cases=len(sc.CASES),
            instances=sum(p["reference"] for p in per),
            status="draft, expert review pending, not certified exhaustive",
            spacing_mm=1.5,
            policy_min_origin_diameter_mm=2.0,
            match_tolerance_mm=sc.TOL_MM,
        ),
        overall={k: v for k, v in overall.items() if k != "per_case"},
        per_case=per,
        leave_one_out=loo,
        inter_expert_mm=2.5,
    )
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "scores.json"), "w") as fh:
        json.dump(payload, fh, indent=1)
    return payload


def pl_classify(rec):
    import pipeline as pl
    return pl.classify(rec["candidates"], rec["stats"], rec["aorta_length_mm"])


if __name__ == "__main__":
    p = build()
    print(json.dumps(p["overall"], indent=1))
    for row in p["per_case"]:
        print(" ", row)
