"""
Local validation — approximate the official rubric on the dev set.

Owner: Person 4

Independent of the rest of the pipeline (just compares two JSON files) --
good to build early so it's ready to score outputs as soon as any other
module starts producing real predictions.
"""

from __future__ import annotations

import os
import json
import glob
import math
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class MatchResult:
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ostium_distances_mm: list[float]


def match_predictions_to_references(
    pred_json: dict,
    ref_json: dict,
    tolerance_mm: float,
) -> MatchResult:
    """
    One-to-one bipartite matching between predicted and reference
    daughters based on ostium_xyz_mm distance, within `tolerance_mm`.
    Unmatched predictions = false positives; unmatched references =
    false negatives.
    """
    pred_daughters = pred_json.get("daughters", [])
    ref_daughters = ref_json.get("daughters", [])

    if not pred_daughters and not ref_daughters:
        return MatchResult(0, 0, 0, [])
    if not pred_daughters:
        return MatchResult(0, 0, len(ref_daughters), [])
    if not ref_daughters:
        return MatchResult(0, len(pred_daughters), 0, [])

    # Create cost matrix for bipartite matching using Euclidean distance
    cost_matrix = np.zeros((len(pred_daughters), len(ref_daughters)))
    for i, p in enumerate(pred_daughters):
        for j, r in enumerate(ref_daughters):
            p_xyz = p["ostium_xyz_mm"]
            r_xyz = r["ostium_xyz_mm"]
            cost_matrix[i, j] = math.sqrt(sum((a - b) ** 2 for a, b in zip(p_xyz, r_xyz)))

    # Optimal 1-to-1 matching to prevent duplicate detections counting as TPs
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    tp, matched_distances = 0, []
    matched_preds, matched_refs = set(), set()

    for r, c in zip(row_ind, col_ind):
        dist = cost_matrix[r, c]
        if dist <= tolerance_mm:
            tp += 1
            matched_distances.append(dist)
            matched_preds.add(r)
            matched_refs.add(c)

    fp = len(pred_daughters) - len(matched_preds)
    fn = len(ref_daughters) - len(matched_refs)

    return MatchResult(tp, fp, fn, matched_distances)


def score_case(pred_json: dict, ref_json: dict, tolerance_mm: float) -> dict:
    """
    Compute precision/recall/F1 (branch discovery, 45%) and mean matched
    ostium distance (ostium localisation, 25%) for a single case.
    """
    match_res = match_predictions_to_references(pred_json, ref_json, tolerance_mm)
    
    tp = match_res.true_positives
    fp = match_res.false_positives
    fn = match_res.false_negatives
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    mean_dist = sum(match_res.matched_ostium_distances_mm) / tp if tp > 0 else 0.0
    
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_ostium_dist_mm": round(mean_dist, 4),
        "tp": tp, "fp": fp, "fn": fn
    }


def score_all(pred_dir: str, ref_dir: str, tolerance_mm: float) -> dict:
    """
    Run `score_case` over every case with both a prediction and a
    reference file, and return an aggregate summary (mean/median P, R,
    F1, ostium distance across all dev cases).
    """
    pred_files = glob.glob(os.path.join(pred_dir, "*.json"))
    all_metrics = []
    
    for p_file in pred_files:
        filename = os.path.basename(p_file)
        r_file = os.path.join(ref_dir, filename)
        
        if os.path.exists(r_file):
            with open(p_file, 'r') as f:
                pred_json = json.load(f)
            with open(r_file, 'r') as f:
                ref_json = json.load(f)
            
            all_metrics.append(score_case(pred_json, ref_json, tolerance_mm))
            
    if not all_metrics:
        return {"error": "No matching JSON files found in the provided directories."}
        
    avg_f1 = sum(m["f1"] for m in all_metrics) / len(all_metrics)
    cases_with_tp = [m for m in all_metrics if m["tp"] > 0]
    avg_dist = sum(m["mean_ostium_dist_mm"] for m in cases_with_tp) / max(1, len(cases_with_tp))
    
    return {
        "cases_evaluated": len(all_metrics),
        "average_f1": round(avg_f1, 4),
        "average_ostium_distance_mm": round(avg_dist, 4)
    }


# Standalone CLI execution block for immediate local testing
if __name__ == "__main__":
    import sys
    # Direct evaluation shortcut: python validate.py <prediction.json> <reference.json>
    if len(sys.argv) == 3:
        p_path, r_path = sys.argv[1], sys.argv[2]
        if os.path.exists(p_path) and os.path.exists(r_path):
            with open(p_path) as f1, open(r_path) as f2:
                print(json.dumps(score_case(json.load(f1), json.load(f2), tolerance_mm=5.0), indent=2))
