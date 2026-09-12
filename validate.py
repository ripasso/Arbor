"""
Local validation — approximate the official rubric on the dev set.

Owner: Person 4

Independent of the rest of the pipeline (just compares two JSON files) --
good to build early so it's ready to score outputs as soon as any other
module starts producing real predictions.
"""

from __future__ import annotations

from dataclasses import dataclass


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
    raise NotImplementedError


def score_case(pred_json: dict, ref_json: dict, tolerance_mm: float) -> dict:
    """
    Compute precision/recall/F1 (branch discovery, 45%) and mean matched
    ostium distance (ostium localisation, 25%) for a single case.
    """
    raise NotImplementedError


def score_all(pred_dir: str, ref_dir: str, tolerance_mm: float) -> dict:
    """
    Run `score_case` over every case with both a prediction and a
    reference file, and return an aggregate summary (mean/median P, R,
    F1, ostium distance across all dev cases).
    """
    raise NotImplementedError
