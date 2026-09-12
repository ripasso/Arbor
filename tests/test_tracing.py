"""
Tests for tracing.py.

Owner: Person 4

Use a synthetic straight-line "centerline" (e.g. a diagonal of known
physical length) to unit-test arc-length and eligibility math before
ostium.py's real output exists.
"""

import pytest


def test_arc_length_matches_known_straight_line():
    pytest.skip("TODO: a straight synthetic centerline of known length should match arc_length_mm")


def test_eligibility_rejects_short_traces():
    pytest.skip("TODO: a centerline shorter than 5mm should fail check_eligibility")


def test_proximal_trim_stops_at_bifurcation():
    pytest.skip("TODO: trimming should stop early if a bifurcation index is before 10mm")
