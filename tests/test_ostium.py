"""
Tests for ostium.py.

Owner: Person 3
"""

import pytest


def test_contact_centroid_is_averaged():
    pytest.skip("TODO: centroid of a synthetic contact patch should equal its mean index")


def test_cropped_face_rejected():
    pytest.skip("TODO: a flat synthetic patch at the mask's top z-slice should be flagged")


def test_close_but_distinct_ostia_not_merged():
    pytest.skip("TODO: two separate synthetic components near each other should both survive dedup")


def test_forking_trunk_counted_once():
    pytest.skip("TODO: one component touching wall once, forking downstream -> single instance")
