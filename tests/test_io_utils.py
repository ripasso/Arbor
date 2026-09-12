"""
Tests for io_utils.py.

Owner: Person 1

TODO: point these at a couple of real sample cases from the dataset
(one plain .nii, one gzip-disguised .nii from subjects 016-025) once
they're copied into ../data/.
"""

import pytest


def test_is_gzip_detects_plain_nii():
    pytest.skip("TODO: implement once io_utils.is_gzip exists")


def test_is_gzip_detects_disguised_gzip():
    pytest.skip("TODO: test against a known gzip-disguised subject (016-025)")


def test_load_case_matching_geometry():
    pytest.skip("TODO: verify image/mask share size/spacing/origin/direction")


def test_voxel_to_mm_matches_sitk():
    pytest.skip("TODO: cross-check against image.TransformIndexToPhysicalPoint directly")
