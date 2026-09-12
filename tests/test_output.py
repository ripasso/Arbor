"""
Tests for output.py.

Owner: Person 1

Schema-only tests -- these can be written and passing before any other
module exists.
"""

import pytest


def test_build_json_matches_required_schema():
    pytest.skip("TODO: check keys case_id/parent/daughters and each daughter's required fields")


def test_empty_daughters_list_is_valid():
    pytest.skip("TODO: build_json with zero daughters should still produce valid schema")


def test_direction_is_unit_vector():
    pytest.skip("TODO: enforce/validate direction_xyz has length ~1.0 before writing")
