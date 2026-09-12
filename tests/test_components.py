"""
Tests for components.py.

Owner: Person 3

Use a small synthetic binary numpy array (not real CT data) to unit-test
labeling and adjacency logic in isolation -- this is the stub Person 3
can build against before region_growing.py is ready.
"""

import pytest


def test_label_components_separates_disjoint_blobs():
    pytest.skip("TODO: two disjoint synthetic blobs should get two labels")


def test_filter_touching_aorta_keeps_adjacent_only():
    pytest.skip("TODO: a blob touching a synthetic aorta mask should pass; a floating one should not")
