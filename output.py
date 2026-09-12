"""
Stage H — Output assembly.

Owner: Person 1

Responsibilities:
- Assemble the required JSON schema for one case.
- Write it to disk.

Schema-only module -- doesn't depend on any of the vision/algorithm
modules, so it can be built and unit-tested independently and early.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Daughter:
    instance_id: str                                  # e.g. "branch_001"
    parent_instance_id: str                            # always "aorta"
    ostium_xyz_mm: tuple[float, float, float]
    seed_xyz_mm: tuple[float, float, float]
    radius_mm: float
    direction_xyz: tuple[float, float, float]          # unit vector


def build_json(case_id: str, daughters: list[Daughter]) -> dict:
    """
    Assemble the required output dict:
    {
      "case_id": ...,
      "parent": {"instance_id": "aorta"},
      "daughters": [ {...}, ... ]
    }
    An empty `daughters` list is valid (no eligible branches found).
    """
    raise NotImplementedError


def write_json(data: dict, output_path: str) -> None:
    """Write `data` as pretty-printed JSON to `output_path`."""
    raise NotImplementedError
