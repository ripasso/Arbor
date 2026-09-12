"""
Stage I — Required visual checks.

Owner: Person 4

For at least 3 cases, produce a simple figure showing the aorta mask,
detected ostium points, and daughter-direction arrows. This is a sanity
check, not a polished UI -- simplicity is fine and expected.
"""

from __future__ import annotations

import SimpleITK as sitk

from output import Daughter


def plot_case(
    image: sitk.Image,
    mask: sitk.Image,
    daughters: list[Daughter],
    save_path: str,
) -> None:
    """
    Render a simple sanity-check figure (e.g. an axial slice through the
    aorta, or a 3D scatter of the mask surface) with ostium points marked
    and small arrows for each daughter's direction vector. Save to
    `save_path`.
    """
    raise NotImplementedError
