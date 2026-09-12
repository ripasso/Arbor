"""
Stage I — Required visual checks.

Owner: Person 4

For at least 3 cases, produce a simple figure showing the aorta mask,
detected ostium points, and daughter-direction arrows. This is a sanity
check, not a polished UI -- simplicity is fine and expected.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the 3d projection)

from output import Daughter

_MAX_MASK_POINTS = 20000
_ARROW_LEN_MM = 10.0


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
    # GetArrayFromImage returns (k, j, i) order; flip to (i, j, k) index
    # space, then to physical mm via p = origin + R @ (spacing * index).
    idx_zyx = np.argwhere(sitk.GetArrayFromImage(mask) > 0)
    rng = np.random.default_rng(0)
    if len(idx_zyx) > _MAX_MASK_POINTS:
        idx_zyx = idx_zyx[rng.choice(len(idx_zyx), _MAX_MASK_POINTS, replace=False)]
    idx = idx_zyx[:, ::-1].astype(float)

    spacing = np.asarray(mask.GetSpacing())
    origin = np.asarray(mask.GetOrigin())
    rotation = np.asarray(mask.GetDirection()).reshape(3, 3)
    mask_pts_mm = origin + (idx * spacing) @ rotation.T

    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    if len(mask_pts_mm):
        ax.scatter(
            mask_pts_mm[:, 0], mask_pts_mm[:, 1], mask_pts_mm[:, 2],
            s=1, c="lightgray", alpha=0.3, label="aorta mask",
        )

    ostia = []
    for d in daughters:
        ostium = np.asarray(d.ostium_xyz_mm, dtype=float)
        direction = np.asarray(d.direction_xyz, dtype=float)
        ostia.append(ostium)
        ax.scatter(*ostium, c="red", s=60, depthshade=False)
        ax.quiver(
            *ostium, *(direction * _ARROW_LEN_MM),
            color="blue", arrow_length_ratio=0.25, linewidth=1.5,
        )
        ax.text(*ostium, d.instance_id, fontsize=8, color="red")

    clouds = [p for p in (mask_pts_mm, np.asarray(ostia)) if len(p)]
    if clouds:
        all_pts = np.vstack(clouds)
        lo, hi = all_pts.min(axis=0), all_pts.max(axis=0)
        center, radius = (lo + hi) / 2.0, max((hi - lo).max() / 2.0, 1.0)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.set_title("Aorta mask with ostia and daughter directions")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
