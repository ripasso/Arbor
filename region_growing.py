import SimpleITK as sitk
import numpy as np


def extract_wall_search_shell(
    aorta_mask: sitk.Image,
    shell_radius_mm: float = 4.0,
    end_cap_margin_mm: float = 5.0,
) -> sitk.Image:
    """
    Creates search shell = Dilate(Aorta) - Aorta.
    Zeroes out top and bottom axial slices to ignore scan crop end-caps.
    """
    spacing = aorta_mask.GetSpacing()  # (dx, dy, dz)
    radius_voxels = [
        max(1, int(np.round(shell_radius_mm / spacing[0]))),
        max(1, int(np.round(shell_radius_mm / spacing[1]))),
        max(1, int(np.round(shell_radius_mm / spacing[2]))),
    ]

    # 1. Morphological Dilation in 3D
    dilated_mask = sitk.BinaryDilate(aorta_mask, radius_voxels, sitk.sitkBall)

    # 2. Subtract parent aorta lumen
    shell_mask = sitk.Subtract(dilated_mask, aorta_mask)

    # 3. Suppress axial end-caps (Z-dimension boundaries)
    shell_arr = sitk.GetArrayFromImage(shell_mask)  # (Z, Y, X)
    aorta_arr = sitk.GetArrayFromImage(aorta_mask)

    z_indices = np.where(aorta_arr > 0)[0]
    if z_indices.size > 0:
        z_min, z_max = z_indices.min(), z_indices.max()
        z_cut_voxels = max(1, int(np.ceil(end_cap_margin_mm / spacing[2])))

        # Mask out top and bottom crop boundaries
        shell_arr[: max(0, z_min + z_cut_voxels), :, :] = 0
        shell_arr[min(shell_arr.shape[0], z_max - z_cut_voxels + 1) :, :, :] = 0

    clean_shell = sitk.GetImageFromArray(shell_arr)
    clean_shell.CopyInformation(aorta_mask)
    return clean_shell