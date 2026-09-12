import SimpleITK as sitk
import numpy as np


def crop_aorta_roi(image: sitk.Image, mask: sitk.Image, margin_mm: float = 20.0):
    """
    Crops CT image and aorta mask to a tight bounding box + margin_mm.
    Returns (cropped_image, cropped_mask, crop_index_xyz).
    """
    mask_arr = sitk.GetArrayFromImage(mask)  # Shape: (Z, Y, X)
    non_zero = np.argwhere(mask_arr > 0)

    if non_zero.size == 0:
        return image, mask, [0, 0, 0]

    # NumPy returns (z, y, x)
    z_min, y_min, x_min = non_zero.min(axis=0)
    z_max, y_max, x_max = non_zero.max(axis=0)

    spacing = image.GetSpacing()  # (dx, dy, dz)
    margin_x = int(np.ceil(margin_mm / spacing[0]))
    margin_y = int(np.ceil(margin_mm / spacing[1]))
    margin_z = int(np.ceil(margin_mm / spacing[2]))

    size = image.GetSize()  # (X, Y, Z)
    x_start = max(0, int(x_min) - margin_x)
    x_end = min(size[0], int(x_max) + margin_x + 1)
    y_start = max(0, int(y_min) - margin_y)
    y_end = min(size[1], int(y_max) + margin_y + 1)
    z_start = max(0, int(z_min) - margin_z)
    z_end = min(size[2], int(z_max) + margin_z + 1)

    crop_index = [x_start, y_start, z_start]
    crop_size = [x_end - x_start, y_end - y_start, z_end - z_start]

    cropped_image = sitk.RegionOfInterest(image, crop_size, crop_index)
    cropped_mask = sitk.RegionOfInterest(mask, crop_size, crop_index)

    return cropped_image, cropped_mask, crop_index