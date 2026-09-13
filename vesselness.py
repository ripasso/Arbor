import SimpleITK as sitk
import numpy as np

def detect_candidate_vessels(
    image: sitk.Image,
    shell_mask: sitk.Image,
    aorta_mask: sitk.Image,
    percentile: float = 5.0,     # Safely lowers threshold to the 220-250 HU range
    min_hu: float = 130.0,
) -> sitk.Image:
    """
    Extracts high-contrast structures using a highly sensitive lumen percentile floor.
    Bypasses morphological erosion entirely to protect thin connected vessel branches.
    """
    img_arr = sitk.GetArrayFromImage(image)
    shell_arr = sitk.GetArrayFromImage(shell_mask)
    aorta_arr = sitk.GetArrayFromImage(aorta_mask)

    aorta_voxels = img_arr[aorta_arr > 0]
    if aorta_voxels.size > 0:
        adaptive_thresh = max(
            min_hu, float(np.percentile(aorta_voxels, percentile))
        )
    else:
        adaptive_thresh = min_hu

    candidate_arr = (shell_arr > 0) & (img_arr >= adaptive_thresh)

    candidate_img = sitk.GetImageFromArray(candidate_arr.astype(np.uint8))
    candidate_img.CopyInformation(image)

    # Completely removed BinaryMorphologicalOpening to prevent 1-2 voxel branches from being erased
    return candidate_img
