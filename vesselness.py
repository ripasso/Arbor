import SimpleITK as sitk
import numpy as np


def detect_candidate_vessels(
    image: sitk.Image,
    shell_mask: sitk.Image,
    aorta_mask: sitk.Image,
    percentile: float = 15.0,
    min_hu: float = 130.0,
) -> sitk.Image:
    """
    Identifies contrast-filled candidate voxels inside the search shell using an adaptive threshold.
    """
    img_arr = sitk.GetArrayFromImage(image)
    shell_arr = sitk.GetArrayFromImage(shell_mask)
    aorta_arr = sitk.GetArrayFromImage(aorta_mask)

    # Self-calibrate threshold based on aorta blood brightness
    aorta_voxels = img_arr[aorta_arr > 0]
    if aorta_voxels.size > 0:
        adaptive_thresh = max(
            min_hu, float(np.percentile(aorta_voxels, percentile))
        )
    else:
        adaptive_thresh = min_hu

    # Binary condition: inside search shell AND high HU contrast
    candidate_arr = (shell_arr > 0) & (img_arr >= adaptive_thresh)

    candidate_img = sitk.GetImageFromArray(candidate_arr.astype(np.uint8))
    candidate_img.CopyInformation(image)

    # 3D morphological opening to clear isolated single-voxel noise
    cleaned_candidates = sitk.BinaryMorphologicalOpening(
        candidate_img, [1, 1, 1], sitk.sitkBall
    )

    return cleaned_candidates