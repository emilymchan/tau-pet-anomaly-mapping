import os
import SimpleITK as sitk
import numpy as np


def get_list_of_dirs(target_dir: str, full_path: bool = True) -> list:
    if full_path:
        dirs = sorted(
            [
                os.path.join(target_dir, d)
                for d in os.listdir(target_dir)
                if (
                    os.path.isdir(os.path.join(target_dir, d)) and not d.startswith(".")
                )
            ]
        )
    elif not full_path:
        dirs = sorted(
            [
                d
                for d in os.listdir(target_dir)
                if (
                    os.path.isdir(os.path.join(target_dir, d)) and not d.startswith(".")
                )
            ]
        )

    return dirs


def read_nii_file(path, orientation=None):
    data = sitk.ReadImage(path)
    if orientation is not None:
        data = sitk.DICOMOrient(data, orientation)
    return data


def get_array_from_itk(image):
    data = sitk.GetArrayFromImage(image)
    if len(data.shape) != 3:
        data = data.squeeze()
    data = np.transpose(data, (2, 1, 0))
    return data


def rescale_linear(data, scaling):
    [in_low, in_high, out_low, out_high] = scaling
    m = (out_high - out_low) / (in_high - in_low)
    b = out_low - m * in_low
    data = m * data + b
    data = np.clip(data, out_low, out_high)
    return data


def scale_to_orig(X, X_min, X_max, clip_lower=None):
    X_rescale = (X * (X_max - X_min)) + X_min
    if clip_lower is not None:
        X_rescale = np.clip(X_rescale, a_min=clip_lower, a_max=None)
    return X_rescale


def get_brain_masks(mask_file, schaefer_file):
    # Whole brain, grey matter (GM) mask
    mask_sitk = read_nii_file(mask_file)
    mask_array = get_array_from_itk(mask_sitk)
    mask_array = np.round(mask_array)
    mask_array = mask_array.squeeze()
    wholeBrain_mask = (mask_array > 0).astype(int)
    GM_mask = (mask_array == 2).astype(int)

    # Cortical GM mask
    schaefer_sitk = read_nii_file(schaefer_file)
    schaefer_array = get_array_from_itk(schaefer_sitk)
    schaefer_binary = (schaefer_array > 0).astype(int)
    GM_cortical_mask = np.multiply(GM_mask, schaefer_binary)

    return wholeBrain_mask, GM_mask, GM_cortical_mask, schaefer_binary
