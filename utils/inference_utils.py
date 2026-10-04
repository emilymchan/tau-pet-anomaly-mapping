from pathlib import Path
from glob import glob
import numpy as np


def get_latest_run_dir(input_folder):
    all_run_dirs = sorted(glob(f"{input_folder}/run_*"))
    if not all_run_dirs:
        raise FileExistsError(f"No run_ folders found in {input_folder}")
    latest_dir = Path(all_run_dirs[-1])
    print(f"Using {latest_dir}")
    return latest_dir


def dice_coefficient(y_true, y_pred):
    intersection = np.nansum(y_true * y_pred)
    return (2.0 * intersection) / (np.nansum(y_true) + np.nansum(y_pred))

