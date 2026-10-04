import argparse
import logging
import os
from time import sleep
from pathlib import Path
import copy

import yaml


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_file", type=str, default="config.yaml")
    args = parser.parse_args()

    return args


def get_param_files(path):
    path = Path(path)
    if path.is_dir():
        files = sorted([ff for ff in path.iterdir() if ff.suffix == ".yml"])
    elif path.suffix == ".yml":
        files = [path]
    else:
        raise ValueError("non valid parameter files (must be folder or .yml)")
    return files


def get_config(config_path: str) -> dict:
    with open(config_path) as file:
        config = yaml.safe_load(file)
    return config


def set_logger(log_path, output="file+console"):
    """
    output options - 'file', 'console' or 'file+console'
    """
    if not "file" in output and not "console" in output:
        raise NotImplementedError

    logger = logging.getLogger(os.path.basename(log_path))
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        if "file" in output:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s:%(levelname)s: %(message)s")
            )
            logger.addHandler(file_handler)

        if "console" in output:
            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(stream_handler)
    sleep(1)  # ensure that filenames based on DateTime are not identical

    return logger


class Args:
    """convert dict to class variables (yaml to parser like)"""

    def __init__(self, d):
        for key, value in d.items():
            if type(value) == dict:
                value = Args(value)
            if value == "None":
                value = None
            setattr(self, key, value)

    def __str__(self):
        return str(self.__dict__)


def copy_files(files, path):
    path = Path(path)
    for f in files:
        if isinstance(f, list):
            name = Path(f[1]).name
            content = Path(f[0]).read_text()
        else:
            name = Path(f).name
            content = Path(f).read_text()
        with open(path / name, "w") as dest:
            dest.write(content)
        (path / name).chmod(0o444)


def get_output_patch_folder_name(cfg):
    rescale = cfg.patches.rescale
    normalise = cfg.patches.normalise
    spacing = cfg.patches.spacing
    modalities = copy.deepcopy(cfg.patches.modalities)

    # If using input segmentation of GM
    if "GM_mask" in modalities:
        modalities.remove("GM_mask")
    if "GM_mask_cortical" in modalities:
        modalities.remove("GM_mask_cortical")

    space_str = "x".join([str(ii) for ii in spacing]).replace(".", "")
    modality_str = "_".join(modalities)  # e.g. ['tau', 'mr'] -> tau_mr
    output_folder = Path(cfg.device.project_dir, modality_str)

    if rescale:
        remove_char = set(".[]")
        rescale_str = "".join(str(c) for c in str(rescale) if c not in remove_char)
        rescale_str = rescale_str.split(", ")
        output_str = "rescale"
        for m in range(len(modalities)):
            in_low = rescale_str[0 + 4 * m]
            in_high = rescale_str[1 + 4 * m]
            out_low = rescale_str[2 + 4 * m]
            out_high = rescale_str[3 + 4 * m]
            output_str += f"_in{in_low}_{in_high}_out{out_low}_{out_high}"
        output_str += f"_{space_str}"
        output_folder = Path(output_folder, output_str)

    elif normalise:  # Doesn't work for multimodal
        mean = normalise[0]
        std = normalise[1]
        mean_str = str(mean).replace(".", "")
        std_str = str(std).replace(".", "")
        output_folder = Path(
            output_folder, f"normalised_m{mean_str}_std{std_str}_{space_str}"
        )

    return output_folder
