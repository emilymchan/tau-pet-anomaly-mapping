"""
Adapted from MAEMI /training/train.py
https://github.com/LangDaniel/MAEMI

Trains the MAEMI model.
Takes in a config file of the arguments.
"""

from pathlib import Path

import hydra
import omegaconf
from omegaconf import DictConfig, open_dict

import sys

sys.path.insert(0, "../../..")

from utils.train_utils import get_output_folder
from utils.io import get_output_patch_folder_name
from data.dataset import MAEDataset
from MAEMI.training.mae3d import main_pretrain

from utils.common_utils import rescale_linear


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    sub = "run"
    model_args = cfg.model
    output_folder = get_output_patch_folder_name(cfg)

    train_data_args = DictConfig({**cfg.data.general, **cfg.data.train})
    train_data_args.modalities = cfg.patches.modalities
    train_data_args.patch_file = Path(output_folder, cfg.data.general.patch_file)
    train_data_args.split_file = Path(
        cfg.device.project_dir, cfg.data.general.split_file
    )

    if model_args.ignore_abnormal_amyloid:
        train_data_args.input_mask_file = Path(
            cfg.device.adni_dir, train_data_args.cerebellum_mask
        )
    else:
        train_data_args.input_mask_file = None

    dataset_train = MAEDataset(train_data_args, "train")

    # Only works if tau is first imaging modality
    if model_args.ignore_abnormal_tau_training:
        tau_threshold = rescale_linear(model_args.tau_threshold, cfg.patches.rescale[0])
    else:
        tau_threshold = None

    with open_dict(model_args):
        model_args.output_dir = get_output_folder(output_folder, sub)
        model_args.log_dir = model_args.output_dir
        model_args.input_size = train_data_args.shape
        model_args.tau_threshold = tau_threshold

    if model_args.output_dir:
        Path(model_args.output_dir).mkdir(parents=True, exist_ok=True)
    omegaconf.OmegaConf.save(cfg, Path(model_args.output_dir, "config.yaml"))

    main_pretrain.main(model_args, dataset_train)


if __name__ == "__main__":
    main()
