"""
Adapted from MAEMI /inference/mae_predict.py
https://github.com/LangDaniel/MAEMI

Given the trained model directory, generate:
- Predicted pseudo-healthy reconstructions
- Anomaly maps

Actually only uses input config to get the device, correct input folder and whether to take
abnormal tau into account during the pseudo-healthy reconstruction.
Otherwise, the config stored within the run dir is used to ensure the correct params are used.
"""

import sys

sys.path.insert(0, "../..")

from inference.slices import SliceIterator

import argparse
from pathlib import Path
from glob import glob
import copy

import torch
import numpy as np
import h5py
from scipy import ndimage as ndi
import yaml
import hydra
from omegaconf import DictConfig
import SimpleITK as sitk

from training.mae3d import models_mae
from utils.io import Args
from data.dataset import MAEDataset
from utils.io import get_output_patch_folder_name
from analysis.analysis_constants import RIDS
from utils.common_utils import rescale_linear


def get_latest_model(input_folder):
    all_models = sorted(glob(f"{input_folder}/*.pth"))
    if not all_models:
        raise FileExistsError(f"No models found to test in {input_folder}")
    checkpoints = np.array(
        [int(x.split("checkpoint-")[-1].replace(".pth", "")) for x in all_models]
    )
    latest_model_ind = checkpoints.argmax()
    print(f"Using checkpoint {checkpoints.max()}")
    return Path(all_models[latest_model_ind])


def uniform_size(data, max_differ=(1, 1, 1), strict=True):
    shape = np.ones(3) * np.inf
    first = True
    for d in data:
        shp = np.minimum(d.shape, shape).astype(int)
        if not first:
            diff = abs(shape - shp)
            if (diff > max_differ).any():
                msg = f"difference {diff} larger than max"
                if strict:
                    raise ValueError(msg)
                print(msg)
        shape = shp
        first = False
    slz = np.s_[: shape[0], : shape[1], : shape[2]]
    out = []
    for d in data:
        out.append(d[slz])
    return out


def get_modality(pid, patch_file, modality):
    data = []
    for mod in modality:
        with h5py.File(patch_file, "r") as ff:
            img = ff[pid][mod][:]
        data.append(img)

    data = uniform_size(data, strict=False)
    data = np.array(data)
    return data


class Predictor:
    def __init__(
        self,
        model,
        patch_file,
        no_norm_patch_file,
        dataset,
        mask_ratio,
        stride,
        repeat,
        filtr,
        device,
        rescale,
        normalise,
        ignore_abnormal_tau=False,
        ignore_abnormal_amyloid=False,
        input_mask=None,
    ):
        self.model = model
        self.patch_file = patch_file
        self.no_norm_patch_file = no_norm_patch_file
        self.dataset = dataset
        self.mask_ratio = mask_ratio
        self.stride = stride
        self.repeat = repeat
        self.filtr = filtr
        self.device = device
        self.rescale = rescale
        self.normalise = normalise
        self.ignore_abnormal_tau = ignore_abnormal_tau
        self.ignore_abnormal_amyloid = ignore_abnormal_amyloid
        self.input_mask = input_mask

    @staticmethod
    def zero_pad(img, shape):
        # zero pad if too small
        padw = np.zeros(img.ndim * 2)
        for ax in range(img.ndim):
            diff = shape[ax] - img.shape[ax]
            if diff < 0:
                continue
            padw[::2][ax] = int(np.ceil(diff / 2))
            padw[1::2][ax] = diff // 2
        padw = padw.reshape(-1, 2).astype(int)
        if padw.any():
            img = np.pad(img, pad_width=padw)
        return img

    def forward(self, img, mask_patch=None):
        x = torch.tensor(img)
        x = x.unsqueeze(dim=0)
        x = x.to(self.device)
        if mask_patch is not None:
            mask_patch = torch.tensor(mask_patch)
            mask_patch = mask_patch.unsqueeze(dim=0)
            mask_patch = mask_patch.to(self.device)

        loss, y, mask = self.model(
            x.float(),
            mask_ratio=self.mask_ratio,
            input_mask=mask_patch,
        )
        y = self.model.unpatchify(y)
        y = y.detach().cpu()

        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, int(np.prod(self.model.patch_size)))
        mask = self.model.unpatchify(mask)
        return y.cpu().numpy(), mask.cpu().numpy()

    @staticmethod
    def inverse_rescale(img, scaling):
        if len(img.shape):
            img = np.expand_dims(img, axis=0)
        for mod in range(img.shape[0]):
            [in_low, in_high, out_low, out_high] = scaling[mod]

            m = (out_high - out_low) / (in_high - in_low)
            b = out_low - m * in_low
            img[mod] = (img[mod] - b) / m
        return np.squeeze(img)

    def compute_anomaly(self, img_, no_norm_img, mask_patch_=None):
        # TO DO: make reverting to original scale work for normalisation too
        y, mask = self.forward(img_, mask_patch_)

        if self.rescale:
            y_orig_scale = self.inverse_rescale(y.squeeze(), np.array(self.rescale))
        if self.normalise:
            raise NotImplementedError

        diff = (no_norm_img.squeeze() - y_orig_scale) ** 2
        for ii in range(diff.shape[0]):
            if len(diff[ii].shape) == 2:
                diff[ii] = self.filtr(diff[ii])
            elif len(diff[ii].shape) == 3:
                for jj in range(diff.shape[1]):
                    diff[ii, jj, :] = self.filtr(diff[ii, jj, :])

        return diff, y_orig_scale, mask.squeeze()

    def one_iteration(self, image, no_norm_img, patch_size, stride, input_mask=None):
        """
        image: [C, W, H, D]
        patch_size: [W, H, D]
        stride: [W, H, D]
        """
        diff = np.zeros(image.shape)
        pred = np.zeros(image.shape)
        count = np.ones(image.shape)

        shape = list(image.shape[-3:])

        img_iter = SliceIterator(shape, patch_size, stride)

        for slz in img_iter:
            slz = (slice(None), *slz)
            if input_mask is not None:
                mask_patch = input_mask[slz]
            else:
                mask_patch = None
            diff_current_slz, y, m = self.compute_anomaly(
                image[slz], no_norm_img[slz], mask_patch
            )

            diff[slz] += np.multiply(diff_current_slz, m)
            pred[slz] += np.multiply(y, m)
            count[slz] += m

        # --------------------------
        diff = np.divide(diff, count, out=np.zeros_like(diff), where=count != 0)
        pred = np.divide(pred, count, out=np.zeros_like(pred), where=count != 0)

        return diff, pred, count

    def get_anomaly_map(self, pid, modality):
        img = self.dataset.get_modality(pid, self.patch_file, modality)
        no_norm_img = self.dataset.get_modality(pid, self.no_norm_patch_file, modality)

        # zero pad
        iz = self.model.img_size
        shape = [self.model.in_chans, iz[0] + 1, iz[1] + 1, iz[2] + 1]
        img = self.zero_pad(img, shape)

        diff = np.zeros(img.shape)
        pred = np.zeros(img.shape)
        count = np.zeros(img.shape)
        for ii in range(self.repeat):
            d, p, c = self.one_iteration(
                img, no_norm_img, self.model.img_size, self.stride, self.input_mask
            )
            diff += d
            pred += p
            count += 1
        diff /= count
        pred /= count

        return diff, pred, count, img


def main_wrapper(run_dir):
    @hydra.main(config_path="../config", config_name="config", version_base=None)
    def _main(cfg: DictConfig) -> None:
        print("Generating MAE predictions")

        modality_dir = get_output_patch_folder_name(cfg)
        model_dir = Path(modality_dir, run_dir)
        print(model_dir)
        config_file = glob(f"{model_dir}/config.yaml")[0]
        with open(config_file, "r") as ff:
            par = ff.read()
            args = yaml.safe_load(par)

        ckpt_file = get_latest_model(model_dir)
        trained_dir = ckpt_file.parent
        ckpt = ckpt_file.name
        print(ckpt)

        args["data"]["train"]["training"] = False
        args["data"]["general"]["augment"] = False
        args["data"]["general"]["pathology"] = "anomaly"
        args["data"]["general"]["margin"] = [0, 0, 0]
        train_data_args = Args({**args["data"]["general"], **args["data"]["train"]})

        rescale = args["patches"]["rescale"]
        normalise = args["patches"]["normalise"]

        data_args = Args({**args["data"]["general"], **args["data"]["train"]})
        data_args.split_file = Path(cfg.device.project_dir, data_args.split_file)
        data_args.patch_file = Path(modality_dir, data_args.patch_file)
        data_args.modalities = args["patches"]["modalities"]
        modalities = copy.deepcopy(args["patches"]["modalities"])
        if "GM_mask" in modalities:
            modalities.remove("GM_mask")
        modality_str = "_".join(modalities)
        data_args.no_norm_patch_file = Path(
            cfg.device.project_dir, modality_str, data_args.no_norm_patch_file
        )

        device = args["model"]["device"]
        ignore_abnormal_tau = args["model"][
            "ignore_abnormal_tau_inf"
        ] and "tauPET" in "\t".join(data_args.modalities)
        ignore_abnormal_amyloid = (
            args["model"]["ignore_abnormal_amyloid"]
            and "amyloidPET" in data_args.modalities
        )
        stride = args["data"]["inference"]["stride"]
        repeat = args["data"]["inference"]["repeat"]
        min_filter_size = args["data"]["inference"]["min_filter_size"]
        filtr = lambda img: ndi.minimum_filter(img, size=tuple(min_filter_size))
        filtr_str = "min_" + "x".join(str(x) for x in min_filter_size)

        model_args = Args(args["model"])
        model_args.device = device
        model_args.input_size = train_data_args.shape

        if ignore_abnormal_amyloid:
            train_data_args.input_mask_file = Path(
                cfg.device.adni_dir, train_data_args.cerebellum_mask
            )
        else:
            train_data_args.input_mask_file = None

        if ignore_abnormal_tau:
            tau_threshold = rescale_linear(
                model_args.tau_threshold, cfg.patches.rescale[0]
            )
        else:
            tau_threshold = None

        # load model
        model = models_mae.__dict__[model_args.model](
            img_size=model_args.input_size,
            patch_size=model_args.patch_size,
            norm_pix_loss=model_args.norm_pix_loss,
            in_chans=args["model"]["in_chans"],
        )

        checkpoint = torch.load(trained_dir / ckpt, map_location=device)
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(msg)

        model.ignore_abnormal_tau = ignore_abnormal_tau
        model.tau_threshold = tau_threshold
        print("Ignore abnormal tau:", ignore_abnormal_tau)
        print("tau threshold: ", tau_threshold)

        _ = model.to(device)

        stride_str = "stride-" + "x".join([str(ff) for ff in stride])
        info_str = f"predictions_{stride_str}_repeat-{repeat}_fltr-{filtr_str}"
        if ignore_abnormal_tau:
            info_str = info_str + "_ignoreAbnormalTau"
        if ignore_abnormal_amyloid:
            info_str = info_str + "_ignoreAbnormalAmyloid"
            input_mask_file = Path(
                cfg.device.adni_dir, args["model"]["cerebellum_mask"]
            )
            data_args.input_mask_file = input_mask_file
            ds = sitk.ReadImage(input_mask_file)
            input_mask = sitk.GetArrayFromImage(ds)
            input_mask = np.transpose(input_mask, (2, 1, 0))
            input_mask = np.expand_dims(input_mask, axis=0)
        else:
            input_mask = None
            data_args.input_mask_file = None

        is_external_dataset = args["data"]["general"].get(
            "is_external_dataset", False
        )
        if is_external_dataset:
            splits = [None]
        else:
            splits = ["train", "val", "test"]

        for split in splits:
            dataset = MAEDataset(data_args, split=split)
            dataset.get_modality = get_modality

            predictor = Predictor(
                model=model,
                patch_file=data_args.patch_file,
                no_norm_patch_file=data_args.no_norm_patch_file,
                dataset=dataset,
                mask_ratio=args["model"]["mask_ratio"],
                stride=stride,
                repeat=repeat,
                filtr=filtr,
                device=device,
                rescale=rescale,
                normalise=normalise,
                ignore_abnormal_tau=ignore_abnormal_tau,
                ignore_abnormal_amyloid=ignore_abnormal_amyloid,
                input_mask=input_mask,
            )

            if split is None:
                output_file = Path(trained_dir, info_str, "pred.h5")
                split_name = "external"
            else:
                output_file = Path(
                    trained_dir,
                    info_str,
                    f"pred_{split}.h5"
                )
                split_name = split

            print(f"\nProcessing {split_name}")
            print(f"Number of subjects: {len(dataset.label_df)}")

            pids = dataset.label_df["identifier"]
            for pid_ind, pid in enumerate(pids):
                # if pid.split("sub-")[1][:4] in RIDS:
                if True:
                    print(f"[{pid_ind+1}/{len(pids)}]: {pid}")
                    with h5py.File(output_file, "a") as ff:
                        if pid in ff.keys():
                            if "diff" in ff[pid].keys() and "pred" in ff[pid].keys():
                                print("Already run, skipping.")
                                continue

                    diff, pred, count, ref = predictor.get_anomaly_map(
                        pid, data_args.modalities
                    )
                    bbox = dataset.get_modality(
                        pid, data_args.patch_file, data_args.modalities
                    )

                    with h5py.File(output_file, "a") as ff:
                        try:
                            ff.create_dataset(f"{pid}/diff", data=diff)
                            ff.create_dataset(f"{pid}/pred", data=pred)
                        except (Exception, ArithmeticError) as e:
                            raise ValueError(f"{pid} failed: {e}")

        # ---------------------------------------------------------
        # Create unified link file
        # ---------------------------------------------------------
        if not is_external_dataset:
            link_file = Path(trained_dir, info_str, "pred.h5")
            with h5py.File(link_file, "w") as f:
                for split in ["train", "val", "test"]:
                    split_file = Path(
                        trained_dir,
                        info_str,
                        f"pred_{split}.h5"
                    )

                    if not split_file.exists():
                        print(f"Warning: {split_file} does not exist, skipping.")
                        continue

                    with h5py.File(split_file, "r") as split_h5:
                        for pid in split_h5.keys():
                            f[pid] = h5py.ExternalLink(
                                split_file.name,
                                f"/{pid}",
                            )

            print(f"Created link file: {link_file}")
        print("Finished mae predict")

    _main()


if __name__ == "__main__":
    run_dir = "run0"
    main_wrapper(run_dir)
