"""
Adapted from ci-ber/2024-miccai-bercea-thor/projects/thor/DDPMTrainer.py
Implementation of the abstract class "Trainer" from core module.
Includes training and validation routines
"""

from zmq import device
from core.Trainer import Trainer
from time import time
import os
from scipy import ndimage as ndi
import wandb
import logging
import matplotlib.pyplot as plt
from net_utils.simplex_noise import generate_noise
from optim.losses.image_losses import *
from torch.cuda.amp import GradScaler, autocast
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
from torchmetrics.image import PeakSignalNoiseRatio as psnr
from torch.utils.data import DataLoader
from analysis.analysis_constants import RIDS
import h5py
from pathlib import Path


def print_mem(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[{tag}] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def generate_patch_slices(vol_shape, patch_size, stride):
    xs = list(range(0, vol_shape[0] - patch_size[0] + 1, stride[0]))
    ys = list(range(0, vol_shape[1] - patch_size[1] + 1, stride[1]))
    zs = list(range(0, vol_shape[2] - patch_size[2] + 1, stride[2]))

    if xs[-1] != vol_shape[0] - patch_size[0]:
        xs.append(vol_shape[0] - patch_size[0])
    if ys[-1] != vol_shape[1] - patch_size[1]:
        ys.append(vol_shape[1] - patch_size[1])
    if zs[-1] != vol_shape[2] - patch_size[2]:
        zs.append(vol_shape[2] - patch_size[2])

    for x in xs:
        for y in ys:
            for z in zs:
                yield np.s_[x:x+patch_size[0],
                            y:y+patch_size[1],
                            z:z+patch_size[2]]



class Predictor:
    def __init__(
        self,
        model,
        patch_file,
        no_norm_patch_file,
        dataset,
        stride,
        repeat,
        filtr,
        device,
        rescale,
        normalise
    ):
        self.model = model
        self.patch_file = patch_file
        self.no_norm_patch_file = no_norm_patch_file
        self.dataset = dataset
        self.stride = stride
        self.repeat = repeat
        self.filtr = filtr
        self.device = device
        self.rescale = rescale
        self.normalise = normalise
        self.h5_patch = h5py.File(patch_file, "r")
        self.h5_no_norm = h5py.File(no_norm_patch_file, "r")

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

    
    def forward_batch(self, imgs):
        """
        imgs: numpy array [B, C, W, H, D]
        """
        x = torch.from_numpy(imgs).to(self.device)

        x = (x * 2) - 1
        x = x.float()

        with torch.no_grad(), torch.autocast("cuda"):
            y, _ = self.model.sample_from_image(
                x, noise_level=self.model.noise_level_recon
            )

        y = (y + 1) / 2
        return y.detach().cpu().numpy()


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
        y = self.forward(img_)

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

        return diff, y_orig_scale
    

    def one_iteration(self, image, no_norm_img, patch_size, stride, input_mask=None):

        diff = np.zeros_like(image)
        pred = np.zeros_like(image)
        count = np.zeros_like(image)

        shape = image.shape[-3:]
        batch_size = 16

        patches = []
        patches_no_norm = []
        coords = []

        for slz in generate_patch_slices(shape, patch_size, stride):

            slz_full = (slice(None), *slz)

            patches.append(image[slz_full])
            patches_no_norm.append(no_norm_img[slz_full])
            coords.append(slz_full)

            if len(patches) == batch_size:

                patches_np = np.stack(patches)
                preds = self.forward_batch(patches_np)

                for i in range(len(coords)):

                    y = preds[i]

                    if self.rescale:
                        y_orig = self.inverse_rescale(
                            y.squeeze(), np.array(self.rescale)
                        )
                    else:
                        y_orig = y.squeeze()

                    diff_current = (patches_no_norm[i].squeeze() - y_orig) ** 2

                    diff[coords[i]] += diff_current
                    pred[coords[i]] += y_orig
                    count[coords[i]] += 1

                patches = []
                patches_no_norm = []
                coords = []

        # leftover batch
        if len(patches) > 0:

            patches_np = np.stack(patches)
            preds = self.forward_batch(patches_np)

            for i in range(len(coords)):

                y = preds[i]

                if self.rescale:
                    y_orig = self.inverse_rescale(
                        y.squeeze(), np.array(self.rescale)
                    )
                else:
                    y_orig = y.squeeze()

                diff_current = (patches_no_norm[i].squeeze() - y_orig) ** 2

                diff[coords[i]] += diff_current
                pred[coords[i]] += y_orig
                count[coords[i]] += 1

        diff /= np.maximum(count, 1)
        pred /= np.maximum(count, 1)

        return diff, pred, count


    def get_anomaly_map(self, pid, modality):
        img = get_modality(pid, self.h5_patch, modality)
        no_norm_img = get_modality(pid, self.h5_no_norm, modality)

        iz = [72, 72, 72]
        shape = [1, iz[0] + 1, iz[1] + 1, iz[2] + 1]
        img = self.zero_pad(img, shape)

        diff = np.zeros(img.shape)
        pred = np.zeros(img.shape)
        count = np.zeros(img.shape)
        for ii in range(self.repeat):
            d, p, c = self.one_iteration(
                img, no_norm_img, [72, 72, 72], self.stride, None
            )
            diff += d
            pred += p
            count += 1
        diff /= count
        pred /= count

        return diff, pred, count, img


class PPredictor(Trainer):
    def __init__(self, training_params, model, data, device, log_wandb=True):
        super(PPredictor, self).__init__(training_params, model, data, device, log_wandb)
        self.val_interval = training_params["val_interval"]
        self.patch_file = Path(*training_params["patch_file"])
        self.no_norm_patch_file = Path(*training_params["no_norm_patch_file"])

    def predict(self, model_weights, test_data, task="test", weight_dir=None, epoch=0):
        """
        :param model_weights: weights of the global model
        :return: dict
            metric_name : values
            e.g.:
             metrics = {
                'test_loss_rec': 0,
                'test_total': 0
            }
        """ 
        self.test_model.load_state_dict(model_weights)
        self.test_model.to(self.device)
        self.test_model.eval() 

        modality = ['tauPET'] # CHANGE

        weight_dir = Path(weight_dir).parent
        output_file = Path(weight_dir, f"pred_{task}.h5")
        logging.info(f"Saving predictions to {output_file}")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        patch_file = self.patch_file
        no_norm_patch_file = self.no_norm_patch_file
        
        filtr = lambda img: ndi.minimum_filter(img, size=tuple([3, 3]))
        patch_predictor = Predictor(
                model=self.test_model,
                patch_file=patch_file,
                no_norm_patch_file=no_norm_patch_file,
                dataset=test_data.dataset,
                stride=[20, 20, 20],
                repeat=3,
                filtr=filtr,
                device=self.device,
                rescale=[[0, 5, 0, 1]],
                normalise=False
            )

        pids = test_data.dataset.label_df["identifier"]
        self.h5 = h5py.File(patch_file, "r")
        with torch.no_grad():
            for pid_ind, pid in enumerate(pids):
                if True:   #pid.split("sub-")[1][:4] in RIDS:
                    logging.info(f"[{pid_ind+1}/{len(pids)}]: {pid}")
                    with h5py.File(output_file, "a") as ff:
                        if pid in ff.keys():
                            if "diff" in ff[pid].keys() and "pred" in ff[pid].keys():
                                logging.info("Already run, skipping.")
                                continue
                
                    diff, pred, count, ref = patch_predictor.get_anomaly_map(
                        pid, modality
                    )

                    with h5py.File(output_file, "a") as ff:
                        try:
                            ff.create_dataset(f"{pid}/diff", data=diff)
                            ff.create_dataset(f"{pid}/pred", data=pred)
                        except (Exception, ArithmeticError) as e:
                            raise ValueError(f"{pid} failed: {e}")


    def inverse_rescale(self, img, scaling):
        [in_low, in_high, out_low, out_high] = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        img = (img - b) / m

        return img
    

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


def get_modality(pid, h5_file, modality):

    data = []

    if isinstance(modality, str):
        modality = [modality]

    for mod in modality:
        img = h5_file[pid][mod][:]
        data.append(img)

    data = uniform_size(data, strict=False)
    data = np.array(data)

    return data