"""
Adapted from MAEMI /training/sequence.py
https://github.com/LangDaniel/MAEMI
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
import pandas as pd
import h5py
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from torch.utils.data import Dataset
import torch


class MAEDataset(Dataset):
    def __init__(self, args, split="train", with_label=False):
        if split not in ["train", "val", "test", None]:
            raise ValueError(
                'Expecting train, val, test or None for variable "split"'
            )

        self.args = args
        self.with_label = with_label

        patch_file = Path(args.patch_file)
        with h5py.File(patch_file, "r") as ff:
            all_pids = list(ff.keys())

        if split is None:
            # External dataset: use all subjects in the patch file
            label_df = pd.DataFrame({"identifier": all_pids})
        else:
            split_file = Path(args.split_file)
            label_df = pd.read_csv(split_file)
            label_df = label_df[label_df["split"] == split]

        self.label_df = label_df.reset_index(drop=True)

        if split != "train":
            self.args.training = False

        self.total_count = len(self.label_df)
        if not self.total_count:
            raise ValueError("no cases were found: check label naming")
        print(f"len {self.total_count}")

        self.ignore_abnormal_amyloid = args.ignore_abnormal_amyloid
        if self.ignore_abnormal_amyloid:
            self.input_mask_file = args.input_mask_file

    # augmentation
    # ----------------------------------------------------------------

    @staticmethod
    def rescale_linear(data, scaling):
        in_low, in_high, out_low, out_high = scaling

        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        data = m * data + b
        data = np.clip(data, out_low, out_high)
        return data

    @staticmethod
    def add_gaussian_noise(data, noise_var):
        variance = np.random.uniform(noise_var[0], noise_var[1])
        data = data + np.random.normal(0.0, variance, size=data.shape)

        return data

    @staticmethod
    def gaussian_blur(data, sigma):
        sig = np.random.uniform(sigma[0], sigma[1])
        return gaussian_filter(data, sig)

    @staticmethod
    def add_offset(data, var):
        off = np.random.uniform(var[0], var[1])
        data = data + off

        return data

    @staticmethod
    def zoom_to_fixed_size(image, zoom_factor):
        original_shape = np.array(image.shape)
        zoomed_image = zoom(
            image, zoom_factor, order=1
        )  # Use bilinear interpolation (order=1)

        # Get the new shape after zoom
        zoomed_shape = np.array(zoomed_image.shape)

        # If zoomed in (zoom_factor > 1), crop the center
        if zoom_factor > 1:
            crop_start = (zoomed_shape - original_shape) // 2
            crop_end = crop_start + original_shape
            slices = tuple(
                slice(start, end) for start, end in zip(crop_start, crop_end)
            )
            return zoomed_image[slices]

        # If zoomed out (zoom_factor < 1), pad the image
        elif zoom_factor < 1:
            pad_width = (original_shape - zoomed_shape) // 2
            pad_width = [
                (pad, pad) for pad in pad_width
            ]  # Create padding tuple for each axis

            # Ensure correct final size (in case of odd dimensions)
            for i in range(len(pad_width)):
                pad_width[i] = (
                    pad_width[i][0],
                    original_shape[i] - (zoomed_shape[i] + pad_width[i][0]),
                )

            return np.pad(
                zoomed_image, pad_width, mode="constant", constant_values=0
            )  # Padding with zeros

        return zoomed_image  # If zoom factor == 1, return unchanged image

    def get_aug_func(self):
        """returns a list with all the augmentations"""
        aug = []
        if self.args.augment.flip:
            # flip on sagittal plane
            if np.random.rand() < 0.5:
                aug.append(lambda data: np.flip(data, axis=0))
            # flip on coronal plane
            if np.random.rand() < 0.5:
                aug.append(lambda data: np.flip(data, axis=1))

        if self.args.augment.rot:
            for axes in [(0, 1), (0, 2), (1, 2)]:
                if np.random.rand() < 0.5:
                    angle = np.random.uniform(
                        -self.args.augment.rot, self.args.augment.rot
                    )
                    aug.append(
                        lambda data: rotate(data, angle, axes=axes, reshape=False)
                    )

        if self.args.augment.zoom:
            zoom_factor = np.random.uniform(
                self.args.augment.zoom[0],
                self.args.augment.zoom[1],
            )
            aug.append(lambda data: self.zoom_to_fixed_size(data, zoom_factor))

        if self.args.augment.offset:
            aug.append(lambda data: self.add_offset(data, self.args.augment.offset))

        if self.args.augment.blur:
            if self.args.augment.blur.type.lower() == "gaussian":
                aug.append(
                    lambda data: self.gaussian_blur(data, self.args.augment.blur.sigma)
                )
            else:
                raise ValueError("blurring not registered")

        if self.args.augment.noise:
            if self.args.augment.noise.type.lower() == "gaussian":
                aug.append(
                    lambda data: self.add_gaussian_noise(
                        data, self.args.augment.noise.variance
                    )
                )
            else:
                raise ValueError("noise not registered")

        return aug

    @staticmethod
    def get_valid_auc_func(augmentation):
        """returns a list with all the augmentations"""
        if augmentation == "unity":
            return lambda data: data
        if augmentation == "flip_coronal":
            return lambda data: np.flip(data, axis=0)
        if augmentation == "flip_sagittal":
            return lambda data: np.flip(data, axis=1)
        if augmentation == "rot_90":
            return lambda data: np.rot90(data, axes=(1, 2), k=1)
        if augmentation == "zoom_1.1":
            return lambda data: zoom(data, zoom=1.1, order=1)
        if augmentation == "zoom_0.9":
            return lambda data: zoom(data, zoom=0.9, order=1)
        raise ValueError("validation augmentation not listed")

    def augmentation(self, data, aug_func):
        for func in aug_func:
            data = func(data)
        return data

    # ----------------------------------------------------------------

    # crop patches
    # ----------------------------------------------------------------
    def get_index(self, patch_shape):
        init = np.zeros(len(patch_shape))
        fin = np.zeros(len(patch_shape))

        # use random indices for training
        if self.args.training:
            for ii in range(0, len(patch_shape)):
                ## choose a range to pick from
                # mh: high margin, ml: left margin
                ml, mh = self.args.margin[ii]
                m = ml + mh
                if m:
                    m_max = patch_shape[ii] - self.args.shape[ii] - 1
                    m_max = max(0, m_max)
                    ml_max = ml * (m_max / m)
                    mh_max = m_max - ml
                    mh_max = max(0, mh_max)
                    ml = min(ml_max, ml)
                    mh = min(mh_max, mh)
                low = ml
                high = patch_shape[ii] - self.args.shape[ii] - mh + 1
                # diff = patch_shape[ii] - self.args.shape[ii] - 2*m
                # m += min(diff//2, 0)
                # low = m
                # high = patch_shape[ii] - self.args.shape[ii] - m + 1
                init[ii] = np.random.randint(
                    low=low,
                    high=high,
                    # low=0,
                    # high=patch_shape[ii] - self.args.shape[ii] + 1
                )

        # center for validation
        else:
            for ii in range(0, len(patch_shape)):
                init[ii] = (patch_shape[ii] - self.args.shape[ii]) // 2

        fin = init + np.array(self.args.shape)

        init = init.astype(int)
        fin = fin.astype(int)

        out = np.empty(len(init) + len(fin))
        out[::2] = init
        out[1::2] = fin

        return out.astype(int)

    def crop_patch(self, data, idx):
        if len(idx) == 6:
            return data[idx[0] : idx[1], idx[2] : idx[3], idx[4] : idx[5]]
        elif len(idx) == 4:
            return data[idx[0] : idx[1], idx[2] : idx[3]]
        raise ValueError("patch_shape not implemented")

    # ----------------------------------------------------------------

    # get data
    # ----------------------------------------------------------------

    @staticmethod
    def to_slice(shape):
        slz = []
        for shp in shape:
            slz.append(np.s_[:shp])
        return tuple(slz)

    def zero_pad(self, img, shape):
        # zero pad if too small
        padw = np.zeros(6)
        for ax in range(img.ndim):
            diff = shape[ax] - img.shape[ax]
            if diff < 0:
                continue
            padw[::2][ax] = int(np.ceil(diff / 2))
            padw[1::2][ax] = diff // 2
        padw = padw.reshape(3, 2).astype(int)
        if padw.any():
            img = np.pad(img, pad_width=padw)
        return img

    def assert_size(self, img):
        img = img[self.to_slice(self.args.shape)]
        return img

    def get_modality(self, pid, modality):
        with h5py.File(self.args.patch_file, "r") as ff:
            data = ff[f"{pid}"][modality][:]

        if self.args.max_value or self.args.min_value:
            data = np.clip(
                data,
                a_min=self.args.min_value,
                a_max=self.args.max_value,
            )

        # rescale intensity values linear
        if self.args.scale:
            data = self.rescale_linear(
                data,
                self.args.scale,
            )

        return data.squeeze()

    def get_patch(self, idx):

        # Use the same index and crop_patch function
        # Pass out the mask

        pid = self.label_df.loc[idx]["identifier"]

        patch = np.empty((len(self.args.modalities), *self.args.shape))

        for ii, mod in enumerate(self.args.modalities):
            modality = self.get_modality(pid, mod)
            modality = self.zero_pad(modality, self.args.shape)
            if not ii:
                shape = modality.shape
                index = self.get_index(shape)
                if self.args.training:
                    aug_func = self.get_aug_func()
            else:
                modality = self.zero_pad(modality, shape)
            p = self.crop_patch(modality, index)
            p = self.assert_size(p)
            if self.args.training:
                p = self.augmentation(p, aug_func)
            patch[ii] = p

        patch = np.nan_to_num(patch)

        if self.ignore_abnormal_amyloid:
            ds = sitk.ReadImage(self.input_mask_file)
            input_mask = sitk.GetArrayFromImage(ds)
            input_mask = np.transpose(input_mask, (2, 1, 0))
            input_mask = np.expand_dims(input_mask, axis=0)

            # input_mask = self.zero_pad(input_mask, self.args.shape)
            mask_patch = self.crop_patch(input_mask.squeeze(), index)
            mask_patch = np.expand_dims(mask_patch, axis=0)
            return patch.astype(np.float32), mask_patch.astype(np.float32)

        return patch.astype(np.float32)

    def get_label(self, idx):
        label = self.label_df.loc[idx][self.args.label_column]
        map_dict = self.args.label_mapping.__dict__
        if map_dict:
            label = map_dict[label]
        if self.args.one_hot:
            n_classes = len(np.unique(list(map_dict.values())))
            label = np.eye(n_classes)[int(label)].astype(np.float32)
        else:
            label = np.array(label, dtype=int)

        return label

    # ----------------------------------------------------------------

    # torch Dataset stuff
    # ----------------------------------------------------------------
    def __len__(self):
        """cases involved"""
        return self.total_count

    def get_case(self, idx, information=False):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.ignore_abnormal_amyloid:
            patch, mask_patch = self.get_patch(idx)
            return patch, mask_patch
        else:
            patch = self.get_patch(idx)
            return patch, None
        # if information:
        #     info = self.label_df.loc[idx][information].to_dict()
        #     return patch, info

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        patch, mask_patch = self.get_case(idx)
        # if self.with_label:
        #     label = self.get_label(idx)
        #     return patch, label
        if mask_patch is None:
            return patch
        else:
            return patch, mask_patch

    # ----------------------------------------------------------------
