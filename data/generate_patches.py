"""
Adapted from MAEMI /utils/generate_patches/image_patches.py
https://github.com/LangDaniel/MAEMI

This script does not do any cropping of the images, it just applies normalisation etc
The patches are all stored in one patches.h5 file
"""

from pathlib import Path
import pickle

import hydra
from omegaconf import DictConfig
import SimpleITK as sitk
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

from utils.io import get_output_patch_folder_name


class GetPatches:

    def __init__(
        self,
        data_dir,
        input_csv,
        output_file,
        modalities,
        spacing,
        clip_before_norm=[False, False],
        bias_correction=[],
        rescale=False,
        normalise=False,
        schaefer_atlas=None,
    ):
        self.data_dir = data_dir
        self.input_csv = input_csv
        self.output_file = Path(output_file)

        # CURRENTLY ASSUMES YOU WANT THE SAME SPACING FOR ALL MODALITIES
        self.spacing = np.array(spacing)

        self.clip_before_norm = clip_before_norm
        self.bias_correction = bias_correction
        self.rescale = rescale
        self.normalise = normalise
        self.modalities = modalities
        self.schaefer_atlas = schaefer_atlas

        # Leave for later (MR)
        # for bc in bias_correction:
        #     if bc not in modalities:
        #         raise ValueError(f'bias correction {bc} not in modalities: {modalities}')
        # for norm in normalise:
        #     if norm not in modalities:
        #         raise ValueError(f'normalise {norm} not in modalities: {modalities}')

        if not rescale and not normalise:
            print("no rescaling and no normalisation used!")
        if rescale and normalise:
            raise ValueError("Rescaling and normalisation cannot both be enabled")

    ######################## IO stuff ##########################
    def get_itk_from_path(self, path, orientation=None):
        path = Path(path)
        if path.is_dir():
            return self.read_dcm_folder(path, orientation)
        elif (path.suffix == ".gz") or (path.suffix == ".nii"):
            return self.read_nii_file(path, orientation)
        raise ValueError("unknown file format")

    @staticmethod
    def read_dcm_folder(path, orientation=None):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(path))
        reader.SetFileNames(dicom_names)

        data = reader.Execute()
        if orientation is not None:
            data = sitk.DICOMOrient(data, orientation)

        return data

    @staticmethod
    def read_nii_file(path, orientation=None):
        data = sitk.ReadImage(path)
        if orientation is not None:
            data = sitk.DICOMOrient(data, orientation)
        return data

    @staticmethod
    def get_array_from_itk(image):
        data = sitk.GetArrayFromImage(image)
        data = np.transpose(data, (2, 1, 0))
        return data

    ######################### crop ##############################
    def resample_img(self, itk_image, is_label=False):

        original_spacing = itk_image.GetSpacing()
        original_size = itk_image.GetSize()

        spacing = np.zeros(3)
        for ii in range(0, 3):
            if self.spacing[ii] == False:
                spacing[ii] = original_spacing[ii]
            else:
                spacing[ii] = self.spacing[ii]

        out_size = [
            int(np.round(original_size[0] * (original_spacing[0] / spacing[0]))),
            int(np.round(original_size[1] * (original_spacing[1] / spacing[1]))),
            int(np.round(original_size[2] * (original_spacing[2] / spacing[2]))),
        ]

        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(spacing)
        resample.SetSize(out_size)
        resample.SetOutputDirection(itk_image.GetDirection())
        resample.SetOutputOrigin(itk_image.GetOrigin())
        resample.SetTransform(sitk.Transform())
        if is_label:
            resample.SetDefaultPixelValue(0)
            resample.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
            resample.SetInterpolator(sitk.sitkBSpline)

        return resample.Execute(itk_image)

    @staticmethod
    def bias_field_correction(image):
        mask = sitk.OtsuThreshold(image, 0, 1, 200)
        mask = sitk.Cast(mask, sitk.sitkUInt8)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()

        corrected_image = corrector.Execute(
            sitk.Cast(image, sitk.sitkFloat32),
            mask,
        )
        return corrected_image

    @staticmethod
    def rescale_linear(data, scaling):
        [in_low, in_high, out_low, out_high] = scaling
        m = (out_high - out_low) / (in_high - in_low)
        b = out_low - m * in_low
        data = m * data + b
        data = np.clip(data, out_low, out_high)
        return data

    @staticmethod
    def normalise_image(img):
        raise NotImplementedError("normalisation function not implemented")

    @staticmethod
    def clip_itk(img, clip_values=[False, False]):
        clamp = sitk.ClampImageFilter()
        if clip_values[0] is not False:
            clamp.SetLowerBound(clip_values[0])
        if clip_values[1] is not False:
            clamp.SetUpperBound(clip_values[1])
        return clamp.Execute(img)

    def get_case(self, pid, modality, _df):
        if modality == "tauPET" or modality == "tauPET_raw":
            path_column = "tau_img"
        elif modality == "amyloidPET":
            path_column = "amyloid_img"
        elif modality == "anat":
            path_column = "anat_img"
        else:
            raise NotImplementedError

        img_path = _df.loc[_df["identifier"] == pid, path_column].item()
        img_path = Path(self.data_dir, img_path)
        seg_path = _df.loc[_df["identifier"] == pid, "anat_mask"].item()
        seg_path = Path(self.data_dir, seg_path)

        try:
            # read and resample image
            ds = self.get_itk_from_path(img_path)
            seg_itk = self.get_itk_from_path(seg_path)
            schaefer_atlas_itk = self.get_itk_from_path(self.schaefer_atlas)

            if self.spacing.any():
                orig_spacing = ds.GetSpacing()
                if not (self.spacing == orig_spacing).all():
                    ds = self.resample_img(ds, is_label=False)
                    seg_itk = self.resample_img(seg_itk, is_label=True)
                    schaefer_atlas_itk = self.resample_img(schaefer_atlas_itk, is_label=True)

            ds = self.clip_itk(ds, self.clip_before_norm)

            # intensity modifications
            if modality in self.bias_correction:
                print("bias correction")
                ds = self.bias_field_correction(ds)

            if self.normalise:
                ds = self.normalise_image(ds)

            img = self.get_array_from_itk(ds)
            seg = self.get_array_from_itk(seg_itk)
            sgmt = (seg > 0).astype(int)

            # Get non white matter brain regions
            if modality == "tauPET":
                seg = np.round(seg)
                GM_mask = (seg == 2).astype(int)
                schaefer_atlas = self.get_array_from_itk(schaefer_atlas_itk)
                schaefer_atlas = (schaefer_atlas > 0).astype(int)
                GM_mask_cortical = np.multiply(GM_mask, schaefer_atlas)

            # Get whole brain region
            if not modality == "tauPET_raw":
                img = np.multiply(img, sgmt)

            if self.rescale:
                img = self.rescale_linear(img, self.mod_rescale)
        except (Exception, ArithmeticError) as e:
            raise ValueError(f"{pid} failed: {e}")
        if modality == "tauPET":
            return img, GM_mask, GM_mask_cortical
        else:
            return img

    def to_disk(self):
        if self.output_file.exists():
            raise ValueError(f"Output file exists: {self.output_file}")
        if not self.output_file.parent.exists():
            self.output_file.parent.mkdir(parents=True)

        disk_df = pd.DataFrame(columns=["identifier", "modality"])
        disk_idx = 0

        with h5py.File(self.output_file, "w") as ff:
            for m, mod in enumerate(self.modalities):
                if self.rescale:
                    self.mod_rescale = self.rescale[m]
                if self.normalise:
                    self.mod_normalise = self.normalise[m]

                df = self.input_csv.drop_duplicates(subset=["identifier"])
                for pid in df["identifier"]:
                    print(f"{pid}-{mod}")
                    try:
                        if mod == "tauPET":
                            data, GM_mask, GM_mask_cortical = self.get_case(
                                pid, mod, df
                            )
                            ff.create_dataset(f"{pid}/GM_mask", data=GM_mask)
                            ff.create_dataset(
                                f"{pid}/GM_mask_cortical", data=GM_mask_cortical
                            )
                        else:
                            data = self.get_case(pid, mod, df)
                        ff.create_dataset(f"{pid}/{mod}", data=data)
                    except (Exception, ArithmeticError) as e:
                        print(f"failed get_case: {e}")
                        continue
                    disk_df.loc[disk_idx] = [pid, mod]
                    disk_idx += 1

        disk_df.to_csv(
            self.output_file.parent / (self.output_file.stem + ".csv"), index=False
        )


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    data_dir = cfg.device.data_dir
    spacing = cfg.patches.spacing
    rescale = cfg.patches.rescale
    normalise = cfg.patches.normalise
    clip_before_norm = cfg.patches.clip_before_norm
    bias_correction = cfg.patches.bias_correction
    modalities = cfg.patches.modalities
    data_split_csv = Path(cfg.device.project_dir, cfg.data.general.split_file)
    case_df = pd.read_csv(data_split_csv)
    schaefer_atlas_file = Path(cfg.device.adni_dir, cfg.data.general.schaefer_atlas)

    if normalise:
        def normalise_image(data, std, mean):
            f = sitk.NormalizeImageFilter()
            data = f.Execute(data)
            return (std * data) + mean

    output_folder = get_output_patch_folder_name(cfg)
    output_file = Path(output_folder, "patches.h5")

    patch_gen = GetPatches(
        data_dir=data_dir,
        input_csv=case_df,
        output_file=output_file,
        clip_before_norm=clip_before_norm,
        bias_correction=bias_correction,
        rescale=rescale,
        normalise=normalise,
        spacing=spacing,
        modalities=modalities,
        schaefer_atlas=schaefer_atlas_file
    )

    ### DOESN'T WORK FOR MULTIMODAL
    if normalise:
        mean = normalise[0]
        std = normalise[1]
        patch_gen.normalise_image = lambda img: normalise_image(img, std=std, mean=mean)

    patch_gen.to_disk()

    output_file = Path(output_file)
    script = output_file.parent / (output_file.stem + ".py")
    with open(script, "w") as ff:
        ff.write(Path(__file__).read_text())

    to_dump = {"spacing": spacing, "rescale": rescale, "normalise": normalise}

    dump_file = output_file.parent / (output_file.stem + ".pkl")
    with open(dump_file, "wb") as ff:
        pickle.dump(to_dump, ff)
    print("Finished generating patches")


if __name__ == "__main__":
    main()
