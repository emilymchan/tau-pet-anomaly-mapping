
import sys
from pathlib import Path

import h5py
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.common_utils import read_nii_file, get_array_from_itk


def roi_analysis(
    data_dir,
    project_dir,
    input_type,
    input_file,
    outfile,
    df,
    cfg,
    key,
    threshold=None,
    evaluation_split="test",
):
    # Select participants. If no split is specified, use the full dataset.
    if evaluation_split is None:
        pids = df["identifier"]
    else:
        pids = df.loc[
            df["split"] == evaluation_split, "identifier"
        ]

    # Check that each identifier corresponds to exactly one metadata row.
    participants = df.set_index("identifier", verify_integrity=True)

    # Schaefer atlas
    schaefer_file = Path(
        cfg.device.adni_dir,
        cfg.data.general.schaefer_atlas,
    )
    schaefer_sitk = read_nii_file(schaefer_file)
    schaefer_array = get_array_from_itk(schaefer_sitk)
    schaefer_binary = (schaefer_array > 0).astype(int)

    # Get Schaefer labels for each Braak ROI
    schaefer_braak_file = Path(
        cfg.device.adni_dir,
        cfg.schaefer_braak,
    )
    df_schaefer_braak = pd.read_excel(schaefer_braak_file)

    braak12_schaefer = [
        s
        for s, b in zip(
            df_schaefer_braak["Schaefer"],
            df_schaefer_braak["Braak"],
        )
        if b in [1, 2]
    ]
    braak34_schaefer = [
        s
        for s, b in zip(
            df_schaefer_braak["Schaefer"],
            df_schaefer_braak["Braak"],
        )
        if b in [3, 4]
    ]
    braak56_schaefer = [
        s
        for s, b in zip(
            df_schaefer_braak["Schaefer"],
            df_schaefer_braak["Braak"],
        )
        if b in [5, 6]
    ]

    # Get Braak Stage ROI masks
    braak12_mask = np.isin(
        schaefer_array, braak12_schaefer
    ).astype(int)
    braak34_mask = np.isin(
        schaefer_array, braak34_schaefer
    ).astype(int)
    braak56_mask = np.isin(
        schaefer_array, braak56_schaefer
    ).astype(int)

    print("Starting roi_analysis")
    output_dict = []

    with h5py.File(input_file, "r") as f:
        for pid_ind, pid in enumerate(pids):
            print(f"[{pid_ind + 1}/{len(pids)}]: {pid}")

            # Get metadata for this participant
            participant = participants.loc[pid]

            # Get grey matter (GM) mask and cortical GM mask
            mask_file = Path(
                data_dir,
                participant["anat_mask"],
            )
            mask_sitk = read_nii_file(mask_file)
            mask_array = get_array_from_itk(mask_sitk)
            mask_array = np.round(mask_array)
            mask_array = mask_array.squeeze()

            GM_mask = (mask_array == 2).astype(int)
            GM_cortical_mask = np.multiply(
                GM_mask, schaefer_binary
            )

            # Get input image
            img = f[pid][key][()]
            img = img.squeeze()

            # Thresholding, if applicable
            if "threshold" in input_type:
                img = (img > threshold).astype(int)

            if "thresholded_tau" in input_type:
                original_img_file = (
                    Path(project_dir) / cfg.no_norm_patch_file
                )
                with h5py.File(original_img_file, "r") as f_orig:
                    original_img = f_orig[pid]["tauPET"][()]
                    original_img = original_img.squeeze()

                img = np.multiply(img, original_img)

            # Get masked images
            img_GM_cortical = np.where(
                GM_cortical_mask, img, np.nan
            )
            img_braak12 = np.where(
                braak12_mask, img, np.nan
            )
            img_braak34 = np.where(
                braak34_mask, img, np.nan
            )
            img_braak56 = np.where(
                braak56_mask, img, np.nan
            )

            # Get diagnosis, amyloid and tau statuses if available
            dx_granular = participant.get("dx_granular")

            if pd.notna(dx_granular):
                tau = dx_granular.split("_")[-1]
                amy = dx_granular.split("_")[1]
            else:
                tau = participant.get("tau")
                amy = participant.get("amy")

            # Get values within each ROI
            if "voxels" in input_type:
                agg_func = np.nansum  # Count anomalous voxels
            else:
                agg_func = np.nanmean

            # Get diagnosis, amyloid and tau statuses
            dx_granular = participant.get("dx_granular")

            if pd.notna(dx_granular):
                # ADNI: derive amyloid and tau status from dx_granular
                tau = dx_granular.split("_")[-1]
                amy = dx_granular.split("_")[1]
            else:
                # A4: use the existing metadata columns
                amy = participant.get("amy")
                tau = participant.get("tau")

            # Fill output dictionary
            row_data = {
                "pid": pid,
                "split_UAD": participant.get("split"),
                "split_classify": participant.get("split_classify"),
                "dx": participant.get("dx"),
                "amy": amy,
                "tau": tau,
                "DX_bl": participant.get("DX_bl"),
                "DX_MCI_granular": participant.get("DX_MCI_granular"),
                "VISCODE": participant.get("VISCODE"),
            }

            if "GM" not in input_type:
                wholeBrain_mask = (
                    mask_array > 0
                ).astype(int)
                img_wholeBrain = np.where(
                    wholeBrain_mask, img, np.nan
                )
                row_data["whole_brain"] = agg_func(
                    img_wholeBrain
                )

            if "cortical" not in input_type:
                img_GM = np.where(
                    GM_mask, img, np.nan
                )
                row_data["GM"] = agg_func(img_GM)

            row_data.update(
                {
                    "GM_cortical": agg_func(
                        img_GM_cortical
                    ),
                    "Braak12": agg_func(img_braak12),
                    "Braak34": agg_func(img_braak34),
                    "Braak56": agg_func(img_braak56),
                }
            )

            output_dict.append(row_data)

    df_out = pd.DataFrame(output_dict)
    df_out.to_csv(outfile, index=False)
    print("Finished roi_analysis")


@hydra.main(
    config_path="../config",
    config_name="config_analysis",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    input_types = cfg.input_types
    data_dir = cfg.device.data_dir
    evaluation_split = cfg.data.inference.evaluation_split

    all_imgs_file = Path(
        cfg.device.project_dir,
        cfg.meta_data_file,
    )
    df = pd.read_csv(all_imgs_file)

    # Some evaluation types use different thresholds
    options_thresholds = {
        "anomaly_thresholded_voxels": cfg.anomaly_thresholds,
        "tauSUVR_thresholded_voxels": cfg.tau_thresholds,
        "anomaly_thresholded_tau": cfg.anomaly_thresholds,
        "tauSUVR_thresholded_tau": cfg.tau_thresholds,
    }

    options = {
        k: (
            options_thresholds[k]
            if "threshold" in k
            else [None]
        )
        for k in input_types
    }

    project_dir = Path(
        cfg.device.project_dir,
        "tauPET",
    )

    for input_type, thresholds in options.items():
        for threshold in thresholds:

            if "tauSUVR" in input_type:
                input_folder = project_dir
                input_file = Path(
                    project_dir,
                    cfg.no_norm_patch_file,
                )
                key = "tauPET"
            else:
                input_folder = Path(
                    cfg.device.project_dir,
                    "tauPET",
                    cfg.anomaly_eval_folder,
                )
                input_file = Path(
                    input_folder,
                    cfg.pred_file,
                )
                key = "diff"

            # Output folder
            output_folder = Path(
                input_folder,
                "outputs",
            )
            output_folder.mkdir(
                exist_ok=True,
                parents=True,
            )

            # Output file
            outfile = Path(
                output_folder,
                f"roi_analysis_{input_type}.csv",
            )

            if "threshold" in input_type:
                outfile = outfile.parent / (
                    outfile.stem
                    + f"_{str(threshold).replace('.', '_')}.csv"
                )

            print(outfile)

            roi_analysis(
                data_dir,
                project_dir,
                input_type,
                input_file,
                outfile,
                df,
                cfg,
                key,
                threshold,
                evaluation_split,
            )

    print("Finished all roi analyses")


if __name__ == "__main__":
    main()
