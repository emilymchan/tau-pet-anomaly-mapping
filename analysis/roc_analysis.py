"""
Code to compute ROC AUC variance using DeLong's method from the following:
@author: yandexdataschool

Original Code found in:
https://github.com/yandexdataschool/roc_comparison

updated: Raul Sanchez-Vazquez
"""

from pathlib import Path
from itertools import combinations

import pandas as pd
import hydra
from omegaconf import DictConfig
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import numpy as np
from scipy import stats
import scipy.stats
import matplotlib.pyplot as plt

from analysis.analysis_constants import STRATIFY_STAGES, ROIS, Y_LABELS


def fast_pairwise_delong(all_preds, all_labels):
    """
    Fast pairwise DeLong + FDR correction.
    """

    results = []

    # group keys by (ROI, comparison)
    grouped = {}
    for (roi, cls, model) in all_preds:
        grouped.setdefault((roi, cls), []).append(model)

    for (roi, cls), models in grouped.items():

        labels = all_labels[(roi, cls)]

        # precompute ordering once per dataset
        order, label_1_count, _ = compute_ground_truth_statistics(labels, None)

        model_pairs = list(combinations(models, 2))

        if len(model_pairs) == 0:
            continue

        pvals = []
        meta = []

        # build stacked prediction matrix once per model pair
        for m1, m2 in model_pairs:

            p1 = all_preds[(roi, cls, m1)]
            p2 = all_preds[(roi, cls, m2)]

            # stack + reorder ONCE
            preds = np.vstack([p1, p2])[:, order]

            aucs, cov = fastDeLong_no_weights(preds, label_1_count)

            # DeLong test
            l = np.array([[1, -1]])

            variance = float((l @ cov @ l.T).item())
            z = np.abs(aucs[0] - aucs[1]) / np.sqrt(variance)

            pval = float(2 * scipy.stats.norm.sf(z))

            pvals.append(pval)

            meta.append({
                "ROI": roi,
                "Classification": cls,
                "Model_A": m1,
                "Model_B": m2,
                "AUC_A": aucs[0],
                "AUC_B": aucs[1],
            })

        # -------- FDR correction (Benjamini–Hochberg) --------
        pvals = np.array(pvals)
        _, pvals_fdr, _, _ = multipletests(pvals, method="fdr_bh")

        for i, m in enumerate(meta):
            m["pval_raw"] = float(pvals[i])
            m["pval_fdr"] = float(pvals_fdr[i])
            results.append(m)

    return results


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float64)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def compute_midrank_weight(x, sample_weight):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    cumulative_weight = np.cumsum(sample_weight[J])
    N = len(x)
    T = np.zeros(N, dtype=np.float64)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = cumulative_weight[i:j].mean()
        i = j
    T2 = np.empty(N, dtype=np.float64)
    T2[J] = T
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count, sample_weight):
    if sample_weight is None:
        return fastDeLong_no_weights(predictions_sorted_transposed, label_1_count)
    else:
        return fastDeLong_weights(
            predictions_sorted_transposed, label_1_count, sample_weight
        )


def fastDeLong_weights(predictions_sorted_transposed, label_1_count, sample_weight):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank_weight(positive_examples[r, :], sample_weight[:m])
        ty[r, :] = compute_midrank_weight(negative_examples[r, :], sample_weight[m:])
        tz[r, :] = compute_midrank_weight(
            predictions_sorted_transposed[r, :], sample_weight
        )
    total_positive_weights = sample_weight[:m].sum()
    total_negative_weights = sample_weight[m:].sum()
    pair_weights = np.dot(sample_weight[:m, np.newaxis], sample_weight[np.newaxis, m:])
    total_pair_weights = pair_weights.sum()
    aucs = (sample_weight[:m] * (tz[:, :m] - tx)).sum(axis=1) / total_pair_weights
    v01 = (tz[:, :m] - tx[:, :]) / total_negative_weights
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / total_positive_weights
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def fastDeLong_no_weights(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Oerating
              Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float64)
    ty = np.empty([k, n], dtype=np.float64)
    tz = np.empty([k, m + n], dtype=np.float64)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def calc_pvalue(aucs, sigma):
    """Computes log(10) of p-values.
    Args:
       aucs: 1D array of AUCs
       sigma: AUC DeLong covariances
    Returns:
       log10(pvalue)
    """
    l = np.array([[1, -1]])
    z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(l, sigma), l.T))
    return np.log10(2) + scipy.stats.norm.logsf(z, loc=0, scale=1) / np.log(10)


def compute_ground_truth_statistics(ground_truth, sample_weight):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    if sample_weight is None:
        ordered_sample_weight = None
    else:
        ordered_sample_weight = sample_weight[order]

    return order, label_1_count, ordered_sample_weight


def delong_roc_variance(ground_truth, predictions, sample_weight=None):
    """
    Computes ROC AUC variance for a single set of predictions
    Args:
       ground_truth: np.array of 0 and 1
       predictions: np.array of floats of the probability of being class 1
    """
    order, label_1_count, ordered_sample_weight = compute_ground_truth_statistics(
        ground_truth, sample_weight
    )
    predictions_sorted_transposed = predictions[np.newaxis, order]
    aucs, delongcov = fastDeLong(
        predictions_sorted_transposed, label_1_count, ordered_sample_weight
    )
    assert (
        len(aucs) == 1
    ), "There is a bug in the code, please forward this to the developers"
    return aucs[0], delongcov


def do_roc_analysis(
    df,
    stratifications,
    input_type,
    threshold,
    outcsv,
    bl_only=False,
    combine_SMC_CN=False,
    all_preds=None,
    all_labels=None,
):
    results = []

    for roi in ROIS:
        for stratify in stratifications:
            print("-" * 40)
            print(f"ROC for {stratify} in ROI {roi}")

            if roi not in list(df):
                print("ROI not present, skipping")
                continue

            stratify_stages = STRATIFY_STAGES[stratify]
            if combine_SMC_CN:
                df = df.replace({"SMC": "CN"}, regex=True)

            comparisons = list(combinations(stratify_stages, 2))

            for stage0, stage1 in comparisons:
                print(f"{stage0} vs {stage1}")
                if stratify == "dx_amy":
                    stage0_dx, stage0_amy = stage0.split("_")
                    stage1_dx, stage1_amy = stage1.split("_")
                    stage0_flag = (df["dx"] == stage0_dx) & (df["amy"] == stage0_amy)
                    stage1_flag = (df["dx"] == stage1_dx) & (df["amy"] == stage1_amy)
                    df_stages = df[stage0_flag | stage1_flag]

                    if bl_only:
                        df_stages = df_stages[df_stages["VISCODE"] == "bl"]

                    labels = (
                        (df_stages["dx"] == stage1_dx)
                        & (df_stages["amy"] == stage1_amy)
                    ).astype(int)
                elif stratify == "dx_tau":
                    stage0_dx, stage0_tau = stage0.split("_")
                    stage1_dx, stage1_tau = stage1.split("_")
                    stage0_flag = (df["dx"] == stage0_dx) & (df["tau"] == stage0_tau)
                    stage1_flag = (df["dx"] == stage1_dx) & (df["tau"] == stage1_tau)
                    df_stages = df[stage0_flag | stage1_flag]

                    if bl_only:
                        df_stages = df_stages[df_stages["VISCODE"] == "bl"]

                    labels = (
                        (df_stages["dx"] == stage1_dx)
                        & (df_stages["tau"] == stage1_tau)
                    ).astype(int)
                else:
                    df_stages = df[df[stratify].isin([stage0, stage1])]
                    labels = df_stages[stratify].map({stage0: 0, stage1: 1}).values
                roi_values = df_stages[roi].values

                input_str = input_type
                if "threshold" in input_type:
                    input_str += f"_{str(threshold).replace('.', '_')}"

                # Store all predictions and labels for later comparison
                classification = f"{stage0} vs {stage1}"

                pred_key = (roi, classification, input_str)
                label_key = (roi, classification)

                all_preds[pred_key] = np.asarray(roi_values)
                all_labels[label_key] = np.asarray(labels)

                # ROC & AUPR
                auroc = roc_auc_score(labels, roi_values)
                auprc = average_precision_score(labels, roi_values)

                # Delong's Method
                alpha = 0.95
                auc, auc_cov = delong_roc_variance(labels, roi_values)
                auc_std = np.sqrt(auc_cov)
                lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
                ci = stats.norm.ppf(lower_upper_q, loc=auc, scale=auc_std)
                ci[ci > 1] = 1

                # Sensitivity, Specificity
                fpr, tpr, thresholds = roc_curve(labels, roi_values)
                idx = np.argmax(tpr - fpr)
                youden_threshold = thresholds[idx]
                preds = roi_values > youden_threshold

                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                specificity = tn / (tn + fp)
                sensitivity = tp / (tp + fn)

                results.append(
                    {
                        "ROI": roi,
                        "Classification": f"{stage0} vs {stage1}",
                        "Input_Type": input_str,
                        "AUROC": round(auroc, 3),
                        "95% AUC CI": ci,
                        "SEN": round(sensitivity, 3),
                        "SPE": round(specificity, 3),
                        "AUPRC": round(auprc, 3),
                    }
                )

    # Save individual roc results
    df_results = pd.DataFrame(results)
    df_results.to_csv(outcsv, index=False, mode="a", header=not outcsv.exists())


def plot_roc_graphs(df_roc, output_folder):
    print("Plotting ROC results")
    # plot over all different input types
    plot_stratifications = [
        "tauNeg vs tauPos",
        "amyNeg vs amyPos",
        "CN vs MCI",
        "CN vs Dementia",
        "MCI vs Dementia",
    ]

    for roi in ROIS:
        out_png = Path(output_folder, f"auroc_aupr_{roi}.png")
        print(out_png)

        df_roi = df_roc[df_roc["ROI"] == roi]
        fig, axs = plt.subplots(2, 3, constrained_layout=True)

        for stratification, axs_f in zip(plot_stratifications, axs.flatten()):
            df_stratify = df_roi[df_roi["Classification"] == stratification]
            df_stratify.plot.bar(
                x="Input_Type",
                y=["AUROC", "AUPRC"],
                rot=40,
                ax=axs_f,
                ylim=[0, 1],
                legend=False,
            )

        axs[-1, -1].axis("off")
        handles, labels = axs[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right")
        plt.suptitle(roi, fontsize=11)
        plt.savefig(out_png)

    print("Finished plotting")


@hydra.main(config_path="../config", config_name="config_analysis", version_base=None)
def main(cfg: DictConfig) -> None:
    stratifications = cfg.stratifications

    if "DX_bl" in stratifications:
        baseline_only = True
    else:
        baseline_only = False

    # Some evaluation types are run with different thresholds
    options_thresholds = {
        "anomaly_thresholded_voxels": cfg.anomaly_thresholds,
        "tauSUVR_thresholded_voxels": cfg.tau_thresholds,
        "anomaly_thresholded_tau": cfg.anomaly_thresholds,
        "tauSUVR_thresholded_tau": cfg.tau_thresholds,
        "histMatched_wholeBrain_thresholded": cfg.hist_matched_thresholds,
        "histMatched_GM_thresholded": cfg.hist_matched_thresholds,
        "histMatched_GM_cortical_thresholded": cfg.hist_matched_thresholds,
    }
    options = {
        k: (options_thresholds[k] if "threshold" in k else [None])
        for k in cfg.input_types
    }

    # Output folder
    output_folder = Path(
        cfg.device.project_dir, "tauPET", cfg.anomaly_eval_folder, "outputs"
    )
    # output_folder = Path(cfg.anomaly_eval_folder)
    outfile = Path(output_folder, f"roc_analysis.csv")
    if baseline_only:
        outfile = outfile.parent / (outfile.stem + "_bl.csv")
        if cfg.combine_SMC_CN:
            outfile = outfile.parent / (outfile.stem + "_combine_SMC_CN.csv")
            STRATIFY_STAGES["DX_bl"].remove("SMC")
    elif "DX_MCI_granular" in stratifications:
        outfile = outfile.parent / (outfile.stem + "_MCI_granular.csv")
    print(f"Writing to: {outfile}")

    all_preds = {}
    all_labels = {}

    for input_type, thresholds in options.items():
        for threshold in thresholds:
            print("-" * 40)
            print(f"{input_type}: threshold {threshold}")
            if "anomaly" in input_type:
                input_folder = Path(
                    cfg.device.project_dir, "tauPET", cfg.anomaly_eval_folder, "outputs"
                )
                # input_folder = cfg.anomaly_eval_folder
            else:
                input_folder = Path(cfg.device.project_dir, "tauPET/outputs")
            input_file = Path(input_folder, f"roi_analysis_{input_type}.csv")
            if "threshold" in input_type:
                input_file = input_file.parent / (
                    input_file.stem + f"_{str(threshold).replace('.', '_')}.csv"
                )

            df = pd.read_csv(input_file)

            do_roc_analysis(
                df,
                stratifications,
                input_type,
                threshold,
                outfile,
                baseline_only,
                combine_SMC_CN=cfg.combine_SMC_CN,
                all_preds=all_preds,
                all_labels=all_labels,
            )

    comparison_results = fast_pairwise_delong(all_preds, all_labels)
    df_comp = pd.DataFrame(comparison_results)
    df_comp.to_csv(
        outfile.parent / "roc_pairwise_comparisons.csv",
        index=False,
    )
    # df_roc = pd.read_csv(outfile)
    # plot_roc_graphs(df_roc, output_folder)

    print("Finished all roc analyses")


if __name__ == "__main__":
    main()
