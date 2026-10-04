from pathlib import Path

import pandas as pd
import hydra
from omegaconf import DictConfig
from scipy.stats import spearmanr, norm, zscore
import statsmodels.api as sm
import numpy as np

from analysis.analysis_constants import COGNITIVE_TESTS, ROIS


# ---------------------------------------------------------------------
# Correlation functions
def do_correlation(df_merged, input_type, threshold, outcsv):
    results = []
    for roi in ROIS:
        for cog_test in COGNITIVE_TESTS.keys():
            print(f"{cog_test} for ROI {roi}")
            if roi not in list(df_merged):
                print("ROI not present, skipping")
                continue

            df_cog = df_merged.dropna(subset=[cog_test])
            roi_values = df_cog[roi].values
            cog_values = df_cog[cog_test].values

            corr, pvalue = spearmanr(roi_values, cog_values)

            # if cog_test in ["MOCA", "MMSE"] and roi in ["GM_cortical", "Braak12"]:
            #     out_png = "correlation_" + input_type + "_" + roi + "_" + cog_test + ".png"
            #     out_png = outcsv.parent / out_png
            #     plot_correlation(cog_values, roi_values, cog_test, out_png, roi, input_type=="anomaly")

            input_str = input_type
            if "threshold" in input_type:
                input_str += f"_{str(threshold).replace('.', '_')}"

            results.append(
                {
                    "ROI": roi,
                    "Input_Type": input_str,
                    "Cog_test": cog_test,
                    "Corr": corr,
                    "pvalue": pvalue,
                }
            )

    df_results = pd.DataFrame(results)
    df_results.to_csv(outcsv, index=False, mode="a", header=not outcsv.exists())


def remove_outliers(df, column, method='zscore', threshold=2.0):
    """
    Removes outliers from a DataFrame based on a single column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the column to check for outliers.
        method (str): 'zscore' or 'iqr'.
        threshold (float): Threshold for detecting outliers.
                          For z-score, common is 2 or 3.
                          For IQR, common multiplier is 1.5.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    if method == 'zscore':
        z = zscore(df[column])
        return df[np.abs(z) < threshold]
    
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    else:
        raise ValueError("method must be either 'zscore' or 'iqr'")


def plot_correlation(x, y, measure, out_png, roi, anomaly=True):
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    teal = (60/255, 180/255, 175/255)
    purple = (155/255, 60/255, 145/255)
    fontsize = 26

    if anomaly:
        y_title = "Anomaly Score"
        color = teal
    else:
        y_title = "Tau SUVr > 1.3"
        color = purple
    df = pd.DataFrame({measure: x, y_title: y})
    from matplotlib.legend_handler import HandlerTuple

    # df = remove_outliers(df, column=y_title, method='iqr', threshold=1.5)

    plt.rcParams.update({'font.size': fontsize}) 
    plt.figure(figsize=(8, 8), dpi=300)

    # out_png = out_png.stem + "_noOutliers" + out_png.suffix
    corr, pvalue = spearmanr(df[measure], df[y_title])
    ax = sns.regplot(x=df[measure].values, y=df[y_title].values, line_kws={'color': color}, scatter_kws={'color': color}, ci=95)
    # custom_label = Line2D([0], [0], color='none', label="r={:.3f}, p={:.3f}".format(corr, pvalue))
    # plt.legend(handles=[Line2D([], [], marker='', linestyle='', label=f"r={corr:.3f}, p<0.001")], loc='upper left', bbox_to_anchor=(0, 0.95))
    legend_handle = Line2D([], [], marker='', linestyle='', label=f"r={corr:.3f}, p<0.001")
    ax.legend(handles=[legend_handle], handlelength=0, handletextpad=0, loc='upper left', bbox_to_anchor=(0, 0.95))
    plt.xlabel(measure)
    plt.ylabel(y_title)
    # plt.text(x=2, y=2, s="r={:.3f}, p={:.3f}".format(corr, pvalue), fontsize=fontsize)
    # plt.legend(handles=[Line2D([], [], color='none', label="r={:.3f}, p<0.001".format(corr))])
    if "GM" in roi:
        plt.title("Cortical GM", fontsize=fontsize)
    else:
        plt.title(roi, fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()


def steiger_z_test(r_xy, r_xz, r_yz, n):
    """
    Steiger's Z-test for comparing dependent correlations.

    r_xy: corr(X, Y)
    r_xz: corr(X, Z)
    r_yz: corr(Y, Z)
    n: sample size
    """
    numerator = (r_xy - r_xz) * np.sqrt((n - 3) * (1 + r_yz))
    denominator = np.sqrt(2 * (1 - r_xy**2 - r_xz**2 - r_yz**2 + 2 * r_xy * r_xz * r_yz))

    z = numerator / denominator
    p = 2 * (1 - norm.cdf(abs(z)))
    return z, p


# ---------------------------------------------------------------------
# Regression functions
def compare_tau_anomaly(df, roi, cog_test, tau_col, anomaly_col, outcsv):
    results = []

    df_clean = df.dropna(subset=[roi, cog_test, tau_col, anomaly_col])

    if len(df_clean) < 10:
        return  # avoid unstable stats

    y = df_clean[cog_test]
    tau = df_clean[tau_col]
    anomaly = df_clean[anomaly_col]

    # --- correlations ---
    r_tau, p_tau = spearmanr(tau, y)
    r_anom, p_anom = spearmanr(anomaly, y)
    r_tau_anom, _ = spearmanr(tau, anomaly)

    # --- steiger test ---
    z, p_steiger = steiger_z_test(r_tau, r_anom, r_tau_anom, len(df_clean))

    # --- regressions ---
    model_tau, r2_tau = run_regression_models(df_clean, cog_test, [tau_col])
    model_anom, r2_anom = run_regression_models(df_clean, cog_test, [anomaly_col])
    model_both, r2_both = run_regression_models(df_clean, cog_test, [tau_col, anomaly_col])

    results.append({
        "ROI": roi,
        "Cog_test": cog_test,

        "r_tau": r_tau,
        "p_tau": p_tau,

        "r_anomaly": r_anom,
        "p_anomaly": p_anom,

        "steiger_z": z,
        "steiger_p": p_steiger,

        "R2_tau": r2_tau,
        "R2_anomaly": r2_anom,
        "R2_both": r2_both,

        "delta_R2": r2_both - r2_tau
    })

    df_out = pd.DataFrame(results)
    df_out.to_csv(outcsv, index=False, mode="a", header=not outcsv.exists())


def run_regression_models(df, y_var, x_vars):
    """
    Fits OLS regression and returns model + R².
    """
    df_clean = df.dropna(subset=[y_var] + x_vars)
    X = df_clean[x_vars]
    X = sm.add_constant(X)
    y = df_clean[y_var]

    model = sm.OLS(y, X).fit()
    return model, model.rsquared


@hydra.main(config_path="../config", config_name="config_analysis_ddpm", version_base=None)
def main(cfg: DictConfig) -> None:
    meta_file = Path(cfg.device.project_dir, cfg.meta_data_file)
    df_cog_tests = pd.read_csv(meta_file)
    df_cog_tests.rename(columns={"identifier": "pid"}, inplace=True)

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
    outfile = Path(output_folder, f"correlation_with_cognitive_tests.csv")
    print(f"Writing to: {outfile}")

    # ---------------------------------------------------------------------
    # Correlation analysis
    for input_type, thresholds in options.items():
        for threshold in thresholds:
            print("-" * 40)
            print(f"{input_type}: threshold {threshold}")
            if "anomaly" in input_type:
                input_folder = Path(
                    cfg.device.project_dir, "tauPET", cfg.anomaly_eval_folder, "outputs"
                )
            else:
                input_folder = Path(cfg.device.project_dir, "tauPET/outputs")
            input_file = Path(input_folder, f"roi_analysis_{input_type}.csv")
            if "threshold" in input_type:
                input_file = input_file.parent / (
                    input_file.stem + f"_{str(threshold).replace('.', '_')}.csv"
                )

            df = pd.read_csv(input_file)
            df_merged = df.merge(df_cog_tests, on="pid")

            do_correlation(df_merged, input_type, threshold, outfile)

            print("Finished all correlation analyses")

    
    # ---------------------------------------------------------------------
    # Regression analysis
    print("Running combined regression comparisons...")

    # Example: load tau + anomaly explicitly
    tau_file = Path(cfg.device.project_dir, "tauPET/outputs", "roi_analysis_tauSUVR.csv")
    anom_file = Path(cfg.device.project_dir, "tauPET", cfg.anomaly_eval_folder, "outputs", "roi_analysis_anomaly.csv")

    df_tau = pd.read_csv(tau_file)
    df_anom = pd.read_csv(anom_file)

    df_combined = df_tau.merge(df_anom, on="pid", suffixes=("_tau", "_anom"))
    df_combined = df_combined.merge(df_cog_tests, on="pid")

    outcsv_compare = Path(output_folder, "regression_comparison.csv")

    for roi in ROIS:
        tau_col = f"{roi}_tau"
        anomaly_col = f"{roi}_anom"

        if tau_col not in df_combined or anomaly_col not in df_combined:
            continue

        for cog_test in COGNITIVE_TESTS.keys():  # CHANGE so ignores if cog_test not present
            compare_tau_anomaly(
                df_combined,
                roi,
                cog_test,
                tau_col,
                anomaly_col,
                outcsv_compare
            )


if __name__ == "__main__":
    main()
