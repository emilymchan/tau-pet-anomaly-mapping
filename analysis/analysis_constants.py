ROIS = {
    "whole_brain": "Whole Brain",
    "GM": "Grey Matter",
    "GM_cortical": "Cortical Grey Matter",
    "Braak12": "Braak1",
    "Braak34": "Braak34",
    "Braak56": "Braak56",
}

Y_LABELS = {
    "tauSUVR": "Mean tau SUVR",
    "tauSUVR_thresholded": "Sum of voxels tau SUVR > ",
    "anomaly": "Mean anomaly score",
    "anomaly_thresholded": "Sum of voxels anomaly score > ",
}

STRATIFY_STAGES = {
    "dx": ["CN", "MCI", "Dementia"],
    "DX_bl": ["CN", "SMC", "EMCI", "LMCI", "AD"],
    "DX_MCI_granular": ["CN", "EMCI", "LMCI", "Dementia"],
    "tau": ["tauNeg", "tauPos"],
    "amy": ["amyNeg", "amyPos"],
    "dx_amy": [
        "CN_amyNeg",
        "CN_amyPos",
        "MCI_amyNeg",
        "MCI_amyPos",
        "Dementia_amyNeg",
        "Dementia_amyPos",
    ],
    "dx_tau": [
        "CN_tauNeg",
        "CN_tauPos",
        "MCI_tauNeg",
        "MCI_tauPos",
        "Dementia_tauNeg",
        "Dementia_tauPos",
    ],
}

DX_AMY_TICKS = [
    "CN (A-)",
    "CN (A+)",
    "MCI (A-)",
    "MCI (A+)",
    "Dem (A-)",
    "Dem (A+)",
]

COGNITIVE_TESTS = {
    "MMSE": "MMSE",
    "CDR_sum_of_boxes": "CSR_SB",
    "ADAS11_TOTAL": "ADAS11",
    "ADAS13_TOTAL": "ADAS13",
    "Q4SCORE": "ADASQ4",
    "MOCA": "MOCA",
    # "PACC_raw": "PACC"
}


RIDS = [  # Specific ids to visualise if short on time - representative across stage
    # "5093",  # CN_amyNeg_tauNeg-train
    # "4580",  # CN_amyNeg_tauNeg-train
    # "6043",  # CN_amyNeg_tauNeg-train
    # "6202",  # CN_amyNeg_tauNeg-train
    # "6273",  # CN_amyNeg_tauNeg-train
    # "4076",  # train
    # "4369",  # train
    # "4951",  # CN_amyNeg_tauNeg-test
    # "5083",  # CN_amyPos_tauNeg
    # "5097",  # CN_amyPos_tauNeg
    # "5177",
    # "6330",
    # "6572",
    # "6716",
    "6701",  # CN_amyPos_tauPos
    "6598",  # MCI_amyPos_tauPos
    "6606",  # MCI_amyPos_tauPos
    "6632",  # MCI_amyPos_tauPos
    # "5277",  # CN_amyPos_tauNeg
    # "6061",  # CN_amyPos_tauNeg
    # "6068",  # MCI_amyPos_tauNeg
    # "0908",  # MCI_amyNeg_tauNeg
    # "1052",  # MCI_amyNeg_tauNeg
    # "2119",  # MCI_amyNeg_tauNeg
    # "4115",  # Dem_amyNeg_tauNeg
    # "4162",
    # "4293",  # Dem_amyNeg_tauNeg
    "4302",  # MCI_amyPos_tauPos
    # "4404",  # Dem_amyPos_tauNeg
    "4414",  # Dem_amyPos_tauPos
    "4521",  # Dem_amyPos_tauPos
    # "6173",  # CN/MCI_amyPos_tauNeg
    "6179",  # Dem_amyPos_tauNeg
    # "6241",  # Dem/MCI_amyPos_tauPos/Neg
    "6303",  # Dem_amyPos_tauNeg
    "6595",  # Dem_amyPos_tauPos
    "6600",  # Dem_amyPos_tauPos
    "6661",
]
