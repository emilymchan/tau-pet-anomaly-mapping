"""Entry point for MAEMI preprocessing, training, inference, and analysis.

Patch generation and training use the standard Hydra configuration.

Inference uses the standard configuration to locate the modality folder,
then loads the configuration saved with the experiment.

Analysis uses config_analysis.yaml.
"""

import argparse
import sys
from pathlib import Path

MAEMI_DIR = Path(__file__).resolve().parent
REPO_ROOT = MAEMI_DIR.parent

sys.path.insert(0, str(MAEMI_DIR))
sys.path.insert(0, str(REPO_ROOT))

from data import generate_patches
from training import train
from inference import mae_predict
from analysis import get_roi_values, roc_analysis, correlate_with_cognitive_tests


def main():
    parser = argparse.ArgumentParser(description="MAEMI pipeline")

    parser.add_argument(
        "task",
        choices=[
            "generate-patches",
            "train",
            "predict",
            "roi-values",
            "roc",
            "correlate",
        ],
        help="Operation to run",
    )

    parser.add_argument(
        "--run-dir",
        help="Experiment directory for inference",
    )

    args, remaining_args = parser.parse_known_args()

    # Hydra-decorated functions read arguments from sys.argv.
    sys.argv = [sys.argv[0]] + remaining_args

    if args.task == "generate-patches":
        generate_patches.main()

    elif args.task == "train":
        train.main()

    elif args.task == "predict":
        if not args.run_dir:
            parser.error("--run-dir is required for prediction")

        mae_predict.main_wrapper(args.run_dir)

    elif args.task == "roi-values":
        get_roi_values.main()

    elif args.task == "roc":
        roc_analysis.main()

    elif args.task == "correlate":
        correlate_with_cognitive_tests.main()


if __name__ == "__main__":
    main()