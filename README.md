# Individual-Specific 3D Tau PET Anomaly Mapping

Code accompanying the manuscript **“Individual-Specific Three-Dimensional Tau PET Anomaly Mapping: External Validation and Comparison with Expert Visual Assessment”** (in preparation).

This repository contains deep learning pipelines for generating individual-specific, three-dimensional tau PET anomaly maps using two approaches:

- **Masked autoencoder (MAE):** Reconstruction-based anomaly detection using the MAEMI framework.
- **Denoising diffusion probabilistic model (DDPM):** Reconstruction-based anomaly detection using a diffusion model.

The resulting anomaly maps are evaluated using regional measurements and compared with expert visual assessment.

## Repository structure

- `MAEMI/` — MAE training, inference and analysis.
- `DDPM/` — DDPM training, inference and analysis.
- `MAEMI/data/` — Data preprocessing and patch generation.
- `MAEMI/analysis/` — Regional analysis and evaluation.
- `MAEMI/utils/` — Supporting utilities.

The repository also includes notebooks used to produce selected analyses and figures for the manuscript.

## Data availability

The study uses tau PET data from the Alzheimer's Disease Neuroimaging Initiative (ADNI) and the A4 study.

Configure local data paths before running the pipelines.

## Environment setup

Create an environment using the mae.yml file.

```bash
conda env create -f MAEMI/mae.yml
conda activate mae
```

The original MAEMI implementation uses `timm==0.3.2`. To resolve a compatibility issue with newer PyTorch versions, run:

```bash
python MAEMI/scripts/patch_timm.py
```

## Configuration

Both pipelines require users to configure their local data and output directories before running the code.

**MAE:** Configuration is managed using [Hydra](https://hydra.cc/). Set the required environment variables for your data and project directories, and adjust the configuration files in `MAEMI/config/` as needed.

**DDPM:** Configuration is managed through `DDPM/projects/tauPET_anomaly/config/ddpm.yaml`. Update `project_dir` to point to your local project directory and `experiment.weights` to the model checkpoint used for inference. The remaining data and output paths are defined relative to `project_dir`.

The analysis notebooks may also require local data, metadata and results directories to be configured before execution.

## Running the code

The MAE pipeline provides a command-line entry point in `MAEMI/main.py`. Run it from the `MAEMI/` directory:

```bash
cd MAEMI
python main.py --help
```

The available commands cover the relevant stages of training, inference and analysis. See the configuration files for experiment-specific settings.

The DDPM implementation and its corresponding scripts are located in `DDPM/`.

## Implementation and acknowledgements

The MAE implementation is adapted from [MAEMI](https://github.com/LangDaniel/MAEMI). The DDPM implementation builds on the standard denoising diffusion probabilistic model, with components adapted from [MONAI](https://github.com/project-monai/monai) and from [ci-ber's implementation of DDPM](https://github.com/compai-lab/2024-miccai-bercea-thor). Original attribution and licence information are retained in the relevant source files.

Please refer to the original implementations for further details and acknowledge them when reusing the corresponding code.

## License

This repository is licensed under the Apache License 2.0. See LICENSE for details. Adapted third-party code remains subject to its original copyright notices and applicable licences.

## Citation

If you use this repository, please cite the accompanying manuscript once it is available:

*Individual-Specific Three-Dimensional Tau PET Anomaly Mapping: External Validation and Comparison with Expert Visual Assessment.* Manuscript in preparation.