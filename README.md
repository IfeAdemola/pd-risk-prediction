# PD Risk Prediction

A research framework for developing multimodal machine learning models for risk prediction and early diagnosis of Parkinson's disease.

## Project Objectives

This project investigates how multimodal clinical data can be integrated to predict an individual's future risk of developing Parkinson's disease.

Main research directions include:

* Survival-based risk prediction
* Competing-risk survival analysis
* Multimodal data integration
* Longitudinal disease progression modelling
* Handling missing clinical data
* Model evaluation and clinical interpretation

## Repository Structure

```text
configs/        Experiment configurations
data/           Dataset files (not tracked by Git)
src/            Main Python package
scripts/        Executable scripts
output/         Generated experiment results, artifacts, and figures (not tracked by Git)
notebooks/      Exploratory analysis notebooks
tests/          Automated tests
```

## Current Statistical Framework

The primary estimand is the **two-year cumulative incidence of Parkinson's disease**, with death before Parkinson's disease treated as a competing event.

Event types are encoded as:

* `0` — censored
* `1` — Parkinson's disease
* `2` — death before Parkinson's disease

The survival model uses cause-specific Cox regression to estimate the cause-specific hazards for Parkinson's disease and death. These hazards are combined to obtain the predicted Parkinson's disease cumulative incidence.

Model evaluation includes:

* Uno's C-index
* IPCW Brier score
* Null-model Brier score
* Brier skill score
* Aalen-Johansen cumulative incidence
* Calibration
* Predicted-risk groups
* Gray's test for differences in cumulative incidence between risk groups

## Running an Experiment

The main entry point for running an experiment is:

```text
scripts/run_experiment.py
```

From the project root directory, run:

```bash
python scripts/run_experiment.py
```

For example, if the project is located at:

```text
C:\Users\username\pd-risk-prediction
```

open a terminal in that directory:

```bash
cd C:\Users\username\pd-risk-prediction
```

and run:

```bash
python scripts/run_experiment.py
```

The script connects the complete modelling pipeline, including:

1. Loading and preparing the modelling dataset
2. Selecting the configured modalities
3. Preprocessing each modality
4. Fusing multimodal features
5. Performing cross-validation
6. Fitting the competing-risk survival model
7. Generating out-of-fold predictions
8. Evaluating model performance
9. Fitting the final model on the complete modelling cohort
10. Saving experiment artifacts and figures

The experiment configuration is controlled through the files in `configs/`.

## Experiment Outputs

Each experiment is saved in a timestamped directory under:

```text
output/experiments/
```

Experiment directories follow a convention based on the experiment configuration, for example:

```text
output/experiments/
└── 20260825_120952_clinical_cause_specific_cox/
```

This avoids requiring a manually unique experiment name for every run.

Depending on the artifact configuration, an experiment directory may contain:

```text
config.yaml
metadata.json
summary.json
fold_results.csv
oof_predictions.csv
risk_groups.csv
cumulative_incidence.csv
gray_test.json
calibration.csv

model/
    pd.csv
    death.csv

plots/
    cumulative_incidence.png
    calibration.png
```

The `oof_predictions.csv` file contains the out-of-fold predictions together with participant identifiers, event information, risk scores, and predicted Parkinson's disease cumulative incidence.

## Current Status

The project infrastructure and primary competing-risk survival modelling pipeline are implemented.

The current workflow supports unimodal and multimodal experiment configurations, cross-validated out-of-fold evaluation, competing-risk survival modelling, and automated experiment artifact generation.

Further work will focus on statistical feasibility, model stability, uncertainty estimation, scientific interpretation, and comparison of unimodal and multimodal models.
