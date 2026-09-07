# `scripts/test_artifacts.py`

from pathlib import Path
from types import SimpleNamespace
import shutil

import numpy as np
import pandas as pd

from pd_risk.modelling.artifacts import (
    ExperimentArtifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_OUTPUT = (
    PROJECT_ROOT
    / "output"
    / "artifact_test"
)


def main():

    print("=" * 60)
    print("TESTING EXPERIMENT ARTIFACTS")
    print("=" * 60)

    # ---------------------------------------------------------
    # Clean previous test output
    # ---------------------------------------------------------

    if TEST_OUTPUT.exists():
        shutil.rmtree(TEST_OUTPUT)

    # ---------------------------------------------------------
    # Fake OOF predictions
    # ---------------------------------------------------------

    oof_predictions = pd.DataFrame(
        {
            "PATNO": [1, 2, 3, 4],
            "fold": [1, 1, 2, 2],
            "time_to_event": [
                2.0,
                1.2,
                2.0,
                0.8,
            ],
            "event_type": [
                1,
                2,
                0,
                1,
            ],
            "risk_score": [
                1.5,
                0.8,
                0.4,
                1.2,
            ],
            "pd_cif": [
                0.40,
                0.15,
                0.10,
                0.35,
            ],
            "predicted_risk": [
                0.40,
                0.15,
                0.10,
                0.35,
            ],
            "risk_group": [
                4,
                2,
                1,
                3,
            ],
        }
    )

    # ---------------------------------------------------------
    # Fake evaluation outputs
    # ---------------------------------------------------------

    risk_groups = pd.DataFrame(
        {
            "risk_group": [1, 2, 3, 4],
            "n": [1, 1, 1, 1],
            "events": [0, 1, 1, 1],
            "event_rate": [
                0.0,
                1.0,
                1.0,
                1.0,
            ],
            "mean_risk": [
                0.10,
                0.15,
                0.35,
                0.40,
            ],
        }
    )

    cumulative_incidence = pd.DataFrame(
        {
            "risk_group": [1, 2, 3, 4],
            "time": [2, 2, 2, 2],
            "pd_cif": [
                0.05,
                0.15,
                0.30,
                0.45,
            ],
        }
    )

    calibration = pd.DataFrame(
        {
            "calibration_group": [1, 2, 3, 4],
            "n": [1, 1, 1, 1],
            "mean_predicted_risk": [
                0.10,
                0.15,
                0.35,
                0.40,
            ],
            "observed_risk": [
                0.08,
                0.18,
                0.31,
                0.43,
            ],
            "horizon": [2, 2, 2, 2],
        }
    )

    gray_test = {
        "statistic": 4.601412,
        "degrees_of_freedom": 1,
        "p_value": 0.0319456,
        "cause": 1,
    }

    # ---------------------------------------------------------
    # Fake model summary
    # ---------------------------------------------------------

    model_summary = {
        "pd": pd.DataFrame(
            {
                "covariate": [
                    "feature_1",
                    "feature_2",
                ],
                "coef": [
                    0.25,
                    -0.10,
                ],
                "hazard_ratio": [
                    1.284,
                    0.905,
                ],
            }
        ),
        "death": pd.DataFrame(
            {
                "covariate": [
                    "feature_1",
                    "feature_2",
                ],
                "coef": [
                    0.15,
                    -0.05,
                ],
                "hazard_ratio": [
                    1.162,
                    0.951,
                ],
            }
        ),
    }

    # ---------------------------------------------------------
    # Fake figures
    # ---------------------------------------------------------

    import matplotlib.pyplot as plt

    ci_fig, ci_ax = plt.subplots()

    ci_ax.plot(
        [0, 1, 2],
        [0, 0.2, 0.4],
    )

    ci_ax.set_title(
        "Test cumulative incidence"
    )

    calibration_fig, calibration_ax = (
        plt.subplots()
    )

    calibration_ax.plot(
        [0, 1],
        [0, 1],
    )

    calibration_ax.set_title(
        "Test calibration"
    )

    # ---------------------------------------------------------
    # Fake experiment
    # ---------------------------------------------------------

    experiment = SimpleNamespace(

        modality_data={
            "clinical": pd.DataFrame(),
            "dat": pd.DataFrame(),
        },

        final_model=SimpleNamespace(
            __class__=SimpleNamespace(
                __name__="CauseSpecificCoxModel"
            )
        ),

        summary={
            "dataset": {
                "n_samples": 4,
                "n_features": 2,
                "n_censored": 1,
                "n_pd_events": 2,
                "n_deaths": 1,
                "time_horizon_years": 2,
            },

            "modalities": [
                "clinical",
                "dat",
            ],

            "validation": {
                "n_folds": 2,
                "strategy": "stratified",
            },

            "performance": {
                "uno_c_index": {
                    "mean": 0.82,
                    "std": 0.03,
                },
            },
        },

        fold_results=[
            {
                "fold": 1,
                "train_samples": 2,
                "validation_samples": 2,
                "uno_c_index": 0.80,
            },
            {
                "fold": 2,
                "train_samples": 2,
                "validation_samples": 2,
                "uno_c_index": 0.84,
            },
        ],

        oof_predictions=oof_predictions,

        evaluation={
            "risk_groups":
                risk_groups,

            "cumulative_incidence":
                cumulative_incidence,

            "gray_test":
                gray_test,

            "calibration":
                calibration,

            "cumulative_incidence_plot":
                ci_fig,

            "calibration_plot":
                calibration_fig,
        },

        final_model_summary=
            model_summary,
    )

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    config = {

        "experiment": {
            "name": None,
        },

        "artifacts": {

            "directory":
                "output/artifact_test",

            "save": {

                "config": True,

                "metadata": True,

                "summary": True,

                "fold_results": True,

                "predictions": True,

                "risk_groups": True,

                "cumulative_incidence": True,

                "gray_test": True,

                "calibration": True,

                "model_summary": True,

                "plots": True,
            },
        },
    }

    # ---------------------------------------------------------
    # Create artifact manager
    # ---------------------------------------------------------

    artifacts = ExperimentArtifacts(
        experiment=experiment,
        config=config,
        project_root=PROJECT_ROOT,
    )

    print("\n[1] Saving artifacts")

    experiment_dir = artifacts.save()

    print(
        f"Created: {experiment_dir}"
    )

    assert experiment_dir.exists()

    print("PASS")

    # ---------------------------------------------------------
    # Check files
    # ---------------------------------------------------------

    expected_files = [

        "config.yaml",

        "metadata.json",

        "summary.json",

        "fold_results.csv",

        "oof_predictions.csv",

        "risk_groups.csv",

        "cumulative_incidence.csv",

        "gray_test.json",

        "calibration.csv",

        "model/pd.csv",

        "model/death.csv",

        "plots/cumulative_incidence.png",

        "plots/calibration.png",
    ]

    print("\n[2] Checking expected files")

    for relative_path in expected_files:

        path = (
            experiment_dir
            / relative_path
        )

        assert path.exists(), (
            f"Missing artifact: {path}"
        )

        print(
            f"PASS: {relative_path}"
        )

    # ---------------------------------------------------------
    # Check OOF predictions
    # ---------------------------------------------------------

    print(
        "\n[3] Checking OOF predictions"
    )

    saved_predictions = pd.read_csv(
        experiment_dir
        / "oof_predictions.csv"
    )

    assert len(saved_predictions) == 4

    assert (
        "event_type"
        in saved_predictions.columns
    )

    assert set(
        saved_predictions["event_type"]
    ) == {0, 1, 2}

    print("PASS")

    # ---------------------------------------------------------
    # Check summary
    # ---------------------------------------------------------

    print(
        "\n[4] Checking summary"
    )

    import json

    with open(
        experiment_dir
        / "summary.json",
        "r",
        encoding="utf-8",
    ) as f:

        summary = json.load(f)

    assert (
        summary["dataset"]["n_samples"]
        == 4
    )

    assert (
        summary["dataset"]["n_pd_events"]
        == 2
    )

    assert (
        summary["dataset"]["n_deaths"]
        == 1
    )

    print("PASS")

    # ---------------------------------------------------------
    # Check Gray's test
    # ---------------------------------------------------------

    print(
        "\n[5] Checking Gray's test"
    )

    with open(
        experiment_dir
        / "gray_test.json",
        "r",
        encoding="utf-8",
    ) as f:

        saved_gray = json.load(f)

    assert (
        saved_gray["cause"] == 1
    )

    assert np.isclose(
        saved_gray["p_value"],
        0.0319456,
    )

    print("PASS")

    # ---------------------------------------------------------
    # Check model summaries
    # ---------------------------------------------------------

    print(
        "\n[6] Checking model summaries"
    )

    pd_summary = pd.read_csv(
        experiment_dir
        / "model"
        / "pd.csv"
    )

    death_summary = pd.read_csv(
        experiment_dir
        / "model"
        / "death.csv"
    )

    assert len(pd_summary) == 2

    assert len(death_summary) == 2

    print("PASS")

    # ---------------------------------------------------------
    # Check plots
    # ---------------------------------------------------------

    print(
        "\n[7] Checking plots"
    )

    assert (
        (
            experiment_dir
            / "plots"
            / "cumulative_incidence.png"
        ).stat().st_size > 0
    )

    assert (
        (
            experiment_dir
            / "plots"
            / "calibration.png"
        ).stat().st_size > 0
    )

    print("PASS")

    # ---------------------------------------------------------
    # Finish
    # ---------------------------------------------------------

    ci_fig.clf()
    calibration_fig.clf()

    print()
    print("=" * 60)
    print("ARTIFACT TEST PASSED")
    print("=" * 60)

    print(
        f"Artifact directory:\n{experiment_dir}"
    )


if __name__ == "__main__":
    main()
