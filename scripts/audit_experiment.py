from pathlib import Path
import json

import pandas as pd


def require_file(
    experiment_dir: Path,
    filename: str,
):
    path = experiment_dir / filename

    if not path.exists():
        raise AssertionError(
            f"Missing artifact: {path}"
        )

    print(f"[OK] {filename}")

    return path


def main():

    # Change this to the experiment directory
    # you want to audit.
    experiment_dir = Path(
        "output/experiments/20260816_201921_cox_ph_confirmed_pd"
    )

    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"Experiment directory does not exist: "
            f"{experiment_dir}"
        )

    print(
        f"\nAuditing experiment:\n"
        f"{experiment_dir}\n"
    )

    # -------------------------
    # Required artifacts
    # -------------------------

    require_file(
        experiment_dir,
        "config.yaml",
    )

    require_file(
        experiment_dir,
        "summary.json",
    )

    require_file(
        experiment_dir,
        "cv_results.csv",
    )

    require_file(
        experiment_dir,
        "predictions.csv",
    )

    require_file(
        experiment_dir,
        "risk_groups.csv",
    )

    require_file(
        experiment_dir,
        "metadata.json",
    )

    require_file(
        experiment_dir,
        "calibration.csv",
    )

    calibration_plot = (
        experiment_dir
        / "plots"
        / "calibration.png"
    )

    if not calibration_plot.exists():
        raise AssertionError(
            f"Missing calibration plot: "
            f"{calibration_plot}"
        )

    print(
        "[OK] plots/calibration.png"
    )

    # -------------------------
    # Load tabular artifacts
    # -------------------------

    cv_results = pd.read_csv(
        experiment_dir / "cv_results.csv"
    )

    predictions = pd.read_csv(
        experiment_dir / "predictions.csv"
    )

    risk_groups = pd.read_csv(
        experiment_dir / "risk_groups.csv"
    )

    calibration = pd.read_csv(
        experiment_dir / "calibration.csv"
    )

    with open(
        experiment_dir / "summary.json",
        "r",
        encoding="utf-8",
    ) as f:
        summary = json.load(f)

    with open(
        experiment_dir / "metadata.json",
        "r",
        encoding="utf-8",
    ) as f:
        metadata = json.load(f)

    # -------------------------
    # CV checks
    # -------------------------

    n_folds = summary[
        "validation"
    ]["n_folds"]

    if len(cv_results) != n_folds:
        raise AssertionError(
            "CV result count does not match "
            "summary n_folds"
        )

    print(
        f"[OK] CV folds: {len(cv_results)}"
    )

    # -------------------------
    # Prediction checks
    # -------------------------

    expected_samples = summary[
        "dataset"
    ]["n_samples"]

    if len(predictions) != expected_samples:
        raise AssertionError(
            "Prediction count does not match "
            "dataset sample count"
        )

    print(
        f"[OK] predictions: {len(predictions)}"
    )

    required_prediction_columns = {
        "PATNO",
        "fold",
        "time_to_event",
        "event",
        "risk_score",
        "survival_probability",
        "predicted_risk",
        "risk_group",
    }

    missing = (
        required_prediction_columns
        - set(predictions.columns)
    )

    if missing:
        raise AssertionError(
            f"Missing prediction columns: {missing}"
        )

    print(
        "[OK] prediction columns"
    )

    # -------------------------
    # Event checks
    # -------------------------

    n_events = int(
        predictions["event"].sum()
    )

    expected_events = summary[
        "dataset"
    ]["n_events"]

    if n_events != expected_events:
        raise AssertionError(
            "Prediction event count does not "
            "match dataset event count"
        )

    print(
        f"[OK] events: {n_events}"
    )

    # -------------------------
    # Risk checks
    # -------------------------

    if not (
        (
            predictions[
                "predicted_risk"
            ] >= 0
        )
        & (
            predictions[
                "predicted_risk"
            ] <= 1
        )
    ).all():

        raise AssertionError(
            "Predicted risks outside [0, 1]"
        )

    if not (
        (
            predictions[
                "survival_probability"
            ] >= 0
        )
        & (
            predictions[
                "survival_probability"
            ] <= 1
        )
    ).all():

        raise AssertionError(
            "Survival probabilities outside [0, 1]"
        )

    print(
        "[OK] probabilities in [0, 1]"
    )

    # Check consistency between risk
    # and survival probability.

    if not (
        (
            predictions[
                "predicted_risk"
            ]
            + predictions[
                "survival_probability"
            ]
        ).sub(1).abs() < 1e-10
    ).all():

        raise AssertionError(
            "predicted_risk is not "
            "1 - survival_probability"
        )

    print(
        "[OK] risk/probability consistency"
    )

    # -------------------------
    # Risk-group checks
    # -------------------------

    prediction_groups = set(
        predictions["risk_group"]
        .dropna()
        .astype(int)
        .unique()
    )

    summary_groups = set(
        risk_groups["risk_group"]
        .astype(int)
        .unique()
    )

    if prediction_groups != summary_groups:
        raise AssertionError(
            "Risk groups in predictions and "
            "risk_groups.csv do not match"
        )

    print(
        "[OK] risk groups"
    )

    # -------------------------
    # Calibration checks
    # -------------------------

    calibration_config = summary[
        "calibration"
    ]

    if calibration_config["enabled"]:

        expected_horizon = (
            calibration_config["horizon"]
        )

        expected_groups = (
            calibration_config["n_groups"]
        )

        print(
            "\nCalibration audit values:"
        )

        print(
            "Expected horizon from summary:",
            expected_horizon,
        )

        print(
            "Horizons in calibration.csv:",
            calibration["horizon"].unique(),
        )

        print(
            "Calibration dtypes:"
        )

        print(
            calibration.dtypes
        )

        if not (
            calibration["horizon"]
            == expected_horizon
        ).all():

            raise AssertionError(
                "Calibration horizon does not "
                "match summary"
            )

        if len(calibration) != expected_groups:
            raise AssertionError(
                "Calibration group count does "
                "not match summary"
            )

        print(
            "[OK] calibration horizon"
        )

        print(
            "[OK] calibration groups"
        )

        if not (
            calibration[
                "mean_predicted_risk"
            ].between(0, 1)
        ).all():

            raise AssertionError(
                "Calibration predicted risks "
                "outside [0, 1]"
            )

        if not (
            calibration[
                "observed_risk"
            ].between(0, 1)
        ).all():

            raise AssertionError(
                "Calibration observed risks "
                "outside [0, 1]"
            )

        print(
            "[OK] calibration probabilities"
        )

    # -------------------------
    # Performance checks
    # -------------------------

    performance = summary[
        "performance"
    ]

    c_index = performance[
        "c_index"
    ]["mean"]

    brier_score = performance[
        "brier_score"
    ]["mean"]

    if not 0 <= c_index <= 1:
        raise AssertionError(
            f"Invalid C-index: {c_index}"
        )

    if brier_score < 0:
        raise AssertionError(
            f"Invalid Brier score: "
            f"{brier_score}"
        )

    print(
        f"[OK] mean C-index: {c_index:.4f}"
    )

    print(
        f"[OK] mean Brier score: "
        f"{brier_score:.4f}"
    )

    # -------------------------
    # Metadata checks
    # -------------------------

    if not metadata.get(
        "run_id"
    ):
        raise AssertionError(
            "Metadata missing run_id"
        )

    print(
        "[OK] metadata"
    )

    logrank_path = experiment_dir / "logrank.json"

    if not logrank_path.exists():
        raise AssertionError(
            "Missing logrank.json"
        )

    with open(
        logrank_path,
        "r",
        encoding="utf-8",
    ) as f:
        logrank = json.load(f)

    assert "test_statistic" in logrank
    assert "p_value" in logrank

    assert (
        0 <= logrank["p_value"] <= 1
    )

    print(
        f"[OK] log-rank p-value: "
        f"{logrank['p_value']:.4g}"
    )

    km_plot_path = (
        experiment_dir
        / "plots"
        / "kaplan_meier.png"
    )

    if not km_plot_path.exists():
        raise AssertionError(
            "Missing Kaplan-Meier plot"
        )

    print(
        "[OK] plots/kaplan_meier.png"
    )

    # -------------------------
    # Final result
    # -------------------------

    print(
        "\n================================"
    )

    print(
        "EXPERIMENT AUDIT PASSED"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()