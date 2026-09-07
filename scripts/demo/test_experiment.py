# Test SurvivalExperiment — Competing-Risk Orchestration

import numpy as np
import pandas as pd

from pd_risk.modelling.experiment import SurvivalExperiment


# =========================================================
# Simple test components
# =========================================================

class TestPreprocessor:
    """Minimal preprocessor for experiment orchestration testing."""

    def fit_transform(self, X):
        return X.drop(columns=["PATNO"]).copy()

    def transform(self, X):
        return X.drop(columns=["PATNO"]).copy()


class TestFusion:
    """Minimal fusion implementation."""

    def __init__(self):
        self.feature_names = None

    def fit_transform(self, modalities):
        frames = []

        for modality, df in modalities.items():
            df = df.copy()

            renamed = {
                column: f"{modality}__{column}"
                for column in df.columns
            }

            df = df.rename(columns=renamed)
            frames.append(df.reset_index(drop=True))

        result = pd.concat(
            frames,
            axis=1,
        )

        self.feature_names = list(
            result.columns
        )

        return result

    def transform(self, modalities):
        frames = []

        for modality, df in modalities.items():
            df = df.copy()

            renamed = {
                column: f"{modality}__{column}"
                for column in df.columns
            }

            df = df.rename(columns=renamed)
            frames.append(df.reset_index(drop=True))

        return pd.concat(
            frames,
            axis=1,
        )

    def get_feature_names(self):
        return self.feature_names


class TestModel:
    """
    Minimal competing-risk model double.

    It deliberately uses event_type to demonstrate that
    the experiment passes competing-risk targets through
    to the survival model.
    """

    def __init__(self):
        self.is_fitted = False
        self.fit_event_types = None

    def fit(self, X, y):

        self.is_fitted = True

        self.fit_event_types = set(
            y["event_type"].unique()
        )

        if not self.fit_event_types.issubset(
            {0, 1, 2}
        ):
            raise ValueError(
                "Unexpected event types."
            )

        return self

    def predict_risk(self, X):

        # Deterministic finite risk score.
        return np.arange(
            1,
            len(X) + 1,
            dtype=float,
        )

    def predict_pd_cumulative_incidence(
        self,
        X,
        time,
    ):

        # Deterministic values in [0, 1].
        return np.linspace(
            0.10,
            0.40,
            len(X),
        )

    def summary(self):

        return {
            "test_model": True,
            "model_type": "competing_risk",
        }


class TestSplitter:
    """Two-fold deterministic splitter."""

    def split(self, X, y):

        n = len(X)
        midpoint = n // 2

        yield (
            np.arange(
                0,
                midpoint,
            ),
            np.arange(
                midpoint,
                n,
            ),
        )

        yield (
            np.arange(
                midpoint,
                n,
            ),
            np.arange(
                0,
                midpoint,
            ),
        )


class TestEvaluator:
    """
    Minimal evaluator implementing the interface
    required by SurvivalExperiment.
    """

    def evaluate(
        self,
        y_train,
        y_test,
        risk_scores,
        predicted_pd_cif,
        horizon,
    ):

        return {
            "uno_c_index": 0.75,
            "brier_score": 0.10,
            "null_brier_score": 0.12,
            "brier_skill_score": 0.1667,
        }

    def assign_risk_groups(
        self,
        predictions,
        n_groups=4,
    ):

        predictions = predictions.copy()

        predictions["risk_group"] = (
            pd.qcut(
                predictions["risk_score"],
                q=n_groups,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

        return predictions

    def build_risk_groups(
        self,
        predictions,
    ):

        return (
            predictions
            .groupby("risk_group")
            .agg(
                n=("event_type", "size"),
                pd_events=(
                    "event_type",
                    lambda x: int(
                        (x == 1).sum()
                    ),
                ),
                deaths=(
                    "event_type",
                    lambda x: int(
                        (x == 2).sum()
                    ),
                ),
                mean_predicted_risk=(
                    "pd_cif",
                    "mean",
                ),
            )
            .reset_index()
        )

    def build_cumulative_incidence(
        self,
        predictions,
        horizon,
    ):

        return {
            "horizon": horizon,
            "cause": 1,
            "n": len(predictions),
        }

    def build_gray_test(
        self,
        predictions,
        cause,
    ):

        return {
            "test_statistic": 1.5,
            "degrees_of_freedom": 1,
            "p_value": 0.20,
            "cause": cause,
        }

    def build_calibration(
        self,
        predictions,
        horizon,
    ):

        return pd.DataFrame(
            {
                "calibration_group": [1, 2],
                "mean_predicted_risk": [
                    0.15,
                    0.35,
                ],
                "observed_risk": [
                    0.12,
                    0.31,
                ],
                "horizon": [
                    horizon,
                    horizon,
                ],
            }
        )

    def plot_cumulative_incidence(
        self,
        cumulative_incidence,
    ):

        return "test_cif_plot"

    def plot_calibration(
        self,
        calibration,
        horizon,
    ):

        return "test_calibration_plot"


# =========================================================
# Test cohort
# =========================================================

def build_test_data():

    n = 12

    patno = np.arange(
        1001,
        1001 + n,
    )

    event_type = np.array(
        [
            0, 1, 2,
            0, 1, 2,
            0, 1, 2,
            0, 1, 2,
        ]
    )

    time_to_event = np.array(
        [
            0.5, 1.0, 1.5,
            2.0, 2.5, 3.0,
            0.7, 1.2, 1.8,
            2.2, 2.8, 3.5,
        ]
    )

    event = (
        event_type != 0
    ).astype(int)

    y = pd.DataFrame(
        {
            "PATNO": patno,
            "time_to_event":
                time_to_event,
            "event": event,
            "event_type": event_type,
        }
    )

    identifiers = pd.DataFrame(
        {
            "PATNO": patno,
        }
    )

    clinical = pd.DataFrame(
        {
            "PATNO": patno,
            "age": np.linspace(
                55,
                70,
                n,
            ),
            "score": np.linspace(
                1,
                12,
                n,
            ),
        }
    )

    dat = pd.DataFrame(
        {
            "PATNO": patno,
            "dat_feature": np.linspace(
                0.1,
                1.2,
                n,
            ),
        }
    )

    modality_data = {
        "clinical": clinical,
        "dat": dat,
    }

    return (
        modality_data,
        y,
        identifiers,
    )


# =========================================================
# Run test
# =========================================================

def main():

    print("=" * 60)
    print("TESTING SURVIVAL EXPERIMENT")
    print("=" * 60)

    (
        modality_data,
        y,
        identifiers,
    ) = build_test_data()

    config = {
        "analysis": {
            "evaluation": {
                "time_horizon": 2,
            },
            "validation": {
                "strategy": "test_split",
            },
            "model": {
                "name": "CauseSpecificCoxModel",
            },
        }
    }

    preprocessor_factories = {
        "clinical":
            lambda: TestPreprocessor(),

        "dat":
            lambda: TestPreprocessor(),
    }

    fusion_factory = (
        lambda: TestFusion()
    )

    model_factory = (
        lambda: TestModel()
    )

    splitter = TestSplitter()

    evaluator = TestEvaluator()

    # =====================================================
    # Construct experiment
    # =====================================================

    print("\n[1] Construct experiment")

    experiment = SurvivalExperiment(
        modality_data=modality_data,
        y=y,
        identifiers=identifiers,
        splitter=splitter,
        preprocessor_factories=(
            preprocessor_factories
        ),
        fusion_factory=fusion_factory,
        model_factory=model_factory,
        evaluator=evaluator,
        config=config,
    )

    print("PASS")

    # =====================================================
    # Run
    # =====================================================

    print("\n[2] Run experiment")

    summary = experiment.run()

    print("PASS")

    # =====================================================
    # OOF predictions
    # =====================================================

    print("\n[3] OOF predictions")

    predictions = (
        experiment.oof_predictions
    )

    assert predictions is not None

    assert len(predictions) == len(y)

    assert predictions["PATNO"].nunique() == len(y)

    print(
        f"OOF predictions: {len(predictions)}"
    )

    print("PASS")

    # =====================================================
    # Event type preservation
    # =====================================================

    print("\n[4] Event types")

    assert "event_type" in predictions.columns

    observed_event_types = set(
        predictions["event_type"].unique()
    )

    assert observed_event_types == {
        0,
        1,
        2,
    }

    print(
        f"Event types: "
        f"{sorted(observed_event_types)}"
    )

    print("PASS")

    # =====================================================
    # Competing-risk predictions
    # =====================================================

    print("\n[5] PD cumulative incidence")

    assert "pd_cif" in predictions.columns

    assert (
        predictions["pd_cif"]
        .between(0, 1)
        .all()
    )

    assert np.isfinite(
        predictions["pd_cif"]
    ).all()

    print(
        "PD CIF values valid"
    )

    print("PASS")

    # =====================================================
    # Risk score
    # =====================================================

    print("\n[6] Risk scores")

    assert "risk_score" in predictions.columns

    assert np.isfinite(
        predictions["risk_score"]
    ).all()

    print("PASS")

    # =====================================================
    # Fold results
    # =====================================================

    print("\n[7] Fold results")

    assert len(
        experiment.fold_results
    ) == 2

    for result in experiment.fold_results:

        assert (
            "uno_c_index"
            in result
        )

        assert (
            "brier_score"
            in result
        )

        assert (
            "null_brier_score"
            in result
        )

        assert (
            "brier_skill_score"
            in result
        )

    print(
        f"Folds: "
        f"{len(experiment.fold_results)}"
    )

    print("PASS")

    # =====================================================
    # Evaluation
    # =====================================================

    print("\n[8] Cohort-level evaluation")

    assert experiment.evaluation is not None

    expected_outputs = {
        "risk_groups",
        "cumulative_incidence",
        "gray_test",
        "calibration",
        "cumulative_incidence_plot",
        "calibration_plot",
    }

    assert expected_outputs.issubset(
        experiment.evaluation.keys()
    )

    print("PASS")

    # =====================================================
    # Final model
    # =====================================================

    print("\n[9] Final model")

    assert (
        experiment.final_model
        is not None
    )

    assert (
        experiment.final_fusion
        is not None
    )

    assert (
        len(
            experiment.final_preprocessors
        )
        == 2
    )

    print("PASS")

    # =====================================================
    # Summary
    # =====================================================

    print("\n[10] Summary")

    assert summary is not None

    assert (
        summary["dataset"]["n_samples"]
        == len(y)
    )

    assert (
        summary["dataset"]["n_pd_events"]
        == int(
            (y["event_type"] == 1).sum()
        )
    )

    assert (
        summary["dataset"]["n_deaths"]
        == int(
            (y["event_type"] == 2).sum()
        )
    )

    assert (
        summary["dataset"]["n_censored"]
        == int(
            (y["event_type"] == 0).sum()
        )
    )

    assert (
        summary["dataset"]
        ["time_horizon_years"]
        == 2
    )

    assert (
        "uno_c_index"
        in summary["performance"]
    )

    assert (
        "brier_score"
        in summary["performance"]
    )

    assert (
        "null_brier_score"
        in summary["performance"]
    )

    assert (
        "brier_skill_score"
        in summary["performance"]
    )

    print("PASS")

    # =====================================================
    # Final report
    # =====================================================

    print()
    print("=" * 60)
    print("SURVIVAL EXPERIMENT TEST PASSED")
    print("=" * 60)

    print(
        f"Participants: "
        f"{summary['dataset']['n_samples']}"
    )

    print(
        f"PD events: "
        f"{summary['dataset']['n_pd_events']}"
    )

    print(
        f"Deaths: "
        f"{summary['dataset']['n_deaths']}"
    )

    print(
        f"Censored: "
        f"{summary['dataset']['n_censored']}"
    )

    print(
        f"Horizon: "
        f"{summary['dataset']['time_horizon_years']} years"
    )

    print(
        f"OOF predictions: "
        f"{len(predictions)}"
    )

    print()
    print("PASS")


if __name__ == "__main__":
    main()
