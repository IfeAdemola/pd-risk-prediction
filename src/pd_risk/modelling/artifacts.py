# `src/pd_risk/modelling/artifacts.py`

from __future__ import annotations

import re
import json
import platform
from datetime import datetime
from pathlib import Path

import numpy as np 
import pandas as pd
import yaml


class ExperimentArtifacts:
    """
    Save outputs from a competing-risk survival experiment.

    The artifact layer is responsible only for organising and
    persisting experiment outputs. Statistical calculations and
    model fitting remain in the experiment/evaluation/model layers.
    """

    def __init__(
        self,
        experiment,
        config,
        project_root=None,
    ):
        self.experiment = experiment
        self.config = config

        if project_root is None:
            project_root = Path.cwd()

        self.project_root = Path(project_root)

        self.artifact_config = (
            config.get("artifacts", {})
        )

        self.save_config = (
            self.artifact_config
            .get("save", {})
        )

        self.output_root = (
            self.project_root
            / self.artifact_config.get(
                "directory",
                "output/experiments",
            )
        )

        self.experiment_name = (
            config.get("experiment", {})
            .get("name")
        )

        self.experiment_dir = None

    # =========================================================
    # Public API
    # =========================================================

    def save(self):
        """
        Save the configured experiment artifacts.

        Returns
        -------
        Path
            Experiment output directory.
        """

        self._create_experiment_directory()

        if self._enabled("config"):
            self._save_config()

        if self._enabled("metadata"):
            self._save_metadata()

        if self._enabled("summary"):
            self._save_summary()

        if self._enabled("fold_results"):
            self._save_fold_results()

        if self._enabled("predictions"):
            self._save_predictions()

        if self._enabled("risk_groups"):
            self._save_risk_groups()

        if self._enabled("cumulative_incidence"):
            self._save_cumulative_incidence()

        if self._enabled("gray_test"):
            self._save_gray_test()

        if self._enabled("calibration"):
            self._save_calibration()

        if self._enabled("model_summary"):
            self._save_model_summary()

        if self._enabled("plots"):
            self._save_plots()

        return self.experiment_dir

    # =========================================================
    # Directory
    # =========================================================

    def _create_experiment_directory(self):

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        if self.experiment_name:

            name = str(
                self.experiment_name
            ).strip()

        else:

            name = self._generate_experiment_name()

        directory_name = (
            f"{timestamp}_{name}"
        )

        self.experiment_dir = (
            self.output_root
            / directory_name
        )

        self.experiment_dir.mkdir(
            parents=True,
            exist_ok=False,
        )

        return self.experiment_dir


    def _generate_experiment_name(self):

        modalities = (
            self.experiment.modality_data.keys()
        )

        modality_name = "_".join(
            str(modality)
            for modality in modalities
        )

        model_name = (
            self.experiment
            .final_model
            .__class__
            .__name__
        )

        model_name = (
            model_name
            .replace("Model", "")
        )

        model_name = re.sub(
            r"(?<!^)(?=[A-Z])",
            "_",
            model_name,
        ).lower()

        return (
            f"{modality_name}_{model_name}"
        )
    # =========================================================
    # Configuration
    # =========================================================

    def _save_config(self):

        path = (
            self.experiment_dir
            / "config.yaml"
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            yaml.safe_dump(
                self.config,
                f,
                sort_keys=False,
            )

    # =========================================================
    # Metadata
    # =========================================================

    def _save_metadata(self):

        summary = (
            self.experiment.summary
        )

        metadata = {
            "timestamp": (
                datetime.now()
                .isoformat()
            ),

            "experiment_name":
                self.experiment_name,

            "generated_directory":
                self.experiment_dir.name,

            "python_version":
                platform.python_version(),

            "platform":
                platform.platform(),

            "model_class":
                (
                    self.experiment
                    .final_model
                    .__class__
                    .__name__
                ),

            "modalities":
                list(
                    self.experiment
                    .modality_data
                    .keys()
                ),

            "dataset":
                summary.get(
                    "dataset",
                    {},
                ),

            "validation":
                summary.get(
                    "validation",
                    {},
                ),

            "time_horizon_years":
                summary.get(
                    "dataset",
                    {}
                ).get(
                    "time_horizon_years"
                ),

            "event_definitions": {
                "0": "censored",
                "1": "PD",
                "2": "death before PD",
            },
        }

        self._write_json(
            "metadata.json",
            metadata,
        )

    # =========================================================
    # Summary
    # =========================================================

    def _save_summary(self):

        self._write_json(
            "summary.json",
            self.experiment.summary,
        )

    # =========================================================
    # Fold results
    # =========================================================

    def _save_fold_results(self):

        fold_results = (
            self.experiment.fold_results
        )

        if not fold_results:
            return

        pd.DataFrame(
            fold_results
        ).to_csv(
            self.experiment_dir
            / "fold_results.csv",
            index=False,
        )

    # =========================================================
    # OOF predictions
    # =========================================================

    def _save_predictions(self):

        predictions = (
            self.experiment
            .oof_predictions
        )

        if predictions is None:
            return

        predictions.to_csv(
            self.experiment_dir
            / "oof_predictions.csv",
            index=False,
        )

    # =========================================================
    # Risk groups
    # =========================================================

    def _save_risk_groups(self):

        evaluation = (
            self.experiment.evaluation
        )

        if not evaluation:
            return

        risk_groups = (
            evaluation.get(
                "risk_groups"
            )
        )

        if risk_groups is None:
            return

        if isinstance(
            risk_groups,
            pd.DataFrame,
        ):

            risk_groups.to_csv(
                self.experiment_dir
                / "risk_groups.csv",
                index=False,
            )

    # =========================================================
    # Cumulative incidence
    # =========================================================

    def _save_cumulative_incidence(self):

        evaluation = (
            self.experiment.evaluation
        )

        if not evaluation:
            return

        cumulative_incidence = (
            evaluation.get(
                "cumulative_incidence"
            )
        )

        if cumulative_incidence is None:
            return

        if isinstance(
            cumulative_incidence,
            pd.DataFrame,
        ):

            cumulative_incidence.to_csv(
                self.experiment_dir
                / "cumulative_incidence.csv",
                index=False,
            )

    # =========================================================
    # Gray's test
    # =========================================================

    def _save_gray_test(self):

        evaluation = (
            self.experiment.evaluation
        )

        if not evaluation:
            return

        gray_test = (
            evaluation.get(
                "gray_test"
            )
        )

        if gray_test is None:
            return

        self._write_json(
            "gray_test.json",
            gray_test,
        )

    # =========================================================
    # Calibration
    # =========================================================

    def _save_calibration(self):

        evaluation = (
            self.experiment.evaluation
        )

        if not evaluation:
            return

        calibration = (
            evaluation.get(
                "calibration"
            )
        )

        if calibration is None:
            return

        if isinstance(
            calibration,
            pd.DataFrame,
        ):

            calibration.to_csv(
                self.experiment_dir
                / "calibration.csv",
                index=False,
            )

    # =========================================================
    # Model summary
    # =========================================================

    def _save_model_summary(self):

        model_summary = (
            self.experiment
            .final_model_summary
        )

        if model_summary is None:
            return

        model_dir = (
            self.experiment_dir
            / "model"
        )

        model_dir.mkdir(
            exist_ok=True
        )

        if isinstance(
            model_summary,
            dict,
        ):

            for cause, summary in (
                model_summary.items()
            ):

                if isinstance(
                    summary,
                    pd.DataFrame,
                ):

                    summary.to_csv(
                        model_dir
                        / f"{cause}.csv",
                        index=False,
                    )

        elif isinstance(
            model_summary,
            pd.DataFrame,
        ):

            model_summary.to_csv(
                model_dir
                / "model_summary.csv",
                index=False,
            )

    # =========================================================
    # Plots
    # =========================================================

    def _save_plots(self):

        evaluation = (
            self.experiment.evaluation
        )

        if not evaluation:
            return

        plot_dir = (
            self.experiment_dir
            / "plots"
        )

        plot_dir.mkdir(
            exist_ok=True
        )

        plot_mapping = {
            "cumulative_incidence_plot":
                "cumulative_incidence.png",

            "calibration_plot":
                "calibration.png",

            "risk_group_plot":
                "risk_groups.png",
        }

        for key, filename in (
            plot_mapping.items()
        ):

            figure = evaluation.get(key)

            if figure is None:
                continue

            figure.savefig(
                plot_dir / filename,
                bbox_inches="tight",
            )

    # =========================================================
    # Helpers
    # =========================================================

    def _enabled(self, key):

        return bool(
            self.save_config.get(
                key,
                False,
            )
        )

    def _write_json(
        self,
        filename,
        data,
    ):

        path = (
            self.experiment_dir
            / filename
        )

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self._make_json_serialisable(
                    data
                ),
                f,
                indent=2,
            )

    def _make_json_serialisable(
        self,
        value,
    ):

        if isinstance(
            value,
            dict,
        ):

            return {
                str(key):
                    self._make_json_serialisable(
                        item
                    )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):

            return [
                self._make_json_serialisable(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            (np.integer,),
        ):

            return int(value)

        if isinstance(
            value,
            (np.floating,),
        ):

            return float(value)

        if isinstance(
            value,
            np.ndarray,
        ):

            return value.tolist()

        if isinstance(
            value,
            pd.Timestamp,
        ):

            return value.isoformat()

        return value
