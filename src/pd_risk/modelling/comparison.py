from pathlib import Path
import json

import pandas as pd


class ExperimentComparator:
    """
    Compare completed survival experiments from
    their saved artifact directories.
    """

    def __init__(self, experiments_dir):
        self.experiments_dir = Path(
            experiments_dir
        )

    def discover(self):
        """
        Discover completed experiment directories.
        """

        if not self.experiments_dir.exists():
            raise FileNotFoundError(
                f"Experiments directory does not exist: "
                f"{self.experiments_dir}"
            )

        experiments = []

        for path in sorted(
            self.experiments_dir.iterdir()
        ):

            if not path.is_dir():
                continue

            summary_file = (
                path / "summary.json"
            )

            if not summary_file.exists():
                continue

            experiments.append(path)

        return experiments

    def load_summary(self, experiment_dir):
        """
        Load summary.json for one experiment.
        """

        summary_file = (
            Path(experiment_dir)
            / "summary.json"
        )

        if not summary_file.exists():
            raise FileNotFoundError(
                f"Missing summary: {summary_file}"
            )

        with open(
            summary_file,
            "r",
            encoding="utf-8",
        ) as f:

            return json.load(f)

    def build_comparison(self):
        """
        Build a comparison table from all
        completed experiments.
        """

        rows = []

        for experiment_dir in self.discover():

            summary = self.load_summary(
                experiment_dir
            )

            dataset = summary.get(
                "dataset",
                {}
            )

            model = summary.get(
                "model",
                {}
            )

            validation = summary.get(
                "validation",
                {}
            )

            performance = summary.get(
                "performance",
                {}
            )

            c_index = performance.get(
                "c_index",
                {}
            )

            brier = performance.get(
                "brier_score",
                {}
            )

            calibration = summary.get(
                "calibration",
                {}
            )

            rows.append(
                {
                    "experiment":
                        experiment_dir.name,

                    "model":
                        model.get("name"),

                    "model_class":
                        model.get("class"),

                    "n_samples":
                        dataset.get("n_samples"),

                    "n_features":
                        dataset.get("n_features"),

                    "n_events":
                        dataset.get("n_events"),

                    "event_rate":
                        dataset.get("event_rate"),

                    "validation_strategy":
                        validation.get("strategy"),

                    "n_folds":
                        validation.get("n_folds"),

                    "c_index_mean":
                        c_index.get("mean"),

                    "c_index_std":
                        c_index.get("std"),

                    "brier_score_mean":
                        brier.get("mean"),

                    "brier_score_std":
                        brier.get("std"),

                    "horizon":
                        brier.get("horizon"),

                    "calibration_enabled":
                        calibration.get(
                            "enabled"
                        ),

                    "calibration_groups":
                        calibration.get(
                            "n_groups"
                        ),
                }
            )

        return pd.DataFrame(rows)

    def load_cv_results(
        self,
        experiment_dir,
    ):
        """
        Load fold-level CV results.
        """

        results_file = (
            Path(experiment_dir)
            / "cv_results.csv"
        )

        if not results_file.exists():
            raise FileNotFoundError(
                f"Missing CV results: "
                f"{results_file}"
            )

        return pd.read_csv(
            results_file
        )

    def build_fold_comparison(self):
        """
        Combine fold-level results from all
        completed experiments.
        """

        rows = []

        for experiment_dir in self.discover():

            try:
                cv_results = (
                    self.load_cv_results(
                        experiment_dir
                    )
                )
            except FileNotFoundError:
                continue

            cv_results = cv_results.copy()

            cv_results.insert(
                0,
                "experiment",
                experiment_dir.name,
            )

            rows.append(
                cv_results
            )

        if not rows:
            return pd.DataFrame()

        return pd.concat(
            rows,
            ignore_index=True,
        )

    def build_performance_ranking(
    self,
    comparison=None,
    ):
        """
        Rank comparable experiments by survival
        discrimination and prediction error.

        Higher C-index is better.
        Lower Brier score is better.
        """

        if comparison is None:
            comparison = self.build_comparison()

        self.validate_comparability(
            comparison
        )

        ranking = comparison.copy()

        ranking["c_index_rank"] = (
            ranking["c_index_mean"]
            .rank(
                ascending=False,
                method="min",
            )
        )

        ranking["brier_score_rank"] = (
            ranking["brier_score_mean"]
            .rank(
                ascending=True,
                method="min",
            )
        )

        ranking["mean_rank"] = (
            ranking[
                [
                    "c_index_rank",
                    "brier_score_rank",
                ]
            ]
            .mean(axis=1)
        )

        ranking = (
            ranking
            .sort_values(
                [
                    "mean_rank",
                    "c_index_rank",
                    "brier_score_rank",
                ]
            )
            .reset_index(drop=True)
        )

        return ranking

    def build_pareto_frontier(
    self,
    comparison=None,
    ):
        """
        Identify non-dominated experiments using
        C-index and Brier score.

        Higher C-index is better.
        Lower Brier score is better.
        """

        if comparison is None:
            comparison = self.build_comparison()

        self.validate_comparability(
            comparison
        )

        comparison = comparison.copy()

        dominated = set()

        for i, candidate in comparison.iterrows():

            for j, other in comparison.iterrows():

                if i == j:
                    continue

                other_is_at_least_as_good = (
                    other["c_index_mean"]
                    >= candidate["c_index_mean"]
                    and
                    other["brier_score_mean"]
                    <= candidate["brier_score_mean"]
                )

                other_is_strictly_better = (
                    other["c_index_mean"]
                    > candidate["c_index_mean"]
                    or
                    other["brier_score_mean"]
                    < candidate["brier_score_mean"]
                )

                if (
                    other_is_at_least_as_good
                    and
                    other_is_strictly_better
                ):
                    dominated.add(i)
                    break

        comparison["dominated"] = (
            comparison.index.isin(
                dominated
            )
        )

        comparison["pareto_optimal"] = (
            ~comparison["dominated"]
        )

        return comparison

    def validate_comparability(self, comparison=None):
        """
        Validate that experiments are comparable.

        Experiments must use the same:
        - dataset size
        - feature count
        - event count
        - validation strategy
        - number of folds
        - evaluation horizon
        """

        if comparison is None:
            comparison = self.build_comparison()

        if comparison.empty:
            raise ValueError(
                "No experiments available for comparison."
            )

        fields = [
            "n_samples",
            "n_features",
            "n_events",
            "validation_strategy",
            "n_folds",
            "horizon",
        ]

        differences = {}

        for field in fields:

            values = (
                comparison[field]
                .dropna()
                .unique()
            )

            if len(values) > 1:
                differences[field] = (
                    values.tolist()
                )

        if differences:

            raise ValueError(
                "Experiments are not comparable. "
                f"Differences: {differences}"
            )

        return True