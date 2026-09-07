from pathlib import Path

import pandas as pd


class ExperimentReporter:
    """
    Generate human-readable reports from experiment
    comparison results.
    """

    def __init__(
        self,
        comparator,
    ):
        self.comparator = comparator

    def build_report(
        self,
        comparison=None,
    ):
        """
        Build a structured experiment comparison report.
        """

        if comparison is None:
            comparison = (
                self.comparator
                .build_comparison()
            )

        self.comparator.validate_comparability(
            comparison
        )

        ranking = (
            self.comparator
            .build_performance_ranking(
                comparison
            )
        )

        pareto = (
            self.comparator
            .build_pareto_frontier(
                comparison
            )
        )

        report = {

            "experiments": {
                "n_experiments":
                    len(comparison),
            },

            "configuration": {
                "model": comparison[
                    "model"
                ].iloc[0],

                "model_class": comparison[
                    "model_class"
                ].iloc[0],

                "n_samples": int(
                    comparison[
                        "n_samples"
                    ].iloc[0]
                ),

                "n_features": int(
                    comparison[
                        "n_features"
                    ].iloc[0]
                ),

                "n_events": int(
                    comparison[
                        "n_events"
                    ].iloc[0]
                ),

                "validation_strategy":
                    comparison[
                        "validation_strategy"
                    ].iloc[0],

                "n_folds": int(
                    comparison[
                        "n_folds"
                    ].iloc[0]
                ),

                "horizon": int(
                    comparison[
                        "horizon"
                    ].iloc[0]
                ),
            },

            "performance": {
                "ranking":
                    ranking[
                        [
                            "experiment",
                            "c_index_mean",
                            "c_index_std",
                            "brier_score_mean",
                            "brier_score_std",
                            "c_index_rank",
                            "brier_score_rank",
                            "mean_rank",
                        ]
                    ].to_dict(
                        orient="records"
                    ),
            },

            "pareto": {
                "experiments":
                    pareto.loc[
                        pareto["pareto_optimal"],
                        "experiment",
                    ].tolist(),
            },
        }

        return report

    def to_markdown(
        self,
        report,
    ):
        """
        Convert a structured experiment report
        into a human-readable Markdown report.
        """

        lines = []

        lines.append(
            "# Experiment Comparison Report"
        )

        lines.append("")

        lines.append(
            "## Overview"
        )

        lines.append("")

        lines.append(
            "## Configuration"
        )

        lines.append("")

        lines.append(
            f"Model: {report['configuration']['model']}"
        )

        lines.append("")

        lines.append(
            f"Model Class: {report['configuration']['model_class']}"
        )

        lines.append("")

        lines.append(
            f"Number of Samples: {report['configuration']['n_samples']}"
        )

        lines.append("")

        lines.append(
            f"Number of Features: {report['configuration']['n_features']}"
        )

        lines.append("")

        lines.append(
            f"Number of Events: {report['configuration']['n_events']}"
        )

        lines.append("")

        lines.append(
            f"Validation Strategy: {report['configuration']['validation_strategy']}"
        )

        lines.append("")

        lines.append(
            f"Number of Folds: {report['configuration']['n_folds']}"
        )

        lines.append("")

        lines.append(
            f"Number of experiments: "
            f"{report['experiments']['n_experiments']}"
        )

        lines.append("")

        lines.append(
            "## Performance Ranking"
        )

        lines.append("")

        ranking = pd.DataFrame(
            report["performance"]["ranking"]
        )

        if ranking.empty:

            lines.append(
                "No experiments available."
            )

        else:

            display_columns = [
                "experiment",
                "c_index_mean",
                "c_index_std",
                "brier_score_mean",
                "brier_score_std",
                "c_index_rank",
                "brier_score_rank",
                "mean_rank",
            ]

            table = ranking[
                display_columns
            ].copy()

            table = table.rename(
                columns={
                    "experiment":
                        "Experiment",
                    "c_index_mean":
                        "C-index Mean",
                    "c_index_std":
                        "C-index SD",
                    "brier_score_mean":
                        "Brier Mean",
                    "brier_score_std":
                        "Brier SD",
                    "c_index_rank":
                        "C-index Rank",
                    "brier_score_rank":
                        "Brier Rank",
                    "mean_rank":
                        "Mean Rank",
                }
            )


            headers = list(
                table.columns
            )

            lines.append(
                "| "
                + " | ".join(headers)
                + " |"
            )

            lines.append(
                "| "
                + " | ".join(
                    ["---"] * len(headers)
                )
                + " |"
            )

            for _, row in table.iterrows():

                values = []

                for value in row:

                    if isinstance(
                        value,
                        float,
                    ):
                        values.append(
                            f"{value:.4f}"
                        )

                    else:
                        values.append(
                            str(value)
                        )

                lines.append(
                    "| "
                    + " | ".join(values)
                    + " |"
                )

        lines.append("")

        lines.append(
            "## Pareto-Optimal Experiments"
        )

        lines.append("")

        pareto_experiments = (
            report["pareto"]["experiments"]
        )

        if not pareto_experiments:

            lines.append(
                "No Pareto-optimal experiments."
            )

        else:

            for experiment in (
                pareto_experiments
            ):

                lines.append(
                    f"- {experiment}"
                )

        lines.append("")

        return "\n".join(lines)


    def save_markdown(
    self,
    report,
    output_path,
    ):
        """
        Save an experiment comparison report
        as a Markdown file.
        """

        output_path = Path(
            output_path
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        markdown = self.to_markdown(
            report
        )

        output_path.write_text(
            markdown,
            encoding="utf-8",
        )

        return output_path