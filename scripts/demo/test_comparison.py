from pd_risk.modelling.comparison import (
    ExperimentComparator
)


def main():

    comparator = ExperimentComparator(
        "output/experiments"
    )

    experiments = (
        comparator.discover()
    )

    print(
        "\nDiscovered experiments:"
    )

    for experiment in experiments:
        print(
            f" - {experiment.name}"
        )


    comparison = (
        comparator.build_comparison()
    )

    print(
        "\nComparability:"
    )

    print(
        comparator.validate_comparability(
            comparison
        )
    )

    print(
        "\nExperiment comparison:"
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    fold_comparison = (
        comparator.build_fold_comparison()
    )

    print(
        "\nFold-level comparison:"
    )

    print(
        fold_comparison.to_string(
            index=False
        )
    )

    ranking = (
        comparator.build_performance_ranking(
            comparison
        )
    )

    print(
        "\nPerformance ranking:"
    )

    print(
        ranking[
            [
                "experiment",
                "c_index_mean",
                "brier_score_mean",
                "c_index_rank",
                "brier_score_rank",
                "mean_rank",
            ]
        ].to_string(
            index=False
        )
    )

    pareto = (
        comparator.build_pareto_frontier(
            comparison
        )
    )

    print(
        "\nPareto frontier:"
    )

    print(
        pareto[
            [
                "experiment",
                "c_index_mean",
                "brier_score_mean",
                "dominated",
                "pareto_optimal",
            ]
        ].to_string(
            index=False
        )
    )


    import pandas as pd
    synthetic = pd.DataFrame(
        {
            "experiment": [
                "A",
                "B",
                "C",
            ],
            "n_samples": [2247, 2247, 2247],
            "n_features": [19, 19, 19],
            "n_events": [162, 162, 162],
            "validation_strategy": [
                "stratified_kfold",
                "stratified_kfold",
                "stratified_kfold",
            ],
            "n_folds": [5, 5, 5],
            "horizon": [3, 3, 3],
            "c_index_mean": [
                0.85,   # A: best C-index
                0.82,   # B: worse than A
                0.84,   # C: good C-index
            ],
            "brier_score_mean": [
                0.10,   # A: best Brier
                0.14,   # B: worse than A
                0.12,   # C: worse than A
            ],
        }
    )

    synthetic_pareto = (
        comparator.build_pareto_frontier(
            synthetic
        )
    )

    print(
        "\nSynthetic Pareto test:"
    )

    print(
        synthetic_pareto[
            [
                "experiment",
                "c_index_mean",
                "brier_score_mean",
                "dominated",
                "pareto_optimal",
            ]
        ].to_string(index=False)
    )

if __name__ == "__main__":
    main()