from pathlib import Path

import yaml

from pd_risk.modelling.dataset import (
    ModelDatasetBuilder,
)

from pd_risk.missingness.missingness import (
    MissingnessAnalyzer,
)


def main():

    with open(
        "configs/prediction/ppmi.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        prediction_config = (
            yaml.safe_load(f)
        )


    with open(
        "configs/analysis/missingness.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        missingness_config = (
            yaml.safe_load(f)
        )


    builder = ModelDatasetBuilder(
        prediction_file=(
            "data/processed/prediction/"
            "ppmi_prediction_dataset_v1.csv"
        ),
        prediction_config=(
            prediction_config
        ),
    )

    builder.build()


    analyzer = MissingnessAnalyzer(
        X=builder.X,

        identifiers=(
            builder.identifiers
        ),

        y=builder.y,

        modality="clinical",

        config=(
            missingness_config[
                "missingness"
            ]
        ),
    )


    feature_summary = (
        analyzer.feature_summary()
    )

    participant_summary = (
        analyzer.participant_summary()
    )

    pattern_summary = (
        analyzer.missingness_patterns()
    )


    print(
        "\n=== Feature missingness ==="
    )

    print(
        feature_summary.to_string(
            index=False
        )
    )


    print(
        "\n=== Participant missingness ==="
    )

    print(
        participant_summary[
            [
                "n_features",
                "n_missing",
                "n_observed",
                "missing_rate",
            ]
        ].describe()
    )


    print(
        "\n=== Missingness patterns ==="
    )

    print(
        pattern_summary.to_string(
            index=False
        )
    )

    print(
        "\n=== Participant missingness distribution ==="
    )

    print(
        participant_summary[
            "n_missing"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()