from pathlib import Path

import pandas as pd
import numpy as np
import yaml

from pd_risk.modelling.dataset import (
    ModelDatasetBuilder,
)

from pd_risk.missingness.missingness import (
    MissingnessAnalyzer,
)


def main():

    # -------------------------
    # Load configuration
    # -------------------------

    with open(
        "configs/prediction/ppmi.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        prediction_config = yaml.safe_load(f)


    with open(
        "configs/analysis/missingness.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        missingness_config = (
            yaml.safe_load(f)
        )


    # -------------------------
    # Build dataset
    # -------------------------

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

    features = list(builder.X.columns)


    # -------------------------
    # Missingness analyzer
    # -------------------------

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

    cooccurrence = analyzer.cooccurrence()

    print(
        "\n=== Missingness co-occurrence ==="
    )

    print(cooccurrence)

    print(
        "\n=== Non-zero off-diagonal "
        "co-occurrences ==="
    )

    non_zero = []

    for i, feature_a in enumerate(
        cooccurrence.index
    ):

        for feature_b in cooccurrence.columns[
            i + 1:
        ]:

            count = cooccurrence.loc[
                feature_a,
                feature_b,
            ]

            if count > 0:

                non_zero.append(
                    {
                        "feature_a":
                            feature_a,

                        "feature_b":
                            feature_b,

                        "n_cooccurring":
                            int(count),
                    }
                )

    non_zero_df = pd.DataFrame(
        non_zero
    )

    if non_zero_df.empty:

        print(
            "No feature pairs have "
            "co-occurring missingness."
        )

    else:

        print(
            non_zero_df.sort_values(
                "n_cooccurring",
                ascending=False,
            )
            .to_string(index=False)
        )


    # -------------------------
    # Sanity checks
    # -------------------------

    assert isinstance(
        cooccurrence,
        pd.DataFrame,
    )

    assert list(
        cooccurrence.index
    ) == features

    assert list(
        cooccurrence.columns
    ) == features

    assert (
        cooccurrence
        .equals(
            cooccurrence.T
        )
    )

    # Diagonal must equal feature
    # missing counts.
    expected_diagonal = (
        builder.X[features]
        .isna()
        .sum()
    )

    actual_diagonal = (
        pd.Series(
            [
                cooccurrence.loc[
                    feature,
                    feature,
                ]
                for feature in features
            ],
            index=features,
        )
    )

    assert (
        actual_diagonal
        .equals(
            expected_diagonal
        )
    )

    print(
        "\n[OK] Co-occurrence matrix"
    )

    print(
        "[OK] Matrix is symmetric"
    )

    print(
        "[OK] Diagonal matches "
        "feature missing counts"
    )

    similarity = (
        analyzer.cooccurrence_similarity()
    )

    print(
        "\n=== Missingness Jaccard similarity ==="
    )

    print(
        similarity
    )


    print(
        "\n=== Strongest missingness similarities ==="
    )

    pairs = []

    for i, feature_a in enumerate(
        similarity.index
    ):

        for j, feature_b in enumerate(
            similarity.columns
        ):

            if j <= i:
                continue

            value = similarity.loc[
                feature_a,
                feature_b,
            ]

            if value > 0:

                pairs.append(
                    {
                        "feature_a": feature_a,
                        "feature_b": feature_b,
                        "jaccard_similarity": value,
                    }
                )

    similarity_pairs = (
        pd.DataFrame(pairs)
        .sort_values(
            "jaccard_similarity",
            ascending=False,
        )
    )

    print(
        similarity_pairs.to_string(
            index=False
        )
    )

    assert (
        list(similarity.index)
        == list(builder.X.columns)
    )

    assert (
        list(similarity.columns)
        == list(builder.X.columns)
    )

    assert (
        similarity.shape
        == cooccurrence.shape
    )

    assert np.allclose(
        similarity.values,
        similarity.values.T,
    )

    assert np.all(
        similarity.values >= 0
    )

    assert np.all(
        similarity.values <= 1
    )

    print(
        "[OK] Jaccard similarity matrix"
    )

    print(
        "[OK] Similarity matrix is symmetric"
    )

    print(
        "[OK] Similarity values are in [0, 1]"
    )

    for feature in builder.X.columns:

        n_missing = (
            builder.X[feature]
            .isna()
            .sum()
        )

        expected = (
            1.0
            if n_missing > 0
            else 0.0
        )

        assert (
            similarity.loc[
                feature,
                feature,
            ]
            == expected
        )

    print(
        "[OK] Similarity diagonal"
    )

    output_directory = (
        missingness_config[
            "missingness"
        ]["output"]["directory"]
    )

    paths = analyzer.save_results(
        output_directory
    )

    print(
        "\n=== Saved missingness results ==="
    )

    for name, path in paths.items():

        print(
            f"[OK] {name}: {path}"
        )

    for path in paths.values():

        assert Path(path).exists()

    print(
        "[OK] All missingness result files exist"
    )


if __name__ == "__main__":
    main()