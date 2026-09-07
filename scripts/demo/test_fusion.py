# scripts/test_fusion.py

from pathlib import Path

import pandas as pd
import yaml

from pd_risk.modelling.dataset import ModelDatasetBuilder
from pd_risk.modelling.preprocessing import (
    ClinicalPreprocessor,
    DATPreprocessor,
    MRIPreprocessor,
    BiospecimenPreprocessor,
    GeneticsPreprocessor,
)
from pd_risk.modelling.fusion import build_fusion


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ======================================================================
# Configuration
# ======================================================================

def load_prediction_config():

    config_file = (
        PROJECT_ROOT
        / "configs"
        / "prediction"
        / "ppmi.yaml"
    )

    with open(
        config_file,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


def load_features_config(
    config,
    modality,
):

    features_file = Path(
        config["modalities"][modality]["features_file"]
    )

    if not features_file.is_absolute():

        features_file = (
            PROJECT_ROOT
            / features_file
        )

    with open(
        features_file,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


# ======================================================================
# Preprocessor factory
# ======================================================================

def build_preprocessor(
    modality,
    builder,
    config,
):

    features_config = load_features_config(
        config,
        modality,
    )

    metadata = (
        builder.modality_features[
            modality
        ]
    )

    types = metadata["types"]

    if modality == "clinical":

        return ClinicalPreprocessor(
            continuous_features=(
                types["continuous"]
            ),
            binary_features=(
                types["binary"]
            ),
            categorical_features=(
                types["categorical"]
            ),
            config=config["preprocessing"],
        )

    if modality == "dat":

        return DATPreprocessor(
            features_config=features_config,
            config=config.get(
                "preprocessing"
            ),
        )

    if modality == "mri":

        return MRIPreprocessor(
            features_config=features_config,
            config=config.get(
                "preprocessing"
            ),
        )

    if modality == "biospecimen":

        return BiospecimenPreprocessor(
            features_config=features_config,
            config=config.get(
                "preprocessing"
            ),
        )

    if modality == "genetics":

        return GeneticsPreprocessor(
            features_config=features_config,
            config=config.get(
                "preprocessing"
            ),
        )

    raise ValueError(
        f"No preprocessor defined for "
        f"modality '{modality}'."
    )


# ======================================================================
# Main
# ======================================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "FUSION INTEGRATION TEST"
    )

    print(
        "=" * 70
    )

    config = load_prediction_config()

    # ------------------------------------------------------------------
    # Build model dataset
    # ------------------------------------------------------------------

    print(
        "\nBuilding model dataset..."
    )

    builder = ModelDatasetBuilder(
        prediction_config=config,
        project_root=PROJECT_ROOT,
    )

    builder.build()

    print(
        "✓ Model dataset built"
    )

    # ------------------------------------------------------------------
    # Enabled modalities
    # ------------------------------------------------------------------

    enabled_modalities = [
        modality
        for modality, modality_config
        in config["modalities"].items()
        if modality_config.get(
            "enabled",
            False,
        )
    ]

    print(
        "\nEnabled modalities:"
    )

    for modality in enabled_modalities:

        print(
            f"  - {modality}"
        )

    # ------------------------------------------------------------------
    # Preprocess modalities
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPROCESSING"
    )

    print(
        "=" * 70
    )

    preprocessed_data = {}

    for modality in enabled_modalities:

        print(
            f"\n{modality.upper()}"
        )

        df = (
            builder.modality_data[
                modality
            ].copy()
        )

        # PATNO belongs to the builder layer.
        # It should not be passed into preprocessing/fusion.

        assert "PATNO" in df.columns, (
            f"{modality} builder output "
            "does not contain PATNO."
        )

        feature_df = df.drop(
            columns=["PATNO"]
        )

        print(
            "  Builder input:",
            df.shape,
        )

        print(
            "  Feature input:",
            feature_df.shape,
        )

        preprocessor = build_preprocessor(
            modality,
            builder,
            config,
        )

        X = (
            preprocessor
            .fit_transform(feature_df)
        )

        # --------------------------------------------------------------
        # Preprocessed output checks
        # --------------------------------------------------------------

        assert isinstance(
            X,
            pd.DataFrame,
        ), (
            f"{modality} preprocessor did not "
            "return a DataFrame."
        )

        assert len(X) == len(
            feature_df
        ), (
            f"{modality} preprocessing changed "
            "the number of rows."
        )

        assert X.index.equals(
            feature_df.index
        ), (
            f"{modality} preprocessing changed "
            "the dataframe index."
        )

        assert not X.columns.duplicated().any(), (
            f"{modality} preprocessing produced "
            "duplicate columns."
        )

        assert not X.isna().any().any(), (
            f"{modality} preprocessing produced "
            "missing values."
        )

        assert all(
            pd.api.types.is_numeric_dtype(
                X[column]
            )
            for column in X.columns
        ), (
            f"{modality} preprocessing produced "
            "non-numeric output."
        )

        preprocessed_data[
            modality
        ] = X

        print(
            "  Preprocessed:",
            X.shape,
        )

        print(
            f"  ✓ {modality} ready for fusion"
        )

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "EARLY FUSION"
    )

    print(
        "=" * 70
    )

    fusion = build_fusion(
        config
    )

    print(
        "\nFusion object:",
        type(fusion).__name__,
    )

    print(
        "Configured strategy:",
        config["fusion"]["strategy"],
    )

    # ------------------------------------------------------------------
    # Fit + transform
    # ------------------------------------------------------------------

    X_fused = fusion.fit_transform(
        preprocessed_data
    )

    print(
        "\nFused output:"
    )

    print(
        "  Type:",
        type(X_fused),
    )

    print(
        "  Shape:",
        X_fused.shape,
    )

    # ------------------------------------------------------------------
    # Output type
    # ------------------------------------------------------------------

    assert isinstance(
        X_fused,
        pd.DataFrame,
    ), (
        "Fusion did not return a pandas DataFrame."
    )

    print(
        "✓ Fused output is a DataFrame"
    )

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    expected_rows = len(
        builder.identifiers
    )

    assert len(X_fused) == expected_rows, (
        "Fused output row count does not "
        "match the modelling cohort."
    )

    print(
        f"✓ Fused participants: "
        f"{len(X_fused):,}"
    )

    # ------------------------------------------------------------------
    # Index alignment
    # ------------------------------------------------------------------

    reference_index = next(
        iter(
            preprocessed_data.values()
        )
    ).index

    assert X_fused.index.equals(
        reference_index
    ), (
        "Fused output index does not match "
        "the modality indices."
    )

    print(
        "✓ Fused index is aligned"
    )

    # ------------------------------------------------------------------
    # Feature count
    # ------------------------------------------------------------------

    expected_features = sum(
        df.shape[1]
        for df in preprocessed_data.values()
    )

    assert X_fused.shape[1] == (
        expected_features
    ), (
        "Fused feature count does not match "
        "the sum of modality feature counts."
    )

    print(
        f"✓ Fused features: "
        f"{X_fused.shape[1]:,}"
    )

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    expected_feature_names = []

    for modality in enabled_modalities:

        expected_feature_names.extend(
            preprocessed_data[
                modality
            ].columns.tolist()
        )

    assert X_fused.columns.tolist() == (
        expected_feature_names
    ), (
        "Fused feature columns do not match "
        "the expected modality order."
    )

    assert fusion.get_feature_names() == (
        expected_feature_names
    ), (
        "Fusion feature names do not match "
        "the fused dataframe."
    )

    print(
        "✓ Fused feature names are correct"
    )

    # ------------------------------------------------------------------
    # Duplicate columns
    # ------------------------------------------------------------------

    assert not X_fused.columns.duplicated().any(), (
        "Fused output contains duplicate feature names."
    )

    print(
        "✓ No duplicate fused feature names"
    )

    # ------------------------------------------------------------------
    # Numeric output
    # ------------------------------------------------------------------

    assert all(
        pd.api.types.is_numeric_dtype(
            X_fused[column]
        )
        for column in X_fused.columns
    ), (
        "Fused output contains non-numeric columns."
    )

    print(
        "✓ Fused output is numeric"
    )

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    assert not X_fused.isna().any().any(), (
        "Fused output contains missing values."
    )

    print(
        "✓ Fused output contains no missing values"
    )

    # ------------------------------------------------------------------
    # Fit/transform consistency
    # ------------------------------------------------------------------

    fusion_b = build_fusion(
        config
    )

    fusion_b.fit(
        preprocessed_data
    )

    X_fused_b = fusion_b.transform(
        preprocessed_data
    )

    pd.testing.assert_frame_equal(
        X_fused,
        X_fused_b,
    )

    print(
        "✓ fit_transform == fit + transform"
    )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata = fusion.get_metadata()

    assert isinstance(
        metadata,
        dict,
    ), (
        "Fusion metadata is not a dictionary."
    )

    assert metadata[
        "strategy"
    ] == "early_fusion"

    assert metadata[
        "modalities"
    ] == enabled_modalities

    assert metadata[
        "n_modalities"
    ] == len(enabled_modalities)

    assert metadata[
        "n_features"
    ] == X_fused.shape[1]

    assert metadata[
        "output_features"
    ] == X_fused.columns.tolist()

    assert metadata[
        "fitted"
    ] is True

    print(
        "✓ Fusion metadata valid"
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL FUSION TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()