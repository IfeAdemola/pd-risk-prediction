from pathlib import Path

import pandas as pd
import yaml

from pd_risk.modelling.dataset import ModelDatasetBuilder
from pd_risk.modelling.splitting import (
    CrossValidationSplitter,
)
from pd_risk.modelling.preprocessing import (
    ClinicalPreprocessor,
    DATPreprocessor,
    MRIPreprocessor,
    BiospecimenPreprocessor,
    GeneticsPreprocessor,
)
from pd_risk.modelling.fusion import (
    build_fusion,
)


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

    types = (
        builder.modality_features[
            modality
        ]["types"]
    )

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
# Main integration test
# ======================================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "MULTIMODAL INTEGRATION TEST"
    )

    print(
        "=" * 70
    )

    config = load_prediction_config()

    # ------------------------------------------------------------------
    # 1. Build dataset
    # ------------------------------------------------------------------

    print(
        "\n[1] Building model dataset..."
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
    # 2. Basic dataset checks
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

    assert set(
        builder.modality_data.keys()
    ) == set(
        enabled_modalities
    )

    assert "PATNO" in builder.identifiers.columns

    assert "PATNO" in builder.targets_df.columns

    assert set(
        builder.identifiers["PATNO"]
    ) == set(
        builder.targets_df["PATNO"]
    )

    print(
        f"  Participants: "
        f"{len(builder.identifiers):,}"
    )

    print(
        f"  Modalities: "
        f"{enabled_modalities}"
    )

    print(
        "✓ Dataset structure valid"
    )

    # ------------------------------------------------------------------
    # 3. Configure cross-validation
    # ------------------------------------------------------------------

    print(
        "\n[2] Creating cross-validation splitter..."
    )

    validation_config = (
        config["analysis"]["validation"]
    )

    splitter = CrossValidationSplitter(
        validation_config
    )

    # Use the first modality only as the
    # row-index carrier for splitting.
    #
    # The actual stratification target comes
    # from the survival targets.

    reference_modality = (
        enabled_modalities[0]
    )

    reference_df = (
        builder.modality_data[
            reference_modality
        ]
    )

    targets = builder.targets_df

    assert reference_df.index.equals(
        targets.index
    ), (
        "Reference modality and targets "
        "do not share the same row index."
    )

    # ------------------------------------------------------------------
    # 4. Run one complete fold
    # ------------------------------------------------------------------

    print(
        "\n[3] Running first complete fold..."
    )

    train_idx, val_idx = next(
        splitter.split(
            reference_df,
            targets["event"],
        )
    )

    print(
        f"  Train participants: "
        f"{len(train_idx):,}"
    )

    print(
        f"  Validation participants: "
        f"{len(val_idx):,}"
    )

    train_events = (
        targets.iloc[train_idx]["event"]
        .sum()
    )

    val_events = (
        targets.iloc[val_idx]["event"]
        .sum()
    )

    print(
        f"  Train events: {train_events:,}"
    )

    print(
        f"  Validation events: {val_events:,}"
    )

    assert len(train_idx) > 0
    assert len(val_idx) > 0

    assert set(train_idx).isdisjoint(
        set(val_idx)
    )

    print(
        "✓ Train/validation split valid"
    )

    # ------------------------------------------------------------------
    # 5. Prepare modality-specific train/validation data
    # ------------------------------------------------------------------

    print(
        "\n[4] Preprocessing modalities..."
    )

    train_processed = {}
    val_processed = {}

    for modality in enabled_modalities:

        print(
            f"\n  {modality.upper()}"
        )

        df = (
            builder.modality_data[
                modality
            ]
        )

        # --------------------------------------------------------------
        # Split using the indices generated by the splitter
        # --------------------------------------------------------------

        train_df = df.iloc[
            train_idx
        ].copy()

        val_df = df.iloc[
            val_idx
        ].copy()

        assert len(train_df) == len(
            train_idx
        )

        assert len(val_df) == len(
            val_idx
        )

        # --------------------------------------------------------------
        # PATNO is retained in the raw modality data
        # but is not passed into preprocessing.
        # --------------------------------------------------------------

        assert "PATNO" in train_df.columns
        assert "PATNO" in val_df.columns

        train_patnos = train_df[
            "PATNO"
        ].copy()

        val_patnos = val_df[
            "PATNO"
        ].copy()

        X_train_input = train_df.drop(
            columns=["PATNO"]
        )

        X_val_input = val_df.drop(
            columns=["PATNO"]
        )

        # --------------------------------------------------------------
        # Fit ONLY on training data
        # --------------------------------------------------------------

        preprocessor = build_preprocessor(
            modality,
            builder,
            config,
        )

        X_train = (
            preprocessor
            .fit_transform(
                X_train_input
            )
        )

        # --------------------------------------------------------------
        # Transform validation using the
        # training-fitted preprocessor.
        # --------------------------------------------------------------

        X_val = (
            preprocessor
            .transform(
                X_val_input
            )
        )

        # --------------------------------------------------------------
        # Basic checks
        # --------------------------------------------------------------

        assert isinstance(
            X_train,
            pd.DataFrame,
        )

        assert isinstance(
            X_val,
            pd.DataFrame,
        )

        assert len(X_train) == len(
            train_df
        )

        assert len(X_val) == len(
            val_df
        )

        assert X_train.index.equals(
            train_df.index
        )

        assert X_val.index.equals(
            val_df.index
        )

        assert not X_train.isna().any().any()
        assert not X_val.isna().any().any()

        assert all(
            pd.api.types.is_numeric_dtype(
                X_train[column]
            )
            for column in X_train.columns
        )

        assert all(
            pd.api.types.is_numeric_dtype(
                X_val[column]
            )
            for column in X_val.columns
        )

        # PATNO remains available from the
        # original split but is not a model feature.

        assert len(train_patnos) == len(X_train)
        assert len(val_patnos) == len(X_val)

        print(
            f"    Train output: {X_train.shape}"
        )

        print(
            f"    Validation output: {X_val.shape}"
        )

        print(
            "    ✓ preprocessing passed"
        )

        train_processed[
            modality
        ] = X_train

        val_processed[
            modality
        ] = X_val

    # ------------------------------------------------------------------
    # 6. Check multimodal alignment after preprocessing
    # ------------------------------------------------------------------

    print(
        "\n[5] Checking multimodal alignment..."
    )

    reference_train_index = (
        train_processed[
            enabled_modalities[0]
        ].index
    )

    reference_val_index = (
        val_processed[
            enabled_modalities[0]
        ].index
    )

    for modality in enabled_modalities:

        assert train_processed[
            modality
        ].index.equals(
            reference_train_index
        ), (
            f"Training indices are not aligned "
            f"for modality '{modality}'."
        )

        assert val_processed[
            modality
        ].index.equals(
            reference_val_index
        ), (
            f"Validation indices are not aligned "
            f"for modality '{modality}'."
        )

    print(
        "✓ All modalities remain aligned"
    )

    # ------------------------------------------------------------------
    # 7. Early fusion — training data
    # ------------------------------------------------------------------

    print(
        "\n[6] Testing early fusion..."
    )

    fusion = build_fusion(
        config
    )

    X_train_fused = (
        fusion.fit_transform(
            train_processed
        )
    )

    print(
        f"  Fused training shape: "
        f"{X_train_fused.shape}"
    )

    assert isinstance(
        X_train_fused,
        pd.DataFrame,
    )

    assert len(
        X_train_fused
    ) == len(train_idx)

    assert not X_train_fused.isna().any().any()

    assert all(
        pd.api.types.is_numeric_dtype(
            X_train_fused[column]
        )
        for column in X_train_fused.columns
    )

    # PATNO must NOT be a model feature.

    assert "PATNO" not in (
        X_train_fused.columns
    )

    print(
        "✓ Training early fusion passed"
    )

    # ------------------------------------------------------------------
    # 8. Early fusion — validation data
    # ------------------------------------------------------------------

    X_val_fused = (
        fusion.transform(
            val_processed
        )
    )

    print(
        f"  Fused validation shape: "
        f"{X_val_fused.shape}"
    )

    assert isinstance(
        X_val_fused,
        pd.DataFrame,
    )

    assert len(
        X_val_fused
    ) == len(val_idx)

    assert not X_val_fused.isna().any().any()

    assert X_val_fused.columns.tolist() == (
        X_train_fused.columns.tolist()
    )

    assert X_val_fused.index.equals(
        reference_val_index
    )

    print(
        "✓ Validation early fusion passed"
    )

    # ------------------------------------------------------------------
    # 9. Feature metadata
    # ------------------------------------------------------------------

    print(
        "\n[7] Checking fusion metadata..."
    )

    feature_names = (
        fusion.get_feature_names()
    )

    metadata = (
        fusion.get_metadata()
    )

    assert len(feature_names) == (
        X_train_fused.shape[1]
    )

    assert metadata["fitted"] is True

    assert metadata["n_modalities"] == (
        len(enabled_modalities)
    )

    assert metadata["n_features"] == (
        X_train_fused.shape[1]
    )

    assert metadata["modalities"] == (
        enabled_modalities
    )

    print(
        f"  Fused features: "
        f"{len(feature_names):,}"
    )

    print(
        f"  Fused modalities: "
        f"{metadata['modalities']}"
    )

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
        "ALL MULTIMODAL INTEGRATION TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()
