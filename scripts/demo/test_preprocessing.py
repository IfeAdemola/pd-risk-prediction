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
    """
    Load the feature configuration associated with a modality.
    """

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
    """
    Construct the appropriate preprocessor for a modality.

    The ModelDatasetBuilder remains responsible for loading and
    selecting the modality data. The preprocessor is responsible
    only for modality-specific preprocessing.
    """

    features_config = load_features_config(
        config,
        modality,
    )

    modality_metadata = (
        builder.modality_features[
            modality
        ]
    )

    types = modality_metadata["types"]

    # ------------------------------------------------------------------
    # Clinical
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # DAT
    # ------------------------------------------------------------------

    if modality == "dat":

        return DATPreprocessor(
            features_config=features_config,
            config=config.get(
                "preprocessing"
            ),
        )

    # ------------------------------------------------------------------
    # MRI
    # ------------------------------------------------------------------

    if modality == "mri":

        return MRIPreprocessor(
            features_config=features_config,
            config=config.get(
                "preprocessing"
            ),
        )

    # ------------------------------------------------------------------
    # Biospecimen
    # ------------------------------------------------------------------

    if modality == "biospecimen":

        return BiospecimenPreprocessor(
            features_config=features_config,
            config=config.get(
                "preprocessing"
            ),
        )

    # ------------------------------------------------------------------
    # Genetics
    # ------------------------------------------------------------------

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
# Generic validation
# ======================================================================

def validate_basic_output(
    modality,
    X_input,
    X_processed,
):
    """
    Generic checks applicable to every modality.
    """

    print(
        f"\n  Input shape:  {X_input.shape}"
    )

    print(
        f"  Output shape: {X_processed.shape}"
    )

    # ------------------------------------------------------------------
    # Type
    # ------------------------------------------------------------------

    assert isinstance(
        X_processed,
        pd.DataFrame,
    ), (
        f"{modality} preprocessor did not "
        "return a pandas DataFrame."
    )

    # ------------------------------------------------------------------
    # Row count
    # ------------------------------------------------------------------

    assert len(X_processed) == len(
        X_input
    ), (
        f"{modality} preprocessing changed "
        "the number of rows."
    )

    # ------------------------------------------------------------------
    # Index
    # ------------------------------------------------------------------

    assert X_processed.index.equals(
        X_input.index
    ), (
        f"{modality} preprocessing changed "
        "the dataframe index."
    )

    # ------------------------------------------------------------------
    # Duplicate output columns
    # ------------------------------------------------------------------

    assert not X_processed.columns.duplicated().any(), (
        f"{modality} preprocessing produced "
        "duplicate output feature names."
    )

    # ------------------------------------------------------------------
    # Missingness
    # ------------------------------------------------------------------

    missing = (
        X_processed
        .isna()
        .sum()
        .sum()
    )

    print(
        f"  Missing values after: {missing:,}"
    )

    assert missing == 0, (
        f"{modality} preprocessing produced "
        "missing values."
    )

    # ------------------------------------------------------------------
    # Numeric output
    # ------------------------------------------------------------------

    non_numeric = [
        column
        for column in X_processed.columns
        if not pd.api.types.is_numeric_dtype(
            X_processed[column]
        )
    ]

    assert not non_numeric, (
        f"{modality} preprocessing produced "
        f"non-numeric output columns: "
        f"{non_numeric}"
    )

    print(
        f"  Output features: {X_processed.shape[1]:,}"
    )


# ======================================================================
# Fit/transform consistency
# ======================================================================

def test_fit_transform_consistency(
    modality,
    df,
    builder,
    config,
):
    """
    Verify that:

        fit_transform(X)

    gives the same result as:

        fit(X)
        transform(X)
    """

    preprocessor_a = build_preprocessor(
        modality,
        builder,
        config,
    )

    X_a = (
        preprocessor_a
        .fit_transform(df)
    )

    preprocessor_b = build_preprocessor(
        modality,
        builder,
        config,
    )

    preprocessor_b.fit(df)

    X_b = (
        preprocessor_b
        .transform(df)
    )

    pd.testing.assert_frame_equal(
        X_a,
        X_b,
    )

    print(
        "  ✓ fit_transform == fit + transform"
    )


# ======================================================================
# Clinical checks
# ======================================================================

def validate_clinical(
    df,
    X_processed,
    builder,
):
    """
    Clinical-specific validation.
    """

    metadata = (
        builder.modality_features[
            "clinical"
        ]
    )

    categorical = (
        metadata["types"]["categorical"]
    )

    binary = (
        metadata["types"]["binary"]
    )

    continuous = (
        metadata["types"]["continuous"]
    )

    print(
        f"  Continuous: {len(continuous):,}"
    )

    print(
        f"  Binary:     {len(binary):,}"
    )

    print(
        f"  Categorical:{len(categorical):,}"
    )

    # Clinical preprocessing should not expose
    # the original categorical strings.

    for column in categorical:

        matching = [
            output_column
            for output_column
            in X_processed.columns
            if column in output_column
        ]

        assert matching, (
            f"Clinical categorical feature "
            f"'{column}' has no encoded output."
        )

    print(
        "  ✓ Clinical categorical features encoded"
    )


# ======================================================================
# DAT checks
# ======================================================================

def validate_dat(
    df,
    X_processed,
    builder,
):
    """
    DAT-specific validation.
    """

    metadata = (
        builder.modality_features[
            "dat"
        ]
    )

    continuous = (
        metadata["types"]["continuous"]
    )

    assert len(X_processed.columns) == len(
        continuous
    ), (
        "DAT output feature count does not "
        "match configured continuous features."
    )

    print(
        f"  DAT features: {len(continuous):,}"
    )

    print(
        "  ✓ DAT continuous preprocessing valid"
    )


# ======================================================================
# MRI checks
# ======================================================================

def validate_mri(
    df,
    X_processed,
    builder,
):
    """
    MRI-specific validation.
    """

    columns = X_processed.columns

    # ------------------------------------------------------------------
    # eTIV must not be a model feature
    # ------------------------------------------------------------------

    etiv_columns = [
        column
        for column in columns
        if "EstimatedTotalIntraCranialVol"
        in column
    ]

    assert not etiv_columns, (
        "eTIV is exposed as a model feature "
        "after MRI preprocessing."
    )

    print(
        "  ✓ eTIV is not exposed as a feature"
    )

    # ------------------------------------------------------------------
    # Expected MRI namespaces
    # ------------------------------------------------------------------

    cth_columns = [
        column
        for column in columns
        if column.startswith(
            "cortical_thickness__"
        )
    ]

    sa_columns = [
        column
        for column in columns
        if column.startswith(
            "cortical_surface_area__"
        )
    ]

    volume_columns = [
        column
        for column in columns
        if column.startswith(
            "regional_volume__"
        )
    ]

    print(
        f"  CTH features:    {len(cth_columns):,}"
    )

    print(
        f"  SA features:     {len(sa_columns):,}"
    )

    print(
        f"  Volume features: {len(volume_columns):,}"
    )

    assert cth_columns, (
        "No cortical-thickness features "
        "found after MRI preprocessing."
    )

    assert sa_columns, (
        "No surface-area features "
        "found after MRI preprocessing."
    )

    assert volume_columns, (
        "No regional-volume features "
        "found after MRI preprocessing."
    )

    # ------------------------------------------------------------------
    # Namespace separation
    # ------------------------------------------------------------------

    namespaces = [
        set(cth_columns),
        set(sa_columns),
        set(volume_columns),
    ]

    for i in range(len(namespaces)):

        for j in range(i + 1, len(namespaces)):

            assert not (
                namespaces[i]
                & namespaces[j]
            ), (
                "MRI namespaces overlap."
            )

    print(
        "  ✓ MRI namespaces are distinct"
    )

    # ------------------------------------------------------------------
    # All MRI outputs should be numeric
    # ------------------------------------------------------------------

    assert all(
        pd.api.types.is_numeric_dtype(
            X_processed[column]
        )
        for column in X_processed.columns
    )

    print(
        "  ✓ MRI output is numeric"
    )


# ======================================================================
# Biospecimen checks
# ======================================================================

def validate_biospecimen(
    df,
    X_processed,
    builder,
):
    """
    Biospecimen-specific validation.

    Currently SAA is the only actively preprocessed
    biospecimen feature.

    The source column is expected to be namespaced
    because the biospecimen modality contains multiple
    source files.
    """

    saa_columns = [
        column
        for column in df.columns
        if column.lower()
        .endswith(
            "saa_status"
        )
    ]

    assert saa_columns, (
        "Could not find namespaced SAA_Status "
        "in biospecimen input."
    )

    print(
        "  SAA input column:",
        saa_columns,
    )

    # ------------------------------------------------------------------
    # Output should contain encoded SAA categories
    # ------------------------------------------------------------------

    saa_output_columns = [
        column
        for column in X_processed.columns
        if "saa_status" in column.lower()
    ]

    assert saa_output_columns, (
        "No processed SAA features found."
    )

    print(
        f"  SAA output features: "
        f"{len(saa_output_columns):,}"
    )

    print(
        "  ✓ Biospecimen SAA preprocessing valid"
    )


# ======================================================================
# Genetics checks
# ======================================================================

def validate_genetics(
    df,
    X_processed,
    builder,
):
    """
    Genetics-specific validation.

    Validation is based on the actual namespaced genetics
    columns produced by ModelDatasetBuilder.
    """

    # ------------------------------------------------------------------
    # Pathogenic variants
    # ------------------------------------------------------------------

    pathogenic_genes = [
        column
        for column in df.columns
        if column.startswith(
            "pathogenic_variants__"
        )
        and column != (
            "pathogenic_variants__PATHVAR_COUNT"
        )
        and column != (
            "pathogenic_variants__APOE"
        )
    ]

    assert pathogenic_genes, (
        "No pathogenic-variant columns found."
    )

    for column in pathogenic_genes:

        raw_gene = column.replace(
            "pathogenic_variants__",
            "",
            1,
        )

        output_column = (
            f"{raw_gene}_carrier"
        )

        assert output_column in X_processed.columns, (
            f"Processed genetics output is missing "
            f"carrier column '{output_column}'."
        )

    print(
        "  ✓ All pathogenic-variant columns present"
    )

    # ------------------------------------------------------------------
    # Pathogenic-variant encoding semantics
    # ------------------------------------------------------------------

    for column in pathogenic_genes:

        raw_gene = column.replace(
            "pathogenic_variants__",
            "",
            1,
        )

        output_column = (
            f"{raw_gene}_carrier"
        )

        normalized = (
            df[column]
            .astype("string")
            .str.strip()
        )

        missing_mask = (
            normalized.isna()
            | normalized.eq("")
        )

        zero_mask = (
            ~missing_mask
            & normalized.eq("0")
        )

        carrier_mask = (
            ~missing_mask
            & ~normalized.eq("0")
        )

        if zero_mask.any():

            assert (
                X_processed.loc[
                    zero_mask,
                    output_column,
                ] == 0
            ).all(), (
                f"Explicit 0 values in {column} "
                f"were not encoded as 0."
            )

        if carrier_mask.any():

            assert (
                X_processed.loc[
                    carrier_mask,
                    output_column,
                ] == 1
            ).all(), (
                f"Non-zero variant values in {column} "
                f"were not encoded as 1."
            )

    print(
        "  ✓ Pathogenic-variant carrier encoding semantics"
    )

    # ------------------------------------------------------------------
    # APOE
    # ------------------------------------------------------------------

    apoe_column = (
        "pathogenic_variants__APOE"
    )

    assert apoe_column in df.columns, (
        f"Genetics input is missing "
        f"'{apoe_column}'."
    )

    assert "APOE_e4_carrier" in X_processed.columns, (
        "Processed genetics output is missing "
        "'APOE_e4_carrier'."
    )

    print(
        "  ✓ APOE present"
    )

    # ------------------------------------------------------------------
    # PATHVAR_COUNT
    # ------------------------------------------------------------------

    count_column = (
        "pathogenic_variants__PATHVAR_COUNT"
    )

    assert count_column in df.columns, (
        f"Genetics input is missing "
        f"'{count_column}'."
    )

    assert "PATHVAR_COUNT" in X_processed.columns, (
        "Processed genetics output is missing "
        "'PATHVAR_COUNT'."
    )

    pathvar = (
        df[count_column]
        .dropna()
    )

    assert (
        pathvar >= 0
    ).all(), (
        f"{count_column} contains negative values."
    )

    assert (
        pathvar <= 3
    ).all(), (
        f"{count_column} contains values "
        "outside the expected 0–3 range."
    )

    print(
        "  ✓ PATHVAR_COUNT remains within 0–3"
    )

    # ------------------------------------------------------------------
    # Polygenic risk / PCs
    # ------------------------------------------------------------------

    prs_columns = [
        column
        for column in df.columns
        if column.startswith(
            "polygenic_risk__Genetic_PRS_PRS"
        )
        or column.startswith(
            "polygenic_risk__Genetic_PRS_PC"
        )
    ]

    assert prs_columns, (
        "No namespaced PRS/PC features found."
    )

    print(
        f"  PRS/PC features: {len(prs_columns):,}"
    )

    # ------------------------------------------------------------------
    # InfPop
    # ------------------------------------------------------------------

    infpop_columns = [
        column
        for column in df.columns
        if column.endswith(
            "Genetic_PRS_InfPop"
        )
    ]

    assert infpop_columns, (
        "Genetic_PRS_InfPop not found."
    )

    print(
        "  ✓ InfPop present"
    )

    # ------------------------------------------------------------------
    # Processed output must be numeric
    # ------------------------------------------------------------------

    assert all(
        pd.api.types.is_numeric_dtype(
            X_processed[column]
        )
        for column in X_processed.columns
    ), (
        "Genetics output contains non-numeric columns."
    )

    print(
        "  ✓ Genetics output is numeric"
    )

# ======================================================================
# Main
# ======================================================================

def main():

    print(
        "\n" + "=" * 70
    )

    print(
        "PREPROCESSING INTEGRATION TEST"
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
    # Preprocessing
    # ------------------------------------------------------------------

    for modality in enabled_modalities:

        print(
            "\n" + "=" * 70
        )

        print(
            modality.upper()
        )

        print(
            "=" * 70
        )

        assert modality in (
            builder.modality_data
        ), (
            f"{modality} is enabled but "
            "not present in builder.modality_data."
        )

        df = (
            builder.modality_data[
                modality
            ]
            .copy()
        )


        print(
            f"\nInput shape: {df.shape}"
        )

        # --------------------------------------------------------------
        # PATNO
        # --------------------------------------------------------------

        assert "PATNO" in df.columns, (
            f"{modality} input does not "
            "contain PATNO."
        )

        input_patnos = df[
            "PATNO"
        ].copy()

        # --------------------------------------------------------------
        # Build preprocessor
        # --------------------------------------------------------------

        preprocessor = build_preprocessor(
            modality,
            builder,
            config,
        )

        # --------------------------------------------------------------
        # Fit + transform
        # --------------------------------------------------------------

        X_model = (
            preprocessor
            .fit_transform(df)
        )

        # --------------------------------------------------------------
        # PATNO should not silently disappear
        # --------------------------------------------------------------
        #
        # The current preprocessors operate on
        # features, not identifiers. Therefore
        # PATNO is intentionally not passed through
        # the sklearn transformer.
        #
        # The test verifies that the row alignment
        # is still intact.

        assert X_model.index.equals(
            df.index
        ), (
            f"{modality} output index "
            "does not match input index."
        )

        # --------------------------------------------------------------
        # Generic checks
        # --------------------------------------------------------------

        validate_basic_output(
            modality,
            df.drop(
                columns=["PATNO"]
            ),
            X_model,
        )

        # --------------------------------------------------------------
        # Modality-specific checks
        # --------------------------------------------------------------

        if modality == "clinical":

            validate_clinical(
                df,
                X_model,
                builder,
            )

        elif modality == "dat":

            validate_dat(
                df,
                X_model,
                builder,
            )

        elif modality == "mri":

            validate_mri(
                df,
                X_model,
                builder,
            )

        elif modality == "biospecimen":

            validate_biospecimen(
                df,
                X_model,
                builder,
            )

        elif modality == "genetics":

            validate_genetics(
                df,
                X_model,
                builder,
            )

        # --------------------------------------------------------------
        # Fit/transform consistency
        # --------------------------------------------------------------

        test_fit_transform_consistency(
            modality,
            df,
            builder,
            config,
        )

        # --------------------------------------------------------------
        # Metadata
        # --------------------------------------------------------------

        metadata = (
            preprocessor
            .get_metadata()
        )

        assert isinstance(
            metadata,
            dict,
        ), (
            f"{modality} preprocessor "
            "metadata is not a dictionary."
        )

        print(
            "  ✓ Metadata generated"
        )

        print(
            f"\n✓ {modality.upper()} PASSED"
        )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL PREPROCESSING TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()