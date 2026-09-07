# scripts/test_model_dataset.py

from pathlib import Path

import pandas as pd
import yaml

from pd_risk.modelling.dataset import ModelDatasetBuilder


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def main():

    print("\n" + "=" * 70)
    print("MODEL DATASET BUILDER TEST")
    print("=" * 70)

    config = load_prediction_config()

    # --------------------------------------------------------------
    # Initialise builder
    # --------------------------------------------------------------

    builder = ModelDatasetBuilder(
        prediction_config=config,
        project_root=PROJECT_ROOT,
    )

    print("\nBuilder initialised.")

    # --------------------------------------------------------------
    # Build
    # --------------------------------------------------------------

    builder.build()

    print("\n" + "=" * 70)
    print("BUILD COMPLETED")
    print("=" * 70)

    # --------------------------------------------------------------
    # Basic structure
    # --------------------------------------------------------------

    print("\nAttributes")
    print("-" * 70)

    print(
        "modality_data:",
        type(builder.modality_data)
    )

    print(
        "modality_features:",
        type(builder.modality_features)
    )

    print(
        "targets_df:",
        type(builder.targets_df)
    )

    print(
        "identifiers:",
        type(builder.identifiers)
    )

    print(
        "metadata:",
        type(builder.metadata)
    )

    print("\nBuilder object data types")
    print("-" * 70)

    for name, value in {
        "modality_data": builder.modality_data,
        "modality_features": builder.modality_features,
        "targets_df": builder.targets_df,
        "identifiers": builder.identifiers,
        "metadata": builder.metadata,
    }.items():

        print(
            f"{name}: {type(value).__name__}"
        )

    # --------------------------------------------------------------
    # Modality datasets
    # --------------------------------------------------------------

    print("\nModality data")
    print("-" * 70)

    for modality, df in builder.modality_data.items():

        print(f"\n{modality}")
        print(f"  shape: {df.shape}")
        print(f"  participants: {df['PATNO'].nunique():,}")
        print(f"  columns: {len(df.columns) - 1:,}")
        print(f"  duplicate PATNO: {df['PATNO'].duplicated().sum():,}")

        print("\n  columns:")
        print(
            df.columns.tolist()
        )

        print(
            f"\n{modality}"
        )

        print(
            f"  DataFrame shape: {df.shape}"
        )

        print(
            f"  PATNO included: {'PATNO' in df.columns}"
        )

        assert "PATNO" in df.columns, (
            f"{modality} does not contain PATNO."
        )

        print(
            "  PATNO dtype:",
            df["PATNO"].dtype,
        )

        print(
            "  Feature dtypes:"
        )

        print(
            df.drop(
                columns=["PATNO"]
            ).dtypes
        )

    # --------------------------------------------------------------
    # Feature metadata
    # --------------------------------------------------------------
    print("\nModality feature metadata")
    print("-" * 70)

    for modality, metadata in builder.modality_features.items():

        print(f"\n{modality}")

        print(
            "  domains:",
            metadata["domains"]
        )

        print(
            "  number of features:",
            len(metadata["columns"])
        )

        print(
            "  continuous:",
            len(
                metadata["types"]["continuous"]
            )
        )

        print(
            "  binary:",
            len(
                metadata["types"]["binary"]
            )
        )

        print(
            "  categorical:",
            len(
                metadata["types"]["categorical"]
            )
        )

    # --------------------------------------------------------------
    # Modality-specific modelling datasets
    # --------------------------------------------------------------

    print("\nModality-specific modelling datasets")
    print("-" * 70)

    expected_cohort_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "cohort"
        / "ppmi_risk_cohort_multimodal_cdmgb.csv"
    )

    expected_cohort = pd.read_csv(
        expected_cohort_file,
        usecols=["PATNO"],
    )

    expected_patnos = set(
        expected_cohort["PATNO"]
    )

    print(
        "Expected modelling cohort:",
        len(expected_patnos),
    )

    print(
        "Builder modelling cohort:",
        len(builder.identifiers),
    )


    # --------------------------------------------------------------
    # Cohort checks
    # --------------------------------------------------------------

    print("\nCohort checks")
    print("-" * 70)

    actual_patnos = set(
        builder.identifiers["PATNO"]
    )

    assert actual_patnos == expected_patnos, (
        "Builder modelling cohort does not match "
        "the configured modelling cohort."
    )

    assert not builder.identifiers[
        "PATNO"
    ].duplicated().any(), (
        "Modelling cohort contains duplicate PATNOs."
    )

    print(
        "✓ Modelling cohort matches cohort file"
    )

    print(
        f"✓ Participants: {len(actual_patnos):,}"
    )


    # --------------------------------------------------------------
    # Target checks
    # --------------------------------------------------------------

    print("\nTargets")
    print("-" * 70)

    targets = builder.targets_df

    print(
        "Shape:",
        targets.shape,
    )

    print(
        "Participants:",
        targets["PATNO"].nunique(),
    )

    print(
        "Columns:",
        targets.columns.tolist(),
    )

    print(
        "\nTarget dtypes:"
    )

    print(
        targets.dtypes
    )

    print(
        "PATNO included:",
        "PATNO" in targets.columns,
    )

    assert "PATNO" in targets.columns, (
        "Targets do not contain PATNO."
    )

    print(
        "\nEvent distribution:"
    )

    print(
        targets["event"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nEvent type distribution:"
    )

    print(
        targets["event_type"]
        .value_counts()
        .sort_index()
    )

    print(
        "\nTime-to-event:"
    )

    print(
        targets["time_to_event"]
        .describe()
    )


    assert set(
        targets["PATNO"]
    ) == expected_patnos, (
        "Target participants do not match "
        "the modelling cohort."
    )

    assert not targets[
        "PATNO"
    ].duplicated().any(), (
        "Targets contain duplicate PATNOs."
    )

    assert not targets[
        "event"
    ].isna().any(), (
        "Missing event values detected."
    )

    assert not targets[
        "time_to_event"
    ].isna().any(), (
        "Missing survival times detected."
    )

    assert (
        targets["time_to_event"] >= 0
    ).all(), (
        "Negative survival times detected."
    )

    print(
        "✓ Targets match modelling cohort"
    )

    print(
        "✓ Survival targets valid"
    )


    # --------------------------------------------------------------
    # Enabled modalities
    # --------------------------------------------------------------

    print("\nEnabled modalities")
    print("-" * 70)

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
        enabled_modalities
    )

    assert set(
        builder.modality_data.keys()
    ) == set(
        enabled_modalities
    ), (
        "Builder modalities do not match "
        "enabled modalities in configuration."
    )

    print(
        "✓ All enabled modalities loaded"
    )


    # --------------------------------------------------------------
    # Modality checks
    # --------------------------------------------------------------

    print("\nModality checks")
    print("-" * 70)

    for modality in enabled_modalities:

        print(
            f"\n{modality.upper()}"
        )

        assert modality in (
            builder.modality_data
        ), (
            f"Enabled modality '{modality}' "
            "is missing from modality_data."
        )

        df = (
            builder.modality_data[
                modality
            ]
        )

        metadata = (
            builder.modality_features[
                modality
            ]
        )

        # ----------------------------------------------------------
        # Basic structure
        # ----------------------------------------------------------

        assert "PATNO" in df.columns, (
            f"{modality} does not contain PATNO."
        )

        assert not df[
            "PATNO"
        ].duplicated().any(), (
            f"{modality} contains duplicate PATNOs "
            "after extraction."
        )

        modality_patnos = set(
            df["PATNO"]
        )

        assert modality_patnos == expected_patnos, (
            f"{modality} does not contain exactly "
            "the modelling cohort."
        )

        # ----------------------------------------------------------
        # Feature columns
        # ----------------------------------------------------------

        feature_columns = (
            metadata["columns"]
        )

        actual_feature_columns = [
            column
            for column in df.columns
            if column != "PATNO"
        ]

        configured_but_missing = (
            set(feature_columns)
            - set(actual_feature_columns)
        )

        present_but_not_configured = (
            set(actual_feature_columns)
            - set(feature_columns)
        )

        print(
            f"  Configured features: "
            f"{len(feature_columns):,}"
        )

        print(
            f"  Actual feature columns: "
            f"{len(actual_feature_columns):,}"
        )

        if configured_but_missing:

            print(
                "\n  Configured but missing:"
            )

            print(
                sorted(
                    configured_but_missing
                )[:20]
            )

        if present_but_not_configured:

            print(
                "\n  Present but not configured:"
            )

            print(
                sorted(
                    present_but_not_configured
                )[:20]
            )

        assert not configured_but_missing, (
            f"{modality} has configured features "
            "missing from the dataframe."
        )

        assert not present_but_not_configured, (
            f"{modality} contains columns that are "
            "not present in the feature configuration."
        )

        assert not df[
            actual_feature_columns
        ].columns.duplicated().any(), (
            f"{modality} contains duplicate feature columns."
        )

        # ----------------------------------------------------------
        # Feature type metadata
        # ----------------------------------------------------------

        continuous = (
            metadata["types"]["continuous"]
        )

        binary = (
            metadata["types"]["binary"]
        )

        categorical = (
            metadata["types"]["categorical"]
        )

        typed_features = (
            continuous
            + binary
            + categorical
        )

        assert len(
            typed_features
        ) == len(
            set(typed_features)
        ), (
            f"{modality} has features assigned "
            "to multiple feature types."
        )

        # ----------------------------------------------------------
        # Missingness
        # ----------------------------------------------------------

        missing = (
            df[
                actual_feature_columns
            ]
            .isna()
            .sum()
        )

        # ----------------------------------------------------------
        # Report
        # ----------------------------------------------------------

        print(
            f"  Participants: "
            f"{df['PATNO'].nunique():,}"
        )

        print(
            f"  Features: "
            f"{len(actual_feature_columns):,}"
        )

        print(
            f"  Continuous: "
            f"{len(continuous):,}"
        )

        print(
            f"  Binary: "
            f"{len(binary):,}"
        )

        print(
            f"  Categorical: "
            f"{len(categorical):,}"
        )

        print(
            f"  Features with missing values: "
            f"{(missing > 0).sum():,}"
        )

        print(
            f"  Maximum missing values in one feature: "
            f"{missing.max():,}"
        )

        print(
            f"  ✓ {modality} passed"
        )


    # --------------------------------------------------------------
    # MRI namespace check
    # --------------------------------------------------------------

    if "mri" in builder.modality_data:

        print("\nMRI feature namespace")
        print("-" * 70)

        mri_columns = (
            builder.modality_data[
                "mri"
            ].columns
        )

        cortical_thickness_columns = [
            column
            for column in mri_columns
            if column.startswith(
                "cortical_thickness__"
            )
        ]

        cortical_surface_area_columns = [
            column
            for column in mri_columns
            if column.startswith(
                "cortical_surface_area__"
            )
        ]

        regional_volume_columns = [
            column
            for column in mri_columns
            if column.startswith(
                "regional_volume__"
            )
        ]

        print(
            "Cortical thickness:",
            len(
                cortical_thickness_columns
            ),
        )

        print(
            "Cortical surface area:",
            len(
                cortical_surface_area_columns
            ),
        )

        print(
            "Regional volume:",
            len(
                regional_volume_columns
            ),
        )

        assert cortical_thickness_columns, (
            "MRI cortical thickness features "
            "were not namespaced."
        )

        assert cortical_surface_area_columns, (
            "MRI cortical surface area features "
            "were not namespaced."
        )

        assert regional_volume_columns, (
            "MRI regional volume features "
            "were not namespaced."
        )

        # Check that the three MRI namespaces
        # do not collide.

        namespace_sets = [
            set(cortical_thickness_columns),
            set(cortical_surface_area_columns),
            set(regional_volume_columns),
        ]

        for i in range(
            len(namespace_sets)
        ):

            for j in range(
                i + 1,
                len(namespace_sets),
            ):

                assert not (
                    namespace_sets[i]
                    & namespace_sets[j]
                ), (
                    "MRI feature namespaces overlap."
                )

        print(
            "✓ MRI feature namespaces are distinct"
        )


    # --------------------------------------------------------------
    # Feature missingness by modality
    # --------------------------------------------------------------

    print("\nFeature missingness by modality")
    print("-" * 70)

    for modality, df in (
        builder.modality_data.items()
    ):

        feature_columns = [
            column
            for column in df.columns
            if column != "PATNO"
        ]

        missing = (
            df[
                feature_columns
            ]
            .isna()
            .sum()
            .sort_values(
                ascending=False
            )
        )

        print(
            f"\n{modality}:"
        )

        print(
            missing.head(10)
        )


    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    print("\nMetadata")
    print("-" * 70)

    assert isinstance(
        builder.metadata,
        dict,
    ), (
        "Metadata should be a dictionary."
    )

    assert "cohort" in (
        builder.metadata
    ), (
        "Metadata is missing cohort information."
    )

    assert "modalities" in (
        builder.metadata
    ), (
        "Metadata is missing modality information."
    )

    assert "survival" in (
        builder.metadata
    ), (
        "Metadata is missing survival information."
    )

    print(
        "✓ Metadata generated"
    )


    # --------------------------------------------------------------
    # Architecture checks
    # --------------------------------------------------------------

    print("\nArchitecture checks")
    print("-" * 70)

    assert not hasattr(
        builder,
        "model_dataset",
    ), (
        "Builder should not create a fused "
        "model_dataset before fusion."
    )

    print(
        "✓ No fused model_dataset created"
    )

    print(
        "✓ Modality datasets remain separate"
    )

    print(
        "✓ Fusion has not occurred"
    )


    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL MODEL DATASET TESTS PASSED"
    )

    print(
        "=" * 70
    )

if __name__ == "__main__":
    main()