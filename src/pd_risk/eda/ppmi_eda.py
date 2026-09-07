"""
PPMI exploratory data analysis utilities.

Provides reusable functions for inspecting PPMI modality datasets during
the initial data exploration stage. The utilities load dataset and feature
configurations, inspect dataset structure, identifiers, duplicates, visit
patterns, cohort overlap, and configured feature properties.

The module is intended for exploratory analysis and data-quality checks.
It does not modify or create the processed modelling datasets.

Typical usage:
    inspect_modality(
        modality="dat",
        feature_config="configs/features/ppmi_dat.yaml",
        clinical_file="data/processed/cohort/ppmi_risk_cohort_longitudinal.csv",
    )

For multi-file modalities, specify the dataset/feature-group name:
    inspect_modality(
        modality="mri",
        dataset="cortical_thickness",
        feature_config="configs/features/ppmi_mri.yaml",
        clinical_file="data/processed/cohort/ppmi_risk_cohort_longitudinal.csv",
    )
"""

from pathlib import Path

import pandas as pd
import yaml


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _extract_feature_columns(feature_config, feature_group=None):
    """Extract feature columns from a specific feature group."""

    if feature_group is None:
        sections = feature_config.get("features", {})
    else:
        sections = {
            feature_group: feature_config
            .get("features", {})
            .get(feature_group, {})
        }

        if not sections[feature_group]:
            raise ValueError(
                f"Feature group '{feature_group}' not found in "
                "feature config."
            )

    columns = []

    def walk(obj):
        if isinstance(obj, dict):
            if "columns" in obj:
                columns.extend(obj["columns"])

            for value in obj.values():
                walk(value)

        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(sections)

    return list(dict.fromkeys(columns))


def _print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------
# Generic checks
# ---------------------------------------------------------------------

def _dataset_overview(df):
    print(f"Rows:                {len(df):,}")
    print(f"Columns:             {df.shape[1]:,}")

    if "PATNO" in df.columns:
        print(f"Unique participants: {df['PATNO'].nunique():,}")

    if "EVENT_ID" in df.columns:
        print(f"Unique EVENT_IDs:    {df['EVENT_ID'].nunique():,}")


def _check_identifiers(df):
    identifiers = [
        "PATNO",
        "EVENT_ID",
        "DATSCAN_DATE",
    ]

    print("\nIdentifier check:")

    for col in identifiers:
        if col not in df.columns:
            continue

        n_missing = df[col].isna().sum()

        if n_missing == 0:
            print(f"  ✓ {col}: no missing values")
        else:
            print(f"  ⚠ {col}: {n_missing:,} missing")


def _check_duplicates(df):
    keys = [c for c in ["PATNO", "EVENT_ID"] if c in df.columns]

    if not keys:
        return

    print("\nDuplicate check:")
    print(f"  Checking {' + '.join(keys)} combinations...")

    duplicates = df.duplicated(subset=keys, keep=False)

    if not duplicates.any():
        print("  ✓ No duplicate participant/visit combinations found.")
    else:
        n_combinations = (
            df.loc[duplicates, keys]
            .drop_duplicates()
            .shape[0]
        )

        print(
            f"  ⚠ {n_combinations:,} duplicate "
            f"{' + '.join(keys)} combinations found."
        )


def _check_visits(df):
    if "PATNO" not in df.columns or "EVENT_ID" not in df.columns:
        return

    print("\nVisit structure:")

    patient_events = (
        df[["PATNO", "EVENT_ID"]]
        .drop_duplicates()
        .groupby("PATNO")["EVENT_ID"]
        .nunique()
    )

    if patient_events.max() <= 1:
        print("  ✓ Cross-sectional data.")
    else:
        print("  ✓ Longitudinal data detected.")

    print(
        f"  Participants with >1 event: "
        f"{(patient_events > 1).sum():,}"
    )

    event_summary = (
        df.groupby("EVENT_ID")["PATNO"]
        .nunique()
        .sort_index()
    )

    print("\n  Participants by EVENT_ID:")
    print(event_summary.to_string())


def _describe_features(df, features):
    print("\nFeature description:")

    missing_features = [c for c in features if c not in df.columns]

    if missing_features:
        print("\n  ⚠ Features missing from raw file:")
        for col in missing_features:
            print(f"    - {col}")

    features = [c for c in features if c in df.columns]

    if not features:
        print("  No configured features found.")
        return pd.DataFrame()

    summary = pd.DataFrame(index=features)

    summary["dtype"] = df[features].dtypes.astype(str)
    summary["unique"] = df[features].nunique()
    summary["missing"] = df[features].isna().sum()
    summary["missing_%"] = (
        df[features].isna().mean() * 100
    ).round(2)

    numeric = df[features].select_dtypes(include="number")

    if not numeric.empty:
        summary.loc[numeric.columns, "mean"] = numeric.mean().round(3)
        summary.loc[numeric.columns, "std"] = numeric.std().round(3)
        summary.loc[numeric.columns, "min"] = numeric.min().round(3)
        summary.loc[numeric.columns, "median"] = numeric.median().round(3)
        summary.loc[numeric.columns, "max"] = numeric.max().round(3)

    print(summary.to_string())

    return summary

def _check_cohort_overlap(
    df,
    clinical_file,
    name="Dataset",
):
    """Compare dataset participants with the clinical study cohort."""

    if "PATNO" not in df.columns:
        return

    clinical = pd.read_csv(
        clinical_file,
        usecols=["PATNO"],
    )

    clinical_patients = set(
        clinical["PATNO"].dropna().unique()
    )

    patients = set(
        df["PATNO"].dropna().unique()
    )

    overlap = patients & clinical_patients

    _print_header(f"{name} — Cohort overlap")

    print(f"Dataset participants:          {len(patients):,}")
    print(f"Study cohort (at-risk) participants:     {len(clinical_patients):,}")
    print(f"Participants in both:          {len(overlap):,}")


# ---------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------

def inspect_modality(
    modality,
    feature_config,
    dataset=None,
    clinical_file=None,
    dataset_config="configs/datasets/ppmi.yaml",
    project_root=None,
):
    """
    Perform initial exploratory checks for a PPMI dataset.

    For single-file modalities:
        dataset can be None.

    For multi-file modalities:
        dataset specifies the dataset/feature-group name shared
        between the dataset and feature configurations.
    """

    if project_root is None:
        project_root = Path.cwd().parent
    else:
        project_root = Path(project_root)

    dataset_config_path = project_root / dataset_config
    feature_config_path = project_root / feature_config

    datasets = _load_yaml(dataset_config_path)
    features_config = _load_yaml(feature_config_path)

    # ---------------------------------------------------------------
    # Find modality
    # ---------------------------------------------------------------

    if modality not in datasets:
        raise ValueError(
            f"Modality '{modality}' not found in "
            f"{dataset_config_path}"
        )

    modality_config = datasets[modality]

    # ---------------------------------------------------------------
    # Resolve dataset/file
    # ---------------------------------------------------------------

    if "file" in modality_config:

        # Single-file modality
        if dataset is not None:
            raise ValueError(
                f"'{modality}' is a single-file modality. "
                "Do not specify dataset."
            )

        data_config = modality_config
        feature_group = None

    else:

        # Multi-file modality
        if dataset is None:
            available = [
                key
                for key, value in modality_config.items()
                if isinstance(value, dict) and "file" in value
            ]

            raise ValueError(
                f"'{modality}' contains multiple datasets. "
                f"Specify dataset=.\n"
                f"Available datasets: {available}"
            )

        if dataset not in modality_config:
            available = [
                key
                for key, value in modality_config.items()
                if isinstance(value, dict) and "file" in value
            ]

            raise ValueError(
                f"Dataset '{dataset}' not found under '{modality}'.\n"
                f"Available datasets: {available}"
            )

        data_config = modality_config[dataset]
        feature_group = dataset

    # ---------------------------------------------------------------
    # Locate file
    # ---------------------------------------------------------------

    file_path = project_root / data_config["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found:\n{file_path}"
        )

    print(f"\nLoading: {file_path.name}")

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    df = pd.read_csv(
        file_path,
        low_memory=False,
    )

    # ---------------------------------------------------------------
    # Load relevant features
    # ---------------------------------------------------------------

    features = _extract_feature_columns(
        features_config,
        feature_group=feature_group,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    display_name = modality.upper()

    if dataset is not None:
        display_name += f" / {dataset}"

    _print_header(f"{display_name} — Dataset overview")

    print(f"File: {file_path.name}")
    print()

    _dataset_overview(df)

    _print_header(f"{display_name} — Data checks")

    _check_identifiers(df)
    _check_duplicates(df)
    _check_visits(df)

    _print_header(f"{display_name} — Feature description")

    feature_summary = _describe_features(
        df,
        features,
    )

    if clinical_file is not None:
        _check_cohort_overlap(
            df,
            clinical_file,
            display_name,
        )

    return {
        "data": df,
        "features": features,
        "feature_summary": feature_summary,
        "modality": modality,
        "dataset": dataset,
        "file": file_path,
    }