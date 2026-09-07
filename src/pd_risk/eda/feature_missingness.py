"""
Feature-level missingness analysis for the PPMI risk cohort.

This module evaluates feature-level data availability within individual
modalities for the baseline study population.

The baseline cohort file defines the reference study population. It is
not used as the source of modality feature values. Modality-specific
feature values are loaded from the files defined in the dataset
configuration.

A modality may consist of one or multiple configured data files. For
example, MRI contains cortical thickness, cortical surface area, and
regional volume datasets. All feature groups belonging to the same
modality are therefore combined for modality-level feature missingness.

Feature definitions are read from the modality feature configuration,
while modality data files are resolved from the dataset configuration.
This avoids hardcoding feature names or file paths in the analysis code.

When different modality datasets contain identically named features,
the feature group name is used as a prefix to preserve a unique feature
namespace. For example:

    cortical_thickness__lh_bankssts
    cortical_surface_area__lh_bankssts

For each configured feature, the analysis reports:

    - number of study participants with a non-missing value
    - number of study participants with a missing value
    - percentage of study participants with a non-missing value

A feature is considered available when its value is non-missing.

The analysis is intended as a feature-level quality-control step following
participant-, visit-, baseline-, and modality-level alignment. It helps
identify heterogeneous feature availability within a modality and assess
whether modality-level availability indicators adequately represent the
underlying features.

The analysis does not perform imputation, feature selection, or modelling.
"""

from pathlib import Path

import pandas as pd
import yaml


# ------------------------------------------------------------------
# Configuration helpers
# ------------------------------------------------------------------

def _load_yaml(path):
    """Load a YAML configuration file."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _get_datasets(config):
    """
    Return configured datasets.

    Supports the project dataset configuration structure where a
    top-level modality may either directly contain a file definition
    or contain multiple sub-datasets.
    """
    return config


def _get_modality_datasets(
    dataset_config,
    modality,
    project_root,
):
    """
    Resolve all data files belonging to a modality.

    Returns
    -------
    list of dict
        Each entry contains:

            name
            file
            key
    """

    config = _load_yaml(dataset_config)

    if modality not in config:
        raise ValueError(
            f"Modality '{modality}' not found in dataset configuration."
        )

    modality_config = config[modality]

    # --------------------------------------------------------------
    # Single-file modality
    # --------------------------------------------------------------

    if isinstance(modality_config, dict) and "file" in modality_config:

        file_path = Path(modality_config["file"])

        if not file_path.is_absolute():
            file_path = project_root / file_path

        return [
            {
                "name": modality,
                "file": file_path,
                "key": modality_config.get("key", []),
            }
        ]

    # --------------------------------------------------------------
    # Multi-file modality
    # --------------------------------------------------------------

    datasets = []

    for dataset_name, dataset_cfg in modality_config.items():

        if not isinstance(dataset_cfg, dict):
            continue

        if "file" not in dataset_cfg:
            continue

        file_path = Path(dataset_cfg["file"])

        if not file_path.is_absolute():
            file_path = project_root / file_path

        datasets.append(
            {
                "name": dataset_name,
                "file": file_path,
                "key": dataset_cfg.get("key", []),
            }
        )

    if not datasets:
        raise ValueError(
            f"No data files found for modality '{modality}'."
        )

    return datasets


# ------------------------------------------------------------------
# Feature configuration
# ------------------------------------------------------------------

def _get_features(feature_config):
    """
    Extract feature definitions grouped by feature dataset.

    Returns
    -------
    dict
        Mapping of feature group name to feature columns.

    Example
    -------
    {
        "cortical_thickness": [...],
        "cortical_surface_area": [...],
        "regional_volume": [...]
    }
    """

    config = _load_yaml(feature_config)

    if "features" not in config:
        raise ValueError(
            f"Feature configuration does not contain a 'features' "
            f"section: {feature_config}"
        )

    feature_groups = {}

    for group_name, group_config in config["features"].items():

        if not isinstance(group_config, dict):
            continue

        columns = []

        for feature_type in group_config.values():

            if not isinstance(feature_type, dict):
                continue

            group_columns = feature_type.get(
                "columns",
                [],
            )

            if group_columns:
                columns.extend(group_columns)

        columns = list(dict.fromkeys(columns))

        if columns:
            feature_groups[group_name] = columns

    if not feature_groups:
        raise ValueError(
            f"No feature columns found in feature configuration: "
            f"{feature_config}"
        )

    return feature_groups


# ------------------------------------------------------------------
# Feature-to-dataset matching
# ------------------------------------------------------------------

def _match_features_to_datasets_older(
    modality_datasets,
    feature_groups,
):
    """
    Match configured feature groups to configured modality datasets.

    Matching is based on the dataset name and feature-group name.

    For example:

        dataset:
            cortical_thickness

        feature group:
            cortical_thickness

    Returns
    -------
    list of dict
        One entry per modality dataset.
    """

    matched = []

    for dataset in modality_datasets:

        dataset_name = dataset["name"]

        if dataset_name not in feature_groups:
            raise ValueError(
                f"Dataset '{dataset_name}' is defined for the modality "
                f"but no matching feature group was found in the feature "
                f"configuration."
            )

        matched.append(
            {
                **dataset,
                "features": feature_groups[dataset_name],
            }
        )

    return matched

def _match_features_to_datasets_old(
    modality_datasets,
    feature_groups,
):
    """
    Match configured feature groups to configured modality datasets.

    Matching is based on the dataset name and feature-group name.

    Datasets that do not yet have a corresponding feature group are
    skipped with a warning. This allows a modality to contain datasets
    that are not yet ready for feature-level missingness analysis.

    For example:

        dataset:
            cortical_thickness

        feature group:
            cortical_thickness

    Returns
    -------
    list of dict
        One entry per modality dataset with a matching feature group.
    """

    matched = []

    for dataset in modality_datasets:

        dataset_name = dataset["name"]

        if dataset_name not in feature_groups:
            print(
                f"  WARNING: Dataset '{dataset_name}' is defined for "
                f"the modality but no matching feature group was found "
                f"in the feature configuration — skipped."
            )
            continue

        matched.append(
            {
                **dataset,
                "features": feature_groups[dataset_name],
            }
        )

    return matched

def _match_features_to_datasets(
    modality_datasets,
    feature_groups,
):
    """
    Match configured feature groups to configured modality datasets.

    Supports two configurations:

    1. Single-file modalities
       ----------------------
       All feature groups belong to the same data file.

       Example:
           dataset:
               dat:
                   file: ...

           features:
               caudate:
               putamen:
               striatum:

       In this case, all feature groups are combined for the dataset.

    2. Multi-file modalities
       ----------------------
       Each dataset corresponds to a feature group.

       Example:
           datasets:
               cortical_thickness:
               cortical_surface_area:
               regional_volume:

           features:
               cortical_thickness:
               cortical_surface_area:
               regional_volume:

       Datasets without a corresponding feature group are skipped
       with a warning.

    Returns
    -------
    list of dict
        Each entry contains the dataset definition and its features.
    """

    # --------------------------------------------------------------
    # Single-file modality
    # --------------------------------------------------------------

    if len(modality_datasets) == 1:

        dataset = modality_datasets[0]

        all_features = []

        for features in feature_groups.values():
            all_features.extend(features)

        # Remove duplicates while preserving order.
        all_features = list(
            dict.fromkeys(all_features)
        )

        if not all_features:
            print(
                f"  WARNING: No feature groups found for "
                f"dataset '{dataset['name']}' — skipped."
            )
            return []

        return [
            {
                **dataset,
                "features": all_features,
            }
        ]

    # --------------------------------------------------------------
    # Multi-file modality
    # --------------------------------------------------------------

    matched = []

    for dataset in modality_datasets:

        dataset_name = dataset["name"]

        if dataset_name not in feature_groups:

            print(
                f"  WARNING: Dataset '{dataset_name}' is defined "
                f"for the modality but no matching feature group "
                f"was found in the feature configuration — skipped."
            )

            continue

        matched.append(
            {
                **dataset,
                "features": feature_groups[dataset_name],
            }
        )

    return matched


# ------------------------------------------------------------------
# Feature-level missingness
# ------------------------------------------------------------------

def inspect_feature_missingness(
    modality,
    feature_config,
    dataset_config,
    baseline_file=(
        "data/processed/cohort/"
        "ppmi_risk_cohort_baseline.csv"
    ),
    project_root=None,
):
    """
    Inspect feature-level missingness for a modality in the baseline
    study population.

    Parameters
    ----------
    modality : str
        Modality name as defined in the dataset configuration.

    feature_config : str or Path
        Path to the modality feature configuration.

    dataset_config : str or Path
        Path to the dataset configuration.

    baseline_file : str or Path
        Baseline cohort file defining the study population.

    project_root : str or Path, optional
        Project root used to resolve relative paths.

    Returns
    -------
    dict
        Dictionary containing:

            modality
            data
            summary
            n_study_participants
            n_modality_participants
            n_features
    """

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    feature_config = Path(feature_config)
    dataset_config = Path(dataset_config)
    baseline_file = Path(baseline_file)

    if not feature_config.is_absolute():
        feature_config = project_root / feature_config

    if not dataset_config.is_absolute():
        dataset_config = project_root / dataset_config

    if not baseline_file.is_absolute():
        baseline_file = project_root / baseline_file

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{modality.upper()} FEATURE-LEVEL MISSINGNESS")
    print("=" * 60)

    print(f"Feature configuration: {feature_config.name}")
    print(f"Dataset configuration: {dataset_config.name}")
    print(f"Baseline cohort:       {baseline_file.name}")

    # ------------------------------------------------------------------
    # Load baseline study population
    # ------------------------------------------------------------------

    baseline = pd.read_csv(
        baseline_file,
        usecols=["PATNO"],
        low_memory=False,
    )

    study_patnos = set(
        baseline["PATNO"]
        .dropna()
        .drop_duplicates()
    )

    n_study = len(study_patnos)

    print(
        f"\nStudy population: {n_study:,}"
    )

    # ------------------------------------------------------------------
    # Resolve modality datasets
    # ------------------------------------------------------------------

    modality_datasets = _get_modality_datasets(
        dataset_config=dataset_config,
        modality=modality,
        project_root=project_root,
    )

    print(
        f"Configured {modality} datasets: "
        f"{len(modality_datasets)}"
    )

    for dataset in modality_datasets:
        print(
            f"  - {dataset['name']}: "
            f"{dataset['file'].name}"
        )

    # ------------------------------------------------------------------
    # Load feature configuration
    # ------------------------------------------------------------------

    feature_groups = _get_features(
        feature_config
    )

    # ------------------------------------------------------------------
    # Match datasets to feature groups
    # ------------------------------------------------------------------

    matched_datasets = _match_features_to_datasets(
        modality_datasets=modality_datasets,
        feature_groups=feature_groups,
    )

    # ------------------------------------------------------------------
    # Load and combine modality datasets
    # ------------------------------------------------------------------

    combined = None
    all_feature_names = []

    modality_participant_sets = []

    for dataset in matched_datasets:

        dataset_name = dataset["name"]
        file_path = dataset["file"]
        features = dataset["features"]

        print("\n" + "-" * 60)
        print(f"Loading {dataset_name}")
        print("-" * 60)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}"
            )

        # --------------------------------------------------------------
        # Check columns
        # --------------------------------------------------------------

        columns = pd.read_csv(
            file_path,
            nrows=0,
        ).columns.tolist()

        if "PATNO" not in columns:
            raise ValueError(
                f"'PATNO' not found in {file_path.name}"
            )

        missing_features = [
            feature
            for feature in features
            if feature not in columns
        ]

        if missing_features:
            raise ValueError(
                f"The following configured features are missing from "
                f"'{file_path.name}':\n"
                + "\n".join(
                    f"  - {feature}"
                    for feature in missing_features
                )
            )

        # --------------------------------------------------------------
        # Load data
        # --------------------------------------------------------------

        data = pd.read_csv(
            file_path,
            usecols=["PATNO"] + features,
            low_memory=False,
        )

        data = data.dropna(
            subset=["PATNO"]
        )

        # Restrict to study population.
        data = data.loc[
            data["PATNO"].isin(study_patnos)
        ].copy()

        participants = set(
            data["PATNO"].unique()
        )

        modality_participant_sets.append(
            participants
        )

        print(
            f"Study participants found: "
            f"{len(participants):,}"
        )

        # --------------------------------------------------------------
        # Prefix features with dataset name
        # --------------------------------------------------------------

        renamed = data.rename(
            columns={
                feature:
                    f"{dataset_name}__{feature}"
                for feature in features
            }
        )

        renamed = renamed[
            [
                "PATNO"
            ]
            + [
                f"{dataset_name}__{feature}"
                for feature in features
            ]
        ]

        prefixed_features = [
            f"{dataset_name}__{feature}"
            for feature in features
        ]

        all_feature_names.extend(
            prefixed_features
        )

        # --------------------------------------------------------------
        # Merge into combined modality table
        # --------------------------------------------------------------

        if combined is None:

            combined = renamed

        else:

            combined = combined.merge(
                renamed,
                on="PATNO",
                how="outer",
            )

    # ------------------------------------------------------------------
    # Combined modality participant count
    # ------------------------------------------------------------------

    if combined is None:
        raise ValueError(
            f"No data could be loaded for modality '{modality}'."
        )

    modality_participants = set(
        combined["PATNO"].unique()
    )

    n_modality = len(
        modality_participants
    )

    print("\n" + "=" * 60)
    print("COMBINED MODALITY")
    print("=" * 60)

    print(
        f"Study participants:       {n_study:,}"
    )

    print(
        f"{modality.upper()} participants: "
        f"{n_modality:,} "
        f"({100 * n_modality / n_study:.1f}%)"
    )

    print(
        f"Total configured features: "
        f"{len(all_feature_names):,}"
    )

    # ------------------------------------------------------------------
    # Feature-level availability
    # ------------------------------------------------------------------

    rows = []

    for feature in all_feature_names:

        available_patnos = set(
            combined.loc[
                combined[feature].notna(),
                "PATNO",
            ].unique()
        )

        n_available = len(
            available_patnos & study_patnos
        )

        n_missing = (
            n_study - n_available
        )

        rows.append(
            {
                "modality": modality,
                "feature": feature,
                "n_study_participants": n_study,
                "n_available": n_available,
                "n_missing": n_missing,
                "percentage_available": (
                    100 * n_available / n_study
                ),
                "percentage_missing": (
                    100 * n_missing / n_study
                ),
            }
        )

    summary = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\nFeature availability:")

    print(
        summary[
            [
                "feature",
                "n_available",
                "n_missing",
                "percentage_available",
            ]
        ].to_string(index=False)
    )

    return {
        "modality": modality,
        "data": combined,
        "summary": summary,
        "n_study_participants": n_study,
        "n_modality_participants": n_modality,
        "n_features": len(all_feature_names),
        "features": all_feature_names,
    }