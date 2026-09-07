# eda/qc.py

from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from pd_risk.eda.ppmi_eda import (
    _load_yaml,
    _extract_feature_columns,
)


# =====================================================================
# Dataset resolution
# =====================================================================

def _resolve_dataset(
    modality,
    dataset,
    datasets,
    feature_config,
    project_root,
):
    """
    Resolve the data file and feature group for a modality.

    Handles both:
        single-file modalities
        multi-file modalities
    """

    if modality not in datasets:
        raise ValueError(
            f"Modality '{modality}' not found in dataset configuration."
        )

    modality_config = datasets[modality]

    # ---------------------------------------------------------------
    # Single-file modality
    # ---------------------------------------------------------------

    if "file" in modality_config:

        if dataset is not None:
            raise ValueError(
                f"'{modality}' is a single-file modality. "
                "Do not specify dataset."
            )

        data_config = modality_config
        feature_group = None

    # ---------------------------------------------------------------
    # Multi-file modality
    # ---------------------------------------------------------------

    else:

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

    file_path = project_root / data_config["file"]

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found:\n{file_path}"
        )

    features = _extract_feature_columns(
        feature_config,
        feature_group=feature_group,
    )

    return file_path, data_config, features


# =====================================================================
# Generic feature checks
# =====================================================================

def _check_numeric_feature(
    series,
    expected_min=None,
    expected_max=None,
    allow_zero=True,
):
    """
    QC checks for continuous/numeric features.

    Returns counts only. Does not modify the data.
    """

    numeric = pd.to_numeric(series, errors="coerce")

    original_nonmissing = series.notna().sum()

    n_non_numeric = (
        original_nonmissing
        - numeric.notna().sum()
    )

    n_non_finite = np.isinf(
        numeric.dropna()
    ).sum()

    valid = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    n_negative = (valid < 0).sum()
    n_zero = (valid == 0).sum()

    n_below_min = 0
    if expected_min is not None:
        n_below_min = (valid < expected_min).sum()

    n_above_max = 0
    if expected_max is not None:
        n_above_max = (valid > expected_max).sum()

    if not allow_zero:
        n_unexpected_zero = n_zero
    else:
        n_unexpected_zero = 0

    return {
        "n": len(series),
        "n_missing": series.isna().sum(),
        "n_non_numeric": n_non_numeric,
        "n_non_finite": int(n_non_finite),
        "n_negative": int(n_negative),
        "n_zero": int(n_zero),
        "n_unexpected_zero": int(n_unexpected_zero),
        "n_below_min": int(n_below_min),
        "n_above_max": int(n_above_max),
        "min": valid.min(),
        "max": valid.max(),
        "mean": valid.mean(),
        "sd": valid.std(),
    }


def _check_categorical_feature(
    series,
    allowed_values=None,
):
    """
    QC checks for categorical features.
    """

    values = series.dropna()

    unique_values = sorted(
        values.astype(str).unique().tolist()
    )

    if allowed_values is None:
        unexpected = []

    else:
        allowed = set(
            str(value)
            for value in allowed_values
        )

        unexpected = [
            value
            for value in unique_values
            if value not in allowed
        ]

    return {
        "n": len(series),
        "n_missing": series.isna().sum(),
        "n_unique": len(unique_values),
        "unique_values": unique_values,
        "unexpected_values": unexpected,
        "n_unexpected": len(
            values[
                ~values.astype(str).isin(
                    allowed if allowed_values is not None
                    else set(unique_values)
                )
            ]
        )
        if allowed_values is not None
        else 0,
    }


# =====================================================================
# Distribution checks
# =====================================================================

def _check_extremes(series):
    """
    Flag statistically extreme observations using the IQR rule.

    This is a FLAG only.

    No observations are removed or modified.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    ).dropna()

    if numeric.empty:
        return {
            "n_extreme_low": 0,
            "n_extreme_high": 0,
        }

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return {
        "n_extreme_low": int(
            (numeric < lower).sum()
        ),
        "n_extreme_high": int(
            (numeric > upper).sum()
        ),
    }


# =====================================================================
# DAT-specific rules
# =====================================================================

def _dat_feature_qc(series):
    """
    DAT SPECT-specific QC.

    DAT SBR values should not be negative.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    return {
        "n_negative_sbr": int(
            (numeric < 0).sum()
        ),
    }


# =====================================================================
# Main QC function
# =====================================================================

def qc_modality(
    modality,
    feature_config,
    dataset=None,
    dataset_config="configs/datasets/ppmi.yaml",
    project_root=None,
    feature_type="continuous",
    expected_min=None,
    expected_max=None,
    allow_zero=True,
    allowed_values=None,
    check_extremes=True,
):
    """
    Perform quality-control checks on selected features
    from a PPMI modality.

    This function is diagnostic only.

    It does NOT:
        - remove observations
        - impute values
        - transform variables
        - normalize variables
        - scale variables

    Parameters
    ----------
    modality : str
        Top-level modality in the dataset configuration.

    feature_config : str or Path
        Feature configuration file.

    dataset : str, optional
        Dataset name for multi-file modalities.

    dataset_config : str or Path
        Dataset configuration file.

    project_root : str or Path, optional
        Project root.

    feature_type : {"continuous", "categorical"}
        Type of features being checked.

    expected_min : float, optional
        Minimum scientifically valid value.

    expected_max : float, optional
        Maximum scientifically valid value.

    allow_zero : bool
        Whether zero is considered valid.

    allowed_values : list, optional
        Valid categorical values.

    check_extremes : bool
        Whether to flag IQR-based statistical extremes.

    Returns
    -------
    dict
        QC results and the loaded data.
    """

    # ---------------------------------------------------------------
    # Project root
    # ---------------------------------------------------------------

    if project_root is None:
        project_root = Path.cwd().parent
    else:
        project_root = Path(project_root)

    dataset_config_path = (
        project_root / dataset_config
    )

    feature_config_path = (
        project_root / feature_config
    )

    # ---------------------------------------------------------------
    # Load configurations
    # ---------------------------------------------------------------

    datasets = _load_yaml(
        dataset_config_path
    )

    features_config = _load_yaml(
        feature_config_path
    )

    # ---------------------------------------------------------------
    # Resolve dataset
    # ---------------------------------------------------------------

    (
        file_path,
        data_config,
        features,
    ) = _resolve_dataset(
        modality=modality,
        dataset=dataset,
        datasets=datasets,
        feature_config=features_config,
        project_root=project_root,
    )

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print(f"\nLoading: {file_path.name}")

    df = pd.read_csv(
        file_path,
        low_memory=False,
    )

    display_name = modality.upper()

    if dataset is not None:
        display_name += f" / {dataset}"

    print(
        f"\n{'=' * 70}"
    )
    print(
        f"{display_name} — Quality Control"
    )
    print(
        f"{'=' * 70}"
    )

    print(
        f"File: {file_path.name}"
    )

    # ---------------------------------------------------------------
    # Check requested features exist
    # ---------------------------------------------------------------

    missing_features = [
        feature
        for feature in features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "The following configured features "
            "are missing from the dataset:\n"
            f"{missing_features}"
        )

    # ---------------------------------------------------------------
    # Feature-level QC
    # ---------------------------------------------------------------

    results = []

    for feature in features:

        series = df[feature]

        result = {
            "feature": feature,
            "type": feature_type,
        }

        if feature_type == "continuous":

            result.update(
                _check_numeric_feature(
                    series,
                    expected_min=expected_min,
                    expected_max=expected_max,
                    allow_zero=allow_zero,
                )
            )

            if check_extremes:
                result.update(
                    _check_extremes(series)
                )

            # -------------------------------------------------------
            # DAT-specific checks
            # -------------------------------------------------------

            if modality.lower() == "dat":

                result.update(
                    _dat_feature_qc(series)
                )

        elif feature_type == "categorical":

            result.update(
                _check_categorical_feature(
                    series,
                    allowed_values=allowed_values,
                )
            )

        else:

            raise ValueError(
                f"Unsupported feature_type: "
                f"{feature_type}"
            )

        # -----------------------------------------------------------
        # MRI-specific checks
        # -----------------------------------------------------------

        if modality.lower() == "mri":

            result.update(
                _mri_feature_qc(
                    series,
                    dataset,
                )
            )

        # -----------------------------------------------------------
        # Overall status
        # -----------------------------------------------------------

        qc_columns = [
            "n_non_numeric",
            "n_non_finite",
            "n_negative",
            "n_unexpected_zero",
            "n_below_min",
            "n_above_max",
            "n_unexpected",
            "n_nonpositive",
        ]

        issues = sum(
            result.get(column, 0)
            for column in qc_columns
        )

        if modality.lower() == "dat":
            issues += result.get(
                "n_negative_sbr",
                0,
            )

        result["status"] = (
            "PASS"
            if issues == 0
            else "FLAG"
        )

        results.append(result)

    feature_qc = pd.DataFrame(results)

    # ---------------------------------------------------------------
    # Print summary
    # ---------------------------------------------------------------

    print(
        f"\nFeatures checked: {len(features)}"
    )

    print(
        f"Features flagged: "
        f"{(feature_qc['status'] == 'FLAG').sum()}"
    )

    print()

    display(feature_qc)

    # ---------------------------------------------------------------
    # Return
    # ---------------------------------------------------------------

    return {
        "data": df,
        "features": features,
        "feature_qc": feature_qc,
        "modality": modality,
        "dataset": dataset,
        "file": file_path,
    }



# =====================================================================
# MRI-specific QC
# =====================================================================

def _mri_feature_qc(series, dataset):
    """
    MRI-specific QC for FreeSurfer-derived features.

    Returns diagnostic flags only.
    No observations are modified or removed.
    """

    numeric = pd.to_numeric(
        series,
        errors="coerce",
    )

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    result = {
        "n_negative": int((numeric < 0).sum()),
        "n_zero": int((numeric == 0).sum()),
    }

    # ---------------------------------------------------------------
    # Cortical thickness
    # ---------------------------------------------------------------

    if dataset == "cortical_thickness":

        # Thickness should be strictly positive.
        result["n_nonpositive"] = int(
            (numeric <= 0).sum()
        )

    # ---------------------------------------------------------------
    # Cortical surface area
    # ---------------------------------------------------------------

    elif dataset == "cortical_surface_area":

        # Existing cortical parcels should have positive area.
        result["n_nonpositive"] = int(
            (numeric <= 0).sum()
        )

    # ---------------------------------------------------------------
    # Regional volume
    # ---------------------------------------------------------------

    elif dataset == "regional_volume":

        # Negative volumes are impossible.
        result["n_negative"] = int(
            (numeric < 0).sum()
        )

        # Zero is potentially legitimate for some aseg structures.
        # Therefore we do NOT flag zero as automatically invalid.

    return result