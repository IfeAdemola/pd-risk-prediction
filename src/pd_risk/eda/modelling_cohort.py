"""
Cohort selection utilities for defining modelling populations.

This module selects participant-level modelling cohorts from the
baseline modality availability manifest.

The baseline manifest contains availability indicators for individual
datasets or feature blocks. These are collapsed into higher-level
modalities before cohort selection.

For example:

    mri_cortical_thickness
    mri_cortical_surface_area
    mri_regional_volume

are treated collectively as the MRI modality.

A participant is considered to have a modality when at least one of
the underlying dataset blocks is available.

Selected cohorts are saved as participant-level files containing only
PATNO. The filename encodes the selected modality combination.

Example:

    ppmi_risk_cohort_multimodal_cdmgb.csv

where:

    c = clinical
    d = dat
    m = mri
    g = genetics
    b = biospecimen
"""

from pathlib import Path

import pandas as pd


# ----------------------------------------------------------------------
# Modality definitions
# ----------------------------------------------------------------------

MODALITY_GROUPS = {
    "clinical": [
        "clinical",
    ],
    "dat": [
        "dat",
    ],
    "mri": [
        "mri_cortical_thickness",
        "mri_cortical_surface_area",
        "mri_regional_volume",
    ],
    "genetics": [
        "genetics_pathogenic_variants",
        "genetics_polygenic_risk",
    ],
    "biospecimen": [
        "biospecimen_biomarkers",
        "biospecimen_saa",
    ],
}


MODALITY_ABBREVIATIONS = {
    "clinical": "c",
    "dat": "d",
    "mri": "m",
    "genetics": "g",
    "biospecimen": "b",
}


# ----------------------------------------------------------------------
# Build modality-level availability
# ----------------------------------------------------------------------

def _build_modality_availability(
    manifest,
    modality_groups=None,
):
    """
    Collapse dataset-level availability into modality-level availability.

    A participant is considered available for a modality when at least
    one of the configured dataset blocks belonging to that modality is
    available.

    Parameters
    ----------
    manifest : pd.DataFrame
        Baseline availability manifest.

    modality_groups : dict, optional
        Mapping from modality names to their underlying manifest
        columns. If omitted, MODALITY_GROUPS is used.

    Returns
    -------
    pd.DataFrame
        Copy of the manifest containing one additional binary column
        per modality.
    """

    if modality_groups is None:
        modality_groups = MODALITY_GROUPS

    result = manifest.copy()

    for modality, columns in modality_groups.items():

        available_columns = [
            column
            for column in columns
            if column in result.columns
        ]

        if not available_columns:
            raise ValueError(
                f"No manifest columns found for modality '{modality}'. "
                f"Expected one or more of: {columns}"
            )

        # Modality availability = any underlying dataset available.
        result[modality] = (
            result[available_columns]
            .eq(1)
            .any(axis=1)
            .astype(int)
        )

    return result


# ----------------------------------------------------------------------
# Cohort selection
# ----------------------------------------------------------------------

def select_modelling_cohort(
    baseline_manifest,
    required_modalities,
    output_dir="data/processed/cohort",
    output_prefix="ppmi_risk_cohort_multimodal",
    modality_groups=None,
    project_root=None,
):
    """
    Select a modelling cohort from the baseline availability manifest.

    Dataset-level availability is first collapsed into modality-level
    availability using an "any available dataset" rule.

    A participant is retained only when every requested modality is
    available.

    Parameters
    ----------
    baseline_manifest : str, Path, or pd.DataFrame
        Baseline participant-level modality availability manifest.

        Expected to contain PATNO and dataset-level availability columns,
        such as:

            clinical
            dat
            mri_cortical_thickness
            mri_cortical_surface_area
            mri_regional_volume
            genetics_pathogenic_variants
            genetics_polygenic_risk
            biospecimen_biomarkers
            biospecimen_saa

    required_modalities : list of str
        Modalities that must be available.

        Example:

            [
                "clinical",
                "dat",
                "mri",
                "genetics",
                "biospecimen",
            ]

    output_dir : str or Path, default="data/processed/cohort"
        Directory where the selected cohort will be saved.

    output_prefix : str, default="ppmi_risk_cohort_multimodal"
        Prefix used for the output filename.

    modality_groups : dict, optional
        Custom mapping of modality names to manifest columns.

    project_root : str or Path, optional
        Project root used to resolve relative paths.

    Returns
    -------
    pd.DataFrame
        Selected participant-level cohort containing PATNO only.
    """

    # ------------------------------------------------------------------
    # Resolve project root
    # ------------------------------------------------------------------

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    # ------------------------------------------------------------------
    # Validate requested modalities
    # ------------------------------------------------------------------

    if not required_modalities:
        raise ValueError(
            "required_modalities must contain at least one modality."
        )

    required_modalities = list(required_modalities)

    if modality_groups is None:
        modality_groups = MODALITY_GROUPS

    unknown_modalities = [
        modality
        for modality in required_modalities
        if modality not in modality_groups
    ]

    if unknown_modalities:
        raise ValueError(
            f"Unknown modality name(s): {unknown_modalities}. "
            f"Available modalities: {list(modality_groups)}"
        )

    if len(required_modalities) != len(set(required_modalities)):
        raise ValueError(
            "required_modalities contains duplicate modalities."
        )

    # ------------------------------------------------------------------
    # Load baseline manifest
    # ------------------------------------------------------------------

    if isinstance(baseline_manifest, pd.DataFrame):

        manifest = baseline_manifest.copy()

    else:

        baseline_manifest = Path(baseline_manifest)

        if not baseline_manifest.is_absolute():
            baseline_manifest = project_root / baseline_manifest

        if not baseline_manifest.exists():
            raise FileNotFoundError(
                f"Baseline manifest not found:\n{baseline_manifest}"
            )

        manifest = pd.read_csv(
            baseline_manifest,
            low_memory=False,
        )

    # ------------------------------------------------------------------
    # Validate PATNO
    # ------------------------------------------------------------------

    if "PATNO" not in manifest.columns:
        raise ValueError(
            "Baseline manifest must contain a 'PATNO' column."
        )

    if manifest["PATNO"].duplicated().any():
        duplicate_count = manifest["PATNO"].duplicated().sum()

        raise ValueError(
            "Baseline manifest must contain one row per participant. "
            f"Found {duplicate_count:,} duplicated PATNO rows."
        )

    # ------------------------------------------------------------------
    # Build modality-level availability
    # ------------------------------------------------------------------

    modality_manifest = _build_modality_availability(
        manifest,
        modality_groups=modality_groups,
    )

    # ------------------------------------------------------------------
    # Select participants with all required modalities
    # ------------------------------------------------------------------

    selected_mask = (
        modality_manifest[required_modalities]
        .eq(1)
        .all(axis=1)
    )

    selected = modality_manifest.loc[
        selected_mask,
        ["PATNO"],
    ].copy()

    selected = selected.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Generate modality suffix
    # ------------------------------------------------------------------

    modality_suffix = "".join(
        MODALITY_ABBREVIATIONS[modality]
        for modality in required_modalities
    )

    filename = (
        f"{output_prefix}_{modality_suffix}.csv"
    )

    output_dir = Path(output_dir)

    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    output_file = output_dir / filename

    # ------------------------------------------------------------------
    # Save selected cohort
    # ------------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    selected.to_csv(
        output_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODELLING COHORT SELECTION")
    print("=" * 60)

    print(
        f"Source participants: {len(manifest):,}"
    )

    print("\nRequired modalities:")

    for modality in required_modalities:
        print(f"  {modality}")

    print(
        f"\nSelected participants: {len(selected):,}"
    )

    print(
        f"\nSaved to:\n  {output_file}"
    )

    print("=" * 60)

    return selected








def compare_modelling_populations(
    manifest_file,
    outcome_file,
    population_definitions,
    outcome_column="pd_event",
    project_root=None,
):
    """
    Compare candidate modelling populations defined by modality/feature
    availability against the study outcome.

    The function is intended for modelling-cohort selection after
    participant-level and modality-level missingness analysis.

    Each population definition specifies the manifest columns that must
    be available (= 1). This allows comparison of increasingly restrictive
    modelling populations, for example:

        - all 752 multimodal participants
        - multimodal + PRS
        - multimodal + pathogenic variants
        - multimodal + PRS + pathogenic variants

    Parameters
    ----------
    manifest_file : str or Path
        Participant-level baseline manifest containing PATNO and
        availability indicators.

    outcome_file : str or Path
        Cross-sectional outcome file containing PATNO and the specified
        outcome column.

    population_definitions : dict
        Mapping from population name to a list of manifest columns that
        must be available.

        Example
        -------
        {
            "Core multimodal": [],
            "Core multimodal + PRS": [
                "genetics_polygenic_risk"
            ],
            "Core multimodal + pathogenic variants": [
                "genetics_pathogenic_variants"
            ],
            "Core multimodal + PRS + pathogenic variants": [
                "genetics_polygenic_risk",
                "genetics_pathogenic_variants",
            ],
        }

        The manifest itself is assumed to already represent the required
        core multimodal population.

    outcome_column : str, default="pd_event"
        Column in the outcome file indicating whether a participant
        experienced the PD event.

    project_root : str or Path, optional
        Project root used to resolve relative paths.

    Returns
    -------
    pd.DataFrame
        One row per candidate modelling population with:

            population
            n_participants
            n_converters
            n_non_converters
            percentage_of_source
            converter_rate
            converters_lost
            participants_lost

    Notes
    -----
    Availability is determined from the manifest using value == 1.

    Participants are counted only when they are present in both the
    manifest and the outcome file and have a non-missing outcome.

    The function does not perform feature-level imputation, feature
    selection, or modelling.
    """

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    manifest_file = Path(manifest_file)
    outcome_file = Path(outcome_file)

    if not manifest_file.is_absolute():
        manifest_file = project_root / manifest_file

    if not outcome_file.is_absolute():
        outcome_file = project_root / outcome_file

    # ------------------------------------------------------------------
    # Load files
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("MODELLING POPULATION COMPARISON")
    print("=" * 60)

    print(f"Manifest: {manifest_file.name}")
    print(f"Outcome:  {outcome_file.name}")

    manifest = pd.read_csv(
        manifest_file,
        low_memory=False,
    )

    outcomes = pd.read_csv(
        outcome_file,
        low_memory=False,
    )

    # ------------------------------------------------------------------
    # Validate required columns
    # ------------------------------------------------------------------

    if "PATNO" not in manifest.columns:
        raise ValueError(
            "Manifest must contain a 'PATNO' column."
        )

    if "PATNO" not in outcomes.columns:
        raise ValueError(
            "Outcome file must contain a 'PATNO' column."
        )

    if outcome_column not in outcomes.columns:
        raise ValueError(
            f"Outcome file does not contain "
            f"'{outcome_column}'."
        )

    # ------------------------------------------------------------------
    # Prepare outcome data
    # ------------------------------------------------------------------

    outcomes = outcomes[
        ["PATNO", outcome_column]
    ].drop_duplicates(
        subset=["PATNO"]
    )

    # Keep only participants with a defined outcome.
    outcomes = outcomes.loc[
        outcomes[outcome_column].notna()
    ].copy()

    # Normalise boolean-like outcome values.
    if outcomes[outcome_column].dtype == object:

        outcomes[outcome_column] = (
            outcomes[outcome_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                }
            )
        )

    outcomes = outcomes.loc[
        outcomes[outcome_column].notna()
    ].copy()

    outcomes[outcome_column] = (
        outcomes[outcome_column]
        .astype(bool)
    )

    # ------------------------------------------------------------------
    # Merge manifest and outcome
    # ------------------------------------------------------------------

    data = manifest.merge(
        outcomes,
        on="PATNO",
        how="inner",
    )

    if data.empty:
        raise ValueError(
            "No participants could be matched between the "
            "manifest and outcome file."
        )

    # ------------------------------------------------------------------
    # Source population
    # ------------------------------------------------------------------

    n_source = len(data)

    n_source_converters = int(
        data[outcome_column].sum()
    )

    print(
        f"\nSource participants with defined outcome: "
        f"{n_source:,}"
    )

    print(
        f"Source converters: "
        f"{n_source_converters:,}"
    )

    # ------------------------------------------------------------------
    # Validate population definitions
    # ------------------------------------------------------------------

    for population_name, required_columns in (
        population_definitions.items()
    ):

        for column in required_columns:

            if column not in manifest.columns:
                raise ValueError(
                    f"Population '{population_name}' requires "
                    f"manifest column '{column}', but it was not found."
                )

    # ------------------------------------------------------------------
    # Evaluate populations
    # ------------------------------------------------------------------

    rows = []

    for population_name, required_columns in (
        population_definitions.items()
    ):

        if required_columns:

            availability = data[
                required_columns
            ].eq(1).all(axis=1)

        else:

            # No additional requirements:
            # use the entire source manifest population.
            availability = pd.Series(
                True,
                index=data.index,
            )

        population = data.loc[
            availability
        ].copy()

        n_participants = len(population)

        n_converters = int(
            population[outcome_column].sum()
        )

        n_non_converters = (
            n_participants - n_converters
        )

        percentage_of_source = (
            100 * n_participants / n_source
        )

        converter_rate = (
            100 * n_converters / n_participants
            if n_participants > 0
            else 0.0
        )

        participants_lost = (
            n_source - n_participants
        )

        converters_lost = (
            n_source_converters - n_converters
        )

        rows.append(
            {
                "population": population_name,
                "n_participants": n_participants,
                "n_converters": n_converters,
                "n_non_converters": n_non_converters,
                "percentage_of_source": percentage_of_source,
                "converter_rate": converter_rate,
                "participants_lost": participants_lost,
                "converters_lost": converters_lost,
            }
        )

    summary = pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    print("\nCandidate modelling populations:")

    display_columns = [
        "population",
        "n_participants",
        "n_converters",
        "n_non_converters",
        "percentage_of_source",
        "converter_rate",
        "participants_lost",
        "converters_lost",
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False,
            formatters={
                "percentage_of_source": "{:.1f}".format,
                "converter_rate": "{:.1f}".format,
            },
        )
    )

    return summary