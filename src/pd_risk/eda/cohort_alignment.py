"""
PPMI cohort alignment utilities.

Creates a participant-level cohort manifest showing which participants
from the PPMI at-risk study cohort are represented in each configured
PPMI modality dataset.

The manifest records both participant presence and the number of records
available for each dataset. It is an exploratory checkpoint and does not
modify raw PPMI files or create modality-specific subsets.

The current implementation performs participant-level and visit-level alignment only.
missingness analysis is handled separately."""

from pathlib import Path

import pandas as pd
import yaml


def _load_yaml(path):
    """Load a YAML configuration file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _get_datasets(dataset_config):
    """
    Return all configured datasets as (name, config) pairs.

    Supports both:
        modality:
          file: ...

    and:
        modality:
          dataset_a:
            file: ...
          dataset_b:
            file: ...
    """
    datasets = []

    for modality, config in dataset_config.items():

        if modality == "clinical":
            continue

        # Single-file modality
        if isinstance(config, dict) and "file" in config:
            datasets.append(
                (modality, config)
            )
            continue

        # Multi-file modality
        if isinstance(config, dict):
            for dataset_name, dataset_config in config.items():

                if (
                    isinstance(dataset_config, dict)
                    and "file" in dataset_config
                ):
                    datasets.append(
                        (f"{modality}_{dataset_name}", dataset_config)
                    )

    return datasets


def create_cohort_manifest(
    clinical_file,
    dataset_config="configs/datasets/ppmi.yaml",
    output_file="output/manifest/ppmi_risk_cohort_participant_manifest.csv",
    project_root=None,
):
    """
    Create a participant-level PPMI cohort alignment manifest.

    The manifest contains one row per participant in the study cohort.

    For each configured dataset, two columns are created:

    - <dataset> (modality): binary indicator of whether the participant is present
      in that modality dataset (1 = present, 0 = absent).

    - <dataset>_n_records: number of rows/records belonging to that
      participant in the dataset. This represents the number of records,
      not necessarily the number of visits, since the meaning depends on
      the structure of the source dataset. For example, a long-format
      biospecimen dataset may contain multiple records for the same
      participant and visit.

    Parameters
    ----------
    clinical_file : str or Path
        Processed clinical cohort containing the study population.

    dataset_config : str or Path
        YAML file containing the PPMI dataset file definitions.

    output_file : str or Path
        Location where the manifest will be saved.

    project_root : str or Path, optional
        Project root directory.
    """

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    clinical_file = project_root / clinical_file
    dataset_config = project_root / dataset_config
    output_file = project_root / output_file

    # ---------------------------------------------------------------
    # Load clinical cohort
    # ---------------------------------------------------------------

    print("\nLoading study cohort ...")

    clinical = pd.read_csv(
        clinical_file,
        usecols=["PATNO"],
    )

    cohort_participants = set(
        clinical["PATNO"].dropna().unique()
    )

    print(
        f"Study cohort (at-risk) participants: {len(cohort_participants):,}"
    )

    # ---------------------------------------------------------------
    # Load dataset configuration
    # ---------------------------------------------------------------

    config = _load_yaml(dataset_config)
    datasets = _get_datasets(config)

    print(
        f"Configured datasets: {len(datasets):,}"
    )

    # ---------------------------------------------------------------
    # Build manifest
    # ---------------------------------------------------------------

    manifest = pd.DataFrame(
        {"PATNO": sorted(cohort_participants)}
    )

    manifest["clinical"] = (
        manifest["PATNO"]
        .isin(cohort_participants)
        .astype(int)
    )

    for dataset_name, dataset_cfg in datasets:

        file_path = project_root / dataset_cfg["file"]

        print(f"\nChecking {dataset_name}...")
        print(f"  File: {file_path.name}")

        if not file_path.exists():
            print("  WARNING: file not found — skipped")
            continue

        # Only PATNO is needed for this checkpoint
        data = pd.read_csv(
            file_path,
            usecols=["PATNO"],
            low_memory=False,
        )

        data = data.dropna(subset=["PATNO"])

        # Number of records for each participant
        record_counts = (
            data.groupby("PATNO")
            .size()
            .rename(f"{dataset_name}_n_records")
        )

        # Participants represented in the dataset
        participants = set(
            data["PATNO"].unique()
        )

        manifest[dataset_name] = (
            manifest["PATNO"]
            .isin(participants)
            .astype(int)
        )

        manifest = manifest.join(
            record_counts,
            on="PATNO",
        )

        manifest[f"{dataset_name}_n_records"] = (
            manifest[f"{dataset_name}_n_records"]
            .fillna(0)
            .astype(int)
        )

        print(
            f"  Cohort participants present: "
            f"{len(participants & cohort_participants):,}"
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("COHORT MANIFEST CREATED")
    print("=" * 60)
    print(f"Participants: {len(manifest):,}")
    print(f"Columns:      {len(manifest.columns):,}")
    print(f"Saved to:     {output_file}")

    return manifest

def create_visit_manifest(
    clinical_file,
    dataset_config="configs/datasets/ppmi.yaml",
    output_file="output/manifest/ppmi_risk_cohort_visit_manifest.csv",
    project_root=None,
):
    """
    Create a visit/record-level PPMI cohort alignment manifest.

    The manifest is based on the records present in the clinical study
    cohort and contains one row for each unique combination of the
    clinical cohort identifiers PATNO and EVENT_ID.

    For each configured dataset:

    - <dataset> (modality): binary indicator of whether the corresponding participant
      and visit/record key is present in that dataset e.g. dat (1 = present,
      0 = absent).

    The alignment key is defined by the `key` field in the dataset
    configuration. Different datasets may therefore use different keys.
    For example, longitudinal datasets may use [PATNO, EVENT_ID], while
    static participant-level datasets may use [PATNO].

    This manifest is an alignment checkpoint. It does not modify or
    subset the raw PPMI data.
    """

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    clinical_file = Path(clinical_file)
    dataset_config = Path(dataset_config)
    output_file = Path(output_file)

    if not clinical_file.is_absolute():
        clinical_file = project_root / clinical_file

    if not dataset_config.is_absolute():
        dataset_config = project_root / dataset_config

    if not output_file.is_absolute():
        output_file = project_root / output_file

    # ---------------------------------------------------------------
    # Load clinical cohort
    # ---------------------------------------------------------------

    print("\nLoading study cohort...")

    clinical = pd.read_csv(
        clinical_file,
        usecols=["PATNO", "EVENT_ID"],
        low_memory=False,
    )

    clinical = clinical.drop_duplicates(
        subset=["PATNO", "EVENT_ID"]
    )

    manifest = clinical.copy()

    clinical_set = set(
        zip(
            clinical["PATNO"],
            clinical["EVENT_ID"],
        )
    )

    manifest["clinical"] = [
        int((patno, event_id) in clinical_set)
        for patno, event_id in zip(
            manifest["PATNO"],
            manifest["EVENT_ID"],
        )
    ]

    print(
        f"Clinical cohort records: {len(manifest):,}"
    )
    print(
        f"Clinical cohort participants: "
        f"{manifest['PATNO'].nunique():,}"
    )

    # ---------------------------------------------------------------
    # Load dataset configuration
    # ---------------------------------------------------------------

    config = _load_yaml(dataset_config)
    datasets = _get_datasets(config)

    # ---------------------------------------------------------------
    # Check each dataset
    # ---------------------------------------------------------------

    for dataset_name, dataset_cfg in datasets:

        file_path = project_root / dataset_cfg["file"]
        key = dataset_cfg.get("key")

        print(f"\nChecking {dataset_name}...")
        print(f"  File: {file_path.name}")

        if not file_path.exists():
            print("  WARNING: file not found — skipped")
            continue

        if not key:
            print("  WARNING: no alignment key defined — skipped")
            continue

        print(f"  Alignment key: {key}")

        # -----------------------------------------------------------
        # Participant + visit datasets
        # -----------------------------------------------------------

        if key == ["PATNO", "EVENT_ID"]:

            data = pd.read_csv(
                file_path,
                usecols=key,
                low_memory=False,
            )

            data = data.dropna(
                subset=key
            ).drop_duplicates(
                subset=key
            )

            available = pd.MultiIndex.from_frame(
                data[key]
            )

            clinical_index = pd.MultiIndex.from_frame(
                manifest[key]
            )

            manifest[dataset_name] = (
                clinical_index.isin(available)
                .astype(int)
            )

        # -----------------------------------------------------------
        # Participant-level datasets
        # -----------------------------------------------------------

        elif key == ["PATNO"]:

            data = pd.read_csv(
                file_path,
                usecols=key,
                low_memory=False,
            )

            participants = set(
                data["PATNO"].dropna().unique()
            )

            manifest[dataset_name] = (
                manifest["PATNO"]
                .isin(participants)
                .astype(int)
            )

        # -----------------------------------------------------------
        # Other key structures
        # -----------------------------------------------------------

        else:

            data = pd.read_csv(
                file_path,
                usecols=key,
                low_memory=False,
            )

            data = data.dropna(
                subset=key
            ).drop_duplicates(
                subset=key
            )

            # Match columns shared with the clinical manifest.
            common_keys = [
                col for col in key
                if col in manifest.columns
            ]

            if len(common_keys) != len(key):
                print(
                    f"  WARNING: key {key} cannot be fully "
                    f"matched to clinical cohort — skipped"
                )
                continue

            available = pd.MultiIndex.from_frame(
                data[common_keys]
            )

            clinical_index = pd.MultiIndex.from_frame(
                manifest[common_keys]
            )

            manifest[dataset_name] = (
                clinical_index.isin(available)
                .astype(int)
            )

        print(
            f"  Cohort records present: "
            f"{manifest[dataset_name].sum():,}"
        )
        print(
            f"  Cohort records absent: "
            f"{(manifest[dataset_name] == 0).sum():,}"
        )

    # ---------------------------------------------------------------
    # Save manifest
    # ---------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("VISIT-LEVEL COHORT MANIFEST CREATED")
    print("=" * 60)
    print(f"Records:      {len(manifest):,}")
    print(f"Participants: {manifest['PATNO'].nunique():,}")
    print(f"Columns:      {len(manifest.columns):,}")
    print(f"Saved to:     {output_file}")

    return manifest

def create_baseline_manifest(
    clinical_file,
    dataset_config="configs/datasets/ppmi.yaml",
    output_file="output/manifest/ppmi_risk_cohort_baseline_manifest.csv",
    project_root=None,
):
    """
    Create a participant-level baseline alignment manifest.

    The manifest contains one row per participant in the clinical study
    cohort at the BL visit.

    For each configured dataset, availability is determined automatically
    from the structure of the raw data:

    - Participant-level/static datasets:
      If each participant has only one record, the dataset is treated as
      participant-level/static regardless of whether an event/time column
      is present.

    - Longitudinal/cross-sectional datasets with an event identifier:
      If participants have multiple records and the dataset key contains
      EVENT_ID or CLINICAL_EVENT, the corresponding column is used to
      identify BL records. Only BL records are considered valid baseline
      measurements.

      Screening (SC) measurements are not substituted for missing BL
      measurements for non-static datasets.

    - Repeated datasets without an event identifier:
      If participants have multiple records but no configured event/time
      identifier is available, participant-level availability is recorded,
      but baseline-specific alignment cannot be established.

    The function reports the inferred structure of each dataset before
    creating its baseline availability indicator.

    Output columns:
        PATNO
        clinical
        <dataset>

    Availability values:
        1 = participant has valid data according to the baseline alignment
            rules above
        0 = participant does not have valid data according to those rules

    The clinical column is derived from the clinical baseline cohort and
    therefore reflects whether the participant is present at BL in the
    reference clinical dataset.
    """

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    clinical_file = Path(clinical_file)
    dataset_config = Path(dataset_config)
    output_file = Path(output_file)

    if not clinical_file.is_absolute():
        clinical_file = project_root / clinical_file

    if not dataset_config.is_absolute():
        dataset_config = project_root / dataset_config

    if not output_file.is_absolute():
        output_file = project_root / output_file

    # ------------------------------------------------------------------
    # Load clinical baseline cohort
    # ------------------------------------------------------------------

    print("\nLoading clinical baseline cohort...")

    clinical = pd.read_csv(
        clinical_file,
        usecols=["PATNO", "EVENT_ID"],
        low_memory=False,
    )

    clinical_bl = (
        clinical.loc[
            clinical["EVENT_ID"].eq("BL"),
            ["PATNO"]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    manifest = clinical_bl.copy()

    clinical_bl_participants = set(
        clinical_bl["PATNO"]
    )

    # Derive clinical availability from the reference cohort rather
    # than explicitly assigning 1.
    manifest["clinical"] = (
        manifest["PATNO"]
        .isin(clinical_bl_participants)
        .astype(int)
    )

    print(
        f"Baseline participants: "
        f"{len(manifest):,}"
    )

    # ------------------------------------------------------------------
    # Load dataset configuration
    # ------------------------------------------------------------------

    config = _load_yaml(dataset_config)
    datasets = _get_datasets(config)

    # ------------------------------------------------------------------
    # Inspect each dataset
    # ------------------------------------------------------------------

    for dataset_name, dataset_cfg in datasets:

        file_path = project_root / dataset_cfg["file"]
        key = dataset_cfg.get("key", [])

        print(f"\n{'=' * 60}")
        print(f"Checking {dataset_name}")
        print(f"{'=' * 60}")
        print(f"File: {file_path.name}")

        # --------------------------------------------------------------
        # Validate file
        # --------------------------------------------------------------

        if not file_path.exists():
            print("  WARNING: file not found — skipped")
            continue

        # --------------------------------------------------------------
        # Identify participant and event columns
        # --------------------------------------------------------------

        if "PATNO" not in key:
            print(
                "  WARNING: PATNO not present in key — skipped"
            )
            continue

        event_column = None

        if "EVENT_ID" in key:
            event_column = "EVENT_ID"

        elif "CLINICAL_EVENT" in key:
            event_column = "CLINICAL_EVENT"

        # --------------------------------------------------------------
        # Load required columns
        # --------------------------------------------------------------

        usecols = ["PATNO"]

        if event_column is not None:
            usecols.append(event_column)

        data = pd.read_csv(
            file_path,
            usecols=usecols,
            low_memory=False,
        )

        data = data.dropna(
            subset=["PATNO"]
        )

        n_participants = data["PATNO"].nunique()

        print(
            f"Unique participants: "
            f"{n_participants:,}"
        )

        if event_column is not None:
            print(
                f"Event column: "
                f"{event_column}"
            )
        else:
            print(
                "Event column: none"
            )

        # --------------------------------------------------------------
        # Determine records per participant
        # --------------------------------------------------------------

        records_per_participant = (
            data.groupby("PATNO")
            .size()
        )

        is_static = (
            records_per_participant.max() == 1
        )

        # ==============================================================
        # CASE 1: STATIC / PARTICIPANT-LEVEL DATA
        # ==============================================================

        if is_static:

            print(
                "  ✓ Inferred as static "
                "participant-level data."
            )

            if event_column is not None:

                events = (
                    data[event_column]
                    .dropna()
                    .unique()
                )

                print(
                    f"  {event_column}s present: "
                    f"{', '.join(map(str, events)) if len(events) else 'none'}"
                )

            participants = set(
                data["PATNO"].unique()
            )

            manifest[dataset_name] = (
                manifest["PATNO"]
                .isin(participants)
                .astype(int)
            )

        # ==============================================================
        # CASE 2 / 3: REPEATED DATA
        # ==============================================================

        else:

            print(
                "  Multiple records per participant."
            )

            # ----------------------------------------------------------
            # CASE 2: Repeated data with event identifier
            # ----------------------------------------------------------

            if event_column is not None:

                n_events = (
                    data[event_column]
                    .nunique()
                )

                participants_multiple_events = (
                    records_per_participant > 1
                ).sum()

                print(
                    f"  Unique {event_column}s: "
                    f"{n_events:,}"
                )

                print(
                    f"  Participants with >1 record: "
                    f"{participants_multiple_events:,}"
                )

                # ------------------------------------------------------
                # Keep BL only
                # ------------------------------------------------------

                baseline = data.loc[
                    data[event_column].eq("BL"),
                    ["PATNO"]
                ].drop_duplicates()

                baseline_participants = set(
                    baseline["PATNO"]
                )

                manifest[dataset_name] = (
                    manifest["PATNO"]
                    .isin(baseline_participants)
                    .astype(int)
                )

                print(
                    f"  ✓ Using {event_column} == BL only. "
                    "SC is not substituted for missing BL."
                )

            # ----------------------------------------------------------
            # CASE 3: Repeated data without event identifier
            # ----------------------------------------------------------

            else:

                print(
                    "  WARNING: multiple records per "
                    "participant but no event/time "
                    "identifier."
                )

                print(
                    "  Using participant-level "
                    "availability only; "
                    "baseline-specific alignment "
                    "cannot be established."
                )

                participants = set(
                    data["PATNO"].unique()
                )

                manifest[dataset_name] = (
                    manifest["PATNO"]
                    .isin(participants)
                    .astype(int)
                )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------

        present = manifest[dataset_name].sum()
        absent = len(manifest) - present

        print(
            f"  Baseline participants present: "
            f"{present:,}"
        )

        print(
            f"  Baseline participants absent:  "
            f"{absent:,}"
        )

    # ------------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("BASELINE COHORT MANIFEST CREATED")
    print("=" * 60)

    print(
        f"Participants: "
        f"{len(manifest):,}"
    )

    print(
        f"Columns:      "
        f"{len(manifest.columns):,}"
    )

    print(
        f"Saved to:     "
        f"{output_file}"
    )

    return manifest

def create_baseline_manifest_old(
    clinical_file,
    dataset_config="configs/datasets/ppmi.yaml",
    output_file="output/manifest/ppmi_risk_cohort_baseline_manifest.csv",
    project_root=None,
):
    """
    Create a participant-level baseline alignment manifest.

    The manifest contains one row per participant in the clinical study
    cohort at the BL visit.

    For each configured dataset, availability is determined automatically
    from the structure of the raw data:

    - Participant-level/static datasets:
      If each participant has only one record, the available record is
      treated as a static feature source, regardless of its EVENT_ID
      (e.g. SC or BL).

    - Longitudinal/cross-sectional datasets:
      If participants have multiple records or visits, only records with
      EVENT_ID == 'BL' are considered valid baseline measurements.

      Screening (SC) measurements are not substituted for missing BL
      measurements for non-static datasets.

    The function reports the inferred structure of each dataset before
    creating its baseline availability indicator.

    Output columns:
        PATNO
        <dataset>  (1 = valid baseline data available, 0 = unavailable)
    """

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    clinical_file = Path(clinical_file)
    dataset_config = Path(dataset_config)
    output_file = Path(output_file)

    if not clinical_file.is_absolute():
        clinical_file = project_root / clinical_file

    if not dataset_config.is_absolute():
        dataset_config = project_root / dataset_config

    if not output_file.is_absolute():
        output_file = project_root / output_file

    # ------------------------------------------------------------------
    # Load clinical baseline cohort
    # ------------------------------------------------------------------

    print("\nLoading clinical baseline cohort...")

    clinical = pd.read_csv(
        clinical_file,
        usecols=["PATNO", "EVENT_ID"],
        low_memory=False,
    )

    clinical_bl = (
        clinical.loc[clinical["EVENT_ID"].eq("BL"), ["PATNO"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    manifest = clinical_bl.copy()

    clinical_bl_participants = set(clinical_bl["PATNO"])

    manifest["clinical"] = (
        manifest["PATNO"]
        .isin(clinical_bl_participants)
        .astype(int)
    )

    print(f"Baseline participants: {len(manifest):,}")

    # ------------------------------------------------------------------
    # Load dataset configuration
    # ------------------------------------------------------------------

    config = _load_yaml(dataset_config)
    datasets = _get_datasets(config)

    # ------------------------------------------------------------------
    # Inspect each dataset
    # ------------------------------------------------------------------

    for dataset_name, dataset_cfg in datasets:

        file_path = project_root / dataset_cfg["file"]
        key = dataset_cfg.get("key", [])

        print(f"\n{'=' * 60}")
        print(f"Checking {dataset_name}")
        print(f"{'=' * 60}")
        print(f"File: {file_path.name}")

        if not file_path.exists():
            print("  WARNING: file not found — skipped")
            continue

        if "PATNO" not in key:
            print("  WARNING: PATNO not present in key — skipped")
            continue

        # We need EVENT_ID when available to distinguish static from
        # longitudinal/cross-sectional data.
        usecols = ["PATNO"]

        if "EVENT_ID" in key:
            usecols.append("EVENT_ID")

        data = pd.read_csv(
            file_path,
            usecols=usecols,
            low_memory=False,
        )

        data = data.dropna(subset=["PATNO"])

        n_participants = data["PATNO"].nunique()

        print(f"Unique participants: {n_participants:,}")

        # ------------------------------------------------------------------
        # Participant-level/static dataset
        # ------------------------------------------------------------------

        records_per_participant = data.groupby("PATNO").size()

        is_static = records_per_participant.max() == 1

        if is_static:

            print("  ✓ Inferred as static participant-level data.")

            if "EVENT_ID" in data.columns:
                events = data["EVENT_ID"].dropna().unique()

                print(
                    f"  EVENT_IDs present: "
                    f"{', '.join(map(str, events)) if len(events) else 'none'}"
                )

            participants = set(data["PATNO"].unique())

            manifest[dataset_name] = (
                manifest["PATNO"]
                .isin(participants)
                .astype(int)
            )

        # ------------------------------------------------------------------
        # Longitudinal/cross-sectional dataset
        # ------------------------------------------------------------------

        else:

            if "EVENT_ID" not in data.columns:
                print(
                    "  WARNING: multiple records per participant but "
                    "no EVENT_ID — skipped"
                )
                continue

            n_events = data["EVENT_ID"].nunique()
            participants_multiple_events = (
                records_per_participant > 1
            ).sum()

            print(f"Unique EVENT_IDs: {n_events:,}")
            print(
                f"Participants with >1 record: "
                f"{participants_multiple_events:,}"
            )

            # Keep BL only.
            baseline = data.loc[
                data["EVENT_ID"].eq("BL"),
                ["PATNO"]
            ].drop_duplicates()

            baseline_participants = set(
                baseline["PATNO"]
            )

            manifest[dataset_name] = (
                manifest["PATNO"]
                .isin(baseline_participants)
                .astype(int)
            )

            print(
                "  ✓ Using BL records only. "
                "SC is not substituted for missing BL."
            )

        # ------------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------------

        present = manifest[dataset_name].sum()
        absent = len(manifest) - present

        print(f"  Baseline participants present: {present:,}")
        print(f"  Baseline participants absent:  {absent:,}")

    # ------------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest.to_csv(
        output_file,
        index=False,
    )

    print("\n" + "=" * 60)
    print("BASELINE COHORT MANIFEST CREATED")
    print("=" * 60)
    print(f"Participants: {len(manifest):,}")
    print(f"Columns:      {len(manifest.columns):,}")
    print(f"Saved to:     {output_file}")

    return manifest