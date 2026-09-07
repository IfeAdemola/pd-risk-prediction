from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


class PPMICohortBuilder:
    """
    Builds the PPMI at-risk cohort.

    Responsibilities
    ----------------
    - Load curated clinical data.
    - Apply cohort inclusion criteria.
    - Record participant exclusions.
    - Record cohort statistics.
    - Save cohort artefacts.
    """

    def __init__(self, dataset_config: dict, cohort_config: dict):

        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------

        self.dataset_config = dataset_config
        self.cohort_config = cohort_config

        # ------------------------------------------------------------------
        # Data
        # ------------------------------------------------------------------

        self.clinical_df = None
        self.baseline_cohort_df = None
        self.longitudinal_cohort_df = None

        self.exclusions_df = pd.DataFrame(
            columns=[
                "PATNO",
                "stage",
                "reason",
            ]
        )

        # ------------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------------

        self.metadata = {}

        self.history = []

    # ======================================================================
    # Public API
    # ======================================================================

    def build(self):
        """Create the cohort."""

        self._load_clinical_data()
        self._select_baseline()
        self._filter_at_risk()
        self._filter_minimum_visits()
        self._create_longitudinal_cohort()
        self._build_metadata()

        assert self.baseline_cohort_df["PATNO"].nunique() == \
               self.longitudinal_cohort_df["PATNO"].nunique()

        assert set(
            self.baseline_cohort_df["PRIMDIAG"].dropna().unique()
        ).issubset(
            set(self.cohort_config["at_risk_diagnoses"])
        )

    def save(self):
        output_config = self.cohort_config["output"]

        output_dir = Path(
            output_config["directory"]
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Baseline cohort
        self.baseline_cohort_df.to_csv(
            output_dir / output_config["baseline_file"],
            index=False,
        )

        # Longitudinal cohort
        self.longitudinal_cohort_df.to_csv(
            output_dir / output_config["longitudinal_file"],
            index=False,
        )

        # Exclusions
        self.exclusions_df.to_csv(
            output_dir / output_config["exclusions_file"],
            index=False,
        )

        # Metadata
        with open(
            output_dir / output_config["metadata_file"],
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                self.metadata,
                f,
                indent=4,
            )

    def summary(self):
        """Print cohort summary."""

        print("=" * 50)
        print("PPMI Risk Cohort Summary")
        print("=" * 50)

        print()

        print(f"Dataset: {self.metadata['dataset']}")
        print(
            f"Data version: {self.metadata['data_version']}"
        )

        print()

        print("Participants")
        print("-" * 20)

        print(
            f"Initial: {self.metadata['participants']['initial']}"
        )

        print(
            f"Final:   {self.metadata['participants']['final']}"
        )

        print()

        print("Rows")
        print("-" * 20)

        print(
            f"Baseline:      {self.metadata['rows']['baseline']}"
        )

        print(
            f"Longitudinal:  {self.metadata['rows']['longitudinal']}"
        )

        print()

        print("Exclusions")
        print("-" * 20)

        if self.exclusions_df.empty:

            print("No exclusions")

        else:

            print(
                self.exclusions_df["reason"]
                .value_counts()
                .rename_axis(None)
                .to_string()
            )

        print()

        print("Cohort history")
        print("-" * 20)

        for step in self.history:

            print(
                f"{step['step']:<25}"
                f"{step['before']:>6}"
                f" -> "
                f"{step['after']:<6}"
            )

        print("=" * 50)

    # ======================================================================
    # Private methods
    # ======================================================================

    def _load_clinical_data(self):
        clinical = self.dataset_config["clinical"]

        self.clinical_df = pd.read_excel(
            clinical["file"],
            sheet_name=clinical["sheet"],
        )

        self.metadata["data_version"] = clinical["data_version"]
        self.metadata["n_rows"] = len(self.clinical_df)
        self.metadata["n_columns"] = self.clinical_df.shape[1]

    def _select_baseline(self):
        before = self.clinical_df["PATNO"].nunique()

        self.baseline_cohort_df = self.clinical_df[
            self.clinical_df["EVENT_ID"]
            == self.cohort_config["baseline_event"]
        ].copy()

        after = self.baseline_cohort_df["PATNO"].nunique()

        self._record_history(
            "baseline",
            before,
            after,
        )

    def _filter_at_risk(self):
        before = self.baseline_cohort_df["PATNO"].nunique()

        eligible = self.cohort_config["at_risk_diagnoses"]

        excluded_df = self.baseline_cohort_df[
            ~self.baseline_cohort_df["PRIMDIAG"].isin(eligible)
        ].copy()

        self._record_exclusions(
            excluded_df,
            "baseline_diagnosis",
            "baseline_not_at_risk",
        )

        self.baseline_cohort_df = self.baseline_cohort_df[
            self.baseline_cohort_df["PRIMDIAG"].isin(eligible)
        ].copy()

        after = self.baseline_cohort_df["PATNO"].nunique()

        self._record_history(
            "at_risk_diagnosis",
            before,
            after,
        )

    def _filter_minimum_visits(self):
        before = self.baseline_cohort_df["PATNO"].nunique()

        minimum_visits = self.cohort_config["minimum_visits"]

        visit_counts = (
            self.clinical_df
            .groupby("PATNO")["EVENT_ID"]
            .nunique()
        )

        eligible_patnos = visit_counts[
            visit_counts >= minimum_visits
        ].index

        excluded_df = self.baseline_cohort_df[
            ~self.baseline_cohort_df["PATNO"].isin(eligible_patnos)
        ].copy()

        self._record_exclusions(
            excluded_df,
            "follow_up",
            "insufficient_follow_up",
        )

        self.baseline_cohort_df = self.baseline_cohort_df[
            self.baseline_cohort_df["PATNO"].isin(eligible_patnos)
        ].copy()

        after = self.baseline_cohort_df["PATNO"].nunique()

        self._record_history(
            "minimum_visits",
            before,
            after,
        )

    def _record_exclusions(self, excluded_df, stage, reason):
        if excluded_df.empty:
            return

        exclusions = pd.DataFrame(
            {
                "PATNO": excluded_df["PATNO"].unique(),
                "stage": stage,
                "reason": reason,
            }
        )

        self.exclusions_df = pd.concat(
            [
                self.exclusions_df,
                exclusions,
            ],
            ignore_index=True,
        )

    def _record_history(self, step, before, after):
        self.history.append(
        {
            "step": step,
            "before": before,
            "after": after,
        }
    )

    def _create_longitudinal_cohort(self):
        eligible_patnos = self.baseline_cohort_df["PATNO"].unique()

        self.longitudinal_cohort_df = self.clinical_df[
            self.clinical_df["PATNO"].isin(eligible_patnos)
        ].copy()

    def _build_metadata(self):
        self.metadata = {

            "dataset": self.dataset_config["dataset_name"],

            "data_version": (
                self.dataset_config["clinical"]["data_version"]
            ),

            "cohort_definition": {

                "baseline_event": (
                    self.cohort_config["baseline_event"]
                ),

                "at_risk_diagnoses": (
                    self.cohort_config["at_risk_diagnoses"]
                ),

                "minimum_visits": (
                    self.cohort_config["minimum_visits"]
                ),

                "include_swedd": (
                    self.cohort_config["include_swedd"]
                ),
            },

            "participants": {

                "initial": (
                    self.clinical_df["PATNO"].nunique()
                ),

                "final": (
                    self.baseline_cohort_df["PATNO"].nunique()
                ),
            },

            "rows": {

                "baseline": (
                    len(self.baseline_cohort_df)
                ),

                "longitudinal": (
                    len(self.longitudinal_cohort_df)
                ),
            },

            "history": self.history,
        }