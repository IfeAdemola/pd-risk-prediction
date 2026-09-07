"""
PPMI outcome definition and extraction.

This module builds participant-level outcomes from
the longitudinal PPMI risk cohort.
"""


from __future__ import annotations

from pathlib import Path
import json

import pandas as pd


class PPMIOutcomeBuilder:
    """
    Build participant-level outcomes from longitudinal PPMI data.

    Outcomes include:
    - Parkinson's disease diagnosis
    - Multiple system atrophy diagnosis
    - Dementia with Lewy bodies diagnosis
    - Death

    The output is one row per participant.
    """

    def __init__(
        self,
        longitudinal_file: str | Path,
        outcome_config: dict,
    ):

        self.longitudinal_file = Path(
            longitudinal_file
        )

        self.outcome_config = outcome_config

        self.df = None
        self.outcome_df = None
        self.metadata = {}
    

    def build(self):
        """
        Build participant-level outcome table.
        """

        self._load_data()

        self._prepare_data()

        self._build_participant_outcomes()

        self._build_metadata()


    def save(self):
        """
        Save outcome dataset and metadata.
        """

        output_config = self.outcome_config["output"]

        output_dir = Path(
            output_config["directory"]
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Outcomes
        self.outcome_df.to_csv(
            output_dir / output_config["outcome_file"],
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
                default=str,
            )

        print(
            f"Saved outcomes to {output_dir}"
        )

    def summary(self):
        """
        Print outcome summary.
        """

        print("=" * 50)
        print("PPMI Outcome Summary")
        print("=" * 50)

        print()

        print("Participants")
        print("--------------------")
        print(
            len(self.outcome_df)
        )

        print()

        print("Outcomes")
        print("--------------------")

        print(
            self.outcome_df[
                [
                    "pd_event",
                    "msa_event",
                    "dlb_event",
                    "death_event",
                ]
            ]
            .sum()
        )

        print()

        print("PD trajectory")
        print("--------------------")

        print(
            self.outcome_df[
                "pd_case_type"
            ]
            .value_counts()
        )

        print()

        print("Death timing")
        print("--------------------")

        print(
            self.outcome_df[
                "death_timing"
            ]
            .value_counts()
        )
    

    # ======================================================================
    # Private methods
    # ======================================================================

    def _load_data(self):
        """
        Load longitudinal cohort data.
        """

        self.df = pd.read_csv(
            self.longitudinal_file
        )

        return self.df

    def _prepare_data(self):
            """
            Prepare longitudinal data.
    
            - Convert dates
            - Add chronological visit order
            """
    
            self.df = self.df.copy()
    
            self.df["visit_date"] = pd.to_datetime(
                self.df["visit_date"],
                format="%m/%Y",
                errors="coerce",
            )
    
            self.df["Death_Date"] = pd.to_datetime(
                self.df["Death_Date"],
                format="%m/%Y",
                errors="coerce",
            )
    
            event_order = {"BL": 0}
    
            for i in range(1, 100):
                event_order[f"V{i:02d}"] = i
    
            self.df["visit_order"] = (
                self.df["EVENT_ID"]
                .map(event_order)
            )
    
            self.df = (
                self.df
                .sort_values(
                    [
                        "PATNO",
                        "visit_order",
                    ]
                )
            )
    
            return self.df

    def _first_diagnosis_event(
        self,
        group: pd.DataFrame,
        code: int,
    ):
        """
        Return first occurrence of diagnosis code.

        Returns
        -------
        dict
            Event information.
        """

        rows = group[
            group["PRIMDIAG"] == code
        ]

        if rows.empty:

            return {
                "event": False,
                "visit": None,
                "date": None,
            }

        first = rows.iloc[0]

        return {
            "event": True,
            "visit": first["EVENT_ID"],
            "date": first["visit_date"],
        }

    def _build_participant_outcomes(self):
        """
        Generate one outcome row per participant.
        """

        outcomes = []

        pd_code = self.outcome_config["outcomes"]["pd"]["diagnosis_code"]
        msa_code = self.outcome_config["outcomes"]["msa"]["diagnosis_code"]
        dlb_code = self.outcome_config["outcomes"]["dlb"]["diagnosis_code"]


        for patno, group in self.df.groupby("PATNO"):

            pd_event = self._first_diagnosis_event(
                group,
                pd_code,
            )

            msa_event = self._first_diagnosis_event(
                group,
                msa_code,
            )

            dlb_event = self._first_diagnosis_event(
                group,
                dlb_code,
            )

            follow_up = self._extract_follow_up(group)

            death = self._extract_death(group)

            trajectory = (
                self._classify_pd_trajectory(group)
            )

            death_timing = self._classify_death_timing(
                pd_event["date"],
                death["death_event"],
                death["death_date"],
            )


            outcomes.append(
                {
                    "PATNO": patno,

                    # PD
                    "pd_event":
                        pd_event["event"],

                    "pd_event_visit":
                        pd_event["visit"],

                    "pd_event_date":
                        pd_event["date"],


                    # MSA
                    "msa_event":
                        msa_event["event"],

                    "msa_event_visit":
                        msa_event["visit"],

                    "msa_event_date":
                        msa_event["date"],


                    # DLB
                    "dlb_event":
                        dlb_event["event"],

                    "dlb_event_visit":
                        dlb_event["visit"],

                    "dlb_event_date":
                        dlb_event["date"],

                    # Follow-up

                    "n_visits":
                        follow_up["n_visits"],

                    "baseline_date":
                        follow_up["baseline_date"],

                    "last_visit":
                        follow_up["last_visit"],

                    "last_visit_date":
                        follow_up["last_visit_date"],

                    "follow_up_years":
                        follow_up["follow_up_years"],


                    # Death

                    "death_event":
                        death["death_event"],

                    "death_date":
                        death["death_date"],

                    "death_timing":
                        death_timing,

                    # Trajectory subtyping (PD case type)
                    "pd_case_type":
                        trajectory["pd_case_type"],

                    "pd_n_visits":
                        trajectory["pd_n_visits"],
                }
            )

        self.outcome_df = pd.DataFrame(
            outcomes
        )


    def _extract_follow_up(
        self,
        group: pd.DataFrame,
    ):
        """
        Extract participant follow-up information.
        """

        baseline_date = (
            group["visit_date"]
            .iloc[0]
        )

        last_visit_date = (
            group["visit_date"]
            .iloc[-1]
        )

        follow_up_years = None

        if (
            pd.notna(baseline_date)
            and pd.notna(last_visit_date)
        ):
            follow_up_years = (
                last_visit_date - baseline_date
            ).days / 365.25


        return {
            "n_visits": len(group),

            "baseline_date": baseline_date,

            "last_visit":
                group["EVENT_ID"].iloc[-1],

            "last_visit_date":
                last_visit_date,

            "follow_up_years":
                follow_up_years,
        }

    def _extract_death(
        self,
        group: pd.DataFrame,
    ):
        """
        Extract death information.
        """

        death_rows = group[
            group[
                self.outcome_config["outcomes"]["death"]["status_column"]
            ]
            == True
        ]

        if death_rows.empty:

            return {
                "death_event": False,
                "death_date": None,
            }


        death_date = (
            death_rows["Death_Date"]
            .dropna()
        )

        return {
            "death_event": True,

            "death_date":
                death_date.iloc[0]
                if not death_date.empty
                else None,
        }

    def _classify_pd_trajectory(
        self,
        group: pd.DataFrame,
    ):
        """
        Classify longitudinal PD diagnostic trajectory.

        Categories:

        persistent_pd:
            PD diagnosis followed by continued PD diagnosis.

        pd_end_of_followup:
            First PD diagnosis occurs at final observed visit.

        pd_reversal:
            PD diagnosis followed by a non-PD diagnosis.

        pd_competing_diagnosis:
            PD followed by MSA or DLB.
        """

        diagnoses = (
            group["PRIMDIAG"]
            .dropna()
            .astype(int)
            .tolist()
        )


        if 1 not in diagnoses:
            return {
                "pd_case_type": "no_pd",
                "pd_n_visits": 0,
            }


        first_pd_index = diagnoses.index(1)

        after_pd = diagnoses[first_pd_index + 1:]


        # PD only observed at last visit

        if len(after_pd) == 0:

            return {
                "pd_case_type":
                    "pd_end_of_followup",

                "pd_n_visits":
                    diagnoses.count(1),
            }


        # Competing diagnoses

        if any(
            d in [5, 11]
            for d in after_pd
        ):

            return {
                "pd_case_type":
                    "pd_competing_diagnosis",

                "pd_n_visits":
                    diagnoses.count(1),
            }


        # Persistent PD

        if all(
            d == 1
            for d in after_pd
        ):

            return {
                "pd_case_type":
                    "persistent_pd",

                "pd_n_visits":
                    diagnoses.count(1),
            }


        # Everything else

        return {
            "pd_case_type":
                "pd_reversal",

            "pd_n_visits":
                diagnoses.count(1),
        }

    def _classify_death_timing(
        self,
        pd_event_date,
        death_event,
        death_date,
    ):
        """
        Classify timing of death relative to PD diagnosis.
        """

        if not death_event:
            return "no_death"

        if pd_event_date is None:
            return "death_without_pd"

        if death_date is None:
            return "death_without_pd"

        if pd_event_date <= death_date:
            return "pd_then_death"

        return "death_before_pd"


    def _build_metadata(self):
        """
        Build metadata summary for outcome dataset.
        """

        self.metadata = {

            "dataset":
                self.outcome_config["dataset"],

            "participants":
                len(self.outcome_df),

            "outcomes": {

                "pd":
                    int(
                        self.outcome_df["pd_event"]
                        .sum()
                    ),

                "msa":
                    int(
                        self.outcome_df["msa_event"]
                        .sum()
                    ),

                "dlb":
                    int(
                        self.outcome_df["dlb_event"]
                        .sum()
                    ),

                "death":
                    int(
                        self.outcome_df["death_event"]
                        .sum()
                    ),
            },


            "pd_trajectory":
                (
                    self.outcome_df["pd_case_type"]
                    .value_counts()
                    .to_dict()
                ),


            "death_timing":
                (
                    self.outcome_df["death_timing"]
                    .value_counts()
                    .to_dict()
                ),
        }

        return self.metadata



