"""
Explore longitudinal diagnosis trajectories in the PPMI risk cohort.

This script is intended for exploratory analyses before implementing
the outcome definition.
"""

from pathlib import Path

import pandas as pd


def load_longitudinal_data(filepath: str | Path) -> pd.DataFrame:
    """
    Load the longitudinal cohort.

    Parameters
    ----------
    filepath
        Path to the longitudinal cohort CSV.

    Returns
    -------
    pandas.DataFrame
        Longitudinal cohort.
    """

    df = pd.read_csv(filepath)

    return df

def parse_dates(df):
    """
    Convert date columns to datetime format.

    visit_date and Death_Date are stored as
    month/year in PPMI.
    """

    df = df.copy()

    df["visit_date"] = pd.to_datetime(
        df["visit_date"],
        format="%m/%Y",
        errors="coerce"
    )

    df["Death_Date"] = pd.to_datetime(
        df["Death_Date"],
        format="%m/%Y",
        errors="coerce"
    )

    return df

def add_visit_order(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add numerical visit order based on EVENT_ID.

    BL -> 0
    V01 -> 1
    V02 -> 2
    ...
    """

    event_order = {"BL": 0}

    for i in range(1, 100):
        event_order[f"V{i:02d}"] = i

    df = df.copy()

    df["visit_order"] = df["EVENT_ID"].map(event_order)

    print(
    df[df["visit_order"].isna()]["EVENT_ID"]
    .value_counts()
    )

    return df

def diagnosis_counts(df: pd.DataFrame) -> None:
    """
    Print diagnosis frequencies across all visits.
    """

    print("\nDiagnosis counts\n")

    print(
        df["PRIMDIAG"]
        .value_counts(dropna=False)
    )

def missing_diagnosis_summary(df: pd.DataFrame) -> None:
    """
    Summarise missing PRIMDIAG values.
    """

    missing_rows = df["PRIMDIAG"].isna().sum()

    missing_participants = (
        df.groupby("PATNO")["PRIMDIAG"]
        .apply(lambda x: x.isna().any())
        .sum()
    )

    baseline_missing = (
        df.loc[df["EVENT_ID"] == "BL", "PRIMDIAG"]
        .isna()
        .sum()
    )

    print("\nMissing diagnosis summary\n")

    print(f"Missing rows: {missing_rows}")
    print(f"Participants affected: {missing_participants}")
    print(f"Missing baseline diagnoses: {baseline_missing}")

def participants_with_pd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return participants who receive a PD diagnosis
    at any visit.
    """

    pd_patnos = df.loc[
        df["PRIMDIAG"] == 1,
        "PATNO"
    ].unique()

    return df[
        df["PATNO"].isin(pd_patnos)
    ].copy()

def print_patient_trajectory(
    df: pd.DataFrame,
    patno: int,
) -> None:
    """
    Print the diagnosis trajectory for one participant.
    """

    patient = (
        df[df["PATNO"] == patno]
        .sort_values("visit_order")
    )

    print()

    print(patient[
        [
            "PATNO",
            "EVENT_ID",
            "visit_order",
            "YEAR",
            "visit_date",
            "PRIMDIAG",
        ]
    ].to_string(index=False))

def diagnosis_patterns(df: pd.DataFrame) -> pd.Series:
    """
    Count diagnosis trajectories.
    """

    patterns = {}

    for _, group in df.groupby("PATNO"):

        group = group.sort_values("visit_order")

        sequence = tuple(
            group["PRIMDIAG"]
            .dropna()
            .astype(int)
            .tolist()
        )

        if len(set(sequence)) > 1:

            patterns[sequence] = (
                patterns.get(sequence, 0) + 1
            )

    patterns = (
        pd.Series(patterns)
        .sort_values(ascending=False)
    )

    return patterns

def build_trajectory_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one participant-level trajectory summary.

    Each row represents one participant and contains:
    - diagnosis trajectory
    - PD conversion information
    - competing diagnoses
    - death information
    - follow-up information
    """

    summaries = []

    for patno, group in df.groupby("PATNO"):

        group = (
            group
            .sort_values("visit_order")
            .copy()
        )

        diagnoses = (
            group["PRIMDIAG"]
            .dropna()
            .astype(int)
            .tolist()
        )

        baseline_diagnosis = (
            diagnoses[0]
            if diagnoses
            else None
        )

        final_diagnosis = (
            diagnoses[-1]
            if diagnoses
            else None
        )

        # -------------------------
        # Follow-up information
        # -------------------------

        n_visits = len(group)

        last_visit = (
            group["EVENT_ID"]
            .iloc[-1]
        )

        last_visit_date = (
            group["visit_date"]
            .iloc[-1]
        )

        baseline_date = (
            group["visit_date"]
            .iloc[0]
        )

        follow_up_years = None

        if pd.notna(baseline_date) and pd.notna(last_visit_date):
            follow_up_years = (
                last_visit_date - baseline_date
            ).days / 365.25


        # -------------------------
        # Diagnosis events
        # -------------------------

        def first_diagnosis_event(code):
            """
            Return first visit/date of diagnosis code.
            """

            rows = group[
                group["PRIMDIAG"] == code
            ]

            if rows.empty:
                return None, None

            return (
                rows["EVENT_ID"].iloc[0],
                rows["visit_date"].iloc[0],
            )


        first_pd_visit, first_pd_date = (
            first_diagnosis_event(1)
        )

        first_pd_position = (
            diagnoses.index(1)
            if 1 in diagnoses
            else None
        )

        first_msa_visit, first_msa_date = (
            first_diagnosis_event(11)
        )

        first_dlb_visit, first_dlb_date = (
            first_diagnosis_event(5)
        )


        # -------------------------
        # Death information
        # -------------------------

        death_rows = group[
            group["Death_Status"] == True
        ]

        death = not death_rows.empty

        death_date = None

        if death:
            death_date = (
                death_rows["Death_Date"]
                .dropna()
                .iloc[0]
                if not death_rows["Death_Date"]
                    .dropna()
                    .empty
                else None
            )


        pd_before_death = False
        death_before_pd = False

        if death_date is not None:

            if first_pd_date is not None:

                pd_before_death = (
                    first_pd_date <= death_date
                )

            else:

                death_before_pd = True


        summaries.append(
            {

                "PATNO": patno,

                # Follow-up
                "n_visits": n_visits,
                "last_visit": last_visit,
                "last_visit_date": last_visit_date,
                "follow_up_years": follow_up_years,


                # Diagnosis trajectory
                "baseline_diagnosis":
                    baseline_diagnosis,

                "final_diagnosis":
                    final_diagnosis,

                "n_unique_diagnoses":
                    len(set(diagnoses)),

                "sequence":
                    tuple(diagnoses),


                # PD
                "ever_pd":
                    1 in diagnoses,

                "first_pd_visit":
                    first_pd_visit,

                "first_pd_date":
                    first_pd_date,

                "first_pd_position":
                    first_pd_position,

                "n_pd_visits":
                    diagnoses.count(1),

                
                # MSA
                "ever_msa":
                    11 in diagnoses,

                "first_msa_visit":
                    first_msa_visit,

                "first_msa_date":
                    first_msa_date,


                # DLB
                "ever_dlb":
                    5 in diagnoses,

                "first_dlb_visit":
                    first_dlb_visit,

                "first_dlb_date":
                    first_dlb_date,


                # Death
                "death":
                    death,

                "death_date":
                    death_date,

                "death_before_pd":
                    death_before_pd,

                "pd_before_death":
                    pd_before_death,
            }
        )


    return pd.DataFrame(summaries)

def save_trajectory_summary(
    summary_df: pd.DataFrame,
    filepath: str | Path,
) -> None:
    """
    Save participant diagnosis trajectory summary.

    Parameters
    ----------
    summary_df
        Output from build_trajectory_summary()

    filepath
        Destination CSV path.
    """

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_df.to_csv(
        filepath,
        index=False,
    )

    print(
        f"Saved trajectory summary to {filepath}"
    )

def death_summary(df: pd.DataFrame) -> None:
    """
    Summarise death information in the cohort.

    Reports:
    - number of participants with death recorded
    - number of death records
    - distribution of death dates
    """

    death_df = (
        df[df["Death_Status"] == True]
        .copy()
    )

    print("\nDeath summary\n")

    print(
        f"Participants with death recorded: "
        f"{death_df['PATNO'].nunique()}"
    )

    print(
        f"Death rows: {len(death_df)}"
    )

    print(
        "\nDeath date summary"
    )

    print(
        death_df["Death_Date"]
        .describe()
    )


# ----------------------------------------
# Trajectory summary statistics
# ----------------------------------------
def event_distribution(summary_df: pd.DataFrame) -> None:
    """
    Summarise major observed events.

    Counts:
    - PD diagnosis
    - MSA diagnosis
    - DLB diagnosis
    - death
    """

    print("\nEvent distribution\n")

    print(
        "Ever PD:"
    )
    print(
        summary_df["ever_pd"]
        .value_counts()
    )

    print(
        "\nEver MSA:"
    )
    print(
        summary_df["ever_msa"]
        .value_counts()
    )

    print(
        "\nEver DLB:"
    )
    print(
        summary_df["ever_dlb"]
        .value_counts()
    )

    print(
        "\nDeath:"
    )
    print(
        summary_df["death"]
        .value_counts()
    )

def final_diagnosis_distribution(
    summary_df: pd.DataFrame
) -> None:
    """
    Distribution of final recorded diagnosis.
    """

    print("\nFinal diagnosis distribution\n")

    print(
        summary_df["final_diagnosis"]
        .value_counts(dropna=False)
    )

def classify_pd_case(row):
    """
    Classify PD trajectories according to what happens
    after the first PD diagnosis.
    """

    if not row["ever_pd"]:
        return None

    sequence = list(row["sequence"])

    first_pd = row["first_pd_position"]

    after_pd = sequence[first_pd + 1:]

    if len(after_pd) == 0:
        return "pd_end_of_followup"

    if any(d in (5, 11) for d in after_pd):
        return "pd_competing_diagnosis"

    if all(d == 1 for d in after_pd):
        return "persistent_pd"

    return "pd_reversal"

def classify_pd_reversal_pattern(row):
    """
    Further classify reversal trajectories.
    """

    if row["pd_case_type"] != "pd_reversal":
        return None

    final_diag = row["final_diagnosis"]

    if final_diag == 1:
        return "returns_to_pd"

    return "reverts_from_pd"

def pd_trajectory_distribution(summary_df):

    summary_df = summary_df.copy()

    summary_df["pd_trajectory_type"] = (
        summary_df.apply(
            classify_pd_trajectory,
            axis=1
        )
    )

    print("\nPD trajectory types\n")

    print(
        summary_df["pd_trajectory_type"]
        .value_counts()
    )

    return summary_df

def single_pd_visit_cases(summary_df):
    """PD diagnosed only once"""

    cases = summary_df[
        summary_df["n_pd_visits"] == 1
    ]

    print(
        "Single PD diagnosis cases:",
        len(cases)
    )

def pd_not_final_cases(summary_df):
    """PD appears but not at last visit"""
    cases = summary_df[
        (summary_df["ever_pd"]) &
        (summary_df["final_diagnosis"] != 1)
    ]

    print(
        "PD not final diagnosis:",
        len(cases)
    )

def competing_diagnosis_summary(summary_df):
    """competing events"""
    print("\nCompeting diagnoses\n")

    print(
        "MSA:"
    )
    print(
        summary_df["ever_msa"]
        .value_counts()
    )

    print(
        "\nDLB:"
    )
    print(
        summary_df["ever_dlb"]
        .value_counts()
    )

def extract_pd_reversal_cases(
    summary_df: pd.DataFrame,
    filepath: str | Path,
) -> pd.DataFrame:
    """
    Extract participants who received a PD diagnosis
    but did not remain PD at the end of follow-up.

    These represent diagnostic uncertainty cases
    such as:
    25 -> 1 -> 25
    17 -> 1 -> 17
    23 -> 1 -> 11
    """

    cases = summary_df[
        (summary_df["ever_pd"]) &
        (summary_df["final_diagnosis"] != 1)
    ].copy()

    cases = cases[
        [
            "PATNO",
            "sequence",
            "n_visits",
            "first_pd_visit",
            "first_pd_date",
            "final_diagnosis",
            "death",
            "death_date",
        ]
    ]

    filepath = Path(filepath)

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cases.to_csv(
        filepath,
        index=False,
    )

    print(
        f"\nSaved PD reversal cases: {len(cases)}"
    )

    return cases

def death_timing_summary(
    summary_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classify death timing relative to PD diagnosis.

    Categories:
    - no_death
    - death_before_pd
    - pd_then_death
    - death_without_pd
    """

    summary_df = summary_df.copy()

    def classify(row):

        if not row["death"]:
            return "no_death"

        if row["ever_pd"]:

            if row["pd_before_death"]:
                return "pd_then_death"

            return "death_before_pd"

        return "death_without_pd"


    summary_df["death_timing"] = (
        summary_df.apply(
            classify,
            axis=1,
        )
    )


    print("\nDeath timing relative to PD\n")

    print(
        summary_df["death_timing"]
        .value_counts()
    )

    return summary_df

def classify_pd_case(row):
    """
    Classify participants who ever received a PD diagnosis.

    Classification is based on what happens after the
    first PD diagnosis.

    Categories
    ----------
    persistent_pd
        All subsequent diagnoses remain PD.

    pd_end_of_followup
        PD is observed only at the final recorded visit,
        so there is no subsequent follow-up.

    pd_reversal
        A non-PD diagnosis occurs after PD.

    pd_competing_diagnosis
        MSA or DLB occurs after PD.
    """

    sequence = list(row["sequence"])

    if not row["ever_pd"]:
        return None

    # first PD diagnosis
    first_pd = sequence.index(1)

    after_pd = sequence[first_pd + 1:]

    # PD occurs at final observed visit
    if len(after_pd) == 0:
        return "pd_end_of_followup"

    # competing diagnosis after PD
    if any(d in (5, 11) for d in after_pd):
        return "pd_competing_diagnosis"

    # every remaining diagnosis is PD
    if all(d == 1 for d in after_pd):
        return "persistent_pd"

    # anything else is a reversal
    return "pd_reversal"


def create_pd_conversion_table(summary):
    """
    Build participant-level PD conversion table.
    """

    pd_cases = (
        summary.loc[summary["ever_pd"]]
        .copy()
    )

    pd_cases["pd_case_type"] = (
        pd_cases.apply(
            classify_pd_case,
            axis=1,
        )
    )

    pd_cases["pd_reversal_pattern"] = (
        pd_cases.apply(
            classify_pd_reversal_pattern,
            axis=1,
        )
    )

    return pd_cases

def summarise_pd_conversion_table(pd_cases):
    """
    Print PD conversion summary.
    """

    print("\nPD case types\n")

    print(
        pd_cases["pd_case_type"]
        .value_counts()
    )

    print("\nPD reversal patterns\n")

    print(
        pd_cases["pd_reversal_pattern"]
        .value_counts(dropna=False)
    )

def main():

    df = load_longitudinal_data(
        "data/processed/cohort/ppmi_risk_cohort_longitudinal.csv"
    )

    df = parse_dates(df)

    df = add_visit_order(df)

    diagnosis_counts(df)

    death_summary(df)

    missing_diagnosis_summary(df)

    pd_df = participants_with_pd(df)

    patterns = diagnosis_patterns(pd_df)

    # print("\nPD diagnosis patterns\n")
    # print(patterns)

    summary = build_trajectory_summary(df)
    save_trajectory_summary(
    summary,
    "data/exploratory/outcomes/ppmi_trajectory_summary.csv",
    )

    pd_summary = build_trajectory_summary(pd_df)
    save_trajectory_summary(
        pd_summary,
        "data/exploratory/outcomes/ppmi_pd_trajectory_summary.csv",
    )

    # print("\nTrajectory summary\n")
    # print(summary.head())
    # print(summary.columns.tolist())
    # print(summary.shape)
    # print(summary["death"].value_counts())
    # print(summary["ever_pd"].value_counts())

    extract_pd_reversal_cases(
    summary,
    "data/exploratory/outcomes/pd_reversal_cases.csv",
    )

    pd_conversion = create_pd_conversion_table(
        summary
    )

    save_trajectory_summary(
    pd_conversion,
    "data/exploratory/outcomes/ppmi_pd_conversion_candidates.csv",
    )

    summarise_pd_conversion_table(pd_conversion)

    save_trajectory_summary(
        pd_conversion,
        "data/exploratory/outcomes/ppmi_pd_conversion_candidates_summary.csv",
    )

    summary = death_timing_summary(summary)
        
    # Trajectory statistics
    # event_distribution(summary)
    # final_diagnosis_distribution(summary)

    # summary = pd_trajectory_distribution(summary)

    # single_pd_visit_cases(summary)
    # pd_not_final_cases(summary)
    # competing_diagnosis_summary(summary)




if __name__ == "__main__":
    main()