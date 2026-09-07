"""
Baseline multimodal completeness and intersection analysis for the PPMI
at-risk cohort.

This module operates on the baseline cohort manifest produced by
cohort_alignment.py. It combines modality-specific feature sets into
higher-level modalities, calculates baseline modality completeness and
participant-level modality intersections, and provides compact plots.

MRI can be represented in two ways:
    - "any": participant has at least one MRI feature set
    - "complete": participant has all MRI feature sets

Clinical is the reference cohort, so all participants in the manifest
are assumed to have clinical baseline data.
"""

# TODO: Make more dynamic. Don't define modalities here, as more would be added later on

from pathlib import Path

import pandas as pd

from itertools import combinations

import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MRI_COMPONENTS = [
    "mri_cortical_thickness",
    "mri_cortical_surface_area",
    "mri_regional_volume",
]

GENETICS_COMPONENTS = [
    "genetics_pathogenic_variants",
    "genetics_polygenic_risk",
]

# TODO: Include biospecimen_biomarkers when I fix it
BIOSPECIMEN_COMPONENTS = [
    "biospecimen_saa",
    #"biospecimen_biomarkers"
    ]

# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze_baseline_modalities(
    manifest,
    mri_mode="complete",
):
    """
    Analyze baseline multimodal availability.

    Parameters
    ----------
    manifest : pd.DataFrame or str/Path
        Baseline cohort manifest.

    mri_mode : {"any", "complete"}
        How MRI availability is defined.

        "any":
            At least one MRI feature set is available.

        "complete":
            All three MRI feature sets are available.

    Returns
    -------
    dict
        Contains the modality availability table, combination table,
        and the analysis dataframe.
    """

    if mri_mode not in {"any", "complete"}:
        raise ValueError("mri_mode must be 'any' or 'complete'.")

    # --------------------------------------------------------------
    # Load manifest if necessary
    # --------------------------------------------------------------

    if isinstance(manifest, (str, Path)):
        manifest = pd.read_csv(manifest)

    manifest = manifest.copy()

    if "PATNO" not in manifest.columns:
        raise ValueError("Manifest must contain a PATNO column.")

    print("\n" + "=" * 60)
    print("BASELINE MULTIMODAL ANALYSIS")
    print("=" * 60)

    print(f"Participants: {len(manifest):,}")
    print(f"MRI definition: {mri_mode}")

    # --------------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------------

    required = (
        ["clinical"]
        + MRI_COMPONENTS
        + GENETICS_COMPONENTS
        + BIOSPECIMEN_COMPONENTS
        + ["dat"]

    )

    missing = [
        column for column in required
        if column not in manifest.columns
    ]

    if missing:
        raise ValueError(
            "Manifest is missing required columns: "
            + ", ".join(missing)
        )

    # --------------------------------------------------------------
    # Convert availability columns to boolean
    # --------------------------------------------------------------

    for column in required:
        manifest[column] = manifest[column].fillna(0).astype(bool)

    # --------------------------------------------------------------
    # Combine MRI feature sets
    # --------------------------------------------------------------

    if mri_mode == "any":
        manifest["mri"] = manifest[MRI_COMPONENTS].any(axis=1)
    else:
        manifest["mri"] = manifest[MRI_COMPONENTS].all(axis=1)

    # --------------------------------------------------------------
    # Combine genetics feature sets
    # --------------------------------------------------------------

    # A participant is considered to have genetics if either
    # pathogenic-variant data or PRS data are available.
    manifest["genetics"] = manifest[GENETICS_COMPONENTS].any(axis=1)

    # TODO: Uncomment, when I figure out biomarkers and want to use it
    manifest["biospecimen"] = manifest[BIOSPECIMEN_COMPONENTS] #.any(axis=1)

    # --------------------------------------------------------------
    # Modality availability
    # --------------------------------------------------------------

    modalities = [
        "clinical",
        "dat",
        "mri",
        "genetics",
        "biospecimen",
    ]

    availability = []

    for modality in modalities:
        n = int(manifest[modality].sum())

        availability.append(
            {
                "modality": modality,
                "n": n,
                "percentage": 100 * n / len(manifest),
            }
        )

    availability = pd.DataFrame(availability)

    print("\nModality availability:")
    for row in availability.itertuples(index=False):
        print(
            f"  {row.modality:<12} "
            f"{row.n:>5,} / {len(manifest):,} "
            f"({row.percentage:5.1f}%)"
        )

    # --------------------------------------------------------------
    # Modality combinations
    # --------------------------------------------------------------

    combination_columns = modalities

    combination_df = (
        manifest[combination_columns]
        .astype(int)
        .groupby(combination_columns)
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )

    combination_df["percentage"] = (
        100 * combination_df["n"] / len(manifest)
    )

    # Human-readable combination name
    combination_df["combination"] = combination_df.apply(
        lambda row: " + ".join(
            modality
            for modality in modalities
            if row[modality] == 1
        ),
        axis=1,
    )

    print("\nModality combinations:")
    for row in combination_df.itertuples(index=False):
        print(
            f"  {row.combination:<45} "
            f"{row.n:>5,} ({row.percentage:5.1f}%)"
        )

    return {
        "data": manifest,
        "availability": availability,
        "combinations": combination_df,
        "mri_mode": mri_mode,
    }


# --------------------------------------------------------------
# Build modality set for modality intersection. 
# Not in use currently. 
# Created 127 combinations
# --------------------------------------------------------------
def _build_modality_sets(
    manifest,
    reference_modality="clinical",
    id_columns=("PATNO",),
    include_modalities=None,
):
    """
    Build modality combinations anchored on a reference modality.

    The reference modality is included in every combination.

    Parameters
    ----------
    manifest : pd.DataFrame
        Participant-level modality manifest.

    reference_modality : str
        Modality that must be present in every combination.

    id_columns : tuple
        Columns that should not be treated as modalities.

    include_modalities : list, optional
        Modalities to include. If None, all non-ID columns are used.

    Returns
    -------
    dict
        Mapping of human-readable combination names to modality columns.
    """

    if reference_modality not in manifest.columns:
        raise ValueError(
            f"Reference modality '{reference_modality}' "
            "is not present in the manifest."
        )

    if include_modalities is None:
        modalities = [
            column
            for column in manifest.columns
            if column not in id_columns
            and column != reference_modality
        ]
    else:
        modalities = [
            modality
            for modality in include_modalities
            if modality in manifest.columns
            and modality != reference_modality
        ]

    modality_sets = {
        reference_modality.capitalize(): [reference_modality]
    }

    for r in range(1, len(modalities) + 1):
        for combination in combinations(modalities, r):

            combination = (
                reference_modality,
                *combination,
            )

            label = " + ".join(
                modality.capitalize()
                for modality in combination
            )

            modality_sets[label] = list(combination)

    return modality_sets



# --------------------------------------------------------------
# Modality intersection
# --------------------------------------------------------------
def calculate_modality_intersections(results):
    """
    Calculate overlapping baseline cohorts for predefined
    modality intersections.

    Unlike exclusive modality patterns, intersections overlap.
    For example, Clinical + Genetics includes participants who
    also have MRI, DAT, SAA, etc.

    Returns
    -------
    pd.DataFrame
        One row per modality intersection.
    """

    data = results["data"]

    modality_sets = {
        "Clinical": ["clinical"],

        "Clinical + DAT": ["clinical", "dat"],
        "Clinical + MRI": ["clinical", "mri"],
        "Clinical + Genetics": ["clinical", "genetics"],
        "Clinical + SAA": ["clinical", "biospecimen_saa"],

        "Clinical + DAT + MRI": [
            "clinical", "dat", "mri"
        ],
        "Clinical + DAT + Genetics": [
            "clinical", "dat", "genetics"
        ],
        "Clinical + MRI + Genetics": [
            "clinical", "mri", "genetics"
        ],
        "Clinical + SAA + Genetics": [
            "clinical", "biospecimen_saa", "genetics"
        ],

        "Clinical + DAT + MRI + Genetics": [
            "clinical", "dat", "mri", "genetics"
        ],
    }

    rows = []

    for label, columns in modality_sets.items():
        mask = data[columns].all(axis=1)
        n = int(mask.sum())

        rows.append({
            "intersection": label,
            "n": n,
            "percentage": 100 * n / len(data),
        })

    intersections = pd.DataFrame(rows)

    print("\n" + "=" * 60)
    print("BASELINE MODALITY INTERSECTIONS")
    print("=" * 60)

    for row in intersections.itertuples(index=False):
        print(
            f"  {row.intersection:<40}"
            f"{row.n:>5,} ({row.percentage:5.1f}%)"
        )

    return intersections


# ---------------------------------------------------------------------
# Completeness plot
# ---------------------------------------------------------------------

def plot_modality_completeness(
    results,
    figsize=(7.0, 4.5),
    bar_color="#4C78A8",
    xlabel="Participants with baseline data (%)",
    title=None,
    dpi=300,
):
    """
    Plot baseline participant completeness across modalities.

    Parameters
    ----------
    results : dict
        Results dictionary containing:
        - results["availability"]: DataFrame with columns
          ["modality", "percentage"]
        - results["mri_mode"]: MRI acquisition mode, used in the
          optional title.
    figsize : tuple, default=(7.0, 4.5)
        Figure dimensions in inches.
    bar_color : str, default="#4C78A8"
        Bar colour.
    xlabel : str, default="Participants with baseline data (%)"
        X-axis label.
    title : str or None, default=None
        Optional figure title. For publication figures, it is generally
        preferable to omit the title and put this information in the
        figure caption.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    ax : matplotlib.axes.Axes
        Matplotlib axes.
    """

    availability = (
        results["availability"]
        .loc[:, ["modality", "percentage"]]
        .dropna()
        .sort_values("percentage", ascending=True)
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    # -----------------------------------------------------------------
    # Bars
    # -----------------------------------------------------------------

    bars = ax.barh(
        availability["modality"],
        availability["percentage"],
        height=0.62,
        color=bar_color,
        edgecolor="none",
        zorder=3,
    )

    # -----------------------------------------------------------------
    # Percentage labels
    # -----------------------------------------------------------------

    for bar, percentage in zip(bars, availability["percentage"]):
        ax.text(
            percentage + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{percentage:.1f}%",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            clip_on=False,
        )

    # -----------------------------------------------------------------
    # Axes
    # -----------------------------------------------------------------

    ax.set_xlabel(
        xlabel,
        fontsize=10,
        labelpad=8,
    )

    ax.set_ylabel("")

    ax.set_xlim(0, 105)

    # Keep the x-axis visually quiet.
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(
        axis="x",
        which="major",
        length=3,
        width=0.8,
        labelsize=9,
    )

    ax.tick_params(
        axis="y",
        which="major",
        length=0,
        labelsize=9,
        pad=6,
    )

    # -----------------------------------------------------------------
    # Grid
    # -----------------------------------------------------------------

    ax.set_axisbelow(True)

    ax.grid(
        axis="x",
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
    )

    # -----------------------------------------------------------------
    # Spines
    # -----------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # -----------------------------------------------------------------
    # Optional title
    # -----------------------------------------------------------------

    if title is not None:
        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=12,
        )

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------

    fig.subplots_adjust(
        left=0.25,
        right=0.92,
        bottom=0.18,
        top=0.94 if title is None else 0.88,
    )

    return fig, ax



# ---------------------------------------------------------------------
# Combination plot
# ---------------------------------------------------------------------

def plot_modality_combinations(
    results,
    top_n=None,
    figsize=(8.0, 5.5),
    bar_color="#4C78A8",
    xlabel="Number of baseline participants",
    title=None,
    dpi=300,
):
    """
    Plot participant counts for baseline modality combinations.

    Clinical is included in the underlying combinations because it is
    the reference cohort, but it is removed from the displayed labels.

    Parameters
    ----------
    results : dict
        Results dictionary containing:
        - results["combinations"]: DataFrame with columns
          ["combination", "n", "percentage"]
        - results["mri_mode"]: MRI acquisition mode.
    top_n : int or None, default=None
        Plot only the top N combinations by participant count.
    figsize : tuple, default=(8.0, 5.5)
        Figure dimensions in inches.
    bar_color : str, default="#4C78A8"
        Bar colour.
    xlabel : str, default="Number of baseline participants"
        X-axis label.
    title : str or None, default=None
        Optional figure title. For publication figures, it is generally
        preferable to omit the title and put this information in the
        figure caption.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    ax : matplotlib.axes.Axes
        Matplotlib axes.
    """

    combinations = results["combinations"].copy()

    # -----------------------------------------------------------------
    # Clean displayed combination names
    # -----------------------------------------------------------------

    combinations["display_combination"] = (
        combinations["combination"]
        .str.replace("clinical + ", "", regex=False)
        .str.replace("clinical", "", regex=False)
        .str.strip()
    )

    # Remove empty combinations.
    combinations = combinations[
        combinations["display_combination"].ne("")
    ].copy()

    # -----------------------------------------------------------------
    # Select top N BEFORE sorting for display
    # -----------------------------------------------------------------

    if top_n is not None:
        combinations = (
            combinations
            .nlargest(top_n, "n")
            .copy()
        )

    # Plot smallest → largest so the largest combination appears at top.
    combinations = (
        combinations
        .sort_values("n", ascending=True)
        .reset_index(drop=True)
    )

    if combinations.empty:
        raise ValueError(
            "No modality combinations available to plot."
        )

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    bars = ax.barh(
        combinations["display_combination"],
        combinations["n"],
        height=0.62,
        color=bar_color,
        edgecolor="none",
        zorder=3,
    )

    # -----------------------------------------------------------------
    # Labels: n + percentage
    # -----------------------------------------------------------------

    max_n = combinations["n"].max()

    for bar, row in zip(bars, combinations.itertuples()):
        ax.text(
            row.n + max_n * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{row.n:,} ({row.percentage:.1f}%)",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            clip_on=False,
        )

    # -----------------------------------------------------------------
    # Axes
    # -----------------------------------------------------------------

    ax.set_xlabel(
        xlabel,
        fontsize=10,
        labelpad=8,
    )

    ax.set_ylabel(
        "Modality combination",
        fontsize=10,
        labelpad=8,
    )

    # Give annotations some room to breathe.
    ax.set_xlim(
        0,
        max_n * 1.18,
    )

    ax.tick_params(
        axis="x",
        which="major",
        length=3,
        width=0.8,
        labelsize=9,
    )

    ax.tick_params(
        axis="y",
        which="major",
        length=0,
        labelsize=9,
        pad=6,
    )

    # -----------------------------------------------------------------
    # Grid
    # -----------------------------------------------------------------

    ax.set_axisbelow(True)

    ax.grid(
        axis="x",
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
    )

    # -----------------------------------------------------------------
    # Spines
    # -----------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # -----------------------------------------------------------------
    # Optional title
    # -----------------------------------------------------------------

    if title is not None:
        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=12,
        )

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------

    fig.subplots_adjust(
        left=0.30,
        right=0.94,
        bottom=0.18,
        top=0.94 if title is None else 0.88,
    )

    return fig, ax


# ---------------------------------------------------------------------
# Intersection plot
# ---------------------------------------------------------------------

def plot_modality_intersections(
    intersections,
    figsize=(8.0, 5.5),
    bar_color="#4C78A8",
    xlabel="Number of baseline participants",
    title=None,
    dpi=300,
):
    """
    Plot overlapping baseline modality intersections.

    Parameters
    ----------
    intersections : pandas.DataFrame
        DataFrame containing:
        - "intersection": modality intersection label
        - "n": number of participants
        - "percentage": percentage of the baseline cohort
    figsize : tuple, default=(8.0, 5.5)
        Figure dimensions in inches.
    bar_color : str, default="#4C78A8"
        Bar colour.
    xlabel : str, default="Number of baseline participants"
        X-axis label.
    title : str or None, default=None
        Optional figure title. For publication figures, it is generally
        preferable to omit the title and put this information in the
        figure caption.
    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.
    ax : matplotlib.axes.Axes
        Matplotlib axes.
    """

    # -----------------------------------------------------------------
    # Prepare data
    # -----------------------------------------------------------------

    plot_data = (
        intersections
        .loc[:, ["intersection", "n", "percentage"]]
        .dropna()
        .sort_values("n", ascending=True)
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No modality intersections available to plot."
        )

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    bars = ax.barh(
        plot_data["intersection"],
        plot_data["n"],
        height=0.62,
        color=bar_color,
        edgecolor="none",
        zorder=3,
    )

    # -----------------------------------------------------------------
    # Labels: n + percentage
    # -----------------------------------------------------------------

    max_n = plot_data["n"].max()

    for bar, row in zip(bars, plot_data.itertuples()):
        ax.text(
            row.n + max_n * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{row.n:,} ({row.percentage:.1f}%)",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            clip_on=False,
        )

    # -----------------------------------------------------------------
    # Axes
    # -----------------------------------------------------------------

    ax.set_xlabel(
        xlabel,
        fontsize=10,
        labelpad=8,
    )

    ax.set_ylabel(
        "Modality intersection",
        fontsize=10,
        labelpad=8,
    )

    # Reserve space for the n (%) annotations.
    ax.set_xlim(
        0,
        max_n * 1.18,
    )

    ax.tick_params(
        axis="x",
        which="major",
        length=3,
        width=0.8,
        labelsize=9,
    )

    ax.tick_params(
        axis="y",
        which="major",
        length=0,
        labelsize=9,
        pad=6,
    )

    # -----------------------------------------------------------------
    # Grid
    # -----------------------------------------------------------------

    ax.set_axisbelow(True)

    ax.grid(
        axis="x",
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
    )

    # -----------------------------------------------------------------
    # Spines
    # -----------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # -----------------------------------------------------------------
    # Optional title
    # -----------------------------------------------------------------

    if title is not None:
        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=12,
        )

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------

    fig.subplots_adjust(
        left=0.30,
        right=0.94,
        bottom=0.18,
        top=0.94 if title is None else 0.88,
    )

    return fig, ax


# -------------------------------------------------------------------------
# UPSET combination plot
# -------------------------------------------------------------------------
def plot_modality_upset(
    results,
    min_subset_size=1,
    figsize=(16, 6),
):
    from upsetplot import UpSet, from_indicators
    import matplotlib.pyplot as plt

    data = results["data"]

    modalities = [
        "clinical",
        "dat",
        "mri",
        "genetics",
        "biospecimen",
    ]

    upset_data = from_indicators(
        modalities,
        data[modalities].astype(bool),
    )

    print("\nGenerating UpSet plot of baseline modality intersections...")

    fig = plt.figure(figsize=figsize)

    upset = UpSet(
        upset_data,
        subset_size="count",
        min_subset_size=min_subset_size,
        show_counts=False,
        sort_by="cardinality",
        element_size=None,
    )

    axes = upset.plot(fig=fig)

    intersection_ax = axes["intersections"]

    # Manual counts
    for bar in intersection_ax.patches:
        height = bar.get_height()

        if height <= 0:
            continue

        intersection_ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height):,}",
            ha="center",
            va="bottom",
        )

    #fig.suptitle(
    #    "Baseline Multimodal Intersections",
    #    y=1.02,
    #)

    plt.show()






# ------------------------------------
# Convert based modality analysis
# ------------------------------------
import pandas as pd
import matplotlib.pyplot as plt


def analyse_modality_combinations_by_outcome(
    results,
    outcome_file,
    project_root=None,
):
    """
    Compare exclusive baseline modality combinations between the
    full at-risk cohort and participants who subsequently convert
    to Parkinson's disease.

    Parameters
    ----------
    results : dict
        Output from the baseline multimodal analysis. Must contain
        ``results["data"]`` with one row per participant and modality
        availability indicators.

    outcome_file : str or Path
        Cross-sectional outcome file containing PATNO and pd_event.

    project_root : str or Path, optional
        Project root used to resolve relative paths.

    Returns
    -------
    pd.DataFrame
        One row per exclusive modality combination with cohort size,
        converter count, percentages, and converter rate.
    """

    from pathlib import Path

    if project_root is None:
        project_root = Path.cwd()
    else:
        project_root = Path(project_root)

    outcome_file = Path(outcome_file)

    if not outcome_file.is_absolute():
        outcome_file = project_root / outcome_file

    data = results["data"].copy()

    # --------------------------------------------------------------
    # Load outcome
    # --------------------------------------------------------------

    outcomes = pd.read_csv(
        outcome_file,
        usecols=["PATNO", "pd_event"],
    )

    outcomes["pd_event"] = (
        outcomes["pd_event"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    # --------------------------------------------------------------
    # Merge outcome onto modality manifest
    # --------------------------------------------------------------

    data = data.merge(
        outcomes,
        on="PATNO",
        how="left",
        validate="one_to_one",
    )

    if data["pd_event"].isna().any():
        n_missing = data["pd_event"].isna().sum()

        raise ValueError(
            f"{n_missing:,} participants in the modality cohort "
            "do not have an outcome."
        )

    # --------------------------------------------------------------
    # Define exclusive modality combination
    # --------------------------------------------------------------

    modalities = [
        "clinical",
        "dat",
        "mri",
        "genetics",
        "biospecimen",
    ]

    def combination(row):
        available = [
            modality
            for modality in modalities
            if row[modality] == 1
        ]

        return " + ".join(
            modality.capitalize()
            for modality in available
        )

    data["modality_combination"] = data.apply(
        combination,
        axis=1,
    )

    # --------------------------------------------------------------
    # Summarise
    # --------------------------------------------------------------

    summary = (
        data
        .groupby("modality_combination")
        .agg(
            n=("PATNO", "size"),
            n_converters=("pd_event", "sum"),
        )
        .reset_index()
    )

    total_n = len(data)
    total_converters = int(data["pd_event"].sum())

    summary["pct_cohort"] = (
        100 * summary["n"] / total_n
    )

    summary["pct_all_converters"] = (
        100 * summary["n_converters"] / total_converters
    )

    summary["converter_rate"] = (
        100 * summary["n_converters"] / summary["n"]
    )

    # Sort by cohort size
    summary = summary.sort_values(
        "n",
        ascending=False,
    ).reset_index(drop=True)

    return summary

# ---------------------------------------------------------------------
# Modality combinations by outcome
# ---------------------------------------------------------------------

def plot_modality_combinations_by_outcome(
    summary,
    figsize=(7.2, 5.2),
    colors=("#9AA0A6", "#2F5597"),
    xlabel="Participants with data in block (%)",
    ylabel="Baseline modality combination",
    title=None,
    dpi=300,
):
    """
    Plot baseline modality combinations as percentages of the
    at-risk cohort and converter cohort.

    Parameters
    ----------
    summary : pandas.DataFrame
        DataFrame containing:
        - "modality_combination"
        - "pct_cohort"
        - "pct_all_converters"

    figsize : tuple, default=(7.2, 5.2)
        Figure dimensions in inches.

    colors : tuple, default=("#9AA0A6", "#2F5597")
        Colours for the at-risk cohort and converters.

    xlabel : str, default="Participants with data in block (%)"
        X-axis label.

    ylabel : str, default="Baseline modality combination"
        Y-axis label.

    title : str or None, default=None
        Optional figure title. For publication figures, it is generally
        preferable to omit the title and describe the analysis in the
        figure caption.

    dpi : int, default=300
        Figure resolution.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure.

    ax : matplotlib.axes.Axes
        Matplotlib axes.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    # -----------------------------------------------------------------
    # Validate input
    # -----------------------------------------------------------------

    required_columns = {
        "modality_combination",
        "pct_cohort",
        "pct_all_converters",
    }

    missing = required_columns.difference(summary.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    # -----------------------------------------------------------------
    # Prepare data
    # -----------------------------------------------------------------

    plot_data = (
        summary[
            [
                "modality_combination",
                "pct_cohort",
                "pct_all_converters",
            ]
        ]
        .dropna(subset=["modality_combination"])
        .copy()
        .sort_values("pct_cohort", ascending=True)
        .reset_index(drop=True)
    )

    if plot_data.empty:
        raise ValueError(
            "No modality-combination data available to plot."
        )

    y = np.arange(len(plot_data))

    # -----------------------------------------------------------------
    # Figure
    # -----------------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=dpi,
    )

    bar_height = 0.30

    color_at_risk, color_converter = colors

    # -----------------------------------------------------------------
    # At-risk cohort
    # -----------------------------------------------------------------

    ax.barh(
        y - bar_height / 2,
        plot_data["pct_cohort"],
        height=bar_height,
        color=color_at_risk,
        edgecolor="none",
        label="At-risk cohort",
        zorder=3,
    )

    # -----------------------------------------------------------------
    # Converters
    # -----------------------------------------------------------------

    ax.barh(
        y + bar_height / 2,
        plot_data["pct_all_converters"],
        height=bar_height,
        color=color_converter,
        edgecolor="none",
        label="Converters",
        zorder=3,
    )

    # -----------------------------------------------------------------
    # Y-axis
    # -----------------------------------------------------------------

    ax.set_yticks(y)
    ax.set_yticklabels(
        plot_data["modality_combination"],
        fontsize=9,
    )

    ax.set_ylabel(
        ylabel,
        fontsize=10,
        labelpad=8,
    )

    ax.tick_params(
        axis="y",
        which="major",
        length=0,
        pad=6,
    )

    # -----------------------------------------------------------------
    # X-axis
    # -----------------------------------------------------------------

    ax.set_xlabel(
        xlabel,
        fontsize=10,
        labelpad=8,
    )

    max_pct = max(
        plot_data["pct_cohort"].max(),
        plot_data["pct_all_converters"].max(),
    )

    # Dynamically scale the axis while keeping enough room
    # for percentage annotations. Never exceed 100%.
    x_max = min(
        100,
        max_pct * 1.20,
    )

    # Avoid an excessively narrow axis for very small percentages.
    x_max = max(x_max, 10)

    ax.set_xlim(0, x_max)

    # Adaptive tick spacing.
    if x_max <= 30:
        tick_interval = 5
    elif x_max <= 60:
        tick_interval = 10
    else:
        tick_interval = 20

    ax.xaxis.set_major_locator(
        MultipleLocator(tick_interval)
    )

    ax.tick_params(
        axis="x",
        which="major",
        length=3,
        width=0.8,
        labelsize=9,
    )

    # -----------------------------------------------------------------
    # Grid
    # -----------------------------------------------------------------

    ax.set_axisbelow(True)

    ax.grid(
        axis="x",
        which="major",
        linestyle="-",
        linewidth=0.6,
        alpha=0.25,
        zorder=0,
    )

    # -----------------------------------------------------------------
    # Spines
    # -----------------------------------------------------------------

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)

    # -----------------------------------------------------------------
    # Percentage labels
    # -----------------------------------------------------------------

    label_offset = max_pct * 0.015

    for i, row in plot_data.iterrows():

        # At-risk cohort
        if row["pct_cohort"] > 0:
            ax.text(
                row["pct_cohort"] + label_offset,
                i - bar_height / 2,
                f"{row['pct_cohort']:.1f}%",
                ha="left",
                va="center",
                fontsize=8.5,
                color="#444444",
                clip_on=False,
            )

        # Converters
        if row["pct_all_converters"] > 0:
            ax.text(
                row["pct_all_converters"] + label_offset,
                i + bar_height / 2,
                f"{row['pct_all_converters']:.1f}%",
                ha="left",
                va="center",
                fontsize=8.5,
                color=color_converter,
                fontweight="medium",
                clip_on=False,
            )

    # -----------------------------------------------------------------
    # Legend
    # -----------------------------------------------------------------

    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=2,
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.5,
        borderaxespad=0,
    )

    # -----------------------------------------------------------------
    # Optional title
    # -----------------------------------------------------------------

    if title is not None:
        ax.set_title(
            title,
            fontsize=11,
            fontweight="bold",
            loc="left",
            pad=32,
        )

    # -----------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------

    fig.subplots_adjust(
        left=0.30,
        right=0.94,
        bottom=0.18,
        top=0.88,
    )

    return fig, ax
