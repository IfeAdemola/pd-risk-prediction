"""
Standalone sanity checks for CauseSpecificCoxModel.

This is intentionally NOT a pytest test.

It checks that the competing-risk Cox model:

1. Fits successfully.
2. Uses event_type correctly.
3. Produces PD and death cumulative incidence predictions.
4. Produces event-free survival.
5. Keeps predictions within valid probability bounds.
6. Produces non-decreasing CIFs as the horizon increases.
7. Satisfies approximately:

       PD CIF + Death CIF + Event-free survival = 1

8. Produces distinct PD and death risk scores.
"""

import numpy as np
import pandas as pd

from pd_risk.modelling.survival import (
    CauseSpecificCoxModel,
)


def build_synthetic_data(
    n=120,
    seed=42,
):
    """
    Build a small synthetic competing-risk dataset.

    event_type:
        0 = censored
        1 = PD
        2 = death
    """

    rng = np.random.default_rng(seed)

    # ---------------------------------------------------------
    # Features
    # ---------------------------------------------------------

    age = rng.normal(
        loc=65,
        scale=6,
        size=n,
    )

    biomarker = rng.normal(
        loc=0,
        scale=1,
        size=n,
    )

    X = pd.DataFrame(
        {
            "age": age,
            "biomarker": biomarker,
        }
    )

    # ---------------------------------------------------------
    # Event times
    # ---------------------------------------------------------

    # PD hazard increases with biomarker.
    pd_rate = (
        0.08
        * np.exp(
            0.55 * biomarker
        )
    )

    # Death hazard increases with age.
    death_rate = (
        0.035
        * np.exp(
            0.08 * (age - 65)
        )
    )

    pd_time = (
        rng.exponential(
            scale=1 / pd_rate
        )
    )

    death_time = (
        rng.exponential(
            scale=1 / death_rate
        )
    )

    # Administrative / loss-to-follow-up censoring.
    censor_time = rng.uniform(
        1.0,
        4.0,
        size=n,
    )

    # ---------------------------------------------------------
    # Determine observed event
    # ---------------------------------------------------------

    time_to_event = np.minimum.reduce(
        [
            pd_time,
            death_time,
            censor_time,
        ]
    )

    event_type = np.zeros(
        n,
        dtype=int,
    )

    pd_first = (
        (pd_time <= death_time)
        & (pd_time <= censor_time)
    )

    death_first = (
        (death_time < pd_time)
        & (death_time <= censor_time)
    )

    event_type[pd_first] = 1
    event_type[death_first] = 2

    y = pd.DataFrame(
        {
            "time_to_event": time_to_event,
            "event_type": event_type,
        }
    )

    return X, y


def assert_probability(
    values,
    name,
):
    if not np.isfinite(values).all():
        raise AssertionError(
            f"{name} contains non-finite values."
        )

    if (
        (values < -1e-10).any()
        or (values > 1 + 1e-10).any()
    ):
        raise AssertionError(
            f"{name} contains values outside [0, 1]."
        )


def main():

    print("=" * 70)
    print("CAUSE-SPECIFIC COX MODEL SANITY CHECK")
    print("=" * 70)

    # ---------------------------------------------------------
    # Generate data
    # ---------------------------------------------------------

    X, y = build_synthetic_data()

    print()
    print("Synthetic dataset")
    print("-" * 70)

    print(
        f"Participants: {len(X)}"
    )

    print(
        "\nEvent types:"
    )

    print(
        y["event_type"]
        .value_counts()
        .sort_index()
    )

    # ---------------------------------------------------------
    # Basic checks
    # ---------------------------------------------------------

    expected_event_types = {
        0,
        1,
        2,
    }

    observed_event_types = set(
        y["event_type"]
        .unique()
    )

    assert observed_event_types <= (
        expected_event_types
    )

    assert (
        y["event_type"] == 1
    ).sum() > 0

    assert (
        y["event_type"] == 2
    ).sum() > 0

    print(
        "\n✓ Dataset contains censored, PD, "
        "and death observations."
    )

    # ---------------------------------------------------------
    # Fit model
    # ---------------------------------------------------------

    model = CauseSpecificCoxModel(
        penalizer=0.01,
    )

    model.fit(
        X,
        y,
    )

    print(
        "\n✓ Cause-specific Cox models fitted."
    )

    # ---------------------------------------------------------
    # Risk scores
    # ---------------------------------------------------------

    pd_risk = (
        model.predict_risk(X)
    )

    death_risk = (
        model.predict_death_risk(X)
    )

    assert len(pd_risk) == len(X)
    assert len(death_risk) == len(X)

    assert np.isfinite(
        pd_risk
    ).all()

    assert np.isfinite(
        death_risk
    ).all()

    print(
        "✓ PD and death risk scores generated."
    )

    # ---------------------------------------------------------
    # Predictions at 2 years
    # ---------------------------------------------------------

    horizon = 2.0

    pd_cif = (
        model.predict_pd_cumulative_incidence(
            X,
            time=horizon,
        )
    )

    death_cif = (
        model.predict_death_cumulative_incidence(
            X,
            time=horizon,
        )
    )

    survival = (
        model.predict_survival(
            X,
            time=horizon,
        )
    )

    # ---------------------------------------------------------
    # Probability checks
    # ---------------------------------------------------------

    assert_probability(
        pd_cif,
        "PD cumulative incidence",
    )

    assert_probability(
        death_cif,
        "Death cumulative incidence",
    )

    assert_probability(
        survival,
        "Event-free survival",
    )

    print(
        "✓ All predictions are valid probabilities."
    )

    # ---------------------------------------------------------
    # CIF monotonicity
    # ---------------------------------------------------------

    horizons = [
        0.5,
        1.0,
        1.5,
        2.0,
    ]

    pd_cifs = []

    death_cifs = []

    for time in horizons:

        pd_cifs.append(
            model.predict_pd_cumulative_incidence(
                X,
                time=time,
            )
        )

        death_cifs.append(
            model.predict_death_cumulative_incidence(
                X,
                time=time,
            )
        )

    pd_cifs = np.asarray(
        pd_cifs
    )

    death_cifs = np.asarray(
        death_cifs
    )

    pd_monotonic = np.all(
        np.diff(
            pd_cifs,
            axis=0,
        ) >= -1e-10
    )

    death_monotonic = np.all(
        np.diff(
            death_cifs,
            axis=0,
        ) >= -1e-10
    )

    assert pd_monotonic, (
        "PD CIF is not monotonic."
    )

    assert death_monotonic, (
        "Death CIF is not monotonic."
    )

    print(
        "✓ PD and death CIFs are non-decreasing."
    )

    # ---------------------------------------------------------
    # Probability decomposition
    # ---------------------------------------------------------

    total_probability = (
        pd_cif
        + death_cif
        + survival
    )

    decomposition_error = np.max(
        np.abs(
            total_probability
            - 1.0
        )
    )

    print()
    print(
        "Probability decomposition"
    )
    print("-" * 70)

    print(
        f"Maximum absolute error: "
        f"{decomposition_error:.6e}"
    )

    assert decomposition_error < 1e-6, (
        "PD CIF + death CIF + survival "
        "does not equal approximately 1."
    )

    print(
        "✓ Probability decomposition holds."
    )

    # ---------------------------------------------------------
    # Check that the two risk scores are not identical
    # ---------------------------------------------------------

    if np.allclose(
        pd_risk,
        death_risk,
    ):
        raise AssertionError(
            "PD and death risk scores are identical. "
            "This suggests the two cause-specific models "
            "may not have been fitted independently."
        )

    print(
        "✓ PD and death risk scores are distinct."
    )

    # ---------------------------------------------------------
    # Display prediction summary
    # ---------------------------------------------------------

    prediction_summary = pd.DataFrame(
        {
            "pd_cif_2y": pd_cif,
            "death_cif_2y": death_cif,
            "event_free_survival_2y": survival,
        }
    )

    print()
    print(
        "Prediction summary"
    )
    print("-" * 70)

    print(
        prediction_summary.describe()
    )

    # ---------------------------------------------------------
    # Model summaries
    # ---------------------------------------------------------

    summaries = (
        model.summary()
    )

    print()
    print(
        "PD cause-specific model"
    )
    print("-" * 70)

    print(
        summaries["pd"][
            [
                "coef",
                "hazard_ratio",
                "hazard_ratio_lower",
                "hazard_ratio_upper",
            ]
        ]
    )

    print()
    print(
        "Death cause-specific model"
    )
    print("-" * 70)

    print(
        summaries["death"][
            [
                "coef",
                "hazard_ratio",
                "hazard_ratio_lower",
                "hazard_ratio_upper",
            ]
        ]
    )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()
