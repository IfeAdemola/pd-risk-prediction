import numpy as np
import pandas as pd

from pd_risk.modelling.evaluation import SurvivalEvaluator


def main():

    print("=" * 70)
    print("TESTING SURVIVAL EVALUATOR")
    print("=" * 70)

    # ---------------------------------------------------------
    # Small synthetic competing-risk dataset
    # ---------------------------------------------------------

    y_train = pd.DataFrame(
        {
            "time_to_event": [
                0.5,
                0.8,
                1.0,
                1.2,
                1.5,
                1.8,
                2.0,
                2.5,
                3.0,
                3.5,
                4.0,
                4.5,
            ],
            "event_type": [
                1,  # PD
                2,  # death
                0,  # censored
                1,  # PD
                2,  # death
                0,  # censored
                1,  # PD
                0,  # censored
                2,  # death
                0,  # censored
                1,  # PD
                0,  # censored
            ],
        }
    )

    y_test = pd.DataFrame(
        {
            "time_to_event": [
                0.6,
                0.9,
                1.1,
                1.4,
                1.7,
                2.2,
                2.8,
                3.2,
            ],
            "event_type": [
                1,  # PD before 2 years
                2,  # death before 2 years
                0,  # censored before 2 years
                1,  # PD before 2 years
                2,  # death before 2 years
                0,  # censored after 2 years
                1,  # PD after horizon
                0,  # censored after horizon
            ],
        }
    )

    horizon = 2.0

    # Synthetic model predictions.
    #
    # Higher risk score = greater predicted PD risk.
    risk_scores = np.array(
        [
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
            0.40,
            0.30,
            0.20,
        ]
    )

    # Predicted 2-year PD cumulative incidence.
    predicted_pd_cif = np.array(
        [
            0.80,
            0.70,
            0.60,
            0.55,
            0.45,
            0.35,
            0.30,
            0.20,
        ]
    )

    evaluator = SurvivalEvaluator()

    # ---------------------------------------------------------
    # 1. Target validation
    # ---------------------------------------------------------

    print("\n[1] Target validation")

    evaluator._validate_targets(y_train)
    evaluator._validate_targets(y_test)

    print("PASS")

    # ---------------------------------------------------------
    # 2. Check event types
    # ---------------------------------------------------------

    print("\n[2] Event types")

    print(
        "Training:",
        y_train["event_type"].value_counts().sort_index().to_dict(),
    )

    print(
        "Validation:",
        y_test["event_type"].value_counts().sort_index().to_dict(),
    )

    assert set(
        y_train["event_type"]
    ).issubset({0, 1, 2})

    assert set(
        y_test["event_type"]
    ).issubset({0, 1, 2})

    print("PASS")

    # ---------------------------------------------------------
    # 3. Null risk
    # ---------------------------------------------------------

    print("\n[3] Null 2-year PD risk")

    null_risk = evaluator._null_risk(
        y_train=y_train,
        horizon=horizon,
    )

    print(
        f"Null 2-year PD CIF: {null_risk:.6f}"
    )

    assert 0.0 <= null_risk <= 1.0

    print("PASS")

    # ---------------------------------------------------------
    # 4. Brier score
    # ---------------------------------------------------------

    print("\n[4] Brier score")

    brier = evaluator._brier_score(
        y_train=y_train,
        y_test=y_test,
        predicted_pd_cif=predicted_pd_cif,
        horizon=horizon,
    )

    print(
        f"2-year PD Brier score: {brier:.6f}"
    )

    assert np.isfinite(brier)
    assert brier >= 0.0

    print("PASS")

    # ---------------------------------------------------------
    # 5. Null Brier score
    # ---------------------------------------------------------

    print("\n[5] Null Brier score")

    null_brier = evaluator._null_brier_score(
        y_train=y_train,
        y_test=y_test,
        horizon=horizon,
    )

    print(
        f"Null Brier score: {null_brier:.6f}"
    )

    assert np.isfinite(null_brier)
    assert null_brier >= 0.0

    print("PASS")

    # ---------------------------------------------------------
    # 6. Brier skill score
    # ---------------------------------------------------------

    print("\n[6] Brier skill score")

    skill = evaluator._brier_skill_score(
        brier_score=brier,
        null_brier_score=null_brier,
    )

    print(
        f"Brier skill score: {skill:.6f}"
    )

    assert np.isfinite(skill)

    print("PASS")

    # ---------------------------------------------------------
    # 7. Uno C-index
    # ---------------------------------------------------------

    print("\n[7] Uno C-index")

    c_index = evaluator._uno_c_index(
        y_train=y_train,
        y_test=y_test,
        risk_scores=risk_scores,
        horizon=horizon,
    )

    print(
        f"Uno C-index: {c_index:.6f}"
    )

    assert np.isfinite(c_index)
    assert 0.0 <= c_index <= 1.0

    print("PASS")

    # ---------------------------------------------------------
    # 8. Complete evaluation
    # ---------------------------------------------------------

    print("\n[8] Complete evaluation")

    results = evaluator.evaluate(
        y_train=y_train,
        y_test=y_test,
        risk_scores=risk_scores,
        predicted_pd_cif=predicted_pd_cif,
        horizon=horizon,
    )

    print("\nResults:")

    for key, value in results.items():
        print(
            f"  {key}: {value:.6f}"
        )

    assert "uno_c_index" in results
    assert "brier_score" in results
    assert "null_brier_score" in results
    assert "brier_skill_score" in results

    print("\nPASS")

    # ---------------------------------------------------------
    # 9. Final result
    # ---------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL EVALUATOR TESTS PASSED")
    print("=" * 70)


if __name__ == "__main__":
    main()