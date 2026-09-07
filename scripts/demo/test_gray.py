import numpy as np
import pandas as pd

from pd_risk.modelling.evaluation import SurvivalEvaluator


print("=" * 70)
print("TESTING GRAY'S TEST")
print("=" * 70)


# ------------------------------------------------------------
# Synthetic competing-risk dataset
# ------------------------------------------------------------
#
# event_type:
#   0 = censored
#   1 = PD
#   2 = death
#
# We deliberately make group 2 have substantially more
# PD events than group 1.
#

predictions = pd.DataFrame(
    {
        "time_to_event": [
            0.5, 1.0, 1.5, 2.0, 2.5,
            0.7, 1.2, 1.8, 2.2, 2.8,
            0.9, 1.4, 2.1, 2.6, 3.0,
            0.6, 1.1, 1.7, 2.4, 2.9,
        ],

        "event_type": [
            # Group 1
            1, 0, 2, 0, 2,

            # Group 1
            0, 2, 0, 2, 0,

            # Group 2
            1, 1, 1, 2, 0,

            # Group 2
            1, 1, 2, 1, 0,
        ],

        "risk_group": [
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1,
            2, 2, 2, 2, 2,
            2, 2, 2, 2, 2,
        ],
    }
)


# ------------------------------------------------------------
# Evaluator
# ------------------------------------------------------------

evaluator = SurvivalEvaluator()


print()
print("[1] Event types")

print(
    predictions[
        "event_type"
    ].value_counts().sort_index()
)

print("PASS")


# ------------------------------------------------------------
# Gray's test
# ------------------------------------------------------------

print()
print("[2] Gray's test")

result = evaluator.build_gray_test(
    predictions,
    cause=1,
)

assert result["cause"] == 1

try:
    evaluator.build_gray_test(
        predictions,
        cause=3,
    )
except ValueError:
    print("PASS: invalid cause rejected")
else:
    raise AssertionError(
        "Invalid cause was not rejected"
    )

print(
    f"Statistic: "
    f"{result['statistic']:.6f}"
)

print(
    f"Degrees of freedom: "
    f"{result['degrees_of_freedom']}"
)

print(
    f"P-value: "
    f"{result['p_value']:.6g}"
)


# ------------------------------------------------------------
# Sanity check
# ------------------------------------------------------------

assert result["statistic"] >= 0

assert (
    result["degrees_of_freedom"] == 1
)

assert (
    0 <= result["p_value"] <= 1
)

print("PASS")


print()
print("=" * 70)
print("GRAY'S TEST PASSED")
print("=" * 70)