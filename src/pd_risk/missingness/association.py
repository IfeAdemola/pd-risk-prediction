from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.stats.multitest import multipletests

class MissingnessAssociationAnalyzer:
    """
    Analyze associations between feature missingness and:

    - observed baseline covariates
    - time-to-event outcomes

    Missingness is represented by binary indicators:
        1 = feature is missing
        0 = feature is observed
    """

    def __init__(
        self,
        X: pd.DataFrame,
        identifiers: pd.DataFrame,
        y: Any,
        modality: str,
        config: dict,
    ) -> None:

        self.X = X.copy()

        self.identifiers = identifiers.copy()

        self.y = y

        self.modality = modality

        self.config = config

        self._validate_inputs()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_inputs(self) -> None:

        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame.")

        if not isinstance(self.identifiers, pd.DataFrame):
            raise TypeError(
                "identifiers must be a pandas DataFrame."
            )

        if len(self.X) != len(self.identifiers):
            raise ValueError(
                "X and identifiers must have the same number "
                "of rows."
            )

        if len(self.X) != len(self.y):
            raise ValueError(
                "X and y must have the same number of rows."
            )

        if not self.modality:
            raise ValueError(
                "modality must be a non-empty string."
            )

    # ------------------------------------------------------------------
    # Missingness indicators
    # ------------------------------------------------------------------

    def missingness_indicators(self) -> pd.DataFrame:

        return self.X.isna().astype(int)

    # ------------------------------------------------------------------
    # Outcome associations
    # ------------------------------------------------------------------

    def feature_outcome_associations(
        self,
    ) -> pd.DataFrame:

        """
        Compare outcome distributions between participants
        with observed versus missing values for each feature.

        For survival outcomes, the initial implementation reports
        descriptive event rates and group sizes. Statistical
        survival testing can be added without changing the
        missingness representation.
        """

        missing = self.missingness_indicators()

        event = self._extract_event_indicator()

        rows = []

        for feature in missing.columns:

            indicator = missing[feature]

            missing_mask = indicator == 1
            observed_mask = indicator == 0

            missing_n = int(missing_mask.sum())
            observed_n = int(observed_mask.sum())

            missing_events = int(
                event.loc[missing_mask].sum()
            )

            observed_events = int(
                event.loc[observed_mask].sum()
            )

            missing_event_rate = (
                missing_events / missing_n
                if missing_n > 0
                else np.nan
            )

            observed_event_rate = (
                observed_events / observed_n
                if observed_n > 0
                else np.nan
            )

            rows.append(
                {
                    "feature": feature,
                    "modality": self.modality,
                    "missing_n": missing_n,
                    "observed_n": observed_n,
                    "missing_events": missing_events,
                    "observed_events": observed_events,
                    "missing_event_rate": (
                        missing_event_rate
                    ),
                    "observed_event_rate": (
                        observed_event_rate
                    ),
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Covariate associations
    # ------------------------------------------------------------------

    def feature_covariate_associations(
        self,
        covariates: pd.DataFrame,
    ) -> pd.DataFrame:

        """
        Test whether feature missingness is associated with
        observed covariates.

        Numeric covariates:
            Mann-Whitney U test

        Binary/categorical covariates:
            Chi-square test when appropriate, otherwise Fisher's
            exact test for 2x2 tables.
        """

        if not isinstance(covariates, pd.DataFrame):
            raise TypeError(
                "covariates must be a pandas DataFrame."
            )

        if len(covariates) != len(self.X):
            raise ValueError(
                "covariates and X must have the same number "
                "of rows."
            )

        missing = self.missingness_indicators()

        rows = []

        for feature in missing.columns:

            indicator = missing[feature]

            for covariate in covariates.columns:

                values = covariates[covariate]

                valid = values.notna()

                indicator_valid = indicator.loc[valid]

                values_valid = values.loc[valid]

                if indicator_valid.nunique() < 2:
                    continue

                if pd.api.types.is_numeric_dtype(
                    values_valid
                ):

                    missing_values = values_valid.loc[
                        indicator_valid == 1
                    ]

                    observed_values = values_valid.loc[
                        indicator_valid == 0
                    ]

                    if (
                        len(missing_values) == 0
                        or len(observed_values) == 0
                    ):
                        continue

                    statistic, p_value = (
                        stats.mannwhitneyu(
                            missing_values,
                            observed_values,
                            alternative="two-sided",
                        )
                    )

                    test = "mann_whitney_u"

                    missing_summary = (
                        float(missing_values.median())
                    )

                    observed_summary = (
                        float(observed_values.median())
                    )

                else:

                    table = pd.crosstab(
                        indicator_valid,
                        values_valid,
                    )

                    if table.shape == (2, 2):

                        statistic, p_value = (
                            stats.fisher_exact(table)
                        )

                        test = "fisher_exact"

                    else:

                        statistic, p_value, _, _ = (
                            stats.chi2_contingency(
                                table
                            )
                        )

                        test = "chi_square"

                    missing_summary = (
                        int(
                            (
                                indicator_valid == 1
                            ).sum()
                        )
                    )

                    observed_summary = (
                        int(
                            (
                                indicator_valid == 0
                            ).sum()
                        )
                    )

                rows.append(
                    {
                        "feature": feature,
                        "modality": self.modality,
                        "covariate": covariate,
                        "test": test,
                        "missing_group_summary": (
                            missing_summary
                        ),
                        "observed_group_summary": (
                            observed_summary
                        ),
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                    }
                )

        results = pd.DataFrame(rows)

        if not results.empty:
            reject, q_values, _, _ = multipletests(
                results["p_value"],
                alpha=0.05,
                method="fdr_bh",
            )

            results["q_value"] = q_values
            results["significant_fdr"] = reject

        return results

    # ------------------------------------------------------------------
    # Outcome extraction
    # ------------------------------------------------------------------

    def _extract_event_indicator(self) -> pd.Series:

        if not isinstance(self.y, pd.DataFrame):
            raise TypeError(
                "y must be a pandas DataFrame."
            )

        if "event" not in self.y.columns:
            raise ValueError(
                "y must contain an 'event' column."
            )

        return (
            self.y["event"]
            .reset_index(drop=True)
            .astype(int)
        )
    # ------------------------------------------------------------------
    # Multiple-testing correction
    # ------------------------------------------------------------------

    @staticmethod
    def adjust_p_values(
        results: pd.DataFrame,
        p_column: str = "p_value",
        method: str = "fdr_bh",
    ) -> pd.DataFrame:

        """
        Apply Benjamini-Hochberg FDR correction.

        Kept separate from the individual analyses so that
        correction can be applied across a clearly defined
        family of hypotheses.
        """

        if results.empty:
            return results.copy()

        from statsmodels.stats.multitest import (
            multipletests,
        )

        adjusted = results.copy()

        valid = adjusted[p_column].notna()

        adjusted.loc[valid, "p_value_adjusted"] = (
            multipletests(
                adjusted.loc[valid, p_column],
                method=method,
            )[1]
        )

        return adjusted