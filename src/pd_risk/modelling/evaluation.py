import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sksurv.metrics import concordance_index_ipcw
from surpyval import gray_test


class SurvivalEvaluator:
    """
    Evaluation of competing-risk prediction models.

    Scientific estimand
    -------------------
    Two-year cumulative incidence of Parkinson's disease (PD),
    with death before PD treated as a competing event.

    Event types
    -----------
    0 : censored
    1 : PD
    2 : death before PD
    """

    PD_EVENT = 1
    DEATH_EVENT = 2
    CENSORED = 0

    def __init__(
        self,
        cause=1,
        n_risk_groups=4,
    ):

        if cause != self.PD_EVENT:
            raise ValueError(
                "This evaluator is configured for "
                "PD as the event of interest "
                "(event_type=1)."
            )

        self.cause = cause
        self.n_risk_groups = n_risk_groups

        self.risk_groups = None
        self.gray_test_result = None
        self.calibration = None
        self.cif_results = None

    # =========================================================
    # Target validation
    # =========================================================

    def validate_targets(
        self,
        y,
    ):
        """
        Validate competing-risk survival targets.
        """

        required = {
            "time_to_event",
            "event_type",
        }

        missing = (
            required
            - set(y.columns)
        )

        if missing:
            raise ValueError(
                "Missing required target columns: "
                f"{sorted(missing)}"
            )

        if y.empty:
            raise ValueError(
                "Survival target dataframe is empty."
            )

        if (
            y["time_to_event"]
            .isna()
            .any()
        ):
            raise ValueError(
                "time_to_event contains missing values."
            )

        if (
            y["time_to_event"] < 0
        ).any():
            raise ValueError(
                "time_to_event cannot be negative."
            )

        observed_types = set(
            y["event_type"].unique()
        )

        unexpected = (
            observed_types
            - {
                self.CENSORED,
                self.PD_EVENT,
                self.DEATH_EVENT,
            }
        )

        if unexpected:
            raise ValueError(
                "Unexpected event_type values: "
                f"{sorted(unexpected)}."
            )

        return True

    # =========================================================
    # Complete evaluation
    # =========================================================

    def evaluate(
        self,
        y_train,
        y_test,
        risk_scores,
        pd_cif,
        horizon,
    ):
        """
        Perform the complete competing-risk evaluation.
        """

        self.validate_targets(y_train)
        self.validate_targets(y_test)

        risk_scores = np.asarray(
            risk_scores,
            dtype=float,
        )

        pd_cif = np.asarray(
            pd_cif,
            dtype=float,
        )

        if len(pd_cif) != len(y_test):
            raise ValueError(
                "pd_cif and y_test "
                "must have the same length."
            )

        # -----------------------------------------------------
        # Primary metrics
        # -----------------------------------------------------

        uno_c_index = self._uno_c_index(
            y_train=y_train,
            y_test=y_test,
            risk_scores=risk_scores,
        )

        brier_score = self._brier_score(
            y_train=y_train,
            y_test=y_test,
            predicted_risk=pd_cif,
            horizon=horizon,
        )

        null_brier_score = self._null_brier_score(
            y_train=y_train,
            y_test=y_test,
            horizon=horizon,
        )

        brier_skill_score = (
            self._brier_skill_score(
                brier_score=brier_score,
                null_brier_score=null_brier_score,
            )
        )

        # -----------------------------------------------------
        # Null prediction
        # -----------------------------------------------------

        null_risk = self._null_risk(
            y_train=y_train,
            horizon=horizon,
        )

        return {
            "uno_c_index":
                float(uno_c_index),

            "brier_score":
                float(brier_score),

            "null_brier_score":
                float(null_brier_score),

            "brier_skill_score":
                float(brier_skill_score),

            "null_pd_cif":
                float(null_risk),

            "horizon":
                float(horizon),

            "cause":
                int(self.cause),
        }

    # =========================================================
    # Uno C-index
    # =========================================================

    def _uno_c_index(
        self,
        y_train,
        y_test,
        risk_scores,
    ):
        """
        Uno's concordance index for PD as the event of interest.

        Death before PD is treated as a competing event rather
        than as a PD event.
        """

        train_event = (
            y_train["event_type"]
            == self.cause
        )

        test_event = (
            y_test["event_type"]
            == self.cause
        )

        train_structured = np.array(
            list(
                zip(
                    train_event.astype(bool),
                    y_train["time_to_event"],
                )
            ),
            dtype=[
                ("event", "?"),
                ("time", "f8"),
            ],
        )

        test_structured = np.array(
            list(
                zip(
                    test_event.astype(bool),
                    y_test["time_to_event"],
                )
            ),
            dtype=[
                ("event", "?"),
                ("time", "f8"),
            ],
        )

        result = concordance_index_ipcw(
            train_structured,
            test_structured,
            risk_scores,
        )

        return float(
            result[0]
        )

    # =========================================================
    # Null PD cumulative incidence
    # =========================================================

    def _null_risk(
        self,
        y_train,
        horizon,
    ):
        """
        Estimate population-level PD cumulative incidence
        at the evaluation horizon.

        Uses the Aalen-Johansen estimator.
        """

        return float(
            self.build_cumulative_incidence(
                y=y_train,
                horizon=horizon,
                cause=self.cause,
            )
        )

    # =========================================================
    # Brier score
    # =========================================================

    def _brier_score(
        self,
        y_train,
        y_test,
        predicted_risk,
        horizon,
    ):
        """
        IPCW Brier score for PD cumulative incidence.

        Individuals who experience PD before the horizon:
            outcome = 1

        Individuals who are alive and PD-free at the horizon:
            outcome = 0

        Individuals who die before PD:
            outcome = 0

        Individuals censored before the horizon:
            contribution = 0

        The denominator is the full test-set size, as required
        by the Graf IPCW estimator.
        """

        predicted_risk = np.asarray(
            predicted_risk,
            dtype=float,
        )

        if len(predicted_risk) != len(y_test):
            raise ValueError(
                "predicted_risk and y_test "
                "must have the same length."
            )

        censoring_survival = (
            self._censoring_survival_function(
                y_train
            )
        )

        test_time = (
            y_test["time_to_event"]
            .to_numpy(dtype=float)
        )

        test_event = (
            y_test["event_type"]
            .to_numpy()
        )

        # -----------------------------------------------------
        # PD event before horizon
        # -----------------------------------------------------

        pd_before_horizon = (
            (test_time <= horizon)
            & (test_event == self.cause)
        )

        # -----------------------------------------------------
        # Known non-PD status at horizon
        #
        # This includes:
        #   - no event by horizon
        #   - death before horizon
        #
        # These are observed to be non-PD by horizon.
        # -----------------------------------------------------

        known_non_pd = (
            (
                test_time >= horizon
            )
            |
            (
                (test_time < horizon)
                & (test_event == self.DEATH_EVENT)
            )
        )

        # -----------------------------------------------------
        # IPCW weights
        # -----------------------------------------------------

        weights = np.zeros(
            len(y_test),
            dtype=float,
        )

        if pd_before_horizon.any():

            weights[
                pd_before_horizon
            ] = (
                1
                / censoring_survival(
                    test_time[
                        pd_before_horizon
                    ]
                )
            )

        if known_non_pd.any():

            weights[
                known_non_pd
            ] = (
                1
                / censoring_survival(
                    np.full(
                        known_non_pd.sum(),
                        horizon,
                    )
                )
            )

        if np.any(
            weights <= 0
        ):
            invalid = (
                (weights < 0)
                | ~np.isfinite(weights)
            )

            if invalid.any():
                return np.nan

        observed_risk = np.zeros(
            len(y_test),
            dtype=float,
        )

        observed_risk[
            pd_before_horizon
        ] = 1.0

        squared_error = (
            observed_risk
            - predicted_risk
        ) ** 2

        return float(
            np.sum(
                weights
                * squared_error
            )
            / len(y_test)
        )

    # =========================================================
    # Null Brier score
    # =========================================================

    def _null_brier_score(
        self,
        y_train,
        y_test,
        horizon,
    ):
        """
        Brier score for a null model that predicts the same
        population PD cumulative incidence for everybody.
        """

        null_risk = self._null_risk(
            y_train=y_train,
            horizon=horizon,
        )

        null_predictions = np.full(
            len(y_test),
            null_risk,
            dtype=float,
        )

        return self._brier_score(
            y_train=y_train,
            y_test=y_test,
            predicted_risk=null_predictions,
            horizon=horizon,
        )

    # =========================================================
    # Brier skill score
    # =========================================================

    def _brier_skill_score(
        self,
        brier_score,
        null_brier_score,
    ):
        """
        Relative improvement over the null model.
        """

        if (
            null_brier_score <= 0
            or not np.isfinite(
                null_brier_score
            )
        ):
            return np.nan

        return float(
            1
            - (
                brier_score
                / null_brier_score
            )
        )

    # =========================================================
    # Censoring distribution
    # =========================================================

    def _censoring_survival_function(
        self,
        y,
    ):
        """
        Estimate the survival function of the censoring
        distribution.

        event_type == 0 is censoring.

        PD and death are observed outcomes and therefore
        are not treated as censoring here.
        """

        times = (
            y["time_to_event"]
            .to_numpy(dtype=float)
        )

        censoring_event = (
            y["event_type"]
            == self.CENSORED
        ).astype(int)

        unique_times = np.sort(
            np.unique(times)
        )

        survival = []
        current_survival = 1.0

        for t in unique_times:

            at_risk = (
                times >= t
            ).sum()

            censoring_events = (
                (times == t)
                & (censoring_event == 1)
            ).sum()

            if at_risk > 0:

                current_survival *= (
                    1
                    - (
                        censoring_events
                        / at_risk
                    )
                )

            survival.append(
                current_survival
            )

        unique_times = np.asarray(
            unique_times
        )

        survival = np.asarray(
            survival,
            dtype=float,
        )

        def predict(query_times):

            query_times = np.asarray(
                query_times,
                dtype=float,
            )

            indices = np.searchsorted(
                unique_times,
                query_times,
                side="right",
            ) - 1

            result = np.ones(
                len(query_times),
                dtype=float,
            )

            valid = (
                indices >= 0
            )

            result[valid] = (
                survival[
                    indices[valid]
                ]
            )

            if np.any(
                result <= 0
            ):
                raise ValueError(
                    "Censoring survival probability "
                    "reached zero."
                )

            return result

        return predict

    # =========================================================
    # Aalen-Johansen CIF
    # =========================================================

    def build_cumulative_incidence(
        self,
        y,
        horizon,
        cause,
    ):
        """
        Estimate the cumulative incidence of a competing event
        at a fixed time horizon using the Aalen-Johansen estimator.

        Parameters
        ----------
        y : pandas.DataFrame
            Survival targets containing:
            - time_to_event
            - event_type

        horizon : float
            Time horizon in years.

        cause : int
            Event type for which cumulative incidence is estimated.
            1 = Parkinson's disease
            2 = death before Parkinson's disease

        Returns
        -------
        float
            Estimated cumulative incidence of the specified cause
            by the given horizon.
        """

        time = (
            y["time_to_event"]
            .to_numpy(dtype=float)
        )

        event_type = (
            y["event_type"]
            .to_numpy()
        )

        order = np.argsort(time)

        time = time[order]
        event_type = event_type[order]

        survival = 1.0
        cif = 0.0

        unique_times = np.unique(
            time[
                time <= horizon
            ]
        )

        for t in unique_times:

            at_risk = (
                time >= t
            ).sum()

            if at_risk == 0:
                continue

            cause_events = (
                (time == t)
                & (event_type == cause)
            ).sum()

            total_events = (
                (time == t)
                & (
                    event_type != self.CENSORED
                )
            ).sum()

            cif += (
                survival
                * (
                    cause_events
                    / at_risk
                )
            )

            survival *= (
                1
                - (
                    total_events
                    / at_risk
                )
            )

        return cif

    # =========================================================
    # CIF curve
    # =========================================================

    def build_cumulative_incidence_curve(
        self,
        y,
        cause,
        horizon=None,
    ):
        """
        Return an Aalen-Johansen cumulative-incidence curve.
        """

        if horizon is None:
            horizon = float(
                y["time_to_event"].max()
            )

        time = (
            y["time_to_event"]
            .to_numpy(dtype=float)
        )

        event_type = (
            y["event_type"]
            .to_numpy()
        )

        unique_times = np.sort(
            np.unique(
                time[
                    time <= horizon
                ]
            )
        )

        survival = 1.0
        cif = 0.0

        curve_time = [0.0]
        curve_cif = [0.0]

        for t in unique_times:

            at_risk = (
                time >= t
            ).sum()

            if at_risk == 0:
                continue

            cause_events = (
                (time == t)
                & (event_type == cause)
            ).sum()

            total_events = (
                (time == t)
                & (
                    event_type != self.CENSORED
                )
            ).sum()

            cif += (
                survival
                * (
                    cause_events
                    / at_risk
                )
            )

            survival *= (
                1
                - (
                    total_events
                    / at_risk
                )
            )

            curve_time.append(
                float(t)
            )

            curve_cif.append(
                float(cif)
            )

        return pd.DataFrame(
            {
                "time": curve_time,
                "cumulative_incidence":
                    curve_cif,
            }
        )

    # =========================================================
    # Risk groups
    # =========================================================

    def assign_risk_groups(
        self,
        predictions,
        n_groups=None,
    ):
        """
        Assign patients to predicted-risk quantile groups.
        """

        if n_groups is None:
            n_groups = self.n_risk_groups

        predictions = predictions.copy()

        predictions[
            "risk_group"
        ] = (
            pd.qcut(
                predictions[
                    "pd_cif"
                ],
                q=n_groups,
                labels=False,
                duplicates="drop",
            )
            + 1
        )

        return predictions

    # =========================================================
    # Risk-group summary
    # =========================================================

    def build_risk_groups(
        self,
        predictions,
        horizon,
    ):
        """
        Summarise predicted-risk groups at the evaluation
        horizon.
        """

        required = {
            "time_to_event",
            "event_type",
            "pd_cif",
            "risk_group",
        }

        missing = (
            required
            - set(predictions.columns)
        )

        if missing:
            raise ValueError(
                "Missing columns for risk-group "
                f"summary: {sorted(missing)}"
            )

        rows = []

        for group in sorted(
            predictions["risk_group"]
            .dropna()
            .unique()
        ):

            group_data = predictions[
                predictions["risk_group"]
                == group
            ]

            pd_cif = (
                self.build_cumulative_incidence(
                    group_data,
                    horizon=horizon,
                    cause=self.PD_EVENT,
                )
            )

            death_cif = (
                self.build_cumulative_incidence(
                    group_data,
                    horizon=horizon,
                    cause=self.DEATH_EVENT,
                )
            )

            rows.append(
                {
                    "risk_group":
                        int(group),

                    "n":
                        len(group_data),

                    "pd_events":
                        int(
                            (
                                group_data[
                                    "event_type"
                                ]
                                == self.PD_EVENT
                            ).sum()
                        ),

                    "death_events":
                        int(
                            (
                                group_data[
                                    "event_type"
                                ]
                                == self.DEATH_EVENT
                            ).sum()
                        ),

                    "censored":
                        int(
                            (
                                group_data[
                                    "event_type"
                                ]
                                == self.CENSORED
                            ).sum()
                        ),

                    "mean_pd_cif":
                        float(
                            group_data[
                                "pd_cif"
                            ].mean()
                        ),

                    "median_pd_cif":
                        float(
                            group_data[
                                "pd_cif"
                            ].median()
                        ),

                    "observed_pd_cif":
                        float(pd_cif),

                    "observed_death_cif":
                        float(death_cif),

                    "horizon":
                        float(horizon),
                }
            )

        self.risk_groups = pd.DataFrame(
            rows
        )

        return self.risk_groups

    # =========================================================
    # Gray's test
    # =========================================================

    def build_gray_test(
        self,
        predictions,
        cause=None,
    ):
        """
        Compare PD cumulative incidence functions across
        predicted-risk groups using Gray's test.
        """

        if cause is None:
            cause = self.cause

        if cause != self.PD_EVENT:
            raise ValueError(
                "Gray's test is configured for "
                "PD (event_type=1)."
            )

        required = {
            "time_to_event",
            "event_type",
            "risk_group",
        }

        missing = (
            required
            - set(predictions.columns)
        )

        if missing:
            raise ValueError(
                "Missing columns for Gray's test: "
                f"{sorted(missing)}"
            )

        groups = (
            predictions[
                "risk_group"
            ].to_numpy()
        )

        if len(
            np.unique(groups)
        ) < 2:
            raise ValueError(
                "Gray's test requires at least "
                "two risk groups."
            )

        result = gray_test(
            predictions[
                "time_to_event"
            ].to_numpy(),

            predictions[
                "event_type"
            ].to_numpy(),

            groups,

            cause=cause,
        )

        self.gray_test_result = {
            "statistic":
                float(result.statistic),

            "degrees_of_freedom":
                int(result.df),

            "p_value":
                float(result.p_value),

            "cause":
                int(cause),

            "n_groups":
                int(
                    len(
                        np.unique(groups)
                    )
                ),
        }

        return self.gray_test_result

    # =========================================================
    # Calibration
    # =========================================================

    def build_calibration(
        self,
        predictions,
        horizon,
        n_groups=None,
    ):
        """
        Compare mean predicted PD CIF with observed PD CIF
        within predicted-risk groups.
        """

        if n_groups is None:
            n_groups = self.n_risk_groups

        df = self.assign_risk_groups(
            predictions,
            n_groups=n_groups,
        )

        rows = []

        for group in sorted(
            df["risk_group"].unique()
        ):

            group_data = df[
                df["risk_group"]
                == group
            ]

            observed_pd_cif = (
                self.build_cumulative_incidence(
                    group_data,
                    horizon=horizon,
                    cause=self.PD_EVENT,
                )
            )

            rows.append(
                {
                    "risk_group":
                        int(group),

                    "n":
                        len(group_data),

                    "mean_pd_cif":
                        float(
                            group_data[
                                "pd_cif"
                            ].mean()
                        ),

                    "observed_pd_cif":
                        float(
                            observed_pd_cif
                        ),

                    "horizon":
                        float(horizon),
                }
            )

        self.calibration = pd.DataFrame(
            rows
        )

        return self.calibration

    # =========================================================
    # Calibration plot
    # =========================================================

    def plot_calibration(
        self,
        calibration,
        horizon,
        ax=None,
    ):
        """
        Plot predicted versus observed PD CIF.
        """

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(7, 6)
            )
        else:
            fig = ax.get_figure()

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            label="Perfect calibration",
        )

        ax.scatter(
            calibration[
                "mean_pd_cif"
            ],
            calibration[
                "observed_pd_cif"
            ],
            s=50,
            label="Risk groups",
        )

        ax.set_xlabel(
            f"Predicted {horizon:g}-year PD cumulative incidence"
        )

        ax.set_ylabel(
            f"Observed {horizon:g}-year PD cumulative incidence"
        )

        ax.set_title(
            f"{horizon:g}-year PD calibration"
        )

        ax.set_xlim(
            0,
            1,
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.legend()

        return fig

    # =========================================================
    # Risk-group CIF plot
    # =========================================================

    def plot_cumulative_incidence(
        self,
        predictions,
        horizon,
        ax=None,
    ):
        """
        Plot observed PD cumulative incidence by predicted-risk
        group.
        """

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(8, 6)
            )
        else:
            fig = ax.get_figure()

        for group in sorted(
            predictions[
                "risk_group"
            ].unique()
        ):

            group_data = predictions[
                predictions[
                    "risk_group"
                ] == group
            ]

            curve = (
                self.build_cumulative_incidence_curve(
                    group_data,
                    cause=self.PD_EVENT,
                    horizon=horizon,
                )
            )

            ax.step(
                curve["time"],
                curve[
                    "cumulative_incidence"
                ],
                where="post",
                label=f"Risk group {group}",
            )

        ax.set_xlabel(
            "Time (years)"
        )

        ax.set_ylabel(
            "PD cumulative incidence"
        )

        ax.set_title(
            f"PD cumulative incidence by predicted-risk group"
        )

        ax.set_xlim(
            0,
            horizon,
        )

        ax.set_ylim(
            0,
            1,
        )

        ax.legend()

        return fig