import numpy as np
import pandas as pd

from lifelines import CoxPHFitter


class CauseSpecificCoxModel:
    """
    Cause-specific Cox proportional hazards model
    for competing-risk survival analysis.

    Event types
    -----------
    0 : censored
    1 : Parkinson's disease (PD)
    2 : death before PD

    Two Cox models are fitted:

    - PD cause-specific hazard
    - Death cause-specific hazard

    The two cause-specific hazards are combined to estimate
    cumulative incidence functions (CIFs).
    """

    PD_EVENT = 1
    DEATH_EVENT = 2
    CENSORED = 0

    VALID_EVENT_TYPES = {
        CENSORED,
        PD_EVENT,
        DEATH_EVENT,
    }

    def __init__(
        self,
        penalizer=0.0,
    ):
        self.penalizer = penalizer

        self.pd_model = CoxPHFitter(
            penalizer=penalizer
        )

        self.death_model = CoxPHFitter(
            penalizer=penalizer
        )

        self.is_fitted = False

        self._time_grid = None
        self._pd_hazard_jumps = None
        self._death_hazard_jumps = None

    # =========================================================
    # FIT
    # =========================================================

    def fit(
        self,
        X,
        y,
    ):
        """
        Fit the two cause-specific Cox models.

        Parameters
        ----------
        X : pandas.DataFrame
            Preprocessed feature matrix.

        y : pandas.DataFrame
            Must contain:

            - time_to_event
            - event_type

        Returns
        -------
        CauseSpecificCoxModel
            Fitted model.
        """

        self._validate_inputs(
            X,
            y,
        )

        event_type = (
            y["event_type"]
            .to_numpy()
        )

        time = (
            y["time_to_event"]
            .to_numpy()
        )

        # -----------------------------------------------------
        # PD cause-specific model
        # -----------------------------------------------------
        #
        # PD      -> event
        # Death   -> censored
        # Censor  -> censored

        pd_df = X.copy()

        pd_df["time_to_event"] = time

        pd_df["event"] = (
            event_type == self.PD_EVENT
        ).astype(int)

        # -----------------------------------------------------
        # Death cause-specific model
        # -----------------------------------------------------
        #
        # Death   -> event
        # PD      -> censored
        # Censor  -> censored

        death_df = X.copy()

        death_df["time_to_event"] = time

        death_df["event"] = (
            event_type == self.DEATH_EVENT
        ).astype(int)

        # -----------------------------------------------------
        # Fit models
        # -----------------------------------------------------

        self.pd_model.fit(
            pd_df,
            duration_col="time_to_event",
            event_col="event",
        )

        self.death_model.fit(
            death_df,
            duration_col="time_to_event",
            event_col="event",
        )

        # -----------------------------------------------------
        # Build common baseline hazard representation
        # -----------------------------------------------------

        self._build_baseline_hazard_jumps()

        self.is_fitted = True

        return self

    # =========================================================
    # VALIDATION
    # =========================================================

    def _validate_inputs(
        self,
        X,
        y,
    ):
        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame."
            )

        if not isinstance(
            y,
            pd.DataFrame,
        ):
            raise TypeError(
                "y must be a pandas DataFrame."
            )

        required_columns = {
            "time_to_event",
            "event_type",
        }

        missing = (
            required_columns
            - set(y.columns)
        )

        if missing:
            raise ValueError(
                "y is missing required columns: "
                f"{sorted(missing)}"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same "
                "number of observations."
            )

        if X.index.equals(y.index) is False:
            raise ValueError(
                "X and y must have aligned indices."
            )

        event_types = set(
            np.unique(
                y["event_type"]
            )
        )

        unexpected = (
            event_types
            - self.VALID_EVENT_TYPES
        )

        if unexpected:
            raise ValueError(
                "Unexpected event_type values: "
                f"{sorted(unexpected)}. "
                "Expected 0 (censored), "
                "1 (PD), or 2 (death)."
            )

        if y["time_to_event"].isna().any():
            raise ValueError(
                "time_to_event contains missing values."
            )

        if (
            y["time_to_event"] < 0
        ).any():
            raise ValueError(
                "time_to_event cannot contain "
                "negative values."
            )

        if not np.isfinite(
            y["time_to_event"].to_numpy()
        ).all():
            raise ValueError(
                "time_to_event contains non-finite values."
            )

    # =========================================================
    # BASELINE HAZARD REPRESENTATION
    # =========================================================

    def _build_baseline_hazard_jumps(self):
        """
        Construct baseline cumulative-hazard jumps for
        the PD and death models on a common event-time grid.

        The Cox model gives us cumulative baseline hazards.
        For CIF calculation we need their increments.
        """

        pd_cum = (
            self.pd_model
            .baseline_cumulative_hazard_
            .iloc[:, 0]
        )

        death_cum = (
            self.death_model
            .baseline_cumulative_hazard_
            .iloc[:, 0]
        )

        pd_times = (
            pd_cum.index.to_numpy(
                dtype=float
            )
        )

        death_times = (
            death_cum.index.to_numpy(
                dtype=float
            )
        )

        common_times = np.unique(
            np.concatenate(
                [
                    pd_times,
                    death_times,
                ]
            )
        )

        common_times.sort()

        # -----------------------------------------------------
        # Map cumulative hazards onto common event times.
        #
        # Forward-fill cumulative hazard values. A cause has
        # zero increment at times where it has no event.
        # -----------------------------------------------------

        pd_cum_values = self._step_function(
            pd_times,
            pd_cum.to_numpy(),
            common_times,
        )

        death_cum_values = self._step_function(
            death_times,
            death_cum.to_numpy(),
            common_times,
        )

        pd_previous = np.concatenate(
            [
                np.array([0.0]),
                pd_cum_values[:-1],
            ]
        )

        death_previous = np.concatenate(
            [
                np.array([0.0]),
                death_cum_values[:-1],
            ]
        )

        self._time_grid = common_times

        self._pd_hazard_jumps = (
            pd_cum_values
            - pd_previous
        )

        self._death_hazard_jumps = (
            death_cum_values
            - death_previous
        )

    @staticmethod
    def _step_function(
        source_times,
        source_values,
        target_times,
    ):
        """
        Evaluate a right-continuous step function.

        Values remain constant between observed event times.
        """

        indices = np.searchsorted(
            source_times,
            target_times,
            side="right",
        ) - 1

        values = np.zeros(
            len(target_times),
            dtype=float,
        )

        valid = indices >= 0

        values[valid] = (
            source_values[
                indices[valid]
            ]
        )

        return values

    # =========================================================
    # PD RISK SCORE
    # =========================================================

    def predict_risk(
        self,
        X,
    ):
        """
        Return the PD cause-specific linear-risk score.

        This is a relative risk score, not a probability.

        Higher values indicate higher PD cause-specific hazard.
        """

        self._check_fitted()

        return (
            self.pd_model
            .predict_partial_hazard(X)
            .to_numpy()
        )

    # =========================================================
    # DEATH RISK SCORE
    # =========================================================

    def predict_death_risk(
        self,
        X,
    ):
        """
        Return the death cause-specific relative-risk score.
        """

        self._check_fitted()

        return (
            self.death_model
            .predict_partial_hazard(X)
            .to_numpy()
        )

    # =========================================================
    # PD CUMULATIVE INCIDENCE
    # =========================================================

    def predict_pd_cumulative_incidence(
        self,
        X,
        time,
    ):
        """
        Predict cumulative incidence of PD by `time`.

        Death before PD is treated as a competing event.

        Returns
        -------
        numpy.ndarray
            Probability of developing PD by `time`.
        """

        self._check_fitted()

        self._validate_prediction_time(
            time
        )

        pd_hazard_ratio = (
            self.predict_risk(X)
        )

        death_hazard_ratio = (
            self.predict_death_risk(X)
        )

        return self._calculate_cif(
            pd_hazard_ratio=pd_hazard_ratio,
            death_hazard_ratio=death_hazard_ratio,
            time=time,
            target_cause="pd",
        )

    # =========================================================
    # DEATH CUMULATIVE INCIDENCE
    # =========================================================

    def predict_death_cumulative_incidence(
        self,
        X,
        time,
    ):
        """
        Predict cumulative incidence of death by `time`.

        PD is treated as a competing event.
        """

        self._check_fitted()

        self._validate_prediction_time(
            time
        )

        pd_hazard_ratio = (
            self.predict_risk(X)
        )

        death_hazard_ratio = (
            self.predict_death_risk(X)
        )

        return self._calculate_cif(
            pd_hazard_ratio=pd_hazard_ratio,
            death_hazard_ratio=death_hazard_ratio,
            time=time,
            target_cause="death",
        )

    # =========================================================
    # OVERALL EVENT-FREE SURVIVAL
    # =========================================================

    def predict_survival(
        self,
        X,
        time,
    ):
        """
        Predict probability of remaining free from both
        PD and death through `time`.

        This is overall event-free survival.

        It is NOT the probability of remaining free from PD
        while ignoring competing death.
        """

        self._check_fitted()

        self._validate_prediction_time(
            time
        )

        pd_hazard_ratio = (
            self.predict_risk(X)
        )

        death_hazard_ratio = (
            self.predict_death_risk(X)
        )

        times = (
            self._time_grid[
                self._time_grid <= time
            ]
        )

        if len(times) == 0:
            return np.ones(
                len(X),
                dtype=float,
            )

        pd_jumps = (
            self._pd_hazard_jumps[
                self._time_grid <= time
            ]
        )

        death_jumps = (
            self._death_hazard_jumps[
                self._time_grid <= time
            ]
        )

        total_hazard = (
            pd_hazard_ratio[:, None]
            * pd_jumps[None, :]
            +
            death_hazard_ratio[:, None]
            * death_jumps[None, :]
        )

        cumulative_hazard = (
            total_hazard.sum(
                axis=1
            )
        )

        return np.exp(
            -cumulative_hazard
        )

    # =========================================================
    # CIF CALCULATION
    # =========================================================

    def _calculate_cif(
        self,
        pd_hazard_ratio,
        death_hazard_ratio,
        time,
        target_cause,
    ):
        """
        Calculate a cause-specific cumulative incidence function.

        Uses the estimated cause-specific baseline hazard jumps
        and combines them through the overall survival function.
        """

        mask = (
            self._time_grid <= time
        )

        times = (
            self._time_grid[mask]
        )

        if len(times) == 0:
            return np.zeros(
                len(pd_hazard_ratio),
                dtype=float,
            )

        pd_jumps = (
            self._pd_hazard_jumps[mask]
        )

        death_jumps = (
            self._death_hazard_jumps[mask]
        )

        n = len(
            pd_hazard_ratio
        )

        cumulative_incidence = (
            np.zeros(
                n,
                dtype=float,
            )
        )

        survival = (
            np.ones(
                n,
                dtype=float,
            )
        )

        for i in range(
            len(times)
        ):

            pd_increment = (
                pd_hazard_ratio
                * pd_jumps[i]
            )

            death_increment = (
                death_hazard_ratio
                * death_jumps[i]
            )

            total_increment = (
                pd_increment
                + death_increment
            )

            # Probability of an event occurring during
            # this small hazard interval.
            event_probability = (
                1.0
                - np.exp(
                    -total_increment
                )
            )

            target_increment = (
                pd_increment
                if target_cause == "pd"
                else death_increment
            )

            cause_fraction = np.divide(
                target_increment,
                total_increment,
                out=np.zeros_like(
                    target_increment
                ),
                where=(
                    total_increment > 0
                ),
            )

            cumulative_incidence += (
                survival
                * event_probability
                * cause_fraction
            )

            survival *= (
                np.exp(
                    -total_increment
                )
            )

        return cumulative_incidence

    # =========================================================
    # MODEL SUMMARY
    # =========================================================

    def summary(self):
        """
        Return summaries for both cause-specific Cox models.
        """

        self._check_fitted()

        return {
            "pd": self._cox_summary(
                self.pd_model
            ),
            "death": self._cox_summary(
                self.death_model
            ),
        }

    @staticmethod
    def _cox_summary(
        model,
    ):
        summary = (
            model.summary
            .copy()
        )

        summary["hazard_ratio"] = (
            np.exp(
                summary["coef"]
            )
        )

        summary["hazard_ratio_lower"] = (
            np.exp(
                summary["coef lower 95%"]
            )
        )

        summary["hazard_ratio_upper"] = (
            np.exp(
                summary["coef upper 95%"]
            )
        )

        return summary

    # =========================================================
    # INTERNAL VALIDATION
    # =========================================================

    def _check_fitted(self):
        if not self.is_fitted:
            raise RuntimeError(
                "CauseSpecificCoxModel must be fitted "
                "before prediction."
            )

    @staticmethod
    def _validate_prediction_time(
        time,
    ):
        if not np.isfinite(time):
            raise ValueError(
                "time must be finite."
            )

        if time <= 0:
            raise ValueError(
                "time must be greater than zero."
            )

