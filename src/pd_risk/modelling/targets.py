from pathlib import Path

import pandas as pd


class SurvivalTargetBuilder:
    """
    Build survival analysis targets.

    Creates:
    - event indicator
    - time-to-event

    Outcome definitions are controlled
    through configuration.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
    ):

        self.df = df.copy()

        self.config = config

        self.targets = None


    def build(self):

        self._define_event()

        self._calculate_time()

        self._build_target_dataframe()

        self._validate()

        return self.targets

    def _define_event(self):
        """
        Define PD event and competing-event status.

        event:
            1 = PD
            0 = no PD before end of observation

        event_type:
            0 = censored
            1 = PD
            2 = death before PD
        """

        method = (
            self.config["outcome"]
            ["definition"]
            ["method"]
        )

        if method == "persistent_pd":

            pd_event = (
                self.df["pd_case_type"]
                == "persistent_pd"
            )

        elif method == "any_pd":

            pd_event = (
                self.df["pd_event"]
                .astype(bool)
            )

        elif method == "confirmed_pd":

            pd_event = (
                self.df["pd_case_type"]
                .isin(
                    [
                        "persistent_pd",
                        "pd_end_of_followup",
                    ]
                )
            )

        else:

            raise ValueError(
                f"Unknown PD definition: {method}"
            )

        pd_event = pd_event.astype(int)

        death_before_pd = (
            self.df["death_event"].astype(bool)
            & (pd_event == 0)
        )

        self.df["event"] = pd_event

        self.df["event_type"] = 0

        self.df.loc[
            pd_event == 1,
            "event_type"
        ] = 1

        self.df.loc[
            death_before_pd,
            "event_type"
        ] = 2

    def _calculate_time(self):
        """
        Calculate survival time.

        For participants without PD:
        - death before PD -> death date
        - otherwise -> last follow-up date
        """

        baseline = pd.to_datetime(
            self.df["baseline_date"]
        )

        pd_event_date = pd.to_datetime(
            self.df["pd_event_date"]
        )

        last_visit = pd.to_datetime(
            self.df["last_visit_date"]
        )

        death_date = pd.to_datetime(
            self.df["death_date"]
        )

        end_date = pd.Series(
            pd.NaT,
            index=self.df.index,
            dtype="datetime64[ns]",
        )

        # PD event
        pd_mask = self.df["event"] == 1

        end_date.loc[pd_mask] = (
            pd_event_date.loc[pd_mask]
        )

        # No PD: death or censoring
        no_pd_mask = ~pd_mask

        death_mask = (
            no_pd_mask
            & death_date.notna()
        )

        end_date.loc[death_mask] = (
            death_date.loc[death_mask]
        )

        censor_mask = (
            no_pd_mask
            & ~death_mask
        )

        end_date.loc[censor_mask] = (
            last_visit.loc[censor_mask]
        )

        self.df["time_to_event"] = (
            (
                end_date
                - baseline
            ).dt.days / 365.25
        )

    def _build_target_dataframe(self):

        self.targets = (
            self.df[
                [
                    "PATNO",
                    "time_to_event",
                    "event",
                    "event_type",
                ]
            ]
            .copy()
        )

    def _validate(self):

        if self.targets["event"].isna().any():
            raise ValueError(
                "Missing event values"
            )

        if self.targets["event_type"].isna().any():
            raise ValueError(
                "Missing event_type values"
            )

        if self.targets["time_to_event"].isna().any():
            raise ValueError(
                "Missing survival times"
            )

        if (
            self.targets["time_to_event"] < 0
        ).any():
            raise ValueError(
                "Negative survival times detected"
            )

        valid_event_types = {0, 1, 2}

        if not set(
            self.targets["event_type"].unique()
        ).issubset(valid_event_types):
            raise ValueError(
                "Invalid event_type values detected"
            )

        print(
            "Survival target validation passed"
        )

        print(
            self.targets["event_type"]
            .value_counts()
            .sort_index()
        )