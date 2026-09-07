from __future__ import annotations

from pathlib import Path
import json
import yaml
import pandas as pd

from pd_risk.prediction.base import PredictionDatasetBuilder

from pd_risk.utils.features import (
    parse_feature_config
)

class PPMIPredictionDatasetBuilder(PredictionDatasetBuilder):
    """
    Build prediction datasets from PPMI data.

    Combines baseline predictor variables with participant-level
    outcomes for survival modelling.
    """

    def __init__(
        self,
        prediction_config: dict,
    ):

        self.config = prediction_config

        self.clinical_df = None
        self.outcomes_df = None
        self.index_df = None
        self.features_df = None
        self.feature_config = None

    
        self.dataset = None
        self.metadata = {}
        self.feature_domains = {}

        self.feature_columns = []
        self.continuous_features = []
        self.binary_features = []
        self.categorical_features = []


    def build(self):
        self._load_inputs()
        self._extract_index_visit()
        self._extract_features()
        self._merge_outcomes()
        self._validate_dataset()
        self._build_metadata()

    def save(self):
        """
        Save prediction dataset and metadata.

        Output files are controlled by the
        prediction configuration YAML.
        """

        if self.dataset is None:
            raise RuntimeError(
                "Prediction dataset has not been built."
            )


        output_config = (
            self.config["output"]
        )


        output_dir = Path(
            output_config["directory"]
        )


        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        # -------------------------
        # Prediction dataset
        # -------------------------

        self.dataset.to_csv(
            output_dir
            /
            output_config["dataset_file"],
            index=False,
        )


        # -------------------------
        # Metadata
        # -------------------------

        with open(
            output_dir
            /
            output_config["metadata_file"],
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.metadata,
                f,
                indent=4,
            )


        print(
            f"Saved prediction dataset to {output_dir}"
        )

    def summary():
        pass

    def index_summary(self):

        if self.index_df is None:
            raise RuntimeError(
                "Index visit has not been extracted."
            )

        print("\nIndex visit summary\n")

        print(
            f"Rows: {len(self.index_df)}"
        )

        print(
            f"Participants: "
            f"{self.index_df['PATNO'].nunique()}"
        )

        print(
            self.index_df["EVENT_ID"]
            .value_counts()
        )

    def _load_inputs(self):
        """
        Load all input files required for prediction dataset creation.

        Loads:
        - modality data
        - outcome table
        - feature configuration
        """

        # -------------------------
        # Load clinical modality
        # -------------------------

        clinical_config = (
            self.config["modalities"]["clinical"]
        )

        if clinical_config["enabled"]:

            self.clinical_df = pd.read_csv(
                clinical_config["file"]
            )

        # -------------------------
        # Load outcomes
        # -------------------------

        outcome_file = (
            self.config["inputs"]["outcomes"]["file"]
        )

        self.outcomes_df = pd.read_csv(
            outcome_file
        )

        # -------------------------
        # Load feature configuration
        # -------------------------

        feature_file = (
            clinical_config["features"]
        )

        with open(
            feature_file,
            "r",
            encoding="utf-8",
        ) as f:

            self.feature_config = yaml.safe_load(f)

    def _extract_index_visit(self):
        """
        Extract the prediction index visit.

        The index visit represents the information available
        at the time prediction is made.

        For the current PPMI experiment:
            index event = BL (baseline)

        Creates:
            self.index_df

        Validation:
        - every participant should have an index visit
        - one row per participant
        - no duplicate participants
        """

        index_config = self.config["index"]["event"]

        index_column = index_config["column"]
        index_value = index_config["value"]


        index_df = (
            self.clinical_df[
                self.clinical_df[index_column] == index_value
            ]
            .copy()
        )


        # -------------------------
        # Validation
        # -------------------------

        n_participants = (
            self.clinical_df["PATNO"]
            .nunique()
        )

        n_index_participants = (
            index_df["PATNO"]
            .nunique()
        )


        duplicates = (
            index_df["PATNO"]
            .duplicated()
            .sum()
        )


        if n_participants != n_index_participants:

            raise ValueError(
                f"Missing index visits. "
                f"Expected {n_participants}, "
                f"found {n_index_participants}."
            )


        if duplicates > 0:

            raise ValueError(
                f"Duplicate index visits found: "
                f"{duplicates}"
            )


        self.index_df = index_df

    def _extract_features(self):
        """
        Extract baseline predictor variables according to
        feature configuration.

        The feature YAML separates:
        - identifiers
        - labels
        - predictor domains

        Only variables under `features` are used as predictors.
        """

        # -------------------------
        # Extract features
        # -------------------------

        parsed_features = (
            parse_feature_config(
                self.feature_config
            )
        )

        self.feature_columns = (
            parsed_features["all"]
        )

        self.continuous_features = (
            parsed_features["continuous"]
        )

        self.binary_features = (
            parsed_features["binary"]
        )

        self.categorical_features = (
            parsed_features["categorical"]
        )


        # -------------------------
        # Validate columns exist
        # -------------------------

        missing_columns = [
            col
            for col in self.feature_columns
            if col not in self.index_df.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing feature columns: "
                f"{missing_columns}"
            )


        # -------------------------
        # Add participant identifier
        # -------------------------

        identifier_columns = (
            self.feature_config["identifiers"]
        )


        selected_columns = (
            identifier_columns
            +
            self.feature_columns
        )


        # -------------------------
        # Create feature dataframe
        # -------------------------

        self.features_df = (
            self.index_df[
                selected_columns
            ]
            .copy()
        )


        # Store feature structure
        self.feature_domains = (
            self.feature_config["features"]
        )

    def _merge_outcomes(self):
        """
        Merge baseline clinical features with
        participant-level outcomes.

        The merge is performed at participant level
        using PATNO.
        """

        if self.features_df is None:
            raise RuntimeError(
                "Features have not been extracted."
            )

        if self.outcomes_df is None:
            raise RuntimeError(
                "Outcomes have not been loaded."
            )


        self.dataset = (
            self.features_df
            .merge(
                self.outcomes_df,
                on="PATNO",
                how="inner",
                validate="one_to_one",
            )
        )

    def _validate_dataset(self):
        """
        Validate final prediction dataset.

        Checks:
        - one row per participant
        - required outcome columns exist
        - basic event counts
        """

        if self.dataset is None:
            raise RuntimeError(
                "Prediction dataset has not been created."
            )


        # -------------------------
        # Participant uniqueness
        # -------------------------

        n_rows = len(self.dataset)

        n_participants = (
            self.dataset["PATNO"]
            .nunique()
        )


        if n_rows != n_participants:
            raise ValueError(
                "Prediction dataset contains duplicated participants."
            )


        # -------------------------
        # Required outcome columns
        # -------------------------

        required_columns = [
            "pd_event",
            "death_event",
        ]


        missing_columns = [
            col
            for col in required_columns
            if col not in self.dataset.columns
        ]


        if missing_columns:

            raise ValueError(
                "Missing required outcome columns: "
                f"{missing_columns}"
            )


        # -------------------------
        # Store validation metadata
        # -------------------------

        self.metadata["validation"] = {

            "n_rows":
                n_rows,

            "n_participants":
                n_participants,

            "pd_events":
                int(
                    self.dataset["pd_event"]
                    .sum()
                ),

            "deaths":
                int(
                    self.dataset["death_event"]
                    .sum()
                ),
        }


        print("\nDataset validation")

        print(
            f"Rows: {n_rows}"
        )

        print(
            f"Participants: {n_participants}"
        )

        print(
            f"PD events: "
            f"{self.metadata['validation']['pd_events']}"
        )

        print(
            f"Deaths: "
            f"{self.metadata['validation']['deaths']}"
        )


    def _build_metadata(self):
        """
        Build metadata describing the prediction dataset.

        Includes:
        - experiment configuration
        - index definition
        - feature information
        - outcome summary
        - validation information
        """

        self.metadata.update(

            {

                # -------------------------
                # Experiment
                # -------------------------

                "experiment": {

                    "cohort":
                        self.config["population"]["cohort"],

                    "analysis_type":
                        self.config["analysis"]["type"],

                    "task":
                        self.config["analysis"]["task"]["name"],
                },


                # -------------------------
                # Index definition
                # -------------------------

                "index": {

                    "column":
                        self.config["index"]["event"]["column"],

                    "value":
                        self.config["index"]["event"]["value"],
                },


                # -------------------------
                # Outcome definition
                # -------------------------

                "outcome": {

                    "name":
                        self.config["outcome"]["name"],

                    "definition":
                        self.config["outcome"]["definition"],

                    "event_column":
                        self.config["analysis"]["event_column"],

                    "time_column":
                        self.config["analysis"].get(
                            "time_column",
                            None
                        ),
                },


                # -------------------------
                # Feature information
                # -------------------------

                "features": {

                    "feature_set":
                        self.feature_config["name"],

                    "n_features":
                        len(self.feature_columns),

                    "columns":
                        self.feature_columns,

                    "types": {

                        "continuous":
                            self.continuous_features,

                        "binary":
                            self.binary_features,

                        "categorical":
                            self.categorical_features,

                    },

                    "domains":
                        {
                            domain: len(columns)

                            for domain, columns
                            in self.feature_domains.items()
                        },
                },


                # -------------------------
                # Dataset summary
                # -------------------------

                "dataset": {

                    "n_rows":
                        len(self.dataset),

                    "n_participants":
                        (
                            self.dataset["PATNO"]
                            .nunique()
                        ),

                    "pd_events":
                        int(
                            self.dataset["pd_event"]
                            .sum()
                        ),

                    "death_events":
                        int(
                            self.dataset["death_event"]
                            .sum()
                        ),
                },
            }
        )