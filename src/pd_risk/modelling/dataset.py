
from pathlib import Path

import pandas as pd
import yaml

from pd_risk.modelling.targets import SurvivalTargetBuilder
from pd_risk.utils.features import parse_feature_config


class ModelDatasetBuilder:
    """
    Build modality-specific modelling datasets and survival targets.

    The builder:

    1. Loads the explicitly defined modelling cohort.
    2. Loads each enabled modality from the prediction configuration.
    3. Selects configured feature domains.
    4. Restricts each modality to the modelling cohort.
    5. Extracts the requested timepoint.
    6. Builds participant-level survival targets.
    7. Keeps modality data separate.

    Fusion is intentionally not performed here.

    Attributes
    ----------
    modality_data : dict
        Mapping from modality name to its modelling dataframe.

    modality_features : dict
        Feature metadata for each modality.

    targets_df : pandas.DataFrame
        Participant-level survival targets.

    identifiers : pandas.DataFrame
        PATNO identifiers for the modelling cohort.

    metadata : dict
        Reproducibility metadata.
    """

    def __init__(
        self,
        prediction_config: dict,
        project_root: str | Path | None = None,
    ):

        self.config = prediction_config

        if project_root is None:
            self.project_root = Path.cwd()
        else:
            self.project_root = Path(project_root)

        self.modality_data = {}
        self.modality_features = {}

        self.targets_df = None
        self.identifiers = None

        self.cohort = None
        self.cohort_patnos = set()

        self.metadata = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self):
        """
        Build modality-specific modelling data and survival targets.
        """

        self._load_cohort()

        self._load_modalities()

        self._build_targets()

        self._validate()

        self._build_metadata()

        return self

    # ------------------------------------------------------------------
    # Cohort
    # ------------------------------------------------------------------

    def _load_cohort(self):
        """
        Load the explicitly defined modelling cohort.

        The cohort file is expected to contain one row per participant
        and a PATNO identifier.
        """

        cohort_config = self.config.get("cohort")

        if not cohort_config:
            raise ValueError(
                "Prediction configuration must define a 'cohort' section."
            )

        cohort_file = Path(
            cohort_config["file"]
        )

        if not cohort_file.is_absolute():
            cohort_file = (
                self.project_root / cohort_file
            )

        identifier = cohort_config.get(
            "identifier",
            "PATNO",
        )

        if identifier != "PATNO":
            raise ValueError(
                "Only PATNO is currently supported as the modelling "
                "cohort identifier."
            )

        if not cohort_file.exists():
            raise FileNotFoundError(
                f"Modelling cohort file not found: {cohort_file}"
            )

        self.cohort = pd.read_csv(
            cohort_file,
            usecols=["PATNO"],
            low_memory=False,
        )

        if self.cohort["PATNO"].isna().any():
            raise ValueError(
                "Modelling cohort contains missing PATNO values."
            )

        if self.cohort["PATNO"].duplicated().any():
            raise ValueError(
                "Modelling cohort contains duplicate PATNO values."
            )

        self.cohort_patnos = set(
            self.cohort["PATNO"]
        )

        self.identifiers = (
            self.cohort[
                ["PATNO"]
            ]
            .copy()
        )

        print(
            "\n" + "=" * 60
        )
        print(
            "MODELLING COHORT"
        )
        print(
            "=" * 60
        )

        print(
            f"Cohort file: "
            f"{cohort_file.name}"
        )

        print(
            f"Participants: "
            f"{len(self.cohort_patnos):,}"
        )

    # ------------------------------------------------------------------
    # Modalities
    # ------------------------------------------------------------------

    def _load_modalities(self):
        """
        Load all enabled modalities.
        """

        modalities = self.config.get(
            "modalities"
        )

        if not modalities:
            raise ValueError(
                "Prediction configuration does not contain any modalities."
            )

        for modality_name, modality_config in modalities.items():

            if not modality_config.get(
                "enabled",
                False,
            ):
                print(
                    f"\nSkipping disabled modality: "
                    f"{modality_name}"
                )
                continue

            print(
                "\n" + "=" * 60
            )
            print(
                f"LOADING {modality_name.upper()}"
            )
            print(
                "=" * 60
            )

            modality_df, feature_metadata = (
                self._load_modality(
                    modality_name,
                    modality_config,
                )
            )

            self.modality_data[
                modality_name
            ] = modality_df

            self.modality_features[
                modality_name
            ] = feature_metadata

    # ------------------------------------------------------------------
    # Individual modality
    # ------------------------------------------------------------------

    def _load_modality(
        self,
        modality_name,
        modality_config,
    ):
        """
        Load and assemble one modality.

        Each source file is first:

        1. restricted to the modelling cohort
        2. reduced to the requested timepoint when applicable
        3. given a source-specific feature namespace when the modality
        contains multiple data files

        Only then are multiple files merged within the modality.
        """

        modality_type = modality_config.get(
            "type",
            "tabular",
        )

        if modality_type != "tabular":
            raise ValueError(
                f"Unsupported modality type for "
                f"'{modality_name}': {modality_type}"
            )

        # --------------------------------------------------------------
        # Feature configuration
        # --------------------------------------------------------------

        features_file = Path(
            modality_config["features_file"]
        )

        if not features_file.is_absolute():
            features_file = (
                self.project_root
                / features_file
            )

        (
            selected_features,
            continuous_features,
            binary_features,
            categorical_features,
            selected_domains,
        ) = self._load_feature_config(
            features_file,
            modality_config,
        )

        # --------------------------------------------------------------
        # Data files
        # --------------------------------------------------------------

        data_files = modality_config.get(
            "data_files"
        )

        # Support single-file configuration
        if (
            not data_files
            and modality_config.get("data_file")
        ):
            data_files = {
                "data": {
                    "file": modality_config["data_file"]
                }
            }

        if not data_files:
            raise ValueError(
                f"No data file(s) configured for modality "
                f"'{modality_name}'."
            )

        # Determine whether feature names need a source namespace.
        #
        # Example:
        #
        # MRI with three files:
        #   cortical_thickness__lh_bankssts
        #   cortical_surface_area__lh_bankssts
        #   regional_volume__lh_bankssts
        #
        # Single-file modality:
        #   lh_bankssts
        #
        prefix_features = len(data_files) > 1

        combined = None

        # Keep track of the actual feature names returned after
        # source-specific prefixing.
        combined_feature_columns = []

        for dataset_name, dataset_config in data_files.items():

            file_path = Path(
                dataset_config["file"]
            )

            if not file_path.is_absolute():
                file_path = (
                    self.project_root
                    / file_path
                )

            dataset_df, feature_mapping = (
                self._load_modality_file(
                    modality_name=modality_name,
                    dataset_name=dataset_name,
                    file_path=file_path,
                    selected_features=selected_features,
                    modality_config=modality_config,
                    extraction=dataset_config.get(
                        "extraction"
                    ),
                    prefix_features=prefix_features,
                )
            )

            combined_feature_columns.extend(
                [
                    column
                    for column in dataset_df.columns
                    if column != "PATNO"
                ]
            )

            if combined is None:

                combined = dataset_df

            else:

                combined = self._merge_modality_data(
                    combined,
                    dataset_df,
                )

    
        if combined is None:
            raise ValueError(
                f"No data could be loaded for modality "
                f"'{modality_name}'."
            )

        combined = self._align_to_cohort(
            combined,
            modality_name,
        )


        # --------------------------------------------------------------
        # Final modality information
        # --------------------------------------------------------------

        print(
            f"Final participants: "
            f"{combined['PATNO'].nunique():,}"
        )

        print(
            f"Features: "
            f"{len(combined_feature_columns):,}"
        )

        # --------------------------------------------------------------
        # Feature metadata
        # --------------------------------------------------------------

        #
        # The feature configuration contains the canonical feature names,
        # while the actual dataframe may contain namespaced versions.
        #
        # Therefore map the configured feature types onto the actual
        # dataframe columns.
        #

        if prefix_features:

            def namespace_features(features):

                namespaced = []

                for feature in features:

                    matching = [
                        column
                        for column in combined_feature_columns
                        if column.endswith(
                            f"__{feature}"
                        )
                    ]

                    namespaced.extend(
                        matching
                    )

                return namespaced

            metadata_continuous = namespace_features(
                continuous_features
            )

            metadata_binary = namespace_features(
                binary_features
            )

            metadata_categorical = namespace_features(
                categorical_features
            )

        else:

            metadata_continuous = [
                feature
                for feature in continuous_features
                if feature in combined.columns
            ]

            metadata_binary = [
                feature
                for feature in binary_features
                if feature in combined.columns
            ]

            metadata_categorical = [
                feature
                for feature in categorical_features
                if feature in combined.columns
            ]

        feature_metadata = {
            "domains": selected_domains,

            # Actual dataframe feature columns
            "columns": combined_feature_columns,

            "types": {
                "continuous": metadata_continuous,
                "binary": metadata_binary,
                "categorical": metadata_categorical,
            },
        }

        return combined, feature_metadata

    # ------------------------------------------------------------------
    # Feature configuration
    # ------------------------------------------------------------------

    def _load_feature_config(
        self,
        features_file,
        modality_config,
    ):
        """
        Load one modality feature configuration and select domains.

        The feature YAML is expected to have the structure:

            features:
            domain_name:
                continuous:
                columns:
                    - feature_1
                binary:
                columns:
                    - feature_2
                categorical:
                columns:
                    - feature_3

        Empty or commented-out feature groups are treated as containing
        no features. Features are deduplicated while preserving order.
        """

        # ------------------------------------------------------------------
        # 1. Validate feature configuration file
        # ------------------------------------------------------------------

        if not features_file.exists():
            raise FileNotFoundError(
                f"Feature configuration not found: {features_file}"
            )

        with open(
            features_file,
            "r",
            encoding="utf-8",
        ) as f:
            feature_config = yaml.safe_load(f)

        if not isinstance(feature_config, dict):
            raise ValueError(
                f"Feature configuration must contain a YAML mapping: "
                f"{features_file}"
            )

        if "features" not in feature_config:
            raise ValueError(
                f"Feature configuration does not contain "
                f"a 'features' section: {features_file}"
            )

        available_domains = feature_config["features"]

        if not isinstance(available_domains, dict):
            raise ValueError(
                f"'features' must be a mapping of domains in: "
                f"{features_file}"
            )

        # ------------------------------------------------------------------
        # 2. Determine which domains to include
        # ------------------------------------------------------------------

        selected_domains = (
            modality_config
            .get("domains", {})
            .get("include", [])
        )

        # Be tolerant of:
        #
        #   include:
        #
        # or
        #
        #   include: null
        #
        # which YAML loads as None.
        selected_domains = selected_domains or []

        if not isinstance(selected_domains, list):
            raise ValueError(
                f"'domains.include' must be a list in modality configuration."
            )

        # ------------------------------------------------------------------
        # 3. Validate requested domains
        # ------------------------------------------------------------------

        missing_domains = [
            domain
            for domain in selected_domains
            if domain not in available_domains
        ]

        if missing_domains:
            raise ValueError(
                f"Unknown feature domains for modality: "
                f"{missing_domains}"
            )

        # ------------------------------------------------------------------
        # 4. Collect features
        # ------------------------------------------------------------------

        selected_features = []
        continuous_features = []
        binary_features = []
        categorical_features = []

        for domain in selected_domains:

            domain_config = available_domains[domain]

            # An empty/commented-out domain is simply ignored.
            if not domain_config:
                continue

            if not isinstance(domain_config, dict):
                raise ValueError(
                    f"Feature domain '{domain}' must be a mapping "
                    f"of feature types."
                )

            for feature_type, values in domain_config.items():

                # Ignore empty/commented-out feature groups.
                #
                # For example:
                #
                #   categorical:
                #     columns:
                #
                # YAML loads this as:
                #
                #   {"columns": None}
                #
                if not values:
                    continue

                if not isinstance(values, dict):
                    raise ValueError(
                        f"Feature type '{feature_type}' in domain "
                        f"'{domain}' must be a mapping."
                    )

                columns = values.get("columns") or []

                # Empty/commented-out columns are valid.
                if not columns:
                    continue

                if not isinstance(columns, list):
                    raise ValueError(
                        f"'columns' for '{domain}.{feature_type}' "
                        f"must be a list."
                    )

                # Validate feature names.
                invalid_columns = [
                    column
                    for column in columns
                    if not isinstance(column, str)
                    or not column.strip()
                ]

                if invalid_columns:
                    raise ValueError(
                        f"Invalid feature names in "
                        f"'{domain}.{feature_type}': "
                        f"{invalid_columns}"
                    )

                # Remove duplicates within the group while preserving order.
                columns = list(dict.fromkeys(columns))

                # Every selected feature belongs to the overall feature list.
                selected_features.extend(columns)

                # Keep the type-specific lists.
                if feature_type == "continuous":

                    continuous_features.extend(columns)

                elif feature_type == "binary":

                    binary_features.extend(columns)

                elif feature_type == "categorical":

                    categorical_features.extend(columns)

                else:
                    raise ValueError(
                        f"Unknown feature type '{feature_type}' "
                        f"in domain '{domain}'. "
                        f"Expected one of: "
                        f"continuous, binary, categorical."
                    )

        # ------------------------------------------------------------------
        # 5. Deduplicate while preserving YAML order
        # ------------------------------------------------------------------

        selected_features = list(
            dict.fromkeys(selected_features)
        )

        continuous_features = list(
            dict.fromkeys(continuous_features)
        )

        binary_features = list(
            dict.fromkeys(binary_features)
        )

        categorical_features = list(
            dict.fromkeys(categorical_features)
        )

        # ------------------------------------------------------------------
        # 6. Ensure no feature has multiple types
        # ------------------------------------------------------------------

        typed_feature_groups = {
            "continuous": set(continuous_features),
            "binary": set(binary_features),
            "categorical": set(categorical_features),
        }

        feature_type_assignments = {}

        for feature_type, features in typed_feature_groups.items():

            for feature in features:

                feature_type_assignments.setdefault(
                    feature,
                    [],
                ).append(feature_type)

        conflicting_features = {
            feature: feature_types
            for feature, feature_types
            in feature_type_assignments.items()
            if len(feature_types) > 1
        }

        if conflicting_features:
            raise ValueError(
                f"Feature appears in multiple feature types in "
                f"{features_file}: {conflicting_features}"
            )

        # ------------------------------------------------------------------
        # 7. Sanity-check selected features
        # ------------------------------------------------------------------

        all_typed_features = (
            continuous_features
            + binary_features
            + categorical_features
        )

        if set(selected_features) != set(all_typed_features):
            untyped_features = (
                set(selected_features)
                - set(all_typed_features)
            )

            if untyped_features:
                raise ValueError(
                    f"Selected features have no recognized feature type "
                    f"in {features_file}: {sorted(untyped_features)}"
                )

        return (
            selected_features,
            continuous_features,
            binary_features,
            categorical_features,
            selected_domains,
        )


    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_modality_file(
        self,
        modality_name,
        dataset_name,
        file_path,
        selected_features,
        modality_config,
        extraction,
        prefix_features,
    ):
        """
        Load one configured modality data file.

        The file is:

        1. restricted to the modelling cohort
        2. reduced to the requested timepoint when applicable
        3. given a source-specific feature namespace when the modality
        contains multiple data files

        Static files without temporal columns are retained as-is.
        """

        print(
            f"\n  Loading {dataset_name}: "
            f"{file_path.name}"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {file_path}"
            )

        # --------------------------------------------------------------
        # Inspect columns
        # --------------------------------------------------------------

        columns = pd.read_csv(
            file_path,
            nrows=0,
        ).columns.tolist()

        if "PATNO" not in columns:
            raise ValueError(
                f"'PATNO' not found in "
                f"{file_path.name}"
            )

        file_features = [
            feature
            for feature in selected_features
            if feature in columns
        ]

        if not file_features:
            raise ValueError(
                f"No configured features for modality "
                f"'{modality_name}' were found in "
                f"'{file_path.name}'."
            )

        # --------------------------------------------------------------
        # Determine temporal structure
        # --------------------------------------------------------------

        if "EVENT_ID" in columns:

            time_column = "EVENT_ID"

        elif "CLINICAL_EVENT" in columns:

            time_column = "CLINICAL_EVENT"

        else:

            # Static modality
            time_column = None

        # --------------------------------------------------------------
        # Load data
        # --------------------------------------------------------------

        use_columns = [
            "PATNO",
            *file_features,
        ]

        if time_column is not None:
            use_columns.append(
                time_column
            )

        use_columns = list(
            dict.fromkeys(
                use_columns
            )
        )

        data = pd.read_csv(
            file_path,
            usecols=use_columns,
            low_memory=False,
        )

        data = data.dropna(
            subset=["PATNO"]
        )

        # --------------------------------------------------------------
        # Restrict to modelling cohort
        # --------------------------------------------------------------

        data = data.loc[
            data["PATNO"].isin(
                self.cohort_patnos
            )
        ].copy()

        print(
            f"    Cohort participants found: "
            f"{data['PATNO'].nunique():,}"
        )

        # --------------------------------------------------------------
        # Extract requested timepoint
        # --------------------------------------------------------------

        data = self._extract_timepoint(
            data=data,
            modality_name=modality_name,
            dataset_name=dataset_name,
            extraction=extraction,
        )

        # --------------------------------------------------------------
        # Validate participant-level structure
        # --------------------------------------------------------------

        if data["PATNO"].duplicated().any():

            duplicated_patnos = (
                data.loc[
                    data["PATNO"].duplicated(
                        keep=False
                    ),
                    "PATNO",
                ]
                .nunique()
            )

            raise ValueError(
                f"File '{file_path.name}' contains "
                f"multiple rows per PATNO after "
                f"timepoint extraction "
                f"({duplicated_patnos:,} participants)."
            )

        # --------------------------------------------------------------
        # Remove temporal index columns
        # --------------------------------------------------------------

        temporal_columns = [
            column
            for column in [
                "EVENT_ID",
                "CLINICAL_EVENT",
            ]
            if column in data.columns
        ]

        if temporal_columns:

            data = data.drop(
                columns=temporal_columns
            )

        # --------------------------------------------------------------
        # Apply source-specific feature namespace
        # --------------------------------------------------------------

        rename_map = {}

        if prefix_features:

            rename_map = {
                feature: (
                    f"{dataset_name}__{feature}"
                )
                for feature in file_features
            }

            data = data.rename(
                columns=rename_map
            )

            output_features = [
                rename_map[feature]
                for feature in file_features
            ]

        else:

            output_features = list(
                file_features
            )

        # --------------------------------------------------------------
        # Return
        # --------------------------------------------------------------

        return (
            data[
                ["PATNO"] + output_features
            ],
            rename_map,
        )

    # ------------------------------------------------------------------
    # Modality merging
    # ------------------------------------------------------------------

    def _merge_modality_data(
        self,
        left,
        right,
    ):
        """
        Merge two data sources belonging to the same modality.

        Currently the modelling data are baseline participant-level
        data, so PATNO is the merge key.

        Duplicate feature names are rejected rather than silently
        overwritten.
        """

        overlapping = (
            set(left.columns)
            & set(right.columns)
        ) - {"PATNO"}

        if overlapping:
            raise ValueError(
                "Duplicate feature columns encountered while "
                "combining modality data: "
                f"{sorted(overlapping)}"
            )

        return left.merge(
            right,
            on="PATNO",
            how="outer",
            validate="one_to_one",
        )

    # ------------------------------------------------------------------
    # Timepoint extraction
    # ------------------------------------------------------------------

    def _extract_timepoint(
        self,
        data,
        modality_name,
        dataset_name,
        extraction,
    ):
        """
        Extract the observation required for modelling.

        extraction:
            BL     -> baseline observation
            SC     -> screening observation
            static -> no timepoint filtering

        The temporal column is inferred from the data:
            EVENT_ID
            CLINICAL_EVENT

        Static datasets must not contain multiple rows per PATNO.
        """

        if extraction is None:
            raise ValueError(
                f"No extraction mode configured for modality "
                f"'{modality_name}', dataset '{dataset_name}'. "
                f"Expected one of: BL, SC, static."
            )

        extraction = str(
            extraction
        ).strip().lower()

        valid_modes = {
            "bl",
            "sc",
            "static",
        }

        if extraction not in valid_modes:
            raise ValueError(
                f"Invalid extraction mode '{extraction}' for "
                f"modality '{modality_name}', dataset "
                f"'{dataset_name}'. Expected one of: "
                f"BL, SC, static."
            )

        # --------------------------------------------------------------
        # Static data
        # --------------------------------------------------------------

        if extraction == "static":

            if "PATNO" not in data.columns:
                raise ValueError(
                    f"Static dataset '{dataset_name}' in modality "
                    f"'{modality_name}' does not contain PATNO."
                )

            if data["PATNO"].duplicated().any():

                duplicate_patnos = (
                    data.loc[
                        data["PATNO"].duplicated(
                            keep=False
                        ),
                        "PATNO",
                    ]
                    .unique()
                    .tolist()
                )

                raise ValueError(
                    f"Static dataset '{dataset_name}' in modality "
                    f"'{modality_name}' contains multiple rows for "
                    f"{len(duplicate_patnos)} participants. "
                    f"Example PATNOs: "
                    f"{duplicate_patnos[:10]}"
                )

            return data.copy()

        # --------------------------------------------------------------
        # Determine temporal column
        # --------------------------------------------------------------

        if "EVENT_ID" in data.columns:

            time_column = "EVENT_ID"

        elif "CLINICAL_EVENT" in data.columns:

            time_column = "CLINICAL_EVENT"

        else:

            raise ValueError(
                f"Cannot perform '{extraction}' extraction for "
                f"modality '{modality_name}', dataset "
                f"'{dataset_name}': neither EVENT_ID nor "
                f"CLINICAL_EVENT is present."
            )

        # --------------------------------------------------------------
        # Extract requested observation
        # --------------------------------------------------------------

        return self._extract_observation(
            data=data,
            modality_name=modality_name,
            dataset_name=dataset_name,
            time_column=time_column,
            extraction=extraction,
        )

    def _extract_observation(
        self,
        data,
        modality_name,
        dataset_name,
        time_column,
        extraction,
    ):
        """
        Extract the configured observation from a temporal dataset.

        extraction:
            bl -> BL
            sc -> SC
        """

        extraction_value = extraction.upper()

        values = (
            data[time_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        matching = (
            values == extraction_value
        )

        if not matching.any():

            raise ValueError(
                f"No '{extraction_value}' observations found for "
                f"modality '{modality_name}', dataset "
                f"'{dataset_name}' using column "
                f"'{time_column}'."
            )

        extracted = (
            data.loc[
                matching
            ]
            .copy()
        )

        # --------------------------------------------------------------
        # One observation per participant
        # --------------------------------------------------------------

        duplicate_mask = (
            extracted["PATNO"]
            .duplicated(
                keep=False
            )
        )

        if duplicate_mask.any():

            duplicate_patnos = (
                extracted.loc[
                    duplicate_mask,
                    "PATNO",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Multiple '{extraction_value}' observations found "
                f"for {len(duplicate_patnos)} participants in "
                f"modality '{modality_name}', dataset "
                f"'{dataset_name}' using column "
                f"'{time_column}'. "
                f"Example PATNOs: "
                f"{duplicate_patnos[:10]}"
            )

        # --------------------------------------------------------------
        # Extraction/index column is not a predictor
        # --------------------------------------------------------------

        extracted = extracted.drop(
            columns=[
                time_column
            ]
        )

        return extracted

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def _build_targets(self):
        """
        Build participant-level survival targets.

        The outcome file is independent of predictor extraction.
        """

        outcome_config = (
            self.config
            .get("inputs", {})
            .get("outcomes")
        )

        if not outcome_config:
            raise ValueError(
                "Prediction configuration must define "
                "'inputs.outcomes.file'."
            )

        outcome_file = Path(
            outcome_config["file"]
        )

        if not outcome_file.is_absolute():
            outcome_file = (
                self.project_root
                / outcome_file
            )

        if not outcome_file.exists():
            raise FileNotFoundError(
                f"Outcome file not found: "
                f"{outcome_file}"
            )

        outcomes = pd.read_csv(
            outcome_file,
            low_memory=False,
        )

        if "PATNO" not in outcomes.columns:
            raise ValueError(
                f"'PATNO' not found in outcome file: "
                f"{outcome_file}"
            )

        outcomes = outcomes.loc[
            outcomes["PATNO"].isin(
                self.cohort_patnos
            )
        ].copy()

        target_builder = SurvivalTargetBuilder(
            df=outcomes,
            config=self.config,
        )

        self.targets_df = (
            target_builder.build()
        )

        if self.targets_df["PATNO"].duplicated().any():

            duplicate_patnos = (
                self.targets_df.loc[
                    self.targets_df["PATNO"].duplicated(
                        keep=False
                    ),
                    "PATNO",
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Survival target builder produced duplicate "
                f"PATNO values: {duplicate_patnos[:10]}"
            )


        target_patnos = set(
            self.targets_df["PATNO"]
        )

        missing_targets = (
            self.cohort_patnos
            - target_patnos
        )

        if missing_targets:
            raise ValueError(
                f"{len(missing_targets):,} modelling cohort "
                "participants are missing outcome targets."
            )

        self.targets_df = (
            self.identifiers
            .merge(
                self.targets_df,
                on="PATNO",
                how="left",
                validate="one_to_one",
                sort=False,
            )
        )

        print(
            "\nTargets:"
        )

        print(
            f"  Participants: "
            f"{len(self.targets_df):,}"
        )

        print(
            f"  Events: "
            f"{int(self.targets_df['event'].sum()):,}"
        )


    # ------------------------------------------------------------------
    # Cohort alignmnet
    # ------------------------------------------------------------------

    def _align_to_cohort(
        self,
        data: pd.DataFrame,
        modality_name: str,
    ) -> pd.DataFrame:
        """
        Align a modality to the modelling cohort.

        The returned dataframe contains exactly one row per modelling
        participant, in the same order as self.identifiers.

        Missing feature values are preserved as NaN. Missing participants
        are not silently discarded.
        """

        if "PATNO" not in data.columns:
            raise ValueError(
                f"Modality '{modality_name}' does not contain PATNO."
            )

        if data["PATNO"].duplicated().any():
            raise ValueError(
                f"Modality '{modality_name}' contains duplicate PATNO values."
            )

        aligned = (
            self.identifiers
            .merge(
                data,
                on="PATNO",
                how="left",
                validate="one_to_one",
                sort=False,
            )
        )

        return aligned


    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate(self):
        """
        Validate cohort, modalities, and targets.
        """

        if self.targets_df is None:
            raise ValueError(
                "Targets have not been built."
            )

        target_patnos = set(
            self.targets_df["PATNO"]
        )

        missing_targets = (
            self.cohort_patnos
            - target_patnos
        )

        if missing_targets:
            raise ValueError(
                f"{len(missing_targets):,} modelling cohort "
                f"participants are missing outcome targets."
            )

        duplicate_targets = (
            self.targets_df["PATNO"]
            .duplicated()
            .any()
        )

        if duplicate_targets:
            raise ValueError(
                "Duplicate PATNO values found in targets."
            )

        for modality, data in (
            self.modality_data.items()
        ):

            if "PATNO" not in data.columns:
                raise ValueError(
                    f"Modality '{modality}' does not contain PATNO."
                )

            if data["PATNO"].duplicated().any():
                raise ValueError(
                    f"Modality '{modality}' contains multiple "
                    f"rows per PATNO after extraction."
                )

            unknown_patnos = (
                set(data["PATNO"])
                - self.cohort_patnos
            )

            if unknown_patnos:
                raise ValueError(
                    f"Modality '{modality}' contains participants "
                    f"outside the modelling cohort."
                )

        cohort_patnos = self.identifiers["PATNO"].tolist()

        if self.targets_df["PATNO"].tolist() != cohort_patnos:
            raise ValueError(
                "Target participant ordering does not match "
                "the modelling cohort."
            )

        for modality, data in self.modality_data.items():

            if data["PATNO"].tolist() != cohort_patnos:
                raise ValueError(
                    f"Modality '{modality}' participant ordering does "
                    "not match the modelling cohort."
                )

        print(
            "\n" + "=" * 60
        )
        print(
            "MODEL DATASET VALIDATION"
        )
        print(
            "=" * 60
        )

        print(
            "Validation passed."
        )

        print(
            f"Modelling cohort: "
            f"{len(self.cohort_patnos):,}"
        )

        for modality, data in (
            self.modality_data.items()
        ):

            print(
                f"  {modality}: "
                f"{len(data):,} participants, "
                f"{len(data.columns) - 1:,} features"
            )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _build_metadata(self):
        """
        Build reproducibility metadata.
        """

        self.metadata = {

            "experiment_config": self.config,

            "cohort": {
                "file": str(
                    self.config["cohort"]["file"]
                ),
                "identifier": "PATNO",
                "n_participants": len(
                    self.cohort_patnos
                ),
            },

            "data_extraction": self.config.get(
                "data_extraction",
                {},
            ),

            "modalities": {},

            "survival": {
                "time_column": "time_to_event",
                "event_column": "event",
                "event_type_column": "event_type",
            },

            "dataset_summary": {
                "n_samples": len(
                    self.cohort_patnos
                ),
                "n_events": int(
                    self.targets_df["event"].sum()
                ),
                "n_censored": int(
                    (
                        self.targets_df["event"]
                        == 0
                    ).sum()
                ),
            },
        }

        for modality, feature_metadata in (
            self.modality_features.items()
        ):

            self.metadata[
                "modalities"
            ][modality] = feature_metadata
