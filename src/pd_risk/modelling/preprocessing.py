# pd_risk/modelling/preprocessing.py

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class BasePreprocessor:
    """
    Base class for modality-specific preprocessing.

    Responsibilities:
        - common feature-config handling
        - feature extraction
        - column validation
        - configurable missing-value handling
        - fitted-state management
        - common fit/transform interface

    Modality-specific preprocessing belongs in subclasses.

    Missing-value handling is driven entirely by the modelling
    configuration. Subclasses should not hard-code imputation rules.
    """

    def __init__(self, config=None):
        self.config = config or {}
        self.is_fitted = False

        # Fitted missing-value transformers.
        self.missing_imputers = {}

    # ------------------------------------------------------------------
    # Feature configuration
    # ------------------------------------------------------------------

    @staticmethod
    def load_feature_config(path):
        path = Path(path)

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_feature_columns(
        features_config,
        domain,
        feature_type,
    ):
        """
        Return columns declared for a domain and feature type.
        """

        try:
            return (
                features_config["features"]
                [domain]
                [feature_type]
                ["columns"]
            )
        except KeyError as exc:
            raise KeyError(
                f"Could not find feature configuration for "
                f"domain='{domain}', type='{feature_type}'."
            ) from exc

    # ------------------------------------------------------------------
    # Prefix handling
    # ------------------------------------------------------------------

    @staticmethod
    def prefix_features(
        features: Iterable[str],
        prefix: str | None = None,
    ) -> list[str]:
        """
        Resolve raw feature names to ModelDatasetBuilder column names.

        Single-file modalities:
            feature

        Multi-file modalities:
            prefix__feature
        """

        if prefix is None:
            return list(features)

        return [
            f"{prefix}__{feature}"
            for feature in features
        ]

    # ------------------------------------------------------------------
    # Missing-value configuration
    # ------------------------------------------------------------------

    def get_missing_strategy(
        self,
        feature_type,
    ):
        """
        Return the configured missing-value strategy for a feature type.

        Expected configuration:

            preprocessing:
              numeric:
                missing:
                  strategy: median

              binary:
                missing:
                  strategy: most_frequent

              categorical:
                missing:
                  strategy: most_frequent
        """

        type_config = self.config.get(
            feature_type,
            {},
        )

        missing_config = type_config.get(
            "missing",
            {},
        )

        return missing_config.get(
            "strategy"
        )

    def build_missing_imputer(
        self,
        feature_type,
    ):
        """
        Build a missing-value imputer from configuration.

        Supported strategies are delegated to sklearn's
        SimpleImputer.

        A strategy of None or 'passthrough' means that no
        imputation is performed.
        """

        strategy = self.get_missing_strategy(
            feature_type
        )

        if strategy is None:
            return None

        if strategy == "passthrough":
            return None

        valid_strategies = {
            "mean",
            "median",
            "most_frequent",
            "constant",
        }

        if strategy not in valid_strategies:
            raise ValueError(
                f"Unsupported missing-value strategy "
                f"'{strategy}' for feature type "
                f"'{feature_type}'. Supported strategies: "
                f"{sorted(valid_strategies)}."
            )

        return SimpleImputer(
            strategy=strategy
        )

    def fit_missing_imputer(
        self,
        df,
        columns,
        feature_type,
    ):
        """
        Fit a missing-value transformer for a feature group.

        Returns the fitted transformer, or None when missing-value
        handling is configured as passthrough.
        """

        columns = list(columns)

        if not columns:
            return None

        imputer = self.build_missing_imputer(
            feature_type
        )

        if imputer is None:
            self.missing_imputers[
                feature_type
            ] = None

            return None

        self.validate_columns(
            df,
            columns,
        )

        imputer.fit(
            df[columns]
        )

        self.missing_imputers[
            feature_type
        ] = imputer

        return imputer

    def transform_missing(
        self,
        df,
        columns,
        feature_type,
    ):
        """
        Apply the fitted missing-value transformer to a feature group.

        The original DataFrame index and columns are preserved.
        """

        columns = list(columns)

        if not columns:
            return df.copy()

        if feature_type not in self.missing_imputers:
            raise RuntimeError(
                f"Missing-value transformer for feature type "
                f"'{feature_type}' has not been fitted."
            )

        imputer = self.missing_imputers[
            feature_type
        ]

        if imputer is None:
            return df.copy()

        self.validate_columns(
            df,
            columns,
        )

        result = df.copy()

        transformed = imputer.transform(
            result[columns]
        )

        result[columns] = transformed

        return result

    def fit_missing_handling(
        self,
        df,
        feature_groups,
    ):
        """
        Fit missing-value transformers for multiple feature groups.

        Parameters
        ----------
        df:
            Input dataframe.

        feature_groups:
            Mapping of feature type to columns, e.g.

                {
                    "numeric": [...],
                    "binary": [...],
                    "categorical": [...],
                }
        """

        for feature_type, columns in (
            feature_groups.items()
        ):
            self.fit_missing_imputer(
                df,
                columns,
                feature_type,
            )

        return self

    def transform_missing_handling(
        self,
        df,
        feature_groups,
    ):
        """
        Apply fitted missing-value handling to multiple
        feature groups.
        """

        result = df.copy()

        for feature_type, columns in (
            feature_groups.items()
        ):
            result = self.transform_missing(
                result,
                columns,
                feature_type,
            )

        return result

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def validate_columns(
        df: pd.DataFrame,
        columns: Iterable[str],
    ):
        columns = list(columns)

        missing = [
            column
            for column in columns
            if column not in df.columns
        ]

        if missing:
            raise ValueError(
                "Required preprocessing columns are missing: "
                f"{missing}"
            )

    @staticmethod
    def validate_no_missing_output(df):
        if df.isna().any().any():
            missing = (
                df.isna()
                .sum()
                .loc[lambda x: x > 0]
                .to_dict()
            )

            raise ValueError(
                "Preprocessed data contains missing values: "
                f"{missing}"
            )

    @staticmethod
    def validate_finite_output(df):
        numeric = df.select_dtypes(
            include=np.number
        )

        if not np.isfinite(
            numeric.to_numpy()
        ).all():
            raise ValueError(
                "Preprocessed data contains "
                "non-finite numeric values."
            )

    # ------------------------------------------------------------------
    # Generic interface
    # ------------------------------------------------------------------

    def fit(self, df):
        raise NotImplementedError

    def transform(self, df):
        raise NotImplementedError

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)

    def get_feature_names(self):
        raise NotImplementedError

    def get_metadata(self):

        missing_metadata = {}

        for feature_type, imputer in (
            self.missing_imputers.items()
        ):

            missing_metadata[
                feature_type
            ] = {
                "strategy": (
                    None
                    if imputer is None
                    else imputer.strategy
                )
            }

        return {
            "preprocessor":
                self.__class__.__name__,
            "fitted":
                self.is_fitted,
            "missing_value_handling":
                missing_metadata,
            "output_features":
                self.get_feature_names(),
        }


# ======================================================================
# Clinical
# ======================================================================


class ClinicalPreprocessor(BasePreprocessor):
    """
    Preprocessing pipeline for clinical predictors.
    """

    def __init__(
        self,
        continuous_features,
        binary_features,
        categorical_features,
        config,
    ):
        super().__init__(config=config)

        self.continuous_features = list(
            continuous_features
        )
        self.binary_features = list(
            binary_features
        )
        self.categorical_features = list(
            categorical_features
        )

        self.pipeline = None

    def build(self):

        numeric_config = self.config["numeric"]
        binary_config = self.config["binary"]
        categorical_config = self.config["categorical"]

        numeric_missing = (
            numeric_config["missing"]["strategy"]
        )

        scaling_enabled = (
            numeric_config["scaling"]["enabled"]
        )

        binary_missing = (
            binary_config["missing"]["strategy"]
        )

        categorical_missing = (
            categorical_config["missing"]["strategy"]
        )

        encoding_method = (
            categorical_config["encoding"]["method"]
        )

        handle_unknown = (
            categorical_config["encoding"]
            .get("handle_unknown", "ignore")
        )

        if encoding_method != "one_hot":
            raise ValueError(
                f"Unsupported categorical encoding: "
                f"{encoding_method}"
            )

        continuous_steps = [
            (
                "imputer",
                SimpleImputer(
                    strategy=numeric_missing
                ),
            )
        ]

        if scaling_enabled:
            continuous_steps.append(
                (
                    "scaler",
                    StandardScaler(),
                )
            )

        continuous_pipeline = Pipeline(
            steps=continuous_steps
        )

        binary_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=binary_missing
                    ),
                )
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy=categorical_missing
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown=handle_unknown,
                        sparse_output=False,
                    ),
                ),
            ]
        )

        self.pipeline = ColumnTransformer(
            transformers=[
                (
                    "continuous",
                    continuous_pipeline,
                    self.continuous_features,
                ),
                (
                    "binary",
                    binary_pipeline,
                    self.binary_features,
                ),
                (
                    "categorical",
                    categorical_pipeline,
                    self.categorical_features,
                ),
            ]
        )

        return self

    def fit(self, df):

        self.validate_columns(
            df,
            (
                self.continuous_features
                + self.binary_features
                + self.categorical_features
            ),
        )

        if self.pipeline is None:
            self.build()

        self.pipeline.fit(df)

        self.is_fitted = True

        return self

    def transform(self, df):

        if not self.is_fitted:
            raise RuntimeError(
                "Clinical preprocessor has not been fitted."
            )

        X = self.pipeline.transform(df)

        result = pd.DataFrame(
            X,
            columns=self.get_feature_names(),
            index=df.index,
        )

        self.validate_no_missing_output(result)
        self.validate_finite_output(result)

        return result

    def get_feature_names(self):

        if self.pipeline is None:
            raise RuntimeError(
                "Pipeline has not been built."
            )

        return (
            self.pipeline
            .get_feature_names_out()
            .tolist()
        )

    def get_metadata(self):

        return {
            "preprocessor": self.__class__.__name__,
            "continuous_features":
                self.continuous_features,
            "binary_features":
                self.binary_features,
            "categorical_features":
                self.categorical_features,
            "output_features":
                self.get_feature_names(),
        }


# ======================================================================
# DAT
# ======================================================================


class DATPreprocessor(BasePreprocessor):
    """
    Preprocessing for DAT imaging-derived features.

    All continuous DAT features are handled according to the
    numeric preprocessing configuration.

    Pipeline:

        missing-value handling
            ↓
        optional standardization

    PATNO is never treated as a feature.
    """

    def __init__(
        self,
        features_config,
        config=None,
    ):
        super().__init__(config=config)

        self.features_config = features_config

        # ----------------------------------------------------------
        # Collect continuous features from all DAT domains
        # ----------------------------------------------------------

        self.features = []

        for domain_config in (
            features_config["features"].values()
        ):
            self.features.extend(
                domain_config
                .get("continuous", {})
                .get("columns", [])
            )

        if not self.features:
            raise ValueError(
                "No continuous DAT features found "
                "in feature configuration."
            )

        # ----------------------------------------------------------
        # Configuration
        # ----------------------------------------------------------

        numeric_config = self.config["numeric"]

        self.missing_strategy = (
            numeric_config["missing"]["strategy"]
        )

        self.scaling_enabled = (
            numeric_config["scaling"]["enabled"]
        )

        # ----------------------------------------------------------
        # Learned preprocessing objects
        # ----------------------------------------------------------

        self.imputer = None
        self.scaler = None

    # --------------------------------------------------------------
    # Fit
    # --------------------------------------------------------------

    def fit(self, df):

        self.validate_columns(
            df,
            self.features,
        )

        # Missing-value handling
        self.imputer = SimpleImputer(
            strategy=self.missing_strategy
        )

        X = self.imputer.fit_transform(
            df[self.features]
        )

        # Scaling
        if self.scaling_enabled:

            self.scaler = StandardScaler()

            self.scaler.fit(X)

        self.is_fitted = True

        return self

    # --------------------------------------------------------------
    # Transform
    # --------------------------------------------------------------

    def transform(self, df):

        if not self.is_fitted:
            raise RuntimeError(
                "DAT preprocessor has not been fitted."
            )

        self.validate_columns(
            df,
            self.features,
        )

        # Missing values
        X = self.imputer.transform(
            df[self.features]
        )

        # Scaling
        if self.scaler is not None:

            X = self.scaler.transform(X)

        result = pd.DataFrame(
            X,
            columns=self.features,
            index=df.index,
        )

        self.validate_no_missing_output(
            result
        )

        self.validate_finite_output(
            result
        )

        return result

    # --------------------------------------------------------------
    # Feature names
    # --------------------------------------------------------------

    def get_feature_names(self):

        return list(self.features)

    # --------------------------------------------------------------
    # Metadata
    # --------------------------------------------------------------

    def get_metadata(self):

        return {
            "preprocessor":
                self.__class__.__name__,

            "missing": {
                "strategy":
                    self.missing_strategy,
            },

            "scaling": {
                "enabled":
                    self.scaling_enabled,
                "method": (
                    "standard_scaler"
                    if self.scaling_enabled
                    else None
                ),
            },

            "features":
                self.features,

            "output_features":
                self.get_feature_names(),
        }
# ======================================================================
# MRI
# ======================================================================

class MRIPreprocessor(BasePreprocessor):
    """
    Preprocessing for FreeSurfer MRI-derived features.

    Input columns are prefixed because MRI consists of multiple files:

        cortical_thickness__*
        cortical_surface_area__*
        regional_volume__*

    Transformations:

        Cortical thickness:
            z-transform

        Surface area:
            eTIV adjustment
            z-transform

        Regional volume:
            eTIV adjustment
            z-transform

    Missing-value handling is controlled by the shared numeric
    preprocessing configuration.

    eTIV is used as an auxiliary adjustment variable and is never
    returned as a model feature.
    """

    CTH_PREFIX = "cortical_thickness"
    SA_PREFIX = "cortical_surface_area"
    VOL_PREFIX = "regional_volume"

    ETIV_RAW = "EstimatedTotalIntraCranialVol"

    def __init__(
        self,
        features_config,
        config=None,
    ):
        super().__init__(config=config)

        self.features_config = features_config

        # --------------------------------------------------------------
        # Feature configuration
        # --------------------------------------------------------------

        cth_raw = self.get_feature_columns(
            features_config,
            "cortical_thickness",
            "continuous",
        )

        sa_raw = self.get_feature_columns(
            features_config,
            "cortical_surface_area",
            "continuous",
        )

        volume_raw = self.get_feature_columns(
            features_config,
            "regional_volume",
            "continuous",
        )

        self.cth_features = self.prefix_features(
            cth_raw,
            self.CTH_PREFIX,
        )

        self.sa_features = self.prefix_features(
            sa_raw,
            self.SA_PREFIX,
        )

        self.etiv_column = (
            f"{self.VOL_PREFIX}__{self.ETIV_RAW}"
        )

        self.volume_features = [
            column
            for column in self.prefix_features(
                volume_raw,
                self.VOL_PREFIX,
            )
            if column != self.etiv_column
        ]

        # --------------------------------------------------------------
        # Fitted MRI parameters
        # --------------------------------------------------------------

        self.sa_betas = {}
        self.volume_betas = {}

        self.etiv_mean = None

        self.cth_scaler = None
        self.sa_scaler = None
        self.volume_scaler = None

    # ------------------------------------------------------------------
    # Numeric feature groups
    # ------------------------------------------------------------------

    @property
    def numeric_features(self):
        """
        All numeric columns requiring missing-value handling.

        eTIV is included because it is required for MRI adjustment,
        even though it is excluded from the final model matrix.
        """

        return (
            self.cth_features
            + self.sa_features
            + self.volume_features
            + [self.etiv_column]
        )

    # ------------------------------------------------------------------
    # eTIV adjustment
    # ------------------------------------------------------------------

    @staticmethod
    def _fit_adjustment(
        df,
        features,
        etiv_column,
    ):
        """
        Fit the linear eTIV adjustment coefficient for each feature.

        beta is estimated as:

            Cov(eTIV, feature) / Var(eTIV)

        Missing values must already have been handled before this
        method is called.
        """

        etiv = df[
            etiv_column
        ].to_numpy(
            dtype=float
        )

        variance = np.var(
            etiv,
            ddof=0,
        )

        if not np.isfinite(variance):
            raise ValueError(
                "eTIV variance is non-finite after "
                "missing-value handling."
            )

        if variance == 0:
            raise ValueError(
                "eTIV has zero variance; "
                "cannot perform MRI adjustment."
            )

        etiv_mean = np.mean(etiv)

        betas = {}

        for feature in features:

            x = df[
                feature
            ].to_numpy(
                dtype=float
            )

            if not np.isfinite(x).all():
                raise ValueError(
                    "MRI feature contains non-finite values "
                    "during eTIV adjustment: "
                    f"'{feature}'."
                )

            x_mean = np.mean(x)

            covariance = np.mean(
                (etiv - etiv_mean)
                * (x - x_mean)
            )

            beta = covariance / variance

            if not np.isfinite(beta):
                raise ValueError(
                    "Non-finite eTIV adjustment coefficient "
                    f"for MRI feature '{feature}'."
                )

            betas[feature] = beta

        return betas

    @staticmethod
    def _adjust_features(
        df,
        features,
        betas,
        etiv_column,
        etiv_mean,
    ):
        df = df.copy()

        etiv = df[
            etiv_column
        ]

        for feature in features:

            beta = betas[feature]

            df[feature] = (
                df[feature]
                - beta
                * (
                    etiv
                    - etiv_mean
                )
            )

        return df

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df):

        self.validate_columns(
            df,
            self.numeric_features,
        )

        # --------------------------------------------------------------
        # Missing-value handling
        #
        # Strategy is entirely controlled by the shared configuration.
        # --------------------------------------------------------------

        feature_groups = {
            "numeric": self.numeric_features,
        }

        self.fit_missing_handling(
            df,
            feature_groups,
        )

        df = self.transform_missing_handling(
            df,
            feature_groups,
        )

        # --------------------------------------------------------------
        # eTIV reference value
        # --------------------------------------------------------------

        self.etiv_mean = (
            df[
                self.etiv_column
            ].mean()
        )

        if not np.isfinite(
            self.etiv_mean
        ):
            raise ValueError(
                "eTIV mean is non-finite after "
                "missing-value handling."
            )


        # --------------------------------------------------------------
        # Cortical thickness
        # --------------------------------------------------------------

        self.cth_scaler = StandardScaler()

        self.cth_scaler.fit(
            df[
                self.cth_features
            ]
        )

        # --------------------------------------------------------------
        # Surface area
        # --------------------------------------------------------------

        self.sa_betas = (
            self._fit_adjustment(
                df,
                self.sa_features,
                self.etiv_column,
            )
        )

        sa_adjusted = (
            self._adjust_features(
                df,
                self.sa_features,
                self.sa_betas,
                self.etiv_column,
                self.etiv_mean,
            )
        )

        self.sa_scaler = StandardScaler()

        self.sa_scaler.fit(
            sa_adjusted[
                self.sa_features
            ]
        )

        # --------------------------------------------------------------
        # Regional volume
        # --------------------------------------------------------------

        self.volume_betas = (
            self._fit_adjustment(
                df,
                self.volume_features,
                self.etiv_column,
            )
        )

        volume_adjusted = (
            self._adjust_features(
                df,
                self.volume_features,
                self.volume_betas,
                self.etiv_column,
                self.etiv_mean,
            )
        )

        self.volume_scaler = StandardScaler()

        self.volume_scaler.fit(
            volume_adjusted[
                self.volume_features
            ]
        )

        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df):

        if not self.is_fitted:
            raise RuntimeError(
                "MRI preprocessor has not been fitted."
            )

        self.validate_columns(
            df,
            self.numeric_features,
        )

        df = df.copy()

        # --------------------------------------------------------------
        # Missing-value handling
        #
        # Uses the transformer fitted during fit().
        # No statistics are learned from this dataframe.
        # --------------------------------------------------------------

        df = self.transform_missing_handling(
            df,
            {
                "numeric": self.numeric_features,
            },
        )
        # --------------------------------------------------------------
        # Cortical thickness
        # --------------------------------------------------------------

        df[
            self.cth_features
        ] = (
            self.cth_scaler.transform(
                df[
                    self.cth_features
                ]
            )
        )

        # --------------------------------------------------------------
        # Surface area
        # --------------------------------------------------------------

        df = self._adjust_features(
            df,
            self.sa_features,
            self.sa_betas,
            self.etiv_column,
            self.etiv_mean,
        )

        df[
            self.sa_features
        ] = (
            self.sa_scaler.transform(
                df[
                    self.sa_features
                ]
            )
        )

        # --------------------------------------------------------------
        # Regional volume
        # --------------------------------------------------------------

        df = self._adjust_features(
            df,
            self.volume_features,
            self.volume_betas,
            self.etiv_column,
            self.etiv_mean,
        )

        df[
            self.volume_features
        ] = (
            self.volume_scaler.transform(
                df[
                    self.volume_features
                ]
            )
        )

        # --------------------------------------------------------------
        # Final model matrix
        # --------------------------------------------------------------

        output_features = (
            self.cth_features
            + self.sa_features
            + self.volume_features
        )

        result = df[
            output_features
        ].copy()

        # eTIV is deliberately absent from result.

        self.validate_no_missing_output(
            result
        )

        self.validate_finite_output(
            result
        )

        return result

    # ------------------------------------------------------------------
    # Metadata / feature names
    # ------------------------------------------------------------------

    def get_feature_names(self):

        return (
            self.cth_features
            + self.sa_features
            + self.volume_features
        )

    def get_metadata(self):

        metadata = super().get_metadata()

        metadata.update({
            "cortical_thickness": {
                "transformation":
                    "standard_scaler",
                "n_features":
                    len(self.cth_features),
            },

            "surface_area": {
                "transformation":
                    "eTIV_adjustment_then_standard_scaler",
                "n_features":
                    len(self.sa_features),
            },

            "regional_volume": {
                "transformation":
                    "eTIV_adjustment_then_standard_scaler",
                "n_features":
                    len(self.volume_features),
            },

            "etiv_column":
                self.etiv_column,

            "etiv_used_as_feature":
                False,
        })

        return metadata
    
# ======================================================================
# Biospecimen
# ======================================================================


class BiospecimenPreprocessor(BasePreprocessor):
    """
    Preprocessing for biospecimen data.

    Currently implemented:
        SAA status

    Raw representation:
        Positive / Negative / Inconclusive

    Because biospecimen contains multiple files, SAA columns are
    prefixed:

        saa__SAA_Status
    """

    SAA_PREFIX = "saa"

    VALID_SAA_STATUS = {
        "positive",
        "negative",
        "inconclusive",
    }

    def __init__(
        self,
        features_config,
        config=None,
    ):
        super().__init__(config=config)

        self.features_config = features_config

        saa_features = self.get_feature_columns(
            features_config,
            "saa",
            "categorical",
        )

        if len(saa_features) != 1:
            raise ValueError(
                "Expected exactly one SAA categorical "
                f"feature, got: {saa_features}"
            )

        self.saa_raw_feature = saa_features[0]

        self.saa_column = (
            f"{self.SAA_PREFIX}__"
            f"{self.saa_raw_feature}"
        )

        self.encoder = None

    def _normalize_saa(self, df):

        df = df.copy()

        values = (
            df[self.saa_column]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        non_missing = values.notna()

        invalid = (
            non_missing
            & ~values.isin(
                self.VALID_SAA_STATUS
            )
        )

        if invalid.any():

            invalid_values = (
                values.loc[invalid]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                "Unexpected SAA status values: "
                f"{invalid_values}"
            )

        df[self.saa_column] = values

        return df

    def fit(self, df):

        self.validate_columns(
            df,
            [self.saa_column],
        )

        df = self._normalize_saa(df)

        self.encoder = OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )

        self.encoder.fit(
            df[[self.saa_column]]
        )

        self.is_fitted = True

        return self

    def transform(self, df):

        if not self.is_fitted:
            raise RuntimeError(
                "Biospecimen preprocessor has not been fitted."
            )

        df = self._normalize_saa(df)

        transformed = self.encoder.transform(
            df[[self.saa_column]]
        )

        result = pd.DataFrame(
            transformed,
            columns=self.get_feature_names(),
            index=df.index,
        )

        self.validate_no_missing_output(result)
        self.validate_finite_output(result)

        return result

    def get_feature_names(self):

        if self.encoder is None:
            raise RuntimeError(
                "SAA encoder has not been fitted."
            )

        return (
            self.encoder
            .get_feature_names_out(
                [self.saa_column]
            )
            .tolist()
        )

    def get_metadata(self):

        return {
            "preprocessor":
                self.__class__.__name__,
            "saa_input":
                self.saa_column,
            "normalization":
                "strip_and_lowercase",
            "encoding":
                "one_hot",
            "valid_values":
                sorted(self.VALID_SAA_STATUS),
            "output_features":
                self.get_feature_names(),
        }


# ======================================================================
# Genetics
# ======================================================================

class GeneticsPreprocessor(BasePreprocessor):
    """
    Preprocessing for genetic predictors.

    Processing is configuration-driven: only features present in the
    feature configuration are processed.

    Pathogenic variants:
        0       -> non-carrier (0)
        variant -> carrier (1)
        missing -> missing

    APOE:
        genotype -> APOE-e4 carrier 0/1

    PATHVAR_COUNT:
        unchanged

    PRS / Genetic PCs:
        unchanged

    InfPop:
        one-hot encoded

    Because genetics contains multiple files, input columns are prefixed:

        pathogenic_variants__*
        polygenic_risk__*

    Optional feature groups:
        - APOE
        - PATHVAR_COUNT
        - Genetic_PRS_InfPop
        - individual pathogenic-variant genes
        - PRS / PC features

    The YAML configuration determines which features are used.
    This class determines how selected features are transformed.
    """

    PATHVAR_PREFIX = "pathogenic_variants"
    PRS_PREFIX = "polygenic_risk"

    VARIANT_GENES = [
        "LRRK2",
        "GBA",
        "SNCA",
        "PRKN",
        "PARK7",
        "PINK1",
        "VPS35",
    ]

    APOE = "APOE"
    PATHVAR_COUNT = "PATHVAR_COUNT"
    INFPOP = "Genetic_PRS_InfPop"

    def __init__(
        self,
        features_config,
        config=None,
    ):
        super().__init__(config=config)

        self.features_config = features_config

        # --------------------------------------------------------------
        # Raw feature names from YAML
        #
        # get_feature_columns() should return [] when a feature group
        # is absent, empty, or commented out.
        # --------------------------------------------------------------

        self.raw_variant_features = (
            self.get_feature_columns(
                features_config,
                "pathogenic_variants",
                "categorical",
            ) or []
        )

        self.raw_count_features = (
            self.get_feature_columns(
                features_config,
                "pathogenic_variants",
                "continuous",
            ) or []
        )

        self.raw_prs_features = (
            self.get_feature_columns(
                features_config,
                "polygenic_risk",
                "continuous",
            ) or []
        )

        self.raw_infpop_features = (
            self.get_feature_columns(
                features_config,
                "polygenic_risk",
                "categorical",
            ) or []
        )

        # --------------------------------------------------------------
        # Resolve actual ModelDatasetBuilder columns
        # --------------------------------------------------------------

        self.variant_features = self.prefix_features(
            self.raw_variant_features,
            self.PATHVAR_PREFIX,
        )

        self.count_features = self.prefix_features(
            self.raw_count_features,
            self.PATHVAR_PREFIX,
        )

        self.prs_features = self.prefix_features(
            self.raw_prs_features,
            self.PRS_PREFIX,
        )

        self.infpop_features = self.prefix_features(
            self.raw_infpop_features,
            self.PRS_PREFIX,
        )

        # --------------------------------------------------------------
        # Configuration validation
        # --------------------------------------------------------------

        if len(self.raw_count_features) > 1:
            raise ValueError(
                "At most one PATHVAR_COUNT feature may be configured."
            )

        if self.raw_count_features:
            if self.raw_count_features[0] != self.PATHVAR_COUNT:
                raise ValueError(
                    "The continuous pathogenic-variant feature must be "
                    f"'{self.PATHVAR_COUNT}'."
                )

        if len(self.raw_infpop_features) > 1:
            raise ValueError(
                "At most one InfPop feature may be configured."
            )

        if self.raw_infpop_features:
            if self.raw_infpop_features[0] != self.INFPOP:
                raise ValueError(
                    "The categorical polygenic-risk feature must be "
                    f"'{self.INFPOP}'."
                )

        self.encoder = None

    # ------------------------------------------------------------------
    # Carrier encoding
    # ------------------------------------------------------------------

    def _get_carrier_columns(self):
        """
        Return carrier columns for configured pathogenic-variant genes.

        APOE is excluded because it has its own dedicated encoding.
        """

        return [
            f"{feature}_carrier"
            for feature in self.raw_variant_features
            if feature != self.APOE
        ]

    @staticmethod
    def _encode_carrier(series):
        """
        Encode explicit non-carrier / carrier / missing states.

            0       -> 0
            variant -> 1
            missing -> pd.NA

        Missing values are never interpreted as carriers.
        """

        normalized = (
            series
            .astype("string")
            .str.strip()
        )

        result = pd.Series(
            pd.NA,
            index=series.index,
            dtype="Int64",
        )

        missing = (
            normalized.isna()
            | normalized.eq("")
        )

        explicit_zero = normalized.eq("0")

        result.loc[explicit_zero] = 0

        carrier = (
            ~missing
            & ~explicit_zero
        )

        result.loc[carrier] = 1

        return result

    @staticmethod
    def _encode_apoe_e4(series):
        """
        Encode APOE-e4 carrier status.

            E2/E2 -> 0
            E2/E3 -> 0
            E2/E4 -> 1
            E3/E3 -> 0
            E3/E4 -> 1
            E4/E4 -> 1

        Missing values remain missing.
        """

        normalized = (
            series
            .astype("string")
            .str.strip()
            .str.upper()
        )

        result = pd.Series(
            pd.NA,
            index=series.index,
            dtype="Int64",
        )

        missing = (
            normalized.isna()
            | normalized.eq("")
        )

        result.loc[
            ~missing
            & normalized.str.contains(
                "E4",
                regex=False,
                na=False,
            )
        ] = 1

        result.loc[
            ~missing
            & ~normalized.str.contains(
                "E4",
                regex=False,
                na=False,
            )
        ] = 0

        return result

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, df):
        """
        Fit preprocessing components using the configured features.
        """

        required = (
            self.variant_features
            + self.count_features
            + self.prs_features
            + self.infpop_features
        )

        self.validate_columns(
            df,
            required,
        )

        carrier_df = pd.DataFrame(
            index=df.index
        )

        # --------------------------------------------------------------
        # Pathogenic variants
        # --------------------------------------------------------------

        for raw_feature, column in zip(
            self.raw_variant_features,
            self.variant_features,
        ):

            # APOE has dedicated encoding below.
            if raw_feature == self.APOE:
                continue

            carrier_df[
                f"{raw_feature}_carrier"
            ] = self._encode_carrier(
                df[column]
            )

        # --------------------------------------------------------------
        # APOE
        #
        # Only run if APOE was explicitly configured.
        # --------------------------------------------------------------

        if self.APOE in self.raw_variant_features:

            apoe_column = (
                f"{self.PATHVAR_PREFIX}__"
                f"{self.APOE}"
            )

            carrier_df[
                "APOE_e4_carrier"
            ] = self._encode_apoe_e4(
                df[apoe_column]
            )

        # --------------------------------------------------------------
        # Configuration-driven missing-value handling
        #
        # Pathogenic variants and APOE are intentionally excluded.
        # Their missingness has semantic meaning and is handled by
        # their dedicated encoders.
        # --------------------------------------------------------------

        feature_groups = {
            "numeric": (
                self.count_features
                + self.prs_features
            ),
            "categorical": self.infpop_features,
        }

        feature_groups = {
            group: columns
            for group, columns in feature_groups.items()
            if columns
        }

        if feature_groups:

            self.fit_missing_handling(
                df,
                feature_groups,
            )

            df = self.transform_missing_handling(
                df,
                feature_groups,
            )

        # --------------------------------------------------------------
        # Carrier missing-value handling
        #
        # Only run when carrier features actually exist.
        # --------------------------------------------------------------

        if len(carrier_df.columns) > 0:

            self.fit_missing_imputer(
                carrier_df,
                carrier_df.columns,
                "binary",
            )

        # --------------------------------------------------------------
        # InfPop encoder
        #
        # Only create and fit an encoder when InfPop is configured.
        # --------------------------------------------------------------

        self.encoder = None

        if self.infpop_features:

            self.encoder = OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            )

            self.encoder.fit(
                df[self.infpop_features]
            )

        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, df):
        """
        Transform a dataframe using the fitted preprocessing components.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Genetics preprocessor has not been fitted."
            )

        required = (
            self.variant_features
            + self.count_features
            + self.prs_features
            + self.infpop_features
        )

        self.validate_columns(
            df,
            required,
        )

        df = df.copy()

        # --------------------------------------------------------------
        # Apply fitted missing-value handling
        #
        # Pathogenic variants and APOE remain untouched because their
        # missingness is handled by their dedicated encoders.
        # --------------------------------------------------------------

        feature_groups = {
            "numeric": (
                self.count_features
                + self.prs_features
            ),
            "categorical": self.infpop_features,
        }

        feature_groups = {
            group: columns
            for group, columns in feature_groups.items()
            if columns
        }

        if feature_groups:

            df = self.transform_missing_handling(
                df,
                feature_groups,
            )

        output = pd.DataFrame(
            index=df.index
        )

        # --------------------------------------------------------------
        # Pathogenic variants
        # --------------------------------------------------------------

        for raw_feature, column in zip(
            self.raw_variant_features,
            self.variant_features,
        ):

            if raw_feature == self.APOE:
                continue

            output[
                f"{raw_feature}_carrier"
            ] = self._encode_carrier(
                df[column]
            )

        # --------------------------------------------------------------
        # APOE
        #
        # Only run if APOE was configured.
        # --------------------------------------------------------------

        if self.APOE in self.raw_variant_features:

            apoe_column = (
                f"{self.PATHVAR_PREFIX}__"
                f"{self.APOE}"
            )

            output[
                "APOE_e4_carrier"
            ] = self._encode_apoe_e4(
                df[apoe_column]
            )

        # --------------------------------------------------------------
        # Carrier missing-value handling
        # --------------------------------------------------------------

        carrier_columns = (
            self._get_carrier_columns()
        )

        if self.APOE in self.raw_variant_features:
            carrier_columns.append(
                "APOE_e4_carrier"
            )

        if carrier_columns:

            output = self.transform_missing(
                output,
                carrier_columns,
                "binary",
            )

        # --------------------------------------------------------------
        # PATHVAR_COUNT
        #
        # Only run if configured.
        # --------------------------------------------------------------

        if self.count_features:

            count_column = self.count_features[0]

            output[
                self.PATHVAR_COUNT
            ] = df[count_column]

        # --------------------------------------------------------------
        # PRS + PCs
        #
        # Passed through unchanged.
        # --------------------------------------------------------------

        for column in self.prs_features:

            output[
                column
            ] = df[column]

        # --------------------------------------------------------------
        # InfPop
        #
        # Only run if configured and fitted.
        # --------------------------------------------------------------

        if (
            self.infpop_features
            and self.encoder is not None
        ):

            encoded = self.encoder.transform(
                df[self.infpop_features]
            )

            encoded_df = pd.DataFrame(
                encoded,
                columns=(
                    self.encoder
                    .get_feature_names_out(
                        self.infpop_features
                    )
                ),
                index=df.index,
            )

            output = pd.concat(
                [
                    output,
                    encoded_df,
                ],
                axis=1,
            )

        # --------------------------------------------------------------
        # Validation
        # --------------------------------------------------------------

        carrier_columns = (
            self._get_carrier_columns()
        )

        if self.APOE in self.raw_variant_features:
            carrier_columns.append(
                "APOE_e4_carrier"
            )

        for column in carrier_columns:

            values = (
                output[column]
                .dropna()
                .unique()
            )

            if not set(values).issubset(
                {0, 1}
            ):
                raise ValueError(
                    f"Invalid carrier values in "
                    f"{column}: {values}"
                )

        return output

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def get_feature_names(self):
        """
        Return the final transformed feature names.

        The returned features depend entirely on the active feature
        configuration.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "Genetics preprocessor has not been fitted."
            )

        feature_names = []

        # --------------------------------------------------------------
        # Variant carrier features
        # --------------------------------------------------------------

        feature_names.extend(
            self._get_carrier_columns()
        )

        # --------------------------------------------------------------
        # APOE
        # --------------------------------------------------------------

        if self.APOE in self.raw_variant_features:

            feature_names.append(
                "APOE_e4_carrier"
            )

        # --------------------------------------------------------------
        # PATHVAR_COUNT
        # --------------------------------------------------------------

        if self.count_features:

            feature_names.append(
                self.PATHVAR_COUNT
            )

        # --------------------------------------------------------------
        # PRS / PCs
        # --------------------------------------------------------------

        feature_names.extend(
            self.prs_features
        )

        # --------------------------------------------------------------
        # InfPop
        # --------------------------------------------------------------

        if self.encoder is not None:

            feature_names.extend(
                self.encoder
                .get_feature_names_out(
                    self.infpop_features
                )
                .tolist()
            )

        return feature_names

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self):

        metadata = super().get_metadata()

        metadata.update({
            "pathogenic_variants":
                "binary_carrier",

            "apoe":
                "e4_carrier",

            "pathvar_count":
                "unchanged",

            "prs":
                "unchanged",

            "genetic_pcs":
                "unchanged",

            "infpop":
                "one_hot",

            "missing_variant_values":
                "preserved_as_missing",
        })

        return metadata
