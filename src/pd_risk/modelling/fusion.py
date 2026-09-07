from __future__ import annotations

import pandas as pd

def build_fusion(config):

    fusion_config = config.get(
        "fusion",
        {},
    )

    strategy = fusion_config.get(
        "strategy"
    )

    if strategy is None:
        raise ValueError(
            "No fusion strategy configured. "
            "Expected 'fusion.strategy'."
        )


    if strategy == "early":
        return EarlyFusion(
        )

    raise ValueError(
        f"Unsupported fusion strategy: "
        f"'{strategy}'."
    )


class BaseFusion:
    """
    Base interface for multimodal feature fusion.

    Fusion operates on already-preprocessed modality data.
    Splitting and preprocessing are handled upstream.
    """

    def __init__(self):
        self.is_fitted = False

    def fit(self, modality_data):
        raise NotImplementedError

    def transform(self, modality_data):
        raise NotImplementedError

    def fit_transform(self, modality_data):
        self.fit(modality_data)
        return self.transform(modality_data)

    def get_feature_names(self):
        raise NotImplementedError

    def get_metadata(self):
        raise NotImplementedError


class EarlyFusion(BaseFusion):
    """
    Early fusion through column-wise concatenation.

    Parameters
    ----------
    modality_data:
        Dictionary mapping modality names to preprocessed
        pandas DataFrames.

        Example:

            {
                "clinical": clinical_X,
                "mri": mri_X,
                "dat": dat_X,
                "genetics": genetics_X,
            }

    Notes
    -----
    Each modality must already have been preprocessed.

    Fusion does not:
        - split data
        - fit preprocessing
        - modify modality values
        - use PATNO as a model feature
    """

    def __init__(
        self,
    ):
        super().__init__()


        self.modality_names = []
        self.feature_names = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_modality_data(modality_data):

        if not isinstance(modality_data, dict):
            raise TypeError(
                "modality_data must be a dictionary mapping "
                "modality names to pandas DataFrames."
            )

        if not modality_data:
            raise ValueError(
                "No modality data provided for fusion."
            )

        for modality_name, df in modality_data.items():

            if not isinstance(df, pd.DataFrame):
                raise TypeError(
                    f"Modality '{modality_name}' must be a "
                    "pandas DataFrame."
                )

            if df.empty:
                raise ValueError(
                    f"Modality '{modality_name}' contains "
                    "no rows."
                )

            if df.columns.duplicated().any():

                duplicated = (
                    df.columns[
                        df.columns.duplicated()
                    ]
                    .tolist()
                )

                raise ValueError(
                    f"Modality '{modality_name}' contains "
                    f"duplicate feature columns: {duplicated}"
                )

    @staticmethod
    def _validate_alignment(modality_data):

        modality_names = list(
            modality_data.keys()
        )

        reference_name = modality_names[0]

        reference = modality_data[
            reference_name
        ]

        for modality_name in modality_names[1:]:

            current = modality_data[
                modality_name
            ]

            if len(current) != len(reference):

                raise ValueError(
                    "Modality row counts do not match: "
                    f"'{reference_name}' has "
                    f"{len(reference)} rows, while "
                    f"'{modality_name}' has "
                    f"{len(current)} rows."
                )

            if not current.index.equals(
                reference.index
            ):

                raise ValueError(
                    "Modality indices are not aligned: "
                    f"'{reference_name}' and "
                    f"'{modality_name}' have different "
                    "row indices."
                )

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, modality_data):

        self._validate_modality_data(
            modality_data
        )

        self._validate_alignment(
            modality_data
        )

        self.modality_names = list(
            modality_data.keys()
        )

        feature_names = []

        for modality_name in (
            self.modality_names
        ):

            df = modality_data[
                modality_name
            ]

            feature_names.extend(
                df.columns.tolist()
            )

        duplicated = (
            pd.Index(feature_names)
            .duplicated()
        )

        if duplicated.any():

            duplicate_features = (
                pd.Index(feature_names)[
                    duplicated
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                "Duplicate feature names detected "
                "across modalities: "
                f"{duplicate_features}"
            )

        self.feature_names = feature_names

        self.is_fitted = True

        return self

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------

    def transform(self, modality_data):

        if not self.is_fitted:

            raise RuntimeError(
                "Early fusion has not been fitted."
            )

        self._validate_modality_data(
            modality_data
        )

        self._validate_alignment(
            modality_data
        )

        received_modalities = list(
            modality_data.keys()
        )

        if received_modalities != (
            self.modality_names
        ):

            raise ValueError(
                "Modality set does not match the "
                "modalities used during fitting. "
                f"Expected {self.modality_names}, "
                f"got {received_modalities}."
            )

        X = pd.concat(
            [
                modality_data[
                    modality_name
                ]
                for modality_name
                in self.modality_names
            ],
            axis=1,
        )

        if X.columns.tolist() != (
            self.feature_names
        ):

            raise ValueError(
                "Fused feature columns do not match "
                "the feature columns learned during fitting."
            )

        return X

    # ------------------------------------------------------------------
    # Feature names
    # ------------------------------------------------------------------

    def get_feature_names(self):

        if not self.is_fitted:

            raise RuntimeError(
                "Early fusion has not been fitted."
            )

        return list(
            self.feature_names
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self):

        return {
            "fusion":
                self.__class__.__name__,

            "strategy":
                "early_fusion",

            "modalities":
                list(self.modality_names),

            "n_modalities":
                len(self.modality_names),

            "n_features":
                len(self.feature_names),

            "output_features":
                list(self.feature_names),

            "fitted":
                self.is_fitted,
        }
