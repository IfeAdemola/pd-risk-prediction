import pandas as pd
from pathlib import Path

class MissingnessAnalyzer:
    """
    Descriptive analysis of missing data.

    The analyzer is modality-agnostic and can be
    applied to clinical, imaging, genetic, or
    other feature matrices.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        identifiers: pd.DataFrame = None,
        y: pd.DataFrame = None,
        modality: str = None,
        config: dict = None,
    ):

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame"
            )

        self.X = X.copy()

        self.identifiers = identifiers

        self.y = y

        self.modality = modality

        self.config = config or {}

        self.feature_summary_df = None

        self.participant_summary_df = None

        self.pattern_summary_df = None


    def feature_summary(self):
        """
        Summarise missingness for each feature.
        """

        n = len(self.X)

        missing = self.X.isna().sum()

        summary = pd.DataFrame(
            {
                "feature":
                    self.X.columns,

                "n":
                    n,

                "n_missing":
                    missing.values,

                "n_observed":
                    (
                        n - missing
                    ).values,

                "missing_rate":
                    (
                        missing / n
                    ).values,
            }
        )

        summary["observed_rate"] = (
            1
            - summary["missing_rate"]
        )

        summary["modality"] = (
            self.modality
        )

        self.feature_summary_df = (
            summary
        )

        return summary


    def participant_summary(self):
        """
        Summarise missingness for each participant.
        """

        n_features = (
            self.X.shape[1]
        )

        n_missing = (
            self.X.isna().sum(
                axis=1
            )
        )

        summary = pd.DataFrame(
            {
                "n_features":
                    n_features,

                "n_missing":
                    n_missing,

                "n_observed":
                    (
                        n_features
                        - n_missing
                    ),

                "missing_rate":
                    (
                        n_missing
                        / n_features
                    ),
            }
        )

        if self.identifiers is not None:

            identifiers = (
                self.identifiers
                .reset_index(
                    drop=True
                )
            )

            summary = pd.concat(
                [
                    identifiers,
                    summary.reset_index(
                        drop=True
                    ),
                ],
                axis=1,
            )

        summary["modality"] = (
            self.modality
        )

        self.participant_summary_df = (
            summary
        )

        return summary


    def missingness_patterns(self):
        """
        Identify and summarise recurring
        feature-level missingness patterns.
        """

        pattern = (
            self.X.isna()
            .astype(int)
        )

        pattern_strings = (
            pattern.astype(str)
            .agg(
                "".join,
                axis=1,
            )
        )

        pattern_counts = (
            pattern_strings
            .value_counts()
            .rename_axis(
                "pattern"
            )
            .reset_index(
                name="n"
            )
        )

        pattern_counts["proportion"] = (
            pattern_counts["n"]
            / len(self.X)
        )

        feature_names = (
            self.X.columns.tolist()
        )

        pattern_counts[
            "missing_features"
        ] = pattern_counts[
            "pattern"
        ].apply(
            lambda pattern: [
                feature
                for feature, value
                in zip(
                    feature_names,
                    pattern,
                )
                if value == "1"
            ]
        )

        pattern_counts[
            "n_missing_features"
        ] = pattern_counts[
            "missing_features"
        ].apply(len)

        max_patterns = (
            self.config
            .get(
                "patterns",
                {}
            )
            .get(
                "max_patterns",
                20,
            )
        )

        summary = (
            pattern_counts
            .head(max_patterns)
            .reset_index(
                drop=True
            )
        )

        summary["modality"] = (
            self.modality
        )

        self.pattern_summary_df = (
            summary
        )

        return summary

    def cooccurrence(self) -> pd.DataFrame:
        """
        Calculate pairwise missingness co-occurrence.

        Each cell contains the number of observations for which
        both features are missing.

        Returns
        -------
        pd.DataFrame
            Symmetric feature-by-feature matrix.

            Diagonal:
                Number of missing values for that feature.

            Off-diagonal:
                Number of rows where both features are missing.
        """

        missing = self.X.isna() # data[X.features]

        cooccurrence = (
            missing.astype(int).T
            @ missing.astype(int)
        )

        return cooccurrence

    def cooccurrence_similarity(self):
        """
        Calculate Jaccard similarity between feature
        missingness indicators.

        The Jaccard similarity between two features is:

            |A ∩ B|
            -------
            |A ∪ B|

        where A and B are the sets of participants for
        whom each feature is missing.

        Returns
        -------
        pandas.DataFrame
            Symmetric feature-by-feature Jaccard similarity
            matrix.
        """

        missing = self.X.isna()

        features = list(missing.columns)

        similarity = pd.DataFrame(
            0.0,
            index=features,
            columns=features,
        )

        for feature_a in features:

            missing_a = missing[feature_a]

            for feature_b in features:

                missing_b = missing[feature_b]

                intersection = (
                    missing_a
                    & missing_b
                ).sum()

                union = (
                    missing_a
                    | missing_b
                ).sum()

                if union == 0:
                    similarity.loc[
                        feature_a,
                        feature_b,
                    ] = 0.0

                else:
                    similarity.loc[
                        feature_a,
                        feature_b,
                    ] = (
                        intersection / union
                    )

        return similarity

    def save_results(
    self,
    output_directory,
    ):
        """
        Save missingness analysis results to CSV files.

        Parameters
        ----------
        output_directory:
            Directory in which missingness analysis
            outputs will be written.

        Returns
        -------
        dict
            Mapping of result names to output paths.
        """

        output_directory = (
            Path(output_directory)
            / self.modality
        )

        output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {}

        # -------------------------
        # Feature-level missingness
        # -------------------------

        feature_missingness = (
            self.feature_summary()
        )

        feature_path = (
            output_directory
            / "feature_missingness.csv"
        )

        feature_missingness.to_csv(
            feature_path,
            index=False,
        )

        paths["feature_missingness"] = (
            feature_path
        )

        # -------------------------
        # Participant-level missingness
        # -------------------------

        participant_missingness = (
            self.participant_summary()
        )

        participant_path = (
            output_directory
            / "participant_missingness.csv"
        )

        participant_missingness.to_csv(
            participant_path,
            index=False,
        )

        paths["participant_missingness"] = (
            participant_path
        )

        # -------------------------
        # Missingness patterns
        # -------------------------

        patterns = (
            self.missingness_patterns()
        )

        pattern_path = (
            output_directory
            / "missingness_patterns.csv"
        )

        patterns.to_csv(
            pattern_path,
            index=False,
        )

        paths["patterns"] = pattern_path

        # -------------------------
        # Missingness co-occurrence
        # -------------------------

        cooccurrence = (
            self.cooccurrence()
        )

        cooccurrence_path = (
            output_directory
            / "missingness_cooccurrence.csv"
        )

        cooccurrence.to_csv(
            cooccurrence_path
        )

        paths["cooccurrence"] = (
            cooccurrence_path
        )

        # -------------------------
        # Jaccard similarity
        # -------------------------

        similarity = (
            self.cooccurrence_similarity()
        )

        similarity_path = (
            output_directory
            / "missingness_jaccard.csv"
        )

        similarity.to_csv(
            similarity_path
        )

        paths["jaccard"] = (
            similarity_path
        )

        return paths