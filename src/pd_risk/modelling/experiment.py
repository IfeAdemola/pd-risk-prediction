from typing import Callable

import numpy as np
import pandas as pd


class SurvivalExperiment:
    """
    Orchestrate a competing-risk survival experiment.

    Statistical modelling is delegated to the survival model
    and evaluation is delegated to SurvivalEvaluator.

    The experiment operates on event_type:

        0 = censored
        1 = PD
        2 = death before PD
    """

    def __init__(
        self,
        modality_data,
        y,
        identifiers,
        splitter,
        preprocessor_factories,
        fusion_factory,
        model_factory,
        evaluator,
        config,
    ):

        self.modality_data = modality_data
        self.y = y.copy()
        self.identifiers = identifiers.copy()

        self.splitter = splitter

        self.preprocessor_factories = (
            preprocessor_factories
        )

        self.fusion_factory = (
            fusion_factory
        )

        self.model_factory = (
            model_factory
        )

        self.evaluator = evaluator

        self.config = config

        # -------------------------
        # Outputs
        # -------------------------

        self.fold_results = []

        self.oof_predictions = None

        self.final_model = None
        self.final_fusion = None
        self.final_preprocessors = {}

        self.summary = None

        self.final_model_summary = None

        self.evaluation = None

    # =========================================================
    # Public API
    # =========================================================

    def run(self):
        """
        Run the complete competing-risk experiment.
        """

        self._validate_inputs()

        self._cross_validate()

        self._build_oof_evaluation()

        self._fit_final_model()

        self._build_summary()

        return self.summary

    # =========================================================
    # Validation
    # =========================================================

    def _validate_inputs(self):
        """
        Validate the experiment inputs.
        """

        required_target_columns = {
            "PATNO",
            "time_to_event",
            "event_type",
        }

        missing = (
            required_target_columns
            - set(self.y.columns)
        )

        if missing:
            raise ValueError(
                "Missing required target columns: "
                f"{sorted(missing)}"
            )

        if len(self.y) != len(self.identifiers):
            raise ValueError(
                "y and identifiers must contain "
                "the same number of rows."
            )

        if len(self.y) == 0:
            raise ValueError(
                "Experiment cohort is empty."
            )

        # -------------------------
        # Validate event types
        # -------------------------

        event_types = set(
            self.y["event_type"].dropna().unique()
        )

        unexpected = (
            event_types - {0, 1, 2}
        )

        if unexpected:
            raise ValueError(
                "Unexpected event_type values: "
                f"{sorted(unexpected)}. "
                "Expected 0=censored, "
                "1=PD, 2=death."
            )

        if self.y["time_to_event"].isna().any():
            raise ValueError(
                "time_to_event contains missing values."
            )

        # -------------------------
        # Validate modalities
        # -------------------------

        if not self.modality_data:
            raise ValueError(
                "No modality data supplied."
            )

        for modality, df in self.modality_data.items():

            if len(df) != len(self.y):
                raise ValueError(
                    f"Modality '{modality}' has "
                    f"{len(df)} rows but targets have "
                    f"{len(self.y)} rows."
                )

            if "PATNO" not in df.columns:
                raise ValueError(
                    f"Modality '{modality}' does not "
                    "contain PATNO."
                )

        # -------------------------
        # Validate factories
        # -------------------------

        for modality in self.modality_data:

            if modality not in (
                self.preprocessor_factories
            ):
                raise ValueError(
                    f"No preprocessor factory for "
                    f"modality '{modality}'."
                )

        if self.fusion_factory is None:
            raise ValueError(
                "fusion_factory is required."
            )

        if self.model_factory is None:
            raise ValueError(
                "model_factory is required."
            )

    # =========================================================
    # Cross-validation
    # =========================================================

    def _cross_validate(self):
        """
        Perform cross-validation and generate
        out-of-fold predictions.
        """

        self.fold_results = []

        oof_predictions = []

        # -----------------------------------------------------
        # Evaluation horizon
        # -----------------------------------------------------

        horizon = (
            self.config
            .get("analysis", {})
            .get("time_horizon", 2)
        )

        # -----------------------------------------------------
        # Splitter
        # -----------------------------------------------------

        # The splitter only needs an array with one row
        # per participant. It does not need to receive
        # the multimodal feature matrix.
        split_reference = np.arange(
            len(self.y)
        )

        for fold, (
            train_idx,
            val_idx,
        ) in enumerate(
            self.splitter.split(
                split_reference,
                self.y["event_type"],
            ),
            start=1,
        ):

            print(
                f"Running fold {fold}"
            )

            # =================================================
            # Split targets
            # =================================================

            y_train = (
                self.y.iloc[train_idx]
                .reset_index(drop=True)
            )

            y_val = (
                self.y.iloc[val_idx]
                .reset_index(drop=True)
            )

            # =================================================
            # Split modalities
            # =================================================

            X_train_modalities = {}
            X_val_modalities = {}

            for modality, df in (
                self.modality_data.items()
            ):

                X_train_modalities[modality] = (
                    df.iloc[train_idx]
                    .reset_index(drop=True)
                    .copy()
                )

                X_val_modalities[modality] = (
                    df.iloc[val_idx]
                    .reset_index(drop=True)
                    .copy()
                )

            # =================================================
            # Preprocess each modality independently
            # =================================================

            train_processed = {}
            val_processed = {}

            fold_preprocessors = {}

            for modality in self.modality_data:

                preprocessor = (
                    self.preprocessor_factories[
                        modality
                    ]()
                )

                X_train = (
                    X_train_modalities[
                        modality
                    ]
                )

                X_val = (
                    X_val_modalities[
                        modality
                    ]
                )

                X_train_processed = (
                    preprocessor
                    .fit_transform(X_train)
                )

                X_val_processed = (
                    preprocessor
                    .transform(X_val)
                )

                train_processed[
                    modality
                ] = X_train_processed

                val_processed[
                    modality
                ] = X_val_processed

                fold_preprocessors[
                    modality
                ] = preprocessor

            # =================================================
            # Fusion
            # =================================================

            fusion = (
                self.fusion_factory()
            )

            X_train_fused = (
                fusion.fit_transform(
                    train_processed
                )
            )

            X_val_fused = (
                fusion.transform(
                    val_processed
                )
            )

            # =================================================
            # Model
            # =================================================

            model = (
                self.model_factory()
            )

            model.fit(
                X_train_fused,
                y_train,
            )

            # =================================================
            # Predictions
            # =================================================

            risk_score = (
                model.predict_risk(
                    X_val_fused
                )
            )

            pd_cif = (
                model.predict_pd_cumulative_incidence(
                    X_val_fused,
                    time=horizon,
                )
            )

            # =================================================
            # OOF predictions
            # =================================================

            fold_predictions = pd.DataFrame(
                {
                    "PATNO":
                        self.identifiers
                        .iloc[val_idx]["PATNO"]
                        .to_numpy(),

                    "fold":
                        fold,

                    "time_to_event":
                        y_val[
                            "time_to_event"
                        ].to_numpy(),


                    "event_type":
                        y_val[
                            "event_type"
                        ].to_numpy(),

                    "risk_score":
                        np.asarray(
                            risk_score
                        ),

                    "pd_cif":
                        np.asarray(
                            pd_cif
                        ),

                }
            )

            oof_predictions.append(
                fold_predictions
            )

            # =================================================
            # Fold evaluation
            # =================================================

            metrics = (
                self.evaluator.evaluate(
                    y_train=y_train,
                    y_test=y_val,
                    risk_scores=risk_score,
                    pd_cif=pd_cif,
                    horizon=horizon,
                )
            )

            fold_result = {
                "fold": fold,

                "train_samples":
                    len(train_idx),

                "validation_samples":
                    len(val_idx),

                "train_pd_events":
                    int(
                        (
                            y_train["event_type"]
                            == 1
                        ).sum()
                    ),

                "validation_pd_events":
                    int(
                        (
                            y_val["event_type"]
                            == 1
                        ).sum()
                    ),

                "train_deaths":
                    int(
                        (
                            y_train["event_type"]
                            == 2
                        ).sum()
                    ),

                "validation_deaths":
                    int(
                        (
                            y_val["event_type"]
                            == 2
                        ).sum()
                    ),

                **metrics,
            }

            self.fold_results.append(
                fold_result
            )

        # =====================================================
        # Combine OOF predictions
        # =====================================================

        self.oof_predictions = (
            pd.concat(
                oof_predictions,
                ignore_index=True,
            )
        )

        # Sanity check: every participant appears once
        if len(self.oof_predictions) != len(
            self.y
        ):
            raise RuntimeError(
                "OOF prediction count does not match "
                "the modelling cohort."
            )

        if (
            self.oof_predictions["PATNO"]
            .nunique()
            != len(self.y)
        ):
            raise RuntimeError(
                "OOF predictions do not contain "
                "exactly one prediction per participant."
            )

    # =========================================================
    # OOF evaluation
    # =========================================================

    def _build_oof_evaluation(self):
        """
        Build all cohort-level evaluation outputs
        from out-of-fold predictions.
        """

        predictions = (
            self.oof_predictions
        )

        horizon = (
            self.config
            .get("analysis", {})
            .get("time_horizon", 2)
        )

        # -----------------------------------------------------
        # Risk groups
        # -----------------------------------------------------

        predictions = (
            self.evaluator
            .assign_risk_groups(
                predictions
            )
        )

        self.oof_predictions = (
            predictions
        )

        # -----------------------------------------------------
        # Risk stratification
        # -----------------------------------------------------

        risk_groups = (
            self.evaluator
            .build_risk_groups(
                predictions,
                horizon=horizon,
            )
        )

        # -----------------------------------------------------
        # Kaplan-Meier / cumulative incidence
        # -----------------------------------------------------

        cumulative_incidence = (
            self.evaluator
            .build_cumulative_incidence(
                predictions,
                horizon=horizon,
                cause=self.evaluator.PD_EVENT,
            )
        )

        # -----------------------------------------------------
        # Gray's test
        # -----------------------------------------------------

        gray_result = (
            self.evaluator
            .build_gray_test(
                predictions,
                cause=1,
            )
        )

        # -----------------------------------------------------
        # Calibration
        # -----------------------------------------------------

        calibration = (
            self.evaluator
            .build_calibration(
                predictions,
                horizon=horizon,
            )
        )

        # -----------------------------------------------------
        # Plots
        # -----------------------------------------------------

        cumulative_incidence_plot = (
            self.evaluator
            .plot_cumulative_incidence(
                predictions,
                horizon=horizon,
            )
        )

        calibration_plot = (
            self.evaluator
            .plot_calibration(
                calibration,
                horizon=horizon,
            )
        )

        self.evaluation = {
            "risk_groups":
                risk_groups,

            "cumulative_incidence":
                cumulative_incidence,

            "gray_test":
                gray_result,

            "calibration":
                calibration,

            "cumulative_incidence_plot":
                cumulative_incidence_plot,

            "calibration_plot":
                calibration_plot,
        }

    # =========================================================
    # Final model
    # =========================================================

    def _fit_final_model(self):
        """
        Fit preprocessing, fusion and competing-risk model
        using the complete modelling cohort.

        This model is NOT used to generate the OOF performance
        metrics. Those metrics come exclusively from CV.
        """

        processed_modalities = {}

        self.final_preprocessors = {}

        # -----------------------------------------------------
        # Preprocessing
        # -----------------------------------------------------

        for modality, df in (
            self.modality_data.items()
        ):

            preprocessor = (
                self.preprocessor_factories[
                    modality
                ]()
            )

            processed_modalities[
                modality
            ] = (
                preprocessor
                .fit_transform(df)
            )

            self.final_preprocessors[
                modality
            ] = preprocessor

        # -----------------------------------------------------
        # Fusion
        # -----------------------------------------------------

        fusion = (
            self.fusion_factory()
        )

        X_fused = (
            fusion.fit_transform(
                processed_modalities
            )
        )

        self.final_fusion = fusion

        # -----------------------------------------------------
        # Final model
        # -----------------------------------------------------

        model = (
            self.model_factory()
        )

        model.fit(
            X_fused,
            self.y,
        )

        self.final_model = model

        # -----------------------------------------------------
        # Model summary
        # -----------------------------------------------------

        self.final_model_summary = (
            model.summary()
        )

    # =========================================================
    # Summary
    # =========================================================

    def _build_summary(self):
        """
        Build a concise experiment summary.

        Performance is based on OOF predictions.
        """

        results_df = pd.DataFrame(
            self.fold_results
        )

        horizon = (
            self.config
            .get("analysis", {})
            .get("time_horizon", 2)
        )

        event_counts = (
            self.y["event_type"]
            .value_counts()
            .to_dict()
        )

        n_features = (
            len(
                self.final_fusion
                .get_feature_names()
            )
        )

        # -----------------------------------------------------
        # Aggregate fold metrics
        # -----------------------------------------------------

        performance = {}

        metric_columns = [
            "uno_c_index",
            "brier_score",
            "null_brier_score",
            "brier_skill_score",
        ]

        for metric in metric_columns:

            if metric in results_df.columns:

                performance[metric] = {
                    "mean":
                        float(
                            results_df[
                                metric
                            ].mean()
                        ),

                    "std":
                        float(
                            results_df[
                                metric
                            ].std()
                        ),
                }

        # -----------------------------------------------------
        # Summary
        # -----------------------------------------------------

        self.summary = {

            "dataset": {

                "n_samples":
                    len(self.y),

                "n_features":
                    n_features,

                "n_censored":
                    int(
                        event_counts.get(
                            0,
                            0,
                        )
                    ),

                "n_pd_events":
                    int(
                        event_counts.get(
                            1,
                            0,
                        )
                    ),

                "n_deaths":
                    int(
                        event_counts.get(
                            2,
                            0,
                        )
                    ),

                "time_horizon_years":
                    horizon,
            },

            "modalities": list(
                self.modality_data.keys()
            ),

            "model": {

                "class":
                    self.final_model
                    .__class__
                    .__name__,

                "configuration":
                    self.config
                    .get(
                        "analysis",
                        {}
                    )
                    .get(
                        "model",
                        {},
                    ),
            },

            "validation": {

                "n_folds":
                    len(results_df),

                "strategy":
                    self.config
                    .get(
                        "analysis",
                        {}
                    )
                    .get(
                        "validation",
                        {}
                    )
                    .get(
                        "strategy"
                    ),
            },

            "performance":
                performance,

            "risk_stratification":
                {
                    "gray_test":
                        (
                            self.evaluation[
                                "gray_test"
                            ]
                            if self.evaluation
                            else None
                        ),
                },
        }

        return self.summary