# python scripts/run_experiment.py

import yaml
from pathlib import Path

from pd_risk.modelling.dataset import ModelDatasetBuilder

from pd_risk.modelling.preprocessing import (
    ClinicalPreprocessor,
    DATPreprocessor,
    MRIPreprocessor,
    BiospecimenPreprocessor,
    GeneticsPreprocessor,
)

from pd_risk.modelling.fusion import build_fusion

from pd_risk.modelling.survival import (
    CauseSpecificCoxModel,
)

from pd_risk.modelling.splitting import (
    CrossValidationSplitter,
)

from pd_risk.modelling.evaluation import (
    SurvivalEvaluator,
)

from pd_risk.modelling.experiment import (
    SurvivalExperiment,
)

from pd_risk.modelling.artifacts import (
    ExperimentArtifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "prediction"
    / "ppmi.yaml"
)


# =========================================================
# Configuration
# =========================================================

def load_config():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


# =========================================================
# Dataset
# =========================================================

def build_dataset(config):

    builder = ModelDatasetBuilder(
        prediction_config=config,
        project_root=PROJECT_ROOT,
    )

    builder.build()

    return builder


# =========================================================
# Enabled modalities
# =========================================================

def get_enabled_modalities(config):

    return [
        modality
        for modality, modality_config
        in config["modalities"].items()
        if modality_config.get(
            "enabled",
            False,
        )
    ]


# =========================================================
# Feature configuration
# =========================================================

def load_features_config(
    config,
    modality,
):

    features_file = Path(
        config["modalities"][modality][
            "features_file"
        ]
    )

    if not features_file.is_absolute():

        features_file = (
            PROJECT_ROOT
            / features_file
        )

    with open(
        features_file,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


# =========================================================
# Preprocessors
# =========================================================

def build_preprocessor_factories(
    config,
    builder,
    enabled_modalities,
):

    def preprocessor_factory(
        modality,
    ):

        modality_metadata = (
            builder.modality_features[
                modality
            ]
        )

        types = modality_metadata[
            "types"
        ]

        # -------------------------------------------------
        # Clinical
        # -------------------------------------------------

        if modality == "clinical":

            return ClinicalPreprocessor(
                continuous_features=(
                    types["continuous"]
                ),

                binary_features=(
                    types["binary"]
                ),

                categorical_features=(
                    types["categorical"]
                ),

                config=config[
                    "preprocessing"
                ],
            )

        # -------------------------------------------------
        # Feature-config modalities
        # -------------------------------------------------

        features_config = (
            load_features_config(
                config,
                modality,
            )
        )

        preprocessing_config = (
            config.get(
                "preprocessing"
            )
        )

        if modality == "dat":

            return DATPreprocessor(
                features_config=(
                    features_config
                ),

                config=(
                    preprocessing_config
                ),
            )

        if modality == "mri":

            return MRIPreprocessor(
                features_config=(
                    features_config
                ),

                config=(
                    preprocessing_config
                ),
            )

        if modality == "biospecimen":

            return BiospecimenPreprocessor(
                features_config=(
                    features_config
                ),

                config=(
                    preprocessing_config
                ),
            )

        if modality == "genetics":

            return GeneticsPreprocessor(
                features_config=(
                    features_config
                ),

                config=(
                    preprocessing_config
                ),
            )

        raise ValueError(
            f"No preprocessor defined for "
            f"modality '{modality}'."
        )

    return {
        modality: (
            lambda modality=modality:
            preprocessor_factory(
                modality
            )
        )

        for modality
        in enabled_modalities
    }


# =========================================================
# Experiment
# =========================================================

def build_experiment(
    config,
    builder,
    enabled_modalities,
):

    # -----------------------------------------------------
    # Modality data
    # -----------------------------------------------------

    modality_data = {
        modality:
            builder.modality_data[
                modality
            ].copy()

        for modality
        in enabled_modalities
    }

    # -----------------------------------------------------
    # Targets
    # -----------------------------------------------------

    y = builder.targets_df[
        [
            "PATNO",
            "time_to_event",
            "event_type",
        ]
    ].copy()

    # -----------------------------------------------------
    # Identifiers
    # -----------------------------------------------------

    identifiers = (
        builder.identifiers.copy()
    )

    # -----------------------------------------------------
    # Preprocessors
    # -----------------------------------------------------

    preprocessor_factories = (
        build_preprocessor_factories(
            config=config,
            builder=builder,
            enabled_modalities=(
                enabled_modalities
            ),
        )
    )

    # -----------------------------------------------------
    # Fusion
    # -----------------------------------------------------

    fusion_factory = (
        lambda: build_fusion(config)
    )

    # -----------------------------------------------------
    # Cause-specific Cox model
    # -----------------------------------------------------

    model_factory = (
        lambda: CauseSpecificCoxModel(
            penalizer=(
                config[
                    "analysis"
                ][
                    "model"
                ][
                    "penalizer"
                ]
            )
        )
    )

    # -----------------------------------------------------
    # Validation
    # -----------------------------------------------------

    splitter = CrossValidationSplitter(
        config[
            "analysis"
        ][
            "validation"
        ]
    )

    # -----------------------------------------------------
    # Evaluator
    # -----------------------------------------------------

    evaluator = SurvivalEvaluator(n_risk_groups=config["analysis"]["calibration"]["n_groups"])

    # -----------------------------------------------------
    # Experiment
    # -----------------------------------------------------

    return SurvivalExperiment(

        modality_data=modality_data,

        y=y,

        identifiers=identifiers,

        splitter=splitter,

        preprocessor_factories=(
            preprocessor_factories
        ),

        fusion_factory=fusion_factory,

        model_factory=model_factory,

        evaluator=evaluator,

        config=config,
    )


# =========================================================
# Reporting
# =========================================================

def print_completion_report(
    experiment,
    experiment_dir,
):

    summary = (
        experiment.summary
    )

    dataset = (
        summary["dataset"]
    )

    performance = (
        summary["performance"]
    )

    print()
    print("=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)

    print(
        f"Experiment directory: "
        f"{experiment_dir}"
    )

    print()

    print(
        f"Samples: "
        f"{dataset['n_samples']}"
    )

    print(
        f"Features: "
        f"{dataset['n_features']}"
    )

    print(
        f"PD events: "
        f"{dataset['n_pd_events']}"
    )

    print(
        f"Deaths: "
        f"{dataset['n_deaths']}"
    )

    print(
        f"Censored: "
        f"{dataset['n_censored']}"
    )

    print(
        f"Horizon: "
        f"{dataset['time_horizon_years']} years"
    )

    print()

    if "uno_c_index" in performance:

        print(
            "Mean Uno C-index: "
            f"{performance['uno_c_index']['mean']:.4f}"
            f" ± {performance['uno_c_index']['std']:.4f}"
        )

    if "brier_score" in performance:

        print(
            "Mean Brier score: "
            f"{performance['brier_score']['mean']:.4f}"
            f" ± {performance['brier_score']['std']:.4f}"
        )

    if "null_brier_score" in performance:

        print(
            "Mean null Brier score: "
            f"{performance['null_brier_score']['mean']:.4f}"
            f" ± {performance['null_brier_score']['std']:.4f}"
        )

    if "brier_skill_score" in performance:

        print(
            "Mean Brier skill score: "
            f"{performance['brier_skill_score']['mean']:.4f}"
            f" ± {performance['brier_skill_score']['std']:.4f}"
        )

    print()
    print("=" * 60)


# =========================================================
# Main
# =========================================================

def main():

    # -----------------------------------------------------
    # Load configuration
    # -----------------------------------------------------

    config = load_config()

    # -----------------------------------------------------
    # Determine modalities
    # -----------------------------------------------------

    enabled_modalities = (
        get_enabled_modalities(
            config
        )
    )

    print()
    print("Enabled modalities:")

    for modality in enabled_modalities:

        print(
            f"  - {modality}"
        )

    # -----------------------------------------------------
    # Build dataset
    # -----------------------------------------------------

    builder = build_dataset(
        config
    )

    # -----------------------------------------------------
    # Build experiment
    # -----------------------------------------------------

    experiment = build_experiment(
        config=config,
        builder=builder,
        enabled_modalities=(
            enabled_modalities
        ),
    )

    # -----------------------------------------------------
    # Run experiment
    # -----------------------------------------------------

    experiment.run()

    # -----------------------------------------------------
    # Save artifacts
    # -----------------------------------------------------

    artifacts = ExperimentArtifacts(
        experiment=experiment,
        config=config,
    )

    experiment_dir = (
        artifacts.save()
    )

    # -----------------------------------------------------
    # Report
    # -----------------------------------------------------

    print_completion_report(
        experiment=experiment,
        experiment_dir=(
            experiment_dir
        ),
    )


if __name__ == "__main__":

    main()