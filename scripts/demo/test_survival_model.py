import yaml

from pd_risk.modelling.dataset import (
    ModelDatasetBuilder
)

from pd_risk.modelling.preprocessing import (
    ClinicalPreprocessor
)

from pd_risk.modelling.survival import (
    CoxPHModel
)


def main():

    # -------------------------
    # Load config
    # -------------------------

    with open(
        "configs/prediction/ppmi.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)


    # -------------------------
    # Build model dataset
    # -------------------------

    builder = ModelDatasetBuilder(
        prediction_file=(
            "data/processed/prediction/"
            "ppmi_prediction_dataset_v1.csv"
        ),
        prediction_config=config,
    )

    builder.build()


    print(
        "Original X:",
        builder.X.shape
    )


    # -------------------------
    # Preprocess
    # -------------------------

    preprocessor = ClinicalPreprocessor(
        continuous_features=(
            builder.continuous_features
        ),

        binary_features=(
            builder.binary_features
        ),

        categorical_features=(
            builder.categorical_features
        ),
)

    X_processed = (
        preprocessor
        .fit_transform(
            builder.X
        )
    )


    print(
        "Processed X:",
        X_processed.shape
    )


    # -------------------------
    # Fit Cox model
    # -------------------------

    model = CoxPHModel()


    model.fit(
        X_processed,
        builder.y,
    )


    print(
        "Model fitted"
    )


    # -------------------------
    # Predict risk
    # -------------------------

    risk_scores = (
        model
        .predict_risk(
            X_processed
        )
    )


    print(
        "Risk scores:"
    )

    print(
        risk_scores[:10]
    )


if __name__ == "__main__":
    main()