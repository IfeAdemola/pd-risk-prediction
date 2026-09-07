# scripts/test_splitter.py

from pathlib import Path

import yaml

from pd_risk.modelling.dataset import (
    ModelDatasetBuilder,
)

from pd_risk.modelling.splitting import (
    CrossValidationSplitter,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_prediction_config():

    config_file = (
        PROJECT_ROOT
        / "configs"
        / "prediction"
        / "ppmi.yaml"
    )

    with open(
        config_file,
        "r",
        encoding="utf-8",
    ) as f:

        return yaml.safe_load(f)


def main():

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SPLITTER TEST")
    print("=" * 70)

    # --------------------------------------------------------------
    # Load configuration
    # --------------------------------------------------------------

    config = load_prediction_config()

    # --------------------------------------------------------------
    # Build model dataset
    # --------------------------------------------------------------

    builder = ModelDatasetBuilder(
        prediction_config=config,
        project_root=PROJECT_ROOT,
    )

    builder.build()

    X = builder.identifiers
    y = builder.targets_df

    print("\nDataset")
    print("-" * 70)

    print(
        "X type:",
        type(X),
    )

    print(
        "X shape:",
        X.shape,
    )

    print(
        "y type:",
        type(y),
    )

    print(
        "y shape:",
        y.shape,
    )

    print(
        "Participants:",
        len(X),
    )

    print(
        "Events:",
        int(y["event"].sum()),
    )

    print(
        "Non-events:",
        int((y["event"] == 0).sum()),
    )

    # --------------------------------------------------------------
    # Validation configuration
    # --------------------------------------------------------------

    validation_config = (
        config["analysis"]["validation"]
    )

    print("\nValidation configuration")
    print("-" * 70)

    print(
        validation_config
    )

    # --------------------------------------------------------------
    # Initialise splitter
    # --------------------------------------------------------------

    splitter = CrossValidationSplitter(
        validation_config
    )

    # --------------------------------------------------------------
    # Generate folds
    # --------------------------------------------------------------

    folds = list(
        splitter.split(
            X,
            y["event"],
        )
    )

    print("\nFolds")
    print("-" * 70)

    assert len(folds) == validation_config["folds"], (
        "Number of generated folds does not match "
        "the configured number of folds."
    )

    # --------------------------------------------------------------
    # Fold checks
    # --------------------------------------------------------------

    validation_indices = []

    overall_event_rate = (
        y["event"].mean()
    )

    print(
        f"Overall event rate: "
        f"{overall_event_rate:.3f}"
    )

    for fold, (train_idx, val_idx) in enumerate(
        folds,
        start=1,
    ):

        train_events = (
            y.iloc[train_idx]["event"]
            .sum()
        )

        val_events = (
            y.iloc[val_idx]["event"]
            .sum()
        )

        train_event_rate = (
            y.iloc[train_idx]["event"]
            .mean()
        )

        val_event_rate = (
            y.iloc[val_idx]["event"]
            .mean()
        )

        print(
            f"\nFold {fold}"
        )

        print(
            "  Train:",
            len(train_idx),
            "events:",
            int(train_events),
            f"event rate: {train_event_rate:.3f}",
        )

        print(
            "  Validation:",
            len(val_idx),
            "events:",
            int(val_events),
            f"event rate: {val_event_rate:.3f}",
        )

        # Train and validation must not overlap.

        assert not (
            set(train_idx)
            & set(val_idx)
        ), (
            f"Fold {fold} contains overlapping "
            "train and validation indices."
        )

        # Every fold must contain observations.

        assert len(train_idx) > 0
        assert len(val_idx) > 0

        validation_indices.extend(
            val_idx.tolist()
        )

    # --------------------------------------------------------------
    # PATNO tracking
    # --------------------------------------------------------------

    print("\nPATNO tracking")
    print("-" * 70)

    assert "PATNO" in X.columns, (
        "PATNO is missing from the splitter input."
    )

    print(
        "✓ PATNO is available for participant tracking."
    )

    for fold, (train_idx, val_idx) in enumerate(
        folds,
        start=1,
    ):

        train_patnos = (
            X.iloc[train_idx]["PATNO"]
        )

        val_patnos = (
            X.iloc[val_idx]["PATNO"]
        )

        print(
            f"\nFold {fold}"
        )

        print(
            "  Train PATNOs:",
            len(train_patnos),
        )

        print(
            "  Validation PATNOs:",
            len(val_patnos),
        )

        assert not (
            set(train_patnos)
            & set(val_patnos)
        ), (
            f"Fold {fold} contains the same PATNO "
            "in both train and validation."
        )

    # --------------------------------------------------------------
    # Cross-fold validation coverage
    # --------------------------------------------------------------

    expected_indices = list(
        range(len(X))
    )

    assert sorted(validation_indices) == expected_indices, (
        "Validation folds do not contain every participant "
        "exactly once."
    )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "ALL CROSS-VALIDATION SPLITTER TESTS PASSED"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()