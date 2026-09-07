import yaml
import pandas as pd

from pd_risk.modelling.targets import SurvivalTargetBuilder


def main():

    # -------------------------
    # Load prediction dataset
    # -------------------------

    prediction_file = (
        "data/processed/prediction/"
        "ppmi_prediction_dataset_v1.csv"
    )

    df = pd.read_csv(
        prediction_file
    )


    # -------------------------
    # Load config
    # -------------------------

    config_file = (
        "configs/prediction/ppmi.yaml"
    )

    with open(
        config_file,
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)


    # -------------------------
    # Build targets
    # -------------------------

    builder = SurvivalTargetBuilder(
        df=df,
        config=config,
    )


    targets = builder.build()


    # -------------------------
    # Inspect
    # -------------------------

    print("\nTarget shape")
    print("----------------")
    print(targets.shape)


    print("\nColumns")
    print("----------------")
    print(targets.columns.tolist())


    print("\nHead")
    print("----------------")
    print(targets.head())


    print("\nEvent distribution")
    print("----------------")
    print(
        targets["event"]
        .value_counts()
    )


    print("\nSummary")
    print("----------------")
    print(
        targets["time_to_event"]
        .describe()
    )


    print("\nMissing values")
    print("----------------")
    print(
        targets.isna()
        .sum()
    )

    print(
        pd.crosstab(
            df["pd_case_type"],
            targets["event"]
        )
    )


if __name__ == "__main__":
    main()