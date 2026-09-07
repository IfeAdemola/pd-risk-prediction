from pathlib import Path
import yaml

from pd_risk.prediction import PPMIPredictionDatasetBuilder


def main():

    # Move this load outcomes file to the builder
    with open(
        "configs/prediction/ppmi.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        config = yaml.safe_load(f)


    builder = PPMIPredictionDatasetBuilder(
        config
    )

    builder.build()
    builder.save()

    import pprint

    pprint.pp(builder.metadata)

    print(builder.features_df.shape)

    print(builder.feature_columns)

    print(builder.continuous_features)

    print(builder.binary_features)

    print(builder.categorical_features)



    
if __name__ == "__main__":
    main()