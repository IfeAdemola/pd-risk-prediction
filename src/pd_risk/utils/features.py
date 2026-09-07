def parse_feature_config(feature_config):
    """
    Parse feature configuration.

    Returns:
    - all predictor columns
    - continuous columns
    - binary columns
    - categorical columns
    """

    features = feature_config["features"]

    all_features = []
    continuous = []
    binary = []
    categorical = []


    for domain, types in features.items():

        for feature_type, values in types.items():

            columns = values["columns"]

            all_features.extend(columns)

            if feature_type == "continuous":

                continuous.extend(columns)

            elif feature_type == "binary":

                binary.extend(columns)

            elif feature_type == "categorical":

                categorical.extend(columns)

            else:

                raise ValueError(
                    f"Unknown feature type: {feature_type}"
                )


    return {
        "all": all_features,
        "continuous": continuous,
        "binary": binary,
        "categorical": categorical,
    }