from pathlib import Path

import yaml

from pd_risk.modelling.dataset import ModelDatasetBuilder
from pd_risk.missingness.association import (
    MissingnessAssociationAnalyzer,
)
from pd_risk.outcomes import builder

def main():

    with open(
        "configs/prediction/ppmi.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        prediction_config = yaml.safe_load(f)

    with open(
        "configs/analysis/missingness.yaml",
        "r",
        encoding="utf-8",
    ) as f:

        missingness_config = yaml.safe_load(f)

    builder = ModelDatasetBuilder(
        prediction_file=(
            "data/processed/prediction/"
            "ppmi_prediction_dataset_v1.csv"
        ),
        prediction_config=prediction_config,
    )

    builder.build()


    analyzer = MissingnessAssociationAnalyzer(
        X=builder.X,
        identifiers=builder.identifiers,
        y=builder.y,
        modality="clinical",
        config=missingness_config["missingness"],
    )

    # --------------------------------------------------------------
    # Outcome association
    # --------------------------------------------------------------

    outcome_results = (
        analyzer.feature_outcome_associations()
    )

    print(
        "\n=== Missingness / outcome associations ==="
    )

    print(
        outcome_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Covariate association
    # --------------------------------------------------------------

    covariates = builder.X.copy()

    covariate_results = (
        analyzer.feature_covariate_associations(
            covariates
        )
    )

    print(
        "\n=== Missingness / covariate associations ==="
    )

    print(
        covariate_results.to_string(
            index=False
        )
    )

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    assert "feature" in outcome_results.columns

    assert (
        "missing_event_rate"
        in outcome_results.columns
    )

    assert "feature" in covariate_results.columns

    assert "covariate" in covariate_results.columns

    print(
        "\n[OK] Association analysis completed."
    )

    # --------------------------------------------------------------
    # Basic validation of FDR correction
    # --------------------------------------------------------------

    print(
        "\n=== FDR-significant missingness / "
        "covariate associations ==="
    )

    fdr_significant = covariate_results[
        covariate_results["significant_fdr"]
    ].copy()

    if fdr_significant.empty:
        print(
            "No associations remain significant "
            "after FDR correction."
        )
    else:
        print(
            fdr_significant[
                [
                    "feature",
                    "modality",
                    "covariate",
                    "test",
                    "p_value",
                    "q_value",
                    "significant_fdr",
                ]
            ].to_string(index=False)
        )

    #--------------------------------------------------------------
    # Basic validation of FDR correction
    #--------------------------------------------------------------
    
    assert "q_value" in covariate_results.columns
    assert "significant_fdr" in covariate_results.columns

    assert covariate_results["q_value"].notna().all()
    assert covariate_results["q_value"].between(0, 1).all()

    assert (
        covariate_results["significant_fdr"].dtype
        == bool
    )

    assert (
        covariate_results.loc[
            covariate_results["significant_fdr"],
            "q_value",
        ]
        < 0.05
    ).all()

if __name__ == "__main__":
    main()