from pd_risk.utils.config import load_yaml
from pd_risk.outcomes import PPMIOutcomeBuilder


def main():

    outcome_config = load_yaml(
        "configs/outcomes/ppmi.yaml"
    )

    builder = PPMIOutcomeBuilder(
        "data/processed/cohort/ppmi_risk_cohort_longitudinal.csv",
        outcome_config,
    )

    builder.build()
    builder.save()

    builder.summary()

if __name__ == "__main__":
    main()