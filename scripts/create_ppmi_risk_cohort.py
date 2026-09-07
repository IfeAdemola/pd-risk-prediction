"""
Loads curated clinical file
Filters participants
Applies criteria
Saves cohort files
"""

from pd_risk.utils.config import load_yaml
from pd_risk.data.cohort import PPMICohortBuilder


def main():

    dataset_config = load_yaml(
        "configs/datasets/ppmi.yaml"
    )

    cohort_config = load_yaml(
        "configs/cohort/ppmi.yaml"
    )

    builder = PPMICohortBuilder(
        dataset_config,
        cohort_config,
    )

    builder.build()
    builder.save()
    builder.summary()


if __name__ == "__main__":
    main()