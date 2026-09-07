"""
This is to run multiple experiments, changing parameters in the config, at once.
Still a lot of work to be done here.
"""


import itertools
import copy
import subprocess
from pathlib import Path
import yaml


BASE_CONFIG = "configs/config.yaml"
CONFIG_DIR = Path("configs/generated")

MODALITIES = ["clinical", "dat", "mri"]

# Maximum number of experiments to run simultaneously
MAX_PARALLEL = 3


def generate_configs():
    with open(BASE_CONFIG, "r") as f:
        base_config = yaml.safe_load(f)

    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    configs = []

    # Generate combinations of size 1, 2, ..., N
    for r in range(1, len(MODALITIES) + 1):

        for combination in itertools.combinations(MODALITIES, r):

            config = copy.deepcopy(base_config)

            # Enable/disable modalities
            for modality in MODALITIES:
                config["modalities"][modality]["enabled"] = (
                    modality in combination
                )

            name = "_".join(combination)
            config_file = CONFIG_DIR / f"{name}.yaml"

            with open(config_file, "w") as f:
                yaml.safe_dump(config, f, sort_keys=False)

            configs.append((name, config_file))

    return configs


def main():

    configs = generate_configs()

    print("Experiments to run:")
    for name, config_file in configs:
        print(f"  {name}: {config_file}")

    processes = []

    for name, config_file in configs:

        print(f"Starting experiment: {name}")

        process = subprocess.Popen([
            "python",
            "run_experiment.py",
            "--config",
            str(config_file),
        ])

        processes.append((name, process))

        # Limit number of simultaneous experiments
        if len(processes) >= MAX_PARALLEL:

            name_finished, process_finished = processes.pop(0)

            process_finished.wait()

            print(f"Finished experiment: {name_finished}")

    # Wait for remaining experiments
    for name, process in processes:
        process.wait()
        print(f"Finished experiment: {name}")


if __name__ == "__main__":
    main()

import itertools
import subprocess

modalities = ["clinical", "dat", "mri"]

for r in range(1, len(modalities) + 1):

    for combination in itertools.combinations(modalities, r):

        print("Running:", combination)

        subprocess.Popen([
            "python",
            "run_experiment.py",
            "--config", "configs/config.yaml",
            "--modalities", *combination
        ])