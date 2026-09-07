from pathlib import Path
import yaml


def load_yaml(path):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    path : str or Path
        Path to YAML file.

    Returns
    -------
    dict
        Configuration contents.
    """

    path = Path(path)

    with path.open("r") as file:
        config = yaml.safe_load(file)

    return config