from abc import ABC, abstractmethod


class PredictionDatasetBuilder(ABC):
    """
    Abstract base class for building prediction datasets.

    A prediction dataset combines predictor variables with
    participant-level outcomes to produce a machine-learning-ready
    dataset.
    """

    @abstractmethod
    def build(self):
        """Build the prediction dataset."""
        pass

    @abstractmethod
    def save(self):
        """Save the prediction dataset and metadata."""
        pass

    @abstractmethod
    def summary(self):
        """Print a summary of the prediction dataset."""
        pass