from sklearn.model_selection import StratifiedKFold

class CrossValidationSplitter:
    """
    Cross-validation splitter for survival analysis.

    Stratifies folds based on event status to preserve
    the proportion of observed events.
    """

    def __init__(
        self,
        validation_config: dict,

    ):

        self.strategy = (
            validation_config["strategy"]
        )

        self.n_splits = (
            validation_config["folds"]
        )

        self.random_state = (
            validation_config["random_state"]
        )


        if self.strategy != "stratified_kfold":

            raise ValueError(
                f"Unsupported validation strategy: "
                f"{self.strategy}"
            )


        self.cv = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )


    def split(
        self,
        X,
        event_type,
    ):
        """
        Generate train/validation indices.

        Parameters
        ----------
        X:
            Feature matrix.

        event:
            Binary event indicator.

        Returns
        -------
        Generator of train/validation indices.
        """

        return self.cv.split(
            X,
            event_type,
        )