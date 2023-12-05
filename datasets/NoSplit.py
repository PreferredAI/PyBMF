from .BaseSplit import BaseSplit


class NoSplit(BaseSplit):
    '''No split, used in reconstruction tasks

    For a reconstruction task, training and testing will use the same full set of samples.
    '''
    def __init__(self, X) -> None:
        super().__init__(X)

        self.X_train = self.X
        self.X_val = self.X
        self.X_test = self.X