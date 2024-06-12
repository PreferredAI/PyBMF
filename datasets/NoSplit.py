from .BaseSplit import BaseSplit
from .RatioSplit import RatioSplit
from scipy.sparse import spmatrix


class NoSplit(BaseSplit):
    '''No split, used in reconstruction tasks.

    Designed for reconstruction tasks, where training, validation and testing use the same full set of samples.
    `NoSplit` supports negative sampling.
    '''
    def __init__(self, X: spmatrix, seed=None):
        super().__init__(X)
        print("[I] NoSplit, sampling positives")     
        # self.check_params(seed=seed)

        self.rs = RatioSplit(self.X, seed=seed)
        self.pos_size = self.rs.pos_train_size

        self.X_train = self.X
        self.X_val = self.X
        self.X_test = self.X


    def negative_sample(self, size, type='uniform', seed=None):
        
        self.rs.negative_sample(train_size=size, type=type, seed=seed)
        self.neg_size = self.rs.neg_train_size

        self.X_train = self.rs.X_train
        self.X_val = self.rs.X_train
        self.X_test = self.rs.X_train