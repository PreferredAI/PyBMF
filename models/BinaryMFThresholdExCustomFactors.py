from .BinaryMFThreshold import BinaryMFThreshold
from .BinaryMFThresholdExWeighted import BinaryMFThresholdExWeighted
from utils import multiply, step, sigmoid, to_dense
import numpy as np
from scipy.optimize import line_search


class BinaryMFThresholdExCustomFactors(BinaryMFThresholdExWeighted):
    '''Weighted BMF Thresholding algorithm with initial factors provided by the user (experimental)
    '''
    def __init__(self, U, V, lamda=None, u=None, v=None, eps=None, max_iter=None):
        self.check_params(lamda=lamda, u=u, v=v, eps=eps, max_iter=max_iter, algorithm='threshold')
        assert U.shape[1] == V.shape[1], "Factor dimensions don't match."
        self.U_dense, self.V_dense = to_dense(U), to_dense(V)
        self.k = U.shape[1]


    def initialize(self):
        self.X_train = to_dense(self.X_train)
        self.U = self.U_dense.astype(float)
        self.V = self.V_dense.astype(float)
        print("[I] After initialization: max U: {:.3f}, max V: {:.3f}".format(self.U.max(), self.V.max()))
