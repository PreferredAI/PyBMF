from .BinaryMFThreshold import BinaryMFThreshold
from .BaseCollectiveModel import BaseCollectiveModel
from utils import multiply, step, sigmoid, dot, to_dense
from utils import split_factor_list, get_factor_list
import numpy as np
from scipy.optimize import line_search


class BinaryMFThresholdExCollective(BaseCollectiveModel, BinaryMFThreshold):
    '''Collective thresholding algorithm (experimental)
    '''
    def __init__(self, k=None, Us=None, lamda=None, u=None, v=None, eps=None, max_iter=None):
        super().check_params(k=k, lamda=lamda, u=u, v=v, eps=eps, max_iter=max_iter, algorithm='threshold')
        if Us is None:
            # to be initialized by NMF
            self.is_initialized = False
            assert k is not None, "Missing k."
        else:
            self.is_initialized = True
            self.Us_dense = [to_dense(U).astype(float) for U in Us]
            self.k = Us[0].shape[1]


    def _fit(self):
        # W only masks Xs[0]: we assume that auxilary information matrices do not have missing values
        self.W = self.Xs_train[0].copy()
        self.W.data = np.ones(self.W.nnz)
        self.W = to_dense(self.W)
        super()._fit()


    def initialize(self):
        if self.is_initialized:
            self.X_train = to_dense(self.X_train)
            self.Us = self.Us_dense
            print("[I] After initialization: max Us: {}".format([U.max() for U in self.Us]))
        else:
            super().initialize()


    def normalize(self):
        rows, cols = split_factor_list(self.Xs_train_factors)
        factor_list = get_factor_list(self.Xs_train_factors)
        return super().normalize()
        
    

    def F(self, params):
        '''
        params = [u, v]
        return = F(u, v)
        '''
        u, v = params
        # reconstruction
        U = sigmoid((self.U - u) * self.lamda)
        V = sigmoid((self.V - v) * self.lamda)

        diff = self.X_train - U @ V.T
        F = 0.5 * np.sum(multiply(self.W, diff) ** 2)        
        return F
    

    def dF(self, params):
        '''
        params = [u, v]
        return = [dF(u, v)/du, dF(u, v)/dv], the ascend direction
        '''
        u, v = params
        sigmoid_U = sigmoid((self.U - u) * self.lamda)
        sigmoid_V = sigmoid((self.V - v) * self.lamda)

        dFdU = multiply(self.W, self.X_train) @ sigmoid_V - multiply(self.W, sigmoid_U @ sigmoid_V.T) @ sigmoid_V
        dUdu = self.dXdx(sigmoid_U, u) # original paper
        # dUdu = self.dXdx(self.U, u) # authors' implementation
        dFdu = multiply(dFdU, dUdu)

        dFdV = sigmoid_U.T @ multiply(self.W, self.X_train) - sigmoid_U.T @ multiply(self.W, sigmoid_U @ sigmoid_V.T)
        dVdv = self.dXdx(sigmoid_V, v) # original paper
        # dVdv = self.dXdx(self.V, v) # authors' implementation
        dFdv = multiply(dFdV, dVdv.T)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF
