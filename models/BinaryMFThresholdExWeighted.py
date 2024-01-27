from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, step, sigmoid, dot, to_dense
import numpy as np
from scipy.optimize import line_search


class BinaryMFThresholdExWeighted(BinaryMFThreshold):
    '''Weighted BMF thresholding algorithm (experimental)
    '''      
    def _fit(self):
        self.W = self.X_train.copy()
        self.W.data = np.ones(self.W.nnz)
        self.W = to_dense(self.W)
        super()._fit()
    

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
