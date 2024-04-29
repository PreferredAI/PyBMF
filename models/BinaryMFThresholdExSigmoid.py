from .BinaryMFThreshold import BinaryMFThreshold
from .NMF import NMF
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract, d_sigmoid
import numpy as np
from scipy.sparse import spmatrix


class BinaryMFThresholdExSigmoid(BinaryMFThreshold):
    '''Binary matrix factorization, thresholding algorithm, sigmoid link function (experimental).
    '''
    def F(self, params):
        '''
        Parameters
        ----------
        params : [u, v]

        Returns
        -------
        F : F(u, v)
        '''
        u, v = params

        U = sigmoid(subtract(self.U, u) * self.lamda)
        V = sigmoid(subtract(self.V, v) * self.lamda)

        # sigmoid link function
        diff = self.X_train - sigmoid(100 * subtract(U @ V.T, 1/2, sparse=False, boolean=False))
        F = 0.5 * np.sum(power(multiply(self.W, diff), 2))
        return F
    

    def dF(self, params):
        '''
        Parameters
        ----------
        params : [u, v]

        Returns
        -------
        dF : [dF(u, v)/du, dF(u, v)/dv], the ascend direction
        '''
        u, v = params
        U = sigmoid(subtract(self.U, u) * self.lamda)
        V = sigmoid(subtract(self.V, v) * self.lamda)
        
        # X_gt = multiply(self.W, self.X_train)
        # X_pd = multiply(self.W, U @ V.T)

        # sigmoid link function
        S = 10 * subtract(U @ V.T, 1/2) # input of sigmoid
        dFds = self.X_train - sigmoid(S) # considered '-' and '^2'
        dFds = multiply(self.W, dFdS)
        dFdS = multiply(dFdS, d_sigmoid(S))

        # dFdU = X_gt @ V - X_pd @ V
        dFdU = dFdS @ V
        dUdu = self.dXdx(self.U, u)
        dFdu = multiply(dFdU, dUdu) # (m, k)

        # dFdV = U.T @ X_gt - U.T @ X_pd
        dFdV = U.T @ dFdS
        dVdv = self.dXdx(self.V, v)
        dFdv = multiply(dFdV, dVdv.T) # (k, n)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF


    def dXdx(self, X, x):
        '''The fractional term in the gradient.

                      dU*     dV*     dW*     dH*
        This computes --- and --- (or --- and --- as in the paper).
                      du      dv      dw      dh
        
        Parameters
        ----------
        X : X*, sigmoid(X - x) in the paper
        '''
        diff = subtract(X, x)
        num = np.exp(-self.lamda * subtract(X, x)) * self.lamda
        denom_inv = sigmoid(diff * self.lamda) ** 2
        return num * denom_inv
