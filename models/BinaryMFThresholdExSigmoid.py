from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract, d_sigmoid
import numpy as np


class BinaryMFThresholdExSigmoid(BinaryMFThreshold):
    '''Binary matrix factorization, thresholding algorithm, sigmoid link function (experimental).
    '''
    def __init__(self, k, U, V, W='mask', u=0.5, v=0.5, link_lamda=10, lamda=100, min_diff=1e-3, max_iter=30, init_method='custom', seed=None):
        '''
        Parameters
        ----------
        link_lamda : float
            The 'lambda' in sigmoid link function.
        '''
        self.check_params(k=k, U=U, V=V, W=W, u=u, v=v, link_lamda=link_lamda, lamda=lamda, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        # self.set_params(['link_lamda'], **kwargs)


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
        X_gt = self.X_train
        S = subtract(U @ V.T, 1/2) * self.link_lamda # input of sigmoid
        X_pd = sigmoid(S) # approximation of X_gt

        diff = X_gt - X_pd
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
        
        # sigmoid link function
        X_gt = self.X_train
        S = subtract(U @ V.T, 1/2) * self.link_lamda # input of sigmoid
        X_pd = sigmoid(S) # approximation of X_gt

        dFdX = X_gt - X_pd # considered '-' and '^2'
        dFdX = multiply(self.W, dFdX) # dF/dX_pd
        
        dXdS = d_sigmoid(S) # dX_pd/dS
        dFdS = multiply(dFdX, dXdS)

        # dFdU = X_gt @ V - X_pd @ V
        dFdU = dFdS @ V # include dS/dU
        dUdu = self.dXdx(self.U, u)
        dFdu = multiply(dFdU, dUdu) # (m, k)

        # dFdV = U.T @ X_gt - U.T @ X_pd
        dFdV = U.T @ dFdS # include dS/dV
        dVdv = self.dXdx(self.V, v)
        dFdv = multiply(dFdV, dVdv.T) # (k, n)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF


    # def dXdx(self, X, x):
    #     '''The fractional term in the gradient.

    #                   dU*     dV*     dW*     dH*
    #     This computes --- and --- (or --- and --- as in the paper).
    #                   du      dv      dw      dh
        
    #     Parameters
    #     ----------
    #     X : X*, sigmoid(X - x) in the paper
    #     '''
    #     diff = subtract(X, x)
    #     num = np.exp(-self.lamda * subtract(X, x)) * self.lamda
    #     denom_inv = sigmoid(diff * self.lamda) ** 2
    #     return num * denom_inv
