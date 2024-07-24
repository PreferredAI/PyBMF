from .BinaryMFThresholdExColumnwise import BinaryMFThresholdExColumnwise, get_prediction_with_sigmoid
from .BinaryMFThreshold import BinaryMFThreshold
from ..utils import multiply, subtract, sigmoid, power, add, d_sigmoid, ignore_warnings, ismat, isnum
from scipy.sparse import lil_matrix
import numpy as np
from tqdm import tqdm
from ..solvers import line_search


class BinaryMFThresholdExSigmoidColumnwise(BinaryMFThresholdExColumnwise):
    '''Binary matrix factorization, thresholding algorithm (experimental).

    - solvers: projected line search
    - sigmoid link function
    - columnwise thresholds
    '''
    def __init__(self, k, U, V, W='full', us=0.5, vs=0.5, link_lamda=10, lamda=10, min_diff=1e-3, max_iter=30, solver='line-search', init_method='custom', seed=None):
        '''
        Parameters
        ----------
        us, vs : list of length k, float
            Initial thresholds for `U` and `V.
            If float is provided, it will be extended to a list of k thresholds.
        '''
        self.check_params(k=k, U=U, V=V, W=W, us=us, vs=vs, link_lamda=link_lamda, lamda=lamda, min_diff=min_diff, max_iter=max_iter, solver=solver, init_method=init_method, seed=seed)
        

    def check_params(self, **kwargs):
        super(BinaryMFThreshold, self).check_params(**kwargs)

        assert self.solver in ['line-search']
        assert self.init_method in ['custom']

        if 'W' in kwargs:
            assert ismat(self.W) or self.W in ['mask', 'full']
        if 'us' in kwargs and isnum(self.us):
            self.us = [self.us] * self.k
        if 'vs' in kwargs and isnum(self.vs):
            self.vs = [self.vs] * self.k


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(BinaryMFThreshold, self).fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        self._fit_line_search()


    @ignore_warnings
    def F(self, params):
        '''
        Parameters
        ----------
        params : [u1, ..., uk, v1, ..., vk]

        Returns
        -------
        F : F(u1, ..., uk, v1, ..., vk)
        '''
        us, vs = params[:self.k], params[self.k:]

        X_gt = self.X_train
        X_pd, _ = get_prediction_with_sigmoid_sigmoid(U=self.U, V=self.V, us=us, vs=vs, lamda=self.lamda, link_lamda=self.link_lamda)

        diff = X_gt - X_pd
        F = 0.5 * np.sum(power(multiply(self.W, diff), 2))
        return F
    

    @ignore_warnings
    def dF(self, params):
        '''
        Parameters
        ----------
        params : [u1, ..., uk, v1, ..., vk]

        Returns
        -------
        dF : dF/d(u1, ..., uk, v1, ..., vk), the ascend direction
        '''
        us, vs = params[:self.k], params[self.k:]

        dF = np.zeros(self.k * 2)

        X_gt = self.X_train
        X_pd, S = get_prediction_with_sigmoid_sigmoid(U=self.U, V=self.V, us=us, vs=vs, lamda=self.lamda, link_lamda=self.link_lamda)

        dFdX = X_gt - X_pd # considered '-' and '^2'
        dFdX = multiply(self.W, dFdX) # dF/dX_pd

        # sigmoid link function
        dXdS = d_sigmoid(S) # dX_pd/dS
        dFdS = multiply(dFdX, dXdS)

        for i in range(self.k):
            U = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)

            # dFdU = X_gt @ V - X_pd @ V
            # dFdU = dFdX @ V
            dFdU = dFdS @ V # include dS/dU
            dUdu = self.dXdx(self.U[:, i], us[i])
            dFdu = multiply(dFdU, dUdu)

            # dFdV = U.T @ X_gt - U.T @ X_pd
            # dFdV = U.T @ dFdX
            dFdV = U.T @ dFdS # include dS/dV
            dVdv = self.dXdx(self.V[:, i], vs[i])
            dFdv = multiply(dFdV, dVdv.T)

            dF[i] = np.sum(dFdu)
            dF[i + self.k] = np.sum(dFdv)

        return dF


    # @ignore_warnings
    # def approximate_X(self, us, vs):
    #     '''`BaseModel.predict_X()` with sigmoid relations and sigmoid link function.

    #     S : input of sigmoid.
    #     X_approx : approximation of X_gt.
    #     '''
    #     U, V = self.U.copy(), self.V.copy()
    #     for i in range(self.k):
    #         U[:, i] = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
    #         V[:, i] = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)
            
    #     self.S = subtract(U @ V.T, 1/2) * self.link_lamda
    #     self.X_approx = sigmoid(self.S)


@ignore_warnings
def get_prediction_with_sigmoid_sigmoid(U, V, us, vs, lamda, link_lamda):
    '''Get sigmoid(S) and S.

    S = (sigmoid(U * λ) @ sigmoid(V * λ) - 1/2) * λ_link
    '''
    UV = get_prediction_with_sigmoid(U, V, us, vs, lamda)
    S = subtract(UV, 1/2) * link_lamda
    return sigmoid(S), S
