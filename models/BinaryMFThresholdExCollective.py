from .BinaryMFThreshold import BinaryMFThreshold
from .ContinuousCollectiveModel import ContinuousCollectiveModel
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract, binarize, matmul, isnum, d_sigmoid
import numpy as np
# from scipy.optimize import line_search
from solvers import line_search
from scipy.sparse import spmatrix, lil_matrix, issparse


class BinaryMFThresholdExCollective(ContinuousCollectiveModel):
    '''Collective thresholding algorithm (experimental).
    '''
    def __init__(self, k, Us, alpha, Ws='full', us=0.5, sigmoid_link=True, link_lamda=10, columnwise=False, lamda=10, lamda_rate=1.0, min_diff=1e-3, max_iter=50, init_method='custom', seed=None):
        '''
        Parameters
        ----------
        Us : list of ndarray or spmatrix
            Initial factors.
        alpha : list of float
            Importance weights of each matrix.
        Ws : list of ndarray or spmatrix, or str in ['mask', 'full']
        us : list of float, or float
            Initial thresholds for `Us`. 
            If `columnwise` is True, it is a list of length `n_factors` * `k` thresholds.
            If `columnwise` is False, it is a list of length `n_factors` thresholds.
            If float is provided, it will be extended to a proper structure.
        sigmoid_link : bool
            Whether to use sigmoid link function.
        link_lamda : float
            The shape parameter for sigmoid link function.
        columnwise : bool
            Whether to use columnwise thresholds.
        lamda : float
            The initial `lamda` in the objective function. 
        lamda_rate : float
            The rate of `lamda` update.
        '''
        self.check_params(k=k, Us=Us, alpha=alpha, Ws=Ws, us=us, sigmoid_link=sigmoid_link, link_lamda=link_lamda, columnwise=columnwise, lamda=lamda, lamda_rate=lamda_rate, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        # only accept custom factors
        assert self.init_method in ['custom']


    def fit(self, Xs_train, factors, Xs_val=None, Xs_test=None, **kwargs):
        super().fit(Xs_train, factors, Xs_val, Xs_test, **kwargs)

        self._fit()
        # self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result) # todo


    def init_model(self):
        super().init_model()

        # check thresholds
        if isnum(self.us):
            self.us = [self.us] * (self.n_factors * self.k) if self.columnwise else [self.us] * self.n_factors
        if self.columnwise:
            assert len(self.us) == self.n_factors * self.k
        else:
            assert len(self.us) == self.n_factors
    
    

    def _fit(self):
        '''The gradient descent method.
        '''
        params = self.us
        x_last = np.array(params) # initial threshold u, v
        p_last = -self.dF(x_last) # initial gradient dF(u, v)

        n_iter = 0
        is_improving = True
        while is_improving:
            xk = x_last # starting point
            pk = p_last # searching direction
            pk = pk / np.sqrt(np.sum(pk ** 2)) # debug: normalize

            print("[I] iter: {}".format(n_iter))

            alpha, fc, gc, new_fval, old_fval, new_slope = line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk, maxiter=50, c1=0.1, c2=0.4)

            if alpha is None:
                self._early_stop("search direction is not a descent direction.")
                break

            x_last = xk + alpha * pk
            p_last = -new_slope # descent direction
            
            self.us = x_last
            diff = np.sqrt(np.sum((alpha * pk) ** 2))
            
            print("[I] Wolfe line search for iter   : {}".format(n_iter))
            print("    num of function evals made   : {}".format(fc))
            print("    num of gradient evals made   : {}".format(gc))
            print("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))

            str_xk = ', '.join('{:.2f}'.format(x) for x in xk)
            str_x_last = ', '.join('{:.2f}'.format(x) for x in x_last)
            print("    threshold update             :")
            print("        [{}]".format(str_xk))
            print("     -> [{}]".format(str_x_last))
            
            str_pk = ', '.join('{:.2f}'.format(p) for p in pk)
            print("    threshold update direction   :")
            print("        [{}]".format(str_pk))
            
            print("    threshold difference         : {:.3f}".format(diff))

            # evaluate
            self.predict_Xs()
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'us': self.us, 'F': new_fval})

            # display
            if self.display and n_iter % 10 == 0:
                self._show_matrix(title=f"iter {n_iter}")

            is_improving = self.early_stop(diff=diff)
            n_iter += 1


    def approximate_Xs(self, us):
        '''Counterpart of `BaseCollectiveModel.predict_Xs()`.
        '''
        # init X_approx
        if not hasattr(self, 'Xs_approx'):
            self.Xs_approx = [None] * self.n_matrices
        # init S, input of sigmoid
        if self.sigmoid_link and not hasattr(self, 'S'):
            self.S = [None] * self.n_matrices
        # load Us
        Us = self.Us.copy()
        # reformat us
        if isnum(us):
            us = [us] * (self.n_factors * self.k)
        # replicate us
        if len(us) == self.n_factors:
            us = [u for u in us for _ in range(self.k)]
        # binarize
        for j in range(self.n_factors):
            for i in range(self.k):
                Us[j][:, i] = sigmoid(subtract(self.Us[j][:, i], us[i + j * self.k]) * self.lamda)
        # generate prediction
        for i, factors in enumerate(self.factors):
            a, b = factors
            if self.sigmoid_link:
                self.S[i] = subtract(Us[a] @ Us[b].T, 1/2) * self.link_lamda
                self.Xs_approx[i] = sigmoid(self.S[i])
            else:
                self.Xs_approx[i] = Us[a] @ Us[b].T


    def F(self, params):
        '''
        Parameters
        ----------
        params : [u01, ..., u0k, ..., un1, ..., unk]

        Returns
        -------
        F : F(u01, ..., u0k, ..., un1, ..., unk)
        '''

        us = params

        F = 0
        self.approximate_Xs(us)
        for m in range(self.n_matrices):
            X_gt, X_pd = self.Xs_train[m], self.Xs_approx[m]

            diff = X_gt - X_pd
            F += 0.5 * self.alpha[m] * np.sum(power(multiply(self.Ws[m], diff), 2))
        return F
    

    def dF(self, params):
        '''
        Parameters
        ----------
        params : [u01, ..., u0k, ..., un1, ..., unk]

        Returns
        -------
        dF : dF/d(u01, ..., u0k, ..., un1, ..., unk), the ascend direction
        '''
        us = params

        dF = np.zeros(self.n_factors * self.k) if self.columnwise else np.zeros(self.n_factors)
        self.approximate_Xs(us)
        for m, factors in enumerate(self.factors):
            a, b = factors

            X_gt, X_pd = self.Xs_train[m], self.Xs_approx[m]

            dFdX = X_gt - X_pd # considered '-' and '^2'
            dFdX = multiply(self.Ws[m], dFdX) # dF/dX_pd

            if self.sigmoid_link:
                # sigmoid link function
                dXdS = d_sigmoid(self.S[m]) # dX_pd/dS
                dFdS = multiply(dFdX, dXdS)
            else:
                dFdS = dFdX

            if self.columnwise:
                for i in range(self.k):
                    U = sigmoid(subtract(self.Us[a][:, i], us[i + a * self.k]) * self.lamda)
                    V = sigmoid(subtract(self.Us[b][:, i], us[i + b * self.k]) * self.lamda)

                    # dFdU = X_gt @ V - X_pd @ V
                    dFdU = dFdX @ V
                    dUdu = self.dXdx(self.Us[a][:, i], us[i + a * self.k])
                    dFdu = multiply(dFdU, dUdu)

                    # dFdV = U.T @ X_gt - U.T @ X_pd
                    dFdV = U.T @ dFdX
                    dVdv = self.dXdx(self.Us[b][:, i], us[i + b * self.k])
                    dFdv = multiply(dFdV, dVdv.T)

                    dF[i + a * self.k] += np.sum(dFdu)
                    dF[i + b * self.k] += np.sum(dFdv)
            else:
                U = sigmoid(subtract(self.Us[a], us[a]) * self.lamda)
                V = sigmoid(subtract(self.Us[b], us[b]) * self.lamda)

                # dFdU = X_gt @ V - X_pd @ V
                dFdU = dFdX @ V
                dUdu = self.dXdx(self.Us[a], us[a])
                dFdu = multiply(dFdU, dUdu)

                # dFdV = U.T @ X_gt - U.T @ X_pd
                dFdV = U.T @ dFdX
                dVdv = self.dXdx(self.Us[b], us[b])
                dFdv = multiply(dFdV, dVdv.T)

                dF[a] += np.sum(dFdu)
                dF[b] += np.sum(dFdv)

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
