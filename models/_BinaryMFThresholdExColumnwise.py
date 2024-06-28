from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract, binarize, matmul
from utils import isnum, index_to_bool
import numpy as np
from scipy.sparse import spmatrix, lil_matrix


class BinaryMFThresholdExColumnwise(BinaryMFThreshold):
    '''Binary matrix factorization, Thresholding algorithm, columnwise thresholds (experimental)
    '''
    def __init__(self, k, U, V, W='mask', us=0.5, vs=0.5, lamda=100, min_diff=1e-2, max_iter=30, init_method='custom', seed=None):
        '''
        Parameters
        ----------
        us, vs : list of length k, float
            Initial thresholds for `U` and `V.
            If float is provided, it will be extended to a list of k thresholds.
        '''
        self.check_params(k=k, U=U, V=V, W=W, us=us, vs=vs, lamda=lamda, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)
        

    def check_params(self, **kwargs):
        super(BinaryMFThreshold, self).check_params(**kwargs)

        # self.set_params(['us', 'vs', 'lamda'], **kwargs)

        if 'W' in kwargs:
            assert isinstance(self.W, spmatrix) or self.W in ['mask', 'full']
        if 'us' in kwargs and isnum(self.us):
            self.us = [self.us] * self.k
        if 'vs' in kwargs and isnum(self.vs):
            self.vs = [self.vs] * self.k


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(BinaryMFThreshold, self).fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        '''The gradient descent method.
        '''
        params = self.us + self.vs
        x_last = np.array(params) # initial threshold u, v
        p_last = -self.dF(x_last) # initial gradient dF(u, v)

        n_iter = 0
        is_improving = True
        while is_improving:
            xk = x_last # starting point
            pk = p_last # searching direction
            pk = pk / np.sqrt(np.sum(pk ** 2)) # debug: normalize

            print("[I] iter: {}".format(n_iter))

            alpha, fc, gc, new_fval, old_fval, new_slope = self.line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk, maxiter=50, c1=0.1, c2=0.4)

            if alpha is None:
                self._early_stop("search direction is not a descent direction.")
                break

            x_last = xk + alpha * pk
            p_last = -new_slope # descent direction
            
            self.us, self.vs = x_last[:self.k], x_last[self.k:]
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
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'us': self.us, 'vs': self.vs, 'F': new_fval})

            # display
            if self.display and n_iter % 10 == 0:
                self._show_matrix(title=f"iter {n_iter}")

            is_improving = self.early_stop(diff=diff)
            is_
            n_iter += 1


    def predict_X(self):
        U, V = self.U.copy(), self.V.copy()
        for i in range(self.k):
            U[:, i] = binarize(U[:, i], self.us[i])
            V[:, i] = binarize(V[:, i], self.vs[i])
        self.X_pd = matmul(U, V.T, boolean=True, sparse=True)
    

    def F(self, params):
        '''
        Parameters
        ----------
        params : [u0, u1, ..., v0, v1, ...]

        Returns
        -------
        F : F(u0, u1, ..., v0, v1, ...)
        '''

        us, vs = params[:self.k], params[self.k:]

        X_pd = lil_matrix((self.m, self.n))
        for i in range(self.k):
            U = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)
            X_pd = add(X_pd, U @ V.T)

        diff = self.X_train - X_pd
        F = 0.5 * np.sum(power(multiply(self.W, diff), 2))
        return F
    

    def dF(self, params):
        '''
        Parameters
        ----------
        params : [u0, u1, ..., v0, v1, ...]

        Returns
        -------
        dF : dF/d(u0, u1, ..., v0, v1, ...), the ascend direction
        '''
        us, vs = params[:self.k], params[self.k:]

        dF = np.zeros(self.k * 2)
        for i in range(self.k):
            X_gt = lil_matrix(np.zeros((self.m, self.n)))
            for j in range(self.k):
                if j == i:
                    continue
                U = sigmoid(subtract(self.U[:, j], us[j]) * self.lamda)
                V = sigmoid(subtract(self.V[:, j], vs[j]) * self.lamda)
                X_gt = add(X_gt, U @ V.T)
            X_gt = multiply(self.W, self.X_train - X_gt) # X_gt: the i-th residual

            U = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)
            X_pd = multiply(self.W, U @ V.T) # X_pd: the i-th pattern

            dFdU = X_gt @ V - X_pd @ V
            dUdu = self.dXdx(self.U[:, i], us[i])
            dFdu = multiply(dFdU, dUdu)

            dFdV = U.T @ X_gt - U.T @ X_pd
            dVdv = self.dXdx(self.V[:, i], vs[i])
            dFdv = multiply(dFdV, dVdv.T)

            dF[i] = np.sum(dFdu)
            dF[i + self.k] = np.sum(dFdv)

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
