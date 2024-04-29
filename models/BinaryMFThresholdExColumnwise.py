from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract, binarize, matmul, isnum, ismat
import numpy as np
from scipy.sparse import spmatrix, lil_matrix


class BinaryMFThresholdExColumnwise(BinaryMFThreshold):
    '''Binary matrix factorization, thresholding algorithm, columnwise thresholds (experimental).
    '''
    def __init__(self, k, U, V, W='mask', us=0.5, vs=0.5, lamda=100, min_diff=1e-3, max_iter=30, init_method='custom', seed=None):
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

        self.set_params(['us', 'vs', 'lamda'], **kwargs)

        if 'W' in kwargs:
            assert ismat(self.W) or self.W in ['mask', 'full']
        if 'us' in kwargs and isnum(self.us):
            self.us = [self.us] * self.k
        if 'vs' in kwargs and isnum(self.vs):
            self.vs = [self.vs] * self.k


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(BinaryMFThreshold, self).fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish()


    def _fit(self):
        '''The gradient descent method.
        '''
        n_iter = 0
        is_improving = True

        params = self.us + self.vs
        x_last = np.array(params) # initial point
        p_last = -self.dF(x_last) # descent direction
        new_fval = self.F(x_last) # initial value

        # initial evaluation
        self.predict_X(us=self.us, vs=self.vs, boolean=True)
        self.evaluate(df_name='updates', head_info={'iter': n_iter, 'us': self.us, 'vs': self.vs, 'F': new_fval})
        while is_improving:
            n_iter += 1
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
            
            self.print_msg("[I] Wolfe line search for iter   : {}".format(n_iter))
            self.print_msg("    num of function evals made   : {}".format(fc))
            self.print_msg("    num of gradient evals made   : {}".format(gc))
            self.print_msg("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))
            str_xk = ', '.join('{:.2f}'.format(x) for x in xk)
            str_x_last = ', '.join('{:.2f}'.format(x) for x in x_last)
            self.print_msg("    threshold update             :")
            self.print_msg("        [{}]".format(str_xk))
            self.print_msg("     -> [{}]".format(str_x_last))
            str_pk = ', '.join('{:.2f}'.format(p) for p in pk)
            self.print_msg("    threshold update direction   :")
            self.print_msg("        [{}]".format(str_pk))
            self.print_msg("    threshold difference         : {:.3f}".format(diff))

            # evaluate
            self.predict_X(us=self.us, vs=self.vs, boolean=True)
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'us': self.us, 'vs': self.vs, 'F': new_fval})

            # display
            if self.verbose and self.display and n_iter % 10 == 0:
                self._show_matrix(title=f"iter {n_iter}")

            # early stop detection
            is_improving = self.early_stop(diff=diff, n_iter=n_iter)


    def debug(self, n_iter):
        flag = False
        if (self.U == 0).sum() + (self.U == 1).sum() == self.U.shape[0] * self.U.shape[1]:
            print("[{}] U is all zeros or ones.".format(n_iter))
            flag = True
        if (self.V == 0).sum() + (self.V == 1).sum() == self.V.shape[0] * self.V.shape[1]:
            print("[{}] V is all zeros or ones.".format(n_iter))
            flag = True
        if not flag:
            print("[{}] OK.".format(n_iter))



    def approximate_X(self, us, vs):
        '''`BaseModel.predict_X()` with sigmoid relations.
        '''
        U, V = self.U.copy(), self.V.copy()
        for i in range(self.k):
            U[:, i] = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V[:, i] = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)
        self.X_approx = U @ V.T
    

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

        self.approximate_X(us, vs)
        diff = self.X_train - self.X_approx
        F = 0.5 * np.sum(power(multiply(self.W, diff), 2))
        return F
    

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

        self.approximate_X(us, vs)
        diff = self.X_train - self.X_approx
        dFdX = multiply(self.W, diff)

        for i in range(self.k):
            U = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)

            # dFdU = X_gt @ V - X_pd @ V
            dFdU = dFdX @ V
            dUdu = self.dXdx(U, us[i])
            # dUdu = self.dXdx(self.U[:, i], us[i])
            dFdu = multiply(dFdU, dUdu)

            # dFdV = U.T @ X_gt - U.T @ X_pd
            dFdV = U.T @ dFdX
            dVdv = self.dXdx(V, us[i])
            # dVdv = self.dXdx(self.V[:, i], vs[i])
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
