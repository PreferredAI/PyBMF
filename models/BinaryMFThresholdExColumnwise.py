from .BinaryMFThreshold import BinaryMFThreshold
from utils import multiply, power, sigmoid, to_dense, dot, add, subtract, binarize, matmul, isnum, ismat, ignore_warnings
import numpy as np
from scipy.sparse import spmatrix, lil_matrix
from tqdm import tqdm


class BinaryMFThresholdExColumnwise(BinaryMFThreshold):
    '''Binary matrix factorization, thresholding algorithm, columnwise thresholds (experimental).

    Implemented solver includes projected line search and projected coordinate descent.
    '''
    def __init__(self, k, U, V, W='mask', us=0.5, vs=0.5, lamda=100, min_diff=1e-3, max_iter=30, init_method='custom', solver='line-search', seed=None):
        '''
        Parameters
        ----------
        us, vs : list of length k, float
            Initial thresholds for `U` and `V.
            If float is provided, it will be extended to a list of k thresholds.
        solver : str, ['line-search', 'cd']
        '''
        self.check_params(k=k, U=U, V=V, W=W, us=us, vs=vs, lamda=lamda, min_diff=min_diff, max_iter=max_iter, init_method=init_method, solver=solver, seed=seed)
        

    def check_params(self, **kwargs):
        super(BinaryMFThreshold, self).check_params(**kwargs)

        # self.set_params(['us', 'vs', 'lamda', 'solver'], **kwargs)
        
        assert self.solver in ['line-search', 'cd']
        assert self.init_method in ['custom']

        if 'W' in kwargs:
            assert ismat(self.W) or self.W in ['mask', 'full']
        if 'us' in kwargs and isnum(self.us):
            self.us = [self.us] * self.k
        if 'vs' in kwargs and isnum(self.vs):
            self.vs = [self.vs] * self.k


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super(BinaryMFThreshold, self).fit(X_train, X_val, X_test, **kwargs)

        if self.solver == 'line-search':
            self._fit_line_search()
        elif self.solver == 'cd':
            self._fit_coordinate_descent()

        self.finish()


    def _fit_line_search(self):
        '''The gradient descent method. A line search algorithm with Wolfe conditions.
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

            # debug: normalize
            # pk = pk / np.sqrt(np.sum(pk ** 2))

            print("[I] iter: {}".format(n_iter))

            alpha, fc, gc, new_fval, old_fval, new_slope = self.line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk, maxiter=50, c1=0.1, c2=0.4)

            if alpha is None:
                self._early_stop("search direction is not a descent direction.")
                break

            x_last = xk + alpha * pk

            # debug: put on constraint so that x_last is in [0, 1]
            # if np.any(x_last < 0) or np.any(x_last > 1):
            #     alpha = np.where(x_last < 0, -xk / pk, (1 - xk) / pk)
            #     x_last = xk + alpha * pk
            #     alpha = np.min(alpha)
            #     new_slope = self.dF(x_last)
            x_last[x_last <= 0] = np.finfo(np.float64).eps
            x_last[x_last >= 1] = 1 - np.finfo(np.float64).eps

            new_slope = self.dF(x_last)
            new_fval = self.F(x_last)

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
            str_pk = ', '.join('{:.4f}'.format(p) for p in alpha * pk)
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


    @ignore_warnings
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
        X_gt, X_pd = self.X_train, self.X_approx

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

        self.approximate_X(us, vs)
        X_gt, X_pd = self.X_train, self.X_approx

        dFdX = X_gt - X_pd # considered '-' and '^2'
        dFdX = multiply(self.W, dFdX) # dF/dX_pd

        for i in range(self.k):
            U = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)

            # dFdU = X_gt @ V - X_pd @ V
            dFdU = dFdX @ V
            dUdu = self.dXdx(self.U[:, i], us[i])
            dFdu = multiply(dFdU, dUdu)

            # dFdV = U.T @ X_gt - U.T @ X_pd
            dFdV = U.T @ dFdX
            dVdv = self.dXdx(self.V[:, i], vs[i])
            dFdv = multiply(dFdV, dVdv.T)

            dF[i] = np.sum(dFdu)
            dF[i + self.k] = np.sum(dFdv)

        return dF


    def _fit_coordinate_descent(self):
        '''The coordinate descent algorithm.

        Step size is determined by the inverse of Hessian matrix.
        '''
        n_iter = 0
        is_improving = True

        params = self.us + self.vs
        violation_init = 0
        violation_last = 1

        with tqdm(total=1.0) as pbar:

            while is_improving:
                n_iter += 1

                grad = self.dF(params)

                projected_grad = np.zeros(len(grad))
                hess = self.d2F(params)

                for i, g in enumerate(grad):
                    projected_grad[i] = np.min(0, g) if params[i] == 0 else g
                    if hess[i] != 0:
                        params[i] = np.max(params[i] - g / hess[i], 0)

                violation = np.sum(projected_grad)

                if n_iter == 1:
                    violation_init = violation
                if violation_init == 0:
                    break

                self.us, self.vs = params[:self.k], params[self.k:]

                violation_ratio = violation / violation_init

                # update progress bar
                pbar.update(violation_last - violation_ratio)

                # early stop detection
                is_improving = self.early_stop(diff=violation_ratio, n_iter=n_iter)

                violation_last = violation_ratio

                # evaluate
                fval = self.F(params)
                self.predict_X(us=self.us, vs=self.vs, boolean=True)
                self.evaluate(df_name='updates', head_info={'iter': n_iter, 'us': self.us, 'vs': self.vs, 'F': fval})

                # display
                # if self.verbose and self.display and n_iter % 20 == 0:
                #     self._show_matrix(title=f"iter {n_iter}")
        

    def d2F(self, params):
        us, vs = params[:self.k], params[self.k:]

        d2F = np.zeros(self.k * 2)

        self.approximate_X(us, vs)
        X_gt, X_pd = self.X_train, self.X_approx

        dFdX = X_gt - X_pd # considered '-' and '^2'
        dFdX = multiply(self.W, dFdX) # dF/dX_pd

        for i in range(self.k):
            U = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)

            # dFdU = X_gt @ V - X_pd @ V
            dFdU = dFdX @ V
            dUdu = self.dXdx(self.U[:, i], us[i])
            dFdu = multiply(dFdU, dUdu)

            # dFdV = U.T @ X_gt - U.T @ X_pd
            dFdV = U.T @ dFdX
            dVdv = self.dXdx(self.V[:, i], vs[i])
            dFdv = multiply(dFdV, dVdv.T)

            d2Udu2 = self.d2Xdx2(self.U[:, i], us[i])
            d2Fdu2 = multiply(dFdU, d2Udu2)

            d2Vdv2 = self.d2Xdx2(self.V[:, i], vs[i])
            d2Fdv2 = multiply(dFdV, d2Vdv2)

            d2F[i] = np.sum(d2Fdu2)
            d2F[i + self.k] = np.sum(d2Fdv2)

        return d2F


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


    def d2Xdx2(self, X, x):
        '''
        '''
        e = np.exp(-self.lamda * subtract(X, x)) # exp(-lamda * (X - x))
        ep1 = add(e, 1) # 1 + exp(-lamda * (X - x))

        num = multiply(self.lamda ** 2, e)
        denom = power(ep1, 2)
        d2Xdx2 = num / denom

        num *= multiply(2 * self.lamda ** 2, power(e, 2))
        denom = power(ep1, 3)
        d2Xdx2 += -num / denom

        return d2Xdx2
