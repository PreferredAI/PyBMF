from .BinaryMFThresholdExColumnwise import BinaryMFThresholdExColumnwise
from .BinaryMFThreshold import BinaryMFThreshold
from ..utils import multiply, power, sigmoid, to_dense, dot, add, subtract, binarize, matmul, isnum, ismat, ignore_warnings
import numpy as np
from scipy.sparse import spmatrix, lil_matrix
from tqdm import tqdm


class BinaryMFThresholdExSigmoidColumnwiseLamda(BinaryMFThresholdExColumnwise):
    '''Binary matrix factorization, thresholding algorithm, sigmoid link function, columnwise thresholds, varying `lamda` (experimental).

    - solver: projected line search
    '''
    def __init__(self, k, U, V, W='mask', us=0.5, vs=0.5, u_lamda=1, v_lamda=1, link_lamda=10, min_diff=1e-3, max_iter=30, init_method='custom', solver='line-search', seed=None):
        '''
        Parameters
        ----------
        us, vs : list of length k, float
            Initial thresholds for `U` and `V.
            If float is provided, it will be extended to a list of k thresholds.
        solver : str, ['line-search', 'cd']
        u_lamda, v_lamda : int, float
            The initial `lamda`. Here `lamda` is to be optimized. It's better to start with a small value.
        link_lamda : float
        '''
        self.check_params(k=k, U=U, V=V, W=W, us=us, vs=vs, u_lamda=u_lamda, v_lamda=v_lamda, link_lamda=link_lamda, min_diff=min_diff, max_iter=max_iter, init_method=init_method, solver=solver, seed=seed)
        

    def check_params(self, **kwargs):
        super(BinaryMFThreshold, self).check_params(**kwargs)

        # self.set_params(['us', 'vs', 'u_lamda', 'v_lamda', 'solver', 'link_lamda'], **kwargs)

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

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        if self.solver == 'line-search':
            self._fit_line_search()
        elif self.solver == 'cd':
            self._fit_coordinate_descent()        


    def _fit_line_search(self):
        '''The gradient descent method. A line search algorithm with Wolfe conditions.
        '''
        n_iter = 0
        is_improving = True

        params = self.us + self.vs
        params.append(self.u_lamda)
        params.append(self.v_lamda)

        x_last = np.array(params) # initial point
        p_last = -self.dF(x_last) # descent direction
        new_fval = self.F(x_last) # initial value

        # initial evaluation
        self.predict_X(us=self.us, vs=self.vs, boolean=True)
        self.evaluate(df_name='updates', head_info={'iter': n_iter, 'us': self.us, 'vs': self.vs, 'u_lamda': self.u_lamda, 'v_lamda': self.v_lamda, 'F': new_fval})
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

            # debug: scheduled update for u_lamda and v_lamda
            # x_last = xk + alpha * pk
            x_last[-2] = x_last[-2] * 1.2
            x_last[-1] = x_last[-1] * 1.2


            # debug: put on constraint so that x_last is in [0, 1]
            lb, ub = np.finfo(np.float64).eps, 1.0 - np.finfo(np.float64).eps
            for i in range(self.k * 2):
                x_last[i] = max(lb, min(ub, x_last[i]))
            # debug: put on constraint so that lamda is > 0
            # for i in [-2, -1]:
            #     x_last[i] = max(lb, x_last[i])

            new_slope = self.dF(x_last)
            new_fval = self.F(x_last)

            p_last = -new_slope # descent direction
            
            self.us, self.vs, self.u_lamda, self.v_lamda = x_last[:self.k], x_last[self.k:-2], x_last[-2], x_last[-1]
            diff = np.sqrt(np.sum((alpha * pk[:self.k * 2]) ** 2))
            
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
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'us': self.us, 'vs': self.vs, 'u_lamda': self.u_lamda, 'v_lamda': self.v_lamda, 'F': new_fval})

            # display
            if self.verbose and self.display and n_iter % 10 == 0:
                self._show_matrix(title=f"iter {n_iter}")

            # early stop detection
            is_improving = self.early_stop(diff=diff, n_iter=n_iter)


    @ignore_warnings
    def approximate_X(self, us, vs, u_lamda, v_lamda):
        '''`BaseModel.predict_X()` with sigmoid relations.
        '''
        U, V = self.U.copy(), self.V.copy()

        for i in range(self.k):
            U[:, i] = sigmoid(subtract(self.U[:, i], us[i]) * u_lamda)
            V[:, i] = sigmoid(subtract(self.V[:, i], vs[i]) * v_lamda)
        self.X_approx = U @ V.T
    

    def F(self, params):
        '''
        Parameters
        ----------
        params : [u1, ..., uk, v1, ..., vk, u_lamda, v_lamda]

        Returns
        -------
        F : F(u1, ..., uk, v1, ..., vk, u_lamda, v_lamda)
        '''
        us, vs, u_lamda, v_lamda = params[:self.k], params[self.k:self.k*2], params[-2], params[-1]

        self.approximate_X(us=us, vs=vs, u_lamda=u_lamda, v_lamda=v_lamda)
        X_gt, X_pd = self.X_train, self.X_approx

        diff = X_gt - X_pd
        F = 0.5 * np.sum(power(multiply(self.W, diff), 2))
        return F
    

    @ignore_warnings
    def dF(self, params):
        '''
        Parameters
        ----------
        params : [u1, ..., uk, v1, ..., vk, u_lamda, v_lamda]

        Returns
        -------
        dF : dF/d(u1, ..., uk, v1, ..., vk, u_lamda, v_lamda), the ascend direction
        '''
        us, vs, u_lamda, v_lamda = params[:self.k], params[self.k:self.k*2], params[-2], params[-1]

        dF = np.zeros(self.k * 2 + 2)

        self.approximate_X(us=us, vs=vs, u_lamda=u_lamda, v_lamda=v_lamda)
        X_gt, X_pd = self.X_train, self.X_approx

        dFdX = X_gt - X_pd # considered '-' and '^2'
        dFdX = multiply(self.W, dFdX) # dF/dX_pd

        for i in range(self.k):
            U = sigmoid(subtract(self.U[:, i], us[i]) * u_lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * v_lamda)

            # dFdU = X_gt @ V - X_pd @ V
            dFdU = dFdX @ V
            dUdu = self.dXdx(self.U[:, i], us[i], u_lamda)
            dFdu = multiply(dFdU, dUdu)

            # dFdV = U.T @ X_gt - U.T @ X_pd
            dFdV = U.T @ dFdX
            dVdv = self.dXdx(self.V[:, i], vs[i], v_lamda)
            dFdv = multiply(dFdV, dVdv.T)

            dF[i] = np.sum(dFdu)
            dF[i + self.k] = np.sum(dFdv)

        # dFdu_lamda and dFdv_lamda
        dUdu_lamda = self.dXdlamda(self.U[:, i], us[i], u_lamda)
        dFdU_lamda = multiply(dFdU, dUdu_lamda)

        dVdv_lamda = self.dXdlamda(self.V[:, i], vs[i], v_lamda)
        dFdV_lamda = multiply(dFdV, dVdv_lamda)

        dF[-2] = np.sum(dFdU_lamda)
        dF[-1] = np.sum(dFdV_lamda)

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
            U = sigmoid(subtract(self.U[:, i], us[i]) * self.u_lamda)
            V = sigmoid(subtract(self.V[:, i], vs[i]) * self.v_lamda)

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


    def dXdx(self, X, x, lamda):
        '''The fractional term in the gradient.

                      dU*     dV*     dW*     dH*
        This computes --- and --- (or --- and --- as in the paper).
                      du      dv      dw      dh
        
        Parameters
        ----------
        X :
            X*, sigmoid(X - x) in the paper
        lamda : int, float
            The shape parameter of the sigmoid function.
        '''
        diff = subtract(X, x)
        num = np.exp(-lamda * subtract(X, x)) * lamda
        denom_inv = sigmoid(diff * lamda) ** 2
        return num * denom_inv
    

    def dXdlamda(self, X, x, lamda):
        '''
                        dU*          dV*
        This computes -------- and --------.
                      du_lamda     dv_lamda
        '''
        diff = subtract(X, x)
        num = multiply(np.exp(-lamda * subtract(X, x)), subtract(X, x))
        denom_inv = sigmoid(diff * lamda) ** 2
        return multiply(num, denom_inv)


    def d2Xdx2(self, X, x, lamda):
        '''
        '''
        e = np.exp(-lamda * subtract(X, x)) # exp(-lamda * (X - x))
        ep1 = add(e, 1) # 1 + exp(-lamda * (X - x))

        num = multiply(lamda ** 2, e)
        denom = power(ep1, 2)
        d2Xdx2 = num / denom

        num *= multiply(2 * lamda ** 2, power(e, 2))
        denom = power(ep1, 3)
        d2Xdx2 += -num / denom

        return d2Xdx2
