from .BinaryMFThresholdExColumnwise import BinaryMFThresholdExColumnwise
from utils import multiply, subtract, sigmoid, power, add, d_sigmoid, ignore_warnings
from scipy.sparse import lil_matrix
import numpy as np
from tqdm import tqdm


class BinaryMFThresholdExSigmoidColumnwise(BinaryMFThresholdExColumnwise):
    '''Binary matrix factorization, thresholding algorithm, columnwise thresholds, sigmoid link function (experimental).
    '''
    def __init__(self, k, U, V, W='mask', us=0.5, vs=0.5, link_lamda=10, lamda=100, min_diff=1e-3, max_iter=30, init_method='custom', seed=None):
        '''
        Parameters
        ----------
        us, vs : list of length k, float
            Initial thresholds for `U` and `V.
            If float is provided, it will be extended to a list of k thresholds.
        '''
        self.check_params(k=k, U=U, V=V, W=W, us=us, vs=vs, link_lamda=link_lamda, lamda=lamda, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        self.set_params(['link_lamda'], **kwargs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)


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


    @ignore_warnings
    def approximate_X(self, us, vs):
        '''`BaseModel.predict_X()` with sigmoid relations and sigmoid link function.

        S : input of sigmoid.
        X_approx : approximation of X_gt.
        '''
        U, V = self.U.copy(), self.V.copy()
        for i in range(self.k):
            U[:, i] = sigmoid(subtract(self.U[:, i], us[i]) * self.lamda)
            V[:, i] = sigmoid(subtract(self.V[:, i], vs[i]) * self.lamda)
            
        self.S = subtract(U @ V.T, 1/2) * self.link_lamda
        self.X_approx = sigmoid(self.S)
    

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

        # sigmoid link function
        dXdS = d_sigmoid(self.S) # dX_pd/dS
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

