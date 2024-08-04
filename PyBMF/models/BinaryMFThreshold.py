from .ContinuousModel import ContinuousModel
from ..utils import multiply, power, sigmoid, ignore_warnings, subtract, get_prediction, get_prediction_with_threshold, ismat, show_factor_distribution
import numpy as np
from ..solvers import line_search, limit_step_size
from tqdm import tqdm


class BinaryMFThreshold(ContinuousModel):
    '''Binary matrix factorization, thresholding algorithm with line search.
    
    .. topic:: Reference
    
        Binary Matrix Factorization with Applications.
        
        Algorithms for Non-negative Matrix Factorization.

    .. note::

        To be released:

        For sigmoid link function, please use ``BinaryMFThresholdExSigmoid``.

        For columnwise thresholds, please use ``BinaryMFThresholdExColumnwise``.
    
        For both, please use ``BinaryMFThresholdExSigmoidColumnwise``.

    Parameters
    ----------
    u : float
        Initial threshold for ``U``.
    v : float
        Initial threshold for ``V``.
    lamda : float
        The 'lambda' in sigmoid function.
    '''
    def __init__(self, k, U, V, W='mask', u=0.5, v=0.5, lamda=100, solver="line-search", min_diff=1e-3, max_iter=100, init_method='custom', normalize_method=None, seed=None):
        self.check_params(k=k, U=U, V=V, W=W, u=u, v=v, lamda=lamda, solver=solver, min_diff=min_diff, max_iter=max_iter, init_method=init_method, normalize_method=normalize_method, seed=seed)
        

    def check_params(self, **kwargs):
        '''Check the validity of parameters.
        '''
        super().check_params(**kwargs)

        assert self.solver in ['line-search']
        assert self.init_method in ['custom']
        assert self.normalize_method in ['balance', 'matrixwise-normalize', 'columnwise-normalize', 'matrixwise-mapping', 'columnwise-mapping', None]
        assert ismat(self.W) or self.W in ['mask', 'full']


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        '''Fit the model.
        '''
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()

        self.X_pd = get_prediction_with_threshold(U=self.U, V=self.V, u=self.u, v=self.v)
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def threshold_to_x(self):
        return np.array([self.u, self.v])
    
    
    def x_to_threshold(self, x_last):
        self.u, self.v = x_last[0], x_last[1]


    def x_bounds(self):
        eps = 1e-6
        x_min = np.array([self.U.min() + eps, self.V.min() + eps])
        x_max = np.array([self.U.max() - eps, self.V.max() - eps])
        return x_min, x_max


    def evaluate_with_threshold(self, n_iter, new_fval):
        self.X_pd = get_prediction_with_threshold(U=self.U, V=self.V, u=self.u, v=self.v)
        self.evaluate(df_name='updates', head_info={'iter': n_iter, 'u': self.u, 'v': self.v, 'F': new_fval})


    def _fit(self):
        '''The gradient descent method.
        '''
        n_iter = 0 # init n_iter
        is_improving = True # init flag

        x_last = self.threshold_to_x() # init threshold
        p_last = -self.dF(x_last) # descent direction
        new_fval = self.F(x_last) # init value

        self.evaluate_with_threshold(n_iter, new_fval) # init evaluation
        
        if self.verbose is False:
            pbar = tqdm(total=self.max_iter, desc='[I] F: -') # init pbar

        while is_improving:
            n_iter += 1 # update n_iter

            xk = x_last # starting point
            pk = p_last # searching direction

            alpha, fc, gc, new_fval, old_fval, new_slope = line_search(f=self.F, myfprime=self.dF, xk=xk, pk=pk, maxiter=50)

            if alpha is None:
                print("[W] Search direction is not a descent direction.")
                break

            # get x_last, p_last
            x_last = xk + alpha * pk
            p_last = -new_slope # descent direction

            # refine x_last, p_last and new_fval by limiting alpha
            x_min, x_max = self.x_bounds()
            x_last, alpha = limit_step_size(x_min=x_min, x_max=x_max, x_last=x_last, xk=xk, pk=pk, alpha=alpha)
            p_last = -self.dF(x_last)
            new_fval = self.F(x_last)

            # measurements
            diff = np.abs(new_fval - old_fval) # the difference of function value
            # diff = np.sqrt(np.sum((alpha * pk) ** 2)) # the difference of threshold

            self.print_msg("  Wolfe line search iter         : {}".format(n_iter))
            self.print_msg("    num of function evals        : {}".format(fc))
            self.print_msg("    num of gradient evals        : {}".format(gc))
            self.print_msg("    function value update        : {:.3f} -> {:.3f}".format(old_fval, new_fval))

            str_xk = ', '.join('{:.2f}'.format(x) for x in list(xk))
            str_x_last = ', '.join('{:.2f}'.format(x) for x in list(x_last))
            self.print_msg("    threshold update             :")
            self.print_msg("      [{}]".format(str_xk))
            self.print_msg("    ->[{}]".format(str_x_last))
            
            str_pk = ', '.join('{:.4f}'.format(p) for p in alpha * pk)
            self.print_msg("    threshold update direction   :")
            self.print_msg("      [{}]".format(str_pk))

            self.x_to_threshold(x_last) # update threshold
            self.evaluate_with_threshold(n_iter, new_fval) # update evaluation

            # update pbar
            if self.verbose is False:
                pbar.update(1)
                pbar.set_description(f"[I] F: {new_fval:.6f}")

            # early stop detection
            is_improving = self.early_stop(n_iter=n_iter, diff=diff)
    

    @ignore_warnings
    def F(self, params):
        '''The objective function.

        Parameters
        ----------
        params : list of 2 floats
            The thresholds [``u``, ``v``].

        Returns
        -------
        F : float
            The value of the objective function F(``u``, ``v``).
        '''
        u, v = params

        U = sigmoid(subtract(self.U, u) * self.lamda)
        V = sigmoid(subtract(self.V, v) * self.lamda)

        diff = self.X_train - U @ V.T
        F = 0.5 * np.sum(power(multiply(self.W, diff), 2))
        return F
    

    @ignore_warnings
    def dF(self, params):
        '''The gradient of the objective function.

        Parameters
        ----------
        params : list of 2 floats
            The thresholds [``u``, ``v``].

        Returns
        -------
        dF : float
            The value of the gradient (ascend direction) of the objective function [dF(``u``, ``v``)/du, dF(``u``, ``v``)/dv].
        '''
        u, v = params
        
        U = sigmoid(subtract(self.U, u) * self.lamda)
        V = sigmoid(subtract(self.V, v) * self.lamda)
        
        X_gt = self.X_train
        X_pd = U @ V.T

        dFdX = multiply(self.W, X_gt - X_pd)

        dFdU = dFdX @ V
        dUdu = self.dXdx(self.U, u)
        dFdu = multiply(dFdU, dUdu) # (m, k)

        dFdV = U.T @ dFdX
        dVdv = self.dXdx(self.V, v)
        dFdv = multiply(dFdV, dVdv.T) # (k, n)

        dF = np.array([np.sum(dFdu), np.sum(dFdv)])
        return dF


    # @ignore_warnings
    def dXdx(self, X, x):
        '''The fractional term in the gradient.

                      dU*     dV*     dW*     dH*
        This computes --- and --- (or --- and --- as in the paper).
                      du      dv      dw      dh
        
        Parameters
        ----------
        X : ndarray
            The ``X*``, sigmoid(``X`` - ``x``) in the paper.
        '''
        diff = subtract(X, x)
        num = np.exp(-self.lamda * subtract(X, x)) * self.lamda
        denom_inv = sigmoid(diff * self.lamda) ** 2

        return num * denom_inv