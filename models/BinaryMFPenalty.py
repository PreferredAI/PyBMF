from .ContinuousModel import ContinuousModel
from utils import binarize, to_dense, power, multiply, ignore_warnings
import numpy as np
from scipy.sparse import spmatrix


class BinaryMFPenalty(ContinuousModel):
    '''Binary matrix factorization, Penalty algorithm
    
    Reference
    ---------
    Binary Matrix Factorization with Applications
    Algorithms for Non-negative Matrix Factorization
    '''
    def __init__(self, k, U=None, V=None, W='mask', reg=2.0, reg_growth=3, tol=0.01, min_diff=0.0, max_iter=100, init_method='nmf_sklearn', seed=None):
        '''
        Parameters
        ----------
        reg : float
            The regularization weight 'lambda' in the paper.
        reg_growth : float
            The growing rate of regularization weight.
        tol : float
            The error tolerance 'epsilon' in the paper.
        '''
        self.check_params(k=k, U=U, V=V, W=W, reg=reg, reg_growth=reg_growth, tol=tol, min_diff=min_diff, max_iter=max_iter, init_method=init_method, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        
        self.set_params(['reg_growth'], **kwargs)
        assert self.init_method in ['normal', 'uniform', 'custom']


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish()

    
    def init_model(self):
        '''Initialize factors and logging variables.
        '''
        super().init_model()

        self.init_UV()
        self.normalize_UV(method="balance")


    def _fit(self):
        '''The alternative minimization algorithm.
        '''
        n_iter = 0
        is_improving = True

        # compute error
        error_old, rec_error_old, reg_error_old = self.error()

        # evaluate with naive threshold
        u, v = 0.5, 0.5
        self.predict_X(u=u, v=v, boolean=True)
        self.evaluate(df_name='updates', head_info={'iter': n_iter, 'error': error_old, 'rec_error': rec_error_old, 'reg': float(self.reg), 'reg_error': reg_error_old})

        while is_improving:
            # update n_iter, U, V
            n_iter += 1
            self.update_V()
            self.update_U()

            # compute error, diff
            error_new, rec_error_new, reg_error_new = self.error()
            diff = abs(reg_error_old - reg_error_new)
            error_old, rec_error_old, reg_error_old = error_new, rec_error_new, reg_error_new

            # evaluate with naive threshold
            u, v = 0.5, 0.5
            self.predict_X(u=u, v=v, boolean=True)
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'error': error_new, 'rec_error': rec_error_new, 'reg': float(self.reg), 'reg_error': reg_error_new})

            # display
            self.print_msg("iter: {}, error: {:.2e}, rec_error: {:.2e}, reg: {:.2e}, reg_error: {:.2e}".format(n_iter, error_new, rec_error_new, self.reg, reg_error_new))
            if self.verbose and self.display and n_iter % 10 == 0:
                self.show_matrix(u=u, v=v, title=f"iter {n_iter}")

            # early stop detection
            is_improving = self.early_stop(error=reg_error_old, diff=diff, n_iter=n_iter)

            # update reg
            self.reg *= self.reg_growth


    @ignore_warnings
    def update_U(self):
        '''Multiplicative update of U.
        '''
        num = multiply(self.W, self.X_train) @ self.V + 3 * self.reg * power(self.U, 2)
        denom = multiply(self.W, self.U @ self.V.T) @ self.V + 2 * self.reg * power(self.U, 3) + self.reg * self.U
        denom[denom == 0] = np.finfo(np.float64).eps
        self.U = multiply(self.U, num / denom)


    @ignore_warnings
    def update_V(self):
        '''Multiplicative update of V.
        '''
        num = multiply(self.W, self.X_train).T @ self.U + 3 * self.reg * power(self.V, 2)
        denom = multiply(self.W, self.U @ self.V.T).T @ self.U + 2 * self.reg * power(self.V, 3) + self.reg * self.V
        denom[denom == 0] = np.finfo(np.float64).eps
        self.V = multiply(self.V, num / denom)


    def error(self):
        '''Error for penalty function algorithm.

        In the paper, only reg_error is considered and minimized.
        '''
        diff = self.X_train - self.U @ self.V.T
        rec_error = np.sum(power(multiply(self.W, diff), 2))
        reg_error = np.sum(power(power(self.U, 2) - self.U, 2)) + np.sum(power(power(self.V, 2) - self.V, 2))

        error = 1 * rec_error + 0.5 * self.reg * reg_error
        return error, rec_error, reg_error
