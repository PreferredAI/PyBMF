from .ContinuousModel import ContinuousModel
from ..utils import binarize, to_dense, power, multiply, ignore_warnings, get_prediction, get_prediction_with_threshold, show_factor_distribution
import numpy as np
from scipy.sparse import spmatrix
from tqdm import tqdm


class BinaryMFPenalty(ContinuousModel):
    '''Binary matrix factorization, penalty function algorithm.
    
    .. topic:: Reference

        Binary Matrix Factorization with Applications.

        Algorithms for Non-negative Matrix Factorization.

        Solving the problem with multiplicative update:

        min 1/2 * ||X - U @ V.T||_F^2 + 1/2 * reg * ||U^2 - U||_F^2 + 1/2 * reg * ||V^2 - V||_F^2

    Parameters
    ----------
    reg : float
        The regularization weight 'lambda' in the paper.
    reg_growth : float
        The growing rate of regularization weight.
    max_reg : float
        The upper bound of regularization weight.
    tol : float
        The error tolerance 'epsilon' in the paper.
    '''
    def __init__(self, k, U=None, V=None, W='full', beta_loss="frobenius", solver="mu", reg=2.0, reg_growth=3, max_reg=1e10, tol=0.01, min_diff=0.0, max_iter=100, init_method='custom', normalize_method='balance', seed=None):
        self.check_params(k=k, U=U, V=V, W=W, reg=reg, beta_loss=beta_loss, solver=solver, reg_growth=reg_growth, max_reg=max_reg, tol=tol, min_diff=min_diff, max_iter=max_iter, init_method=init_method, normalize_method=normalize_method, seed=seed)
        

    def check_params(self, **kwargs):
        '''Check the validity of parameters.
        '''
        super().check_params(**kwargs)
        
        assert self.beta_loss in ['frobenius']
        assert self.solver in ['mu']
        assert self.init_method in ['normal', 'uniform', 'custom']
        assert self.normalize_method in ['balance', None]
        
        self.reg, self.reg_growth, self.max_reg = np.float64(self.reg), np.float64(self.reg_growth), np.float64(self.max_reg)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        '''Fit the model.
        '''
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()

        # boolean product with fixed threshold
        self.X_pd = get_prediction_with_threshold(U=self.U, V=self.V, u=0.5, v=0.5)
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        '''The multiplicative update of factor matrices U, V.
        '''
        n_iter = 0
        is_improving = True

        # initial evaluation
        self.X_pd = self.get_prediction()
        error_old, rec_error_old, reg_error_old = error(
            X_gt=self.X_train, X_pd=self.X_pd, W=self.W, U=self.U, V=self.V, reg=self.reg)
        self.evaluate(df_name='updates', head_info={'iter': n_iter, 'error': error_old, 'rec_error': rec_error_old, 'reg': float(self.reg), 'reg_error': reg_error_old}, metrics=['RMSE', 'MAE'])

        # initial boolean evaluation
        self.X_pd = get_prediction_with_threshold(U=self.U, V=self.V, u=0.5, v=0.5)
        self.evaluate(df_name='boolean')

        # init pbar
        desc = "[I] error: {:.2e}, rec_error: {:.2e}, reg: {:.2e}, reg_error: {:.2e}".format(error_old, rec_error_old, self.reg, reg_error_old)
        pbar = tqdm(total=self.max_iter, position=0, desc=desc)

        while is_improving:
            # update n_iter, U, V
            n_iter += 1

            self.update_V()
            self.update_U()

            # compute error, diff
            self.X_pd = self.get_prediction()
            error_new, rec_error_new, reg_error_new = error(
                X_gt=self.X_train, X_pd=self.X_pd, W=self.W, U=self.U, V=self.V, reg=self.reg
            )
            diff = abs(reg_error_old - reg_error_new) # difference of reg_error
            error_old, rec_error_old, reg_error_old = error_new, rec_error_new, reg_error_new

            # evaluation
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'error': error_new, 'rec_error': rec_error_new, 'reg': float(self.reg), 'reg_error': reg_error_new}, metrics=['RMSE', 'MAE'])

            # boolean evaluation
            self.X_pd = get_prediction_with_threshold(U=self.U, V=self.V, u=0.5, v=0.5)
            self.evaluate(df_name='boolean')

            # display
            desc = "[I] error: {:.2e}, rec_error: {:.2e}, reg: {:.2e}, reg_error: {:.2e}".format(error_new, rec_error_new, self.reg, reg_error_new)
            pbar.set_description(desc)
            pbar.update(1)
            if self.verbose and self.display and n_iter % 5 == 0:
                self.show_matrix(boolean=True, u=0.5, v=0.5, colorbar=True, title=f"iter {n_iter}")
                show_factor_distribution(U=self.U, V=self.V, resolution=100)

            # early stop detection
            is_improving = self.early_stop(error=reg_error_old, diff=diff, n_iter=n_iter)

            # update reg
            self.reg = min(self.reg * self.reg_growth, self.max_reg)


    def get_prediction(self):
        '''Get prediction matrix.
        '''
        return get_prediction(U=self.U, V=self.V, boolean=False)
    
    
    def update_U(self):
        '''The update process of U.
        '''
        self.U = update_U(X=self.X_train, W=self.W, U=self.U, V=self.V, reg=self.reg)
        

    def update_V(self):
        '''The update process of V.
        '''
        self.V = update_V(X=self.X_train, W=self.W, U=self.U, V=self.V, reg=self.reg)


def update_U(X, W, U, V, reg, solver='mu', beta_loss='frobenius'):
    '''Multiplicative update of factor U.
    '''
    num = multiply(W, X) @ V
    num += 3 * reg * power(U, 2)

    denom = multiply(W, U @ V.T) @ V
    denom += 2 * reg * power(U, 3) + reg * U
    denom[denom == 0] = np.finfo(np.float64).eps

    U_next = multiply(U, num / denom)
    U_next[U_next == 0] = np.finfo(float).eps
    return U_next


def update_V(X, W, U, V, reg, solver='mu', beta_loss='frobenius'):
    '''Multiplicative update of factor V.
    '''
    num = multiply(W, X).T @ U
    num += 3 * reg * power(V, 2)

    denom = multiply(W, U @ V.T).T @ U
    denom += 2 * reg * power(V, 3) + reg * V
    denom[denom == 0] = np.finfo(np.float64).eps

    V_next = multiply(V, num / denom)
    V_next[V_next == 0] = np.finfo(float).eps
    return V_next


def error(X_gt, X_pd, W, U, V, reg):
    '''Error for penalty function algorithm.
    '''
    rec_err = rec_error(X_gt, X_pd, W)
    reg_err = reg * (reg_error(U) + reg_error(V))
    err = rec_err + reg_err
    return err, rec_err, reg_err


def rec_error(X_gt, X_pd, W):
    '''Reconstruction error.
    '''
    rec_error = 0.5 * np.sum(multiply(W, power(X_gt - X_pd, 2)))
    return rec_error


def reg_error(X):
    '''The regularization function.
    '''
    reg_error = 0.5 * np.sum(power(power(X, 2) - X, 2))
    return reg_error
