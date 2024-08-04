from tqdm import tqdm
from .ContinuousModel import ContinuousModel
from ..utils import to_sparse, multiply, power, ignore_warnings, subtract, check_sparse, add, get_prediction
import numpy as np
from scipy.sparse import lil_matrix


class WNMF(ContinuousModel):
    '''Weighted Nonnegative Matrix Factorization.

    .. topic:: Reference

        Weighted Nonnegative Matrix Factorization and Face Feature Extraction.

        For scipy implementation:

        Projected Gradient Methods for Non-negative Matrix Factorization.

        https://github.com/scikit-learn/scikit-learn/blob/a95203b249c1cf392f86d001ad999e29b2392739/sklearn/decomposition/nmf.py#L158

    Parameters
    ----------
    U : ndarray, spmatrix
        Need to be prepared if ``init_method`` is 'custom'. 
    V : ndarray, spmatrix
        Need to be prepared if ``init_method`` is 'custom'.
    '''
    def __init__(self, k, U=None, V=None, W='mask', beta_loss='frobenius', init_method='normal', solver='mu', tol=0.0, min_diff=0.0, max_iter=30, seed=None):
        self.check_params(k=k, U=U, V=V, W=W, beta_loss=beta_loss, init_method=init_method, solver=solver, tol=tol, min_diff=min_diff, max_iter=max_iter, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        assert self.beta_loss in ['frobenius', 'kullback-leibler']
        assert self.solver in ['mu']
        assert self.init_method in ['uniform', 'normal', 'custom']
        
    
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        '''Fit the model.
        '''
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()

        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=False)
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        '''The alternative minimization algorithm.
        '''
        n_iter = 0
        is_improving = True

        # compute error
        error_old = self.error()

        # evaluate
        # self.predict_X(boolean=False)
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=False)
        self.evaluate(df_name='updates', head_info={'iter': n_iter, 'error': error_old}, metrics=['RMSE', 'MAE'])

        pbar = tqdm(total=self.max_iter, desc="[I] error: -")
        while is_improving:
            # update n_iter, U, V
            n_iter += 1
            self.update()

            # compute error, diff
            error_new = self.error()
            diff = abs(error_old - error_new)
            error_old = error_new

            # evaluate
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=False)
            self.evaluate(df_name='updates', head_info={'iter': n_iter, 'error': error_new}, metrics=['RMSE', 'MAE'])

            # display
            if self.verbose and self.display and n_iter % 10 == 0:
                self.show_matrix(boolean=False, title=f"iter {n_iter}")

            # update pbar
            pbar.update(1)
            pbar.set_description(f"[I] error: {error_new:.6e}")

            # early stop detection
            is_improving = self.early_stop(error=error_old, diff=diff, n_iter=n_iter)


    @ignore_warnings
    def update(self):
        '''Multiplicative update.
        '''
        if self.beta_loss == 'frobenius':
            # update V
            num = multiply(self.W, self.X_train).T @ self.U
            denom = multiply(self.W, self.U @ self.V.T).T @ self.U
            denom[denom == 0] = np.finfo(np.float64).eps

            self.V = multiply(self.V, num / denom)

            # update U
            num = multiply(self.W, self.X_train) @ self.V
            denom = multiply(self.W, self.U @ self.V.T) @ self.V
            denom[denom == 0] = np.finfo(np.float64).eps

            self.U = multiply(self.U, num / denom)

        elif self.beta_loss == 'kullback-leibler':
            WX = multiply(self.W, self.X_train)
            O = lil_matrix(np.ones(self.X_train.shape))

            # update V
            UV = self.U @ self.V.T
            num = (WX / UV).T @ self.U
            denom = O.T @ self.U
            denom[denom == 0] = np.finfo(np.float64).eps

            self.V = multiply(self.V, num / denom)

            # update U
            UV = self.U @ self.V.T
            num = (WX / UV) @ self.V
            denom = O @ self.V
            denom[denom == 0] = np.finfo(np.float64).eps

            self.U = multiply(self.U, num / denom)
                

    @ignore_warnings
    def error(self):
        '''The error function.
        '''
        X_gt = self.X_train
        X_pd = self.U @ self.V.T

        X_gt[X_gt == 0] = np.finfo(np.float64).eps
        X_pd[X_pd == 0] = np.finfo(np.float64).eps

        if self.beta_loss == 'frobenius':
            rec_error = 0.5 * np.sum(multiply(self.W, power(X_gt - X_pd, 2)))
            error = rec_error
            
        elif self.beta_loss == 'kullback-leibler':
            rec_error = np.sum(multiply(self.W, multiply(X_gt, np.log(X_gt / X_pd)) - X_gt + X_pd))
            error = rec_error
            
        return error