from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from utils import multiply, bool_to_index, ERR, get_prediction, show_matrix, get_residual, matmul, add
import numpy as np


class GreConD(BaseModel):
    '''The GreConD algorithm for exact Boolean decomposition.
    
    Reference
    ---------
    Discovery of optimal factors in binary data via a novel method of matrix decomposition.
    '''
    def __init__(self, k=None, tol=0):
        self.check_params(k=k, tol=tol)
        
        
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):

        k = 0
        is_factorizing = True
        self.X_rs = lil_matrix(self.X_train.copy())
        while is_factorizing:
            score, u, v = get_concept(self.X_train, self.X_rs)
            if score == 0:
                is_factorizing = self.early_stop(msg="No pattern found", k=k)
                break

            # update factors
            self.set_factors(k, u=u, v=v)

            # evaluate
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

            self.evaluate(
                df_name='updates', 
                head_info={
                    'k': k, 
                    'score': score, 
                    'shape': [u.sum(), v.sum()], 
                }, 
            )

            # early stop detection
            error = ERR(gt=self.X_train, pd=self.X_pd)
            print("[I] k: {}, score: {}, error: {:.3f}, shape: [{}, {}]".format(k, score, error, u.sum(), v.sum()))
            is_factorizing = self.early_stop(error=error, n_factor=k+1, k=k)

            k += 1



def get_concept(X_gt, X_rs):
    '''Get a concept/pattern from the residual matrix.

    Parameters
    ----------
    X_gt : spmatrix
        The ground-truth matrix.
    X_rs : spmatrix
        The residual matrix.

    Returns
    -------
    score : float
        The TP coverage of the pattern over X_gt.
    u, v : (m, 1), (n, 1) spmatrix
        The factors.
        If the pattern is not found, they'll be zero vectors.
    '''
    m, n = X_gt.shape
    score, best_score = 0, 0
    u, best_u = None, lil_matrix(np.ones((m, 1)))
    v, best_v = None, lil_matrix((n, 1))

    # column ids of residual matrix
    j_rs = bool_to_index(X_rs.sum(axis=0) > 0)

    n_iter = 0
    is_improving = True
    while is_improving:

        last_score = best_score
        j_list = [j for j in j_rs if not best_v[j] == 1]

        for j in j_list:

            u = multiply(lil_matrix(X_gt[:, j]), best_u)

            if u.sum() * n > score: # check the score upper bound
                u_idx = bool_to_index(u)
                v_idx = bool_to_index(X_gt[u_idx, :].sum(axis=0) == u.sum())

                v = lil_matrix((n, 1))
                v[v_idx] = 1

                if u.sum() * v.sum() > score: # check the score upper bound
                    score = X_rs[u_idx][:, v_idx].sum()
                    if score > best_score:
                        best_score = score
                        best_u = u
                        best_v = v
        
        if last_score == best_score:
            is_improving = False
        n_iter += 1

    return best_score, best_u, best_v

                    