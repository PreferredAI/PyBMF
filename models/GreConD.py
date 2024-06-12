from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from utils import multiply, bool_to_index
import numpy as np


class GreConD(BaseModel):
    '''
    
    Reference
    ---------
    Discovery of optimal factors in binary data via a novel method of matrix decomposition.
    '''
    def __init__(self, k=None, tol=0):
        self.check_params(k=k, tol=tol)
        
        
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish()


    def _fit(self):

        self.X_rs = lil_matrix(self.X_train)
        is_factorizing = True
        k = 0
        while is_factorizing:
            score, u, v = get_concept(self.X_train, self.X_rs)
            if score == 0:
                is_factorizing = self.early_stop(msg="No pattern found", k=k)
            else:
                is_factorizing = self.early_stop(n_factor=k+1, k=k)
                self.set_factors(k, u=u, v=v)
            k += 1



def get_concept(X_train, X_rs):
    '''Get a concept/pattern from the residual matrix.

    Parameters
    ----------
    X_train : scipy.sparse.csr_matrix
        The training data matrix.
    X_rs : scipy.sparse.csr_matrix
        The residual matrix.
    '''
    m, n = X_train.shape
    score, u, v = 0, None, None
    best_score, best_u, best_v = 0, lil_matrix(np.ones(m, 1)), lil_matrix((n, 1))

    # column ids of residual matrix
    j_rs = bool_to_index(X_rs.sum(axis=1) > 0)

    is_improving = True
    n_iter = 0
    while is_improving:

        last_score = best_score
        j_list = [j for j in j_rs if best_v[j] == 0]

        for j in j_list:

            u = multiply(lil_matrix(X_train[:, j]), best_u)

            if u.sum() * n > score: # check the score upper bound
                u_idx = bool_to_index(u)
                v_idx = bool_to_index(X_rs[u_idx, :].sum(axis=1) == u.sum())

                v = lil_matrix((n, 1))
                v[v_idx] = 1

                if u.sum() * v.sum() > score: # check the score upper bound
                    score = X_rs[u_idx, v_idx].sum()
                    if score > best_score:
                        best_score = score
                        best_u = u
                        best_v = v
        
        if last_score == best_score:
            is_improving = False
        n_iter += 1

    return best_score, best_u, best_v

                    