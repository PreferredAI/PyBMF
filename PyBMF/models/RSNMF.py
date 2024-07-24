from sklearn import decomposition
from .ContinuousModel import ContinuousModel
from ..utils import to_sparse


class RSNMF(ContinuousModel):
    '''Regularized single-element-based NMF.

    Under development.

    Reference
    ---------
    An Efficient Non-Negative Matrix-Factorization-Based Approach to Collaborative Filtering for Recommender Systems
    https://github.com/zhengyaochang/RSNMF
    '''
    def __init__(self, k, U=None, V=None, init_method='nndsvd', tol=1e-4, max_iter=1000, seed=None):
        '''
        Parameters
        ----------
        U, V : numpy.ndarray, spmatrix
            Need to be prepared if `init_method` is 'custom'.
        '''
        self.check_params(k=k, U=U, V=V, init_method=init_method, tol=tol, max_iter=max_iter, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        # check if init_method is valid
        assert self.init_method in ['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom']
        
    
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()


    def _fit(self):
        pass