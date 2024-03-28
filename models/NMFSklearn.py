from sklearn import decomposition
from .BaseModel import BaseModel
from utils import to_sparse


class NMFSklearn(BaseModel):
    '''NMF by scikit-learn.
    '''
    def __init__(self, k, init_method='nndsvd', tol=1e-4, max_iter=1000, seed=None):
        self.check_params(k=k, init_method=init_method, tol=tol, max_iter=max_iter, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        assert self.init_method in ['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom']
        
    
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)
        self._fit()


    def _fit(self):
        self.model = decomposition.NMF(
            n_components=self.k, 
            init=self.init_method, 
            random_state=self.rng, 
            solver="cd",
            beta_loss="frobenius",
            tol=self.tol,
            max_iter=self.max_iter,
            alpha_W=0.0,
            alpha_H="same",
            l1_ratio=0.0,
            verbose=0,
            shuffle=False
        )
        self.U = self.model.fit_transform(self.X_train)
        self.V = self.model.components_.T

        self.U, self.V = to_sparse(self.U), to_sparse(self.V)
