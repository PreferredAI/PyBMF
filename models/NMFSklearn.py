from sklearn import decomposition
from .BaseModel import BaseModel


class NMFSklearn(BaseModel):
    def __init__(self, k, init='nndsvd', max_iter=1000, seed=None):
        self.check_params(k=k, init=init, max_iter=max_iter, seed=seed)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        self.set_params(['k', 'init', 'max_iter', 'seed'], **kwargs)
        assert self.init in ['random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom']
        
    
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)
        self._fit()


    def _fit(self):
        self.model = decomposition.NMF(
            n_components=self.k, 
            init=self.init, 
            random_state=self.rng, 
            solver="cd",
            beta_loss="frobenius",
            tol=1e-4,
            max_iter=self.max_iter,
            alpha_W=0.0,
            alpha_H="same",
            l1_ratio=0.0,
            verbose=0,
            shuffle=False
        )
        self.U = self.model.fit_transform(self.X_train)
        self.V = self.model.components_.T
