from sklearn import decomposition
from .BaseModel import BaseModel


class NMF(BaseModel):
    def __init__(self, k, init='random', max_iter=1000,seed=None) -> None:
        self.check_params(k=k, init=init, max_iter=max_iter, seed=seed)
        # other parameters are set as below by default
        self.model = decomposition.NMF(n_components=self.k, 
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
                                       shuffle=False)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        self.init = kwargs.get("init")
        print("[I] init         :", self.init)
        self.max_iter = kwargs.get("max_iter")
        print("[I] max_iter     :", self.max_iter)
        
    
    def fit(self, X_train, X_val=None):
        self.load_dataset(X_train, X_val)
        self.U = self.model.fit_transform(self.X_train)
        self.V = self.model.components_
        self.V = self.V.T
