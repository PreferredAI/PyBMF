from .BaseModel import BaseModel
from .wnmf import wNMF
import numpy as np
from utils import to_dense


class WeightedNMF(BaseModel):
    '''
    https://github.com/asn32/weighted-nmf
    '''
    def __init__(self, k, init='random', max_iter=1000,seed=None) -> None:
        self.check_params(k=k, init=init, max_iter=max_iter, seed=seed)
        # other parameters are set as below by default
        self.model = wNMF(n_components=self.k, 
                          init=self.init, 
                          random_state=self.seed, 
                          # solver="cd",
                          beta_loss="frobenius",
                          tol=1e-4,
                          max_iter=self.max_iter,
                          # alpha_W=0.0,
                          # alpha_H="same",
                          # l1_ratio=0.0,
                          verbose=1,
                          # shuffle=False,
                          rescale=False,
                          track_error=False)
        

    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        self.init = kwargs.get("init")
        print("[I] init         :", self.init)
        self.max_iter = kwargs.get("max_iter")
        print("[I] max_iter     :", self.max_iter)
        
    
    def fit(self, X_train, W, X_val=None):
        self.load_dataset(X_train, X_val)

        self.X_train = to_dense(X_train).astype(float)
        self.W = to_dense(W).astype(float)
        
        self.coefficients_ = self.model.fit_transform(X=self.X_train, W=self.W)
        self.V = self.model.V.T
        self.U = self.model.U
