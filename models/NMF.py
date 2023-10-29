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
        
    
    def fit(self, train_set):
        self.check_dataset(train_set=train_set)
        self.U = self.model.fit_transform(self.X_train) # W, as in the paper
        self.V = self.model.components_ # H, as in the paper


    def check_dataset(self, train_set):
        # load train and val set
        # most heuristics don't use val_set for auto tuning
        if train_set is None:
            print("[E] Missing training set.")
            return
        self.train_set = train_set
        self.import_UVX()


    def import_UVX(self):
        self.U_info = self.train_set.factor['U']
        self.V_info = self.train_set.factor['V']
        # NMF accepts csr and csc sparse formats
        self.X_train = self.train_set.matrix['X'].csr_matrix
        self.m = self.train_set.matrix['X'].m
        self.n = self.train_set.matrix['X'].n
        
