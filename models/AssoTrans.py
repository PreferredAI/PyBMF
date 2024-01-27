from .Asso import Asso
import numpy as np
from utils import to_triplet

class AssoTrans(Asso):
    '''The Asso algorithm with transpose
    
    Reference:
        The discrete basis problem
    '''
    def fit(self, X_train, X_val=None, **kwargs):
        X_train = X_train.T
<<<<<<< HEAD
        X_val = X_val.T if X_val is not None else None
=======
        if X_val is not None:
            X_val = X_val.T
>>>>>>> 8ea583386c050f827fd03c38c626ea0e080fd29f
        super().fit(X_train, X_val, **kwargs)
        self.U, self.V = self.V, self.U
