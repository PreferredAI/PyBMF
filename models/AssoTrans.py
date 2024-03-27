from .Asso import Asso
import numpy as np
from utils import to_triplet

class AssoTrans(Asso):
    '''The Asso algorithm with transpose.
    
    Reference
    ---------
    The discrete basis problem. Zhang et al. 2007.
    '''
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        X_train = X_train.T
        X_val = X_val.T if X_val is not None else None
        X_test = X_test.T if X_test is not None else None

        super().fit(X_train, X_val, X_test, **kwargs)
        
        self.U, self.V = self.V, self.U
