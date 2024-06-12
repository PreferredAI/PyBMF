from .Asso import Asso
import numpy as np
from utils import to_triplet
from .BaseModel import BaseModel

class TransposedModel(Asso):
    '''The model with transposed input.
    '''
    def __init__(self, model, **kwargs):
        self.check_params(model=model, **kwargs)

        assert isinstance(self.model, BaseModel), 'The model must be an instance of BaseModel.'


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        
        X_train = X_train.T
        X_val = X_val.T if X_val is not None else None
        X_test = X_test.T if X_test is not None else None

        self.model.fit(X_train, X_val, X_test, **kwargs)
        
        self.U, self.V = self.model.V, self.model.U
