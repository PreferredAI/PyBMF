from .Asso import Asso
import numpy as np
from utils import to_triplet

class AssoTrans(Asso):
    '''The Asso algorithm with transpose
    
    From the paper 'The discrete basis problem'.
    '''
    def __init__(self, k, tau=None, w=None):
        super().__init__(k=k, tau=tau, w=w)


    def import_UVX(self):
        '''Transposed X
        '''
        self.U_info = self.train_set.factor['V']
        self.V_info = self.train_set.factor['U']
        self.X_train = self.train_set.matrix['X'].matrix.T.tolil()
        self.X_val = None if self.val_set is None else to_triplet(self.val_set.matrix['X'].matrix.T)
        self.m = self.train_set.matrix['X'].n
        self.n = self.train_set.matrix['X'].m


    def fit(self, train_set, val_set=None, display=False):
        super().fit(train_set=train_set, val_set=val_set, display=display)
        self.U, self.V = self.V.T, self.U.T
