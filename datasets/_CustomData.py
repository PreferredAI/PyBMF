from .BaseData import BaseData
from utils import to_sparse
from scipy import sparse
from typing import Union
import numpy as np

class CustomData(BaseData):
    def __init__(self, X: Union[np.ndarray, sparse.spmatrix]):
        super().__init__()
        
        self.X = to_sparse(X, type='csr')

        U_order = np.array([i for i in range(self.X.shape[0])]).astype(int)
        V_order = np.array([i for i in range(self.X.shape[1])]).astype(int)

        U_info = (U_order, U_order, U_order.astype(str))
        V_info = (V_order, V_order, V_order.astype(str))
        
        self.factor_info = [U_info, V_info]
        
