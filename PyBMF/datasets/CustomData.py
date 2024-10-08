from .BaseData import BaseData
import numpy as np
from scipy.sparse import spmatrix


class CustomData(BaseData):
    '''Load custom dataset.

    Parameters
    ----------
    X : spmatrix
        The matrix.
    name : str
        The name of the dataset. Used for naming pickles.
    factor_info : list, default: None
        The list of factor info. Leave it to `None` to generate trivial factors info.
    '''
    def __init__(self, X, name, factor_info=None):
        super().__init__()

        self.is_single = True
        self.name = name

        assert isinstance(X, spmatrix), "X must be a sparse matrix."

        m, n = X.shape

        print("Using custom data of shape: {}, type: {}".format(X.shape, type(X)))

        if factor_info is None:
            u, v = np.array([i for i in range(m)]), np.array([i for i in range(n)])
            u_order, u_idmap, u_alias = u.astype(int), u.astype(int), u.astype(str)
            v_order, v_idmap, v_alias = v.astype(int), v.astype(int), v.astype(str)
            u_info, v_info = [u_order, u_idmap, u_alias], [v_order, v_idmap, v_alias]
            factor_info = [u_info, v_info]

        self.X = X
        self.factor_info = factor_info


    def read_data(self):
        '''Read data.
        '''
        pass


    def load_data(self):
        '''Load data.
        '''
        pass