from .BaseModel import BaseModel
import numpy as np
from utils import binarize, matmul, to_dense, to_sparse
from scipy.sparse import csr_matrix


class ContinuousModel(BaseModel):
    '''Continuous binary matrix factorization.
    
    Reference
    ---------
    Binary Matrix Factorization with Applications
    Algorithms for Non-negative Matrix Factorization
    '''
    def __init__(self):
        raise NotImplementedError("This is a template class.")
    

    def init_model(self):
        '''Wrapper of `BaseModel.init_model()` for continuous models.
        '''
        if self.init_method != 'custom':
            # init factors and logging variables
            super().init_model()
        else:
            # do not overwrite factors if they've been imported
            self._init_logs()

        if hasattr(self, 'W'):
            self.init_W()


    def init_W(self):
        '''Initialize masking weights.

        Turning codenames into matrix.
        '''
        if isinstance(self.W, str):
            if self.W == 'mask':
                self.W = self.X_train.copy()
                self.W.data = np.ones(self.X_train.data.shape)
            elif self.W == 'full':
                self.W = np.ones((self.m, self.n))

        self.W = to_sparse(self.W)
        

    def init_UV(self):
        '''Initialize factors.
        '''
        if self.init_method == "normal":
            avg = np.sqrt(self.X_train.mean() / self.k)
            V = avg * self.rng.standard_normal(size=(self.n, self.k))
            U = avg * self.rng.standard_normal(size=(self.m, self.k))
            self.U, self.V = np.abs(U), np.abs(V)
        elif self.init_method == "uniform":
            avg = np.sqrt(self.X_train.mean() / self.k)
            self.V = self.rng.uniform(low=0, high=avg * 2, size=(self.n, self.k))
            self.U = self.rng.uniform(low=0, high=avg * 2, size=(self.m, self.k))
        elif self.init_method == "custom":
            assert hasattr(self, 'U') and hasattr(self, 'V') # U and V must be provided at this point

        self.U, self.V = to_sparse(self.U), to_sparse(self.V)


    def normalize_UV(self, method='balance'):
        '''Normalize factors.

        Parameters
        ----------
        method : str, ['balance', 'normalize']
            'balance': used in `BinaryMFPenalty`.
            'normalize': used in `BinaryMFThreshold`.
        '''
        a, c = [self.U.min(), self.U.max()], [self.V.min(), self.V.max()]

        if method == 'balance':
            diag_U = to_dense(np.sqrt(np.max(self.U, axis=0))).flatten()
            diag_V = to_dense(np.sqrt(np.max(self.V, axis=0))).flatten()
            for i in range(self.k):
                self.U[:, i] = self.U[:, i] * diag_V[i] / diag_U[i]
                self.V[:, i] = self.V[:, i] * diag_U[i] / diag_V[i]
        elif method == 'normalize':
            for i in range(self.k):
                self.U[:, i] = self.U[:, i] / self.U[:, i].max()
                self.V[:, i] = self.V[:, i] / self.V[:, i].max()

        b, d = [self.U.min(), self.U.max()], [self.V.min(), self.V.max()]

        print("[I] Normalized U: {} -> {}, V: {} -> {}".format(a, b, c, d))


    def show_matrix(self, settings=None, u=None, v=None, boolean=True, **kwargs):
        '''Wrapper of `BaseModel.show_matrix()` with thresholds `u` and `v`.
        '''
        if settings is None:
            U = binarize(self.U, u) if boolean and u is not None else self.U
            V = binarize(self.V, v) if boolean and v is not None else self.V
            X = matmul(U, V.T, boolean=boolean)
            settings = [(X, [0, 0], "X"), (U, [0, 1], "U"), (V.T, [1, 0], "V")]
        super().show_matrix(settings, **kwargs)
