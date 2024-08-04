from .BaseModel import BaseModel
import numpy as np
from ..utils import binarize, matmul, to_dense, to_sparse, ismat
from scipy.sparse import csr_matrix


class ContinuousModel(BaseModel):
    '''Base class for continuous binary matrix factorization models.
    '''
    def __init__(self):
        raise NotImplementedError("This is a template class.")
    

    def init_model(self):
        '''The ``BaseModel.init_model()`` for continuous models.
        '''
        self._start_timer()
        self._make_name()
        self._init_logs()

        # avoid init factors when custom factors are provided
        if not (hasattr(self, 'init_method') and self.init_method == 'custom'):
            self._init_factors()
        
        self.init_W()
        self.init_UV()
        self.normalize_UV()

        self._to_dense()
        self._to_float()

        # replace zeros in U, V
        if hasattr(self, 'solver') and self.solver == 'mu' and hasattr(self, 'U') and self.U is not None and hasattr(self, 'V') and self.V is not None:
            eps = np.finfo(np.float64).eps
            self.U[self.U == 0] = eps
            self.V[self.V == 0] = eps


    def init_W(self):
        '''Initialize masking weight matrix for models that accept masking weights. 
        
        This turns codenames into matrix.

        If ``W`` is 'mask': ``W`` will be assigned 1 for any entrances in ``X_train``, no matter if the value is 1, or 0 from negative sampling.

        If ``W`` is 'full': ``W`` is full 1 matrix. The loss will take the whole matrix into consideration.

        If ``W`` is ndarray or spmatrix: ``W`` will be used as the mask matrix.
        '''
        if hasattr(self, 'W'):

            assert self.W in ['mask', 'full'] or ismat(self.W)

            if isinstance(self.W, str):

                if self.W == 'mask':
                    self.W = self.X_train.copy()
                    self.W.data = np.ones(self.X_train.data.shape)

                elif self.W == 'full':
                    self.W = np.ones((self.m, self.n))

            self.W = to_sparse(self.W, type='csr')
        

    def init_UV(self, init_method="normal", ):
        '''Initialize factors U and V with given ``init_method``.
        '''
        if hasattr(self, 'init_method'):

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
                # U and V must be provided at this point
                assert hasattr(self, 'U') and hasattr(self, 'V')


    def normalize_UV(self):
        '''Normalize factors U and V with given ``normalize_method``.

        .. topic:: Reference

            The method 'balance' comes from the paper behind model ``BinaryMFPenalty``:

            Binary Matrix Factorization with Applications.

        If 'balance': balance each pair of factors, used in `BinaryMFPenalty`.
        This does not necessarily map the factors to an interval within [0, 1].

        If 'matrixwise-normalize': normalize the whole factor matrix to [0, 1], used in thresholding methods.
        This will maintain the relative magnitude of the values within the whole factor matrix.

        If 'columnwise-normalize': normalize each factor vector to [0, 1], used in thresholding methods. 
        This will maintain the relative magnitude of the values within each factor vector.

        If 'matrixwise-mapping': map unique values in the whole factor matrix to an athmetic sequence in [0, 1].
        This will maintain the relative magnitude of the values within the whole factor matrix.

        If 'columnwise-mapping': map unique values in each factor vector to an athmetic sequence in [0, 1].
        This will maintain the relative magnitude of the values within each factor vector.

        If None: do nothing.
        '''
        if hasattr(self, 'normalize_method'):

            U0_min, U0_max, V0_min, V0_max = self.U.min(), self.U.max(), self.V.min(), self.V.max()

            if self.normalize_method == 'balance':
                diag_U = to_dense(np.sqrt(np.max(self.U, axis=0))).flatten()
                diag_V = to_dense(np.sqrt(np.max(self.V, axis=0))).flatten()

                for i in range(self.k):
                    self.U[:, i] = self.U[:, i] * diag_V[i] / diag_U[i]
                    self.V[:, i] = self.V[:, i] * diag_U[i] / diag_V[i]

            elif self.normalize_method == 'matrixwise-normalize':
                self.U, self.V = self.U / self.U.max(), self.V / self.V.max()

            elif self.normalize_method == 'columnwise-normalize':
                for i in range(self.k):
                    self.U[:, i] = self.U[:, i] / self.U[:, i].max()
                    self.V[:, i] = self.V[:, i] / self.V[:, i].max()

            elif self.normalize_method == 'matrixwise-mapping':
                self.U = unique_values_mapping(to_dense(self.U))
                self.V = unique_values_mapping(to_dense(self.V))

            elif self.normalize_method == 'columnwise-mapping':
                for i in range(self.k):
                    self.U[:, i] = unique_values_mapping(to_dense(self.U[:, i]))
                    self.V[:, i] = unique_values_mapping(to_dense(self.V[:, i]))
            
            elif self.normalize_method is None:
                return

            U1_min, U1_max, V1_min, V1_max = self.U.min(), self.U.max(), self.V.min(), self.V.max()

            print("[I] Normalized from: U: [{:.4f}, {:.4f}], V: [{:.4f}, {:.4f}]".format(U0_min, U0_max, V0_min, V0_max))
            print("[I]              to: U: [{:.4f}, {:.4f}], V: [{:.4f}, {:.4f}]".format(U1_min, U1_max, V1_min, V1_max))


    def show_matrix(self, settings=None, u=None, v=None, boolean=True, **kwargs):
        '''Wrapper of ``BaseModel.show_matrix()`` with thresholds ``u`` and ``v``.
        '''
        if settings is None:
            U = binarize(self.U, u) if boolean and u is not None else self.U
            V = binarize(self.V, v) if boolean and v is not None else self.V
            X = matmul(U, V.T, boolean=boolean)
            settings = [(X, [0, 0], "X"), (U, [0, 1], "U"), (V.T, [1, 0], "V")]
        super().show_matrix(settings, **kwargs)
    

    def _show_matrix(self):
        '''Wrapper for ``BaseModel._show_matrix()``.
        '''
        settings = [(self.X_train, [0, 0], 'gt'), (self.X_pd, [0, 1], 'pd')]
        self.show_matrix(settings, colorbar=True, discrete=False, keep_nan=False)


    def _to_dense(self):
        '''Turn X, W, U and V into dense matrices.

        For temporary use in development.
        '''
        self.X_train = to_dense(self.X_train)
        if self.X_val is not None:
            self.X_val = to_dense(self.X_val)
        if self.X_test is not None:
            self.X_test = to_dense(self.X_test)
        if hasattr(self, 'W'):
            self.W = to_dense(self.W)
        if hasattr(self, 'U') and self.U is not None:
            self.U = to_dense(self.U)
        if hasattr(self, 'V') and self.V is not None:
            self.V = to_dense(self.V)



    def _to_float(self):
        '''Turn X, W, U and V into float matrices.

        For temporary use in development.
        '''
        self.X_train = self.X_train.astype(np.float64)
        if self.X_val is not None:
            self.X_val = self.X_val.astype(np.float64)
        if self.X_test is not None:
            self.X_test = self.X_test.astype(np.float64)
        if hasattr(self, 'W'):
            self.W = self.W.astype(np.float64)
        if hasattr(self, 'U') and self.U is not None:
            self.U = self.U.astype(np.float64)
        if hasattr(self, 'V') and self.V is not None:
            self.V = self.V.astype(np.float64)



    def _to_bool(self):
        '''Turn X, W, U and V into bool matrices.

        For temporary use in development.
        '''
        self.X_train = self.X_train.astype(bool)
        if self.X_val is not None:
            self.X_val = self.X_val.astype(bool)
        if self.X_test is not None:
            self.X_test = self.X_test.astype(bool)
        if hasattr(self, 'W'):
            self.W = self.W.astype(bool)
        if hasattr(self, 'U') and self.U is not None:
            self.U = self.U.astype(bool)
        if hasattr(self, 'V') and self.V is not None:
            self.V = self.V.astype(bool)


def unique_values_mapping(arr):
    '''Map unique values in a matrix to [0, 1] interval.
    '''
    unique_values = np.unique(arr)
    mapping = {val: idx / len(unique_values) for idx, val in enumerate(unique_values)}
    projected_arr = np.vectorize(mapping.get)(arr)
    return projected_arr