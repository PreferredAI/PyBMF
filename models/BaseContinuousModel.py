from .BaseModel import BaseModel
from .NMFSklearn import NMFSklearn
import numpy as np
from utils import binarize, matmul, to_dense, to_sparse
from scipy.sparse import csr_matrix


class BaseContinuousModel(BaseModel):
    '''Binary matrix factorization.
    
    Reference
    ---------
    Binary Matrix Factorization with Applications
    Algorithms for Non-negative Matrix Factorization
    '''
    def __init__(self):
        raise NotImplementedError("This is a template class.")


    def _init_W(self):
        if isinstance(self.W, str):
            if self.W == 'mask':
                self.W = self.X_train.copy()
                self.W.data = np.ones(self.X_train.data.shape)
            elif self.W == 'full':
                self.W = np.ones((self.m, self.n))

        self.W = to_sparse(self.W)
        

    def _init_UV(self):
        if self.init_method == 'nmf_sklearn':
            model = NMFSklearn(k=self.k, init_method='nndsvd', seed=self.seed)
            model.fit(X_train=self.X_train)
            self.U, self.V = model.U, model.V
        elif self.init_method == 'nmf':
            pass
            # model = NMF(k=self.k, init_method='nndsvd', seed=self.seed)
            # model.fit(X_train=self.X_train)
            # self.U, self.V = model.U, model.V
        elif self.init_method == "normal":
            avg = np.sqrt(self.X_train.mean() / self.k)
            V = avg * self.rng.standard_normal(size=(self.n, self.k))
            U = avg * self.rng.standard_normal(size=(self.m, self.k))
            self.U, self.V = np.abs(U), np.abs(V)
        elif self.init_method == "uniform":
            pass
        elif self.init_method == "import":
            self.U, self.V = self.model.U, self.model.V

        self.U, self.V = to_sparse(self.U), to_sparse(self.V)


    def normalize(self):
        """Normalize factors.
        """
        diag_U = to_dense(np.sqrt(np.max(self.U, axis=0))).flatten()
        diag_V = to_dense(np.sqrt(np.max(self.V, axis=0))).flatten()
        for i in range(self.k):
            self.U[:, i] = self.U[:, i] * diag_V[i] / diag_U[i]
            self.V[:, i] = self.V[:, i] * diag_U[i] / diag_V[i]

        # display
        print("[I] max U: {:.3f}, max V: {:.3f}".format(self.U.max(), self.V.max()))
        if self.display:
            self.show_matrix(title="normalization", colorbar=True)


    def show_matrix(self, u=None, v=None, boolean=True, **kwargs):
        '''
        Parameters
        ----------
        boolean : bool
            Boolean or inner product.
        '''
        U = binarize(self.U, u) if u is not None else self.U
        V = binarize(self.V, v) if v is not None else self.V
        X = matmul(U, V.T, boolean=boolean)
        settings = [(X, [0, 0], "X"), (U, [0, 1], "U"), (V.T, [1, 0], "V")]
        super().show_matrix(settings, **kwargs)
