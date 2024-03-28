from .BaseContinuousModel import BaseContinuousModel
from .NMFSklearn import NMFSklearn
import numpy as np
from utils import binarize, matmul, to_dense, to_sparse
from scipy.sparse import csr_matrix


class BinaryMF(BaseContinuousModel):
    '''Binary MF template class.

    Instantiate BinaryMFPenalty or BinaryMFThreshold instead.
    '''
    def __init__(self):
        raise NotImplementedError("This is a template class.")
    

    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)
        self.init_model()
        self._normalize_UV()
        self._fit()


    def init_model(self):
        """Initialize factors and logging variables.
        """
        super().init_model()
        self._init_W()
        self._init_UV()
        # display
        print("[I] max U: {:.3f}, max V: {:.3f}".format(self.U.max(), self.V.max()))
        if self.display:
            self.show_matrix(title="initialization", colorbar=True)


    def _normalize_UV(self):
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
