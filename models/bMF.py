from .BaseModel import BaseModel
from .NMF import NMF
import numpy as np

from utils import multiply, matmul, to_dense, to_sparse
from scipy.linalg import inv
# from scipy.sparse.linalg import inv
from scipy.sparse import csr_matrix


class bMF(BaseModel):
    '''Binary matrix factorization (bMF, binaryMF)
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''
    def __init__(self, k,
                 reg=None, reg_growth=None,     # for 'penalty' only
                 lamda=None, u=None, v=None,    # for 'threshold' only
                 eps=None, max_iter=None,       # shared params
                ) -> None:
        raise NotImplementedError("Instantiate bMFPenalty or bMFThreshold instead.")
    

    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.check_dataset(X_train, X_val)
        self._fit()


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if 'algorithm' in kwargs:
            self.algorithm = kwargs.get('algorithm')
            print("[I] algorithm    :", self.algorithm)
            if self.algorithm == 'penalty':
                # check reg
                self.reg = kwargs.get('reg') # 'lambda' in paper
                if self.reg is None:
                    self.reg = 10
                print("[I] reg          :", self.reg)
                # check reg_growth
                self.reg_growth = kwargs.get('reg_growth') # reg growing rate
                if self.reg_growth is None:
                    self.reg_growth = 10
                print("[I] reg_growth   :", self.reg_growth)
                # check eps
                self.eps = kwargs.get('eps') # error tolerance 'epsilon' in paper
                if self.eps is None:
                    self.eps = 0.01
                print("[I] eps          :", self.eps)
                # check max_iter
                self.max_iter = kwargs.get('max_iter') # max iteration
                if self.max_iter is None:
                    self.max_iter = 100
                print("[I] max_iter     :", self.max_iter)
            elif self.algorithm == 'threshold':
                # initial threshold u and v
                self.u = kwargs.get("u")
                self.v = kwargs.get("v")
                self.u = 0.8 if self.u is None else self.u
                self.v = 0.8 if self.v is None else self.v
                print("[I] initial u, v :", [self.u, self.v])
                # check lamda
                self.lamda = kwargs.get('lamda') # 'lambda' for sigmoid function
                if self.lamda is None:
                    self.lamda = 10
                print("[I] lamda        :", self.lamda)
                # check eps
                self.eps = kwargs.get('eps') # step size threshold
                if self.eps is None:
                    self.eps = 1e-6
                print("[I] eps          :", self.eps)
                # check max_iter
                self.max_iter = kwargs.get('max_iter') # max iteration
                if self.max_iter is None:
                    self.max_iter = 100
                print("[I] max_iter     :", self.max_iter)
        

    def initialize(self, nmf_max_iter=2000, nmf_seed=None):
        nmf = NMF(k=self.k, init='random', max_iter=nmf_max_iter, seed=nmf_seed)
        nmf.fit(train_set=self.X_train)

        self.U = to_sparse(nmf.U, type='csr')
        self.V = to_sparse(nmf.V, type='csr')

        print("[I] After initialization: max U: {:.3f}, max V: {:.3f}".format(np.max(self.U), np.max(self.V)))
        self.show_matrix(title="after initialization", colorbar=True)


    def normalize(self):
        diag_U = np.sqrt(np.max(self.U, axis=0)).toarray().flatten()
        diag_V = np.sqrt(np.max(self.V, axis=1)).toarray().flatten()
        
        for i in range(self.k):
            self.U[:, i] = self.U[:, i] * diag_V[i] / diag_U[i]
            self.V[i, :] = self.V[i, :] * diag_U[i] / diag_V[i]

        print("[I] After normalization: max U: {:.3f}, max V: {:.3f}".format(np.max(self.U), np.max(self.V)))
        self.show_matrix(title="after normalization", colorbar=True)


    def show_matrix(self, title=None, scaling=1.0, pixels=5, colorbar=False):
        if self.display:        
            if self.algorithm == "penalty":
                u, v = 0.5, 0.5
            elif self.algorithm == "threshold":
                u, v = self.u, self.v

            U = self.step_function(X=self.U, threshold=u)
            V = self.step_function(X=self.V, threshold=v)
            X = matmul(U, V, sparse=False, boolean=True)
            U, V = to_dense(U), to_dense(V)

            settings = [(U, [0, 1], "U"), (V.T, [1, 0], "V"), (X, [0, 0], "X")]
            super().show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar)
