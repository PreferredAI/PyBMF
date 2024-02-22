from .BaseModel import BaseModel
from .NMF import NMF
import numpy as np
from utils import binarize, matmul, to_dense, to_sparse


class BinaryMF(BaseModel):
    '''Binary matrix factorization
    
    From the papers:
        'Binary Matrix Factorization with Applications', 
        'Algorithms for Non-negative Matrix Factorization'.
    '''
    def __init__(self, k,
                 reg=None, reg_growth=None,     # for 'penalty' only
                 lamda=None, u=None, v=None,    # for 'threshold' only
                 eps=None, max_iter=None,       # shared params
                ) -> None:
        raise NotImplementedError("Instantiate BinaryMFPenalty or BinaryMFThreshold instead.")
    

    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train, X_val)
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
                self.u = 0.5 if self.u is None else self.u
                self.v = 0.5 if self.v is None else self.v
                print("[I] initial u, v :", [self.u, self.v])
                # check lamda
                self.lamda = kwargs.get('lamda') # 'lambda' for sigmoid function
                if self.lamda is None:
                    self.lamda = 100
                print("[I] lamda        :", self.lamda)
                # check eps
                self.eps = kwargs.get('eps') # step size threshold
                if self.eps is None:
                    self.eps = 1e-4
                print("[I] eps          :", self.eps)
                # check max_iter
                self.max_iter = kwargs.get('max_iter') # max iteration
                if self.max_iter is None:
                    self.max_iter = 100
                print("[I] max_iter     :", self.max_iter)
        

    def initialize(self, nmf_max_iter=1000, nmf_seed=None):
        nmf = NMF(k=self.k, init='random', max_iter=nmf_max_iter, seed=nmf_seed)
        nmf.fit(X_train=self.X_train)

        if self.algorithm == 'penalty':
            # penalty supports sparse matrix
            self.U = to_sparse(nmf.U, type='csr')
            self.V = to_sparse(nmf.V, type='csr')
        elif self.algorithm == 'threshold':
            self.X_train = to_dense(self.X_train)
            self.U = to_dense(nmf.U)
            self.V = to_dense(nmf.V)
            # todo: sparsity support

        print("[I] After initialization: max U: {:.3f}, max V: {:.3f}".format(self.U.max(), self.V.max()))
        self.show_matrix(title="after initialization", colorbar=True)


    def normalize(self):
        diag_U = to_dense(np.sqrt(np.max(self.U, axis=0))).flatten()
        diag_V = to_dense(np.sqrt(np.max(self.V, axis=0))).flatten()
        
        for i in range(self.k):
            self.U[:, i] = self.U[:, i] * diag_V[i] / diag_U[i]
            self.V[:, i] = self.V[:, i] * diag_U[i] / diag_V[i]

        print("[I] After normalization: max U: {:.3f}, max V: {:.3f}".format(self.U.max(), self.V.max()))
        self.show_matrix(title="after normalization", colorbar=True)

        if self.U.max() > 1 or self.V.max() > 1:
            print("[W] Normalization failed. Re-try will help.")


    def show_matrix(self, title=None, colorbar=None):
        if not self.display:
            return
        if self.algorithm == "penalty":
            u, v = 0.5, 0.5
        elif self.algorithm == "threshold":
            u, v = self.u, self.v

        U = self.U.copy()
        V = self.V.copy()
        X_inner = matmul(U, V.T, sparse=False, boolean=False)

        U = binarize(X=U, threshold=u)
        V = binarize(X=V, threshold=v)
        X_bool = matmul(U, V.T, sparse=False, boolean=True)

        settings = [(to_dense(U), [0, 3], "U thresholded"), 
                    (to_dense(V).T, [1, 2], "V thresholded"), 
                    (X_bool, [0, 2], "X boolean product"),
                    (to_dense(self.U), [0, 1], "U"), 
                    (to_dense(self.V).T, [1, 0], "V"), 
                    (X_inner, [0, 0], "X inner product"),
                    ]
        super().show_matrix(settings=settings, title=title, colorbar=colorbar, 
                            scaling=self.scaling, pixels=self.pixels)
