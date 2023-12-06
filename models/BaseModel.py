import numpy as np
from utils import show_matrix, reverse_index, dot, matmul, check_sparse, TPR, FPR, PPV, ACC, TP, FP, ERR, F1
from utils import to_dense, to_sparse, to_triplet, get_metrics
import time
from scipy.sparse import isspmatrix, spmatrix, lil_matrix, csr_matrix
from typing import Union, List, Tuple


class BaseModel():
    def __init__(self) -> None:
        # model parameters
        self.U = None # csr_matrix
        self.V = None # csr_matrix


    def check_params(self, **kwargs):
        '''Popular parameters used by most algorithms
        '''
        if "k" in kwargs: # some algorithms have no predefined k or don't need k at all
            self.k = kwargs.get("k")
            print("[I] k            :", self.k)
        if "display" in kwargs:
            self.display = kwargs.get("display")
            print("[I] display      :", self.display)
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if seed is None and not hasattr(self,'seed'): # use time as self.seed
                seed = int(time.time())
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            elif seed is not None: # overwrite self.seed
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            else: # self.rng remains unchanged
                pass


    def fit(self, X_train: Union[np.ndarray, spmatrix], X_val: Union[np.ndarray, spmatrix]=None):
        """Fit the model to observations
        
        X_train: data for factorization.
        X_val: data for model selection.
        """
        raise NotImplementedError("[E] Missing fit method.")
    

    def check_dataset(self, X_train: Union[np.ndarray, spmatrix], X_val: Union[np.ndarray, spmatrix]=None):
        """Load train and val data
        """
        if X_train is None:
            raise TypeError("[E] Missing training data.")
        if X_val is None:
            print("[W] Missing validation data.")
        self.X_train = to_sparse(X_train, 'csr')
        self.X_val = None if X_val is None else to_sparse(X_val, 'csr')
        self.m, self.n = self.X_train.shape

        self.U = lil_matrix((self.m, self.k), dtype=float)
        self.V = lil_matrix((self.n, self.k), dtype=float)


    def cover(self, X=None, Y=None, w=None, axis=None) -> Union[float, np.ndarray]:
        '''Measure the coverage of X using Y
        '''
        if X is None:
            X = self.X_train
        if Y is None:
            Y = matmul(self.U, self.V.T, sparse=True, boolean=True)
        covered = TP(X, Y, axis=axis)
        overcovered = FP(X, Y, axis=axis)
        w = self.w if w is None else w
        return w[0] * covered - w[1] * overcovered


    def error(self, X=None, Y=None, axis=None) -> Union[float, np.ndarray]:
        '''Measure the coverage error of X using Y
        '''
        if X is None:
            X = self.X_train
        if Y is None:
            Y = matmul(self.U, self.V.T, sparse=True, boolean=True)
        return ERR(X, Y, axis)
    

    def score(self, U_idx, V_idx):
        """Predict the scores/ratings of a user for an item
        """
        return dot(self.U[U_idx], self.V[V_idx], boolean=True)
    

    def eval(self, X_test: Union[np.ndarray, spmatrix], metrics: List[str], task='prediction'):
        """Evaluation

        Task 'prediction' or 'reconstruction' shall give the same result.
        'prediction' uses triplet while 'reconstruction' uses spmatrix.
        """
        # assert task in ['prediction', 'reconstruction'], "Eval task is either 'prediction' or 'reconstruction'."
        if task == 'prediction':
            U_idx, V_idx, gt_data = to_triplet(X_test)
            pd_num = len(gt_data)
            pd_data = np.zeros(pd_num, dtype=int)
            for i in range(pd_num):
                pd_data[i] = self.score(U_idx=U_idx[i], V_idx=V_idx[i])
        elif task == 'reconstruction':
            gt_data = to_sparse(X_test, type='csr')
            pd_data = matmul(U=self.U, V=self.V.T, sparse=True, boolean=True)
        else: # debug
            U = to_sparse(self.U, 'csr')
            V = to_sparse(self.V, 'csr')
            gt_data = to_dense(X_test).flatten()
            pd_data = matmul(U=U, V=V.T, sparse=False, boolean=True).flatten()
            
        results = get_metrics(gt=gt_data, pd=pd_data, metrics=metrics)
        return results


    def show_matrix(self, settings: Union[Tuple, np.ndarray, spmatrix]=None, 
                    factor_info: List[Tuple]=None,
                    scaling=1.0, pixels=5, title=None, colorbar=False):
        """Show matrix
        """
        if not self.display:
            return
        if settings is None:
            if factor_info is not None:
                U_info, V_info = factor_info
                U_order = reverse_index(idx=U_info[0])
                V_order = reverse_index(idx=V_info[0])
                U, V = self.U[U_order], self.V[V_order]
            else:
                U, V = self.U, self.V
            X = matmul(U, V.T, boolean=True, sparse=False)
            U, V = to_dense(U), to_dense(V)
            settings = [(X, [0, 0], "X"), (U, [0, 1], "U"),  (V.T, [1, 0], "V")]
        elif isspmatrix(settings) or isinstance(settings, np.ndarray):
            # function overloading when settings is a matrix
            settings = [(check_sparse(settings, sparse=False), [0, 0], title)]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar)
            