import numpy as np
from utils import show_matrix, reverse_index, dot, matmul, check_sparse, TPR, FPR, PPV, ACC, TP, FP, ERR, F1
from utils import to_dense, to_sparse, to_triplet, get_metrics
import time
from scipy.sparse import isspmatrix, spmatrix
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
            print("[I] display :", self.display)
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
        self.X_val = to_sparse(X_val, 'csr')
        self.m, self.n = self.X_train.shape

        # self.import_UVX()


    # def import_UVX(self):
    #     self.U_info = self.train_set.factor['U']
    #     self.V_info = self.train_set.factor['V']
    #     self.X_train = self.train_set.matrix['X'].lil_matrix
    #     self.X_val = None if self.val_set is None else self.val_set.matrix['X'].triplet
    #     self.m = self.train_set.matrix['X'].m
    #     self.n = self.train_set.matrix['X'].n


    # implement import_UVX_UTY, import_UVX_WVZ, etc. as you want

    def cover(self, X=None, Y=None, w=None, axis=None) -> Union[float, np.ndarray]:
        '''Measure the coverage of X using Y
        '''
        if X is None:
            X = self.X_train
        if Y is None:
            Y = matmul(self.U, self.V, sparse=True, boolean=True)
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
            Y = matmul(self.U, self.V, sparse=True, boolean=True)
        return ERR(X, Y, axis)
    

    def score(self, U_idx, V_idx):
        """Predict the scores/ratings of a user for an item
        """
        return dot(self.U[U_idx], self.V[V_idx], boolean=True)
    

    def eval(self, X_test: Union[np.ndarray, spmatrix], metrics: List[str], task='prediction'):
        assert task in ['prediction', 'reconstruction'], "[E] Eval task is either 'prediction' or 'reconstruction'."
        if task == 'prediction':
            U_idx, V_idx, gt_data = to_triplet(X_test)
            pd_num = len(gt_data)
            pd_data = np.zeros(pd_num, dtype=int)
            for i in range(pd_num):
                pd_data[i] = self.score(U_idx=U_idx[i], V_idx=V_idx[i])
        else:
            gt_data = to_sparse(X_test, type='csr')
            pd_data = matmul(U=self.U, V=self.V, sparse=True, boolean=True)
            
        results = get_metrics(gt=gt_data, pd=pd_data, metrics=metrics)
        return results


    def show_matrix(self, settings: Union[Tuple, np.ndarray, spmatrix]=None, 
                    scaling=1.0, pixels=5, title=None, colorbar=False):
        if self.display is False:
            return
        
        if settings is None:
            if hasattr(self, 'factor_info'):
                U_info = self.factor_info[0]
                V_info = self.factor_info[1]
                U_order = reverse_index(idx=U_info[0])
                V_order = reverse_index(idx=V_info[0])
                U = self.U[U_order]
                V = self.V[V_order]
            else:
                U, V = self.U, self.V

            X = matmul(U, V.T, boolean=True, sparse=False)
            U, V = to_dense(U), to_dense(V)

            settings = [
                (X, [0, 0], "X" + str(self.X_train.shape)),
                (U, [0, 1], "U"), 
                (V.T, [1, 0], "V")
            ]
        elif isspmatrix(settings) or isinstance(settings, np.ndarray):
            # function overloading when settings is a matrix
            settings = [(check_sparse(settings, sparse=False), [0, 0], title)]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar)
            