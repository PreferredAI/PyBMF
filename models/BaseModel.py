import numpy as np
from utils import show_matrix, reverse_index, dot, matmul, TP, FP, ERR, add_log
from utils import to_dense, to_sparse, to_triplet, get_metrics
import time
from scipy.sparse import spmatrix, lil_matrix, csr_matrix
from typing import Union, List, Tuple
import pandas as pd


class BaseModel():
    def __init__(self) -> None:
        # model parameters
        self.U = None # csr_matrix
        self.V = None # csr_matrix


    def check_params(self, **kwargs):
        '''Popular parameters used by most algorithms

        kwargs:
            k: rank. some algorithms have no predefined k or don't need k at all.
            seed: random seed shared by the whole model.
            display: whether to show matrix at each step during the fitting and after finish.
                scaling: rescale the matrix plot.
                pixels: set the resolution of matrix plot.
            verbose: whether to show matrix and print result at each step during the fitting.
        '''
        if "k" in kwargs:
            k = kwargs.get("k")
            if k is None:
                print("[I] Running without k.")
            self.k = k
            print("[I] k            :", self.k)

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

        if "task" in kwargs:
            task = kwargs.get("task")
            assert task in ['prediction', 'reconstruction'], "Eval task is either 'prediction' or 'reconstruction'."
            if not hasattr(self, 'task') or self.task != task: # first call or upon changes
                self.task = task
                print("[I] task         :", self.task)

        # params that automatically resets when not called
        if "verbose" in kwargs:
            verbose = kwargs.get("verbose")
            if not hasattr(self, 'verbose') or self.verbose != verbose: # first call or upon changes
                self.verbose = verbose
                print("[I] verbose      :", self.verbose)
        else:
            self.verbose = False # auto reset to False

        if "display" in kwargs:
            self.display = kwargs.get("display")
            print("[I] display      :", self.display)
        else:
            self.display = False # auto reset to False

        if "scaling" in kwargs and self.display:
            self.scaling = kwargs.get("scaling")
            print("[I]   scaling    :", self.scaling)
        else:
            self.scaling = 1.0 # auto reset to 1.0

        if "pixels" in kwargs and self.display:
            self.pixels = kwargs.get("pixels")
            print("[I]   pixels     :", self.pixels)
        else:
            self.pixels = 2


    def fit(self, X_train: Union[np.ndarray, spmatrix], X_val: Union[np.ndarray, spmatrix]=None, **kwargs):
        """Fit the model to observations
        
        X_train:
            data for factorization.
        X_val:
            data for model selection.
        kwargs:
            seed: random seed shared by the whole model.
            display: whether to show matrix at each step during the fitting and after finish.
                scaling: rescale the matrix plot.
                pixels: set the resolution of matrix plot.
            verbose: whether to show matrix and print result at each step during the fitting.
        """
        raise NotImplementedError("Missing fit method.")
    

    def check_dataset(self, X_train: Union[np.ndarray, spmatrix], X_val: Union[np.ndarray, spmatrix]=None):
        """Load train and val data

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not modified, csr or csc is preferred.
        """
        if X_train is None:
            raise TypeError("Missing training data.")
        if X_val is None:
            print("[W] Missing validation data.")
        self.X_train = to_sparse(X_train, 'csr')
        self.X_val = None if X_val is None else to_sparse(X_val, 'csr')

        self.m, self.n = self.X_train.shape

        self.U = lil_matrix(np.zeros((self.m, self.k)))
        self.V = lil_matrix(np.zeros((self.n, self.k)))


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
    

    def eval(self, X_test: Union[np.ndarray, spmatrix], metrics: List[str], task: str):
        """Evaluation

        'prediction' uses triplet while 'reconstruction' uses the whole spmatrix.
        The triplet may contain 0's, depending on the negative sampling used.
        """
        self.check_params(task=task)

        if self.task == 'prediction':
            U_idx, V_idx, gt_data = to_triplet(X_test)
            pd_num = len(gt_data)
            pd_data = np.zeros(pd_num, dtype=int)
            for i in range(pd_num):
                pd_data[i] = self.score(U_idx=U_idx[i], V_idx=V_idx[i])
                
        elif self.task == 'reconstruction':
            gt_data = to_sparse(X_test, type='csr')
            pd_data = matmul(U=self.U, V=self.V.T, sparse=True, boolean=True)
            
        results = get_metrics(gt=gt_data, pd=pd_data, metrics=metrics)
        return results
    

    def validate(self, names: List[str]=[], metrics: List[str]=[], values=[], **kwargs):
        """Called anytime when X_val needs to be evaluated and recorded

        To validate, please provide validation set.

        names:
            names of the values.
        values:
            usually model parameters.
        metrics:
            list of metrics to evaluate.
        kwargs:
            verbose: print results or not.
        """
        if self.X_val is None:
            return
        
        self.check_params(**kwargs)

        if not hasattr(self, 'df_validation') or self.df_validation is None:
            self.df_validation = pd.DataFrame(columns=names+metrics)
        results = self.eval(self.X_val, metrics=metrics, task=self.task)
        add_log(df=self.df_validation, line=values+results, verbose=self.verbose)


    def early_stop(self, msg: str, k: int=None):
        print("[W] Stopped in advance: " + msg)
        if k is not None:
            print("[W]   got {} factor(s).".format(k))
            self.U = self.U[:, :k]
            self.V = self.V[:, :k]
    
    
    def print_msg(self, msg):
        if self.verbose:
            print("[I] " + msg)


    def show_matrix(self, settings: List[Tuple]=None, 
                    matrix: Union[np.ndarray, spmatrix]=None, 
                    factors: List[List[int]]=None,
                    factor_info: List[Tuple]=None,
                    scaling=None, pixels=None, title=None, colorbar=False):
        """show_matrix() for models

        settings: list of tuples
            matrices together with position and labels.
        matrix: np.ndarray, spmatrix
            a single matrix.
        factor_info: list of tuples
            the info tuples of each factor.
        factors: list of int lists
            the links of a matrix to its factors.
        """
        if not self.display:
            return
        
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None and matrix is None:
            if factor_info is not None: # use the order provided in factor_info
                U_info, V_info = factor_info
                U_order = reverse_index(idx=U_info[0])
                V_order = reverse_index(idx=V_info[0])
                U, V = self.U[U_order], self.V[V_order]
            else:
                U, V = self.U, self.V

            X = matmul(U, V.T, boolean=True, sparse=False)
            U, V = to_dense(U), to_dense(V)
            settings = [(X, [0, 0], "X"), (U, [0, 1], "U"),  (V.T, [1, 0], "V")]

        elif matrix is not None:
            settings = [(to_dense(matrix), [0, 0], title)]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, 
                    title=title, colorbar=colorbar)
            