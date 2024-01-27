import numpy as np
from utils import show_matrix, reverse_index, dot, matmul, TP, FP, ERR, add_log
from utils import to_dense, to_sparse, to_triplet, get_metrics
import time
from scipy.sparse import spmatrix, lil_matrix, isspmatrix
from typing import Union, List, Tuple
import pandas as pd
from tqdm import tqdm


class BaseModel():
    def __init__(self) -> None:
        raise NotImplementedError("Missing init method.")


    def check_params(self, **kwargs):
        self.check_model_params(**kwargs)
        self.check_fit_params(**kwargs)
        self.check_eval_params(**kwargs)


    def check_fit_params(self, **kwargs):
        '''
        display: 
            to show matrix at each step during fitting and after finish.
        verbose: 
            to print result at each step during fitting.
        '''
        if not hasattr(self, 'verbose'):
            self.verbose = False
            print("[I] verbose      :", self.verbose)
        if not hasattr(self, 'display'):
            self.display = False
            print("[I] display      :", self.display)

        if "verbose" in kwargs:
            verbose = kwargs.get("verbose")
            if verbose != self.verbose:
                self.verbose = verbose
                print("[I] verbose      :", self.verbose)
        if "display" in kwargs:
            display = kwargs.get("display")
            if display != self.display:
                self.display = display
                print("[I] display      :", self.display)
            
        if "scaling" in kwargs and self.display:
            self.scaling = kwargs.get("scaling")
            print("[I]   scaling    :", self.scaling)
        else: # auto reset
            self.scaling = 1.0
        if "pixels" in kwargs and self.display:
            self.pixels = kwargs.get("pixels")
            print("[I]   pixels     :", self.pixels)
        else: # auto reset
            self.pixels = 2


    def check_model_params(self, **kwargs):
        '''
        k: 
            rank, number of patterns.
        seed: 
            random seed shared by the whole model.
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


    def check_eval_params(self, **kwargs):
        if "task" in kwargs:
            task = kwargs.get("task")
            assert task in ['prediction', 'reconstruction'], "Eval task is either 'prediction' or 'reconstruction'."
            self.task = task
            print("[I] task         :", self.task)


    def fit(self, X_train, X_val=None, **kwargs):
        """Fit the model to observations, with necessary validation
        
        X_train : np.ndarray, spmatrix
            data for factorization.
        X_val : np.ndarray, spmatrix
            data for model selection.
        kwargs :
            seed, verbose, task, display, scaling, pixels
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
            print("[I] Missing validation data.")
        self.X_train = to_sparse(X_train, 'csr')
        self.X_val = None if X_val is None else to_sparse(X_val, 'csr')

        self.m, self.n = self.X_train.shape


    def check_outcome(self):
        """Initialize outcomes
        """
        self.U = lil_matrix(np.zeros((self.m, self.k)))
        self.V = lil_matrix(np.zeros((self.n, self.k)))
        self.df = {} # dataframes
        self.U_history = {}
        self.V_history = {}


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
    

    def eval(self, X_gt, metrics, task):
        """Evaluation

        X_gt : np.ndarray, spmatrix
            ground truth matrix. can be X_train, X_val or X_test.
        metrics : list of str
            list of metric names.
        task : str
            prediction: 
                to use triplet data only.
                the triplet may contain 0's, depending on whether negative sampling has been used.
            reconstruction:
                to use the whole matrix, which considers missing values as zeros in spmatrix.
        """
        if task == 'prediction':
            U_idx, V_idx, gt_data = to_triplet(X_gt)
            if len(gt_data) < 5000:
                pd_data = np.zeros(len(gt_data), dtype=int)
                for i in tqdm(range(len(gt_data)), position=0, desc="[I] Making predictions"):
                    pd_data[i] = self.score(U_idx=U_idx[i], V_idx=V_idx[i])
            else:
                X_pd = matmul(U=self.U, V=self.V.T, sparse=True, boolean=True)
                pd_data = np.zeros(len(gt_data), dtype=int)
                for i in tqdm(range(len(gt_data)), position=0, desc="[I] Making predictions"):
                    pd_data[i] = X_pd[U_idx[i], V_idx[i]]
        elif task == 'reconstruction':
            gt_data = to_sparse(X_gt, type='csr')
            pd_data = matmul(U=self.U, V=self.V.T, sparse=True, boolean=True)
            
        results = get_metrics(gt=gt_data, pd=pd_data, metrics=metrics)
        return results
    

    def evaluate(self, X_gt, df, task, metrics=[], extra_metrics=[], extra_results=[], **kwargs):
        """Evaluate metrics with given ground truth data and save results to a dataframe 

        X_gt:
            ground truth matrix, like X_train, X_val or X_test.
        df:
            name of dataframe that the results are saved.
        names:
            names of the values.
        values:
            usually model parameters.
        metrics:
            list of metrics to evaluate.
        kwargs:
            verbose.
        """
        self.check_params(**kwargs)

        extra_metrics = ['time'] + extra_metrics
        extra_results = [pd.Timestamp.now().strftime("%d/%m/%y %I:%M:%S")] + extra_results

        if not df in self.df:
            self.df[df] = pd.DataFrame(columns=extra_metrics+metrics)

        results = self.eval(X_gt, metrics=metrics, task=task)
        add_log(df= self.df[df], line=extra_results+results, verbose=self.verbose)


    def early_stop(self, msg: str, k: int=None):
        print("[W] Stopped in advance: " + msg)
        if k is not None:
            print("[W]   got {} factor(s).".format(k))
            self.U = self.U[:, :k]
            self.V = self.V[:, :k]
    
    
    def print_msg(self, msg):
        if self.verbose:
            print("[I] " + msg)


    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        """The show_matrix() wrapper for BMF models

        If both settings is None, show the factors and their boolean product.
        """
        if not self.display:
            return
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None:
            X = matmul(self.U, self.V.T, boolean=True, sparse=True)
            settings = [(X, [0, 0], "X"), (self.U, [0, 1], "U"),  (self.V.T, [1, 0], "V")]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, **kwargs)
