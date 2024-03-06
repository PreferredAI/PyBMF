import numpy as np
from utils import show_matrix, matmul, to_sparse
from utils import header, record, eval
import time
from scipy.sparse import lil_matrix
from itertools import product
import pandas as pd
from itertools import product


class BaseModel():
    def __init__(self):
        raise NotImplementedError("Missing init method.")
    

    def check_params(self, **kwargs):
        '''Check parameters.

        Used in model initialization, fitting and evaluation.

        Parameters
        ----------
        k : int
        seed : int
        task : str, {'prediction', 'reconstruction'}
        display : bool, default: False
        verbose : bool, default: False
        scaling : float, default: 1.0
        pixels : int, default: 2
        '''
        # trigger when it's mentioned in kwargs
        if "task" in kwargs:
            task = kwargs.get("task")
            assert task in ['prediction', 'reconstruction'], "Eval task is either 'prediction' or 'reconstruction'."
            self.task = task
            print("[I] task         :", self.task)
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
        # trigger during initialization
        if not hasattr(self, 'verbose'):
            self.verbose = False
            print("[I] verbose      :", self.verbose)
        if not hasattr(self, 'display'):
            self.display = False
            print("[I] display      :", self.display)
        # trigger when it's getting changed
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
        # trigger no matter if it's mantioned or not
        if "scaling" in kwargs and self.display:
            self.scaling = kwargs.get("scaling")
            print("[I]   scaling    :", self.scaling)
        else:
            self.scaling = 1.0
        if "pixels" in kwargs and self.display:
            self.pixels = kwargs.get("pixels")
            print("[I]   pixels     :", self.pixels)
        else:
            self.pixels = 2


    def fit(self, X_train, X_val=None, **kwargs):
        """Fit the model to observations, with validation if necessary.

        Please implement your own fit method.
        
        X_train : array, spmatrix
            Data for matrix factorization.
        X_val : array, spmatrix
            Data for model selection.
        kwargs :
            Common parameters that are checked and set in `BaseModel.check_params()` and the model-specific parameters included in `self.check_params()`.
        """
        raise NotImplementedError("Missing fit method.")
    

    def load_dataset(self, X_train, X_val=None):
        """Load train and validation data.

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not getting modified, csr or csc is preferred.

        X_train : array, spmatrix
        X_val : array, spmatrix
        """
        if X_train is None:
            raise TypeError("Missing training data.")
        if X_val is None:
            print("[I] Missing validation data.")
        self.X_train = to_sparse(X_train, 'csr')
        self.X_val = None if X_val is None else to_sparse(X_val, 'csr')

        self.m, self.n = self.X_train.shape


    def init_model(self):
        """Initialize factors and logging variables.

        U : spmatrix
        V : spmatrix
        logs : dict
            The dict containing dataframes, arrays and lists.
        """
        self.U = lil_matrix(np.zeros((self.m, self.k)))
        self.V = lil_matrix(np.zeros((self.n, self.k)))
        self.logs = {}


    def early_stop(self, msg: str, k: int=None):
        print("[W] Stopped in advance: " + msg)
        if k is not None:
            print("[W]   got {} factor(s).".format(k))
            self.U = self.U[:, :k]
            self.V = self.V[:, :k]
    
    
    def print_msg(self, msg, type='I'):
        if self.verbose:
            print("[{}] {}".format(type, msg))


    def predict_X(self):
        self.X_pd = matmul(self.U, self.V.T, boolean=True, sparse=True)


    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        """The show_matrix() wrapper for BMF models.

        If `settings` is missing, show the factors and their boolean product.
        """
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None:
            self.predict_X()
            settings = [(self.X_pd, [0, 0], "X"), 
                        (self.U, [0, 1], "U"), 
                        (self.V.T, [1, 0], "V")]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, **kwargs)


    def evaluate(self, df_name, names=[], values=[], metrics=['Recall', 'Precision', 'Accuracy', 'F1']):
        self.predict_X()
        
        results_train = eval(X_gt=self.X_train, X_pd=self.X_pd, 
            metrics=metrics, task=self.task)
        columns = header(names) + list(product(['train'], metrics))
        results = values + results_train
        
        if self.X_val is not None:
            results_val = eval(X_gt=self.X_val, X_pd=self.X_pd, 
                metrics=metrics, task=self.task)
            columns = columns + list(product(['val'], metrics))
            results = results + results_val
        
        record(df_dict=self.logs, df_name=df_name, columns=columns, records=results, verbose=self.verbose)
