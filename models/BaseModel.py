import numpy as np
from utils import show_matrix, matmul, to_sparse
from utils import header, record, eval, binarize
import time
from scipy.sparse import lil_matrix
from itertools import product
import pandas as pd
from itertools import product


class BaseModel():
    def __init__(self):
        raise NotImplementedError("This is a template class.")
    

    def check_params(self, **kwargs):
        '''Shared parameters called upon model initialization, fitting and evaluation.

        Model parameters
        ----------------
        k : int
            Rank.
        W : ndarray, spmatrix or str in {'mask', 'full'}
            Masking weight matrix. 
            For 'mask', it'll use all samples in `X_train` (both 1's and 0's) as a mask. 
            For 'full', it refers to a full 1's matrix.
        Ws : list of spmatrix, str in {'mask', 'full'}
            Masking weight matrices.
        alpha : list of float
            Importance weights for matrices.
        lf : float
            Learning rate.
        reg : float
            Regularization weight.
        tol : float
            Error tolerance.
        min_diff : float
            Minimal difference.
        max_iter : int
            Maximal number of iterations.
        init_method : str
            Initialization method.

        System parameters
        -----------------
        task : str, {'prediction', 'reconstruction'}
            The type of task when evaluating.
        seed : int
            Model seed.
        display : bool, default: False
            Switch for visualization.
        verbose : bool, default: False
            Switch for verbosity.
        scaling : float, default: 1.0
            Scaling of images in visualization.
        pixels : int, default: 2
            Resolution of images in visualization.
        '''
        # frequently used
        self.set_params(['k', 'W', 'Ws', 'alpha', 'lr', 'reg', 'tol', 'min_diff', 'max_iter', 'init_method'], **kwargs)
        # triggered when it's mentioned in kwargs
        if "task" in kwargs:
            task = kwargs.get("task")
            assert task in ['prediction', 'reconstruction'], "Eval task must be 'prediction' or 'reconstruction'."
            self.task = task
            print("[I] task         :", self.task)
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if seed is None and not hasattr(self,'seed'):
                # use time as self.seed
                seed = int(time.time())
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            elif seed is not None:
                # overwrite self.seed
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            else:
                # self.rng remains unchanged
                pass
        # triggered upon initialization
        if not hasattr(self, 'verbose'):
            self.verbose = False
            print("[I] verbose      :", self.verbose)
        if not hasattr(self, 'display'):
            self.display = False
            print("[I] display      :", self.display)
        # triggered when it's getting changed
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
        # triggered no matter if it's mantioned or not
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


    def set_params(self, params, **kwargs):
        '''Set parameters without checking.

        Parameters
        ----------
        params : str or list of str
        '''
        for param in list(params):
            if param in kwargs:
                value = kwargs.get(param)
                setattr(self, param, value)
                print("[I] {:<12} : {}".format(param, value.shape if hasattr(value, 'shape') else value))


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        """Fit the model to observations, with validation and prediction if necessary.

        Here are the preparations for a normal fitting procedure
        """
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val, X_test=X_test)
        self.init_model()


    def _finish(self):
        for log in self.logs.values():
            if isinstance(log, pd.DataFrame):
                display(log)
    

    def load_dataset(self, X_train, X_val=None, X_test=None):
        """Load train and validation data.

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not getting modified, csr or csc is preferred.

        X_train : array, spmatrix
            Data for matrix factorization.
        X_val : array, spmatrix
            Data for model selection.
        X_test : ndarray, spmatrix
            Data for prediction.
        """
        if X_train is None:
            raise TypeError("Missing training data.")
        if X_val is None:
            print("[I] Missing validation data.")
        if X_test is None:
            print("[W] Missing testing data.")

        self.X_train = to_sparse(X_train, 'csr')
        self.X_val = None if X_val is None else to_sparse(X_val, 'csr')
        self.X_test = None if X_test is None else to_sparse(X_test, 'csr')

        self.m, self.n = self.X_train.shape


    def import_model(self, **kwargs):
        """Import or inherit variables and parameters from another model.
        """
        for attr in kwargs:
            setattr(self, attr, kwargs.get(attr))
            action = "Overwrote" if hasattr(self, attr) else "Imported"
            self.print_msg("{} model parameter: {}".format(action, attr))


    def init_model(self):
        """Initialize factors and logging variables.

        logs : dict
            The ``dict`` containing the logging data recorded in ``dataframe``, ``array`` or ``list``.
        """
        if not hasattr(self, 'U') or not hasattr(self, 'V'):
            self.U = lil_matrix(np.zeros((self.m, self.k)))
            self.V = lil_matrix(np.zeros((self.n, self.k)))

        if not hasattr(self, 'logs'):
            self.logs = {}


    def early_stop(self, error=None, diff=None, n_iter=None):
        """Early stop detection.
        """
        should_continue = True

        if error is not None and hasattr(self, 'tol') and error < self.tol:
            self._early_stop("Error lower than tolerance")
            should_continue = False
        if n_iter is not None and hasattr(self, 'max_iter') and n_iter > self.max_iter:
            self._early_stop("Reach maximum iteration")
            should_continue = False
        if diff is not None and hasattr(self, 'min_diff') and diff < self.min_diff:
            self._early_stop("Difference lower than threshold")
            should_continue = False

        return should_continue
        

    def _early_stop(self, msg: str, k: int=None):
        '''To deal with early covergence or stop.

        Parameters
        ----------
        msg : str
            The message to be displayed.
        k : int, optional
            The number of factors obtained.
        '''
        print("[W] Stopped in advance: " + msg)
        if k is not None:
            print("[W]   got {} factor(s).".format(k))
            self.U = self.U[:, :k]
            self.V = self.V[:, :k]
    
    
    def print_msg(self, msg, type='I'):
        if self.verbose:
            print("[{}] {}".format(type, msg))


    def predict_X(self, U=None, V=None, u=None, v=None, boolean=True):
        U = self.U if U is None else U
        V = self.V if V is None else V
        U = binarize(U, u) if u is not None else U
        V = binarize(V, v) if v is not None else V
        self.X_pd = matmul(U, V.T, boolean=boolean, sparse=True)


    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        """The show_matrix() wrapper for BMF models.

        If `settings` is missing, show the factors and their boolean product.
        """
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None:
            settings = [(self.X_pd, [0, 0], "X"), (self.U, [0, 1], "U"), (self.V.T, [1, 0], "V")]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, **kwargs)


    def evaluate(self, df_name, 
            head_info={}, train_info={}, val_info={}, test_info={}, 
            metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
            train_metrics=None, val_metrics=None, test_metrics=None):
        '''Evaluate a BMF model on the given train, val and test daatset.

        Parameters
        ----------
        df_name : str
            The name of ``dataframe`` to record with.
        head_info : dict
            The names and values of shared information at the head of each record.
        train_info : dict
            The names and values of external information measured on training data.
        val_info : dict
            The names and values of external information measured on validation data.
        test_info : dict
            The names and values of external information measured on testing data.
        metrics : list of str, default: ['Recall', 'Precision', 'Accuracy', 'F1']
            The metrics to be measured. For metric names check `utils.get_metrics`.
        train_metrics : list of str, optional
            The metrics to be measured on training data. Will use `metrics` instead if it's `None`.
        val_metrics : list of str, optional
            The metrics to be measured on validation data. Will use `metrics` instead if it's `None`.
        test_metrics : list of str, optional
            The metrics to be measured on testing data. Will use `metrics` instead if it's `None`.
        '''
        train_metrics = metrics if train_metrics is None else train_metrics
        val_metrics = metrics if val_metrics is None else val_metrics
        test_metrics = metrics if test_metrics is None else test_metrics

        columns = header(list(head_info.keys()), levels=3)
        results = list(head_info.values())

        c, r = self._evaluate('train', train_info, train_metrics)
        columns += c
        results += r

        if self.X_val is not None:
            c, r = self._evaluate('val', val_info, val_metrics)
            columns += c
            results += r
            
        if self.X_test is not None:
            c, r = self._evaluate('test', test_info, test_metrics)
            columns += c
            results += r
        
        record(df_dict=self.logs, df_name=df_name, columns=columns, records=results, verbose=self.verbose)


    def _evaluate(self, name, info, metrics):
        '''Evaluate on a given dataset.

        Parameters
        ----------
        name : str in ['train', 'val', 'test']
        info : dict of list
        metrics : list of str
        '''
        X_gt = getattr(self, 'X_' + name)
        results = eval(X_gt=X_gt, X_pd=self.X_pd, metrics=metrics, task=self.task)
        columns = list(product([name], [0], list(info.keys()) + metrics))
        results = list(info.values()) + results
        
        return columns, results
