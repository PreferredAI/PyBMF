import numpy as np
from utils import show_matrix, matmul, to_sparse
from utils import header, record, eval, binarize
import time
from scipy.sparse import lil_matrix
from itertools import product
import pandas as pd
from itertools import product
from .BaseTools import BaseTools


class BaseModel(BaseTools):
    '''The base class for all MF models.    
    '''
    def __init__(self, **kwargs):
        '''Initialize the model with parameters.
        '''
        self.check_params(**kwargs)
        raise NotImplementedError('This is a template class.')
    

    def check_params(self, **kwargs):
        '''Check parameters upon model initialization and fitting.

        Model parameters are those frequently used when initializing the model.
        For now, they are: 'k', 'W', 'Ws', 'alpha', 'lr', 'reg', 'tol', 'min_diff', 'max_iter', 'init_method'.
        For the extra parameters you need, you can wrap this method into your own `check_params()`.

        System configurations are those involved when calling the `fit()` method.
        They controls the random seed generator and the verbosity and display settings.
        They also identify the type of task the model is dealing with, which affects the evaluation procedure.
        
        E.g. in your model class, you can do:
        .. code-block:: python
            def __init__(self, k, W, alpha):
                self.check_params(k=k, W=W, alpha=alpha)

            def fit(self, X_train, X_val=None, X_test=None, **kwargs):
                self.check_params(**kwargs)

            model.fit(X_train, X_val, X_test, seed=2024, task='prediction', verbose=False, display=True)
        '''
        self.set_params(['k', 'U', 'V', 'Us', 'W', 'Ws', 'alpha', 'lr', 'reg', 'tol', 'min_diff', 'max_iter', 'init_method'], **kwargs)
        self.set_config(**kwargs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        '''Fit the model to observations, with validation and prediction (experimental).

        The default preparations for a fitting procedure, followed by `_fit()` and `finish()`.
        Simply overwrite this method if you want to drop any parts or include more procedures.
        '''
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val, X_test=X_test)
        self.init_model()
        # attach these in your models:
        # self._fit()
        # self.finish()


    def init_model(self):
        '''Initialize the model.

        The default initialization procedure.
        Simply overwrite this method if you want to drop any parts or include more procedures.
        E.g. when you choose to import the factors from another model.
        '''
        self._init_factors()
        self._init_logs()


    def _fit(self):
        '''Where the tedious fitting procedure takes place.
        '''
        raise NotImplementedError('This is a template method.')


    def finish(self, show_logs=True, show_matrix=True, save_model=True):
        '''Called when the fitting is over.

        The default finishing procedure.
        Simply overwrite this method if you want to drop any parts or include more procedures.
        You can attach this to the end of `fit()` or simply call from outside.
        '''
        self._show_logs()
        self._show_matrix()
        self._save_model()


    def load_dataset(self, X_train, X_val=None, X_test=None):
        '''Load train and validation data.

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not getting modified, csr or csc is preferred.

        Parameters
        ----------
        X_train : array, spmatrix
            Data for matrix factorization.
        X_val : array, spmatrix
            Data for model selection.
        X_test : ndarray, spmatrix
            Data for prediction.
        '''
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


    def predict_X(self, U=None, V=None, u=None, v=None, boolean=True):
        '''Update `X_pd`.
        '''
        U = self.U if U is None else U
        V = self.V if V is None else V
        U = binarize(U, u) if u is not None else U
        V = binarize(V, v) if v is not None else V
        self.X_pd = matmul(U, V.T, boolean=boolean, sparse=True)


    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        '''The show_matrix() wrapper for BMF models.

        If the `settings` is missing, show the factors and their boolean product by default.
        '''
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
            The name of `dataframe` to record with.
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
