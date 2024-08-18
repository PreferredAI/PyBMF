import numpy as np
from ..utils import show_matrix, matmul, to_sparse
from ..utils import header, record, eval, binarize
import pandas as pd
from itertools import product
from .BaseModelTools import BaseModelTools
import time


class BaseModel(BaseModelTools):
    '''The base class for all the models.
    
    Initialize the model with parameters.
    '''
    def __init__(self, **kwargs):
        raise NotImplementedError('This is a template class.')
    

    def check_params(self, **kwargs):
        '''Check and load model parameters and experiment configurations.
        
        Called upon model initialization and fitting.
        
        .. code-block:: python

            # include this in your model class:

            def __init__(self, k, tol, alpha):
                self.check_params(k=k, tol=tol, alpha=alpha)

            def fit(self, X_train, X_val=None, X_test=None, **kwargs):
                self.check_params(**kwargs)

            # call them when initializing and fitting:

            model = MyModel(k=10, W='mask', alpha=0.1, seed=1997)
            
            model.fit(X_train, X_val, X_test, seed=2024, task='prediction', verbose=False, display=True)
        '''
        self.set_params(**kwargs)
        self.set_config(**kwargs)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        '''Fit the model to observations, with validation and test if necessary.

        For a fitting procedure, implement and append your ``_fit()`` and ``finish()``.

        Simply overwrite this method if you want to drop or add any parts of the procedures.

        Parameters
        ----------
        X_train : ndarray
            Training data.
        X_val : ndarray
            Validation data.
        X_test : ndarray
            Test data.
        **kwargs : dict
            Other parameters.
        '''
        # these are the common routines when the fitting starts:
        
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val, X_test=X_test)
        self.init_model()

        # attach these in your models:

        # self._fit()
        # self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def init_model(self):
        '''Initialize the model.
        
        Called after params are set and datasets are loaded.

        Simply overwrite this method if you want to drop or add any parts of the procedures.
        '''
        self._init_factors()
        self._init_logs()
        self._start_timer()
        self._make_name()

        # attach more in your models if needed, for example:

        # if you have more initialization methods:
        # self.init_U() or self.init_Us()

        # if you accept masking matrices:
        # self.init_W() or self.init_Ws()

        # if you want to force the model to work with dense `ndarray`:
        # self._to_dense()


    def _fit(self):
        '''Where the tedious fitting procedure takes place.
        '''
        raise NotImplementedError('This is a template method.')


    def finish(self, show_logs=True, save_model=True, show_result=True):
        '''Called when the fitting is over.

        The default finishing procedure.

        Simply overwrite this method if you want to drop or add any parts of the procedures.

        You can attach this to the end of ``fit()`` or simply call from outside.
        '''
        self._stop_timer()
        if save_model:
            self._save_model()
        if show_result:
            self._show_result()
        if show_logs:
            self._show_logs()



    def load_dataset(self, X_train, X_val=None, X_test=None):
        '''Load train and validation data.

        For matrices that are modified frequently, ``lil`` (LIst of List) or ``coo`` is preferred.

        For matrices that are not getting modified, ``csr`` or ``csc`` is preferred.

        Parameters
        ----------
        X_train : ndarray, spmatrix
            Data for matrix factorization.
        X_val : ndarray, spmatrix
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


    def predict_X(self, U=None, V=None, u=None, v=None, us=None, vs=None, boolean=True):
        '''Update the prediction ``X_pd``.

        Parameters
        ----------
        U : ndarray, spmatrix
            Factor matrix.
        V : ndarray, spmatrix
            Factor matrix.
        u : float
            The shared threshold for factors in ``U``. 
        v : float
            The shared threshold for factors in ``V``.
        us : list of k floats
            The thresholds for each factor in ``U``.
        vs : list of k floats
            The thresholds for each factor in ``V``.
        boolean : bool
            Whether to apply Boolean multiplication.
        '''
        U = self.U.copy() if U is None else U.copy()
        V = self.V.copy() if V is None else V.copy()

        if us is not None:
            assert len(us) == U.shape[1]
            for i in range(U.shape[1]):
                U[:, i] = binarize(U[:, i], us[i])
        elif u is not None:
            U = binarize(U, u)

        if vs is not None:
            assert len(vs) == V.shape[1]
            for i in range(V.shape[1]):
                V[:, i] = binarize(V[:, i], vs[i])
        elif v is not None:
            V = binarize(V, v)

        self.X_pd = matmul(U, V.T, boolean=boolean, sparse=True)




    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        '''The ``show_matrix()`` wrapper for BMF models.

        If ``settings`` is missing, show the factors ``U``, ``V`` and ``X_pd`` by default.
        '''
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None:
            settings = [(self.X_pd, [0, 0], "X"), (self.U, [0, 1], "U"), (self.V.T, [1, 0], "V")]

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, **kwargs)


    def evaluate(self, df_name, 
            head_info={}, train_info={}, val_info={}, test_info={}, 
            metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
            train_metrics=None, val_metrics=None, test_metrics=None, verbose=False):
        '''Evaluate a BMF model on the given train, val and test dataset.

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
            The metrics to be measured. For metric names check ``utils.get_metrics``.
        train_metrics : list of str, optional
            The metrics to be measured on training data. Will use `metrics` instead if it's ``None``.
        val_metrics : list of str, optional
            The metrics to be measured on validation data. Will use `metrics` instead if it's ``None``.
        test_metrics : list of str, optional
            The metrics to be measured on testing data. Will use `metrics` instead if it's ``None``.
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

        record(df_dict=self.logs, df_name=df_name, columns=columns, records=results, verbose=verbose)


    def _evaluate(self, name, info, metrics):
        '''Evaluate on a given dataset.

        Parameters
        ----------
        name : str in ['train', 'val', 'test']
            Which matrix to evaluate.
        info : dict of list
            The extra information to be recorded.
        metrics : list of str
            The metrics to be evaluated and recorded.
        '''
        X_gt = getattr(self, 'X_' + name)
        results = eval(X_gt=X_gt, X_pd=self.X_pd, metrics=metrics, task=self.task)
        columns = list(product([name], [0], list(info.keys()) + metrics))
        results = list(info.values()) + results
        
        return columns, results
