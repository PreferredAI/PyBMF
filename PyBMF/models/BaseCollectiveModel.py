from .BaseModel import BaseModel
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix, lil_matrix
from itertools import product
from ..utils import split_factor_list, get_factor_list, get_matrices, get_settings, get_factor_dims, concat_Xs_into_X, get_factor_starts, power
from ..utils import to_sparse, dot, matmul, show_matrix, get_metrics, record, eval, header, to_triplet, binarize, isnum
from ..utils import weighted_score, harmonic_score
from tqdm import tqdm


class BaseCollectiveModel(BaseModel):
    def __init__(self) -> None:
        raise NotImplementedError("This is a template class.")
    

    def fit(self, Xs_train, factors, Xs_val=None, Xs_test=None, **kwargs):
        '''Fit the model to observations, with validation and prediction if necessary.
        '''
        # these are the common routines when the fitting starts:

        self.check_params(**kwargs)
        self.load_dataset(Xs_train, factors, Xs_val, Xs_test)
        self.init_model()
        
        # attach these in your models:

        # self._fit()
        # self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)
        
        
    def load_dataset(self, Xs_train, factors, Xs_val=None, Xs_test=None):
        '''Load train and val data.

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not modified, csr or csc is preferred.

        Parameters
        ----------
        Xs_train : list of np.ndarray or spmatrix
            List of Boolean matrices for training.
        factors : list of int list
            List of factor id pairs, indicating the row and column factors of each matrix.
        Xs_val : list of np.ndarray, spmatrix and None, optional
            List of Boolean matrices for validation. It should have the same length of `Xs_train`.
        Xs_test : list of np.ndarray, spmatrix and None, optional
            List of Boolean matrices for testing. It should have the same length of `Xs_train`.
        '''
        if Xs_train is None:
            raise TypeError("Missing training data.")
        if factors is None:
            raise TypeError("Missing factors.")
        if Xs_val is None:
            print("[W] Missing validation data.")
        if Xs_test is None:
            print("[W] Missing testing data.")

        self.Xs_train = [to_sparse(X, 'csr') for X in Xs_train]
        self.factors = factors
        self.Xs_val = None if Xs_val is None else [to_sparse(X, 'csr') for X in Xs_val]
        self.Xs_test = None if Xs_test is None else [to_sparse(X, 'csr') for X in Xs_test]

        self.X_train = concat_Xs_into_X(Xs_train, factors)
        self.matrices = get_matrices(factors)
        self.factor_list = get_factor_list(factors)
        self.factor_dims = get_factor_dims(Xs_train, factors)
        self.row_factors, self.col_factors = split_factor_list(factors)
        self.row_starts, self.col_starts = get_factor_starts(Xs_train, factors)

        self.n_factors = len(self.factor_list)
        self.n_matrices = len(Xs_train)
        

    def init_model(self):
        '''The `BaseModel.init_model()` for collective models.
        '''
        self._init_factors()
        self._init_logs()


    def _init_factors(self):
        if not hasattr(self, 'Us'):
            self.Us = [lil_matrix((dim, self.k)) for dim in self.factor_dims]


    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        '''The show_matrix() wrapper for CMF models.

        If `settings` is None, show the factors and their boolean product.
        '''
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None:
            settings = get_settings(Xs=self.Xs_pd, factors=self.factors, Us=self.Us)

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, **kwargs)


    def predict_Xs(self, Us=None, us=None, boolean=True):
        '''Get predictions.

        us : list of float or float, optional
            If not None, the `us` are used to binarize the factors.
            If len(us) == self.n_factors, `us` are extended to `self.n_factors` * `self.k`.
        '''
        # init Xs_pd
        if not hasattr(self, 'Xs_pd'):
            self.Xs_pd = [None] * self.n_matrices
        # load Us
        if Us is None:
            Us = self.Us.copy()
        # reformat us
        if isnum(us):
            us = [us] * (self.n_factors * self.k)
        # replicate us
        if us is not None and len(us) == self.n_factors:
            us = [u for u in us for _ in range(self.k)]
        print(us)
        # binarize
        if us is not None:
            for j in range(self.n_factors):
                for i in range(self.k):
                    Us[j][:, i] = binarize(Us[j][:, i], us[i + j * self.k])
        # generate prediction
        for i, factors in enumerate(self.factors):
            a, b = factors
            X = matmul(U=Us[a], V=Us[b].T, boolean=boolean, sparse=None)
            self.Xs_pd[i] = X


    def evaluate(self, df_name, 
            head_info={}, train_info={}, val_info={}, test_info={}, 
            metrics=['Recall', 'Precision', 'Accuracy', 'F1'], 
            train_metrics=None, val_metrics=None, test_metrics=None, verbose=False):
        '''Evaluate a CMF model on the given train, val and test daatset.
        '''
        train_metrics = metrics if train_metrics is None else train_metrics
        val_metrics = metrics if val_metrics is None else val_metrics
        test_metrics = metrics if test_metrics is None else test_metrics

        columns = header(list(head_info.keys()), levels=3)
        results = list(head_info.values())

        c, r = self._evaluate('train', train_info, train_metrics)
        columns += c
        results += r

        if self.Xs_val is not None:
            c, r = self._evaluate('val', val_info, val_metrics)
            columns += c
            results += r
            
        if self.Xs_test is not None:
            c, r = self._evaluate('test', test_info, test_metrics)
            columns += c
            results += r
        
        record(df_dict=self.logs, df_name=df_name, columns=columns, records=results, verbose=verbose)


    def _evaluate(self, name, info, metrics):
        '''Evaluate on a given dataset.

        Parameters
        ----------
        name : str in ['train', 'val', 'test']
        info : dict of list
            If a value in the `dict` is a `list`, and the length equals `n_matrices`, the values in the `list` will be assigned to the head of each matrix in the dataset. Make sure the order matches the order of matrices.
        metrics : list of str
        '''
        columns, results = [], []
        r_array = np.zeros((self.n_matrices, len(info) + len(metrics)))

        for m in range(self.n_matrices):
            Xs_gt = getattr(self, 'Xs_' + name)
            m_results = eval(X_gt=Xs_gt[m], X_pd=self.Xs_pd[m], metrics=metrics, task=self.task)
            m_info = [v[m] for v in info.values()]

            columns += list(product([name], [m], list(info.keys()) + metrics))

            results += m_info + m_results
            
            r_array[m] = m_info + m_results

        w_results = weighted_score(r_array, self.alpha).flatten()
        h_results = harmonic_score(r_array).flatten()

        columns += list(product([name], ['weighted', 'harmonic'], list(info.keys()) + metrics))
        results += list(w_results) + list(h_results)

        return columns, results