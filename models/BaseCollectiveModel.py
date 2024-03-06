from .BaseModel import BaseModel
import pandas as pd
import numpy as np
from scipy.sparse import spmatrix, lil_matrix
from itertools import product
from utils import split_factor_list, get_factor_list, get_matrices, get_settings, get_factor_dims, concat_Xs_into_X, get_factor_starts
from utils import to_sparse, dot, matmul, show_matrix, get_metrics, record, eval, header, to_triplet
from tqdm import tqdm


class BaseCollectiveModel(BaseModel):
    def __init__(self) -> None:
        raise NotImplementedError("Missing init method.")
    

    def fit(self, Xs_train, factors, Xs_val=None, **kwargs):
        """Fit the model to observations, with validation if necessary.

        Please implement your own fit method.
        
        X_train : ndarray, spmatrix
            Data for matrix factorization.
        X_val : ndarray, spmatrix
            Data for model selection.
        kwargs :
            Common parameters that are checked and set in `BaseModel.check_params()` and the model-specific parameters included in `self.check_params()`.
        """
        raise NotImplementedError("Missing fit method.")
        
        
    def load_dataset(self, Xs_train, factors, Xs_val=None):
        """Load train and val data.

        For matrices that are modified frequently, lil (LIst of List) or coo is preferred.
        For matrices that are not modified, csr or csc is preferred.

        Parameters
        ----------
        Xs_train : list of np.ndarray or spmatrix
            List of Boolean matrices for training.
        factors : list of int list
            List of factor id pairs, indicating the row and column factors of each matrix.
        Xs_val : list of np.ndarray, spmatrix and None, optional
            List of Boolean matrices for validation. It should have the same length of Xs_train. In the ``list``, `None` can be used as placeholders of those matrices that are not being validated. When `Xs_val` is `None`, the matrix with factor id [0, 1] is used as the only matrix being validated.
        """
        if Xs_train is None:
            raise TypeError("Missing training data.")
        if factors is None:
            raise TypeError("Missing factors.")
        if Xs_val is None:
            print("[W] Missing validation data.")

        self.Xs_train = [to_sparse(X, 'csr') for X in Xs_train]
        self.factors = factors
        self.Xs_val = None if Xs_val is None else [to_sparse(X, 'csr') for X in Xs_val]

        self.X_train = concat_Xs_into_X(Xs_train, factors)
        self.matrices = get_matrices(factors)
        self.factor_list = get_factor_list(factors)
        self.factor_dims = get_factor_dims(Xs_train, factors)
        self.row_factors, self.col_factors = split_factor_list(factors)
        self.row_starts, self.col_starts = get_factor_starts(Xs_train, factors)

        self.n_factors = len(self.factor_list)
        self.n_matrices = len(Xs_train)
        

    def init_model(self):
        """Initialize factors and logging variables.

        Us : list of spmatrix
        logs : dict
            The dict containing dataframes, arrays and lists.
        """
        self.Us = []
        for dim in self.factor_dims:
            U = lil_matrix(np.zeros((dim, self.k)))
            self.Us.append(U)
        self.logs = {}


    def show_matrix(self, settings=None, scaling=None, pixels=None, **kwargs):
        """The show_matrix() wrapper for CMF models.

        If `settings` is None, show the factors and their boolean product.
        """
        if not self.display:
            return
        scaling = self.scaling if scaling is None else scaling
        pixels = self.pixels if pixels is None else pixels

        if settings is None:
            self.predict_Xs()
            settings = get_settings(Xs=self.Xs_pd, factors=self.factors, Us=self.Us)

        show_matrix(settings=settings, scaling=scaling, pixels=pixels, **kwargs)


    def predict_Xs(self):
        if not hasattr(self, 'Xs'):
            self.Xs_pd = [None] * self.n_matrices
        for i, factors in enumerate(self.factors):
            a, b = factors
            X = matmul(U=self.Us[a], V=self.Us[b].T, boolean=True, sparse=True)
            self.Xs_pd[i] = X


    #### collective score, eval and evaluate() ####


    def collective_score(self, m, U_idx, V_idx):
        """Predict the scores/ratings of a user for an item.
        """
        a, b = self.factors[m]
        return dot(self.Us[a][U_idx], self.Us[b][V_idx], boolean=True)
    

    def collective_eval(self, X_gt, m, metrics, task):
        """Evaluation with given metrics.

        X_gt : array, spmatrix
            The ground-truth matrix.
        metrics : list of str
            List of metric names.
        task : str, {'prediction', 'reconstruction'}
            prediction: 
                Ignore the missing values and only use the triplet from the ``spmatrix``. The triplet may contain zeros, depending on whether negative sampling has been used.
            reconstruction:
                Use the whole matrix, which considers missing values as zeros in ``spmatrix``.
        """
        if task == 'prediction':
            U_idx, V_idx, gt_data = to_triplet(X_gt)
            if len(gt_data) < 5000:
                pd_data = np.zeros(len(gt_data), dtype=int)
                for i in tqdm(range(len(gt_data)), position=0, desc="[I] Making predictions"):
                    pd_data[i] = self.collective_score(m=m, U_idx=U_idx[i], V_idx=V_idx[i])
            else:
                # X_pd = matmul(U=self.U, V=self.V.T, sparse=True, boolean=True)
                X_pd = self.Xs_pd[m] # make sure it's updated
                pd_data = np.zeros(len(gt_data), dtype=int)
                for i in tqdm(range(len(gt_data)), position=0, desc="[I] Making predictions"):
                    pd_data[i] = X_pd[U_idx[i], V_idx[i]]
        elif task == 'reconstruction':
            gt_data = to_sparse(X_gt, type='csr')
            # pd_data = matmul(U=self.U, V=self.V.T, sparse=True, boolean=True)
            pd_data = self.Xs_pd[m] # make sure it's updated
            
        results = get_metrics(gt=gt_data, pd=pd_data, metrics=metrics)
        return results
    

    def collective_evaluate(self, X_gt, m, df_name, task, metrics=[], extra_metrics=[], extra_results=[], **kwargs):
        """Evaluate metrics with given ground-truth data and save results to a dataframe.

        X_gt : array, spmatrix
        m : int
            Matrix id.
        df_name : str
            Name of the dataframe that the results are going to be saved under.
        task : str
        matrics : list of str
            List of metric names to be evaluated.
        extra_metrics : list of str
            List of extra metric names.
        extra_results : list of int and float
            List of results of extra metrics.
        kwargs :
            Common parameters that are checked and set in `BaseModel.check_params()`.
        """
        self.check_params(**kwargs)

        if X_gt is None:
            return

        extra_metrics = ['time'] + extra_metrics
        extra_results = [pd.Timestamp.now().strftime("%d/%m/%y %I:%M:%S")] + extra_results

        if not df_name in self.logs:
            self.logs[df_name] = pd.DataFrame(columns=extra_metrics+metrics)

        results = self.collective_eval(X_gt, m=m, metrics=metrics, task=task)

        add_log(df=self.logs[df_name], line=extra_results+results, verbose=self.verbose, caption=df_name)

        return results # newly added
    

    def evaluate(self, df_name, names=[], values=[], metrics=['Recall', 'Precision', 'Accuracy', 'F1']):
        self.predict_Xs()
        
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