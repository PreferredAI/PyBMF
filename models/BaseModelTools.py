import numpy as np
import pandas as pd
import os
from scipy.sparse import lil_matrix, hstack
import pickle
import time
from IPython.display import display
from utils import _make_name, ismat, get_prediction

class BaseModelTools():
    '''The helper class for BaseModel.
    '''
    def __init__(self):
        raise NotImplementedError('This is a helper class.')
    
    def set_params(self, **kwargs):
        '''Model parameters.

        The parameter list shows the commonly used meanings of them.

        Model parameters
        ----------------
        k : int
            Rank.
        U, V, Us : ndarray, spmatrix
            Initial factors when `init_method` is 'custom'.
        W : ndarray, spmatrix or str in {'mask', 'full'}
            Masking weight matrix. 
            For 'mask', it'll use all samples in `X_train` (both 1's and 0's) as a mask. 
            For 'full', it refers to a full 1's matrix.
        Ws : list of spmatrix, str in {'mask', 'full'}
            Masking weight matrices.
        alpha : list of float
            Importance weights for matrices.
        lr : float
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
        '''
        kwconfigs = ['task', 'seed', 'display', 'verbose', 'scaling', 'pixels']
        for param in kwargs:
            if param in kwconfigs:
                continue

            value = kwargs.get(param)
            setattr(self, param, value)

            # display
            if isinstance(value, list):
                value = len(value)
            if ismat(value):
                value = value.shape

            print("[I] {:<12} : {}".format(param, value))


    def set_config(self, **kwargs):
        '''Set system configurations.

        System configurations are those involved when calling the `fit()` method.
        They controls the global random seed generator, the verbosity and display settings.
        They also identify the type of task the model is dealing with, which affects the evaluation procedure.

        System configurations
        ---------------------
        task : str, {'prediction', 'reconstruction'}
            The type of evaluation task.
            When the datasets (`X_train`, `X_val` and `X_test`) are provided as `csr_matrix`, prediction tasks only measure the entries in the sparse matrix (these entries can be 0 or 1, see `negative_sampling()`), while reconstruction tasks measure the whole matrix (treat sparse matrix as numpy array).
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

        # triggered no matter if it's mentioned or not

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


    def _show_logs(self):
        '''Show logs.
        '''
        for log in self.logs.values():
            if isinstance(log, pd.DataFrame):
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    display(log)


    def _show_matrix(self):
        '''Show matrices.
        '''
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        settings = [(self.X_train, [0, 0], 'gt'), (self.X_pd, [0, 1], 'pd')]
        self.show_matrix(settings, colorbar=True, discrete=True, clim=[0, 1])


    def _save_model(self, path='../saved_models/', name=None):
        '''Save the model.
        '''
        name = _make_name(self)
        data = self.__dict__
        path = os.path.join(path, name + '.pickle')

        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("[I] model saved as: {}.pickle".format(name))

    def import_model(self, **kwargs):
        '''Import or inherit variables and parameters from another model.
        '''
        for attr in kwargs:
            setattr(self, attr, kwargs.get(attr))
            action = "Overwrote" if hasattr(self, attr) else "Imported"
            self.print_msg("{} model parameter: {}".format(action, attr))


    def _init_factors(self):
        '''Initialize the factors.
        '''
        if hasattr(self, 'U') or hasattr(self, 'V'):
            print("[I] U, V existed. Skipping initialization.")
            return
        if hasattr(self, 'k') and self.k is not None:
            self.U = lil_matrix((self.m, self.k))
            self.V = lil_matrix((self.n, self.k))
        else:
            self.U = lil_matrix((self.m, 1))
            self.V = lil_matrix((self.n, 1))


    def _init_logs(self):
        '''Initialize the logs.

        The `logs` is a `dict` that holds the records in one place.
        The types of records include but are not limited to `dataframe`, `array` and `list`.
        '''
        if not hasattr(self, 'logs'):
            self.logs = {}


    def early_stop(self, error=None, diff=None, n_iter=None, n_factor=None, msg=None, k=None):
        '''Stopping criteria detection and early stop.

        Parameters
        ----------
        k : int
            The number of factors to obtain. This will keep the first `k` columns in `self.U` and `self.V`.
        error : float
            Current error. To be compared with error tolerance `self.tol`.
        diff : float
            Current update difference. To be compared with difference threshold `self.min_diff`.
        n_iter : int
            Current number of iterations. To be compared with maximum number of iterations `self.max_iter`.
        n_factor : int
            Current number of factors. To be compared with maximum number of factors `self.k`.
        k ; int
            The number of factors to obtain.

        Returns
        -------
        is_improving : bool
            Whether the model is improving or not.
        '''
        is_improving = True

        if error is not None and hasattr(self, 'tol') and error < self.tol:
            self._early_stop(msg="Error lower than tolerance", k=k)
            is_improving = False
        if n_iter is not None and hasattr(self, 'max_iter') and n_iter > self.max_iter:
            self._early_stop(msg="Reach maximum iteration", k=k)
            is_improving = False
        if diff is not None and hasattr(self, 'min_diff') and diff < self.min_diff:
            self._early_stop(msg="Difference lower than threshold", k=k)
            is_improving = False
        if n_factor is not None and (hasattr(self, 'k') and self.k is not None) and n_factor >= self.k:
            self._early_stop(msg="Reach requested factor")
            is_improving = False
        if msg is not None:
            self._early_stop(msg=msg, k=k)
            is_improving = False

        return is_improving
        

    def _early_stop(self, msg, k=None):
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
            print("[W] Obtained {} factor(s).".format(k))
            self.truncate_factors(k)


    def set_factors(self, k, u, v):
        '''Add new factor (k = 0, 1, ...).
        '''
        if self.U.shape[1] < k + 1:
            self.extend_factors(k + 1)
        self.U[:, k] = u
        self.V[:, k] = v


    def truncate_factors(self, k):
        '''Get the first k factors (k = 1, 2, ...).
        '''
        self.U = self.U[:, :k]
        self.V = self.V[:, :k]


    def extend_factors(self, k):
        '''Increase the number of factors to k (k = 1, 2, ...).
        '''
        self.U = hstack([self.U, lil_matrix((self.m, k - self.U.shape[1]))]).tolil()
        self.V = hstack([self.V, lil_matrix((self.n, k - self.V.shape[1]))]).tolil()
    
    
    def print_msg(self, msg, type='I'):
        if self.verbose:
            print("[{}] {}".format(type, msg))