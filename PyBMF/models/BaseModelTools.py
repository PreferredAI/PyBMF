import numpy as np
import pandas as pd
import os
from scipy.sparse import lil_matrix, hstack
import pickle
import time
from IPython.display import display
from ..utils import _make_name, ismat, get_config

class BaseModelTools():
    '''The helper class for ``BaseModel``.
    '''
    def __init__(self):
        raise NotImplementedError('This is a helper class.')
    
    def set_params(self, **kwargs):
        '''Model parameters.

        The parameter list shows the commonly used meanings of them.

        Model parameters
        ----------------
        k : int
            The rank.
        U : ndarray, spmatrix
            Initial factor matrix when ``init_method`` is ``'custom'``.
        V : ndarray, spmatrix
            Initial factor matrix when ``init_method`` is ``'custom'``.
        Us : ndarray, spmatrix
            For collective matrix factorization. 
            Initial factor matrices when ``init_method`` is ``'custom'``.
        W : ndarray, spmatrix or str in {'mask', 'full'}
            Masking weight matrix. 
            For 'mask', it'll use all samples in ``X_train`` (both 1's and 0's) as a mask. 
            For 'full', it refers to a full 1's matrix.
        Ws : list of spmatrix, str in {'mask', 'full'}
            For collective matrix factorization. 
            Masking weight matrices.
        alpha : list of floats
            For collective matrix factorization. 
            Importance weights for matrices.
        lr : float
            The learning rate.
        reg : float
            The regularization parameter.
        tol : float
            The error tolerance.
            Fitting will stop when the specified error is below ``tol``.
        min_diff : float
            The minimal difference.
            Fitting will stop when the specified change is below ``min_diff``.
        max_iter : int
            The maximal number of iterations.
        init_method : str
            The initialization method.
        '''
        kwconfigs = ['task', 'seed', 'display', 'verbose', 'scaling', 'pixels', 'show_logs', 'save_model', 'show_result']
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

        System configurations are those involved when calling the ``fit()`` method.

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

        if "show_logs" in kwargs:
            self.show_logs = kwargs.get("show_logs")
            print("[I] show_logs    :", self.show_logs)
        else:
            self.show_logs = True
        if "save_model" in kwargs:
            self.save_model = kwargs.get("save_model")
            print("[I] save_model   :", self.save_model)
        else:
            self.save_model = True
        if "show_result" in kwargs:
            self.show_result = kwargs.get("show_result")
            print("[I] show_result  :", self.show_result)
        else:
            self.show_result = True


    def _start_timer(self):
        '''Start timer.
        '''
        self.time = time.time()


    def _make_name(self):
        '''Make name.
        '''
        if not hasattr(self, 'name'):
            self.name = _make_name(model=self)
            print("[I] name         :", self.name)


    def _stop_timer(self):
        '''Stop timer.
        '''
        if not hasattr(self, 'time'):
            print('[W] Timer not started.')
            return
        
        self.time = time.time() - self.time
        
        # convert elapsed time to hours, minutes, and seconds
        hours, seconds = divmod(self.time, 3600)
        minutes, seconds = divmod(seconds, 60)

        # format the elapsed time
        formatted_time = ""
        if hours > 0:
            formatted_time += f"{int(hours)}h"
        if minutes > 0:
            formatted_time += f"{int(minutes)}m"
        formatted_time += f"{int(seconds)}s"
        
        print("[I] time elapsed : ", formatted_time)
        self.time = formatted_time


    def _show_logs(self):
        '''Display all the dataframes in ``self.logs``.
        '''
        for log in self.logs.values():
            if isinstance(log, pd.DataFrame):
                with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    display(log)


    def _show_result(self):
        '''Display the prediction.

        Make sure ``self.X_pd`` is set properly before calling.

        For example:

        >>> self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        '''
        settings = [(self.X_train, [0, 0], 'gt'), (self.X_pd, [0, 1], 'pd')]
        self.show_matrix(settings, colorbar=True, discrete=True, clim=[0, 1])


    def _save_model(self, path=None, name=None):
        '''Save the model.

        Parameters
        ----------
        path : str
            Path to save the model.
        name : str
            Name of the model.
        '''
        name = self.name
        data = self.__dict__

        path = get_config(key="saved_models") if path is None else path
        path = os.path.join(path, name + '.pickle')

        self.pickle_path = path

        with open(path, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("[I] model saved as: {}.pickle".format(name))

    def import_model(self, **kwargs):
        '''Import or inherit variables and parameters from another model.

        Parameters
        ----------
        **kwargs
            The variables and parameters to be imported.
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

        The ``logs`` is a ``dict`` that holds the records in one place.
        The types of records include but are not limited to ``dataframe``, ``ndarray`` and ``list``.
        '''
        if not hasattr(self, 'logs'):
            self.logs = {}


    def early_stop(self, error=None, diff=None, n_iter=None, n_factor=None, msg=None, k=None, verbose=True):
        '''Stopping criteria detection and early stop.

        Parameters
        ----------
        k : int
            The number of factors to obtain. This will keep the first ``k`` columns in ``self.U`` and ``self.V``.
        error : float
            Current error. To be compared with error tolerance ``self.tol``.
        diff : float
            Current update difference. To be compared with difference threshold ``self.min_diff``.
        n_iter : int
            Current number of iterations. To be compared with maximum number of iterations ``self.max_iter``.
        n_factor : int
            Current number of factors. To be compared with maximum number of factors ``self.k``.
        k : int
            The number of factors to obtain.
        verbose : bool
            Whether to print the message.

        Returns
        -------
        is_improving : bool
            Whether the fitting should continue or not.
        '''
        is_improving = True

        if error is not None and hasattr(self, 'tol') and error <= self.tol:
            self._early_stop(msg="Error <= tolerance", verbose=verbose, k=k)
            is_improving = False
        if n_iter is not None and hasattr(self, 'max_iter') and n_iter > self.max_iter:
            self._early_stop(msg="Reach maximum iteration", verbose=verbose, k=k)
            is_improving = False
        if diff is not None and hasattr(self, 'min_diff') and diff < self.min_diff:
            self._early_stop(msg="Difference lower than threshold", verbose=verbose, k=k)
            is_improving = False
        if n_factor is not None and (hasattr(self, 'k') and self.k is not None) and n_factor >= self.k:
            self._early_stop(msg="Reach requested factor", verbose=verbose)
            is_improving = False
        if msg is not None:
            # forced early stop without reason
            self._early_stop(msg=msg, k=k)
            is_improving = False

        return is_improving
        

    def _early_stop(self, msg, verbose, k=None):
        '''To deal with early covergence or stop.

        Parameters
        ----------
        msg : str
            The message to be displayed.
        verbose : bool
            Whether to print the message.
        k : int, optional
            The number of factors obtained.
        '''
        if verbose:
            print("[W] Stopped in advance: " + msg)
        if k is not None:
            if verbose:
                print("[W] Obtained {} factor(s).".format(k))
            self.truncate_factors(k)


    def set_factors(self, k, u, v):
        '''Add new factor (k = 0, 1, ...).

        Parameters
        ----------
        k : int
            The number of factor to be added.
        u : numpy.ndarray
            The user factor to be added.
        v : numpy.ndarray
            The item factor to be added.
        '''
        if self.U.shape[1] < k + 1:
            self.extend_factors(k + 1)
        self.U[:, k] = u
        self.V[:, k] = v


    def truncate_factors(self, k):
        '''Get the first k factors (k = 1, 2, ...).

        Parameters
        ----------
        k : int
            The number of factors to obtain.
        '''
        self.U = self.U[:, :k]
        self.V = self.V[:, :k]


    def extend_factors(self, k):
        '''Increase the number of factors to k (k = 1, 2, ...).

        Parameters
        ----------
        k : int
            The number of factors to obtain.
        '''
        self.U = hstack([self.U, lil_matrix((self.m, k - self.U.shape[1]))]).tolil()
        self.V = hstack([self.V, lil_matrix((self.n, k - self.V.shape[1]))]).tolil()
    
    
    def print_msg(self, msg, type='I'):
        '''Print message.

        Parameters
        ----------
        msg : str
            The message to be printed.
        type : str
            The type of message, e.g. 'I' for info, 'W' for warning, 'E' for error.
        '''
        if self.verbose:
            print("[{}] {}".format(type, msg))