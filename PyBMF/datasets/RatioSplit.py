from .BaseSplit import BaseSplit
from math import ceil
import numpy as np


class RatioSplit(BaseSplit):
    '''Ratio split, usually used in prediction tasks.

    Parameters
    ----------
    X : ndarray, spmatrix
        The data matrix.
    test_size : int or float
        If it is int, ``test_size`` is the integer size of dataset. 
        If it is float, ``test_size`` is the fraction size of dataset.
    val_size : int or float
        If it is int, ``val_size`` is the integer size of dataset.
        If it is float, ``val_size`` is the fraction size of dataset.
    seed : int
        Random seed.
    '''
    def __init__(self, X, test_size=None, val_size=None, seed=None):
        super().__init__(X)
        print("[I] RatioSplit, sampling positives")     
        self.check_params(seed=seed)

        train_size, val_size, test_size = self.get_size(
            val_size=val_size, test_size=test_size, n_ratings=self.X.nnz)

        print("[I]   train_size   :", train_size)
        print("[I]   val_size     :", val_size)
        print("[I]   test_size    :", test_size)
        print("[I]   seed         :", self.seed)

        data_idx = self.rng.permutation(self.X.nnz)

        train_idx, val_idx, test_idx = self.get_indices(
            data_idx, train_size, test_size)

        self.load_pos_data(train_idx, val_idx, test_idx)


    def negative_sample(self, train_size=None, test_size=None, val_size=None, seed=None, type='uniform'):
        '''Select and append negative samples onto train, val and test set.

        Parameters
        ----------
        train_size : int or float
            If it is int, ``train_size`` is the integer size of dataset.
            If it is float, ``train_size`` is the fraction size of dataset.
        test_size : int or float
            If it is int, ``test_size`` is the integer size of dataset.
            If it is float, ``test_size`` is the fraction size of dataset.
        val_size : int or float
            If it is int, ``val_size`` is the integer size of dataset.
            If it is float, ``val_size`` is the fraction size of dataset.
        seed : int
            Random seed.
        type : str in {'uniform', 'popularity'}
            Type of negative sampling.
        '''
        print("[I] RatioSplit, sampling negatives")        
        self.check_params(seed=seed)
        m, n = self.X.shape
        all_negatives = m * n - self.X.nnz

        train_size, val_size, test_size = self.get_size(
            train_size=train_size, val_size=val_size, test_size=test_size, n_ratings=all_negatives)
    
        n_negatives = train_size + val_size + test_size

        print("[I]   all_negatives:", all_negatives)
        print("[I]   n_negatives  :", n_negatives)
        print("[I]   train_size   :", train_size)
        print("[I]   val_size     :", val_size)
        print("[I]   test_size    :", test_size)
        print("[I]   seed         :", self.seed)

        U_neg, V_neg = self.get_neg_indices(n_negatives, type)

        data_idx = self.rng.permutation(n_negatives)
        
        train_idx, val_idx, test_idx = self.get_indices(
            data_idx, train_size, test_size)

        self.load_neg_data(train_idx, val_idx, test_idx, U_neg, V_neg)


    @staticmethod
    def get_size(val_size, test_size, n_ratings, train_size=None):
        '''Get size of train, val and test set.

        Used in both ``RatioSplit`` and ``RatioSplit.negative_sample``.

        Parameters
        ----------
        val_size : int or float
            If it is int, ``val_size`` is the integer size of dataset.
            If it is float, ``val_size`` is the fraction size of dataset.
        test_size : int or float
            If it is int, ``test_size`` is the integer size of dataset.
            If it is float, ``test_size`` is the fraction size of dataset.
        n_ratings : int
            Number of ratings.
        train_size : int, float or None
            If ``None``, use the rest of data.
            If ``0.0``, empty training set. Used in negative sampling if there's no need to append negative samples to the training set.
        '''
        # validate val_size
        if val_size is None:
            val_size = 0.0
        elif val_size < 0 or val_size >= n_ratings:
            raise ValueError("Invalid val_size.")
        elif val_size < 1:
            val_size = ceil(val_size * n_ratings)

        # validate test_size
        if test_size is None:
            test_size = 0.0
        elif test_size < 0 or test_size >= n_ratings:
            raise ValueError("Invalid test_size.")
        elif test_size < 1:
            test_size = ceil(test_size * n_ratings)

        # validate train_val_size
        if train_size is None:
            if val_size + test_size > n_ratings:
                raise ValueError("Sum of val_size and test_size exceeds n_ratings.")
            train_size = n_ratings - (val_size + test_size)
        elif train_size < 0 or train_size >= n_ratings:
            raise ValueError("Invalid train_size.")
        elif train_size < 1:
            train_size = ceil(train_size * n_ratings)

        # final validation
        if train_size + val_size + test_size > n_ratings:
            raise ValueError("Sum of train_size, val_size and test_size exceeds n_ratings.")

        return int(train_size), int(val_size), int(test_size)
        

    @staticmethod
    def get_indices(data_idx, train_size, test_size):
        '''Get indices for train, val and test set.

        Used in ``RatioSplit`` and ``RatioSplit.negative_sampling``.

        Parameters
        ----------
        data_idx : ndarray
            The indices of dataset.
        train_size : int
            The size of training set.
        test_size : int
            The size of test set.
        '''
        train_idx = data_idx[:train_size]
        val_idx = data_idx[train_size:-test_size]
        test_idx = data_idx[-test_size:] if test_size > 0 else np.array([])

        return train_idx, val_idx, test_idx