from .BaseSplit import BaseSplit
from math import ceil
import numpy as np


class CrossValidation(BaseSplit):
    '''K-fold cross-validation, used in prediction tasks

    Parameters
    ----------
    X : ndarray, spmatrix
        The data matrix.
    test_size : int or float
        If it is int, ``test_size`` is the integer size of dataset. 
        If it is float, ``test_size`` is the fraction size of dataset.
    n_folds : int
        Number of folds.
    current_fold : int
        Index of the current fold.
    '''
    def __init__(self, X, test_size=None, n_folds=None, seed=None):
        super().__init__(X)
        print("[I] CrossValidation, sampling positives")
        
        self.check_params(seed=seed)

        self.cv_pos_partition, test_size = self.get_partition(
            n_folds=n_folds, test_size=test_size, n_ratings=self.X.nnz)
        
        self.cv_pos_data_idx = self.rng.permutation(self.X.nnz)
        self.n_folds = n_folds
        self.pos_train_val_size = self.X.nnz - test_size
        self.pos_test_size = test_size
        self.ns_initialized = False

        print("[I]   n_folds      :", self.n_folds)
        print("[I]   partition    :", self.cv_pos_partition)
        print("[I]   train + val  :", self.pos_train_val_size)
        print("[I]   test_size    :", self.pos_test_size)



    def get_fold(self, current_fold):
        '''Get current fold.

        Parameters
        ----------
        current_fold : int
            Index of the current fold.
        '''
        print("[I] CrossValidation, current fold :", current_fold)
        train_idx, val_idx, test_idx = self.get_indices(
            data_idx=self.cv_pos_data_idx, 
            partition=self.cv_pos_partition, 
            current_fold=current_fold)
        fold_size =  (len(train_idx), len(val_idx), len(test_idx))
        print("[I]   fold size            :", fold_size)
        self.load_pos_data(train_idx, val_idx, test_idx)

        if not self.ns_initialized:
            print("[W]   No negative sampling config.")
            return

        train_idx, val_idx, test_idx = self.get_indices(
            data_idx=self.cv_neg_data_idx, 
            partition=self.cv_neg_partition, current_fold=current_fold)
        fold_size =  (len(train_idx), len(val_idx), len(test_idx))
        print("[I]   fold neg sample size :", fold_size)
        self.load_neg_data(train_idx, val_idx, test_idx, self.U_neg, self.V_neg)


    def negative_sample(self, test_size, train_val_size, seed=None, type='uniform'):
        '''Negative sampling for cross-validation.

        Parameters
        ----------
        test_size : int
            Number of test samples.
        train_val_size : int
            Number of train and validation samples.
        seed : int
            Random seed.
        type : str
            Type of negative sampling.
        '''
        print("[I] CrossValidation, sampling negatives")

        self.check_params(seed=seed)

        m, n = self.X.shape
        all_negatives = m * n - self.X.nnz

        # TODO: deal with fractional test_size and train_val_size
        n_negatives = train_val_size + test_size
        assert n_negatives <= all_negatives, "No enough negatives."

        self.cv_neg_partition, test_size = self.get_partition(
            n_folds=self.n_folds, 
            train_val_size=train_val_size,
            test_size=test_size, 
            n_ratings=all_negatives)
        
        self.U_neg, self.V_neg = self.get_neg_indices(n_negatives, type)

        self.cv_neg_data_idx = self.rng.permutation(n_negatives)
        self.neg_train_val_size = train_val_size
        self.neg_test_size = test_size
        self.ns_initialized = True

        print("[I]   n_folds      :", self.n_folds)
        print("[I]   partition    :", self.cv_neg_partition)
        print("[I]   train + val  :", self.neg_train_val_size)
        print("[I]   test_size    :", self.neg_test_size)


    @staticmethod
    def get_partition(n_folds, test_size, n_ratings, train_val_size=None):
        '''Get partition for cross-validation.

        Used in ``CrossValidation`` and ``CrossValidation.cv_negative_sample``.

        Parameters
        ----------
        n_folds : int
            Number of folds.
        test_size : int or float
            If it is int, ``test_size`` is the integer size of dataset. 
            If it is float, ``test_size`` is the fraction size of dataset.
        train_val_size : int, float or None
            If it is ``None``, use the remaining data outside ``test_size``.
            If it is int, ``train_val_size`` is the integer size of dataset. 
            If it is float, ``train_val_size`` is the fraction size of dataset.
            Note that ``0.0`` is not valid.

        Return
        ------
        partition : ndarray
            An array of starting indices of each fold and the test set.
        test_size : int
            The size of test set.
        '''
        # validate test_size
        if test_size is None:
            test_size = 0.0
        elif test_size < 0 or test_size >= n_ratings:
            raise ValueError("Invalid test_size.")
        elif test_size < 1:
            test_size = ceil(test_size * n_ratings)
        # validate train_val_size
        if train_val_size is None:
            train_val_size = n_ratings - test_size
        elif train_val_size <= 0 or train_val_size >= n_ratings:
            raise ValueError("Invalid train_val_size.")
        elif train_val_size < 1:
            train_val_size = ceil(train_val_size * n_ratings)
        # final validation
        if train_val_size + test_size > n_ratings:
            raise ValueError("Sum of train_size, val_size and test_size exceeds n_ratings.")
            
        fold_size = int(train_val_size / n_folds)
        remain_size = train_val_size - fold_size * n_folds

        partition = [0] * (n_folds + 1)
        for i in range(n_folds):
            partition[i+1] = partition[i] + fold_size + (i < remain_size)

        return partition, test_size


    @staticmethod
    def get_indices(data_idx, partition, current_fold):
        '''Get indices for current fold.

        Parameters
        ----------
        data_idx : ndarray
            The indices of dataset.
        partition : ndarray
            An array of starting indices of each fold and the test set.
        current_fold : int
            The index of current fold.

        Return
        ------
        train_idx : ndarray
            The indices of training data.
        val_idx : ndarray
            The indices of validation data.
        test_idx : ndarray
            The indices of test data.
        '''
        print("[I] CrossValidation, get indices for current fold")
        a = partition[current_fold] # start of val
        b = partition[current_fold+1] # end of val
        c = partition[-1] # start of test

        if current_fold >= 0 and current_fold < len(data_idx):
            print("[I]   current fold         :", current_fold)
            print("[I]   current train size   :", c - b + a)
            print("[I]   current val size     :", b - a)
        else:
            print("[E]   current_fold should lie in [1, n_fold]")

        train_idx = np.concatenate((data_idx[:a], data_idx[b:c]))
        test_idx = data_idx[c:]
        val_idx = data_idx[a:b]

        return train_idx, val_idx, test_idx