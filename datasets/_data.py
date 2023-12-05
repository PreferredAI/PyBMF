from utils import safe_indexing, to_sparse, Factor, Matrix
from scipy.sparse import issparse
import time
import numpy as np
from math import ceil
from tqdm import tqdm


class Data:
    '''A Data object loads raw matrix data from sourse

    One or more Data objects are then merged into a Dataset object.
    '''
    def __init__(self) -> None:
        '''
        Use Matrix.update and Factor.update to update the following:
        '''
        self.X = Matrix()
        self.U = Factor()
        self.V = Factor()
        '''
        Use Data.no_split, Data.split or Data.cross_validation to update the following:
        '''
        self.train_data = Matrix()
        self.test_data = Matrix()
        self.val_data = Matrix()


    def no_split(self):
        '''No split, used for reconstruction tasks

        For a reconstruction task, training and testing will use the full set of samples.
        '''
        self.train_data = self.X
        self.test_data = self.X
        self.val_data = self.X


    def split(self, test_size=None, val_size=None, seed=None):
        '''Ratio split, used for prediction tasks

        test_size, val_size:
            int, integer size of dataset.
            float, fraction size of dataset.
        '''
        print("[I] ratio split, sampling positives")     
        self.check_params(seed=seed)

        train_size, val_size, test_size = self.get_size(
            val_size=val_size, test_size=test_size, n_ratings=self.X.r)

        print("[I]   train_size   :", train_size)
        print("[I]   val_size     :", val_size)
        print("[I]   test_size    :", test_size)
        print("[I]   seed         :", self.seed)

        data_idx = self.rng.permutation(self.X.r)

        train_idx, val_idx, test_idx = self.get_indices(
            data_idx, train_size, test_size)

        self.load_pos_data(train_idx, val_idx, test_idx)


    def negative_sample(self, train_size=None, test_size=None, val_size=None, seed=None, type='uniform'):
        '''Select and append negative samples onto train, val and test set

        Used with ratio split.

        type: how negative item will be sampled, should be 'uniform' or 'popularity'.
        '''
        print("[I] ratio split, sampling negatives")        
        self.check_params(seed=seed)
        m = self.X.m
        n = self.X.n
        all_negatives = m * n - self.X.r

        train_size, val_size, test_size = self.get_size(
            train_size=train_size, val_size=val_size, test_size=test_size, n_ratings=all_negatives)
        
        n_negatives = train_size + val_size + test_size

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


    def cross_validation(self, test_size=None, n_folds=None, current_fold=None, seed=None):
        '''K-fold cross-validation, used for prediction tasks

        test_size:
            int, integer size of dataset.
            float, fraction size of dataset.
        n_folds, current_fold:
            int, number of folds or index of the current fold.
        '''
        print("[I] cross-validation, sampling positives")
        if hasattr(self, 'cv_initialized'):
            print("[I]   Re-using original config, seed will be ignored.")
        else:
            # first call
            self.cv_initialized = True

            self.check_params(seed=seed)

            self.cv_pos_partition = self.cv_get_partition(
                n_folds=n_folds, test_size=test_size, n_ratings=self.X.r)
            
            self.cv_pos_data_idx = self.rng.permutation(self.X.r)

            print("[I]   partition    :", self.cv_pos_partition)
            print("[I]   test_size    :", test_size)
            print("[I]   seed         :", self.seed)

        train_idx, val_idx, test_idx = self.cv_get_indices(
            data_idx=self.cv_pos_data_idx, partition=self.cv_pos_partition, current_fold=current_fold)

        self.load_pos_data(train_idx, val_idx, test_idx)


    def cv_negative_sample(self, test_size=None, n_folds=None, current_fold=None, train_val_size=None, seed=None, type='uniform'):
        print("[I] cross-validation, sampling negatives")
        if hasattr(self, 'cv_ns_initialized'):
            print("[I]   Re-using original config, seed will be ignored.")
        else:
            # first call
            self.cv_ns_initialized = True

            self.check_params(seed=seed)

            m = self.X.m
            n = self.X.n
            all_negatives = m * n - self.X.r

            if train_val_size is None:
                n_negatives = all_negatives
            else:
                n_negatives = train_val_size + test_size
                if n_negatives > all_negatives:
                    print("[W] No enough negatives.")
                    n_negatives = all_negatives

            self.cv_neg_partition = self.cv_get_partition(
                n_folds=n_folds, train_val_size=train_val_size, test_size=test_size, n_ratings=all_negatives)
            
            self.U_neg, self.V_neg = self.get_neg_indices(n_negatives, type)

            self.cv_neg_data_idx = self.rng.permutation(n_negatives)

            print("[I]   n_negatives  :", n_negatives)
            print("[I]   test_size    :", test_size)
            print("[I]   seed         :", self.seed)

        train_idx, val_idx, test_idx = self.cv_get_indices(
            data_idx=self.cv_neg_data_idx, partition=self.cv_neg_partition, current_fold=current_fold)

        self.load_neg_data(train_idx, val_idx, test_idx, self.U_neg, self.V_neg)
            

    def _get_neg_indices(self, n_negatives, type):
        m = self.X.m
        n = self.X.n
        if type == "uniform":
            U_indices = np.arange(m)
            V_indices = np.arange(n)
        elif type == "popularity":
            U_indices = self.X.triplet[0]
            V_indices = self.X.triplet[1]
        else:
            raise ValueError("Unsupported negative sampling option: {}".format(type))

        U_neg = []
        V_neg = []
        for _ in tqdm(range(n_negatives)):
            # randomly choose a row and column
            u = self.rng.choice(U_indices)
            v = self.rng.choice(V_indices)
            # check if the randomly chosen entry is a positive sample
            while self.X.matrix[u, v] == 1 or (u, v) in zip(U_neg, V_neg):
                u = self.rng.choice(U_indices)
                v = self.rng.choice(V_indices)
            # add the negative sample to the list
            U_neg.append(u)
            V_neg.append(v)
        return U_neg, V_neg
    

    def get_neg_indices(self, n_negatives, type):
        m = self.X.m
        n = self.X.n
        if type == "uniform":
            indices = self.rng.choice(a=m*n, size=n_negatives, replace=False)
        elif type == "popularity":
            p = np.zeros((m, n))
            pu, pv = self.X.sum
            s = self.X.matrix.sum()
            pu = pu / s
            pv = pv / s
            for r in range(m):
                p[r] = pu[r] * pv
            p[self.X == 1] = 0
            p = p.flatten()
            indices = self.rng.choice(a=m*n, size=n_negatives, replace=False, p=p)
        else:
            raise ValueError("Unsupported negative sampling option: {}".format(type))
        
        U_neg = (indices / n).astype(int)
        V_neg = (indices % n).astype(int)
        return U_neg, V_neg


    @staticmethod
    def get_size(val_size, test_size, n_ratings, train_size=None):
        """
        Used in ratio split and negative_sample.

        train_size:
            None, use the rest of data.
            0.0, empty training set. used in negative sampling if there's no need to append negative samples to the training set.
        """
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
        '''
        Used in ratio split and negative sampling.
        '''
        train_idx = data_idx[:train_size]
        test_idx = data_idx[-test_size:]
        val_idx = data_idx[train_size:-test_size]

        return train_idx, val_idx, test_idx
    

    @staticmethod
    def cv_get_partition(n_folds, test_size, n_ratings, train_val_size=None):
        '''
        Used in cross_validation and cv_negative_sample.

        train_val_size:
            None, use the rest of data.
            0.0, not valid.

        Return
        ------
        partition:
            An array of starting indices of each fold and the test set.
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

        return partition
    

    @staticmethod
    def cv_get_indices(data_idx, partition, current_fold):
        print("[I] getting cross-validation indices")
        a = partition[current_fold - 1] # start of val
        b = partition[current_fold] # end of val
        c = partition[-1] # start of test

        if current_fold < 1 or current_fold > len(data_idx) - 1:
            print("[E]   current_fold should lie in [1, n_fold]")
        else:
            print("[I]   current fold         :", current_fold)
            print("[I]   current train size   :", c - b + a)
            print("[I]   current val size     :", b - a)

        train_idx = np.concatenate((data_idx[:a], data_idx[b:c]))
        test_idx = data_idx[c:]
        val_idx = data_idx[a:b]

        return train_idx, val_idx, test_idx
    

    def load_pos_data(self, train_idx, val_idx, test_idx):
        self.train_data.matrix = safe_indexing(self.X.matrix, train_idx)
        self.test_data.matrix = safe_indexing(self.X.matrix, test_idx) if len(test_idx) > 0 else None
        self.val_data.matrix = safe_indexing(self.X.matrix, val_idx) if len(val_idx) > 0 else None
        
        self.pos_train_size = len(train_idx)
        self.pos_test_size = len(test_idx)
        self.pos_val_size = len(val_idx)

        if self.X.name is not None:
            self.train_data.name = self.X.name
            self.test_data.name = self.X.name
            self.val_data.name = self.X.name


    def load_neg_data(self, train_idx, val_idx, test_idx, U_neg, V_neg):
        print(train_idx)
        print(val_idx)
        print(test_idx)
        # coo type does not support item assignment, force to use csr
        self.train_data.matrix = to_sparse(self.train_data.matrix, type='csr')
        self.test_data.matrix = to_sparse(self.test_data.matrix, type='csr')
        self.val_data.matrix = to_sparse(self.val_data.matrix, type='csr')

        # remove zeros in sparse matrices
        self.train_data.matrix.eliminate_zeros()
        self.test_data.matrix.eliminate_zeros()
        self.val_data.matrix.eliminate_zeros()

        # faster assignment
        self.train_data.matrix = to_sparse(self.train_data.matrix, type='lil')
        self.test_data.matrix = to_sparse(self.test_data.matrix, type='lil')
        self.val_data.matrix = to_sparse(self.val_data.matrix, type='lil')
        
        for idx in train_idx:
            self.train_data.matrix[U_neg[idx], V_neg[idx]] = 0
        for idx in test_idx:
            self.test_data.matrix[U_neg[idx], V_neg[idx]] = 0
        for idx in val_idx:
            self.val_data.matrix[U_neg[idx], V_neg[idx]] = 0

        # back to csr
        self.train_data.matrix = to_sparse(self.train_data.matrix, type='csr')
        self.test_data.matrix = to_sparse(self.test_data.matrix, type='csr')
        self.val_data.matrix = to_sparse(self.val_data.matrix, type='csr')

        self.neg_train_size = len(train_idx)
        self.neg_test_size = len(test_idx)
        self.neg_val_size = len(val_idx)


    def check_params(self, **kwargs):
        # check seed
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if seed is None and not hasattr(self,'seed'): # use time as self.seed
                seed = int(time.time())
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I]   Data seed    :", self.seed)
            elif seed is not None: # overwrite self.seed
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I]   Data seed    :", self.seed)
            else: # self.rng remains unchanged
                pass


    def sample(self, axis, idx=None, n=None, seed=None):
        if idx is not None:
            print("[I] sampling with given indices")
            if axis == 0:
                self._sample_U(idx)
            elif axis == 1:
                self._sample_V(idx)
        elif n is not None:
            print("[I] sampling to size", n)
            assert self.X.m >= n, "[E] Target length is greater than the original."
            self.check_params(seed=seed)
            if axis == 0:
                idx = [True] * n + [False] * (self.X.m - n)
                self.rng.shuffle(idx)
                self._sample_U(idx)
            elif axis == 1:
                idx = [True] * n + [False] * (self.X.n - n)
                self.rng.shuffle(idx)
                self._sample_V(idx)
            return idx


    def _sample_U(self, idx):
        self.X.matrix = self.X.matrix[idx, :]
        self.X.reset()
        order, idmap, alias = self.U.order[idx], self.U.idmap[idx], self.U.alias[idx]
        self.U.update(order=order, idmap=idmap, alias=alias, ignore=True)
        self.U.sort_order()


    def _sample_V(self, idx):
        self.X.matrix = self.X.matrix[:, idx]
        self.X.reset()
        order, idmap, alias = self.V.order[idx], self.V.idmap[idx], self.V.alias[idx]
        self.V.update(order=order, idmap=idmap, alias=alias, ignore=True)
        self.V.sort_order()


    