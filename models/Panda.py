from .BaseModel import BaseModel
from tqdm import tqdm
import numpy as np
from utils import sum, invert, multiply, to_dense, matmul, bool_to_index
from utils import header, record, eval
from itertools import product
from scipy.sparse import lil_matrix


class Panda(BaseModel):
    """PaNDa and PaNDa+ algorithm.

    Reference
    ---------
    Mining Top-K Patterns from Binary Datasets in presence of Noise.
    A unifying framework for mining approximate top-k binary patterns.
    """
    def __init__(self, k, rho=None, w=None, method=None):
        """
        Parameters
        ----------
        k : int
            The rank.
        rho : float
            The regularization parameter.
        w : list
            The penalty weights for under-coverage and over-coverage.
            This is not included in original works of Panda or Panda+.
        method : str
            How items are sorted.
        """
        self.check_params(k=k, rho=rho, w=w, method=method)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "rho" in kwargs:
            rho = kwargs.get("rho")
            if rho is None:
                rho = 1.0
            self.rho = rho
            print("[I] rho          :", self.rho)
        if "w" in kwargs:
            w = kwargs.get("w")
            if w is None:
                w = [1.0, 1.0]
            self.w = w
            print("[I] weights      :", self.w)
        if "method" in kwargs:
            method = kwargs.get("method")
            if method is None:
                method = 'frequency'
            assert method in ['frequency', 'couples-frequency', 'correlation'], "Sorting method should be 'frequency', 'couples-frequency' or 'correlation'."
            self.method = method
            print("[I] method       :", self.method)


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val)
        self.init_model()
        
        self._fit()

        display(self.logs['updates'])
        self.show_matrix(colorbar=True, discrete=True, clim=[0, 1], title="result")


    def _fit(self):
        """
        X_uncovered : spmatrix
            X_train but covered cells are set to 0.
        X_covered : spmatrix
            X_train but covered cells are set to 1.
        E : list
            The item extension list.
        I : list
            The item list.
        T : list
            The user or transaction list.
        """
        self.X_uncovered = self.X_train.copy()
        self.X_covered = self.X_train.copy()

        self.I = lil_matrix(np.zeros((self.n, 1)))
        self.T = lil_matrix(np.zeros((self.m, 1)))
        cost_old = self.cost(self.I, self.T)

        for k in range(self.k): # tqdm(range(self.k), position=0):
            print(f"[I] k            : {k}")
            print(f"[I]   init cost  : {cost_old}")

            self.find_core()
            print(f"[I]   find core  : {self.cost_now}") # self.cost(self.I, self.T)

            self.extend_core()
            print(f"[I]   extension  : {self.cost_now}") # self.cost(self.I, self.T)

            if self.cost_now > cost_old:
                self.early_stop(msg="Cost stops improving", k=k)
                break
            else:
                # update U and V
                self.U[:, k] = self.T
                self.V[:, k] = self.I
                cost_old = self.cost_now

                # update residual and coverage
                pattern = matmul(self.T, self.I.T, sparse=True, boolean=True).astype(bool)
                self.X_uncovered[pattern] = 0
                self.X_covered[pattern] = 1

                self.evaluate(names=['cost'], values=[self.cost_now], df_name='updates')


    def find_core(self):
        """Find a dense core (T, I) and its extension list E.
        
        E : array
            The item extension list.
        I : spmatrix
            The item list.
        T : sparse matrix
            The user or transaction list.
        """
        self.E = list(range(self.n)) # start with all items
        self.I = lil_matrix(np.zeros((self.n, 1)))
        self.T = lil_matrix(np.zeros((self.m, 1)))

        # initialize extension list
        if self.method == 'frequency' or self.method == 'correlation':
            self.sort_items(method='frequency')
        elif self.method == 'couples-frequency':
            self.sort_items(method='couples-frequency')

        # add the first item from E to I
        self.I[self.E[0]] = 1
        self.T = self.X_uncovered[:, self.E[0]]
        self.E.pop(0)

        cost_old = self.cost(self.I, self.T)

        i = 0
        while i < len(self.E):
            I = self.I.copy()
            if self.method == 'correlation':
                self.sort_items(method='correlation') # re-order extension list
                I[self.E[0]] = 1 # use item with highest correlation
                T = multiply(self.T, self.X_uncovered[:, self.E[0]])
            else:
                I[self.E[i]] = 1
                T = multiply(self.T, self.X_uncovered[:, self.E[i]])

            # check cost by self.cost(): this is expensive
            # cost_new = self.cost(I, T)

            # check cost by cost difference: this is faster
            w0, h0 = self.I.sum(), self.T.sum()
            w1, h1 = self.I.sum() + 1, T.sum()
            d_cost = 1 + (h1 - h0) - self.rho * self.w[0] * ((w1 * h1) - (w0 * h0))

            if d_cost <= 0: # cost_new <= cost_old
                cost_new = cost_old + d_cost
                self.print_msg(f"[I]   add column : {cost_old} -> {cost_new} (d_cost: {d_cost})")
                cost_old = cost_new

                # update I, T and E
                self.I = I
                self.T = T
                if self.method == 'correlation':
                    self.E.pop(0)
                else:
                    self.E.pop(i)
            else:
                i += 1
        
        self.cost_now = cost_old


    def extend_core(self):
        cost_old = self.cost_now
        
        for i in range(len(self.E)):
            # update I
            I = self.I.copy()
            I[self.E[i]] = 1
            cost_new = self.cost(I, self.T)
            if cost_new <= cost_old:
                self.print_msg(f"[I]   add column : {cost_old} -> {cost_new} (col id: {i})")
                self.I = I
                cost_old = cost_new
            else:
                continue

            # faster T update check by computing cost difference
            idx_I = bool_to_index(self.I) # indices of items in itemset
            idx_T = bool_to_index(invert(self.T)) # indices of transactions outside T
            d_fn = -self.X_uncovered[idx_T][:, idx_I].sum(axis=1) # newly added false negatives
            d_fp = self.I.sum() - self.X_covered[idx_T][:, idx_I].sum(axis=1) # newly added false positives
            d_cost = 1 + self.rho * (self.w[0] * d_fn + self.w[1] * d_fp) # cost difference

            # update T
            idx = to_dense(d_cost, squeeze=True) <= 0
            idx_T = idx_T[idx] # row indices whose d_cost <= 0
            self.T[idx_T] = 1

            if idx.sum() > 0: # at least 1 transaction is added to T
                d_cost = d_cost[idx].sum()
                cost_new = cost_old + d_cost
                self.print_msg(f"[I]   add row    : {cost_old} -> {cost_new} (d_cost: {d_cost}, row(s) added: {idx.sum()})")
                cost_old = cost_new

        self.cost_now = cost_old # update current cost


    def sort_items(self, method):
        """Sort the extension list by descending scores.

        frequency :
            Sort items in extension list by frequency.
        couples-frequency :
            Sort items in extension list by frequency of item-pairs that include an item.
        correlation :
            Sort items in extension list by correlation with current transactions T.
        prefix-child :
            To-do.
        """
        if method == 'frequency':
            scores = sum(self.X_uncovered[:, self.E], axis=0)
        elif method == 'couples-frequency':
            scores = np.zeros(len(self.E))
            for i in range(len(self.E)):
                T = self.X_uncovered[:, self.E[i]] # T of i-th item in E
                idx_T = bool_to_index(T) # indices of transactions
                scores[i] = sum(self.X_uncovered[idx_T, :], axis=0).sum() # sum of 1 and 2-itemset frequency
            scores = scores - sum(self.X_uncovered[:, self.E], axis=0) # sum of 2-itemset frequency
        elif method == 'correlation':
            idx_T = bool_to_index(self.T)
            scores = sum(self.X_uncovered[idx_T][:, self.E], axis=0)
        elif method == 'prefix-child':
            pass

        idx = np.flip(np.argsort(scores)).astype(int)
        self.E = np.array(self.E)[idx].tolist()


    def cost(self, I, T):
        """MDL cost function.

        T : (m, 1) boolean spmatrix
        I : (n, 1) boolean spmatrix
        """
        idx_I = bool_to_index(I)
        idx_T = bool_to_index(T)
        width, height = I.sum(), T.sum()
        cost_width = self.U.sum() + width
        cost_height = self.V.sum() + height

        fn = self.X_uncovered.sum() - self.X_uncovered[idx_T][:, idx_I].sum()
        fp = width * height - self.X_covered[idx_T][:, idx_I].sum()

        cost_noise = self.w[0] * fn + self.w[1] * fp
        cost = (cost_width + cost_height) + self.rho * cost_noise
        return cost


    def evaluate(self, df_name, names=[], values=[]):
        self.predict()
        metrics = ['Recall', 'Precision', 'Accuracy', 'F1']
        
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