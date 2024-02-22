from .BaseModel import BaseModel
from tqdm import tqdm
import numpy as np
from utils import sum, invert, multiply, to_dense, matmul, bool_to_index, show_matrix
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
        k:
            rank.
        rho:
            regularization parameter.
        w: 
            penalty weights for under-coverage and over-coverage.
            this is not included in Panda or Panda+.
        method:
            how items are sorted.
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
        self.load_dataset(X_train=X_train, X_val=X_val)
        self.check_params(**kwargs)
        self._fit()


    def _fit(self):
        """
        X_res:
            X_train but covered cells are set to 0
        X_cover:
            X_train but covered cells are set to 1
        E: 
            item extension list
        I: 
            item list
        T: 
            user or transaction list
        """
        self.X_res = self.X_train.copy()
        self.X_cover = self.X_train.copy()

        self.I = lil_matrix(np.zeros((self.n, 1)))
        self.T = lil_matrix(np.zeros((self.m, 1)))
        cost_before = self.cost(self.I, self.T)

        for k in range(self.k): # tqdm(range(self.k), position=0):

            print(f"[I] {k}\tinit cost\t: {cost_before}")

            self.find_core()

            print(f"[I] {k}\tfind core\t: {self.cost(self.I, self.T)} ({self.current_cost})")

            self.extend_core()

            print(f"[I] {k}\textension\t: {self.cost(self.I, self.T)} ({self.current_cost})")

            if self.current_cost > cost_before:
                self.early_stop(msg="Cost stops improving", k=k)
                break
            else:
                # update U and V
                self.U[:, k] = self.T
                self.V[:, k] = self.I
                cost_before = self.current_cost

                # update residual and coverage
                pattern = matmul(self.T, self.I.T, sparse=True, boolean=True).astype(bool)
                self.X_res[pattern] = 0
                self.X_cover[pattern] = 1

                # display
                print(f"[I] \tpattern shape\t: {(self.T.sum(), self.I.sum())}")
                if self.display:
                    settings = [
                        (to_dense(self.X_res), [0, 0], 'residual'), 
                        (to_dense(self.X_cover), [0, 1], 'coverage'), 
                        (to_dense(pattern), [0, 2], 'new pattern'), 
                        (matmul(self.U, self.V.T, sparse=False, boolean=True), [0, 3], 'X')]
                    show_matrix(settings=settings, title=str(k), scaling=self.scaling)


    def find_core(self):
        """Find a dense core (T, I) and its extension list E
        
        E: np array, item extension list
        I: sparse matrix, item list
        T: sparse matrix, user or transaction list
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
        self.T = self.X_res[:, self.E[0]]
        self.E.pop(0)

        cost_before = self.cost(self.I, self.T)

        i = 0
        while i < len(self.E):
            I = self.I.copy()
            if self.method == 'correlation':
                self.sort_items(method='correlation') # re-order extension list
                I[self.E[0]] = 1 # use item with highest correlation
                T = multiply(self.T, self.X_res[:, self.E[0]])
            else:
                I[self.E[i]] = 1
                T = multiply(self.T, self.X_res[:, self.E[i]])

            # check cost: it's more expensive
            # cost_after = self.cost(I, T)

            # check cost difference: it's faster
            w0, h0 = self.I.sum(), self.T.sum()
            w1, h1 = self.I.sum() + 1, T.sum()
            d_cost = 1 + (h1 - h0) - self.rho * self.w[0] * ((w1 * h1) - (w0 * h0))

            if d_cost <= 0: # cost_after <= cost_before
                cost_after = cost_before + d_cost
                self.print_msg(f"\tadd col\t\t: {cost_before} -> {cost_after} ({d_cost})")
                cost_before = cost_after

                # update I, T and E
                self.I = I
                self.T = T
                if self.method == 'correlation':
                    self.E.pop(0)
                else:
                    self.E.pop(i)
            else:
                i += 1

        self.current_cost = cost_before


    def extend_core(self):
        cost_before = self.current_cost

        for i in range(len(self.E)):
            # update I
            I = self.I.copy()
            I[self.E[i]] = 1
            cost_after = self.cost(I, self.T)
            if cost_after <= cost_before:
                self.print_msg(f"\tadd col c={i}\t: {cost_before} -> {cost_after}")
                self.I = I
                cost_before = cost_after
            else:
                continue

            # faster T update check by computing cost difference
            idx_I = bool_to_index(self.I) # indices of items in itemset
            idx_T = bool_to_index(invert(self.T)) # indices of transactions outside T
            d_fn = -self.X_res[idx_T][:, idx_I].sum(axis=1) # newly added false negatives
            d_fp = self.I.sum() - self.X_cover[idx_T][:, idx_I].sum(axis=1) # newly added false positives
            d_cost = 1 + self.rho * (self.w[0] * d_fn + self.w[1] * d_fp) # cost difference

            # update T
            idx = to_dense(d_cost, squeeze=True) <= 0
            idx_T = idx_T[idx] # row indices whose d_cost <= 0
            self.T[idx_T] = 1

            if idx.sum() > 0: # at least 1 transaction is added to T
                d_cost = d_cost[idx].sum()
                cost_after = cost_before + d_cost
                self.print_msg(f"\tadd row\t\t: {cost_before} -> {cost_after} ({d_cost}) {idx.sum()} row(s) added")
                cost_before = cost_after
                
        self.current_cost = cost_before # update current cost


    def sort_items(self, method):
        """Sort the extension list by descending scores

        frequency:
            sort items in extension list by frequency.
        couples-frequency:
            sort items in extension list by frequency of item-pairs that include an item.
        correlation:
            sort items in extension list by correlation with current transactions T.
        prefix-child:
            tbd.
        """
        if method == 'frequency':
            scores = sum(self.X_res[:, self.E], axis=0)
        elif method == 'couples-frequency':
            scores = np.zeros(len(self.E))
            for i in range(len(self.E)):
                T = self.X_res[:, self.E[i]] # T of i-th item in E
                idx_T = bool_to_index(T) # indices of transactions
                scores[i] = sum(self.X_res[idx_T, :], axis=0).sum() # sum of 1 and 2-itemset frequency
            scores = scores - sum(self.X_res[:, self.E], axis=0) # sum of 2-itemset frequency
        elif method == 'correlation':
            idx_T = bool_to_index(self.T)
            scores = sum(self.X_res[idx_T][:, self.E], axis=0)
        elif method == 'prefix-child':
            pass

        idx = np.flip(np.argsort(scores)).astype(int)
        self.E = np.array(self.E)[idx].tolist()


    def cost(self, I, T):
        """MDL cost function

        T: m * 1 boolean sparse matrix
        I: n * 1 boolean sparse matrix
        """
        idx_I = bool_to_index(I)
        idx_T = bool_to_index(T)
        width, height = I.sum(), T.sum()
        cost_width = self.U.sum() + width
        cost_height = self.V.sum() + height

        fn = self.X_res.sum() - self.X_res[idx_T][:, idx_I].sum()
        fp = width * height - self.X_cover[idx_T][:, idx_I].sum()

        cost_noise = self.w[0] * fn + self.w[1] * fp
        cost = (cost_width + cost_height) + self.rho * cost_noise
        return cost
