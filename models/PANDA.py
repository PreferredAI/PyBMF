from .BaseModel import BaseModel
from tqdm import tqdm
import numpy as np
from utils import sum, invert, multiply, to_dense, matmul, bool_to_index, show_matrix
from scipy.sparse import csr_matrix, lil_matrix


class Panda(BaseModel):
    """PaNDa and PaNDa+ algorithms

    Reference:
        Mining Top-K Patterns from Binary Datasets in presence of Noise
        A unifying framework for mining approximate top-k binary patterns
    """
    def __init__(self, k, rho=None, w=None):
        """
        k:
            rank.
        rho:
            regularization parameter.
        w: 
            penalty weights for under-coverage and over-coverage.
            this is not included in Panda or Panda+.
        """
        self.check_params(k=k, rho=rho, w=w)


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


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_dataset(X_train=X_train, X_val=X_val)
        self.check_params(**kwargs)
        self._fit()


    def _fit(self):
        """
        X_res:
            X_train but covered cells are set to 0
        X_cover:
            X_train but covered cells are set to 1
        """
        self.X_res = self.X_train.copy()
        self.X_cover = self.X_train.copy()

        self.I = lil_matrix(np.zeros((self.n, 1)))
        self.T = lil_matrix(np.zeros((self.m, 1)))
        cost_before = self.cost(self.I, self.T)

        for k in range(self.k): # tqdm(range(self.k), position=0):
            
            self.print_msg(f"{k}\tstart\t\t: {cost_before}")
            self.find_core()
            self.print_msg(f"{k}\tfind_core\t: {self.cost(self.I, self.T)}  ( {self.current_cost} )")
            self.extend_core()
            self.print_msg(f"{k}\textension\t: {self.cost(self.I, self.T)}  ( {self.current_cost} )")

            if self.current_cost > cost_before:
                self.early_stop(msg="Cost stops improving", k=k) 
                break
            else:
                self.U[:, k] = self.T
                self.V[:, k] = self.I
                cost_before = self.current_cost

                pattern = matmul(self.T, self.I.T, sparse=True, boolean=True).astype(bool)
                self.X_res[pattern] = 0
                self.X_cover[pattern] = 1

                print("pattern", pattern.sum())

                settings = [(to_dense(self.X_res), [0, 0], 'X_res'), 
                            (to_dense(self.X_cover), [0, 1], 'X_cover'), 
                            (to_dense(pattern), [0, 2], 'pattern'), 
                            (matmul(self.U, self.V.T, sparse=False, boolean=True), [0, 3], 'X')]
                show_matrix(settings=settings, title=str(k), scaling=0.3)


    def find_core(self):
        """Find a dense core (T, I) and its extension list E
        
        E: np array, item extension list
        I: np array, item list
        T: sparse matrix, user or transaction list
        """
        self.E = list(range(self.n))
        self.I = lil_matrix(np.zeros((self.n, 1)))
        self.T = lil_matrix(np.zeros((self.m, 1)))
        self.sort_items(method='frequency')

        # add the first item from E to I
        self.I[self.E[0]] = 1
        self.T = self.X_res[:, self.E[0]]
        self.E.pop(0)

        cost_before = self.cost(self.I, self.T)

        i = 0
        while i < len(self.E):
            I = self.I.copy()
            I[self.E[i]] = 1
            T = multiply(self.T, self.X_res[:, self.E[i]])

            cost_after = self.cost(I, T)

            # fast item set update
            w0, h0 = self.I.sum(), self.T.sum()
            w1, h1 = self.I.sum() + 1, T.sum()
            delta = 1 + (h1 - h0) - self.rho * self.w[0] * ((w1 * h1) - (w0 * h0))

            if cost_after <= cost_before:
                self.print_msg(f"\tadd col\t\t: {cost_before} -> {cost_after}, delta: {delta}")
                self.I = I
                self.T = T
                self.E.pop(i)
                cost_before = cost_after
            else:
                i += 1

        self.current_cost = cost_before


    def extend_core(self):
        cost_before = self.current_cost
        for i in range(len(self.E)):
            I = self.I.copy()
            I[self.E[i]] = 1
            cost_after = self.cost(I, self.T)

            if cost_after <= cost_before:
                self.print_msg(f"\tadd col c={i}\t: {cost_before} -> {cost_after}")
                self.I = I
                cost_before = cost_after
            else:
                continue

            # fast transaction update
            idx_I = bool_to_index(self.I)
            idx_T = bool_to_index(invert(self.T))
            d_fn = -self.X_res[idx_T][:, idx_I].sum(axis=1)
            d_fp = self.I.sum() - self.X_cover[idx_T][:, idx_I].sum(axis=1)
            d_cost = 1 + self.rho * (self.w[0] * d_fn + self.w[1] * d_fp)
            idx = to_dense(d_cost, squeeze=True) <= 0
            idx_T = idx_T[idx]
            self.T[idx_T] = 1
            if idx.sum() > 0:
                cost_after = self.cost(self.I, self.T)
                cost_after_incremental = cost_before + d_cost[idx].sum()
                self.print_msg(f"\tadd row\t\t: {cost_before} -> {cost_after}  ( {cost_after_incremental} ) {idx.sum()} row(s) added")
                cost_before = cost_after

        self.current_cost = cost_before


    def sort_items(self, method='frequency'):
        """Sort the extension list by descending scores
        """
        if method == 'frequency':
            scores = sum(self.X_res[:, self.E], axis=0)
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
        cost = 0 * (cost_width + cost_height) + self.rho * cost_noise
        return cost
