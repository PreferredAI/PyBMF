import numpy as np
from utils import matmul, multiply, sum, bool_to_index, multiply, ERR
from .BaseModel import BaseModel
from scipy.sparse import issparse, lil_matrix, csr_matrix


class MEBF(BaseModel):
    '''Median Expansion for Boolean Factorization
    
    From the paper 'Fast And Efficient Boolean Matrix Factorization By Geometric Segmentation'.
    '''
    def __init__(self, k=None, t=None, w=None) -> None:
        """
        k:
            suggested rank.
        t:
            threshold.
        w:
            reward and penalty parameters.
            w[0]: reward the covered true-positives.
            w[1]: penalize the over-covered false-positives.
            this is not included in the original paper, which is equivalent to the default w = [1, 1].
        """
        self.check_params(k=k, t=t, w=w)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if 't' in kwargs:
            t = kwargs.get("t")
            if t is None:
                t = 0.5
            self.t = t
            print("[I] t            :", self.t)
        if 'w' in kwargs:
            w = kwargs.get("w")
            if w is None:
                w = [1, 1]
            self.w = w
            print("[I] w            :", self.w)


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.k = min(X_train.shape) if self.k is None else self.k
        self.load_dataset(X_train=X_train, X_val=X_val)
        self._fit()


    def _fit(self):
        self.X_res = self.X_train.copy()
        self.X_cover = self.X_train.copy()
        self.cost = self.X_res.sum()

        self.u = lil_matrix(np.zeros((self.m, 1)))
        self.v = lil_matrix(np.zeros((self.n, 1)))

        k = 0

        while True:
            self.bidirectional_growth()
            if self.d_cost > 0: # cost increases
                self.print_msg("[I] k: {}, cost increases by {}".format(k, self.d_cost))
                self.weak_signal_detection() # fall back to small pattern
                if self.d_cost > 0: # cost still increases
                    self.early_stop(msg="Cost stops decreasing", k=k)
                    break
            self.U[:, k] = self.u
            self.V[:, k] = self.v
            self.cost = self.cost + self.d_cost
            self.print_msg("[I] k: {}, pattern: {}, d_cost: {}, cost: {}".format(
                k, (self.u.sum(), self.v.sum()), self.d_cost, self.cost))

            # debug: verify cost (slow)
            # X = matmul(self.U, self.V.T, sparse=True, boolean=True)
            # err = ERR(self.X_train, X) * self.m * self.n
            # err = np.round(err)
            # print("[I] k: {}, pattern: {}, d_cost: {}, cost: {}  ( {} )".format(k, (self.u.sum(), self.v.sum()), self.d_cost, self.cost, err))

            pattern = matmul(self.u, self.v.T, sparse=True, boolean=True).astype(bool)
            self.X_res[pattern] = 0
            self.X_cover[pattern] = 1

            k += 1
            
            if k == self.k:
                self.early_stop(msg="Reached requested rank", k=k)
                break


    def bidirectional_growth(self):
        """Bi-directional growth algorithm
        """
        u_0, v_0 = self.get_factor(axis=0)
        d_cost_0 = self.get_factor_cost(u_0, v_0)
        
        u_1, v_1 = self.get_factor(axis=1)
        d_cost_1 = self.get_factor_cost(u_1, v_1)

        if d_cost_0 <= d_cost_1:
            self.u, self.v, self.d_cost = u_0, v_0, d_cost_0
        else:
            self.u, self.v, self.d_cost = u_1, v_1, d_cost_1


    def weak_signal_detection(self):
        """Weak signal detection algorithm
        """
        u_0, v_0 = self.get_weak_signal(axis=0)
        d_cost_0 = self.get_factor_cost(u_0, v_0)
        
        u_1, v_1 = self.get_weak_signal(axis=1)
        d_cost_1 = self.get_factor_cost(u_1, v_1)

        if d_cost_0 <= d_cost_1:
            self.u, self.v, self.d_cost = u_0, v_0, d_cost_0
        else:
            self.u, self.v, self.d_cost = u_1, v_1, d_cost_1


    def get_factor_cost(self, u, v):
        """Get the difference of cost d_cost given the new pattern
        """
        pattern = matmul(u, v.T, sparse=True, boolean=True).astype(bool)
        tp = self.X_res[pattern].sum() # covered
        fp = pattern.sum() - self.X_cover[pattern].sum() # over-covered
        d_cost = self.w[1] * fp - self.w[0] * tp

        return d_cost
    
    
    def get_factor(self, axis):
        """Get factor for bi-directional growth

        axis:
            0, sort cols, find middle u and grow on v
            1, sort rows, find middle v and grow on u
        a, b: np.matrix
        """
        scores = sum(X=self.X_res, axis=axis)
        idx = np.flip(np.argsort(scores)).astype(int)
        idx = idx[scores > 0]
        mid = idx[int(len(idx) / 2)]
        a = self.X_res[:, mid] if axis == 0 else self.X_res[mid, :]
        idx = bool_to_index(a)
        X_sub = self.X_res[idx, :] if axis == 0 else self.X_res[:, idx]
        b = sum(X=X_sub, axis=axis) > self.t * a.sum()
        b = csr_matrix(b)
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)

        return (a, b) if axis == 0 else (b, a)
    

    def get_weak_signal(self, axis):
        """Get factor for weak signal detection

        axis:
            0, find u and grow on v
            1, find v and grow on u
        a, b: np.matrix
        """
        scores = sum(X=self.X_res, axis=axis)
        idx = np.flip(np.argsort(scores)).astype(int)
        first, second = idx[0], idx[1]
        if axis == 0:
            a = multiply(self.X_res[:, first], self.X_res[:, second], boolean=True)
        else:
            a = multiply(self.X_res[first, :], self.X_res[second, :], boolean=True)
        idx = bool_to_index(a)
        X_sub = self.X_res[idx, :] if axis == 0 else self.X_res[:, idx]
        b = sum(X=X_sub, axis=axis) > self.t * a.sum()
        b = csr_matrix(b)
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)

        return (a, b) if axis == 0 else (b, a)

