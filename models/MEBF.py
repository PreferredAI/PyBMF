import numpy as np
from utils import matmul, multiply, error
from .BaseModel import BaseModel
from scipy.sparse import issparse, lil_matrix


class MEBF(BaseModel):
    '''Median Expansion for Boolean Factorization
    
    From the paper 'Fast And Efficient Boolean Matrix Factorization By Geometric Segmentation'.
    '''
    def __init__(self, tau, k=None, p=None) -> None:
        self.check_params(k=k, tau=tau, p=p)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        # check threshold tau
        self.tau = kwargs.get("tau")
        if self.tau == None:
            self.tau = 0.5
        print("[I] tau          :", self.tau)
        # check percentage p
        self.p = kwargs.get("p")
        if self.p == None:
            self.p = 0.8
        print("[I] p            :", self.p)


    def fit(self, train_set, val_set=None, display_flag=False):
        super().check_params(display_flag=display_flag)
        super().check_dataset(train_set=train_set, val_set=val_set)
        self.U = lil_matrix(np.zeros((self.m, self.k)))
        self.V = lil_matrix(np.zeros((self.k, self.n)))
        self.solve()

    

    def solve():
        # initialize
        pass

    @staticmethod
    def bidirectional_growth(self, X, t):
        """
        """
        # build Upper Triangular-Like (UTL) matrix by sorting rows and columns by their sum
        U_sum = np.array(X.sum(axis=1)).squeeze()
        V_sum = np.array(X.sum(axis=0)).squeeze()
        # get reversed order by sum
        U_order = np.argsort(U_sum[::-1])
        V_order = np.argsort(V_sum[::-1])
        # exclude empty rows and columns
        U_order = U_order[U_sum > 0]
        V_order = V_order[V_sum > 0]
        # mid-point
        U_mid = int((max(U_order)-min(U_order)) / 2)
        V_mid = int((max(V_order)-min(V_order)) / 2)
        # extract middle column and row vector
        u = X[U_order.index(U_mid)]
        v = X[V_order.index(V_mid)]
        # find v_tmp
        corr = matmul(U=u, V=np.ones(1, self.n), sparse=True, boolean=True)
        corr = multiply(X, corr)
        corr = corr.sum(axis=0)
        v_tmp = corr > (t * self.n)
        # find u_tmp
        corr = matmul(U=np.ones(self.m, 1), V=v, sparse=True, boolean=True)
        corr = multiply(X, corr)
        corr = corr.sum(axis=1)
        u_tmp = corr > (t * self.m)
        # cost u, v_tmp
        error(gt=X, pd=matmul(U=U, V=V, sparse=True, boolean=True))


        # u, v_tmp = self._get_factors(X=X, axis=1, t=t)
        # u_tmp, v = self._get_factors(X=X, axis=0, t=t)
        return u, v
    
    # @staticmethod
    # def _get_factors(X, axis, t):
    #     sum = np.array(X.sum(axis=axis)).squeeze()
    #     order = np.argsort(sum[::-1])
    #     order = order[sum > 0]
    #     mid = int((max(order)-min(order)) / 2)
    #     a = X[order.index(mid)]

    #     return a, b

    @staticmethod
    def cost(X, U, V):
        UV = 
        cost = 


    def weak_signal_detection():
        pass


