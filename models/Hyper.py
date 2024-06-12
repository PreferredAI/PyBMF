from .BaseModel import BaseModel
from mlxtend.frequent_patterns import apriori
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, hstack
from tqdm import tqdm
from utils import matmul, get_prediction, get_residual


class Hyper(BaseModel):
    def __init__(self, min_support):
        '''
        Parameters
        ----------
        min_support : float
            The 'alpha' in the paper. The min support of frequent itemsets.
        '''
        self.check_params(min_support=min_support)
    

    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish()


    def init_model(self):
        self._init_factors()
        self._init_logs()

        self.init_itemsets()
        self.init_transactions()
        self.sort_by_cost()


    def init_itemsets(self):
        '''Initialize candidate itemsets with Apriori.

        I : list of int list
        '''
        X_df = pd.DataFrame.sparse.from_spmatrix(self.X_train.astype(bool))
        itemsets = apriori(X_df, min_support=self.min_support)
        itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))
        itemsets = itemsets[itemsets['length'] > 1]
        L = len(itemsets)
        if L == 0:
            print("[W] No itemset discovered outside singletons. Try to decrease min_support.")
        else:
            print(f"[I] Found {L} itemsets, max size: {itemsets['length'].max()}")
        self.I = [[i] for i in range(self.n)]
        for i in range(L):
            self.I.append(list(itemsets['itemsets'].values[i]))


    def init_transactions(self):
        '''Initialize transactions with cost.

        T : list of int list
        c : list of float
        X_rs : spmatrix
        '''
        self.T = []
        self.c = []
        i = 0
        progress = tqdm(range(len(self.I)), position=0, desc="[I] Initializing transactions")
        for _ in progress:
            t, c = self.find_hyper(I=self.I[i], X_gt=self.X_train, X_rs=self.X_train)
            if t == []:
                self.I.pop(i)
                # progress.reset(total=len(self.I))
            else:
                self.T.append(t)
                self.c.append(c)
                i += 1


    def sort_by_cost(self):
        '''Sort `T`, `I` and `c` lists in the ascending order of cost `c`.
        '''
        order = np.argsort(self.c)
        self.T = [self.T[i] for i in order]
        self.I = [self.I[i] for i in order]
        self.c = [self.c[i] for i in order]


    def _fit(self):
        self.T_final, self.I_final = [], []
        self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

        k = 0
        progress = tqdm(range(len(self.I)), position=0, desc=f"[I] Finding exact decomposition")
        for _ in progress:
            self.T[0], self.c[0] = self.find_hyper(I=self.I[0], X_gt=self.X_train, X_rs=self.X_rs)
            while self.T[0] == []:
                self.T.pop(0)
                self.I.pop(0)
                self.c.pop(0)
                # progress.reset(total=len(self.I))
            
            n_iter = 0
            while self.c[0] > self.c[1]:
                self.sort_by_cost()
                self.T[0], self.c[0] = self.find_hyper(I=self.I[0], X_gt=self.X_train, X_rs=self.X_rs)
                n_iter += 1

            # record lists T, I
            self.T_final.append(self.T[0])
            self.I_final.append(self.I[0])

            # update factors U, V
            U = lil_matrix((self.m, 1))
            V = lil_matrix((self.n, 1))
            U[self.T[0]] = 1
            V[self.I[0]] = 1

            self.set_factors(k, u=U, v=V)

            # evaluate
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)
            self.evaluate(df_name='updates', head_info={
                'k': k, 
                'iter': n_iter, 
                'shape': [U.sum(), V.sum()],
                })

            if self.X_rs.sum() == 0:
                break

            self.T[0], self.c[0] = self.find_hyper(I=self.I[0], X_gt=self.X_train, X_rs=self.X_rs)
            self.sort_by_cost()
            k += 1

        self.k = self.U.shape[1]
        self.T = self.T_final
        self.I = self.I_final


    @staticmethod
    def find_hyper(I, X_gt, X_rs):
        '''
        queue : list
            The indices of rows with non-zero uncoverage, in descending order. Row must be a support of I.
        '''
        covered = X_gt[:, I].sum(axis=1)
        covered = np.array(covered).flatten()

        uncovered = X_rs[:, I].sum(axis=1)
        uncovered = np.array(uncovered).flatten()

        idx = np.argsort(uncovered)[::-1]
        exact = (covered == len(I)) & (uncovered > 0)
        exact = exact[idx]
        queue = idx[exact].tolist()

        if len(queue) == 0:
            return [], np.inf
        t = queue.pop(0)
        T = [t]
        cost_old = Hyper.cost(T, I, X_rs)
        while len(queue) > 0:
            t = queue.pop(0)
            cost_new = Hyper.cost(T + [t], I, X_rs)
            if cost_new <= cost_new:
                T.append(t)
                cost_old = cost_new
            else:
                break
        return T, cost_old


    @staticmethod
    def cost(T, I, X_rs):
        '''The cost function (gamma) in Hyper.
        '''
        cost = len(T) + len(I)
        cost = cost / X_rs[T, :][:, I].sum()
        return cost
