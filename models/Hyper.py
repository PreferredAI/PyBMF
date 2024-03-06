from .BaseModel import BaseModel
from mlxtend.frequent_patterns import apriori
import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, hstack
from tqdm import tqdm
from utils import matmul


class Hyper(BaseModel):
    def __init__(self, alpha):
        self.check_params(alpha=alpha)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if 'alpha' in kwargs:
            alpha = kwargs.get('alpha')
            self.alpha = alpha
            print("[I] alpha        :", self.alpha)
    

    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train, X_val)
        self.init_model()

        self.init_itemsets()
        self.init_transactions()
        self.sort_by_cost()

        self._fit()

        self.evaluate(names=['k'], values=[self.k], df_name='results')
        self.show_matrix(colorbar=True, discrete=True, clim=[0, 1], title="result")


    def init_model(self):
        self.U = None
        self.V = None
        self.logs = {}


    def init_itemsets(self):
        '''Initialize candidate itemsets with Apriori.

        I : list of int list
        '''
        X_df = pd.DataFrame.sparse.from_spmatrix(self.X_train.astype(bool))
        itemsets = apriori(X_df, min_support=self.alpha)
        itemsets['length'] = itemsets['itemsets'].apply(lambda x: len(x))
        itemsets = itemsets[itemsets['length'] > 1]
        L = len(itemsets)
        if L == 0:
            print("[W] No itemset discovered outside singletons. Try to decrease alpha.")
        else:
            print(f"[I] Found {L} itemsets, max size: {itemsets['length'].max()}")
        self.I = [[i] for i in range(self.n)]
        for i in range(L):
            self.I.append(list(itemsets['itemsets'].values[i]))


    def init_transactions(self):
        '''Initialize transactions with cost.

        T : list of int list
        c : list of float
        X_uncovered : spmatrix
        '''
        self.T = []
        self.c = []
        i = 0
        progress = tqdm(range(len(self.I)), position=0, desc="[I] Initializing transactions")
        for _ in progress:
            t, c = find_hyper(I=self.I[i], X_gt=self.X_train, X_uncovered=self.X_train)
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
        self.X_uncovered = self.X_train.copy().tolil()

        k = 0
        progress = tqdm(range(len(self.I)), position=0, desc=f"[I] Finding exact decomposition")
        for _ in progress:
            self.T[0], self.c[0] = find_hyper(I=self.I[0], X_gt=self.X_train, X_uncovered=self.X_uncovered)
            while self.T[0] == []:
                self.T.pop(0)
                self.I.pop(0)
                self.c.pop(0)
                # progress.reset(total=len(self.I))
            
            i = 0
            while self.c[0] > self.c[1]:
                self.sort_by_cost()
                self.T[0], self.c[0] = find_hyper(I=self.I[0], X_gt=self.X_train, X_uncovered=self.X_uncovered)
                i += 1

            self.T_final.append(self.T[0])
            self.I_final.append(self.I[0])

            # update factors U, V
            U = lil_matrix(np.zeros((self.m, 1)))
            V = lil_matrix(np.zeros((self.n, 1)))
            U[self.T[0]] = 1
            V[self.I[0]] = 1

            self.U = U if k == 0 else hstack([self.U, U])
            self.V = V if k == 0 else hstack([self.V, V])
                
            # update residual X_uncovered
            pattern = matmul(U, V.T, sparse=True, boolean=True).astype(bool)
            self.X_uncovered[pattern] = 0

            self.evaluate(
                names=['k', 'iter', 'size', 'uncovered'], 
                values=[k, i, pattern.sum(), self.X_uncovered.sum()], 
                df_name='updates')

            if self.X_uncovered.sum() == 0:
                break

            self.T[0], self.c[0] = find_hyper(I=self.I[0], X_gt=self.X_train, X_uncovered=self.X_uncovered)
            self.sort_by_cost()
            k += 1

        self.k = self.U.shape[1]


def find_hyper(I, X_gt, X_uncovered):
    '''
    queue : list
        The indices of rows with non-zero uncoverage, in descending order. Row must be a support of I.
    '''
    covered = X_gt[:, I].sum(axis=1)
    covered = np.array(covered).flatten()

    uncovered = X_uncovered[:, I].sum(axis=1)
    uncovered = np.array(uncovered).flatten()

    idx = np.argsort(uncovered)[::-1]
    exact = (covered == len(I)) & (uncovered > 0)
    exact = exact[idx]
    queue = idx[exact].tolist()

    if len(queue) == 0:
        return [], np.inf
    t = queue.pop(0)
    T = [t]
    cost_old = cost(T, I, X_uncovered)
    while len(queue) > 0:
        t = queue.pop(0)
        cost_new = cost(T + [t], I, X_uncovered)
        if cost_new <= cost_new:
            T.append(t)
            cost_old = cost_new
        else:
            break
    return T, cost_old


def cost(T, I, X_uncovered):
    '''The cost function (gamma) in Hyper.
    '''
    cost = len(T) + len(I)
    cost = cost / X_uncovered[T, :][:, I].sum()
    return cost
