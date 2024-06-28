from .BaseModel import BaseModel
from tqdm import tqdm
import numpy as np
from utils import sum, invert, multiply, to_dense, ERR, bool_to_index
from utils import get_prediction, get_residual, ignore_warnings, to_sparse, description_length
from itertools import product
from scipy.sparse import lil_matrix, hstack


class Panda(BaseModel):
    '''PaNDa and PaNDa+ algorithm.

    PaNDa and PaNDa+ both fix `w_fp`, `w_fn` to [1, 1].
    PaNDa fixes `w_model` to 1 while PaNDa+ does not.

    Reference
    ---------
    Mining Top-K Patterns from Binary Datasets in presence of Noise.
    A unifying framework for mining approximate top-k binary patterns.
    '''
    def __init__(self, k=None, tol=0, w_model=1, w_fp=1, w_fn=1, init_method='correlation', exact_decomp=False):
        '''
        Parameters
        ----------
        k : int, optional
            The target rank.
            If `None`, it will factorize until the error is smaller than `tol`, or when other stopping criteria is met.
        tol : float, default: 0
            The target error.
        w_model : float
            The model code weight `rho` in the Panda+ paper.
        w_fp, w_fn : float
            The penalty weights for FP and FN, respectively.
            They are added on top of the original works of Panda and Panda+.
        init_method : str
            How items are sorted.
        exact_decomp : bool
            Exact decomposition.
        '''
        self.check_params(k=k, tol=tol, w_model=w_model, w_fp=w_fp, w_fn=w_fn, init_method=init_method, exact_decomp=exact_decomp)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        assert self.init_method in ['frequency', 'couples-frequency', 'correlation']

        if self.exact_decomp:
            print("[I] Exact decomposition mode.")
            self.w_model = 0
            self.init_method = 'frequency'


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)
        
        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        '''
        E : list
            The item extension list.
        I : list
            The item list.
        T : list
            The user or transaction list.
        '''
        # update residual and coverage
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

        self.I = lil_matrix((self.n, 1))
        self.T = lil_matrix((self.m, 1))

        cost_old = description_length(
            gt=self.X_train, 
            U=hstack([self.U, self.T]).tolil(), 
            V=hstack([self.V, self.I]).tolil(), 
            w_model=self.w_model, 
            w_fp=self.w_fp, 
            w_fn=self.w_fn, 
        )

        k = 0
        is_improving = True
        if self.verbose is False:
            pbar = tqdm(total=self.k, position=0)

        while is_improving:
            desc = f"[I] k: {k} - [{cost_old}]" # init cost
            self.find_core()
            desc += f" -> [{self.cost_now}]" # core cost
            if not self.exact_decomp:
                # disable extend_core step to find dense cores only
                self.extend_core()
                desc += f" -> [{self.cost_now}]" # extension cost

            if self.verbose:
                print(desc)
            else:
                pbar.set_description(desc)

            # early stop detection
            # the original Panda stops when cost starts increasing
            # we disble this to keep the algorithm running
            if self.cost_now > cost_old:
                # is_improving = self.early_stop(msg='Cost starts increasing.', k=k)
                # continue
                print("[W] Cost increased.")
            cost_old = self.cost_now

            # update factors
            self.set_factors(k, u=self.T, v=self.I)
            
            # update residual and coverage
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

            # evaluate
            self.evaluate(
                df_name='updates', 
                head_info={
                    'cost': self.cost_now, 
                    'shape': [int(self.T.sum()), int(self.I.sum())], 
                }
            )
            
            # early stop detection
            is_improving = self.early_stop(
                error=ERR(gt=self.X_train, pd=self.X_pd), 
                n_factor=k+1, 
                k=k)
            
            # update pbar and k
            if self.verbose is False:
                pbar.update(1)
            k += 1


    @ignore_warnings
    def find_core(self):
        '''Find a dense core (T, I) and its extension list E.
        
        E : list
            The item extension list.
        I : spmatrix
            The item list.
        T : spmatrix
            The user or transaction list.
        '''
        self.E = list(range(self.n)) # start with all items
        self.I = lil_matrix((self.n, 1))
        self.T = lil_matrix((self.m, 1))

        # initialize extension list
        if self.init_method == 'frequency':
            self.sort_items(method='frequency')
        elif self.init_method == 'couples-frequency' or self.init_method == 'correlation':
            self.sort_items(method='couples-frequency')

        # add the first item from E to I
        self.I[self.E[0]] = 1
        self.T = self.X_rs[:, self.E[0]]
        self.E.pop(0)

        cost_old = description_length(
            gt=self.X_train, 
            U=hstack([self.U, self.T]).tolil(), 
            V=hstack([self.V, self.I]).tolil(), 
            w_model=self.w_model, 
            w_fp=self.w_fp, 
            w_fn=self.w_fn, 
        )

        i = 0
        while i < len(self.E):
            I = self.I.copy()
            if self.init_method == 'correlation':
                self.sort_items(method='correlation') # re-order extension list
                I[self.E[0]] = 1 # use item with highest correlation
                T = multiply(self.T, self.X_rs[:, self.E[0]])
            else:
                I[self.E[i]] = 1
                T = multiply(self.T, self.X_rs[:, self.E[i]])

            # check cost by description_length(): this is expensive
            # cost_new = description_length()

            # check cost by cost difference: this is faster
            w0, h0 = self.I.sum(), self.T.sum()
            w1, h1 = self.I.sum() + 1, T.sum()
            d_cost = self.w_model * ((w1 + h1) - (w0 + h0)) - self.w_fn * ((w1 * h1) - (w0 * h0))

            if d_cost <= 0: # cost_new <= cost_old
                cost_new = cost_old + d_cost
                self.print_msg(f"       [{cost_old}] -> [{cost_new}] (find_core: add col, d_cost: {d_cost})")
                cost_old = cost_new

                # update I, T and E
                self.I = I
                self.T = T
                if self.init_method == 'correlation':
                    self.E.pop(0)
                else:
                    self.E.pop(i)
            else:
                i += 1
       
        self.cost_now = cost_old


    @ignore_warnings
    def extend_core(self):
        cost_old = self.cost_now
        
        for i in range(len(self.E)):
            # update I
            I = self.I.copy()
            I[self.E[i]] = 1
            cost_new = description_length(
                gt=self.X_train, 
                U=hstack([self.U, self.T]).tolil(), 
                V=hstack([self.V, I]).tolil(), 
                w_model=self.w_model, 
                w_fp=self.w_fp, 
                w_fn=self.w_fn, 
            )

            if cost_new <= cost_old:
                self.print_msg(f"       [{cost_old}] -> [{cost_new}] (extend_core: add col)")
                self.I = I
                cost_old = cost_new
            else:
                continue

            # check cost by cost difference: this is faster
            idx_I = bool_to_index(self.I) # indices of items in itemset I
            idx_T = bool_to_index(invert(self.T)) # indices of transactions outside T

            d_fn = - self.X_rs[idx_T][:, idx_I].sum(axis=1) # newly added false negatives
            d_fp = self.I.sum() - self.X_pd[idx_T][:, idx_I].sum(axis=1) - self.X_rs[idx_T][:, idx_I].sum(axis=1) # newly added false positives
            d_cost = self.w_model * 1 + (self.w_fn * d_fn + self.w_fp * d_fp) # cost difference

            # update T
            idx = to_dense(d_cost, squeeze=True) <= 0
            idx_T = idx_T[idx] # row indices whose d_cost <= 0
            self.T[idx_T] = 1

            if idx.sum() > 0: # at least 1 transaction is added to T
                d_cost = d_cost[idx].sum()
                cost_new = cost_old + d_cost
                self.print_msg(f"       [{cost_old}] -> [{cost_new}] (extend_core: add row, d_cost: {d_cost}, row(s) added: {idx.sum()})")
                cost_old = cost_new

        self.cost_now = cost_old # update current cost


    def sort_items(self, method):
        '''Sort the extension list by descending scores.

        frequency :
            Sort items in extension list by frequency.
        couples-frequency :
            Sort items in extension list by frequency of item-pairs that include an item.
        correlation :
            Sort items in extension list by correlation with current transactions T.
        '''
        if method == 'frequency':
            scores = sum(self.X_rs[:, self.E], axis=0)
        elif method == 'couples-frequency':
            scores = np.zeros(len(self.E))
            for i in range(len(self.E)):
                T = self.X_rs[:, self.E[i]] # T of i-th item in E
                idx_T = bool_to_index(T) # indices of transactions
                scores[i] = sum(self.X_rs[idx_T, :], axis=0).sum() # sum of 1 and 2-itemset frequency
            scores = scores - sum(self.X_rs[:, self.E], axis=0) # sum of 2-itemset frequency
        elif method == 'correlation':
            idx_T = bool_to_index(self.T)
            scores = sum(self.X_rs[idx_T][:, self.E], axis=0)

        idx = np.flip(np.argsort(scores)).astype(int)
        self.E = np.array(self.E)[idx].tolist()
