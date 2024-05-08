from .Hyper import Hyper
from utils import FPR, matmul, FP
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, hstack


class HyperPlus(Hyper):
    '''The Hyper+ algorithm.
    
    Hyper+ is used after fitting a Hyper model. It's a relaxation of the exact decomposition algorithm.
    
    Reference
    ---------
    Summarizing Transactional Databases with Overlapped Hyperrectangles. Xiang et al. SIGKDD 2011.
    '''
    def __init__(self, model, samples=100, beta=None, target_k=None):
        '''
        model : Hyper class
            The fitted Hyper model.
        beta : float
            The upper limit of the false positive rate.
        samples : int, default: all possible samples
            Number of pairs to be merged during trials. 
        target_k : int, default: half of the original `k`
            The target number of factors.
        '''
        self.check_params(model=model, beta=beta, target_k=target_k, samples=samples)


    def check_params(self, **kwargs):
        super(Hyper, self).check_params(**kwargs)
        # self.set_params(['beta', 'target_k', 'samples'], **kwargs)
        assert self.beta is not None or self.target_k is not None, "Please specify beta or target_k or both."

        # import model
        if 'model' in kwargs:
            model = kwargs.get('model')
            assert isinstance(model, Hyper), "Please import a Hyper model."
            self.import_model(U=model.U, V=model.V, T=model.T, I=model.I, k=model.k, logs=model.logs)
            print("[I] k from model :", self.k)
            
        # default number of pairs for sampling
        if self.samples is None:
            print("[W] Missing samples. Will sample in the whole space.")
            self.samples = self.k * (self.k - 1) / 2
            print("[I] samples      :", self.samples)
        if self.samples > self.k * (self.k - 1) / 2:
            print("[W] Too many samples. Will sample in the whole space.")
            self.samples = self.k * (self.k - 1) / 2
            print("[I] samples      :", self.samples)     

    
    
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        # super(Hyper, self).fit(X_train, X_val, X_test, **kwargs)
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val, X_test=X_test)

        self._fit()
        self.finish()


    def init_model(self):
        # do not re-init model since the model is imported from another model
        pass


    def _fit(self):
        self.predict_X()
        fpr = FPR(gt=self.X_train, pd=self.X_pd)
        self.X_covered = self.X_train.copy().tolil()


        while fpr <= self.beta and self.k > self.target_k:
            print(fpr, self.k)
            # sample pairs
            a = int(self.k * (self.k - 1) / 2)
            pairs_idx = np.random.choice(a, size=self.samples, replace=False)
            pairs = []
            # for m in tqdm(range(self.k), position=0, leave=True, desc="[I] Sampling pairs"):
            for m in range(self.k):
                idx_0 = 0.5 * m * ((self.k - 1) + (self.k - m))
                idx_1 = 0.5 * (m + 1) * ((self.k - 1) + (self.k - (m + 1)))
                idx = pairs_idx[(pairs_idx >= idx_0) & (pairs_idx < idx_1)]
                for i in idx:
                    n = int(i - idx_0)
                    pairs.append([m, n])

            # trials
            best_m, best_n = None, None
            best_T, best_I = None, None
            best_U, best_V = None, None
            best_savings = 0
            # for m, n in tqdm(pairs, position=1, leave=False, desc="[I] Merging"):
            for m, n in pairs:
                # debug
                if m >= len(self.T) or n >= len(self.T):
                    print(self.k, len(self.T), m, n)
                T = list(set(self.T[m] + self.T[n]))
                I = list(set(self.I[m] + self.I[n]))
                U = lil_matrix((self.m, 1))
                V = lil_matrix((self.n, 1))
                U[T] = 1
                V[I] = 1
                
                pattern = matmul(U, V.T, sparse=True, boolean=True).astype(bool)
                X_covered = self.X_covered.copy()
                X_covered[pattern] = 1

                if FPR(gt=self.X_train, pd=X_covered) > self.beta:
                    continue
                savings = cost_savings(self.T[m], self.I[m], self.T[n], self.I[n], self.X_covered)
                if savings > best_savings:
                    best_m, best_n = m, n
                    best_T, best_I = T, I
                    best_U, best_V = U, V
                    best_savings = savings

            # update T, I
            idx = [i for i in range(self.k) if i not in [best_m, best_n]]
            self.T = [self.T[i] for i in idx]
            self.I = [self.I[i] for i in idx]
            self.T.append(best_T)
            self.I.append(best_I)
            
            # update U, V
            self.U = self.U[:, idx]
            self.V = self.V[:, idx]
            self.U = hstack([self.U, best_U], format='lil')
            self.V = hstack([self.V, best_V], format='lil')

            # update X_covered, fpr and k
            pattern = matmul(best_U, best_V.T, sparse=True, boolean=True).astype(bool)
            self.X_covered[pattern] = 1
            fpr = FPR(gt=self.X_train, pd=self.X_covered)
            self.k -= 1

            # evaluate
            self.predict_X()
            self.evaluate(df_name='refinements', head_info={'k': self.k, 'savings': best_savings, 'FPR': fpr})
            

def cost_savings(T_0, I_0, T_1, I_1, X_covered):
    '''Compute the cost savings of merging `H_0` and `H_1`.

    `H_0` = [`T_0`, `I_0`], `H_1` = [`T_1`, `I_1`]
    '''
    T = list(set(T_0 + T_1))
    I = list(set(I_0 + I_1))
    denominator = len(T) * len(I) - X_covered[T, :][:, I].sum()
    if denominator == 0:
        savings = np.Inf
        return savings
    else:
        numerator = len(T_0) + len(T_1) + len(I_0) + len(I_1) - len(T) - len(I)
        savings = numerator / denominator
        return savings