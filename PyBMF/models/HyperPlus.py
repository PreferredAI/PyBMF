from .Hyper import Hyper
from ..utils import FPR, matmul, FP, get_prediction, get_residual, ignore_warnings
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, hstack


class HyperPlus(Hyper):
    '''The Hyper+ algorithm.
    
    Hyper is an exact decomposition algorithm. Hyper+ is used after fitting a Hyper model. It's a relaxation of the exact decomposition.

    .. topic:: Reference

        Summarizing Transactional Databases with Overlapped Hyperrectangles. Xiang et al. SIGKDD 2011.

    Parameters
    ----------
    model : Hyper class
        The fitted ``Hyper`` model.
    target_k : int
        The target number of factors. 
        By default, it's 1.
        This will ask the model to factorize all the way down to ``k`` = 1.
        The last pattern will always be full 1 matrix.
    samples : int, default: all possible samples
        Number of pairs to be merged during trials. 
        Smaller ``samples`` will be faster but more likely to surpass the false positive rate ``beta``.
    beta : float
        The upper limit of false positive / ground truth. 
        1.0 means no limit.
    '''
    def __init__(self, model, target_k, samples=500, beta=np.inf):
        self.check_params(model=model, beta=beta, target_k=target_k, samples=samples)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)

        assert self.beta is not None or self.target_k is not None, "Please specify beta or target_k or both."

        # import model
        if 'model' in kwargs:
            model = kwargs.get('model')
            assert isinstance(model, Hyper), "Please import a Hyper model."
            self.import_model(U=model.U, V=model.V, T=model.T, I=model.I, k=model.k, logs=model.logs)
            print("[I] k from model :", self.k)
    
    
    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val, X_test=X_test)
        # do not re-init model factors and logs since the model is imported
        self._start_timer()

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)

    
    @ignore_warnings
    def _fit(self):

        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

        fpr = FPR(gt=self.X_train, pd=self.X_pd) # fp rate, FP / N
        fpb = FP(gt=self.X_train, pd=self.X_pd) / self.X_train.sum() # fp budget, FP / P

        # record U, V
        self.logs['U'] = []
        self.logs['V'] = []

        is_improving = True
        while is_improving:

            # sample pairs

            a = int(self.k * (self.k - 1) / 2)
            size = int(min(self.samples, a))
            pairs_idx = np.random.choice(a=a, size=size, replace=False)

            pairs = []
            for m in tqdm(range(self.k - 1), position=0, leave=True, desc="[I] Sampling pairs... Current FP budget: {:.3f}".format(fpb)):
            # for m in range(self.k - 1):
                idx_0 = 0.5 * m * ((self.k - 1) + (self.k - m))
                idx_1 = 0.5 * (m + 1) * ((self.k - 1) + (self.k - (m + 1)))
                idx = pairs_idx[(pairs_idx >= idx_0) & (pairs_idx < idx_1)]

                for i in idx:
                    n = int(i - idx_0 + m + 1)
                    pairs.append([m, n])

            # start trials
            
            best_m, best_n = None, None
            best_T, best_I = None, None
            best_U, best_V = None, None
            best_savings = 0
            # for m, n in tqdm(pairs, position=1, leave=False, desc="[I] Merging"):
            for m, n in pairs:

                T = list(set(self.T[m] + self.T[n]))
                I = list(set(self.I[m] + self.I[n]))
                U = lil_matrix((self.m, 1))
                V = lil_matrix((self.n, 1))
                U[T] = 1
                V[I] = 1
                
                # X_pd with merged pattern
                idx = [i for i in range(self.U.shape[1]) if i != m and i != n]
                pattern = matmul(U, V.T, sparse=True, boolean=True)
                X_pd = get_prediction(U=self.U[:, idx], V=self.V[:, idx], boolean=True)
                X_pd[pattern.astype(bool)] = 1

                fpr = FPR(gt=self.X_train, pd=X_pd) # fp rate, FP / N
                fpb = FP(gt=self.X_train, pd=X_pd) / self.X_train.sum() # fp budget, FP / P

                if fpb > self.beta:
                    continue

                savings = cost_savings(self.T[m], self.I[m], self.T[n], self.I[n], X_pd)

                if savings > best_savings:
                    best_m, best_n = m, n
                    best_T, best_I = T, I
                    best_U, best_V = U, V
                    best_savings = savings

            if best_m is None:
                is_improving = False
                break

            # update T, I
            idx = [i for i in range(self.k) if i not in [best_m, best_n]]
            self.T = [self.T[i] for i in idx]
            self.I = [self.I[i] for i in idx]
            self.T.append(best_T)
            self.I.append(best_I)
            
            # update U, V
            self.U = hstack([self.U[:, idx], best_U], format='lil')
            self.V = hstack([self.V[:, idx], best_V], format='lil')

            self.logs['U'].append(self.U.copy())
            self.logs['V'].append(self.V.copy())
          
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

            fpr = FPR(gt=self.X_train, pd=self.X_pd) # fp rate, FP / N
            fpb = FP(gt=self.X_train, pd=self.X_pd) / self.X_train.sum() # fp budget, FP / P
            ocr = FP(gt=self.X_train, pd=self.X_pd) / self.X_pd.sum() # oc rate, FP / predicted P

            # self.k -= 1
            self.k = self.U.shape[1]

            # evaluate
            self.evaluate(df_name='refinements', head_info={
                'k': self.U.shape[1], 
                'savings': best_savings, 
                'FPR': fpr, 
                'FPB': fpb, 
                'OCR': ocr, 
                })

            is_improving = fpb <= self.beta and self.k >= self.target_k
            

def cost_savings(T_0, I_0, T_1, I_1, X_pd):
    '''Compute the cost savings of merging `H_0` and `H_1`.

    savings = model description savings / exclusive area of merged pattern

    `H_0` = [`T_0`, `I_0`], `H_1` = [`T_1`, `I_1`]
    '''
    T = list(set(T_0 + T_1))
    I = list(set(I_0 + I_1))
    denominator = len(T) * len(I) - X_pd[T, :][:, I].sum()
    if denominator == 0:
        savings = np.Inf
        return savings
    else:
        numerator = len(T_0) + len(T_1) + len(I_0) + len(I_1) - len(T) - len(I)
        savings = numerator / denominator
        return savings
    