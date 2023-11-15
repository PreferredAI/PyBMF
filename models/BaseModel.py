import numpy as np
from utils import show_matrix, reverse_index, dot, matmul, check_sparse, TPR, FPR, PPV, ACC, F1
import time
from scipy.sparse import isspmatrix


class BaseModel():
    def __init__(self) -> None:
        # model parameters
        self.U = None # csr_matrix
        self.V = None # csr_matrix


    def check_params(self, **kwargs):
        if "k" in kwargs: # some algorithms have no predefined k or don't need k at all
            self.k = kwargs.get("k")
            print("[I] k            :", self.k)
        if "display" in kwargs:
            self.display = kwargs.get("display")
            print("[I] display :", self.display)
        if "seed" in kwargs:
            seed = kwargs.get("seed")
            if seed is None and not hasattr(self,'seed'): # use time as self.seed
                seed = int(time.time())
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            elif seed is not None: # overwrite self.seed
                self.seed = seed
                self.rng = np.random.RandomState(seed)
                print("[I] seed         :", self.seed)
            else: # self.rng remains unchanged
                pass


    def fit(self, train_set, val_set=None):
        """Fit the model to observations
        
        train_set: Preference data.
        val_set: Preference data for model selection purposes.
        """
        raise NotImplementedError("[E] Missing fit method.")
    

    def check_dataset(self, train_set, val_set=None):
        # load train and val set
        # most heuristics don't use val_set for auto tuning
        if train_set is None:
            print("[E] Missing training set.")
            return
        if val_set is None:
            print("[W] Missing validation set.")
        self.train_set = train_set
        self.val_set = val_set
        self.import_UVX()


    def import_UVX(self):
        self.U_info = self.train_set.factor['U']
        self.V_info = self.train_set.factor['V']
        self.X_train = self.train_set.matrix['X'].lil_matrix
        self.X_val = None if self.val_set is None else self.val_set.matrix['X'].triplet
        self.m = self.train_set.matrix['X'].m
        self.n = self.train_set.matrix['X'].n


    # implement import_UVX_UTY, import_UVX_WVZ, etc. as you want


    def score(self, U_idx, V_idx):
        """Predict the scores/ratings of a user for an item
        
        Parameters
        ----------
        U_idx: int, required
            The index of the user for whom to perform score prediction.

        V_idx: int, required
            The index of the item for which to perform score prediction.

        Returns
        -------
        score : int,
            Relative score that the user gives to the item or to all known items
        """
        return dot(self.U[U_idx, :], self.V[:, V_idx].T, boolean=True)
    

    def eval(self, test_set=None):
        if test_set is not None:
            '''
            using test_set.matrix['X'] for testing
            '''
            U_idx = test_set.matrix['X'].triplet[0]
            V_idx = test_set.matrix['X'].triplet[1]
            gt_data = test_set.matrix['X'].triplet[2]
            pd_num = len(gt_data)
            pd_data = np.zeros(pd_num, dtype=int)

            for i in range(pd_num):
                pd = self.score(U_idx=U_idx[i], V_idx=V_idx[i])
                pd_data[i] = pd

            tpr = TPR(gt=gt_data, pd=pd_data)
            fpr = FPR(gt=gt_data, pd=pd_data)
            ppv = PPV(gt=gt_data, pd=pd_data)
            acc = ACC(gt=gt_data, pd=pd_data)
            f1 = 2 * ppv * tpr / (ppv + tpr)
            
            return tpr, fpr, ppv, acc, f1
        else:
            '''
            using train_set.matrix['X'] for testing for reconstruction tasks
            '''
            pass


    def show_matrix(self, settings=None, scaling=1.0, pixels=5, title=None, colorbar=False):
        if self.display is True:
            if settings is None:
                U_order = reverse_index(idx=self.U_info.order)
                V_order = reverse_index(idx=self.V_info.order)
                U = self.U[U_order, :]
                V = self.V[:, V_order]
                # to dense
                U = check_sparse(U, sparse=False)
                V = check_sparse(V, sparse=False)
                X = matmul(U, V, boolean=True, sparse=False)
                settings = [(U, [1, 0], "U"), 
                            (V, [0, 1], "V"), 
                            (X, [1, 1], "X:" + str(self.X_train.shape))]
            elif isspmatrix(settings) or isinstance(settings, np.ndarray):
                # when settings is a matrix
                settings = [(check_sparse(settings, sparse=False), [0, 0], title)]
            show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title, colorbar=colorbar)
            