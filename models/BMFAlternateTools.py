from .BaseModel import BaseModel
from scipy.sparse import lil_matrix
from utils import ignore_warnings, matmul, to_sparse, bool_to_index, to_dense
import numpy as np
from .Asso import Asso

class BMFAlternateTools(BaseModel):
    def init_basis_asso(self, X, n_basis, tau):
        '''Init basis candidates using `X` as in `Asso`.

        `Asso.build_basis()` will drop empty rows, thus n_basis may < n.
        '''
        assoc = Asso.build_assoc(X=X, dim=1) # real-valued association matrix
        basis = Asso.build_basis(assoc=assoc, tau=tau) # binary-valued basis candidates

        if n_basis is None or n_basis > basis.shape[0]:
            n_basis = basis.shape[0]
            print("[W] n_basis updated to: {}".format(n_basis))

        if n_basis < basis.shape[0]: # down-sampling
            p = to_dense(basis.sum(axis=1), squeeze=True).astype(np.float64)
            p /= p.sum()
            idx = self.rng.choice(basis.shape[0], size=n_basis, replace=False, p=p)
            basis = basis[idx, :]

        return basis, n_basis


    def init_basis_random_rows(self, X, n_basis, axis=1):
        '''Init basis candidates by randomly picking rows from `X`.

        This will drop empty rows, thus n_basis may < m.
        Can also pick non-empty columns if `axis=0`.
        '''
        idx = X.sum(axis=axis) > 0
        idx = bool_to_index(idx)
        basis = X[idx, :] if axis else X[:, idx].T
        
        if n_basis is None or n_basis > basis.shape[0]:
            n_basis = basis.shape[0]
            print("[W] n_basis updated to: {}".format(n_basis))

        if n_basis < basis.shape[0]: # down-sampling
            p = to_dense(basis.sum(axis=1), squeeze=True).astype(np.float64)
            p /= p.sum()
            idx = self.rng.choice(basis.shape[0], size=n_basis, replace=False, p=p)
            basis = basis[idx, :]

        return basis, n_basis
    

    def init_basis_random_bits(self, X, n_basis, p):
        '''Init basis candidates using random binary vectors with density `p`.
        '''
        # sampling
        basis = self.rng.choice([0, 1], size=(n_basis, X.shape[1]), p=[1 - p, p])
        basis = lil_matrix(basis, dtype=np.float64)

        return basis, n_basis


    @ignore_warnings
    def update_cover(self, U, V):
        assert U.shape[1] == V.shape[1], "U and V.T should be muliplicable"

        pattern = matmul(U, V.T, sparse=True, boolean=True).astype(bool)
        self.X_uncovered[pattern] = 0
        self.X_covered[pattern] = 1


    def init_cover(self):
        self.X_uncovered = lil_matrix(self.X_train)
        self.X_covered = lil_matrix(self.X_train.shape)


    def init_basis(self):
        if self.init_method == 'asso':
            assert self.tau is not None
            self.basis, self.n_basis = self.init_basis_asso(
                X=self.X_uncovered, n_basis=self.n_basis, tau=self.tau)

        elif self.init_method == 'random_rows':            
            self.basis, self.n_basis = self.init_basis_random_rows(
                X=self.X_uncovered, n_basis=self.n_basis)
            
        elif self.init_method == 'random_bits':
            assert self.n_basis is not None
            assert self.p is not None
            self.basis, self.n_basis = self.init_basis_random_bits(
                X=self.X_uncovered, n_basis=self.n_basis, p=self.p)
        
        # debug
        title = 'init_method: {}, n_basis: {}'.format(self.init_method, self.n_basis)
        if self.init_method == 'asso':
            title += ', tau: {}'.format(self.tau)
        elif self.init_method == 'random_bits':
            title += ', p: {}'.format(self.p)
        settings = [(self.basis, [0, 0], title)]
        self.show_matrix(settings, colorbar=False, clim=[0, 1], title=None)
        settings = [(self.X_uncovered, [0, 0], title)]
        self.show_matrix(settings, colorbar=False, clim=[0, 1], title=None)

    def init_basis_list(self):
        self.basis_list = [lil_matrix((self.n_basis, self.m)), self.basis]


class w_scheduler:
    '''Scheduler of the weight `w`.
    '''
    def __init__(self, w_list):
        self.w_list = w_list
        self.i = -1

    def step(self):
        if self.i < len(self.w_list) - 1:
            self.i += 1
        return self.w_list[self.i]
    
    def reset(self, i=-1):
        self.i = i


class w_schedulers:
    '''Manage schedulers of the weight `w`.
    '''
    def __init__(self, w_list, n_basis):
        self.w_list = w_list
        self.scheduler_list = [w_scheduler(w_list) for _ in range(n_basis)]

    def step(self, basis_id):
        return self.scheduler_list[basis_id].step()
    
    def reset(self, basis_id, i=-1):
        self.scheduler_list[basis_id].reset(i=i)

    def add(self, w_list=None):
        w_list = self.w_list if w_list is None else w_list
        self.scheduler_list.append(w_scheduler(w_list))


