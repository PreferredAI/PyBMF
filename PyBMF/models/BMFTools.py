from .BaseModel import BaseModel
from scipy.sparse import lil_matrix, vstack
from ..utils import ignore_warnings, matmul, record, bool_to_index, to_dense, add
import numpy as np
from .Asso import Asso


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

    def remove(self, basis_id):
        other_ids = [j for j in range(len(self.scheduler_list)) if basis_id != j]
        self.scheduler_list = self.scheduler_list[other_ids]


class BMFTools(BaseModel):
    '''Tools for BMFAlternate and its parallel version..
    '''
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


    def init_cover(self):
        self.X_uncovered = lil_matrix(self.X_train)
        self.X_covered = lil_matrix(self.X_train.shape)


    @ignore_warnings
    def update_cover(self, U, V):
        assert U.shape[1] == V.shape[1], "U and V.T should be muliplicable"

        pattern = matmul(U, V.T, sparse=True, boolean=True).astype(bool)
        self.X_uncovered[pattern] = 0
        self.X_covered[pattern] = 1


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
        
        self.basis_list = [lil_matrix((self.n_basis, self.m)), self.basis]

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



    def reinit_basis(self, basis_ids):
        '''
        basis_ids: list of int
            To be re-initialized.
        '''
        # re-init patterns

        other_ids = [j for j in range(self.n_basis) if j not in basis_ids]
        self.update_cover(U=self.basis_list[0][other_ids].T, V=self.basis_list[1][other_ids].T)        

        for i in basis_ids:
            self.basis_dim[i] = 1

            self.w_schedulers.reset(basis_id=i)
            self.w_now[i] = self.w_schedulers.step(basis_id=i)

        # re_init basis

        n_basis = len(basis_ids)

        if self.init_method == 'asso':
            basis, n_basis = self.init_basis_asso(
                X=self.X_uncovered, n_basis=n_basis, tau=self.tau)

        elif self.init_method == 'random_rows':            
            basis, n_basis = self.init_basis_random_rows(
                X=self.X_uncovered, n_basis=n_basis)
            
        elif self.init_method == 'random_bits':
            basis, n_basis = self.init_basis_random_bits(
                X=self.X_uncovered, n_basis=n_basis, p=self.p)
            
        assert n_basis == len(basis_ids)

        self.basis_list[1][basis_ids] = basis
        self.basis_list[0][basis_ids] = 0


    def get_X_pd_rest(self, basis_list, basis_id):
        '''Make X_pd_rest with other patterns using the copy from last lazy update.
        '''
        other_id = [j for j in range(self.n_basis) if basis_id != j]
        U=basis_list[0][other_id].T
        V=basis_list[1][other_id].T
        X_pd_rest = matmul(U, V.T, sparse=True, boolean=True)

        return X_pd_rest

    # MANAGE BASIS DIM ========================================================    

    def get_basis_dim(self, basis_id):
        basis_dim = self.basis_dim[basis_id]
        target_dim = 1 - basis_dim
        return basis_dim, target_dim
    

    def update_basis_dim(self, i=None, basis_dim=None):
        if i is None:
            for i in range(self.n_basis):
                self.basis_dim[i] = 1 - self.basis_dim[i]
        else:
            self.basis_dim[i] = basis_dim

    # RECORD & DISPLAY ========================================================

    def _record(self, n_iter):
        '''Temporary recording function with following items.

        - weight
        - exclusive desc len (decrease of desc len with i-th pattern)
        - exclusive score upon update (rt = real time)
        - exclusive score after updating all patterns
        - density of exclusive pattern
        '''
        for i in range(self.n_basis):
            _, target_dim = self.get_basis_dim(basis_id=i)
            records = [n_iter, target_dim]
            columns = ['n_iter', 'target_dim']

            # dl_0, dl_1, u, v, tp, fp = self.exclusive_dl(basis_id=i)

            columns += [str(i) + suffix for suffix in [
                "_w", 
                # "_excl_dl", 
                "_excl_score_rt", 
                # "_excl_score", 
                # "_dl_1v0", 
                ' ']]
            
            records += [
                self.w_now[i], 
                # dl_0 - dl_1, 
                self.scores[i], 
                # tp - fp, 
                # dl_1 / dl_0, 
                ' ']

        record(df_dict=self.logs, df_name='updates', columns=columns, records=records, verbose=self.verbose)


    def _show_patterns(self, basis_list, n_iter):
        X = lil_matrix(self.X_train.shape)
        for i in range(self.n_basis):
            pattern = matmul(U=basis_list[0][i].T, V=basis_list[1][i], sparse=True, boolean=True)
            pattern = pattern * (i + 1)
            X[pattern > 0] = pattern[pattern > 0]

        name = f'n_iter: {n_iter}, basis: {i+1}/{self.n_basis}'

        self.show_matrix(
            settings=[(X, [0, 0], name)], 
            cmap="tab20", 
            colorbar=True, 
            clim=[0, self.n_basis], 
            discrete=True, 
            center=True)
        
    # CHECK CONVERGE / REMOVE / MERGE / OVERLAP ===============================
    def check_converge(self, now, last, diff_threshold):
        '''Return the pattern ids whose score difference `abs(now - last)` is smaller than `diff_threshold`.
        '''
        converged = []
        for i in range(self.n_basis):
            if last[i] == 0:
                continue
            else:
                diff = abs(now[i] - last[i]) / abs(last[i])

            if diff < diff_threshold:
                converged.append(i)
        return converged


    def check_overlap(self, basis_ids=None, lazy_update=False, greater_id_only=False):
        if len(basis_ids) == 0:
            return []
        
        basis_list = self.basis_list.copy()

        rng = range(self.n_basis) if basis_ids is None else basis_ids
        for i in rng:
            overlaps, overlap_ratio = self.get_overlappings(
                basis_list if lazy_update else self.basis_list, basis_id=i, greater_id_only=greater_id_only, overlap_dims=2)

            print(f"[I] overlappings (on both dims) of {i}: ", overlaps, overlap_ratio)

            # overlap_id = overlaps[np.argmax(overlap_ratio)] # largest overlapping

            for overlap_id in overlaps:

                self.refine_overlapping(basis_id=i, overlap_id=overlap_id)


    def check_remove(self, basis_ids=None, lazy_update=False):
        if len(basis_ids) == 0:
            return []

        to_remove = []
        to_keep = []

        rng = range(self.n_basis) if basis_ids is None else basis_ids
        for i in rng:
            dl_none, dl_pattern, dl_merged = self.exclusive_dl(basis_ids=[i])

            print(f"[I] check_remove of {i}: ", dl_none, dl_pattern, dl_merged)

            if dl_pattern > dl_none: # remove pattern
                to_remove.append(i)
            else:
                to_keep.append(i)

        print(f"[I] removing patterns: {to_remove}, keeping patterns: {to_keep}")

        self.reinit_basis(basis_ids=to_remove)

        return to_keep


    def check_merge(self, basis_ids=None, lazy_update=False, greater_id_only=False):
        if len(basis_ids) == 0:
            return []
        
        rng = range(self.n_basis) if basis_ids is None else basis_ids
        for i in rng:
            overlaps, overlap_ratio = self.get_overlappings(
                basis_list=self.basis_list, 
                basis_id=i, 
                greater_id_only=greater_id_only, 
                overlap_dims=1)
            
            print(f"[I] overlappings (on any dim) of {i}: ", overlaps, overlap_ratio)

            for j in overlaps:
                basis_id = i
                overlap_id = j

                u0, v0 = self.basis_list[0][basis_id], self.basis_list[1][basis_id]
                u1, v1 = self.basis_list[0][overlap_id], self.basis_list[1][overlap_id]

                U, V = vstack([u0, u1], format='lil').T, vstack([v0, v1], format='lil').T

                dl_none, dl_pattern, dl_merged = self.exclusive_dl(basis_ids=[basis_id, overlap_id], U=U, V=V)

                print(f"[I] overlapping of {i}, {j}: dl_none: {dl_none}, dl_pattern: {dl_pattern}, dl_merged: {dl_merged}")

                # if dl_merged < dl_pattern:
                #     # self.merge(basis_id, overlap_id)
                #     self.basis_list[1][basis_id] = add(v0, v1, boolean=True, sparse=True)
                #     self.basis_list[0][basis_id] = add(u0, u1, boolean=True, sparse=True)

                #     to_remove = [overlap_id]
                #     self.reinit_basis(basis_ids=to_remove)