import numpy as np
from ..utils import matmul, multiply, sum, bool_to_index, multiply, weighted_error, get_prediction, get_residual, ERR
from .BaseModel import BaseModel
from scipy.sparse import issparse, lil_matrix, csr_matrix, hstack


class MEBF(BaseModel):
    '''Median Expansion for Boolean Factorization
    
    From the paper 'Fast And Efficient Boolean Matrix Factorization By Geometric Segmentation'.
    '''
    def __init__(self, k=None, tol=0, t=None, w_fp=1, w_fn=1):
        '''
        k : int, optional
            The target rank.
            If `None`, it will factorize until the error is smaller than `tol`, or when other stopping criteria is met.
        tol : float, default: 0
            The target error.
        t :
            Threshold.
        w_fp, w_fn : float
            The penalty weights for FP and FN, respectively. 
        '''
        self.check_params(k=k, tol=tol, t=t, w_fp=w_fp, w_fn=w_fn)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train=X_train, X_val=X_val, X_test=X_test)
        self.init_model()
        
        self._fit()

        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):
        # update residual and coverage
        self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
        self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

        self.cost = self.X_train.sum()

        self.u = lil_matrix((self.m, 1))
        self.v = lil_matrix((self.n, 1))

        k = 0
        is_improving = True
        while is_improving:
            self.bidirectional_growth()

            if self.u.sum() == 0 or self.v.sum() == 0:
                is_improving = self.early_stop(msg="No pattern found", k=k)
                break

            if self.d_cost > 0: # cost increases
                self.print_msg("k: {}, cost increases by {}".format(k, self.d_cost))
                self.weak_signal_detection() # fall back to small pattern

                if self.d_cost > 0: # cost still increases
                    is_improving = self.early_stop(msg="Cost stops decreasing", k=k)
                    break

            if self.u.sum() == 0 or self.v.sum() == 0:
                is_improving = self.early_stop(msg="No pattern found", k=k)
                break

            self.set_factors(k, u=self.u, v=self.v)

            self.cost = self.cost + self.d_cost

            # update residual and coverage
            self.X_pd = get_prediction(U=self.U, V=self.V, boolean=True)
            self.X_rs = get_residual(X=self.X_train, U=self.U, V=self.V)

            self.print_msg("k: {}, pattern: {}, d_cost: {}, cost: {}, e: {}, rs: {}, err: {}".format(
                k, 
                [self.u.sum(), self.v.sum()], 
                self.d_cost, 
                self.cost, 
                np.round(ERR(gt=self.X_train, pd=self.X_pd) * self.m * self.n), 
                self.X_rs.sum(), 
                ERR(gt=self.X_train, pd=self.X_pd), 
            ))

            # evaluate
            self.evaluate(
                df_name='updates', 
                head_info={
                    'cost': self.cost, 
                    'shape': [self.u.sum(), self.v.sum()], 
                    'rs': self.X_rs.sum(), 
                }
            )

            # early stop detection
            is_improving = self.early_stop(error=ERR(gt=self.X_train, pd=self.X_pd), k=k)
            is_improving = self.early_stop(n_factor=k+1)
            if self.X_rs.sum() == 0:
                break

            k += 1


    def bidirectional_growth(self):
        '''Bi-directional growth algorithm.
        '''      
        error = weighted_error(gt=self.X_train, pd=self.X_pd, w_fp=self.w_fp, w_fn=self.w_fn)

        u_0, v_0 = self.get_factor(axis=0)
        if u_0.sum() == 0 or v_0.sum() == 0:
            e_0 = error
        else:
            U_0, V_0 = hstack([self.U, u_0]), hstack([self.V, v_0])
            X_0 = matmul(U=U_0, V=V_0.T, sparse=True, boolean=True)
            e_0 = weighted_error(gt=self.X_train, pd=X_0, w_fp=self.w_fp, w_fn=self.w_fn)
        
        u_1, v_1 = self.get_factor(axis=1)
        if u_1.sum() == 0 or v_1.sum() == 0:
            e_1 = error
        else:
            U_1, V_1 = hstack([self.U, u_1]), hstack([self.V, v_1])
            X_1 = matmul(U=U_1, V=V_1.T, sparse=True, boolean=True)
            e_1 = weighted_error(gt=self.X_train, pd=X_1, w_fp=self.w_fp, w_fn=self.w_fn)

        if e_0 <= e_1:
            self.u, self.v, self.d_cost = u_0, v_0, e_0 - error
        else:
            self.u, self.v, self.d_cost = u_1, v_1, e_1 - error


    def weak_signal_detection(self):
        '''Weak signal detection algorithm.
        '''
        error = weighted_error(gt=self.X_train, pd=self.X_pd, w_fp=self.w_fp, w_fn=self.w_fn)

        u_0, v_0 = self.get_weak_signal(axis=0)
        if u_0.sum() == 0 or v_0.sum() == 0:
            e_0 = error
        else:
            U_0, V_0 = hstack([self.U, u_0]), hstack([self.V, v_0])
            X_0 = matmul(U=U_0, V=V_0.T, sparse=True, boolean=True)
            e_0 = weighted_error(gt=self.X_train, pd=X_0, w_fp=self.w_fp, w_fn=self.w_fn)
        
        u_1, v_1 = self.get_weak_signal(axis=0)
        if u_1.sum() == 0 or v_1.sum() == 0:
            e_1 = error
        else:
            U_1, V_1 = hstack([self.U, u_1]), hstack([self.V, v_1])
            X_1 = matmul(U=U_1, V=V_1.T, sparse=True, boolean=True)
            e_1 = weighted_error(gt=self.X_train, pd=X_1, w_fp=self.w_fp, w_fn=self.w_fn)

        if e_0 <= e_1:
            self.u, self.v, self.d_cost = u_0, v_0, e_0 - error
        else:
            self.u, self.v, self.d_cost = u_1, v_1, e_1 - error
    
    
    def get_factor(self, axis):
        '''Get factor for bi-directional growth.

        axis :
            0, sort cols, find middle u and grow on v
            1, sort rows, find middle v and grow on u
        a, b : np.matrix
        '''
        scores = sum(X=self.X_rs, axis=axis)
        idx = np.flip(np.argsort(scores)).astype(int)
        idx = [i for i in idx if scores[i] > 0]
        
        if len(idx) == 0:
            return (csr_matrix([]), csr_matrix([]))

        mid = idx[int(np.floor(len(idx) / 2))]
        a = self.X_rs[:, mid] if axis == 0 else self.X_rs[mid, :]
        idx = bool_to_index(a)
        X_sub = self.X_rs[idx, :] if axis == 0 else self.X_rs[:, idx]
        b = sum(X=X_sub, axis=axis) > self.t * a.sum()
        b = csr_matrix(b)
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)

        return (a, b) if axis == 0 else (b, a)
    

    def get_weak_signal(self, axis):
        '''Get factor for weak signal detection.

        axis :
            0, find u and grow on v
            1, find v and grow on u
        a, b : np.matrix
        '''
        scores = sum(X=self.X_rs, axis=axis)
        idx = np.flip(np.argsort(scores)).astype(int)
        first, second = idx[0], idx[1]
        if axis == 0:
            a = multiply(self.X_rs[:, first], self.X_rs[:, second], boolean=True)
        else:
            a = multiply(self.X_rs[first, :], self.X_rs[second, :], boolean=True)
        idx = bool_to_index(a)
        X_sub = self.X_rs[idx, :] if axis == 0 else self.X_rs[:, idx]
        b = sum(X=X_sub, axis=axis) > self.t * a.sum()
        b = csr_matrix(b)
        a = a.reshape(-1, 1)
        b = b.reshape(-1, 1)

        return (a, b) if axis == 0 else (b, a)

