from .BaseCollectiveModel import BaseCollectiveModel
import numpy as np
from utils import to_dense, to_sparse, concat_Xs_into_X, split_U_into_Us


class ContinuousCollectiveModel(BaseCollectiveModel):
    '''Continuous collective matrix factorization.
    '''
    def __init__(self):
        raise NotImplementedError("This is a template class.")
    

    def init_model(self):
        '''The `BaseModel.init_model()` for continuous collective models.
        '''
        if hasattr(self, 'init_method') and self.init_method == 'custom':
            # init logging variables
            self._init_logs()
        else:
            # init factors and logging variables
            self._init_factors()
            self._init_logs()

        if hasattr(self, 'Ws'):
            self.init_Ws()


    def init_Ws(self):
        '''Wrapper of `ContinuousModel.init_W()` for collective models.

        `Ws` is a list of sparse matrices. Some entries can be codename "mask" or "full".
        To keep things short, if all codenames are same, `Ws` is a single `str`.
        Sometimes we only masks Xs[0]: we assume that auxilary information matrices do not have missing values. Then ['mask', 'full', 'full', ...] is required.
        '''
        if isinstance(self.Ws, str):
            self.Ws = [self.Ws] * self.n_matrices

        for i in range(self.n_matrices):
            if self.Ws[i] == 'mask':
                self.Ws[i] = self.Xs_train[i].copy()
                self.Ws[i].data = np.ones(self.Xs_train[i].data.shape) # self.X.nnz
            elif self.Ws[i] == 'full':
                self.Ws[i] = np.ones(self.Xs_train[i].shape)

            self.Ws[i] = to_sparse(self.Ws[i])


    def init_Us(self):
        '''Wrapper of `ContinuousModel.init_UV()` for collective models.
        '''
        if self.init_method == 'bmf':
            # init factors using single binary mf

            pass
        elif self.init_method == "normal": # TODO: positive constraint
            # init factors randomly with standard normal distribution
            self.X_train = concat_Xs_into_X(Xs=self.Xs_train, factors=self.factors)
            avg = np.sqrt(self.X_train.mean() / self.k)
            self.Us = [avg * self.rng.standard_normal(size=(dim, self.k)) for dim in self.factor_dims]
        elif self.init_method == "uniform":
            # init factors randomly with uniform distribution
            avg = np.sqrt(1 / self.k)
            self.Us = [avg * self.rng.rand(dim, self.k) for dim in self.factor_dims]
        elif self.init_method == "custom":
            assert hasattr(self, 'Us') # Us must be provided at this point
    

    def _to_dense(self):
        '''Turn the Xs and Ws into dense matrices.

        For temporary use during development.
        '''
        for i in range(self.n_matrices):
            self.Xs_train[i] = to_dense(self.Xs_train[i])
            self.Ws[i] = to_dense(self.Ws[i])
            if self.Xs_val is not None:
                self.Xs_val[i] = to_dense(self.Xs_val[i])
            if self.Xs_test is not None:
                self.Xs_test[i] = to_dense(self.Xs_test[i])