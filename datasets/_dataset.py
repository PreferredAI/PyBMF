from utils import show_matrix, reverse_index, check_sparse, to_sparse, Factor, Matrix
from .data import Data
from collections import OrderedDict
from typing import Union


class Dataset:
    def __init__(self):
        '''Dataset object is the container of matrices and factor info extracted from Data
        
        It fits BMF and 2-CMF, 3-CMF. You can add more matrices and factors as you want.
        All matrix indices are pruned when it's at Data stage, i.e., no empty rows and columns.

        The recommended naming tradition:

            X = U @ V # user - item matrix
            Y = U @ T # user - user attribute matrix
            Z = W @ V # item attribute - item matrix
        '''
        self.matrix = OrderedDict()
        self.factor = OrderedDict()


    def load_data(self, X: Union[Matrix, Data], U: Factor=None, V: Factor=None):
        
        if isinstance(X, Data):
            X, U, V = X.X, X.U, X.V

        self.matrix[X.name] = Matrix(matrix=X, name=X.name)
        self.factor[U.name] = Factor(factor=U, name=U.name)
        self.factor[V.name] = Factor(factor=V, name=V.name)
        
        self.matrix[X.name].update(factors=[self.factor[U.name], self.factor[V.name]])
        self.factor[U.name].update(matrices=self.matrix[X.name])
        self.factor[V.name].update(matrices=self.matrix[X.name])


    def summarize(self, display=True, ordered=True, scaling=1.0, pixels=2, title=None):
        settings = []
        pos_list = {'X': [1, 1], 'Y': [1, 0], 'Z': [0, 1]}
        for name in self.matrix:
            m = self.matrix[name]
            # u, v = m.factors[0], m.factors[1]
            u_mean, v_mean = m.mean
            u_median, v_median = m.median
            print('[I] Summary of {}{}:'.format(m.name, m.shape))
            print('[I]   row/col mean ({:.1f}, {:.1f}), row/col median ({}, {})'.format(v_mean, u_mean, v_median, u_median))
            # print('[I] Mean num of {}s per {} in {}: {:.1f}'.format(v.name, u.name, m.name, u_mean))
            # print('[I] Mean num of {}s per {} in {}: {:.1f}'.format(u.name, v.name, m.name, v_mean))
            
            # print('[I] Median num of {}s per {} in {}: {}'.format(v.name, u.name, m.name, u_median))
            # print('[I] Median num of {}s per {} in {}: {}'.format(u.name, v.name, m.name, v_median))

            if display and m.name in pos_list:
                X = self.matrix_to_show(X=m, ordered=ordered)
                pos = pos_list[m.name]
                settings.append(tuple([X, pos, m.name]))

        if display:
            show_matrix(settings=settings, scaling=scaling, pixels=pixels, title=title)


    @staticmethod
    def matrix_to_show(X: Matrix, ordered: bool):
        has_negative_sampling = any(X.coo_matrix.data == 0)
        if has_negative_sampling:
            x = X.coo_matrix
            x.data = (x.data - 1) * 2
            x = to_sparse(x, type='csr')
        else:
            x = X.csr_matrix

        if ordered:
            U_order = reverse_index(idx=X.factors[0].order)
            V_order = reverse_index(idx=X.factors[1].order)
            x = x[U_order, :]
            x = x[:, V_order]
        return check_sparse(x, sparse=False)


