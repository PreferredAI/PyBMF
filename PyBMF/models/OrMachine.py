import sys
import os

# current_path = os.path.dirname(os.path.abspath(__file__))
# pybmf_path = os.path.abspath(os.path.join(current_path, os.pardir))
# external_model_path = os.path.abspath(os.path.join(current_path, 'bmf_ormachine_rakut'))
# sys.path.insert(1, pybmf_path)
# sys.path.insert(1, current_path)
# sys.path.insert(1, external_model_path)

from .ContinuousModel import ContinuousModel
from ..utils import show_matrix, matmul

try:
    import ormachine
except ImportError:
    print('[E] Missing package: ormachine. Please install it first.')
    pass


class OrMachine(ContinuousModel):
    def __init__(self, k):
        self.check_params(k=k)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._to_dense()
        self._to_float()

        self._fit()       
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):

        X = 2 * self.X_train - 1

        # invoke machine object
        orm = ormachine.machine()
        data = orm.add_matrix(val=X, sampling_indicator=False)

        # add layer 
        layer1 = orm.add_layer(size=self.k, child=data, lbda_init=2)

        # run inference
        orm.infer()

        self.U = orm.members[1].val
        self.U = (self.U + 1) / 2

        self.V = orm.members[2].val
        self.V = (self.V + 1) / 2

        self.X_pd = matmul(self.U, self.V.T, boolean=True, sparse=True)
        self.evaluate(df_name='boolean')