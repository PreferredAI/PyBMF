import sys
import os

# current_path = os.path.dirname(os.path.abspath(__file__))
# pybmf_path = os.path.abspath(os.path.join(current_path, os.pardir))
# external_model_path = os.path.abspath(os.path.join(current_path, 'bmf_mp_ravanba'))
# sys.path.insert(1, pybmf_path)
# sys.path.insert(1, current_path)
# sys.path.insert(1, external_model_path)

from .ContinuousModel import ContinuousModel
from ..utils import show_matrix, matmul
from .bmf_mp_ravanba.matrix_completion import MatrixCompletion
from .bmf_mp_ravanba.utilities import *


class MessagePassing(ContinuousModel):
    def __init__(self, k, W='mask', prior_u=0.5, prior_v=0.5, channel_pos=0.99, channel_neg=0.99, tol=1e-4, max_iter=500, lr=0.2):
        self.check_params(k=k, W=W, prior_u=prior_u, prior_v=prior_v, channel_pos=channel_pos, channel_neg=channel_neg, tol=tol, max_iter=max_iter, lr=lr)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._to_dense()
        self._to_bool()

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):

        self.model = MatrixCompletion(
            O=self.X_train, 
            K=self.k, 
            mask=self.W, 
            min_sum=True, 
            verbose=self.verbose, 
            tol=self.tol, 
            learning_rate=self.lr,
            max_iter=self.max_iter, 
            p_x_1=self.prior_u, 
            p_y_1=self.prior_v, 
            p_1_given_1=self.channel_pos, 
            p_0_given_0=self.channel_neg
        )

        self.model.run()

        self.X_pd = matmul(self.model.X, self.model.Y.T, boolean=True, sparse=True)
        self.evaluate(df_name='boolean')