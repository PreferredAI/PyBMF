from .data import Data
from utils import to_sparse
from generators import BaseBooleanMatrix

class GeneratedData(Data):
    def __init__(self, bm: BaseBooleanMatrix, X_name: str, U_name: str, V_name: str) -> None:
        super().__init__()
        self.X.update(matrix=to_sparse(bm.X, type='csr'), name=X_name)
        self.U.update(order=bm.U_order,
                      idmap=bm.U_order,
                      alias=bm.U_order.astype(str).tolist(),
                      name=U_name,
                      ignore=True)
        self.V.update(order=bm.V_order,
                      idmap=bm.V_order,
                      alias=bm.V_order.astype(str).tolist(),
                      name=V_name,
                      ignore=True)
