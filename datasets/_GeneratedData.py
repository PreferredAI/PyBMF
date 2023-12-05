from .BaseData import BaseData
from utils import to_sparse
from generators import BaseBooleanMatrix

class GeneratedData(BaseData):
    def __init__(self, boolmat: BaseBooleanMatrix):
        super().__init__()

        self.X = to_sparse(boolmat.X, type='csr')
        U_info = (boolmat.U_order, boolmat.U_order, boolmat.U_order.astype(str))
        V_info = (boolmat.V_order, boolmat.V_order, boolmat.V_order.astype(str))
        self.factor_info = [U_info, V_info]
