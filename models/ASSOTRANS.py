from .ASSO import ASSO
import numpy as np

class ASSOTRANS(ASSO):
    def __init__(self, X, k, tau, w=[0.5, 0.5], U_idx=None, V_idx=None, display_flag=False):
        super().__init__(X=X.T, k=k, tau=tau, w=w, U_idx=V_idx, V_idx=U_idx, display_flag=display_flag)
    
    def solve(self):
        super().solve()
        self.U, self.V = self.V.T, self.U.T
