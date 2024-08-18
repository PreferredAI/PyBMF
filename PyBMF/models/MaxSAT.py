from .ContinuousModel import ContinuousModel
import sys
import os
from ..utils import to_dense, matmul
import subprocess
import numpy as np
import pandas as pd


class MaxSAT(ContinuousModel):
    def __init__(self, k, mode='fast_undercover'):
        self.check_params(k=k, mode=mode)


    def fit(self, X_train, X_val=None, X_test=None, **kwargs):
        super().fit(X_train, X_val, X_test, **kwargs)

        self._fit()
        self.finish(show_logs=self.show_logs, save_model=self.save_model, show_result=self.show_result)


    def _fit(self):

        X_train = to_dense(self.X_train)
        # X_train.tofile("D:/Dropbox/PyBMF/models/bmf_maxsat_avellaneda/input.csv", sep = ',')
        df = pd.DataFrame.sparse.from_spmatrix(self.X_train)
        df.to_csv("D:/Dropbox/PyBMF/models/bmf_maxsat_avellaneda/input.csv", index=False, header=False)

        cp = subprocess.run([
            "wsl", 
            "~", 
            "-e", 
            "/mnt/d/Dropbox/PyBMF/models/bmf_maxsat_avellaneda/inferbmf", 
            "-k", 
            str(self.k), 
            "fromFile",
            "-o", 
            "/mnt/d/Dropbox/PyBMF/models/bmf_maxsat_avellaneda/result",
            "/mnt/d/Dropbox/PyBMF/models/bmf_maxsat_avellaneda/input.csv"
            ], capture_output=True, shell=True)

        print("=== stdout ===")
        print(cp.stdout.decode())

        print("=== stderr ===")
        print(cp.stderr.decode())

        self.U = np.genfromtxt('D:/Dropbox/PyBMF/models/bmf_maxsat_avellaneda/result.A.csv', delimiter=',')
        self.V = np.genfromtxt('D:/Dropbox/PyBMF/models/bmf_maxsat_avellaneda/result.B.csv', delimiter=',').T  

        self.X_pd = matmul(self.U, self.V.T, boolean=True, sparse=True)
        self.evaluate(df_name='boolean')