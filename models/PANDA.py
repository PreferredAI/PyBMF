from .BaseModel import BaseModel
from tqdm import tqdm


class Panda(BaseModel):
    """PaNDa and PaNDa+ algorithms

    Reference:
        Mining Top-K Patterns from Binary Datasets in presence of Noise
        A unifying framework for mining approximate top-k binary patterns
    """
    def __init__(self, k, rho):
        self.check_params(k=k, rho=rho)


    def check_params(self, **kwargs):
        super().check_params(**kwargs)
        if "rho" in kwargs:
            self.rho = kwargs.get("rho")
            print("[I] rho          :", self.rho)


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_dataset(X_train=X_train, X_val=X_val)
        self.check_params(**kwargs)
        self._fit()


    def _fit(self):
        for i in tqdm(range(self.k), position=0):
            self.find_core()
            self.extend_core()


    def find_core(self):
        self.extension_list = []
        pass

    def extend_core(self):
        pass

    def sort_items(self):
        # self.sorted_list = 

    def cost(self):
        pass