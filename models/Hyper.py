from .BaseModel import BaseModel


class Hyper(BaseModel):
    def __init__(self):
        super().__init__()


    def fit(self, X_train, X_val=None, **kwargs):
        self.check_params(**kwargs)
        self.load_dataset(X_train, X_val)
        self._fit()