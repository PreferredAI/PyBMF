from datasets import Dataset
from models import BaseModel

class Experiment:
    def __init__(self,
                 dataset:Dataset,
                 split:str,
                 model:BaseModel,
                 grid_search:str,
                 metrics) -> None:
        pass


    def run(self):
        self.dataset

        pass
