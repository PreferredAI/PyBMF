class BaseData:
    def __init__(self):
        """Built-in datasets

        X: csr_matrix
            will be passed to NoSplit, RatioSplit or CrossValidation later
        factor_info: list of tuples
            [(order, idmap, alias), (order, idmap, alias)]
        """
        self.X = None
        self.factor_info = None