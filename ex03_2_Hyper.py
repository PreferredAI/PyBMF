# use project files rather than the installed PyBMF

import sys
sys.path.insert(0, '../')

# # generating synthetic data

# from PyBMF.generators import BlockDiagonalMatrixGenerator

# data = BlockDiagonalMatrixGenerator(m=300, n=500, k=5, overlap=[0.3, 0.2])
# data.generate(seed=1000)
# data.add_noise(noise=[0.4, 0.1], seed=2000)

# data.show_matrix(scaling=0.5)
# loading real-world data

from PyBMF.datasets import MovieLensData

data = MovieLensData(size="100k")
data.load()

idx = data.sample(factor_id=0, n_samples=300, seed=1000)
idx = data.sample(factor_id=1, n_samples=500, seed=1000)

data.show_matrix(scaling=0.5)
# splitting the data into train, validation and test

from PyBMF.datasets import RatioSplit, NoSplit

# split = RatioSplit(X=data.X, val_size=0.1, test_size=0.2, seed=1997)
split = NoSplit(X=data.X)

# split.negative_sample(
#     train_size=split.pos_train_size,
#     val_size=split.pos_val_size,
#     test_size=split.pos_test_size,
#     seed=2023, type='popularity')

X_train, X_val, X_test = split.X_train, split.X_val, split.X_test
# # `Hyper` for exact decomposition

# Since `Hyper` is an exact decomposition algorithm, there will be absolutely no coverage on validation and test set.
# from PyBMF.models import Hyper

# min_support = 0.2

# model = Hyper(min_support=min_support)
# model.fit(X_train=X_train, X_val=X_val, X_test=X_test, task='reconstruction', verbose=False, display=False)
# `HyperPlus` for approximate decomposition
from PyBMF.models import Hyper, HyperPlus
import warnings

import numpy as np
beta = np.inf
samples = 10
target_k = 200

model = Hyper(min_support=0.2)
model.fit(X_train=X_train, X_val=X_val, X_test=X_test, task='reconstruction', verbose=False, display=False)

model_plus = HyperPlus(model=model, beta=beta, samples=samples, target_k=target_k)
model_plus.fit(X_train=X_train, X_val=X_val, X_test=X_test, task='reconstruction', verbose=False, display=True, save_model=False)
