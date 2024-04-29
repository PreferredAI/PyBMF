# Model `BinaryMFThresholdExColumnwise`
# %load_ext autoreload
# %autoreload 2
import sys
sys.path.append('D:/Dropbox/PyBMF/')
# from generators import BlockDiagonalMatrixGenerator

# n_row, n_col, k, seed = 300, 500, 5, None

# data = BlockDiagonalMatrixGenerator(m=n_row, n=n_col, k=k, overlap=[0.2, 0.1, 0.2, 0.1])
# data.generate(seed=seed) # if no seed assigned, use time instead

# data.show_matrix(title="gen_diag_sorted")
from datasets import MovieLensData

data = MovieLensData(size="100k")
data.load()

idx = data.sample(factor_id=0, n_samples=300, seed=1000)
idx = data.sample(factor_id=1, n_samples=500, seed=1000)

# idx = data.sample(factor_id=0, n_samples=300)
# idx = data.sample(factor_id=1, n_samples=500)
from datasets import RatioSplit

split = RatioSplit(X=data.X, val_size=0.1, test_size=0.2, seed=1997)
split.negative_sample(
    train_size=split.pos_train_size, 
    val_size=split.pos_val_size, 
    test_size=split.pos_test_size, 
    seed=2023, type='popularity')
# Init with `NMFSklearn`
X_train, X_val, X_test = split.X_train, split.X_val, split.X_test

k = 5
reg = 1
reg_growth = 3

from models import NMFSklearn, BinaryMFPenalty

model_nmf = NMFSklearn(k=k, init_method='nndsvd', max_iter=1000, seed=2024)
model_nmf.fit(X_train=X_train)

U, V = model_nmf.U, model_nmf.V
U.toarray()
U.max()
from models import BinaryMFThresholdExColumnwise

k = 5
us, vs = 0.1, 0.1
# W = 'full'
W = 'mask'
init_method = 'custom'

X_train, X_val, X_test = split.X_train, split.X_val, split.X_test

model = BinaryMFThresholdExColumnwise(k=k, U=U, V=V, us=us, vs=vs, W=W, init_method=init_method)
# model.fit(X_train=X_train, X_val=X_val, X_test=X_test, task='prediction', verbose=True, display=False)
model.U.toarray()
model.fit(X_train=X_train, X_val=X_val, X_test=X_test, task='prediction', verbose=True, display=False)
U
model.U.toarray()
model._fit()
# Init with `BinaryMFPenalty`
# model_bmf = BinaryMFPenalty(k=k, U=U, V=V, reg=reg, reg_growth=reg_growth, init_method='custom', max_iter=100, seed=2024)
# model_bmf.fit(X_train=X_train, X_val=X_val, X_test=X_test, task='prediction', verbose=False, display=False)

# U, V = model_bmf.U, model_bmf.V
from models import BinaryMFThresholdExColumnwise

k = 5
us, vs = 0.1, 0.1
# W = 'full'
W = 'mask'
init_method = 'custom'

X_train, X_val, X_test = split.X_train, split.X_val, split.X_test

model = BinaryMFThresholdExColumnwise(k=k, U=U, V=V, us=us, vs=vs, W=W, init_method=init_method)
model.fit(X_train=X_train, X_val=X_val, X_test=X_test, task='prediction', verbose=False, display=False)
# Visualize F
from mpl_toolkits import mplot3d
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

N = 10
dpi = 1000
x = np.linspace(-1.1, 2.1, N)
y = np.linspace(-1.1, 2.1, N)
Z = np.zeros([N, N])
X, Y = np.meshgrid(x, y)
for i in tqdm(range(N)):
    for j in range(N):
        params = [X[i, j]] * k + [Y[i, j]] * k
        Z[i, j] = model.F(params)
        
# fig = plt.figure(dpi=dpi)
# ax = plt.axes(projection='3d')
# ax.contour3D(X, Y, Z, 50, cmap='hot')
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# levels = np.logspace(0, 10, num=100, endpoint=True, base=10)
# levels = (levels - levels.min()) / (levels.max() - levels.min())
# levels = levels * (Z.max() - Z.min()) + Z.min()

# plt.figure(dpi=dpi)
# cp = plt.contour(X, Y, Z, levels=levels)
# plt.clabel(cp, inline=1, fontsize=10)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()

#########################################

# Visualize dF
# x = np.linspace(0.0, 1.2, N)
# y = np.linspace(0.0, 1.2, N)
# dx = np.zeros([N, N])
# dy = np.zeros([N, N])
# X, Y = np.meshgrid(x, y)
# for i in tqdm(range(N)):
#     for j in range(N):
#         dx[i, j], dy[i, j] = model.dF([X[i, j], Y[i, j]])

#         # re-scale
#         k = 6 / np.sqrt(dx[i, j]**2 + dy[i, j]**2)
#         dx[i, j] *= k
#         dy[i, j] *= k

# plt.figure(dpi=dpi) 
# fig = plt.quiver(X, Y, dx, dy)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()