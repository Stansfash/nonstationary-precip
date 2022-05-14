# Sparse GP benchmarks

import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood
from torch.utils.data import TensorDataset, DataLoader

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

import urllib.request
import os
import numpy as np
from math import floor
import pandas as pd

import SGP.sgpr as sgpr

data_df = pd.read_csv('khyber_2000_2010_tp.csv')

data = torch.Tensor(data_df.values)
X = data[:, 1:-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()


train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

# Inducing points
idx = np.random.randint(len(train_x), size=int(len(train_x)*0.10)+1)
Z_init = train_x[idx]

# Kernels
SE_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3))
custom_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3) + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0])* gpytorch.kernels.PeriodicKernel(active_dims=[0]))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = sgpr.SparseGPR(train_x, train_y, likelihood, Z_init, custom_kernel)

optimizer = torch.optim.Adam(model.parameters(), lr=0.05)


# Train
with gpytorch.settings.cholesky_jitter(1e-1):
    losses = model.train_model(optimizer, max_steps=5000)

from metrics import rmse, nlpd

test_pred = model.posterior_predictive(test_x)
y_std = torch.tensor([1.0]) ## did not scale y-values

rmse_test = np.round(rmse(test_pred.loc, test_y, y_std).item(), 4)
nlpd_test = np.round(nlpd(test_pred, test_y, y_std).item(), 4)

print(rmse_test, nlpd_test)
