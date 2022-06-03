import torch
import tqdm
import gpytorch
import urllib.request
import os
from math import floor
import pandas as pd
import numpy as np
import models.dgps as m
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import DeepApproximateMLL
from gpytorch.kernels import RBFKernel, ScaleKernel, PeriodicKernel, MaternKernel
from sklearn.utils import shuffle
import scipy.stats
from scipy.special import inv_boxcox
from utils.metrics import nlpd, rmse

filepath = 'data/uib_2000_2010_tp.csv'

# Kernel
kernel = gpytorch.kernels.ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[1,2])+ gpytorch.kernels.RBFKernel(active_dims=[0]) * gpytorch.kernels.PeriodicKernel(active_dims=[0]))

df = pd.read_csv(filepath)
#data_df = shuffle(df, random_state=random_state)
data = torch.Tensor(df.values)

X = data[:394*5,:-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:394*5, -1]
y_tr, bc_param = scipy.stats.boxcox(y)
y_tr = torch.Tensor(y_tr)

stdy_tr, _ = torch.std_mean(y_tr)
stdy, _ = torch.std_mean(y_tr)

train_n = 394*4
train_x = X[:train_n, :].contiguous()
train_y = y_tr[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y [train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y, X = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda(), X.cuda()


#### Model
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = m.ExactGPModel(train_x, train_y, likelihood, kernel)

# Initialize lengthscale and outputscale to mean of priors
#model.covar_module.outputscale = outputscale_prior.mean

#### Training
training_iter = 200

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Parameters to optimise
# training_parameters = [p for name, p in model.named_parameters()if not name.startswith('covar_module.base_kernel.kernels.1.kernels.0.raw_period')]
training_parameters =  model.parameters() # all parameters

# Use the adam optimizer
optimizer = torch.optim.Adam(training_parameters , lr=0.1)  # Includes GaussianLikelihood parameters
# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)   

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f'  # lengthscale: %.3f   noise: %.3f' 
        % (i + 1, training_iter, loss.item()))
        #model.covar_module.base_kernel.lengthscale.item(),
        #model.likelihood.noise.item()))
    optimizer.step()


#### Metrics

model.eval()
with torch.no_grad():
    pred_y_test = likelihood(model(test_x)) 
    y_mean = pred_y_test.loc.detach()
    y_var = pred_y_test.covariance_matrix.diag().sqrt().detach()

# Inverse transform predictions
# pred_y_test_tr = torch.Tensor(inv_boxcox(pred_y_test, bc_param))
y_mean_tr = torch.Tensor(inv_boxcox(y_mean, bc_param))
y_var_tr = torch.Tensor(inv_boxcox(y_var + y_mean, bc_param,)) - y_mean_tr
test_y_tr = torch.Tensor(inv_boxcox(test_y, bc_param))

## Metrics
rmse_test = rmse(y_mean_tr, test_y_tr, stdy)
nlpd_test = nlpd(pred_y_test, test_y, stdy_tr)

print(f"RMSE: {rmse_test.item()}, NLPD: {nlpd_test.item()}")
