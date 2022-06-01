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

filepath = 'data/uib_2000_2010_tp.csv'

# Kernel
kernel = gpytorch.kernels.ScaleKernel(RBFKernel(ard_num_dims=2, active_dims=[1,2])+ gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0]) * gpytorch.kernels.PeriodicKernel(active_dims=[0]))

df = pd.read_csv(filepath)
#data_df = shuffle(df, random_state=random_state)
data = torch.Tensor(df.values)

X = data[:394*5,:-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:394*5, -1]
y_tr, bc_param = scipy.stats.boxcox(y + 0.001)
y_tr = torch.Tensor(y_tr)

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
training_iter = 50

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
    #print('Iter %d/%d - Loss: %.3f'  # lengthscale: %.3f   noise: %.3f' 
    #    % (i + 1, training_iter, loss.item(),
        #model.covar_module.base_kernel.lengthscale.item(),
        #model.likelihood.noise.item()))
    optimizer.step()


#### Metrics
def negative_log_predictive_density(test_y, predicted_mean, predicted_var): 
    # Vector of log-predictive density per test point    
    lpd = torch.distributions.Normal(predicted_mean, torch.sqrt(predicted_var)).log_prob(test_y)
    # return the average
    return -torch.mean(lpd)

def sqrt_mean_squared_error(test_y, predicted_mean):  
    return torch.sqrt(torch.mean((test_y - predicted_mean)**2))

model.eval()
with torch.no_grad():
    trained_pred_dist = likelihood(model(test_x))
    predictive_mean = trained_pred_dist.mean
    predictive_variances = trained_pred_dist.variance

# Inverse transform predictions
predictive_means_tr = torch.Tensor(inv_boxcox(predictive_mean, bc_param))
vanilla_variances = torch.Tensor(np.ones((len(predictive_variances))))

rmse = sqrt_mean_squared_error(test_y, predictive_means_tr)
nlpd = negative_log_predictive_density(test_y, predictive_means_tr, vanilla_variances)

print(f"RMSE: {rmse.item()}, NLPD: {nlpd.item()}")
