import torch
import tqdm
import gpytorch
import urllib.request
import os
from math import floor
import pandas as pd
import numpy as np
import models as m
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import DeepApproximateMLL
from sklearn.utils import shuffle

#filepath = 'data/uib_jan2000_tp.csv'
filepath = 'data/uib_lat32_lon81_tp.csv'
rmses = []
nlpds = []

# Kernels
# kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0]) * gpytorch.kernels.PeriodicKernel(active_dims=[0]))
# kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=3) + gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0])*gpytorch.kernels.PeriodicKernel(active_dims=[0]))


for random_state in range(10):
    print('random_state = ', random_state)

    df = pd.read_csv(filepath)
    data_df = shuffle(df, random_state=random_state)
    data = torch.Tensor(data_df.values)

    X = data[:,:-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]

    train_n = int(floor(0.75 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y [train_n:].contiguous()

    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y, X = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda(), X.cuda()


    #### Model

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = m.ExactGPModel(train_x, train_y, likelihood, kernel)

    #### Training
    training_iter = 50

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
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

    rmse = sqrt_mean_squared_error(test_y, predictive_mean)
    nlpd = negative_log_predictive_density(test_y, predictive_mean, predictive_variances)
    print(f"RMSE: {rmse.item()}, NLPD: {nlpd.item()}")
    rmses.append(rmse.item())
    nlpds.append(nlpd.item())

    with torch.no_grad():
        trained_pred_dist = likelihood(model(X))
        all_predictive_means = trained_pred_dist.mean

    df1 = pd.DataFrame()
    df1['pred'] = all_predictive_means
    df1['y'] = data[:,-1]
    #df1['lat'] = data[:,1]
    #df1['lon'] = data[:,0]
    df1['time'] = data[:,0]
    df1.to_csv('data/SEARD_uib_32lat_81lon.csv')
    #df1.to_csv('data/SEARD_u_uib_jan2000.csv')

print(np.mean(rmses), '±', np.std(rmses)/np.sqrt(10))
print(np.mean(nlpds), '±', np.std(nlpds)/np.sqrt(10))
