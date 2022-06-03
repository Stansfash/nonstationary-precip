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

filepath = 'data/uib_jan2000_tp.csv'
# filepath = 'data/uib_lat32_lon81_tp.csv'
rmses = []
nlpds = []

# Kernels
kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
'''
# kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=1, active_dims=[0]) * gpytorch.kernels.PeriodicKernel(active_dims=[0]))
# Kernels for timeseries modelling
kernel1 = ScaleKernel(RBFKernel()*PeriodicKernel())
kernel2 = ScaleKernel(MaternKernel(0.5)*PeriodicKernel())
'''

for random_state in range(10):
    print('random_state = ', random_state)

    df = pd.read_csv(filepath)
    data_df = shuffle(df, random_state=random_state)
    data = torch.Tensor(data_df.values)

    X = data[:,:-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]
    y_tr, bc_param = scipy.stats.boxcox(y + 0.001)
    y_tr = torch.Tensor(y_tr)

    stdy_tr, _ = torch.std_mean(y_tr)
    stdy, _ = torch.std_mean(y_tr)

    train_n = int(floor(0.75 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y_tr[:train_n].contiguous()

    test_x = X[train_n:, :].contiguous()
    test_y = y_tr[train_n:].contiguous()

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
    rmses.append(rmse_test.item())
    nlpds.append(nlpd_test.item())

    '''

    with torch.no_grad():
        trained_pred_dist = likelihood(model(X))
        all_predictive_means = trained_pred_dist.mean

    df1 = pd.DataFrame()
    df1['pred'] = all_predictive_means
    df1['y'] = data[:,-1]
    #df1['lat'] = data[:,1]
    #df1['lon'] = data[:,0]
    df1['time'] = data[:,0]
    df1.to_csv('data/softk_uib_32lat_81lon_.csv')
    df1.to_csv('data/SEARD_transform_uib_jan2000.csv')
'''
print(np.mean(rmses), '±', np.std(rmses)/np.sqrt(10))
print(np.mean(nlpds), '±', np.std(nlpds)/np.sqrt(10))
