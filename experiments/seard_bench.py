#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Modelling spatial precipitation at a single time-point across UIB

'''

import torch
import gpytorch
from math import floor
import pandas as pd
import numpy as np
import models.dgps as m
from gpytorch.kernels import RBFKernel, ScaleKernel, MaternKernel
from sklearn.utils import shuffle
import scipy.stats
from scipy.special import inv_boxcox
from utils.metrics import nlpd, rmse
import utils.dataprep as dp

filepath = 'data/uib_spatial.csv'
dataset = dp.download_data(filepath)

transform = 'whitening' # or 'boxcox'

## lists to save the metrics after each run
rmses = []
nlpds = []

for random_state in range(10):
    
    print('random_state = ', random_state)

    data = shuffle(dataset, random_state=random_state)

    if transform == 'whitening':    
        x_tr, y_tr, meanx, stdx, meany, stdy = dp.whitening_transform(data)
    elif transform == 'boxcox': 
        x_tr, y_tr, bc_param = dp.box_cox_transform(data)
        
    train_x, train_y, test_x, test_y = dp.train_test_split(x_tr, y_tr, 0.8)
    
    #### Model

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2))
    model = m.ExactGPModel(train_x, train_y, likelihood, kernel)
    
    # Initialize lengthscale and outputscale to mean of priors

    #### Training
    training_iter = 200

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()
    
    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
        model = model.cuda()
        likelihood = likelihood.cuda()

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
        if i%50 == 0:
            print('Iter %d/%d - Loss: %.3f  lengthscale: %s   noise: %.3f' 
                % (i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale,
                model.likelihood.noise.item()))
        optimizer.step()

    #### Metrics

    model.eval()
    with torch.no_grad():
        pred_y_test = likelihood(model(test_x)) 
        y_mean = pred_y_test.loc.detach()
        y_var = pred_y_test.covariance_matrix.diag().sqrt().detach()

    # Inverse transform predictions
    # pred_y_test_tr = torch.Tensor(inv_boxcox(pred_y_test, bc_param))
    #y_mean_tr = torch.Tensor(inv_boxcox(y_mean, bc_param))
    #y_var_tr = torch.Tensor(inv_boxcox(y_var + y_mean, bc_param,)) - y_mean_tr
    #test_y_tr = torch.Tensor(inv_boxcox(test_y, bc_param))

    ## Metrics
    rmse_test = rmse(y_mean, test_y, stdy)
    nlpd_test = nlpd(pred_y_test, test_y, stdy)

    print(f"RMSE: {rmse_test.item()}, NLPD: {nlpd_test.item()}")
    
    rmses.append(rmse_test.item())
    nlpds.append(nlpd_test.item())

    # df1 = pd.DataFrame()
    # df1['pred'] = y_mean
    # df1['std'] =  np.sqrt(y_var)
    # df1['y'] = data[:,-1]
    # df1['lat'] = data[:,1]
    # df1['lon'] = data[:,0]
    # #df1['time'] = data[:,0]
    # #df1.to_csv('data/softk_uib_32lat_81lon_.csv')
    # df1.to_csv('data/SEARD_transform_uib_jan2000.csv')
    
print(np.mean(rmses), '±', np.std(rmses)/np.sqrt(10))
print(np.mean(nlpds), '±', np.std(nlpds)/np.sqrt(10))
