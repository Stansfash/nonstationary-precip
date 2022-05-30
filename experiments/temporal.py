#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Khyber time-series at a single point

"""
import numpy as np
import math
import torch 
import gpytorch
import os
import gpytorch
import pandas as pd
import matplotlib
import cartopy.crs as ccrs
import matplotlib.pylab as plt
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
from models.multivariate_gibbs_kernel import MultivariateGibbsKernel
gpytorch.settings.cholesky_jitter(1e-5)

from utils.config import BASE_SEED
from utils.config import DATASET_DIR

rng = np.random.default_rng(BASE_SEED)
torch.manual_seed(BASE_SEED)

def load_khyber_timeseries():
    
    fname = str(DATASET_DIR) + '/khyber_time_series.csv'
    data = pd.read_csv(fname)
    return torch.Tensor(np.array(data))[:,0], torch.Tensor(np.array(data)[:,-1])

class KhyberTemporal(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel()*PeriodicKernel())
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    

if __name__ == "__main__":
    
    x, y = load_khyber_timeseries()
    
    with torch.no_grad():
        stdx, meanx = torch.std_mean(x)
        x_norm = (x -  meanx) / stdx
        stdy, meany = torch.std_mean(y)
        y_norm = (y - meany) / stdy
        
    num_train = math.ceil(80/100 * y.shape[0])
    idx = np.arange(0, y.shape[0], 1)
    rng.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    x_train = x_norm[..., train_idx].detach()
    y_train = y_norm[..., train_idx].detach()
    x_test = x_norm[..., test_idx].detach()
    y_test = y_norm[..., test_idx].detach()
    
     ## Training 
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = KhyberTemporal(x_train, y_train, likelihood)

    model.train()
    likelihood.train()
    
    n_iter = 5000
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        losses.append(loss.item())
        if i%5 == 0:
            print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                i + 1, n_iter, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    
     ## Testing
     
    model.eval()
    likelihood.eval()
    
    ## Pred full
    
    pred_f = model(x)
    
    f_mean = pred_f.loc.detach()
    f_var = pred_f.covariance_matrix.diag().detach()
    
    ## viz
    f_true = f_mean*stdy + meany
    f_sigma = torch.sqrt(f_var)*stdy
        
    plt.plot(x,y,label='Observations')
    plt.plot(x,f_true,label='Posterior mean')
    plt.fill_between(x, y - 2*f_sigma, y + 2*f_sigma, color='orange', alpha=0.5)
    plt.legend(fontsize='small')
    plt.title('Temporal Model')
    
