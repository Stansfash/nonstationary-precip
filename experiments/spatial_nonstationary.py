#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Khyber Spatial model

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
from gpytorch.kernels import ScaleKernel, RBFKernel
from models.multivariate_gibbs_kernel import MultivariateGibbsKernel
gpytorch.settings.cholesky_jitter(1e-4)

torch.manual_seed(13)
rng = np.random.default_rng(13)

def sqrt_mean_squared_error(test_y, predicted_mean):
    
    return torch.sqrt(torch.mean((test_y - predicted_mean)**2))

def negative_log_predictive_density(test_y, predicted_mean, predicted_var):
    
    # Vector of log-predictive density per test point    
    lpd = torch.distributions.Normal(predicted_mean,  torch.sqrt(predicted_var)).log_prob(test_y)
    # return the average
    return -torch.mean(lpd)

def load_khyber_data():
    
    fname = '/home/vr308/Desktop/Workspace/Kernel_Learning_Latent_GPs/data/khyber_jan.csv'
    data = pd.read_csv(fname)
    return data, torch.Tensor(np.array(data))[:,0:2], torch.Tensor(np.array(data)[:,-1])

class KhyberSpatial(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(MultivariateGibbsKernel(train_x,2))
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
    
class KhyberSpatialStat(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=2))
        
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
        
if __name__ == "__main__":
    
    data, x, y = load_khyber_data()
    
    with torch.no_grad():
        stdx, meanx = torch.std_mean(x, dim=-2)
        x = (x -  meanx) / stdx
        stdy, meany = torch.std_mean(y)
        y = (y - meany) / stdy
        
    num_train = math.ceil(80/100 * y.shape[0])
    idx = np.arange(0, y.shape[0], 1)
    rng.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    x_train = x[..., train_idx, :].detach()
    y_train = y[..., train_idx].detach()
    x_test = x[..., test_idx, :].detach()
    y_test = y[..., test_idx].detach()
     
    ## Training 
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    #model = KhyberSpatial(x_train, y_train, likelihood)
    model = KhyberSpatial(x_train, y_train, likelihood)

    model.train()
    likelihood.train()
    
    n_iter = 5000
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)  
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
            #print(model.base_covar_module.base_kernel.inducing_locations)
        optimizer.step()
        #model.base_covar_module.inducing_locations = model.covar_module.inducing_points
    
     ## Testing
     
    model.eval()
    likelihood.eval()
    
    ## Pred full
    
    pred_f = model(x)
    
    f_mean = pred_f.loc.detach()
    f_var = pred_f.covariance_matrix.diag().detach()
    
    #### Viz
    data = pd.read_csv('/home/vr308/Desktop/Workspace/Kernel_Learning_Latent_GPs/results/khyber_mean.csv')
    
    df = data.set_index(['lat', 'lon']) ## with ground truth tp
    
    df_recon = df.copy()
    #df_recon['tp'] = f_mean*stdy + meany   ## overwrite with predictions
    da = df_recon.to_xarray()
    
    plt.figure()
    ax = plt.subplot(projection=ccrs.PlateCarree())
    ax.set_extent([71, 83, 30, 38])
    g = da.tp.plot(vmin=0, vmax=7,cbar_kwargs={
            "label": "Precipitation [mm/day]",
            "extend": "neither", "pad": 0.10})
    g.cmap.set_under("white")
    g.colorbar.vmin = 0
    g.colorbar.vmax = 6
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.show()
    
    ### pred test
  
    pred_f_test = model(x_test)
    
    f_mean_test = pred_f_test.loc.detach()
    f_var_test = pred_f_test.covariance_matrix.diag().detach()
    
    ## metrics 
    
    rmse_test = sqrt_mean_squared_error(y, f_mean)
    nlpd = negative_log_predictive_density(y, f_mean, f_var)
    
    
    