#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Khyber Spatio-Temporal model

"""

import numpy as np
import math
import torch 
import gpytorch
import pandas as pd
import cartopy.crs as ccrs
import getopt
import sys
import pymc3 as pm
import matplotlib.pylab as plt
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel, InducingPointKernel
from models.gibbs_kernels import LogNormalPriorProcess, PositivePriorProcess, GibbsKernel, GibbsSafeScaleKernel
from models.spatio_temporal_models import SpatioTemporal_Stationary, SparseSpatioTemporal_Nonstationary
from gpytorch.constraints import GreaterThan

from utils.config import BASE_SEED, EPSILON, DATASET_DIR
from utils.metrics import rmse, nlpd, get_trainable_param_names

rng = np.random.default_rng(BASE_SEED+2)
torch.manual_seed(BASE_SEED+5)
gpytorch.settings.cholesky_jitter(EPSILON)

## helper methods and classes 

def load_khyber_data():
        
    fname = str(DATASET_DIR) + '/khyber_spatio_temporal.csv'
    data = pd.read_csv(fname, index_col=0)
    return data, torch.Tensor(np.array(data))[:,0:3], torch.Tensor(np.array(data)[:,-1])

def load_train_test(test_year):
        
    data, x, y = load_khyber_data()
    train_test_data = data[data['time'] < test_year+1]
    x, y = torch.Tensor(np.array(train_test_data))[:,0:3], torch.Tensor(np.array(train_test_data)[:,-1])
    
    with torch.no_grad():
        stdx, meanx = torch.std_mean(x, dim=-2)
        x_norm = (x -  meanx) / stdx
        stdy, meany = torch.std_mean(y)
        y_norm = (y - meany) / stdy
        
    split_idx = len(np.where(train_test_data['time'] < test_year)[0])
    test_idx = np.where(train_test_data['time'].astype('int') == test_year)[0]
    x_train, y_train = x_norm[0:split_idx], y_norm[0:split_idx]
    x_test, y_test = x_norm[split_idx:], y_norm[split_idx:]
    return train_test_data, x_train, y_train, x_test, y_test, meany, stdy
    
def parse_args(argv):
    
    """Get command line arguments, set defaults and return a dict of settings."""

    args_dict = {
        ## File paths and logging
        
            'data'          : '/data/',            # relative path to data dir from root
            'device'        : 'cpu',               # cpu or cuda
        
        ## Training options
        
            'model'         : 'Non-Stationary',        # 'Stationary' / 'Non-stationary'
            'lr'            : '1e-2',              # learning rate          
            'max_iters'     : 1000,
            'threshold'     : 1e-6,                # improvement after which to stop
            'M'             : 500,                # Number of inducing points (sparse regression only)
            'prior_scale'   : 1,                   # initial value for the prior outputscale (same for both dims)
            'prior_ell'     : 1.3,                 # Initial value for the prior's lengthscale (same for both dims)
            'prior_mean'    : 0.3,                 # Initial value for the prior's mean (same for both dims)
            'noise'         : 0.0,                 # 0 for optimised noise, else fixed through training
            'scale'         : 0 ,                  # 0 for optimised output scale, else fixed through training
    }

    try:
        opts, _ = getopt.getopt(argv, '', [name + '=' for name in args_dict.keys()])
    except getopt.GetoptError:
        helpstr = 'Check options. Permitted long options are '
        print(helpstr, args_dict.keys())

    for opt, arg in opts:
        opt_name = opt[2:] # remove initial --
        args_dict[opt_name] = arg
    
    return args_dict
        
if __name__ == "__main__":
    
    args = parse_args(sys.argv[1:])
    device = args['device']
    max_cg_iterations = 4000
    
    #### Loading and prep data
    test_year = 2004
    
    data, x_train, y_train, x_test, y_test, meany, stdy = load_train_test(test_year)
    
    num_inducing = 500  
    z = torch.tensor(pm.gp.util.kmeans_inducing_points(num_inducing, np.array(x_train)))

    ## Initialising model-set up and prior settings
    
    if args['model'] == 'Non-stationary':
        
        prior = LogNormalPriorProcess(input_dim=2, active_dims=(0,1)).to(device)
            #### change the prior settings here if desired
        prior.covar_module.outputscale = float(args['prior_scale']) * torch.ones_like(prior.covar_module.outputscale)
        prior.covar_module.base_kernel.lengthscale = float(args['prior_ell']) * torch.ones_like(
                                                                prior.covar_module.base_kernel.lengthscale)
        prior.mean_module.constant = torch.nn.Parameter(
            math.log(float(args['prior_mean'])) * torch.ones_like(prior.mean_module.constant)
                                )
     
        for p in prior.parameters():
            p.requires_grad = False
        
    ## Training 
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if args['model'] == 'Stationary':
        model = SpatioTemporal_Stationary(x_train, y_train, likelihood, z) 
    else:
        model = SparseSpatioTemporal_Nonstationary(x_train, y_train, likelihood, prior, z, num_dim=2) 

    # #noise hyper
    # if float(args['noise']) > 0:
    #     model.likelihood.noise = float(args['noise'])
    #     for p in model.likelihood.noise_covar.parameters():
    #         p.requires_grad = False
    # # outputscale hyper
    # if float(args['scale']) > 0:
    #     model.covar_module.outputscale = float(args['scale'])
    #     model.covar_module._parameters['raw_outputscale'].requires_grad = False
           
    model.train()
    likelihood.train()
    
    n_iter = 20
     
    optimizer = torch.optim.Adam(model.parameters(), lr=0.015)  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        with gpytorch.settings.max_cg_iterations(max_cg_iterations):
            output = model(x_train)
            loss = -mll(output, y_train)
        loss.backward()
        losses.append(loss.item())
        if i%50 == 0:
            print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                i + 1, n_iter, loss.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    
     ## Testing
     
    model.eval()
    likelihood.eval()
    
    if args['model'] == 'Non-stationary':
        pred_y_test = likelihood(model.predict(x_test)) 
    else:
        pred_y_test = likelihood(model(x_test)) 
     
    y_mean = pred_y_test.loc.detach()
    y_var = pred_y_test.covariance_matrix.diag().sqrt().detach()
    
    ## Metrics
    rmse_test = rmse(y_mean, y_test, stdy)
    nlpd_test = nlpd(pred_y_test, y_test, stdy)
    
    print('RMSE test =  ' + str(rmse_test))
    print('NLPD test = ' + str(nlpd_test))
    
    ### Pred full
    
    # if args['model'] == 'Non-stationary':
    #     pred_f = likelihood(model.predict(x_norm))
    # else:
    #     pred_f = likelihood(model(x))
    
    # f_mean = pred_f.loc.detach()
    # f_var = pred_f.covariance_matrix.diag().detach()
    
    #### Viz
    
    # #data['time'] = data.time.astype('int')
    # df = data.set_index(['lat', 'lon', 'time']) ## with ground truth tp
    # #df_recon['tp'] = f_mean*stdy + meany   ## overwrite with predictions

    # years = [2000, 2001, 2002, 2003, 2004]
    # plt.figure(figsize=(10,3))
    
    # for i in [0,1,2,3]:
        
    #     sub = df.xs(years[i], level=2)
    #     da = sub.to_xarray()
        
    #     ax = plt.subplot(1,4,i+1,projection=ccrs.PlateCarree())
    #     ax.set_extent([71, 83, 30, 38])
    #     g = da.tp.plot(vmin=0, vmax=7,cbar_kwargs={
    #             "label": "Precipitation [mm/day]",
    #             "extend": "neither", "pad": 0.10})
    #     g.cmap.set_under("white")
    #     g.colorbar.vmin = 0
    #     g.colorbar.vmax = 6
    #     gl = ax.gridlines(draw_labels=True)
    #     gl.top_labels = False
    #     gl.right_labels = False
    #     ax.set_xlabel("Longitude")
    #     ax.set_ylabel("Latitude")
    #     plt.show()
   
    