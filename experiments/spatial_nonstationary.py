#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Khyber Spatial model

"""

import numpy as np
import math
import torch 
import gpytorch
import pandas as pd
import cartopy.crs as ccrs
import getopt
import sys
import matplotlib.pylab as plt
from gpytorch.kernels import ScaleKernel, RBFKernel
from models.multivariate_gibbs_kernel import MultivariateGibbsKernel
from models.gibbs import LogNormalPriorProcess, PositivePriorProcess, GibbsKernel, GibbsSafeScaleKernel
from models.nonstationary import DiagonalExactGP
from gpytorch.constraints import GreaterThan

from utils.config import BASE_SEED, EPSILON, DATASET_DIR
from utils.metrics import rmse, nlpd

rng = np.random.default_rng(BASE_SEED)
torch.manual_seed(BASE_SEED)
gpytorch.settings.cholesky_jitter(EPSILON)

## helper methods and classes 

def load_khyber_data():
        
    fname = str(DATASET_DIR) + '/khyber_spatial.csv'
    data = pd.read_csv(fname)
    return data, torch.Tensor(np.array(data))[:,0:2], torch.Tensor(np.array(data)[:,-1])

# class KhyberSpatial(gpytorch.models.ExactGP):
    
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = ScaleKernel(MultivariateGibbsKernel(train_x,2))
        
#     def forward(self, x):
#         mean = self.mean_module(x)
#         covar = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)
    
# class KhyberSpatialStat(gpytorch.models.ExactGP):
    
#     def __init__(self, train_x, train_y, likelihood):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ZeroMean()
#         self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=2))
        
#     def forward(self, x):
#         mean = self.mean_module(x)
#         covar = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean, covar)
    
def parse_args(argv):
    """Get command line arguments, set defaults and return a dict of settings."""

    args_dict = {
        ## File paths and logging
            'data'          : '/data/',             # relative path to data dir from root
            'root'          : '.',                 # path to root directory from which paths are relative
            'device'        : 'cpu',               # cpu or cuda
            'logdir'        : 'experiments/logs/', # relative path to logs from root
            'log_interval'  : 1,                   # how often to log train data
            'test_interval' : 1,                   # how often to log test metrics
            'plot_interval' : 10,                  # how often to generate plots
            'name'          : None,                # name for experiment
            'test_type'      : 'random',           # random or 'censored'
        
        ## Training options
            'model'         : 'DiagonalML',        # 'DiagonalML', 'FullML', 'DiagonalGibbs'
            'inference'     : 'exact',             # 'exact' or 'sparse'
            'train_percent' : '80',                # percentage of data to use for training
            'lr'            : '1e-1',              # learning rate          
            'max_iters'     : 1000,
            'threshold'     : 1e-6,                # improvement after which to stop
            'M'             : 1000,                  # Number of inducing points (sparse regression only)
            'prior_scale'   : 1,                    # initial value for the prior outputscale (same for both dims)
            'prior_ell'     : 1.5,                  # Initial value for the prior's lengthscale (same for both dims)
            'prior_mean'    : 0.3,                  # Initial value for the prior's mean (same for both dims)
            'noise'         : 0,                     # 0 for optimised noise, else fixed through training
            'scale'         : 1 ,                   # 0 for optimised output scale, else fixed through training
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
    
    #### Loading and prep data
    
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
    
    ## Initialising model-set up and prior settings
    
    prior = LogNormalPriorProcess(input_dim=2).to(device)
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
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=GreaterThan(0.001))
    model = DiagonalExactGP(x_train, y_train, likelihood, prior, num_dim=2).to(device)
    
    # noise hyper
    if float(args['noise']) > 0:
        model.likelihood.noise = float(args['noise'])
        for p in model.likelihood.noise_covar.parameters():
            p.requires_grad = False
    # outputscale hyper
    if float(args['scale']) > 0:
        model.covar_module.outputscale = float(args['scale'])
        model.covar_module._parameters['raw_outputscale'].requires_grad = False
           
    model.train()
    likelihood.train()
    
    n_iter = 2000
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(x_train)
        loss = -mll(output, y_train)
        loss.backward()
        losses.append(loss.item())
        if i%50 == 0:
            print('Iter %d/%d - Loss: %.3f  amplitude: %.3f noise: %.3f' % (
                i + 1, n_iter, loss.item(),
                model.covar_module.outputscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
    
     ## Testing
     
    model.eval()
    likelihood.eval()
    
    ## Pred full
    
    pred_f = model.predict(x)
    
    f_mean = pred_f.loc.detach()
    f_var = pred_f.covariance_matrix.diag().detach()
    
    #### Viz
    
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
    
    
    