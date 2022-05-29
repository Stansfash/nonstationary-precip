#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Khyber Spatio-temporal dataset analysis

"""

import gpytorch 
import torch
import numpy as np
import matplotlib.pylab as plt
from utils.data_tools import load_khyber_data
from gpytorch.kernels import InducingPointKernel
from kernels.latent_priors import LatentGpPrior
gpytorch.settings.cholesky_jitter(1e-5)

class DenseRBFModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood, rank=1, interval1=3.):
        super(DenseRBFModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module_projection = gpytorch.kernels.RBFKernel(
            ard_num_dims=rank,
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1),
        )
        proj = torch.randn(train_x.shape[-1], rank)
        proj /= (proj**2).sum()
        proj.detach_().requires_grad_()
        self.register_parameter(
            "projection", torch.nn.Parameter(proj)
        )
        self.covar_module_ard = gpytorch.kernels.RBFKernel(
            ard_num_dims=train_x.shape[-1],
            lengthscale_constraint=gpytorch.constraints.Interval(0.01, interval1)
        )

    def forward(self, x):
        proj_x = x.matmul(self.projection)
        mean_x = self.mean_module(x)
        # this kernel is exp(-l_1^2 (x - x')P P^T(x - x') - l_2^2 (x - x')D(x - x'))
        # because we compute the product elementwise
        covar_x = self.covar_module_projection(proj_x) * self.covar_module_ard(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class DiagonalRBFModel(gpytorch.models.ExactGP):
    
    def __init__(self, train_x, train_y, likelihood):
        super(DiagonalRBFModel, self).__init__(train_x, train_y, likelihood)
        
        self.mean_module = gpytorch.means.ConstantMean()
        # Optionally specify ard_num_dims in RBFKernel to get non-constant diagonal matrix D.
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(active_dims=(0))*gpytorch.kernels.RBFKernel(ard_num_dims=2, active_dims=(1,2)))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":

    X_raw, X_norm, Y_raw, Y_norm, bc = load_khyber_data()
    
    # initialize likelihood and model
    train_range = np.arange(700)
    test_range = np.arange(700,1000,1)
    
    X_train = X[train_range].to(torch.float32)
    Y_train = torch.Tensor(Y[train_range])
    X_test = X[test_range].to(torch.float32)
    Y_test = torch.Tensor(Y[test_range])
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    
    #likelihood.noise = 1e-3  # Some small value, but don't make it too small or numerical performance will suffer. I recommend 1e-4.
    #likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.
    
    model= DiagonalRBFModel(X_train, Y_train, likelihood)
    #model = DenseRBFModel(X_train, Y_train.flatten(), likelihood)
    #model.covar_module_projection.raw_lengthscale.requires_grad = False
    #model.covar_module_projection.lengthscale = torch.tensor([1.0])

    # Find optimal model hyperparameters   
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Includes GaussianLikelihood parameters
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    training_iter = 3000
    losses = []
    
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X_train)
        # Calc loss and backprop gradients
        loss = -mll(output, Y_train)
        loss.backward()
        losses.append(loss.detach().item())
        if i%500 == 0: 
            print('Iter %d/%d - Loss: %.3f  lengthscale: %.3f noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.kernels[1].lengthscale[0,0].item(),
                model.likelihood.noise.item()
            ))
            #print('Ls: ' + str(model.covar_module_ard.lengthscale))
        optimizer.step()
        
    # # #### Plot losses
    
    plt.figure()
    losses = [x.cpu().detach().item() for x in losses]
    plt.plot(losses)
        
    
    # # # #### Prediction 
    
    with torch.no_grad():
        model.eval()
        likelihood.eval()
        y_preds = likelihood(model(X_test))
        
    
    fig = plt.figure(figsize=(10,4))
    j = 0 
    time_steps = np.unique(X_test[:,0])

    min_p = Y_test.min()
    max_p =  Y_test.max()
    
    levels = np.linspace(min_p, 13)
    
    for i in time_steps[0:8]:
        
        plt.subplot(2,4,j+1)
     
        j = j+1
        # slice by time-step
        df = df1[np.float32(df1.time) == i]
        
        m = Basemap(projection='lcc', resolution='l', 
                  lat_0=35.0, lon_0=73.5, width=0.4e+6, height=(1e+6)*0.4)
        m.etopo()
        # draw parallels.
        parallels = np.arange(33,40,1)
        m.drawparallels(parallels,labels=[1,0,0,0],fontsize=6)
        # draw meridians
        meridians = np.arange(72,77,2.5)
        m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=6)
        
        ## scatter 
        m.scatter(df['lon'].values, df['lat'].values, latlon=True,
        marker='x', color='r', s=2)
        
        points = np.vstack((df['lat'], df['lon'])).T
        values = y_preds.loc
        
        x1 = np.linspace(72,76,50)   # lons
        x2 = np.linspace(34,36,50)   # lats
        
        lon_grid, lats_grid = np.meshgrid(x1, x2) 
        grid_z = griddata(points, values, (lats_grid, lon_grid), method='cubic')
        # plot an interpolated surface
        cf = m.contourf(lon_grid, lats_grid, grid_z, latlon=True,alpha=0.7, levels=levels, cmap='RdBu')
        cbar = m.colorbar()
        cbar.set_ticks(np.linspace(min_p, 13,5))
        cbar.ax.tick_params(labelsize=6)
        plt.title('t = ' + str(j), fontsize='small')

    
    # # # ##### Visualisation 
    
    plt.figure()
    plt.contourf(xx, yy, f_preds.loc.reshape(140,100).detach().cpu(), levels=50, cmap=plt.get_cmap('jet'))
    plt.scatter(X_train[:,0].cpu().numpy(), X_train[:,1].cpu().numpy(), c='k', marker='x')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    ############################
    
    prior = LatentGpPrior(3, x)
    gpk = GibbsProductKernel(x, 3, prior)
    K_gibbs = gpk.forward(x, x)
        
    f = gpytorch.distributions.MultivariateNormal(torch.zeros(len(x)), covariance_matrix=K_gibbs + torch.eye(len(x))*1e-5).sample()

    ### Preparing training and test data
    
    train_index = np.random.randint(0,500,100)
    train_x = x[train_index]
    test_f = f 
    train_y = f[train_index] + 0.4*torch.randn(len(train_index))
    
    Z_init = torch.linspace(-10,10,5).double()
    
    normal_prior = LatentGpPrior(1, train_x)
    sgp_prior = LatentGpPrior(1, Z_init)
    
    ## Training 
    
    likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
    model = ExactGPModel(train_x, train_y, likelihood, normal_prior).double()
    
    #likelihood.noise = 1e-4  # Some small value, but don't make it too small or numerical performance will suffer. I recommend 1e-4.
    #likelihood.noise_covar.raw_noise.requires_grad_(False)  # Mark that we don't want to train the noise.

    #model = SGPRModel(train_x, train_y, likelihood, sgp_prior)
    
    #plt.figure()
    #plt.plot(train_x, train_y, 'bo')
    
    model.train()
    likelihood.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    n_iter = 5000
    
    losses = []
    for i in range(n_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        losses.append(loss.item())
        print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
            i + 1, n_iter, loss.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    
     ## Testing
     
    model.eval()
    likelihood.eval()
    
    pred_f = model(x)
    
    f_mean = pred_f.loc.detach()
    f_var = pred_f.covariance_matrix.diag().detach()
    
    ls_dist=model.covar_module.get_conditional_log_lengthscale_dist(x)
    #ls_sparse_dist = model.base_covar_module.get_conditional_log_lengthscale_dist(x)
    ls_mean = torch.exp(ls_dist.loc.detach())
    
    ## Visualisation
    
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.plot(x,f, label='True f')
    plt.plot(x, f_mean, label='Predicted f')
    plt.fill_between(x, f_mean - 1.96*np.sqrt(f_var), f_mean + 1.96*np.sqrt(f_var), alpha=0.6, color='orange')
    plt.scatter(train_x, train_y, c='k', marker='x')
    plt.legend(fontsize='small')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('1D regression', fontsize='small')
    
    plt.subplot(122)
    plt.plot(x, gpk.lengthscale.detach(), label='True lengthscale', c='r')
    plt.plot(x, ls_mean, label='MAP Est.', c='g')
    plt.xlabel('x')
    plt.ylabel('lengthscale')
    plt.legend(fontsize='small')
    plt.title('Lengthscale Process', fontsize='small')
    
    plt.figure(figsize=(10,4))
    plt.subplot(121)
    plt.imshow(K_gibbs.detach())
    plt.title('GT Kernel')
    plt.xticks([])
    plt.yticks([])
    
    K_learnt = model.covar_module.forward(x,x).detach()
    plt.subplot(122)
    plt.imshow(K_learnt.detach())
    plt.title('Learnt Kernel')
    plt.xticks([])
    plt.yticks([])
    