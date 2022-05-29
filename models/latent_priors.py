#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Latent Model Priors 

"""

import gpytorch 
import gpytorch.priors as priors
import torch

global jitter 

jitter = 1e-5

class LearnedSoftPlus(torch.nn.Module):
    def __init__(self, init_beta=1.0, threshold=20):
        super().__init__()
        # keep beta > 0
        self.log_beta = torch.nn.Parameter(torch.tensor(float(init_beta)).log())
        self.threshold = 20
    def forward(self, x):
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where(beta_x < 20, torch.log1p(beta_x.exp()) / beta, x)

class MatrixVariateNormalPrior(gpytorch.priors.MultivariateNormalPrior):
    
    ''' 
    Matrix Normal Prior for a N x D matrix of reals
    
    :param loc: Matrix of size N x D 
    :param row_covariance_matrix: Matrix of size N x N 
    :param column_covariance_matrix: Matrix of size D x D 
    
    '''
    def __init__(self, loc, row_covariance_matrix, column_covariance_matrix):
        
        n =  row_covariance_matrix.shape[0] ## row dim
        d = column_covariance_matrix.shape[0] ## col dim

        # the parent class is initialised with the vectorisation of loc (same size as the data over which 
        # the prior is defined)
        vec_loc = loc.flatten()
        kron_cov = torch.kron(row_covariance_matrix+torch.eye(n)*jitter, column_covariance_matrix)
        kron_cov_inv = torch.kron(column_covariance_matrix.inverse(), (row_covariance_matrix+torch.eye(n)*jitter).inverse())
        
        super().__init__(loc=vec_loc.double(), covariance_matrix=kron_cov.double())
        
        #self.loc = loc 
        self.row_covariance_matrix = row_covariance_matrix
        self.col_covariance_matrix = column_covariance_matrix
        self.vec_loc = vec_loc
        self.kron_cov = kron_cov
        self.kron_cov_inv = kron_cov_inv
        self.n = n
        self.d = d
        
    def sample_n(self, num_samples):
        vec_sample = super().sample_n(num_samples).T
        return vec_sample.reshape(self.n, self.d)
    
    def log_prob(self, x):
       return super().log_prob(x.T.flatten())
        
class LatentGpPrior(gpytorch.priors.MultivariateNormalPrior):
    
    ''' 
    GP Prior (1d) for the lengthscale process/amplitude process of the product Gibbs kernel
    
    '''
    def __init__(self, input_dim, X, sig_f, ls, kernel_func=None):
        
            mean_module = gpytorch.means.ZeroMean()
            if kernel_func is None:
               covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=input_dim))
               #covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.PeriodicKernel(ard_num_dims=input_dim))
               
            covar_module.outputscale = sig_f
            covar_module.base_kernel.lengthscale = ls
            
            covar_matrix = covar_module(X).evaluate().detach() + torch.eye(X.shape[0])*jitter
            super().__init__(loc=mean_module(X), covariance_matrix=covar_matrix)
            
            self.X = X
            self.mean_module = mean_module
            self.covar_module = covar_module
            self.covar_matrix = covar_matrix
            self.covar_module.base_kernel.raw_lengthscale.requires_grad_(False)
            self.covar_module.raw_outputscale.requires_grad_(False)
            #self.covar_module.outputscale = 2
            #self.covar_module.base_kernel.lengthscale = 1
            
    def forward(self, X):
        
            mean_x = self.mean_module(X).detach()
            covar_x = self.covar_matrix
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    
if __name__ == "__main__":

    #     x = torch.linspace(-10, 10, 1000)
    #     l = LatentGpPrior(1, torch.linspace(-10, 10, 1000))
    import numpy as np
    from gpytorch.kernels import MaternKernel
    
    num_grid = 30 
    X = np.linspace(-2, 2, num_grid)
    jitter = np.eye(num_grid**2)*1e-6
    X_grid = np.meshgrid(X, X)
    X = torch.Tensor(np.vstack((X_grid[0].flatten(), X_grid[1].flatten())).T)

    row_covar_kernel = MaternKernel(nu=2.5, ard_num_dims = 2)
    #col_covar_kernel = torch.eye(self.d)
       
    ## we place a joint matrix variate prior on the sigma matrix 
    loc = torch.zeros(900, 2)        # N x D
    
    row_covar = row_covar_kernel(X).evaluate() # N x N 
    col_covar =  torch.eye(2) # D x D
    
    sigma_matrix_prior = MatrixVariateNormalPrior(loc, row_covariance_matrix=row_covar, column_covariance_matrix=col_covar)    
    
#     ss=sigma_matrix_prior.sample_n(1)
#     #self.batch_log_ls_func = [LatentGpPrior(se
    
#     fig, axs = plt.subplots(20, 7)
#     fig.suptitle('Reconstructions')
#     k = 1
#     for i in range(20):
#         for j in range(7):
#             k += 1
#             axs[i, j].imshow(sigmas[k])
#             axs[i, j].axis('off')
#     plt.tight_layout()
#     plt.suptitle('Train Reconstructions [10% missing pixels]', fontsize='small')

        
