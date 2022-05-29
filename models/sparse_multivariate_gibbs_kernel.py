#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sparse Multivariate Gibbs kernel

"""

import gpytorch 
import torch
import pymc3 as pm
from kernels.latent_priors import MatrixVariateNormalPrior
from gpytorch.kernels import RBFKernel, ScaleKernel
from utils.metrics import get_trainable_param_names
import numpy as np

global jitter
jitter = 1e-5
softplus = torch.nn.Softplus()

class SparseMultivariateGibbsKernel(gpytorch.kernels.Kernel):
    
    """
    Multivariate Gibbs kernel -> GP prior combined with Paciorek and Scheverish [2003] form.
    
    """
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, Z, input_dim, Z_init, **kwargs):
        super().__init__(**kwargs)
        
        self.inducing_locations = Z
        self.d = input_dim
        self.m = self.inducing_locations.shape[0]
        
        if input_dim == 1: 
            raise ValueError('Use gibbs 1d kernel for dim 1')
        
        else: # input_dim > 1
        
            ### Setting prior parameters based on kernels for smoothly evolving 
            ### sigma matrix
            
            self.row_covar_kernel = ScaleKernel(RBFKernel(ard_num_dims = self.d, lengthscale=torch.Tensor([1.3,1.1])))
            #col_covar_kernel = torch.eye(self.d)
            self.row_covar_kernel.requires_grad_(False)
           
            ## we place a joint matrix variate prior on the sigma matrix 
            self.loc = torch.zeros(self.m, self.d) # M x D
            
            self.row_covar = self.row_covar_kernel(self.inducing_locations).evaluate() # M x M 
            
            self.static_row_covar = self.row_covar_kernel(Z_init).evaluate()
            Z_init.requires_grad_(False)
            
            self.col_covar = torch.Tensor([[1,0.0],[0.0,1]]) # D x D
            
            self.H_matrix_prior = MatrixVariateNormalPrior(self.loc, row_covariance_matrix=self.static_row_covar, column_covariance_matrix=self.col_covar)    
            
            H_init = self.H_matrix_prior.sample_n(1)
            self.register_parameter(name='H', parameter=torch.nn.Parameter(H_init.to(torch.float32)))
            self.register_prior('prior_H', self.H_matrix_prior, 'H')
            
            D_init = torch.diag(torch.randn(2))
            self.register_parameter(name='D', parameter=torch.nn.Parameter(D_init.to(torch.float32)))
            
    def expectation_conditional_matrix_variate_dist(self, x_star):
                
        kron_cov_inv = torch.kron(self.col_covar.inverse(), (self.row_covar+torch.eye(self.m)*jitter).inverse())
    
        #kron_cov_inv = self.prior_H.kron_cov_inv
        
        row_cross_covar = self.row_covar_kernel(x_star, self.inducing_locations).evaluate() # N* x N 
        col_cross_covar = self.col_covar
        
        cross_covar = torch.kron(col_cross_covar, row_cross_covar)
                
        matrix_cond_mean_vec = torch.matmul(torch.matmul(cross_covar, kron_cov_inv), self.H.T.flatten())

        return matrix_cond_mean_vec.reshape(self.d, len(x_star)).T # hack to achieve column-wise reshape
                
    def forward(self, x1, x2, diag=False, **params):
                
        if torch.equal(x1, x2):
            
            try: 
                assert(len(x1) == self.H.shape[0])
            
                Hx = self.H.detach()
                N1 = N2 = len(x1)
                
            except AssertionError: 
                
                # this means the size of the lengthscale and size of the inputs don't match but the two 
                # inputs are the same -> kernel on test inputs K_{**} / K_{nn} is being computed.
                
                ########### use current inducing locations to compute the conditional ################
                Hx = self.expectation_conditional_matrix_variate_dist(x1).detach()
                N1 = N2 = len(x1)
                
            # x1 is NxD
            
            sigma_xs = softplus(torch.Tensor([np.outer(x,x.T)**2 for x in Hx])) + self.D**2
            
            ## expanding to get dimensions NxN
            self.sigma_matrix_i = torch.einsum('ijkl->jikl', sigma_xs.expand(N1,N1,self.d,self.d)) # each row has a DxD matrix
            self.sigma_matrix_j = torch.einsum('ijkl->jikl', self.sigma_matrix_i)
            
            sigma_dets_matrix = torch.det(self.sigma_matrix_i).pow(0.25) 
            det_product = torch.mul(sigma_dets_matrix, sigma_dets_matrix.T) ## N x N
        
        else:
            
            ## x1 and x2 are different data blocks -- computing the test-train cross covariance
            print('Detected x1 and x2 are different - need cross covariance computation')
                    
            if x1.shape[0] == self.H.shape[0]:
                
                x_star = x2
                Hx1 = self.H.detach() 
                Hx2 = self.expectation_conditional_matrix_variate_dist(x_star).detach()
                
            elif x2.shape[0] == self.H.shape[0]:
                
                x_star = x1
                Hx2 = self.H.detach()
                Hx1 = self.expectation_conditional_matrix_variate_dist(x_star).detach()
                            
            N1 = len(x1)
            N2 = len(x2)
            
            sigma_x1 = softplus(torch.Tensor([np.outer(x,x.T)**2 for x in Hx1])) + self.D**2
            sigma_x2 = softplus(torch.Tensor([np.outer(x,x.T)**2 for x in Hx2]))  + self.D**2

            ## expanding to get dimensions NxN
            
            sigma_matrix_i = torch.einsum('ijkl->jikl', sigma_x1.expand(N1,N1,self.d,self.d)) # each row has a DxD matrix
            self.sigma_matrix_i = sigma_matrix_i[:, 0:N2, :,:]
            self.sigma_matrix_j = sigma_x2
            
            sigma_dets_matrix_i = torch.det(self.sigma_matrix_i).pow(0.25) 
            sigma_dets_matrix_j = torch.det(self.sigma_matrix_j).pow(0.25) 
            det_product = torch.mul(sigma_dets_matrix_i, sigma_dets_matrix_j) ## N1 x N2
            
        avg_kernel_matrix = (self.sigma_matrix_i + self.sigma_matrix_j)/2
        avg_kernel_det = torch.det(avg_kernel_matrix).pow(-0.5) 
        prefactor = torch.mul(det_product, avg_kernel_det) ## N1 x N2
        
        self.sig_inv = torch.inverse(avg_kernel_matrix + jitter*torch.eye(self.d)) ## N1 x N2 x D x D
        self.diff = (x1.unsqueeze(-2) - x2.unsqueeze(-3)) 
        first_prod = torch.matmul(self.diff.unsqueeze(-2), self.sig_inv)
        final_prod = torch.matmul(first_prod, self.diff.unsqueeze(-1)).reshape(N1,N2) ## N1xN2

        return torch.mul(prefactor, torch.exp(-final_prod)) ## N1 x N2
    
# if __name__ == "__main__":
    
#     import matplotlib.pyplot as plt

#     num_grid = 30
#     X = np.linspace(2, 3, num_grid)
#     #jitter = np.eye(num_grid**2)*1e-6
#     X_grid = np.meshgrid(X, X)
#     X = torch.Tensor(np.vstack((X_grid[0].flatten(), X_grid[1].flatten())).T)
    
#     mgk = SparseMultivariateGibbsKernel(X, input_dim=2)
#     K = mgk.forward(X, X).detach()
#     #f = pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K + np.eye(num_grid**2)*1e-5, shape=(num_grid**2,)).random(size=1).T
#     plt.matshow(K.detach())