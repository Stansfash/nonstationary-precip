#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multivariate Gibbs Kernel -> Matrix GP prior with Paciorek and Scheverish form.

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

class MultivariateGibbsKernel(gpytorch.kernels.Kernel):
    
    """
    Multivariate Gibbs kernel -> GP prior combined with Paciorek and Scheverish [2003] form.
    """
    is_stationary = False

    # We will register the parameter when initializing the kernel
    def __init__(self, x, input_dim, **kwargs):
        super().__init__(**kwargs)
        
        self.x = x
        self.n = len(x)
        self.d = input_dim
        
        #if Z is not None:
        #    self.inducing
        
        if input_dim == 1: 
            raise ValueError('Use gibbs 1d kernel for dim 1')
        
        else: # input_dim > 1
        
            ### Setting prior parameters based on kernels for smoothly evolving 
            ### sigma matrix
            
            self.row_covar_kernel = RBFKernel(ard_num_dims = self.d, lengthscale=torch.Tensor([0.2,0.2]))
            #col_covar_kernel = torch.eye(self.d)
            self.row_covar_kernel.requires_grad_(False)
           
            ## we place a joint matrix variate prior on the sigma matrix 
            self.loc = torch.zeros(self.n, self.d) # N x D
            
            self.row_covar = self.row_covar_kernel(self.x).evaluate() # N x N 
            self.col_covar = torch.Tensor([[5,0],[0,5]]) # D x D
            
            self.H_matrix_prior = MatrixVariateNormalPrior(self.loc, row_covariance_matrix=self.row_covar, column_covariance_matrix=self.col_covar)    
            
            H_init = self.H_matrix_prior.sample_n(1)
            self.register_parameter(name='H', parameter=torch.nn.Parameter(H_init.to(torch.float32)))
            self.register_prior('prior_H', self.H_matrix_prior, 'H')
            
            D_init = torch.diag(torch.randn(2))
            self.register_parameter(name='D', parameter=torch.nn.Parameter(D_init.to(torch.float32)))
            
    def expectation_conditional_matrix_variate_dist(self, x_star):
                
        kron_cov_inv = self.prior_H.kron_cov_inv
        row_cross_covar = self.row_covar_kernel(x_star, self.x).evaluate() # N* x N 
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
                # inputs are the same -> kernel on test inputs K_{**} is being computed.
                
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
    
if __name__ == "__main__":
    
    import matplotlib.pyplot as plt

    num_grid = 30
    X = np.linspace(2, 3, num_grid)
    #jitter = np.eye(num_grid**2)*1e-6
    X_grid = np.meshgrid(X, X)
    X = torch.Tensor(np.vstack((X_grid[0].flatten(), X_grid[1].flatten())).T)
    
    X2 = X[torch.randint(0,900,(200,))] + 0.5*torch.randn(200,2)

    mgk = MultivariateGibbsKernel(X,2)
    K = mgk.forward(X, X2).detach()
    f = pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K + np.eye(num_grid**2)*1e-5, shape=(num_grid**2,)).random(size=1).T
    plt.matshow(K.detach())
    
    plt.figure()
    plt.contourf(X_grid[0], X_grid[1], f[:,0].reshape(num_grid, num_grid), cmap=plt.get_cmap('jet'), alpha=0.7)

#     # ##### Visualisation 
    # import matplotlib.pyplot as plt
    # f_list = []
    
    # plt.style.use('seaborn-deep')
    # fig, axs = plt.subplots(3, 7)
    # fig.suptitle('Multivariate Gibbs Kernel', fontsize='small')
    # for i in range(3):
    #     for j in range(7):
    #         mgk = MultivariateGibbsKernel(X,2)
    #         K = mgk.forward(X, X).detach()
    #         f = pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K + np.eye(num_grid**2)*1e-4, shape=(num_grid**2,)).random(size=1).T
    #         #plt.contourf(X_grid[0], X_grid[1], f[:,0].reshape(num_grid, num_grid), cmap=plt.get_cmap('jet'), alpha=0.7)
    #         axs[i, j].imshow(K)
    #         f_list.append(f)
    #         #axs[i, j].axis('off')
    #         axs[i,j].set_xticks([])
    #         axs[i,j].set_yticks([])
    # plt.subplots_adjust(wspace=0, hspace=0.02)
    
    # f_stack = np.array(f_list).squeeze()
    
    # plt.style.use('seaborn-deep')
    # fig, axs = plt.subplots(3, 7)
    # fig.suptitle('Multivariate Gibbs Kernel - 2d Function samples', fontsize='small')
    # k=0
    # for i in range(3):
    #     for j in range(7):
    #         #f = pm.MvNormal.dist(mu=np.zeros(K.shape[0]), cov=K + np.eye(num_grid**2)*1e-3, shape=(num_grid**2,)).random(size=1).T
    #         #plt.contourf(X_grid[0], X_grid[1], f[k].reshape(num_grid, num_grid), cmap=plt.get_cmap('jet'), alpha=0.7)
    #         axs[i, j].contourf(X_grid[0], X_grid[1], f_stack[k].reshape(num_grid, num_grid), cmap=plt.get_cmap('jet'), alpha=0.7)
    #         #axs[i, j].axis('off')
    #         axs[i,j].set_xticks([])
    #         axs[i,j].set_yticks([])
    #         k += 1 
    # plt.subplots_adjust(wspace=0, hspace=0.02)


    # from mpl_toolkits.axes_grid1 import ImageGrid
    # fig = plt.figure(figsize=(16,3))
    # grid1 = ImageGrid(fig, 133, nrows_ncols=(3,7), axes_pad=0.001)
    # k = 10
    # #images = model(samples[[k], :]).loc[:, 0]
    # for axs in grid1:
    #           k += 1
    #           axs.imshow(ngk.sigma_matrix_i[5,k,:,:].detach())
    #           axs.axis('off')
    # grid1.axes_all[3].set_title('Reconstructions', fontsize='small')
            
    # def compute_prefactor_term_per_dim(self, ls_processes):
        
    #         ls1 = ls_processes[:,None]
    #         ls2 = ls_processes[:,None]
            
    #         square_term = self.compute_square_term_per_dim(ls1, ls2)
    #         self.square_terms.append(square_term)
    #         prod_term = 2 * np.outer(ls1, ls2)[:,:,None]
    #         prefactor = (prod_term / square_term).pow(0.5).prod(dim=-1) 
    #         return prefactor 
              
    # def compute_prefactor(self, ls_processes):
        
    #     if self.input_dim == 1:
            
    #         prefactor = self.compute_prefactor_term_per_dim(ls_processes)
    #         return prefactor
              
    #     else:
    #         # product of prefactor terms over d dims
    #         cumulative_prefactor = torch.ones_like(torch.eye(ls_processes.shape[0]))
    #         for i in range(self.input_dim):
    #             prefactor = self.compute_prefactor_term_1d(ls_processes[:,i])
    #             cumulative_prefactor = torch.mul(cumulative_prefactor, prefactor)
                
    #         return cumulative_prefactor
            