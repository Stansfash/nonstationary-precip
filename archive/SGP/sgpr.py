#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SGPR
Reference: Michalis Titsias 2009, Sparse Gaussian processes using inducing points.
"""

import gpytorch
import torch
import numpy as np
from gpytorch.means import ZeroMean
from gpytorch.kernels import ScaleKernel, RBFKernel, InducingPointKernel
from gpytorch.distributions import MultivariateNormal
#from utils.metrics import get_trainable_param_names
torch.manual_seed(45)
np.random.seed(37)

def func(x):
    return np.sin(x * 3) + 0.3 * np.cos(x * 4 * 3.14) 

class SparseGPR(gpytorch.models.ExactGP):

    def __init__(self, train_x, train_y, likelihood, Z_init, kernel):

        """The sparse GP class for regression with the collapsed bound.
           q*(u) is implicit.
        """
        super(SparseGPR, self).__init__(train_x, train_y, likelihood)
        self.train_x = train_x
        self.train_y = train_y
        self.inducing_points = Z_init
        self.num_inducing = len(Z_init)
        self.likelihood = likelihood
        self.mean_module = ZeroMean()
        self.base_covar_module = kernel
        self.covar_module = InducingPointKernel(self.base_covar_module, inducing_points=Z_init, likelihood=self.likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    @staticmethod
    def optimization_trace(trace_states, states, grad_params):
        trace_states.append({param_name: param.numpy() for param_name, param in states.items() if param_name in grad_params})
        return trace_states

    def train_model(self, optimizer, combine_terms=True, n_restarts=10, max_steps=10000):

        self.train()
        self.likelihood.train()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

        #grad_params = get_trainable_param_names(self)

        #trace_states = []
        losses = []

        for j in range(max_steps):
              optimizer.zero_grad()
              output = self.forward(self.train_x)
              if combine_terms:
                  loss = -mll(output, self.train_y).sum()
              else:
                  loss = -self.elbo(output, self.train_y)
              losses.append(loss.item())
              loss.backward()
              if j%10 == 0:
                        print('Iter %d/%d - Loss: %.3f   outputscale: %.3f  lengthscale: %s   noise: %.3f ' % (
                        j + 1, max_steps, loss.item(),
                        self.base_covar_module.outputscale.item(),
                        self.base_covar_module.base_kernel.lengthscale,
                        self.likelihood.noise.item()))
                        #self.covar_module.inducing_points[0:5]))
              optimizer.step()
        return losses


    def optimal_q_u(self):
       return self(self.covar_module.inducing_points)

    def posterior_predictive(self, test_x):

        ''' Returns the posterior predictive multivariate normal '''

        self.eval()
        self.likelihood.eval()

        # Make predictions by feeding model through likelihood
        with torch.no_grad():
            y_star = self.likelihood(self(test_x))
        return y_star
    

if __name__ == "__main__":
    
    N = 1000  # Number of training observations

    X = torch.randn(N) * 2 - 1  # X values
    Y = func(X) + 0.2 * torch.randn(N)  # Noisy Y values

    # Initial inducing points
    Z_init = torch.randn(12)
    
    # Initialise model and likelihood
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SparseGPR(X[:,None], Y, likelihood, Z_init)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
        
    # Train
    losses = model.train_model(optimizer, max_steps=5000)
    
    # # Test 
    test_x = torch.linspace(-8, 8, 1000)
    test_y = func(test_x)
    
    ## predictions
    test_pred = model.posterior_predictive(test_x)
    
    from metrics import rmse, nlpd
    
    y_std = torch.tensor([1.0]) ## did not scale y-values

    rmse_test = np.round(rmse(test_pred.loc, test_y,y_std).item(), 4)
    nlpd_test = np.round(nlpd(test_pred, test_y, y_std).item(), 4)
    


