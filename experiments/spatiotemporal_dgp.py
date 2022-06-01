import torch
import tqdm
import gpytorch
import urllib.request
import os
from math import floor
import pandas as pd
import numpy as np
import models.dgps as m
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import DeepApproximateMLL
from sklearn.utils import shuffle
import scipy.stats
from scipy.special import inv_boxcox

num_epochs = 200
num_samples = 10
num_layers = 3

print('num_epochs = ', num_epochs)
print('num_samples = ', num_samples)
print('num_layers = ', num_layers)


filepath = 'data/uib_2000_2010_tp.csv'

df = pd.read_csv(filepath)
data = torch.Tensor(df.values)

X = data[:394*5,:-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:394*5, -1]
y_tr, bc_param = scipy.stats.boxcox(y + 0.001)
y_tr = torch.Tensor(y_tr)

train_n = 394*4
train_x = X[:train_n, :].contiguous()
train_y = y_tr[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y [train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


#### Model
if num_layers == 2:
    model = m.DeepGP2(train_x.shape)
if num_layers == 3:
    model = m.DeepGP3(train_x.shape)
if num_layers == 5:
    model = m.DeepGP5(train_x.shape)

if torch.cuda.is_available():
    model = model.cuda()


#### Training
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
], lr=0.01)
mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
for i in epochs_iter:
    # Within each iteration, we will go over each minibatch of data
    minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
    for x_batch, y_batch in minibatch_iter:
        with gpytorch.settings.num_likelihood_samples(num_samples):
            optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())


#### Metrics
def negative_log_predictive_density(test_y, predicted_mean, predicted_var):
    # Vector of log-predictive density per test point    
    lpd = torch.distributions.Normal(predicted_mean, torch.sqrt(predicted_var)).log_prob(test_y)
    # return the average
    return -torch.mean(lpd)
def sqrt_mean_squared_error(test_y, predicted_mean):
    return torch.sqrt(torch.mean((test_y - predicted_mean)**2))


test_dataset = TensorDataset(test_x, test_y)
test_loader = DataLoader(test_dataset, batch_size=1024)

model.eval()
predictive_means, predictive_variances, test_lls = model.predict(test_loader)

# Inverse transform predictions
predictive_means_tr = torch.Tensor(inv_boxcox(predictive_means, bc_param))
vanilla_variances = torch.Tensor(np.ones((len(predictive_variances), len(predictive_variances[0]))))

rmse = sqrt_mean_squared_error(test_y, predictive_means_tr)
nlpd = negative_log_predictive_density(test_y, predictive_means_tr, vanilla_variances)

print(f"RMSE: {rmse.item()}, NLPD: {nlpd.item()}")


