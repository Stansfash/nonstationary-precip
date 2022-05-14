import torch
import tqdm
import gpytorch
import urllib.request
import os
from math import floor
import pandas as pd
import DGP.dgps as dgps
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import DeepApproximateMLL

data_df = pd.read_csv('khyber_2000_2010_tp.csv')

data = torch.Tensor(data_df.values)
X = data[:, 1:-1]
X = X - X.min(0)[0]
X = 2 * (X / X.max(0)[0]) - 1
y = data[:, -1]

train_n = int(floor(0.8 * len(X)))
train_x = X[:train_n, :].contiguous()
train_y = y[:train_n].contiguous()

test_x = X[train_n:, :].contiguous()
test_y = y[train_n:].contiguous()

if torch.cuda.is_available():
    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()


train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)


#### Model 
model = dgps.DeepGP2(train_x.shape)

if torch.cuda.is_available():
    model = model.cuda()


#### Training
num_epochs = 20
num_samples = 400

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

rmse = sqrt_mean_squared_error(test_y, predictive_means)
nlpd = negative_log_predictive_density(test_y, predictive_means, predictive_variances)

print(f"RMSE: {rmse.item()}, NLPD: {nlpd.item()}")