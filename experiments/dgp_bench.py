import torch
import tqdm
import gpytorch
import urllib.request
import os
from math import floor
import math
import pandas as pd
import numpy as np
import models.dgps as m
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.mlls import DeepApproximateMLL
from sklearn.utils import shuffle
import scipy.stats
from scipy.special import inv_boxcox
from utils.metrics2 import nlpd, rmse
from utils.config import BASE_SEED, EPSILON, DATASET_DIR
from utils.metrics import rmse, nlpd, get_trainable_param_names

def load_khyber_data():
        
    fname = str(DATASET_DIR) + '/uib_spatial.csv'
    data = pd.read_csv(fname, dtype=np.float64)
    return data, torch.Tensor(np.array(data))[:,0:2], torch.Tensor(np.array(data)[:,-1])

num_epochs = 150
num_samples = 3
num_layers = 3
filepath = 'data/khyber_spatial.csv'
#filepath = 'data/uib_lat32_lon81_tp.csv'

print('num_epochs = ', num_epochs)
print('num_samples = ', num_samples)
print('num_layers = ', num_layers)

rmses = []
nlpds = []

for i in range(10):
    
    print('random_state = ', i)

    data, x, y = load_khyber_data()
    
    rng = np.random.default_rng(BASE_SEED+i)
    torch.manual_seed(BASE_SEED+i)

    # X = data[:,:-1]
    # X = X - X.min(0)[0]
    # X = 2 * (X / X.max(0)[0]) - 1
    # y = data[:, -1]
    # y_tr, bc_param = scipy.stats.boxcox(y)
    # y_tr = torch.Tensor(y_tr)

    #stdy_tr = 1. #torch.std_mean(y_tr)
    #stdy = 1. #torch.std_mean(y)
    
    # train_n = int(floor(0.80 * len(X)))
    # train_x = X[:train_n, :].contiguous()
    # train_y = y[:train_n].contiguous()

    # test_x = X[train_n:, :].contiguous()
    # test_y = y[train_n:].contiguous()
    
    with torch.no_grad():
            stdx, meanx = torch.std_mean(x, dim=-2)
            x_norm = (x -  meanx) / stdx
            stdy, meany = torch.std_mean(y)
            y_norm = (y - meany) / stdy
           
    num_train = math.ceil(80/100 * y.shape[0])
    idx = np.arange(0, y.shape[0], 1)
    rng.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    train_x = x_norm[..., train_idx, :].detach()
    train_y = y_norm[..., train_idx].detach()
    test_x = x_norm[..., test_idx, :].detach()
    test_y = y_norm[..., test_idx].detach()


    #if torch.cuda.is_available():
    #    train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=315, shuffle=True)


    #### Model
    if num_layers == 2:
        model = m.DeepGP2(train_x.shape)
    if num_layers == 3:
        model = m.DeepGP3(train_x.shape)
    if num_layers == 5:
        model = m.DeepGP5(train_x.shape)

    #if torch.cuda.is_available():
    #    model = model.cuda()


    #### Training
    model.train()
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},
    ], lr=0.01)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))

    epochs_iter = tqdm.tqdm(range(num_epochs), desc="Epoch")
    for i in epochs_iter:
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=True)
        for x_batch, y_batch in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch)
                print(loss)
                loss.backward()
                optimizer.step()
                minibatch_iter.set_postfix(loss=loss.item())


    #### Metrics

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=315)

    model.eval()
    with torch.no_grad():
        pred_y, y_means, y_var, test_lls = model.predict(test_loader)

    # Inverse transform predictions
    # pred_y_test_tr = torch.Tensor(inv_boxcox(pred_y_test, bc_param))
    #y_mean_raw = torch.Tensor(inv_boxcox(y_means, bc_param))
    #y_var_ = torch.Tensor(inv_boxcox(y_var + y_mean, bc_param,)) - y_mean_tr
    #test_y_tr = torch.Tensor(inv_boxcox(test_y, bc_param))
        
    #y_mean_raw = inv_boxcox(y_means, bc_param)
    #y_test_raw = inv_boxcox(test_y, bc_param)
    
    ## Metrics
    rmse_test = rmse(y_means, test_y, stdy)
    nlpd_test = nlpd(pred_y, test_y, stdy).mean()
    
    # Metrics RMSE on original / raw values, NLPD on bc trans. values, both stdy=1
    #rmse_test = rmse(y_mean_raw, y_test_raw, torch.Tensor([1.0]))
    #nlpd_test = nlpd(pred_y, test_y, torch.Tensor([1.0])).mean()


    print(f"RMSE: {rmse_test.item()}, NLPD: {nlpd_test.item()}")
    rmses.append(rmse_test.item())
    nlpds.append(nlpd_test.item())

    plt_dataset = TensorDataset(x, y)
    plt_loader = DataLoader(plt_dataset, batch_size=1024)
    plt_pred_y, plt_y_means, plt_y_var, plt_test_lls = model.predict(plt_loader)
    
# df1 = pd.DataFrame()
# df1['pred'] = plt_y_means.mean(axis=0)
# df1['var'] =  np.sqrt(plt_y_var.mean(axis=0))
# df1['lat'] = data[:,1]
# df1['lon'] = data[:,0]
# #df1['time'] = data[:,0]
# #df1.to_csv('data/DGP'+ str('num_layers')+'_'+ str(num_samples)+'samples_uib_32lat_81lon.csv')
# df1.to_csv('results/DGP'+ str('num_layers')+'_uib_jan2000.csv')

print(np.mean(rmses), '±', np.std(rmses)/np.sqrt(10))
print(np.mean(nlpds), '±', np.std(nlpds)/np.sqrt(10))
