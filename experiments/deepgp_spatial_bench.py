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
import utils.dataprep as dp

filepath = 'data/uib_spatial.csv'
dataset = dp.download_data(filepath)

transform = 'whitening' # or 'boxcox'

# def load_khyber_data():
        
#     fname = str(DATASET_DIR) + '/uib_spatial.csv'
#     data = pd.read_csv(fname, dtype=np.float64)
#     return data, torch.Tensor(np.array(data))[:,0:2], torch.Tensor(np.array(data)[:,-1])

num_epochs = 200
num_samples = 3
num_layers = 2
filepath = 'data/uib_spatial.csv'

print('num_epochs = ', num_epochs)
print('num_samples = ', num_samples)
print('num_layers = ', num_layers)

rmses = []
nlpds = []

for random_state in range(1):
    
    print('random_state = ', random_state)

    data = shuffle(dataset, random_state=random_state)

    if transform == 'whitening':    
        x_tr, y_tr, meanx, stdx, meany, stdy = dp.whitening_transform(data)
    elif transform == 'boxcox': 
        x_tr, y_tr, bc_param = dp.box_cox_transform(data)
        stdy = 1.0
        
    train_x, train_y, test_x, test_y = dp.train_test_split(x_tr, y_tr, 0.8)

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=315, shuffle=True)

    #### Model
    model = m.DeepGP(num_layers, train_x.shape)

    if torch.cuda.is_available():
         train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
         model = model.cuda()
         model.likelihood = model.likelihood.cuda()

    #### Training
    model.train()
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
                #print(loss)
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

    # plt_dataset = TensorDataset(x, y)
    # plt_loader = DataLoader(plt_dataset, batch_size=1024)
    # plt_pred_y, plt_y_means, plt_y_var, plt_test_lls = model.predict(plt_loader)
    
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
