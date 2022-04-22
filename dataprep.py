# Data preparation

import pandas as pd
import scipy as sp
import torch
from torch.utils.data import TensorDataset


def download_data(filepath):
    df = pd.read_csv(filepath)
    data = torch.Tensor(df.values)
    return data

def prep_inputs(data):
    X = data[:, 1:-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    return X

def prep_outputs(data):
    y = data[:, -1]
    # performs Box-Cox transformation to make y distribution more Gaussian
    y_tr, bc_param = sp.stats.boxcox(y)
    # y = sp.special.inv_boxcox(y_tr, bc_param)
    return y_tr, bc_param

def test_train_split(X, y):
    train_n = int(floor(0.8 * len(X)))
    train_x = X[:train_n, :].contiguous()
    train_y = y[:train_n].contiguous()
    test_x = X[train_n:, :].contiguous()
    test_y = y[train_n:].contiguous()
    return train_x, train_y, test_x, test_y 
    
    
if __name__ == "__main__":
    
    filepath = 'khyber_2000_2010_tp.csv'
    data = download_data(filepath)
    
    X = prep_inputs(data)
    y , boxcox_param = prep_outputs(data)
    
    train_x, train_y, test_x, test_y = test_train_split(X, y)
    
    if torch.cuda.is_available():
        train_x, train_y, test_x, test_y = train_x.cuda(), train_y.cuda(), test_x.cuda(), test_y.cuda()
    
    train_dataset = TensorDataset(train_x, train_y)
    