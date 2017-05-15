#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import autograd.numpy as np

def stochastic_update_Adam(w,grad_w,mt,vt,lrate,iteration):
    beta1 = 0.9;
    beta2 = 0.999;
    epsilon = 1e-8;

    mt = mt*beta1 + (1.0-beta1)*grad_w;
    vt = vt*beta2 + (1.0-beta2)*grad_w**2;

    mt_hat = mt/(1.0-beta1**iteration);
    vt_hat = vt/(1.0-beta2**iteration);

    scal = 1.0/(np.sqrt(vt_hat) + epsilon);

    w = w - lrate*mt_hat*scal;
    
    return w,mt,vt

def Normalize(X, X_m, X_s):
    return (X-X_m)/(X_s)
     
def Denormalize(X, X_m, X_s):    
    return X_s*X + X_m

def fetch_minibatch_rnn(X,Y,N_batch):
    N = X.shape[1]
    idx = np.random.choice(N, N_batch, replace=False)
    X_batch = X[:,idx,:]
    Y_batch = Y[idx,:]
    return X_batch, Y_batch

def activation(x):
    return np.tanh(x)
    
def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)

def create_dataset(data, lags):
    N = len(data)-lags
    data_dim = data.shape[1]
    X = np.zeros((lags, N, data_dim))
    Y = np.zeros((N, data_dim))
    for i in range(0,N):
        X[:,i,:] = data[i:(i+lags), :]
        Y[i,:] = data[i + lags, :]
    return X, Y
