#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from RecurrentNeuralNetworks import RecurrentNeuralNetworks
from Utilities import create_dataset

np.random.seed(1234)
    
if __name__ == "__main__":
    
    # generate the dataset
    def f(x):
        return np.sin(np.pi*x)
    
    dataset = f(np.arange(0,10,0.1)[:,None])
    
    # Training Data
    train_size = int(len(dataset) * (2/3))
    train = dataset[0:train_size,:]
    
    # reshape X and Y
    # X has the form lags x data x dim
    # Y has the form data x dim
    lags = 5
    X, Y = create_dataset(train, lags)
    
    # Model creation
    hidden_dim = 4
    model = RecurrentNeuralNetworks(X, Y, hidden_dim, 
                 max_iter = 10000, N_batch = 1, 
                 monitor_likelihood = 10, lrate = 1e-3)    
    
    model.train()
    
    # Prediction
    pred = np.zeros((len(dataset)-lags, Y.shape[-1]))
    X_tmp =  np.copy(X[:,0:1,:])
    for i in range(0, len(dataset)-lags):
        pred[i] = model.forward_pass(X_tmp, model.hyp)
        X_tmp[:-1,:,:] = X_tmp[1:,:,:] 
        X_tmp[-1,:,:] = pred[i]
        
    plt.figure(1)
    plt.rcParams.update({'font.size': 14})
    plt.plot(dataset[lags:], 'b-', linewidth = 2, label = "Exact")
    plt.plot(pred, 'r--', linewidth = 3, label = "Prediction")
    plt.plot(X.shape[1]*np.ones((2,1)), np.linspace(-1.75,1.75,2), 'k--', linewidth=2)
    plt.axis('tight')
    plt.xlabel('$t$')
    plt.ylabel('$y_t$')
    plt.legend(loc='lower left')
    
    plt.savefig('Example_RNN.eps', format='eps', dpi=1000)