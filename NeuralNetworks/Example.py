#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from NeuralNetworks import NeuralNetworks
from Utilities import Normalize

np.random.seed(1234)
    
if __name__ == "__main__":
    
    N = 100
    X_dim = 1
    Y_dim = 1
    layers = np.array([X_dim,50,50,Y_dim])
    noise = 0.1
    
    Normalize_input_data = 1
    Normalize_output_data = 1
    
    # Generate Training Data   
    def f(x):
        return x*np.sin(4.0*np.pi*x)
    
    lb = 0.0*np.ones((1,X_dim))
    ub = 1.0*np.ones((1,X_dim)) 
    
    X = lb + (ub-lb)*lhs(X_dim, N)
    Y = f(X) + noise*np.random.randn(N,Y_dim)
    
    # Generate Test Data
    N_star = 1000
    X_star = lb + (ub-lb)*np.linspace(0,1,N_star)[:,None]
    Y_star = f(X_star)
    
    #  Normalize Input Data
    if Normalize_input_data == 1:
        X_m = np.mean(X, axis = 0)
        X_s = np.std(X, axis = 0)   
        X = Normalize(X, X_m, X_s)
        X_star = Normalize(X_star, X_m, X_s)
        
    #  Normalize Output Data
    if Normalize_output_data == 1:
        Y_m = np.mean(Y, axis = 0)
        Y_s = np.std(Y, axis = 0)   
        Y = Normalize(Y, Y_m, Y_s)
        Y_star = Normalize(Y_star, Y_m, Y_s)
        
    # Model creation
    model = NeuralNetworks(X, Y, layers, 
                 max_iter = 10000, N_batch = 10, 
                 monitor_likelihood = 10, lrate = 1e-3)
        
    model.train()
    
    mean_star = model.forward_pass(X_star, model.layers, model.hyp)
    
    plt.figure(1)
    plt.rcParams.update({'font.size': 14})
    plt.plot(X_star, Y_star, 'b-', linewidth=2)
    plt.plot(X_star, mean_star, 'r--', linewidth=3)
    plt.scatter(X, Y, alpha = 1)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(['$f(x)$', 'prediction', '%d training data' % N], loc='lower left')
        
    plt.savefig('Example_NN.eps', format='eps', dpi=1000)
