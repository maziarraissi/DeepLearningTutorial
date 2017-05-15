#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import autograd.numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
from ConditionalVariationalAutoencoders import ConditionalVariationalAutoencoders
from Utilities import Normalize

np.random.seed(1234)
    
if __name__ == "__main__":
    
    N = 100
    X_dim = 1
    Y_dim = 1
    Z_dim = 2
    noise = 0.1
    
    layers_encoder_0 = np.array([X_dim,50,50,Z_dim])
    layers_encoder_1 = np.array([Y_dim,50,50,Z_dim])
    layers_decoder = np.array([Z_dim,50,50,Y_dim])
    
    Normalize_input_data = 1
    Normalize_output_data = 1
    
    lb = 0.0*np.ones((1,X_dim))
    ub = 1.0*np.ones((1,X_dim)) 
    
    # Generate training data
    def f(x):
        return (x<-0.5) + 1.0 + 1.5*(x>0.5)
    
    X = lb + (ub-lb)*lhs(X_dim, N)
    Y = f(X) + noise*np.random.randn(N,Y_dim)
    
    # Generate test data
    N_star = 400
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
    model = ConditionalVariationalAutoencoders(X, Y, layers_encoder_0, layers_encoder_1, layers_decoder, 
                 max_iter = 5000, N_batch = 5, monitor_likelihood = 10, 
                 lrate = 1e-3)
        
    # Training
    model.train()
      
    # Prediction
    N_samples = 100
    plt.figure(1)
    plt.rcParams.update({'font.size': 14})
    for i in range(0, N_samples):
        mean_star, _ = model.generate_samples(X_star, 1)
        plt.plot(X_star, mean_star)
        
    plt.scatter(X, Y, alpha = 1, label = '%d training data' % N)
    plt.xlabel('$x$')
    plt.ylabel('$f(x)$')
    plt.legend(loc='upper left')
    
    plt.savefig('Example_CVAE.eps', format='eps', dpi=1000)