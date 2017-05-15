#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import autograd.numpy as np
from autograd import value_and_grad
from Utilities import fetch_minibatch, stochastic_update_Adam, activation

class NeuralNetworks:
    
    def __init__(self, X, Y, layers, 
                 max_iter = 2000, N_batch = 1, monitor_likelihood = 10, lrate = 1e-3):
        
        self.X = X
        self.Y = Y
        self.layers = layers
        
        self.max_iter = max_iter
        self.N_batch = N_batch
        self.monitor_likelihood = monitor_likelihood
        
        self.hyp = self.initialize_NN(self.layers)
        
        # Adam optimizer parameters
        self.mt_hyp = np.zeros(self.hyp.shape)
        self.vt_hyp = np.zeros(self.hyp.shape)
        self.lrate = lrate
        
        print("Total number of parameters: %d" % (self.hyp.shape[0]))  
        
    def initialize_NN(self, Q):
        hyp = np.array([])
        layers = Q.shape[0]
        for layer in range(0,layers-1):
            A = -np.sqrt(6.0/(Q[layer]+Q[layer+1])) + 2.0*np.sqrt(6.0/(Q[layer]+Q[layer+1]))*np.random.rand(Q[layer],Q[layer+1])
            b = np.zeros((1,Q[layer+1]))
            hyp = np.concatenate([hyp, A.ravel(), b.ravel()])
        
        return hyp
        
    def forward_pass(self, X, Q, hyp):
        H = X
        idx_3 = 0
        layers = Q.shape[0]   
        for layer in range(0,layers-2):        
            idx_1 = idx_3
            idx_2 = idx_1 + Q[layer]*Q[layer+1]
            idx_3 = idx_2 + Q[layer+1]
            A = np.reshape(hyp[idx_1:idx_2], (Q[layer],Q[layer+1]))
            b = np.reshape(hyp[idx_2:idx_3], (1,Q[layer+1]))
            H = activation(np.matmul(H,A) + b)
            
        idx_1 = idx_3
        idx_2 = idx_1 + Q[-2]*Q[-1]
        idx_3 = idx_2 + Q[-1]
        A = np.reshape(hyp[idx_1:idx_2], (Q[-2],Q[-1]))
        b = np.reshape(hyp[idx_2:idx_3], (1,Q[-1]))
        mu = np.matmul(H,A) + b
                
        return mu
    
    def MSE(self, hyp):
        X = self.X_batch
        Y = self.Y_batch                          
        mu = self.forward_pass(X, self.layers, hyp)                
        return np.mean((Y-mu)**2)
    
    def train(self):
        
        # Gradients from autograd 
        MSE = value_and_grad(self.MSE)
        
        for i in range(1,self.max_iter+1):
            # Fetch minibatch
            self.X_batch, self.Y_batch = fetch_minibatch(self.X, self.Y, self.N_batch)
            
            # Compute MSE and gradients 
            MSE_value, D_MSE = MSE(self.hyp)
            
            # Update hyper-parameters
            self.hyp, self.mt_hyp, self.vt_hyp = stochastic_update_Adam(self.hyp, D_MSE, self.mt_hyp, self.vt_hyp, self.lrate, i)
            
            if i % self.monitor_likelihood == 0:
                print("Iteration: %d, MSE: %.5e" % (i, MSE_value))