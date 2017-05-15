#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Maziar Raissi
"""

import autograd.numpy as np
from autograd import value_and_grad
from Utilities import fetch_minibatch, stochastic_update_Adam, activation

class ConditionalVariationalAutoencoders:
    def __init__(self, X, Y, layers_encoder_0, layers_encoder_1, layers_decoder, 
                 max_iter = 2000, N_batch = 1, monitor_likelihood = 10, lrate = 1e-3): 
        self.X = X
        self.Y = Y
        self.Y_dim = Y.shape[1]
        self.Z_dim = layers_encoder_0[-1]
        self.layers_encoder_0 = layers_encoder_0
        self.layers_encoder_1 = layers_encoder_1
        self.layers_decoder = layers_decoder
        
        self.max_iter = max_iter
        self.N_batch = N_batch
        self.monitor_likelihood = monitor_likelihood
        
        # Initialize encoder_0
        hyp =  self.initialize_NN(layers_encoder_0)
        self.idx_encoder_0 = np.arange(hyp.shape[0])
        
        # Initialize encoder_1
        hyp = np.concatenate([hyp, self.initialize_NN(layers_encoder_1)])
        self.idx_encoder_1 = np.arange(self.idx_encoder_0[-1]+1, hyp.shape[0])
        
        # Initialize decoder
        hyp = np.concatenate([hyp, self.initialize_NN(layers_decoder)])
        self.idx_decoder = np.arange(self.idx_encoder_1[-1]+1, hyp.shape[0])
                
        self.hyp = hyp
        
        # Adam optimizer parameters
        self.mt_hyp = np.zeros(hyp.shape)
        self.vt_hyp = np.zeros(hyp.shape)
        self.lrate = lrate
        
        print("Total number of parameters: %d" % (hyp.shape[0]))
        
        
    def initialize_NN(self, Q):
        hyp = np.array([])
        layers = Q.shape[0]
        for layer in range(0,layers-2):
            A = -np.sqrt(6.0/(Q[layer]+Q[layer+1])) + 2.0*np.sqrt(6.0/(Q[layer]+Q[layer+1]))*np.random.rand(Q[layer],Q[layer+1])
            b = np.zeros((1,Q[layer+1]))
            hyp = np.concatenate([hyp, A.ravel(), b.ravel()])

        A = -np.sqrt(6.0/(Q[-2]+Q[-1])) + 2.0*np.sqrt(6.0/(Q[-2]+Q[-1]))*np.random.rand(Q[-2],Q[-1])
        b = np.zeros((1,Q[-1]))
        hyp = np.concatenate([hyp, A.ravel(), b.ravel()])
        
        A = -np.sqrt(6.0/(Q[-2]+Q[-1])) + 2.0*np.sqrt(6.0/(Q[-2]+Q[-1]))*np.random.rand(Q[-2],Q[-1])
        b = np.zeros((1,Q[-1]))
        hyp = np.concatenate([hyp, A.ravel(), b.ravel()])
        
        return hyp
        
    def neural_net(self, X, Q, hyp):
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

        idx_1 = idx_3
        idx_2 = idx_1 + Q[-2]*Q[-1]
        idx_3 = idx_2 + Q[-1]
        A = np.reshape(hyp[idx_1:idx_2], (Q[-2],Q[-1]))
        b = np.reshape(hyp[idx_2:idx_3], (1,Q[-1]))
        Sigma = np.exp(np.matmul(H,A) + b)
        
        return mu, Sigma    
    
    def likelihood(self, hyp):
        X = self.X_batch
        Y = self.Y_batch     
            
        # Encode X
        mu_0, Sigma_0 = self.neural_net(X, self.layers_encoder_0, hyp[self.idx_encoder_0]) 
        
        # Encode Y
        mu_1, Sigma_1 = self.neural_net(Y, self.layers_encoder_1, hyp[self.idx_encoder_1]) 
        
        # Reparametrization trick
        epsilon = np.random.randn(self.N_batch,self.Z_dim)        
        z = mu_1 + epsilon*np.sqrt(Sigma_1)
        
        # Decode
        mu_2, Sigma_2 = self.neural_net(z, self.layers_decoder, hyp[self.idx_decoder])
        
        # Log-determinants
        log_det_0 = np.sum(np.log(Sigma_0))
        log_det_1 = np.sum(np.log(Sigma_1))
        log_det_2 = np.sum(np.log(Sigma_2))
        
        # KL[q(z|y) || p(z|x)]
        KL = 0.5*(np.sum(Sigma_1/Sigma_0) + np.sum((mu_0-mu_1)**2/Sigma_0) - self.Z_dim + log_det_0 - log_det_1)
        
        # -log p(y|z)
        NLML = 0.5*(np.sum((Y-mu_2)**2/Sigma_2) + log_det_2 + np.log(2.*np.pi)*self.Y_dim*self.N_batch)
                   
        return NLML + KL
    

    def train(self):
        
        # Gradients from autograd 
        NLML = value_and_grad(self.likelihood)
        
        for i in range(1,self.max_iter+1):
            # Fetch minibatch
            self.X_batch, self.Y_batch = fetch_minibatch(self.X, self.Y, self.N_batch) 
            
            # Compute likelihood_UB and gradients 
            NLML_value, D_NLML = NLML(self.hyp)
            
            # Update hyper-parameters
            self.hyp, self.mt_hyp, self.vt_hyp = stochastic_update_Adam(self.hyp, D_NLML, self.mt_hyp, self.vt_hyp, self.lrate, i)
            
            if i % self.monitor_likelihood == 0:
                print("Iteration: %d, likelihood: %.2f" % (i, NLML_value))
        
        
    def generate_samples(self, X_star, N_samples):
        
        # Encode X_star
        mu_0, Sigma_0 = self.neural_net(X_star, self.layers_encoder_0, self.hyp[self.idx_encoder_0]) 
                   
        # Reparametrization trick
        epsilon = np.random.randn(N_samples,self.Z_dim)        
        Z = mu_0 + epsilon*np.sqrt(Sigma_0)
                
        # Decode
        mean_star, var_star = self.neural_net(Z, self.layers_decoder, self.hyp[self.idx_decoder]) 
                    
        return mean_star, var_star
    