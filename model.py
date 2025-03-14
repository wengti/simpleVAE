# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:29:47 2025

@author: User
"""

import torch
import torch.nn as nn


class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.common_fc = nn.Sequential(nn.Flatten(),
                                       nn.Linear(in_features = 28*28,
                                                 out_features = 196),
                                       nn.Tanh(),
                                       nn.Linear(in_features = 196,
                                                 out_features = 48),
                                       nn.Tanh())
                                       
        
        self.mean_fc = nn.Sequential(nn.Linear(in_features = 48,
                                               out_features = 16),
                                     nn.Tanh(),
                                     nn.Linear(in_features = 16,
                                               out_features = 2))
        
        self.log_var_fc = nn.Sequential(nn.Linear(in_features = 48,
                                                  out_features = 16),
                                        nn.Tanh(),
                                        nn.Linear(in_features = 16,
                                                  out_features = 2))
        
        self.decoder_fc = nn.Sequential(nn.Linear(in_features = 2,
                                                 out_features = 16),
                                       nn.Tanh(),
                                       nn.Linear(in_features = 16,
                                                 out_features = 48),
                                       nn.Tanh(),
                                       nn.Linear(in_features = 48,
                                                 out_features = 196),
                                       nn.Tanh(),
                                       nn.Linear(in_features = 196,
                                                 out_features = 28*28),
                                       nn.Tanh())
    
    def encoder(self, x):
        common_out = self.common_fc(x)
        mean = self.mean_fc(common_out)
        log_var = self.log_var_fc(common_out)
        return mean, log_var
    
    def reparameterization(self, mean, log_var):
        std = torch.exp(log_var) ** 0.5
        z = torch.randn_like(std)
        out = mean + z*std
        return out
    
    def decoder(self, x):
        out = self.decoder_fc(x)
        out_reshape = out.reshape((x.shape[0], 1, 28, 28))
        return out_reshape
        
    def forward(self, x):
        mean, log_var = self.encoder(x)
        out_reparam = self.reparameterization(mean, log_var)
        out = self.decoder(out_reparam)
        return out, mean, log_var