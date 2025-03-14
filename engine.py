# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 10:04:30 2025

@author: User
"""

import torch
import torch.nn as nn

def train_step(model, dataloader, device, optimizer):
    
    recon_loss_fn = nn.MSELoss()
    train_loss = 0
    
    for batch, (X, _) in enumerate(dataloader):
        
        X = X.to(device)
        
        X_recon, mean, log_var = model(X)
        
        recon_loss = recon_loss_fn(X_recon, X)
        kl_loss = torch.mean(torch.sum(1 + log_var - mean**2 - torch.exp(log_var), dim=1) * (-0.5))
        loss = recon_loss + 0.00001 * kl_loss
        
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(dataloader)
    
    return {"model_name": model.__class__.__name__,
            "loss": train_loss}


def test_step(model, dataloader, device):
    
    recon_loss_fn = nn.MSELoss()
    test_loss = 0
    
    model.eval()
    with torch.inference_mode():
        
        for batch, (X, _) in enumerate(dataloader):
            
            X = X.to(device)
            
            X_recon, mean, log_var = model(X)
            
            recon_loss = recon_loss_fn(X_recon, X)
            kl_loss = torch.mean(torch.sum(1 + log_var - mean**2 - torch.exp(log_var), dim=1) * (-0.5))
            loss = recon_loss + 0.00001 * kl_loss
            
            test_loss += loss
        
        test_loss /= len(dataloader)
    
    return {"model_name": model.__class__.__name__,
            "loss": test_loss}


        