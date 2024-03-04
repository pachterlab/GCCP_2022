# math
import numpy as np
np.random.seed(30703)
import random
random.seed(30703)

import matplotlib.pyplot as plt
# pytorch
import torch
torch.manual_seed(30703)
import torch.nn as nn
import torch.nn.functional as F

# iterables 
from collections.abc import Iterable

# miscellaneous 
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
eps = 1e-18

# x : [b, beta, gamma, n, mu_1 .... mu_npdf, std_1 ... sdt_npdf]
# x shape : samples by 4+2*npdf

# simplest model, learns weights for basis functions and hyperparameter for scaling std dev
class MLP_weights_hyp(nn.Module):

    def __init__(self, input_size, npdf, num_layers, num_units,
                 activate = 'relu'):
        super().__init__()
        
        self.npdf = npdf
        self.module_list = nn.ModuleList([])
        self.module_list.append(nn.Linear(input_size,num_units,bias=True))
        for k in range(num_layers-1):
            self.module_list.append(nn.Linear(num_units, num_units,bias=True))
        self.module_list.append(nn.Linear(num_units,npdf))

        self.hyp_layer =  nn.Linear(num_units,1)

        self.softmax = nn.Softmax(dim=1)
        self.activate = activate
        

    def forward(self, x):
        
        unscaled_means = x[:,4:4+self.npdf]
        unscaled_stds = x[:,4+self.npdf:]
        
        for f in self.module_list[:-1]:
            x = f(x)
            if self.activate == 'relu':
                x = F.relu(x)
            elif self.activate == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.activate == 'sin':
                x = torch.sin(x)
                
        # apply softmax
        w_pred = self.softmax(self.module_list[-1](x))
        
        # produce hyperparameter, which is set to be between 1 and 6, which scales the standard deviations                      
        hyp = torch.sigmoid(self.hyp_layer(x))*5 + 1
                              
        # scale the standard deviations                      
        scaled_stds = unscaled_stds*hyp
                              
        return w_pred,unscaled_means,scaled_stds
    

# this model learns weights AND returns scaled means and scaled standard deviations
class MLP_weights_scale(nn.Module):

    def __init__(self, input_size, npdf, num_layers, num_units, activate = 'relu',
                final_activation = 'sigmoid', max_mv = 'update', max_val = 2.0):
                   
        super().__init__()
        self.module_list = nn.ModuleList([])
        self.module_list.append(nn.Linear(input_size,num_units,bias=False))
        for k in range(num_layers-1):
            self.module_list.append(nn.Linear(num_units, num_units,bias=False))
        self.module_list.append(nn.Linear(num_units,npdf))
        
        self.scaling_mean = nn.Linear(num_units,npdf)
        self.scaling_std = nn.Linear(num_units,npdf)
        
        self.npdf = npdf
        self.activate = activate
        self.max_mv = max_mv                        
        self.final_activation = final_activation
        self.softmax = nn.Softmax(dim=1)
                              
        if self.max_mv == 'update':
            self.max_mean = nn.Parameter(torch.tensor(max_val,dtype=torch.float32),requires_grad=True)
            self.max_std = nn.Parameter(torch.tensor(max_val,dtype=torch.float32),requires_grad=True)
        else:
            self.max_mean = torch.tensor(max_val,dtype=torch.float32)
            self.max_std = torch.tensor(max_val,dtype=torch.float32)



    def forward(self, x):
        
        # store the unscaled means and standard deviations
        unscaled_means = x[:,4:4+self.npdf]
        unscaled_stds = x[:,4+self.npdf:]
        
        # pass through first layer 
        x = F.relu(self.module_list[0](x))

        # pass through next layers                      
        for f in self.module_list[1:-1]:
            x = f(x)
            if self.activate == 'relu':
                x = F.relu(x)
            elif self.activate == 'sigmoid':
                x = torch.sigmoid(x)
            elif self.activate == 'sin':
                x = torch.sin(x)
                
        scaling_factors_mean = torch.sigmoid(self.scaling_mean(x))
        scaling_factors_std = torch.sigmoid(self.scaling_std(x))  
        
        # apply softmax to weights
        w_pred = self.softmax(self.module_list[-1](x))   
        
        # calculate the scaled means and stds                       
        c_mean = scaling_factors_mean*(self.max_mean-(1/self.max_mean)) + (1/self.max_mean)
        c_std = scaling_factors_std*(self.max_std-(1/self.max_std)) + (1/self.max_std)
        scaled_means = c_mean*unscaled_means
        scaled_stds = c_std*unscaled_stds    

            
        return w_pred,scaled_means,scaled_stds
                              
                              
                            