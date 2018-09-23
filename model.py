import torch
import torch.nn as nn
import torch.nn.functional as F
from  torch.distributions.normal import Normal
from  torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import scipy
from scipy.linalg import cholesky

class Net_h(nn.Module):
    def __init__(self, input_dim, layers_dim):
        super(Net_h, self).__init__()
        net_h = [nn.Linear(input_dim, layers_dim[0])]
        net_h_ = [nn.Linear(layers_dim[i-1], layers_dim[i])
                  for i in range(1, len(layers_dim))]
        
        self.net = nn.ModuleList(np.concatenate((net_h, net_h_)))

    def forward(self, x):
        for i, l in enumerate(self.net):
            print(x)
            for m, p in l.named_parameters():
                print(m, p)
            x = l(x)
            if i+1 < len(self.net):
                x = F.relu(x)
#                x = F.sigmoid(x)
        return x
    
        
class Net_g(nn.Module):
    def __init__(self, layers_dim, io_dims):
        super(Net_g, self).__init__()
        layers_dim[0] += io_dims[0]
        net_g = [nn.Linear(layers_dim[i-1], layers_dim[i]) 
                 for i in range(1, len(layers_dim))]
        net_g_ = [nn.Linear(layers_dim[-1], io_dims[1]*2)]
        
        self.net = nn.ModuleList(np.concatenate((net_g, net_g_)))

    def forward(self, x):
        for i, l in enumerate(self.net):
            x = l(x)
            if i+1 < len(self.net):
                x = F.relu(x)
#                x = F.sigmoid(x)
        return x 
    
    
class CNP_Net(nn.Module):
    def __init__(self, io_dims=[1,1], \
                 layers_dim={'h':[8, 32, 128], 'g':[128, 64, 32, 16, 8]}):
        super(CNP_Net, self).__init__()
        self.io_dims = io_dims

        self.net_h = Net_h(np.sum(io_dims), layers_dim['h'])
        self.net_g = Net_g(layers_dim['g'], io_dims)
        self.operator = torch.mean
        self.softplus = nn.Softplus()

    def forward(self, O, T):
        print(O, T)
        self.r = self.operator(self.net_h(O), dim=0).expand(T.shape[0], -1)
        self.xr = torch.cat((self.r, T[:,:self.io_dims[0]]), dim=1)
        self.phi = self.net_g(self.xr)

#        if self.io_dims[1] != 1: 
#            print("multivariate regression is not supported")
#            return
        self.mu = self.phi[:,:self.io_dims[1]]
        self.sig = self.softplus(self.phi[:,self.io_dims[1]:])
    
        normals = [MultivariateNormal(mu, torch.diag(cov)) for mu, cov in 
                zip(self.mu, self.sig**2)]
        log_probs = [normals[i].log_prob(t[self.io_dims[0]:]) for i, t in enumerate(T)]

        log_prob = 0
        for p in log_probs:
            log_prob += p
        
        return self.phi, log_prob/len(log_probs)

