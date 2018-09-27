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
        import time
        t = time.time()
        self.r = self.operator(self.net_h(O), dim=0).expand(T.shape[0], -1)
        self.xr = torch.cat((self.r, T[:,:self.io_dims[0]]), dim=1)
        self.phi = self.net_g(self.xr)

#        if self.io_dims[1] != 1: 
#            print("multivariate regression is not supported")
#            return
        t = time.time()
        self.mu = self.phi[:,:self.io_dims[1]]
        self.sig = self.softplus(self.phi[:,self.io_dims[1]:])
        
        t = time.time()
        log_probs = -0.5*torch.log(2*np.pi*self.sig**2) - 0.5*(self.mu-T[:,self.io_dims[0]:])**2/self.sig**2

        """
        log_probs = []
        def func(m, s, t):
            normal = MultivariateNormal(m, torch.diag(s**2))
            return normal.log_prob(t)
#        print(self.mu)
#        print(self.sig)
#        print(T)
#        print(T[:, self.io_dims[0]:])
        diffs = T[:, self.io_dims[0]:] - self.mu

        n = self.sig[0].size(-1)
        cholesky = torch.stack([C.potrf(upper=False) for C in bmat.reshape((-1, n, n))])
        scale_tril = cholesky.view(bmat.shape)
        scale_tril = _batch_portf_lower(covariance_matrix)

        flat_L = L.unsqueeze(0).reshape((-1,) + L.shape[-2:])
        L_inv = torch.stack([torch.inverse(Li.t()) for Li in flat_L]).view(L.shape)
        M = (x.unsqueeze(-1) * L_inv).sum(-2).pow(2.0).sum(-1)
        M = _batch_mahanobis(self.scale_tril, diff)

        log_det = _batch_diag(self.scale_tril).abs().log().sum(-1)
        log_probs =  -0.5 * (M + self.loc.size(-1) * math.log(2 * math.pi)) - log_det
            
#        log_probs = list(map(func, self.mu, self.sig, T[:, self.io_dims[0]:]))
#        for m, s, t in zip(self.mu, self.sig, T):
#            #            normal = MultivariateNormal(m, torch.diag(s**2))
##            log_probs.append(normal.log_prob(t[self.io_dims[0]:]))
#            print(m)
#            log_probs.append(t.log_normal_(m[0].cpu().numpy().tolist(), s[0].cpu().numpy().tolist()**2))
#            log_probs.append(normal.log_prob(t[self.io_dims[0]:]))

    
#        t = time.time()
#        normals = [MultivariateNormal(mu, torch.diag(cov)) for mu, cov in 
#                zip(self.mu, self.sig**2)]
#        print('normals', time.time() - t)
#        t = time.time()
#        log_probs = [normals[i].log_prob(t[self.io_dims[0]:]) for i, t in enumerate(T)]
#        print('log_probs', time.time() - t)

        """
        log_prob = 0
        for p in log_probs:
            log_prob += p
        
        return self.phi, log_prob/len(log_probs)

