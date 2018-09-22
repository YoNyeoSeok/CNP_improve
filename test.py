"""
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log", action='store_false')
args = parser.parse_args()

print(args.log)
print(type(args.log))
"""



import numpy as np
import torch
from  torch.distributions.multivariate_normal import MultivariateNormal

#mu = torch.tensor(4*(np.random.rand(51)-.5))
x = 4*(np.random.rand(51)-.5)
x2 = 4*(np.random.rand(51)-.5)
#x = torch.tensor([x, x2])
y = 4*(np.random.rand(51)-.5)
T = torch.tensor(np.concatenate(([x], [y]), axis=0))
print(T)
x, y = T
print(x, y)
mu = torch.tensor([4*(np.random.rand(51)-.5)])
cov = torch.tensor([4*(np.random.rand(51)-.5)])**2
#self.mu = self.phi[:,:self.io_dims[1]]
#self.sig = self.softplus(self.phi[:,self.io_dims[1]:])
normals = [MultivariateNormal(mu, torch.diag(cov)) for mu_, cov_ in 
    zip(mu, cov)]
log_probs = [normals[i].log_prob(t[1:]) for i, t in enumerate(T)]

#normals = MultivariateNormal(mu, torch.diag(cov))
#print(normals)
