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

mu = torch.tensor(4*(np.random.rand(51)-.5))
x = torch.tensor(4*(np.random.rand(51)-.5))
cov = torch.tensor(4*(np.random.rand(51)-.5))**2

normals = MultivariateNormal(mu, torch.diag(cov))
print(normals)
"""

