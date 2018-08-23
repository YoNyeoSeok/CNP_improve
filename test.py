import numpy as np
import torch
from  torch.distributions.multivariate_normal import MultivariateNormal

mu = torch.tensor(4*(np.random.rand(51)-.5))
x = torch.tensor(4*(np.random.rand(51)-.5))
cov = torch.tensor(4*(np.random.rand(51)-.5))**2

normals = MultivariateNormal(mu, torch.diag(cov))
print(normals)


