import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import numpy as np

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = weightNorm(nn.Linear(n_feature, n_hidden))
        self.predict = weightNorm(nn.Linear(n_hidden, n_output))

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
