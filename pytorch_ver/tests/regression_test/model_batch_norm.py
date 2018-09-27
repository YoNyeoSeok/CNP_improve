import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Net(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.bn_input = nn.BatchNorm1d(1)
        self.hidden = nn.Linear(n_feature, n_hidden)
        self.bn = nn.BatchNorm1d(n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.bn_input(x)
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x
