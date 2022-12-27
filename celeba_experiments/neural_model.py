import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torch.optim as optim
from torchvision import models
from torch.nn.functional import upsample
from copy import deepcopy
import torch.nn.functional as F


class Nonlinearity(torch.nn.Module):
    def __init__(self):
        super(Nonlinearity, self).__init__()

    def forward(self, x):
        return F.relu(x)


class Net(nn.Module):

    def __init__(self, dim, num_classes=2):
        super(Net, self).__init__()
        bias = False
        k = 1024
        self.dim = dim
        self.width = k
        self.first = nn.Linear(dim, k, bias=bias)
        self.fc = nn.Sequential(Nonlinearity(),
                                nn.Linear(k, k, bias=bias),
                                Nonlinearity(),
                                nn.Linear(k, num_classes, bias=bias))

    def forward(self, x):
        return self.fc(self.first(x))
