import torch as T
import torch.nn as nn

import pandas as pd
import numpy as np

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(Policy, self).__init__()
        
        # Layers
        self.linear1 = nn.Linear(obs_dim, 50)
        self.linear2 = nn.Linear(50, 30)
        self.linear3 = nn.Linear(30, act_dim)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        
        return x