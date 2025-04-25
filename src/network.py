import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class FeedForwardNN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)


    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
  
        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)  
        
        return output



class BetaPolicyNetwork(nn.Module):
    def __init__(self, in_dim, action_dim):
        super(BetaPolicyNetwork, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        
        # Output layers for alpha and beta (each of shape [batch, action_dim])
        self.alpha_head = nn.Linear(64, action_dim)
        self.beta_head = nn.Linear(64, action_dim)

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)

        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))

        # Ensure α, β > 1 by using softplus + 1 (as in the paper)
        alpha = F.softplus(self.alpha_head(x)) + 1.0
        beta = F.softplus(self.beta_head(x)) + 1.0

        return alpha, beta