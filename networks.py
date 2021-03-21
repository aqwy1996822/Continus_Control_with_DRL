import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1 / math.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    
    def __init__(self, state_size, action_size, batchnorm, initialize, hidden=[256, 256]):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0], hidden[1])
        self.fc3 = nn.Linear(hidden[1], action_size)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.bn2 = nn.BatchNorm1d(hidden[1])
        self.batchnorm = batchnorm
        if initialize:
            self.initialize()
        
    def forward(self, x):
        if self.batchnorm:
            x = F.relu(self.bn1(self.fc1(x)))
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x
    
    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
        
class Critic(nn.Module):
    
    def __init__(self, state_size, action_size, batchnorm, initialize, hidden=[256, 256]):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden[0])
        self.fc2 = nn.Linear(hidden[0]+action_size, hidden[1])
        self.fc3 = nn.Linear(hidden[1], 1)
        self.bn1 = nn.BatchNorm1d(hidden[0])
        self.batchnorm = batchnorm
        if initialize:
            self.initialize()
    
    def forward(self, state, action):
        if self.batchnorm:
            x = F.relu(self.bn1(self.fc1(state)))
        else:
            x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def initialize(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc1.bias.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc2.bias.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-4, 3e-4)
        self.fc3.bias.data.uniform_(-3e-4, 3e-4)