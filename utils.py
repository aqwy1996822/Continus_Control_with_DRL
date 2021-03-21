import random
import numpy as np
from collections import deque
import copy
import torch

from config import DEVICE as device

class ReplayBuffer:

    def __init__(self, maxlen):
        
        self.memory = deque(maxlen=maxlen)  
    
    def add(self, state, action, reward, next_state, done):
        
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batchsize):
        
        experiences = random.sample(self.memory, k=batchsize)
        
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        return states, actions, rewards, next_states, dones
    
class OUNoise:

    def __init__(self, action_size, mu=0, theta=0.15, sigma=0.05):
        
        self.action_size = action_size
        self.mu = mu * np.ones(action_size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        
        self.state = copy.copy(self.mu)

    def sample(self):
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(size=self.action_size)
        self.state = x + dx
        return self.state