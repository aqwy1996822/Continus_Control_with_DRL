import torch

#device, CPU or the GPU of your choice
GPU = 0
DEVICE = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#environment names
RAM_ENV_NAME = 'LunarLanderContinuous-v2'
VISUAL_ENV_NAME = None

#Agent parameters
BATCH_SIZE = 256
LR1 = 0.0001
LR2 = 0.001
SPEED1 = 1
SPEED2 = 1
STEP = 1
TAU = 0.001
LEARNING_TIME = 1
OUN=True, 
BN=False, 
CLIP=False,
INIT=True,
HIDDEN=[400, 300]

#Training parameters
L = 10
N = 1