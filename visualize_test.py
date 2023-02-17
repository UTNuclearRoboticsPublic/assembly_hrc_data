import torch
import torchvision
from torch.utils.data import DataLoader
from dataloader import AssemblyDataset
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import visualize

epochs = []
error1 = []
error2 = []
error3 = []
error4 = []

for x in range(1,16):
    error1.append(random.randrange(1,10))
    error2.append(random.randrange(1,10))
    error3.append(random.randrange(1,10))
    error4.append(random.randrange(1,10))
    epochs.append(x)

visualize(epochs, error1, error2, error3, error4)