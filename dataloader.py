# dataloader for images and labels

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import math

transforms = transforms.compose ({
    transforms.resize((224, 224)),
    # random changes so it doesn't learn biased (only facing one direction)
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    ## convert it to a tensor (generalized as a multidimensional array)
    transforms.ToTensor(),
    
    # normalization - (image-mean)/std
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
})

dataset = torchvision.datasets.ImageFolder(root='./train', transform = transforms)

class AssemblyDataset(Dataset):
    def __init__(self):
        # data loading
        data = np.loadtxt()
        self.images = 1
        self.masks = 1

    def __getitem__(self, index):
        # dataset[0]
        return self.images[index], self.masks[index]
  
    def __len__(self):
        #len(dataset)
        return len(self.images)

dataset = AssemblyDataset()
dataloader = DataLoader(dataset=dataset, batch_size = 4, shuffle = True, num_workers = 2)