# dataloader for images and labels

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import math
import os
import cv2

transforms = transforms.Compose ({
    transforms.Resize((224, 224)),
    # random changes so it doesn't learn biased (only facing one direction)
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    ## convert it to a tensor (generalized as a multidimensional array)
    transforms.ToTensor(),
    
    # normalization - (image-mean)/std
    # transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
})

# dataset = torchvision.datasets.ImageFolder(root='./train', transform = transforms)

class AssemblyDataset(Dataset):
    def __init__(self):
        # data loading
        image_dir = 'images'
        image_files = os.listdir(r'./Labelled/train/images')
        self.images = [cv2.imread(os.path.join(image_dir, file)) for file in image_files if file.endswith('.png')]
        self.masks = os.path.join(r"./Labelled", r"/train/labels")

    def __getitem__(self, index):
        # dataset[0]
        return self.images[index], self.masks[index]
  
    def __len__(self):
        #len(dataset)
        return len(self.images)

dataset = AssemblyDataset()
dataloader = DataLoader(dataset=dataset, batch_size = 4, shuffle = True, num_workers = 2)
