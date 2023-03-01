"""model.py contains the UNet architecture"""
## dropout imports
import warnings
warnings.filterwarnings("ignore")

# import numpy as np
# import pandas as pd
# import time
# import h5py
# from scipy.ndimage.interpolation import rotate

# import matplotlib
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# import matplotlib.gridspec as gridspec

# import seaborn as sns

import torch
# import torchvision
# from torchvision import datasets
# from torchvision import transforms
# from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms.functional as TF
# import torch.optim as optim
# from torch.utils.data.sampler import SubsetRandomSampler

# import pymc3 as pm


class DoubleConv(nn.Module):
    ## added droprate
    def __init__(self, in_channels, out_channels, droprate):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.Dropout2d(p=droprate),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=droprate),
        )

    def forward(self, x):
        return self.conv(x)

class UNET_Dropout(nn.Module):
    def __init__(
            self, in_channels, out_channels, droprate, features=[64, 128, 256, 512],
    ):
        super(UNET_Dropout, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature, droprate))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature, droprate))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2, droprate)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    x = torch.randn((7, 3, 161, 161))
    model = UNET_Dropout(in_channels=3, out_channels=3)
    preds = model(x)
    print(preds.shape == x.shape)

if __name__ == "__main__":
    test()