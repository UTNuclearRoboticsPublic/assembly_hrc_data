"""utils used in train.py"""

import torch
import torchvision
from torch.utils.data import DataLoader
from dataloader import AssemblyDataset
from torch import nn
import matplotlib.pyplot as plt
import numpy as np


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """save_checkpoint saves a checkpoint for a trained model"""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """load_checkpoint allows you to load a previously trained model"""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(batch_size):
    train_ds = AssemblyDataset(0, 3)
    train_loader = DataLoader(dataset=train_ds, batch_size = batch_size, num_workers=4, shuffle = True)

    val_ds = AssemblyDataset(0, 3)
    val_loader = DataLoader(dataset=val_ds, batch_size = batch_size, num_workers=4, shuffle = False)

    return train_loader, val_loader

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)

        with torch.no_grad():
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1).detach().cpu()

        """ Shows distribution of the predictions"""
        print(f"Unique redictions are {torch.unique(preds.flatten())}")

        folder = "saved_images/predictions/"

        # to save prediciton image
        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )

        """Testing shape size for fitting"""
        # print("hello")
        # print(f"y shape {torch.unique(y)}")
        # print(f"preds shape {preds.shape}")
        # print(f"preds 1 shape {preds[0].shape} and unique {torch.unique(preds[0])}")

        """save ground truth"""
        # img = TF.to_pil_image(preds)
        # for j in range(x.shape[0]):
        #     print(torch.unique(preds[1]))
        #     plt.imshow(preds[0])
        #     # plt.imshow(np.transpose(y[j], (1, 2, 0)))
        #     folder = f"saved_images/ground_truth/image{j}.jpg"
        #     plt.savefig(folder)
        #     plt.show()
        plt.imshow(preds[0])
        plt.show()
    model.train()
