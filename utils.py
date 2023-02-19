"""utils used in train.py"""

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from dataloader import AssemblyDataset
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

## this may be causing some errors
"""if so, remove"""
writer = SummaryWriter()

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """save_checkpoint saves a checkpoint for a trained model"""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """load_checkpoint allows you to load a previously trained model"""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

## tracking test values
def test(loader, model, loss, device="cuda"):
    model.eval()
    
    test_loss, test_acc  = 0, 0

    with torch.inference_mode():
        for idx, (X, y) in enumerate(loader):
            X, y = X.to(device=device), y.to(device=device)
            test_outputs = model(X)
            loss = loss(test_outputs, y)
            test_loss+=loss.item()

            test_preds = torch.argmax(test_outputs, dim=1).detach().cpu()
            # add test acc
        
    test_loss = test_loss/len(loader)
    test_acc = test_acc/len(loader)
    return test_loss, test_acc
    # plt.imshow(preds[0])
    # folder = f"./image.jpg"
    # plt.savefig(folder)

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
            print(f"the shape of outputs is {outputs.shape}")
            preds = torch.argmax(outputs, dim=1).detach().cpu()

        """ Shows distribution of the predictions"""
        print(f"Unique predictions are {torch.unique(preds)}")

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
        folder = f"./image.jpg"
        plt.savefig(folder)
    model.train()

def create_writer(experiment_name:str, model_name:str, extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():
    
    # Get timestamp of current date (all experiments on certain day live in same folder)
    timestamp = datetime.now().strftime("%Y-%m-%d") # returns current date in YYYY-MM-DD format

    if extra:
        # Create log directory path
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_dir = os.path.join("runs", timestamp, experiment_name, model_name)
    
    print(f"[INFO] Created SummaryWriter, saving to: {log_dir}...")
    return SummaryWriter(log_dir=log_dir)

