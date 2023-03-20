"""utils used in train.py"""

import torch
import torchvision
from torchmetrics.classification import MulticlassJaccardIndex
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision.utils import draw_segmentation_masks
from torch.utils.data import DataLoader
from dataloader import AssemblyDataset
from torch import nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
from datetime import datetime
import dill

from EgoHands_Dataset.get_meta_by import get_meta_by
from EgoHands_Dataset.dataset import EgoHandsDataset


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
    metric = MulticlassJaccardIndex(num_classes=3).to(device=device)

    with torch.inference_mode():
        for idx, (X, y) in enumerate(loader):
            X, y = X.to(device=device), y.to(device=device)
            test_outputs = model(X)
            batch_loss = loss(test_outputs, y.long())
            test_loss+=batch_loss.item()

            test_outputs = torch.argmax(test_outputs, dim=1).detach() ## removed .cpu
            test_acc += metric(test_outputs, y.long()) 
            # add test acc
        
    test_loss = test_loss/len(loader)
    test_acc = test_acc/len(loader)
    return test_loss, test_acc
    # plt.imshow(preds[0])
    # folder = f"./image.jpg"
    # plt.savefig(folder)

def to_uint8(x):
    return (x * 255).int().to(torch.uint8)

transform3 = transforms.Compose([
    transforms.Resize(size=(1258, 1260)),
    transforms.ToTensor(),
    dill.loads(dill.dumps(to_uint8))
])

def get_loaders(batch_size, train_set, test_set):
    if train_set=="assembly":
        train_ds = AssemblyDataset(0, 3)
        train_loader = DataLoader(dataset=train_ds, batch_size = batch_size, num_workers=4, shuffle = True)

    elif train_set=="egohands":
            
        # we should change these transforms to what we will mention in the paper as not to skew the data
        train_transform = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Rotate(limit=35, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )
        
        # training dataset
        train_ds = EgoHandsDataset(
            get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'B', 'Partner', 'S'),
            train_transform
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
        )

    if test_set=="assembly":
        val_ds = AssemblyDataset(0, 3)
        val_loader = DataLoader(dataset=val_ds, batch_size = batch_size, num_workers=4, shuffle = False)

        clean_val_ds = AssemblyDataset(0, 3, transform2=transform3)
        clean_val_loader = DataLoader(dataset=clean_val_ds, batch_size = batch_size, num_workers=4, shuffle = False)

    elif test_set == "egohands":
        # validation dataset

        IMAGE_HEIGHT = 90
        IMAGE_WIDTH = 160

        val_transforms = A.Compose(
            [
                A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ],
        )

        val_ds = EgoHandsDataset(
            # switched S and B
            get_meta_by('Location', 'COURTYARD', 'Activity', 'PUZZLE', 'Viewer', 'S', 'Partner', 'B'),
            val_transforms
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True,
            shuffle=False,
        )

    return train_loader, val_loader, clean_val_loader

def save_predictions_as_imgs(train_set, clean_loader, loader, model, folder="saved_images/", device="cuda", epochs=3, loss=0):
    model.eval()
    if train_set == "assembly":
        for idx, (loader_item, clean_loader_item) in enumerate(zip(loader, clean_loader)):
            x1, y1 = clean_loader_item
            x, y = loader_item
            x = x.to(device=device)

            outputs = model(x)
            outputs = F.interpolate(outputs, size=(1258, 1260), mode = 'nearest')


            with torch.no_grad():
                preds = torch.nn.functional.softmax(outputs, dim=1)
                preds = torch.argmax(preds, dim=1).detach().cpu()

            print(f"shape of preds is {preds.shape}")

            preds = preds.cpu()

            preds = (preds == torch.arange(4)[:, None, None])
            preds = ~preds.bool()

            # preds= preds.swapaxes(0, 1)
            # preds = preds.permute(1, 0, 2)
            print(f"shape of preds is {preds.shape}")

            hands_with_masks = [
                draw_segmentation_masks(img, masks=mask, alpha=0.4, colors=["red"])
                for img, mask in zip(x1, preds)
            ]

            images = [
                img.permute(2, 0, 1)
                for img, mask in zip(x1, preds)
            ]

            masks = [
                mask
                for img, mask in zip(x1, y)
            ]

            fig, axs = plt.subplots(1, 3, figsize=(10, 5))
            axs[1].imshow(hands_with_masks[0].permute(1, 2, 0))
            axs[1].set_title('Predicted Segmentation Mask')
            axs[1].axis('off')

            axs[0].imshow(images[0].permute(2, 0, 1))
            axs[0].set_title('Image')
            axs[0].axis('off')

            axs[2].imshow(masks[0])
            axs[2].set_title('Ground Truth Mask')
            axs[2].axis('off')

            # plt.imshow(preds[0,:, :, :].permute(1, 2, 0), alpha = 0.6)
            plt.savefig("img3.jpg", dpi=300)
            plt.show()

    elif train_set=="egohands":
        # write code for when merged with EgoHands here
        model.eval()
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)
            print(x.shape)
            with torch.no_grad():
                preds = torch.sigmoid(model(x))
                preds = (preds>0.5).float()
            y = torch.movedim(y, 3, 1)
            torchvision.utils.save_image(y.float(), f"{folder}{idx}.png")

            torchvision.utils.save_image(
                preds, f"{folder}/pred_{idx}.png"
            )

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


def ensemble_predict(loader, models, folder="saved_images/", device="cuda"):
    predictions = []
    for model in models:
        model.eval()
        model = model.to(device=device)
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)

            with torch.no_grad():
                outputs = model(x)

            folder = "saved_images/predictions/"
        
        # outputs = outputs.detach.numpy()
        outputs = outputs.cpu().detach().numpy()
        predictions.append(outputs)
        
    outputs = np.average(predictions, axis=0)
    outputs = torch.from_numpy(outputs)
    preds = torch.nn.functional.softmax(outputs, dim=1)
    preds = torch.argmax(outputs, dim=1).detach().cpu()

    plt.imshow(preds[0])
    folder = f"./image2.jpg"
    plt.savefig(folder)
    