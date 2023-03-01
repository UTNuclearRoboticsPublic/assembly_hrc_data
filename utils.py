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
import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
from datetime import datetime

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

# transform3 = transforms.Compose([
#     transforms.PILToTensor()
# ])

transform3 = transforms.Compose([
    transforms.Resize(size=(1258, 1260)),
    # lambda x: x.mul(255).round().div(255)
    transforms.ToTensor(),
    lambda x: (x * 255).int().to(torch.uint8)
])

def get_loaders(batch_size):
    train_ds = AssemblyDataset(0, 3)
    train_loader = DataLoader(dataset=train_ds, batch_size = batch_size, num_workers=4, shuffle = True)

    val_ds = AssemblyDataset(0, 3)
    val_loader = DataLoader(dataset=val_ds, batch_size = batch_size, num_workers=4, shuffle = False)

    clean_val_ds = AssemblyDataset(0, 3, transform2=transform3)
    clean_val_loader = DataLoader(dataset=clean_val_ds, batch_size = batch_size, num_workers=4, shuffle = False)

    return train_loader, val_loader, clean_val_loader

def save_predictions_as_imgs(clean_loader, loader, model, folder="saved_images/", device="cuda", epochs=3, loss=0):
    model.eval()

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
        # preds= preds.swapaxes(0, 1)
        preds = preds.permute(1, 0, 2)
        print(f"shape of preds is {preds.shape}")

        hands_with_masks = [
            draw_segmentation_masks(img, masks=mask, alpha=0.5, colors=["green", "yellow", "blue"])
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

    # for idx, (x, y) in enumerate(loader):
    #     x = x.to(device=device)

    #     with torch.no_grad():
    #         outputs = model(x)
    #         preds = torch.nn.functional.softmax(outputs, dim=1)
    #         preds = torch.argmax(outputs, dim=1).detach().cpu()

    #     """ Shows distribution of the predictions"""
    #     print(f"Unique predictions are {torch.unique(preds)}")

    #     folder = "saved_images/predictions/"

    #     # to save prediciton image
    #     # torchvision.utils.save_image(
    #     #     preds, f"{folder}/pred_{idx}.png"
    #     # )

    #     """Testing shape size for fitting"""
    #     # print("hello")
    #     # print(f"y shape {torch.unique(y)}")
    #     # print(f"preds shape {preds.shape}")
    #     # print(f"preds 1 shape {preds[0].shape} and unique {torch.unique(preds[0])}")

    #     """save ground truth"""
    #     # img = TF.to_pil_image(preds)
    #     # for j in range(x.shape[0]):
    #     #     print(torch.unique(preds[1]))
    #     #     plt.imshow(preds[0])
    #     #     # plt.imshow(np.transpose(y[j], (1, 2, 0)))
    #     #     folder = f"saved_images/ground_truth/image{j}.jpg"
    #     #     plt.savefig(folder)
    #     #     plt.show()

    #     # fig.add_subplot(2, 2, 1)
    #     plt.imshow(preds[0])
    #     plt.title("Predicted Mask")

    #     folder = f"./image.jpg"
    #     plt.savefig(folder)
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
    
