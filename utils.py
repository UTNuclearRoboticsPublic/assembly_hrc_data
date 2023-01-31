import torch
import torchvision
from torch.utils.data import DataLoader
from dataloader import AssemblyDataset


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """save_checkpoint saves a checkpoint for a trained model"""
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    """load_checkpoint allows you to load a previously trained model"""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(batch_size):
    train_ds = AssemblyDataset(0, 7)
    train_loader = DataLoader(dataset=train_ds, batch_size = batch_size, num_workers=4, shuffle = True)

    val_ds = AssemblyDataset(0, 7)
    val_loader = DataLoader(dataset=val_ds, batch_size = batch_size, num_workers=4, shuffle = False)

    return train_loader, val_loader

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        # print(x.shape)
        # with torch.no_grad():
        #     preds = torch.sigmoid(model(x))
        #     preds = (preds>0.5).float()
        # y = torch.movedim(y, 3, 1)
        # torchvision.utils.save_image(y.float(), f"{folder}{idx}.png")

        # torchvision.utils.save_image(
        #     preds, f"{folder}/pred_{idx}.png"
        # )

        # with torch.no_grad():
        torchvision.utils.save_image(
            model(x), f"{folder}/pred_{idx}.png"
        )

        # print(y.shape)
    model.train()

