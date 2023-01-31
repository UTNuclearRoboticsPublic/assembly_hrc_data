import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import (load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 30
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
PIN_MEMORY = True
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """train_fn trains the model in model.py with the specified loader, model
    optimizer, loss function, and scaler value
    """

    loop = tqdm(loader)

    for index, batch in enumerate(loop):
        data, targets = batch
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        # for target in len(targets):
        #     target = torch.argmax(target, dim=1)
        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)
        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    model = UNET(in_channels=4, out_channels=4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(BATCH_SIZE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        if LOAD_MODEL is not True:
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":optimizer.state_dict(),
            }
            save_checkpoint(checkpoint)

            save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
if __name__ == "__main__":
    main()