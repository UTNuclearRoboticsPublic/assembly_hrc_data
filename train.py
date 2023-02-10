import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
import numpy as np
from utils import (load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs)
import matplotlib.pyplot as plt

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 200
NUM_NETS = 2
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
PIN_MEMORY = True
LOAD_MODEL = False
losses = []

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """train_fn trains the model in model.py with the specified loader, model
    optimizer, loss function, and scaler value
    """

    loop = tqdm(loader)

    for index, batch in enumerate(loop):
        data, targets = batch
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)
        model = model.to(device=DEVICE)

        weight1 = torch.sum(targets==0)
        weight2 = torch.sum(targets==1)
        weight3 = torch.sum(targets==2)

        weights = torch.FloatTensor([weight1, weight2, weight3])
        weights = torch.div(1.0, weights).to(device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        # print(f"shape is {targets.shape}") #checking shape

        # print(f"unique targets are {torch.unique(targets)}") #checking shape

        with torch.cuda.amp.autocast():
            predictions = model(data)

            # checking size when fixing tensor shape errors
            # print(f"Shape of targets {(targets.shape)}") # 2 (batch), 224, 224

            ## the target should be a LongTensor with the shape [batch_size, height, width] 
            ## and contain the class indices for each pixel location in the range [0, nb_classes-1] 

            # print(f"shape of predictions {predictions.shape}") # 2 (batch), 3, 224, 224 (this is good)
            # print(f"shape of predictions {torch.unique(predictions)}")

            ## just removed long
            loss = loss_fn(predictions, targets.long())
            losses.append(loss.cpu().detach().numpy())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    # model = UNET(in_channels=3, out_channels=3).to(DEVICE)
    nets = []
    optimizers = []
    for _ in range(NUM_NETS):
        net = UNET(in_channels=3, out_channels=3)
        nets.append(net)
        optimizers.append(optim.Adam(net.parameters(), lr=LEARNING_RATE))


    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader = get_loaders(BATCH_SIZE)

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    scaler = torch.cuda.amp.GradScaler()
    epochs = []

    for i, net in enumerate(nets):
        optimizer = optimizers[i]
        model = nets[i].to(DEVICE)
        for epoch in range(NUM_EPOCHS):
            if LOAD_MODEL is not True:
                model.train()
                train_fn(train_loader, model, optimizer, loss_fn, scaler)

                # save model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                save_checkpoint(checkpoint)

                save_predictions_as_imgs(
                val_loader, model, folder="saved_images/", device=DEVICE, 
            )
            epochs.append(epoch)
            plt.clf()
            if epoch%10 == 0 :
                plt.plot(epochs, losses, label="Train loss")
                plt.title("Training Loss Curve")
                plt.ylabel("Loss")
                plt.xlabel("Epochs")
                plt.show()
        
if __name__ == "__main__":
    main()