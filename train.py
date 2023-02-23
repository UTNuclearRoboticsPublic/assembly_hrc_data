import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from model import UNET
import numpy as np
from utils import (load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs, test, create_writer)
import matplotlib as plt

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

experiment_name="unet_TEST3percent"
model_name="UNET_Dropout"
extra="200_epochsTHISTIME"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """train_fn trains the model in model.py with the specified loader, model
    optimizer, loss function, and scaler value
    """
    train_loss, train_acc = 0, 0

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

        

        with torch.cuda.amp.autocast():
            predictions = model(data)

            ## just removed long
            loss = loss_fn(predictions, targets.long())
            losses.append(loss.cpu().detach().numpy())

            train_loss+=loss.item()

            ## train accuracy and loss writing
            predictions = torch.argmax(predictions, dim=1).detach()
            metric = MulticlassJaccardIndex(num_classes=3).to(device = DEVICE)
            train_acc += metric(predictions, targets.long()) 

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    train_loss = train_loss/len(loader)
    train_acc = train_acc / len(loader)

    return train_loss, train_acc



def main():
    nets = []
    optimizers = []
    for _ in range(NUM_NETS):
        net = UNET(in_channels=3, out_channels=3, droprate=0.5)
        nets.append(net)
        optimizers.append(optim.Adam(net.parameters(), lr=LEARNING_RATE))


    loss_fn = nn.CrossEntropyLoss()

    train_loader, val_loader = get_loaders(BATCH_SIZE)

    if LOAD_MODEL:
        load_checkpoint(torch.load(experiment_name + model_name + extra), model)

    scaler = torch.cuda.amp.GradScaler()
    epochs = []

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    # Create an example writer
    writer = create_writer(
        experiment_name,
        model_name,
        extra
    )

    for i, net in enumerate(nets):
        optimizer = optimizers[i]
        model = nets[i].to(DEVICE)
        for epoch in range(NUM_EPOCHS):
            if LOAD_MODEL is not True:
                model.train()
                train_loss, train_acc = train_fn(train_loader, model, optimizer, loss_fn, scaler)
            test_loss, test_acc = test(val_loader, model, loss_fn)

            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            writer.add_scalars(main_tag="Loss", 
                           tag_scalar_dict={"train_loss": train_loss,
                                            "test_loss": test_loss},
                           global_step=epoch)

            # Add accuracy results to SummaryWriter
            writer.add_scalars( main_tag="Accuracy", 
                                tag_scalar_dict={"train_acc": train_acc,
                                                "test_acc": test_acc}, 
                                global_step=epoch)

            writer.close()

                # save model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer":  optimizer.state_dict(),
                }
                save_checkpoint(checkpoint, filename = experiment_name + model_name + extra)

            save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )
    writer.close()
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