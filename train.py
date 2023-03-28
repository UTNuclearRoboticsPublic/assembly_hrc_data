import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from UNET import UNET
from UNET_Dropout import UNET_Dropout
import numpy as np
from utils import (load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs, test, create_writer, ensemble_predict)
import matplotlib as plt
from fast_scnn_model import FastSCNN

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 150
NUM_NETS = 1 # set to 1 if you don't want to use deep ensembles
NUM_WORKERS = 2
IMAGE_HEIGHT = 64
IMAGE_WIDTH = 64
PIN_MEMORY = True
LOAD_MODEL = False # decide if you want to use a saved model

## Choose model
architecture="UNET" # or "UNET_Dropout" or "FastSCNN"

## Choose how you will label the experiment
train_set = "assembly" # either "assembly" or "egohands"
test_set = "assembly" # either "assembly" or "egohands"


## Auto-set values that will be used to save your experiment
experiment_name=f"{architecture}_architecute"
model_name=f"{architecture}"
extra= f"{NUM_EPOCHS}-epochs_{train_set}-train_{test_set}-test"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    """train_fn trains the model in model.py with the specified loader, model
    optimizer, loss function, and scaler value
    """
    train_loss, train_acc = 0, 0

    loop = tqdm(loader)

    if train_set=="assembly":
        for index, batch in enumerate(loop):
            
            data, targets = batch
            data = data.to(device=DEVICE)
            targets = targets.to(device=DEVICE)
            model = model.to(device=DEVICE)

            # removed because now we are doing Binary Only

            # weight1 = torch.sum(targets==0)
            # weight2 = torch.sum(targets==1)
            # weight3 = torch.sum(targets==2)

            # weights = torch.FloatTensor([weight1, weight2, weight3])
            # weights = torch.div(1.0, weights).to(device=DEVICE)
            # loss_fn = nn.CrossEntropyLoss(weight=weights)


            with torch.cuda.amp.autocast():
                predictions = model(data)

                if architecture=="FastSCNN":
                    predictions = predictions[0]

                ## just removed long
                loss = loss_fn(predictions, targets.long())

                train_loss+=loss.item()

                predictions = torch.sigmoid(predictions)
                preds = (preds>0.5).float()
                metric = BinaryJaccardIndex()
                train_acc += metric(preds, targets)
                train_loss += loss


                ## train accuracy and loss writing
                # predictions = torch.argmax(predictions, dim=1).detach() # removed an addition .cpu() at the end
                # predictions = predictions.to(device=DEVICE)
                # metric = MulticlassJaccardIndex(num_classes=3).to(device = DEVICE)
                # train_acc += metric(predictions, targets.long()) 

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item())
    
    elif train_set == "egohands":
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)
            targets = targets[:, :, :, :, 0]/255

            # forward
            with torch.cuda.amp.autocast():
                predictions = model(data)
                loss = loss_fn(predictions, targets)

                # for train accuracy and loss tracking
                preds = torch.sigmoid(predictions)
                preds = (preds>0.5).float()
                y2 = torch.movedim(targets, 3, 1).float()
                metric = BinaryJaccardIndex()
                train_acc += metric(predictions, targets)
                train_loss += loss



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
        if architecture == "UNET_Dropout":
            net = UNET_Dropout(in_channels=3, out_channels=3, droprate=0.5)
        elif architecture == "UNET":
            net = UNET(in_channels=3, out_channels=3)
        elif architecture == "FastSCNN":
            ## adjust this when adding EgoHands to the Model
            net = FastSCNN(in_channels=3, out_channels=1).to(DEVICE)
        nets.append(net)
        optimizers.append(optim.Adam(net.parameters(), lr=LEARNING_RATE))

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader, val_loader, clean_val_loader = get_loaders(BATCH_SIZE, train_set, test_set)

    if LOAD_MODEL:
        load_checkpoint(torch.load(experiment_name + model_name + extra), net)

    scaler = torch.cuda.amp.GradScaler()
    epochs = []

    if LOAD_MODEL is not True:  
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
            test_loss, test_acc = test(architecture, val_loader, model, loss_fn)


            if LOAD_MODEL is not True:
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

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer":  optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename = experiment_name + model_name + extra)

            save_predictions_as_imgs(
            test_set, clean_val_loader, val_loader, model, architecture, folder="saved_images/", device=DEVICE
        )
    if NUM_NETS > 1:
        ensemble_predict(
            test_set, val_loader, nets, architecure, folder="saved_images/", device=DEVICE
        )
        
    writer.close()
        
if __name__ == "__main__":
    main()