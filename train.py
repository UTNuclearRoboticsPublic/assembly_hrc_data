import torch
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import MulticlassJaccardIndex
from torchmetrics.classification import BinaryJaccardIndex
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from models.unet import UNET
from models.unet_dropout import UNET_Dropout
from models.ensemble import ensemble_predict
import numpy as np
import yaml
from utils import (load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs, test, create_writer)
import matplotlib as plt
from models.fast_scnn_model import FastSCNN
from metrics import iou

with open("./config/config.yml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

LEARNING_RATE = config["learning_rate"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = config["batch_size"]
NUM_EPOCHS = config["num_epochs"]
NUM_NETS = config["num_nets"] # set to 1 if you don't want to use deep ensembles
NUM_WORKERS = config["num_workers"]
PIN_MEMORY = config["pin_memory"]
LOAD_MODEL = config["load_model"] # decide if you want to use a saved model

## Choose model
architecture=config["architecture"] # or "UNET_Dropout" or "FastSCNN" or "UNET"

## Choose how you will label the experiment
train_set = config["train_set"] # either "assembly" or "egohands"
test_set = config["test_set"] # either "assembly" or "egohands"

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
            # targets = targets.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)
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

                # print(f"Shape of the predictions(1) is {predictions.shape}")
                # print(f"Shape of the targets(1) is {targets.shape} and unique {targets.unique}")

                ## just removed long
                loss = loss_fn(predictions, targets)

                train_loss+=loss.item()

                predictions = torch.sigmoid(predictions)
                preds = (predictions>0.5).float()
                # metric = BinaryJaccardIndex().to(device=DEVICE)
                train_acc += iou(preds, targets, device=DEVICE)
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
                if architecture == "FastSCNN":
                    predictions = predictions[0]
                loss = loss_fn(predictions, targets)

                # for train accuracy and loss tracking
                preds = torch.sigmoid(predictions)
                preds = (preds>0.5).float()
                y2 = torch.movedim(targets, 3, 1).float()
                # metric = BinaryJaccardIndex().to(device=DEVICE)
                train_acc += iou(predictions, targets, device=DEVICE)
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

    ## Auto-set values that will be used to save your experiment
    experiment_name=f"{architecture}_architecute"
    model_name=f"{architecture}"
    extra= f"{NUM_EPOCHS}-epochs_{train_set}-train_{test_set}-test1"
    
    nets = []
    optimizers = []
    for _ in range(NUM_NETS):
        if architecture == "UNET_Dropout":
            net = UNET_Dropout(in_channels=3, out_channels=1, droprate=0.5)
        elif architecture == "UNET":
            net = UNET(in_channels=3, out_channels=1)
        elif architecture == "FastSCNN":
            ## adjust this when adding EgoHands to the Model
            net = FastSCNN(in_channels=3, out_channels=1)
        nets.append(net)
        optimizers.append(optim.Adam(net.parameters(), lr=LEARNING_RATE))

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader, val_loader, clean_val_loader = get_loaders(BATCH_SIZE, train_set, test_set)

    if LOAD_MODEL:
        extra= f"{NUM_EPOCHS}-epochs_{train_set}-train_{train_set}-test"
        load_checkpoint(torch.load(experiment_name + model_name + extra), net)

    scaler = torch.cuda.amp.GradScaler()
    epochs = []

    if LOAD_MODEL is not True:  
        results = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
            "test_ece": [],
            "test_ace": [],
            "test_f1": [],
            "test_auc": [],
            "test_entropy": [],
            "test_variance": []
            
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
            test_loss, test_acc, test_ece, test_ace, test_f1, test_auc, test_entropy, test_variance = test(architecture, val_loader, model, loss_fn, test_set, device=DEVICE)


            if LOAD_MODEL is not True:
                results["train_loss"].append(train_loss)
                results["train_acc"].append(train_acc)
                results["test_loss"].append(test_loss)
                results["test_acc"].append(test_acc)

                results["test_ece"].append(test_ece)
                results["test_ace"].append(test_ace)
                results["test_f1"].append(test_f1)
                results["test_auc"].append(test_auc)
                results["test_entropy"].append(test_entropy)
                results["test_variance"].append(test_variance)




                writer.add_scalars(main_tag="Loss", 
                            tag_scalar_dict={"train_loss": train_loss,
                                                "test_loss": test_loss},
                            global_step=epoch)

                # Add accuracy results to SummaryWriter
                writer.add_scalars( main_tag="Accuracy (IoU)", 
                                    tag_scalar_dict={"train_acc": train_acc,
                                                    "test_acc": test_acc}, 
                                    global_step=epoch)
                
                writer.add_scalars( main_tag="Calibration Error", 
                                    tag_scalar_dict={"test_ace": test_ace,
                                                    "test_ece": test_ece}, 
                                    global_step=epoch)

                writer.add_scalars( main_tag="Precision Recall Metrics", 
                    tag_scalar_dict={"test_f1": test_f1,
                                    "test_auc": test_auc}, 
                    global_step=epoch)

                writer.add_scalars( main_tag="Entropy", 
                    tag_scalar_dict={"test_entropy": test_entropy}, 
                    global_step=epoch)
                
                writer.add_scalars( main_tag="Variance", 
                    tag_scalar_dict={"variance": test_variance}, 
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
            test_set, val_loader, nets, architecture, folder="saved_images/", device=DEVICE
        )
        
    writer.close()
        
if __name__ == "__main__":
    main()