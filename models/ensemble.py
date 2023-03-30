import torch
import matplotlib.pyplot as plt
import numpy as np

def ensemble_predict(test_set, loader, models, architecture, folder="saved_images/", device="cuda"):
    predictions = []
    for model in models:
        model.eval()
        model = model.to(device=device)
        for idx, (x, y) in enumerate(loader):
            x = x.to(device=device)

            with torch.no_grad():
                outputs = model(x)
            
            if architecture == "FastSCNN":
                outputs = outputs[0]

            folder = "saved_images/predictions/"
        
        # outputs = outputs.detach.numpy()
        outputs = outputs.cpu().detach().numpy()
        predictions.append(outputs)
        
    if test_set == 'assemblyhrc':
        outputs = np.average(predictions, axis=0)
        outputs = torch.from_numpy(outputs)
        # preds = torch.nn.functional.softmax(outputs, dim=1)
        # preds = torch.argmax(outputs, dim=1).detach().cpu()

        # uncomment to make it all binary classification
        preds = torch.sigmoid(outputs)
        preds = (preds>0.5).float()

        plt.imshow(preds[0])
        folder = f"./image2.jpg"
        plt.savefig(folder)

    elif test_set=="egohands":
        # outputs = outputs.detach.numpy()
        outputs = outputs.cpu().detach().numpy()
        predictions.append(outputs)

        with torch.no_grad():
            outputs = np.average(predictions, axis=0)
            outputs = torch.from_numpy(outputs)
            preds = torch.sigmoid(outputs)
            preds = (preds>0.5).float()

        plt.imshow(preds[0])
        folder = f"./image2.jpg"
        plt.savefig(folder)