from torchmetrics.classification import BinaryJaccardIndex, BinaryCalibrationError, BinaryPrecisionRecallCurve
import numpy as np
import torch
from scipy.stats import entropy

# for all of these, pass in preds after sigmoid. Tested on EgoHands test function

def expected_calibration_error(predictions, targets, device="cuda"):
    metric = BinaryCalibrationError(norm='l1').to(device=device)
    bce = metric(predictions, targets)
    return bce

def iou(outputs, targets, device="cuda"):
    metric = BinaryJaccardIndex().to(device=device)
    return metric(outputs, targets)

# metrics for the precision recall curve
def metrics_pr(predictions, targets):
        
    print(f"predictions shape is {predictions.shape}")
    num_batches = ((predictions[:, 0, 0, 0].shape))[0]
    height = ((predictions[0, 0, :, 0].shape))[0]
    width = ((predictions[0, 0, 0, :].shape))[0]

    preds = torch.reshape(predictions, (num_batches, height, width))
    targets = torch.reshape(targets, (num_batches, height, width))

    # preds = predictions

    # returns f1 score and area under curve for the precision recall curve
    pr_curve = BinaryPrecisionRecallCurve()
    precision, recall, thresholds = pr_curve(preds, targets.long())

    # doing at zero because the above evaluates at multiple thresholds. Change if you want a specific threshold.
    f1 = 2 * ((torch.mul(precision[0], recall[0])) / (torch.add(precision[0], recall[0])))
    auc = torch.trapezoid(precision, recall)

    return f1, auc

def adaptive_calibration_error(confidences: torch.Tensor,
                  true_labels: torch.Tensor,
                  n_bins: int = 15,
                  threshold: float = 0) -> float:
    """
        How to use - 
        preds2 = torch.reshape(preds[0, :, :, :], (161*161, 1))
        t2 = torch.reshape(targets[0, :, :, :], (161*161,))
        print(f"the ACE is {adaptive_calibration_error(preds2, t2)}")
        
        confidences - a tensor [N, K] of predicted probs
        true_labels- a tensor [N,] of ground truth labels
        n_bins - the num of bins used
        threshold - keep this to 0 for ACE, change for TACE
    """
    
    confidences = torch.reshape(confidences[0, :, :, :], (161*161, 1))

    true_labels = torch.reshape(true_labels[0, :, :, :], (161*161,))

    num_objects, num_classes = confidences.size()

    tace = torch.zeros(1, device=confidences.device)
    for current_class in range(num_classes):
        current_class_conf = confidences[:, current_class]

        targets_sorted = true_labels[current_class_conf.argsort()]
        current_class_conf_sorted = current_class_conf.sort()[0]

        targets_sorted = targets_sorted[current_class_conf_sorted > threshold]
        current_class_conf_sorted = current_class_conf_sorted[current_class_conf_sorted > threshold]

        bin_size = len(current_class_conf_sorted) // n_bins

        # going through and summing up each of the bins
        for bin_i in range(n_bins):
            bin_start_index = bin_i * bin_size
            if bin_i < n_bins - 1:
                bin_end_index = bin_start_index + bin_size
            else:
                bin_end_index = len(targets_sorted)
                bin_size = bin_end_index - bin_start_index
            bin_accuracy = (targets_sorted[bin_start_index:bin_end_index] == current_class)
            bin_confidence = current_class_conf_sorted[bin_start_index:bin_end_index]

            # calculating confidence and accuracy for this part of the summation
            avg_confidence_in_bin = torch.mean(bin_confidence.float())
            avg_accuracy_in_bin = torch.mean(bin_accuracy.float())

            # subtracting the two before using the summation in the bin
            delta = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            tace += delta * bin_size / (num_objects * num_classes)

    return tace.item()

    # taces  = []

    # num_batches = ((confidences[:, 0, 0, 0].shape))[0]
    
    # for i in range(num_batches):
    #     confidences = torch.reshape(confidences[i, :, :, :], (161*161, 1))

    #     true_labels = torch.reshape(true_labels[i, :, :, :], (161*161,))

    #     num_objects, num_classes = confidences.size()

    #     tace = torch.zeros(1, device=confidences.device)
    #     for current_class in range(num_classes):
    #         current_class_conf = confidences[:, current_class]

    #         targets_sorted = true_labels[current_class_conf.argsort()]
    #         current_class_conf_sorted = current_class_conf.sort()[0]

    #         targets_sorted = targets_sorted[current_class_conf_sorted > threshold]
    #         current_class_conf_sorted = current_class_conf_sorted[current_class_conf_sorted > threshold]

    #         bin_size = len(current_class_conf_sorted) // n_bins

    #         # going through and summing up each of the bins
    #         for bin_i in range(n_bins):
    #             bin_start_index = bin_i * bin_size
    #             if bin_i < n_bins - 1:
    #                 bin_end_index = bin_start_index + bin_size
    #             else:
    #                 bin_end_index = len(targets_sorted)
    #                 bin_size = bin_end_index - bin_start_index
    #             bin_accuracy = (targets_sorted[bin_start_index:bin_end_index] == current_class)
    #             bin_confidence = current_class_conf_sorted[bin_start_index:bin_end_index]

    #             # calculating confidence and accuracy for this part of the summation
    #             avg_confidence_in_bin = torch.mean(bin_confidence.float())
    #             avg_accuracy_in_bin = torch.mean(bin_accuracy.float())

    #             # subtracting the two before using the summation in the bin
    #             delta = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
    #             tace += delta * bin_size / (num_objects * num_classes)

    #         taces.append(tace)

    # return torch.mean(taces)

def shannon_entropy(probabilities):
    # right after the sigmoid
    num_batches = ((probabilities[:, 0, 0, 0].shape))[0]
    entropy_list = []

    for i in range(num_batches):
        inputs = probabilities[i, :, :, :].cpu().numpy()
        # flatten the image to a 1d array to prepare for entropy
        inputs = inputs.flatten()
        entropy2 = entropy(inputs)
        image_avg_entropy = np.average(entropy2)

        #append average entropy for this image to the entropy list
        entropy_list.append(image_avg_entropy)

    # get the average entropy across the batch
    avg_entropy_per_image = np.average(entropy_list)
    return avg_entropy_per_image

def avg_variance_per_image(probabilities):
    # check which dimension you need to do this over
    var = []
    num_batches = ((probabilities[:, 0, 0, 0].shape))[0]
    for i in range(num_batches):
        probability = probabilities[i, :, :, :].cpu().numpy()
        probability = probability.flatten()
        variance = np.var(probability)
        var.append(variance)

    return np.average(var)