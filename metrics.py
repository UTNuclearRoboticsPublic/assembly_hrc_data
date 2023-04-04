from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import BinaryCalibrationError
import numpy as np
import torch
from scipy.stats import entropy

def expected_calibration_error(predictions, targets):
    metric = BinaryCalibrationError(norm='l1')
    bce = metric(predictions, targets)
    return bce

def iou(outputs, targets, device="cuda"):
    metric = BinaryJaccardIndex().to(device=device)
    return metric(outputs, targets)

# Adaptive calibration error
# def adaptive_calibration_error(predictions, targets):
#     # make sure to change this part of the code to get average adaptive calibration error
#     predictions = predictions[0, :, :, :]
#     targets = targets[0, :, :, :]
#     predictions2 = (predictions>0.5).float().cpu().numpy()
#     predictions = predictions.cpu().numpy()
#     targets = targets.cpu().numpy()

#     ace = 0.0
#     bins = 15
#     rows = len(predictions)
#     columns = len(predictions[0])

#     # assuming sigmoid has already been completed (predictions = softmax(output, axis=1))
#     # also assuming BEFORE all predictions have been converted to 0s or 1s
    

#     # probabilities should be after the sigmoid/softmax but not before the argmax/floating
#     probabilities = predictions
#     predictions = predictions2
#     confidences = np.max(probabilities, axis=1)
#     accuracies = np.equal(predictions, targets)


#     # will never be less than zero when we take a sigmoid
#     probabilities[probabilities < 0] = 0

#     pred_matrix = np.zeros([rows, columns])
#     label_matrix = np.zeros([rows, columns])

#     print(f"datatype is {(np.arange(rows)).dtype}")
#     print(f"datatype is {(predictions).astype(np.int32).dtype}")

#     pred_matrix[np.arange(rows), predictions.astype(np.int32)] = 1
#     label_matrix[np.arange(rows), targets.astype(np.int32)] = 1

#     acc_matrix = np.equal(pred_matrix, label_matrix)

#     probabilities = probabilities[:, 0]

#     # if the probabilities array is empty (no confidence score)
#     if probabilities.size == 0:
#         boundaries = np.linspace(0, 1,  bins + 1)
#         low_bounds = boundaries[:-1]
#         high_bounds = boundaries[1:]
#     # if prob. array not empty create bounds equally spaced 
#     # with probabilities
#     else:
#         bin_n = int(rows/bins)
#         boundaries = np.array([])
#         probabilities_sort = np.sort(probabilities)
#         for i in range(0, bins):
#             boundaries = np.append(boundaries, probabilities_sort[i*bin_n])
#         boundaries = np.append(boundaries, 1.0)
#         low_bounds = boundaries[:-1]
#         high_bounds = boundaries[1:]

#     # now computing the error
#     bin_proportion = np.zeros(bins)
#     bin_accuracy = np.zeros(bins)
#     bin_confidence = np.zeros(bins)
#     bin_score = np.zeros(bins)

#     # confidences = confidences[:, 0]
#     # accuracies = accuracies[:, 0]
#     confidences = probabilities[:, 0]
#     accuracies = acc_matrix[:, 0]

#     print(f" the lower is {low_bounds.size} and high is {high_bounds.size}")

#     for i, (lower_bound, upper_bound) in enumerate(zip(low_bounds, high_bounds)):
#         # boolean masks for confidence values below upper and above lower bound
#         in_bin = np.greater(confidences, lower_bound.item()) * np.less_equal(confidences, upper_bound.item())
#         bin_proportion[i] = np.mean(in_bin)
#         # calc |acc(r, k) - conf(r, k)|
#         if bin_proportion[i].item()>0:
#             bin_accuracy[i] = np.mean(accuracies[in_bin])
#             bin_confidence[i] = np.mean(confidences[in_bin])
#             bin_score[i] = np.abs(bin_confidence[i] - bin_accuracy[i])
    
#     # bin_proportion is 1/KR
#     # bin score is the summations afterwards
#     ace += np.dot(bin_proportion, bin_score)

#     return ace

def adaptive_calibration_error(confidences: torch.Tensor,
                  true_labels: torch.Tensor,
                  n_bins: int = 15,
                  threshold: float = 1e-3) -> float:
    """
    Args:
        confidences (Tensor): a tensor of shape [N, K] of predicted confidences.
        true_labels (Tensor): a tensor of shape [N,] of ground truth labels.
        n_bins (int): the number of bins used by the histrogram binning.
        threshold (float): the value for thresholding to avoid tiny predictions.
    
    Returns:
        tace (float): thresholded adaptive calibration error of predictions.
    """

    ## How to use - 
    # preds2 = torch.reshape(preds[0, :, :, :], (161*161, 1))
    # t2 = torch.reshape(targets[0, :, :, :], (161*161,))
    # print(f"the ACE is {adaptive_calibration_error(preds2, t2)}")

    n_objects, n_classes = confidences.size()

    tace = torch.zeros(1, device=confidences.device)
    for cur_class in range(n_classes):
        cur_class_conf = confidences[:, cur_class]

        targets_sorted = true_labels[cur_class_conf.argsort()]
        cur_class_conf_sorted = cur_class_conf.sort()[0]

        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]

        bin_size = len(cur_class_conf_sorted) // n_bins
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins - 1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind
            bin_acc = (targets_sorted[bin_start_ind:bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind:bin_end_ind]
            avg_confidence_in_bin = torch.mean(bin_conf.float())
            avg_accuracy_in_bin = torch.mean(bin_acc.float())
            delta = torch.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
            tace += delta * bin_size / (n_objects * n_classes)

    return tace.item()

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