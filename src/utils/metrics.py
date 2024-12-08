import logging
from typing import List, Tuple
from pathlib import Path
import numpy as np
import torch
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff
from typing import Union
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score as f1
from sklearn.metrics import accuracy_score


HAUSSDORF = "Haussdorf distance"
DICE = "DICE"
SENS = "Sensitivity"
SPEC = "Specificity"
ACC = "Accuracy"
JACC = "Jaccard index"
PREC = "Precision"
METRICS = [HAUSSDORF, DICE, SENS, SPEC, ACC, JACC, PREC]


def calculate_metrics(ground_truth: np.ndarray, segmentation: np.ndarray, patient: str) -> dict:
    """
    This function computes the Jaccard index, Accuracy, Haussdorf, DICE score, Sensitivity, Specificity and Precision.

    True Positive: Predicted presence of tumor and there were tumor in ground truth
    True Negative: Predicted not presence of tumor and there were no tumor in ground truth
    False Positive: Predicted presence of tumor and there were no tumor in ground truth
    False Negative: Predicted not presence of tumor and there were tumor in ground truth

    Params
    ******
        - ground_truth (torch.Tensor): Torch tensor ground truth of size 1*C*Z*Y*X
        - segmentation (torch.Tensor): Torch tensor predicted of size 1*C*Z*Y*X
        - patient (String): The patient ID

    Returns
    *******
        - metrics (List[dict]): Dict where each key represents a metric {metric:value}
    """

    assert segmentation.shape == ground_truth.shape, "Predicted segmentation and ground truth do not have the same size"

    # initializing metrics
    metrics = dict(patient_id=patient)

    # Ground truth and segmentation for region i-th (et, tc, wt)
    gt = ground_truth.astype(float)
    seg = segmentation.astype(float)

    #  cardinalities metrics tp, tn, fp, fn
    tp = float(np.sum(l_and(seg, gt)))
    tn = float(np.sum(l_and(l_not(seg), l_not(gt))))
    fp = float(np.sum(l_and(seg, l_not(gt))))
    fn = float(np.sum(l_and(l_not(seg), gt)))

    # If a region is not present in the ground truth some metrics are not defined
    # if np.sum(gt) == 0:
    #     logging.info(f"Tumor not present for {patient}")

    # Computing all metrics
    metrics[HAUSSDORF] = haussdorf_distance(gt, seg)
    metrics[DICE] = dice_score(tp, fp, fn, gt, seg)
    metrics[SENS] = sentitivity(tp, fn)
    metrics[SPEC] = specificity(tn, fp)
    metrics[ACC] = accuracy(tp, tn, fp, fn)
    metrics[JACC] = jaccard_index(tp, fp, fn, gt, seg)
    metrics[PREC] = precision(tp, fp)

    return metrics


def calculate_metrics_multiclass_segmentation(
        ground_truth: np.ndarray,
        segmentation: np.ndarray,
        patient: str,
        num_classes: int = 3,
        skip_background: bool = True,
        averaging: bool = True
) -> dict:

    assert segmentation.shape == ground_truth.shape, "Predicted segmentation and ground truth do not have the same size"

    var = 0
    if skip_background:
        var = 1

    # initializing dictionary of metrics
    metrics_dict = dict(patient_id=patient)
    for metric in METRICS:
        metrics_dict[metric] = []

    # iterating over all the regions
    for i in range(var, num_classes):
        # Ground truth and segmentation for region i-th
        gt = (ground_truth == i).astype(float)
        seg = (segmentation == i).astype(float)

        #  cardinalities metrics tp, tn, fp, fn
        tp = float(np.sum(l_and(seg, gt)))
        tn = float(np.sum(l_and(l_not(seg), l_not(gt))))
        fp = float(np.sum(l_and(seg, l_not(gt))))
        fn = float(np.sum(l_and(l_not(seg), gt)))

        # Computing all metrics
        metrics_dict[HAUSSDORF].append(haussdorf_distance(gt, seg))
        metrics_dict[DICE].append(dice_score(tp, fp, fn, gt, seg))
        metrics_dict[SENS].append(sentitivity(tp, fn))
        try:
            metrics_dict[SPEC].append(specificity(tn, fp))
        except:
            metrics_dict[SPEC].append(0)
        metrics_dict[ACC].append(accuracy(tp, tn, fp, fn))
        metrics_dict[JACC].append(jaccard_index(tp, fp, fn, gt, seg))
        metrics_dict[PREC].append(precision(tp, fp))

    if not averaging:
        return metrics_dict

    # Averaging all the classes to yield a single value for each metric
    for k in metrics_dict.keys():
        if k != "patient_id":
            metrics_dict[k] = np.nanmean(metrics_dict[k])

    return metrics_dict


def save_metrics(
        metrics: List[torch.Tensor],
        current_epoch: int,
        loss: float,
        regions: Tuple[str],
        save_folder: Path = None
):
    """
    This function is called after every validation epoch to store metrics into .txt file.


    Params:
    *******
        - metrics (torch.nn.Module): model used to compute the segmentation
        - current_epoch (int): number of current epoch
        - loss (float): averaged validation loss
        - classes (List[String]): regions to predict
        - save_folder (Path): path where the model state is saved

    Return:
    *******
        - It does not return anything. However, it generates a .txt file where the results got in the
        validation step are stored. filename = validation_error.txt
    """

    metrics = list(zip(*metrics))
    metrics = [torch.tensor(metric, device="cpu").numpy() for metric in metrics]
    metrics = {key: value for key, value in zip(regions, metrics)}
    logging.info(f"\nEpoch {current_epoch} -> "
                 f"Val: {[f'{key.upper()} : {np.nanmean(value):.4f}' for key, value in metrics.items()]} -> "
                 f"Average: {np.mean([np.nanmean(value) for key, value in metrics.items()]):.4f} "
                 f"\t Loss Average: {loss:.4f} "
                 )

    # Saving progress in a file
    with open(f"{save_folder}/validation_error.txt", mode="a") as f:
        print(f"Epoch {current_epoch} -> "
              f"Val: {[f'{key.upper()} : {np.nanmean(value):.4f}' for key, value in metrics.items()]} -> "
              f"Average: {np.mean([np.nanmean(value) for key, value in metrics.items()]):.4f}"
              f"\t Loss Average: {loss:.4f} ",
              file=f)


def sentitivity(tp: float, fn: float) -> float:
    """
    The sentitivity is intuitively the ability of the classifier to find all tumor voxels.
    """

    if tp == 0:
        sensitivity = np.nan
    else:
        sensitivity = tp / (tp + fn)

    return sensitivity


def specificity(tn: float, fp: float) -> float:
    """
    The specificity is intuitively the ability of the classifier to find all non-tumor voxels.
    """

    spec = tn / (tn + fp)

    return spec


def precision(tp: float, fp: float) -> float:

    if tp == 0:
        prec = np.nan
    else:
        prec = tp / (tp + fp)

    return prec


def accuracy(tp: float, tn: float, fp: float, fn: float) -> float:

    return (tp + tn) / (tp + tn + fp + fn)


def f1_score(tp: float, fp: float, fn: float) -> float:

    return (2 * tp) / (2 * tp + fp + fn)


def dice_score(tp: float, fp: float, fn: float, gt: np.ndarray, seg: np.ndarray) -> float:

    if np.sum(gt) == 0:
        dice = 1 if np.sum(seg) == 0 else 0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)

    return dice


def jaccard_index(tp: float, fp: float, fn: float, gt: np.ndarray, seg: np.ndarray) -> float:

    if np.sum(gt) == 0:
        jac = 1 if np.sum(seg) == 0 else 0
    else:
        jac = tp / (tp + fp + fn)

    return jac


def haussdorf_distance(gt: np.ndarray, seg: np.ndarray) -> float:
    try:
        gt = np.asarray(gt, dtype=bool)[0, 0, :, :]
        seg = np.asarray(seg, dtype=bool)[0, 0, :, :]
    except:
        pass

    if np.sum(gt) == 0 and np.sum(seg) == 0:
        hd = 0
    if (np.sum(gt) == 0 and np.sum(seg) != 0) | (np.sum(gt) != 0 and np.sum(seg) == 0):
        hd = np.nan
    else:
        hd = max(directed_hausdorff(seg, gt)[0], directed_hausdorff(gt, seg)[0])

    return hd


def dice_score_from_tensor(gt: torch.tensor, seg: torch.tensor) -> float:
    gt = gt.double()
    seg = seg.double()
    tp = torch.sum(torch.logical_and(seg, gt)).double()
    fp = torch.sum(torch.logical_and(seg, torch.logical_not(gt))).double()
    fn = torch.sum(torch.logical_and(torch.logical_not(seg), gt)).double()

    if torch.sum(gt) == 0:
        dice = 1 if torch.sum(seg) == 0 else 0
    else:
        dice = 2 * tp / (2 * tp + fp + fn)

    return dice


def accuracy_from_tensor(ground_truth, prediction) -> float:

    tp = torch.sum(torch.logical_and(prediction, ground_truth)).double()
    tn = torch.sum(torch.logical_and(torch.logical_not(prediction), torch.logical_not(ground_truth))).double()
    fp = torch.sum(torch.logical_and(prediction, torch.logical_not(ground_truth))).double()
    fn = torch.sum(torch.logical_and(torch.logical_not(prediction), ground_truth)).double()

    return (tp + tn) / (tp + tn + fp + fn)


def f1_score_from_tensor(ground_truth, prediction) -> float:

    tp = torch.sum(torch.logical_and(prediction, ground_truth)).double()
    fp = torch.sum(torch.logical_and(prediction, torch.logical_not(ground_truth))).double()
    fn = torch.sum(torch.logical_and(torch.logical_not(prediction), ground_truth)).double()

    return (2 * tp) / (2 * tp + fp + fn)


##################################################################################################################
###############################         MULTICLASS SEGMENTATION METRICS          #################################
##################################################################################################################


def DICE_coefficient_multiclass(
        prediction: Union[np.ndarray, torch.tensor],
        ground_truth: Union[np.ndarray, torch.tensor],
        num_classes: int = 3,
        skip_background: bool = True
) -> Union[float, torch.tensor]:
    """
    This function calculates the DICE coefficient for multi-class semantic segmentation.

    :param prediction: could be either arrays or tensors with shape (batch_size, num_classes, width, height)
    :param ground_truth: could be either arrays or tensors with shape (batch_size, num_classes, width, height)
    :param num_classes: number of classes within the segmentation
    :param skip_background: if true, background will not be taken into when averaging the mean DICE

    :return mean DICE over all the classes (except background if skipped)
    """

    var = 0
    if skip_background:
        var = 1

    if isinstance(prediction, np.ndarray) and isinstance(ground_truth, np.ndarray):
        dice = np.zeros(num_classes - var)
        for i in range(var, num_classes):
            mask = (prediction == i).astype(float)
            gt = (ground_truth == i).astype(float)
            intersection = np.sum(mask * gt)
            union = np.sum(mask) + np.sum(gt)
            dice[i - var] = 2.0 * intersection / union if union > 0 else 1.0

        return np.mean(dice)
    elif isinstance(prediction, torch.Tensor) and isinstance(ground_truth, torch.Tensor):
        dice = torch.zeros(num_classes - var, device=ground_truth.device)
        for i in range(var, num_classes):
            mask = (prediction == i).float()
            gt = (ground_truth == i).float()
            intersection = torch.sum(mask * gt)
            union = torch.sum(mask) + torch.sum(gt)
            dice[i - var] = 2.0 * intersection / union if union > 0 else 1.0

        return torch.mean(dice)
    else:
        raise ValueError("Inputs must be either numpy arrays or torch tensors.")


def accuracy_multiclass(
        prediction: Union[np.ndarray, torch.tensor],
        ground_truth: Union[np.ndarray, torch.tensor],
        num_classes: int = 3,
        skip_background: bool = True
) -> Union[float, torch.tensor]:

    var = 0
    if skip_background:
        var = 1

    if isinstance(prediction, np.ndarray) and isinstance(ground_truth, np.ndarray):
        acc = np.zeros(num_classes - var)
        for i in range(var, num_classes):
            mask = (prediction == i).float()
            gt = (ground_truth == i).float()

            tp = np.sum(np.logical_and(mask, gt)).astype(float)
            tn = np.sum(np.logical_and(np.logical_not(mask), np.logical_not(gt))).astype(float)
            fp = np.sum(np.logical_and(mask, np.logical_not(gt))).astype(float)
            fn = np.sum(np.logical_and(np.logical_not(mask), gt)).astype(float)

            acc[i - var] = (tp + tn) / (tp + tn + fp + fn)

        return np.mean(acc)

    elif isinstance(prediction, torch.Tensor) and isinstance(ground_truth, torch.Tensor):
        acc = torch.zeros(num_classes - var, device=ground_truth.device)
        for i in range(var, num_classes):
            mask = (prediction == i).float()
            gt = (ground_truth == i).float()

            tp = torch.sum(torch.logical_and(mask, gt)).double()
            tn = torch.sum(torch.logical_and(torch.logical_not(mask), torch.logical_not(gt))).double()
            fp = torch.sum(torch.logical_and(mask, torch.logical_not(gt))).double()
            fn = torch.sum(torch.logical_and(torch.logical_not(mask), gt)).double()

            acc[i - var] = (tp + tn) / (tp + tn + fp + fn)

        return torch.mean(acc)
    else:
        raise ValueError("Inputs must be either numpy arrays or torch tensors.")


################################################################################################################
###############################         BINARY CLASSIFICATION METRICS          #################################
################################################################################################################

def binary_classification_metrics(ground_truth, predictions):
    metrics = {}

    # getting confusion matrix
    cm = confusion_matrix(y_true=ground_truth, y_pred=predictions).ravel()
    tn, fp, fn, tp = cm.ravel()

    metrics["Precision"] = precision(tp, fp)
    metrics["Sensitivity"] = sentitivity(tp, fn)
    metrics["Specificity"] = specificity(tn, fp)
    metrics["Accuracy"] = accuracy(tp, tn, fp, fn)
    metrics["F1 score"] = f1_score(tp, fp, fn)

    return metrics


####################################################################################################################
###############################         MULTICLASS CLASSIFICATION METRICS          #################################
####################################################################################################################

def multiclass_classification_metrics(ground_truth, predictions, labels=None):
    if labels is None:
        labels = [0, 1, 2]

    precisions = calculate_precision_multiclass(ground_truth, predictions, labels)
    recalls = calculate_recall_multiclass(ground_truth, predictions, labels)
    f1_scores = calculate_f1_multiclass(ground_truth, predictions, labels)
    acc = {"accuracy": accuracy_score(ground_truth, predictions)}

    return {**precisions, **recalls, **f1_scores, **acc}


def calculate_precision_multiclass(ground_truth, predictions, labels):
    precisions = {}

    individual_precisions = precision_score(ground_truth, predictions, labels=labels, average=None)
    for n, value in enumerate(individual_precisions):
        precisions[f"precision_class_{n}"] = value

    precisions['precision_macro'] = precision_score(ground_truth, predictions, labels=labels, average='macro')
    precisions['precision_micro'] = precision_score(ground_truth, predictions, labels=labels, average='micro')
    precisions['precision_weighted'] = precision_score(ground_truth, predictions, labels=labels, average='weighted')

    return precisions


def calculate_recall_multiclass(ground_truth, predictions, labels):
    recalls = {}

    individual_recalls = recall_score(ground_truth, predictions, labels=labels, average=None)
    for n, value in enumerate(individual_recalls):
        recalls[f"recall_class_{n}"] = value

    recalls['recall_macro'] = recall_score(ground_truth, predictions, labels=labels, average='macro')
    recalls['recall_micro'] = recall_score(ground_truth, predictions, labels=labels, average='micro')
    recalls['recall_weighted'] = recall_score(ground_truth, predictions, labels=labels, average='weighted')

    return recalls


def calculate_f1_multiclass(ground_truth, predictions, labels):
    f1_scores = {}

    individual_f1_scores = f1(ground_truth, predictions, labels=labels, average=None)
    for n, value in enumerate(individual_f1_scores):
        f1_scores[f"f1_class_{n}"] = value

    f1_scores['f1_macro'] = f1(ground_truth, predictions, labels=labels, average='macro')
    f1_scores['f1_micro'] = f1(ground_truth, predictions, labels=labels, average='micro')
    f1_scores['f1_weighted'] = f1(ground_truth, predictions, labels=labels, average='weighted')

    return f1_scores
