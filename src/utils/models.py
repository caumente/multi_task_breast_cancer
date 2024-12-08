import warnings
warnings.filterwarnings('ignore')
import numpy as np
import logging
import os
import pandas as pd
import cv2
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.utils.metrics import calculate_metrics
from src.utils.metrics import calculate_metrics_multiclass_segmentation
from src.utils.images import count_pixels
from src.utils.images import postprocess_semantic_segmentation
from src.utils.images import postprocess_binary_segmentation
from scipy.ndimage import binary_fill_holes


def load_pretrained_model(model: nn.Module, ckpt_path: str):
    """
    It restores a pretrained state model

    :param model: PyTorch module to be used
    :param ckpt_path: Path to the checkpoint

    :return: Model with a state loaded
    """
    if os.path.isfile(ckpt_path):
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logging.info(f"Loaded checkpoint '{ckpt_path}'. Last epoch: {checkpoint['epoch']}")
    else:
        raise ValueError(f"\n\t-> No checkpoint found at '{ckpt_path}'")

    return model


def inference_binary_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
        fill_holes: bool = True
):
    """
    It performs binary inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU
    :param fill_holes: fill holes in the predicted segmentation

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['class'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        features_map = model(test_images)
        if isinstance(features_map, list):
            for n, ds in enumerate(reversed(features_map)):
                save_features_map(seg=torch.sigmoid(ds), path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png")
            features_map = features_map[-1]  # in case that deep supervision is being used we got the last output
        else:
            save_features_map(seg=features_map, path=f"{path}/features_map/{label}_{patient_id}_seg.png")
        test_outputs = (torch.sigmoid(features_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.detach().cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()

        if fill_holes:
            test_outputs = test_outputs.astype(np.uint8)[0, 0, :, :]
            test_masks = test_masks.astype(np.uint8)[0, 0, :, :]
            test_outputs = binary_fill_holes(test_outputs).astype(int)

        # getting metrics
        metrics = calculate_metrics(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_binary_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")

    # saving metrics results
    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    return results


def inference_multilabel_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
        postprocessing: bool = False
):
    """
    It performs multilabel inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled from 0 to the number of classes.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU
    :param postprocessing: boolean to decide whether labelling all the pixels as the majority class

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class', 'predicted_class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['class'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        features_map = model(test_images)
        if isinstance(features_map, list):
            for n, ds in enumerate(reversed(features_map)):
                save_features_map(seg=ds, path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png")
            features_map = features_map[-1]  # in case that deep supervision is being used we got the last output
        else:
            save_features_map(seg=features_map, path=f"{path}/features_map/{label}_{patient_id}_seg.png")
        test_outputs = torch.nn.functional.softmax(features_map)

        # converting tensors to numpy arrays
        test_masks = torch.argmax(test_masks, dim=1, keepdim=True).float().detach().cpu().numpy()
        test_outputs = torch.argmax(test_outputs, dim=1, keepdim=True).float().detach().cpu().numpy()
        if postprocessing:
            test_outputs_postprocessed = postprocess_semantic_segmentation(test_outputs)

        # getting predicted class
        counter = count_pixels(test_outputs)
        benign_pixels, malignant_pixels = counter.get(1, 0), counter.get(2, 0)
        if benign_pixels >= malignant_pixels:
            predicted_class = 'benign'
        else:
            predicted_class = 'malignant'

        # getting segmentation metrics
        if postprocessing:
            metrics = calculate_metrics_multiclass_segmentation(test_masks, test_outputs_postprocessed, patient_id)
        else:
            metrics = calculate_metrics_multiclass_segmentation(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        metrics['predicted_class'] = predicted_class
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_multilabel_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")
        if postprocessing:
            save_multilabel_segmentation(seg=test_outputs_postprocessed,
                                         path=f"{path}/segs/{label}_{patient_id}_seg_postprocessed.png")

    # applying mapping for classification
    mapping_class = {
        'benign': 0,
        'malignant': 1
    }
    results['numerical_class'] = results['class'].map(mapping_class)
    results['numerical_class_predicted'] = results['predicted_class'].map(mapping_class)

    results.to_csv(f'{path}/results.csv', index=False)

    return results


def inference_multitask_binary_classification_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu'
):
    """
    It performs multitask inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1 as well as the image will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['class'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        pred_class, features_map = model(test_images)
        if isinstance(features_map, list):
            for n, ds in enumerate(reversed(features_map)):
                save_features_map(seg=ds, path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png")
            features_map = features_map[-1]  # in case that deep supervision is being used we got the last output
        else:
            save_features_map(seg=features_map, path=f"{path}/features_map/{label}_{patient_id}_seg.png")
        test_outputs = (torch.sigmoid(features_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.detach().cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()

        # getting metrics
        metrics = calculate_metrics(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_binary_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")

    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    # classification
    patients = []
    ground_truth_label = []
    predicted_label = []
    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['label'][0]
        test_images = test_data['image'].to(device)

        # generating segmentation
        test_outputs, segs = model(test_images)
        if isinstance(test_outputs, list):
            test_outputs = torch.mean(torch.stack(test_outputs, dim=0), dim=0)
        test_outputs = (torch.sigmoid(test_outputs) > .5).double()

        # converting tensors to numpy arrays
        patients.append(patient_id)
        ground_truth_label.append(label.detach().cpu().numpy()[0])
        predicted_label.append(test_outputs.detach().cpu().numpy()[0][0])

    # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return results, metrics


def inference_multitask_multiclass_classification_segmentation(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
        threshold: int = 0,
        overlap_seg_based_on_class: bool = False,
        overlap_class_based_on_seg: bool = False
):
    """
    It performs multitask inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1 as well as the image will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU
    :param threshold: CPU or GPU
    :param overlap_seg_based_on_class: CPU or GPU
    :param overlap_class_based_on_seg: CPU or GPU

    :return: CSV file containing the main metrics
    """

    results = pd.DataFrame(columns=['patient_id', 'Haussdorf distance', 'DICE', 'Sensitivity', 'Specificity',
                                    'Accuracy', 'Jaccard index', 'Precision', 'class'])

    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['class'][0]
        test_images = test_data['image'].to(device)
        test_masks = test_data['mask'].to(device)

        # generating segmentation
        pred_class, features_map = model(test_images)
        if isinstance(features_map, list):
            for n, ds in enumerate(reversed(features_map)):
                save_features_map(seg=ds, path=f"{path}/features_map/{label}_{patient_id}_ds_{n}.png")
            features_map = features_map[-1]  # in case that deep supervision is being used we got the last output
        else:
            save_features_map(seg=features_map, path=f"{path}/features_map/{label}_{patient_id}_seg.png")
        test_outputs = (torch.sigmoid(features_map) > .5).float()

        # converting tensors to numpy arrays
        test_masks = test_masks.detach().cpu().numpy()
        test_outputs = test_outputs.detach().cpu().numpy()

        if threshold > 0:
            test_outputs = postprocess_binary_segmentation(test_outputs, threshold)

        if overlap_seg_based_on_class:
            if isinstance(features_map, list):
                pred_class = torch.mean(torch.stack(pred_class, dim=0), dim=0)
            # prob_pred_normal = pred_class[0][0][2].item()
            pred_class = [pl.argmax() for pl in pred_class]
            if pred_class[0].item() == 2:
            # if prob_pred_normal > .85:
                test_outputs[test_outputs > 0] = 0

        # getting metrics
        metrics = calculate_metrics(test_masks, test_outputs, patient_id)
        metrics['class'] = label
        results = results.append(metrics, ignore_index=True)

        # saving segmentation
        save_binary_segmentation(seg=test_outputs, path=f"{path}/segs/{label}_{patient_id}_seg.png")

    results.to_csv(f'{path}/results_segmentation.csv', index=False)

    # classification
    patients = []
    ground_truth_label = []
    predicted_label = []
    predicted_probabilities = []
    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        # label = test_data['label'][0]
        test_label = test_data['label'].to(device)
        test_label = torch.nn.functional.one_hot(test_label.flatten().to(torch.int64), num_classes=3).to(torch.float)
        test_images = test_data['image'].to(device)

        # generating segmentation
        test_outputs, segs = model(test_images)
        if isinstance(segs, list):
            test_outputs = torch.mean(torch.stack(test_outputs, dim=0), dim=0)
        test_label = [l.argmax() for l in test_label]
        predicted_probabilities.append(test_outputs.detach().cpu().numpy().tolist()[0])
        test_outputs = [pl.argmax() for pl in test_outputs]

        # counting tumor pixels
        segs = (torch.sigmoid(segs[-1]) > .5).float()
        counter_tumor_pixels = count_pixels(segs.detach().cpu().numpy()).get(1, 0)

        # converting tensors to numpy arrays
        patients.append(patient_id)
        if len(test_outputs) > 1:
            for r, p in zip(test_label, test_outputs):
                if overlap_class_based_on_seg and counter_tumor_pixels == 0:
                    ground_truth_label.append(int(r.detach().cpu().numpy()[0]))
                    predicted_label.append(2)
                else:
                    ground_truth_label.append(int(r.detach().cpu().numpy()[0]))
                    predicted_label.append(int(p.detach().cpu().numpy()[0]))
        else:
            if overlap_class_based_on_seg and counter_tumor_pixels == 0:
                ground_truth_label.append(int(test_label[0].detach().cpu().numpy()))
                predicted_label.append(2)
            else:
                ground_truth_label.append(int(test_label[0].detach().cpu().numpy()))
                predicted_label.append(int(test_outputs[0].detach().cpu().numpy()))

    # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })
    metrics[["prob_benign", "prob_malignant", "prob_normal"]] = predicted_probabilities
    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return results, metrics


def inference_multiclass_classification(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu',
):
    """
    It performs multitask inference over PyTorch dataloader by means of a trained model. It means that pixels will be
    labeled as 0 or 1 as well as the image will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    # classification
    patients = []
    ground_truth_label = []
    predicted_label = []
    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        test_label = test_data['label'].to(device)
        test_label = torch.nn.functional.one_hot(test_label.flatten().to(torch.int64), num_classes=3).to(torch.float)
        test_images = test_data['image'].to(device)

        # generating segmentation
        test_outputs = model(test_images)
        if isinstance(test_outputs, list):
            test_outputs = torch.mean(torch.stack(test_outputs, dim=0), dim=0)
        test_label = [l.argmax() for l in test_label]
        test_outputs = [pl.argmax() for pl in test_outputs]

        # converting tensors to numpy arrays
        patients.append(patient_id)
        if len(test_outputs) > 1:
            for r, p in zip(test_label, test_outputs):
                ground_truth_label.append(int(r.detach().cpu().numpy()[0]))
                predicted_label.append(int(p.detach().cpu().numpy()[0]))
        else:
            ground_truth_label.append(int(test_label[0].detach().cpu().numpy()))
            predicted_label.append(int(test_outputs[0].detach().cpu().numpy()))

    # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results_classification.csv', index=False)

    return metrics


def inference_binary_classification(
        model: torch.nn.Module,
        test_loader: torch.utils.data.DataLoader,
        path: str,
        device: str = 'cpu'
):
    """
    It performs binary classification inference over PyTorch dataloader by means of a trained model. It means the image
    will be classified as benign or malignant.

    :param model: PyTorch module used to evaluate the images
    :param test_loader: Test dataloader to be evaluated
    :param path: path to store the segmentations
    :param device: CPU or GPU

    :return: CSV file containing the main metrics
    """

    patients = []
    ground_truth_label = []
    predicted_label = []
    for i, test_data in enumerate(test_loader):

        # load information from patient
        patient_id = test_data['patient_id'].item()
        label = test_data['label'][0]
        test_images = test_data['image'].to(device)

        # generating segmentation
        test_outputs = model(test_images)
        test_outputs = (torch.sigmoid(test_outputs) > .5).double()

        # converting tensors to numpy arrays
        patients.append(patient_id)
        ground_truth_label.append(label.detach().cpu().numpy()[0])
        predicted_label.append(test_outputs.detach().cpu().numpy()[0][0])

        # getting metrics
    metrics = pd.DataFrame({
        'patient_id': patients,
        'ground_truth': ground_truth_label,
        'predicted_label': predicted_label
    })

    metrics.to_csv(f'{path}/results.csv', index=False)

    return metrics


def save_binary_segmentation(seg: np.array, path: str, value_non_zero: int = 255):
    """
    It saves a NumPy array as a binary image

    :param seg: Image to be saved
    :param path: path to save the image
    :param value_non_zero: value to assign all non-zero values. Typically, it will be 255 or 1.
    """

    n_dims = len(seg.shape)

    assert n_dims <= 4, "Numpy array must have less than 5 dimensions to be able to be stored"

    if n_dims == 4:
        seg = seg[0, 0, :, :].astype(int)
    elif n_dims == 3:
        seg = seg[0, :, :].astype(int)

    seg[seg > 0] = value_non_zero
    cv2.imwrite(path, seg)


def save_multilabel_segmentation(seg: np.array, path: str):
    """
    It saves a NumPy array as multilabel image

    :param seg: Image to be saved
    :param path: path to save the image
    """

    n_dims = len(seg.shape)

    assert n_dims <= 4, "Numpy array must have less than 5 dimensions to be able to be stored"

    if n_dims == 4:
        seg = seg[0, 0, :, :].astype(int)
    elif n_dims == 3:
        seg = seg[0, :, :].astype(int)

    if len(seg.shape) == 2:
        seg = seg.astype(int)
        cv2.imwrite(path, seg)
    else:
        seg = seg[0, 0, :, :].astype(int)
        cv2.imwrite(path, seg)


def save_features_map(seg: np.array, path: str):
    seg = seg.detach().cpu().numpy()
    seg = seg[0, 0, :, :].astype(float)
    cv2.imwrite(path, seg)


def count_parameters(model: torch.nn.Module) -> int:
    """
    This function counts the trainable parameters of a model.

    :param model: Torch model
    :return: Number of parameters
    """

    return sum(p.numel() for p in model.parameters() if p.requires_grad)
