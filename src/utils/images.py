import numpy as np
import torch


def count_pixels(segmentation: np.array):

    unique, counts = np.unique(segmentation, return_counts=True)
    pixels_dict = dict(zip(unique, counts))

    return pixels_dict


def min_max_scaler(image: torch.tensor) -> torch.tensor:
    """ Min max scaler function for tensors."""

    min_, max_ = torch.min(image), torch.max(image)
    image = (image - min_) / (max_ - min_)

    return image


def postprocess_semantic_segmentation(segmentation):
    """
    OLD FUNCTION
    It replaced the pixels from minority class to majority class.

    """
    segmentation_postprocessed = segmentation.copy()

    counter = count_pixels(segmentation)
    benign_pixels, malignant_pixels = counter.get(1, 0), counter.get(2, 0)

    if benign_pixels >= malignant_pixels:
        segmentation_postprocessed[segmentation_postprocessed == 2] = 1
    else:
        segmentation_postprocessed[segmentation_postprocessed == 1] = 2

    return segmentation_postprocessed


def postprocess_binary_segmentation(segmentation, threshold):
    """
    OLD FUNCTION
    It replaced the pixels from minority class to majority class.

    """
    segmentation_postprocessed = segmentation.copy()

    counter = count_pixels(segmentation)
    tumor_pixels = counter.get(1, 0)

    if tumor_pixels <= threshold:
        segmentation_postprocessed[segmentation_postprocessed == 1] = 0

    return segmentation_postprocessed
