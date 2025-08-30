from __future__ import print_function, division

import random

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import rotate, hflip, vflip

from src.utils.images import min_max_scaler
from src.utils.custom_transforms import apply_SOBEL_filter


class BUSI(Dataset):
    """BUSI (Breast UltraSound Image) dataset."""

    def __init__(
            self,
            mapping_file: pd.DataFrame,
            transforms=None,
            augmentations=None,
            normalization=None,
            semantic_segmentation=False
    ):
        super(BUSI, self).__init__()

        if augmentations is None:
            augmentations = {}

        self.mapping_file = mapping_file
        self.transforms = transforms
        self.semantic_segmentation = semantic_segmentation
        self.transforms_applied = {}
        self.augmentations = True if sum([v for k, v in augmentations.items()]) else False
        if augmentations:
            self.CLAHE = augmentations.get("CLAHE", False)
            self.SOBEL = augmentations.get("SOBEL", False)
            self.brightness_brighter = augmentations.get("brightness_brighter", False)
            self.brightness_darker = augmentations.get("brightness_darker", False)
            self.contrast_high = augmentations.get("contrast_high", False)
            self.contrast_low = augmentations.get("contrast_low", False)

        self.normalization = normalization

        self.data = []
        for index, row in self.mapping_file.iterrows():
            # loading image and mask
            image = cv2.imread(row['img_path'], 0)
            if semantic_segmentation:
                mask = cv2.imread(row['mask_path'], 1).transpose((2, 0, 1))
            else:
                mask = cv2.imread(row['mask_path'], 0)
                mask[mask == 255] = 1

            # loading other features
            patient_id = row['id']
            class_ = row['class']
            dim1 = row['dim1']
            dim2 = row['dim2']
            tumor_pixels = row['tumor_pixels']
            if self.semantic_segmentation:
                if class_ == 'benign':
                    label = torch.ones(1)
                elif class_ == 'normal':
                    label = torch.zeros(1)
                elif class_ == 'malignant':
                    label = 2 * torch.ones(1)
                else:
                    raise Exception(f"\n\t-> Unknown class: {row['class']}")
            else:
                if class_ == 'malignant':
                    label = torch.ones(1)
                elif class_ == 'benign':
                    label = torch.zeros(1)
                elif class_ == 'normal':
                    label = 2 * torch.ones(1)
                else:
                    raise Exception(f"\n\t-> Unknown class: {row['class']}")

            # appending information in a list
            self.data.append({
                'patient_id': patient_id,
                'label': label,
                'class_': class_,
                'image': image,
                'mask': mask,
                'dim1': dim1,
                'dim2': dim2,
                'tumor_pixels': tumor_pixels
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        patient_info = self.data[idx]

        # adding channel component is necessary
        image = torch.unsqueeze(torch.as_tensor(patient_info['image'], dtype=torch.float32), 0)
        mask = torch.as_tensor(patient_info['mask'], dtype=torch.float32)
        if not self.semantic_segmentation:
            mask = torch.unsqueeze(mask, 0)

        if self.normalization is not None:
            image = min_max_scaler(image)

        # Augmentations
        aumengs = []
        if self.augmentations and not self.semantic_segmentation:

            if self.CLAHE:
                clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(4, 4))
                aumengs.append(torch.unsqueeze(torch.as_tensor(clahe.apply(patient_info['image']),
                                                               dtype=torch.float32), 0))

            if self.SOBEL:
                aumengs.append(torch.unsqueeze(torch.as_tensor(apply_SOBEL_filter(patient_info['image']),
                                                               dtype=torch.float32), 0))

            if self.brightness_brighter:  # brightness
                brightness_matrix = np.ones(patient_info['image'].shape, dtype='uint8') * 80
                img_brighter = cv2.add(patient_info['image'], brightness_matrix)
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_brighter, dtype=torch.float32), 0))
            if self.brightness_darker:  # brightness
                brightness_matrix = np.ones(patient_info['image'].shape, dtype='uint8') * 80
                img_darker = cv2.subtract(patient_info['image'], brightness_matrix)
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_darker, dtype=torch.float32), 0))

            if self.contrast_low:  # contrast
                matrix1 = np.ones(patient_info['image'].shape) * .02
                img_low_contrast = np.uint8(cv2.multiply(np.float64(patient_info['image']), matrix1))
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_low_contrast, dtype=torch.float32), 0))
            if self.contrast_high:  # contrast
                matrix2 = np.ones(patient_info['image'].shape) * 1.5
                img_high_contrast = np.uint8(np.clip(cv2.multiply(np.float64(patient_info['image']), matrix2), 0, 255))
                aumengs.append(torch.unsqueeze(torch.as_tensor(img_high_contrast, dtype=torch.float32), 0))

        # apply transformations without augmentations
        if self.transforms is not None and not self.augmentations:
            joined = self.transforms(torch.cat([mask, image], dim=0))
            # joined, self.transforms_applied = apply_transformations(torch.cat([mask, image], dim=0), self.transforms)
            if not self.semantic_segmentation:
                mask = torch.unsqueeze(joined[0, :, :], 0)
                image = torch.unsqueeze(joined[1, :, :], 0)
            else:
                mask = joined[0:-1, :, :]
                image = torch.unsqueeze(joined[-1, :, :], 0)

        # apply transformations with augmentations
        if self.transforms is not None and self.augmentations and not self.semantic_segmentation:
            joined = torch.cat([mask, image] + aumengs, dim=0)
            joined = self.transforms(joined)
            # joined, self.transforms_applied = apply_transformations(joined, self.transforms)
            mask = torch.unsqueeze(joined[0, :, :], 0)
            image = joined[1:, :, :]

        # applying augmentation but not transformations
        if self.transforms is None and self.augmentations and not self.semantic_segmentation:
            image = torch.cat([image] + aumengs, dim=0)

        # if self.normalization is not None:
        #     # image = torch.cat([image, aug1], dim=0)
        #     image = min_max_scaler(image)

        return {
            'patient_id': patient_info['patient_id'],
            'label': patient_info['label'],
            'class': patient_info['class_'],
            'image': image,
            'mask': mask,
            'dim1': patient_info['dim1'],
            'dim2': patient_info['dim2'],
            'tumor_pixels': patient_info['tumor_pixels'],
            # 'transforms_applied': self.transforms_applied
        }


def testing_apply_transformations(image, transforms_sequential):

    # This will store the transformations applied
    transforms_applied = {'horizontal_flip': False, 'vertical_flip': False, 'rotation': 0}

    # Random horizontal flips
    if random.random() < transforms_sequential.get('horizontal_flip') != .0:
        transforms_applied['horizontal_flip'] = True
        image = hflip(image)

    # Random vertical flips
    if random.random() < transforms_sequential.get('vertical_flip') != .0:
        transforms_applied['vertical_flip'] = True
        image = vflip(image)

    # Random rotations between 0-360 degrees
    if random.random() < transforms_sequential.get('rotation'):
        # angle = random.randint(0, 360)
        angle = int(np.random.choice(range(0, 360)))
        transforms_applied['rotation'] = angle
        image = rotate(image, angle)

    return image, transforms_applied
