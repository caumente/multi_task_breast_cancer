#!/usr/bin/env python
# coding: utf-8

"""
Script to preprocess the BUSI Breast Ultrasound Images dataset:
- Resize images and masks
- Combine multiple masks into one
- Optionally filter using a curated mapping CSV
- Generate mapping CSV with image info, dimensions, tumor pixels, and tumor sizes
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import cv2

# =======================
# User-Configurable Settings
# =======================
ROOT_DATA = "./../data/"
INPUT_FOLDER = "Dataset_BUSI_with_GT"

# Preprocessed BUSI dataset
RESIZE_DIMENSIONS = (128, 128)
OUTPUT_FOLDER = "Dataset_BUSI_with_GT_128"
CLASS_NAMES = ["benign", "malignant", "normal"]

# Curated BUSI dataset
CURATED = False  # Set True to keep only images in the curated mapping
CURATED_MAPPING_FILE = "./../data/mapping_curated_BUSI.csv"
CURATED_OUTPUT_FOLDER = "Curated_BUSI_128"

# =======================
# Utility Functions
# =======================
def count_pixels(segmentation: np.ndarray) -> Dict[int, int]:
    """Count pixels by value in a mask"""
    unique, counts = np.unique(segmentation, return_counts=True)
    return dict(zip(unique, counts))


def size_tumor(seg: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    """Calculate tumor bounding box and size from a mask"""
    y_indexes, x_indexes = np.nonzero(seg != 0)
    if len(y_indexes) == 0 or len(x_indexes) == 0:
        return 0,0,0,0,0,0
    ymin, xmin = [max(0, int(np.min(idx))) for idx in (y_indexes, x_indexes)]
    ymax, xmax = [int(np.max(arr)+1) for arr in (y_indexes, x_indexes)]
    return ymax, ymin, xmax, xmin, ymax - ymin, xmax - xmin


def load_class_dataframe(class_path: Path, class_name: str) -> pd.DataFrame:
    """Load filenames and classify them as mask or image"""
    files = sorted(os.listdir(class_path))
    ids = [f.replace(".png", "").split(" ")[-1].split("_")[0].replace("(", "").replace(")", "") for f in files]
    types = ["mask" if "mask" in f else "img" for f in files]
    return pd.DataFrame({"class": [class_name] * len(ids), "ids": ids, "type": types})


def combine_and_resize_images(class_name: str, class_ids: List[str], two_masks_ids: List[int],
                              path: Path, output_path: Path, curated_ids: List[int] = None) -> None:
    """Combine multiple masks, resize, and optionally filter by curated IDs"""
    for j in set(class_ids):
        j_int = int(j)
        if curated_ids is not None and j_int not in curated_ids:
            continue  # skip if not in curated list

        img_path = path / class_name / f"{class_name} ({j}).png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), 0)
        mask_files = [f"{class_name} ({j})_mask.png"]
        if j_int in two_masks_ids:
            mask_files.append(f"{class_name} ({j})_mask_1.png")

        total_mask = sum(cv2.imread(str(path / class_name / f), 0) for f in mask_files)

        # Resize
        img = cv2.resize(img, RESIZE_DIMENSIONS, interpolation=cv2.INTER_NEAREST)
        total_mask = cv2.resize(total_mask, RESIZE_DIMENSIONS, interpolation=cv2.INTER_NEAREST)

        # Save
        cv2.imwrite(str(output_path / "images" / f"{class_name}_id_{j}.png"), img)
        cv2.imwrite(str(output_path / "masks" / f"{class_name}_id_{j}_mask.png"), total_mask)


# =======================
# Main Processing
# =======================
def main():
    input_path = Path(ROOT_DATA) / INPUT_FOLDER
    output_path = Path(ROOT_DATA) / (CURATED_OUTPUT_FOLDER if CURATED else OUTPUT_FOLDER)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)

    # Load curated IDs if needed
    curated_ids_dict = {}
    if CURATED:
        curated_mapping = pd.read_csv(CURATED_MAPPING_FILE, sep=';')
        for cls in CLASS_NAMES:
            curated_ids_dict[cls] = curated_mapping[curated_mapping['class'] == cls]['id'].astype(int).tolist()
    else:
        curated_ids_dict = {cls: None for cls in CLASS_NAMES}

    # Load class dataframes
    df_classes = [load_class_dataframe(input_path / cls, cls) for cls in CLASS_NAMES]

    # Find images with two masks
    two_masks_b = sorted(df_classes[0].groupby("ids").filter(lambda x: x['ids'].count() == 3)["ids"].astype(int).unique())
    two_masks_m = sorted(df_classes[1].groupby("ids").filter(lambda x: x['ids'].count() == 3)["ids"].astype(int).unique())

    # Process each class
    combine_and_resize_images("benign", df_classes[0]['ids'], two_masks_b, input_path, output_path, curated_ids_dict.get("benign"))
    combine_and_resize_images("malignant", df_classes[1]['ids'], two_masks_m, input_path, output_path, curated_ids_dict.get("malignant"))
    combine_and_resize_images("normal", df_classes[2]['ids'], [], input_path, output_path, curated_ids_dict.get("normal"))

    # Create mapping CSV only for existing files
    img_paths = sorted((output_path / "images").glob("*.png"))
    df_mapping = pd.DataFrame({"img_path": [str(p) for p in img_paths]})
    df_mapping['mask_path'] = df_mapping['img_path'].str.replace("images", "masks").str.replace(".png", "_mask.png")

    df_mapping['class'] = df_mapping['img_path'].apply(lambda x: Path(x).stem.split('_')[0])
    df_mapping['id'] = df_mapping['img_path'].apply(lambda x: int(Path(x).stem.split('_')[-1]))

    # Add dimensions, tumor pixels, and tumor size
    dims1, dims2, tumor_pixels = [], [], []
    ymaxl, yminl, xmaxl, xminl, y_sizel, x_sizel = [],[],[],[],[],[]

    for img_path, mask_path in zip(df_mapping['img_path'], df_mapping['mask_path']):
        img = cv2.imread(img_path, 0)
        dims1.append(img.shape[0])
        dims2.append(img.shape[1])

        mask = cv2.imread(mask_path, 0)
        counting = count_pixels(mask)
        tumor_pixels.append(counting.get(255, 0))

        ymax, ymin, xmax, xmin, y_size, x_size = size_tumor(mask)
        ymaxl.append(ymax)
        yminl.append(ymin)
        xmaxl.append(xmax)
        xminl.append(xmin)
        y_sizel.append(y_size)
        x_sizel.append(x_size)

    df_mapping['dim1'] = dims1
    df_mapping['dim2'] = dims2
    df_mapping['tumor_pixels'] = tumor_pixels
    df_mapping['y_max'] = ymaxl
    df_mapping['y_min'] = yminl
    df_mapping['x_max'] = xmaxl
    df_mapping['x_min'] = xminl
    df_mapping['y_size'] = y_sizel
    df_mapping['x_size'] = x_sizel

    df_mapping = df_mapping.sort_values(by=['class', 'id'])
    df_mapping.to_csv(str(output_path / "mapping.csv"), index=False)
    print(f"Processed dataset saved at {output_path}")


if __name__ == "__main__":
    main()
