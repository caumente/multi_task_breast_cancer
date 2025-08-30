#!/usr/bin/env python
# coding: utf-8

"""
Script to preprocess the BUSI Breast Ultrasound Images dataset:
- Resize images and masks
- Combine multiple masks into one
- Optionally filter using a Curated mapping CSV
- Generate mapping CSV with image info, dimensions, tumor pixels, and tumor sizes
- Logs number of masks per image for each class
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
ROOT_DATA = "./data/"
INPUT_FOLDER = "Dataset_BUSI_with_GT"

# Preprocessed BUSI dataset
RESIZE_DIMENSIONS = (128, 128)
OUTPUT_FOLDER = "Dataset_BUSI_with_GT_128"
CLASS_NAMES = ["benign", "malignant", "normal"]

# Curated BUSI dataset
CURATED = True  # Set True to keep only images in the curated mapping
CURATED_MAPPING_FILE = "./data/mapping_curated_BUSI.csv"
CURATED_OUTPUT_FOLDER = "Curated_BUSI_128"


# =======================
# Utility Functions
# =======================
def count_pixels(segmentation: np.ndarray) -> Dict[int, int]:
    unique, counts = np.unique(segmentation, return_counts=True)
    return dict(zip(unique, counts))


def size_tumor(seg: np.ndarray) -> Tuple[int, int, int, int, int, int]:
    y_indexes, x_indexes = np.nonzero(seg != 0)
    if len(y_indexes) == 0 or len(x_indexes) == 0:
        return 0, 0, 0, 0, 0, 0
    ymin, xmin = [max(0, int(np.min(idx))) for idx in (y_indexes, x_indexes)]
    ymax, xmax = [int(np.max(arr) + 1) for arr in (y_indexes, x_indexes)]
    return ymax, ymin, xmax, xmin, ymax - ymin, xmax - xmin


def load_class_dataframe(class_path: Path, class_name: str) -> pd.DataFrame:
    files = [f for f in sorted(os.listdir(class_path)) if f.endswith(".png")]
    ids = [f.replace(".png", "").split(" ")[-1].split("_")[0].replace("(", "").replace(")", "") for f in files]
    types = ["mask" if "mask" in f else "img" for f in files]
    print(f"[INFO] Loaded {len(ids)} files for class '{class_name}'")
    return pd.DataFrame({"class": [class_name] * len(ids), "ids": ids, "type": types})


def get_mask_counts(df: pd.DataFrame, class_name: str) -> List[int]:
    counts = df.groupby("ids").apply(lambda x: sum(x['type'] == 'mask')).to_dict()
    return [int(k) for k, v in counts.items() if v > 1]


def combine_and_resize_images(class_name: str, class_ids: List[str], multi_mask_ids: List[int],
                              path: Path, output_path: Path, curated_ids: List[int] = None) -> None:
    for j in set(class_ids):
        j_int = int(j)
        if curated_ids is not None and j_int not in curated_ids:
            continue

        img_path = path / class_name / f"{class_name} ({j}).png"
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path), 0)
        mask_files = [f"{class_name} ({j})_mask.png"]
        if j_int in multi_mask_ids:
            mask_files.append(f"{class_name} ({j})_mask_1.png")
        total_mask = sum(cv2.imread(str(path / class_name / f), 0) for f in mask_files)

        img = cv2.resize(img, RESIZE_DIMENSIONS, interpolation=cv2.INTER_NEAREST)
        total_mask = cv2.resize(total_mask, RESIZE_DIMENSIONS, interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(str(output_path / "images" / f"{class_name}_id_{j}.png"), img)
        cv2.imwrite(str(output_path / "masks" / f"{class_name}_id_{j}_mask.png"), total_mask)


def process_all_classes(df_classes, input_path, output_path, curated_ids):
    multi_masks_per_class = []
    for i, cls in enumerate(CLASS_NAMES):
        multi_mask_ids = get_mask_counts(df_classes[i], cls)
        multi_masks_per_class.append(multi_mask_ids)
        combine_and_resize_images(cls, df_classes[i]['ids'], multi_mask_ids, input_path, output_path, curated_ids.get(cls))
    return multi_masks_per_class


def create_mapping_csv(output_path: Path) -> pd.DataFrame:
    img_paths = sorted((output_path / "images").glob("*.png"))
    df_mapping = pd.DataFrame({"img_path": [str(p) for p in img_paths]})
    df_mapping['mask_path'] = df_mapping['img_path'].str.replace("images", "masks", regex=False).str.replace(".png", "_mask.png", regex=False)
    df_mapping['class'] = df_mapping['img_path'].apply(lambda x: Path(x).stem.split('_')[0])
    df_mapping['id'] = df_mapping['img_path'].apply(lambda x: int(Path(x).stem.split('_')[-1]))
    return df_mapping


def add_image_metadata(df_mapping: pd.DataFrame) -> pd.DataFrame:
    dims1, dims2, tumor_pixels = [], [], []
    ymaxl, yminl, xmaxl, xminl, y_sizel, x_sizel = [], [], [], [], [], []

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
    return df_mapping


# =======================
# Main
# =======================
def main():
    print("[INFO] Starting BUSI dataset preprocessing")
    print(f"[INFO] Target resize dimensions: {RESIZE_DIMENSIONS}")
    print(f"[INFO] Curated mode: {CURATED}")

    input_path = Path(ROOT_DATA) / INPUT_FOLDER
    output_path = Path(ROOT_DATA) / (CURATED_OUTPUT_FOLDER if CURATED else OUTPUT_FOLDER)
    (output_path / "images").mkdir(parents=True, exist_ok=True)
    (output_path / "masks").mkdir(parents=True, exist_ok=True)

    curated_ids_dict = {}
    if CURATED:
        curated_mapping = pd.read_csv(CURATED_MAPPING_FILE, sep=';')
        for cls in CLASS_NAMES:
            curated_ids_dict[cls] = curated_mapping[curated_mapping['class'] == cls]['id'].astype(int).tolist()
    else:
        curated_ids_dict = {cls: None for cls in CLASS_NAMES}

    df_classes = [load_class_dataframe(input_path / cls, cls) for cls in CLASS_NAMES]
    process_all_classes(df_classes, input_path, output_path, curated_ids_dict)

    df_mapping = create_mapping_csv(output_path)
    df_mapping = add_image_metadata(df_mapping)
    df_mapping.to_csv(str(output_path / "mapping.csv"), index=False)

    # Resumen final
    total_images = len(df_mapping)
    print(f"[INFO] Preprocessing completed. Processed dataset saved at {output_path}")
    print(f"[INFO] Total images processed: {total_images}")
    for cls in CLASS_NAMES:
        count_cls = len(df_mapping[df_mapping['class'] == cls])
        print(f"[INFO] {cls.capitalize()} images: {count_cls}")


if __name__ == "__main__":
    main()
