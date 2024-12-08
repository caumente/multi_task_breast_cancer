import os
import json
import torch
import random
import pprint
import logging
import argparse
import numpy as np
import pandas as pd
import glob
import yaml
from pprint import pformat
from src.utils.metrics import binary_classification_metrics
from src.utils.metrics import multiclass_classification_metrics


def load_config_file(path: str):
    """
    This function load a config file and return the different sections.


    Parameters
    ----------
    path: Path where the config file is stored

    """
    with open(path) as cf:
        config = yaml.load(cf, Loader=yaml.FullLoader)
        logging.info(pformat(config))
    return config['model'], config['optimizer'], config['loss'], config['training'], config['data']


def save_args(args: argparse.Namespace):
    """
    This function saves parsed arguments into config file.


    Parameters
    ----------
    args (dict{arg:value}): Arguments for this run

    """

    config = vars(args).copy()
    del config['save_folder'], config['seg_folder']

    logging.info(f"Execution for configuration:")
    pprint.pprint(config)

    config_file = args.save_folder / "config_file.json"
    with config_file.open("w") as file:
        json.dump(config, file, indent=4)


def init_log(log_name: str):
    """
    This function initializes a log file.

    Params:
    *******
        - log_name (str): log name

    """
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] - [%(levelname)s] - [%(filename)s:%(lineno)s] --- %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_name,
        filemode='a',
        force=True
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)


def seed_everything(seed: int, cuda_benchmark: bool = False):
    """
    This function initializes all the seeds

    Params:
    *******
        - seed: seed number
        - cuda_benchmark: flag to activate/deactivate CUDA optimization algorithms

    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = cuda_benchmark


def save_segmentation_results(path: str):
    """
    This function combines the segmentation results obtained on each fold into a single file (segmentation_results.xlsx)

    Params:
    *******
        - path: root path of the experiment

    """
    results = []
    for n, f in enumerate(sorted(glob.glob(path + "/fold*/results_segmentation.csv"))):
        df = pd.read_csv(f)
        df["fold"] = n
        results.append(df)

    df = pd.concat(results)
    df_grouped = df.drop(columns="patient_id").groupby('fold').mean().reset_index().drop(columns='fold').T
    df_grouped.columns = [f"fold {c}" for c in df_grouped.columns]
    df_grouped["mean"] = df_grouped.mean(axis=1)
    df_grouped["std"] = df_grouped.std(axis=1)
    df_grouped["latex"] = (round(df_grouped["mean"], 3).astype(str).str.ljust(5, '0') + " $\\pm$ " +
                           round(df_grouped["std"], 3).astype(str).str.ljust(5, '0'))
    df_grouped.to_excel(path + '/results_segmentation.xlsx', index=False)


def save_classification_results(path: str, n_classes: int):
    """
    This function combines the segmentation results obtained on each fold into a single file (segmentation_results.xlsx)

    Params:
    *******
        - path: root path of the experiment
        - n_classes: it defines whether the problem is binary or multiclass classification

    """
    results, metric = [], {}
    for n, f in enumerate(sorted(glob.glob(path + "/fold*/results_classification.csv"))):
        df = pd.read_csv(f)

        if n_classes <= 2:
            metric = binary_classification_metrics(df.ground_truth, df.predicted_label)
        else:
            metric = multiclass_classification_metrics(df.ground_truth, df.predicted_label)

        results.append(pd.DataFrame([metric]))

    df_grouped = pd.concat(results).T
    df_grouped.columns = [f"fold {c}" for c in df_grouped.columns]
    df_grouped["mean"] = df_grouped.mean(axis=1)
    df_grouped["std"] = df_grouped.std(axis=1)
    df_grouped["latex"] = (round(df_grouped["mean"], 3).astype(str).str.ljust(5, '0') + " $\\pm$ " +
                           round(df_grouped["std"], 3).astype(str).str.ljust(5, '0'))

    df_grouped.to_excel(path + '/classification_results.xlsx', index=True)


def write_metrics_file(path_file, text_to_write, close=True):
    """
    This function allows us to write text in a given file

    Params:
    *******
        - path_file: path to the file
        - text_to_write: line to writen within the file

    """
    with open(path_file, 'a') as fm:
        fm.write(text_to_write)
        fm.write("\n")
        if close:
            fm.close()
