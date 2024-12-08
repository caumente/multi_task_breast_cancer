import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms.v2 import RandomResizedCrop, ElasticTransform

from src.dataset.BUSI_dataloader import load_datasets
from src.utils.metrics import dice_score_from_tensor
from src.utils.miscellany import init_log
from src.utils.miscellany import seed_everything
from src.utils.models import inference_binary_segmentation
from src.utils.models import load_pretrained_model
from src.utils.visualization import plot_evolution
from src.utils.miscellany import save_segmentation_results
from src.utils.miscellany import load_config_file
from src.utils.miscellany import write_metrics_file
from src.utils.criterions import apply_criterion_binary_segmentation
from src.utils.experiment_init import load_segmentation_experiment_artefacts
from src.utils.experiment_init import device_setup
from monai.transforms import MaskIntensity, HistogramNormalize, ThresholdIntensity


def train_one_epoch():
    training_loss = 0.
    running_dice = 0.

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, masks = data['image'].to(dev), data['mask'].to(dev)

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        batch_loss = apply_criterion_binary_segmentation(criterion=criterion, ground_truth=masks, segmentation=outputs,
                                                         inversely_weighted=config_loss['inversely_weighted'])
        training_loss += batch_loss.item()

        # Performing backward step through scaler methodology
        batch_loss.backward()
        optimizer.step()

        # measuring DICE
        if isinstance(outputs, list):
            outputs = outputs[-1]
        outputs = torch.sigmoid(outputs) > .5
        dice = dice_score_from_tensor(masks, outputs)
        running_dice += dice

        del batch_loss
        del outputs

    return training_loss / training_loader.__len__(), running_dice / training_loader.__len__()


# initializing times
init_time = time.perf_counter()
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# loading config file
config_model, config_opt, config_loss, config_training, config_data = load_config_file(path='./src/config.yaml')
if config_training['CV'] < 2:
    sys.exit("This code is prepared for receiving a CV greater than 1")

# initializing seed and gpu if possible
seed_everything(config_training['seed'], cuda_benchmark=config_training['cuda_benchmark'])
dev = device_setup()

# initializing folder structures and log
run_path = (f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}_batch_"
            f"{config_data['batch_size']}_{'_'.join(config_data['classes'])}")
Path(f"{run_path}").mkdir(parents=True, exist_ok=True)
init_log(log_name=f"./{run_path}/execution.log")
shutil.copyfile('./src/config.yaml', f'./{run_path}/config.yaml')

# initializing experiment's objects
n_augments = sum([v for k, v in config_data['augmentation'].items()])
transforms = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=360),
)
train_loaders, test_loaders = load_datasets(config_training, config_data, transforms, mode='CV_PROD')


for n, (training_loader, test_loader) in enumerate(zip(train_loaders, test_loaders)):
    logging.info(f"\n\n *********************  FOLD {n}  ********************* \n\n")
    logging.info(f"\n\n ###############  TRAINING PHASE  ###############  \n\n")

    # creating specific paths and experiment's objects for each fold
    fold_time = time.perf_counter()
    Path(f"{run_path}/fold_{n}/segs/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/features_map/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/plots/").mkdir(parents=True, exist_ok=True)

    # artefacts initialization
    model, optimizer, criterion, scheduler = load_segmentation_experiment_artefacts(config_model, config_opt, 
                                                                                    config_loss, n_augments, run_path)
    model = model.to(dev)

    # init metrics file
    write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                       text_to_write=f'epoch,LR,Train,Test,Train_loss')

    best_validation_loss = 1_000_000.
    patience = 0
    for epoch in range(config_training['epochs']):
        start_epoch_time = time.perf_counter()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_train_loss, avg_dice = train_one_epoch()

        # # Update the learning rate at the end of each epoch
        if config_opt['scheduler'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_train_loss)

        # Track the best performance, and save the model's state
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler': 'scheduler',
            'val_loss': best_validation_loss
        }, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}.tar')

        # logging results of current epoch
        end_epoch_time = time.perf_counter()
        test_results = inference_binary_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/",
                                                     device=dev)
        logging.info(f'EPOCH {epoch} --> '
                     f'|| Training loss {avg_train_loss:.4f} '
                     f'|| Training DICE {avg_dice:.4f} '
                     f'|| Patience: {patience} '
                     f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f} '
                     f'|| LR: {optimizer.param_groups[0]["lr"]:.8f}')

        # write metrics
        write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                           text_to_write=f'{epoch},{optimizer.param_groups[0]["lr"]:.8f},'
                                         f'{avg_dice:.4f},{test_results["DICE"].mean():.4f},'
                                         f'{avg_train_loss:.4f}',
                           close=True)

        # early stopping
        if patience > config_training['max_patience']:
            logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
            break

    # store metrics
    metrics = pd.read_csv(f'{run_path}/fold_{n}/metrics.csv')

    """
    INFERENCE PHASE
    """

    # results for test dataset
    logging.info(f"\n\n ###############  TESTING PHASE  ###############  \n\n")
    test_results = inference_binary_segmentation(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/",
                                                 device=dev)
    logging.info(test_results.mean())

    end_time = time.perf_counter()
    logging.info(f"Total time for fold {n}: {end_time - fold_time:.2f}")

    # Clear the GPU memory after evaluating on the test data for this fold
    torch.cuda.empty_cache()


# saving final results as a Excel file
save_segmentation_results(run_path)

# Measuring total time
end_time = time.perf_counter()
logging.info(f"Total time for all of the folds: {end_time - init_time:.2f}")
