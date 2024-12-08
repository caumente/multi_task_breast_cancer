import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from pprint import pformat

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score as f1
from torchvision.transforms import RandomRotation, RandomHorizontalFlip, RandomVerticalFlip

from src.dataset.BUSI_dataloader import load_datasets
from src.utils.criterions import apply_criterion_classification
from src.utils.experiment_init import device_setup
from src.utils.experiment_init import load_classification_experiment_artefacts
from src.utils.metrics import accuracy_from_tensor, f1_score_from_tensor
from src.utils.metrics import binary_classification_metrics
from src.utils.metrics import multiclass_classification_metrics
from src.utils.miscellany import init_log
from src.utils.miscellany import load_config_file
from src.utils.miscellany import save_classification_results
from src.utils.miscellany import seed_everything
from src.utils.miscellany import write_metrics_file
from src.utils.models import inference_binary_classification
from src.utils.models import inference_multiclass_classification
from src.utils.models import load_pretrained_model
from src.utils.visualization import plot_evolution


def train_one_epoch():
    running_training_loss = 0.
    ground_truth_label = []
    predicted_label = []

    # Iterating over training loader
    for k, data in enumerate(training_loader):
        inputs, label = data['image'].to(dev), data['label'].to(dev)
        if len(config_data['classes']) > 2:
            label = torch.nn.functional.one_hot(label.flatten().to(torch.int64), num_classes=3).to(torch.float)

        # Zero your gradients for every batch!
        optimizer.zero_grad(set_to_none=True)

        # Make predictions for this batch
        pred_class = model(inputs)

        # Compute the loss and its gradients
        classification_loss = apply_criterion_classification(classification_criterion, label, pred_class,
                                                             config_loss['inversely_weighted'])
        running_training_loss += classification_loss.item()

        # Performing backward step through scaler methodology
        classification_loss.backward()
        optimizer.step()

        # averaging prediction if deep supervision
        if isinstance(pred_class, list):
            pred_class = torch.mean(torch.stack(pred_class, dim=0), dim=0)

        # this if-else differentiates between multiclass and binary class predictions
        if len(config_data['classes']) > 2:
            label = [torch.argmax(l, keepdim=True).to(torch.float) for l in label]
            pred_class = [torch.argmax(pl, keepdim=True).to(torch.float) for pl in pred_class]

            if isinstance(pred_class, list):
                for la, p in zip(label, pred_class):
                    ground_truth_label.append(la)
                    predicted_label.append(p)
            else:
                ground_truth_label.append(label)
                predicted_label.append(pred_class)
        else:
            # adding ground truth label and predicted label
            if pred_class.shape[0] > 1:  # when batch size > 1, each element is added individually
                for i in range(pred_class.shape[0]):
                    predicted_label.append((torch.sigmoid(pred_class[i, :]) > .5).double())
                    ground_truth_label.append(label[i, :])
            else:
                predicted_label.append((torch.sigmoid(pred_class) > .5).double())
                ground_truth_label.append(label)

        del pred_class

    if len(config_data['classes']) > 2:
        ground_truth_label = [tensor.item() for tensor in ground_truth_label]
        predicted_label = [tensor.item() for tensor in predicted_label]
        running_acc = accuracy_score(ground_truth_label, predicted_label)
        running_f1_score = f1(y_true=ground_truth_label, y_pred=predicted_label, labels=[0, 1, 2], average='micro')
        # macro_f1 = f1_score(y_true=ground_truth_label, y_pred=predicted_label, labels=[0, 1, 2], average='macro')
    else:
        running_acc = accuracy_from_tensor(torch.Tensor(ground_truth_label), torch.Tensor(predicted_label))
        running_f1_score = f1_score_from_tensor(torch.Tensor(ground_truth_label), torch.Tensor(predicted_label))

    return running_training_loss / training_loader.__len__(), running_acc, running_f1_score


@torch.inference_mode()
def validate_one_epoch():
    running_validation_loss = 0.0
    val_ground_truth_label = []
    val_predicted_label = []

    for i, validation_data in enumerate(validation_loader):

        validation_images, validation_masks = validation_data['image'].to(dev), validation_data['mask'].to(dev)
        validation_label = validation_data['label'].to(dev)
        if len(config_data['classes']) > 2:
            validation_label = torch.nn.functional.one_hot(validation_label.flatten().to(torch.int64), num_classes=3).to(torch.float)

        pred_val_class = model(validation_images)

        # Compute the loss and its gradients
        classification_val_loss = apply_criterion_classification(classification_criterion, validation_label,
                                                                 pred_val_class, config_loss['inversely_weighted'])
        running_validation_loss += classification_val_loss.item()

        # averaging prediction if deep supervision
        if isinstance(pred_val_class, list):
            pred_val_class = torch.mean(torch.stack(pred_val_class, dim=0), dim=0)

        # storing ground truth and prediction
        if len(config_data['classes']) > 2:
            validation_label = [l.argmax() for l in validation_label]
            pred_val_class = [pl.argmax() for pl in pred_val_class]
            if isinstance(pred_val_class, list):
                for l, p in zip(validation_label, pred_val_class):
                    val_ground_truth_label.append(l)
                    val_predicted_label.append(p)
            else:
                val_ground_truth_label.append(validation_label)
                val_predicted_label.append(pred_val_class)
        else:
            # adding ground truth label and predicted label
            if pred_val_class.shape[0] > 1:  # when batch size > 1, each element is added individually
                for i in range(pred_val_class.shape[0]):
                    val_predicted_label.append((torch.sigmoid(pred_val_class[i, :]) > .5).double())
                    val_ground_truth_label.append(validation_label[i, :])
            else:
                val_predicted_label.append((torch.sigmoid(pred_val_class) > .5).double())
                val_ground_truth_label.append(validation_label)

        del pred_val_class

    # total segmentation loss and DICE metric for the epoch
    avg_validation_loss = running_validation_loss / validation_loader.__len__()

    # total classifiation loss and accuracy for the epoch
    if len(config_data['classes']) > 2:
        val_ground_truth_label = [tensor.item() for tensor in val_ground_truth_label]
        val_predicted_label = [tensor.item() for tensor in val_predicted_label]
        running_acc = accuracy_score(val_ground_truth_label, val_predicted_label)
        running_f1_score = f1(y_true=val_ground_truth_label, y_pred=val_predicted_label, labels=[0, 1, 2], average='micro')
        # macro_f1 = f1_score(y_true=ground_truth_label, y_pred=predicted_label, labels=[0, 1, 2], average='macro')
    else:
        running_acc = accuracy_from_tensor(torch.Tensor(val_ground_truth_label), torch.Tensor(val_predicted_label))
        running_f1_score = f1_score_from_tensor(torch.Tensor(val_ground_truth_label), torch.Tensor(val_predicted_label))

    return avg_validation_loss, running_acc, running_f1_score


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
# config_training['alpha'] = alpha
alpha = config_training['alpha']
run_path = f"runs/{timestamp}_{config_model['architecture']}_{config_model['width']}" \
           f"_batch_{config_data['batch_size']}_{'_'.join(config_data['classes'])}"
Path(f"{run_path}").mkdir(parents=True, exist_ok=True)
init_log(log_name=f"./{run_path}/execution.log")
shutil.copyfile('./src/config.yaml', f'./{run_path}/config.yaml')

# initializing experiment's objects
n_augments = sum([v for k, v in config_data['augmentation'].items()])
from torchvision.transforms import v2
transforms = torch.nn.Sequential(
    RandomHorizontalFlip(p=0.5),
    RandomVerticalFlip(p=0.5),
    RandomRotation(degrees=360),
)
train_loaders, val_loaders, test_loaders = load_datasets(config_training, config_data, transforms, mode='CV')


for n, (training_loader, validation_loader, test_loader) in enumerate(zip(train_loaders, val_loaders, test_loaders)):
    logging.info(f"\n\n *********************  FOLD {n}  ********************* \n\n")
    logging.info(f"\n\n ###############  TRAINING PHASE  ###############  \n\n")

    # creating specific paths and experiment's objects for each fold
    fold_time = time.perf_counter()
    Path(f"{run_path}/fold_{n}/segs/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/plots/").mkdir(parents=True, exist_ok=True)
    Path(f"{run_path}/fold_{n}/features_map/").mkdir(parents=True, exist_ok=True)

    # artefacts initialization
    model, optimizer, classification_criterion, scheduler = load_classification_experiment_artefacts(config_data, config_model, config_opt, config_loss, n_augments, run_path)
    model = model.to(dev)

    # init metrics file
    write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                       text_to_write=f'epoch,LR,Train_loss,Validation_loss,Train_acc,Train_F1,Validation_acc,Validation_F1')

    best_validation_loss = 1_000_000.
    patience = 0
    for epoch in range(config_training['epochs']):
        current_lr = optimizer.param_groups[0]["lr"]
        start_epoch_time = time.perf_counter()

        # Make sure gradient tracking is on, and do a pass over the data
        model.train(True)
        avg_train_loss, train_acc, train_f1_score = train_one_epoch()

        # We don't need gradients on to do reporting
        model.train(False)
        avg_validation_loss, val_acc, val_f1_score = validate_one_epoch()

        # # Update the learning rate at the end of each epoch
        if config_opt['scheduler'] == 'cosine':
            scheduler.step()
        else:
            scheduler.step(avg_validation_loss)

        # Track best performance, and save the model's state
        if avg_validation_loss < best_validation_loss:
            patience = 0  # restarting patience
            best_validation_loss = avg_validation_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': 'scheduler',
                'val_loss': best_validation_loss
            }, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')
        else:
            patience += 1

        # logging results of current epoch
        end_epoch_time = time.perf_counter()
        logging.info(f'EPOCH {epoch} --> '
                     f'|| Training loss {avg_train_loss:.4f} '
                     f'|| Validation loss {avg_validation_loss:.4f} '
                     f'|| Training ACC {train_acc:.4f} '
                     f'|| Training F1 {train_f1_score:.4f} '
                     f'|| Validation ACC {val_acc:.4f} '
                     f'|| Validation F1 {val_f1_score:.4f} '
                     f'|| Patience: {patience} '
                     f'|| Epoch time: {end_epoch_time - start_epoch_time:.4f}')

        write_metrics_file(path_file=f'{run_path}/fold_{n}/metrics.csv',
                           text_to_write=f'{epoch},{current_lr:.8f},{avg_train_loss:.4f},{avg_validation_loss:.4f},'
                                         f'{train_acc:.4f},'
                                         f'{train_f1_score:.4f},{val_acc:.4f},{val_f1_score:.4f}',
                           close=True)

        # early stopping
        if patience > config_training['max_patience']:
            logging.info(f"\nValidation loss did not improve over the last {patience} epochs. Stopping training")
            break

    # store metrics
    metrics = pd.read_csv(f'{run_path}/fold_{n}/metrics.csv')
    plot_evolution(metrics, columns=['Train_loss', 'Validation_loss'], path=f'{run_path}/fold_{n}/loss_evolution.png')
    plot_evolution(metrics, columns=['Train_acc', 'Train_F1', 'Validation_acc', 'Validation_F1'],
                   path=f'{run_path}/fold_{n}/classification_metrics_evolution.png')

    """
    INFERENCE PHASE
    """

    # results for validation dataset
    logging.info(f"\n\n ###############  VALIDATION PHASE  ###############  \n\n")
    model = load_pretrained_model(model, f'{run_path}/fold_{n}/model_{timestamp}_fold_{n}')

    # results for test dataset
    logging.info(f"\n\n ###############  TESTING PHASE  ###############  \n\n")
    if len(config_data['classes']) <= 2:
        test_results_classification = inference_binary_classification(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev)
    else:
        test_results_classification = inference_multiclass_classification(model=model, test_loader=test_loader, path=f"{run_path}/fold_{n}/", device=dev)

    # classification metrics
    if len(config_data['classes']) <= 2:
        logging.info(f"\nClassification metrics:\n\n{pformat(binary_classification_metrics(test_results_classification.ground_truth, test_results_classification.predicted_label))}")
    else:
        logging.info(f"\nClassification metrics:\n\n{pformat(multiclass_classification_metrics(test_results_classification.ground_truth, test_results_classification.predicted_label))}")

    # Clear the GPU memory after evaluating on the test data for this fold
    torch.cuda.empty_cache()

    del model


# saving final results as a Excel file
save_classification_results(run_path, len(config_data['classes']))

# Measuring total time
end_time = time.perf_counter()
logging.info(f"Total time for all of the folds: {end_time - init_time:.2f}")
