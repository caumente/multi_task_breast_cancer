import warnings
warnings.filterwarnings('ignore')
import logging
import sys
import torch
from pathlib import Path
from src.models.segmentation.BTS_UNet import BTSUNet
from src.models.segmentation.ResidualUNet import ResidualUNet
from src.models.segmentation.nnUNet import nnUNet2021
from src.models.classification.BTS_UNET_classifier import BTSUNetClassifier
from src.models.classification.UnetPlusPlus_Classifier import UNetPlusPlusClassifier
from src.models.classification.nnUNet_classifier import nnUNetClassifier
from monai.networks.nets import EfficientNetBN
from src.models.multitask.Multi_BTS_UNet import Multi_BTS_UNet
from src.models.multitask.Multi_FSB_BTS_UNet import Multi_FSB_BTS_UNet
from src.models.multitask.MTUNetPlusPlus import MTUNetPlusPlus
from src.models.multitask.MTnnUNet import MTnnUNet
from monai.networks.nets import UNet, AttentionUnet, BasicUnetPlusPlus
from monai.losses import DiceLoss, DiceFocalLoss, GeneralizedDiceLoss, DiceCELoss, HausdorffDTLoss, FocalLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from monai.networks.nets import SwinUNETR, SegResNet
from src.utils.models import count_parameters
from src.utils.criterions import FocalLoss as FocalLossFunction


def init_segmentation_model(
        architecture: str,
        sequences: int = 1,
        regions: int = 1,
        width: int = 48,
        save_folder: Path = None,
        deep_supervision: bool = False
) -> torch.nn.Module:
    """
    This function implement the architecture chosen.

    :param architecture: architecture chosen
    :param sequences: number of channels for the input layer
    :param regions: number of channels for the output layer
    :param width: number of channels to use in the first convolutional module
    :param deep_supervision: whether deep supervision is active
    :param save_folder: path to store the model
    :return: PyTorch module
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"The model will be fed with {sequences} sequences")

    if architecture == 'BTSUNet':
        model = BTSUNet(sequences=sequences, regions=regions, width=width, deep_supervision=deep_supervision)
    elif architecture == 'nnUNet':
        model = nnUNet2021(sequences=sequences, regions=regions)
    elif architecture == 'UNet':
        model = UNet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                     channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == "AttentionUNet":
        model = AttentionUnet(spatial_dims=2, in_channels=sequences, out_channels=regions,
                              channels=(width, 2*width, 4*width, 8*width), strides=(2, 2, 2))
    elif architecture == "ResidualUNet":
        model = ResidualUNet(sequences=sequences, regions=regions, width=width)
    elif architecture == "UnetPlusPlus":
        model = BasicUnetPlusPlus(spatial_dims=2, in_channels=sequences, out_channels=regions,
                                  deep_supervision=deep_supervision)
    elif architecture == "SwinUNETR":
        model = SwinUNETR(img_size=(128, 128), in_channels=1, out_channels=1, use_checkpoint=True, spatial_dims=2)
    elif architecture == "SegResNet":
        model = SegResNet(spatial_dims=2, in_channels=sequences, out_channels=1)
    else:
        model = torch.nn.Module()
        assert ("The model selected does not exist. Please, chose some of the following architectures: "
                "BTS U-Net (BTSUNet), nnU-Net (nnUNet), Residual U-Net (ResidualUNet), UNet (UNet), Attention U-Net "
                "(AttentionUNet), UNet++ (UnetPlusPlus), Swin UNETR (SwinUNETR), or SegResNet (SegResNet).")

    # Saving the model scheme in a .txt file
    if save_folder is not None:
        model_file = save_folder / "model.txt"
        with model_file.open("w") as f:
            print(model, file=f)

    logging.info(model)
    logging.info(f"Total number of trainable parameters: {count_parameters(model)}")

    return model


def init_classification_model(
        architecture: str,
        sequences: int = 1,
        n_classes: int = 1,
        width: int = 48,
        save_folder: Path = None
) -> torch.nn.Module:
    """
    This function implement the architecture chosen.

    :param architecture: architecture chosen
    :param sequences: number of channels for the input layer
    :param n_classes: number of classes to predict
    :param width: number of channels to use in the first convolutional module
    :param save_folder: path to store the model
    :return: PyTorch module
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"The model will be fed with {sequences} sequences")

    if architecture == 'BTSUNetClassifier':
        model = BTSUNetClassifier(sequences=sequences, classes=n_classes, width=width)
    elif architecture == 'UNetPlusPlusClassifier':
        model = UNetPlusPlusClassifier(spatial_dims=2, in_channels=sequences, n_classes=n_classes)
    elif architecture == 'nnUNetClassifier':
        model = nnUNetClassifier(sequences=sequences, n_classes=n_classes)
    else:
        model = torch.nn.Module()
        assert ("The model selected does not exist. Please, chose some of the following architectures: nnU-Net "
                "(nnUNetClassifier) or UNet++ (UNetPlusPlusClassifier)")

    # Saving the model scheme in a .txt file
    if save_folder is not None:
        model_file = save_folder / "model.txt"
        with model_file.open("w") as f:
            print(model, file=f)

    logging.info(model)
    logging.info(f"Total number of trainable parameters: {count_parameters(model)}")

    return model


def init_multitask_model(
        architecture: str,
        sequences: int = 1,
        regions: int = 1,
        n_classes: int = 2,
        width: int = 48,
        save_folder: Path = None,
        deep_supervision: bool = False
) -> torch.nn.Module:
    """
    This function implement the architecture chosen.

    :param architecture: architecture chosen
    :param sequences: number of channels for the input layer
    :param regions: number of channels for the output layer
    :param n_classes: number of classes to predict
    :param width: number of channels to use in the first convolutional module
    :param deep_supervision: whether deep supervision is active
    :param save_folder: path to store the model
    :return: PyTorch module
    """

    logging.info(f"Creating {architecture} model")
    logging.info(f"The model will be fed with {sequences} sequences")
    if architecture == 'Multi_BTSUNet':
        model = Multi_BTS_UNet(sequences=sequences, regions=regions, n_classes=n_classes, width=width, deep_supervision=deep_supervision)
    elif architecture == 'MTUNetPlusPlus':
        model = MTUNetPlusPlus(in_channels=sequences, out_channels=regions, n_classes=n_classes, deep_supervision=deep_supervision)
    elif architecture == "MTnnUNet":
        model = MTnnUNet(sequences=sequences, regions=regions, n_classes=n_classes)
    else:
        model = torch.nn.Module()
        assert ("The model selected does not exist. Please, chose some of the following architectures: "
                "Multi-task nnU-Net (MTnnUNet) or Multi-task UNet++ (MTUNetPlusPlus)")

    # Saving the model scheme in a .txt file
    if save_folder is not None:
        model_file = save_folder / "model.txt"
        with model_file.open("w") as f:
            print(model, file=f)

    logging.info(model)
    logging.info(f"Total number of trainable parameters: {count_parameters(model)}")

    return model


def init_optimizer(model: torch.nn.Module, optimizer: str, learning_rate: float = 0.001) -> torch.optim:
    """
    This function initialize the optimizer chosen.

    :param model: architecture used in the experiment
    :param optimizer: optimizer chosen to initialize
    :param learning_rate: learning rate defined
    :return: PyTorch optimizer
    """
    if optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=1e-4)
    elif optimizer == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        logging.info(f"The optimizer '{optimizer}' is not recognized. SGD will be used instead.")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, nesterov=True)

    return optimizer


def init_criterion_segmentation(loss_function: str = "dice") -> torch.nn.Module:
    """
    This function initialize the segmentation criterion chosen.
    Note: All the loss functions are initialized with sigmoid activation by default. Then, the segmentation map
    predicted by the model mustn't to apply the activation in the output layer.

    :param loss_function: criterion chosen
    :return: PyTorch module
    """

    if loss_function == 'DICE':
        loss_function_criterion = DiceLoss(include_background=True, sigmoid=True, smooth_dr=1, smooth_nr=1,
                                           squared_pred=True)
    elif loss_function == 'Hausdorff':
        loss_function_criterion = HausdorffDTLoss(sigmoid=True)
    elif loss_function == "FocalDICE":
        loss_function_criterion = DiceFocalLoss(include_background=True, sigmoid=True, smooth_dr=1, smooth_nr=1,
                                                squared_pred=True)
    elif loss_function == "GeneralizedDICE":
        loss_function_criterion = GeneralizedDiceLoss(include_background=True, sigmoid=True)
    elif loss_function == "CrossentropyDICE":
        loss_function_criterion = DiceCELoss(include_background=True, sigmoid=True, squared_pred=True)
    elif loss_function == "Jaccard":
        loss_function_criterion = DiceLoss(include_background=True, sigmoid=True, jaccard=True, reduction="sum")
    elif loss_function == "FocalLoss":
        loss_function_criterion = FocalLoss(include_background=True, use_softmax=False)
    elif loss_function == "BCE":
        loss_function_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        assert ("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard'"
                ", 'FocalLoss', 'BCE', 'Hausdorff']")
        sys.exit()

    return loss_function_criterion


def init_criterion_classification(
        n_classes: int = 2,
        classes_weighted=None,
        classification_criterion="CE"
) -> torch.nn.Module:
    if n_classes == 2:
        loss_function_criterion = torch.nn.BCEWithLogitsLoss()
    else:
        if classes_weighted:
            class_frequencies = torch.tensor(classes_weighted)

            # Calculate class weights
            class_weights = 1.0 / class_frequencies

            # Create a tensor for the Normalize weights
            weight_tensor = torch.tensor(class_weights / class_weights.sum(), dtype=torch.float)

            # Define the loss function with class weights
            if classification_criterion == "Focal":
                loss_function_criterion = FocalLossFunction(alpha=1, gamma=2, reduction='mean', weight=weight_tensor.to("cuda"))
            else:
                loss_function_criterion = torch.nn.CrossEntropyLoss(reduction='mean', weight=weight_tensor.to("cuda"))
        else:
            if classification_criterion == "Focal":
                loss_function_criterion = FocalLossFunction(alpha=1, gamma=2, reduction='mean')
            else:
                loss_function_criterion = torch.nn.CrossEntropyLoss(reduction='mean')

    return loss_function_criterion


def init_lr_scheduler(
        optimizer,
        scheduler: str = "cosine",
        t_max: int = 20,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        patience: int = 20
) -> torch.optim.lr_scheduler:

    if scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=patience, min_lr=min_lr, verbose=True)
    elif scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=min_lr)
    else:
        print("Select a loss function allowed: ['DICE', 'FocalDICE', 'GeneralizedDICE', 'CrossentropyDICE', 'Jaccard']")
        sys.exit()

    return scheduler


def load_segmentation_experiment_artefacts(config_model, config_opt, config_loss, n_augments, run_path):

    model = init_segmentation_model(architecture=config_model['architecture'],
                                    sequences=config_model['sequences'] + n_augments,
                                    width=config_model['width'], deep_supervision=config_model['deep_supervision'],
                                    save_folder=Path(f'./{run_path}/'))
    optimizer = init_optimizer(model=model, optimizer=config_opt['opt'], learning_rate=config_opt['lr'])
    criterion = init_criterion_segmentation(loss_function=config_loss['function'])
    scheduler = init_lr_scheduler(optimizer=optimizer, scheduler=config_opt['scheduler'], t_max=int(config_opt['t_max']),
                                  patience=int(config_opt['patience']), min_lr=float(config_opt['min_lr']),
                                  factor=float(config_opt['decrease_factor']))

    return model, optimizer, criterion, scheduler


def load_multitask_experiment_artefacts(config_data, config_model, config_opt, config_loss, n_augments, run_path):

    model = init_multitask_model(architecture=config_model['architecture'],
                                 sequences=config_model['sequences'] + n_augments,
                                 width=config_model['width'],
                                 n_classes=len(config_data['classes']),
                                 deep_supervision=config_model['deep_supervision'],
                                 save_folder=Path(f'{run_path}/') if run_path is not None else None)
    optimizer = init_optimizer(model=model, optimizer=config_opt['opt'], learning_rate=config_opt['lr'])
    segmentation_criterion = init_criterion_segmentation(loss_function=config_loss['function'])
    classification_criterion = init_criterion_classification(n_classes=len(config_data['classes']),
                                                             classes_weighted=config_data["classes_weighted"],
                                                             classification_criterion=config_loss['classification_criterion'])
    scheduler = init_lr_scheduler(optimizer=optimizer, scheduler=config_opt['scheduler'],
                                  t_max=int(config_opt['t_max']), patience=int(config_opt['patience']),
                                  min_lr=float(config_opt['min_lr']), factor=float(config_opt['decrease_factor']))

    return model, optimizer, segmentation_criterion, classification_criterion, scheduler


def load_classification_experiment_artefacts(config_data, config_model, config_opt, config_loss, n_augments, run_path):

    model = init_classification_model(architecture=config_model['architecture'],
                                      sequences=config_model['sequences'] + n_augments,
                                      width=config_model['width'],
                                      n_classes=len(config_data['classes']),
                                      save_folder=Path(f'{run_path}/'))
    optimizer = init_optimizer(model=model, optimizer=config_opt['opt'], learning_rate=config_opt['lr'])
    classification_criterion = init_criterion_classification(n_classes=len(config_data['classes']),
                                                             classes_weighted=config_data["classes_weighted"],
                                                             classification_criterion=config_loss['classification_criterion'])
    scheduler = init_lr_scheduler(optimizer=optimizer, scheduler=config_opt['scheduler'],
                                  t_max=int(config_opt['t_max']), patience=int(config_opt['patience']),
                                  min_lr=float(config_opt['min_lr']), factor=float(config_opt['decrease_factor']))

    return model, optimizer, classification_criterion, scheduler


def device_setup():
    if torch.cuda.is_available():
        dev = "cuda:0"
        logging.info("GPU will be used to train the model")
    else:
        dev = "cpu"
        logging.info("CPU will be used to train the model")

    return dev
