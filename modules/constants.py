"""
This module is designed for training, validating, and testing a U-Net model for semantic
segmentation tasks. It sets up the necessary configurations for the model, including directories
for training, validation, and testing datasets, model parameters, and runtime settings. The module
supports running on CUDA-enabled devices if available, falling back to CPU otherwise. It defines
constants for batch sizes, learning rates, class weights, and more, to be used across the training
and evaluation processes.

Attributes:
    VISUALIZER_BATCH_SIZE (int): Batch size for visualizing predictions.
    TRAINING_IMAGE_DIR (str): Directory path for training images.
    TRAINING_MASK_DIR (str): Directory path for training masks.
    VAL_IMAGE_DIR (str): Directory path for validation images.
    VAL_MASK_DIR (str): Directory path for validation masks.
    TEST_IMAGE_DIR (str): Directory path for test images.
    TEST_MASK_DIR (str): Directory path for test masks.
    PARAMS (dict): Configuration parameters for the model including :
        - model type
        - device
        - learning rate
        - class weights
        - batch size
        - number of epochs
"""

import torch

VISUALIZER_BATCH_SIZE = 4

TRAINING_IMAGE_DIR = "../data/train/images/"
TRAINING_MASK_DIR = "../data/train/masks/"

VAL_IMAGE_DIR = "../data/val/images/"
VAL_MASK_DIR = "../data/val/masks/"

TEST_IMAGE_DIR = "./data/test/images/"
TEST_MASK_DIR = "./data/test/masks/"

PARAMS = {
    "model": "UNet11",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 0.003,
    "class_weights": [1.0, 3.0],
    "batch_size": 16,
    "epochs": 25,
}
