"""
This module provides the setup for training, validating, and testing a U-Net model on semantic
segmentation tasks. It prepares the environment by setting directories for datasets, configuring
model parameters, and establishing runtime settings. The module ensures compatibility with 
CUDA-enabled devices, automatically opting for CPU usage if CUDA is unavailable. Constants for 
various configurations such as batch sizes, learning rates, class weights, and more are predefined
for consistency in the training and evaluation phases.

Attributes:
    VISUALIZER_BATCH_SIZE (int): Defines the batch size for generating visual predictions.
    TRAINING_IMAGE_DIR (str): Path to the directory containing training images.
    TRAINING_MASK_DIR (str): Path to the directory containing training masks.
    VAL_IMAGE_DIR (str): Path to the directory containing validation images.
    VAL_MASK_DIR (str): Path to the directory containing validation masks.
    TEST_IMAGE_DIR (str): Path to the directory containing test images.
    TEST_MASK_DIR (str): Path to the directory containing test masks.
    API_IMAGES_DIR (str): Path to the folder that contains the image to predict from and the prediction for the API.
    PARAMS (dict): Holds the model configuration parameters, including:
        - model (str): Specifies the model type.
        - device (str): Sets the device for computation ('cuda' or 'cpu').
        - lr (float): Defines the learning rate.
        - class_weights (list): Lists the weights for each class to handle class imbalance.
        - batch_size (int): Sets the batch size for training.
        - epochs (int): Specifies the number of training epochs.
    PRETRAINED_MODEL_PATH (str): Indicates the file path to a pretrained model.
    YN_ANNOTATION_MASKS_PATH (str): Path to the JSON file containing YN annotations for masks.
    MR_ANNOTATION_MASKS_PATH (str): Path to the NDJSON file containing MR annotations for masks.
    YN_ANNOTATION_TEST_MASKS_PATH (str): Path to the JSON file having annotations for test masks.

Note:
    - The DATA_ROOT_DIR, DATA_IMAGE_DIR, and DATA_MASK_DIR constants define the root directory and 
    subdirectories for organizing the dataset images and masks.
    - The PARAMS dictionary can be adjusted to tune the model's performance and adapt to different
      hardware capabilities.
    - The additional JSON and NDJSON paths for annotations are used for specialized tasks within 
    the semantic segmentation framework, possibly for evaluation or further annotation refinement
    processes.
"""

import torch

VISUALIZER_BATCH_SIZE = 4

DATA_ROOT_DIR = "./data/"

DATA_IMAGE_DIR = DATA_ROOT_DIR + "images/"
DATA_MASK_DIR = DATA_ROOT_DIR + "masks/"

TRAINING_IMAGE_DIR = DATA_ROOT_DIR + "train/images/"
TRAINING_MASK_DIR = DATA_ROOT_DIR + "train/masks/"

VAL_IMAGE_DIR = DATA_ROOT_DIR + "val/images/"
VAL_MASK_DIR = DATA_ROOT_DIR + "val/masks/"

TEST_IMAGE_DIR = DATA_ROOT_DIR + "test/images/"
TEST_MASK_DIR = DATA_ROOT_DIR + "test/masks/"

API_IMAGES_DIR = DATA_ROOT_DIR + "API/"

PARAMS = {
    "model": "UNet11",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 0.003,
    "class_weights": [1.0, 3.0],
    "batch_size": 16,
    "epochs": 25,
}

PRETRAINED_MODEL_PATH = "./cross_entropy_weighted10_batch64_32_16.pth"

YN_ANNOTATION_MASKS_PATH = "./json/mask.json"
MR_ANNOTATION_MASKS_PATH = "./json/mask_maxime.ndjson"
YN_ANNOTATION_TEST_MASKS_PATH = "./json/mask_test.json"