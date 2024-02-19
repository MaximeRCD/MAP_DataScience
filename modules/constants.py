import torch

VISUALIZER_BATCH_SIZE = 4

TRAINING_IMAGE_DIR = "../data/train/images/"
TRAINING_MASK_DIR = "../data/train/masks/"

VAL_IMAGE_DIR = "../data/val/images/"
VAL_MASK_DIR = "../data/val/masks/"

TEST_IMAGE_DIR = "../data/test/images/"
TEST_MASK_DIR = "../data/test/masks/"

PARAMS = {
    "model": "UNet11",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "lr": 0.003,
    "class_weights": [1.0, 3.0],
    "batch_size": 16,
    # "num_workers": 4,
    "epochs": 25,
}