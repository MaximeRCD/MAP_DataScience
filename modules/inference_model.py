from collections import defaultdict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ternausnet.models
import os
import cv2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np
from .constants import PARAMS, TEST_IMAGE_DIR
from .datasets import FreeParkingPlacesInferenceDataset

def load_model(model_file_path):
    model = getattr(ternausnet.models, "UNet11")(pretrained=False)
    model.load_state_dict(torch.load(model_file_path, map_location=PARAMS['device']))
    model = model.to(PARAMS['device'])
    return model

def predict(model, params, test_dataset, batch_size):
    # Initialize DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
    )

    model.eval()  # Set the model to evaluation mode
    predictions = []

    with torch.no_grad():  # No need to track gradients for predictions
        for images, (original_heights, original_widths) in test_loader:
            images = images.to(params["device"], non_blocking=True)
            output = model(images)
            probabilities = torch.sigmoid(output.squeeze(1))
            predicted_masks = (probabilities >= 0.5).float() * 1
            predicted_masks = predicted_masks.cpu().numpy()

            # Process predictions
            for predicted_mask, original_height, original_width in zip(
                predicted_masks, original_heights.numpy(), original_widths.numpy()
            ):
                predictions.append((predicted_mask, original_height, original_width))

    return predictions