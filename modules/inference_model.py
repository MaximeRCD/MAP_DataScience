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


path_to_saved_model = "./cross_entropy_weighted10_batch64_32_16.pth"

model = load_model(path_to_saved_model)
## 4. Test du modèle sur le jeu de test
test_transform = A.Compose(
    [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)
test_dataset = FreeParkingPlacesInferenceDataset(TEST_IMAGE_DIR, transform=test_transform)

predictions = predict(model, PARAMS, test_dataset, batch_size=16)

predicted_masks = []
for predicted_256x256_mask, original_height, original_width in predictions:
    full_sized_mask = A.resize(
        predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST
    )
    predicted_masks.append(full_sized_mask)

# visualizer_worker.display_image_grid(TEST_IMAGE_FILENAMES, TEST_IMAGE_DIR, TEST_MASK_DIR, predicted_masks=predicted_masks)
