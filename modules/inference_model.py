"""
This module contains functions for loading a pre-trained model and making predictions on a dataset
of images for the task of detecting free parking spaces. It utilizes a U-Net model architecture
from the ternausnet library, processes images using albumentations for transformations, and employs
PyTorch for model operations.
"""

import torch.optim
import ternausnet.models
from torch.utils.data import DataLoader
import torch
from .constants import PARAMS


def load_model(model_file_path):
    """
    Loads a model from a specified file path.

    Args:
        model_file_path (str): Path to the model file.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = getattr(ternausnet.models, "UNet11")(pretrained=False)
    model.load_state_dict(torch.load(model_file_path, map_location=PARAMS["device"]))
    model = model.to(PARAMS["device"])
    return model


def predict(model, params, test_dataset, batch_size):
    """
    Generates predictions for a given dataset using the specified model.

    Args:
        model (torch.nn.Module): The model to use for predictions.
        params (dict): Parameters for model and device settings.
        test_dataset (Dataset): The dataset to predict on.
        batch_size (int): The batch size for processing.

    Returns:
        list: A list of tuples containing predicted masks and their original dimensions.
    """
    # Initialize DataLoader for the test dataset
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
