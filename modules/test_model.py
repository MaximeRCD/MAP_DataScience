"""
This module is designed to perform inference on a dataset of parking space images to detect free
parking places.It utilizes a pre-trained model to predict free parking spaces, applies necessary
transformations to the input images, and visualizes the predictions. This script serves as the 
main entry point for the inference process, leveraging the FreeParkingPlacesInferenceDataset for
image loading and preprocessing, and utilizing the Visualizer for displaying the results.
"""

import os
import cv2
import ternausnet.models
import torch
import torch.optim
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from constants import TEST_IMAGE_DIR, TEST_MASK_DIR, PARAMS, PRETRAINED_MODEL_PATH, S3_USER_BUCKET, S3_PRETRAINED_MODEL_NAME
from datasets import FreeParkingPlacesInferenceDataset
from utils import Visualizer
from s3_fs import S3FileManager


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


def main():
    """
    Main function to execute the inference workflow.

    This function initializes the visualizer, loads the test images, and uses a pre-trained model
    to predict free parking spaces on the test dataset. It applies a series of transformations to
    the input images for model compatibility, performs the predictions, resizes the predicted
    masks to their original dimensions, and visualizes the results alongside the original images
    and masks.

    The path to the saved model, the batch size for predictions, and the specific transformations
    are predefined within the function. It demonstrates an end-to-end application of the model
    inference, from loading the model to visualizing the predicted free parking spaces.
    """
    # Initialize the S3FileManager
    manager = S3FileManager()
    manager.import_file_from_ssp_cloud(
        "/".join([S3_USER_BUCKET, S3_PRETRAINED_MODEL_NAME]),
        "/".join([".", PRETRAINED_MODEL_PATH]),
    )
    visualizer_worker = Visualizer()
    test_image_filenames = os.listdir(TEST_IMAGE_DIR)

    path_to_saved_model = PRETRAINED_MODEL_PATH
    model = load_model(path_to_saved_model)

    # Setup for testing the model with predefined transformations
    test_transform = A.Compose(
        [
            A.Resize(256, 256),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
    test_dataset = FreeParkingPlacesInferenceDataset(
        TEST_IMAGE_DIR, transform=test_transform
    )

    predictions = predict(model, PARAMS, test_dataset, batch_size=16)

    predicted_masks = []
    for predicted_256x256_mask, original_height, original_width in predictions:
        full_sized_mask = cv2.resize(
            predicted_256x256_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        predicted_masks.append(full_sized_mask)

    visualizer_worker.display_image_grid(
        test_image_filenames,
        TEST_IMAGE_DIR,
        TEST_MASK_DIR,
        predicted_masks=predicted_masks,
    )


if __name__ == "__main__":
    main()
