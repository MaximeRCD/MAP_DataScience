"""
This module is designed to perform inference on a dataset of parking space images to detect free
parking places.It utilizes a pre-trained model to predict free parking spaces, applies necessary
transformations to the input images, and visualizes the predictions. This script serves as the 
main entry point for the inference process, leveraging the FreeParkingPlacesInferenceDataset for
image loading and preprocessing, and utilizing the Visualizer for displaying the results.
"""

import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from modules.inference_model import load_model, predict
from modules.datasets import FreeParkingPlacesInferenceDataset
from modules.utils import Visualizer
from modules.constants import TEST_IMAGE_DIR, TEST_MASK_DIR, PARAMS


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
    visualizer_worker = Visualizer()
    test_image_filenames = os.listdir(TEST_IMAGE_DIR)

    path_to_saved_model = "./cross_entropy_weighted10_batch64_32_16.pth"
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
