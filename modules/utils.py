"""
This module provides utility functions and classes for loading, processing, and visualizing
data related to detecting free parking spaces. It includes functionality for checking CUDA
availability, preprocessing masks for semantic segmentation, reading image and mask data,
and visualizing results for comparison and evaluation purposes.
"""

import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


def check_cuda_availability():
    """
    Checks and prints if a CUDA-compatible GPU is available. Recommends using Google Colab
    if no CUDA GPU is detected.
    """
    if torch.cuda.is_available():
        print("A CUDA-compatible GPU has been detected.")
    else:
        print("No CUDA-compatible GPU detected. Consider using Google Colab.")


def preprocess_mask(mask):
    """
    Converts a mask image's pixel values from 255 to 1.0 for binary classification tasks.

    Args:
        mask (np.ndarray): The input mask image.

    Returns:
        np.ndarray: The preprocessed mask image.
    """
    mask = mask.astype(np.float32)
    mask[mask == 255.0] = 1.0
    return mask


class DataReader:
    """
    Provides methods for reading images and their corresponding masks from disk.
    """

    @staticmethod
    def read_data(image_path, mask_path=None):
        """
        Reads an image and optionally a mask from given paths.

        Args:
            image_path (str): Path to the image file.
            mask_path (str, optional): Path to the mask file. Defaults to None.

        Returns:
            tuple: The original image and mask (if provided) as numpy arrays.
        """
        image = cv2.imread(image_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask_path:
            original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            return original_image, original_mask
        return original_image, None

    @staticmethod
    def read_mask(mask_path):
        """
        Reads a mask from the given file paths.
        Args:
            mask_path (str): Path to the mask file.
        Returns:
            the original mask as numpy arrays.
        """
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        return original_mask


class Visualizer:
    """
    Contains methods for visualizing images and masks for evaluation and comparison.
    """

    @staticmethod
    def visualize(image, mask, original_image=None, original_mask=None, title=None):
        """
        Displays the original and transformed images and masks side by side for comparison.

        Args:
            image (np.ndarray): The transformed image.
            mask (np.ndarray): The transformed mask.
            original_image (np.ndarray, optional): The original image.
            original_mask (np.ndarray, optional): The original mask.
            title (str, optional): Title for the plot.
        """
        fontsize = 18
        fig, ax = (
            plt.subplots(1,2, figsize=(10, 10))
            if original_image is None
            else plt.subplots(2, 2, figsize=(10, 10))
        )
        if original_image is not None:
            fig.suptitle(title, fontsize=fontsize)
            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title("Original image", fontsize=fontsize)
            ax[1, 0].imshow(original_mask)
            ax[1, 0].set_title("Original mask", fontsize=fontsize)
            ax[0, 1].imshow(image)
            ax[0, 1].set_title("Transformed image", fontsize=fontsize)
            ax[1, 1].imshow(mask)
            ax[1, 1].set_title("Transformed mask", fontsize=fontsize)
        else:
            ax[0].imshow(image)
            ax[0].set_title('Image')
            ax[1].imshow(mask)
            ax[1].set_title('Mask')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_batch(data_loader):
        """
        Visualizes a batch of images and masks from a DataLoader.

        Args:
            data_loader (DataLoader): The DataLoader to visualize.
        """
        _, axs = plt.subplots(3, 2, figsize=(10, 15))
        for i, (images, masks) in enumerate(data_loader):
            if i == 3:
                break
            image = (
                images[i].permute(1, 2, 0) * torch.tensor([0.229, 0.224, 0.225])
                + torch.tensor([0.485, 0.456, 0.406])
            ).clamp(0, 1)
            axs[i, 0].imshow(image.numpy())
            axs[i, 0].set_title(f"Image {i}")
            axs[i, 0].axis("off")
            axs[i, 1].imshow(masks[i], cmap="gray")
            axs[i, 1].set_title(f"Mask {i}")
            axs[i, 1].axis("off")
        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_image_grid(
        images_filenames, images_directory, masks_directory, predicted_masks=None
    ):
        """
        Displays a grid of images, their ground truth masks, and optionally predicted masks.

        Args:
            images_filenames (list): Filenames of images to display.
            images_directory (str): Directory containing the images.
            masks_directory (str): Directory containing the ground truth masks.
            predicted_masks (list, optional): List of predicted masks.
        """
        cols = 3 if predicted_masks else 2
        rows = len(images_filenames)
        _, ax = plt.subplots(rows, cols, figsize=(10, rows * 3), squeeze=False)
        for i, filename in enumerate(images_filenames):
            img = cv2.cvtColor(
                cv2.imread(os.path.join(images_directory, filename)), cv2.COLOR_BGR2RGB
            )
            mask = preprocess_mask(
                cv2.imread(
                    os.path.join(
                        masks_directory, filename.replace(".png", "_mask.png")
                    ),
                    cv2.IMREAD_GRAYSCALE,
                )
            )
            ax[i, 0].imshow(img)
            ax[i, 1].imshow(mask, cmap="gray")
            if predicted_masks:
                ax[i, 2].imshow(predicted_masks[i], cmap="gray")
            for j in range(cols):
                ax[i, j].axis("off")
        plt.tight_layout()
        plt.show()


class DirectoryManager:
    """
    A helper class to manage directory operations such as ensuring directories exist.
    """
    @staticmethod
    def ensure_directory_exists(file_path):
        """
        Ensures that the directory for the given file path exists.
        Creates the directory if it does not exist.
        Args:
            file_path (str): The file path for which the directory needs to be checked/created.
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)


class ImageSaver:
    """
    Helper class for saving images to the disk.

    This class provides a static method to save images in
    various formats to a specified path on the disk.
    It ensures that the necessary directories exist
    before saving the image using OpenCV's imwrite function.
    """

    @staticmethod
    def save_image(image: np.ndarray, file_path: str) -> None:
        """
        Saves an image to the specified file path.

        Parameters:
        image (np.ndarray): The image array to be saved. Expected to be in a format
                            compatible with OpenCV, such as an 8-bit or floating-point 32-bit array.
        file_path (str): The complete target file path where the image will be saved.
                         If the directory structure does not exist, it will be created.

        Returns:
        None: This method does not return any value.

        Raises:
        ValueError: If 'image' is not a valid NumPy ndarray.
        OSError: If the file could not be saved to the specified path due to a file system error.

        Examples:
        >>> img = np.zeros((100, 100), dtype=np.uint8)
        >>> ImageSaver.save_image(img, '/path/to/save/image.png')
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Provided image is not a valid numpy array")

        DirectoryManager.ensure_directory_exists(file_path)

        if not cv2.imwrite(file_path, image):
            raise OSError(
                f"Image could not be saved to {file_path}."
                "There might be an issue with the file path or permissions."
            )
