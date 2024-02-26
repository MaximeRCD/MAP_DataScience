"""
This module provides dataset classes for loading and preprocessing datasets for parking space
detection tasks. It includes two classes: `FreeParkingPlacesDataset` for training and validation 
datasets which include both images and their corresponding masks, and 
`FreeParkingPlacesInferenceDataset` for inference datasets which only require images.

The datasets utilize a custom `DataReader` for reading images and masks from disk, and support 
applying transformations to the data, making them suitable for training and evaluating machine 
learning models for semantic segmentation.
"""

import os
from torch.utils.data import Dataset
import numpy as np
from utils import DataReader

class FreeParkingPlacesDataset(Dataset):
    """
    A dataset class for loading images and their corresponding segmentation masks for parking 
    space detection.

    Attributes:
        images_directory (str): Directory containing the images.
        masks_directory (str): Directory containing the corresponding masks.
        images_filenames (list): List of filenames for the images in the images directory.
        transform (callable, optional): An optional transform to be applied on a sample.

    Args:
        images_directory (str): Path to the directory with images.
        masks_directory (str): Path to the directory with masks.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, images_directory, masks_directory, transform=None):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.images_filenames = os.listdir(images_directory)
        self.transform = transform

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.images_filenames)

    def __getitem__(self, idx):
        """
        Fetches the image and mask at the index `idx` and applies transformations if any.

        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: A tuple containing the transformed image and mask.
        """
        image_filename = self.images_filenames[idx]
        image_path = os.path.join(self.images_directory, image_filename)
        mask_path = os.path.join(self.masks_directory, image_filename.replace(".png", '_mask.png'))
        image, mask = DataReader.read_data(image_path, mask_path)
        mask = self.preprocess_mask(mask)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        return image, mask

    def preprocess_mask(self, mask):
        """
        Preprocesses the mask to normalize its values.

        Args:
            mask (numpy.ndarray): The original mask.

        Returns:
            numpy.ndarray: The preprocessed mask.
        """
        mask = mask.astype(np.float32)
        mask[mask == 255.0] = 1.0
        return mask    

class FreeParkingPlacesInferenceDataset(Dataset):
    """
    A dataset class for loading images for inference, without requiring masks. It is designed for 
    use during the inference phase of parking space detection.

    Attributes:
        images_directory (str): Directory containing the images.
        images_filenames (list): List of filenames for the images in the images directory.
        transform (callable, optional): An optional transform to be applied on a sample.

    Args:
        images_directory (str): Path to the directory with images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """
    def __init__(self, images_directory, transform=None):
        self.images_directory = images_directory
        self.images_filenames = os.listdir(images_directory)
        self.transform = transform

    def __len__(self):
        """Returns the size of the dataset."""
        return len(self.images_filenames)

    def __getitem__(self, idx):
        """
        Fetches the image at the index `idx`, applies transformations if any, and returns the image
        along with its original size.

        Args:
            idx (int): Index of the data to fetch.

        Returns:
            tuple: A tuple containing the transformed image and its original size.
        """
        image_filename = self.images_filenames[idx]
        image_path = os.path.join(self.images_directory, image_filename)
        image, _ = DataReader.read_data(image_path)
        original_size = tuple(image.shape[:2])
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, original_size
    