"""
    A class to perform data augmentation for image and mask pairs.
    It allows for various pixel-level transformations
    and can save augmented images and masks to disk.

"""

import random
import os
from typing import List, Tuple
import numpy as np
import cv2
import albumentations as A
from tqdm import tqdm
from joblib import Parallel, delayed
from utils import DataReader, Visualizer, ImageSaver
from constants import TRAINING_IMAGE_DIR, TRAINING_MASK_DIR, \
                      VAL_IMAGE_DIR, VAL_MASK_DIR

random.seed(42)


class DataAugmentationWorker:
    """
    A class to perform data augmentation for image and mask pairs.
    It allows for various pixel-level transformations
    and can save augmented images and masks to disk.

    Attributes:
        pixel_transformations (dict): A dictionary mapping transformation names to
                                      Albumentations transformation objects.
        image_path (str): Path to the original image.
        mask_path (str): Path to the original mask.
        original_image (np.ndarray): The original image read from image_path.
        original_mask (np.ndarray): The original mask read from mask_path.
        all_transformations (List[A.Compose]): List of all transformation combinations as
                                               Albumentations Compose objects.
        augmented_data_root_path (str): Root path to save augmented images and masks.
    """

    def __init__(
        self, image_path: str, mask_path: str, augmented_data_root_path: str
    ) -> None:
        """
        Initializes the DataAugmentationWorker with image paths
        and a root path to save augmented data.

        Parameters:
            image_path (str): Path to the original image.
            mask_path (str): Path to the original mask.
            augmented_data_root_path (str): Root path to save augmented data.

        Returns:
            None
        """
        self.pixel_transformations = {
            "ChannelShuffle": A.ChannelShuffle(p=1),
            "CLAHE": A.CLAHE(p=1),
            "Equalize": A.Equalize(p=1),
            "RandomBrightnessContrast": A.RandomBrightnessContrast(p=1),
        }
        self.image_path = image_path
        self.mask_path = mask_path
        self.original_image, self.original_mask = DataReader.read_data(
            image_path, mask_path
        )
        self.all_transformations = []
        self.augmented_data_root_path = augmented_data_root_path

    def create_transformations(self) -> List[A.Compose]:
        """
        Create a list of transformations by combining rotation
        and pixel transformations using Albumentations library.

        Returns:
            List[A.Compose]: A list of combined transformation objects.
        """

        all_transformations = []
        for i in range(0, 180, 30):
            random_interval = random.choice([[i, i + 30], [-i, -(i + 30)]])
            random_pixel_transformation = random.choice(
                list(self.pixel_transformations.keys())
            )
            transformation = A.Compose(
                [
                    A.Rotate(limit=random_interval, p=1),
                    self.pixel_transformations[random_pixel_transformation],
                ]
            )
            all_transformations.append(transformation)
        return all_transformations

    def apply_transformations(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Apply the created transformations to the original image and mask.

        Returns:
            List[Tuple[np.ndarray, np.ndarray]]: A list of tuples containing the
                                                 transformed images and masks.
        """
        all_transformed_data = []

        if not self.all_transformations:
            self.all_transformations = self.create_transformations()

        for transformation in self.all_transformations:
            transformed_data = transformation(
                image=self.original_image, mask=self.original_mask
            )
            all_transformed_data.append(
                (transformed_data["image"], transformed_data["mask"])
            )
        return all_transformed_data

    def visualize_all_transformations(self) -> None:
        """
        Visualize all the transformations applied to the original
        image and mask using the Visualizer class.

        Returns:
            None
        """
        for transformed_image, transformed_mask in self.apply_transformations():
            Visualizer.visualize(
                transformed_image,
                transformed_mask,
                self.original_image,
                self.original_mask,
            )

    def save_one_image_mask_couple(
        self, image: np.ndarray, mask: np.ndarray, image_path: str, mask_path: str
    ) -> None:
        """
        Save a single image and mask pair to the specified paths.

        Parameters:
            image (np.ndarray): The image to be saved.
            mask (np.ndarray): The mask to be saved.
            image_path (str): Path to save the image.
            mask_path (str): Path to save the mask.

        Returns:
            None
        """
        ImageSaver.save_image(image, image_path)
        ImageSaver.save_image(mask, mask_path)
        print(f"Saved {image_path}")
        print(f"Saved {mask_path}")

    def save_all_transformations(self) -> None:
        """
        Save all transformed image and mask pairs to the augmentation root path.

        Returns:
            None
        """
        all_transformed_data = self.apply_transformations()
        self.save_one_image_mask_couple(
            cv2.cvtColor(self.original_image, cv2.COLOR_RGB2BGR),
            self.original_mask,
            self.image_path.replace("../data", self.augmented_data_root_path),
            self.mask_path.replace("../data", self.augmented_data_root_path),
        )
        for i, ((transformed_image, transformed_mask), transformation) in enumerate(
            zip(all_transformed_data, self.all_transformations)
        ):

            transformations_names = list(
                map(
                    lambda t: t["__class_fullname__"],
                    transformation.to_dict()["transform"]["transforms"],
                )
            )
            image_mask_name_suffix = "_".join(
                [transformations_names[0], str(30 * i), str(30 * i + 30)]
                + transformations_names[1:]
            ).lower()

            image_path = self.image_path.replace(
                "../data", self.augmented_data_root_path
            ).replace(".png", "_" + image_mask_name_suffix + ".png")
            mask_path = self.mask_path.replace(
                "../data", self.augmented_data_root_path
            ).replace("_mask.png", "_" + image_mask_name_suffix + "_mask.png")

            self.save_one_image_mask_couple(
                transformed_image, transformed_mask, image_path, mask_path
            )


def apply_data_augmentation(
    image_path: str, mask_path: str, augmented_data_root_path: str
) -> None:
    """
    Applies data augmentation to an image and its corresponding
    mask and saves the augmented results.

    This function creates a DataAugmentationWorker instance, which generates a series of augmented
    images and masks based on predefined transformations. The augmented images and masks are then
    saved to a specified root directory.

    Parameters:
        image_path (str): Path to the original image that will be augmented.
        mask_path (str): Path to the original mask that will be augmented.
        augmented_data_root_path (str): Root directory where the
                                        augmented images and masks will be saved.

    Returns:
        None

    Examples:
        >>> apply_data_augmentation(
                image_path='path/to/original/image.png',
                mask_path='path/to/original/mask.png',
                augmented_data_root_path='path/to/save/augmented/data'
            )
    """
    worker = DataAugmentationWorker(
        image_path=image_path,
        mask_path=mask_path,
        augmented_data_root_path=augmented_data_root_path,
    )
    worker.save_all_transformations()


if __name__ == '__main__':

    _ = Parallel(n_jobs=-1)(
        delayed(apply_data_augmentation)(
            os.path.join(TRAINING_IMAGE_DIR, image_name),
            os.path.join(TRAINING_MASK_DIR, image_name.replace(".png", "_mask.png")),
            "../data",
        )
        for image_name in tqdm(os.listdir("../data/images"))
    )

    _ = Parallel(n_jobs=-1)(
        delayed(apply_data_augmentation)(
            os.path.join(VAL_IMAGE_DIR, image_name),
            os.path.join(VAL_MASK_DIR, image_name.replace(".png", "_mask.png")),
            "../data",
        )
        for image_name in tqdm(os.listdir("../data/images"))
   )
