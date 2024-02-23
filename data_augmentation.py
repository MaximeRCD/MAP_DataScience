import random
import os
from typing import Optional, List, Tuple
import numpy as np
from cv2 import cv2
import matplotlib.pyplot as plt
import albumentations as A
from tqdm import tqdm
from joblib import Parallel, delayed

random.seed(42)


class DataReader:
    """
    Helper class for reading image and mask data from specified file paths.

    This class uses OpenCV to read and process image and mask data, returning them as NumPy arrays.
    """

    @staticmethod
    def read_data(image_path: str, mask_path: str) -> tuple:
        """
        Reads an image and its corresponding mask from the provided file paths.

        Parameters:
        image_path (str): Path to the image file.
        mask_path (str): Path to the mask file.

        Returns:
        (np.ndarray, np.ndarray): A tuple containing the original image in RGB format and the mask in grayscale,
                                  both as NumPy arrays.

        Raises:
        FileNotFoundError: If the image_path or mask_path does not correspond to an existing file.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"The image file {image_path} does not exist.")
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if original_mask is None:
            raise FileNotFoundError(f"The mask file {mask_path} does not exist.")
        return original_image, original_mask

    @staticmethod
    def read_mask(mask_path: str) -> np.ndarray:
        """
        Reads a mask from the provided file path.

        Parameters:
        mask_path (str): Path to the mask file.

        Returns:
        np.ndarray: The original mask as a NumPy array in grayscale.

        Raises:
        FileNotFoundError: If the mask_path does not correspond to an existing file.
        """
        original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if original_mask is None:
            raise FileNotFoundError(f"The mask file {mask_path} does not exist.")
        return original_mask


class DirectoryManager:
    """
    Helper class for managing directory operations.

    This class provides a static method to ensure that a directory exists for a given file path,
    creating the directory if necessary. It leverages the `os` library for interacting with the file system.
    """

    @staticmethod
    def ensure_directory_exists(file_path: str) -> None:
        """
        Ensures that the directory for the specified file path exists, creating it if it does not.

        Parameters:
        file_path (str): The full file path for which the directory needs to be verified and potentially created.
                         This is not the path to the directory itself, but to a file within that directory.

        Returns:
        None: This method does not return anything.

        Side Effects:
        Creates a new directory at the specified path if it does not already exist.

        Raises:
        OSError: If the directory cannot be created for reasons such as insufficient permissions or invalid path.
        """
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
            except OSError as e:
                raise OSError(f"Unable to create directory {directory}: {e.strerror}")


class Visualizer:
    """
    Helper class for visualizing images and masks.

    Provides a static method to display images and masks side-by-side for comparison,
    using Matplotlib to create a plot with the original and transformed images and masks.
    """

    @staticmethod
    def visualize(
        image: np.ndarray,
        mask: np.ndarray,
        original_image: Optional[np.ndarray] = None,
        original_mask: Optional[np.ndarray] = None,
        title: Optional[str] = None,
    ) -> None:
        """
        Visualizes the original and transformed images and masks side by side for comparison.

        Parameters:
        image (np.ndarray): The transformed image to be visualized.
        mask (np.ndarray): The transformed mask to be visualized.
        original_image (np.ndarray, optional): The original image for comparison. If None, only the transformed image and mask are shown.
        original_mask (np.ndarray, optional): The original mask for comparison. If None, only the transformed image and mask are shown.
        title (str, optional): The title for the visualization plot. If provided, it adds a title to the entire plot.

        Returns:
        None: The method displays the plot and does not return any value.

        Side Effects:
        Displays a matplotlib figure containing the visualizations of the provided images and masks.

        Raises:
        ValueError: If the provided images or masks are not of type np.ndarray.
        """
        if not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray):
            raise ValueError("Image and mask must be numpy arrays.")

        if original_image is not None and not isinstance(original_image, np.ndarray):
            raise ValueError("Original image must be a numpy array if provided.")

        if original_mask is not None and not isinstance(original_mask, np.ndarray):
            raise ValueError("Original mask must be a numpy array if provided.")

        fontsize = 18
        if original_image is None and original_mask is None:
            f, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].imshow(image)
            ax[1].imshow(mask)
        else:
            f, ax = plt.subplots(2, 2, figsize=(8, 8))
            if title:
                f.suptitle(title, fontsize=fontsize)
            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title("Original image", fontsize=fontsize)
            ax[1, 0].imshow(original_mask)
            ax[1, 0].set_title("Original mask", fontsize=fontsize)
            ax[0, 1].imshow(image)
            ax[0, 1].set_title("Transformed image", fontsize=fontsize)
            ax[1, 1].imshow(mask)
            ax[1, 1].set_title("Transformed mask", fontsize=fontsize)


class ImageSaver:
    """
    Helper class for saving images to the disk.

    This class provides a static method to save images in various formats to a specified path on the disk.
    It ensures that the necessary directories exist before saving the image using OpenCV's imwrite function.
    """

    @staticmethod
    def save_image(image: np.ndarray, file_path: str) -> None:
        """
        Saves an image to the specified file path.

        Parameters:
        image (np.ndarray): The image array to be saved. Expected to be in a format compatible with OpenCV, such as an 8-bit or floating-point 32-bit array.
        file_path (str): The complete target file path where the image will be saved. If the directory structure does not exist, it will be created.

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
                f"Image could not be saved to {file_path}. There might be an issue with the file path or permissions."
            )


class DataAugmentationWorker:
    """
    A class to perform data augmentation for image and mask pairs. 
    It allows for various pixel-level transformations
    and can save augmented images and masks to disk.

    Attributes:
        pixel_transformations (dict): A dictionary mapping transformation names to Albumentations transformation objects.
        image_path (str): Path to the original image.
        mask_path (str): Path to the original mask.
        original_image (np.ndarray): The original image read from image_path.
        original_mask (np.ndarray): The original mask read from mask_path.
        all_transformations (List[A.Compose]): List of all transformation combinations as Albumentations Compose objects.
        augmented_data_root_path (str): Root path to save augmented images and masks.
    """

    def __init__(
        self, image_path: str, mask_path: str, augmented_data_root_path: str
    ) -> None:
        """
        Initializes the DataAugmentationWorker with image paths and a root path to save augmented data.

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
        Create a list of transformations by combining rotation and pixel transformations using Albumentations library.

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
            List[Tuple[np.ndarray, np.ndarray]]: A list of tuples containing the transformed images and masks.
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
        Visualize all the transformations applied to the original image and mask using the Visualizer class.

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
    Applies data augmentation to an image and its corresponding mask and saves the augmented results.

    This function creates a DataAugmentationWorker instance, which generates a series of augmented
    images and masks based on predefined transformations. The augmented images and masks are then
    saved to a specified root directory.

    Parameters:
        image_path (str): Path to the original image that will be augmented.
        mask_path (str): Path to the original mask that will be augmented.
        augmented_data_root_path (str): Root directory where the augmented images and masks will be saved.

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


training_image_root_path = "../data/train/images/"
training_mask_root_path = "../data/train/masks/"

_ = Parallel(n_jobs=-1)(
    delayed(apply_data_augmentation)(
        os.path.join(training_image_root_path, image_name),
        os.path.join(training_mask_root_path, image_name.replace(".png", "_mask.png")),
        "../data",
    )
    for image_name in tqdm(os.listdir("../data/train/images"))
)


val_image_root_path = "../data/val/images/"
val_mask_root_path = "../data/val/masks/"

_ = Parallel(n_jobs=-1)(
    delayed(apply_data_augmentation)(
        os.path.join(val_image_root_path, image_name),
        os.path.join(val_mask_root_path, image_name.replace(".png", "_mask.png")),
        "../data",
    )
    for image_name in tqdm(os.listdir("../data/val/images"))
)
