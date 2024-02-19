import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_cuda_availability():
    """
    Checks for the availability of a CUDA-compatible GPU.

    This function checks if a CUDA-compatible GPU is available on the machine.
    It prints a message indicating whether a CUDA-compatible GPU has been detected or not.

    If a GPU is detected, it means you can perform your tensor computation operations
    on the GPU to speed up calculations. Otherwise, it's suggested to use Google Colab,
    which provides free access to GPU instances.
    """
    if torch.cuda.is_available():
        print("A CUDA-compatible GPU has been detected.")
    else:
        print("No CUDA-compatible GPU has been detected. It's probably better to use a Google Colab instance.")

def preprocess_mask(mask):
    mask = mask.astype(np.float32)
    mask[mask == 255.0] = 1.0
    return mask

class DataReader:
    """
    A helper class to read image and mask data from specified paths.
    """
    @staticmethod
    def read_data(image_path, mask_path=None):
        """
        Reads an image and its corresponding mask from the given file paths.
        Args:
            image_path (str): Path to the image file.
            mask_path (str): Path to the mask file.
        Returns:
            tuple: A tuple containing the original image and mask as numpy arrays.
        """
        image = cv2.imread(image_path)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if mask_path :
          original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
          return original_image, original_mask
        else:
          return original_image, None
        
class Visualizer:
    """
    A helper class for visualizing images and masks.
    """
    @staticmethod
    def visualize(image, mask, original_image=None, original_mask=None, title=None):
        """
        Visualizes the original and transformed images and masks.
        Args:
            image (numpy.ndarray): Transformed image to be visualized.
            mask (numpy.ndarray): Transformed mask to be visualized.
            original_image (numpy.ndarray, optional): Original image for comparison.
            original_mask (numpy.ndarray, optional): Original mask for comparison.
            title (str, optional): Title for the visualization plot.
        """
        fontsize = 18
        if original_image is None and original_mask is None:
            f, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].imshow(image)
            ax[1].imshow(mask)
        else:
            f, ax = plt.subplots(2, 2, figsize=(8, 8))
            f.suptitle(title, fontsize=fontsize)
            ax[0, 0].imshow(original_image)
            ax[0, 0].set_title('Original image', fontsize=fontsize)
            ax[1, 0].imshow(original_mask)
            ax[1, 0].set_title('Original mask', fontsize=fontsize)
            ax[0, 1].imshow(image)
            ax[0, 1].set_title('Transformed image', fontsize=fontsize)
            ax[1, 1].imshow(mask)
            ax[1, 1].set_title('Transformed mask', fontsize=fontsize)


    @staticmethod
    def visualize_batch(data_loader):
        """
        Visualizes the first three images and masks from the given data loader.

        Args:
            data_loader (DataLoader): A PyTorch DataLoader object containing images and masks.
        """
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))  # 3 rows, 2 columns
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        mean = mean[:, None, None]
        std = std[:, None, None]
        for i, (images, masks) in enumerate(data_loader):
            if i == 3:  # Only show first 3 images and masks
                break

            axs[i, 0].imshow((torch.clamp((images[i]*std+mean).permute(1, 2, 0), 0, 1)*255).numpy().astype(np.uint8))  # Change CxHxW to HxWxC
            axs[i, 0].set_title(f'Image {i}')
            axs[i, 0].axis('off')

            axs[i, 1].imshow(masks[i], cmap='gray')
            axs[i, 1].set_title(f'Mask {i} & Pourcentage of parking : {masks[i][masks[i]==1.0].shape[0]/(256*256)}')
            axs[i, 1].axis('off')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def display_image_grid(images_filenames, images_directory, masks_directory, predicted_masks=None):
        cols = 3 if predicted_masks else 2
        rows = len(images_filenames)
        figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
        for i, image_filename in enumerate(images_filenames):
            image = cv2.imread(os.path.join(images_directory, image_filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(masks_directory, image_filename.replace(".png", "_mask.png")), cv2.IMREAD_GRAYSCALE)
            mask = preprocess_mask(mask)
            ax[i, 0].imshow(image)
            ax[i, 1].imshow(mask, interpolation="nearest")

            ax[i, 0].set_title("Image")
            ax[i, 1].set_title("Ground truth mask")

            ax[i, 0].set_axis_off()
            ax[i, 1].set_axis_off()

            if predicted_masks:
                predicted_mask = predicted_masks[i]
                ax[i, 2].imshow(predicted_mask, interpolation="nearest")
                ax[i, 2].set_title("Predicted mask")
                ax[i, 2].set_axis_off()
        plt.tight_layout()
        plt.show()
