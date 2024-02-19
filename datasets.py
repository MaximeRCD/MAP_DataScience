import os
from MAP_DataScience.utils import DataReader
from torch.utils.data import Dataset
import numpy as np

class FreeParkingPlacesDataset(Dataset):
    def __init__(self, images_directory, masks_directory, transform=None):
        self.images_directory = images_directory
        self.masks_directory = masks_directory
        self.images_filenames = os.listdir(images_directory)
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
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
        mask = mask.astype(np.float32)
        mask[mask == 255.0] = 1.0
        return mask
class FreeParkingPlacesInferenceDataset(Dataset):
    def __init__(self, images_directory, transform=None):
        self.images_directory = images_directory
        self.images_filenames = os.listdir(images_directory)
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]
        image_path = os.path.join(self.images_directory, image_filename)
        image, _ = DataReader.read_data(image_path)
        original_size = tuple(image.shape[:2])
        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]
        return image, original_size

    def preprocess_mask(self, mask):
        mask = mask.astype(np.float32)
        mask[mask == 255.0] = 1.0
        return mask
