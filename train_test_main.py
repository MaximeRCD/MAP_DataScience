import os
import cv2
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import matplotlib.pyplot as plt
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ternausnet.models
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from tqdm import tqdm
from modules.inference_model import load_model, predict
from modules.utils import Visualizer, preprocess_mask, check_cuda_availability
from modules.constants import *
from modules.datasets import FreeParkingPlacesDataset, FreeParkingPlacesInferenceDataset


if __name__ == '__main__':
    visualizer_worker = Visualizer()
    TEST_IMAGE_FILENAMES = os.listdir(TEST_IMAGE_DIR)

    # ### 2.5 Definition of Training & Validation DataSets
    # training_ds = FreeParkingPlacesDataset(
    #     images_directory=TRAINING_IMAGE_DIR,
    #     masks_directory=TEST_MASK_DIR,
    #     transform=A.Compose([
    #         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    #         ToTensorV2()])
    # )

    # validation_ds = FreeParkingPlacesDataset(
    #     images_directory=VAL_IMAGE_DIR,
    #     masks_directory=VAL_MASK_DIR,
    #     transform=A.Compose([
    #         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
    #         ToTensorV2()])
    # )

    # train_loader = DataLoader(training_ds, batch_size=VISUALIZER_BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(validation_ds, batch_size=VISUALIZER_BATCH_SIZE, shuffle=False)
    # #### Visualisation des données dans le DataSet de Training
    # #Pour le masque, nous avons décidé d'afficher également le pourcentage de pixels représentant les places de parking pour avoir une idée de la proportion de labels 1 et 0.
    # visualizer_worker.visualize_batch(train_loader)
    # #### Visualisation des données dans le DataSet de Validation
    # #Pour le masque, nous avons également décidé d'afficher le pourcentage de pixels représentant les places de parking pour avoir une idée de la proportion de labels 1 et 0.
    # visualizer_worker.visualize_batch(val_loader)
    # #### Visualisation des données dans le DataSet de Test
    # visualizer_worker.display_image_grid(TEST_IMAGE_FILENAMES, TEST_IMAGE_DIR, TEST_MASK_DIR)

    
    
    path_to_saved_model = "./cross_entropy_weighted10_batch64_32_16.pth"

    model = load_model(path_to_saved_model)
    # 4. Test du modèle sur le jeu de test
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

    visualizer_worker.display_image_grid(TEST_IMAGE_FILENAMES, TEST_IMAGE_DIR, TEST_MASK_DIR, predicted_masks=predicted_masks)
