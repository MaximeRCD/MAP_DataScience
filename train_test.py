# Notebook d'entrainement et de test du modèle de segmentation sémantiques des images de Parking Libre
Pour pouvoir entrainer le modèle U_NET11 fourni par la librairie [Ternausnet](https://pypi.org/project/ternausnet/), nous avons eu besoin d'utiliser des ressources GPU. Pour ce faire nous avons utiliser Google Collab et avons donc besoin de synchroniser l'environnement Google Collab avec notre Drive Partagé afin de pouvoir accéder aux données.
# from google.colab import drive
# drive.mount('/content/drive')
#pip install ternausnet
##  1. Import des librairies nécessaires
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
from collections import defaultdict


## 2. Configuration de l'environnement
### 2.1 Test de la disponibilité d'un GPU compatible avec Pytorch

### 2.2 Définition de classes utilitaires







### 2.3 Définition des classes afin de créer les datasets
### 2.4 Constants & Helper Entities definitions
visualizer_worker = Visualizer()


# training_image_dir = "../data/train/images/"
# training_mask_dir = "../data/train/masks/"

# val_image_dir = "../data/val/images/"
# val_mask_dir = "../data/val/masks/"

# test_image_dir = "../data/test/images/"
# test_mask_dir = "../data/test/masks/"

test_image_filenames = os.listdir("../data/test/images/")



### 2.5 Definition of Training & Validation DataSets
training_ds = FreeParkingPlacesDataset(
    images_directory=training_image_dir,
    masks_directory=training_mask_dir,
    transform=A.Compose([
#         A.Resize(height=256, width=256, p=1), # Needed because all images do not have the same size. Will be deleted !
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()])
)

validation_ds = FreeParkingPlacesDataset(
    images_directory=val_image_dir,
    masks_directory=val_mask_dir,
    transform=A.Compose([
#         A.Resize(height=256, width=256, p=1), # Needed because all images do not have the same size. Will be deleted !
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
        ToTensorV2()])
)

train_loader = DataLoader(training_ds, batch_size=visualizer_batch_size, shuffle=True)
val_loader = DataLoader(validation_ds, batch_size=visualizer_batch_size, shuffle=False)
#### Visualisation des données dans le DataSet de Training
Pour le masque, nous avons décidé d'afficher également le pourcentage de pixels représentant les places de parking pour avoir une idée de la proportion de labels 1 et 0.
visualizer_worker.visualize_batch(train_loader)
#### Visualisation des données dans le DataSet de Validation
Pour le masque, nous avons également décidé d'afficher le pourcentage de pixels représentant les places de parking pour avoir une idée de la proportion de labels 1 et 0.
visualizer_worker.visualize_batch(val_loader)
#### Visualisation des données dans le DataSet de Test
visualizer_worker.display_image_grid(test_image_filenames, test_image_dir, test_mask_dir)
## 3. Définition et Entrainement du model U_Net11
### 3.1 Classes et Fonctions utilitaires
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]

    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )

def train(train_loader, model, criterion, optimizer, epoch, params):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, target) in enumerate(stream, start=1):
        images = images.to(params["device"], non_blocking=True)
        target = target.to(params["device"], non_blocking=True)
        output = model(images).squeeze(1)
        loss = criterion(output, target)
        metric_monitor.update(f"{criterion.__class__()}", loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        stream.set_description(
            "Epoch: {epoch}. Train.      {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
        )

def validate(val_loader, model, criterion, epoch, params):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    with torch.no_grad():
        for i, (images, target) in enumerate(stream, start=1):
            images = images.to(params["device"], non_blocking=True)
            target = target.to(params["device"], non_blocking=True)
            output = model(images).squeeze(1)
            loss = criterion(output, target)
            metric_monitor.update(f"{criterion.__class__()}", loss.item())
            stream.set_description(
                "Epoch: {epoch}. Validation. {metric_monitor}".format(epoch=epoch, metric_monitor=metric_monitor)
            )
def train_and_validate(model, train_dataset, val_dataset, params):
    # Initialize DataLoaders for training and validation datasets
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
    )

    # We introduce Weigth in the loss because parking place class is under represented
    criterion = nn.BCEWithLogitsLoss(pos_weight=10*torch.ones([256])).to(params["device"])


    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])

    # Training and validation loop
    for epoch in range(1, params["epochs"] + 1):
        train(train_loader, model, criterion, optimizer, epoch, params)
        validate(val_loader, model, criterion, epoch, params)

    return model

def predict(model, params, test_dataset, batch_size):
    # Initialize DataLoader for the test dataset
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
    )

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
def create_model():
    model = getattr(ternausnet.models, "UNet11")(pretrained=False)
    model = model.to("cuda")
    return model


def load_model(model_file_path):
    model = getattr(ternausnet.models, "UNet11")(pretrained=False)
    model.load_state_dict(torch.load(model_file_path, map_location=params['device']))
    model = model.to(params['device'])
    return model
### 3.2 Entrainement et sauvegarde d'un modèle U_Net11
#### Initialisation du modèle
# model = create_model()
#### Initialisation des paramètre de la boucle d'entrainement
# all_params = [{
#     "model": "UNet11",
#     "device": "cuda",
#     "lr": 0.01,
#     "class_weights": [1.0, 3.0],
#     "batch_size": 64,
#     # "num_workers": 4,
#     "epochs": 5,
# },
#               {
#     "model": "UNet11",
#     "device": "cuda",
#     "lr": 0.003,
#     "class_weights": [1.0, 3.0],
#     "batch_size": 64,
#     # "num_workers": 4,
#     "epochs": 5,
# },
#               {
#     "model": "UNet11",
#     "device": "cuda",
#     "lr": 0.001,
#     "class_weights": [1.0, 3.0],
#     "batch_size": 64,
#     # "num_workers": 4,
#     "epochs": 5,
# },
#  {
#     "model": "UNet11",
#     "device": "cuda",
#     "lr": 0.001,
#     "class_weights": [1.0, 3.0],
#     "batch_size": 32,
#     # "num_workers": 4,
#     "epochs": 5,
# },
# {
#     "model": "UNet11",
#     "device": "cuda",
#     "lr": 0.001,
#     "class_weights": [1.0, 3.0],
#     "batch_size": 16,
#     # "num_workers": 4,
#     "epochs": 5,
# }]
#### Entrainement et sauvegarde
# for params in all_params:
#     model = train_and_validate(model, training_ds, validation_ds, params)


# torch.save(model.state_dict(), "./drive/MyDrive/ML_with_Python_Project_Segmentation/cross_entropy_weighted10_batch64_32_16_lr_01_001_0001.pth")
### 3.3 Réutilisation d'un modèle sauvegardé
path_to_saved_model = "./cross_entropy_weighted10_batch64_32_16.pth"

model = load_model(path_to_saved_model)
## 4. Test du modèle sur le jeu de test
test_transform = A.Compose(
    [A.Resize(256, 256), A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), ToTensorV2()]
)
test_dataset = FreeParkingPlacesInferenceDataset(test_image_dir, transform=test_transform)

predictions = predict(model, params, test_dataset, batch_size=16)

predicted_masks = []
for predicted_256x256_mask, original_height, original_width in predictions:
    full_sized_mask = A.resize(
        predicted_256x256_mask, height=original_height, width=original_width, interpolation=cv2.INTER_NEAREST
    )
    predicted_masks.append(full_sized_mask)

visualizer_worker.display_image_grid(test_image_filenames, test_image_dir, test_mask_dir, predicted_masks=predicted_masks)
