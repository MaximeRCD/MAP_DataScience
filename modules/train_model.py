from collections import defaultdict
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
import ternausnet.models
import os
import cv2
from torch.utils.data import DataLoader
import torch
import matplotlib.pyplot as plt
import numpy as np

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

def create_model():
    model = getattr(ternausnet.models, "UNet11")(pretrained=False)
    model = model.to("cuda")
    return model


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
