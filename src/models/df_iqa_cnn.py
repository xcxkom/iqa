import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader


def create_regression_head(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 2048),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(2048, 1024),
        nn.ReLU(),
        nn.Dropout(p=0.25),
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 1),
    )

def create_iqa_model(model_name):
    if model_name == "resnet50":
        base_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features = base_model.fc.in_features
        base_model.fc = create_regression_head(in_features)

    elif model_name == "densenet201":
        base_model = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT)
        in_features = base_model.classifier.in_features
        base_model.classifier = create_regression_head(in_features)

    elif model_name == "vgg16":
        base_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = base_model.classifier[0].in_features
        base_model.classifier = create_regression_head(in_features)

    return base_model
