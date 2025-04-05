import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights

def get_resnet_model(num_classes=1, unfreeze_last_layer=True):
    # Use updated weights argument
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

    # Fine-tune last layer (layer4 + fc)
    if unfreeze_last_layer:
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes)
    )
    return model
