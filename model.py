import torch 
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes2, freeze_backbone=True):

    # sd resnet50 
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # freseze backbone 
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # thay đổi lớp fully connected cuối cùng để phù hợp với số lượng lớp của bài toán
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 255),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(255, num_classes2)
    )

    return model