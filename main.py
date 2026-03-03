from Preprocessing import getDataloader
from model import get_model
from train import train_model

train_loader, val_loader, test_loader = getDataloader(
    "chest_xray/train",
    "chest_xray/val",
    "chest_xray/test"
)


model = get_model(num_classes=2, freeze_backbone=False)

train_model(model, train_loader, val_loader, epochs=15, lr=1e-4)