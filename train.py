import torch
import torch.nn as nn
from tqdm import tqdm
from model import get_model
from Preprocessing import getDataloader


def train_model(model, train_loader, val_loader, epochs=10, lr=1e-4):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5
    )

    best_val_loss = float("inf")
    patience = 5
    trigger_times = 0

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")

        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = 100 * correct / total
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_acc = 100 * correct / total
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.2f}%")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")
            trigger_times = 0
        else:
            trigger_times += 1
            print(f"Early stopping counter: {trigger_times}/{patience}")

            if trigger_times >= patience:
                print("Early stopping!")
                break

    print("Training complete.")

if __name__ == "__main__":

    train_loader, val_loader, _ = getDataloader(
        "chest_xray/train",
        "chest_xray/val",
        "chest_xray/test"
    )

    model = get_model(num_classes=2, freeze_backbone=False)

    train_model(model, train_loader, val_loader, epochs=10)