import torch
from torchvision import transforms,datasets
from torch.utils.data import DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225]) # x′= x - u / σ , gia tri goc - gia tri trung binh(mean) / do lech chuan(std),x' = (x - mean) / std

])
"""
def getDataloader(train_dir ,val_dir, test_dir, batch_size=20):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # train preprocessing
    transforms_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), # tranh overfit
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # val preprocessing
    transforms_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # cắt ở chính giữa 
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # test preprosessing
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])


    train_dir = "chest_xray/train"
    val_dir = "chest_xray/val"
    test_dir = "chest_xray/test"


    train_ds = datasets.ImageFolder(train_dir, transform=transforms_train)
    val_ds = datasets.ImageFolder(val_dir, transform=transforms_val)
    test_ds = datasets.ImageFolder(test_dir, transform=test_transform)

    """
    batch_size=20: Số lượng ảnh mỗi lần đưa vào mô hình.
    (Cân bằng giữa tốc độ và bộ nhớ GPU.)
    shuffle=True: Trộn ngẫu nhiên dữ liệu mỗi epoch (chỉ dùng cho train).
    num_workers=2: Số luồng đọc dữ liệu song song (tăng tốc load ảnh).
    pin_memory=True: Tăng hiệu suất khi dùng GPU (copy dữ liệu nhanh hơn sang
    """

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=64, num_workers=4, pin_memory=True)


    return train_loader, val_loader, test_loader



