# Download Datatset in Kaggle 
#### install all framework in requiremments.txt
#### create api token in kaggle
#### mkdir kaggele
#### copy yourpath kaggle.son kaggle\
`kaggle.api.authenticate()`

`kaggle.api.dataset_download_files('ghost5612/chest-x-ray-images-normal-and-pneumonia', unzip=True, path='.')`
# datasrt in kaggle Chest X-ray 

# Model AI 
### Sử dụng model ResNet50 với 50 lớp tích chập


# Intoduction 

### 1.reszing 224,224, 
### 2.Normalization,
### 3.Data Augmentation 

### model Resnet50 

### voi 1 so may su dung gpu thi co the tai thu vien torch tuy bien o day minh co su dung gpu nen se cai rieng 1 thu vien pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 
## con lai chi can pip install -r requirements.txt