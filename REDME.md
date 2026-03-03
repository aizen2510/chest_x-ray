🩻 Chest X-ray Pneumonia Detection using ResNet50
📌 Introduction

This project builds a deep learning model to classify chest X-ray images into:

NORMAL

PNEUMONIA

The model is implemented using PyTorch and deployed with Streamlit.

📂 Dataset

Dataset from Kaggle:

Chest X-ray Images (Normal and Pneumonia)

Download dataset:

kaggle.api.authenticate()
kaggle.api.dataset_download_files(
    'ghost5612/chest-x-ray-images-normal-and-pneumonia',
    unzip=True,
    path='.'
)
⚙️ Installation
1️⃣ Install all required packages
pip install -r requirements.txt

If using GPU (CUDA 12.1):

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
🔑 Kaggle API Setup
pip install kaggle

Create API token from Kaggle

Create folder:

mkdir kaggle

Copy your kaggle.json into that folder

🧠 Model Architecture
ResNet50 (50 convolutional layers)

Pretrained on ImageNet

Transfer Learning

Fine-tuning final layers

🔍 Data Processing

Resize images to 224x224

Convert grayscale to 3 channels

Normalize using ImageNet mean/std

Data Augmentation:

RandomResizedCrop

RandomHorizontalFlip

🚀 Training Strategy

CrossEntropyLoss

Adam optimizer

ReduceLROnPlateau scheduler

Early stopping

GPU acceleration (CUDA)

📊 Results
Model	Accuracy
ResNet50 TL	~93-96%
🖥 Deployment

Run Streamlit app:

streamlit run app.py

Upload X-ray image to get prediction.

📁 Requirements.txt
numpy>=1.21.0
pillow>=8.0.0
tqdm>=4.62.0
opencv-python>=4.5.0
kaggle>=1.5.0
streamlit
fastapi
uvicorn
🎯 Future Improvements

Add Grad-CAM visualization

Try EfficientNet

Deploy on cloud (Render / HuggingFace)

🏁 Conclusion

This project demonstrates:

Deep Learning with PyTorch

Transfer Learning

Model fine-tuning

GPU training

Model deployment with Streamlit

Suitable for Machine Learning / AI Internship applications.