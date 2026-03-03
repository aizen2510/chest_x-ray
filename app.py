import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = get_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

st.title("Chest X-ray Pneumonia Detection")
st.write("Upload a chest X-ray image to predict NORMAL or PNEUMONIA")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)

    class_names = ["NORMAL", "PNEUMONIA"]
    prediction = class_names[predicted.item()]

    st.subheader(f"Prediction: {prediction}")