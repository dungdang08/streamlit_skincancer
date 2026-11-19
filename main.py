"""Streamlit app: classify a skin mole image as benign or malignant with two CNNs."""

import random
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

APP_DIR = Path(__file__).parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class indices follow torchvision's ImageFolder ordering (alphabetical folder names).
CLASS_NAMES = ["BENIGN", "MALIGNANT"]
NUM_SAMPLE_IMAGES = 15

# Same preprocessing used during training (ImageNet normalisation).
TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODELS = {
    "VGG16": {"builder": models.vgg16_bn, "weights": "model_weights.pth"},
    "AlexNet": {"builder": models.alexnet, "weights": "model_weights2.pth"},
}


@st.cache_resource
def load_model(name: str) -> nn.Module:
    """Build the architecture, swap in a 2-class head, and load the trained weights."""
    spec = MODELS[name]
    model = spec["builder"](weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(CLASS_NAMES))
    state_dict = torch.load(APP_DIR / spec["weights"], map_location="cpu")
    model.load_state_dict(state_dict)
    return model.to(DEVICE).eval()


@torch.no_grad()
def predict(model: nn.Module, image: Image.Image) -> str:
    batch = TRANSFORM(image).unsqueeze(0).to(DEVICE)
    return CLASS_NAMES[model(batch).argmax(dim=1).item()]


st.title("TechLabs Project 05: Skin Cancer Classification")
st.write("Using CNNs to classify whether a skin mole is benign or malignant.")

st.sidebar.markdown("""
# CNN models

## Result 1

- Model: pretrained VGG16 (with batch norm)
- Optimiser: SGD with scheduler
- Loss function: Cross Entropy Loss
- Batch size: 16
- Learning rate: 0.001
- Epochs: 20 (early stopping)
- Test accuracy: ~89%

## Result 2

- Model: pretrained AlexNet
- Optimiser: SGD with scheduler
- Loss function: Cross Entropy Loss
- Batch size: 30
- Learning rate: 0.001
- Epochs: 25 (early stopping)
- Test accuracy: ~86%
""")

uploaded = st.file_uploader("Upload a photo of a skin mole", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    caption = "Uploaded image"
else:
    image = Image.open(APP_DIR / "sample_images" / f"image{random.randint(1, NUM_SAMPLE_IMAGES)}.jpg")
    caption = "Random sample image (upload your own above)"

image = image.convert("RGB")
st.image(image, caption=caption, use_column_width=True)

with st.spinner("Making predictions ..."):
    for i, name in enumerate(MODELS, start=1):
        st.subheader(f"Result {i} ({name}):")
        st.write(predict(load_model(name), image))
