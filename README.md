# TechLabs Project 05: Skin Cancer Classification

A Streamlit web app that classifies a photo of a skin mole as **benign** or **malignant** using two fine-tuned convolutional neural networks (VGG16 and AlexNet).

> **Disclaimer:** This is a student project for educational purposes only. It is not a medical device and must not be used for diagnosis. See a dermatologist about any skin concern.

## Models

Both models were pretrained on ImageNet and fine-tuned on the dataset in `data/`, with the final classifier layer replaced by a 2-class output.

| | Result 1 | Result 2 |
|---|---|---|
| Architecture | VGG16 (batch norm) | AlexNet |
| Weights file | `model_weights.pth` | `model_weights2.pth` |
| Optimiser | SGD + LR scheduler | SGD + LR scheduler |
| Loss | Cross Entropy | Cross Entropy |
| Batch size | 16 | 30 |
| Learning rate | 0.001 | 0.001 |
| Epochs | 20 (early stopping) | 25 (early stopping) |
| Test accuracy | ~89% | ~86% |

Input images are resized to 256 px, center-cropped to 224×224 and normalised with ImageNet mean/std.

## Project structure

```
main.py               Streamlit app
requirements.txt      Python dependencies
model_weights.pth     Trained VGG16 weights (~540 MB)
model_weights2.pth    Trained AlexNet weights (~230 MB)
image1.jpg … image15.jpg   Sample images shown when nothing is uploaded
data/
  train/{benign,malignant}/   Training images (1440 / 1197)
  test/{benign,malignant}/    Test images (360 / 300)
```

The weight files are large and are tracked with [Git LFS](https://git-lfs.com/). Run `git lfs pull` after cloning, or the app will fail to load them.

## Getting started

Requires Python 3.8–3.11 (the pinned PyTorch 2.0.1 does not support newer versions).

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run main.py
```

The app opens at http://localhost:8501.

## Usage

1. Upload a `.jpg`, `.jpeg` or `.png` photo of a skin mole.
2. Both models predict **BENIGN** or **MALIGNANT**, and the app shows both results.

If you don't upload anything, the app shows a random sample image and classifies it. The models load once and are cached, so only the first prediction is slow.
