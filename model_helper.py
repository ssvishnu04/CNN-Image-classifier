import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from huggingface_hub import hf_hub_download
import streamlit as st
import torch.nn.functional as F


class_names = ['Front Breakage', 'Front Crushed', 'Front Normal',
               'Rear Breakage', 'Rear Crushed', 'Rear Normal']


class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.model = models.resnet50(weights='DEFAULT')

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def load_model_from_hf():
    token = st.secrets.get("HF_TOKEN")

    if not token:
        raise ValueError("HF_TOKEN not found in Streamlit secrets")

    model_path = hf_hub_download(
        repo_id="vyadavalli3471/car-damage-prediction-cnn",
        filename="saved_model.pth",
        token=token
    )

    model = CarClassifierResNet()
    model.load_state_dict(
        torch.load(model_path, map_location=torch.device("cpu"))
    )
    model.eval()
    return model


@st.cache_resource
def get_model():
    return load_model_from_hf()


def predict(image):
    image = image.convert("RGB")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    model = get_model()

    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    confidence = confidence.item()

    if confidence < 0.5:
        return "Not a valid car image", confidence

    return class_names[predicted_class.item()], confidence
