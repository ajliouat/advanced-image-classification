import torch
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
from src.model.vit import ViT

# Load model
model = ViT(num_classes=10)
model.load_state_dict(torch.load("models/vit_model.pth"))
model.eval()

# Load and preprocess image
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
image = Image.open("<path_to_image>").convert("RGB")
image = transform(image).unsqueeze(0)

# Make prediction
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print(f"Predicted class: {predicted.item()}")