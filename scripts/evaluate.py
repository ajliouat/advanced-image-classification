import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from src.model.vit import ViT

# Load dataset
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
test_dataset = CIFAR10(root="data/cifar-10", train=False, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load model
model = ViT(num_classes=10)
model.load_state_dict(torch.load("models/vit_model.pth"))
model.eval()

# Evaluation
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")