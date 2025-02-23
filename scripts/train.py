import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from transformers import ViTForImageClassification, ViTFeatureExtractor
from src.data_loading.dataset import CIFAR10Dataset
from src.model.vit import ViT

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-4

# Load dataset
transform = Compose([
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dataset = CIFAR10Dataset(root="data/cifar-10", train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize model
model = ViT(num_classes=10)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS}, Batch {batch_idx}, Loss: {loss.item()}")

# Save model
torch.save(model.state_dict(), "models/vit_model.pth")