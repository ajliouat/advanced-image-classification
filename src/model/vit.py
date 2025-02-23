import torch
import torch.nn as nn
from transformers import ViTForImageClassification

class ViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.vit = ViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224",
            num_labels=num_classes
        )

    def forward(self, x):
        return self.vit(x).logits