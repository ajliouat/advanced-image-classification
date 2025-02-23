import torch
from src.model.vit import ViT

def test_model():
    model = ViT(num_classes=10)
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    assert output.shape == (1, 10)