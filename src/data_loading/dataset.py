from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize

class CIFAR10Dataset(CIFAR10):
    def __init__(self, root, train=True, transform=None):
        super().__init__(root, train=train, download=True, transform=transform)