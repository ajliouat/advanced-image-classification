from src.data_loading.dataset import CIFAR10Dataset

def test_data_loading():
    dataset = CIFAR10Dataset(root="data/cifar-10", train=True)
    assert len(dataset) == 50000