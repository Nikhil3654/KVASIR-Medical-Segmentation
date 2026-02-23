import torch
from src.models.deeplabv3 import DeepLabV3MobileNet


def test_deeplab_forward_shape():
    m = DeepLabV3MobileNet(out_channels=1)
    x = torch.randn(2, 3, 128, 128)
    y = m(x)
    assert y.shape[0] == 2
    assert y.shape[1] == 1