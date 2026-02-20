import numpy as np
from src.infer.overlay import overlay_mask_rgb


def test_overlay_shape_and_type():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:5, 2:5] = 1

    out = overlay_mask_rgb(img, mask, alpha=0.5)
    assert out.shape == img.shape
    assert out.dtype == np.uint8