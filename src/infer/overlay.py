from __future__ import annotations

import numpy as np


def overlay_mask_rgb(image_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must be HxWx3")

    m = mask.astype(bool)
    out = image_rgb.copy()

    red = out[:, :, 0].astype(np.float32)
    red[m] = (1 - alpha) * red[m] + alpha * 255.0
    out[:, :, 0] = np.clip(red, 0, 255).astype(np.uint8)
    return out