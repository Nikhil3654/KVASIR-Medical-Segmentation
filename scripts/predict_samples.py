from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np
import torch

from src.data.kvasir_dataset import KvasirSegDataset
from src.infer.overlay import overlay_mask_rgb
from src.models.unet import UNet
from src.models.deeplabv3 import DeepLabV3MobileNet
from pathlib import WindowsPath
import torch.serialization

torch.serialization.add_safe_globals([WindowsPath])

def load_unet(ckpt_path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    # base is not needed if you hardcode base=32. Keep consistent with training.
    model = UNet(in_channels=3, out_channels=1, base=32).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def load_deeplab(ckpt_path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt

    model = DeepLabV3MobileNet(out_channels=1).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_overlay(model: torch.nn.Module, img_t: torch.Tensor, thresh: float = 0.5) -> np.ndarray:
    device = next(model.parameters()).device
    x = img_t.unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    pred = (prob > thresh).astype(np.uint8)

    img = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    over = overlay_mask_rgb(img, pred, alpha=0.45)
    return over


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pairs_csv = Path("data/processed/pairs.csv")
    if not pairs_csv.exists():
        raise RuntimeError("pairs.csv missing. Run python -m scripts.build_pairs first.")

    ckpt_unet = Path("outputs/checkpoints/unet_best.pt")
    ckpt_dl = Path("outputs/checkpoints/deeplabv3_best.pt")
    if not ckpt_unet.exists():
        raise RuntimeError("Missing outputs/checkpoints/unet_best.pt")
    if not ckpt_dl.exists():
        raise RuntimeError("Missing outputs/checkpoints/deeplabv3_best.pt")

    # Use a single image_size for prediction set to keep outputs aligned.
    # Choose 256 to match today's deeplab training.
    ds_val = KvasirSegDataset(pairs_csv=pairs_csv, split="val", image_size=256, seed=42, train_frac=0.85)

    unet = load_unet(ckpt_unet, device)
    dl = load_deeplab(ckpt_dl, device)

    out_dir = Path("outputs/day4/preds")
    out_dir.mkdir(parents=True, exist_ok=True)

    idxs = list(range(len(ds_val)))
    random.seed(42)
    random.shuffle(idxs)
    idxs = idxs[:10]

    for j, i in enumerate(idxs):
        img_t, mask_t = ds_val[i]
        gt = (mask_t.squeeze(0).numpy() > 0.5).astype(np.uint8)
        img = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)

        gt_over = overlay_mask_rgb(img, gt, alpha=0.45)
        unet_over = predict_overlay(unet, img_t)
        dl_over = predict_overlay(dl, img_t)

        cv2.imwrite(str(out_dir / f"{j:02d}_gt.png"), cv2.cvtColor(gt_over, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{j:02d}_unet.png"), cv2.cvtColor(unet_over, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out_dir / f"{j:02d}_deeplab.png"), cv2.cvtColor(dl_over, cv2.COLOR_RGB2BGR))

    print("Saved predictions to:", out_dir)


if __name__ == "__main__":
    main()