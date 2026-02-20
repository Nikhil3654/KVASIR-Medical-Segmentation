from __future__ import annotations

from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.infer.overlay import overlay_mask_rgb


DATA_DIR = Path("data/raw/kvasir_seg")


def index_pairs(base: Path):
    # Try common structure: images/ and masks/
    images = []
    masks = []

    for p in base.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        parent = str(p.parent).lower()
        if "mask" in p.stem.lower() or "masks" in parent:
            masks.append(p)
        else:
            images.append(p)

    return images, masks


def match_mask(img_path: Path, masks: list[Path]) -> Path | None:
    stem = img_path.stem.lower()
    # exact stem match first
    for m in masks:
        if m.stem.lower() == stem:
            return m
    # substring fallback
    for m in masks:
        if stem in m.stem.lower() or m.stem.lower() in stem:
            return m
    return None


def main() -> None:
    if not DATA_DIR.exists():
        raise RuntimeError("Dataset not found. Run scripts/kaggle_download.py first.")

    imgs, masks = index_pairs(DATA_DIR)
    print("Found images:", len(imgs))
    print("Found masks:", len(masks))
    if len(imgs) == 0 or len(masks) == 0:
        raise RuntimeError("Could not locate image and mask files under data/raw/kvasir_seg.")

    img_path = random.choice(imgs)
    mask_path = match_mask(img_path, masks) or random.choice(masks)

    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Failed to read mask: {mask_path}")
    mask_bin = (mask > 127).astype(np.uint8)

    over = overlay_mask_rgb(img_rgb, mask_bin, alpha=0.45)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "day1_overlay.png"

    over_bgr = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), over_bgr)

    print("Sample image:", img_path)
    print("Sample mask:", mask_path)
    print("Image shape:", img_rgb.shape)
    print("Mask shape:", mask.shape)
    print("Saved overlay to:", out_path)


if __name__ == "__main__":
    main()