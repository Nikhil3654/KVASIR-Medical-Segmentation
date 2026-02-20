from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from scripts._bootstrap import ROOT
from src.data.kvasir_dataset import KvasirSegDataset
from src.infer.overlay import overlay_mask_rgb
from pathlib import Path

import cv2
import numpy as np

from src.data.kvasir_dataset import KvasirSegDataset
from src.infer.overlay import overlay_mask_rgb


def main() -> None:
    pairs_csv = Path("data/processed/pairs.csv")
    if not pairs_csv.exists():
        raise RuntimeError("pairs.csv not found. Run scripts/build_pairs.py first.")

    ds = KvasirSegDataset(pairs_csv=pairs_csv, split="train", image_size=352)
    out_dir = Path("outputs/day2")
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(8, len(ds))):
        img_t, mask_t = ds[i]
        img = (img_t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        mask = (mask_t.squeeze(0).numpy() > 0.5).astype(np.uint8)
        over = overlay_mask_rgb(img, mask, alpha=0.45)
        out_bgr = cv2.cvtColor(over, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"sample_{i}.png"), out_bgr)

    print("Saved samples to:", out_dir)


if __name__ == "__main__":
    main()