from __future__ import annotations

from pathlib import Path
import csv

import cv2
import numpy as np
import pandas as pd


def make_grid(img_paths: list[Path], cols: int = 3) -> np.ndarray:
    imgs = [cv2.imread(str(p)) for p in img_paths]
    imgs = [cv2.resize(im, (384, 384)) for im in imgs]
    rows = int(np.ceil(len(imgs) / cols))
    blank = np.zeros_like(imgs[0])
    while len(imgs) < rows * cols:
        imgs.append(blank)

    grid_rows = []
    for r in range(rows):
        row = cv2.hconcat(imgs[r * cols : (r + 1) * cols])
        grid_rows.append(row)
    return cv2.vconcat(grid_rows)


def main() -> None:
    compare_csv = Path("outputs/day4/day4_compare.csv")
    preds_dir = Path("outputs/day4/preds")
    out_dir = Path("outputs/day5")
    out_dir.mkdir(parents=True, exist_ok=True)

    # save a markdown table for README
    df = pd.read_csv(compare_csv)
    md_path = out_dir / "results_table.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(df.to_markdown(index=False))

    # make 3 grids: gt, unet, deeplab (first 6 samples)
    gt = sorted(preds_dir.glob("*_gt.png"))[:6]
    unet = sorted(preds_dir.glob("*_unet.png"))[:6]
    dl = sorted(preds_dir.glob("*_deeplab.png"))[:6]

    if len(gt) and len(unet) and len(dl):
        grid_gt = make_grid(gt, cols=3)
        grid_unet = make_grid(unet, cols=3)
        grid_dl = make_grid(dl, cols=3)

        cv2.imwrite(str(out_dir / "grid_gt.png"), grid_gt)
        cv2.imwrite(str(out_dir / "grid_unet.png"), grid_unet)
        cv2.imwrite(str(out_dir / "grid_deeplab.png"), grid_dl)

    print("Wrote:", md_path)
    print("Wrote grids to:", out_dir)


if __name__ == "__main__":
    main()