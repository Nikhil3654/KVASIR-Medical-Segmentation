from __future__ import annotations

import csv
from pathlib import Path

from scripts._bootstrap import ROOT
from src.data.kvasir_pairs import build_pairs


DATA_DIR = ROOT / "data/raw/kvasir_seg"
OUT_DIR = ROOT / "data/processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

from pathlib import Path
import csv

from src.data.kvasir_pairs import build_pairs


DATA_DIR = Path("data/raw/kvasir_seg")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if not DATA_DIR.exists():
        raise RuntimeError("Dataset not found. Run scripts/kaggle_download.py first.")

    pairs = build_pairs(DATA_DIR)
    out_path = OUT_DIR / "pairs.csv"

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "mask_path"])
        for p in pairs:
            w.writerow([str(p.image_path).replace("\\", "/"), str(p.mask_path).replace("\\", "/")])

    print("Saved:", out_path)
    print("Pairs:", len(pairs))


if __name__ == "__main__":
    main()