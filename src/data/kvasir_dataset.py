from __future__ import annotations

from pathlib import Path
import csv

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _read_pairs_csv(pairs_csv: Path) -> list[tuple[Path, Path]]:
    rows: list[tuple[Path, Path]] = []
    with pairs_csv.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((Path(row["image_path"]), Path(row["mask_path"])))
    return rows


class KvasirSegDataset(Dataset):
    def __init__(
        self,
        pairs_csv: Path,
        split: str,
        image_size: int = 352,
        seed: int = 42,
        train_frac: float = 0.85,
    ):
        all_pairs = _read_pairs_csv(pairs_csv)
        if len(all_pairs) == 0:
            raise ValueError("pairs_csv has no rows.")

        rng = np.random.default_rng(seed)
        idx = np.arange(len(all_pairs))
        rng.shuffle(idx)

        cut = int(len(all_pairs) * train_frac)
        if split == "train":
            chosen = idx[:cut]
        elif split == "val":
            chosen = idx[cut:]
        else:
            raise ValueError("split must be 'train' or 'val'")

        self.pairs = [all_pairs[i] for i in chosen]
        self.image_size = image_size
        self.split = split

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int):
        img_path, mask_path = self.pairs[i]

        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        s = self.image_size
        img_rgb = cv2.resize(img_rgb, (s, s), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (s, s), interpolation=cv2.INTER_NEAREST)

        img = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return img, mask_t