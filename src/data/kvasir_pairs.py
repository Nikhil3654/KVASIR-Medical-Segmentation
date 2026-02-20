from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Pair:
    image_path: Path
    mask_path: Path


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def find_images_and_masks(root: Path) -> tuple[list[Path], list[Path]]:
    images: list[Path] = []
    masks: list[Path] = []

    for p in root.rglob("*"):
        if not _is_image(p):
            continue
        parent = str(p.parent).lower()
        name = p.stem.lower()
        if "mask" in name or "masks" in parent:
            masks.append(p)
        else:
            images.append(p)

    return images, masks


def build_pairs(root: Path) -> list[Pair]:
    images, masks = find_images_and_masks(root)
    mask_by_stem = {m.stem.lower(): m for m in masks}

    pairs: list[Pair] = []
    missing = 0
    for img in images:
        m = mask_by_stem.get(img.stem.lower())
        if m is None:
            missing += 1
            continue
        pairs.append(Pair(img, m))

    if len(images) > 0 and (missing / max(1, len(images))) > 0.10:
        raise RuntimeError(
            f"Pairing failed too often. Images={len(images)} paired={len(pairs)} missing={missing}. "
            "Check dataset folder structure under data/raw/kvasir_seg."
        )

    return pairs