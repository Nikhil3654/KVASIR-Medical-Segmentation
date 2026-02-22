from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.unet import UNet
from src.data.kvasir_dataset import KvasirSegDataset
from src.eval.metrics import dice_iou_from_logits


@dataclass
class TrainConfig:
    pairs_csv: Path = Path("data/processed/pairs.csv")
    image_size: int = 352
    batch_size: int = 4
    lr: float = 1e-3
    epochs: int = 5
    base: int = 32
    seed: int = 42
    train_frac: float = 0.85
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: Path = Path("outputs")
    num_workers: int = 0  # Windows: keep 0 to avoid multiprocessing issues


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    eps = 1e-7
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    dice_loss = 1 - dice.mean()
    return bce + dice_loss


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> dict:
    model.eval()
    dices = []
    ious = []
    losses = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = bce_dice_loss(logits, y)
        d, i = dice_iou_from_logits(logits, y)
        losses.append(loss.item())
        dices.append(d)
        ious.append(i)

    return {
        "val_loss": float(sum(losses) / max(1, len(losses))),
        "val_dice": float(sum(dices) / max(1, len(dices))),
        "val_iou": float(sum(ious) / max(1, len(ious))),
    }


def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)

    if not cfg.pairs_csv.exists():
        raise RuntimeError("pairs.csv missing. Run python -m scripts.build_pairs first.")

    train_ds = KvasirSegDataset(
        pairs_csv=cfg.pairs_csv,
        split="train",
        image_size=cfg.image_size,
        seed=cfg.seed,
        train_frac=cfg.train_frac,
    )
    val_ds = KvasirSegDataset(
        pairs_csv=cfg.pairs_csv,
        split="val",
        image_size=cfg.image_size,
        seed=cfg.seed,
        train_frac=cfg.train_frac,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = UNet(in_channels=3, out_channels=1, base=cfg.base).to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    ckpt_dir = cfg.out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best = {"val_dice": -1.0}
    history = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for x, y in train_loader:
            x = x.to(cfg.device)
            y = y.to(cfg.device)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = bce_dice_loss(logits, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        train_loss = float(sum(losses) / max(1, len(losses)))
        metrics = evaluate(model, val_loader, cfg.device)
        metrics["epoch"] = epoch
        metrics["train_loss"] = train_loss
        metrics["seconds"] = float(time.time() - t0)
        history.append(metrics)

        print(
            f"Epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={metrics['val_loss']:.4f} "
            f"val_dice={metrics['val_dice']:.4f} val_iou={metrics['val_iou']:.4f}"
        )

        if metrics["val_dice"] > best["val_dice"]:
            best = metrics
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "cfg": cfg.__dict__,
                    "best": best,
                },
                ckpt_dir / "unet_best.pt",
            )

    # Save training summary
    run_dir = cfg.out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_dict = dict(cfg.__dict__)
    for k, v in list(cfg_dict.items()):
        if isinstance(v, Path):
            cfg_dict[k] = str(v)

    summary = {"best": best, "history": history, "cfg": cfg_dict}

    with (run_dir / "unet_day3_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

    print("Saved best checkpoint to:", ckpt_dir / "unet_best.pt")
    print("Saved summary to:", run_dir / "unet_day3_summary.json")
    return summary