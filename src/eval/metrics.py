from __future__ import annotations

import torch


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5) -> tuple[float, float]:
    """
    logits: (B,1,H,W)
    targets: (B,1,H,W) float 0/1
    """
    probs = torch.sigmoid(logits)
    preds = (probs > thresh).float()

    preds = preds.view(preds.size(0), -1)
    targets = targets.view(targets.size(0), -1)

    eps = 1e-7
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)

    dice = ((2 * inter + eps) / (union + eps)).mean().item()

    # IoU
    denom = (preds + targets - preds * targets).sum(dim=1)
    iou = ((inter + eps) / (denom + eps)).mean().item()

    return dice, iou