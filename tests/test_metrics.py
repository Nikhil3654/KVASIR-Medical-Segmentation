import torch
from src.eval.metrics import dice_iou_from_logits


def test_metrics_perfect_prediction():
    targets = torch.ones((2, 1, 4, 4))
    logits = torch.full((2, 1, 4, 4), 10.0)
    dice, iou = dice_iou_from_logits(logits, targets)
    assert dice > 0.99
    assert iou > 0.99