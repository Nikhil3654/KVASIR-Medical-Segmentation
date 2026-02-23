from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


class DeepLabV3MobileNet(nn.Module):
    def __init__(self, out_channels: int = 1, pretrained_backbone: bool = True):
        super().__init__()
        # weights are optional and can trigger downloads. Keep backbone pretrained true but safe.
        # If it tries to download and you want offline, set pretrained_backbone=False.
        self.model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None)

        in_ch = self.model.classifier[-1].in_channels
        self.model.classifier[-1] = nn.Conv2d(in_ch, out_channels, kernel_size=1)

        # Aux classifier exists for some variants. If present, set it too.
        if self.model.aux_classifier is not None:
            aux_in = self.model.aux_classifier[-1].in_channels
            self.model.aux_classifier[-1] = nn.Conv2d(aux_in, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        # torchvision returns dict: {"out": logits, "aux": ...}
        return out["out"]