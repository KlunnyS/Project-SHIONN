"""The shallow, wide v1 SHIONN behavior-cloning network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .dataset import BINARY_ACTION_COLUMNS


class ImitationPolicy(nn.Module):
    """Four RGB frames in, independent binary controls and Gaussian mouse out."""
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(384 * 4 * 4, 1024), nn.ReLU(inplace=True),
            nn.Linear(1024, 512), nn.ReLU(inplace=True),
        )
        self.binary_heads = nn.ModuleDict({name: nn.Linear(512, 2) for name in BINARY_ACTION_COLUMNS})
        self.mouse_head = nn.Linear(512, 4)  # dx/dy mean, followed by dx/dy log standard deviation

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        features = self.trunk(frames)
        output = {name: head(features) for name, head in self.binary_heads.items()}
        mouse = self.mouse_head(features)
        output["mouse_mean"] = mouse[:, :2]
        output["mouse_log_std"] = mouse[:, 2:].clamp(-8.0, 4.0)
        return output
