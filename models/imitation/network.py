"""The shallow, wide SHIONN behavior-cloning network."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .dataset import BINARY_ACTION_COLUMNS

ARCHITECTURE_VERSION = "shionn_imitation_v3"


class LegacyImitationPolicy(nn.Module):
    """Exact v1/v2 network retained for loading existing checkpoints."""
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
        self.binary_heads = nn.ModuleDict(
            {name: nn.Linear(512, 2) for name in BINARY_ACTION_COLUMNS}
        )
        self.mouse_head = nn.Linear(512, 4)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        features = self.trunk(frames)
        output = {name: head(features) for name, head in self.binary_heads.items()}
        mouse = self.mouse_head(features)
        output["mouse_mean"] = mouse[:, :2]
        output["mouse_log_std"] = mouse[:, 2:].clamp(-8.0, 4.0)
        return output


class ImitationPolicy(nn.Module):
    """Four RGB frames in, independent binary controls and Gaussian mouse out.

    Group normalization and SiLU keep image-dependent activations alive through
    the trunk. The earlier plain-ReLU stack collapsed to a constant feature
    vector after several layers on the pilot dataset.
    """
    def __init__(self) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 64), nn.SiLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128), nn.SiLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 256), nn.SiLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 256), nn.SiLU(inplace=True),
            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 384), nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
            nn.Linear(384 * 4 * 4, 1024), nn.LayerNorm(1024), nn.SiLU(inplace=True),
            nn.Linear(1024, 512), nn.LayerNorm(512), nn.SiLU(inplace=True),
        )
        self.binary_heads = nn.ModuleDict({name: nn.Linear(512, 2) for name in BINARY_ACTION_COLUMNS})
        self.mouse_head = nn.Linear(512, 4)  # dx/dy mean, followed by dx/dy log standard deviation

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        # Cached inputs arrive in [0, 1]. Centering prevents positive image
        # brightness from overwhelming early convolutional biases.
        features = self.trunk((frames - 0.5) / 0.25)
        output = {name: head(features) for name, head in self.binary_heads.items()}
        mouse = self.mouse_head(features)
        output["mouse_mean"] = mouse[:, :2]
        output["mouse_log_std"] = mouse[:, 2:].clamp(-8.0, 4.0)
        return output
