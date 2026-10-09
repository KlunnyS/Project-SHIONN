"""Versioned SHIONN behavior-cloning networks.

Old classes stay here because a checkpoint is only useful when its exact
architecture remains loadable.  New training uses the residual v5 policy;
v1-v4 checkpoints continue to use their original implementations.
"""

from __future__ import annotations

# Installed dependency: PyTorch layers and tensor types.
import torch
from torch import Tensor, nn

# Sibling module: action names defined by models.imitation.dataset.
from .dataset import BINARY_ACTION_COLUMNS

V3_ARCHITECTURE_VERSION = "shionn_imitation_v3"
V4_BINNED_ARCHITECTURE_VERSION = "shionn_imitation_v4_binned"
ARCHITECTURE_VERSION = "shionn_imitation_v5_residual"
BINNED_ARCHITECTURE_VERSION = "shionn_imitation_v5_residual_binned"


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


class V3ImitationPolicy(nn.Module):
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


class V4BinnedImitationPolicy(nn.Module):
    """The v3 visual trunk with independent discrete dx and dy heads."""

    def __init__(self, dx_classes: int, dy_classes: int) -> None:
        super().__init__()
        if dx_classes < 3 or dy_classes < 3:
            raise ValueError("Each mouse axis needs a negative, zero, and positive bin")
        # Keep the v3 trunk and binary-head initialization identical for a
        # controlled comparison with an otherwise matching Gaussian run.
        base = V3ImitationPolicy()
        self.trunk = base.trunk
        self.binary_heads = base.binary_heads
        self.mouse_dx_head = nn.Linear(512, dx_classes)
        self.mouse_dy_head = nn.Linear(512, dy_classes)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        features = self.trunk((frames - 0.5) / 0.25)
        output = {name: head(features) for name, head in self.binary_heads.items()}
        output["mouse_dx_logits"] = self.mouse_dx_head(features)
        output["mouse_dy_logits"] = self.mouse_dy_head(features)
        return output


class SqueezeExcitation(nn.Module):
    """Cheap channel attention for visually important features."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(16, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: Tensor) -> Tensor:
        return features * self.gate(self.pool(features))


class ResidualVisualBlock(nn.Module):
    """Pre-activation-free residual block suited to small training batches."""

    def __init__(self, channels: int, dropout: float = 0.05) -> None:
        super().__init__()
        groups = 8 if channels < 256 else 16
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            SqueezeExcitation(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, features: Tensor) -> Tensor:
        return self.activation(features + self.layers(features))


def _downsample(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = 8 if out_channels < 256 else 16
    return nn.Sequential(
        nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2,
            padding=1, bias=False,
        ),
        nn.GroupNorm(groups, out_channels),
        nn.SiLU(inplace=True),
    )


class ResidualVisualEncoder(nn.Module):
    """A spatial encoder used by the 13.2M-parameter v5 policies.

    The 3x5 output grid retains the input's 16:9 geometry.  This matters for
    steering: global average pooling alone would discard where an opening or
    wall edge appears in the view.
    """

    output_features = 512

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(8, 64),
            nn.SiLU(inplace=True),
            ResidualVisualBlock(64),
            _downsample(64, 128),
            ResidualVisualBlock(128),
            _downsample(128, 256),
            ResidualVisualBlock(256),
            ResidualVisualBlock(256),
            _downsample(256, 384),
            ResidualVisualBlock(384),
            nn.AdaptiveAvgPool2d((3, 5)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(384 * 3 * 5, 1024),
            nn.LayerNorm(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, self.output_features),
            nn.LayerNorm(self.output_features),
            nn.SiLU(inplace=True),
            nn.Dropout(0.15),
        )

    def forward(self, frames: Tensor) -> Tensor:
        normalized = (frames - 0.5) / 0.25
        return self.projection(self.features(normalized))


class ImitationPolicy(nn.Module):
    """Residual v5 encoder with binary controls and a Gaussian mouse head."""

    def __init__(self) -> None:
        super().__init__()
        self.trunk = ResidualVisualEncoder()
        self.binary_heads = nn.ModuleDict({
            name: nn.Linear(self.trunk.output_features, 2)
            for name in BINARY_ACTION_COLUMNS
        })
        self.mouse_head = nn.Linear(self.trunk.output_features, 4)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        features = self.trunk(frames)
        output = {name: head(features) for name, head in self.binary_heads.items()}
        mouse = self.mouse_head(features)
        output["mouse_mean"] = mouse[:, :2]
        output["mouse_log_std"] = mouse[:, 2:].clamp(-8.0, 4.0)
        return output


class BinnedImitationPolicy(nn.Module):
    """Residual v5 encoder with discrete, multimodal mouse actions."""

    def __init__(self, dx_classes: int, dy_classes: int) -> None:
        super().__init__()
        if dx_classes < 3 or dy_classes < 3:
            raise ValueError("Each mouse axis needs a negative, zero, and positive bin")
        self.trunk = ResidualVisualEncoder()
        self.binary_heads = nn.ModuleDict({
            name: nn.Linear(self.trunk.output_features, 2)
            for name in BINARY_ACTION_COLUMNS
        })
        self.mouse_dx_head = nn.Linear(self.trunk.output_features, dx_classes)
        self.mouse_dy_head = nn.Linear(self.trunk.output_features, dy_classes)

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        features = self.trunk(frames)
        output = {name: head(features) for name, head in self.binary_heads.items()}
        output["mouse_dx_logits"] = self.mouse_dx_head(features)
        output["mouse_dy_logits"] = self.mouse_dy_head(features)
        return output
