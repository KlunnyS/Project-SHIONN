"""Optional persistent crop-conditioned readout over the unchanged CNN/GRU."""

from __future__ import annotations

import torch
from torch import nn

from .dataset import BINARY_ACTION_COLUMNS
from .recurrent import RecurrentBinnedPolicy

VISUAL_GOAL_ARCHITECTURE_VERSION = "shionn_imitation_v5_gru_visual_goal_binned"
VISUAL_GOAL_CONFIG = {
    "schema": "shionn_visual_crop_goal_v1",
    "color_order": "RGB",
    "crop_size": [96, 96],
    "resize": "INTER_AREA",
    "normalization": "uint8_rgb_divide_255",
    "encoder": "shared_visual_trunk_repeat_rgb_four_times",
    "persistence": "until_explicit_goal_change_or_episode_reset",
}


class VisualGoalRecurrentPolicy(nn.Module):
    """Frames [B,T,12,H,W], persistent crop [B,3,96,96], state [1,B,H].

    Forward may alternatively accept an already encoded [B,E] target, for
    no-grad cached inference only. No engine coordinates, triggers or goal IDs
    enter the model. Goal-conditioned residual logits start at exactly zero.
    """

    def __init__(
        self,
        dx_classes,
        dy_classes,
        hidden_size=256,
        feature_gate=False,
        embedding_size=128,
        adapter_size=128,
    ):
        super().__init__()
        if (
            type(embedding_size) is not int
            or embedding_size < 1
            or type(adapter_size) is not int
            or adapter_size < 1
        ):
            raise ValueError("Visual goal dimensions must be positive integers")
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.base = RecurrentBinnedPolicy(
            dx_classes, dy_classes, hidden_size, feature_gate
        )
        self.goal_projection = nn.Sequential(
            nn.Linear(512, embedding_size), nn.LayerNorm(embedding_size), nn.Tanh()
        )
        self.adapter = nn.Sequential(
            nn.Linear(hidden_size + embedding_size, adapter_size), nn.Tanh()
        )
        self.goal_heads = nn.ModuleDict(
            {k: nn.Linear(adapter_size, 2) for k in BINARY_ACTION_COLUMNS}
        )
        self.goal_heads["mouse_dx_logits"] = nn.Linear(adapter_size, dx_classes)
        self.goal_heads["mouse_dy_logits"] = nn.Linear(adapter_size, dy_classes)
        for head in self.goal_heads.values():
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initialize_from_recurrent(self, source_state_dict, *, freeze_base=True):
        self.base.load_state_dict(source_state_dict, strict=True)
        self.base.requires_grad_(not freeze_base)

    def encode_goal(self, crop):
        if crop.ndim != 4 or crop.shape[1:] != (3, 96, 96):
            raise ValueError("Expected crop [B,3,96,96]")
        if (
            not crop.is_floating_point()
            or not torch.isfinite(crop).all()
            or torch.any((crop < 0) | (crop > 1))
        ):
            raise ValueError("Crops must be finite floating RGB in [0,1]")
        # Same visual filters as the scene, no duplicate large CNN parameters.
        return self.goal_projection(self.base.trunk(crop.repeat(1, 4, 1, 1)))

    def forward(
        self, frames, goal_crop=None, hidden=None, valid=None, *, goal_embedding=None
    ):
        if (goal_crop is None) == (goal_embedding is None):
            raise ValueError("Supply exactly one crop or persistent embedding")
        embedding = (
            self.encode_goal(goal_crop) if goal_embedding is None else goal_embedding
        )
        if (
            embedding.shape != (frames.shape[0], self.embedding_size)
            or not torch.isfinite(embedding).all()
        ):
            raise ValueError("Persistent embedding batch/shape/values invalid")
        output, hidden, features = self.base(
            frames, hidden, valid, return_features=True
        )
        goal = embedding[:, None].expand(-1, frames.shape[1], -1)
        residual = self.adapter(torch.cat((features, goal), dim=-1))
        return {
            k: value + self.goal_heads[k](residual) for k, value in output.items()
        }, hidden

    def step(self, frame, goal_embedding, hidden=None):
        output, hidden = self(
            frame[:, None], hidden=hidden, goal_embedding=goal_embedding
        )
        return {k: v[:, 0] for k, v in output.items()}, hidden
