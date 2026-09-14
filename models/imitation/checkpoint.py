"""Portable, resumable checkpoint helpers for the behavior-cloning policy."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch

CHECKPOINT_FORMAT_VERSION = 1
REQUIRED_KEYS = {
    "model_state_dict", "optimizer_state_dict", "epoch", "global_step",
    "best_val_loss", "config",
}


def config_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.with_suffix(".json")


def save_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    config: dict[str, Any],
    step_in_epoch: int = 0,
) -> None:
    """Atomically save model, optimizer, progress and its matching JSON contract."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    temporary_checkpoint = checkpoint_path.with_name(f".{checkpoint_path.name}.tmp")
    torch.save(payload, temporary_checkpoint)
    os.replace(temporary_checkpoint, checkpoint_path)
    metadata = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "config": config,
    }
    json_path = config_path(checkpoint_path)
    temporary_json = json_path.with_name(f".{json_path.name}.tmp")
    temporary_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_json, json_path)


def load_checkpoint(
    checkpoint_path: Path,
    *,
    model: torch.nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """Load a checkpoint and its JSON contract, optionally restoring training state."""
    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    missing = REQUIRED_KEYS.difference(payload)
    if missing:
        raise ValueError(f"{checkpoint_path} is not a resumable SHIONN checkpoint; missing {sorted(missing)}")
    json_path = config_path(checkpoint_path)
    if not json_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint configuration: {json_path}")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if metadata.get("config") != payload["config"]:
        raise ValueError(f"Checkpoint and JSON configuration differ for {checkpoint_path}")
    if model is not None:
        model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
