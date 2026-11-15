"""Opt-in visual-goal imitation integration; no automatic training execution."""

from __future__ import annotations

import copy
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .recurrent import RECURRENT_ARCHITECTURE_VERSION
from .visual_goal import (
    VISUAL_GOAL_ARCHITECTURE_VERSION,
    VISUAL_GOAL_CONFIG,
    VisualGoalRecurrentPolicy,
)
from .visual_targets import sha256


def warm_start_visual_policy(
    source_path: Path, *, device="cpu", embedding_size=128, adapter_size=128
):
    """Strict source architecture/preprocessing load, no optimizer updates."""
    from .mouse_bins import MouseBins
    from .preprocess import PREPROCESSING_CONFIG

    checkpoint = load_checkpoint(source_path, device=torch.device(device))
    config = checkpoint["config"]
    if (
        config.get("architecture") != RECURRENT_ARCHITECTURE_VERSION
        or config.get("preprocessing") != PREPROCESSING_CONFIG
    ):
        raise ValueError("Visual pilot requires a compatible recurrent source")
    memory = config["memory"]
    if memory.get("type") != "gru" or not isinstance(memory.get("feature_gate"), bool):
        raise ValueError("Invalid source memory configuration")
    bins = MouseBins(config["target_processing"]["mouse_bins"])
    model = VisualGoalRecurrentPolicy(
        *bins.class_counts(),
        memory["hidden_size"],
        memory["feature_gate"],
        embedding_size,
        adapter_size,
    ).to(device)
    model.initialize_from_recurrent(checkpoint["model_state_dict"])
    target_config = {
        key: copy.deepcopy(config[key])
        for key in (
            "architecture",
            "preprocessing",
            "target_processing",
            "memory",
            "action_columns",
        )
        if key in config
    }
    target_config["architecture"] = VISUAL_GOAL_ARCHITECTURE_VERSION
    target_config["visual_goal"] = {
        "preprocessing": copy.deepcopy(VISUAL_GOAL_CONFIG),
        "embedding_size": embedding_size,
        "adapter_size": adapter_size,
        "source_checkpoint_sha256": sha256(source_path),
    }
    # Source optimizer/training history must not masquerade as this new run.
    target_config["training"] = {"status": "prepared_not_trained", "freeze_base": True}
    return model, target_config


def visual_goal_batch_loss(
    model,
    batch,
    *,
    burn_in,
    binary_class_weights,
    mouse_bins,
    device="cpu",
    label_smoothing=0.0,
    goal_mode="selected",
):
    """Same BC losses; caller owns updates/checkpoints/budgets, never run here.

    Burn-in is no-grad, hidden is detached, goal encoding is differentiable for
    the supervised segment, and episode/chunk state is always initialized fresh.
    """
    from .train_bc import behavior_cloning_loss

    device = torch.device(device)
    frames = batch["frames"].to(device=device, dtype=torch.float32) / 255.0
    crops = batch["goal_crop"].to(device=device, dtype=torch.float32) / 255.0
    if goal_mode not in {"selected", "constant"}:
        raise ValueError("Unknown visual-goal control mode")
    if goal_mode == "constant":
        crops = torch.zeros_like(crops)
    valid = batch["valid"].to(device=device, dtype=torch.bool)
    targets = batch["targets"].to(device)
    if binary_class_weights is not None:
        binary_class_weights = binary_class_weights.to(device)
    mask = batch["loss_mask"].to(device=device, dtype=torch.bool) & valid
    if type(burn_in) is not int or not 0 <= burn_in < frames.shape[1]:
        raise ValueError("Invalid burn-in length")
    if mask.shape != valid.shape or bool(mask[:, :burn_in].any()):
        raise ValueError("Burn-in must not carry supervised labels")
    hidden = None
    embedding = model.encode_goal(crops)
    if burn_in:
        with torch.no_grad():
            _, hidden = model(
                frames[:, :burn_in],
                valid=valid[:, :burn_in],
                goal_embedding=embedding.detach(),
            )
        hidden = hidden.detach()
    logits, _ = model(
        frames[:, burn_in:],
        hidden=hidden,
        valid=valid[:, burn_in:],
        goal_embedding=embedding,
    )
    supervise = mask[:, burn_in:]
    if not bool(supervise.any()):
        raise ValueError("No supervised visual-goal frames")
    loss, parts = behavior_cloning_loss(
        {k: v[supervise] for k, v in logits.items()},
        targets[:, burn_in:][supervise],
        binary_class_weights,
        None,
        mouse_bins,
        label_smoothing,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite visual goal loss")
    return loss, parts, int(supervise.sum())


def run_visual_goal_epoch(
    model,
    loader,
    *,
    binary_class_weights,
    mouse_bins,
    optimizer=None,
    device="cpu",
    max_batches=32,
    goal_mode="selected",
    max_grad_norm=1.0,
):
    """Bounded opt-in epoch using the dataset's explicit membership contract.

    No command invokes this automatically. Supplying an optimizer constitutes
    training; only an approved TRAIN VisualGoalSequenceDataset may do so.
    """
    from .visual_goal_dataset import VisualGoalSequenceDataset

    if type(max_batches) is not int or not 1 <= max_batches <= 200:
        raise ValueError("Visual epoch must be bounded to 1..200 batches")
    if not isinstance(loader.dataset, VisualGoalSequenceDataset):
        raise ValueError("Use the provenance-validated visual sequence dataset")
    training = optimizer is not None
    if training and loader.dataset.role != "train":
        raise ValueError("Cannot optimize using non-TRAIN data")
    prior_mode = model.training
    model.train(training)
    totals = {}
    count = 0
    steps = 0
    try:
        for index, batch in enumerate(loader):
            if index >= max_batches:
                break
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.set_grad_enabled(training):
                loss, parts, n = visual_goal_batch_loss(
                    model,
                    batch,
                    burn_in=loader.dataset.burn_in,
                    binary_class_weights=binary_class_weights,
                    mouse_bins=mouse_bins,
                    device=device,
                    goal_mode=goal_mode,
                )
            if training:
                loss.backward()
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_grad_norm, error_if_nonfinite=True
                    )
                optimizer.step()
                steps += 1
            for key, value in {"total": loss, **parts}.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * n
            count += n
        if not count:
            raise ValueError("Empty visual goal epoch")
        return {
            "losses": {k: v / count for k, v in totals.items()},
            "frames": count,
            "optimizer_steps": steps,
        }
    finally:
        model.train(prior_mode)
