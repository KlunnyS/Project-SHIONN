"""Train the Stage 1–2 factored behavior-cloning policy."""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import BINARY_ACTION_COLUMNS, BehaviorCloningDataset, discover_cached_episodes, split_episode_manifests
from .network import ARCHITECTURE_VERSION, ImitationPolicy
from .preprocess import PREPROCESSING_CONFIG


def format_duration(seconds: float) -> str:
    """Format elapsed wall time without hiding sub-minute training runs."""
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    if hours:
        return f"{hours}h {minutes:02d}m {remaining:04.1f}s"
    if minutes:
        return f"{minutes}m {remaining:04.1f}s"
    return f"{remaining:.1f}s"


def make_binary_class_weights(
    positive_counts, sample_count: int, max_weight: float = 10.0,
    min_positive_examples: int = 20,
) -> torch.Tensor:
    """Balance usable heads without amplifying a handful of accidental labels."""
    weights = torch.ones((len(BINARY_ACTION_COLUMNS), 2), dtype=torch.float32)
    for index, positive in enumerate(positive_counts):
        negative = sample_count - positive
        if positive < min_positive_examples or negative < min_positive_examples:
            continue
        weights[index, 0] = min(max_weight, sample_count / (2.0 * negative))
        weights[index, 1] = min(max_weight, sample_count / (2.0 * positive))
    return weights


def behavior_cloning_loss(
    prediction: dict[str, torch.Tensor],
    targets: torch.Tensor,
    binary_class_weights: torch.Tensor | None = None,
    mouse_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum balanced binary losses and scaled mouse Gaussian NLL."""
    losses: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=targets.device)
    for index, name in enumerate(BINARY_ACTION_COLUMNS):
        weight = None if binary_class_weights is None else binary_class_weights[index]
        loss = nn.functional.cross_entropy(
            prediction[name], targets[:, index].long(), weight=weight
        )
        losses[name] = loss
        total = total + loss
    if mouse_scale is None:
        mouse_scale = torch.ones(2, device=targets.device)
    mouse_target = targets[:, 8:10] / mouse_scale
    log_std = prediction["mouse_log_std"]
    inverse_variance = torch.exp(-2.0 * log_std)
    mouse_nll = 0.5 * (((mouse_target - prediction["mouse_mean"]) ** 2) * inverse_variance + 2.0 * log_std + math.log(2.0 * math.pi)).sum(dim=1).mean()
    losses["mouse"] = mouse_nll
    return total + mouse_nll, losses


def run_epoch(
    model, loader, optimizer, scaler, device, train: bool,
    binary_class_weights: torch.Tensor, mouse_scale: torch.Tensor,
    global_step: int = 0, on_step=None, skip_batches: int = 0,
    log_every: int = 0,
) -> tuple[dict[str, float], int]:
    model.train(train)
    totals: dict[str, float] = {name: 0.0 for name in (*BINARY_ACTION_COLUMNS, "mouse", "total")}
    batches = 0
    for batch_index, (frames, targets) in enumerate(loader):
        if batch_index < skip_batches:
            continue
        frames = frames.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
        targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(frames)
            loss, parts = behavior_cloning_loss(
                prediction, targets, binary_class_weights, mouse_scale
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if on_step is not None:
                on_step(global_step, batch_index + 1)
        totals["total"] += loss.detach().item()
        for name, value in parts.items():
            totals[name] += value.detach().item()
        batches += 1
        if train and log_every and batches % log_every == 0:
            print(
                f"  batch {batch_index + 1}/{len(loader)} "
                f"loss={totals['total'] / batches:.4f}"
            )
    return {name: value / max(1, batches) for name, value in totals.items()}, global_step


def resolve_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return device


def checkpoint_config(args, binary_class_weights, mouse_scale) -> dict:
    """The inference-critical contract persisted inside and beside every checkpoint."""
    return {
        "architecture": ARCHITECTURE_VERSION,
        "preprocessing": PREPROCESSING_CONFIG,
        "action_columns": list(BINARY_ACTION_COLUMNS) + ["mouse_dx", "mouse_dy"],
        "target_processing": {
            "leading_idle": "exclude_before_first_nonzero_action",
            "binary_class_weights": binary_class_weights.tolist(),
            "mouse_scale": mouse_scale.tolist(),
            "mouse_loss": "gaussian_nll_on_standardized_deltas",
        },
        "training": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
        },
    }


def make_training_loader(dataset, args, pin_memory: bool, epoch: int) -> DataLoader:
    """A deterministic per-epoch order lets periodic checkpoints resume mid-epoch."""
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=args.workers, pin_memory=pin_memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SHIONN's behavior-cloning policy from cached frames.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/datasets/cached_frames"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Tier-1 default for an 8 GB GPU")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for portable Windows/headless operation; raise on Linux after validation")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/imitation/checkpoints_v3"))
    parser.add_argument("--checkpoint-every", type=int, default=1_000, help="Save a resumable checkpoint every N optimizer steps; 0 disables periodic saves")
    parser.add_argument("--log-every", type=int, default=100, help="Print running training loss every N batches; 0 disables batch logs")
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume, including optimizer state")
    args = parser.parse_args()
    run_started_at = time.perf_counter()
    torch.manual_seed(args.seed)
    manifests = discover_cached_episodes(args.cache_dir)
    train_manifests, validation_manifests = split_episode_manifests(manifests, args.validation_fraction, args.seed)
    print("train episodes:", ", ".join(item.name for item in train_manifests))
    print("validation episodes:", ", ".join(item.name for item in validation_manifests))
    if args.workers < 0:
        raise ValueError("--workers must be zero or greater")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be zero or greater")
    if args.log_every < 0:
        raise ValueError("--log-every must be zero or greater")
    device = resolve_device(args.device)
    print(f"device: {device}")
    pin_memory = device.type == "cuda"
    train_dataset = BehaviorCloningDataset(train_manifests)
    validation_dataset = BehaviorCloningDataset(validation_manifests)
    positive_counts, mouse_scale_values = train_dataset.action_statistics()
    sample_count = len(train_dataset)
    binary_class_weights = make_binary_class_weights(positive_counts, sample_count)
    print(
        f"excluded leading idle frames: train={train_dataset.leading_idle_frames} "
        f"validation={validation_dataset.leading_idle_frames}"
    )
    print(
        "binary positive rates: "
        + " ".join(
            f"{name}={100.0 * count / sample_count:.2f}%"
            for name, count in zip(BINARY_ACTION_COLUMNS, positive_counts)
        )
    )
    print(
        f"mouse scales: dx={mouse_scale_values[0]:.3f} dy={mouse_scale_values[1]:.3f}"
    )
    binary_class_weights = binary_class_weights.to(device)
    mouse_scale = torch.as_tensor(mouse_scale_values, device=device)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    model = ImitationPolicy().to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = checkpoint_config(args, binary_class_weights.cpu(), mouse_scale.cpu())
    best_validation = float("inf")
    global_step = 0
    start_epoch = 1
    resume_step_in_epoch = 0
    best_epoch = None
    if args.resume:
        restored = load_checkpoint(args.resume, model=model, optimizer=optimizer, device=device)
        compatibility_keys = ("architecture", "preprocessing", "action_columns", "target_processing")
        if any(restored["config"].get(key) != config.get(key) for key in compatibility_keys):
            raise ValueError("Resume checkpoint model or target processing differs from this run")
        best_validation = float(restored["best_val_loss"])
        global_step = int(restored["global_step"])
        resume_step_in_epoch = int(restored.get("step_in_epoch", 0))
        start_epoch = int(restored["epoch"]) if resume_step_in_epoch else int(restored["epoch"]) + 1
        print(f"resumed {args.resume}: epoch={restored['epoch']} step_in_epoch={resume_step_in_epoch} global_step={global_step} best_val={best_validation:.4f}")
    initial_global_step = global_step
    completed_epochs = 0
    epoch_durations: list[float] = []
    final_train_metrics = None
    final_validation_metrics = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_loader = make_training_loader(train_dataset, args, pin_memory, epoch)
        skip_batches = resume_step_in_epoch if epoch == start_epoch else 0
        def save_periodic(step: int, step_in_epoch: int) -> None:
            if args.checkpoint_every and step % args.checkpoint_every == 0:
                path = args.checkpoint_dir / f"step_{step:09d}.pt"
                save_checkpoint(path, model=model, optimizer=optimizer, epoch=epoch, step_in_epoch=step_in_epoch, global_step=step, best_val_loss=best_validation, config=config)
                save_checkpoint(args.checkpoint_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, step_in_epoch=step_in_epoch, global_step=step, best_val_loss=best_validation, config=config)
        train_metrics, global_step = run_epoch(
            model, train_loader, optimizer, scaler, device, train=True,
            binary_class_weights=binary_class_weights, mouse_scale=mouse_scale,
            global_step=global_step, on_step=save_periodic, skip_batches=skip_batches,
            log_every=args.log_every,
        )
        resume_step_in_epoch = 0
        with torch.no_grad():
            validation_metrics, _ = run_epoch(
                model, validation_loader, optimizer, scaler, device, train=False,
                binary_class_weights=binary_class_weights, mouse_scale=mouse_scale,
                global_step=global_step,
            )
        print(f"epoch {epoch:03d} train={train_metrics['total']:.4f} validation={validation_metrics['total']:.4f} " + " ".join(f"{key}={validation_metrics[key]:.3f}" for key in (*BINARY_ACTION_COLUMNS, "mouse")))
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            save_checkpoint(args.checkpoint_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, best_val_loss=best_validation, config=config)
        save_checkpoint(args.checkpoint_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, best_val_loss=best_validation, config=config)
        completed_epochs += 1
        epoch_durations.append(time.perf_counter() - epoch_started_at)
        final_train_metrics = train_metrics
        final_validation_metrics = validation_metrics

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - run_started_at
    print("\n=== Training complete ===")
    print(f"elapsed wall time: {format_duration(elapsed)}")
    print(f"device: {device}")
    print(f"architecture: {ARCHITECTURE_VERSION}")
    print(f"model parameters: {parameter_count:,}")
    print(
        f"dataset: {len(train_manifests)} train episodes / {len(train_dataset):,} usable samples; "
        f"{len(validation_manifests)} validation episodes / {len(validation_dataset):,} usable samples"
    )
    print(
        f"epochs completed this run: {completed_epochs}; "
        f"optimizer steps this run: {global_step - initial_global_step:,}; "
        f"total optimizer steps: {global_step:,}"
    )
    if epoch_durations:
        print(
            f"average epoch time: {format_duration(sum(epoch_durations) / len(epoch_durations))}"
        )
    if best_epoch is None:
        print(f"best validation loss: {best_validation:.4f} (from resumed checkpoint)")
    else:
        print(f"best validation loss: {best_validation:.4f} (epoch {best_epoch})")
    if final_train_metrics is not None and final_validation_metrics is not None:
        print(f"final train loss: {final_train_metrics['total']:.4f}")
        print(f"final validation loss: {final_validation_metrics['total']:.4f}")
        print(
            "final validation components: "
            + " ".join(
                f"{name}={final_validation_metrics[name]:.4f}"
                for name in (*BINARY_ACTION_COLUMNS, "mouse")
            )
        )
    print(f"best checkpoint: {(args.checkpoint_dir / 'best.pt').resolve()}")
    print(f"latest checkpoint: {(args.checkpoint_dir / 'last.pt').resolve()}")
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"peak CUDA memory allocated: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
