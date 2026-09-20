"""Train the Stage 1–2 factored behavior-cloning policy."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import BINARY_ACTION_COLUMNS, BehaviorCloningDataset, discover_cached_episodes, split_episode_manifests
from .mouse_bins import MouseBins
from .network import ARCHITECTURE_VERSION, BINNED_ARCHITECTURE_VERSION, BinnedImitationPolicy, ImitationPolicy
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


class EarlyStopping:
    """Stop after a configured number of complete epochs without a new best loss."""

    def __init__(self, patience: int, best_loss: float = float("inf")) -> None:
        if patience < 0:
            raise ValueError("early-stop patience must be zero or greater")
        self.patience = patience
        self.best_loss = best_loss
        self.unimproved_epochs = 0

    def update(self, validation_loss: float) -> tuple[bool, bool]:
        improved = validation_loss < self.best_loss
        if improved:
            self.best_loss = validation_loss
            self.unimproved_epochs = 0
        else:
            self.unimproved_epochs += 1
        return improved, self.patience > 0 and self.unimproved_epochs >= self.patience


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


def binary_class_weights_for_mode(
    positive_counts, sample_count: int, mode: str,
) -> torch.Tensor:
    if mode == "balanced":
        return make_binary_class_weights(positive_counts, sample_count)
    if mode == "none":
        return torch.ones((len(BINARY_ACTION_COLUMNS), 2), dtype=torch.float32)
    raise ValueError(f"Unknown binary class weighting mode: {mode}")


def with_jump_positive_weight(weights: torch.Tensor, ratio: float | None) -> torch.Tensor:
    """Optionally set jump-positive weight relative to the no-jump class."""
    if ratio is None:
        return weights
    if ratio <= 0:
        raise ValueError("--jump-positive-weight must be greater than zero")
    result = weights.clone()
    index = BINARY_ACTION_COLUMNS.index("jump")
    result[index, 1] = result[index, 0] * ratio
    return result


def behavior_cloning_loss(
    prediction: dict[str, torch.Tensor],
    targets: torch.Tensor,
    binary_class_weights: torch.Tensor | None = None,
    mouse_scale: torch.Tensor | None = None,
    mouse_bins: MouseBins | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum binary losses and the selected Gaussian or binned mouse loss."""
    losses: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=targets.device)
    for index, name in enumerate(BINARY_ACTION_COLUMNS):
        weight = None if binary_class_weights is None else binary_class_weights[index]
        loss = nn.functional.cross_entropy(
            prediction[name], targets[:, index].long(), weight=weight
        )
        losses[name] = loss
        total = total + loss
    if mouse_bins is not None:
        dx_target, dy_target = mouse_bins.encode(targets[:, 8:10])
        mouse_loss = (
            nn.functional.cross_entropy(prediction["mouse_dx_logits"], dx_target)
            + nn.functional.cross_entropy(prediction["mouse_dy_logits"], dy_target)
        )
    else:
        if mouse_scale is None:
            mouse_scale = torch.ones(2, device=targets.device)
        mouse_target = targets[:, 8:10] / mouse_scale
        log_std = prediction["mouse_log_std"]
        inverse_variance = torch.exp(-2.0 * log_std)
        mouse_loss = 0.5 * (((mouse_target - prediction["mouse_mean"]) ** 2) * inverse_variance + 2.0 * log_std + math.log(2.0 * math.pi)).sum(dim=1).mean()
    losses["mouse"] = mouse_loss
    return total + mouse_loss, losses


def run_epoch(
    model, loader, optimizer, scaler, device, train: bool,
    binary_class_weights: torch.Tensor, mouse_scale: torch.Tensor,
    global_step: int = 0, on_step=None, skip_batches: int = 0,
    log_every: int = 0, mouse_bins: MouseBins | None = None,
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
                prediction, targets, binary_class_weights, mouse_scale, mouse_bins
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


def checkpoint_config(
    args, binary_class_weights, mouse_scale, mouse_bins: MouseBins | None = None,
    mouse_bin_counts: dict[str, list[int]] | None = None,
) -> dict:
    """The inference-critical contract persisted inside and beside every checkpoint."""
    return {
        "architecture": BINNED_ARCHITECTURE_VERSION if mouse_bins is not None else ARCHITECTURE_VERSION,
        "preprocessing": PREPROCESSING_CONFIG,
        "action_columns": list(BINARY_ACTION_COLUMNS) + ["mouse_dx", "mouse_dy"],
        "target_processing": {
            "leading_idle": "exclude_before_first_nonzero_action",
            "binary_class_weights": binary_class_weights.tolist(),
            **({
                "mouse_bins": mouse_bins.config,
                "mouse_bin_counts": mouse_bin_counts,
                "mouse_loss": "cross_entropy_per_axis",
            }
               if mouse_bins is not None else {
                   "mouse_scale": mouse_scale.tolist(),
                   "mouse_loss": "gaussian_nll_on_standardized_deltas",
               }),
        },
        "training": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "validation_fraction": args.validation_fraction,
            "seed": args.seed,
            "sampling": args.sampling,
            "binary_class_weighting": args.binary_class_weighting,
            "jump_positive_weight": args.jump_positive_weight,
            "mouse_head": args.mouse_head,
            "maps": sorted(args.maps),
            "train_extra_cache_dirs": [str(path) for path in args.train_extra_cache_dirs],
            "start_window_frames": args.start_window_frames,
            "start_sampling_boost": args.start_sampling_boost,
            "correction_sampling_fraction": args.correction_sampling_fraction,
        },
    }


def select_maps(manifests, map_names: list[str]):
    """Keep only requested chambers and reject misspelled map names."""
    if not map_names:
        return manifests
    if len(map_names) != len(set(map_names)):
        raise ValueError("--map values must be unique")
    available = {manifest.map_name for manifest in manifests}
    missing = set(map_names) - available
    if missing:
        raise ValueError("No cached episodes for map(s): " + ", ".join(sorted(missing)))
    selected = set(map_names)
    return [manifest for manifest in manifests if manifest.map_name in selected]


def training_sampling_weights(
    dataset: BehaviorCloningDataset, sampling: str, *,
    start_window_frames: int = 8, start_sampling_boost: float = 1.0,
    correction_episode_names: set[str] | None = None,
    correction_sampling_fraction: float = 0.0,
) -> np.ndarray:
    """Set expected opening and correction exposure without changing epoch size."""
    if start_window_frames < 1 or start_sampling_boost < 1:
        raise ValueError("Start sampling needs a positive window and boost at least 1")
    if not 0 <= correction_sampling_fraction < 1:
        raise ValueError("Correction sampling fraction must be in [0, 1)")
    if sampling == "chamber-balanced":
        weights = dataset.chamber_sampling_weights()
    elif sampling == "uniform":
        weights = np.ones(len(dataset), dtype=np.float64)
    else:
        raise ValueError(f"Unknown training sampling mode: {sampling}")
    episode_indices = np.fromiter((episode for episode, _ in dataset.index),
                                  dtype=np.int32, count=len(dataset))
    frame_indices = np.fromiter((frame for _, frame in dataset.index),
                                dtype=np.int32, count=len(dataset))
    first_frames = np.asarray(dataset.first_action_frames, dtype=np.int32)
    opening = frame_indices - first_frames[episode_indices] < start_window_frames
    if start_sampling_boost != 1:
        weights[opening] *= start_sampling_boost
        if sampling == "chamber-balanced":
            # Preserve equal expected map exposure after boosting their openings.
            names = np.asarray(dataset.episode_maps, dtype=object)[episode_indices]
            for map_name in set(dataset.episode_maps):
                selected = names == map_name
                weights[selected] *= len(dataset) / len(set(dataset.episode_maps)) / weights[selected].sum()
    correction_names = correction_episode_names or set()
    correction = np.fromiter(
        (dataset.episode_names[episode] in correction_names for episode in episode_indices),
        dtype=bool, count=len(dataset),
    )
    if correction_sampling_fraction:
        if not correction.any() or correction.all():
            raise ValueError("Correction sampling needs both base and correction frames")
        base_mass, correction_mass = weights[~correction].sum(), weights[correction].sum()
        weights[correction] *= (
            correction_sampling_fraction / (1 - correction_sampling_fraction)
            * base_mass / correction_mass
        )
    weights *= len(dataset) / weights.sum()
    return weights


def make_training_loader(
    dataset, args, pin_memory: bool, epoch: int,
    correction_episode_names: set[str] | None = None,
) -> DataLoader:
    """Use deterministic weighted draws for each resumable epoch."""
    generator = torch.Generator()
    generator.manual_seed(args.seed + epoch)
    sampler = None
    boost = getattr(args, "start_sampling_boost", 1.0)
    correction_fraction = getattr(args, "correction_sampling_fraction", 0.0)
    if args.sampling == "chamber-balanced" or boost != 1 or correction_fraction:
        weights = training_sampling_weights(
            dataset, args.sampling,
            start_window_frames=getattr(args, "start_window_frames", 8),
            start_sampling_boost=boost,
            correction_episode_names=correction_episode_names,
            correction_sampling_fraction=correction_fraction,
        )
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights),
            num_samples=len(dataset), replacement=True, generator=generator,
        )
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, generator=generator,
        num_workers=args.workers, pin_memory=pin_memory,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SHIONN's behavior-cloning policy from cached frames.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/datasets/cached_frames"))
    parser.add_argument("--train-extra-cache-dir", dest="train_extra_cache_dirs",
                        type=Path, action="append", default=[],
                        help="Add cached correction episodes to training only; repeat as needed")
    parser.add_argument("--correction-sampling-fraction", type=float, default=0.0,
                        help="Expected share of optimizer draws from train-extra caches, for example 0.10")
    parser.add_argument("--start-window-frames", type=int, default=8,
                        help="Number of usable frames after each episode's first action eligible for opening boost")
    parser.add_argument("--start-sampling-boost", type=float, default=1.0,
                        help="Multiply opening-frame sampling weight; 1 keeps the original sampler")
    parser.add_argument("--map", dest="maps", action="append", default=[], metavar="NAME",
                        help="Train only on this chamber; repeat for multiple chambers")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Tier-1 default for an 8 GB GPU")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampling", choices=("chamber-balanced", "uniform"),
        default="chamber-balanced",
        help="Equal expected frames per map, or one draw per recorded usable frame.",
    )
    parser.add_argument(
        "--binary-class-weighting", choices=("balanced", "none"), default="balanced",
        help="Balance rare binary actions, or use their raw recorded frequencies.",
    )
    parser.add_argument("--jump-positive-weight", type=float,
                        help="Override jump-positive weight as a ratio to no-jump; 3 and 5 are sweep candidates")
    parser.add_argument("--mouse-head", choices=("gaussian", "binned"), default="gaussian",
                        help="Use the existing Gaussian mouse head or independent dx/dy classification")
    parser.add_argument("--mouse-bins", type=int, default=15,
                        help="Requested odd number of classes per mouse axis when --mouse-head binned")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for portable Windows/headless operation; raise on Linux after validation")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/imitation/checkpoints_v3"))
    parser.add_argument("--checkpoint-every", type=int, default=1_000, help="Save a resumable checkpoint every N optimizer steps; 0 disables periodic saves")
    parser.add_argument("--log-every", type=int, default=100, help="Print running training loss every N batches; 0 disables batch logs")
    parser.add_argument("--early-stop-patience", type=int, default=3, help="Stop after N epochs without improved validation loss; 0 disables early stopping")
    parser.add_argument("--resume", type=Path, help="Checkpoint to resume, including optimizer state")
    args = parser.parse_args()
    run_started_at = time.perf_counter()
    torch.manual_seed(args.seed)
    manifests = select_maps(discover_cached_episodes(args.cache_dir), args.maps)
    train_manifests, validation_manifests = split_episode_manifests(
        manifests, args.validation_fraction, args.seed,
        stratify=args.sampling == "chamber-balanced",
    )
    known_episodes = {item.name for item in manifests}
    correction_episode_names: set[str] = set()
    for extra_dir in args.train_extra_cache_dirs:
        additions = discover_cached_episodes(extra_dir)
        if args.maps:
            additions = [item for item in additions if item.map_name in args.maps]
        if not additions:
            raise ValueError(f"No selected cached correction episodes in {extra_dir}")
        duplicate = known_episodes & {item.name for item in additions}
        if duplicate:
            raise ValueError("Duplicate cached episode name: " + ", ".join(sorted(duplicate)[:3]))
        known_episodes.update(item.name for item in additions)
        correction_episode_names.update(item.name for item in additions)
        train_manifests.extend(additions)
    print("train episodes:", ", ".join(item.name for item in train_manifests))
    print("validation episodes:", ", ".join(item.name for item in validation_manifests))
    if args.workers < 0:
        raise ValueError("--workers must be zero or greater")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be zero or greater")
    if args.log_every < 0:
        raise ValueError("--log-every must be zero or greater")
    if args.early_stop_patience < 0:
        raise ValueError("--early-stop-patience must be zero or greater")
    if args.mouse_bins < 3 or args.mouse_bins % 2 != 1:
        raise ValueError("--mouse-bins must be odd and at least 3")
    if args.jump_positive_weight is not None and args.jump_positive_weight <= 0:
        raise ValueError("--jump-positive-weight must be greater than zero")
    if args.start_window_frames < 1 or args.start_sampling_boost < 1:
        raise ValueError("--start-window-frames must be positive and --start-sampling-boost at least 1")
    if not 0 <= args.correction_sampling_fraction < 1:
        raise ValueError("--correction-sampling-fraction must be in [0, 1)")
    if args.correction_sampling_fraction and not correction_episode_names:
        raise ValueError("--correction-sampling-fraction needs --train-extra-cache-dir")
    device = resolve_device(args.device)
    print(f"device: {device}")
    pin_memory = device.type == "cuda"
    train_dataset = BehaviorCloningDataset(train_manifests)
    validation_dataset = BehaviorCloningDataset(validation_manifests)
    sample_weights = training_sampling_weights(
        train_dataset, args.sampling,
        start_window_frames=args.start_window_frames,
        start_sampling_boost=args.start_sampling_boost,
        correction_episode_names=correction_episode_names,
        correction_sampling_fraction=args.correction_sampling_fraction,
    )
    positive_counts, mouse_scale_values = train_dataset.action_statistics(
        weights=sample_weights
    )
    sample_count = len(train_dataset)
    binary_class_weights = with_jump_positive_weight(
        binary_class_weights_for_mode(positive_counts, sample_count, args.binary_class_weighting),
        args.jump_positive_weight,
    )
    mouse_labels = train_dataset.mouse_deltas() if args.mouse_head == "binned" else None
    mouse_bins = MouseBins.fit(mouse_labels, args.mouse_bins) if mouse_labels is not None else None
    mouse_bin_counts = None
    if mouse_bins is not None:
        dx_classes, dy_classes = mouse_bins.encode(torch.from_numpy(mouse_labels))
        dx_count, dy_count = mouse_bins.class_counts()
        mouse_bin_counts = {
            "dx": torch.bincount(dx_classes, minlength=dx_count).tolist(),
            "dy": torch.bincount(dy_classes, minlength=dy_count).tolist(),
        }
    print(
        f"excluded leading idle frames: train={train_dataset.leading_idle_frames} "
        f"validation={validation_dataset.leading_idle_frames}"
    )
    print(f"training sampling: {args.sampling}; binary class weighting: {args.binary_class_weighting}")
    opening_mask = np.fromiter(
        (frame - train_dataset.first_action_frames[episode] < args.start_window_frames
         for episode, frame in train_dataset.index),
        dtype=bool, count=len(train_dataset),
    )
    print(
        f"opening sampling: {args.start_window_frames} frames, boost {args.start_sampling_boost:g}; "
        f"expected share {sample_weights[opening_mask].sum() / sample_weights.sum():.1%}"
    )
    if correction_episode_names:
        correction_mask = np.fromiter(
            (train_dataset.episode_names[episode] in correction_episode_names
             for episode, _ in train_dataset.index),
            dtype=bool, count=len(train_dataset),
        )
        print(
            f"correction sampling: {len(correction_episode_names)} episodes, "
            f"expected share {sample_weights[correction_mask].sum() / sample_weights.sum():.1%}"
        )
    print("usable training frames by map: " + ", ".join(
        f"{name}={count}" for name, count in sorted(train_dataset.chamber_sample_counts().items())
    ))
    print(
        "binary positive rates: "
        + " ".join(
            f"{name}={100.0 * count / sample_count:.2f}%"
            for name, count in zip(BINARY_ACTION_COLUMNS, positive_counts)
        )
    )
    if mouse_bins is None:
        print(f"mouse scales: dx={mouse_scale_values[0]:.3f} dy={mouse_scale_values[1]:.3f}")
    else:
        print("mouse bins from train actions: " + ", ".join(
            f"{name}={mouse_bins.config[name]['magnitude_edges']}"
            for name in ("dx", "dy")
        ))
        print("mouse bin target counts: " + ", ".join(
            f"{name}={mouse_bin_counts[name]}" for name in ("dx", "dy")
        ))
    print("jump class weights: " + str(binary_class_weights[BINARY_ACTION_COLUMNS.index("jump")].tolist()))
    binary_class_weights = binary_class_weights.to(device)
    mouse_scale = torch.as_tensor(mouse_scale_values, device=device)
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=pin_memory)
    model = (
        BinnedImitationPolicy(*mouse_bins.class_counts()) if mouse_bins is not None
        else ImitationPolicy()
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = checkpoint_config(
        args, binary_class_weights.cpu(), mouse_scale.cpu(), mouse_bins, mouse_bin_counts
    )
    config["training"]["training_episodes"] = [item.name for item in train_manifests]
    config["training"]["validation_episodes"] = [item.name for item in validation_manifests]
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
        if restored["config"].get("training", {}).get("sampling", "uniform") != args.sampling:
            raise ValueError("Resume checkpoint sampling strategy differs from this run")
        for key, default in (
            ("start_window_frames", 8), ("start_sampling_boost", 1.0),
            ("correction_sampling_fraction", 0.0),
        ):
            if restored["config"].get("training", {}).get(key, default) != getattr(args, key):
                raise ValueError(f"Resume checkpoint {key} differs from this run")
        previous_training = restored["config"].get("training", {}).get("training_episodes")
        if previous_training is not None and previous_training != config["training"]["training_episodes"]:
            raise ValueError("Resume checkpoint training episodes differ from this run")
        previous_validation = restored["config"].get("training", {}).get("validation_episodes")
        if previous_validation is not None and previous_validation != config["training"]["validation_episodes"]:
            raise ValueError("Resume checkpoint validation episodes differ from this run")
        best_validation = float(restored["best_val_loss"])
        global_step = int(restored["global_step"])
        resume_step_in_epoch = int(restored.get("step_in_epoch", 0))
        start_epoch = int(restored["epoch"]) if resume_step_in_epoch else int(restored["epoch"]) + 1
        print(f"resumed {args.resume}: epoch={restored['epoch']} step_in_epoch={resume_step_in_epoch} global_step={global_step} best_val={best_validation:.4f}")
    early_stopping = EarlyStopping(args.early_stop_patience, best_validation)
    metrics_path = args.checkpoint_dir / "metrics.jsonl"
    initial_global_step = global_step
    completed_epochs = 0
    epoch_durations: list[float] = []
    final_train_metrics = None
    final_validation_metrics = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started_at = time.perf_counter()
        train_loader = make_training_loader(
            train_dataset, args, pin_memory, epoch, correction_episode_names
        )
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
            log_every=args.log_every, mouse_bins=mouse_bins,
        )
        resume_step_in_epoch = 0
        with torch.no_grad():
            validation_metrics, _ = run_epoch(
                model, validation_loader, optimizer, scaler, device, train=False,
                binary_class_weights=binary_class_weights, mouse_scale=mouse_scale,
                global_step=global_step, mouse_bins=mouse_bins,
            )
        print(f"epoch {epoch:03d} train={train_metrics['total']:.4f} validation={validation_metrics['total']:.4f} " + " ".join(f"{key}={validation_metrics[key]:.3f}" for key in (*BINARY_ACTION_COLUMNS, "mouse")))
        improved, should_stop = early_stopping.update(validation_metrics["total"])
        best_validation = early_stopping.best_loss
        if improved:
            best_epoch = epoch
            save_checkpoint(args.checkpoint_dir / "best.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, best_val_loss=best_validation, config=config)
        save_checkpoint(args.checkpoint_dir / "last.pt", model=model, optimizer=optimizer, epoch=epoch, global_step=global_step, best_val_loss=best_validation, config=config)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps({
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "best_validation_loss": best_validation,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started_at,
                "improved": improved,
            }) + "\n")
        completed_epochs += 1
        epoch_durations.append(time.perf_counter() - epoch_started_at)
        final_train_metrics = train_metrics
        final_validation_metrics = validation_metrics
        if should_stop:
            print(f"early stopping: validation loss did not improve for {args.early_stop_patience} epochs; using best.pt")
            break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - run_started_at
    print("\n=== Training complete ===")
    print(f"elapsed wall time: {format_duration(elapsed)}")
    print(f"device: {device}")
    print(f"architecture: {config['architecture']}")
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
    print(f"epoch metrics: {metrics_path.resolve()}")
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        print(f"peak CUDA memory allocated: {peak_gib:.2f} GiB")


if __name__ == "__main__":
    main()
