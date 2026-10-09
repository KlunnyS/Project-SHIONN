"""Train the Stage 1–2 factored behavior-cloning policy."""

from __future__ import annotations

# Python standard library: training CLI, run metadata, and timing.
import argparse
import json
import math
import time
from pathlib import Path

# Installed dependencies: NumPy statistics and PyTorch optimization/loaders.
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

# Sibling modules in models.imitation (leading dot = this package).
from .checkpoint import load_checkpoint, save_checkpoint
from .dataset import BINARY_ACTION_COLUMNS, BehaviorCloningDataset, discover_cached_episodes, split_episode_manifests
from .frozen_split import DEFAULT_SPLIT_PATH, FrozenSplit, load_frozen_split
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
        return improved, self.should_stop

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self.unimproved_epochs >= self.patience

    def state_dict(self) -> dict[str, float | int]:
        return {
            "best_loss": self.best_loss,
            "unimproved_epochs": self.unimproved_epochs,
        }

    def load_state_dict(self, state: dict) -> None:
        best_loss = float(state.get("best_loss", self.best_loss))
        if not math.isclose(best_loss, self.best_loss):
            raise ValueError("Checkpoint early-stopping best loss is inconsistent")
        unimproved_epochs = int(state.get("unimproved_epochs", 0))
        if unimproved_epochs < 0:
            raise ValueError("Checkpoint early-stopping count must be non-negative")
        self.unimproved_epochs = unimproved_epochs


def consecutive_unimproved_epochs(metrics_path: Path, completed_epoch: int) -> int:
    """Recover legacy early-stopping progress from completed metric rows."""
    if completed_epoch < 1 or not metrics_path.is_file():
        return 0
    rows_by_epoch = {}
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["epoch"]) <= completed_epoch:
            rows_by_epoch[int(row["epoch"])] = row
    count = 0
    for epoch in sorted(rows_by_epoch, reverse=True):
        row = rows_by_epoch[epoch]
        if row.get("improved", False):
            break
        count += 1
    return count


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
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Sum binary losses and the selected Gaussian or binned mouse loss."""
    losses: dict[str, torch.Tensor] = {}
    total = torch.zeros((), device=targets.device)
    for index, name in enumerate(BINARY_ACTION_COLUMNS):
        weight = None if binary_class_weights is None else binary_class_weights[index]
        loss = nn.functional.cross_entropy(
            prediction[name], targets[:, index].long(), weight=weight,
            label_smoothing=label_smoothing,
        )
        losses[name] = loss
        total = total + loss
    if mouse_bins is not None:
        dx_target, dy_target = mouse_bins.encode(targets[:, 8:10])
        mouse_loss = (
            nn.functional.cross_entropy(
                prediction["mouse_dx_logits"], dx_target,
                label_smoothing=label_smoothing,
            )
            + nn.functional.cross_entropy(
                prediction["mouse_dy_logits"], dy_target,
                label_smoothing=label_smoothing,
            )
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


def augment_batch(
    frames: torch.Tensor,
    targets: torch.Tensor,
    *,
    brightness: float = 0.0,
    contrast: float = 0.0,
    horizontal_flip_probability: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply temporally consistent image transforms and matching controls.

    All four stacked frames receive the same transform.  A horizontal flip is
    only valid when strafing labels and horizontal mouse motion are transformed
    with the pixels.
    """
    if brightness < 0 or contrast < 0:
        raise ValueError("Brightness and contrast augmentation must be non-negative")
    if not 0 <= horizontal_flip_probability <= 1:
        raise ValueError("Horizontal flip probability must be in [0, 1]")
    if brightness:
        offset = torch.empty(
            (len(frames), 1, 1, 1), device=frames.device, dtype=frames.dtype
        ).uniform_(-brightness, brightness)
        frames = frames + offset
    if contrast:
        factor = torch.empty(
            (len(frames), 1, 1, 1), device=frames.device, dtype=frames.dtype
        ).uniform_(1.0 - contrast, 1.0 + contrast)
        mean = frames.mean(dim=(1, 2, 3), keepdim=True)
        frames = (frames - mean) * factor + mean
    if brightness or contrast:
        frames = frames.clamp_(0.0, 1.0)
    if horizontal_flip_probability:
        flip = torch.rand(len(frames), device=frames.device) < horizontal_flip_probability
        if flip.any():
            frames = frames.clone()
            targets = targets.clone()
            frames[flip] = frames[flip].flip(-1)
            move_a = targets[flip, 1].clone()
            targets[flip, 1] = targets[flip, 3]
            targets[flip, 3] = move_a
            targets[flip, 8] = -targets[flip, 8]
    return frames, targets


def run_epoch(
    model, loader, optimizer, scaler, device, train: bool,
    binary_class_weights: torch.Tensor, mouse_scale: torch.Tensor,
    global_step: int = 0, on_step=None, skip_batches: int = 0,
    log_every: int = 0, mouse_bins: MouseBins | None = None,
    brightness: float = 0.0, contrast: float = 0.0,
    horizontal_flip_probability: float = 0.0,
    label_smoothing: float = 0.0, max_grad_norm: float = 0.0,
) -> tuple[dict[str, float], int]:
    model.train(train)
    totals: dict[str, float] = {name: 0.0 for name in (*BINARY_ACTION_COLUMNS, "mouse", "total")}
    batches = 0
    samples = 0
    for batch_index, (frames, targets) in enumerate(loader):
        if batch_index < skip_batches:
            continue
        frames = frames.to(device=device, dtype=torch.float32, non_blocking=True)
        if device.type == "cuda":
            frames = frames.contiguous(memory_format=torch.channels_last)
        frames.div_(255.0)
        targets = targets.to(device=device, dtype=torch.float32, non_blocking=True)
        if train:
            frames, targets = augment_batch(
                frames, targets, brightness=brightness, contrast=contrast,
                horizontal_flip_probability=horizontal_flip_probability,
            )
        if train:
            optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=device.type == "cuda"):
            prediction = model(frames)
            loss, parts = behavior_cloning_loss(
                prediction, targets, binary_class_weights, mouse_scale, mouse_bins,
                label_smoothing,
            )
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        if train:
            scaler.scale(loss).backward()
            if max_grad_norm:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if on_step is not None:
                on_step(global_step, batch_index + 1)
        batch_samples = len(frames)
        totals["total"] += loss.detach().item() * batch_samples
        for name, value in parts.items():
            totals[name] += value.detach().item() * batch_samples
        batches += 1
        samples += batch_samples
        if train and log_every and batches % log_every == 0:
            print(
                f"  batch {batch_index + 1}/{len(loader)} "
                f"loss={totals['total'] / samples:.4f}"
            )
    return {name: value / max(1, samples) for name, value in totals.items()}, global_step


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
            "validation_batch_size": args.validation_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "label_smoothing": args.label_smoothing,
            "validation_fraction": args.validation_fraction,
            "holdout_maps": sorted(args.holdout_maps),
            "split_manifest": str(args.split_manifest) if not args.unfrozen_split else None,
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
            "augmentation": {
                "brightness": args.brightness_augmentation,
                "contrast": args.contrast_augmentation,
                "horizontal_flip_probability": args.horizontal_flip_probability,
            },
            "lr_scheduler": {
                "name": args.lr_scheduler,
                "factor": args.lr_reduction_factor,
                "patience": args.lr_reduction_patience,
                "min_learning_rate": args.min_learning_rate,
            },
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


def split_training_manifests(
    manifests, validation_fraction: float, seed: int, stratify: bool,
    holdout_maps: list[str],
):
    """Split episodes normally, or reserve complete chambers for validation."""
    if not holdout_maps:
        return split_episode_manifests(
            manifests, validation_fraction, seed, stratify=stratify
        )
    if len(holdout_maps) != len(set(holdout_maps)):
        raise ValueError("--holdout-map values must be unique")
    available = {manifest.map_name for manifest in manifests}
    missing = set(holdout_maps) - available
    if missing:
        raise ValueError(
            "No cached episodes for holdout map(s): " + ", ".join(sorted(missing))
        )
    held_out = set(holdout_maps)
    train = [item for item in manifests if item.map_name not in held_out]
    validation = [item for item in manifests if item.map_name in held_out]
    if not train:
        raise ValueError("At least one non-holdout chamber is required for training")
    return train, validation


def partition_frozen_training(
    manifests: list, split: FrozenSplit,
) -> tuple[list, list]:
    """Use declared episode roles, excluding evaluation episodes entirely."""
    train = [item for item in manifests if split.role(item, "base") == "train"]
    validation = [item for item in manifests if split.role(item, "base") == "validation"]
    if not train or not validation:
        raise ValueError("Frozen split needs selected training and validation episodes")
    return train, validation


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
    loader_options = {}
    if args.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(
        dataset, batch_size=args.batch_size, shuffle=sampler is None,
        sampler=sampler, generator=generator,
        num_workers=args.workers, pin_memory=pin_memory,
        **loader_options,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SHIONN's behavior-cloning policy from cached frames.")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/datasets/cached_frames"))
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_PATH,
                        help="Versioned episode split; required by default for training")
    parser.add_argument("--unfrozen-split", action="store_true",
                        help="Explicitly bypass the frozen manifest for experimental datasets")
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
    parser.add_argument(
        "--holdout-map", dest="holdout_maps", action="append", default=[],
        metavar="NAME",
        help="Reserve every episode from this chamber for validation; repeat as needed",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8, help="Tier-1 default for an 8 GB GPU")
    parser.add_argument("--validation-batch-size", type=int, default=32,
                        help="Larger inference-only batch used for validation")
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-scheduler", choices=("plateau", "none"), default="plateau",
                        help="Reduce learning rate when validation stops improving")
    parser.add_argument("--lr-reduction-factor", type=float, default=0.25,
                        help="Learning-rate multiplier used by the plateau scheduler")
    parser.add_argument("--lr-reduction-patience", type=int, default=0,
                        help="Unimproved epochs before reducing learning rate")
    parser.add_argument("--min-learning-rate", type=float, default=1e-6,
                        help="Lower bound for plateau learning-rate reductions")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Clip the global gradient norm; 0 disables clipping")
    parser.add_argument("--label-smoothing", type=float, default=0.01)
    parser.add_argument("--brightness-augmentation", type=float, default=0.05,
                        help="Random per-stack brightness offset in normalized image units")
    parser.add_argument("--contrast-augmentation", type=float, default=0.10,
                        help="Random per-stack contrast range around 1.0")
    parser.add_argument("--horizontal-flip-probability", type=float, default=0.0,
                        help="Optional mirrored-view augmentation with A/D and mouse-dx correction")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sampling", choices=("chamber-balanced", "uniform"),
        default="chamber-balanced",
        help="Equal expected frames per map, or one draw per recorded usable frame.",
    )
    parser.add_argument(
        "--binary-class-weighting", choices=("balanced", "none"), default="none",
        help="Balance rare binary actions, or use their raw recorded frequencies.",
    )
    parser.add_argument("--jump-positive-weight", type=float, default=3.0,
                        help="Override jump-positive weight as a ratio to no-jump; 3 and 5 are sweep candidates")
    parser.add_argument("--mouse-head", choices=("gaussian", "binned"), default="binned",
                        help="Use the existing Gaussian mouse head or independent dx/dy classification")
    parser.add_argument("--mouse-bins", type=int, default=15,
                        help="Requested odd number of classes per mouse axis when --mouse-head binned")
    parser.add_argument("--workers", type=int, default=0, help="Use 0 for portable Windows/headless operation; raise on Linux after validation")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("models/imitation/checkpoints_v5"))
    parser.add_argument("--checkpoint-every", type=int, default=10_000, help="Save a resumable checkpoint every N optimizer steps; 0 disables periodic saves")
    parser.add_argument("--log-every", type=int, default=100, help="Print running training loss every N batches; 0 disables batch logs")
    parser.add_argument("--early-stop-patience", type=int, default=2, help="Stop after N epochs without improved validation loss; 0 disables early stopping")
    parser.add_argument("--resume", type=Path,
                        help="Checkpoint to resume, including optimizer and stopping state")
    args = parser.parse_args()
    run_started_at = time.perf_counter()
    torch.manual_seed(args.seed)
    all_manifests = discover_cached_episodes(args.cache_dir)
    frozen_split = None if args.unfrozen_split else load_frozen_split(args.split_manifest)
    evaluation_maps = {"evaluation1", "evaluation2"}
    if frozen_split is None and any(item.map_name in evaluation_maps for item in all_manifests):
        raise ValueError("Evaluation chambers cannot enter training, even with --unfrozen-split")
    if frozen_split is not None:
        frozen_split.verify_cache(all_manifests, "base")
        expected_holdouts = {name for name, role in frozen_split.map_roles.items()
                             if role == "validation"}
        if args.holdout_maps and set(args.holdout_maps) != expected_holdouts:
            raise ValueError("--holdout-map conflicts with frozen split validation chambers")
        args.holdout_maps = sorted(expected_holdouts)
    manifests = select_maps(all_manifests, args.maps)
    if frozen_split is not None:
        train_manifests, validation_manifests = partition_frozen_training(manifests, frozen_split)
    else:
        train_manifests, validation_manifests = split_training_manifests(
            manifests, args.validation_fraction, args.seed,
            stratify=args.sampling == "chamber-balanced",
            holdout_maps=args.holdout_maps,
        )
    known_episodes = {item.name for item in all_manifests}
    correction_episode_names: set[str] = set()
    for extra_dir in args.train_extra_cache_dirs:
        additions = discover_cached_episodes(extra_dir)
        if any(item.map_name in evaluation_maps for item in additions):
            raise ValueError(f"Evaluation chambers cannot enter training from {extra_dir}")
        if frozen_split is not None:
            group = "recovery" if extra_dir.resolve() == Path("data/datasets/recovery_cached_frames").resolve() else extra_dir.name
            frozen_split.verify_cache(additions, group)
        if args.maps:
            additions = [item for item in additions if item.map_name in args.maps]
        if frozen_split is not None:
            blocked = [item.name for item in additions if frozen_split.role(item, group) != "train"]
            if blocked:
                raise ValueError("Extra cache contains non-training episodes: " + ", ".join(blocked[:3]))
        else:
            additions = [item for item in additions if item.map_name not in args.holdout_maps]
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
    if args.batch_size < 1 or args.validation_batch_size < 1:
        raise ValueError("Training and validation batch sizes must be positive")
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
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if not 0 < args.lr_reduction_factor < 1:
        raise ValueError("--lr-reduction-factor must be between zero and one")
    if args.lr_reduction_patience < 0:
        raise ValueError("--lr-reduction-patience must be zero or greater")
    if not 0 <= args.min_learning_rate < args.learning_rate:
        raise ValueError("--min-learning-rate must be non-negative and below --learning-rate")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be non-negative")
    if not 0 <= args.label_smoothing < 1:
        raise ValueError("--label-smoothing must be in [0, 1)")
    if args.brightness_augmentation < 0 or args.contrast_augmentation < 0:
        raise ValueError("Image augmentation strengths must be non-negative")
    if not 0 <= args.horizontal_flip_probability <= 1:
        raise ValueError("--horizontal-flip-probability must be in [0, 1]")
    if args.start_window_frames < 1 or args.start_sampling_boost < 1:
        raise ValueError("--start-window-frames must be positive and --start-sampling-boost at least 1")
    if not 0 <= args.correction_sampling_fraction < 1:
        raise ValueError("--correction-sampling-fraction must be in [0, 1)")
    if args.correction_sampling_fraction and not correction_episode_names:
        raise ValueError("--correction-sampling-fraction needs --train-extra-cache-dir")
    device = resolve_device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    if args.holdout_maps:
        print("whole-chamber validation: " + ", ".join(sorted(args.holdout_maps)))
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
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.validation_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=pin_memory,
        **({"persistent_workers": True, "prefetch_factor": 2} if args.workers else {}),
    )
    model = (
        BinnedImitationPolicy(*mouse_bins.class_counts()) if mouse_bins is not None
        else ImitationPolicy()
    ).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
        fused=device.type == "cuda",
    )
    scheduler = None
    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_reduction_factor,
            patience=args.lr_reduction_patience, min_lr=args.min_learning_rate,
        )
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config = checkpoint_config(
        args, binary_class_weights.cpu(), mouse_scale.cpu(), mouse_bins, mouse_bin_counts
    )
    config["training"]["training_episodes"] = [item.name for item in train_manifests]
    config["training"]["validation_episodes"] = [item.name for item in validation_manifests]
    if frozen_split is not None:
        config["training"]["split_id"] = frozen_split.split_id
        config["training"]["split_sha256"] = frozen_split.digest
    best_validation = float("inf")
    global_step = 0
    start_epoch = 1
    resume_step_in_epoch = 0
    best_epoch = None
    restored_training_state = None
    completed_resume_epoch = 0
    if args.resume:
        restored = load_checkpoint(args.resume, model=model, optimizer=optimizer, device=device)
        compatibility_keys = ("architecture", "preprocessing", "action_columns", "target_processing")
        if any(restored["config"].get(key) != config.get(key) for key in compatibility_keys):
            raise ValueError("Resume checkpoint model or target processing differs from this run")
        if restored["config"].get("training", {}).get("sampling", "uniform") != args.sampling:
            raise ValueError("Resume checkpoint sampling strategy differs from this run")
        restored_training = restored["config"].get("training", {})
        previous_split = restored_training.get("split_id")
        current_split = config["training"].get("split_id")
        if previous_split is not None and previous_split != current_split:
            raise ValueError("Resume checkpoint frozen split differs from this run")
        previous_digest = restored_training.get("split_sha256")
        if previous_digest is not None and previous_digest != config["training"].get("split_sha256"):
            raise ValueError("Resume checkpoint frozen split manifest changed")
        resume_settings = {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "label_smoothing": args.label_smoothing,
            "holdout_maps": sorted(args.holdout_maps),
            "binary_class_weighting": args.binary_class_weighting,
            "jump_positive_weight": args.jump_positive_weight,
            "mouse_head": args.mouse_head,
            "start_window_frames": args.start_window_frames,
            "start_sampling_boost": args.start_sampling_boost,
            "correction_sampling_fraction": args.correction_sampling_fraction,
            "augmentation": {
                "brightness": args.brightness_augmentation,
                "contrast": args.contrast_augmentation,
                "horizontal_flip_probability": args.horizontal_flip_probability,
            },
            "lr_scheduler": {
                "name": args.lr_scheduler,
                "factor": args.lr_reduction_factor,
                "patience": args.lr_reduction_patience,
                "min_learning_rate": args.min_learning_rate,
            },
        }
        for key, current_value in resume_settings.items():
            previous_value = restored_training.get(key)
            if key == "lr_scheduler" and previous_value is None:
                previous_value = {
                    "name": "none", "factor": args.lr_reduction_factor,
                    "patience": args.lr_reduction_patience,
                    "min_learning_rate": args.min_learning_rate,
                }
            if previous_value != current_value:
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
        completed_resume_epoch = int(restored["epoch"]) - (1 if resume_step_in_epoch else 0)
        restored_training_state = restored.get("training_state") or None
        print(f"resumed {args.resume}: epoch={restored['epoch']} step_in_epoch={resume_step_in_epoch} global_step={global_step} best_val={best_validation:.4f}")
    early_stopping = EarlyStopping(args.early_stop_patience, best_validation)
    metrics_path = args.checkpoint_dir / "metrics.jsonl"
    if restored_training_state is not None:
        early_stopping.load_state_dict(restored_training_state.get("early_stopping", {}))
        if scheduler is not None and restored_training_state.get("scheduler") is not None:
            scheduler.load_state_dict(restored_training_state["scheduler"])
    elif args.resume:
        early_stopping.unimproved_epochs = consecutive_unimproved_epochs(
            args.resume.parent / "metrics.jsonl", completed_resume_epoch
        )
        if early_stopping.unimproved_epochs:
            print(
                "restored early-stopping count from metrics: "
                f"{early_stopping.unimproved_epochs} unimproved epoch(s)"
            )
    if args.resume and early_stopping.should_stop:
        print(
            "early stopping was already satisfied at the resume checkpoint; "
            "no additional epochs will run"
        )
        start_epoch = args.epochs + 1

    def current_training_state() -> dict:
        return {
            "early_stopping": early_stopping.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }

    def save_training_checkpoint(
        path: Path, *, epoch: int, step: int, step_in_epoch: int = 0,
    ) -> None:
        save_checkpoint(
            path, model=model, optimizer=optimizer, epoch=epoch,
            step_in_epoch=step_in_epoch, global_step=step,
            best_val_loss=best_validation, config=config,
            training_state=current_training_state(),
        )

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
                save_training_checkpoint(
                    path, epoch=epoch, step=step, step_in_epoch=step_in_epoch,
                )
                save_training_checkpoint(
                    args.checkpoint_dir / "last.pt", epoch=epoch, step=step,
                    step_in_epoch=step_in_epoch,
                )
        epoch_learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics, global_step = run_epoch(
            model, train_loader, optimizer, scaler, device, train=True,
            binary_class_weights=binary_class_weights, mouse_scale=mouse_scale,
            global_step=global_step, on_step=save_periodic, skip_batches=skip_batches,
            log_every=args.log_every, mouse_bins=mouse_bins,
            brightness=args.brightness_augmentation,
            contrast=args.contrast_augmentation,
            horizontal_flip_probability=args.horizontal_flip_probability,
            label_smoothing=args.label_smoothing,
            max_grad_norm=args.max_grad_norm,
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
        if scheduler is not None:
            scheduler.step(validation_metrics["total"])
        next_learning_rate = optimizer.param_groups[0]["lr"]
        if improved:
            best_epoch = epoch
            save_training_checkpoint(
                args.checkpoint_dir / "best.pt", epoch=epoch, step=global_step,
            )
        save_training_checkpoint(
            args.checkpoint_dir / "last.pt", epoch=epoch, step=global_step,
        )
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps({
                "epoch": epoch,
                "global_step": global_step,
                "train": train_metrics,
                "validation": validation_metrics,
                "best_validation_loss": best_validation,
                "learning_rate": epoch_learning_rate,
                "next_learning_rate": next_learning_rate,
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
