"""Compare a checkpoint's actions with held-out human actions on the same frames.

This is an open-loop imitation score. It cannot measure whether the policy
would recover after its own actions change the game state; use live attempts
for that question.
"""

from __future__ import annotations

# Python standard library: CLI, reports, and grouped results.
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Installed dependencies: numerical arrays and PyTorch inference/data loading.
import numpy as np
import torch
from torch.utils.data import DataLoader

# Project modules under models/imitation: cached episodes and policy inference.
from models.imitation.dataset import (
    ACTION_COLUMNS, BINARY_ACTION_COLUMNS, BehaviorCloningDataset,
    EpisodeManifest, discover_cached_episodes,
)
from models.imitation.inference import PolicyInference
from models.imitation.recurrent import RecurrentBinnedPolicy


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ActionMetrics:
    def __init__(self) -> None:
        self.frames = 0
        self.exact_matches = 0
        self.true_positive = np.zeros(8, dtype=np.int64)
        self.false_positive = np.zeros(8, dtype=np.int64)
        self.false_negative = np.zeros(8, dtype=np.int64)
        self.true_negative = np.zeros(8, dtype=np.int64)
        self.mouse_absolute_error = np.zeros(2, dtype=np.float64)

    def add(self, expected: np.ndarray, predicted: np.ndarray) -> None:
        expected_binary = expected[:8].astype(bool)
        predicted_binary = predicted[:8].astype(bool)
        self.frames += 1
        self.exact_matches += int(np.array_equal(expected_binary, predicted_binary))
        self.true_positive += expected_binary & predicted_binary
        self.false_positive += ~expected_binary & predicted_binary
        self.false_negative += expected_binary & ~predicted_binary
        self.true_negative += ~expected_binary & ~predicted_binary
        self.mouse_absolute_error += np.abs(expected[8:10] - predicted[8:10])

    def report(self) -> dict:
        actions = {}
        for index, name in enumerate(BINARY_ACTION_COLUMNS):
            tp = int(self.true_positive[index])
            fp = int(self.false_positive[index])
            fn = int(self.false_negative[index])
            tn = int(self.true_negative[index])
            precision = tp / (tp + fp) if tp + fp else None
            recall = tp / (tp + fn) if tp + fn else None
            f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
            actions[name] = {
                "accuracy": (tp + tn) / self.frames if self.frames else None,
                "precision": precision, "recall": recall, "f1": f1,
                "true_positive": tp, "false_positive": fp,
                "false_negative": fn, "true_negative": tn,
            }
        return {
            "frames": self.frames,
            "exact_binary_match": self.exact_matches / self.frames if self.frames else None,
            "mouse_mae": {
                "dx": self.mouse_absolute_error[0] / self.frames if self.frames else None,
                "dy": self.mouse_absolute_error[1] / self.frames if self.frames else None,
            },
            "actions": actions,
        }


def select_manifests(
    manifests: list[EpisodeManifest], config: dict, subset: str,
    maps: list[str] | None = None,
) -> list[EpisodeManifest]:
    if subset == "validation":
        names = config.get("training", {}).get("validation_episodes")
        if not isinstance(names, list) or not names:
            raise ValueError(
                "Checkpoint has no saved validation episode list. Use a checkpoint "
                "from the updated trainer, or --subset all with a separate evaluation cache."
            )
        available = {item.name: item for item in manifests}
        missing = set(names) - available.keys()
        if missing:
            raise ValueError(f"Validation episodes missing from cache: {', '.join(sorted(missing)[:5])}")
        selected = [available[name] for name in names]
    else:
        selected = manifests
    if maps:
        selected = [item for item in selected if item.map_name in maps]
        missing_maps = set(maps) - {item.map_name for item in selected}
        if missing_maps:
            raise ValueError(f"No selected episodes for maps: {', '.join(sorted(missing_maps))}")
    if not selected:
        raise ValueError("No cached episodes selected for benchmarking")
    return selected


def recurrent_prediction_batches(inference, dataset, score_starts):
    """Yield live-path GRU predictions in episode order, resetting per episode."""
    offset = 0
    for episode_index, (frames, _) in enumerate(dataset.episodes):
        inference.reset()
        for frame_idx in range(len(frames)):
            stacked, target = dataset[offset]
            action, diagnostics = inference.predict_stacked_tensor(
                stacked, history_frames=min(frame_idx + 1, dataset.frame_stack)
            )
            if frame_idx >= score_starts[episode_index]:
                predicted = np.asarray(
                    [[action[name] for name in ACTION_COLUMNS]], dtype=np.float32
                )
                probabilities = np.asarray(
                    [[diagnostics["binary_probabilities"][name]
                      for name in BINARY_ACTION_COLUMNS]], dtype=np.float32
                )
                bin_diagnostics = diagnostics.get("mouse_bin_probabilities")
                mouse_probabilities = (
                    [
                        np.asarray([bin_diagnostics[axis]], dtype=np.float32)
                        for axis in ("dx", "dy")
                    ]
                    if bin_diagnostics is not None else None
                )
                yield (
                    [(episode_index, frame_idx)], target[None], predicted,
                    probabilities, mouse_probabilities,
                )
            offset += 1


def score_checkpoint(
    checkpoint: Path, cache_dir: Path, output_dir: Path, *,
    subset: str = "validation", maps: list[str] | None = None,
    batch_size: int = 8, device: str = "auto", final_frames: int = 24,
    jump_threshold: float | None = None, move_w_threshold: float | None = None,
    mouse_decode: str = "argmax", mouse_move_threshold: float = 0.5,
) -> dict:
    if batch_size <= 0 or final_frames <= 0:
        raise ValueError("batch size and final-frame window must be positive")
    inference = PolicyInference(
        checkpoint, device=device, jump_threshold=jump_threshold,
        move_w_threshold=move_w_threshold, mouse_decode=mouse_decode,
        mouse_move_threshold=mouse_move_threshold,
    )
    mouse_bins = getattr(inference, "mouse_bins", None)
    manifests = select_manifests(
        discover_cached_episodes(cache_dir), inference.config, subset, maps
    )
    recurrent = isinstance(inference.policy, RecurrentBinnedPolicy)
    dataset = BehaviorCloningDataset(manifests, trim_leading_idle=not recurrent)
    score_starts = (
        [dataset._first_action_frame(actions) for _, actions in dataset.episodes]
        if recurrent else None
    )
    loader = None if recurrent else DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    rows_path = output_dir / "action_comparison.csv"
    metrics = defaultdict(lambda: {"all": ActionMetrics(), "final_frames": ActionMetrics()})
    metrics["overall"]
    fieldnames = ["episode", "map", "frame_idx", "final_frames"]
    for name in BINARY_ACTION_COLUMNS:
        fieldnames.extend((f"expected_{name}", f"predicted_{name}", f"prob_{name}"))
    for name in ACTION_COLUMNS[8:]:
        fieldnames.extend((f"expected_{name}", f"predicted_{name}"))
    if mouse_bins is not None:
        fieldnames.extend(("mouse_dx_bin_probabilities", "mouse_dy_bin_probabilities"))

    def prediction_batches():
        if recurrent:
            yield from recurrent_prediction_batches(inference, dataset, score_starts)
            return
        offset = 0
        with torch.no_grad():
            for frames, targets in loader:
                output = inference.policy(
                    frames.to(inference.device, dtype=torch.float32).div_(255.0)
                )
                predicted, probabilities, mouse_probabilities = inference.decode_outputs(output)
                indices = dataset.index[offset:offset + len(targets)]
                yield indices, targets.numpy(), predicted, probabilities, mouse_probabilities
                offset += len(targets)

    with rows_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for indices, expected, predicted, probabilities, mouse_probabilities in prediction_batches():
            for batch_index, (episode_index, frame_idx) in enumerate(indices):
                manifest = manifests[episode_index]
                is_final = frame_idx >= manifest.frame_count - final_frames
                row = {
                    "episode": manifest.name, "map": manifest.map_name,
                    "frame_idx": frame_idx, "final_frames": int(is_final),
                }
                for action_index, name in enumerate(BINARY_ACTION_COLUMNS):
                    row[f"expected_{name}"] = int(expected[batch_index, action_index])
                    row[f"predicted_{name}"] = int(predicted[batch_index, action_index])
                    row[f"prob_{name}"] = float(probabilities[batch_index, action_index])
                for action_index, name in enumerate(ACTION_COLUMNS[8:], start=8):
                    row[f"expected_{name}"] = float(expected[batch_index, action_index])
                    row[f"predicted_{name}"] = int(predicted[batch_index, action_index])
                if mouse_bins is not None and mouse_probabilities is not None:
                    row["mouse_dx_bin_probabilities"] = json.dumps(mouse_probabilities[0][batch_index].tolist())
                    row["mouse_dy_bin_probabilities"] = json.dumps(mouse_probabilities[1][batch_index].tolist())
                writer.writerow(row)
                for key in ("overall", manifest.map_name):
                    metrics[key]["all"].add(expected[batch_index], predicted[batch_index])
                    if is_final:
                        metrics[key]["final_frames"].add(
                            expected[batch_index], predicted[batch_index]
                        )

    report = {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "architecture": inference.config.get("architecture"),
        "scoring_mode": "episode_sequential_step" if recurrent else "feedforward_batched",
        "action_decode": {
            "mouse": inference.mouse_decode if mouse_bins is not None else "gaussian_mean",
            "mouse_move_threshold": (
                inference.mouse_move_threshold
                if mouse_bins is not None and inference.mouse_decode == "hierarchical"
                else None
            ),
            "jump_threshold": getattr(inference, "jump_threshold", None),
            "move_w_threshold": getattr(inference, "move_w_threshold", None),
        },
        "preprocessing": inference.config.get("preprocessing"),
        "cache_dir": str(cache_dir.resolve()),
        "subset": subset,
        "episodes": len(manifests),
        "final_frame_window": final_frames,
        "final_frames_note": "Final frames approximate goal approach only for successful expert episodes.",
        "by_map": {
            name: {segment: value.report() for segment, value in segments.items()}
            for name, segments in sorted(metrics.items()) if name != "overall"
        },
        "overall": {
            segment: value.report() for segment, value in metrics["overall"].items()
        },
        "action_rows": str(rows_path),
        "mouse_bins": mouse_bins.config if mouse_bins is not None else None,
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for name, values in report["by_map"].items():
        all_frames = values["all"]
        print(
            f"{name}: {all_frames['frames']} frames, "
            f"exact binary match={all_frames['exact_binary_match']:.1%}, "
            f"mouse MAE=({all_frames['mouse_mae']['dx']:.2f}, "
            f"{all_frames['mouse_mae']['dy']:.2f})"
        )
    print(f"Expert comparison: {report_path}")
    print(f"Per-frame actions: {rows_path}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score predicted actions on expert frames.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/datasets/cached_frames"))
    parser.add_argument("--subset", choices=("validation", "all"), default="validation")
    parser.add_argument("--map", dest="maps", action="append", metavar="NAME")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--final-frames", type=int, default=24)
    parser.add_argument("--jump-threshold", type=float)
    parser.add_argument("--move-w-threshold", type=float)
    parser.add_argument("--mouse-decode", choices=("argmax", "hierarchical", "expected"), default="argmax")
    parser.add_argument("--mouse-move-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_dir = args.output_dir or (
        Path("model_attempts/benchmarks") / datetime.now().strftime("expert_%Y%m%d_%H%M%S_%f")
    )
    score_checkpoint(
        args.checkpoint, args.cache_dir, output_dir, subset=args.subset,
        maps=args.maps, batch_size=args.batch_size, device=args.device,
        final_frames=args.final_frames, jump_threshold=args.jump_threshold,
        move_w_threshold=args.move_w_threshold, mouse_decode=args.mouse_decode,
        mouse_move_threshold=args.mouse_move_threshold,
    )


if __name__ == "__main__":
    main()
