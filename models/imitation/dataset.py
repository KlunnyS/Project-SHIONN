"""Episode-aware, memory-mapped data loading for behavior cloning."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ACTION_COLUMNS = (
    "move_w", "move_a", "move_s", "move_d", "jump", "use",
    "fire_left", "fire_right", "mouse_dx", "mouse_dy",
)
BINARY_ACTION_COLUMNS = ACTION_COLUMNS[:8]


@dataclass(frozen=True)
class EpisodeManifest:
    name: str
    frames_path: Path
    actions_path: Path
    frame_count: int


def discover_cached_episodes(cache_dir: Path) -> list[EpisodeManifest]:
    manifests: list[EpisodeManifest] = []
    for metadata_path in sorted(cache_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames_path = metadata_path.with_suffix(".npy")
        actions_path = metadata_path.parent / metadata.get("cached_actions", "")
        if not actions_path.is_file():  # Compatibility with caches written before portable labels existed.
            actions_path = Path(metadata["actions_csv"])
        if not frames_path.is_file() or not actions_path.is_file():
            raise FileNotFoundError(f"Incomplete cached episode described by {metadata_path}")
        manifests.append(EpisodeManifest(metadata["episode"], frames_path, actions_path, int(metadata["frame_count"])))
    return manifests


def split_episode_manifests(
    manifests: list[EpisodeManifest], validation_fraction: float = 0.2, seed: int = 0
) -> tuple[list[EpisodeManifest], list[EpisodeManifest]]:
    """Split whole episodes, never adjacent frames, into train and validation."""
    if len(manifests) < 2:
        raise ValueError("At least two cached episodes are required for an episode-level train/validation split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    ordered = list(manifests)
    random.Random(seed).shuffle(ordered)
    validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
    return ordered[validation_count:], ordered[:validation_count]


class BehaviorCloningDataset:
    """Returns a four-frame RGB stack and its action at the newest frame."""
    def __init__(self, manifests: list[EpisodeManifest], frame_stack: int = 4):
        if not manifests:
            raise ValueError("Dataset needs at least one episode")
        if frame_stack != 4:
            raise ValueError("This v1 architecture is specified for exactly four stacked frames")
        self.frame_stack = frame_stack
        self.episodes: list[tuple[np.ndarray, np.ndarray]] = []
        self.index: list[tuple[int, int]] = []
        for manifest in manifests:
            frames = np.load(manifest.frames_path, mmap_mode="r")
            if frames.shape != (manifest.frame_count, 180, 320, 3):
                raise ValueError(f"Unexpected cached frame shape for {manifest.name}: {frames.shape}")
            actions = self._load_actions(manifest.actions_path, manifest.frame_count)
            episode_index = len(self.episodes)
            self.episodes.append((frames, actions))
            self.index.extend((episode_index, frame_idx) for frame_idx in range(manifest.frame_count))

    @staticmethod
    def _load_actions(path: Path, expected_count: int) -> np.ndarray:
        if path.suffix == ".npy":
            values = np.load(path, mmap_mode="r")
            if values.shape != (expected_count, len(ACTION_COLUMNS)):
                raise ValueError(f"{path} has action shape {values.shape}, expected {(expected_count, len(ACTION_COLUMNS))}")
            return values
        import csv
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != expected_count:
            raise ValueError(f"{path} contains {len(rows)} rows, expected {expected_count}")
        values = np.empty((expected_count, len(ACTION_COLUMNS)), dtype=np.float32)
        for index, row in enumerate(rows):
            if int(row["frame_idx"]) != index:
                raise ValueError(f"{path}: frame_idx is not aligned at row {index}")
            values[index] = [float(row[column]) for column in ACTION_COLUMNS]
        return values

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        episode_index, frame_idx = self.index[item]
        frames, actions = self.episodes[episode_index]
        # Repeat the first frame at episode start instead of leaking frames from another episode.
        indices = [max(0, frame_idx - offset) for offset in range(self.frame_stack - 1, -1, -1)]
        stack = np.ascontiguousarray(frames[indices].transpose(0, 3, 1, 2).reshape(12, 180, 320))
        return stack, actions[frame_idx].copy()
