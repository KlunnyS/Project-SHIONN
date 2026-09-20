"""Episode-aware, memory-mapped data loading for behavior cloning."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
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
    map_name: str = "unknown"


def episode_map_name(metadata: dict) -> str:
    """Read the cached map label, falling back to older source recordings."""
    map_name = metadata.get("map")
    if not map_name and metadata.get("actions_csv"):
        source_metadata = Path(metadata["actions_csv"]).parent / "metadata.json"
        if source_metadata.is_file():
            map_name = json.loads(source_metadata.read_text(encoding="utf-8")).get("map")
    return map_name if isinstance(map_name, str) and map_name.strip() else "unknown"


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
        manifests.append(EpisodeManifest(
            metadata["episode"], frames_path, actions_path,
            int(metadata["frame_count"]), episode_map_name(metadata),
        ))
    return manifests


def split_episode_manifests(
    manifests: list[EpisodeManifest], validation_fraction: float = 0.2, seed: int = 0,
    stratify: bool = True,
) -> tuple[list[EpisodeManifest], list[EpisodeManifest]]:
    """Split whole episodes, never adjacent frames, into train and validation."""
    if len(manifests) < 2:
        raise ValueError("At least two cached episodes are required for an episode-level train/validation split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not stratify:
        ordered = list(manifests)
        random.Random(seed).shuffle(ordered)
        validation_count = max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
        return ordered[validation_count:], ordered[:validation_count]
    by_map: dict[str, list[EpisodeManifest]] = defaultdict(list)
    for manifest in manifests:
        by_map[manifest.map_name].append(manifest)
    rng = random.Random(seed)
    train: list[EpisodeManifest] = []
    validation: list[EpisodeManifest] = []
    for map_name in sorted(by_map):
        ordered = sorted(by_map[map_name], key=lambda item: item.name)
        rng.shuffle(ordered)
        validation_count = (
            max(1, min(len(ordered) - 1, round(len(ordered) * validation_fraction)))
            if len(ordered) > 1 else 0
        )
        validation.extend(ordered[:validation_count])
        train.extend(ordered[validation_count:])
    if not validation:
        raise ValueError("At least one chamber needs two episodes for validation")
    rng.shuffle(train)
    rng.shuffle(validation)
    return train, validation


class BehaviorCloningDataset:
    """Returns a four-frame RGB stack and its action at the newest frame."""
    def __init__(
        self,
        manifests: list[EpisodeManifest],
        frame_stack: int = 4,
        trim_leading_idle: bool = True,
    ):
        if not manifests:
            raise ValueError("Dataset needs at least one episode")
        if frame_stack != 4:
            raise ValueError("This v1 architecture is specified for exactly four stacked frames")
        self.frame_stack = frame_stack
        self.episodes: list[tuple[np.ndarray, np.ndarray]] = []
        self.episode_names: list[str] = []
        self.episode_maps: list[str] = []
        self.first_action_frames: list[int] = []
        self.index: list[tuple[int, int]] = []
        self.leading_idle_frames = 0
        for manifest in manifests:
            frames = np.load(manifest.frames_path, mmap_mode="r")
            if frames.shape != (manifest.frame_count, 180, 320, 3):
                raise ValueError(f"Unexpected cached frame shape for {manifest.name}: {frames.shape}")
            actions = self._load_actions(manifest.actions_path, manifest.frame_count)
            episode_index = len(self.episodes)
            self.episodes.append((frames, actions))
            self.episode_names.append(manifest.name)
            self.episode_maps.append(manifest.map_name)
            first_frame = self._first_action_frame(actions) if trim_leading_idle else 0
            self.first_action_frames.append(first_frame)
            self.leading_idle_frames += first_frame
            self.index.extend(
                (episode_index, frame_idx)
                for frame_idx in range(first_frame, manifest.frame_count)
            )
        if not self.index:
            raise ValueError("Dataset contains no non-idle action frames")

    @staticmethod
    def _first_action_frame(actions: np.ndarray) -> int:
        """Discard ambiguous waiting labels before the demonstrator starts."""
        active = np.any(actions[:, :8] != 0, axis=1) | np.any(actions[:, 8:10] != 0, axis=1)
        indices = np.flatnonzero(active)
        return int(indices[0]) if len(indices) else len(actions)

    def action_statistics(
        self, balance_chambers: bool = False, weights: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return action statistics, optionally weighted to match training sampling."""
        targets = np.stack(
            [self.episodes[episode_index][1][frame_idx] for episode_index, frame_idx in self.index]
        )
        if balance_chambers or weights is not None:
            if weights is None:
                weights = self.chamber_sampling_weights()
            if len(weights) != len(targets) or np.any(weights <= 0):
                raise ValueError("Action statistic weights must match usable frames and be positive")
            positive_counts = np.sum(targets[:, :8] * weights[:, None], axis=0, dtype=np.float64)
            mouse_mean = np.average(targets[:, 8:10], axis=0, weights=weights)
            mouse_scale = np.sqrt(np.average(
                (targets[:, 8:10] - mouse_mean) ** 2, axis=0, weights=weights
            ))
        else:
            positive_counts = targets[:, :8].sum(axis=0, dtype=np.float64)
            mouse_scale = targets[:, 8:10].std(axis=0, dtype=np.float64)
        return positive_counts, np.maximum(mouse_scale, 1.0).astype(np.float32)

    def mouse_deltas(self) -> np.ndarray:
        """Return actual usable mouse labels in index order for bin fitting."""
        return np.stack([
            self.episodes[episode_index][1][frame_idx, 8:10]
            for episode_index, frame_idx in self.index
        ])

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

    def chamber_sample_counts(self) -> Counter[str]:
        """Count usable frames by chamber after leading-idle trimming."""
        return Counter(self.episode_maps[episode_index] for episode_index, _ in self.index)

    def chamber_sampling_weights(self) -> np.ndarray:
        """Give each chamber equal expected frame draws while keeping epoch length."""
        counts = self.chamber_sample_counts()
        per_map_scale = len(self.index) / len(counts)
        return np.fromiter(
            (per_map_scale / counts[self.episode_maps[episode_index]]
             for episode_index, _ in self.index),
            dtype=np.float64, count=len(self.index),
        )

    def __getitem__(self, item: int) -> tuple[np.ndarray, np.ndarray]:
        episode_index, frame_idx = self.index[item]
        frames, actions = self.episodes[episode_index]
        # Repeat the first frame at episode start instead of leaking frames from another episode.
        indices = [max(0, frame_idx - offset) for offset in range(self.frame_stack - 1, -1, -1)]
        stack = np.ascontiguousarray(frames[indices].transpose(0, 3, 1, 2).reshape(12, 180, 320))
        return stack, actions[frame_idx].copy()
