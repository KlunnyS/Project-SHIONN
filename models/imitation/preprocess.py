"""Offline video-to-array preprocessing for SHIONN recordings.

The resulting .npy arrays contain uint8 RGB frames shaped (N, 180, 320, 3).
They are memory-mappable, so training never decodes video or resizes frames.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np

TARGET_WIDTH = 320
TARGET_HEIGHT = 180
PREPROCESSING_CONFIG = {
    "source_color_order": "BGR",
    "cached_color_order": "RGB",
    "resize": {"width": TARGET_WIDTH, "height": TARGET_HEIGHT, "interpolation": "INTER_AREA"},
    "normalization": "uint8_rgb_divide_255",
    "frame_stack": 4,
}
ACTION_COLUMNS = (
    "move_w", "move_a", "move_s", "move_d", "jump", "use",
    "fire_left", "fire_right", "mouse_dx", "mouse_dy",
)


def discover_episodes(recordings_dir: Path) -> list[Path]:
    """Return direct child directories containing both recording artifacts."""
    if not recordings_dir.exists():
        return []
    return sorted(
        path for path in recordings_dir.iterdir()
        if path.is_dir() and (path / "video.mp4").is_file() and (path / "actions.csv").is_file()
    )


def read_actions(actions_path: Path) -> list[dict[str, str]]:
    with actions_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{actions_path} has no action rows")
    missing = set(ACTION_COLUMNS).difference(rows[0])
    if missing:
        raise ValueError(f"{actions_path} is missing columns: {', '.join(sorted(missing))}")
    frame_indices = [int(row["frame_idx"]) for row in rows]
    if frame_indices != list(range(len(rows))):
        raise ValueError(
            f"{actions_path} frame_idx must be contiguous and start at zero; "
            "re-record or normalize it before caching."
        )
    return rows


def action_array(rows: list[dict[str, str]]) -> np.ndarray:
    """Make action labels portable with the frame cache, while retaining the CSV source."""
    return np.asarray([[float(row[column]) for column in ACTION_COLUMNS] for row in rows], dtype=np.float32)


def preprocess_bgr_frame(frame_bgr: np.ndarray) -> np.ndarray:
    """Apply exactly the cache color conversion and resize to one captured frame."""
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError(f"Expected a BGR HxWx3 frame, received {frame_bgr.shape}")
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(frame_rgb, (TARGET_WIDTH, TARGET_HEIGHT), interpolation=cv2.INTER_AREA)


def cache_episode(episode_dir: Path, cache_dir: Path, overwrite: bool = False) -> Path:
    """Decode one episode once, resize it, and atomically publish its cache."""
    episode_dir = episode_dir.resolve()
    actions_path = episode_dir / "actions.csv"
    video_path = episode_dir / "video.mp4"
    rows = read_actions(actions_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = cache_dir / f"{episode_dir.name}.npy"
    cached_actions_path = cache_dir / f"{episode_dir.name}.actions.npy"
    metadata_path = cache_dir / f"{episode_dir.name}.json"
    if output_path.exists() and cached_actions_path.exists() and metadata_path.exists() and not overwrite:
        return output_path

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    declared_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if declared_count != len(rows):
        capture.release()
        raise ValueError(
            f"{episode_dir.name}: video has {declared_count} frames but actions.csv has {len(rows)} rows. "
            "Refusing to create a misaligned dataset."
        )

    temporary_path = cache_dir / f".{episode_dir.name}.tmp.npy"
    temporary_actions_path = cache_dir / f".{episode_dir.name}.actions.tmp.npy"
    frames = np.lib.format.open_memmap(
        temporary_path, mode="w+", dtype=np.uint8,
        shape=(len(rows), TARGET_HEIGHT, TARGET_WIDTH, 3),
    )
    decoded = 0
    try:
        while decoded < len(rows):
            ok, frame_bgr = capture.read()
            if not ok:
                raise ValueError(f"{episode_dir.name}: video ended at frame {decoded}, expected {len(rows)}")
            frames[decoded] = preprocess_bgr_frame(frame_bgr)
            decoded += 1
        extra, _ = capture.read()
        if extra:
            raise ValueError(f"{episode_dir.name}: video contains extra frames beyond actions.csv")
    finally:
        capture.release()
        del frames

    os.replace(temporary_path, output_path)
    np.save(temporary_actions_path, action_array(rows))
    os.replace(temporary_actions_path, cached_actions_path)
    metadata: dict[str, Any] = {
        "episode": episode_dir.name,
        "source_video": str(video_path),
        "actions_csv": str(actions_path),
        "cached_actions": cached_actions_path.name,
        "frame_count": decoded,
        "source_size": [source_width, source_height],
        "source_fps": source_fps,
        "preprocessing": PREPROCESSING_CONFIG,
    }
    temporary_metadata = metadata_path.with_suffix(".tmp.json")
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_metadata, metadata_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache SHIONN recording frames at 320x180 RGB.")
    parser.add_argument("--recordings-dir", type=Path, default=Path("episodes"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/datasets/cached_frames"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    episodes = discover_episodes(args.recordings_dir)
    if not episodes:
        raise SystemExit(f"No episodes found in {args.recordings_dir}")
    for episode in episodes:
        result = cache_episode(episode, args.cache_dir, args.overwrite)
        print(f"cached {episode.name}: {result}")


if __name__ == "__main__":
    main()
