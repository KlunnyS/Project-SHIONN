import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from models.imitation.preprocess import ACTION_COLUMNS
from models.imitation.visual_targets import (
    extract_target,
    load_reviewed_episode,
    sha256,
)


class FakeCapture:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        frame = next(self.frames, None)
        return (frame is not None, frame)

    def release(self):
        pass


class VisualTargetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        (self.path / "video.mp4").write_bytes(b"fixture")
        (self.path / "metadata.json").write_text(
            json.dumps({"capture_valid": True, "training_approved": False})
        )
        with (self.path / "actions.csv").open("w") as f:
            w = csv.DictWriter(f, fieldnames=["frame_idx", *ACTION_COLUMNS])
            w.writeheader()
            for i in range(3):
                w.writerow({"frame_idx": i, **{k: 0 for k in ACTION_COLUMNS}})
        self.ep = {
            "frame_count": 3,
            "source_path": str(self.path),
            "split_role": "annotation_only",
            "training_approved": False,
            "episode_id": "fixture",
            "requested_goal_id": "fixture_A",
            "phases_half_open": [{"name": "initial_wait", "start": 0, "end": 1}],
            "provenance": {
                n + "_sha256": sha256(self.path / f)
                for n, f in [
                    ("video", "video.mp4"),
                    ("actions", "actions.csv"),
                    ("metadata", "metadata.json"),
                ]
            },
        }
        self.frames = [np.full((180, 320, 3), i, dtype=np.uint8) for i in range(3)]

    def load(self):
        with patch(
            "models.imitation.visual_targets.cv2.VideoCapture",
            return_value=FakeCapture(self.frames),
        ):
            return load_reviewed_episode(self.ep)

    def test_wait_is_masked_without_removing_history(self):
        data = self.load()
        self.assertEqual(data["frames"].shape, (3, 180, 320, 3))
        self.assertEqual(data["loss_mask"].tolist(), [False, True, True])

    def test_development_data_cannot_enter_training(self):
        with self.assertRaisesRegex(ValueError, "training-approved"):
            load_reviewed_episode(self.ep, for_training=True)

    def test_hash_tampering_rejected(self):
        (self.path / "video.mp4").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.load()

    def test_missing_frame_rejected(self):
        self.frames.pop()
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            self.load()

    def test_frame_budget_and_boundaries_rejected(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            load_reviewed_episode(self.ep, max_frames=2)
        self.ep["phases_half_open"][0]["end"] = 4
        with self.assertRaisesRegex(ValueError, "boundaries"):
            self.load()

    def test_crop_is_persistent_hashed_and_never_overwritten(self):
        output = self.path / "target.png"

        def capture(_):
            return FakeCapture(self.frames)

        with patch(
            "models.imitation.visual_targets.cv2.VideoCapture", side_effect=capture
        ):
            target = extract_target(self.ep, [10, 20, 30, 40], output)
            self.assertTrue(target["persistent"])
            self.assertFalse(target["training_approved"])
            self.assertEqual(target["crop_sha256"], sha256(output))
            with self.assertRaises(FileExistsError):
                extract_target(self.ep, [10, 20, 30, 40], output)
        with self.assertRaises(ValueError):
            extract_target(self.ep, [-1, 20, 30, 40], output)
