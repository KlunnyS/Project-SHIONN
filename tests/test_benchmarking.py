import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from benchmark_expert import score_checkpoint, select_manifests
from benchmark_sequence import summarize_sequence
from models.imitation.dataset import ACTION_COLUMNS, EpisodeManifest


class FakePolicy(torch.nn.Module):
    def forward(self, frames):
        batch_size = len(frames)
        output = {
            name: torch.zeros((batch_size, 2), device=frames.device)
            for name in ACTION_COLUMNS[:8]
        }
        output["move_w"][:, 1] = 2.0
        output["mouse_mean"] = torch.zeros((batch_size, 2), device=frames.device)
        return output


class FakeInference:
    def __init__(self, checkpoint, device="auto"):
        self.device = torch.device("cpu")
        self.config = {"training": {"validation_episodes": ["episode_valid"]}}
        self.mouse_scale = np.ones(2, dtype=np.float32)
        self.policy = FakePolicy()


class ExpertBenchmarkTest(unittest.TestCase):
    def test_validation_selection_never_includes_training_episodes(self):
        manifests = [
            EpisodeManifest("episode_train", Path("a"), Path("b"), 3, "map_a"),
            EpisodeManifest("episode_valid", Path("c"), Path("d"), 3, "map_a"),
        ]
        config = {"training": {"validation_episodes": ["episode_valid"]}}

        selected = select_manifests(manifests, config, "validation")

        self.assertEqual([item.name for item in selected], ["episode_valid"])
        with self.assertRaisesRegex(ValueError, "no saved validation"):
            select_manifests(manifests, {}, "validation")

    @patch("benchmark_expert.PolicyInference", FakeInference)
    def test_scores_expected_and_predicted_actions_and_final_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "episode_valid.npy", np.zeros((3, 180, 320, 3), dtype=np.uint8))
            actions = np.zeros((3, len(ACTION_COLUMNS)), dtype=np.float32)
            actions[:2, 0] = 1
            np.save(cache / "episode_valid.actions.npy", actions)
            (cache / "episode_valid.json").write_text(json.dumps({
                "episode": "episode_valid", "map": "evaluation2", "frame_count": 3,
                "cached_actions": "episode_valid.actions.npy",
            }))

            report = score_checkpoint(
                root / "model.pt", cache, root / "output", batch_size=2, final_frames=1
            )

            self.assertEqual(report["episodes"], 1)
            action = report["by_map"]["evaluation2"]["all"]["actions"]["move_w"]
            self.assertEqual((action["true_positive"], action["false_positive"]), (2, 1))
            self.assertEqual(report["by_map"]["evaluation2"]["final_frames"]["frames"], 1)
            with (root / "output" / "action_comparison.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["expected_move_w"], "0")
            self.assertEqual(rows[-1]["predicted_move_w"], "1")


class SequenceBenchmarkTest(unittest.TestCase):
    def test_reports_success_rate_duration_and_invalid_attempts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            entries = [{"type": "sequence", "jobs": 3}]
            for index, reason in enumerate(("goal_reached", "time_limit", "runner_error"), start=1):
                log = root / f"attempt_{index}.jsonl"
                log.write_text("".join(json.dumps(item) + "\n" for item in (
                    {"type": "metadata", "capture": {"fps": 24}, "dry_run": False},
                    {"type": "summary", "ticks": 240, "focus_losses": 0, "stop_reason": reason},
                )))
                entries.append({
                    "type": "result", "job": index, "model": "candidate",
                    "map": "evaluation2", "repeat": index, "stop_reason": reason,
                    "returncode": 1 if reason == "runner_error" else 0,
                    "log": str(log), "video": None,
                })
            (root / "sequence.jsonl").write_text(
                "".join(json.dumps(item) + "\n" for item in entries)
            )

            report = summarize_sequence(root)

            row = report["summary"][0]
            self.assertEqual(row["valid_attempts"], 2)
            self.assertEqual(row["invalid_attempts"], 1)
            self.assertEqual(row["successes"], 1)
            self.assertEqual(row["success_rate"], 0.5)
            self.assertEqual(row["median_success_seconds"], 10.0)


if __name__ == "__main__":
    unittest.main()
