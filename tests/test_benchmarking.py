"""Regression tests for expert-frame and live-sequence benchmark reports."""

# Python standard library: fixtures, temporary files, and mocks.
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Installed dependencies used by fake inference policies.
import numpy as np
import torch

# Project benchmark and dataset modules under test.
from benchmark_expert import score_checkpoint, select_manifests
from benchmark_sequence import summarize_sequence
from models.imitation.dataset import ACTION_COLUMNS, EpisodeManifest
from models.imitation.inference import PolicyInference
from models.imitation.recurrent import RecurrentBinnedPolicy


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
    decode_outputs = PolicyInference.decode_outputs

    def __init__(self, checkpoint, device="auto", jump_threshold=None, move_w_threshold=None,
                 mouse_decode="argmax", mouse_move_threshold=0.5):
        self.device = torch.device("cpu")
        self.config = {"training": {"validation_episodes": ["episode_valid"]}}
        self.mouse_scale = np.ones(2, dtype=np.float32)
        self.mouse_bins = None
        self.jump_threshold = jump_threshold
        self.move_w_threshold = move_w_threshold
        self.mouse_decode = mouse_decode
        self.mouse_move_threshold = mouse_move_threshold
        self.policy = FakePolicy()


class FakeMouseBins:
    config = {"fake": True}

    def decode(self, axis, class_index):
        return (-5, 0, 5)[class_index]

    def representatives(self, axis):
        return [-5, 0, 5]


class FakeRecurrentPolicy(RecurrentBinnedPolicy):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.seen = []

    def step(self, frames, hidden=None):
        value = round(float(frames[0, 9, 0, 0]) * 255)
        self.seen.append((value, hidden is None))
        if hidden is None:
            hidden = torch.zeros((1, len(frames), 1), device=frames.device)
        hidden = hidden + frames[:, 9, 0, 0].view(1, -1, 1)
        score = hidden[0, :, 0]
        output = {
            name: torch.stack((-score, score), dim=1)
            for name in ACTION_COLUMNS[:8]
        }
        output["mouse_dx_logits"] = torch.stack((score, score * 0, -score), dim=1)
        output["mouse_dy_logits"] = output["mouse_dx_logits"]
        return output, hidden


class FakeRecurrentInference:
    predict_stacked_tensor = PolicyInference.predict_stacked_tensor
    decode_outputs = PolicyInference.decode_outputs
    reset = PolicyInference.reset

    def __init__(self, checkpoint, device="auto", jump_threshold=None, move_w_threshold=None,
                 mouse_decode="argmax", mouse_move_threshold=0.5):
        self.device = torch.device("cpu")
        self.config = {
            "architecture": "shionn_imitation_v5_gru_binned",
            "preprocessing": {"frame_stack": 4},
            "training": {"validation_episodes": ["episode_a", "episode_b"]},
        }
        self.mouse_scale = None
        self.mouse_bins = FakeMouseBins()
        self.jump_threshold = jump_threshold
        self.move_w_threshold = move_w_threshold
        self.mouse_decode = mouse_decode
        self.mouse_move_threshold = mouse_move_threshold
        self.frames = []
        self.hidden = None
        self.policy = FakeRecurrentPolicy()


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
            checkpoint = root / "model.pt"
            checkpoint.write_bytes(b"fake-checkpoint-identity")
            np.save(cache / "episode_valid.npy", np.zeros((3, 180, 320, 3), dtype=np.uint8))
            actions = np.zeros((3, len(ACTION_COLUMNS)), dtype=np.float32)
            actions[:2, 0] = 1
            np.save(cache / "episode_valid.actions.npy", actions)
            (cache / "episode_valid.json").write_text(json.dumps({
                "episode": "episode_valid", "map": "evaluation2", "frame_count": 3,
                "cached_actions": "episode_valid.actions.npy",
            }))

            report = score_checkpoint(
                checkpoint, cache, root / "output", batch_size=2, final_frames=1
            )

            self.assertEqual(report["episodes"], 1)
            self.assertEqual(
                report["checkpoint_sha256"],
                hashlib.sha256(b"fake-checkpoint-identity").hexdigest(),
            )
            action = report["by_map"]["evaluation2"]["all"]["actions"]["move_w"]
            self.assertEqual((action["true_positive"], action["false_positive"]), (2, 1))
            self.assertEqual(report["by_map"]["evaluation2"]["final_frames"]["frames"], 1)
            with (root / "output" / "action_comparison.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[-1]["expected_move_w"], "0")
            self.assertEqual(rows[-1]["predicted_move_w"], "1")

    @patch("benchmark_expert.PolicyInference", FakeInference)
    def test_feedforward_benchmark_uses_live_action_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache = root / "cache"
            cache.mkdir()
            np.save(cache / "episode_valid.npy", np.zeros((2, 180, 320, 3), dtype=np.uint8))
            actions = np.zeros((2, len(ACTION_COLUMNS)), dtype=np.float32)
            actions[:, 0] = 1
            np.save(cache / "episode_valid.actions.npy", actions)
            (cache / "episode_valid.json").write_text(json.dumps({
                "episode": "episode_valid", "map": "evaluation2", "frame_count": 2,
                "cached_actions": "episode_valid.actions.npy",
            }))

            report = score_checkpoint(
                root / "model.pt", cache, root / "output", subset="all",
                final_frames=1, move_w_threshold=0.99,
            )

            action = report["by_map"]["evaluation2"]["all"]["actions"]["move_w"]
            self.assertEqual(action["false_negative"], 2)
            self.assertEqual(report["action_decode"]["move_w_threshold"], 0.99)

    def test_recurrent_scoring_is_sequential_and_resets_hidden_per_episode(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cache = root / "cache"
            cache.mkdir()
            for name, map_name, pixel in (
                ("episode_a", "evaluation2", 10),
                ("episode_b", "evaluation3", 40),
            ):
                frames = np.full((3, 180, 320, 3), pixel, dtype=np.uint8)
                frames[1] += 1
                frames[2] += 2
                np.save(cache / f"{name}.npy", frames)
                actions = np.zeros((3, len(ACTION_COLUMNS)), dtype=np.float32)
                actions[1:, 0] = 1
                np.save(cache / f"{name}.actions.npy", actions)
                (cache / f"{name}.json").write_text(json.dumps({
                    "episode": name, "map": map_name, "frame_count": 3,
                    "cached_actions": f"{name}.actions.npy",
                }))

            instances = []

            def make_inference(*args, **kwargs):
                instance = FakeRecurrentInference(*args, **kwargs)
                instances.append(instance)
                return instance

            with patch("benchmark_expert.PolicyInference", side_effect=make_inference):
                report = score_checkpoint(
                    root / "model.pt", cache, root / "output", final_frames=1,
                    mouse_decode="expected",
                )

            self.assertEqual(report["episodes"], 2)
            self.assertEqual(report["overall"]["all"]["frames"], 4)
            self.assertEqual(report["architecture"], "shionn_imitation_v5_gru_binned")
            self.assertEqual(report["scoring_mode"], "episode_sequential_step")
            self.assertEqual(report["action_decode"]["mouse"], "expected")
            self.assertEqual(
                instances[0].policy.seen,
                [(10, True), (11, False), (12, False),
                 (40, True), (41, False), (42, False)],
            )
            with (root / "output" / "action_comparison.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [(row["episode"], row["frame_idx"]) for row in rows],
                [("episode_a", "1"), ("episode_a", "2"),
                 ("episode_b", "1"), ("episode_b", "2")],
            )


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
