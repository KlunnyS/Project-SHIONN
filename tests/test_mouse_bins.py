import tempfile
import unittest
import json
from pathlib import Path

import numpy as np
import torch

from models.imitation.checkpoint import save_checkpoint
from models.imitation.inference import PolicyInference
from models.imitation.mouse_bins import MouseBins
from models.imitation.network import BINNED_ARCHITECTURE_VERSION, BinnedImitationPolicy
from models.imitation.preprocess import PREPROCESSING_CONFIG
from models.imitation.train_bc import behavior_cloning_loss, with_jump_positive_weight
from benchmark_expert import score_checkpoint
from analyze_mouse_validation import analyze_validation


class MouseBinsTest(unittest.TestCase):
    def test_fit_keeps_zero_and_opposite_turns_in_distinct_classes(self):
        values = np.array([
            [-100, -10], [-25, -3], [-5, -1], [0, 0],
            [0, 0], [5, 1], [25, 3], [100, 10],
        ], dtype=np.float32)
        bins = MouseBins.fit(values, bins_per_axis=7)
        classes_x, classes_y = bins.encode(torch.from_numpy(values))
        centers = [count // 2 for count in bins.class_counts()]

        self.assertEqual(classes_x[3].item(), centers[0])
        self.assertEqual(classes_y[3].item(), centers[1])
        self.assertLess(classes_x[0].item(), centers[0])
        self.assertGreater(classes_x[-1].item(), centers[0])
        self.assertLess(bins.decode(0, 0), 0)
        self.assertGreater(bins.decode(0, bins.class_counts()[0] - 1), 0)

    def test_binned_loss_and_jump_weight_override(self):
        bins = MouseBins.fit(np.array([[-10, -3], [0, 0], [10, 3]], dtype=np.float32), 5)
        dx_count, dy_count = bins.class_counts()
        prediction = {
            name: torch.zeros((2, 2), requires_grad=True)
            for name in ("move_w", "move_a", "move_s", "move_d", "jump", "use", "fire_left", "fire_right")
        }
        prediction["mouse_dx_logits"] = torch.zeros((2, dx_count), requires_grad=True)
        prediction["mouse_dy_logits"] = torch.zeros((2, dy_count), requires_grad=True)
        targets = torch.zeros((2, 10))
        targets[1, 8:10] = torch.tensor([10, -3])

        loss, parts = behavior_cloning_loss(prediction, targets, mouse_bins=bins)
        loss.backward()

        self.assertGreater(parts["mouse"].item(), 0)
        self.assertIsNotNone(prediction["mouse_dx_logits"].grad)
        weights = with_jump_positive_weight(torch.ones((8, 2)), 3)
        self.assertEqual(weights[4].tolist(), [1.0, 3.0])

    def test_binned_checkpoint_decodes_argmax_without_sampling(self):
        bins = MouseBins.fit(np.array([[-10, -3], [0, 0], [10, 3]], dtype=np.float32), 5)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "binned.pt"
            model = BinnedImitationPolicy(*bins.class_counts())
            optimizer = torch.optim.AdamW(model.parameters())
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.mouse_dx_head.bias[-1] = 5
                model.mouse_dy_head.bias[0] = 5
            config = {
                "architecture": BINNED_ARCHITECTURE_VERSION,
                "preprocessing": PREPROCESSING_CONFIG,
                "target_processing": {"mouse_bins": bins.config},
            }
            save_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer,
                epoch=1, global_step=1, best_val_loss=1.0, config=config,
            )

            policy = PolicyInference(checkpoint_path, device="cpu")
            action, diagnostics = policy.predict_with_diagnostics(
                np.zeros((1080, 1920, 3), dtype=np.uint8)
            )

            self.assertEqual(action["mouse_dx"], bins.decode(0, bins.class_counts()[0] - 1))
            self.assertEqual(action["mouse_dy"], bins.decode(1, 0))
            self.assertIn("mouse_bin_probabilities", diagnostics)
            self.assertNotIn("mouse_mean", diagnostics)
            calibrated = PolicyInference(checkpoint_path, device="cpu", jump_threshold=0.4)
            self.assertEqual(calibrated.predict(np.zeros((1080, 1920, 3), dtype=np.uint8))["jump"], 1)
            forward = PolicyInference(checkpoint_path, device="cpu", move_w_threshold=0.4)
            self.assertEqual(forward.predict(np.zeros((1080, 1920, 3), dtype=np.uint8))["move_w"], 1)
            with self.assertRaises(ValueError):
                PolicyInference(checkpoint_path, device="cpu", jump_threshold=1.0)
            with self.assertRaises(ValueError):
                PolicyInference(checkpoint_path, device="cpu", move_w_threshold=0.0)

            cache_dir = Path(temporary_directory) / "cache"
            cache_dir.mkdir()
            np.save(cache_dir / "episode.npy", np.zeros((4, 180, 320, 3), dtype=np.uint8))
            actions = np.zeros((4, 10), dtype=np.float32)
            actions[:, 0] = 1
            np.save(cache_dir / "episode.actions.npy", actions)
            (cache_dir / "episode.json").write_text(json.dumps({
                "episode": "episode", "map": "dataset_test9", "frame_count": 4,
                "cached_actions": "episode.actions.npy",
            }))

            report = score_checkpoint(
                checkpoint_path, cache_dir, Path(temporary_directory) / "report",
                subset="all", device="cpu", batch_size=2,
            )
            self.assertEqual(report["overall"]["all"]["frames"], 4)
            ranges = Path(temporary_directory) / "wall_ranges.csv"
            ranges.write_text("episode,start_frame,end_frame\nepisode,0,2\n")
            wall_report, wall_rows = analyze_validation(
                Path(report["action_rows"]),
                Path(temporary_directory) / "report" / "summary.json",
                ranges,
            )
            self.assertEqual(wall_report["wall_frames"], 3)
            self.assertEqual(wall_report["other_frames"], 1)
            self.assertEqual(len(wall_rows), 3)


if __name__ == "__main__":
    unittest.main()
