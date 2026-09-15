import csv
import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import BehaviorCloningDataset, discover_cached_episodes, split_episode_manifests
from models.imitation.inference import PolicyInference
from models.imitation.network import ImitationPolicy
from models.imitation.preprocess import PREPROCESSING_CONFIG
from models.imitation.preprocess import ACTION_COLUMNS, cache_episode, discover_episodes
from models.imitation.train_bc import make_binary_class_weights
from recorder import FFmpegVideoWriter


class ImitationPipelineTest(unittest.TestCase):
    def test_discovers_flat_and_outcome_grouped_episodes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "episodes"
            flat = root / "episode_flat"
            success = root / "goal_reached" / "episode_success"
            incomplete = root / ".in_progress" / "episode_partial"
            for episode in (flat, success, incomplete):
                episode.mkdir(parents=True)
                (episode / "actions.csv").touch()
                (episode / "video.mp4").touch()
            self.assertEqual(discover_episodes(root), [flat, success])

    def test_cache_stack_alignment_and_leading_idle_trim(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            episode = root / "episodes" / "one"
            episode.mkdir(parents=True)
            writer = cv2.VideoWriter(str(episode / "video.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 24, (32, 18))
            self.assertTrue(writer.isOpened())
            for value in range(5):
                writer.write(np.full((18, 32, 3), value * 40, dtype=np.uint8))
            writer.release()
            with (episode / "actions.csv").open("w", newline="", encoding="utf-8") as handle:
                output = csv.DictWriter(handle, fieldnames=["frame_idx", *ACTION_COLUMNS])
                output.writeheader()
                for index in range(5):
                    output.writerow({"frame_idx": index, **{name: index if name.startswith("mouse") else index % 2 for name in ACTION_COLUMNS}})
            cache_episode(episode, root / "cache")
            manifests = discover_cached_episodes(root / "cache")
            self.assertEqual(manifests[0].actions_path.parent, root / "cache")
            dataset = BehaviorCloningDataset(manifests)
            frames, action = dataset[0]
            self.assertEqual(frames.shape, (12, 180, 320))
            self.assertEqual(action.shape, (10,))
            self.assertEqual(action[8], 1)
            self.assertEqual(dataset.leading_idle_frames, 1)
            self.assertTrue(np.array_equal(frames[0:3], frames[3:6]))
            self.assertFalse(np.array_equal(frames[0:3], frames[9:12]))

    def test_action_statistics_ignore_trimmed_waiting_frames(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames_path = root / "episode.npy"
            actions_path = root / "episode.actions.npy"
            np.save(frames_path, np.zeros((4, 180, 320, 3), dtype=np.uint8))
            actions = np.zeros((4, len(ACTION_COLUMNS)), dtype=np.float32)
            actions[2:, 0] = 1
            actions[2, 8:10] = [10, -4]
            np.save(actions_path, actions)
            from models.imitation.dataset import EpisodeManifest
            dataset = BehaviorCloningDataset(
                [EpisodeManifest("episode", frames_path, actions_path, 4)]
            )

            positive_counts, mouse_scale = dataset.action_statistics()

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset.leading_idle_frames, 2)
            self.assertEqual(positive_counts[0], 2)
            self.assertTrue(np.all(mouse_scale >= 1))

    def test_class_weights_balance_supported_actions_without_amplifying_noise(self):
        counts = np.asarray([80, 20, 1, 25, 5, 0, 0, 0], dtype=np.float64)

        weights = make_binary_class_weights(counts, sample_count=100)

        self.assertAlmostEqual(weights[0, 0].item(), 2.5)
        self.assertAlmostEqual(weights[1, 1].item(), 2.5)
        self.assertTrue(torch.equal(weights[2], torch.ones(2)))
        self.assertTrue(torch.equal(weights[4], torch.ones(2)))

    def test_split_keeps_full_episodes_together(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            manifests = []
            for name in ("one", "two", "three"):
                frames = root / f"{name}.npy"
                np.save(frames, np.zeros((1, 180, 320, 3), dtype=np.uint8))
                actions = root / f"{name}.csv"
                actions.write_text("frame_idx,move_w,move_a,move_s,move_d,jump,use,fire_left,fire_right,mouse_dx,mouse_dy\n0,0,0,0,0,0,0,0,0,0,0\n")
                from models.imitation.dataset import EpisodeManifest
                manifests.append(EpisodeManifest(name, frames, actions, 1))
            train, validation = split_episode_manifests(manifests, validation_fraction=0.34, seed=7)
            self.assertEqual(len(train) + len(validation), 3)
            self.assertFalse({item.name for item in train}.intersection(item.name for item in validation))

    def test_ffmpeg_writer_produces_h264_mp4(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "recording.mp4"
            writer = FFmpegVideoWriter(str(output), 64, 64, 24, crf=20)
            for value in range(3):
                writer.write(np.full((64, 64, 3), value * 50, dtype=np.uint8))
            writer.release()
            codec = subprocess.check_output(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,width,height,avg_frame_rate", "-of", "default=noprint_wrappers=1", str(output)],
                text=True,
            )
            self.assertIn("codec_name=h264", codec)
            self.assertIn("width=64", codec)
            self.assertIn("height=64", codec)
            self.assertIn("avg_frame_rate=24/1", codec)
            stream_types = subprocess.check_output(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "stream=codec_type",
                    "-of", "csv=p=0", str(output),
                ],
                text=True,
            ).splitlines()
            self.assertEqual(stream_types, ["video"])

    def test_checkpoint_resumes_optimizer_and_inference_uses_saved_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "model.pt"
            model = ImitationPolicy()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            config = {
                "architecture": "shionn_imitation_v3",
                "preprocessing": PREPROCESSING_CONFIG,
                "action_columns": ["move_w", "move_a", "move_s", "move_d", "jump", "use", "fire_left", "fire_right", "mouse_dx", "mouse_dy"],
                "target_processing": {"mouse_scale": [100.0, 20.0]},
            }
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.zero_()
                model.mouse_head.bias[:2] = torch.tensor([0.5, -0.5])
            save_checkpoint(checkpoint_path, model=model, optimizer=optimizer, epoch=3, global_step=42, best_val_loss=1.25, config=config)
            restored_model = ImitationPolicy()
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
            restored = load_checkpoint(checkpoint_path, model=restored_model, optimizer=restored_optimizer)
            self.assertEqual(restored["global_step"], 42)
            self.assertEqual(restored["epoch"], 3)
            self.assertTrue(checkpoint_path.with_suffix(".json").is_file())
            policy = PolicyInference(checkpoint_path, device="cpu")
            action = policy.predict(np.zeros((1080, 1920, 3), dtype=np.uint8))
            self.assertEqual(set(action), set(config["action_columns"]))
            self.assertEqual(action["mouse_dx"], 50)
            self.assertEqual(action["mouse_dy"], -10)


if __name__ == "__main__":
    unittest.main()
