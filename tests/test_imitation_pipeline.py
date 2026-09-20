import csv
import json
import subprocess
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import cv2
import numpy as np
import torch

from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import BehaviorCloningDataset, discover_cached_episodes, split_episode_manifests
from models.imitation.inference import PolicyInference
from models.imitation.network import ImitationPolicy
from models.imitation.preprocess import PREPROCESSING_CONFIG
from models.imitation.preprocess import ACTION_COLUMNS, cache_episode, discover_episodes
from models.imitation.train_bc import EarlyStopping, format_duration, make_binary_class_weights, make_training_loader
from recorder import FFmpegVideoWriter
from run_imitation import AttemptLog, EscapeKeyMonitor, action_label, apply_predicted_action, classify_terminal_events, frame_change, hyprland_instance_candidates, make_attempt_recording_path, parse_args


class ImitationPipelineTest(unittest.TestCase):
    def test_early_stopping_preserves_best_loss_and_counts_unimproved_epochs(self):
        stopping = EarlyStopping(patience=3)
        self.assertEqual(stopping.update(3.0), (True, False))
        self.assertEqual(stopping.update(2.4), (True, False))
        self.assertEqual(stopping.update(2.5), (False, False))
        self.assertEqual(stopping.update(2.4), (False, False))
        self.assertEqual(stopping.update(2.6), (False, True))
        self.assertEqual(stopping.best_loss, 2.4)
        self.assertEqual(stopping.update(2.3), (True, False))
        self.assertEqual(stopping.unimproved_epochs, 0)

    def test_early_stopping_can_be_disabled_and_reset_for_resume(self):
        stopping = EarlyStopping(patience=0, best_loss=2.4)
        self.assertEqual(stopping.update(3.0), (False, False))
        self.assertEqual(stopping.update(2.3), (True, False))

    def test_training_duration_readout_is_human_readable(self):
        self.assertEqual(format_duration(7.25), "7.2s")
        self.assertEqual(format_duration(65.5), "1m 05.5s")
        self.assertEqual(format_duration(3723.5), "1h 02m 03.5s")

    def test_configured_runner_timeout_ignores_chamber_timeout_failure(self):
        terminal, ignored_timeout = classify_terminal_events(
            "EVT|episode_failed|timeout\n", max_seconds=60.0
        )

        self.assertFalse(terminal)
        self.assertTrue(ignored_timeout)

    def test_unlimited_run_and_other_failures_remain_terminal(self):
        self.assertEqual(
            classify_terminal_events(
                "EVT|episode_failed|timeout\n", max_seconds=0.0
            ),
            (True, False),
        )
        self.assertEqual(
            classify_terminal_events(
                "EVT|episode_failed|out_of_bounds\n", max_seconds=60.0
            ),
            (True, False),
        )
        self.assertEqual(
            classify_terminal_events("EVT|goal_reached|1\n", max_seconds=60.0),
            (True, False),
        )

    @patch("run_imitation.evdev.InputDevice")
    @patch("run_imitation.evdev.list_devices")
    def test_escape_monitor_detects_global_key_press(self, list_devices, input_device):
        import evdev

        keyboard = Mock()
        keyboard.name = "Physical Keyboard"
        keyboard.path = "/dev/input/event-test"
        keyboard.capabilities.return_value = {
            evdev.ecodes.EV_KEY: [
                evdev.ecodes.KEY_ESC,
                evdev.ecodes.KEY_W,
                evdev.ecodes.KEY_A,
            ]
        }
        keyboard.read_loop.return_value = iter([
            SimpleNamespace(
                type=evdev.ecodes.EV_KEY,
                code=evdev.ecodes.KEY_ESC,
                value=1,
            )
        ])
        list_devices.return_value = [keyboard.path]
        input_device.return_value = keyboard
        monitor = EscapeKeyMonitor()

        monitor.start()
        try:
            self.assertTrue(monitor.wait(1.0))
            self.assertTrue(monitor.stop_requested)
        finally:
            monitor.stop()

        keyboard.close.assert_called()

    def test_attempt_recordings_use_a_separate_timestamped_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            recording_dir = Path(temporary_directory) / "model_attempts"

            output = make_attempt_recording_path(
                recording_dir, datetime(2026, 9, 16, 12, 34, 56, 123456)
            )

            self.assertTrue(recording_dir.is_dir())
            self.assertEqual(
                output,
                recording_dir / "attempt_20260916_123456_123456.mp4",
            )

    def test_runner_recording_options_default_to_model_attempts(self):
        args = parse_args(["--record-video", "--verbose", "--keep-focused"])

        self.assertTrue(args.record_video)
        self.assertTrue(args.verbose)
        self.assertTrue(args.keep_focused)
        self.assertEqual(args.recording_dir, Path("model_attempts"))
        self.assertEqual(args.hyprland_instance, "auto")

    def test_explicit_hyprland_instance_does_not_depend_on_session_environment(self):
        self.assertEqual(hyprland_instance_candidates("test-signature"), ["test-signature"])

    def test_attempt_log_writes_line_delimited_json(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "attempt.jsonl"
            log = AttemptLog(path)
            log.write("tick", tick=1, action={"move_w": 1})
            log.write("summary", ticks=1)
            log.close()

            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([record["type"] for record in records], ["tick", "summary"])
            self.assertEqual(records[0]["action"], {"move_w": 1})

    def test_frame_change_and_action_label_are_compact(self):
        first = np.zeros((32, 32, 3), dtype=np.uint8)
        second = np.full((32, 32, 3), 16, dtype=np.uint8)
        change, sample = frame_change(None, first)
        self.assertIsNone(change)
        change, _ = frame_change(sample, second)
        self.assertEqual(change, 16.0)
        self.assertEqual(action_label({"move_w": 1, "jump": 1}), "move_w+jump")

    def test_verbose_dry_run_never_applies_an_action(self):
        controller = Mock()

        applied = apply_predicted_action(
            controller, {"move_w": 1}, dry_run=True, verbose=True
        )

        self.assertFalse(applied)
        controller.apply_action.assert_not_called()

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
            (episode / "metadata.json").write_text('{"map": "dataset_test5"}\n')
            with (episode / "actions.csv").open("w", newline="", encoding="utf-8") as handle:
                output = csv.DictWriter(handle, fieldnames=["frame_idx", *ACTION_COLUMNS])
                output.writeheader()
                for index in range(5):
                    output.writerow({"frame_idx": index, **{name: index if name.startswith("mouse") else index % 2 for name in ACTION_COLUMNS}})
            cache_episode(episode, root / "cache")
            cache_metadata_path = root / "cache" / "one.json"
            cache_metadata = json.loads(cache_metadata_path.read_text())
            self.assertEqual(cache_metadata["map"], "dataset_test5")
            del cache_metadata["map"]
            cache_metadata_path.write_text(json.dumps(cache_metadata))
            with patch("models.imitation.preprocess.cv2.VideoCapture", side_effect=AssertionError("video was decoded again")):
                cache_episode(episode, root / "cache")
            self.assertEqual(json.loads(cache_metadata_path.read_text())["map"], "dataset_test5")
            manifests = discover_cached_episodes(root / "cache")
            self.assertEqual(manifests[0].actions_path.parent, root / "cache")
            self.assertEqual(manifests[0].map_name, "dataset_test5")
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

    def test_existing_cache_reads_map_from_recording_metadata(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "episodes" / "episode_old"
            source.mkdir(parents=True)
            (source / "metadata.json").write_text('{"map": "dataset_test1"}\n')
            cache = root / "cache"
            cache.mkdir()
            (cache / "episode_old.npy").touch()
            (cache / "episode_old.actions.npy").touch()
            (cache / "episode_old.json").write_text(json.dumps({
                "episode": "episode_old", "frame_count": 2,
                "actions_csv": str(source / "actions.csv"),
                "cached_actions": "episode_old.actions.npy",
            }))

            manifests = discover_cached_episodes(cache)

            self.assertEqual(manifests[0].map_name, "dataset_test1")

    def test_balanced_sampling_equalizes_usable_frames_by_map(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            from models.imitation.dataset import EpisodeManifest
            manifests = []
            for name, count in (("short", 2), ("long", 6)):
                frames_path = root / f"{name}.npy"
                actions_path = root / f"{name}.actions.npy"
                np.save(frames_path, np.zeros((count, 180, 320, 3), dtype=np.uint8))
                actions = np.zeros((count, len(ACTION_COLUMNS)), dtype=np.float32)
                actions[:, 0 if name == "short" else 8] = 1
                np.save(actions_path, actions)
                manifests.append(EpisodeManifest(name, frames_path, actions_path, count, name))
            dataset = BehaviorCloningDataset(manifests)

            weights = dataset.chamber_sampling_weights()
            self.assertEqual(dataset.chamber_sample_counts(), {"short": 2, "long": 6})
            self.assertAlmostEqual(weights[:2].sum(), 4.0)
            self.assertAlmostEqual(weights[2:].sum(), 4.0)
            self.assertEqual(dataset.action_statistics()[0][0], 2)
            self.assertAlmostEqual(dataset.action_statistics(balance_chambers=True)[0][0], 4.0)
            args = SimpleNamespace(sampling="chamber-balanced", seed=4, batch_size=2, workers=0)
            loader = make_training_loader(dataset, args, False, epoch=1)
            self.assertEqual(next(iter(loader))[0].shape, (2, 12, 180, 320))
            first = list(make_training_loader(dataset, args, False, epoch=1).sampler)
            second = list(make_training_loader(dataset, args, False, epoch=1).sampler)
            self.assertEqual(first, second)
            self.assertEqual(len(first), len(dataset))

    def test_episode_split_stratifies_chambers(self):
        from models.imitation.dataset import EpisodeManifest
        manifests = [
            EpisodeManifest(f"{name}_{index}", Path("unused"), Path("unused"), 1, name)
            for name in ("map_a", "map_b") for index in range(5)
        ]
        train, validation = split_episode_manifests(manifests, 0.2, seed=3)

        self.assertEqual({name: sum(item.map_name == name for item in validation) for name in ("map_a", "map_b")}, {"map_a": 1, "map_b": 1})
        self.assertFalse({item.name for item in train} & {item.name for item in validation})

        legacy_train, legacy_validation = split_episode_manifests(
            manifests, 0.2, seed=3, stratify=False
        )
        self.assertEqual(len(legacy_train), 8)
        self.assertEqual(len(legacy_validation), 2)

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
            action, diagnostics = policy.predict_with_diagnostics(
                np.zeros((1080, 1920, 3), dtype=np.uint8)
            )
            self.assertEqual(set(action), set(config["action_columns"]))
            self.assertEqual(action["mouse_dx"], 50)
            self.assertEqual(action["mouse_dy"], -10)
            self.assertEqual(set(diagnostics["binary_probabilities"]), set(config["action_columns"][:8]))
            self.assertEqual(diagnostics["mouse_mean"], [50.0, -10.0])
            self.assertEqual(diagnostics["history_frames"], 1)


if __name__ == "__main__":
    unittest.main()
