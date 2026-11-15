"""Fault-injection coverage for recording validity and goal labels."""

import io
import json
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn

from models.imitation.goals import (
    GOAL_ANNOTATION_SCHEMA,
    GOAL_COORDINATE_SYSTEM,
    GoalCondition,
    validate_goal_annotation_manifest,
)
from models.imitation.inference import PolicyInference
from models.imitation.preprocess import discover_episodes
from models.imitation.recurrent import (
    GoalConditionedRecurrentPolicy,
    RecurrentBinnedPolicy,
    initialize_goal_policy_from_recurrent,
)
from policy_decision_rate import DecisionRatePolicy
from recorder import EpisodeRecorder, FrameSample, InputTracker, WaylandCamera


class _StoppedThread:
    def join(self, timeout=None):
        pass

    def is_alive(self):
        return False


class RecordingFailClosedTests(unittest.TestCase):
    def test_input_read_failure_is_recorded_with_device_identity(self):
        keyboard = SimpleNamespace(
            path="/dev/input/event17",
            name="Test Keyboard",
            read_loop=lambda: (_ for _ in ()).throw(OSError("device removed")),
        )
        tracker = InputTracker([keyboard], [])
        tracker._kbd_loop(keyboard)
        self.assertIn("/dev/input/event17", tracker.capture_errors()[0])
        self.assertIn("device removed", tracker.capture_errors()[0])

    def test_camera_partial_frame_eof_is_a_terminal_capture_error(self):
        camera = WaylandCamera(2, 1)
        camera.proc = SimpleNamespace(stdout=io.BytesIO(b"short"))
        camera.running = True
        camera._update()
        self.assertIn("partial frame", camera.capture_error)
        self.assertIsNone(camera.get_latest_frame_sample().frame_id)

    def test_camera_pipe_fragments_are_reassembled(self):
        camera = WaylandCamera(2, 1)

        class FragmentedPipe:
            def __init__(self):
                self.chunks = iter((b"123", b"456", b""))

            def read(self, _size):
                return next(self.chunks)

        camera.proc = SimpleNamespace(stdout=FragmentedPipe())
        camera.running = True
        camera._update()
        self.assertEqual(camera.get_latest_frame_sample().frame_id, 1)

    def test_stale_frame_invalidates_tick_without_writing_black_frame(self):
        recorder = self._sync_recorder(
            FrameSample(np.ones((1, 1, 3), dtype=np.uint8), 1, 1.0)
        )
        with patch("recorder.time.monotonic", return_value=10.0):
            recorder._sync_loop()
        self.assertIn("stale", recorder.capture_failure_reason)
        recorder.video_writer.write.assert_not_called()
        recorder.csv_writer.writerow.assert_not_called()

    def test_missing_frame_invalidates_tick_without_synthetic_observation(self):
        recorder = self._sync_recorder(FrameSample(None, None, None))
        with patch("recorder.time.monotonic", return_value=10.0):
            recorder._sync_loop()
        self.assertIn("complete frame", recorder.capture_failure_reason)
        recorder.video_writer.write.assert_not_called()
        recorder.csv_writer.writerow.assert_not_called()

    def test_required_input_device_error_stops_recording_tick(self):
        recorder = self._sync_recorder(
            FrameSample(np.ones((1, 1, 3), dtype=np.uint8), 1, 10.0)
        )
        recorder.tracker.capture_errors = lambda: (
            "keyboard /dev/input/event17 disconnected",
        )
        with patch("recorder.time.monotonic", return_value=10.0):
            recorder._sync_loop()
        self.assertIn("keyboard", recorder.capture_failure_reason)
        recorder.video_writer.write.assert_not_called()

    def test_writer_failure_becomes_explicit_capture_failure(self):
        recorder = self._sync_recorder(
            FrameSample(np.ones((1, 1, 3), dtype=np.uint8), 1, 10.0)
        )
        recorder.video_writer.write.side_effect = OSError("writer pipe closed")
        with patch("recorder.time.monotonic", return_value=10.0):
            recorder._sync_loop()
        self.assertIn("synchronization failed", recorder.capture_failure_reason)
        self.assertIn("writer pipe closed", recorder.capture_failure_reason)

    def test_frame_id_gap_invalidates_recording(self):
        samples = deque(
            (
                FrameSample(np.ones((1, 1, 3), dtype=np.uint8), 1, 10.0),
                FrameSample(np.ones((1, 1, 3), dtype=np.uint8), 3, 10.0),
            )
        )
        recorder = self._sync_recorder(None)
        recorder.camera.get_latest_frame_sample.side_effect = lambda: samples.popleft()
        with (
            patch("recorder.time.monotonic", return_value=10.0),
            patch("recorder.time.sleep", return_value=None),
        ):
            recorder._sync_loop()
        self.assertIn("frame gap", recorder.capture_failure_reason)
        self.assertEqual(recorder.video_writer.write.call_count, 1)

    @staticmethod
    def _sync_recorder(sample):
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.fps = 24
        recorder.recording = True
        recorder.sync_lock = threading.Lock()
        recorder._capture_failure_reason = None
        recorder.paused = False
        recorder.resume_deadline = None
        recorder.frame_idx = 0
        recorder.input_discard_count = 0
        recorder.tracker = SimpleNamespace(
            capture_errors=lambda: (),
            pause_is_pending=lambda: False,
            get_snapshot_and_reset=lambda: {
                key: 0
                for key in (
                    "move_w",
                    "move_a",
                    "move_s",
                    "move_d",
                    "jump",
                    "crouch",
                    "use",
                    "fire_left",
                    "fire_right",
                    "mouse_dx",
                    "mouse_dy",
                )
            },
        )
        recorder.camera = SimpleNamespace(
            capture_error=None,
            get_latest_frame_sample=Mock(return_value=sample),
        )
        recorder.video_writer = Mock()
        recorder.csv_writer = Mock()
        recorder.csv_file = Mock()
        return recorder

    def test_invalid_recording_is_retained_but_not_discovered_for_training(self):
        with tempfile.TemporaryDirectory() as root:
            progress = Path(root) / ".in_progress" / "episode_test"
            progress.mkdir(parents=True)
            (progress / "metadata.json").write_text('{"map":"dataset_test1"}\n')
            (progress / "actions.csv").touch()
            (progress / "video.mp4").touch()
            recorder = EpisodeRecorder.__new__(EpisodeRecorder)
            recorder.recording, recorder.paused = True, False
            recorder.resume_deadline = None
            recorder.sync_thread = _StoppedThread()
            recorder.video_writer, recorder.csv_file = Mock(), Mock()
            recorder.episodes_root, recorder.episode_name = root, "episode_test"
            recorder.ep_dir = str(progress)
            recorder._capture_failure_reason = "camera frame gap detected (2 to 5)"
            recorder._capture_metadata = None
            recorder.tracker = SimpleNamespace(capture_errors=lambda: ())
            recorder.camera = SimpleNamespace(capture_error=None)
            result = recorder.stop_recording("goal_reached")

            invalid = Path(root) / "invalid" / "episode_test"
            metadata = json.loads((invalid / "metadata.json").read_text())
            self.assertEqual(result, "invalid")
            self.assertFalse(metadata["capture_valid"])
            self.assertIn("frame gap", metadata["capture_failure_reason"])
            self.assertEqual(discover_episodes(Path(root)), [])

    def test_healthy_terminal_recording_still_publishes_normally(self):
        with tempfile.TemporaryDirectory() as root:
            progress = Path(root) / ".in_progress" / "episode_ok"
            progress.mkdir(parents=True)
            (progress / "metadata.json").write_text('{"map":"dataset_test1"}\n')
            (progress / "actions.csv").touch()
            (progress / "video.mp4").touch()
            recorder = EpisodeRecorder.__new__(EpisodeRecorder)
            recorder.recording, recorder.paused = True, False
            recorder.resume_deadline = None
            recorder.sync_thread = _StoppedThread()
            recorder.video_writer, recorder.csv_file = Mock(), Mock()
            recorder.episodes_root, recorder.episode_name = root, "episode_ok"
            recorder.ep_dir = str(progress)
            recorder._capture_failure_reason = None
            recorder._capture_metadata = None
            recorder.tracker = SimpleNamespace(capture_errors=lambda: ())
            recorder.camera = SimpleNamespace(capture_error=None)
            self.assertEqual(recorder.stop_recording("goal_reached"), "goal_reached")
            completed = Path(root) / "goal_reached" / "episode_ok"
            self.assertTrue(completed.is_dir())
            self.assertTrue(
                json.loads((completed / "metadata.json").read_text())["capture_valid"]
            )

    def test_concurrent_key_state_updates_finish_released(self):
        import evdev

        tracker = InputTracker([], [])
        devices = [SimpleNamespace(path=f"/dev/input/event{i}") for i in range(2)]

        def pulse(device):
            for _ in range(500):
                tracker._update_physical_key("W", device, evdev.ecodes.KEY_W, 1)
                tracker.get_snapshot_and_reset()
                tracker._update_physical_key("W", device, evdev.ecodes.KEY_W, 0)

        threads = [threading.Thread(target=pulse, args=(device,)) for device in devices]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(tracker.get_snapshot_and_reset()["move_w"], 0)


class GoalConditioningTests(unittest.TestCase):
    def test_goal_requires_stable_id_and_canonical_invisibility(self):
        goal = GoalCondition("exit-a", 0.4, -0.2, True, "exit")
        self.assertEqual(goal.as_tensor().shape, (1, 4))
        invisible = GoalCondition("exit-a", None, None, False, "exit")
        self.assertEqual(invisible.as_tensor()[0, :2].tolist(), [0.0, 0.0])
        with self.assertRaises(ValueError):
            GoalCondition("exit-a", 0.4, -0.2, False, "exit")

    def test_inference_keeps_goal_across_observations_and_clears_on_reset(self):
        from collections import deque

        policy = PolicyInference.__new__(PolicyInference)
        policy.policy = GoalConditionedRecurrentPolicy(7, 7, hidden_size=5)
        policy.current_goal = None
        policy.frames = deque(maxlen=4)
        policy.hidden = None
        goal = GoalCondition("exit-a", 0.4, -0.2, True, "exit")
        policy.set_goal(goal)
        policy.add_frame(np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertIs(policy.current_goal, goal)
        policy.reset_memory()
        self.assertIs(policy.current_goal, goal)
        policy.reset()
        self.assertIsNone(policy.current_goal)

    def test_decision_rate_memory_reset_preserves_goal_and_drops_held_action(self):
        class FakePolicy:
            current_goal = GoalCondition("exit-a", 0.2, 0.0, True, "exit")

            def reset_memory(self):
                self.memory_reset = True

            def reset(self):
                self.current_goal = None

        inner = FakePolicy()
        policy = DecisionRatePolicy(inner, 3)
        policy.tick = 2
        policy.action = {"move_w": 1}
        policy.reset_memory()
        self.assertEqual(policy.tick, 0)
        self.assertIsNone(policy.action)
        self.assertIsNotNone(inner.current_goal)
        self.assertTrue(inner.memory_reset)

    def test_goal_adapter_starts_exactly_equal_to_recurrent_baseline(self):
        class TinyBase(nn.Module):
            hidden_size = 5

            def forward(self, frames, hidden=None, valid=None, return_features=False):
                batch, steps = frames.shape[:2]
                feature = frames.new_zeros(batch, steps, self.hidden_size)
                keys = (
                    "move_w",
                    "move_a",
                    "move_s",
                    "move_d",
                    "jump",
                    "use",
                    "fire_left",
                    "fire_right",
                )
                output = {name: frames.new_zeros(batch, steps, 2) for name in keys}
                output["mouse_dx_logits"] = frames.new_zeros(batch, steps, 7)
                output["mouse_dy_logits"] = frames.new_zeros(batch, steps, 7)
                state = frames.new_zeros(1, batch, self.hidden_size)
                return (output, state, feature) if return_features else (output, state)

        policy = GoalConditionedRecurrentPolicy(7, 7, hidden_size=5)
        source = RecurrentBinnedPolicy(7, 7, hidden_size=5)
        initialize_goal_policy_from_recurrent(policy, source.state_dict())
        for key, value in source.state_dict().items():
            self.assertTrue(torch.equal(policy.base.state_dict()[key], value))
        self.assertTrue(
            all(not parameter.requires_grad for parameter in policy.base.parameters())
        )
        policy.base = TinyBase()
        frames = torch.zeros(2, 3, 12, 4, 4)
        goals = torch.tensor([[[0.2, 0.0, 1.0, 0.0]] * 3] * 2)
        output, _ = policy(frames, goals)
        self.assertTrue(torch.equal(output["move_w"], torch.zeros(2, 3, 2)))
        self.assertTrue(torch.equal(output["mouse_dx_logits"], torch.zeros(2, 3, 7)))
        step_output, _ = policy.step(frames[:, 0], goals[:, 0])
        self.assertTrue(torch.equal(step_output["move_w"], output["move_w"][:, 0]))
        output["move_w"][..., 1].sum().backward()
        self.assertIsNotNone(policy.binary_goal_heads["move_w"].weight.grad)
        self.assertGreater(
            policy.binary_goal_heads["move_w"].weight.grad.abs().sum().item(), 0
        )

    def test_goal_annotations_require_complete_frames_and_disjoint_groups(self):
        def episode(name, role, session, start):
            return {
                "episode_id": name,
                "split_role": role,
                "training_approved": role == "train",
                "quarantined": False,
                "source_session_id": session,
                "start_group_id": start,
                "map_name": "dataset_test1",
                "frame_count": 1,
                "requested_goal_id": "exit-1",
                "provenance": {
                    key: "0" * 64
                    for key in ("video_sha256", "actions_sha256", "metadata_sha256")
                },
                "annotations": [
                    {
                        "frame_idx": 0,
                        "goal_id": "exit-1",
                        "goal_transition": False,
                        "target_kind": "exit",
                        "visible": True,
                        "x_norm": 0.2,
                        "y_norm": -0.2,
                        "action_owner": "human",
                    }
                ],
            }

        manifest = {
            "schema": GOAL_ANNOTATION_SCHEMA,
            "coordinate_system": GOAL_COORDINATE_SYSTEM,
            "episodes": [
                episode("train-a", "train", "session-a", "start-a"),
                episode("dev-b", "development", "session-b", "start-b"),
            ],
        }
        self.assertTrue(validate_goal_annotation_manifest(manifest)["valid"])
        paired_result = validate_goal_annotation_manifest(
            manifest,
            require_counterfactual_pairs=True,
        )
        self.assertTrue(
            any("counterfactual coverage" in error for error in paired_result["errors"])
        )
        manifest["episodes"][1]["source_session_id"] = "session-a"
        manifest["episodes"][0]["annotations"] = []
        result = validate_goal_annotation_manifest(manifest)
        self.assertFalse(result["valid"])
        self.assertTrue(any("frame_idx" in error for error in result["errors"]))
        self.assertTrue(any("leakage" in error for error in result["errors"]))
        malformed = json.loads(json.dumps(manifest))
        malformed["coordinate_system"] = "pixels_top_left"
        malformed["episodes"][0]["annotations"] = [
            {
                "frame_idx": 0,
                "goal_id": "different-goal",
                "target_kind": "exit",
                "visible": True,
                "x_norm": 2.0,
                "y_norm": 0.0,
                "action_owner": "human",
            }
        ]
        malformed_result = validate_goal_annotation_manifest(malformed)
        self.assertTrue(
            any("coordinate_system" in error for error in malformed_result["errors"])
        )
        self.assertTrue(
            any("explicit transition" in error for error in malformed_result["errors"])
        )
        self.assertTrue(
            any("coordinates" in error for error in malformed_result["errors"])
        )


if __name__ == "__main__":
    unittest.main()
