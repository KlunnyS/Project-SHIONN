"""Tests for physical-input selection and demonstration recording flow."""

# Python standard library: recording fixtures, mocks, and test cases.
import json
import tempfile
import threading
import unittest
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Project recording modules at the repository root.
from record_dataset import (
    EpisodeEvent, EpisodeEventStream, RecordingProgress, event_outcome,
    parse_args, record_episodes, validate_args,
    wait_for_terminal_event,
)
from recorder import EpisodeRecorder, InputTracker, _pointer_score, find_input_devices
from record_recovery import (
    main as record_recovery, pointer_inside_portal, wait_for_manual_start,
    wait_for_physical_mouse,
)


class FakePointerDevice:
    def __init__(self, name, keys, path="/dev/input/event-test", extra_capabilities=None):
        self.name = name
        self.path = path
        self._keys = keys
        self._extra_capabilities = extra_capabilities or {}
        self.closed = False

    def capabilities(self):
        import evdev
        return {evdev.ecodes.EV_KEY: self._keys, **self._extra_capabilities}

    def close(self):
        self.closed = True


class EpisodeEventStreamTest(unittest.TestCase):
    def test_escape_and_k_controls_ignore_key_repeat_and_duplicate_presses(self):
        import evdev

        tracker = InputTracker([], [])
        keyboard = SimpleNamespace(path="/dev/input/event-test")
        escape = SimpleNamespace(code=evdev.ecodes.KEY_ESC, value=1)
        release = SimpleNamespace(code=evdev.ecodes.KEY_ESC, value=0)
        with patch("recorder.time.monotonic", return_value=1.0) as clock:
            tracker._track_control_key(keyboard, escape)
            tracker._track_control_key(keyboard, SimpleNamespace(code=escape.code, value=2))
            tracker._track_control_key(keyboard, release)
            tracker._track_control_key(keyboard, escape)  # Duplicate interface timing.
            self.assertEqual(tracker.pop_control_events(), ["pause"])
            clock.return_value = 1.3
            tracker._track_control_key(keyboard, release)
            tracker._track_control_key(keyboard, escape)
            tracker._track_control_key(keyboard, SimpleNamespace(code=evdev.ecodes.KEY_K, value=1))
        self.assertEqual(tracker.pop_control_events(), ["pause", "stop"])

    def test_r_restart_control_is_only_enabled_for_recovery_sessions(self):
        import evdev

        press = SimpleNamespace(code=evdev.ecodes.KEY_R, value=1)
        press.type = evdev.ecodes.EV_KEY
        keyboard = SimpleNamespace(path="/dev/input/event-test", name="Keyboard",
                                   read_loop=lambda: iter([press]))
        ordinary = InputTracker([], [])
        recovery = InputTracker([], [], restart_enabled=True)
        ordinary._kbd_loop(keyboard)
        recovery._kbd_loop(keyboard)
        self.assertEqual(ordinary.pop_control_events(), [])
        self.assertEqual(recovery.pop_control_events(), ["restart"])

    def test_p_start_control_is_only_enabled_for_manual_recovery(self):
        import evdev

        press = SimpleNamespace(type=evdev.ecodes.EV_KEY,
                                code=evdev.ecodes.KEY_P, value=1)
        keyboard = SimpleNamespace(path="/dev/input/event-test", name="Keyboard",
                                   read_loop=lambda: iter([press]))
        ordinary = InputTracker([], [])
        manual = InputTracker([], [], start_enabled=True)
        ordinary._kbd_loop(keyboard)
        manual._kbd_loop(keyboard)
        self.assertEqual(ordinary.pop_control_events(), [])
        self.assertEqual(manual.pop_control_events(), ["start"])

    def test_pointer_selection_prefers_physical_mouse_over_dongle(self):
        import evdev

        button_keys = [evdev.ecodes.BTN_LEFT]
        physical_mouse = FakePointerDevice("Razer Razer Basilisk V3", button_keys)
        receiver = FakePointerDevice("Corsair Wireless Dongle", button_keys)
        self.assertGreater(_pointer_score(physical_mouse), _pointer_score(receiver))

    @patch("recorder.evdev.InputDevice")
    @patch("recorder.evdev.list_devices")
    def test_device_scan_skips_unreadable_event_nodes(self, list_devices, input_device):
        import evdev

        list_devices.return_value = ["/dev/input/event0", "/dev/input/event1"]
        mouse = FakePointerDevice(
            "Physical Mouse",
            [evdev.ecodes.BTN_LEFT],
            path="/dev/input/event1",
            extra_capabilities={
                evdev.ecodes.EV_REL: [evdev.ecodes.REL_X, evdev.ecodes.REL_Y]
            },
        )
        input_device.side_effect = [PermissionError("denied"), mouse]

        keyboards, mice = find_input_devices()

        self.assertEqual(keyboards, [])
        self.assertEqual(mice, [mouse])

    def test_parses_events_split_across_netconsole_chunks(self):
        stream = EpisodeEventStream()
        stream.feed("console noise\nEVT|chamber_")
        self.assertIsNone(stream.pop())
        stream.feed("ready|123.5\nmore noise\n")
        self.assertEqual(stream.pop(), EpisodeEvent("chamber_ready", "123.5"))
        self.assertIsNone(stream.pop())

    def test_parses_multiple_events_and_outcomes(self):
        stream = EpisodeEventStream()
        stream.feed("EVT|goal_reached|1\nEVT|episode_failed|out_of_bounds\n")
        goal = stream.pop()
        failure = stream.pop()
        self.assertEqual(event_outcome(goal), "goal_reached")
        self.assertEqual(event_outcome(failure), "out_of_bounds")

    def test_sanitizes_failure_reason_for_episode_directory(self):
        event = EpisodeEvent("episode_failed", "bad reason/with spaces")
        self.assertEqual(event_outcome(event), "bad_reason_with_spaces")


class EpisodeRecorderOutcomeTest(unittest.TestCase):
    def test_low_level_recorder_k_stops_session(self):
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.recording = False
        recorder.tracker = Mock()
        recorder.tracker.pop_control_events.return_value = ["stop"]
        recorder.camera = Mock()
        recorder.controller = Mock()
        recorder.controller.connect.return_value = True

        recorder.run()

        recorder.tracker.stop.assert_called_once()
        recorder.camera.stop.assert_called_once()
        recorder.controller.disconnect.assert_called_once()

    def test_pause_skips_frame_and_action_writes(self):
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.recording = True
        recorder.paused = True
        recorder.fps = 24
        recorder.sync_lock = threading.Lock()
        recorder.frame_idx = 0
        recorder.tracker = Mock()
        recorder.video_writer = Mock()
        recorder.csv_writer = Mock()

        with patch("recorder.time.sleep", side_effect=lambda _: setattr(recorder, "recording", False)):
            recorder._sync_loop()

        recorder.tracker.get_snapshot_and_reset.assert_called_once()
        recorder.video_writer.write.assert_not_called()
        recorder.csv_writer.writerow.assert_not_called()
        self.assertEqual(recorder.frame_idx, 0)

    def test_pending_escape_stops_frames_before_main_loop_toggles_pause(self):
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.recording = True
        recorder.paused = False
        recorder.fps = 24
        recorder.frame_idx = 0
        recorder.sync_lock = threading.Lock()
        recorder.tracker = Mock()
        recorder.tracker.pause_is_pending.return_value = True
        recorder.video_writer = Mock()
        recorder.csv_writer = Mock()

        with patch("recorder.time.sleep", side_effect=lambda _: setattr(recorder, "recording", False)):
            recorder._sync_loop()

        recorder.video_writer.write.assert_not_called()
        recorder.csv_writer.writerow.assert_not_called()
        self.assertEqual(recorder.frame_idx, 0)

    def test_resume_keeps_frame_indices_contiguous(self):
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.recording = True
        recorder.paused = False
        recorder.fps = 24
        recorder.frame_idx = 0
        recorder.sync_lock = threading.Lock()
        recorder.tracker = Mock()
        recorder.tracker.get_snapshot_and_reset.return_value = {
            name: 0 for name in (
                "move_w", "move_a", "move_s", "move_d", "jump", "crouch",
                "use", "fire_left", "fire_right", "mouse_dx", "mouse_dy",
            )
        }
        recorder.tracker.pause_is_pending.return_value = False
        recorder.camera = Mock()
        recorder.camera.get_latest_frame.return_value = object()
        recorder.video_writer = Mock()
        recorder.csv_writer = Mock()
        recorder.csv_file = Mock()
        sleeps = [0]

        def change_state(_):
            sleeps[0] += 1
            if sleeps[0] == 1:
                recorder.paused = True
            elif sleeps[0] == 2:
                recorder.paused = False
            else:
                recorder.recording = False

        with patch("recorder.time.sleep", side_effect=change_state):
            recorder._sync_loop()

        self.assertEqual(recorder.frame_idx, 2)
        self.assertEqual(recorder.video_writer.write.call_count, 2)
        self.assertEqual(
            [call.args[0][1] for call in recorder.csv_writer.writerow.call_args_list],
            [0, 1],
        )

    def test_escape_toggles_pause_and_discards_pending_actions(self):
        recorder = EpisodeRecorder.__new__(EpisodeRecorder)
        recorder.recording = True
        recorder.paused = False
        recorder.sync_lock = threading.Lock()
        recorder.tracker = Mock()

        self.assertTrue(recorder.toggle_pause())
        self.assertFalse(recorder.toggle_pause())
        self.assertEqual(recorder.tracker.get_snapshot_and_reset.call_count, 2)

    @patch("recorder.get_default_output", return_value="DP-1")
    @patch("recorder.WaylandCamera")
    @patch("recorder.InputTracker")
    @patch("recorder.find_input_devices", return_value=([Mock()], [Mock()]))
    def test_evaluation_recordings_use_separate_root(
        self, find_devices, tracker, camera, default_output
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "episodes_eval"
            recorder = EpisodeRecorder(episodes_root=root, controller=Mock())

            self.assertEqual(Path(recorder.episodes_root), root)
            self.assertEqual(Path(recorder.in_progress_root), root / ".in_progress")
            self.assertTrue((root / ".in_progress").is_dir())

    @patch("recorder.get_default_output", return_value="DP-1")
    @patch("recorder.WaylandCamera")
    @patch("recorder.InputTracker")
    @patch("recorder.find_input_devices", return_value=([Mock()], [Mock()]))
    def test_test_recorder_does_not_create_episode_directory(
        self, find_devices, tracker, camera, default_output,
    ):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory) / "unused_episodes"
            EpisodeRecorder(episodes_root=root, controller=Mock(), prepare_output=False)
            self.assertFalse(root.exists())

    @patch("recorder.threading.Thread")
    @patch("recorder.FFmpegVideoWriter")
    def test_new_episode_records_map_name(self, video_writer, thread):
        with tempfile.TemporaryDirectory() as temporary_directory:
            recorder = EpisodeRecorder.__new__(EpisodeRecorder)
            recorder.in_progress_root = temporary_directory
            recorder.width = 1920
            recorder.height = 1080
            recorder.fps = 24
            recorder.video_crf = 20

            recorder.start_recording(map_name="dataset_test2", metadata_extra={
                "source": "model_recovery", "source_attempt": "/tmp/attempt.jsonl",
            })
            try:
                metadata = json.loads((Path(recorder.ep_dir) / "metadata.json").read_text())
                self.assertEqual(metadata["map"], "dataset_test2")
                self.assertEqual(metadata["source"], "model_recovery")
                self.assertEqual(metadata["source_attempt"], "/tmp/attempt.jsonl")
                self.assertTrue((Path(recorder.ep_dir) / "actions.csv").is_file())
                thread.return_value.start.assert_called_once()
            finally:
                recorder.recording = False
                recorder.csv_file.close()

    def test_completed_episode_moves_into_its_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            episodes_root = Path(temporary_directory) / "episodes"
            active_dir = episodes_root / ".in_progress" / "episode_test"
            active_dir.mkdir(parents=True)
            (active_dir / "video.mp4").touch()
            (active_dir / "actions.csv").touch()
            (active_dir / "metadata.json").write_text('{"map": "dataset_test1"}\n')

            recorder = EpisodeRecorder.__new__(EpisodeRecorder)
            recorder.recording = True
            recorder.sync_thread = Mock()
            recorder.video_writer = Mock()
            recorder.csv_file = Mock()
            recorder.episodes_root = str(episodes_root)
            recorder.episode_name = "episode_test"
            recorder.ep_dir = str(active_dir)

            recorder.stop_recording("goal_reached")

            completed_dir = episodes_root / "goal_reached" / "episode_test"
            self.assertEqual(Path(recorder.ep_dir), completed_dir)
            self.assertTrue((completed_dir / "video.mp4").is_file())
            self.assertEqual(
                json.loads((completed_dir / "metadata.json").read_text())["map"],
                "dataset_test1",
            )
            self.assertFalse(active_dir.exists())


class RecordingQuotaTest(unittest.TestCase):
    @patch("record_dataset.read_events", return_value=[])
    def test_paused_time_does_not_consume_recording_duration(self, read_events):
        recorder = Mock()
        recorder.paused = False
        events = deque([["pause"], ["pause"]])
        recorder.tracker.pop_control_events.side_effect = (
            lambda: events.popleft() if events else []
        )
        recorder.toggle_pause.side_effect = lambda: setattr(
            recorder, "paused", not recorder.paused
        )
        clock = [0.0]

        def advance(seconds):
            clock[0] += 5.0 if recorder.paused else seconds

        with patch("record_dataset.time.monotonic", side_effect=lambda: clock[0]), \
             patch("record_dataset.time.sleep", side_effect=advance):
            outcome = wait_for_terminal_event(Mock(), EpisodeEventStream(), 0.2, recorder)

        self.assertEqual(outcome, "timeout")
        self.assertGreaterEqual(clock[0], 5.2)
        self.assertEqual(recorder.toggle_pause.call_count, 2)

    @patch("record_dataset.read_events")
    def test_k_stops_without_waiting_for_chamber_event(self, read_events):
        recorder = Mock()
        recorder.paused = True
        recorder.tracker.pop_control_events.return_value = ["stop"]

        self.assertEqual(
            wait_for_terminal_event(Mock(), EpisodeEventStream(), 30.0, recorder),
            "interrupted",
        )
        read_events.assert_not_called()

    @patch("record_dataset.time.sleep")
    @patch("record_dataset.wait_for_terminal_event", return_value="interrupted")
    @patch("record_dataset.load_map_and_wait_for_ready")
    def test_k_saves_partial_episode_and_does_not_restart(
        self, load_map, wait_for_terminal, sleep
    ):
        args = SimpleNamespace(
            episodes=0, map_names=["dataset_test2"], ready_timeout=60.0,
            duration=30.0, restart_delay=1.0,
        )
        recorder = Mock()
        controller = Mock()
        progress = RecordingProgress()

        record_episodes(recorder, controller, EpisodeEventStream(), args, progress)

        load_map.assert_called_once()
        recorder.stop_recording.assert_called_once_with("interrupted")
        self.assertEqual(progress.attempts, 1)
        controller.send_command.assert_not_called()

    def test_repeated_map_flags_share_one_per_map_target(self):
        args = parse_args([
            "--map", "dataset_test5", "--map", "dataset_test6", "--episodes", "3",
            "--episodes-root", "episodes_eval",
        ])
        validate_args(args)
        self.assertEqual(args.map_names, ["dataset_test5", "dataset_test6"])
        self.assertEqual(args.episodes, 3)
        self.assertEqual(args.episodes_root, Path("episodes_eval"))

    def test_duplicate_maps_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_args(parse_args(["--map", "dataset_test5", "--map", "dataset_test5"]))

    @patch("record_dataset.time.sleep")
    @patch("record_dataset.wait_for_terminal_event")
    @patch("record_dataset.load_map_and_wait_for_ready")
    def test_failures_are_retried_and_target_returns_to_menu(
        self, load_map, wait_for_terminal, sleep
    ):
        wait_for_terminal.side_effect = [
            "timeout", "goal_reached", "out_of_bounds", "goal_reached"
        ]
        args = SimpleNamespace(
            episodes=2, map_names=["dataset_test2"], ready_timeout=60.0,
            duration=30.0, restart_delay=1.0,
        )
        recorder = Mock()
        controller = Mock()
        progress = RecordingProgress()

        record_episodes(recorder, controller, EpisodeEventStream(), args, progress)

        self.assertEqual(progress.successes, 2)
        self.assertEqual(progress.attempts, 4)
        self.assertEqual(load_map.call_count, 4)
        self.assertEqual(
            [call.args[0] for call in recorder.stop_recording.call_args_list],
            ["timeout", "goal_reached", "out_of_bounds", "goal_reached"],
        )
        controller.send_command.assert_called_once_with("disconnect")


    @patch("record_dataset.time.sleep")
    @patch("record_dataset.wait_for_terminal_event")
    @patch("record_dataset.load_map_and_wait_for_ready")
    def test_cycles_maps_after_success_and_retries_failure_on_same_map(
        self, load_map, wait_for_terminal, sleep
    ):
        wait_for_terminal.side_effect = [
            "goal_reached", "timeout", "goal_reached", "goal_reached", "goal_reached"
        ]
        args = SimpleNamespace(
            episodes=2, map_names=["dataset_test5", "dataset_test6"],
            ready_timeout=60.0, duration=30.0, restart_delay=1.0,
        )
        recorder = Mock()
        controller = Mock()
        progress = RecordingProgress()

        record_episodes(recorder, controller, EpisodeEventStream(), args, progress)

        expected_maps = [
            "dataset_test5", "dataset_test6", "dataset_test6",
            "dataset_test5", "dataset_test6",
        ]
        self.assertEqual([call.args[2] for call in load_map.call_args_list], expected_maps)
        self.assertEqual(
            [call.kwargs["map_name"] for call in recorder.start_recording.call_args_list],
            expected_maps,
        )
        self.assertEqual(progress.successes_by_map, {"dataset_test5": 2, "dataset_test6": 2})
        self.assertEqual(progress.attempts, 5)
        controller.send_command.assert_called_once_with("disconnect")


class RecoveryRecordingTest(unittest.TestCase):
    @patch("record_recovery.time.sleep")
    @patch("record_recovery.wait_for_terminal_event", return_value="interrupted")
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_k_preserves_partial_recovery_episode(
        self, game_running, recorder_class, connect, camera_ready, terminal, sleep
    ):
        recorder = recorder_class.return_value
        recorder.recording = False

        record_recovery(["--map", "dataset_test9", "--focus-delay", "0"])

        recorder.stop_recording.assert_called_once_with("interrupted")
        self.assertEqual(terminal.call_args.kwargs["recorder"], recorder)

    @patch("record_dataset.time.sleep")
    @patch("record_dataset.read_events")
    def test_zero_recovery_duration_waits_for_game_outcome(self, read_events, sleep):
        read_events.side_effect = [[], [EpisodeEvent("goal_reached", "1")]]
        self.assertEqual(
            wait_for_terminal_event(Mock(), EpisodeEventStream(), 0),
            "goal_reached",
        )
        self.assertEqual(read_events.call_count, 2)

    @patch("record_dataset.read_events",
           return_value=[EpisodeEvent("chamber_ready", "2")])
    def test_recovery_rejects_unexpected_map_reload(self, read_events):
        self.assertEqual(
            wait_for_terminal_event(
                Mock(), EpisodeEventStream(), 0, reject_new_chamber=True
            ),
            "map_reloaded",
        )

    def test_recovery_r_requests_restart_without_completion(self):
        recorder = Mock()
        recorder.paused = False
        recorder.tracker.pop_control_events.return_value = ["restart"]
        self.assertEqual(
            wait_for_terminal_event(Mock(), EpisodeEventStream(), 0, recorder=recorder),
            "restart_requested",
        )

    def test_mouse_check_accepts_motion_even_if_view_returns_to_start(self):
        now = [0.0]
        tracker = Mock()
        tracker.pop_control_events.return_value = []
        angles = iter([(0.0, 91.0, 0.0), (0.0, 90.0, 0.0), (0.0, 90.0, 0.0)])

        def advance(seconds):
            now[0] += seconds

        with patch("record_recovery.time.monotonic", side_effect=lambda: now[0]), \
             patch("record_recovery.time.sleep", side_effect=advance), \
             patch("record_recovery.read_player_angles", side_effect=lambda *_args, **_kwargs: next(angles)):
            self.assertIsNone(wait_for_physical_mouse(
                Mock(), tracker, 0.3, "auto", (0.0, 90.0, 0.0)
            ))

    def test_mouse_check_retries_capture_once_after_no_motion(self):
        now = [0.0]
        tracker = Mock()
        tracker.pop_control_events.return_value = []
        samples = iter([
            (0.0, 90.0, 0.0), (0.0, 90.0, 0.0),
            (0.0, 90.0, 0.0), (0.0, 90.0, 0.0),
            (0.0, 91.0, 0.0), (0.0, 91.0, 0.0),
        ])
        with patch("record_recovery.time.monotonic", side_effect=lambda: now[0]), \
             patch("record_recovery.time.sleep", side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)), \
             patch("record_recovery.read_player_angles",
                   side_effect=lambda *_args, **_kwargs: next(samples)), \
             patch("record_recovery.activate_portal_input",
                   return_value={"portal_focused": True}) as activate:
            self.assertIsNone(wait_for_physical_mouse(
                Mock(), tracker, 0.2, "auto", (0.0, 90.0, 0.0)
            ))
        activate.assert_called_once()

    def test_mouse_check_k_stops_before_recording(self):
        tracker = Mock()
        tracker.pop_control_events.return_value = ["stop"]
        self.assertEqual(wait_for_physical_mouse(
            Mock(), tracker, 5.0, "auto", (0.0, 90.0, 0.0)
        ), "interrupted")

    def test_portal_pointer_check_detects_escape_to_second_monitor(self):
        window = {
            "hyprland_instance": "test-instance", "at": [100, 200],
            "size": [800, 600],
        }
        with patch("record_recovery.query_hyprland_active_window",
                   return_value={"portal_focused": True}), \
             patch("record_recovery.query_hyprland_cursor",
                   side_effect=[(500, 500), (950, 500)]):
            self.assertTrue(pointer_inside_portal(window))
            self.assertFalse(pointer_inside_portal(window))

    def test_mouse_check_restarts_if_pointer_escapes_during_edge_sweep(self):
        window = {"hyprland_instance": "test", "at": [0, 0], "size": [800, 600]}
        now = [0.0]
        tracker = Mock()
        tracker.pop_control_events.return_value = []
        with patch("record_recovery.time.monotonic", side_effect=lambda: now[0]), \
             patch("record_recovery.time.sleep",
                   side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)), \
             patch("record_recovery.pointer_inside_portal", side_effect=[True, False]), \
             patch("record_recovery.read_player_angles", return_value=(0.0, 90.0, 0.0)):
            self.assertEqual(wait_for_physical_mouse(
                Mock(), tracker, 2.0, "auto", (0.0, 90.0, 0.0), window,
            ), "restart_requested")

    def test_manual_focus_waits_for_p_after_delay(self):
        now = [0.0]
        tracker = Mock()
        tracker.pop_control_events.side_effect = [["start"], [], ["start"]]
        with patch("record_recovery.time.monotonic", side_effect=lambda: now[0]), \
             patch("record_recovery.time.sleep",
                   side_effect=lambda seconds: now.__setitem__(0, now[0] + seconds)):
            self.assertEqual(wait_for_manual_start(tracker, 0.2), "start")
        self.assertEqual(tracker.pop_control_events.call_count, 3)

    def test_manual_focus_r_retries_before_recording(self):
        tracker = Mock()
        tracker.pop_control_events.return_value = ["restart"]
        self.assertEqual(wait_for_manual_start(tracker, 5.0), "restart_requested")

    @patch("record_recovery.time.sleep")
    @patch("record_recovery.wait_for_terminal_event", return_value="goal_reached")
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_recovers_from_current_state_without_loading_a_map(
        self, game_running, recorder_class, connect, camera_ready, terminal, sleep
    ):
        with tempfile.TemporaryDirectory() as directory:
            attempt = Path(directory) / "attempt.jsonl"
            attempt.touch()
            recorder = recorder_class.return_value
            recorder.recording = False
            recorder.ep_dir = Path(directory) / "recovery" / "goal_reached" / "episode_test"

            with patch("record_recovery.disable_chamber_timeout") as disable_timeout, \
                 patch("record_recovery.read_player_angles",
                       side_effect=[(0.0, 90.0, 0.0)] * 3), \
                 patch("record_recovery.wait_for_physical_mouse", return_value=None), \
                 patch("record_recovery.pointer_inside_portal", return_value=True), \
                 patch("record_recovery.activate_portal_input",
                       return_value={"portal_focused": True}) as activate_input:
                record_recovery([
                    "--map", "dataset_test9", "--source-attempt", str(attempt),
                    "--episodes-root", str(Path(directory) / "recovery"),
                    "--result-file", str(Path(directory) / "result.json"),
                    "--focus-delay", "2", "--max-seconds", "0",
                    "--keep-focused", "--disable-chamber-timeout",
                ])
            self.assertEqual(disable_timeout.call_count, 2)
            disable_timeout.assert_any_call(recorder.controller, require_active=True)
            activate_input.assert_called_once_with(recorder.controller, "auto")

            recorder.start_recording.assert_called_once_with(
                map_name="dataset_test9",
                metadata_extra={
                    "source": "model_recovery", "source_attempt": str(attempt.resolve()),
                },
            )
            recorder.stop_recording.assert_called_once_with("goal_reached")
            result = json.loads((Path(directory) / "result.json").read_text())
            self.assertEqual(result["outcome"], "goal_reached")
            self.assertEqual(result["episode"], str(recorder.ep_dir.resolve()))
            recorder.controller.load_map.assert_not_called()
            recorder.controller.send_command.assert_not_called()

    @patch("record_recovery.time.sleep")
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_recovery_rejects_uncaptured_physical_mouse(
        self, game_running, recorder_class, connect, camera_ready, sleep,
    ):
        recorder = recorder_class.return_value
        recorder.recording = False
        with patch("record_recovery.read_player_angles", side_effect=[
            (0.0, 90.0, 0.0), (0.0, 90.0, 0.0),
        ]), patch("record_recovery.activate_portal_input",
                  return_value={"portal_focused": True}), \
             patch("record_recovery.wait_for_physical_mouse",
                   side_effect=RuntimeError("Physical mouse did not turn")):
            with self.assertRaisesRegex(RuntimeError, "Physical mouse did not turn"):
                record_recovery([
                    "--map", "dataset_test5", "--keep-focused",
                    "--focus-delay", "5", "--max-seconds", "0",
                ])
        recorder.start_recording.assert_not_called()

    @patch("record_recovery.wait_for_physical_mouse", return_value=None)
    @patch("record_recovery.wait_for_terminal_event", return_value="restart_requested")
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_r_during_correction_discards_clip_and_restores_game_binding(
        self, game_running, recorder_class, connect, camera_ready, terminal, mouse_check,
    ):
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_class.return_value
            recorder.recording = False
            recorder.ep_dir = Path(directory) / "discarded" / "episode_test"
            result_file = Path(directory) / "result.json"
            with patch("record_recovery.read_player_angles",
                       side_effect=[(0.0, 90.0, 0.0)] * 3), \
                 patch("record_recovery.read_portal_binding",
                       return_value="swap_ss_input"), \
                 patch("record_recovery.restore_portal_binding") as restore, \
                 patch("record_recovery.pointer_inside_portal", return_value=True), \
                 patch("record_recovery.activate_portal_input",
                       return_value={"portal_focused": True}):
                record_recovery([
                    "--map", "dataset_test5", "--keep-focused",
                    "--session-controls", "--focus-delay", "2",
                    "--result-file", str(result_file),
                ])
            recorder.stop_recording.assert_called_once_with("discarded")
            restore.assert_called_once_with(
                recorder.controller, "r", "swap_ss_input"
            )
            self.assertEqual(json.loads(result_file.read_text())["outcome"],
                             "restart_requested")

    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_finished_chamber_skips_recovery_without_recording(
        self, game_running, recorder_class, connect, camera_ready,
    ):
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_class.return_value
            recorder.recording = False
            result_file = Path(directory) / "result.json"
            with patch("record_recovery.disable_chamber_timeout", side_effect=RuntimeError(
                "Chamber is already finished or not active for recovery"
            )):
                record_recovery([
                    "--map", "dataset_test1", "--disable-chamber-timeout",
                    "--result-file", str(result_file),
                ])
            self.assertEqual(json.loads(result_file.read_text())["outcome"],
                             "model_chamber_finished")
            recorder.start_recording.assert_not_called()

    @patch("record_recovery.wait_for_physical_mouse", return_value=None)
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_early_mouse_motion_retries_without_error_or_clip(
        self, game_running, recorder_class, connect, camera_ready, mouse_check,
    ):
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_class.return_value
            recorder.recording = False
            result_file = Path(directory) / "result.json"
            with patch("record_recovery.read_player_angles", side_effect=[
                (0.0, 90.0, 0.0), (-21.1, 90.0, 0.0),
            ]), patch("record_recovery.activate_portal_input",
                      return_value={"portal_focused": True}):
                record_recovery([
                    "--map", "dataset_test1", "--keep-focused",
                    "--focus-delay", "2", "--result-file", str(result_file),
                ])
            self.assertEqual(json.loads(result_file.read_text())["outcome"],
                             "restart_requested")
            recorder.start_recording.assert_not_called()
            mouse_check.assert_not_called()

    @patch("record_recovery.wait_for_physical_mouse", return_value=None)
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_finished_during_focus_wait_skips_recovery(
        self, game_running, recorder_class, connect, camera_ready, mouse_check,
    ):
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_class.return_value
            recorder.recording = False
            result_file = Path(directory) / "result.json"
            with patch("record_recovery.disable_chamber_timeout", side_effect=[
                None, RuntimeError("Chamber is already finished or not active for recovery"),
            ]), patch("record_recovery.read_player_angles", side_effect=[
                (0.0, 90.0, 0.0), (0.0, 90.0, 0.0), (0.0, 90.0, 0.0),
            ]), patch("record_recovery.pointer_inside_portal", return_value=True), \
                 patch("record_recovery.activate_portal_input",
                      return_value={"portal_focused": True}):
                record_recovery([
                    "--map", "dataset_test1", "--disable-chamber-timeout",
                    "--keep-focused", "--focus-delay", "2",
                    "--result-file", str(result_file),
                ])
            self.assertEqual(json.loads(result_file.read_text())["outcome"],
                             "model_chamber_finished")
            recorder.start_recording.assert_not_called()

    @patch("record_recovery.wait_for_physical_mouse", return_value=None)
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_large_mouse_movement_during_wait_retries_without_clip(
        self, game_running, recorder_class, connect, camera_ready, mouse_check,
    ):
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_class.return_value
            recorder.recording = False
            result_file = Path(directory) / "result.json"
            with patch("record_recovery.read_player_angles", side_effect=[
                (0.0, 90.0, 0.0), (0.0, 90.0, 0.0), (0.0, 112.0, 0.0),
            ]), patch("record_recovery.pointer_inside_portal", return_value=True), \
                 patch("record_recovery.activate_portal_input",
                      return_value={"portal_focused": True}):
                record_recovery([
                    "--map", "dataset_test1", "--keep-focused",
                    "--focus-delay", "2", "--result-file", str(result_file),
                ])
            self.assertEqual(json.loads(result_file.read_text())["outcome"],
                             "restart_requested")
            recorder.start_recording.assert_not_called()

    @patch("record_recovery.wait_for_manual_start", return_value="start")
    @patch("record_recovery.wait_for_terminal_event", return_value="goal_reached")
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_manual_focus_allows_reframing_then_records_on_p(
        self, game_running, recorder_class, connect, camera_ready, terminal, manual_start,
    ):
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_class.return_value
            recorder.recording = False
            recorder.ep_dir = Path(directory) / "episode"
            result_file = Path(directory) / "result.json"
            with patch("record_recovery.disable_chamber_timeout"), \
                 patch("record_recovery.read_portal_binding", return_value=""), \
                 patch("record_recovery.restore_portal_binding"), \
                 patch("record_recovery.read_player_angles", side_effect=[
                     (0.0, 90.0, 0.0), (-21.1, 90.0, 0.0),
                 ]), patch("record_recovery.activate_portal_input",
                           return_value={"portal_focused": True}), \
                 patch("record_recovery.pointer_inside_portal") as pointer_check, \
                 patch("record_recovery.wait_for_physical_mouse") as mouse_check:
                record_recovery([
                    "--map", "dataset_test1", "--keep-focused",
                    "--disable-chamber-timeout", "--session-controls",
                    "--manual-focus", "--focus-delay", "2",
                    "--result-file", str(result_file),
                ])
            manual_start.assert_called_once_with(recorder.tracker, 2.0)
            mouse_check.assert_not_called()
            pointer_check.assert_not_called()
            recorder.start_recording.assert_called_once()
            self.assertEqual(json.loads(result_file.read_text())["outcome"], "goal_reached")

    @patch("record_recovery.wait_for_manual_start", return_value="start")
    @patch("record_recovery.wait_for_terminal_event", return_value="goal_reached")
    @patch("record_recovery.wait_for_camera")
    @patch("record_recovery.connect_controller")
    @patch("record_recovery.EpisodeRecorder")
    @patch("record_recovery.is_game_running", return_value=True)
    def test_test_mode_monitors_completion_without_saving_episode(
        self, game_running, recorder_class, connect, camera_ready, terminal, manual_start,
    ):
        with tempfile.TemporaryDirectory() as directory:
            recorder = recorder_class.return_value
            recorder.recording = False
            result_file = Path(directory) / "result.json"
            with patch("record_recovery.disable_chamber_timeout"), \
                 patch("record_recovery.read_portal_binding", return_value=""), \
                 patch("record_recovery.restore_portal_binding"), \
                 patch("record_recovery.read_player_angles", side_effect=[
                     (0.0, 90.0, 0.0), (0.0, 90.0, 0.0),
                 ]), patch("record_recovery.activate_portal_input",
                           return_value={"portal_focused": True}):
                record_recovery([
                    "--map", "dataset_test1", "--keep-focused",
                    "--disable-chamber-timeout", "--session-controls",
                    "--manual-focus", "--test-mode", "--focus-delay", "2",
                    "--result-file", str(result_file),
                ])
            recorder.start_recording.assert_not_called()
            recorder.stop_recording.assert_not_called()
            self.assertEqual(json.loads(result_file.read_text()), {
                "outcome": "goal_reached", "episode": None,
            })


if __name__ == "__main__":
    unittest.main()
