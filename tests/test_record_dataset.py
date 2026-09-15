import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from record_dataset import EpisodeEvent, EpisodeEventStream, event_outcome
from recorder import EpisodeRecorder, _pointer_score, find_input_devices


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
    def test_completed_episode_moves_into_its_result_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            episodes_root = Path(temporary_directory) / "episodes"
            active_dir = episodes_root / ".in_progress" / "episode_test"
            active_dir.mkdir(parents=True)
            (active_dir / "video.mp4").touch()
            (active_dir / "actions.csv").touch()

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
            self.assertFalse(active_dir.exists())


if __name__ == "__main__":
    unittest.main()
