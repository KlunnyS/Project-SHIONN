"""Bound recorder shutdown so handoff cannot wait forever on capture exit."""
import subprocess
import unittest
from unittest.mock import Mock
from recorder import WaylandCamera


class CaptureShutdownTest(unittest.TestCase):
    def test_stalled_recorder_is_killed_after_bounded_wait(self):
        camera = WaylandCamera.__new__(WaylandCamera)
        camera.proc = Mock()
        camera.proc.poll.return_value = None
        camera.proc.wait.side_effect = [subprocess.TimeoutExpired('wf-recorder', 3), 0]
        camera.stderr_thread = Mock()
        camera.stop()
        camera.proc.terminate.assert_called_once()
        camera.proc.kill.assert_called_once()
        self.assertEqual(camera.proc.wait.call_count, 2)
        for call in camera.proc.wait.call_args_list:
            self.assertEqual(call.kwargs, {'timeout': 3.0})
        self.assertFalse(camera.running)

    def test_normal_shutdown_does_not_kill(self):
        camera = WaylandCamera.__new__(WaylandCamera)
        camera.proc = Mock()
        camera.proc.poll.return_value = None
        camera.stderr_thread = None
        camera.stop()
        camera.proc.kill.assert_not_called()
        camera.proc.wait.assert_called_once_with(timeout=3.0)
