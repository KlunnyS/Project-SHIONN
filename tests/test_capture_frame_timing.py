"""Camera publication tests use synthetic pipe bytes, without starting capture."""
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from recorder import FrameSample, WaylandCamera


class FrameTimingTests(unittest.TestCase):
    def test_empty_sample_has_unknown_identity_and_receive_time(self):
        camera = WaylandCamera(2, 1)
        self.assertEqual(camera.get_latest_frame_sample(), FrameSample(None, None, None))
        self.assertIsNone(camera.get_latest_frame())

    def test_complete_frames_publish_coherent_ids_times_and_keep_old_sample(self):
        camera = WaylandCamera(2, 1)
        saved = []
        class Pipe:
            count = 0
            def read(self, size):
                self.count += 1
                saved.append(camera.get_latest_frame_sample())
                return bytes([self.count]) * size if self.count <= 2 else b''
        camera.proc = SimpleNamespace(stdout=Pipe())
        camera.running = True
        with patch('recorder.time.monotonic', side_effect=[10., 11.]):
            camera._update()
        first, last = saved[1], camera.get_latest_frame_sample()
        self.assertEqual((first.frame_id, first.received_monotonic), (1, 10.))
        self.assertEqual((last.frame_id, last.received_monotonic), (2, 11.))
        np.testing.assert_array_equal(first.frame, np.full((1, 2, 3), 1))
        np.testing.assert_array_equal(last.frame, np.full((1, 2, 3), 2))
        self.assertIs(camera.get_latest_frame(), last.frame)
        self.assertIs(camera.get_latest_frame_sample(), last)

    def test_identical_pixels_still_get_distinct_ids(self):
        camera = WaylandCamera(2, 1)
        camera.proc = SimpleNamespace(stdout=io.BytesIO(bytes(12)))
        camera.running = True
        with patch('recorder.time.monotonic', side_effect=[1., 2.]):
            camera._update()
        self.assertEqual(camera.get_latest_frame_sample().frame_id, 2)

    def test_incomplete_read_is_not_published_as_a_frame(self):
        camera = WaylandCamera(2, 1)
        camera.proc = SimpleNamespace(stdout=io.BytesIO(bytes(3)))
        camera.running = True
        camera._update()
        self.assertIsNone(camera.get_latest_frame_sample().frame_id)


if __name__ == '__main__':
    unittest.main()
