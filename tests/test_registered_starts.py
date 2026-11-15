import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import record_recovery
import registered_starts as starts


class RegisteredStartTests(unittest.TestCase):
    def setUp(self):
        self.start = starts.load_start(
            Path("configs/registered_starts_v1.json"),
            "goal_choice_interior_01",
            "goal_choice_v1",
        )
        self.pose = {
            "position": self.start["position"],
            "angles": self.start["angles"],
            "measured_monotonic": 1,
        }

    @patch("registered_starts.time.sleep")
    def test_success_wraparound_and_two_samples(self, sleep):
        pose = {**self.pose, "angles": [0, 450, 360]}
        with patch("registered_starts.read_pose", return_value=pose) as read:
            result = starts.verify_start(Mock(), self.start)
        self.assertTrue(result["verified"])
        self.assertEqual(read.call_count, 2)

    def test_tolerance_and_missing_telemetry(self):
        for pose in [
            {**self.pose, "position": [999, 0, 0]},
            {**self.pose, "angles": [0, 91, 0]},
        ]:
            with patch("registered_starts.read_pose", return_value=pose):
                with self.assertRaisesRegex(RuntimeError, "tolerance"):
                    starts.verify_start(Mock(), self.start, timeout=0.005)
        with patch("registered_starts.console_query", return_value="unrelated text"):
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                starts.read_pose(Mock())

    @patch("registered_starts.time.sleep")
    def test_instability_rejected(self, sleep):
        p = self.pose["position"]
        other = {**self.pose, "position": [p[0] + 0.5, *p[1:]]}
        import itertools

        with patch(
            "registered_starts.read_pose",
            side_effect=itertools.cycle([self.pose, other]),
        ):
            with self.assertRaisesRegex(RuntimeError, "stable"):
                starts.verify_start(Mock(), self.start, timeout=0.005)

    @patch("registered_starts.time.sleep")
    def test_unique_markers_chunked_reply(self, sleep):
        controller = Mock()
        marker = "SHIONN_START_test"
        controller.read_console.side_effect = [
            "old setpos 0 0 0;setang 0 0 0\n",
            marker
            + "_BEGIN\nsetpos 320 26.44379 64.03125;setang 0 90 0\n"
            + marker
            + "_EN",
            "D\n",
        ]
        with patch("registered_starts.uuid.uuid4", return_value=Mock(hex="test")):
            result = starts.read_pose(controller)
        self.assertEqual(result["position"], self.start["position"])

    @patch("registered_starts.time.sleep")
    def test_timeout_and_command_failure(self, sleep):
        c = Mock()
        c.read_console.return_value = ""
        with patch("registered_starts.time.monotonic", side_effect=[0, 0, 0, 3]):
            with self.assertRaisesRegex(RuntimeError, "Timed out"):
                starts.console_query(c, "getpos")
        c.send_command.return_value = False
        with self.assertRaisesRegex(RuntimeError, "command failed"):
            starts.console_query(c, "getpos")

    @patch("registered_starts.time.sleep")
    def test_restore_retries_and_restores_cheat_setting(self, sleep):
        with (
            patch(
                "registered_starts.console_query",
                side_effect=['"sv_cheats" = "0"', "", "", '"sv_cheats" = "0"'],
            ) as query,
            patch(
                "registered_starts.verify_start",
                side_effect=[RuntimeError("bad"), {"verified": True}],
            ),
        ):
            result = starts.restore_start(Mock(), self.start)
        self.assertEqual(result["restore_attempts"], 2)
        self.assertIn("sv_cheats 0; sv_cheats", query.call_args.args)

    @patch("registered_starts.time.sleep")
    def test_failed_restore_is_bounded_and_cleans_up(self, sleep):
        with (
            patch(
                "registered_starts.console_query",
                side_effect=['"sv_cheats" = "1"', "", "", '"sv_cheats" = "1"'],
            ) as query,
            patch(
                "registered_starts.verify_start", side_effect=RuntimeError("bad")
            ) as verify,
        ):
            with self.assertRaisesRegex(RuntimeError, "bad"):
                starts.restore_start(Mock(), self.start)
        self.assertEqual(verify.call_count, 2)
        self.assertEqual(query.call_count, 4)

    def test_config_map_and_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            starts.load_start(
                Path("configs/registered_starts_v1.json"),
                "goal_choice_interior_01",
                "wrong_map",
            )
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "starts.json"
            p.write_text(
                json.dumps(
                    {
                        "schema": "shionn_registered_starts_v1",
                        "starts": [
                            {
                                "id": "bad",
                                "map": "map",
                                "position": [float("nan"), 0, 0],
                                "angles": [0, 0, 0],
                            }
                        ],
                    }
                )
            )
            with self.assertRaises(ValueError):
                starts.load_start(p, "bad", "map")

    def test_cli_is_optional_and_implies_reload(self):
        self.assertFalse(
            record_recovery.parse_args(["--map", "goal_choice_v1"]).load_map
        )
        self.assertTrue(
            record_recovery.parse_args(
                [
                    "--map",
                    "goal_choice_v1",
                    "--registered-start",
                    "goal_choice_interior_01",
                ]
            ).load_map
        )

    def test_failed_start_never_begins_recording(self):
        with (
            patch.multiple(
                "record_recovery",
                is_game_running=Mock(return_value=True),
                EpisodeRecorder=Mock(),
                connect_controller=Mock(),
                load_map_and_wait_for_ready=Mock(),
                restore_start=Mock(side_effect=RuntimeError("bad")),
            ),
            patch("record_recovery.write_result") as result,
        ):
            with self.assertRaises(RuntimeError):
                record_recovery.main(
                    [
                        "--map",
                        "goal_choice_v1",
                        "--registered-start",
                        "goal_choice_interior_01",
                    ]
                )
            record_recovery.EpisodeRecorder.return_value.start_recording.assert_not_called()
            result.assert_called_with(None, "start_verification_failed", None)

    def test_success_records_verified_identity_and_preserves_waypoint(self):
        recorder = Mock()
        recorder.recording = False
        recorder.tracker.capture_errors.return_value = []
        recorder.stop_recording.return_value = "waypoint_reached"
        recorder.ep_dir = "/tmp/start_fixture"
        initial = {
            "verified": True,
            "start_id": self.start["id"],
            "config_sha256": self.start["config_sha256"],
        }
        with patch.multiple(
            "record_recovery",
            is_game_running=Mock(return_value=True),
            EpisodeRecorder=Mock(return_value=recorder),
            connect_controller=Mock(),
            load_map_and_wait_for_ready=Mock(),
            restore_start=Mock(return_value=initial),
            verify_start=Mock(return_value={"verified": True}),
            wait_for_camera=Mock(),
            read_portal_binding=Mock(),
            restore_portal_binding=Mock(),
            wait_for_manual_start=Mock(return_value="start"),
            wait_for_terminal_event=Mock(return_value="waypoint_reached"),
        ):
            record_recovery.main(
                [
                    "--map",
                    "goal_choice_v1",
                    "--registered-start",
                    "goal_choice_interior_01",
                    "--requested-waypoint",
                    "A",
                    "--manual-focus",
                    "--session-controls",
                    "--focus-delay",
                    "0",
                ]
            )
            metadata = recorder.start_recording.call_args.kwargs["metadata_extra"]
            self.assertTrue(
                metadata["registered_start"]["pre_recording_verification"]["verified"]
            )
            self.assertEqual(metadata["requested_waypoint"], "A")
            self.assertFalse(metadata["training_approved"])

    def test_final_pose_failure_prevents_recording(self):
        recorder = Mock()
        recorder.recording = False
        recorder.tracker.capture_errors.return_value = []
        with patch.multiple(
            "record_recovery",
            is_game_running=Mock(return_value=True),
            EpisodeRecorder=Mock(return_value=recorder),
            connect_controller=Mock(),
            load_map_and_wait_for_ready=Mock(),
            restore_start=Mock(return_value={"verified": True}),
            verify_start=Mock(side_effect=RuntimeError("moved")),
            wait_for_camera=Mock(),
            read_portal_binding=Mock(),
            restore_portal_binding=Mock(),
            wait_for_manual_start=Mock(return_value="start"),
        ):
            with self.assertRaisesRegex(RuntimeError, "moved"):
                record_recovery.main(
                    [
                        "--map",
                        "goal_choice_v1",
                        "--registered-start",
                        "goal_choice_interior_01",
                        "--manual-focus",
                        "--session-controls",
                        "--focus-delay",
                        "0",
                    ]
                )
            recorder.start_recording.assert_not_called()

    @patch("registered_starts.time.sleep")
    def test_transient_positioning_settles_before_deadline(self, sleep):
        transient = {**self.pose, "position": [320, -10, 96], "angles": [0, 91, 0]}
        with patch(
            "registered_starts.read_pose", side_effect=[transient, self.pose, self.pose]
        ) as read:
            result = starts.verify_start(Mock(), self.start)
        self.assertTrue(result["verified"])
        self.assertEqual(read.call_count, 3)
