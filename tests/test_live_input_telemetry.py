"""Capture diagnostics must preserve uncertainty and stop bad benchmarks."""

import json
import tempfile
import threading
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from benchmark_sequence import capture_review_reasons
from live_input_telemetry import CameraTelemetry, DesktopTelemetry
from run_imitation import (
    AttemptLog, MouseCaptureError, desktop_input_snapshot, mouse_capture_preflight,
)
from wrapper import Portal2Controller
from focus_guard import FocusGuardViolation, HyprlandFocusGuard
import run_imitation


class DesktopTelemetryTest(unittest.TestCase):
    def test_focus_loss_unknown_and_stale_are_distinct(self):
        sample = Mock(side_effect=[
            {"window": {"portal_focused": True}, "cursor_inside_game_window": True},
            {"window": {"portal_focused": False}, "cursor_inside_game_window": False},
            {"window": None, "cursor_inside_game_window": None},
        ])
        telemetry = DesktopTelemetry(sample, Mock())
        with patch("live_input_telemetry.time.monotonic", return_value=10):
            for _ in range(3):
                telemetry.collect()
            self.assertIsNone(telemetry.latest()["portal_focused"])
        self.assertEqual(telemetry.summary()["focus_losses"], 1)
        self.assertEqual(telemetry.summary()["unknown_focus_samples"], 1)
        self.assertEqual(telemetry.summary()["cursor_exits"], 1)
        telemetry.sample = lambda: {"window": {"portal_focused": True}}
        with patch("live_input_telemetry.time.monotonic", return_value=20):
            telemetry.collect()
        with patch("live_input_telemetry.time.monotonic", return_value=21):
            latest = telemetry.latest()
        self.assertTrue(latest["stale"])
        self.assertIsNone(latest["portal_focused"])

    def test_failed_query_records_unknown_and_collection_latency(self):
        log = Mock()
        telemetry = DesktopTelemetry(Mock(side_effect=OSError("IPC unavailable")), log)
        with patch("live_input_telemetry.time.monotonic", side_effect=[10, 10.8, 10.8]):
            record = telemetry.collect()
            latest = telemetry.latest()
        self.assertIsNone(record["portal_focused"])
        self.assertIn("IPC unavailable", record["error"])
        self.assertTrue(latest["stale"])

    @patch("run_imitation.niri_is_active", return_value=False)
    @patch("run_imitation.query_hyprland_cursor", return_value=(300, 300))
    @patch("run_imitation.query_hyprland_active_workspace", return_value={"id": 2})
    def test_unrelated_window_is_never_used_as_game_geometry(self, *_):
        terminal = {"portal_focused": False, "address": "terminal", "at": [0, 0], "size": [1000, 1000]}
        game = {"portal_focused": True, "address": "game", "at": [100, 100], "size": [800, 600]}
        state = desktop_input_snapshot(terminal, "test", game)
        self.assertEqual(state["game_window"]["address"], "game")
        self.assertTrue(state["game_window_geometry_cached"])
        self.assertIsNone(state["cursor_inside_game_window"])

    def test_log_is_valid_under_concurrent_sampling_and_ticks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            log = AttemptLog(path)
            threads = [threading.Thread(target=lambda: [log.write("tick", tick=i) for i in range(100)])
                       for _ in range(3)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            log.close()
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 300)
            self.assertTrue(all("monotonic_seconds" in row for row in records))


class CameraTelemetryTest(unittest.TestCase):
    def test_source_echo_spaces_and_crlf_are_accepted(self):
        log = Mock()
        telemetry = CameraTelemetry(Mock(), log)
        with patch("live_input_telemetry.time.monotonic", return_value=10):
            telemetry.request(1)
            marker = telemetry.pending["marker"]
            telemetry.consume(
                f"{marker}_BEGIN \r\n"
                "setpos 64.000000 -96.000000 64.031250;setang 0.000153 90.000000 0.000000\r\n"
                f"{marker}_END \r\n", 2,
            )
        record = log.write.call_args.kwargs
        self.assertEqual(record["angles"], (0.000153, 90.0, 0.0))
        self.assertIsNone(record["error"])
        self.assertIsNone(telemetry.pending)

    def test_chunked_replies_ignore_stale_angles_and_preserve_correlation(self):
        controller, log = Mock(), Mock()
        telemetry = CameraTelemetry(controller, log)
        with patch("live_input_telemetry.time.monotonic", return_value=10):
            telemetry.request(7)
            marker = telemetry.pending["marker"]
            telemetry.consume("setang 0 99 0\n" + marker + "_BE", 8)
            self.assertFalse(log.write.called)
            telemetry.consume("GIN\nsetang 0 179 0\n" + marker + "_END\nEVT|goal_reached|1\n", 9)
        record = log.write.call_args.kwargs
        self.assertEqual(record["angles"], (0, 179, 0))
        self.assertEqual(record["after_tick"], 7)
        self.assertEqual(record["received_after_tick"], 9)
        with patch("live_input_telemetry.time.monotonic", return_value=11):
            telemetry.request(10)
            marker = telemetry.pending["marker"]
            telemetry.consume(marker + "_BEGIN\nsetang 0 -179 0\n" + marker + "_END\n", 11)
        self.assertEqual(log.write.call_args.kwargs["delta_degrees"], [0, 2, 0])

    def test_timeout_late_reply_and_shutdown_do_not_invent_motion(self):
        log = Mock()
        telemetry = CameraTelemetry(Mock(), log)
        with patch("live_input_telemetry.time.monotonic", return_value=10):
            telemetry.request(1)
        with patch("live_input_telemetry.time.monotonic", return_value=11):
            telemetry.consume("", 2)
        self.assertEqual(log.write.call_args.kwargs["error"], "timeout")
        self.assertIsNone(log.write.call_args.kwargs["angles"])
        with patch("live_input_telemetry.time.monotonic", return_value=12):
            telemetry.request(3)
            marker = telemetry.pending["marker"]
        with patch("live_input_telemetry.time.monotonic", return_value=13):
            telemetry.consume(marker + "_BEGIN\nsetang 0 40 0\n" + marker + "_END\n", 4)
            self.assertEqual(log.write.call_args.kwargs["error"], "late_reply")
            telemetry.request(4)
            telemetry.close(4)
        self.assertEqual(telemetry.summary()["unknown_samples"], 2)
        self.assertIsNone(log.write.call_args.kwargs["angles"])


class MouseCapturePreflightTest(unittest.TestCase):
    def setUp(self):
        self.controller = Mock()
        self.controller.move_mouse.return_value = True
        self.monitor = Mock(stop_requested=False)
        self.monitor.wait.return_value = False
        self.log = Mock()

    @patch("run_imitation.query_hyprland_active_window", return_value={"portal_focused": True})
    @patch("run_imitation.read_player_angles", side_effect=[(0, 179, 0), (0, -178, 0), (0, -178, 0), (0, 179, 0)])
    def test_passes_opposite_turns_across_angle_wrap(self, *_):
        mouse_capture_preflight(self.controller, "test", self.monitor, self.log)
        self.assertEqual(self.controller.move_mouse.call_count, 6)
        self.assertEqual(self.log.write.call_args.kwargs["phase"], "verified")

    @patch("run_imitation.query_hyprland_active_window", return_value={"portal_focused": True})
    @patch("run_imitation.read_player_angles", side_effect=[(0, 0, 0), (0, 0, 0)])
    def test_weak_capture_aborts_before_second_direction(self, *_):
        with self.assertRaises(MouseCaptureError):
            mouse_capture_preflight(self.controller, "test", self.monitor, self.log)
        self.assertEqual(self.controller.move_mouse.call_count, 3)

    @patch("run_imitation.query_hyprland_active_window", return_value=None)
    @patch("run_imitation.read_player_angles", return_value=(0, 0, 0))
    def test_unknown_focus_never_injects(self, *_):
        with self.assertRaises(MouseCaptureError):
            mouse_capture_preflight(self.controller, "test", self.monitor, self.log)
        self.controller.move_mouse.assert_not_called()

    @patch("run_imitation.query_hyprland_active_window", return_value={"portal_focused": True})
    @patch("run_imitation.read_player_angles", return_value=(0, 0, 0))
    def test_failed_injection_and_emergency_stop_abort(self, *_):
        self.controller.move_mouse.return_value = False
        with self.assertRaises(MouseCaptureError):
            mouse_capture_preflight(self.controller, "test", self.monitor, self.log)
        self.controller.move_mouse.reset_mock()
        self.monitor.stop_requested = True
        with self.assertRaises(InterruptedError):
            mouse_capture_preflight(self.controller, "test", self.monitor, self.log)
        self.controller.move_mouse.assert_not_called()


class SubmissionTest(unittest.TestCase):
    def test_socket_failure_does_not_advance_key_state_and_is_retried(self):
        controller = Portal2Controller(log_file=None)
        controller.sock = Mock()
        controller.sock.sendall.side_effect = OSError("disconnected")
        self.assertFalse(controller.apply_action({"move_w": 1}))
        self.assertEqual(controller._policy_action_state["move_w"], 0)
        controller.sock.sendall.side_effect = None
        self.assertTrue(controller.apply_action({"move_w": 1}))
        self.assertEqual(controller._policy_action_state["move_w"], 1)
        self.assertIsNone(controller.last_action_result["game_consumed"])

    def test_uinput_failure_is_reported(self):
        controller = Portal2Controller(log_file=None)
        controller.ui = Mock()
        controller.ui.write.side_effect = OSError("device removed")
        self.assertFalse(controller.apply_action({"mouse_dx": 10}))
        self.assertIs(controller.last_action_result["mouse_sent"], False)

    def test_legacy_or_unknown_capture_is_flagged_for_review(self):
        reasons = capture_review_reasons({})
        self.assertIn("telemetry_missing", reasons)
        self.assertIn("mouse_preflight_not_passed", reasons)
        details = {"mouse_preflight": "passed", "injection_failures": 0,
                   "telemetry": {"samples": 15, "max_sample_gap_seconds": 0.08, "unknown_cursor_samples": 15},
                   "camera_telemetry": {"samples": 15, "max_sample_gap_seconds": 0.08}}
        self.assertEqual(capture_review_reasons(details), ["unknown_cursor_samples"])

    def test_confinement_without_effective_match_evidence_requires_review(self):
        details = {"focus_guard": {"confinement_requested": True},
                   "focus_guard_cleanup": {"status": "restored"}}
        self.assertIn("confinement_rule_match_unverified", capture_review_reasons(details))
        for expected in (False, True):
            details["focus_guard"]["rule_match_probe"] = {"expected": expected, "observed": expected}
            self.assertNotIn("confinement_rule_match_unverified", capture_review_reasons(details))
            details["focus_guard"]["rule_match_probe"]["observed"] = not expected
            self.assertIn("confinement_rule_match_unverified", capture_review_reasons(details))


class RunnerIntegrationTest(unittest.TestCase):
    def run_mocked(self, root, *, preflight=False, fail_preflight=False, lose_focus=False,
                   guarded=False, interrupt=False, confined=False, escape_pointer=False, shared_mouse=None):
        checkpoint = root / "best.pt"
        checkpoint.touch()
        log_path = root / "attempt.jsonl"
        argv = ["--checkpoint", str(checkpoint), "--log-file", str(log_path),
                "--countdown", "0", "--max-seconds", "1", "--keep-focused", "--verbose"]
        if preflight:
            argv += ["--mouse-preflight", "--map", "test1"]
        if guarded:
            argv += ["--hyprland-focus-guard"]
        if confined:
            argv += ["--hyprland-confine-pointer"]
        controller, policy, camera = Mock(), Mock(), Mock()
        if shared_mouse is not None:
            controller.disconnect.side_effect = lambda: Portal2Controller.disconnect(controller)
        controller.read_console.side_effect = ["", "EVT|goal_reached|1\n"]
        controller.apply_action.return_value = True
        controller.last_action_result = {"mouse_sent": True, "game_consumed": None}
        policy.device = "cpu"
        policy.mouse_bins = None
        policy.predict_with_diagnostics.return_value = ({"mouse_dx": 10, "mouse_dy": 0}, {})
        camera.get_latest_frame.return_value = np.zeros((4, 4, 3), dtype=np.uint8)
        camera.output_name = "test-output"
        monitor = Mock(stop_requested=False, session_stop_requested=False,
                       restart_requested=False, handoff_requested=False)
        monitor.wait.return_value = False
        focused = {"portal_focused": True, "class": "steam_app_620", "address": "0x123",
                   "workspace": {"id": 1}, "monitor": 0, "hyprland_instance": "test"}
        good = {"window": focused, "portal_focused": True, "cursor_inside_game_window": True}
        bad = {"window": {"portal_focused": False}, "portal_focused": False}
        if escape_pointer:
            bad = dict(good, cursor_inside_game_window=False)
        desktop = Mock()
        desktop.latest.side_effect = [good, bad if lose_focus or escape_pointer else good, good, good]
        desktop.focus_losses = 0 if guarded else int(lose_focus)
        desktop.cursor_exits = 0
        desktop.summary.return_value = {"samples": 4}
        camera_telemetry = Mock()
        camera_telemetry.summary.return_value = {"samples": 2}
        guard = Mock()
        guard.describe.return_value = {"original_follow_mouse": 1}
        guard.restore_result = {"status": "restored"}
        guard.restore.return_value = guard.restore_result
        guard_logic = HyprlandFocusGuard()
        guard_logic.window = focused
        guard_logic.confine_pointer = confined
        guard.check.side_effect = guard_logic.check
        events = []
        with ExitStack() as stack:
            for name, value in {
                "parse_args": Mock(return_value=run_imitation.parse_args(argv)),
                "is_game_running": Mock(return_value=True),
                "connect_controller": Mock(return_value=controller),
                "WaylandCamera": Mock(return_value=camera),
                "EscapeKeyMonitor": Mock(return_value=monitor),
                "prepare_portal_visual_state": Mock(return_value=focused),
                "activate_portal_input": Mock(return_value=focused),
                "query_hyprland_active_window": Mock(return_value=focused),
                "desktop_input_snapshot": Mock(return_value=good),
                "DesktopTelemetry": Mock(return_value=desktop),
                "CameraTelemetry": Mock(return_value=camera_telemetry),
                "HyprlandFocusGuard": Mock(return_value=guard),
                "get_default_output": Mock(return_value="test-output"),
                "load_recovery_map": Mock(side_effect=lambda *_: events.append("reload")),
                "mouse_capture_preflight": Mock(side_effect=MouseCaptureError("weak yaw") if fail_preflight
                                                 else lambda *_: events.append("preflight")),
            }.items():
                stack.enter_context(patch.object(run_imitation, name, value))
            stack.enter_context(patch("models.imitation.inference.PolicyInference", return_value=policy))
            policy.predict_with_diagnostics.side_effect = KeyboardInterrupt if interrupt else lambda *_: (events.append("predict") or ({"mouse_dx": 10, "mouse_dy": 0}, {}))
            if fail_preflight:
                with self.assertRaises(MouseCaptureError):
                    run_imitation.main(virtual_mouse=shared_mouse)
            elif guarded and (lose_focus or escape_pointer):
                with self.assertRaises(FocusGuardViolation):
                    run_imitation.main(virtual_mouse=shared_mouse)
            else:
                run_imitation.main(virtual_mouse=shared_mouse)
        if guarded:
            guard.activate.assert_called_once_with(focused)
            guard.restore.assert_called_once()
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        return controller, records, events

    def test_shared_virtual_mouse_survives_attempt_cleanup_and_failed_preflight(self):
        mouse = Mock()
        for fail in (False, True):
            with self.subTest(fail=fail), tempfile.TemporaryDirectory() as directory:
                controller, records, _ = self.run_mocked(
                    Path(directory), shared_mouse=mouse, preflight=True, fail_preflight=fail,
                )
            self.assertIsNone(controller.ui)
            mouse.close.assert_not_called()
            metadata = next(r for r in records if r['type'] == 'metadata')
            self.assertIs(metadata['shared_virtual_mouse'], True)

    def test_focus_lost_during_prediction_gates_input_and_final_tick_is_logged(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, records, _ = self.run_mocked(Path(directory), lose_focus=True)
        self.assertEqual(controller.apply_action.call_count, 1)
        ticks = [row for row in records if row["type"] == "tick"]
        self.assertEqual([row["action_applied"] for row in ticks], [False, True])
        self.assertIn("EVT|goal_reached", ticks[-1]["console_events"])
        self.assertEqual(records[-1]["stop_reason"], "goal_reached")

    def test_preflight_reload_precedes_inference_and_failure_sends_no_policy_actions(self):
        with tempfile.TemporaryDirectory() as directory:
            _, records, events = self.run_mocked(Path(directory), preflight=True)
        self.assertEqual(events[:3], ["preflight", "reload", "predict"])
        self.assertEqual(records[-1]["mouse_preflight"], "passed")
        with tempfile.TemporaryDirectory() as directory:
            controller, records, events = self.run_mocked(Path(directory), preflight=True, fail_preflight=True)
        controller.apply_action.assert_not_called()
        self.assertNotIn("predict", events)
        self.assertEqual(records[-1]["stop_reason"], "mouse_capture_failed")

    def test_guard_stops_focus_loss_during_prediction_and_restores_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, records, _ = self.run_mocked(Path(directory), guarded=True, lose_focus=True)
        controller.apply_action.assert_not_called()
        self.assertEqual(records[-1]["stop_reason"], "focus_guard_violation")
        self.assertEqual(records[-1]["focus_guard_cleanup"]["status"], "restored")

    def test_guard_restores_on_normal_exit_and_keyboard_interrupt(self):
        for interrupt in (False, True):
            with self.subTest(interrupt=interrupt), tempfile.TemporaryDirectory() as directory:
                _, records, _ = self.run_mocked(Path(directory), guarded=True, interrupt=interrupt)
            self.assertEqual(records[-1]["focus_guard_cleanup"]["status"], "restored")

    def test_confined_run_reloads_map_and_stops_if_pointer_leaves_during_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, records, events = self.run_mocked(
                Path(directory), guarded=True, confined=True, preflight=True, escape_pointer=True,
            )
        controller.apply_action.assert_not_called()
        self.assertEqual(events[:3], ["preflight", "reload", "predict"])
        self.assertEqual(records[-1]["stop_reason"], "focus_guard_violation")
        self.assertEqual(records[-1]["focus_guard_cleanup"]["status"], "restored")

    def test_confined_run_with_known_containment_proceeds(self):
        with tempfile.TemporaryDirectory() as directory:
            controller, records, _ = self.run_mocked(Path(directory), guarded=True, confined=True, preflight=True)
        self.assertEqual(controller.apply_action.call_count, 2)
        self.assertEqual(records[-1]["stop_reason"], "goal_reached")


if __name__ == "__main__":
    unittest.main()
