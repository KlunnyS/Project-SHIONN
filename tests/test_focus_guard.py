"""Temporary compositor settings must roll back and never chase focus."""

import json
from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from focus_guard import FocusGuardViolation, HyprlandFocusGuard


WINDOW = {"portal_focused": True, "address": "0x123", "class": "gamescope",
          "workspace": {"id": 1}, "monitor": 0, "hyprland_instance": "test-instance"}


class FocusGuardTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.guard = HyprlandFocusGuard(Path(self.directory.name))
        self.addCleanup(self.guard.restore)
        self.value = 1
        self.calls = []
        self.rule_values = {}
        self.tags = []
        self.original_window_no_follow_mouse = False

        def run(*args):
            self.calls.append(args)
            if args[0] == "getoption":
                return json.dumps({"int": self.value})
            if args[0] == "clients":
                return json.dumps([dict(WINDOW, tags=list(self.tags))])
            if args[0] == "getprop":
                # Hyprland's TagKeeper uses exact names, not regex matching.
                name = self.guard.confinement_rule
                key = f"windowrule[{name}]:"
                matches = self.rule_values.get(key + "match:tag") in self.tags
                enabled = self.rule_values.get(key + "enable") == "1"
                value = self.original_window_no_follow_mouse
                if matches and enabled:
                    value = self.rule_values.get(key + "no_follow_mouse") == "on"
                return json.dumps({"no_follow_mouse": value})
            if args[0] == "dispatch":
                tag, address = args[2].split()
                self.assertEqual(address, "address:0x123")
                if tag.startswith("+"):
                    self.tags.append(tag[1:])
                elif tag[1:] in self.tags:
                    self.tags.remove(tag[1:])
            elif args[1] == "input:follow_mouse":
                self.value = int(args[2])
            else:
                self.rule_values[args[1]] = args[2]
            return "ok"

        self.guard._run = Mock(side_effect=run)

    def enable_confinement(self):
        self.guard.confine_pointer = True
        self.guard.confinement_rule = "shionn_cursor_test"

    def test_confinement_targets_one_window_and_is_disabled_before_hover_restoration(self):
        self.enable_confinement()
        self.guard.activate(WINDOW)
        self.assertEqual(self.tags, ["shionn_cursor_test"])
        self.assertEqual(self.rule_values["windowrule[shionn_cursor_test]:enable"], "1")
        self.assertEqual(self.rule_values["windowrule[shionn_cursor_test]:match:tag"], "shionn_cursor_test")
        self.assertEqual(self.rule_values["windowrule[shionn_cursor_test]:confine_pointer"], "on")
        note = json.loads(next(Path(self.directory.name).glob("*.json")).read_text())
        self.assertEqual(len(note["confinement_cleanup_commands"]), 2)
        self.guard.restore()
        self.assertEqual(self.tags, [])
        self.assertEqual(self.rule_values["windowrule[shionn_cursor_test]:enable"], "0")
        self.assertEqual(self.value, 1)
        self.assertEqual(self.guard.rule_match_probe["expected"], self.guard.rule_match_probe["observed"])
        self.assertFalse(self.guard._read_window_hover_property())
        disable = max(i for i, call in enumerate(self.calls) if call == (
            "keyword", "windowrule[shionn_cursor_test]:enable", "0"
        ))
        restore = self.calls.index(("keyword", "input:follow_mouse", "1"))
        self.assertLess(disable, restore)

    def test_accepted_rule_with_regex_tag_is_rejected_before_policy_input(self):
        self.enable_confinement()
        original = self.guard._run.side_effect

        def wrong_match(*args):
            if args[:2] == ("keyword", "windowrule[shionn_cursor_test]:match:tag"):
                args = (*args[:2], "^shionn_cursor_test$")
            return original(*args)

        self.guard._run.side_effect = wrong_match
        with self.assertRaisesRegex(RuntimeError, "did not match"):
            self.guard.activate(WINDOW)
        self.assertEqual(self.tags, ["shionn_cursor_test"])
        self.guard.restore()
        self.assertEqual(self.tags, [])
        self.assertEqual(self.value, 1)

    def test_probe_checks_a_change_when_window_property_was_already_true(self):
        self.enable_confinement()
        self.original_window_no_follow_mouse = True
        self.guard.activate(WINDOW)
        self.assertFalse(self.guard.rule_match_probe["expected"])
        self.assertFalse(self.guard._read_window_hover_property())
        self.guard.restore()
        self.assertTrue(self.guard._read_window_hover_property())

    def test_failed_confinement_enable_rolls_back_tag_and_hover_setting(self):
        self.enable_confinement()
        original = self.guard._run.side_effect

        def reject_enable(*args):
            result = original(*args)
            return "rule rejected" if args == ("keyword", "windowrule[shionn_cursor_test]:enable", "1") else result

        self.guard._run.side_effect = reject_enable
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.guard.activate(WINDOW)
        self.guard.restore()
        self.assertEqual(self.tags, [])
        self.assertEqual(self.value, 1)
        self.assertEqual(self.rule_values["windowrule[shionn_cursor_test]:enable"], "0")

    def test_missing_tag_is_not_treated_as_successful_confinement(self):
        self.enable_confinement()
        original = self.guard._run.side_effect
        # Model a compositor that acknowledges tagging without attaching it.
        self.guard._run.side_effect = lambda *args: "ok" if args[:2] == ("dispatch", "tagwindow") else original(*args)
        with self.assertRaisesRegex(RuntimeError, "verify the confinement tag"):
            self.guard.activate(WINDOW)
        self.guard.restore()
        self.assertEqual(self.tags, [])
        self.assertEqual(self.value, 1)

    def test_cleanup_attempts_tag_removal_and_hover_restore_even_if_rule_disable_fails(self):
        self.enable_confinement()
        self.guard.activate(WINDOW)
        original = self.guard._run.side_effect
        self.guard._run.side_effect = lambda *args: "failed" if args == (
            "keyword", "windowrule[shionn_cursor_test]:enable", "0"
        ) else original(*args)
        self.assertEqual(self.guard.restore()["status"], "failed")
        self.assertEqual(self.tags, [])
        self.assertEqual(self.value, 1)
        self.guard.rule_touched = False
        self.guard.changed = False

    def test_confinement_stops_on_unknown_containment_or_observed_exit(self):
        self.enable_confinement()
        self.guard.window = WINDOW
        good = {"portal_focused": True, "window": WINDOW, "cursor_inside_game_window": True}
        self.guard.check(good)
        for inside in (False, None):
            with self.subTest(inside=inside), self.assertRaises(FocusGuardViolation):
                self.guard.check(dict(good, cursor_inside_game_window=inside))
        with self.assertRaises(FocusGuardViolation):
            self.guard.check(good, cursor_exits=1)

    def test_destroyed_window_does_not_prevent_hover_restoration(self):
        self.enable_confinement()
        self.guard.activate(WINDOW)
        original = self.guard._run.side_effect

        def closed(*args):
            if args[0] == "clients":
                return "[]"
            if args[0] == "dispatch":
                return "Window not found"
            return original(*args)

        self.guard._run.side_effect = closed
        self.assertEqual(self.guard.restore()["status"], "restored")
        self.assertEqual(self.value, 1)

    def test_restores_previous_preference_and_clears_recovery_note(self):
        self.guard.activate(WINDOW)
        self.assertEqual(self.value, 0)
        state = next(Path(self.directory.name).glob("*.json"))
        self.assertEqual(json.loads(state.read_text())["original_follow_mouse"], 1)
        self.guard.restore()
        self.assertEqual(self.value, 1)
        self.assertEqual(state.read_text(), "")
        self.assertEqual(self.guard.restore_result["status"], "restored")

    def test_existing_zero_needs_no_mutation(self):
        self.value = 0
        self.guard.activate(WINDOW)
        self.guard.restore()
        self.assertFalse(any(call[0] == "keyword" for call in self.calls))

    def test_compositor_rejection_rolls_back_ambiguous_write(self):
        original = self.guard._run.side_effect

        def fail_once(*args):
            result = original(*args)
            if args == ("keyword", "input:follow_mouse", "0"):
                raise subprocess.TimeoutExpired("hyprctl", 2)
            return result

        self.guard._run.side_effect = fail_once
        with self.assertRaises(subprocess.TimeoutExpired):
            self.guard.activate(WINDOW)
        self.assertEqual(self.value, 0)
        self.guard.restore()
        self.assertEqual(self.value, 1)

    def test_zero_exit_code_with_error_text_is_rejected(self):
        original = self.guard._run.side_effect
        self.guard._run.side_effect = lambda *args: "invalid option" if args[0] == "keyword" else original(*args)
        with self.assertRaisesRegex(RuntimeError, "rejected"):
            self.guard.activate(WINDOW)

    def test_cannot_own_same_session_twice(self):
        self.guard.activate(WINDOW)
        other = HyprlandFocusGuard(Path(self.directory.name))
        other._run = Mock()
        with self.assertRaisesRegex(RuntimeError, "Another"):
            other.activate(WINDOW)
        other._run.assert_not_called()
        other.restore()
        self.assertEqual(self.value, 0)

    def test_preserves_explicit_preference_change_during_run(self):
        self.guard.activate(WINDOW)
        self.value = 3
        self.assertEqual(self.guard.restore()["status"], "external_change_preserved")
        self.assertEqual(self.value, 3)

    def test_restore_failure_keeps_recovery_note(self):
        self.guard.activate(WINDOW)
        self.guard._run.side_effect = OSError("compositor unavailable")
        result = self.guard.restore()
        self.assertEqual(result["status"], "failed")
        note = next(Path(self.directory.name).glob("*.json"))
        self.assertTrue(json.loads(note.read_text())["active"])
        self.assertEqual(result["restore_command"][-1], "1")
        self.guard.changed = False  # Avoid repeating a simulated outage in cleanup.

    def test_stale_recovery_note_survives_refused_start(self):
        self.guard.activate(WINDOW)
        note = next(Path(self.directory.name).glob("*.json"))
        self.guard.lock.close()  # Simulate kill: OS releases lock, original setting remains zero.
        self.guard.lock = None
        self.guard.changed = False
        previous = note.read_text()
        other = HyprlandFocusGuard(Path(self.directory.name))
        other._run = Mock(return_value=json.dumps({"int": 0}))
        with self.assertRaisesRegex(RuntimeError, "Unrestored"):
            other.activate(WINDOW)
        other.restore()
        self.assertEqual(note.read_text(), previous)

    def test_changes_in_window_workspace_monitor_or_focus_stop(self):
        self.guard.window = WINDOW
        good = {"portal_focused": True, "window": WINDOW}
        self.guard.check(good)
        for key, value in (("address", "0x456"), ("workspace", {"id": 2}), ("monitor", 1)):
            with self.subTest(key=key), self.assertRaises(FocusGuardViolation):
                self.guard.check({"portal_focused": True, "window": dict(WINDOW, **{key: value})})
        for focused in (None, False):
            with self.subTest(focused=focused), self.assertRaises(FocusGuardViolation):
                self.guard.check(dict(good, portal_focused=focused))
        # A brief focus loss counts even when focus is already back by this tick.
        with self.assertRaises(FocusGuardViolation):
            self.guard.check(good, focus_losses=1)

    def test_unknown_original_preference_is_never_overwritten(self):
        self.guard._run.side_effect = None
        self.guard._run.return_value = "{}"
        with self.assertRaises(RuntimeError):
            self.guard.activate(WINDOW)
        self.assertFalse(self.guard.changed)

    def test_real_subprocess_arguments_protect_negative_tag_from_flag_parsing(self):
        guard = HyprlandFocusGuard()
        guard.instance = "test-instance"
        with patch("focus_guard.subprocess.run", return_value=Mock(stdout="ok\n")) as run:
            guard._run("dispatch", "tagwindow", "-shionn_cursor_test address:0x123")
        self.assertEqual(run.call_args.args[0], [
            "hyprctl", "-i", "test-instance", "dispatch", "tagwindow -shionn_cursor_test address:0x123",
        ])

    @unittest.skipUnless(shutil.which("hyprctl"), "Hyprland CLI not installed")
    def test_installed_cli_accepts_negative_tag_payload_without_touching_a_live_session(self):
        result = subprocess.run([
            "hyprctl", "-i", "shionn-nonexistent-test-instance", "dispatch",
            "tagwindow -shionn_cursor_test address:0x123",
        ], capture_output=True, text=True, timeout=3)
        self.assertNotIn("usage: hyprctl", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
