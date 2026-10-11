"""Tests for the model-to-human recovery session handoff."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_recovery_session import main, model_command, parse_args, recovery_command


class RecoverySessionTest(unittest.TestCase):
    def test_commands_keep_model_limit_and_leave_human_unlimited(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            args = parse_args(["--checkpoint", str(checkpoint)])
            model = model_command(args, "dataset_test9", root / "attempt")
            recovery = recovery_command(
                args, "dataset_test9", root / "attempt.jsonl", root / "result.json"
            )
            self.assertEqual(model[model.index("--max-seconds") + 1], "60.0")
            self.assertNotIn("--max-chamber-age-seconds", model)
            self.assertIn("--disable-chamber-timeout", model)
            self.assertEqual(recovery[recovery.index("--max-seconds") + 1], "0.0")
            self.assertIn("--disable-chamber-timeout", recovery)
            self.assertIn("--keep-focused", recovery)
            self.assertIn("--session-controls", recovery)

    def test_manual_focus_flag_reaches_recovery_only(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            args = parse_args(["--checkpoint", str(checkpoint), "--manual-focus"])
            self.assertNotIn("--manual-focus", model_command(args, "dataset_test1", root / "attempt"))
            self.assertIn("--manual-focus", recovery_command(
                args, "dataset_test1", root / "attempt.jsonl", root / "result.json",
            ))

    def test_test_mode_omits_media_and_uses_separate_session_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            args = parse_args(["--checkpoint", str(checkpoint), "--test-mode"])
            model = model_command(args, "dataset_test1", root / "attempt")
            recovery = recovery_command(
                args, "dataset_test1", root / "attempt.jsonl", root / "result.json",
            )
            self.assertEqual(args.session_root, Path("model_attempts/recovery_tests"))
            self.assertNotIn("--record-video", model)
            self.assertIn("--log-actions", model)
            self.assertIn("--test-mode", recovery)

    def test_test_mode_goal_counts_without_episode_path(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()

            def run(command, **_kwargs):
                if command[1].endswith("run_imitation.py"):
                    attempt_dir = Path(command[command.index("--recording-dir") + 1])
                    attempt_dir.mkdir(parents=True)
                    (attempt_dir / "attempt_test.jsonl").write_text(
                        json.dumps({"type": "summary", "stop_reason": "recovery_handoff"}) + "\n",
                        encoding="utf-8",
                    )
                else:
                    Path(command[command.index("--result-file") + 1]).write_text(
                        json.dumps({"outcome": "goal_reached", "episode": None}),
                        encoding="utf-8",
                    )
                return type("Result", (), {"returncode": 0})()

            with patch("run_recovery_session.subprocess.run", side_effect=run):
                result = main([
                    "--checkpoint", str(checkpoint), "--test-mode",
                    "--target-corrections", "1", "--session-root", str(root / "tests"),
                ])
            self.assertEqual(result, 0)
            manifest = next((root / "tests").glob("session_*/session.jsonl"))
            entries = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertTrue(entries[0]["test_mode"])
            self.assertIsNone(entries[-1]["episode"])

    def test_rejects_validation_chambers(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            checkpoint.touch()
            with self.assertRaises(SystemExit):
                parse_args(["--checkpoint", str(checkpoint), "--map", "dataset_test11"])

    def test_p_handoff_records_correction_then_loads_next_map(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            calls = []

            def run(command, **_kwargs):
                calls.append(command)
                if command[1].endswith("run_imitation.py"):
                    attempt_dir = Path(command[command.index("--recording-dir") + 1])
                    attempt_dir.mkdir(parents=True)
                    (attempt_dir / "attempt_test.jsonl").write_text(
                        json.dumps({"type": "summary", "stop_reason": "recovery_handoff"}) + "\n",
                        encoding="utf-8",
                    )
                else:
                    result_file = Path(command[command.index("--result-file") + 1])
                    result_file.write_text(json.dumps({
                        "outcome": "goal_reached", "episode": str(root / "episode_test"),
                    }), encoding="utf-8")
                return type("Result", (), {"returncode": 0})()

            with patch("run_recovery_session.subprocess.run", side_effect=run):
                result = main([
                    "--checkpoint", str(checkpoint), "--map", "dataset_test1",
                    "--map", "dataset_test2", "--target-corrections", "2",
                    "--session-root", str(root / "sessions"),
                    "--episodes-root", str(root / "episodes"),
                ])

            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 4)
            self.assertEqual(
                [calls[index][calls[index].index("--map") + 1] for index in (0, 2)],
                ["dataset_test1", "dataset_test2"],
            )
            self.assertIn("--recovery-handoff", calls[0])
            self.assertEqual(
                calls[1][calls[1].index("--source-attempt") + 1],
                str(Path(calls[0][calls[0].index("--recording-dir") + 1]) / "attempt_test.jsonl"),
            )

    def test_interrupted_recovery_does_not_advance(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            calls = []

            def run(command, **_kwargs):
                calls.append(command)
                if command[1].endswith("run_imitation.py"):
                    attempt_dir = Path(command[command.index("--recording-dir") + 1])
                    attempt_dir.mkdir(parents=True)
                    (attempt_dir / "attempt_test.jsonl").write_text(
                        json.dumps({"type": "summary", "stop_reason": "recovery_handoff"}) + "\n",
                        encoding="utf-8",
                    )
                else:
                    Path(command[command.index("--result-file") + 1]).write_text(
                        json.dumps({"outcome": "interrupted", "episode": "discarded"}),
                        encoding="utf-8",
                    )
                return type("Result", (), {"returncode": 0})()

            with patch("run_recovery_session.subprocess.run", side_effect=run):
                result = main([
                    "--checkpoint", str(checkpoint), "--target-corrections", "2",
                    "--session-root", str(root / "sessions"),
                ])

            self.assertEqual(result, 0)
            self.assertEqual(len(calls), 2)

    def test_model_finishes_during_handoff_and_same_map_restarts_without_correction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            model_maps = []
            recovery_count = 0

            def run(command, **_kwargs):
                nonlocal recovery_count
                if command[1].endswith("run_imitation.py"):
                    map_name = command[command.index("--map") + 1]
                    model_maps.append(map_name)
                    attempt_dir = Path(command[command.index("--recording-dir") + 1])
                    attempt_dir.mkdir(parents=True)
                    (attempt_dir / "attempt_test.jsonl").write_text(
                        json.dumps({"type": "summary", "stop_reason": "recovery_handoff"}) + "\n",
                        encoding="utf-8",
                    )
                else:
                    recovery_count += 1
                    outcome = "model_chamber_finished" if recovery_count == 1 else "goal_reached"
                    Path(command[command.index("--result-file") + 1]).write_text(
                        json.dumps({"outcome": outcome, "episode": None}), encoding="utf-8",
                    )
                return type("Result", (), {"returncode": 0})()

            with patch("run_recovery_session.subprocess.run", side_effect=run):
                result = main([
                    "--checkpoint", str(checkpoint), "--map", "dataset_test1",
                    "--map", "dataset_test2", "--target-corrections", "1",
                    "--session-root", str(root / "sessions"),
                ])
            self.assertEqual(result, 0)
            self.assertEqual(model_maps, ["dataset_test1", "dataset_test1"])
            manifest = next((root / "sessions").glob("session_*/session.jsonl"))
            entries = [json.loads(line) for line in manifest.read_text().splitlines()]
            outcomes = [entry["outcome"] for entry in entries if entry["type"] == "recovery"]
            self.assertEqual(outcomes, ["model_chamber_finished", "goal_reached"])

    def test_model_goal_without_handoff_restarts_same_chamber(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            model_maps = []

            def run(command, **_kwargs):
                if command[1].endswith("run_imitation.py"):
                    model_maps.append(command[command.index("--map") + 1])
                    attempt_dir = Path(command[command.index("--recording-dir") + 1])
                    attempt_dir.mkdir(parents=True)
                    stop_reason = "goal_reached" if len(model_maps) == 1 else "recovery_handoff"
                    (attempt_dir / "attempt_test.jsonl").write_text(
                        json.dumps({"type": "summary", "stop_reason": stop_reason}) + "\n",
                        encoding="utf-8",
                    )
                else:
                    Path(command[command.index("--result-file") + 1]).write_text(
                        json.dumps({"outcome": "goal_reached", "episode": str(root / "episode")}),
                        encoding="utf-8",
                    )
                return type("Result", (), {"returncode": 0})()

            with patch("run_recovery_session.subprocess.run", side_effect=run):
                result = main([
                    "--checkpoint", str(checkpoint), "--map", "dataset_test1",
                    "--map", "dataset_test2", "--target-corrections", "1",
                    "--session-root", str(root / "sessions"),
                ])
            self.assertEqual(result, 0)
            self.assertEqual(model_maps, ["dataset_test1", "dataset_test1"])

    def test_timeout_and_r_retry_same_chamber_before_advancing(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            calls = []
            model_reasons = iter(["time_limit", "restart_requested", "recovery_handoff",
                                  "recovery_handoff", "goal_reached"])
            recovery_outcomes = iter(["restart_requested", "goal_reached"])

            def run(command, **_kwargs):
                calls.append(command)
                if command[1].endswith("run_imitation.py"):
                    attempt_dir = Path(command[command.index("--recording-dir") + 1])
                    attempt_dir.mkdir(parents=True)
                    (attempt_dir / "attempt_test.jsonl").write_text(
                        json.dumps({"type": "summary", "stop_reason": next(model_reasons)}) + "\n",
                        encoding="utf-8",
                    )
                else:
                    result_file = Path(command[command.index("--result-file") + 1])
                    result_file.write_text(json.dumps({
                        "outcome": next(recovery_outcomes), "episode": None,
                    }), encoding="utf-8")
                return type("Result", (), {"returncode": 0})()

            with patch("run_recovery_session.subprocess.run", side_effect=run):
                result = main([
                    "--checkpoint", str(checkpoint), "--map", "dataset_test1",
                    "--map", "dataset_test2", "--target-corrections", "1",
                    "--session-root", str(root / "sessions"),
                ])

            self.assertEqual(result, 0)
            model_maps = [
                command[command.index("--map") + 1] for command in calls
                if command[1].endswith("run_imitation.py")
            ]
            self.assertEqual(model_maps, ["dataset_test1"] * 4)
            self.assertEqual(len(calls), 6)

    def test_k_stops_session_during_model_attempt(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()

            def run(command, **_kwargs):
                attempt_dir = Path(command[command.index("--recording-dir") + 1])
                attempt_dir.mkdir(parents=True)
                (attempt_dir / "attempt_test.jsonl").write_text(
                    json.dumps({"type": "summary", "stop_reason": "session_stop"}) + "\n",
                    encoding="utf-8",
                )
                return type("Result", (), {"returncode": 0})()

            with patch("run_recovery_session.subprocess.run", side_effect=run):
                result = main([
                    "--checkpoint", str(checkpoint),
                    "--session-root", str(root / "sessions"),
                ])

            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
