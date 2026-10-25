"""Tests for checkpoint-by-chamber job planning and sequence results."""

# Python standard library: fake subprocess runs, fixtures, and test cases.
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# Project live-sequence runner at the repository root.
from run_model_sequence import Model, build_jobs, main, parse_args, runner_command, persistent_virtual_mouse, run_persistent_attempt
from focus_guard import FOCUS_GUARD_EXIT_CODE, FocusGuardViolation


class ModelSequenceTest(unittest.TestCase):
    def test_persistent_mouse_survives_both_attempts_and_closes_once_after_sequence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            checkpoint.with_suffix('.json').touch()
            mouse = Mock()
            owner = Mock(ui=mouse)
            owner.init_virtual_mouse.return_value = True
            owner.disconnect.side_effect = mouse.close
            received = []

            def fake_attempt(command, shared):
                self.assertIs(shared, mouse)
                mouse.close.assert_not_called()
                received.append(shared)
                run_dir = Path(command[command.index('--recording-dir') + 1])
                run_dir.mkdir()
                (run_dir / 'attempt_test.jsonl').write_text(json.dumps({
                    'type': 'summary', 'stop_reason': 'time_limit', 'ticks': 120,
                }) + '\n')
                return 0

            with patch('wrapper.Portal2Controller', return_value=owner), \
                 patch('run_model_sequence.time.sleep'), \
                 patch('run_model_sequence.run_persistent_attempt', side_effect=fake_attempt), \
                 patch('run_model_sequence.subprocess.run') as spawn:
                code = main(['--checkpoint', str(checkpoint), '--map', 'test11', '--repeats', '2',
                             '--persistent-mouse', '--recording-root', str(root / 'results')])
            self.assertEqual(code, 0)
            self.assertEqual(len(received), 2)
            owner.init_virtual_mouse.assert_called_once()
            mouse.close.assert_called_once()
            spawn.assert_not_called()

    def test_persistent_mouse_owner_closes_on_error_or_interrupt(self):
        for failure in (RuntimeError('failed attempt'), KeyboardInterrupt()):
            with self.subTest(failure=type(failure).__name__):
                owner = Mock(ui=Mock())
                owner.init_virtual_mouse.return_value = True
                with patch('wrapper.Portal2Controller', return_value=owner), patch('run_model_sequence.time.sleep'):
                    with self.assertRaises(type(failure)):
                        with persistent_virtual_mouse(True):
                            raise failure
                owner.disconnect.assert_called_once()

    def test_persistent_attempt_passes_argv_and_device_and_preserves_failure_code(self):
        mouse = Mock()
        with patch('run_imitation.main') as runner:
            self.assertEqual(run_persistent_attempt(['python', 'run_imitation.py', '--map', 'test11'], mouse), 0)
            runner.assert_called_once_with(['--map', 'test11'], virtual_mouse=mouse)
        with patch('run_imitation.main', side_effect=FocusGuardViolation('lost capture')):
            self.assertEqual(run_persistent_attempt(['python', 'run_imitation.py'], mouse), FOCUS_GUARD_EXIT_CODE)

    def test_completed_sequence_writes_benchmark_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").touch()

            def fake_run(command, *, cwd, check):
                run_dir = Path(command[command.index("--recording-dir") + 1])
                run_dir.mkdir()
                (run_dir / "attempt_test.jsonl").write_text("".join(
                    json.dumps(row) + "\n" for row in (
                        {"type": "metadata", "capture": {"fps": 24}, "dry_run": False},
                        {"type": "summary", "stop_reason": "goal_reached", "ticks": 120},
                    )
                ))
                return SimpleNamespace(returncode=0)

            with patch("run_model_sequence.subprocess.run", side_effect=fake_run):
                exit_code = main([
                    "--checkpoint", str(checkpoint), "--map", "test1",
                    "--recording-root", str(root / "results"),
                ])

            self.assertEqual(exit_code, 0)
            report_path = next((root / "results").glob("sequence_*/benchmark_summary.json"))
            report = json.loads(report_path.read_text())
            self.assertEqual(report["summary"][0]["success_rate"], 1.0)

    def test_map_first_plan_repeats_every_model_chamber_pair(self):
        models = [Model("old", Path("old.pt")), Model("new", Path("new.pt"))]
        jobs = build_jobs(models, ["test1", "test2"], repeats=2, order="map-first")

        self.assertEqual(
            [(job.model.label, job.map_name, job.repeat) for job in jobs],
            [
                ("old", "test1", 1), ("new", "test1", 1),
                ("old", "test2", 1), ("new", "test2", 1),
                ("old", "test1", 2), ("new", "test1", 2),
                ("old", "test2", 2), ("new", "test2", 2),
            ],
        )

    def test_runner_command_keeps_attempts_separate_and_logs_without_video(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "best.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").touch()
            args = parse_args([
                "--checkpoint", f"candidate={checkpoint}", "--map", "dataset_test1",
                "--no-video", "--keep-focused", "--output", "DP-1",
                "--move-w-threshold", "0.2",
                "--mouse-max-abs", "28",
                "--hyprland-focus-guard",
            ])
            job = build_jobs(args.models, args.maps, 1, args.order)[0]
            run_dir = Path(temporary_directory) / "attempt"

            command = runner_command(job, run_dir, args)

            self.assertEqual(command[command.index("--checkpoint") + 1], str(checkpoint))
            self.assertEqual(command[command.index("--map") + 1], "dataset_test1")
            self.assertEqual(command[command.index("--recording-dir") + 1], str(run_dir))
            self.assertIn("--log-actions", command)
            self.assertIn("--keep-focused", command)
            self.assertIn("--mouse-preflight", command)
            self.assertIn("--hyprland-focus-guard", command)
            self.assertEqual(command[command.index("--telemetry-hz") + 1], "15.0")
            self.assertEqual(command[command.index("--move-w-threshold") + 1], "0.2")
            self.assertEqual(command[command.index("--mouse-max-abs") + 1], "28")
            self.assertNotIn("--record-video", command)

    def test_input_and_guard_failures_stop_even_with_continue_on_error(self):
        for reason in ("mouse_capture_failed", "focus_guard_violation", "focus_guard_restore_failed"):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                checkpoint = root / "best.pt"
                checkpoint.touch()
                checkpoint.with_suffix(".json").touch()

                def fail_capture(command, **_):
                    run_dir = Path(command[command.index("--recording-dir") + 1])
                    run_dir.mkdir()
                    (run_dir / "attempt_test.jsonl").write_text(json.dumps({
                        "type": "summary", "stop_reason": reason,
                    }) + "\n")
                    return SimpleNamespace(returncode=1)

                with patch("run_model_sequence.subprocess.run", side_effect=fail_capture) as run:
                    code = main(["--checkpoint", str(checkpoint), "--map", "test1", "--map", "test2",
                                 "--recording-root", str(root / "results"), "--continue-on-error"])
                self.assertEqual(code, 1)
                self.assertEqual(run.call_count, 1)

    def test_dry_run_does_not_enable_mouse_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").touch()
            args = parse_args(["--checkpoint", str(checkpoint), "--map", "test1", "--dry-run"])
            job = build_jobs(args.models, args.maps, 1, args.order)[0]
            self.assertNotIn("--mouse-preflight", runner_command(job, Path(directory), args))

    def test_confinement_implies_guard_and_requires_live_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").touch()
            argv = ["--checkpoint", str(checkpoint), "--map", "test11", "--hyprland-confine-pointer"]
            args = parse_args(argv)
            self.assertTrue(args.hyprland_focus_guard)
            job = build_jobs(args.models, args.maps, 1, args.order)[0]
            command = runner_command(job, Path(directory), args)
            self.assertIn("--hyprland-confine-pointer", command)
            self.assertIn("--hyprland-focus-guard", command)
            for flag in ("--dry-run", "--no-mouse-preflight"):
                with self.subTest(flag=flag), self.assertRaises(SystemExit):
                    parse_args(argv + [flag])

    def test_early_guard_setup_failure_stops_sequence_without_an_attempt_log(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").touch()
            with patch("run_model_sequence.subprocess.run", return_value=SimpleNamespace(returncode=3)) as run:
                result = main([
                    "--checkpoint", str(checkpoint), "--map", "test1", "--repeats", "2",
                    "--continue-on-error", "--recording-root", str(Path(directory) / "results"),
                ])
            self.assertEqual(result, 1)
            self.assertEqual(run.call_count, 1)

    def test_escape_in_one_attempt_stops_the_rest_of_the_sequence(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            checkpoint = root / "best.pt"
            checkpoint.touch()
            checkpoint.with_suffix(".json").touch()
            calls = []

            def fake_run(command, *, cwd, check):
                run_dir = Path(command[command.index("--recording-dir") + 1])
                run_dir.mkdir()
                reason = "episode_failed" if not calls else "escape_key"
                records = [
                    {"type": "metadata", "video": str(run_dir / "attempt.mp4")},
                    {"type": "summary", "stop_reason": reason},
                ]
                (run_dir / "attempt_test.jsonl").write_text(
                    "".join(json.dumps(record) + "\n" for record in records)
                )
                calls.append(command)
                return SimpleNamespace(returncode=0)

            with patch("run_model_sequence.subprocess.run", side_effect=fake_run):
                exit_code = main([
                    "--checkpoint", str(checkpoint),
                    "--map", "test1", "--map", "test2", "--map", "test3",
                    "--recording-root", str(root / "results"), "--pause-seconds", "0",
                ])

            self.assertEqual(exit_code, 130)
            self.assertEqual(len(calls), 2)
            manifest = next((root / "results").glob("sequence_*/sequence.jsonl"))
            records = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual([row["stop_reason"] for row in records[1:]], ["episode_failed", "escape_key"])


if __name__ == "__main__":
    unittest.main()
