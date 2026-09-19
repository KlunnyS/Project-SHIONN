import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_model_sequence import Model, build_jobs, main, parse_args, runner_command


class ModelSequenceTest(unittest.TestCase):
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
            ])
            job = build_jobs(args.models, args.maps, 1, args.order)[0]
            run_dir = Path(temporary_directory) / "attempt"

            command = runner_command(job, run_dir, args)

            self.assertEqual(command[command.index("--checkpoint") + 1], str(checkpoint))
            self.assertEqual(command[command.index("--map") + 1], "dataset_test1")
            self.assertEqual(command[command.index("--recording-dir") + 1], str(run_dir))
            self.assertIn("--log-actions", command)
            self.assertIn("--keep-focused", command)
            self.assertNotIn("--record-video", command)

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
