"""Safety tests for the SHIONN Dash command builder."""

import sys
import unittest
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "dashboard"
sys.path.insert(0, str(APP_DIR))

from backend.project import ROOT, build_command, project_status  # noqa: E402


class DashboardBackendTest(unittest.TestCase):
    def test_status_uses_frozen_v3_counts(self):
        status = project_status()
        self.assertEqual(status["split_id"], "frozen-2026-10-18-v3")
        self.assertEqual(status["train"], 1151)
        self.assertEqual(status["validation"], 200)
        self.assertEqual(status["evaluation"], 0)

    def test_recording_only_allows_training_chambers(self):
        with self.assertRaisesRegex(ValueError, "restricted to training maps"):
            build_command("record", {"mode": "demonstration", "maps": ["dataset_test11"]})

    def test_proactive_turn_command_uses_recorder_flags(self):
        command, label = build_command("record", {
            "mode": "proactive_turn", "maps": ["dataset_test1"],
            "episodes": 2, "duration": 60, "direction": "left",
        })
        self.assertIn("--proactive-turn", command)
        self.assertEqual(command[command.index("--turn-direction") + 1], "left")
        self.assertIn("record_dataset.py", command[1])
        self.assertIn("Dataset recording", label)

    def test_live_benchmark_is_held_out_and_no_launch(self):
        checkpoints = list((ROOT / "models/imitation/checkpoints").glob("runs/*/best.pt"))
        if not checkpoints:
            self.skipTest("No project checkpoint available")
        relpath = str(checkpoints[0].relative_to(ROOT))
        command, _ = build_command("benchmark", {
            "mode": "live", "checkpoint": relpath, "maps": ["evaluation1"],
            "repeats": 1, "max_seconds": 60, "device": "auto",
        })
        self.assertIn("--no-launch", command)
        self.assertIn("--keep-focused", command)
        with self.assertRaisesRegex(ValueError, "validation/evaluation"):
            build_command("benchmark", {
                "mode": "live", "checkpoint": relpath, "maps": ["dataset_test1"],
                "repeats": 1, "max_seconds": 60, "device": "auto",
            })

    def test_training_command_uses_split_v3_and_both_distinct_caches(self):
        command, _ = build_command("train", {
            "epochs": 20, "device": "auto", "run_name": "interface_test_candidate",
            "correction_fraction": 0.1,
            "extra_caches": ["recovery", "proactive_turn"],
        })
        self.assertIn("data/splits/frozen_v3.json", command)
        self.assertEqual(command.count("--train-extra-cache-dir"), 2)
        self.assertNotIn("shell=True", command)
        self.assertIn("models/imitation/checkpoints/runs/interface_test_candidate", command)

    def test_training_rejects_shell_syntax_in_run_name(self):
        with self.assertRaisesRegex(ValueError, "Run name"):
            build_command("train", {
                "run_name": "okay;rm", "extra_caches": ["recovery", "proactive_turn"],
            })


if __name__ == "__main__":
    unittest.main()
