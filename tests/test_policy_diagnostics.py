import json
import tempfile
import unittest
from pathlib import Path

from analyze_mouse_rollout import analyze_attempt, axis_distribution
from benchmark_jump import score_jump_rows


class PolicyDiagnosticsTest(unittest.TestCase):
    def test_wall_report_separates_opposing_peaks_from_flat_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attempt.jsonl"
            records = [{
                "type": "metadata", "map": "dataset_test9",
                "mouse_bins": {
                    "dx": {"representatives": [-10, 0, 10]},
                    "dy": {"representatives": [-3, 0, 3]},
                },
            }]
            for tick, dx in enumerate(([0.49, 0.02, 0.49], [1 / 3] * 3, [0, 1, 0]), start=1):
                records.append({
                    "type": "tick", "tick": tick, "elapsed_seconds": float(tick),
                    "policy": {
                        "mouse_bin_probabilities": {"dx": dx, "dy": [0, 1, 0]},
                        "binary_probabilities": {"jump": 0.8},
                    },
                    "action": {"mouse_dx": -10, "mouse_dy": 0, "jump": tick != 3},
                })
            path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
            report, rows = analyze_attempt(path, [(1.0, 2.0)])

            self.assertEqual(report["wall_frames"], 2)
            self.assertTrue(rows[0]["dx_opposing_top_bins"])
            self.assertAlmostEqual(rows[0]["dx_peak_gap"], 0.0)
            self.assertAlmostEqual(rows[1]["dx_normalized_entropy"], 1.0)
            self.assertEqual(report["jump_action_rate_wall"], 1.0)
            self.assertEqual(report["jump_action_rate_other"], 0.0)

    def test_probability_validation_and_jump_precision_recall(self):
        with self.assertRaises(ValueError):
            axis_distribution([float("nan"), 0, 1], [-1, 0, 1])
        rows = [
            {"map": "dataset_test9", "expected_jump": "1", "prob_jump": "0.8"},
            {"map": "dataset_test9", "expected_jump": "1", "prob_jump": "0.3"},
            {"map": "dataset_test9", "expected_jump": "0", "prob_jump": "0.6"},
        ]
        scores = score_jump_rows(rows, (0.5,))
        overall = next(row for row in scores if row["map"] == "overall")
        self.assertEqual((overall["true_positive"], overall["false_positive"], overall["false_negative"]), (1, 1, 1))
        self.assertEqual(overall["precision"], 0.5)
        self.assertEqual(overall["recall"], 0.5)


if __name__ == "__main__":
    unittest.main()
