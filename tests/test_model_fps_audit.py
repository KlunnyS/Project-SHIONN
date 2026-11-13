"""Synthetic log tests for reporting decision and control timing separately."""
import json
import tempfile
import unittest
from pathlib import Path
from audit_model_fps import summarize


class RateAuditTest(unittest.TestCase):
    def test_distinguishes_decisions_from_held_actions(self):
        rows=[{'type':'metadata','capture':{'fps':24,'model_fps':8}}]
        for i in range(7):
            refreshed=i%3==0
            rows.append({'type':'tick','elapsed_seconds':i/24,
                         'action_applied':True,
                         'policy':{'decision_rate':{'refreshed':refreshed}},
                         'frame_timing':{'inference_started_monotonic':i if refreshed else None,
                                         'inference_finished_monotonic':i+.01 if refreshed else None}})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempt.jsonl'
            path.write_text('\n'.join(json.dumps(row) for row in rows))
            report=summarize(path)
        self.assertEqual(report['model_decisions'],3)
        self.assertEqual(report['held_ticks'],4)
        self.assertAlmostEqual(report['observed_control_hz'],24)
        self.assertAlmostEqual(report['observed_decision_hz'],8)
        self.assertAlmostEqual(report['median_inference_ms'],10)
        self.assertEqual(report['held_ticks_claiming_inference'],0)

    def test_empty_attempt_does_not_invent_rates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempt.jsonl'
            path.write_text(json.dumps({'type':'metadata','capture':{'fps':24}}))
            report=summarize(path)
        self.assertIsNone(report['observed_control_hz'])
        self.assertIsNone(report['observed_decision_hz'])
        self.assertEqual(report['configured_model_fps'],24)
