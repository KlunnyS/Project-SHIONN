"""CPU synthetic protocol checks; no real policy training or game access."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import run_recovery_history_probe as history
import run_recovery_probe as base
from models.imitation.checkpoint import save_checkpoint
from test_recovery_probe import TinyDataset, TinyPolicy, config


class HistoryProbeTests(unittest.TestCase):
    def test_repeats_newest_rgb_without_mutating_input_or_label(self):
        frames = np.arange(48, dtype=np.uint8).reshape(12, 2, 2)
        target = np.arange(10)
        original = frames.copy()
        ds = history.RepeatedHistoryDataset([(frames, target)], [0, 0], [True, False])
        repeated, label = ds[0]
        np.testing.assert_array_equal(repeated, np.tile(original[-3:], (4, 1, 1)))
        self.assertIs(label, target)
        self.assertIs(ds[1][0], frames)
        np.testing.assert_array_equal(frames, original)

    def test_mask_is_per_draw_recovery_only_and_does_not_consume_global_rng(self):
        labels = np.array(["base", "recovery", "proactive_turn"])
        draws = np.tile(np.arange(3), 100)
        torch_before = torch.get_rng_state().clone()
        numpy_before = np.random.get_state()
        mask = history.repetition_mask(draws, labels, .5, 0)
        self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
        after = np.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        np.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
        self.assertFalse(mask[labels[draws] != "recovery"].any())
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.sum(), 100)
        np.testing.assert_array_equal(mask, history.repetition_mask(draws, labels, .5, 0))

    def test_invalid_budget_and_probability_are_rejected(self):
        for args in (["--repeat-probability", "0"], ["--repeat-probability", "nan"],
                     ["--time-budget-seconds", "601"], ["--time-budget-seconds", "inf"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                history.parse_args(args)

    def test_cache_guard_rejects_newer_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "metadata.json"
            p.write_text("{}")
            with patch.object(history, "discover_cached_episodes", return_value=[]):
                with self.assertRaisesRegex(ValueError, "Cache changed"):
                    history.cache_fingerprint({"base": tmp}, p.stat().st_mtime_ns - 1)
                result = history.cache_fingerprint({"base": tmp}, p.stat().st_mtime_ns)
                self.assertEqual(result["files"], 1)
                self.assertFalse(result["historical_full_content_hashes_available"])

    def make_reference(self, root):
        args = base.parse_args(["--execute", "--device", "cpu", "--steps", "2", "--batch-size", "2",
                                "--output", str(root / "reference")])
        args.source = root / "source.pt"
        model = TinyPolicy()
        save_checkpoint(args.source, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                        epoch=3, global_step=42, best_val_loss=1, config=config())
        train = TinyDataset([("base", "a", 8), ("r", "a", 8), ("p", "b", 4)])
        validation = TinyDataset([("v", "val", 3)])
        assessment = TinyDataset([(f"a{i}", "a", 2) for i in range(6)])
        from types import SimpleNamespace
        data = (train, validation, assessment, {"r": "recovery", "p": "proactive_turn"},
                SimpleNamespace(split_id="x", digest="y"))
        plan = base.plan_probe(args, config(), data)
        self.assertEqual(base.execute_probe(args, config(), data, plan, base.Budget(60)), 0)
        return args.output / "report.json", data

    def run_toy(self, root, expired=False):
        reference, data = self.make_reference(root)
        args = history.parse_args(["--execute", "--device", "cpu", "--reference", str(reference),
                                   "--output", str(root / "history")])
        with patch.object(base, "load_data", return_value=data), \
             patch.object(history, "cache_fingerprint", return_value={"test": True}):
            prepared = history.prepare(args)
        immutable = {p: history.file_hash(p) for p in (reference, prepared.endpoint, prepared.common.source)}
        with patch.object(base, "fresh_model", wraps=base.fresh_model) as fresh:
            if expired:
                with patch.object(base, "evaluate", side_effect=base.BudgetExceeded("evaluation deadline")):
                    status = history.execute(prepared, base.Budget(60))
            else:
                status = history.execute(prepared, base.Budget(60))
            source = history.load_checkpoint(prepared.common.source)["model_state_dict"]
            for name, value in source.items():
                self.assertTrue(torch.equal(value, fresh.call_args.args[0][name]))
        for path, digest in immutable.items():
            self.assertEqual(history.file_hash(path), digest)
        return status, json.loads((args.output / "report.json").read_text())

    def test_complete_new_arm_reuses_control_and_starts_from_parent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(base, "BinnedImitationPolicy", TinyPolicy), \
             contextlib.redirect_stdout(io.StringIO()):
            status, report = self.run_toy(Path(tmp))
        self.assertEqual(status, 0)
        self.assertTrue(report["comparison_complete"])
        self.assertFalse(report["plan"]["control_retrained"])
        self.assertEqual(report["new_arm"]["steps_completed"], 2)
        self.assertEqual(report["new_arm"]["consumed_draw_counts"], report["reused_control"]["consumed_draw_counts"])
        self.assertTrue(report["new_arm"]["assessment"]["complete"])

    def test_incomplete_evaluation_is_never_a_complete_comparison(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(base, "BinnedImitationPolicy", TinyPolicy), \
             contextlib.redirect_stdout(io.StringIO()):
            status, report = self.run_toy(Path(tmp), expired=True)
            self.assertTrue((Path(tmp) / "history/partial.pt").exists())
        self.assertEqual(status, 2)
        self.assertFalse(report["comparison_complete"])
        self.assertEqual(report["status"], "budget_exhausted_incomplete")

    def test_reference_rejects_incomplete_changed_helper_and_recipe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(base, "BinnedImitationPolicy", TinyPolicy), \
             contextlib.redirect_stdout(io.StringIO()):
            path, _ = self.make_reference(Path(tmp))
            reference = json.loads(path.read_text())
        mutations = [lambda r: r.update(paired_complete=False),
                     lambda r: r.update(script_sha256="changed"),
                     lambda r: r["plan"].update(group_fractions={"base": 1}),
                     lambda r: r["arms"]["control"]["assessment"].update(complete=False),
                     lambda r: r["arms"]["control"].update(steps_completed=1)]
        for mutate in mutations:
            changed = copy.deepcopy(reference)
            mutate(changed)
            with self.assertRaises(ValueError):
                history.validate_reference(changed, history.file_hash(base.__file__))

    def test_default_mode_never_executes(self):
        from types import SimpleNamespace
        with patch.object(history, "prepare", return_value=SimpleNamespace(plan={})), \
             patch.object(history, "execute") as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(history.main([]), 0)
        execute.assert_not_called()

    def test_changed_reference_is_rejected_before_output_creation(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reference, endpoint = root / "report.json", root / "control.pt"
            reference.write_text("original")
            endpoint.write_text("original")
            plan = {"reference_report_sha256": history.file_hash(reference),
                    "reference_control_sha256": history.file_hash(endpoint),
                    "helper_sha256": history.file_hash(base.__file__)}
            prepared = SimpleNamespace(args=SimpleNamespace(output=root / "new", reference=reference),
                                       common=None, config=None, plan=plan, data=(None,) * 5,
                                       endpoint=endpoint)
            for changed in (reference, endpoint):
                changed.write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed during preparation"):
                    history.execute(prepared, base.Budget(60))
                self.assertFalse(prepared.args.output.exists())
                changed.write_text("original")


if __name__ == "__main__":
    unittest.main()
