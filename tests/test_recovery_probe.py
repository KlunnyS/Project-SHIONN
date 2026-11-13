"""Synthetic, CPU-only checks; no real policy training or game access."""
import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

import run_recovery_probe as probe
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import BehaviorCloningDataset


class TinyDataset(Dataset):
    chamber_sample_counts = BehaviorCloningDataset.chamber_sample_counts
    chamber_sampling_weights = BehaviorCloningDataset.chamber_sampling_weights

    def __init__(self, episodes):
        self.episode_names = [e[0] for e in episodes]
        self.episode_maps = [e[1] for e in episodes]
        self.first_action_frames = [0] * len(episodes)
        self.index = [(e, f) for e, (_, _, n) in enumerate(episodes) for f in range(n)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        targets = np.zeros(10, dtype=np.float32)
        targets[0] = 1
        targets[8] = 1
        return np.full((12, 2, 2), 40, dtype=np.uint8), targets


class TinyPolicy(torch.nn.Module):
    def __init__(self, dx=3, dy=3):
        super().__init__()
        self.heads = torch.nn.ModuleDict({k: torch.nn.Linear(12, 2) for k in probe.BINARY_ACTION_COLUMNS})
        self.dx, self.dy = torch.nn.Linear(12, dx), torch.nn.Linear(12, dy)

    def forward(self, x):
        v = x.mean(dim=(2, 3))
        return {**{k: h(v) for k, h in self.heads.items()}, "mouse_dx_logits": self.dx(v), "mouse_dy_logits": self.dy(v)}


def config():
    return {"architecture": probe.BINNED_ARCHITECTURE_VERSION,
            "preprocessing": copy.deepcopy(probe.PREPROCESSING_CONFIG),
            "action_columns": list(probe.ACTION_COLUMNS),
            "target_processing": {"binary_class_weights": np.ones((8, 2)).tolist(),
                                  "mouse_bins": {axis: {"magnitude_edges": [], "representatives": [-1, 0, 1]} for axis in ("dx", "dy")}},
            "training": {"weight_decay": .0001, "label_smoothing": .01, "max_grad_norm": 1,
                         "augmentation": {"brightness": 0, "contrast": 0, "horizontal_flip_probability": 0}}}


class RecoveryProbeTests(unittest.TestCase):
    def setUp(self):
        self.train = TinyDataset([("base", "a", 8), ("r", "a", 8), ("p", "b", 4)])
        self.validation = TinyDataset([("v", "val", 3)])
        self.assessment = TinyDataset([(f"a{i}", "a", 2) for i in range(6)])
        self.groups = {"r": "recovery", "p": "proactive_turn"}

    def test_only_recovery_openings_change_and_group_mass_is_preserved(self):
        control, mask = probe.recovery_opening_weights(self.train, self.groups, window_frames=2)
        boosted, _ = probe.recovery_opening_weights(self.train, self.groups, window_frames=2, boost=8)
        labels = np.array([self.groups.get(self.train.episode_names[e], "base") for e, _ in self.train.index])
        for group, share in (("base", .9), ("recovery", .05), ("proactive_turn", .05)):
            self.assertAlmostEqual(boosted[labels == group].sum() / boosted.sum(), share)
        np.testing.assert_array_equal(control[labels != "recovery"], boosted[labels != "recovery"])
        self.assertGreater(boosted[mask].sum(), control[mask].sum())
        self.assertAlmostEqual(boosted[8] / boosted[10], 8)

    def test_invalid_budgets_rejected(self):
        for args in (["--steps", "0"], ["--steps", "1001"], ["--time-budget-seconds", "1081"],
                     ["--time-budget-seconds", "nan"], ["--boost", "1"], ["--learning-rate", "nan"]):
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                probe.parse_args(args)

    def test_partial_evaluation_raises_instead_of_returning_a_score(self):
        calls = []
        class SmallBudget:
            def check(self):
                calls.append(1)
                if len(calls) >= 3:
                    raise probe.BudgetExceeded("end")
        with self.assertRaises(probe.BudgetExceeded):
            probe.evaluate(TinyPolicy(), self.validation, torch.device("cpu"), config()["target_processing"], SmallBudget(), 1)

    def test_finite_boost_cannot_silently_overflow_group_sum(self):
        dataset = TinyDataset([("base", "a", 32), ("r", "a", 8), ("p", "b", 4)])
        with self.assertRaisesRegex(ValueError, "overflowed"):
            probe.recovery_opening_weights(dataset, self.groups, window_frames=8, boost=1e308)

    def test_shared_budget_clock_is_not_restarted(self):
        now = [0.0]
        budget = probe.Budget(5, clock=lambda: now[0])
        now[0] = 4
        budget.check()
        now[0] = 5
        with self.assertRaises(probe.BudgetExceeded):
            budget.check()

    def test_evaluation_rounds_mouse_representatives_like_live_inference(self):
        target = config()["target_processing"]
        target["mouse_bins"]["dx"]["representatives"] = [-1.5, 0, 1.5]
        model = TinyPolicy()
        with torch.no_grad():
            model.dx.weight.zero_()
            model.dx.bias.copy_(torch.tensor([0, 0, 10]))
        result = probe.evaluate(model, self.validation, torch.device("cpu"), target, probe.Budget(60), 2)
        self.assertEqual(result["first_frames"][0]["predicted"][8], 2)

    def test_exact_update_count_on_cpu_synthetic_data(self):
        model = TinyPolicy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        progress = {"arm": "test", "steps_completed": 0}
        loader = DataLoader(Subset(self.train, [0, 1, 2, 3]), batch_size=2)
        result = probe.train_probe_steps(model, loader, optimizer, torch.device("cpu"), config(), probe.Budget(60), progress)
        self.assertEqual(progress["steps_completed"], 2)
        self.assertEqual(result["samples"], 4)
        self.assertTrue(all(int(s["step"]) == 2 for s in optimizer.state.values()))

    def test_fresh_models_are_identical_and_dont_mutate_parent_state(self):
        state = TinyPolicy().state_dict()
        before = {k: v.clone() for k, v in state.items()}
        with patch.object(probe, "BinnedImitationPolicy", TinyPolicy):
            a = probe.fresh_model(state, config(), torch.device("cpu"), 0)
            b = probe.fresh_model(state, config(), torch.device("cpu"), 0)
        for k in state:
            torch.testing.assert_close(a.state_dict()[k], b.state_dict()[k])
        next(a.parameters()).data.add_(1)
        for k in state:
            torch.testing.assert_close(state[k], before[k])

    def test_endpoint_preserves_source_targets_and_blocks_generic_resume(self):
        source = config()
        args = probe.parse_args([])
        data = (self.train, self.validation, self.assessment, self.groups, SimpleNamespace(split_id="x", digest="y"))
        plan = probe.plan_probe(args, source, data)
        endpoint = probe.endpoint_config(source, plan, self.train, self.validation, self.groups, "boosted", "hash")
        self.assertEqual(endpoint["target_processing"], source["target_processing"])
        self.assertNotIn("recovery_probe", source["training"])
        with tempfile.TemporaryDirectory() as tmp:
            model = TinyPolicy(); optimizer = torch.optim.AdamW(model.parameters())
            p = Path(tmp) / "endpoint.pt"
            save_checkpoint(p, model=model, optimizer=optimizer, epoch=0, step_in_epoch=2,
                            global_step=2, best_val_loss=1, config=endpoint)
            load_checkpoint(p)  # Inference/warm-start loading remains available.
            with self.assertRaisesRegex(ValueError, "cannot resume"):
                load_checkpoint(p, model=model, optimizer=optimizer)

    def test_assessment_overlap_is_rejected_before_dataset_construction(self):
        base = [SimpleNamespace(name="base", map_name="a"), SimpleNamespace(name="v", map_name="val")]
        recovery, proactive = [SimpleNamespace(name="r")], [SimpleNamespace(name="p")]
        assessment = [SimpleNamespace(name=name) for name in ("base", "a1", "a2", "a3", "a4", "a5")]
        split = SimpleNamespace(digest="digest", episodes={("base", "base"): {}},
                                verify_cache=lambda *args: None, role=lambda *args: "train")
        source = config(); source["training"]["split_sha256"] = "digest"
        with patch.object(probe, "load_frozen_split", return_value=split), \
             patch.object(probe, "discover_cached_episodes", side_effect=[base, recovery, proactive, assessment]), \
             patch.object(probe, "partition_frozen_training", return_value=([base[0]], [base[1]])), \
             patch.object(probe, "BehaviorCloningDataset") as constructor:
            with self.assertRaisesRegex(ValueError, "Assessment episodes overlap"):
                probe.load_data(probe.parse_args([]), source)
            constructor.assert_not_called()

    def test_plan_mode_never_loads_weights_or_creates_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source.pt"; source.write_bytes(b"not loaded in plan mode")
            source.with_suffix(".json").write_text(json.dumps({"config": config()}))
            output = Path(tmp) / "new"
            data = (self.train, self.validation, self.assessment, self.groups, SimpleNamespace(split_id="x", digest="y"))
            with patch.object(probe, "load_data", return_value=data), \
                 patch.object(probe, "load_checkpoint") as load, \
                 patch.object(probe, "execute_probe") as execute, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(probe.main(["--source", str(source), "--output", str(output)]), 0)
            load.assert_not_called(); execute.assert_not_called()
            self.assertFalse(output.exists())

    def run_tiny_pair(self, temporary, evaluation=None):
        args = probe.parse_args(["--execute", "--device", "cpu", "--steps", "2", "--batch-size", "2",
                                 "--output", str(Path(temporary) / "out")])
        args.source = Path(temporary) / "source.pt"
        model = TinyPolicy(); cfg = config(); opt = torch.optim.AdamW(model.parameters())
        save_checkpoint(args.source, model=model, optimizer=opt, epoch=3, global_step=42, best_val_loss=1, config=cfg)
        initial_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
        data = (self.train, self.validation, self.assessment, self.groups, SimpleNamespace(split_id="x", digest="y"))
        plan = probe.plan_probe(args, cfg, data)
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(probe, "BinnedImitationPolicy", TinyPolicy))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            if evaluation:
                stack.enter_context(patch.object(probe, "evaluate", side_effect=evaluation))
            status = probe.execute_probe(args, cfg, data, plan, probe.Budget(60))
        self.assertEqual(hashlib.sha256(args.source.read_bytes()).hexdigest(), initial_hash)
        return args, status, json.loads((args.output / "report.json").read_text())

    def test_complete_cpu_toy_pair_preserves_source_and_saves_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            args, status, report = self.run_tiny_pair(tmp)
            self.assertEqual(status, 0)
            self.assertTrue(report["paired_complete"])
            for arm in ("control", "boosted"):
                self.assertEqual(report["arms"][arm]["steps_completed"], 2)
                self.assertEqual(sum(report["arms"][arm]["consumed_draw_counts"].values()), 4)
                self.assertTrue(report["arms"][arm]["assessment"]["complete"])
                checkpoint = load_checkpoint(args.output / arm / "endpoint.pt")
                self.assertEqual(checkpoint["epoch"], 0)
                self.assertEqual(checkpoint["step_in_epoch"], 2)

    def test_deadline_in_second_arm_is_not_a_completed_pair(self):
        score = {"complete": True, "loss": 1}
        with tempfile.TemporaryDirectory() as tmp:
            args, status, report = self.run_tiny_pair(tmp, [score, score, score, score, probe.BudgetExceeded("during validation")])
            self.assertEqual(status, 2)
            self.assertFalse(report["paired_complete"])
            self.assertEqual(report["status"], "budget_exhausted_incomplete")
            self.assertEqual(report["arms"]["boosted"]["status"], "budget_exhausted_incomplete")
            self.assertTrue((args.output / "boosted/partial.pt").exists())


if __name__ == "__main__":
    unittest.main()
