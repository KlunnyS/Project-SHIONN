"""Contracts for the optional post-encoder feature-gate ablation."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from models.imitation.checkpoint import save_checkpoint
from models.imitation.inference import PolicyInference
from models.imitation.network import (BINNED_ARCHITECTURE_VERSION,
    GATED_BINNED_ARCHITECTURE_VERSION, BinnedImitationPolicy,
    GatedBinnedImitationPolicy, ObservationFeatureGate)
from models.imitation.preprocess import PREPROCESSING_CONFIG


class FeatureGateTest(unittest.TestCase):
    def test_identity_and_learnable_modulation(self):
        gate = ObservationFeatureGate()
        x = torch.randn(4, 512)
        self.assertTrue(torch.equal(x, gate(x)))
        opt = torch.optim.SGD(gate.parameters(), lr=.1)
        loss = gate(x).square().mean()
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in gate.parameters()))
        self.assertGreater(gate.layers[-1].weight.grad.abs().sum().item(), 0)
        opt.step()
        self.assertFalse(torch.equal(x, gate(x)))

    def test_same_seed_preserves_baseline_outputs_at_initialization(self):
        torch.manual_seed(17)
        baseline = BinnedImitationPolicy(3, 3).eval()
        torch.manual_seed(17)
        gated = GatedBinnedImitationPolicy(3, 3).eval()
        for k, v in baseline.state_dict().items():
            self.assertTrue(torch.equal(v, gated.state_dict()[k]), k)
        with torch.inference_mode():
            x = torch.rand(1, 12, 180, 320)
            a, b = baseline(x), gated(x)
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        with self.assertRaises(RuntimeError):
            gated.load_state_dict(baseline.state_dict())

    def test_checkpoint_inference_roundtrip_and_reset(self):
        bins = {k: {'magnitude_edges': [], 'representatives': [-3., 0., 3.]} for k in ('dx', 'dy')}
        for cls, architecture in [(BinnedImitationPolicy, BINNED_ARCHITECTURE_VERSION),
                                   (GatedBinnedImitationPolicy, GATED_BINNED_ARCHITECTURE_VERSION)]:
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as tmp:
                model = cls(3, 3).eval()
                path = Path(tmp) / 'candidate.pt'
                save_checkpoint(path, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                    epoch=0, global_step=0, best_val_loss=1., config={
                        'architecture': architecture, 'preprocessing': PREPROCESSING_CONFIG,
                        'target_processing': {'mouse_bins': bins}})
                policy = PolicyInference(path, device='cpu')
                self.assertIsInstance(policy.policy, cls)
                frame = np.zeros((180, 320, 3), dtype=np.uint8)
                action = policy.predict(frame)
                self.assertEqual(len(action), 10)
                self.assertEqual(len(policy.frames), 1)
                policy.reset()
                self.assertEqual(len(policy.frames), 0)
                for k,v in model.state_dict().items():
                    self.assertTrue(torch.equal(v, policy.policy.state_dict()[k]),k)

    def test_existing_imitation_loss_trains_full_gated_policy(self):
        from models.imitation.mouse_bins import MouseBins
        from models.imitation.train_bc import behavior_cloning_loss
        model = GatedBinnedImitationPolicy(3, 3).train()
        bins = MouseBins({k: {'magnitude_edges': [], 'representatives': [-3., 0., 3.]}
                          for k in ('dx', 'dy')})
        prediction = model(torch.rand(1, 12, 180, 320))
        targets = torch.zeros(1, 10)
        targets[:, 0] = 1
        loss, _ = behavior_cloning_loss(prediction, targets, mouse_bins=bins)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for module in (model.trunk, model.feature_gate, model.mouse_dx_head, model.binary_heads):
            grads = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(grads)
            self.assertTrue(all(torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0)

    def test_gaussian_gate_rejected_before_data_loading(self):
        r = subprocess.run([sys.executable, '-m', 'models.imitation.train_bc',
            '--feature-gate', 'observation', '--mouse-head', 'gaussian'], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn('requires --mouse-head binned', r.stderr)
