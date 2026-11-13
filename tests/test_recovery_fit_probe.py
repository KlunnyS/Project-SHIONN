"""CPU synthetic tests only; no real training or game interaction."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch

import run_recovery_fit_probe as fit
from models.imitation.checkpoint import save_checkpoint
from models.imitation.mouse_bins import MouseBins
from test_recovery_probe import TinyDataset, TinyPolicy, config


class RecoveryFitTests(unittest.TestCase):
    def setUp(self):
        self.raw = TinyDataset([(f'r{i}', 'a', 2) for i in range(31)])
        self.cfg = config()
        self.cfg['training'].update(split_sha256='frozen', training_episodes=self.raw.episode_names)
        self.manifests = [SimpleNamespace(name=n) for n in self.raw.episode_names]

    def prepared(self, root, role='train', config_override=None):
        cfg = self.cfg if config_override is None else config_override
        source = root/'source.pt'
        model = TinyPolicy()
        save_checkpoint(source, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                        config=cfg, epoch=3, global_step=42, best_val_loss=1)
        args = fit.parse_args(['--source',str(source),'--output',str(root/'out'),
                               '--execute','--device','cpu','--steps','2','--batch-size','2'])
        split = SimpleNamespace(digest='frozen', verify_cache=Mock(), role=Mock(return_value=role))
        with patch.object(fit,'load_frozen_split',return_value=split), \
             patch.object(fit,'discover_cached_episodes',return_value=self.manifests), \
             patch.object(fit,'BehaviorCloningDataset',return_value=self.raw):
            return fit.prepare(args)

    def test_balanced_private_draws_have_103_or_104_exposures(self):
        state = torch.get_rng_state().clone()
        draws = fit.balanced_draws(31,3200,0)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(set(np.bincount(draws)),{103,104})
        np.testing.assert_array_equal(draws,fit.balanced_draws(31,3200,0))

    def test_start_tiling_is_newest_rgb_and_preserves_source_targets(self):
        frames = np.arange(48,dtype=np.uint8).reshape(12,2,2)
        target = np.zeros(10,dtype=np.float32);target[8]=1
        raw = SimpleNamespace(index=[(0,0)],episode_names=['r'],first_action_frames=[0])
        class Raw:
            index,episode_names,first_action_frames=raw.index,raw.episode_names,raw.first_action_frames
            def __getitem__(self,i):return frames,target
        ds = fit.RecoveryStarts(Raw())
        np.testing.assert_array_equal(ds[0][0],np.tile(frames[-3:],(4,1,1)))
        self.assertIsNot(ds[0][1],target)
        ds[0][1][8]=7
        self.assertEqual(target[8],1)

    def test_nonfinite_and_all_idle_starts_rejected(self):
        for bad in (np.zeros(10,dtype=np.float32),np.full(10,np.nan,dtype=np.float32)):
            class Bad(TinyDataset):
                def __getitem__(self,i):return super().__getitem__(i)[0],bad
            with self.assertRaisesRegex(ValueError,'Invalid'):
                fit.RecoveryStarts(Bad([('r','a',1)]))

    def test_conflicting_identical_inputs_rejected_by_encoded_class(self):
        ds = fit.RecoveryStarts(self.raw)
        bins = MouseBins(self.cfg['target_processing']['mouse_bins'])
        fit.validate_samples(ds,bins)
        ds.items[1][1][8]=-1
        with self.assertRaisesRegex(ValueError,'conflicting'):
            fit.validate_samples(ds,bins)

    def test_validation_role_and_source_membership_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError,'TRAIN'):
                self.prepared(Path(tmp),role='validation')
            cfg=copy.deepcopy(self.cfg);cfg['training']['training_episodes']=[]
            with self.assertRaisesRegex(ValueError,'source training membership'):
                self.prepared(Path(tmp),config_override=cfg)

    def test_class_scoring_distinguishes_quantization_from_action_error(self):
        cfg=copy.deepcopy(self.cfg)
        cfg['target_processing']['mouse_bins']['dx']['representatives']=[-3,0,3]
        model=TinyPolicy()
        with torch.no_grad():
            for k,h in model.heads.items():
                h.weight.zero_();h.bias.copy_(torch.tensor([0.,9.]) if k=='move_w' else torch.tensor([9.,0.]))
            model.dx.weight.zero_();model.dx.bias.copy_(torch.tensor([0.,0.,9.]))
            model.dy.weight.zero_();model.dy.bias.copy_(torch.tensor([0.,9.,0.]))
        result=fit.score(model,fit.RecoveryStarts(self.raw),cfg,torch.device('cpu'),fit.base.Budget(60),8)
        self.assertTrue(result['fit_goal_met'])
        self.assertEqual(result['full_encoded_correct'],31)
        self.assertEqual(result['dx_mae'],2)
        self.assertEqual(result['target_quantization_dx_mae'],2)

    def test_complete_toy_diagnostic_preserves_source_and_saves_report_only(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fit.base,'BinnedImitationPolicy',TinyPolicy), \
             contextlib.redirect_stdout(io.StringIO()):
            p=self.prepared(Path(tmp));before=fit.file_hash(p.args.source)
            with patch.object(fit.base,'fresh_model',wraps=fit.base.fresh_model) as fresh:
                self.assertEqual(fit.execute(p,fit.base.Budget(60)),0)
            self.assertEqual(fit.file_hash(p.args.source),before)
            fresh.assert_called_once()
            report=json.loads((p.args.output/'report.json').read_text())
            self.assertTrue(report['diagnostic_complete'])
            self.assertEqual(report['progress']['steps_completed'],2)
            self.assertEqual(report['progress']['consumed_draws'],4)
            self.assertEqual(report['before']['samples'],31)
            self.assertEqual(report['after']['samples'],31)
            self.assertEqual([v.name for v in p.args.output.iterdir()],['report.json'])

    def test_budget_exhausted_in_final_score_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(fit.base,'BinnedImitationPolicy',TinyPolicy):
            p=self.prepared(Path(tmp))
            with patch.object(fit,'score',side_effect=[{'complete':True},fit.base.BudgetExceeded('end')]), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(fit.execute(p,fit.base.Budget(60)),2)
            r=json.loads((p.args.output/'report.json').read_text())
            self.assertFalse(r['diagnostic_complete'])
            self.assertEqual(r['status'],'budget_exhausted_incomplete')
            self.assertNotIn('after',r)

    def test_changed_source_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(Path(tmp));p.args.source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):
                fit.execute(p,fit.base.Budget(60))
            self.assertFalse(p.args.output.exists())

    def test_plan_default_never_executes_or_creates_model(self):
        with patch.object(fit,'prepare',return_value=SimpleNamespace(plan={})), \
             patch.object(fit,'execute') as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(fit.main([]),0)
        execute.assert_not_called()

    def test_invalid_budget_steps_and_learning_rate_rejected(self):
        for argv in (['--steps','401'],['--steps','0'],['--batch-size','9'],
                     ['--time-budget-seconds','241'],['--learning-rate','nan'],['--seed','-1']):
            with self.subTest(argv=argv),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                fit.parse_args(argv)


if __name__=='__main__':unittest.main()
