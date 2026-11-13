"""Synthetic CPU checks only; no real candidate training or game access."""
import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_stationary_replay as replay
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from test_recovery_probe import TinyDataset, TinyPolicy, config


class StationaryReplayTests(unittest.TestCase):
    def setUp(self):
        self.recovery = TinyDataset([(f'r{i}', 'a', 2) for i in range(31)])
        self.train = TinyDataset([('base', 'a', 8), *[(f'r{i}', 'a', 2) for i in range(31)], ('p', 'b', 4)])
        self.validation = TinyDataset([('validation', 'v', 3)])
        self.assessment = TinyDataset([(f'assessment{i}', 'v', 2) for i in range(6)])
        self.groups = {**{n: 'recovery' for n in self.recovery.episode_names}, 'p': 'proactive_turn'}
        self.cfg = config()
        self.cfg['training'].update(training_episodes=self.train.episode_names,
                                    validation_episodes=self.validation.episode_names,
                                    split_sha256='frozen', recovery_probe={'generic_resume_supported': False})
        self.data = (self.train, self.validation, self.assessment, self.groups, SimpleNamespace(digest='frozen'))

    def prepared(self, root, steps=2):
        source = root/'source.pt'
        model = TinyPolicy()
        save_checkpoint(source, model=model, optimizer=torch.optim.AdamW(model.parameters()),
                        epoch=0, global_step=1000, step_in_epoch=1000, best_val_loss=1,
                        config=self.cfg, training_state={'probe_status': 'complete_fixed_endpoint'})
        args = replay.parse_args(['--execute', '--source', str(source), '--output', str(root/'out'),
                                  '--steps', str(steps), '--device', 'cpu'])
        with patch.object(replay.base, 'load_data', return_value=self.data), \
             patch.object(replay, 'discover_cached_episodes', return_value=[]), \
             patch.object(replay, 'BehaviorCloningDataset', return_value=self.recovery), \
             patch.object(replay, 'cache_fingerprint', return_value={'toy': True}):
            return replay.prepare(args)

    def test_every_batch_is_exactly_two_training_starts_and_six_regular(self):
        starts = replay.fit.RecoveryStarts(self.recovery)
        ds = replay.ReplayBatches(self.train, starts, np.arange(12), np.array([0,1,2,3]))
        for frames, labels, flags in DataLoader(ds, batch_size=8):
            self.assertEqual(flags.tolist(), [True, True, False, False, False, False, False, False])
            self.assertEqual(len(frames), 8)
        self.assertFalse(set(r['episode'] for r in starts.records) & set(self.assessment.episode_names))
        with self.assertRaises(ValueError):
            replay.ReplayBatches(self.train, starts, [0], [0,1])

    def test_plan_records_conditional_and_total_recipe_and_balanced_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.prepared(Path(tmp), steps=2000)
        self.assertEqual(p.plan['regular_draws'], 12000)
        self.assertEqual(p.plan['stationary_draws'], 4000)
        self.assertEqual(p.plan['start_exposures_min_max'], [129,130])
        self.assertEqual(p.plan['total_expected_group_fractions'], {'base':.675,'recovery':.2875,'proactive_turn':.0375})
        self.assertEqual(p.plan['regular_group_expected_fractions'], {'base':.9,'recovery':.05,'proactive_turn':.05})
        self.assertFalse(p.plan['assessment_used_for_gradient_or_checkpoint_selection'])

    def test_augmentation_changes_only_regular_frames_and_targets(self):
        class TracePolicy(TinyPolicy):
            def forward(self, x):
                self.last = x.detach().clone()
                return super().forward(x)
        starts = replay.fit.RecoveryStarts(self.recovery)
        original = starts.items[0][1].copy()
        ds = replay.ReplayBatches(self.train, starts, np.arange(6), np.array([0,1]))
        def augment(frames, targets, **kwargs):
            changed = targets.clone();changed[:,0] = 0
            return frames + .25, changed
        model = TracePolicy()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        progress = {'steps_completed':0}
        with patch.object(replay.base,'augment_batch',side_effect=augment) as aug, \
             patch.object(replay.base,'behavior_cloning_loss',wraps=replay.base.behavior_cloning_loss) as loss:
            replay.train_steps(model,DataLoader(ds,batch_size=8),optimizer,torch.device('cpu'),
                               self.cfg,replay.base.Budget(60),progress)
        self.assertEqual(aug.call_args.args[0].shape[0],6)
        self.assertTrue(torch.allclose(model.last[:2],torch.full_like(model.last[:2],40/255)))
        self.assertTrue(torch.allclose(model.last[2:],torch.full_like(model.last[2:],40/255+.25)))
        self.assertEqual(loss.call_args.args[1][:,0].tolist(),[1,1,0,0,0,0,0,0])
        np.testing.assert_array_equal(starts.items[0][1],original)
        self.assertEqual(progress['steps_completed'],1)

    def test_complete_toy_run_saves_inference_checkpoint_with_actual_recipe(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(replay.base,'BinnedImitationPolicy',TinyPolicy), \
             patch.object(replay,'cache_fingerprint',return_value={'toy':True}), contextlib.redirect_stdout(io.StringIO()):
            p = self.prepared(Path(tmp));before = replay.fit.file_hash(p.args.source)
            self.assertEqual(replay.execute(p,replay.base.Budget(60)),0)
            self.assertEqual(replay.fit.file_hash(p.args.source),before)
            r = json.loads((p.args.output/'report.json').read_text())
            self.assertTrue(r['candidate_complete'])
            self.assertEqual(r['progress']['steps_completed'],2)
            self.assertEqual(r['progress']['consumed_stationary_draws'],4)
            for phase in ('before','after'):
                self.assertEqual(r[phase]['validation']['frames'],3)
                self.assertEqual(r[phase]['assessment']['frames'],12)
                self.assertEqual(r[phase]['training_starts']['samples'],31)
                self.assertEqual(r[phase]['assessment_starts']['samples'],6)
            self.assertFalse((p.args.output/'pending.pt').exists())
            checkpoint = load_checkpoint(p.args.output/'candidate.pt',model=TinyPolicy())
            self.assertTrue(all(int(v['step']) == 2 for v in checkpoint['optimizer_state_dict']['state'].values()))
            self.assertEqual(checkpoint['global_step'],2)
            self.assertEqual(checkpoint['step_in_epoch'],2)
            self.assertEqual(checkpoint['config']['target_processing'],self.cfg['target_processing'])
            training = checkpoint['config']['training']
            self.assertEqual(training['correction_group_fractions'],{'recovery':.2875,'proactive_turn':.0375})
            self.assertEqual(training['sampling'],'stationary-replay-mixed')
            with self.assertRaisesRegex(ValueError,'cannot resume'):
                load_checkpoint(p.args.output/'candidate.pt',optimizer=torch.optim.AdamW(TinyPolicy().parameters()))

    def test_deadline_during_final_evaluation_never_publishes_candidate(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(replay.base,'BinnedImitationPolicy',TinyPolicy), \
             patch.object(replay,'cache_fingerprint',return_value={'toy':True}), contextlib.redirect_stdout(io.StringIO()):
            p=self.prepared(Path(tmp))
            real = replay.evaluate
            calls=[0]
            def evaluate(*args):
                calls[0]+=1
                if calls[0]==2:raise replay.base.BudgetExceeded('final evaluation')
                return real(*args)
            with patch.object(replay,'evaluate',side_effect=evaluate):
                self.assertEqual(replay.execute(p,replay.base.Budget(60)),2)
            r=json.loads((p.args.output/'report.json').read_text())
            self.assertFalse(r['candidate_complete'])
            self.assertEqual(r['progress']['steps_completed'],2)
            self.assertFalse((p.args.output/'candidate.pt').exists())

    def test_changed_source_or_draws_are_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(Path(tmp));p.regular_draws[0]=(p.regular_draws[0]+1)%len(self.train)
            with self.assertRaisesRegex(ValueError,'draws changed'):
                replay.execute(p,replay.base.Budget(60))
            self.assertFalse(p.args.output.exists())
            p=self.prepared(Path(tmp));p.args.source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Source/code changed'):
                replay.execute(p,replay.base.Budget(60))
            self.assertFalse(p.args.output.exists())

    def test_deadline_after_staging_keeps_final_candidate_absent(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(replay.base,'BinnedImitationPolicy',TinyPolicy), \
             patch.object(replay,'cache_fingerprint',return_value={'toy':True}), contextlib.redirect_stdout(io.StringIO()):
            p=self.prepared(Path(tmp));now=[0.]
            budget=replay.base.Budget(60,clock=lambda:now[0])
            real=replay.save_checkpoint
            def save(*args,**kwargs):
                real(*args,**kwargs)
                now[0]=60.
            with patch.object(replay,'save_checkpoint',side_effect=save):
                self.assertEqual(replay.execute(p,budget),2)
            self.assertTrue((p.args.output/'pending.pt').exists())
            self.assertFalse((p.args.output/'candidate.pt').exists())
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['candidate_complete'])

    def test_binary_progress_mismatch_leaves_incomplete_report(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(replay,'cache_fingerprint',return_value={'toy':True}):
            p=self.prepared(Path(tmp))
            source=load_checkpoint(p.args.source);source['global_step']=999
            with patch.object(replay,'load_checkpoint',return_value=source), self.assertRaisesRegex(ValueError,'config/progress'):
                replay.execute(p,replay.base.Budget(60))
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['candidate_complete'])

    def test_regression_screen_uses_all_criteria_without_promoting(self):
        scores={'segments':{'all':{'exact_binary_match':.85,'mouse_mae':{'dx':6.,'dy':2.}},
                            'first8':{'frames':1600,'predicted_idle_frames':180}}}
        limits={'binary_exact_max_drop':.005,'mouse_mae_max_ratio':1.05,'first8_idle_max_additional_fraction':.01}
        self.assertTrue(replay.validation_screen(scores,scores,limits)['passed'])
        after=copy.deepcopy(scores);after['segments']['all']['mouse_mae']['dx']=7
        self.assertFalse(replay.validation_screen(scores,after,limits)['passed'])

    def test_default_plan_does_not_execute(self):
        with patch.object(replay,'prepare',return_value=SimpleNamespace(plan={})), \
             patch.object(replay,'execute') as execute, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(replay.main([]),0)
        execute.assert_not_called()

    def test_bounds_are_enforced(self):
        for args in (['--steps','2001'],['--steps','0'],['--time-budget-seconds','721'],
                     ['--learning-rate','nan'],['--seed','-1']):
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                replay.parse_args(args)


if __name__=='__main__':unittest.main()
