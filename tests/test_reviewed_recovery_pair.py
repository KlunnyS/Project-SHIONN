"""Paired-data isolation and publication tests; synthetic CPU only."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import run_reviewed_recovery_pair as runner
import tests.test_route_expansion as fixtures


class ReviewedRecoveryTest(unittest.TestCase):
    def prepared(self,tmp):
        p=fixtures.TargetedTest().prepared(tmp)
        p.openings=[0];p.opening_draws=np.zeros(2,dtype=np.int64)
        p.plan['input_sha256'][str(runner.PROTOCOL)]='protocolsha'
        return p

    def run_fake(self,p,fail=False):
        calls=[]
        def train(model,loader,opt,device,cfg,budget,progress):
            self.assertEqual(len(opt.state),0)
            self.assertEqual(len(loader),2)
            self.assertEqual(float(model.weight.detach()),1.)
            for _ in range(2):
                opt.zero_grad();model.weight.square().sum().backward();opt.step()
                progress['steps_completed']+=1
            return {'samples':20}
        def evaluate(*args):
            calls.append(1)
            if fail and len(calls)==3:raise RuntimeError('second arm failure')
            return {'navigation':{'loss':1.}}
        with patch.object(runner,'guard'),patch.object(runner.base,'fresh_model',side_effect=lambda *a:fixtures.Tiny()),patch.object(runner.base,'train_probe_steps',side_effect=train),patch.object(runner,'evaluate',side_effect=evaluate),patch.object(runner,'validation_screen',return_value={'passed':True}):
            return runner.execute(p,runner.base.Budget(20))

    def test_exact_batch_ownership_and_matched_nonhuman_examples(self):
        p=SimpleNamespace(regular=[('base',i) for i in range(20)],corrections=[('new',i) for i in range(4)],control=[('old',i) for i in range(4)],shared=np.arange(16),extra=np.arange(4),correction_draws=np.arange(4),control_draws=np.arange(4),openings=[19],opening_draws=np.zeros(2,dtype=np.int64))
        a,b=runner.PairedRehearsal(p,'control'),runner.PairedRehearsal(p,'correction')
        for i in range(20):
            if i%10<7:self.assertEqual(a[i],b[i]);self.assertEqual(a[i][0],'base')
            elif i%10==7:self.assertEqual(a[i],('base',19));self.assertEqual(a[i],b[i])
            else:self.assertEqual(a[i][0],'old');self.assertEqual(b[i][0],'new')

    def test_complete_pair_has_fresh_optimizers_and_isolated_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);sha=runner.file_hash(p.source)
            report=self.run_fake(p)
            self.assertTrue(report['paired_complete'])
            self.assertEqual(runner.file_hash(p.source),sha)
            for arm in ('control','correction'):
                path=p.args.output/arm/'endpoint.json';cfg=json.loads(path.read_text())['config']['training']
                self.assertEqual(cfg['opening_rehearsal_fraction'],.1)
                self.assertEqual(cfg['training_episodes'],['base','oldhuman'] if arm=='control' else ['base','human'])
                self.assertFalse(cfg['recovery_probe']['generic_resume_supported'])

    def test_failure_does_not_publish_either_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp)
            with self.assertRaisesRegex(RuntimeError,'second arm'):self.run_fake(p,fail=True)
            self.assertFalse(list(p.args.output.glob('*/endpoint.*')))
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['paired_complete'])

    def test_plan_only_never_trains(self):
        with patch.object(runner,'prepare',return_value=SimpleNamespace(plan={})),patch.object(runner,'execute') as execute,patch('builtins.print'):
            runner.main([])
        execute.assert_not_called()

    def test_review_membership_excludes_supplemental_exit(self):
        protocol=json.loads(runner.PROTOCOL.read_text());records=runner.reviewed_records(protocol)
        self.assertEqual(len(records),5)
        self.assertNotIn('episode_20261112_135307_521808',[r['episode'] for r in records])
        self.assertEqual(sum(b-a for a,b in [r['usable_window'] for r in records]),806)
        protocol['new_review_sha256']='invalid'
        with self.assertRaisesRegex(ValueError,'review changed'):runner.reviewed_records(protocol)
