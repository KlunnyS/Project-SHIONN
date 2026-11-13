"""Matched single-factor turn replay and completed-control provenance tests."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import run_reviewed_turn_rehearsal as runner
import tests.test_reviewed_recovery_pair as fixtures
import tests.test_route_expansion as tiny


class TurnRehearsalTest(unittest.TestCase):
    def test_only_last_human_slot_changes(self):
        p=SimpleNamespace(regular=[('base',i) for i in range(20)],corrections=[('human',i) for i in range(4)],control=[('old',i) for i in range(4)],shared=np.arange(16),extra=np.arange(4),correction_draws=np.arange(4),control_draws=np.arange(4),openings=[19],opening_draws=np.zeros(2,dtype=np.int64))
        control=runner.pair.PairedRehearsal(p,'correction');changed=runner.TurnRehearsal(p,[3],np.zeros(2,dtype=np.int64))
        for i in range(20):
            if i%10==9:self.assertEqual(changed[i],('human',3))
            else:self.assertEqual(changed[i],control[i])
        with self.assertRaises(ValueError):runner.TurnRehearsal(p,[3],np.zeros(1,dtype=np.int64))

    def test_reference_must_match_completed_source_data_and_draws(self):
        plan={'mode':'plan_only','draw':'fixed'}
        ref={'status':'complete_no_automatic_promotion','paired_complete':True,'plan':dict(plan,mode='execute'),'arms':{'correction':{'status':'complete','steps_completed':1000,'evaluation':{'navigation':{'complete':True}}}}}
        runner.verify_reference(ref,plan)
        wrong=copy.deepcopy(ref);wrong['plan']['draw']='different'
        with self.assertRaises(ValueError):runner.verify_reference(wrong,plan)
        wrong=copy.deepcopy(ref);wrong['paired_complete']=False
        with self.assertRaises(ValueError):runner.verify_reference(wrong,plan)

    def prepared(self,tmp):
        parent=fixtures.ReviewedRecoveryTest().prepared(tmp)
        parent.shared=np.zeros(8000,dtype=np.int64)
        parent.extra=parent.correction_draws=parent.control_draws=np.zeros(2000,dtype=np.int64)
        parent.opening_draws=np.zeros(1000,dtype=np.int64)
        return SimpleNamespace(parent=parent,args=parent.args,indices=[0],draws=np.zeros(1000,dtype=np.int64),plan={},reference={'before':{'navigation':{}},'arms':{'correction':{'status':'complete'}}})

    def run_fake(self,p,budget):
        def train(model,loader,opt,device,cfg,budget,report):
            self.assertEqual(len(loader),1000);self.assertEqual(len(opt.state),0)
            opt.zero_grad();model.weight.square().sum().backward();opt.step()
            report['steps_completed']=1000;return {'samples':10000}
        with patch.object(runner,'guard'),patch.object(runner.pair.base,'fresh_model',side_effect=lambda *a:tiny.Tiny()),patch.object(runner.pair.base,'train_probe_steps',side_effect=train),patch.object(runner.pair,'evaluate',return_value={'navigation':{'loss':1.}}),patch.object(runner.pair,'validation_screen',return_value={'passed':True}):
            return runner.execute(p,budget)

    def test_isolated_source_initialization_and_checkpoint_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);digest=runner.file_hash(p.parent.source)
            result=self.run_fake(p,runner.pair.base.Budget(20))
            self.assertTrue(result['comparison_complete']);self.assertEqual(runner.file_hash(p.parent.source),digest)
            cfg=json.loads((p.args.output/'endpoint.json').read_text())['config']['training']
            self.assertEqual(cfg['training_episodes'],['base','human'])
            self.assertEqual(cfg['turn_rehearsal_fraction'],.1)
            self.assertEqual(cfg['opening_rehearsal_fraction'],.1)
            self.assertFalse(cfg['recovery_probe']['generic_resume_supported'])

    def test_expiry_before_publication_retains_pending_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);budget=runner.pair.base.Budget(20);original=runner.file_hash
            def hashed(path):
                result=original(path)
                if Path(path).name=='pending.pt':budget.deadline=budget.started
                return result
            with patch.object(runner,'file_hash',side_effect=hashed),self.assertRaises(runner.pair.base.BudgetExceeded):self.run_fake(p,budget)
            self.assertFalse((p.args.output/'endpoint.pt').exists())
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['comparison_complete'])

    def test_default_does_not_execute(self):
        with patch.object(runner,'prepare',return_value=SimpleNamespace(plan={})),patch.object(runner,'execute') as execute,patch('builtins.print'):
            runner.main([])
        execute.assert_not_called()
