"""Reused-control provenance and single-factor rehearsal schedule checks."""
import copy
import json
import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import run_opening_rehearsal as runner


class BackwardRehearsalTest(unittest.TestCase):
    def test_only_declared_slots_change(self):
        p=SimpleNamespace(regular=[('base',i) for i in range(16)],corrections=[('human',i) for i in range(4)],
            shared=np.arange(16),extra=np.arange(4),correction_draws=np.arange(4),control=[],control_draws=np.arange(4))
        reference=runner.prior.MatchedBatches(p.regular,p.corrections,p.shared,p.extra,p.correction_draws,'correction',p.control,p.control_draws)
        candidate=runner.RehearsalBatches(p,[('opening',0)],[0],np.array([0,0]))
        self.assertEqual(len(candidate),20)
        self.assertEqual([i for i in range(20) if reference[i]!=candidate[i]],[7,17])
        with self.assertRaises(ValueError):runner.RehearsalBatches(p,[0],[0],np.array([]))

    def test_reference_must_match_complete_control(self):
        p={'draw_sha256':{'shared':'fixed'},'soft_budget_seconds':600}
        r={'paired_complete':True,'status':'complete_no_automatic_promotion','plan':copy.deepcopy(p),
           'arms':{'correction':{'status':'complete','steps_completed':1000,'evaluation':{'navigation':{'complete':True}}}}}
        runner.verify_reference(r,p)
        bad=copy.deepcopy(p);bad['draw_sha256']['shared']='changed'
        with self.assertRaisesRegex(ValueError,'provenance'):runner.verify_reference(r,bad)
        bad=copy.deepcopy(r);bad['paired_complete']=False
        with self.assertRaisesRegex(ValueError,'incomplete'):runner.verify_reference(bad,p)
        bad=copy.deepcopy(r);bad['arms']['correction']['steps_completed']=999
        with self.assertRaisesRegex(ValueError,'incomplete'):runner.verify_reference(bad,p)

    def test_default_never_executes(self):
        fake=SimpleNamespace(plan={'mode':'plan_only_no_training'})
        with patch.object(runner,'prepare',return_value=fake),patch.object(runner,'execute') as execute,patch('builtins.print'):
            runner.main([])
        execute.assert_not_called()

    def prepared(self,tmp):
        from models.imitation.checkpoint import save_checkpoint
        model=torch.nn.Linear(1,1)
        cfg={'training':{'weight_decay':.0001,'training_episodes':['base','old']}}
        source=Path(tmp)/'source.pt'
        save_checkpoint(source,model=model,optimizer=torch.optim.AdamW(model.parameters()),epoch=3,global_step=100,best_val_loss=1.,config=cfg)
        class Old(list):pass
        old=Old([(np.zeros((12,1,1),np.uint8),np.zeros(10,np.float32))]);old.episode_names=['old'];old.index=[(0,0)]
        parent=SimpleNamespace(source=source,cfg=cfg,regular=old,corrections=old,control=old,control_draws=np.zeros(2000,dtype=np.int64),
            shared=np.zeros(8000,dtype=np.int64),extra=np.zeros(2000,dtype=np.int64),correction_draws=np.zeros(2000,dtype=np.int64),
            plan={'source_progress':{'epoch':3,'global_step':100,'step_in_epoch':0},'base_train_episodes':['base'],
                  'train_corrections':['human'],'held_out_corrections':['held'],'input_sha256':{str(runner.prior.PROTOCOL):'sha'}})
        return SimpleNamespace(parent=parent,args=SimpleNamespace(device='cpu',output=Path(tmp)/'output'),old=old,indices=[0],draws=np.zeros(1000,dtype=np.int64),plan={},reference={'before':{'navigation':{}},'arms':{'correction':{'status':'complete'}}})

    def run_fake(self,p,budget):
        def train(model,loader,opt,device,cfg,budget,report):
            self.assertEqual(len(loader),1000)
            self.assertEqual(len(opt.state),0)
            opt.zero_grad();model(torch.ones(1,1)).square().sum().backward();opt.step()
            report['steps_completed']=1000
            return {'precision':'float32'}
        with patch.object(runner,'guard'),patch.object(runner.prior.base,'fresh_model',side_effect=lambda *a:torch.nn.Linear(1,1)),patch.object(runner.prior.base,'train_probe_steps',side_effect=train),patch.object(runner.prior,'evaluate',return_value={'navigation':{'loss':1.}}),patch.object(runner.prior,'validation_screen',return_value={'passed':True}):
            return runner.execute(p,budget)

    def test_fixed_endpoint_and_parent_source_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);sha=runner.file_hash(p.parent.source)
            result=self.run_fake(p,runner.prior.base.Budget(20))
            self.assertTrue(result['comparison_complete'])
            self.assertEqual(runner.file_hash(p.parent.source),sha)
            meta=json.loads((p.args.output/'endpoint.json').read_text())
            self.assertFalse(meta['config']['training']['recovery_probe']['generic_resume_supported'])
            self.assertNotIn('held',meta['config']['training']['training_episodes'])
            self.assertEqual(meta['config']['training']['training_episodes'],['base','human'])
            self.assertEqual(meta['config']['training']['correction_sampling_fraction'],.2)
            self.assertEqual(meta['config']['training']['opening_rehearsal_fraction'],.1)

    def test_final_hash_timeout_blocks_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);budget=runner.prior.base.Budget(20);original=runner.file_hash
            def hashed(path):
                value=original(path)
                if Path(path).name=='pending.pt':budget.deadline=budget.started
                return value
            with patch.object(runner,'file_hash',side_effect=hashed):
                with self.assertRaises(runner.prior.base.BudgetExceeded):self.run_fake(p,budget)
            self.assertFalse((p.args.output/'endpoint.pt').exists())
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['comparison_complete'])

    def test_final_publication_expiry_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);budget=runner.prior.base.Budget(20);check=budget.check
            def expire():
                if (p.args.output/'endpoint.pt').exists(): raise runner.prior.base.BudgetExceeded('expiry')
                check()
            with patch.object(budget,'check',side_effect=expire):
                with self.assertRaises(runner.prior.base.BudgetExceeded):self.run_fake(p,budget)
            self.assertFalse(list(p.args.output.glob('endpoint.*')))
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['comparison_complete'])

    def test_opening_selection_bounds_and_train_maps(self):
        ds=SimpleNamespace(episode_maps=['dataset_test9'],first_action_frames=[3],index=[(0,2),(0,3),(0,10),(0,11)])
        self.assertEqual(runner.opening_indices(ds),[1,2])
        ds.episode_maps=['dataset_test11']
        with self.assertRaisesRegex(ValueError,'TRAIN'):runner.opening_indices(ds)

    def test_partial_publication_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);replace=Path.replace
            def fail(path,target):
                if str(target).endswith('endpoint.json'):raise OSError('rename failed')
                return replace(path,target)
            with patch.object(Path,'replace',fail):
                with self.assertRaises(OSError):self.run_fake(p,runner.prior.base.Budget(20))
            self.assertFalse(list(p.args.output.glob('endpoint.*')))
