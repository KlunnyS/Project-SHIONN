"""Safety and paired execution contracts, using tiny synthetic CPU policies."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
import run_route_expansion as runner
from run_recovery_fit_probe import file_hash
from models.imitation.checkpoint import save_checkpoint


class Tiny(nn.Module):
    def __init__(self):super().__init__();self.weight=nn.Parameter(torch.ones(1))


class TargetedTest(unittest.TestCase):
    def test_shared_schedule_and_no_held_targets(self):
        base=[('base',i) for i in range(20)];corrections=[('human',i) for i in range(6)]
        shared=np.arange(16);extra=np.arange(4);draw=np.arange(4)
        control=[('old',i) for i in range(6)]
        a=runner.MatchedBatches(base,corrections,shared,extra,draw,'control',control,draw[::-1])
        b=runner.MatchedBatches(base,corrections,shared,extra,draw,'correction',corrections,draw)
        self.assertEqual(len(a),20)
        for i in range(20):
            if i%10<8:self.assertEqual(a[i],b[i])
            else:self.assertEqual(a[i],control[int(draw[::-1][(i//10)*2+i%10-8])]);self.assertEqual(b[i][0],'human')
        with self.assertRaises(ValueError):runner.MatchedBatches(base,corrections,shared,extra,draw[:-1],'correction',corrections,draw)

    def test_review_role_and_group_guards(self):
        protocol=json.loads(runner.PROTOCOL.read_text())
        train,held=runner.validate_review(protocol)
        self.assertEqual((len(train),len(held)),(7,2))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'review.json';r=json.loads(Path(protocol['correction_review_manifest']).read_text())
            r['episodes'][0]['map']='dataset_test11';path.write_text(json.dumps(r))
            protocol.update(correction_review_manifest=str(path),review_manifest_sha256=file_hash(path))
            with self.assertRaisesRegex(ValueError,'Unapproved'):runner.validate_review(protocol)

    def test_invalid_review_hash_rejected(self):
        protocol=json.loads(runner.PROTOCOL.read_text());protocol['review_manifest_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'manifest changed'):runner.validate_review(protocol)

    def prepared(self,tmp):
        cfg={'training':{'weight_decay':.0001,'training_episodes':['base'],'augmentation':{},'label_smoothing':.01},'target_processing':{}}
        src=Path(tmp)/'source.pt'
        save_checkpoint(src,model=Tiny(),optimizer=torch.optim.AdamW(Tiny().parameters()),epoch=3,global_step=100,best_val_loss=1.,config=cfg)
        ds=[(np.zeros((12,1,1),np.uint8),np.zeros(10,np.float32))]
        return SimpleNamespace(args=SimpleNamespace(device='cpu',seed=0,steps=2,output=Path(tmp)/'output'),source=src,cfg=cfg,
            regular=ds,corrections=ds,control=ds,control_draws=np.zeros(4,dtype=np.int64),shared=np.zeros(16,dtype=np.int64),extra=np.zeros(4,dtype=np.int64),correction_draws=np.zeros(4,dtype=np.int64),
            plan={'source_progress':{'epoch':3,'global_step':100,'step_in_epoch':0},'base_train_episodes':['base'],'control_corrections':['oldhuman'],'train_corrections':['human'],
                  'held_out_corrections':['held'],'input_sha256':{str(runner.PROTOCOL):'protocolsha'}})

    def train(self,model,loader,opt,device,cfg,budget,progress):
        for _ in range(2):
            opt.zero_grad();model.weight.square().sum().backward();opt.step();progress['steps_completed']+=1
        return {'samples':20}

    def run_fake(self,p,budget,evaluation=None):
        ev=evaluation or (lambda *a:{'navigation':{'loss':1.}})
        with patch.object(runner,'guard'),patch.object(runner.base,'fresh_model',side_effect=lambda *a:Tiny()),patch.object(runner.base,'train_probe_steps',side_effect=self.train),patch.object(runner,'evaluate',side_effect=ev),patch.object(runner,'validation_screen',return_value={'passed':True}):
            return runner.execute(p,budget)

    def test_pair_fixed_endpoints_and_no_source_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);sha=file_hash(p.source);r=self.run_fake(p,runner.base.Budget(20))
            self.assertTrue(r['paired_complete']);self.assertEqual(file_hash(p.source),sha)
            for arm in ['control','correction']:
                path=p.args.output/arm/'endpoint.pt';self.assertTrue(path.is_file())
                cfg=json.loads(path.with_suffix('.json').read_text())['config']
                self.assertFalse(cfg['training']['recovery_probe']['generic_resume_supported'])
                self.assertNotIn('held',cfg['training']['training_episodes'])
                self.assertEqual(cfg['training']['training_episodes'],['base','human'] if arm=='correction' else ['base','oldhuman'])

    def test_incomplete_second_arm_never_publishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);calls=[]
            def evaluate(*args):
                calls.append(1)
                if len(calls)==3:raise runner.base.BudgetExceeded('test timeout')
                return {'navigation':{'loss':1.}}
            with self.assertRaises(runner.base.BudgetExceeded):self.run_fake(p,runner.base.Budget(20),evaluate)
            self.assertFalse((p.args.output/'control/endpoint.pt').exists())
            report=json.loads((p.args.output/'report.json').read_text());self.assertFalse(report['paired_complete'])

    def test_changed_raw_correction_fails_before_decoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'actions.csv').write_text('changed')
            rec={'path':str(path),'hashes':{'actions.csv':'wrong'}}
            with self.assertRaisesRegex(ValueError,'hash changed'):
                runner.HumanDataset([rec],runner.base.Budget(20))

    def test_expiry_during_final_hash_does_not_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);budget=runner.base.Budget(20);original=runner.file_hash
            def hashed(path):
                value=original(path)
                if Path(path).name=='pending.pt':budget.deadline=budget.started
                return value
            with patch.object(runner,'file_hash',side_effect=hashed):
                with self.assertRaises(runner.base.BudgetExceeded):self.run_fake(p,budget)
            self.assertFalse((p.args.output/'control/endpoint.pt').exists())
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['paired_complete'])

    def test_default_never_trains(self):
        with patch.object(runner,'prepare',return_value=SimpleNamespace(plan={})),patch.object(runner,'execute') as execute,patch('builtins.print'):
            runner.main([])
        execute.assert_not_called()

    def test_new_map_metadata_is_checked_before_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); meta=p/'metadata.json'
            meta.write_text(json.dumps({'source':'model_recovery','map':'dataset_test11','source_attempt':'unused'}))
            rec={'path':str(p),'hashes':{'metadata.json':file_hash(meta)},'map':'dataset_test11','source_attempt':'unused'}
            with self.assertRaisesRegex(ValueError,'ownership/map/source'):
                runner.HumanDataset([rec],runner.base.Budget(10))

    def test_expiry_after_rename_rolls_back_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);budget=runner.base.Budget(20);check=budget.check
            def expire():
                if (p.args.output/'control/endpoint.pt').exists():
                    raise runner.base.BudgetExceeded('publication expiry')
                check()
            with patch.object(budget,'check',side_effect=expire):
                with self.assertRaises(runner.base.BudgetExceeded): self.run_fake(p,budget)
            self.assertFalse(list(p.args.output.glob('*/endpoint.*')))
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['paired_complete'])

    def test_partial_rename_failure_rolls_back_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);replace=Path.replace
            def fail(path,target):
                if str(target).endswith('correction/endpoint.pt'): raise OSError('rename failure')
                return replace(path,target)
            with patch.object(Path,'replace',fail):
                with self.assertRaises(OSError): self.run_fake(p,runner.base.Budget(20))
            self.assertFalse(list(p.args.output.glob('*/endpoint.*')))
