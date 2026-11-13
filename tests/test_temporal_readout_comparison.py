"""Matched temporal-readout contracts; synthetic checks never train game data."""
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
import run_temporal_readout_comparison as pilot
from models.imitation.network import BinnedImitationPolicy
from models.imitation.recurrent import RecurrentBinnedPolicy,SequenceDataset
from models.imitation.checkpoint import save_checkpoint,load_checkpoint

BINS={k:{'magnitude_edges':[],'representatives':[-3.,0.,3.]} for k in ('dx','dy')}
CFG={'target_processing':{'mouse_bins':BINS,'binary_class_weights':[[1.,1.]]*8},'training':{'weight_decay':.0001,'label_smoothing':.01}}


class Data:
    def __init__(self):
        self.first_action_frames=[0,0];self.episode_names=['one','two'];self.episode_maps=['map1','map2']
        self.episodes=[(np.zeros((19,1),np.uint8),np.zeros((19,10),np.float32)),(np.zeros((5,1),np.uint8),np.zeros((5,10),np.float32))]
        self.index=[(e,i) for e,(f,a) in enumerate(self.episodes) for i in range(len(f))]
    def __len__(self):return len(self.index)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            self.trunk=nn.Linear(1,1);self.binary_heads=nn.ModuleDict({'move_w':nn.Linear(1,2)})
            self.mouse_dx_head=nn.Linear(1,3);self.mouse_dy_head=nn.Linear(1,3)
        self.trunk.requires_grad_(False)


class TemporalReadoutTest(unittest.TestCase):
    def test_identical_heads_and_encoder_and_reset_control(self):
        source=BinnedImitationPolicy(3,3)
        models=[pilot.build_model(a,source.state_dict(),CFG,0) for a in pilot.ARMS]
        for m in models:
            for k,v in source.trunk.state_dict().items():torch.testing.assert_close(v,m.trunk.state_dict()[k],rtol=0,atol=0)
            self.assertTrue(all(not p.requires_grad for p in m.trunk.parameters()))
            self.assertFalse(m.trunk.training)
            for k,v in models[0].binary_heads.state_dict().items():torch.testing.assert_close(v,m.binary_heads.state_dict()[k],rtol=0,atol=0)
            for k,v in models[0].mouse_dx_head.state_dict().items():torch.testing.assert_close(v,m.mouse_dx_head.state_dict()[k],rtol=0,atol=0)
        for k,v in models[1].memory.state_dict().items():torch.testing.assert_close(v,models[2].memory.state_dict()[k],rtol=0,atol=0)

    def test_cached_gru_equals_pixel_forward_and_reset_has_no_history(self):
        model=RecurrentBinnedPolicy(3,3,8,False).eval()
        x=torch.rand(1,3,12,180,320)
        with torch.no_grad():
            expected,h=model(x)
            z=model.trunk(x.reshape(3,12,180,320)).reshape(1,3,512)
            out,state=pilot.readout(model,z,'gru')
            for k in out:torch.testing.assert_close(out[k],expected[k],rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(h,state,rtol=1e-4,atol=1e-5)
            reset,_=pilot.readout(model,z,'gru_reset')
            parts=[pilot.readout(model,z[:,i:i+1],'gru')[0] for i in range(3)]
            for k in reset:torch.testing.assert_close(reset[k],torch.cat([p[k] for p in parts],1))

    def test_schedule_roles_masks_and_exposure_are_shared(self):
        data={'base':Data(),'human_train':Data(),'human_held':Data()}
        seq,schedule=pilot.make_schedule(data,10,0)
        self.assertEqual(sum(role=='human_train' for role,i in schedule),4)
        self.assertNotIn('human_held',{r for r,i in schedule})
        bank=pilot.FeatureBank(data)
        for role in data:bank.features[role][:]=1.;bank.ready[role][:]=True
        f,y,valid=bank.chunk('base',seq['base'],1)
        self.assertEqual(valid[16:].sum(),3)
        self.assertEqual(pilot.chunk_indices(seq['base'],2)[16:].tolist(),[19,20,21,22,23]+[-1]*11)
        self.assertEqual(pilot.exposure(seq,schedule),pilot.exposure(seq,copy.deepcopy(schedule)))
        bank.ready['base'][:]=False
        with self.assertRaisesRegex(ValueError,'Missing'):bank.chunk('base',seq['base'],0)

    def test_real_readout_update_masks_padding_and_freezes_encoder(self):
        data={'base':Data()};seq={'base':SequenceDataset(data['base'],16,16)}
        bank=pilot.FeatureBank(data);bank.ready['base'][:]=True;bank.features['base'][:]=np.random.default_rng(0).normal(size=(24,512))
        original=bank.chunk
        def chunk(*a):
            f,y,v=original(*a);y[~v]=99;y[:16]=77
            return f,y,v
        bank.chunk=chunk
        p=SimpleNamespace(parent=SimpleNamespace(cfg=CFG),sequences=seq,schedule=[('base',1),('base',2)])
        model=RecurrentBinnedPolicy(3,3,8,False);model.trunk.requires_grad_(False)
        before=pilot.tensor_hash(model.trunk.state_dict());progress={'steps_completed':0}
        opt,metrics=pilot.train(model,'gru',p,bank,torch.device('cpu'),pilot.parent.base.Budget(20),progress)
        self.assertEqual(progress['steps_completed'],1);self.assertEqual(metrics['supervised_frames'],8)
        self.assertEqual(before,pilot.tensor_hash(model.trunk.state_dict()))
        self.assertTrue(np.isfinite(metrics['loss']))

    def test_delayed_cue_positive_control(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            model=RecurrentBinnedPolicy(3,3,8,False)
            x=torch.zeros(16,8,512);target=torch.arange(16)%2;x[:,0,0]=target*2.-1.
            opt=torch.optim.Adam(list(model.memory.parameters())+list(model.binary_heads['move_w'].parameters()),lr=.02)
            for _ in range(150):
                opt.zero_grad();out,_=pilot.readout(model,x,'gru');loss=nn.functional.cross_entropy(out['move_w'][:,-1],target);loss.backward();opt.step()
            with torch.no_grad():
                out,_=pilot.readout(model,x,'gru');reset,_=pilot.readout(model,x,'gru_reset')
            self.assertGreaterEqual(float((out['move_w'][:,-1].argmax(1)==target).float().mean()),.95)
            self.assertEqual(float((reset['move_w'][:,-1].argmax(1)==target).float().mean()),.5)

    def prepared(self,tmp):
        source=Path(tmp)/'source.pt';model=Tiny()
        save_checkpoint(source,model=model,optimizer=torch.optim.AdamW(model.parameters()),epoch=3,global_step=10,best_val_loss=1.,config=CFG)
        parent=SimpleNamespace(source=source,cfg=CFG)
        plan={'source_progress':{'epoch':3,'global_step':10,'step_in_epoch':0},'train_episodes':['train'],'source_ancestry':['train'],
              'parent_input_sha256':{str(source):'source',str(pilot.parent.PROTOCOL):'protocol'},'data_exposure':{'supervised_frames_by_group':{'base':8}}}
        return SimpleNamespace(args=SimpleNamespace(device='cpu',output=Path(tmp)/'output',steps=2),parent=parent,plan=plan,datasets={},requirements={})

    def execute_fake(self,p,budget):
        def train(model,arm,p,bank,device,budget,progress):
            opt=torch.optim.AdamW([v for v in model.parameters() if v.requires_grad]);progress['steps_completed']=2
            return opt,{'supervised_frames':8,'loss':1.}
        with patch.object(pilot,'guard'),patch.object(pilot.parent.base,'fresh_model',side_effect=lambda *a:Tiny()),patch.object(pilot,'build_model',side_effect=lambda *a:Tiny()),patch.object(pilot.FeatureBank,'populate'),patch.object(pilot,'evaluate',return_value={'navigation':{'loss':1.}}),patch.object(pilot,'train',side_effect=train):
            return pilot.execute(p,budget)

    def test_complete_publication_and_diagnostic_control_is_not_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);result=self.execute_fake(p,pilot.parent.base.Budget(20))
            self.assertTrue(result['comparison_complete'])
            self.assertTrue((p.args.output/'feedforward/endpoint.pt').exists());self.assertTrue((p.args.output/'gru/endpoint.pt').exists())
            self.assertFalse((p.args.output/'gru_reset/endpoint.pt').exists())
            with self.assertRaisesRegex(ValueError,'missing'):load_checkpoint(p.args.output/'gru_reset/diagnostic_state.pt')
            cfg=json.loads((p.args.output/'gru/endpoint.json').read_text())['config']
            self.assertEqual(cfg['memory']['hidden_size'],512)
            self.assertFalse(cfg['training']['recovery_probe']['generic_resume_supported'])
            self.assertEqual(cfg['training']['augmentation'],{'brightness':0.,'contrast':0.,'horizontal_flip_probability':0.})
            self.assertEqual(cfg['training']['readout_comparison']['data_exposure'],p.plan['data_exposure'])

    def test_expiry_at_final_hash_prevents_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);budget=pilot.parent.base.Budget(20);original=pilot.parent.file_hash
            def hashed(path):
                value=original(path)
                if Path(path).name=='pending.pt':budget.deadline=budget.started
                return value
            with patch.object(pilot.parent,'file_hash',side_effect=hashed):
                with self.assertRaises(pilot.parent.base.BudgetExceeded):self.execute_fake(p,budget)
            self.assertFalse((p.args.output/'feedforward/endpoint.pt').exists())
            self.assertFalse(json.loads((p.args.output/'report.json').read_text())['comparison_complete'])

    def test_feature_bank_fingerprint_ignores_unpopulated_rows(self):
        bank=pilot.FeatureBank({'base':Data()})
        bank.ready['base'][0]=True;bank.features['base'][0]=1.
        h=bank.fingerprint();bank.features['base'][1]=777.
        self.assertEqual(bank.fingerprint(),h)
        bank.features['base'][0,0]=2.
        self.assertNotEqual(bank.fingerprint(),h)
