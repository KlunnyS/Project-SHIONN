import subprocess
import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from models.imitation.dataset import BehaviorCloningDataset, EpisodeManifest
from models.imitation.recurrent import SequenceDataset, RecurrentBinnedPolicy, RECURRENT_ARCHITECTURE_VERSION, run_recurrent_epoch
from models.imitation.checkpoint import save_checkpoint
from models.imitation.inference import PolicyInference
from models.imitation.preprocess import PREPROCESSING_CONFIG
from models.imitation.mouse_bins import MouseBins

BINS = {k: {'magnitude_edges': [], 'representatives': [-3., 0., 3.]} for k in ('dx', 'dy')}


def tiny(gate=False):
    model = RecurrentBinnedPolicy(3, 3, 7, gate)
    model.trunk = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(12, 512))
    return model


class RecurrentTest(unittest.TestCase):
    def test_order_boundaries_burn_and_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifests=[]
            for ep,n in enumerate([5,3]):
                f=Path(tmp)/f'{ep}.npy';a=Path(tmp)/f'{ep}.actions.npy'
                frames=np.zeros((n,180,320,3),np.uint8)
                for i in range(n):frames[i]=ep*20+i
                actions=np.zeros((n,10),np.float32);actions[:,0]=1
                np.save(f,frames);np.save(a,actions)
                manifests.append(EpisodeManifest(str(ep),f,a,n,'map'))
            data=SequenceDataset(BehaviorCloningDataset(manifests),3,2)
            self.assertEqual(len(data),3)
            f,a,v=data[0];self.assertEqual(v.tolist(),[False,False,True,True,True])
            self.assertEqual(f[2:,9,0,0].tolist(),[0,1,2])
            f,a,v=data[1];self.assertEqual(v.tolist(),[True,True,True,True,False])
            self.assertEqual(f[:4,9,0,0].tolist(),[1,2,3,4])
            f,a,v=data[2];self.assertEqual(f[2,0,0,0],20)
            self.assertEqual(v.tolist(),[False,False,True,True,True])

    def test_step_sequence_and_masked_state(self):
        for gated in (False,True):
            model=tiny(gated).eval();x=torch.rand(2,3,12,8,8)
            with torch.no_grad():
                seq,h=model(x);state=None;outputs=[]
                for i in range(3):
                    out,state=model.step(x[:,i],state);outputs.append(out)
                for k in seq:torch.testing.assert_close(seq[k],torch.stack([o[k] for o in outputs],1))
                torch.testing.assert_close(h,state)
                _,masked=model(x,valid=torch.tensor([[True,False,False],[True,True,True]]))
                _,first=model(x[:1,:1]);torch.testing.assert_close(masked[:,:1],first)
                self.assertEqual(h.shape,(1,2,7))
            with self.assertRaises(ValueError):model(x,torch.zeros(1,1,7))

    def test_detached_context_blocks_gradient(self):
        model=tiny();x=torch.rand(1,4,12,8,8,requires_grad=True)
        _,state=model(x[:,:2])
        out,_=model(x[:,2:],state.detach())
        sum(v.square().mean() for v in out.values()).backward()
        self.assertEqual(x.grad[:,:2].abs().sum(),0)
        self.assertGreater(x.grad[:,2:].abs().sum(),0)

    def test_masked_training_and_no_grad_burn(self):
        class Data:
            burn_in=1
            def __len__(self):return 2
            def __getitem__(self,i):
                targets = torch.zeros(4,10)
                targets[0] = 77  # burn-in labels must never reach the loss
                targets[3] = 99  # invalid class labels deliberately test masking
                return torch.ones(4,12,8,8),targets,torch.tensor([True,True,True,False])
        model=tiny(True);opt=torch.optim.AdamW(model.parameters(),lr=.001)
        before=model.memory.weight_ih_l0.detach().clone()
        hooks=[]
        model.trunk.register_forward_hook(lambda m,a,o: hooks.append(torch.is_grad_enabled()))
        metrics,step=run_recurrent_epoch(model,DataLoader(Data(),batch_size=2),opt,
            torch.amp.GradScaler('cpu',enabled=False),torch.device('cpu'),True,
            torch.ones(8,2),torch.ones(2),mouse_bins=MouseBins(BINS),brightness=.1,contrast=.1,horizontal_flip_probability=1.)
        self.assertEqual(step,1);self.assertTrue(np.isfinite(metrics['total']))
        self.assertFalse(torch.equal(before,model.memory.weight_ih_l0))
        self.assertFalse(hooks[0]);self.assertTrue(all(hooks[1:]))

    def test_checkpoint_inference_reset_and_sessions(self):
        for gate in (False,True):
            with tempfile.TemporaryDirectory() as tmp:
                model=RecurrentBinnedPolicy(3,3,7,gate).eval();p=Path(tmp)/'r.pt'
                save_checkpoint(p,model=model,optimizer=torch.optim.AdamW(model.parameters()),epoch=0,global_step=0,best_val_loss=1.,config={
                    'architecture':RECURRENT_ARCHITECTURE_VERSION,'preprocessing':PREPROCESSING_CONFIG,
                    'target_processing':{'mouse_bins':BINS},'memory':{'type':'gru','hidden_size':7,'feature_gate':gate,'sequence_length':3,'burn_in':1}})
                policy=PolicyInference(p,'cpu');frame=np.zeros((180,320,3),np.uint8)
                a=policy.predict(frame);h=policy.hidden.clone()
                self.assertEqual(h.shape,(1,1,7));self.assertIsNone(policy.hidden.grad_fn)
                policy.predict(frame);self.assertFalse(torch.equal(h,policy.hidden))
                other=PolicyInference(p,'cpu');self.assertIsNone(other.hidden)
                self.assertEqual(other.predict(frame),a)
                policy.reset();self.assertIsNone(policy.hidden);self.assertEqual(len(policy.frames),0)
                self.assertEqual(policy.predict(frame),a)
                torch.testing.assert_close(policy.hidden,h)

    def test_real_trainer_cli_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=Path(tmp)/'cache';cache.mkdir()
            for i in range(2):
                name=f'episode_{i}'
                np.save(cache/(name+'.npy'),np.zeros((3,180,320,3),np.uint8))
                actions=np.zeros((3,10),np.float32);actions[:,0]=1;actions[:,8]=[-3,0,3]
                np.save(cache/(name+'.actions.npy'),actions)
                (cache/(name+'.json')).write_text(json.dumps({'episode':name,'frame_count':3,'map':'test', 'cached_actions':name+'.actions.npy'}))
            dest=Path(tmp)/'model'
            cmd=[sys.executable,'-m','models.imitation.train_bc','--cache-dir',str(cache),'--unfrozen-split',
                '--checkpoint-dir',str(dest),'--temporal-memory','gru','--feature-gate','observation',
                '--sampling','uniform','--gru-hidden-size','7','--sequence-length','2','--burn-in','1',
                '--mouse-bins','3','--batch-size','1','--validation-batch-size','1','--epochs','1','--device','cpu']
            r=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr+'\n'+r.stdout)
            policy=PolicyInference(dest/'last.pt','cpu')
            self.assertEqual(policy.config['architecture'],RECURRENT_ARCHITECTURE_VERSION)
            self.assertEqual(policy.config['memory']['hidden_size'],7)
            self.assertTrue(policy.config['memory']['feature_gate'])
            r=subprocess.run(cmd+['--resume',str(dest/'last.pt'),'--sequence-length','3'],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('model or target processing differs',r.stderr)
