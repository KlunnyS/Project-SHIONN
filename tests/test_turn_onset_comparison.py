"""Synthetic CPU tests only; no real training or game APIs."""
import contextlib
import copy
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import numpy as np
import torch
from torch.utils.data import DataLoader
import run_turn_onset_comparison as turn
from models.imitation.dataset import ACTION_COLUMNS
from models.imitation.checkpoint import save_checkpoint,load_checkpoint
from test_recovery_probe import TinyDataset,TinyPolicy,config


class TurnOnsetTests(unittest.TestCase):
    def test_private_shared_schedule_changes_only_within_episode_frame_pool(self):
        ds=TinyDataset([(f'p{i}','a',30) for i in range(12)])
        windows={n:(10,18) for n in ds.episode_names};state=torch.get_rng_state().clone()
        draws=turn.draw_schedule(10,22,ds,windows,2000,0,np.ones(10))
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertEqual(set(np.bincount(draws['episodes'])),{166,167})
        for arm in turn.ARMS:
            self.assertEqual([ds.index[i][0] for i in draws[arm]],draws['episodes'].tolist())
        self.assertTrue(all(10<=ds.index[i][1]<18 for i in draws['onset_window']))
        self.assertTrue(any(ds.index[i][1]<10 or ds.index[i][1]>=18 for i in draws['whole_episode']))
        self.assertEqual(len(draws['base']),10000);self.assertEqual(len(draws['recovery']),4000)

    def test_exact_two_static_one_proactive_five_base_and_no_target_mutation(self):
        target=np.zeros(10,dtype=np.float32);target[0]=1
        base=[(np.full((12,2,2),10,dtype=np.uint8),target)]
        starts=[(np.full((12,2,2),20,dtype=np.uint8),target)]
        proactive=[(np.full((12,2,2),30,dtype=np.uint8),target)]
        draws={'base':np.zeros(5,dtype=int),'recovery':np.zeros(2,dtype=int),'whole_episode':np.zeros(1,dtype=int)}
        x,y,flags=next(iter(DataLoader(turn.TurnBatches(base,starts,proactive,draws,'whole_episode'),batch_size=8)))
        self.assertEqual(x[:,0,0,0].tolist(),[20,20,30,10,10,10,10,10])
        self.assertEqual(flags.tolist(),[True,True,False,False,False,False,False,False])
        y.zero_();self.assertEqual(target[0],1)

    def onset_fixture(self,root):
        parent=root/'parent.json';parent.write_text('{}')
        ms=[];anns=[];actions={}
        for i in range(20):
            name=f'p{i}';mapname=f'dataset_test{i//4+1}';frames=root/f'{name}.npy'
            rows=np.zeros((30,10),dtype=np.float32);rows[5:,0]=1;rows[5:,8]=1;actions[name]=rows
            csv_path=root/f'{name}.csv'
            with csv_path.open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['frame_idx',*ACTION_COLUMNS]);writer.writeheader()
                for frame,row in enumerate(rows):writer.writerow({'frame_idx':frame,**dict(zip(ACTION_COLUMNS,row))})
            frames.with_suffix('.json').write_text(json.dumps({'actions_csv':str(csv_path),'source_fps':24}))
            ms.append(SimpleNamespace(name=name,map_name=mapname,frame_count=30,frames_path=frames))
            anns.append({'episode':name,'map':mapname,'frame_count':30,'fps':24,'context_start_frame':0,
                         'turn_start_frame':8,'onset_review_status':'reviewed_main_onset_with_adjustment_excluded' if i==0 else 'reviewed_sampled_images_and_actions',
                         'actions_csv_sha256':turn.rt.file_hash(csv_path)})
        ann=root/'ann.json';ann.write_text(json.dumps({'annotations':anns}))
        train=[m.name for m in ms if m.map_name not in ['dataset_test2','dataset_test5']]
        held=[m.name for m in ms if m.name not in train]
        manifest={'version':1,'experiment_id':'turn-onset-comparison-v1','recovery_protocol':str(parent),
                  'recovery_protocol_sha256':turn.rt.file_hash(parent),'onset_annotations':str(ann),
                  'onset_annotations_sha256':turn.rt.file_hash(ann),'train_proactive_episodes':train,
                  'validation_proactive_episodes':held,'training_batches':{'base':5,'stationary_recovery':2,'proactive':1},'arms':list(turn.ARMS)}
        manifest['window_rule']=turn.WINDOW_RULE
        path=root/'protocol.json';path.write_text(json.dumps(manifest))
        class DS(TinyDataset):
            def __init__(self,manifests):
                super().__init__([(m.name,m.map_name,30) for m in manifests]);self.first_action_frames=[5]*len(manifests)
                self.index=[(e,f) for e in range(len(manifests)) for f in range(5,30)]
                self.episodes=[(None,actions[m.name]) for m in manifests]
        return path,ann,ms,DS

    def test_reviewed_annotations_split_hashes_and_trimmed_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,ann,ms,ds=self.onset_fixture(Path(tmp));args=SimpleNamespace(protocol=path,proactive_cache=Path(tmp))
            p={'reserved_recovery_maps':['dataset_test2','dataset_test5','dataset_test8'],'parent_frozen_split':'unused'}
            cfg={'training':{'training_episodes':['base']}}
            with patch.object(turn,'discover_cached_episodes',return_value=ms),patch.object(turn,'load_frozen_split',return_value=Mock()),patch.object(turn,'BehaviorCloningDataset',ds):
                _,train,held,windows=turn.load_onsets(args,p,cfg)
                self.assertEqual(len(train.episode_names),12);self.assertEqual(len(held.episode_names),8)
                self.assertFalse(set(train.episode_names)&set(held.episode_names))
                self.assertEqual(windows['p0'],(5,16))
                cfg['training']['training_episodes']=['p0']
                with self.assertRaisesRegex(ValueError,'already trained'):turn.load_onsets(args,p,cfg)
                cfg['training']['training_episodes']=[];ann.write_text('changed')
                with self.assertRaisesRegex(ValueError,'changed'):turn.load_onsets(args,p,cfg)

    def prepared_toy(self,root):
        base=TinyDataset([('base','a',8)]);rec=TinyDataset([(f'r{i}','a',2) for i in range(22)])
        proactive=TinyDataset([(f'p{i}','a',4) for i in range(12)])
        rv=TinyDataset([(f'v{i}','b',4) for i in range(9)]);pv=TinyDataset([(f'q{i}','b',4) for i in range(8)])
        diag=TinyDataset([(f'd{i}','a',4) for i in range(6)]);nav=TinyDataset([('nav','v',3)])
        source=root/'source.pt';cfg=config();cfg['training'].update(training_episodes=['base'],validation_episodes=['nav'])
        model=TinyPolicy();optimizer=torch.optim.AdamW(model.parameters());sum(v.sum() for v in model.parameters()).backward();optimizer.step()
        save_checkpoint(source,model=model,optimizer=optimizer,config=cfg,epoch=2,global_step=58804,best_val_loss=1)
        recovery_protocol=root/'recovery.json';recovery_protocol.write_text('{}');ann=root/'onsets.json';ann.write_text('{}')
        protocol=root/'turn.json';protocol.write_text(json.dumps({'recovery_protocol':str(recovery_protocol),'onset_annotations':str(ann)}))
        args=turn.parse_args(['--execute','--device','cpu','--steps','2','--output',str(root/'out'),'--protocol',str(protocol)])
        starts=turn.fit.RecoveryStarts(rec);windows={n:(0,2) for n in proactive.episode_names+pv.episode_names}
        draws=turn.draw_schedule(len(base),22,proactive,windows,2,0,np.ones(len(base)))
        records=[{'episode':n,'role':role,'offline_window':[0,2]} for role,names in [('recovery_validation',rv.episode_names),('diagnostic_only',diag.episode_names)] for n in names]
        p=SimpleNamespace(args=args,source=source,cfg=cfg,base=base,starts=starts,proactive=proactive,
                          recovery_val=rv,proactive_val=pv,navigation=nav,context={'datasets':{'assessment':diag}},
                          windows=windows,draws=draws,protocol={'records':records})
        p.plan={'source_sha256':turn.rt.file_hash(source),'source_sidecar_sha256':turn.rt.file_hash(source.with_suffix('.json')),
                'source_progress':{'epoch':2,'global_step':58804,'step_in_epoch':0},'protocol_sha256':turn.rt.file_hash(protocol),
                'recovery_protocol_sha256':turn.rt.file_hash(recovery_protocol),'onset_annotations_sha256':turn.rt.file_hash(ann),
                'code_sha256':{turn.__file__:turn.rt.file_hash(turn.__file__)},'draw_sha256':turn.fingerprint_draws(draws),
                'cache_paths':{},'cache_fingerprint':{'toy':True},
                'stationary_samples_sha256':turn.fit.validate_samples(starts,turn.MouseBins(cfg['target_processing']['mouse_bins']))}
        return p

    def test_complete_pair_fresh_optimizers_and_correct_membership_metadata(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(turn.base,'BinnedImitationPolicy',TinyPolicy),patch.object(turn,'cache_fingerprint',return_value={'toy':True}),contextlib.redirect_stdout(io.StringIO()):
            p=self.prepared_toy(Path(tmp));source_hash=turn.rt.file_hash(p.source)
            self.assertEqual(turn.execute(p,turn.base.Budget(60)),0)
            r=json.loads((p.args.output/'report.json').read_text());self.assertTrue(r['paired_complete'])
            self.assertEqual(turn.rt.file_hash(p.source),source_hash)
            for arm in turn.ARMS:
                cp=load_checkpoint(p.args.output/arm/'endpoint.pt',model=TinyPolicy())
                self.assertEqual(cp['global_step'],2)
                self.assertTrue(all(int(v['step'])==2 for v in cp['optimizer_state_dict']['state'].values()))
                cfg=cp['config']['training'];self.assertEqual(len(cfg['training_episodes']),35)
                self.assertFalse(set(cfg['training_episodes'])&set(p.recovery_val.episode_names+p.proactive_val.episode_names))
                self.assertEqual(cfg['split_id'],'turn-onset-comparison-v1')
                self.assertFalse(cfg['recovery_probe']['generic_resume_supported'])
                self.assertTrue(r['arms'][arm]['diagnostic_only']['complete'])

    def test_deadline_in_second_arm_never_publishes_either_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(turn.base,'BinnedImitationPolicy',TinyPolicy),patch.object(turn,'cache_fingerprint',return_value={'toy':True}),contextlib.redirect_stdout(io.StringIO()):
            p=self.prepared_toy(Path(tmp));real=turn.evaluate;calls=[0]
            def evaluate(*args):
                calls[0]+=1
                if calls[0]==3:raise turn.base.BudgetExceeded('second arm')
                return real(*args)
            with patch.object(turn,'evaluate',side_effect=evaluate):self.assertEqual(turn.execute(p,turn.base.Budget(60)),2)
            r=json.loads((p.args.output/'report.json').read_text());self.assertFalse(r['paired_complete'])
            self.assertTrue((p.args.output/'whole_episode'/'pending.pt').exists())
            self.assertFalse(list(p.args.output.glob('*/endpoint.pt')))

    def test_changed_sampling_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(turn,'cache_fingerprint',return_value={'toy':True}):
            p=self.prepared_toy(Path(tmp));p.draws['base'][0]=(p.draws['base'][0]+1)%len(p.base)
            with self.assertRaisesRegex(ValueError,'sampling/data'):turn.execute(p,turn.base.Budget(60))
            self.assertFalse(p.args.output.exists())

    def test_deadline_during_final_hashing_never_marks_pair_complete(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(turn.base,'BinnedImitationPolicy',TinyPolicy),patch.object(turn,'cache_fingerprint',return_value={'toy':True}),contextlib.redirect_stdout(io.StringIO()):
            p=self.prepared_toy(Path(tmp));now=[0.];budget=turn.base.Budget(60,clock=lambda:now[0])
            real=turn.rt.file_hash
            def file_hash(path):
                value=real(path)
                if Path(path).name=='pending.pt':now[0]=60.
                return value
            with patch.object(turn.rt,'file_hash',side_effect=file_hash):self.assertEqual(turn.execute(p,budget),2)
            report=json.loads((p.args.output/'report.json').read_text());self.assertFalse(report['paired_complete'])
            self.assertFalse(list(p.args.output.glob('*/endpoint.pt')))

    def test_default_mode_does_not_execute(self):
        with patch.object(turn,'prepare',return_value=SimpleNamespace(plan={})),patch.object(turn,'execute') as execute,contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(turn.main([]),0)
        execute.assert_not_called()

    def test_limits(self):
        for args in [['--steps','1501'],['--time-budget-seconds','1081'],['--learning-rate','nan'],['--seed','-1']]:
            with self.subTest(args=args),contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):turn.parse_args(args)


if __name__=='__main__':unittest.main()
