"""Plan or explicitly execute a bounded matched turn-onset sampling comparison.

Two fixed arms; no game/input APIs or automatic promotion. Default is plan only.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
import run_recovery_probe as base
import run_recovery_fit_probe as fit
import run_stationary_replay as replay
from run_recovery_history_probe import cache_fingerprint
from benchmark_recovery_transfer import decode_actions
from models.imitation import recovery_transfer as rt
from models.imitation.dataset import BehaviorCloningDataset,discover_cached_episodes
from models.imitation.frozen_split import load_frozen_split
from models.imitation.preprocess import read_actions,action_array
from models.imitation.checkpoint import load_checkpoint,save_checkpoint
from models.imitation.mouse_bins import MouseBins

ARMS=('whole_episode','onset_window')
WINDOW_RULE='[max(context_start_frame, first_active_frame), turn_start_frame+8); exclusive end'
OUTPUT=Path('models/imitation/checkpoints/runs/turn_onset_comparison_v1')


def load_onsets(args,protocol,cfg):
    manifest=json.loads(args.protocol.read_text())
    for path,key in ((manifest['recovery_protocol'],'recovery_protocol_sha256'),
                     (manifest['onset_annotations'],'onset_annotations_sha256')):
        if rt.file_hash(path)!=manifest[key]:raise ValueError('Parent annotation/protocol changed')
    if manifest.get('version')!=1 or manifest.get('experiment_id')!='turn-onset-comparison-v1':raise ValueError('Unsupported turn experiment')
    if manifest.get('training_batches')!={'base':5,'stationary_recovery':2,'proactive':1} or manifest.get('arms')!=list(ARMS):raise ValueError('Experiment branch recipe differs')
    if manifest.get('window_rule')!=WINDOW_RULE:raise ValueError('Window rule differs from version1')
    anns=json.loads(Path(manifest['onset_annotations']).read_text())['annotations']
    ms=discover_cached_episodes(args.proactive_cache)
    split=load_frozen_split(Path(protocol['parent_frozen_split']));split.verify_cache(ms,'proactive_turn')
    by={a['episode']:a for a in anns}
    if len(anns)!=20 or len(by)!=20 or set(by)!={m.name for m in ms}:raise ValueError('Requires exactly20 registered onset annotations')
    if set(by)&set(cfg['training']['training_episodes']):raise ValueError('Initial source already trained on proactive episodes')
    reserved=set(protocol['reserved_recovery_maps'])
    expected_train={m.name for m in ms if m.map_name not in reserved}
    expected_val={m.name for m in ms if m.map_name in reserved}
    if (len(expected_train)!=12 or len(expected_val)!=8 or len(manifest['train_proactive_episodes'])!=12
            or len(manifest['validation_proactive_episodes'])!=8 or set(manifest['train_proactive_episodes'])!=expected_train
            or set(manifest['validation_proactive_episodes'])!=expected_val):raise ValueError('Proactive split differs from reserved-map grouping')
    full=BehaviorCloningDataset(ms);windows={}
    for e,m in enumerate(ms):
        a=by[m.name];metadata=json.loads(m.frames_path.with_suffix('.json').read_text())
        if (a['map']!=m.map_name or a['frame_count']!=m.frame_count or a['fps']!=24
                or metadata['source_fps']!=24 or a['onset_review_status'] not in
                ['reviewed_sampled_images_and_actions','reviewed_main_onset_with_adjustment_excluded']):raise ValueError('Unreviewed or mismatched onset')
        if rt.file_hash(metadata['actions_csv'])!=a['actions_csv_sha256']:raise ValueError('Source action CSV changed since onset review')
        if not np.array_equal(action_array(read_actions(Path(metadata['actions_csv']))),full.episodes[e][1]):raise ValueError('Cached actions differ from reviewed CSV')
        labels=full.episodes[e][1]
        if not np.isfinite(labels).all() or not np.isin(labels[:,:8],[0,1]).all():raise ValueError('Invalid cached action values')
        start=max(a['context_start_frame'],full.first_action_frames[e]);end=a['turn_start_frame']+8
        if not 0<=a['context_start_frame']<=a['turn_start_frame']<end<=m.frame_count or not start<=a['turn_start_frame']:raise ValueError('Invalid onset window')
        windows[m.name]=(start,end)
    train=BehaviorCloningDataset([m for m in ms if m.name in expected_train])
    held=BehaviorCloningDataset([m for m in ms if m.name in expected_val])
    return manifest,train,held,windows


def draw_schedule(nbase,nrecovery,proactive,windows,steps,seed,base_weights):
    if nbase<1 or nrecovery<1 or not proactive.episode_names or len(base_weights)!=nbase:raise ValueError('Empty or inconsistent training pools')
    base_draws=np.asarray(list(WeightedRandomSampler(torch.from_numpy(base_weights),steps*5,replacement=True,
                          generator=torch.Generator().manual_seed(seed))),dtype=np.int64)
    recovery_draws=fit.balanced_draws(nrecovery,steps*2,seed+1)
    episodes=fit.balanced_draws(len(proactive.episode_names),steps,seed+2)
    fractions=np.random.default_rng(seed+3).random(steps)
    pools={arm:[] for arm in ARMS}
    for e,name in enumerate(proactive.episode_names):
        all_indices=[i for i,(episode,f) in enumerate(proactive.index) if episode==e]
        lo,hi=windows[name]
        onset=[i for i in all_indices if lo<=proactive.index[i][1]<hi]
        if not all_indices or not onset:raise ValueError('Empty proactive sampling pool')
        pools['whole_episode'].append(all_indices);pools['onset_window'].append(onset)
    proactive_draws={arm:np.array([pools[arm][e][min(int(u*len(pools[arm][e])),len(pools[arm][e])-1)]
                                  for e,u in zip(episodes,fractions)],dtype=np.int64) for arm in ARMS}
    return {'base':base_draws,'recovery':recovery_draws,'episodes':episodes,**proactive_draws}


class TurnBatches(Dataset):
    def __init__(self,base_ds,starts,proactive,draws,arm):
        self.base,self.starts,self.proactive,self.draws,self.arm=base_ds,starts,proactive,draws,arm
        if len(draws['base'])!=len(draws[arm])*5 or len(draws['recovery'])!=len(draws[arm])*2:raise ValueError('Invalid5/2/1 composition')
    def __len__(self):return len(self.draws[self.arm])*8
    def __getitem__(self,i):
        batch,slot=divmod(i,8)
        if slot<2:
            x,y=self.starts[int(self.draws['recovery'][batch*2+slot])];return x,y,True
        if slot==2:
            x,y=self.proactive[int(self.draws[self.arm][batch])];return x,y,False
        x,y=self.base[int(self.draws['base'][batch*5+slot-3])];return x,y,False


def fingerprint_draws(draws):return {k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in draws.items()}


def prepare(args,budget=None):
    manifest=json.loads(args.protocol.read_text());protocol=rt.load_protocol(manifest['recovery_protocol'])
    context=rt.load_context(protocol,args.base_cache,args.recovery_cache,args.assessment_cache)
    allowed,held=rt.training_partitions(protocol,context)
    source=Path(protocol['source']);metadata=json.loads(source.with_suffix('.json').read_text());cfg=metadata['config'];base.validate_source(cfg)
    m,proactive,proactive_val,windows=load_onsets(args,protocol,cfg)
    base_ds=BehaviorCloningDataset(context['base_train']);navigation=BehaviorCloningDataset(context['navigation_validation'])
    recovery=BehaviorCloningDataset(context['recovery_train']);recovery_val=BehaviorCloningDataset(held)
    starts=fit.RecoveryStarts(recovery);start_hash=fit.validate_samples(starts,MouseBins(cfg['target_processing']['mouse_bins']))
    weights=base.training_sampling_weights(base_ds,'chamber-balanced')
    draws=draw_schedule(len(base_ds),len(starts),proactive,windows,args.steps,args.seed,weights)
    paths={name:str(path.resolve()) for name,path in [('base',args.base_cache),('recovery',args.recovery_cache),
           ('proactive',args.proactive_cache),('assessment',args.assessment_cache)]}
    if any(args.output.resolve().is_relative_to(Path(p)) for p in paths.values()):raise ValueError('Output must stay outside caches')
    root=Path(__file__).resolve().parent
    files=['run_turn_onset_comparison.py','run_stationary_replay.py','run_recovery_probe.py','run_recovery_fit_probe.py',
           'run_recovery_history_probe.py','benchmark_recovery_transfer.py','models/imitation/recovery_transfer.py',
           'models/imitation/dataset.py','models/imitation/frozen_split.py','models/imitation/checkpoint.py',
           'models/imitation/train_bc.py','models/imitation/network.py','models/imitation/mouse_bins.py']
    source_progress={k:metadata.get(k,0) for k in ['epoch','global_step','step_in_epoch']}
    exposures=np.bincount(draws['episodes'],minlength=12)
    plan={'mode':'execute' if args.execute else 'plan_only_no_training','source':str(source.resolve()),
          'source_sha256':protocol['source_sha256'],'source_sidecar_sha256':protocol['source_sidecar_sha256'],
          'source_progress':source_progress,'protocol_sha256':rt.file_hash(args.protocol),
          'recovery_protocol_sha256':rt.file_hash(m['recovery_protocol']),'onset_annotations_sha256':rt.file_hash(m['onset_annotations']),
          'code_sha256':{str(root/p):rt.file_hash(root/p) for p in files},
          'libraries':{'torch':str(torch.__version__),'numpy':np.__version__},
          'cache_paths':paths,'cache_fingerprint':cache_fingerprint(paths,2**63-1),'draw_sha256':fingerprint_draws(draws),
          'stationary_samples_sha256':start_hash,'base_train_episodes':len(base_ds.episode_names),
          'recovery_train_episodes':recovery.episode_names,'recovery_validation_episodes':recovery_val.episode_names,
          'proactive_train_episodes':proactive.episode_names,'proactive_validation_episodes':proactive_val.episode_names,
          'diagnostic_episodes':context['datasets']['assessment'].episode_names,
          'batch_composition':{'base':5,'stationary_recovery':2,'proactive':1},'steps_per_arm':args.steps,
          'learning_rate':args.learning_rate,'seed':args.seed,'fresh_optimizer_each_arm':True,'arms':list(ARMS),
          'only_changed_factor':'Proactive frame sampling pool; same episode order and uniform fractions',
          'onset_windows':{k:list(v) for k,v in windows.items()},
          'proactive_episode_draws_min_max':[int(exposures.min()),int(exposures.max())],
          'draws_per_arm':{'base':args.steps*5,'stationary_recovery':args.steps*2,'proactive':args.steps},
          'source_targets_preserved':True,'stationary_augmentation':False,'normal_augmentation':'source settings',
          'assessment_used_for_gradient_or_checkpoint_selection':False,
          'navigation_validation_frames':len(navigation),'heldout_recovery_window_frames':9*48,
          'shared_soft_budget_seconds':args.time_budget_seconds,'estimated_total_runtime_minutes':[12,18],
          'outer_limit':'RuntimeMaxSec=20min','no_automatic_promotion_retry_or_extension':True}
    if budget:budget.check()
    return SimpleNamespace(args=args,protocol=protocol,cfg=cfg,source=source,context=context,base=base_ds,
                           navigation=navigation,starts=starts,recovery_val=recovery_val,proactive=proactive,
                           proactive_val=proactive_val,windows=windows,draws=draws,plan=plan)


def score_windows(model,ds,windows,cfg,device,budget):
    records=[];model.eval();bins=MouseBins(cfg['target_processing']['mouse_bins'])
    for e,name in enumerate(ds.episode_names):
        lo,hi=windows[name];indices=[i for i,(ep,f) in enumerate(ds.index) if ep==e and lo<=f<hi]
        if len(indices)!=hi-lo:raise ValueError('Incomplete scoring window')
        predictions=[];targets=[]
        for offset in range(0,len(indices),32):
            budget.check();items=[ds[i] for i in indices[offset:offset+32]]
            x=torch.from_numpy(np.stack([v[0] for v in items])).to(device,dtype=torch.float32).div_(255)
            if device.type=='cuda':x=x.contiguous(memory_format=torch.channels_last)
            with torch.inference_mode():output=model(x)
            predictions.extend(decode_actions(output,bins));targets.extend(v[1] for v in items)
        records.append({'episode':name,'map':ds.episode_maps[e],'window':[lo,hi],
                        'metrics':rt.episode_metrics(np.asarray(targets),np.asarray(predictions)),
                        'targets':np.asarray(targets).tolist(),'predictions':np.asarray(predictions).tolist()})
    budget.check()
    return {'complete':True,'episodes':len(records),'records':records,
            'aggregate':rt.aggregate([r['metrics'] for r in records]),
            'by_map':{m:rt.aggregate([r['metrics'] for r in records if r['map']==m]) for m in sorted({r['map'] for r in records})}}


def evaluate(model,p,device,budget):
    print('Evaluating full navigation validation and reserved correction windows...',flush=True)
    recovery_windows={r['episode']:r['offline_window'] for r in p.protocol['records'] if r['role']=='recovery_validation'}
    return {'navigation':base.evaluate(model,p.navigation,device,p.cfg['target_processing'],budget,32,p.cfg['training']['label_smoothing']),
            'recovery':score_windows(model,p.recovery_val,recovery_windows,p.cfg,device,budget),
            'proactive':score_windows(model,p.proactive_val,p.windows,p.cfg,device,budget)}


def endpoint_config(p,arm):
    cfg=copy.deepcopy(p.cfg)
    cfg['training'].update(training_episodes=p.base.episode_names+[r['episode'] for r in p.starts.records]+p.proactive.episode_names,
                           validation_episodes=p.navigation.episode_names,learning_rate=p.args.learning_rate,
                           batch_size=8,seed=p.args.seed,sampling='matched-turn-onset-replay',
                           correction_sampling_fraction=0.,correction_group_fractions={'recovery':.25,'proactive_turn':.125},
                           train_extra_cache_dirs=[str(p.args.recovery_cache.resolve()),str(p.args.proactive_cache.resolve())],
                           correction_episode_groups={**{r['episode']:'recovery' for r in p.starts.records},
                                                      **{n:'proactive_turn' for n in p.proactive.episode_names}})
    cfg['training'].update(split_id='turn-onset-comparison-v1',split_sha256=p.plan['protocol_sha256'],
                           recovery_validation_episodes=p.recovery_val.episode_names,
                           proactive_validation_episodes=p.proactive_val.episode_names,
                           diagnostic_only_episodes=p.context['datasets']['assessment'].episode_names,
                           parent_recovery_protocol_sha256=p.plan['recovery_protocol_sha256'])
    cfg['training']['recovery_probe']={'arm':arm,'source':str(p.source),'source_sha256':p.plan['source_sha256'],
        'generic_resume_supported':False,'paired_report_required':True,'fresh_optimizer':True,
        'assessment_used_for_gradient_or_selection':False,'optimizer_steps':p.args.steps,
        'turn_protocol_sha256':p.plan['protocol_sha256'],'source_progress':p.plan['source_progress']}
    return cfg


def execute(p,budget):
    args,plan=p.args,p.plan
    if args.output.exists():raise FileExistsError('Existing comparison refused')
    def unchanged():
        if rt.file_hash(p.source)!=plan['source_sha256'] or rt.file_hash(p.source.with_suffix('.json'))!=plan['source_sidecar_sha256']:
            raise ValueError('Pinned source changed')
        if rt.file_hash(args.protocol)!=plan['protocol_sha256'] or rt.file_hash(json.loads(args.protocol.read_text())['onset_annotations'])!=plan['onset_annotations_sha256']:
            raise ValueError('Experiment annotations changed')
        if rt.file_hash(json.loads(args.protocol.read_text())['recovery_protocol'])!=plan['recovery_protocol_sha256']:raise ValueError('Recovery protocol changed')
        if any(rt.file_hash(path)!=digest for path,digest in plan['code_sha256'].items()):raise ValueError('Implementation changed')
        if fingerprint_draws(p.draws)!=plan['draw_sha256'] or cache_fingerprint(plan['cache_paths'],2**63-1)!=plan['cache_fingerprint']:
            raise ValueError('Prepared sampling/data changed')
        if fit.validate_samples(p.starts,MouseBins(p.cfg['target_processing']['mouse_bins']))!=plan['stationary_samples_sha256']:
            raise ValueError('Prepared stationary samples changed')
    unchanged();device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    budget.check();args.output.mkdir(parents=True,exist_ok=False)
    report={'status':'running','paired_complete':False,'plan':plan,'arms':{}}
    def write():
        report['elapsed_seconds']=round(budget.clock()-budget.started,3)
        tmp=args.output/'report.tmp.json';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output/'report.json')
    write()
    try:
        source=load_checkpoint(p.source,device='cpu')
        if source['config']!=p.cfg or {k:source.get(k,0) for k in plan['source_progress']}!=plan['source_progress']:raise ValueError('Source binary metadata differs')
        state=source['model_state_dict'];del source
        model=base.fresh_model(state,p.cfg,device,args.seed)
        report['before']=evaluate(model,p,device,budget);del model;write()
        for arm in ARMS:
            print(f'Starting {arm}: {args.steps} fixed optimizer updates',flush=True)
            progress={'steps_completed':0,'status':'training'};report['arms'][arm]=progress;write()
            model=base.fresh_model(state,p.cfg,device,args.seed)
            optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=p.cfg['training']['weight_decay'],fused=device.type=='cuda')
            loader=DataLoader(TurnBatches(p.base,p.starts,p.proactive,p.draws,arm),batch_size=8,shuffle=False,num_workers=0)
            progress['training']=replay.train_steps(model,loader,optimizer,device,p.cfg,budget,progress)
            if progress['steps_completed']!=args.steps:raise RuntimeError('Fixed optimizer budget incomplete')
            progress['status']='evaluating';write();progress['evaluation']=evaluate(model,p,device,budget)
            progress['navigation_screen']=replay.validation_screen(report['before']['navigation'],progress['evaluation']['navigation'],
                {'binary_exact_max_drop':.005,'mouse_mae_max_ratio':1.05,'first8_idle_max_additional_fraction':.01})
            budget.check();folder=args.output/arm
            save_checkpoint(folder/'pending.pt',model=model,optimizer=optimizer,epoch=0,global_step=args.steps,
                            step_in_epoch=args.steps,best_val_loss=progress['evaluation']['navigation']['loss'],
                            config=endpoint_config(p,arm),training_state={'probe_status':'complete_fixed_endpoint','paired_report_required':True})
            progress['status']='pending_pair';write();del model,optimizer
        # Diagnostic clips are evaluated only after both fixed arms complete.
        diag=p.context['datasets']['assessment'];windows={r['episode']:r['offline_window'] for r in p.protocol['records'] if r['role']=='diagnostic_only'}
        for arm in ARMS:
            cp=load_checkpoint(args.output/arm/'pending.pt',device='cpu')
            model=base.fresh_model(cp['model_state_dict'],p.cfg,device,args.seed);del cp
            report['arms'][arm]['diagnostic_only']=score_windows(model,diag,windows,p.cfg,device,budget);del model;write()
        unchanged()
        endpoint_hashes={arm:rt.file_hash(args.output/arm/'pending.pt') for arm in ARMS}
        budget.check()
        for arm in ARMS:
            folder=args.output/arm
            (folder/'pending.pt').replace(folder/'endpoint.pt');(folder/'pending.json').replace(folder/'endpoint.json')
            report['arms'][arm].update(status='complete',checkpoint_sha256=endpoint_hashes[arm])
        budget.check()
        report.update(status='complete_no_automatic_promotion',paired_complete=True)
    except base.BudgetExceeded as error:report.update(status='budget_exhausted_incomplete',reason=str(error))
    except BaseException as error:
        report.update(status='interrupted_or_failed_incomplete',reason=f'{type(error).__name__}: {error}');raise
    finally:
        for progress in report['arms'].values():progress['consumed_draws']=progress['steps_completed']*8
        write()
    print(f"Comparison: {report['status']}; {args.output/'report.json'}",flush=True)
    return 0 if report['paired_complete'] else 2


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--protocol',type=Path,default=Path('data/annotations/turn_onset_comparison_v1.json'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    for name,path in [('base','data/datasets/cached_frames'),('recovery','data/datasets/recovery_cached_frames'),('proactive','data/datasets/proactive_turn_cached_frames'),('assessment','model_attempts/recovery_assessment_20261101/cache')]:
        parser.add_argument('--'+name+'-cache',type=Path,default=Path(path))
    parser.add_argument('--steps',type=int,default=1500)
    parser.add_argument('--learning-rate',type=float,default=5e-5)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--time-budget-seconds',type=float,default=1080)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    a=parser.parse_args(argv)
    if not 1<=a.steps<=1500 or not 0<=a.seed<2**63-4:parser.error('At most1500 updates and bounded nonnegative seed required')
    if not math.isfinite(a.learning_rate) or not 0<a.learning_rate<=5e-5:parser.error('Learning rate must be positive finite and at most5e-5')
    if not math.isfinite(a.time_budget_seconds) or not 0<a.time_budget_seconds<=1080:parser.error('Shared budget must be positive finite and at most1080s')
    return a


def main(argv=None):
    args=parse_args(argv);budget=base.Budget(args.time_budget_seconds) if args.execute else None
    p=prepare(args,budget);print(json.dumps(p.plan,indent=2),flush=True)
    return execute(p,budget) if args.execute else 0


if __name__=='__main__':raise SystemExit(main())
