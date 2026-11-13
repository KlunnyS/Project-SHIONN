"""Bounded 10% TRAIN-opening rehearsal; reuse an exact completed control.

Default is plan-only. No game controls, automatic retries or promotion.
"""
import argparse
from collections import Counter
import copy
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import run_route_expansion as prior
from run_recovery_fit_probe import file_hash,balanced_draws
from models.imitation.checkpoint import load_checkpoint,save_checkpoint

REFERENCE=Path('models/imitation/checkpoints/runs/route_expansion_v1/report.json')
OUTPUT=Path('models/imitation/checkpoints/runs/opening_rehearsal_v1')


class RehearsalBatches(prior.MatchedBatches):
    def __init__(self,p,old,indices,draws):
        super().__init__(p.regular,p.corrections,p.shared,p.extra,p.correction_draws,'correction',p.control,p.control_draws)
        self.old,self.indices,self.draws=old,indices,draws
        if len(draws)*2 != len(p.extra):raise ValueError('Requires one replacement every batch')
    def __getitem__(self,i):
        batch,slot=divmod(i,10)
        if slot==7:return self.old[self.indices[int(self.draws[batch])]]
        return super().__getitem__(i)


def opening_indices(dataset):
    if any(m not in {f'dataset_test{i}' for i in range(1,11)} for m in dataset.episode_maps):
        raise ValueError('Opening rehearsal requires original TRAIN maps')
    return [i for i,(e,f) in enumerate(dataset.index) if 0 <= f-dataset.first_action_frames[e] < 8]


def verify_reference(report,plan):
    if not report.get('paired_complete') or report.get('status')!='complete_no_automatic_promotion':
        raise ValueError('Reference pair incomplete')
    ignore={'mode','soft_budget_seconds'}
    if {k:v for k,v in report['plan'].items() if k not in ignore}!={k:v for k,v in plan.items() if k not in ignore}:
        raise ValueError('Reference control provenance/draws/settings differ')
    arm=report['arms']['correction']
    if arm.get('status')!='complete' or arm.get('steps_completed')!=1000 or not arm['evaluation']['navigation']['complete']:
        raise ValueError('Reference correction control is incomplete')


def prepare(args,budget):
    # The completed correction arm is the control: same source and1000 updates,
    # not a warm start. Only1000 slot7 inputs are replaced in the new arm.
    parent_args=SimpleNamespace(execute=False,steps=1000,seed=0,time_budget_seconds=600,
        base_cache=Path('data/datasets/cached_frames'),output=args.output,device=args.device)
    p=prior.prepare(parent_args,budget)
    report=json.loads(REFERENCE.read_text());verify_reference(report,p.plan)
    control=REFERENCE.parent/'correction/endpoint.pt'
    if file_hash(control)!=report['arms']['correction']['checkpoint_sha256']:
        raise ValueError('Reference checkpoint hash differs')
    old=p.regular
    indices=opening_indices(old)
    if len(indices)!=8800:raise ValueError('Expected first8 frames of1100 TRAIN episodes')
    draws=balanced_draws(len(indices),1000,4)
    selected=[{'episode':old.episode_names[old.index[i][0]],'frame':old.index[i][1]} for i in indices]
    files=[Path(__file__),REFERENCE,control,control.with_suffix('.json')]
    plan={'mode':'execute' if args.execute else 'plan_only_no_training','purpose':'Test whether10% original TRAIN-opening rehearsal preserves startup while retaining expanded corrections',
        'parent_plan':p.plan,'reference_report':str(REFERENCE),'reference_arm':'correction',
        'input_sha256':{str(f):file_hash(f) for f in files},
        'opening_examples':selected,
        'realized_opening_episodes':len({old.index[indices[int(d)]][0] for d in draws}),
        'realized_opening_maps':dict(Counter(old.episode_maps[old.index[indices[int(d)]][0]] for d in draws)),
        'opening_pool_semantics':'First8 active-window frames, not guaranteed spawn views; normal original history','replacement_draw_sha256':hashlib.sha256(draws.tobytes()).hexdigest(),
        'source':str(p.source),'steps':1000,'seed':0,'batch_size':10,'learning_rate':1e-5,
        'target_counts':{'base':7000,'expanded_human':2000,'train_openings':1000},
        'changed_factor':'Replace slot7 on every batch with normal-history original TRAIN opening example; all other draws and source settings unchanged',
        'soft_budget_seconds':600,'estimated_minutes':[6,9],'external_hard_cap':'12min',
        'reference_already_failed_opening_idle_guard':True,'automatic_promotion_or_retry':False}
    budget.check()
    return SimpleNamespace(parent=p,args=args,old=old,indices=indices,draws=draws,plan=plan,reference=report)


def guard(p,budget):
    prior.guard(p.parent,budget)
    for path,sha in p.plan['input_sha256'].items():
        if file_hash(path)!=sha:raise ValueError('Reference/script changed')
        budget.check()
    if hashlib.sha256(p.draws.tobytes()).hexdigest()!=p.plan['replacement_draw_sha256']:
        raise ValueError('Rehearsal draws changed')
    selected=[{'episode':p.old.episode_names[p.old.index[i][0]],'frame':p.old.index[i][1]} for i in p.indices]
    if selected!=p.plan['opening_examples']:raise ValueError('Rehearsal indices changed')
    budget.check()


def execute(p,budget):
    args=p.args;parent=p.parent;device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    guard(p,budget)
    source=load_checkpoint(parent.source,device='cpu')
    if source['config']!=parent.cfg or any(source.get(k,0)!=v for k,v in parent.plan['source_progress'].items()):
        raise ValueError('Source payload differs')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'plan':p.plan,'status':'running','comparison_complete':False,'steps_completed':0,
        'source_baseline':p.reference['before'],'reused_control':p.reference['arms']['correction']}
    def write():
        report['elapsed_seconds']=round(budget.clock()-budget.started,3)
        temp=args.output/'report.tmp.json';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output/'report.json')
    write()
    try:
        model=prior.base.fresh_model(source['model_state_dict'],parent.cfg,device,0)
        torch.manual_seed(0)
        if device.type=='cuda':torch.cuda.manual_seed_all(0)
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=parent.cfg['training']['weight_decay'])
        ds=RehearsalBatches(parent,p.old,p.indices,p.draws)
        loader=DataLoader(ds,batch_size=10,shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(0))
        report['arm']='opening_rehearsal'
        report['training']=prior.base.train_probe_steps(model,loader,opt,device,parent.cfg,budget,report)
        if report['steps_completed']!=1000:raise RuntimeError('Incomplete training')
        report['evaluation']=prior.evaluate(model,parent,device,budget)
        report['navigation_screen']=prior.validation_screen(p.reference['before']['navigation'],report['evaluation']['navigation'],
            {'binary_exact_max_drop':.005,'mouse_mae_max_ratio':1.05,'first8_idle_max_additional_fraction':.01})
        cfg=copy.deepcopy(parent.cfg)
        cfg['training'].update(learning_rate=1e-5,batch_size=10,seed=0,sampling='targeted-opening-rehearsal',
            training_episodes=parent.plan['base_train_episodes']+parent.plan['train_corrections'],
            correction_sampling_fraction=.2,correction_group_fractions={},correction_episode_groups={},
            opening_rehearsal_fraction=.1,opening_rehearsal_window=8,split_id='targeted-opening-rehearsal-v1',split_sha256=parent.plan['input_sha256'][str(prior.PROTOCOL)],
            recovery_probe={'generic_resume_supported':False,'fresh_optimizer':True,'source_sha256':prior.SOURCE_SHA,
                'comparison_report_required':True,'parent_training_episodes':parent.cfg['training']['training_episodes'],
                'held_out_corrections':parent.plan['held_out_corrections']})
        save_checkpoint(args.output/'pending.pt',model=model,optimizer=opt,epoch=0,global_step=1000,
            best_val_loss=report['evaluation']['navigation']['loss'],config=cfg,
            training_state={'probe_status':'complete_fixed_endpoint','comparison_report_required':True})
        guard(p,budget);sha=file_hash(args.output/'pending.pt');budget.check()
        for ext in ['pt','json']:(args.output/f'pending.{ext}').replace(args.output/f'endpoint.{ext}')
        budget.check();report.update(status='complete_no_automatic_promotion',comparison_complete=True,checkpoint_sha256=sha);write()
    except BaseException as error:
        for ext in ['pt','json']:
            endpoint=args.output/f'endpoint.{ext}'
            if endpoint.exists():endpoint.replace(args.output/f'pending.{ext}')
        report.update(status='incomplete',error=f'{type(error).__name__}: {error}',comparison_complete=False);write();raise
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda');args=parser.parse_args(argv)
    budget=prior.base.Budget(600);p=prepare(args,budget)
    if args.execute:execute(p,budget)
    else:print(json.dumps(p.plan,indent=2))

if __name__=='__main__':main()
