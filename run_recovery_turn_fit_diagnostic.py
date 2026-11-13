"""Fixed 100-update TRAIN-only fit diagnostic; never exports model weights."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from models.imitation.checkpoint import load_checkpoint
from run_reviewed_recovery_pair import reviewed_records, PROTOCOL, SOURCE_SHA
from run_route_expansion import HumanDataset, base
from run_turn_onset_comparison import score_windows
from run_recovery_fit_probe import file_hash

OUTPUT=Path('docs/reviews/recovery_turn_fit_probe_20261112')


def fit(model, frames, targets, cfg, device, budget):
    """Exactly twenty shuffled exposures per target, 100 batches of eight."""
    if len(frames)!=40 or len(targets)!=40:
        raise ValueError('Requires exactly40 reviewed target frames')
    generator=torch.Generator().manual_seed(0)
    indices=torch.cat([torch.randperm(40,generator=generator) for _ in range(20)])
    loader=DataLoader(TensorDataset(frames[indices],targets[indices]),batch_size=8,
                      shuffle=False,num_workers=0,generator=torch.Generator().manual_seed(0))
    settings=copy.deepcopy(cfg)
    settings['training']['augmentation'].update(brightness=0.,contrast=0.,horizontal_flip_probability=0.)
    torch.manual_seed(0)
    if device.type=='cuda':torch.cuda.manual_seed_all(0)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,
                               weight_decay=cfg['training']['weight_decay'])
    progress={'arm':'train_only_fit','steps_completed':0}
    metrics=base.train_probe_steps(model,loader,optimizer,device,settings,budget,progress)
    if progress['steps_completed']!=100 or metrics['samples']!=800:
        raise RuntimeError('Diagnostic budget incomplete')
    return {'steps_completed':100,'target_exposures':800,'exposures_per_frame':20,
            'draw_sha256':hashlib.sha256(indices.numpy().tobytes()).hexdigest(),'training':metrics}


def prepare(args,budget):
    protocol=json.loads(PROTOCOL.read_text());source=Path(protocol['source'])
    if file_hash(source)!=SOURCE_SHA:raise ValueError('Deployed source changed')
    records=reviewed_records(protocol);ds=HumanDataset(records,budget)
    indices=[i for i,(e,f) in enumerate(ds.index) if f-ds.first_action_frames[e]<8]
    if len(indices)!=40:raise ValueError('Expected40 active-window targets')
    paths=[source,source.with_suffix('.json'),PROTOCOL,Path(protocol['new_review']),
           Path(__file__),Path('run_reviewed_recovery_pair.py'),Path('run_route_expansion.py'),
           Path('run_recovery_probe.py'),Path('run_turn_onset_comparison.py')]
    paths.extend(Path('models/imitation')/n for n in ('network.py','dataset.py','train_bc.py','preprocess.py','mouse_bins.py','checkpoint.py','recovery_transfer.py'))
    for r in records:
        paths.extend(Path(r['path'])/n for n in r['hashes'])
        paths.append(Path(r['source_attempt']))
    if args.output.exists():raise FileExistsError('Diagnostic output exists; no overwrite or retry')
    protected=[source.parent.resolve()]+[Path(r['path']).resolve() for r in records]
    if any(args.output.resolve().is_relative_to(p) for p in protected):raise ValueError('Output overlaps input')
    return ds,indices,source,{str(p):file_hash(p) for p in paths}


def execute(args,ds,indices,source,hashes,budget):
    device=torch.device(args.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no fallback')
    def guard():
        for path,digest in hashes.items():
            if file_hash(Path(path))!=digest:raise ValueError('Pinned input/code changed')
            budget.check()
    guard();checkpoint=load_checkpoint(source,device='cpu');cfg=checkpoint['config']
    model=base.fresh_model(checkpoint['model_state_dict'],cfg,device,0)
    windows={n:[ds.first_action_frames[e],ds.first_action_frames[e]+8] for e,n in enumerate(ds.episode_names)}
    report={'status':'running','complete':False,'purpose':'Fit capability on TRAIN labels only; not a gameplay candidate',
            'source':str(source),'input_sha256':hashes,'learning_rate':1e-5,'seed':0,
            'augmentation':'disabled for this fitting diagnostic; cannot attribute change solely to exposure',
            'weights_exported':False,'heldout_or_gameplay_evaluation':False}
    args.output.mkdir(parents=True,exist_ok=False)
    def write():
        report['elapsed_seconds']=budget.clock()-budget.started
        tmp=args.output/'report.tmp.json';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(args.output/'report.json')
    write()
    try:
        report['before']=score_windows(model,ds,windows,cfg,device,budget)
        items=[ds[i] for i in indices]
        frames=torch.from_numpy(np.stack([x[0] for x in items]));targets=torch.from_numpy(np.stack([x[1] for x in items]))
        report['fit']=fit(model,frames,targets,cfg,device,budget)
        report['after']=score_windows(model,ds,windows,cfg,device,budget)
        guard();report.update(status='complete_diagnostic_only',complete=True);write()
    except BaseException as error:
        report.update(status='incomplete',error=f'{type(error).__name__}: {error}');write();raise
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args(argv);torch.set_num_threads(2);budget=base.Budget(300)
    ds,indices,source,hashes=prepare(args,budget)
    if args.execute:execute(args,ds,indices,source,hashes,budget)
    else:print(json.dumps({'mode':'plan_only','targets':len(indices),'updates':100,'batch_size':8,
                         'exposures_per_frame':20,'weights_exported':False,'output':str(args.output),
                         'input_sha256':hashes},indent=2))

if __name__=='__main__':main()
