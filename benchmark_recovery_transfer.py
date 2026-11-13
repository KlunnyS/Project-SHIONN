"""Offline sustained recovery imitation scorer. Never runs capture or game input."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from models.imitation import recovery_transfer as rt
from models.imitation.dataset import BINARY_ACTION_COLUMNS
from models.imitation.inference import PolicyInference


def decode_actions(output,bins):
    if not all(torch.isfinite(v).all() for v in output.values()):
        raise FloatingPointError('Non-finite model output; refusing to decode')
    binary=np.stack([output[k].argmax(1).cpu().numpy() for k in BINARY_ACTION_COLUMNS],axis=1)
    mouse=np.stack([np.rint(np.asarray(bins.representatives(a))[output[k].argmax(1).cpu().numpy()]) for a,k in enumerate(['mouse_dx_logits','mouse_dy_logits'])],axis=1)
    return np.column_stack((binary,mouse))


def data_fingerprint(args):
    result={}
    for root in [args.recovery_cache,args.assessment_cache]:
        for p in sorted(root.glob('*')):
            if p.suffix not in ['.json','.npy']:continue
            stat=p.stat()
            result[str(p.resolve())]={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,
                                     'sha256':rt.file_hash(p) if p.suffix=='.json' or p.name.endswith('.actions.npy') else None}
    return result


def run(args):
    if args.output.exists():raise FileExistsError('Existing output refused')
    if any(args.output.resolve().is_relative_to(p.resolve()) for p in [args.base_cache,args.recovery_cache,args.assessment_cache]):
        raise ValueError('Report must stay outside data caches')
    protocol=rt.load_protocol(args.protocol)
    context=rt.load_context(protocol,args.base_cache,args.recovery_cache,args.assessment_cache)
    train,held=rt.training_partitions(protocol,context)
    report={'complete':False,'mode':'validation_only' if args.validate_only else 'scoring',
            'meaning':'Expert-frame imitation, not autonomous recovery success',
            'protocol_sha256':rt.file_hash(args.protocol),'eligible_training_episodes':len(train),
            'reserved_recovery_episodes':len(held),'models':{},'data_fingerprint':data_fingerprint(args),
            'code_sha256':{str(p):rt.file_hash(p) for p in [Path(__file__),Path(rt.__file__)]},
            'libraries':{'torch':str(torch.__version__),'numpy':np.__version__}}
    if not args.validate_only:
        if not args.checkpoint:raise ValueError('Provide checkpoints or --validate-only')
        for path in args.checkpoint:
            before_hash=rt.file_hash(path)
            sidecar_hash=rt.file_hash(path.with_suffix('.json'))
            policy=PolicyInference(path,device=args.device)
            model_report={'checkpoint_sha256':before_hash,'sidecar_sha256':sidecar_hash,
                          'exposure':rt.exposure(protocol,policy.config),'histories':{}}
            if policy.mouse_bins is None:raise ValueError('Binned checkpoint required')
            for history in args.history or ['normal']:
                records=[]
                for record in protocol['records']:
                    ds=context['datasets'][record['group']];e=ds.episode_names.index(record['episode'])
                    start,end=record['offline_window']
                    # Full post-idle episode is secondary; initial48 is primary.
                    lookup={f:i for i,(ep,f) in enumerate(ds.index) if ep==e}
                    predictions=[];targets=[]
                    for offset in range(start,record['frame_count'],args.batch_size):
                        items=[ds[lookup[f]] for f in range(offset,min(offset+args.batch_size,record['frame_count']))]
                        x=np.stack([v[0] for v in items]);targets.extend(v[1] for v in items)
                        if history=='repeated':x=np.tile(x[:,-3:],(1,4,1,1))
                        tensor=torch.from_numpy(x).to(policy.device,dtype=torch.float32).div_(255)
                        if policy.device.type=='cuda':tensor=tensor.contiguous(memory_format=torch.channels_last)
                        with torch.inference_mode():out=policy.policy(tensor)
                        predictions.extend(decode_actions(out,policy.mouse_bins))
                    target,pred=np.array(targets),np.array(predictions)
                    records.append({'episode':record['episode'],'map':record['map'],'role':record['role'],
                                    'initial48':rt.episode_metrics(target[:end-start],pred[:end-start]),
                                    'full_reference':rt.episode_metrics(target,pred),
                                    'remainder':rt.episode_metrics(target[end-start:],pred[end-start:]) if len(target)>end-start else None,
                                    'targets':target.tolist(),'predictions':pred.tolist()})
                groups={}
                for role in rt.ROLES:
                    selected=[r for r in records if r['role']==role]
                    groups[role]={segment:rt.aggregate([r[segment] for r in selected if r[segment] is not None]) for segment in ['initial48','full_reference','remainder']}
                    groups[role]['by_map']={m:{seg:rt.aggregate([r[seg] for r in selected if r['map']==m and r[seg] is not None]) for seg in ['initial48','full_reference','remainder']} for m in sorted({r['map'] for r in selected})}
                model_report['histories'][history]={'records':records,'groups':groups}
            if rt.file_hash(path)!=before_hash or rt.file_hash(path.with_suffix('.json'))!=sidecar_hash:
                raise ValueError('Checkpoint/config changed during scoring')
            report['models'][str(path)]=model_report
            del policy
            print(f'Completed {path}',flush=True)
    if data_fingerprint(args)!=report['data_fingerprint'] or rt.file_hash(args.protocol)!=report['protocol_sha256']:
        raise ValueError('Scoring data/protocol changed')
    if any(rt.file_hash(p)!=digest for p,digest in report['code_sha256'].items()):
        raise ValueError('Scoring implementation changed')
    report['complete']=True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temp=args.output.with_suffix('.tmp.json');temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(args.output)
    print(args.output)
    return report


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path('data/annotations/recovery_transfer_development_v1.json'))
    p.add_argument('--base-cache',type=Path,default=Path('data/datasets/cached_frames'))
    p.add_argument('--recovery-cache',type=Path,default=Path('data/datasets/recovery_cached_frames'))
    p.add_argument('--assessment-cache',type=Path,default=Path('model_attempts/recovery_assessment_20261101/cache'))
    p.add_argument('--checkpoint',type=Path,action='append')
    p.add_argument('--history',choices=['normal','repeated'],action='append')
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(argv)
    if not 1<=a.batch_size<=32:p.error('Batch size must be1–32')
    if a.checkpoint and len(set(a.checkpoint))!=len(a.checkpoint):p.error('Duplicate checkpoints')
    if a.history and len(set(a.history))!=len(a.history):p.error('Duplicate history modes')
    return a


if __name__=='__main__':run(parse_args())
