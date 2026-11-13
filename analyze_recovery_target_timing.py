"""Read-only stage-wise target, quantization, prediction and capture-time audit."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from models.imitation.mouse_bins import MouseBins
from run_recovery_fit_probe import file_hash


def metrics(target,prediction,bins):
    if len(target)==0:return {'frames':0}
    expert=target[:,8:10];pred=prediction[:,8:10]
    classes=bins.encode(torch.from_numpy(expert.astype(np.float32)))
    reconstructed=np.stack([[bins.decode(axis,int(i)) for i in classes[axis]] for axis in (0,1)],axis=1)
    result={'frames':len(target),'axes':{}}
    for axis,name in enumerate(('dx','dy')):
        e,p,q=expert[:,axis],pred[:,axis],reconstructed[:,axis]
        significant=abs(e)>=3
        result['axes'][name]={'expert_abs_mean':float(abs(e).mean()),'predicted_abs_mean':float(abs(p).mean()),
            'expert_signed_sum':float(e.sum()),'predicted_signed_sum':float(p.sum()),
            'mouse_mae':float(abs(e-p).mean()),'target_bin_reconstruction_mae':float(abs(e-q).mean()),
            'assigned_bin_match_fraction':float((p==q).mean()),
            'significant_targets':int(significant.sum()),'correct_sign':int(((np.sign(e)==np.sign(p))&significant).sum()),
            'nonzero_target_predicted_zero':int(((e!=0)&(p==0)).sum()),
            'large_target_predicted_tiny':int(((abs(e)>88)&(abs(p)<=12)).sum()) if axis==0 else None,
            'largest_bin_on_small_target':int(((abs(e)<=5)&(abs(p)==134)).sum()) if axis==0 else None}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Existing review refused')
    protocol_path=Path('docs/reviews/recovery_correction_plan_20261112.json');protocol=json.loads(protocol_path.read_text())
    report_path=Path('models/imitation/checkpoints/runs/reviewed_turn_rehearsal_20261112/report.json');report=json.loads(report_path.read_text())
    cfg=json.loads(Path(protocol['source']).with_suffix('.json').read_text())['config'];bins=MouseBins(cfg['target_processing']['mouse_bins'])
    names={Path(p).name for p in protocol['new_training_episodes']};datasets={'deployed':report['source_baseline'],'uniform':report['reused_control']['evaluation'],'turn_rehearsal':report['evaluation']}
    output={'report':str(report_path),'report_sha256':file_hash(report_path),'models':{},'capture_timing':[],
            'limits':['Labelled TRAIN observations only; no new inference, training, gameplay or causal timing repair','Signed displacement is in recorded mouse units, not calibrated game angle','Stage definitions use first active target, not manually annotated route milestones']}
    reference={}
    for name,data in datasets.items():
        selected=[r for r in data['train_corrections']['records'] if r['episode'] in names]
        if len(selected)!=5:raise ValueError('Missing reviewed episodes')
        output['models'][name]={}
        for stage,lo,hi in [('first8',0,8),('next16',8,24),('remainder',24,None),('full',0,None)]:
            target=[];prediction=[]
            for r in selected:
                e=np.asarray(r['targets']);p=np.asarray(r['predictions'])
                if name=='deployed':reference[r['episode']]=e
                elif not np.array_equal(e,reference[r['episode']]):raise ValueError('Unmatched target trajectories')
                target.extend(e[lo:hi]);prediction.extend(p[lo:hi])
            output['models'][name][stage]=metrics(np.asarray(target),np.asarray(prediction),bins)
    for folder in protocol['new_training_episodes']:
        path=Path(folder)/'actions.csv';rows=list(csv.DictReader(path.open()))
        sampled=np.array([float(r['frame_sampled_monotonic']) for r in rows]);received=np.array([float(r['frame_received_monotonic']) for r in rows]);starts=np.array([float(r['input_snapshot_started_monotonic']) for r in rows]);ends=np.array([float(r['input_snapshot_finished_monotonic']) for r in rows])
        output['capture_timing'].append({'episode':Path(folder).name,'actions_sha256':file_hash(path),'raw_frames':len(rows),
            'strictly_ordered_sample_times':bool(np.all(np.diff(sampled)>0)),
            'receive_times_nondecreasing':bool(np.all(np.diff(received)>=0)),
            'snapshot_brackets_valid':bool(np.all(ends>=starts)),
            'nominal_seconds':len(rows)/24,'sample_span_seconds':float(sampled[-1]-sampled[0]),
            'median_frame_age_seconds':float(np.median(sampled-received)),
            'max_frame_age_seconds':float(max(sampled-received)),
            'max_snapshot_duration_seconds':float(max(ends-starts))})
    args.output.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':main()
