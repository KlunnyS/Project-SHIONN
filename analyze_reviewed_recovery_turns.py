"""Read-only source/control/candidate fit on reviewed human recovery windows."""
import argparse
import json
from pathlib import Path
import torch
from models.imitation.checkpoint import load_checkpoint
from run_reviewed_recovery_pair import reviewed_records, PROTOCOL
from run_route_expansion import HumanDataset, base
from run_turn_onset_comparison import score_windows
from run_recovery_fit_probe import file_hash


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Review already exists; no overwrite')
    torch.set_num_threads(2)
    protocol=json.loads(PROTOCOL.read_text());budget=base.Budget(600)
    records=reviewed_records(protocol);ds=HumanDataset(records,budget)
    root=Path('models/imitation/checkpoints/runs/reviewed_recovery_pair_20261112')
    paths={'deployed':Path(protocol['source']),'control':root/'control/endpoint.pt','candidate':root/'correction/endpoint.pt'}
    report={'kind':'offline_labelled_training_fit_diagnostic','training_or_gameplay_performed':False,
            'protocol_sha256':file_hash(PROTOCOL),'checkpoints':{},'results':{},
            'limits':'TRAIN examples only; onset means first active labelled frames, not an inferred optimal intervention. Correct bin/turn fitting does not establish rollout success.'}
    for name,path in paths.items():
        checkpoint=load_checkpoint(path,device='cpu');cfg=checkpoint['config']
        model=base.fresh_model(checkpoint['model_state_dict'],cfg,torch.device('cpu'),0)
        report['checkpoints'][name]={'path':str(path),'sha256':file_hash(path)}
        report['results'][name]={}
        for length in (8,24,None):
            key='full' if length is None else f'first_{length}_active_frames'
            windows={n:[ds.first_action_frames[e],len(ds.episodes[e][0]) if length is None else min(len(ds.episodes[e][0]),ds.first_action_frames[e]+length)] for e,n in enumerate(ds.episode_names)}
            result=score_windows(model,ds,windows,cfg,torch.device('cpu'),budget)
            report['results'][name][key]=result
            print(name,key,json.dumps(result['aggregate']),flush=True)
        del model,checkpoint
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
