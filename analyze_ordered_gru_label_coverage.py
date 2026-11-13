"""Read-only completed-run action exposure; no frame decoding or optimization."""

def main():
    import csv,json,hashlib,collections
    from pathlib import Path
    import numpy as np
    from models.imitation.dataset import discover_cached_episodes,BehaviorCloningDataset,ACTION_COLUMNS
    report=Path('models/imitation/checkpoints/runs/ordered_gru_route_v1/report.json');r=json.loads(report.read_text());assert r['comparison_complete'];plan=r['plan'];review=json.loads(Path('data/annotations/route_expansion_split_20261109.json').read_text())
    base={m.name:m for m in discover_cached_episodes(Path(plan['input_context']['base_cache']))};human={a['episode']:a for a in review['episodes']};groups=collections.defaultdict(list);chunk_backwards=[]
    for name,count in plan['exposure']['frames_by_episode'].items():
     if name in human:
      rec=human[name];rows=list(csv.DictReader((Path(rec['path'])/'actions.csv').open()));actions=np.array([[float(row[k]) for k in ACTION_COLUMNS] for row in rows],np.float32);lo,hi=rec['usable_window'];usable=actions[lo:hi];role='human'
     else:
      m=base[name];actions=BehaviorCloningDataset._load_actions(m.actions_path,m.frame_count);usable=actions[BehaviorCloningDataset._first_action_frame(actions):];role='base'
     assert count%len(usable)==0
     groups[role].extend([usable]*(count//len(usable)))
     for _ in range(count//len(usable)):
      chunk_backwards.extend(int((usable[i:i+32,2]!=0).sum()) for i in range(0,len(usable),32))
    result={}
    for role,arrays in groups.items():
     a=np.concatenate(arrays);result[role]={'frames':len(a),'positive_actions':{k:int((a[:,i]!=0).sum()) for i,k in enumerate(ACTION_COLUMNS[:8])},'backward_fraction':float((a[:,2]!=0).mean()),'mouse_dx_nonzero':int((a[:,8]!=0).sum()),'mouse_dx_abs_ge_3':int((abs(a[:,8])>=3).sum())}
    held=[]
    for rec in human.values():
     if rec['role']!='development_holdout':continue
     rows=list(csv.DictReader((Path(rec['path'])/'actions.csv').open()));lo,hi=rec['usable_window'];a=np.array([[float(row[k]) for k in ACTION_COLUMNS] for row in rows],np.float32)[lo:hi];held.append(a)
    a=np.concatenate(held);result['heldout']={'frames':len(a),'backward_frames':int((a[:,2]!=0).sum()),'backward_fraction':float((a[:,2]!=0).mean())}
    out={'report':str(report),'report_sha256':hashlib.sha256(report.read_bytes()).hexdigest(),'counts':result,'optimizer_batches':len(chunk_backwards),'batches_with_backward_target':sum(n>0 for n in chunk_backwards),'total_backward_targets':sum(chunk_backwards),'binary_class_weights':json.loads(Path(plan['source']).with_suffix('.json').read_text())['config']['target_processing']['binary_class_weights'],'scope':'Post-run TRAIN label exposure audit; does not prove causal failure or authorize fitting validation actions. No training or data changes.'}
    assert len(chunk_backwards)==545
    Path('data/annotations/ordered_gru_route_label_coverage_20261109.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))


if __name__ == "__main__":
    main()
