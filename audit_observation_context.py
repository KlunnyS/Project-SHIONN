"""Read-only context coverage audit. Passing never approves expert labels/training."""
import argparse
import bisect
import csv
import json
import math
from pathlib import Path


def audit(directory, model_log, human_episode, *, decode_video=True):
    directory, model_log, human_episode = map(Path, (directory, model_log, human_episode))
    meta = json.loads((directory / 'metadata.json').read_text())
    frames = [json.loads(l) for l in (directory / 'frames.jsonl').read_text().splitlines()]
    model = [json.loads(l) for l in model_log.read_text().splitlines()]
    human_meta = json.loads((human_episode / 'metadata.json').read_text())
    with (human_episode / 'actions.csv').open(newline='') as f:
        human = list(csv.DictReader(f))
    links = [json.loads(l) for l in (directory / 'links.jsonl').read_text().splitlines()]
    checks = {'unlabelled_only': meta.get('kind') == 'shionn_unlabelled_observation_context_v1'
        and meta.get('expert_targets') is False and meta.get('labels') == 'none'
        and meta.get('training_approved') is False and not (directory / 'actions.csv').exists(),
        'capture_finalized': meta.get('capture_status') == 'complete',
        'frame_count': len(frames) == meta.get('frames') and bool(frames),
        'human_provenance': human_meta.get('source') == 'model_recovery'
            and Path(human_meta.get('source_attempt') or '/missing').resolve() == model_log.resolve(),
        'requested_map': human_meta.get('map') == meta.get('requested_map') and human_meta.get('map') in {f'dataset_test{i}' for i in range(1,11)}
            and next((m.get('map') for m in model if m.get('type')=='metadata'),None) == human_meta.get('map'),
        'orchestration_links': any(l.get('phase')=='model_subprocess_returned' and Path(l.get('attempt_log') or '/missing').resolve()==model_log.resolve() for l in links)
            and any(l.get('phase')=='human_recorder_returned' and Path(l.get('episode') or '/missing').resolve()==human_episode.resolve() for l in links)}
    ticks = [r for r in model if r.get('type') == 'tick']
    handoff = [r for r in model if r.get('type') == 'stop' and r.get('reason') == 'recovery_handoff']
    if not frames or not ticks or not handoff or not human:
        raise ValueError('Context/model handoff/human records missing; cannot audit continuity')
    sampled = [r['sampled_monotonic'] for r in frames]
    received = [r['received_monotonic'] for r in frames]
    ages = [r['frame_age_seconds'] for r in frames]
    checks['ordered_finite_timestamps'] = all(math.isfinite(x) for x in sampled + received + ages)
    checks['ordered_finite_timestamps'] &= all(b > a for a, b in zip(sampled, sampled[1:]))
    checks['ordered_finite_timestamps'] &= all(b >= a for a, b in zip(received, received[1:]))
    checks['frame_indices_and_ownership'] = all(r['frame_idx'] == i and r.get('expert_targets') is False for i, r in enumerate(frames))
    checks['frame_ages_match'] = all(0 <= age and abs((s - r) - age) < 1e-6 for s, r, age in zip(sampled, received, ages))
    gaps = [b - a for a, b in zip(sampled, sampled[1:])]
    checks['camera_ids_ordered'] = all(isinstance(r.get('frame_id'),int) for r in frames) and all(b['frame_id']>=a['frame_id'] for a,b in zip(frames,frames[1:]))
    start = ticks[0]['monotonic_seconds']
    stop = handoff[-1]['monotonic_seconds']
    human_start = float(human[0]['frame_received_monotonic'])
    human_end = float(human[-1]['frame_sampled_monotonic'])
    checks['covers_model_through_human'] = sampled[0] <= start <= stop <= human_start <= human_end <= sampled[-1]
    lo=max(0,bisect.bisect_right(sampled,start)-1)
    hi=min(len(sampled),bisect.bisect_left(sampled,human_end)+1)
    required_gaps=[b-a for a,b in zip(sampled[lo:hi],sampled[lo+1:hi])]
    checks['sample_gaps_bounded'] = bool(required_gaps) and max(required_gaps)<=.25
    checks['frame_ages_match'] &= all(age<=.25 for age in ages[lo:hi])
    between = sum(stop <= t <= human_start for t in sampled)
    checks['handoff_interval_observed'] = between > 0
    decoded = None
    if decode_video:
        import cv2
        cap = cv2.VideoCapture(str(directory / meta['video']))
        decoded = 0
        try:
            while True:
                ok, _ = cap.read()
                if not ok:break
                decoded += 1
        finally:
            cap.release()
        checks['decoded_video_frame_count'] = decoded == len(frames)
    return {'directory': str(directory), 'model_log': str(model_log), 'human_episode': str(human_episode),
        'checks': checks, 'passed': all(checks.values()), 'frames': len(frames), 'decoded_frames': decoded,
        'max_sample_gap_seconds': max(gaps, default=None), 'required_span_max_gap_seconds': max(required_gaps,default=None), 'max_frame_age_seconds': max(ages),
        'handoff_interval_seconds': human_start - stop, 'observations_during_handoff': between,
        'meaning': 'Structural capture coverage only; independent-camera scene/action alignment requires visual review. Context is never an expert target.',
        'training_approved': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--model-log', type=Path, required=True)
    parser.add_argument('--human-episode', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.model_log, args.human_episode)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':main()
