"""Read-only actual control/decision timing report for an attempt JSONL."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path


def summarize(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    metadata = next(row for row in records if row.get('type') == 'metadata')
    if metadata.get('policy_kind')=='expert_replay':raise ValueError('Expert replay has no neural model decision rate')
    capture = metadata['capture']
    ticks = [row for row in records if row.get('type') == 'tick']
    decisions = [row for row in ticks if row.get('policy', {}).get('decision_rate', {}).get('refreshed', True)]
    held = [row for row in ticks if not row.get('policy', {}).get('decision_rate', {}).get('refreshed', True)]
    def rate(rows):
        if len(rows) < 2:
            return None
        duration = rows[-1]['elapsed_seconds'] - rows[0]['elapsed_seconds']
        return (len(rows)-1)/duration if duration > 0 else None
    durations = []
    for row in decisions:
        timing = row.get('frame_timing', {})
        start, end = timing.get('inference_started_monotonic'), timing.get('inference_finished_monotonic')
        if start is not None and end is not None:
            durations.append(1000*(end-start))
    return {
        'attempt': str(path), 'configured_control_fps': capture['fps'],
        'configured_model_fps': capture.get('model_fps', capture['fps']),
        'control_ticks': len(ticks), 'model_decisions': len(decisions), 'held_ticks': len(held),
        'observed_control_hz': rate(ticks), 'observed_decision_hz': rate(decisions),
        'median_inference_ms': statistics.median(durations) if durations else None,
        'held_ticks_claiming_inference': sum(
            row.get('frame_timing', {}).get('inference_started_monotonic') is not None
            or row.get('frame_timing', {}).get('inference_finished_monotonic') is not None for row in held),
        'unapplied_ticks': sum(not row.get('action_applied', False) for row in ticks),
        'meaning': 'Observed timing only; no gameplay improvement or capture-validity approval.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('attempt', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.attempt), indent=2))


if __name__ == '__main__':
    main()
