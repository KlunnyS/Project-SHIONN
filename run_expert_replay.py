"""Plan-only by default: replay one fresh successful human episode, raw vs bins."""
import argparse
import json
import shlex
import sys
from datetime import datetime
from pathlib import Path
from expert_action_replay import load_source,digest


def command(source,encoding,directory):
    o=source['metadata']['oracle'];camera=o['capture']
    return [sys.executable,str(Path(__file__).with_name('run_imitation.py')),
        '--checkpoint',o['checkpoint'],'--expert-replay',source['directory'],'--replay-encoding',encoding,
        '--map',source['metadata']['map'],'--port',str(o['port']),'--fps','24','--model-fps','24',
        '--width',str(camera['width']),'--height',str(camera['height']),'--output',camera['output'],
        '--countdown','3','--post-preflight-settle','3','--max-seconds',str(len(source['actions'])/24+1.5),
        '--keep-focused','--hyprland-focus-guard','--hyprland-confine-pointer','--mouse-preflight',
        '--disable-chamber-timeout','--no-launch','--record-video','--log-actions','--verbose',
        '--recording-dir',str(directory)]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True,type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--recording-root',type=Path,default=Path('model_attempts/expert_oracle_20261112/replays'))
    args=parser.parse_args(argv);source=load_source(args.episode)
    if not source['metadata']['oracle']['capture']['output']:raise ValueError('An explicit captured output is required')
    root=args.recording_root.resolve()
    protected=[Path(source['directory']),Path(source['metadata']['oracle']['checkpoint']).parent]
    if any(root.is_relative_to(p.resolve()) for p in protected):raise ValueError('Replay output overlaps input')
    session=root/datetime.now().strftime('oracle_%Y%m%d_%H%M%S_%f')
    commands=[command(source,encoding,session/encoding) for encoding in ('raw','binned')]
    if not args.execute:
        print(json.dumps({'mode':'plan_only_no_gameplay','source':source['directory'],
            'source_manifest_sha256':source['manifest_sha256'],'commands':[shlex.join(c) for c in commands]},indent=2));return 0
    from run_model_sequence import persistent_virtual_mouse,run_persistent_attempt,inspect_attempt
    session.mkdir(parents=True,exist_ok=False)
    report={'kind':'expert_action_replay_diagnostic','training_approved':False,'neural_policy_used':False,
        'source':source['directory'],'source_manifest_sha256':source['manifest_sha256'],'status':'running','attempts':[],
        'interpretation':'Raw must reproduce the demonstrated route before interpreting a binned replay failure. Pose match is not full world-state equivalence; source/pipeline/timing/scene review required.'}
    def write():
        temp=session/'report.tmp.json';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(session/'report.json')
    write();print(f'Oracle diagnostic: {session}',flush=True)
    try:
        with persistent_virtual_mouse(True) as mouse:
            for encoding,cmd in zip(('raw','binned'),commands):
                current=load_source(args.episode)
                if current['manifest_sha256']!=source['manifest_sha256']:raise ValueError('Source manifest changed between attempts')
                code=run_persistent_attempt(cmd,mouse)
                reason,log,video=inspect_attempt(session/encoding)
                report['attempts'].append({'encoding':encoding,'returncode':code,'outcome':reason,'log':str(log) if log else None,'video':str(video) if video else None})
                write()
                if load_source(args.episode)['manifest_sha256']!=source['manifest_sha256']:raise ValueError('Source changed during oracle attempt')
                if code!=0 or reason not in ('goal_reached','expert_replay_exhausted','episode_failed'):
                    report['status']='stopped_invalid_or_interrupted';write();return 1
                if encoding=='raw' and reason!='goal_reached':
                    report['status']='raw_did_not_reproduce';write()
                    print('Raw actions did not reproduce the route; binned replay skipped pending review.',flush=True)
                    return 0
        if load_source(args.episode)['manifest_sha256']!=source['manifest_sha256']:raise ValueError('Source changed during replay')
        report['status']='completed_pending_review';write()
    except BaseException as error:
        report.update(status='stopped_invalid_or_interrupted',error=f'{type(error).__name__}: {error}');write();raise
    print(f'Finished; raw/quantized diagnostic requires review: {session}',flush=True)
    return 0

if __name__=='__main__':raise SystemExit(main())
