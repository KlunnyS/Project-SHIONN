"""Validated human-action replay diagnostics; no model training or game launch."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import time
from pathlib import Path
import numpy as np

KIND='shionn_expert_replay_source_v1'
class ReplayPoseError(RuntimeError):pass
class ReplayTimingError(RuntimeError):pass
class ReplayResetError(RuntimeError):pass

KEYS=('move_w','move_a','move_s','move_d','jump','use','fire_left','fire_right')
NUMBER=r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
POSE=re.compile(r'setpos(?:_exact)?\s+('+NUMBER+r')\s+('+NUMBER+r')\s+('+NUMBER+r').*?setang\s+('+NUMBER+r')\s+('+NUMBER+r')\s+('+NUMBER+r')(?=\s|;)',re.S)


def digest(path):
    with Path(path).open('rb') as handle:return hashlib.file_digest(handle,'sha256').hexdigest()


def parse_pose(text):
    match=POSE.search(text)
    if not match:raise ValueError('Complete getpos position/angle reply missing')
    values=[float(v) for v in match.groups()]
    if not all(math.isfinite(v) for v in values):raise ValueError('Nonfinite pose')
    return {'position':values[:3],'angles':values[3:]}


def read_pose(controller,timeout=2.):
    controller.read_console(print_to_terminal=False)
    started=time.monotonic();controller.send_command('getpos');output=''
    while time.monotonic()-started<timeout:
        output+=controller.read_console(print_to_terminal=False)
        try:pose=parse_pose(output)
        except ValueError:time.sleep(.01);continue
        return {**pose,'query_started_monotonic':started,'query_finished_monotonic':time.monotonic()}
    raise RuntimeError('Complete getpos pose unavailable')


def verify_pose(expected,actual):
    for pose in (expected,actual):
        if any(len(pose[k])!=3 or not all(math.isfinite(float(x)) for x in pose[k]) for k in ('position','angles')):
            raise ValueError('Invalid pose values')
    distance=math.dist(expected['position'],actual['position'])
    rotation=max(abs((b-a+180)%360-180) for a,b in zip(expected['angles'],actual['angles']))
    if distance>.5 or rotation>.5:raise ReplayPoseError(f'Oracle pose mismatch: position {distance:.3f}, angle {rotation:.3f}; no replay permitted')
    return {'position_distance':distance,'max_angle_error_degrees':rotation}


def read_settings(controller):
    result={}
    for name in ('sensitivity','m_yaw','m_pitch','m_rawinput'):
        controller.read_console(print_to_terminal=False);controller.send_command(name)
        deadline=time.monotonic()+1.;output=''
        pattern=re.compile(r'"?'+name+r'"?\s*(?:=|:)\s*"?('+NUMBER+r')(?=["\s;])')
        while time.monotonic()<deadline:
            output+=controller.read_console(print_to_terminal=False)
            match=pattern.search(output)
            if match:
                value=float(match.group(1))
                if not math.isfinite(value):raise ValueError('Nonfinite game setting')
                result[name]=value;break
            time.sleep(.01)
        else:raise RuntimeError(f'Could not fingerprint game setting {name}')
    return result


class CaptureStop:
    def __init__(self,tracker):self.tracker=tracker;self.stop_requested=False
    def wait(self,seconds):
        deadline=time.monotonic()+seconds
        while True:
            if self.tracker.pop_control_events():self.stop_requested=True
            if self.stop_requested:return True
            remaining=deadline-time.monotonic()
            if remaining<=0:return False
            time.sleep(min(.02,remaining))


def source_metadata(controller,tracker,map_name,map_file,checkpoint,camera):
    from run_imitation import (query_hyprland_active_window,prepare_portal_visual_state,
        activate_portal_input,disable_chamber_timeout,mouse_capture_preflight,load_recovery_map)
    monitor=CaptureStop(tracker);records=[]
    class Log:
        def write(self,kind,**fields):records.append({'type':kind,**fields})
    if monitor.wait(0):raise InterruptedError('Oracle capture cancelled')
    if prepare_portal_visual_state(controller,'auto') is None:raise RuntimeError('Oracle visual preparation failed')
    if not (activate_portal_input(controller,'auto') or {}).get('portal_focused'):raise RuntimeError('Oracle focus unavailable')
    print('Oracle camera preflight: leave all controls untouched.',flush=True)
    mouse_capture_preflight(controller,'auto',monitor,Log())
    load_recovery_map(controller,map_name)
    disable_chamber_timeout(controller,require_active=True)
    if prepare_portal_visual_state(controller,'auto') is None:raise RuntimeError('Oracle visual restore failed')
    if not (activate_portal_input(controller,'auto') or {}).get('portal_focused'):raise RuntimeError('Oracle focus restore failed')
    if monitor.wait(3.):raise InterruptedError('Oracle capture cancelled')
    if not (query_hyprland_active_window('auto') or {}).get('portal_focused'):
        raise RuntimeError('Portal must be focused for oracle capture')
    a=read_pose(controller)
    if monitor.wait(.1):raise InterruptedError('Oracle capture cancelled')
    b=read_pose(controller);verify_pose(a,b)
    snapshot=tracker.get_snapshot_and_reset()
    if any(snapshot.get(k,0) for k in (*KEYS,'crouch','mouse_dx','mouse_dy')):
        raise RuntimeError('Keep all controls released while capturing oracle start pose')
    sidecar=Path(checkpoint).with_suffix('.json');config=json.loads(sidecar.read_text())['config']
    if 'mouse_bins' not in config['target_processing']:raise ValueError('Binned reference required')
    return {'source':KIND,'action_owner':'human','training_approved':False,
        'oracle':{'fps':24,'port':controller.port,'map_file':str(Path(map_file).resolve()),'map_sha256':digest(map_file),
                  'map':map_name,'chamber_timeout_disabled':True,'mouse_preflight':'passed','preflight_records':records,
                  'initial_pose':b,'game_settings':read_settings(controller),'checkpoint':str(Path(checkpoint).resolve()),
                  'checkpoint_sha256':digest(checkpoint),'sidecar_sha256':digest(sidecar),
                  'mouse_bins':config['target_processing']['mouse_bins'],
                  'capture':{'width':camera.width,'height':camera.height,'output':camera.output_name},
                  'initial_idle_required_seconds':3.}}


def wait_idle_frames(recorder,timeout=10.):
    """Do not invite human motion until the writer has emitted the idle prefix."""
    deadline=time.monotonic()+timeout
    while recorder.frame_idx<72:
        if time.monotonic()>=deadline:raise TimeoutError('Oracle recorder did not write72idle frames')
        if recorder.tracker.pop_control_events():raise InterruptedError('Oracle capture cancelled before GO')
        time.sleep(.01)


def finalize_source(directory,outcome):
    directory=Path(directory);meta=json.loads((directory/'metadata.json').read_text())
    manifest={'kind':KIND,'outcome':outcome,'training_approved':False,
              'hashes':{name:digest(directory/name) for name in ('metadata.json','actions.csv','video.mp4')}}
    path=directory/'oracle_source.json'
    with path.open('x') as handle:json.dump(manifest,handle,indent=2);handle.write('\n')


def load_source(directory):
    directory=Path(directory).resolve();manifest=json.loads((directory/'oracle_source.json').read_text())
    if manifest.get('kind')!=KIND or manifest.get('outcome')!='goal_reached' or manifest.get('training_approved') is not False:raise ValueError('Clean successful oracle source required')
    for name in ('metadata.json','actions.csv','video.mp4'):
        if digest(directory/name)!=manifest['hashes'].get(name):raise ValueError('Oracle raw source changed')
    meta=json.loads((directory/'metadata.json').read_text());oracle=meta['oracle']
    if meta.get('source')!=KIND or meta.get('action_owner')!='human' or meta.get('training_approved') is not False:
        raise ValueError('Oracle ownership mismatch')
    verify_pose(oracle['initial_pose'],oracle['initial_pose'])
    if oracle.get('initial_idle_required_seconds')!=3. or not oracle['capture']['output'] or any(not isinstance(oracle['capture'][k],int) or oracle['capture'][k]<=0 for k in ('width','height')):raise ValueError('Invalid oracle capture profile')
    if set(oracle['game_settings'])!={'sensitivity','m_yaw','m_pitch','m_rawinput'} or not all(math.isfinite(v) for v in oracle['game_settings'].values()):raise ValueError('Invalid camera settings fingerprint')
    if not isinstance(oracle['port'],int) or not 1<=oracle['port']<=65535:raise ValueError('Invalid oracle console port')
    if oracle['fps']!=24 or meta['map']!=oracle['map'] or meta['map'] not in {f'dataset_test{i}' for i in range(1,11)}:
        raise ValueError('Oracle TRAIN map/rate mismatch')
    if digest(oracle['map_file'])!=oracle['map_sha256']:raise ValueError('Installed map changed')
    if digest(oracle['checkpoint'])!=oracle['checkpoint_sha256'] or digest(Path(oracle['checkpoint']).with_suffix('.json'))!=oracle['sidecar_sha256']:
        raise ValueError('Bin reference changed')
    reference=json.loads(Path(oracle['checkpoint']).with_suffix('.json').read_text())['config']
    if oracle['mouse_bins']!=reference['target_processing']['mouse_bins'] or (oracle.get('chamber_timeout_disabled') is not True or oracle.get('mouse_preflight')!='passed'):raise ValueError('Oracle reference/timeout mismatch')
    with (directory/'actions.csv').open(newline='') as handle:rows=list(csv.DictReader(handle))
    if not 72<len(rows)<=1440:raise ValueError('Oracle source must include3s idle and at most60s')
    sampled=[];actions=[];previous_received=-math.inf;previous_id=-1
    for i,row in enumerate(rows):
        if int(row['frame_idx'])!=i or int(row['input_discard_count'])!=0 or int(row.get('crouch','0'))!=0:
            raise ValueError('Paused, crouched or discontinuous oracle sequence')
        action={k:int(row[k]) for k in (*KEYS,'mouse_dx','mouse_dy')}
        if any(action[k] not in (0,1) for k in KEYS):raise ValueError('Invalid binary label')
        frame_id=int(row['frame_id'])
        if frame_id<previous_id:raise ValueError('Source frame IDs out of order')
        previous_id=frame_id
        values=[float(row[k]) for k in ('frame_received_monotonic','frame_sampled_monotonic','input_snapshot_started_monotonic','input_snapshot_finished_monotonic')]
        receive,sample,start,end=values
        if not all(math.isfinite(x) for x in values) or not receive<=sample or not start<=end<=sample or not 0<=sample-receive<=.25 or receive<previous_received:
            raise ValueError('Invalid capture/input timestamp')
        previous_received=receive;sampled.append(sample);actions.append(action)
    gaps=np.diff(sampled)
    if not np.all(gaps>0) or max(gaps)>.125 or abs((sampled[-1]-sampled[0])-(len(rows)-1)/24)>.25:
        raise ValueError('Oracle capture cadence invalid')
    if any(any(a.values()) for a in actions[:72]):raise ValueError('Initial oracle idle window was disturbed')
    if not any(any(a.values()) for a in actions[72:]):raise ValueError('No human action after oracle idle window')
    import cv2
    cap=cv2.VideoCapture(str(directory/'video.mp4'));frames=0
    try:
        while cap.read()[0]:frames+=1
    finally:cap.release()
    if frames!=len(actions):raise ValueError('Source video/action frame count differs')
    return {'directory':str(directory),'manifest_sha256':digest(directory/'oracle_source.json'),
            'metadata':meta,'actions':actions}


class ExpertReplay:
    """Actions are observed human labels; replay media never becomes expert data."""
    def __init__(self,directory,encoding):
        from models.imitation.mouse_bins import MouseBins
        self.source=load_source(directory);self.encoding=encoding
        if encoding not in ('raw','binned'):raise ValueError('Unknown replay encoding')
        self.mouse_bins=MouseBins(self.source['metadata']['oracle']['mouse_bins'])
        self.device='expert_replay';self.started=False;self.terminal=False;self.index=0;self.first_tick=None;self.last_tick=None
        self.actions=[dict(a) for a in self.source['actions']]
        if encoding=='binned':
            import torch
            classes=self.mouse_bins.encode(torch.tensor([[a['mouse_dx'],a['mouse_dy']] for a in self.actions],dtype=torch.float32))
            for i,a in enumerate(self.actions):
                a['mouse_dx']=self.mouse_bins.decode(0,int(classes[0][i]));a['mouse_dy']=self.mouse_bins.decode(1,int(classes[1][i]))
    @property
    def exhausted(self):return self.index>=len(self.actions)+12
    def reset(self):
        if self.started and not self.terminal:raise ReplayResetError('Replay reset during execution; aborting rather than restarting labels')
        self.index=0;self.first_tick=None;self.last_tick=None
    def finish(self):self.terminal=True
    def begin(self,controller):
        if self.started:raise ReplayResetError('Replay already started')
        o=self.source['metadata']['oracle']
        if digest(o['map_file'])!=o['map_sha256']:raise RuntimeError('Map changed before replay')
        a=read_pose(controller);time.sleep(.1);b=read_pose(controller)
        verify_pose(a,b);comparison=verify_pose(o['initial_pose'],b)
        if read_settings(controller)!=o['game_settings']:raise RuntimeError('Game camera settings changed')
        self.started=True
        return {'expected_pose':o['initial_pose'],'actual_pose':b,'comparison':comparison}
    def predict_with_diagnostics(self,frame=None):
        if not self.started or self.terminal or self.exhausted:raise RuntimeError('Replay not armed or exhausted')
        now=time.monotonic()
        if self.last_tick is not None and now-self.last_tick<1/48:raise ReplayTimingError('Replay catch-up burst rejected')
        if self.first_tick is None:self.first_tick=now
        drift=(now-self.first_tick)-self.index/24
        if abs(drift)>.125:raise ReplayTimingError(f'Replay timing drift {drift:+.3f}s; result invalid')
        tail=self.index>=len(self.actions)
        action={k:0 for k in (*KEYS,'mouse_dx','mouse_dy')} if tail else dict(self.actions[self.index])
        info={'expert_replay':{'encoding':self.encoding,'source_frame':None if tail else self.index,'phase':'release_tail' if tail else 'source','timing_drift_seconds':drift,'source_raw_action':None if tail else dict(self.source['actions'][self.index]),'expert_targets':False}}
        self.last_tick=now;self.index+=1
        return action,info
    def describe(self):
        return {'encoding':self.encoding,'source':self.source['directory'],'source_manifest_sha256':self.source['manifest_sha256'],
                'bin_reference_checkpoint':self.source['metadata']['oracle']['checkpoint'],'neural_policy_used':False,'expert_targets':False,'training_approved':False,'release_tail_ticks':12}
