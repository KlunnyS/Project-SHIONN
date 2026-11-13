"""Fake-only observer tests: no Wayland, ffmpeg, devices, game or real signals."""
import json
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import observation_context as context
import run_recovery_session as session


class Camera:
    def __init__(self, stale=False):
        self.started = self.stopped = False
        self.calls = 0
        self.stale = stale
    def start(self):self.started = True
    def stop(self):self.stopped = True
    def error_summary(self):return 'fake error'
    def get_latest_frame_sample(self):
        self.calls += 1
        return SimpleNamespace(frame=np.zeros((2,2,3),np.uint8),frame_id=self.calls,
            received_monotonic=time.monotonic() - (3 if self.stale else 0))


class Writer:
    def __init__(self, broken=False, finalize=False):
        self.frames = 0
        self.released = False
        self.broken, self.finalize = broken, finalize
    def write(self, frame):
        if self.broken:raise RuntimeError('fake write failure')
        self.frames += 1
    def release(self):
        self.released = True
        if self.finalize:raise RuntimeError('fake finalize failure')


class ObservationContextTest(unittest.TestCase):
    def test_frames_timestamps_and_no_expert_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            camera, writer = Camera(), Writer()
            result = context.record_observations(Path(tmp),camera,writer,fps=1000,
                stopped=lambda:writer.frames>=3,requested_map='dataset_test1')
            self.assertEqual(result['capture_status'],'complete')
            self.assertFalse(result['expert_targets'])
            self.assertEqual(result['labels'],'none')
            self.assertFalse(result['training_approved'])
            rows=[json.loads(l) for l in (Path(tmp)/'frames.jsonl').read_text().splitlines()]
            self.assertEqual([r['frame_idx'] for r in rows],[0,1,2])
            self.assertTrue(all(r['sampled_monotonic']>=r['received_monotonic'] for r in rows))
            self.assertTrue(all(not r['expert_targets'] for r in rows))
            self.assertTrue(camera.stopped and writer.released)
            self.assertFalse((Path(tmp)/'actions.csv').exists())

    def test_stale_write_and_finalize_failures_remain_failed(self):
        for stale,broken,finalize in [(True,False,False),(False,True,False),(False,False,True)]:
            with self.subTest(stale=stale,broken=broken,finalize=finalize),tempfile.TemporaryDirectory() as tmp:
                camera,writer=Camera(stale),Writer(broken,finalize)
                with self.assertRaises(RuntimeError):
                    context.record_observations(Path(tmp),camera,writer,fps=1000,stopped=lambda:writer.frames>=1)
                state=json.loads((Path(tmp)/'metadata.json').read_text())
                self.assertEqual(state['capture_status'],'failed')
                self.assertTrue(camera.stopped and writer.released)
                self.assertFalse(state['training_approved'])

    def test_parent_wait_is_bounded_and_kills_only_owned_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer=context.ObservationContext(Path(tmp),width=2,height=2,fps=24)
            proc=SimpleNamespace(pid=987654,returncode=-9,poll=lambda:None,
                send_signal=lambda sig:None)
            observer.proc=proc
            def wait(timeout):
                if timeout==8.:raise subprocess.TimeoutExpired('fake observer',timeout)
                return -9
            proc.wait=wait
            with patch.object(context.os,'killpg') as kill:
                with self.assertRaisesRegex(RuntimeError,'finalize'):
                    observer.close()
                self.assertTrue(any(call.args==(987654,signal.SIGKILL) for call in kill.call_args_list))
                self.assertTrue(all(call.args[0]==987654 for call in kill.call_args_list))
            self.assertEqual(observer.state()['capture_status'],'failed')

    def test_healthy_close_and_orchestration_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer=context.ObservationContext(Path(tmp),width=2,height=2,fps=24)
            context.atomic_json(Path(tmp)/'metadata.json',{'capture_status':'recording','heartbeat_monotonic':time.monotonic()})
            proc=SimpleNamespace(pid=987654,returncode=0,poll=lambda:None,send_signal=lambda sig:None)
            def wait(timeout):
                context.atomic_json(Path(tmp)/'metadata.json',{'capture_status':'complete','expert_targets':False})
                return 0
            proc.wait=wait;observer.proc=proc
            observer.ensure_healthy();observer.mark('model_subprocess_returned',attempt_log='attempt.jsonl')
            with patch.object(context.os,'killpg',side_effect=ProcessLookupError):observer.close()
            row=json.loads((Path(tmp)/'links.jsonl').read_text())
            self.assertIn('orchestration_only',row['boundary_precision'])
            self.assertEqual(row['attempt_log'],'attempt.jsonl')

    def test_context_flag_is_opt_in_and_incompatible_with_no_media_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/'best.pt';ck.touch()
            args=session.parse_args(['--checkpoint',str(ck)])
            self.assertFalse(args.record_context)
            with patch('observation_context.ObservationContext') as constructor:
                with session.observation_context(args,Path(tmp)/'attempt','dataset_test1') as observer:
                    self.assertIsNone(observer)
                constructor.assert_not_called()
            with self.assertRaises(SystemExit):session.parse_args(['--checkpoint',str(ck),'--record-context','--test-mode'])

    def test_session_observer_spans_both_subprocesses_and_closes_on_return(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ck=root/'best.pt';ck.touch();calls=[]
            class Observer:
                directory=root/'context'
                active=False
                def __enter__(self):self.active=True;calls.append('enter');return self
                def close(self):
                    if self.active:self.active=False;calls.append('close')
                def __exit__(self,*args):self.close()
                def ensure_healthy(self):assert self.active;calls.append('healthy')
                def mark(self,phase,**links):calls.append(phase)
            observer=Observer()
            def run(command,**kwargs):
                self.assertTrue(observer.active)
                if command[1].endswith('run_imitation.py'):
                    calls.append('model')
                    p=Path(command[command.index('--recording-dir')+1]);p.mkdir(parents=True,exist_ok=True)
                    (p/'attempt_fake.jsonl').write_text(json.dumps({'type':'summary','stop_reason':'recovery_handoff'})+'\n')
                else:
                    calls.append('human')
                    Path(command[command.index('--result-file')+1]).write_text(json.dumps({'outcome':'goal_reached','episode':str(root/'human_episode')}))
                return SimpleNamespace(returncode=0)
            with patch.object(session,'observation_context',return_value=observer),patch.object(session.subprocess,'run',side_effect=run):
                result=session.main(['--checkpoint',str(ck),'--map','dataset_test1','--target-corrections','1','--session-root',str(root/'sessions'),'--record-context'])
            self.assertEqual(result,0)
            self.assertEqual(calls[0],'enter');self.assertEqual(calls[-1],'close')
            self.assertLess(calls.index('model'),calls.index('human'))
            self.assertLess(calls.index('human'),calls.index('close'))
            entries=[json.loads(l) for l in next((root/'sessions').glob('session_*/session.jsonl')).read_text().splitlines()]
            self.assertEqual(entries[-1]['observation_context'],str(observer.directory))

    def test_observer_failure_prevents_human_launch_and_next_attempt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ck=root/'best.pt';ck.touch();calls=[]
            class Observer:
                def __enter__(self):return self
                def __exit__(self,*args):calls.append('closed')
                def ensure_healthy(self):raise RuntimeError('capture failed')
                def mark(self,*args,**kwargs):pass
            with patch.object(session,'observation_context',return_value=Observer()),patch.object(session.subprocess,'run',return_value=SimpleNamespace(returncode=0)) as run:
                with self.assertRaisesRegex(RuntimeError,'capture failed'):
                    session.main(['--checkpoint',str(ck),'--target-corrections','2','--session-root',str(root/'sessions'),'--record-context'])
                self.assertEqual(run.call_count,1)
                self.assertEqual(calls,['closed'])

    def test_startup_failure_reaps_worker_before_policy_can_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            observer=context.ObservationContext(Path(tmp)/'context',width=2,height=2,fps=24)
            proc=SimpleNamespace(pid=987654,returncode=1,poll=lambda:1,wait=lambda timeout:1)
            with patch.object(context.subprocess,'Popen',return_value=proc) as popen,patch.object(context.os,'killpg',side_effect=ProcessLookupError):
                with self.assertRaises(RuntimeError):observer.__enter__()
            self.assertTrue(popen.call_args.kwargs['start_new_session'])
            self.assertEqual(observer.state()['capture_status'],'failed')

    def test_finalize_failure_is_not_counted_as_correction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);ck=root/'best.pt';ck.touch()
            class Observer:
                directory=root/'context'
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def ensure_healthy(self):pass
                def mark(self,*args,**kwargs):pass
                def close(self):raise RuntimeError('finalize failed')
            def run(command,**kwargs):
                if command[1].endswith('run_imitation.py'):
                    p=Path(command[command.index('--recording-dir')+1]);p.mkdir(parents=True,exist_ok=True)
                    (p/'attempt_fake.jsonl').write_text(json.dumps({'type':'summary','stop_reason':'recovery_handoff'})+'\n')
                else:Path(command[command.index('--result-file')+1]).write_text(json.dumps({'outcome':'goal_reached','episode':'human_episode'}))
                return SimpleNamespace(returncode=0)
            with patch.object(session,'observation_context',return_value=Observer()),patch.object(session.subprocess,'run',side_effect=run) as child:
                with self.assertRaisesRegex(RuntimeError,'finalize failed'):
                    session.main(['--checkpoint',str(ck),'--target-corrections','1','--session-root',str(root/'sessions'),'--record-context'])
            self.assertEqual(child.call_count,2)
            rows=[json.loads(l) for l in next((root/'sessions').glob('session_*/session.jsonl')).read_text().splitlines()]
            self.assertFalse(any(r['type']=='recovery' for r in rows))

    def audit_fixture(self, root):
        from audit_observation_context import audit
        directory=root/'context';directory.mkdir()
        source=root/'attempt.jsonl';human=root/'human';human.mkdir()
        context.atomic_json(directory/'metadata.json',{'kind':'shionn_unlabelled_observation_context_v1',
            'expert_targets':False,'labels':'none','training_approved':False,'capture_status':'complete',
            'requested_map':'dataset_test1','frames':9,'video':'context.mp4'})
        frames=[{'frame_idx':i,'frame_id':i+1,'received_monotonic':i*.1,'sampled_monotonic':i*.1+.01,
                 'frame_age_seconds':.01,'expert_targets':False} for i in range(9)]
        (directory/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in frames))
        source.write_text(''.join(json.dumps(r)+'\n' for r in [
            {'type':'metadata','map':'dataset_test1'}, {'type':'tick','monotonic_seconds':.1},
            {'type':'stop','reason':'recovery_handoff','monotonic_seconds':.3}]))
        context.atomic_json(human/'metadata.json',{'map':'dataset_test1','source':'model_recovery','source_attempt':str(source)})
        (human/'actions.csv').write_text('frame_received_monotonic,frame_sampled_monotonic\n0.6,0.61\n0.7,0.71\n')
        (directory/'links.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in [
            {'phase':'model_subprocess_returned','attempt_log':str(source)},
            {'phase':'human_recorder_returned','episode':str(human)}]))
        return audit,directory,source,human

    def test_coverage_audit_never_approves_labels_and_rejects_sampling_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit,directory,source,human=self.audit_fixture(Path(tmp))
            result=audit(directory,source,human,decode_video=False)
            self.assertTrue(result['passed']);self.assertFalse(result['training_approved'])
            self.assertEqual(result['observations_during_handoff'],3)
            rows=[json.loads(l) for l in (directory/'frames.jsonl').read_text().splitlines()]
            rows=[r for r in rows if not .2<r['sampled_monotonic']<.6]
            for i,r in enumerate(rows):r['frame_idx']=i
            (directory/'frames.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            meta=json.loads((directory/'metadata.json').read_text());meta['frames']=len(rows);context.atomic_json(directory/'metadata.json',meta)
            result=audit(directory,source,human,decode_video=False)
            self.assertFalse(result['passed']);self.assertFalse(result['checks']['sample_gaps_bounded'])

    def test_coverage_audit_rejects_wrong_human_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            audit,directory,source,human=self.audit_fixture(Path(tmp))
            meta=json.loads((human/'metadata.json').read_text());meta['source_attempt']='wrong.jsonl';context.atomic_json(human/'metadata.json',meta)
            self.assertFalse(audit(directory,source,human,decode_video=False)['checks']['human_provenance'])
