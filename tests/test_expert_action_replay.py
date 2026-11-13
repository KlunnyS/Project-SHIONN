"""Pure oracle mapping, timing, pose and source-validation tests; no live input."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch,Mock
from expert_action_replay import (ExpertReplay,ReplayPoseError,ReplayResetError,ReplayTimingError,
                                  parse_pose,verify_pose,digest,load_source,finalize_source,KIND,KEYS)


class OracleTest(unittest.TestCase):
    def source(self):
        sidecar=Path('models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.json')
        bins=json.loads(sidecar.read_text())['config']['target_processing']['mouse_bins']
        rows=[dict.fromkeys((*KEYS,'mouse_dx','mouse_dy'),0) for _ in range(3)]
        rows[0].update(move_w=1,mouse_dx=200,mouse_dy=-30)
        return {'directory':'/unused/source','manifest_sha256':'sha','metadata':{'oracle':{'mouse_bins':bins,'checkpoint':'/unused/best.pt'}},'actions':rows}

    def policy(self,encoding='raw'):
        with patch('expert_action_replay.load_source',return_value=self.source()):return ExpertReplay('/unused',encoding)

    def test_raw_and_quantized_preserve_binary_and_original_labels(self):
        raw,binned=self.policy(),self.policy('binned')
        self.assertEqual(raw.actions[0]['mouse_dx'],200)
        self.assertEqual(binned.actions[0]['mouse_dx'],134)
        self.assertEqual(binned.actions[0]['mouse_dy'],-23)
        self.assertEqual(binned.source['actions'][0]['mouse_dx'],200)
        for a,b in zip(raw.actions,binned.actions):
            for k in KEYS:self.assertEqual(a[k],b[k])

    def test_exhaustion_release_tail_and_no_extra_action(self):
        p=self.policy();p.started=True
        with patch('expert_action_replay.time.monotonic',side_effect=[i/24 for i in range(15)]):
            result=[p.predict_with_diagnostics(None) for _ in range(15)]
        self.assertTrue(p.exhausted)
        self.assertTrue(all(not any(a.values()) for a,_ in result[3:]))
        self.assertEqual(result[0][1]['expert_replay']['source_raw_action']['mouse_dx'],200)
        with self.assertRaises(RuntimeError):p.predict_with_diagnostics(None)

    def test_focus_reset_aborts_but_terminal_cleanup_is_allowed(self):
        p=self.policy();p.reset();p.started=True
        with self.assertRaises(ReplayResetError):p.reset()
        p.finish();p.reset()
        with self.assertRaises(RuntimeError):p.predict_with_diagnostics(None)

    def test_timing_drift_and_bursts_are_rejected(self):
        for times in ([0,.2],[0,0]):
            p=self.policy();p.started=True
            with patch('expert_action_replay.time.monotonic',side_effect=times):
                p.predict_with_diagnostics(None)
                with self.assertRaises(ReplayTimingError):p.predict_with_diagnostics(None)

    def test_settings_parser_waits_for_complete_values(self):
        from expert_action_replay import read_settings
        c=Mock();c.read_console.side_effect=['','"sensitivity" = "1','.536"\n','','"m_yaw" = "0.022"\n','','"m_pitch" = "0.022"\n','','"m_rawinput" = "1"\n']
        with patch('expert_action_replay.time.sleep'):
            settings=read_settings(c)
        self.assertEqual(settings['sensitivity'],1.536)
        self.assertEqual(settings['m_rawinput'],1.)

    def test_source_preflight_reload_and_stop_before_capture(self):
        from expert_action_replay import source_metadata
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'best.pt';checkpoint.write_bytes(b'checkpoint')
            checkpoint.with_suffix('.json').write_text(json.dumps({'config':{'target_processing':{'mouse_bins':self.source()['metadata']['oracle']['mouse_bins']}}}))
            mapfile=root/'dataset_test1.bsp';mapfile.write_bytes(b'map')
            c=Mock(port=8020);tracker=Mock();tracker.get_snapshot_and_reset.return_value={}
            camera=Mock(width=1920,height=1080,output_name='test-output');pose={'position':[0,0,0],'angles':[0,90,0]}
            events=[]
            with patch('expert_action_replay.CaptureStop.wait',return_value=False),patch('expert_action_replay.read_pose',return_value=pose),patch('expert_action_replay.read_settings',return_value={}),patch('run_imitation.query_hyprland_active_window',return_value={'portal_focused':True}),patch('run_imitation.prepare_portal_visual_state',return_value={}),patch('run_imitation.activate_portal_input',return_value={'portal_focused':True}),patch('run_imitation.mouse_capture_preflight',side_effect=lambda *a:events.append('preflight')),patch('run_imitation.load_recovery_map',side_effect=lambda *a:events.append('reload')),patch('run_imitation.disable_chamber_timeout') as disable:
                meta=source_metadata(c,tracker,'dataset_test1',mapfile,checkpoint,camera)
            self.assertEqual(events,['preflight','reload'])
            self.assertFalse(meta['training_approved'])
            self.assertEqual(meta['oracle']['initial_pose'],pose)
            disable.assert_called_once_with(c,require_active=True)
            with patch('expert_action_replay.CaptureStop.wait',return_value=True),patch('run_imitation.mouse_capture_preflight') as preflight:
                with self.assertRaises(InterruptedError):source_metadata(c,tracker,'dataset_test1',mapfile,checkpoint,camera)
            preflight.assert_not_called()

    def test_complete_pose_and_wrap_tolerance(self):
        p=parse_pose('setpos -1.2 2 3;setang 0 179.9 0\n')
        q=copy.deepcopy(p);q['angles'][1]=-179.9
        self.assertAlmostEqual(verify_pose(p,q)['max_angle_error_degrees'],.2)
        q['position'][0]+=1
        with self.assertRaises(ReplayPoseError):verify_pose(p,q)
        with self.assertRaises(ValueError):parse_pose('setang 0 90 0')

    def fixture(self,root):
        source=self.source();ref=root/'best.pt';ref.write_bytes(b'checkpoint')
        side=ref.with_suffix('.json');side.write_text(json.dumps({'config':{'target_processing':{'mouse_bins':source['metadata']['oracle']['mouse_bins']}}}))
        mapfile=root/'dataset_test1.bsp';mapfile.write_bytes(b'map');ep=root/'episode';ep.mkdir()
        meta={'source':KIND,'map':'dataset_test1','action_owner':'human','training_approved':False,
              'oracle':{'fps':24,'port':8020,'map':'dataset_test1','map_file':str(mapfile),'map_sha256':digest(mapfile),
                        'initial_pose':{'position':[0,0,0],'angles':[0,90,0]},'capture':{'width':1920,'height':1080,'output':'test-output'},'game_settings':{'sensitivity':1.,'m_yaw':.022,'m_pitch':.022,'m_rawinput':1.},'initial_idle_required_seconds':3.,'mouse_preflight':'passed','chamber_timeout_disabled':True,'checkpoint':str(ref),'checkpoint_sha256':digest(ref),
                        'sidecar_sha256':digest(side),'mouse_bins':source['metadata']['oracle']['mouse_bins']}}
        (ep/'metadata.json').write_text(json.dumps(meta));(ep/'video.mp4').write_bytes(b'video')
        fields=['frame_idx','frame_id',*KEYS,'mouse_dx','mouse_dy','crouch','input_discard_count','frame_received_monotonic','frame_sampled_monotonic','input_snapshot_started_monotonic','input_snapshot_finished_monotonic']
        with (ep/'actions.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
            for i in range(80):
                row=dict.fromkeys(fields,0);t=10+i/24
                row.update(frame_idx=i,frame_id=i,frame_received_monotonic=t-.01,frame_sampled_monotonic=t,input_snapshot_started_monotonic=t-.002,input_snapshot_finished_monotonic=t-.001)
                if i>=72:row['move_w']=1
                w.writerow(row)
        finalize_source(ep,'goal_reached');return ep

    def load(self,ep):
        cap=Mock();cap.read.side_effect=[(True,None)]*80+[(False,None)]
        with patch('cv2.VideoCapture',return_value=cap):return load_source(ep)

    def test_wait_for_written_idle_frames_and_bounded_timeout(self):
        from expert_action_replay import wait_idle_frames
        class Recorder:
            tracker=Mock(pop_control_events=Mock(return_value=[]))
            counts=iter((50,60,72))
            @property
            def frame_idx(self):return next(self.counts)
        with patch('expert_action_replay.time.monotonic',return_value=0),patch('expert_action_replay.time.sleep') as sleep:
            wait_idle_frames(Recorder())
        self.assertEqual(sleep.call_count,2)
        stalled=Mock(frame_idx=0)
        with patch('expert_action_replay.time.monotonic',side_effect=[0,11]):
            with self.assertRaises(TimeoutError):wait_idle_frames(stalled)

    def test_full_three_second_idle_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep=self.fixture(Path(tmp));rows=list(csv.DictReader((ep/'actions.csv').open()))
            rows[65]['mouse_dx']='20'
            with (ep/'actions.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (ep/'oracle_source.json').unlink();finalize_source(ep,'goal_reached')
            with self.assertRaisesRegex(ValueError,'idle window'):self.load(ep)

    def test_source_hash_map_and_pause_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep=self.fixture(Path(tmp));self.assertEqual(len(self.load(ep)['actions']),80)
            (ep/'actions.csv').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source changed'):self.load(ep)
        with tempfile.TemporaryDirectory() as tmp:
            ep=self.fixture(Path(tmp));meta=json.loads((ep/'metadata.json').read_text())
            Path(meta['oracle']['map_file']).write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'map changed'):self.load(ep)
        with tempfile.TemporaryDirectory() as tmp:
            ep=self.fixture(Path(tmp))
            # Explicitly edit a discontinuity field, then refresh manifest to test semantic validation.
            rows=list(csv.DictReader((ep/'actions.csv').open()));rows[-1]['input_discard_count']='1'
            with (ep/'actions.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (ep/'oracle_source.json').unlink();finalize_source(ep,'goal_reached')
            with self.assertRaisesRegex(ValueError,'discontinuous'):self.load(ep)
