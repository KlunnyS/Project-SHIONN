"""Workflow tests use fake files/controllers; never access a desktop or input."""
import contextlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock,patch
import run_expert_replay as replay
import record_dataset as record


class WorkflowTest(unittest.TestCase):
    def source(self):
        return {'directory':'/not-real/source','manifest_sha256':'sha','metadata':{'map':'dataset_test1','oracle':{'checkpoint':'/not-real/reference/best.pt','port':8020,'capture':{'width':1920,'height':1080,'output':'test-output'}}},'actions':[{}]*80}

    def test_replay_profile_rejects_missing_safeguards_before_game_access(self):
        import run_imitation
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint=Path(tmp)/'best.pt';checkpoint.touch()
            with patch.object(run_imitation,'is_game_running') as game,patch.object(run_imitation,'connect_controller') as connect:
                with self.assertRaisesRegex(ValueError,'Oracle replay requires'):
                    run_imitation.main(['--checkpoint',str(checkpoint),'--expert-replay',str(Path(tmp)/'source')])
            game.assert_not_called();connect.assert_not_called()

    def test_plan_only_never_touches_input(self):
        with patch.object(replay,'load_source',return_value=self.source()),patch('run_model_sequence.persistent_virtual_mouse') as mouse,patch('builtins.print'):
            self.assertEqual(replay.main(['--episode','/unused']),0)
        mouse.assert_not_called()
        cmd=replay.command(self.source(),'binned',Path('/unused/output'))
        for flag in ('--mouse-preflight','--hyprland-focus-guard','--hyprland-confine-pointer','--disable-chamber-timeout','--no-launch'):
            self.assertIn(flag,cmd)
        self.assertNotIn('--mouse-max-abs',cmd)
        self.assertEqual(cmd[cmd.index('--replay-encoding')+1],'binned')

    def test_raw_failure_stops_before_quantized_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(replay,'load_source',return_value=self.source()),patch('run_model_sequence.persistent_virtual_mouse',return_value=contextlib.nullcontext(object())),patch('run_model_sequence.run_persistent_attempt',return_value=0) as run,patch('run_model_sequence.inspect_attempt',return_value=('expert_replay_exhausted',Path('/log'),None)),patch('builtins.print'):
                self.assertEqual(replay.main(['--episode','/unused','--execute','--recording-root',tmp]),0)
            self.assertEqual(run.call_count,1)
            report=json.loads(next(Path(tmp).glob('*/report.json')).read_text())
            self.assertEqual(report['status'],'raw_did_not_reproduce')
            self.assertFalse(report['neural_policy_used'])

    def test_source_change_after_raw_failure_invalidates_early_result(self):
        import copy
        original=self.source();changed=copy.deepcopy(original);changed['manifest_sha256']='changed'
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(replay,'load_source',side_effect=[original,original,changed]),patch('run_model_sequence.persistent_virtual_mouse',return_value=contextlib.nullcontext(object())),patch('run_model_sequence.run_persistent_attempt',return_value=0) as run,patch('run_model_sequence.inspect_attempt',return_value=('expert_replay_exhausted',Path('/log'),None)),patch('builtins.print'):
                with self.assertRaisesRegex(ValueError,'Source changed'):
                    replay.main(['--episode','/unused','--execute','--recording-root',tmp])
            self.assertEqual(run.call_count,1)
            report=json.loads(next(Path(tmp).glob('*/report.json')).read_text())
            self.assertEqual(report['status'],'stopped_invalid_or_interrupted')

    def test_clean_raw_and_binned_require_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(replay,'load_source',return_value=self.source()),patch('run_model_sequence.persistent_virtual_mouse',return_value=contextlib.nullcontext(object())),patch('run_model_sequence.run_persistent_attempt',return_value=0) as run,patch('run_model_sequence.inspect_attempt',side_effect=[('goal_reached',Path('/rawlog'),None),('expert_replay_exhausted',Path('/binlog'),None)]),patch('builtins.print'):
                self.assertEqual(replay.main(['--episode','/unused','--execute','--recording-root',tmp]),0)
            self.assertEqual(run.call_count,2)
            report=json.loads(next(Path(tmp).glob('*/report.json')).read_text())
            self.assertEqual(report['status'],'completed_pending_review')
            self.assertFalse(report['training_approved'])

    def test_oracle_record_profile_and_single_attempt_finalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mapfile=root/'dataset_test1.bsp';mapfile.touch()
            checkpoint=root/'reference.pt';checkpoint.touch();checkpoint.with_suffix('.json').touch()
            argv=['--oracle-source','--map','dataset_test1','--episodes','1','--no-launch','--output','test-output',
                  '--oracle-map-file',str(mapfile),'--oracle-checkpoint',str(checkpoint),'--episodes-root',str(root/'sources')]
            args=record.parse_args(argv);record.validate_args(args)
            recorder=Mock();recorder.ep_dir=str(root/'finished');controller=Mock();stream=record.EpisodeEventStream();progress=record.RecordingProgress()
            pose={'position':[0,0,0],'angles':[0,90,0]}
            with patch.object(record,'load_map_and_wait_for_ready'),patch('expert_action_replay.source_metadata',return_value={'oracle':{'initial_pose':pose}}),patch('expert_action_replay.read_pose',return_value=pose),patch('expert_action_replay.wait_idle_frames'),patch('expert_action_replay.finalize_source') as finalize,patch.object(record,'wait_for_terminal_event',return_value='goal_reached') as wait,patch.object(record.time,'sleep'),patch('builtins.print'):
                record.record_episodes(recorder,controller,stream,args,progress)
            finalize.assert_called_once_with(recorder.ep_dir,'goal_reached')
            self.assertEqual(progress.attempts,1);self.assertEqual(progress.successes,1)
            self.assertTrue(wait.call_args.kwargs['require_focus']);self.assertTrue(wait.call_args.kwargs['reject_new_chamber'])
            args.episodes=2
            with self.assertRaises(ValueError):record.validate_args(args)
