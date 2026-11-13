"""Offline decision-rate contracts; no game, capture or input devices."""
import unittest
from policy_decision_rate import DecisionRatePolicy, decision_interval


class FakePolicy:
    def __init__(self):
        self.frames = []
        self.calls = []
    def add_frame(self, frame):
        self.frames.append(frame)
    def reset(self):
        self.frames.clear()
    def predict_with_diagnostics(self):
        self.calls.append(list(self.frames[-4:]))
        return {'move_w': 1, 'mouse_dx': 7, 'mouse_dy': -2}, {'history': list(self.frames[-4:])}


class DecisionRateTest(unittest.TestCase):
    def test_rates_preserve_history_and_mouse_distance(self):
        for fps, count in ((24,24),(12,12),(8,8)):
            with self.subTest(fps=fps):
                source = FakePolicy()
                policy = DecisionRatePolicy(source, decision_interval(24,fps))
                actions = [policy.predict_with_diagnostics(i)[0] for i in range(24)]
                self.assertEqual(len(source.calls),count)
                self.assertEqual(source.frames,list(range(24)))
                self.assertEqual(sum(a['mouse_dx'] for a in actions),168)
                interval = 24//fps
                self.assertEqual(source.calls[1],list(range(max(0,interval-3),interval+1)))

    def test_reset_forces_fresh_decision_and_clears_history(self):
        source = FakePolicy(); policy = DecisionRatePolicy(source,3)
        policy.predict_with_diagnostics(1)
        policy.reset()
        _, info = policy.predict_with_diagnostics(9)
        self.assertEqual(source.calls,[[1],[9]])
        self.assertTrue(info['decision_rate']['refreshed'])
        self.assertEqual(info['decision_rate']['decision_index'],0)

    def test_cached_results_are_isolated_and_marked(self):
        policy = DecisionRatePolicy(FakePolicy(),3)
        action, info = policy.predict_with_diagnostics(0)
        action['mouse_dx']=999; info['history'].append(999)
        action, info = policy.predict_with_diagnostics(1)
        self.assertEqual(action['mouse_dx'],7)
        self.assertEqual(info['history'],[0])
        self.assertFalse(info['decision_rate']['refreshed'])
        self.assertEqual(info['decision_rate']['source_observation_tick'],0)

    def test_validation(self):
        self.assertEqual(decision_interval(24,None),1)
        for rate in (0,-1,25,10,float('nan'),float('inf'),1e-320):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                decision_interval(24,rate)
        with self.assertRaises(ValueError): decision_interval(float('nan'),12)

    def test_sequence_cli_forwards_rate(self):
        import tempfile
        from pathlib import Path
        from run_model_sequence import parse_args, build_jobs, runner_command
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'best.pt'; checkpoint.touch(); checkpoint.with_suffix('.json').touch()
            args=parse_args(['--checkpoint',str(checkpoint),'--map','test1','--model-fps','8'])
            self.assertEqual(args.model_fps,8)
            job=build_jobs(args.models,args.maps,1,args.order)[0]
            command=runner_command(job,Path(directory),args)
            self.assertEqual(command[command.index("--model-fps")+1],"8.0")

if __name__ == '__main__': unittest.main()
