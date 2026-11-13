"""Fixed diagnostic exposure, source preservation and no-export contracts."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
import run_recovery_turn_fit_diagnostic as runner
import tests.test_route_expansion as fixtures


class FitDiagnosticTest(unittest.TestCase):
    def test_fixed_exposure_and_original_settings_preserved(self):
        cfg={'training':{'weight_decay':.0001,'augmentation':{'brightness':.2,'contrast':.2,'horizontal_flip_probability':.5}}}
        model=torch.nn.Linear(1,1);seen=[]
        def train(model,loader,opt,device,settings,budget,progress):
            self.assertEqual(len(opt.state),0)
            self.assertEqual(settings['training']['augmentation'],{'brightness':0.,'contrast':0.,'horizontal_flip_probability':0.})
            for x,y in loader:seen.extend(x[:,0].tolist());self.assertEqual(len(x),8)
            progress['steps_completed']=100
            return {'samples':800}
        with patch.object(runner.base,'train_probe_steps',side_effect=train):
            r=runner.fit(model,torch.arange(40.).reshape(40,1),torch.zeros(40,1),cfg,torch.device('cpu'),runner.base.Budget(10))
        self.assertEqual(r['steps_completed'],100)
        self.assertEqual([seen.count(float(i)) for i in range(40)],[20]*40)
        self.assertEqual(cfg['training']['augmentation']['brightness'],.2)

    def test_wrong_target_count_refused(self):
        with self.assertRaises(ValueError):runner.fit(None,torch.zeros(39,1),torch.zeros(39,1),{},torch.device('cpu'),runner.base.Budget(10))

    def test_execute_never_exports_weights_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=fixtures.TargetedTest().prepared(tmp);digest=runner.file_hash(p.source)
            args=SimpleNamespace(output=Path(tmp)/'diagnostic',device='cpu')
            ds=SimpleNamespace(episode_names=['one'],first_action_frames=[0])
            # The fixture dataset supplies tiny labelled samples via __getitem__.
            class DS:
                episode_names=['one'];first_action_frames=[0]
                def __getitem__(self,i):return torch.zeros(1).numpy(),torch.zeros(1).numpy()
            def fitted(*args):return {'steps_completed':100,'target_exposures':800}
            with patch.object(runner.base,'fresh_model',return_value=fixtures.Tiny()),patch.object(runner,'score_windows',return_value={'complete':True}),patch.object(runner,'fit',side_effect=fitted),patch('torch.save',side_effect=AssertionError('weight export forbidden')):
                report=runner.execute(args,DS(),list(range(40)),p.source,{str(p.source):digest},runner.base.Budget(10))
            self.assertTrue(report['complete']);self.assertFalse(report['weights_exported'])
            self.assertEqual(runner.file_hash(p.source),digest)
            self.assertEqual([p.name for p in args.output.iterdir()],['report.json'])

    def test_plan_only_does_not_execute(self):
        with patch.object(runner,'prepare',return_value=(None,list(range(40)),None,{})),patch.object(runner,'execute') as execute,patch('builtins.print'):
            runner.main([])
        execute.assert_not_called()
