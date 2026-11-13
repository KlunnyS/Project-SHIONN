"""Synthetic CPU contracts: no gameplay or real demonstration optimization."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

import run_ordered_gru_comparison as runner
from models.imitation.checkpoint import load_checkpoint, save_checkpoint
from models.imitation.dataset import BINARY_ACTION_COLUMNS
from models.imitation.network import BinnedImitationPolicy
from models.imitation.recurrent import RecurrentBinnedPolicy
from models.imitation.preprocess import PREPROCESSING_CONFIG
from models.imitation.inference import PolicyInference

BINS = {k: {'magnitude_edges': [], 'representatives': [-3., 0., 3.]} for k in ('dx', 'dy')}
CFG = {'preprocessing': PREPROCESSING_CONFIG, 'target_processing': {'mouse_bins': BINS, 'binary_class_weights': [[1., 1.]] * 8},
       'training': {'weight_decay': .0001, 'label_smoothing': .01, 'training_episodes': ['ancestor']}}


class Data:
    def __init__(self, lengths=(41, 5), prefix='base'):
        self.first_action_frames = [1] * len(lengths)
        self.episode_names = [f'{prefix}{i}' for i in range(len(lengths))]
        self.episode_maps = ['dataset_test1' if i % 2 == 0 else 'dataset_test9' for i in range(len(lengths))]
        self.episodes = []
        for length in lengths:
            x = np.arange(length, dtype=np.uint8).reshape(length, 1, 1, 1)
            y = np.zeros((length, 10), np.float32)
            y[1:, 0] = 1
            self.episodes.append((x, y))
        self.index = [(e, f) for e, (frames, _) in enumerate(self.episodes) for f in range(1, len(frames))]
    def __getitem__(self, i):
        e, f = self.index[i]
        return self.episodes[e][0][f], self.episodes[e][1][f]


class Tiny(nn.Module):
    def __init__(self, arm='gru'):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(37)
            self.trunk = nn.Sequential(nn.Flatten(1), nn.Linear(1, 4), nn.Tanh())
            self.memory = nn.GRU(4, 4, batch_first=True)
            self.hidden_size = 4
            self.binary_heads = nn.ModuleDict({k: nn.Linear(4, 2) for k in BINARY_ACTION_COLUMNS})
            self.mouse_dx_head = nn.Linear(4, 3)
            self.mouse_dy_head = nn.Linear(4, 3)


class OrderedGRUTest(unittest.TestCase):
    def test_complete_ordered_chunks_and_masks(self):
        ds = Data()
        stream = runner.EpisodeStream(ds)
        self.assertEqual(list(stream.chunks(0)), [list(range(32)), list(range(32, 40))])
        self.assertEqual(list(stream.chunks(1)), [list(range(40, 44))])
        x, y, mask = stream.batch(list(range(32, 40)))
        self.assertEqual(int(mask.sum()), 8)
        self.assertEqual(x[:8, 0, 0, 0].tolist(), list(range(33, 41)))
        self.assertFalse(mask[8:].any())
        broken = copy.deepcopy(ds)
        broken.index[3] = (1, 2)
        with self.assertRaisesRegex(ValueError, 'episode-local'):
            runner.EpisodeStream(broken)

    def test_schedule_contains_only_whole_train_episodes(self):
        streams = {'base': runner.EpisodeStream(Data()), 'human': runner.EpisodeStream(Data((8, 9), 'human'))}
        schedule = runner.make_schedule(streams, 200, 7)
        self.assertEqual(schedule, runner.make_schedule(streams, 200, 7))
        stats = runner.exposure(streams, schedule)
        self.assertGreaterEqual(stats['supervised_frames'], 200)
        self.assertEqual(stats['supervised_frames'], sum(streams[g].offsets[e][1] for g, e in schedule))
        for g, e in schedule:
            self.assertEqual(sum(map(len, streams[g].chunks(e))), streams[g].offsets[e][1])
        streams['human'].dataset.episode_maps[0] = 'dataset_test11'
        with self.assertRaisesRegex(ValueError, 'TRAIN'):
            runner.make_schedule(streams, 200)

    def test_state_carry_matches_one_pass_padding_and_reset_control(self):
        torch.manual_seed(4)
        model = Tiny().eval()
        x = torch.rand(6, 1, 1, 1)
        with torch.no_grad():
            whole, state = runner.forward_chunk(model, x, 'gru')
            first, carried = runner.forward_chunk(model, x[:3], 'gru')
            second, final = runner.forward_chunk(model, x[3:], 'gru', carried)
            for k in whole:
                torch.testing.assert_close(torch.cat((first[k], second[k]), 1), whole[k])
            torch.testing.assert_close(state, final)
            padded = torch.cat((x, torch.full_like(x[:2], float('nan'))))
            out, tail = runner.forward_chunk(model, padded, 'gru', valid=torch.tensor([True] * 6 + [False] * 2))
            torch.testing.assert_close(tail, state)
            for k in whole:
                torch.testing.assert_close(out[k][:, :6], whole[k])
            reset, _ = runner.forward_chunk(model, x, 'gru_reset')
            self.assertFalse(torch.allclose(reset['move_w'], whole['move_w']))
            each = [runner.forward_chunk(model, row[None], 'gru')[0]['move_w'] for row in x]
            torch.testing.assert_close(torch.cat(each, 1), reset['move_w'])

    def test_detach_blocks_prior_chunk_input_gradients(self):
        model = Tiny()
        first = torch.rand(2, 1, 1, 1, requires_grad=True)
        second = torch.rand(2, 1, 1, 1, requires_grad=True)
        _, hidden = runner.forward_chunk(model, first, 'gru')
        out, _ = runner.forward_chunk(model, second, 'gru', hidden.detach())
        out['move_w'].square().sum().backward()
        self.assertIsNone(first.grad)
        self.assertIsNotNone(second.grad)
        self.assertTrue(torch.isfinite(second.grad).all())

    def prepared(self, tmp):
        base, human = Data(), Data((8, 9), 'human')
        streams = {'base': runner.EpisodeStream(base), 'human': runner.EpisodeStream(human)}
        schedule = [('base', 0), ('base', 0), ('human', 1)]
        source = Path(tmp) / 'source.pt'
        model = Tiny()
        save_checkpoint(source, model=model, optimizer=torch.optim.AdamW(model.parameters()), epoch=3, global_step=100, best_val_loss=1., config=CFG)
        plan = {'exposure': runner.exposure(streams, schedule), 'source_ancestry': ['ancestor'],
            'state_policy': 'ordered_episode_state_carry_detach_each_chunk', 'train_episodes': ['base0', 'human1'],
            'schedule_sha256': runner.json_hash(schedule), 'target_trace_sha256': runner.target_trace(streams, schedule), 'input_sha256': {str(Path(runner.__file__)): 'scriptsha'}}
        parent = SimpleNamespace(source=source, cfg=copy.deepcopy(CFG), plan={'source_progress': {'epoch': 3, 'global_step': 100, 'step_in_epoch': 0}, 'held_out_corrections': ['held']})
        return SimpleNamespace(args=SimpleNamespace(output=Path(tmp) / 'output', device='cpu'), parent=parent,
            streams=streams, schedule=schedule, plan=plan, reference={'before': {'navigation': {'loss': 1.}}})

    def test_actual_imitation_updates_encoder_carries_and_resets(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.prepared(tmp)
            model = Tiny()
            before = model.trunk[1].weight.detach().clone()
            original = runner.forward_chunk
            states = []
            def forward(*args, **kwargs):
                h = args[3]
                states.append(h)
                if h is not None:
                    self.assertIsNone(h.grad_fn)
                return original(*args, **kwargs)
            progress = {'steps_completed': 0}
            with patch.object(runner, 'forward_chunk', side_effect=forward):
                _, result = runner.train(model, 'gru', p, torch.device('cpu'), runner.parent.base.Budget(20), progress)
            self.assertEqual([h is None for h in states], [True, False, True, False, True])
            self.assertEqual(result['supervised_frames'], 88)
            self.assertEqual(progress['steps_completed'], 5)
            self.assertFalse(torch.equal(before, model.trunk[1].weight))

    def test_initialization_and_native_recurrent_inference(self):
        torch.set_num_threads(1)
        source = BinnedImitationPolicy(3, 3)
        models = [runner.build_model(a, source.state_dict(), CFG, torch.device('cpu')) for a in runner.ARMS]
        for model in models:
            self.assertTrue(all(p.requires_grad for p in model.trunk.parameters()))
            self.assertEqual(runner.temporal.tensor_hash(model.trunk.state_dict()), runner.temporal.tensor_hash(source.trunk.state_dict()))
            self.assertEqual(runner.temporal.tensor_hash(model.binary_heads.state_dict()), runner.temporal.tensor_hash(models[0].binary_heads.state_dict()))
        self.assertEqual(runner.temporal.tensor_hash(models[1].memory.state_dict()), runner.temporal.tensor_hash(models[2].memory.state_dict()))
        model = models[1].eval()
        x = torch.rand(2, 12, 180, 320)
        with torch.no_grad():
            actual, hidden = runner.forward_chunk(model, x, 'gru')
            expected, state = model(x[None])
            for k in actual:
                torch.testing.assert_close(actual[k], expected[k], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(hidden, state, rtol=1e-4, atol=1e-5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            cfg = runner.endpoint_config(self.prepared(tmp), 'gru')
            save_checkpoint(path, model=model, optimizer=torch.optim.AdamW(model.parameters()), epoch=0, global_step=0, best_val_loss=1., config=cfg)
            restored = RecurrentBinnedPolicy(3, 3, 512).eval()
            load_checkpoint(path, model=restored)
            with torch.no_grad():
                out, h = restored.step(x[:1])
            for k in out:
                torch.testing.assert_close(out[k], actual[k][:, 0], rtol=1e-4, atol=1e-5)
            policy = PolicyInference(path, 'cpu')
            stacks, actions = [], []
            rng = np.random.default_rng(3)
            for _ in range(4):
                actions.append(policy.predict(rng.integers(0,256,(180,320,3),dtype=np.uint8)))
                history = list(policy.frames)
                while len(history)<4:history.insert(0,history[0])
                stacks.append(np.ascontiguousarray(np.stack(history).transpose(0,3,1,2).reshape(12,180,320)))
            with torch.no_grad():
                logits, state = runner.forward_chunk(model,torch.from_numpy(np.stack(stacks)).float()/255,'gru')
            decoded = runner.decode(logits,runner.MouseBins(BINS))
            keys=list(BINARY_ACTION_COLUMNS)+['mouse_dx','mouse_dy']
            self.assertEqual(decoded.tolist(),[[a[k] for k in keys] for a in actions])
            torch.testing.assert_close(policy.hidden,state,rtol=1e-4,atol=1e-5)

    def fake_execute(self, p, budget):
        with patch.object(runner, 'guard'), patch.object(runner, 'build_model', side_effect=lambda *args: Tiny()), \
             patch.object(runner, 'evaluate', return_value={'navigation': {'loss': 1.}}), \
             patch.object(runner.parent, 'validation_screen', return_value={'passed': True}), \
             patch.object(runner, 'development_screen', return_value={'passed': False}):
            return runner.execute(p, budget)

    def test_matched_exposure_publication_metadata_and_diagnostic_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.prepared(tmp)
            source_sha = runner.parent.file_hash(p.parent.source)
            r = self.fake_execute(p, runner.parent.base.Budget(30))
            self.assertTrue(r['comparison_complete'])
            self.assertEqual(source_sha, runner.parent.file_hash(p.parent.source))
            self.assertEqual(len({a['training']['target_trace_sha256'] for a in r['arms'].values()}), 1)
            for arm in ('feedforward', 'gru'):
                path = p.args.output / arm / 'endpoint.pt'
                ck = load_checkpoint(path)
                t = ck['config']['training']
                self.assertEqual(t['training_episodes'], ['base0', 'human1'])
                self.assertNotIn('held', t['training_episodes'])
                self.assertFalse(t['encoder_frozen'])
                self.assertFalse(t['recovery_probe']['generic_resume_supported'])
                self.assertEqual(t['optimizer_parameter_group_lrs'], {'encoder': 1e-5, 'heads_memory': 1e-3})
                self.assertEqual(t['lr_scheduler'], {'name':'none'})
                self.assertEqual(t['train_extra_cache_dirs'], [])
            with self.assertRaisesRegex(ValueError, 'missing'):
                load_checkpoint(p.args.output / 'gru_reset' / 'diagnostic_state.pt')
            self.assertFalse((p.args.output / 'gru_reset' / 'endpoint.pt').exists())

    def test_incomplete_third_arm_publishes_no_endpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.prepared(tmp)
            original = runner.train
            def fail(model, arm, *args):
                if arm == 'gru_reset':
                    raise runner.parent.base.BudgetExceeded('third arm')
                return original(model, arm, *args)
            with patch.object(runner, 'train', side_effect=fail):
                with self.assertRaises(runner.parent.base.BudgetExceeded):
                    self.fake_execute(p, runner.parent.base.Budget(30))
            self.assertFalse(list(p.args.output.glob('*/endpoint.*')))
            self.assertFalse(json.loads((p.args.output / 'report.json').read_text())['comparison_complete'])

    def test_final_publication_failure_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = self.prepared(tmp)
            replace = Path.replace
            def fail(path, target):
                if str(target).endswith('gru/endpoint.pt'):
                    raise OSError('publication failure')
                return replace(path, target)
            with patch.object(Path, 'replace', fail):
                with self.assertRaises(OSError):
                    self.fake_execute(p, runner.parent.base.Budget(30))
            self.assertFalse(list(p.args.output.glob('*/endpoint.*')))

    def test_default_only_plans(self):
        with patch.object(runner, 'prepare', return_value=SimpleNamespace(plan={})), \
             patch.object(runner, 'execute') as execute, patch('builtins.print'):
            runner.main([])
        execute.assert_not_called()

    def test_reference_rejects_changed_provenance_and_incomplete_pair(self):
        plan={'mode':'plan_only','soft_budget_seconds':100,'draw':'fixed'}
        r={'paired_complete':True,'status':'complete_no_automatic_promotion','plan':dict(plan),
            'before':{'navigation':{'complete':True,'frames':39769}},
            'arms':{a:{'status':'complete','steps_completed':1000,'evaluation':{'navigation':{'complete':True}}} for a in ['control','correction']}}
        runner.verify_source_reference(r,plan)
        wrong=dict(plan,draw='different')
        with self.assertRaisesRegex(ValueError,'provenance'):runner.verify_source_reference(r,wrong)
        r['arms']['control']['steps_completed']=999
        with self.assertRaisesRegex(ValueError,'incomplete'):runner.verify_source_reference(r,plan)

    def test_ordered_evaluation_reset_and_opening_denominators(self):
        ds=Data((7,5));stream=runner.EpisodeStream(ds);model=Tiny().eval()
        result=runner.evaluate_dataset(model,'gru',stream,CFG,torch.device('cpu'),runner.parent.base.Budget(20))
        self.assertEqual(result['frames'],10)
        self.assertEqual(result['episodes'],2)
        expected=[]
        with torch.no_grad():
            for e in range(2):
                ix=list(stream.chunks(e))[0];x,y,m=stream.batch(ix)
                out,h=runner.forward_chunk(model,torch.from_numpy(x).float()/255,'gru',valid=torch.from_numpy(m))
                expected.append(runner.decode(out,runner.MouseBins(BINS))[:len(ix)].tolist())
        self.assertEqual([r['predictions'] for r in result['records']],expected)
        self.assertEqual(result['segments']['first8']['frames'],10)

    def test_development_gate_does_not_accept_accuracy_alone(self):
        def agg(binary,tp=8,fp=2):return {'episode_mean':{'binary_exact':binary},'backward_tp':tp,'false_backward':fp}
        report={'source_baseline':{'development_holdout':{'aggregate':agg(.69)}},'arms':{}}
        for arm,score in [('feedforward',.66),('gru',.70),('gru_reset',.67)]:
            report['arms'][arm]={'navigation_screen':{'passed':True},'after':{'human_held':{'aggregate':agg(score)}}}
        self.assertTrue(runner.development_screen(report)['passed'])
        report['arms']['gru']['navigation_screen']['passed']=False
        self.assertFalse(runner.development_screen(report)['passed'])

    def test_expiry_after_publication_rolls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=self.prepared(tmp);budget=runner.parent.base.Budget(30);check=budget.check
            def expire():
                if (p.args.output/'feedforward/endpoint.pt').exists():raise runner.parent.base.BudgetExceeded('publication expiry')
                check()
            with patch.object(budget,'check',side_effect=expire):
                with self.assertRaises(runner.parent.base.BudgetExceeded):self.fake_execute(p,budget)
            self.assertFalse(list(p.args.output.glob('*/endpoint.*')))

    def test_category_exposure_does_not_change_targets(self):
        base,human=Data(),Data((8,9),'human')
        human.episode_categories={'human0':'exit_finalization','human1':'route_recovery'}
        streams={'base':runner.EpisodeStream(base),'human':runner.EpisodeStream(human)}
        schedule=[('base',0),('human',0),('human',1)]
        result=runner.exposure(streams,schedule)
        self.assertEqual(result['frames_by_category'],{'original_navigation':40,'exit_finalization':7,'route_recovery':8})
        self.assertEqual(sum(result['frames_by_category'].values()),result['supervised_frames'])
