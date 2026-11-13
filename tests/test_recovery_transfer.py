import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from models.imitation import recovery_transfer as rt
from models.imitation.mouse_bins import MouseBins
from benchmark_recovery_transfer import decode_actions


class RecoveryTransferTests(unittest.TestCase):
    def test_nonfinite_logits_are_rejected_before_argmax(self):
        from models.imitation.dataset import BINARY_ACTION_COLUMNS
        bins=MouseBins({axis:{'magnitude_edges':[],'representatives':[-1,0,1]} for axis in ['dx','dy']})
        output={k:torch.zeros((1,2)) for k in BINARY_ACTION_COLUMNS}
        output.update(mouse_dx_logits=torch.tensor([[float('nan'),0.,0.]]),mouse_dy_logits=torch.zeros((1,3)))
        with self.assertRaises(FloatingPointError):decode_actions(output,bins)
    def test_protocol_rejects_overlap_shifted_windows_and_forged_map(self):
        original=json.loads(Path('data/annotations/recovery_transfer_development_v1.json').read_text())
        changes=[lambda p:p['train_recovery_episodes'].__setitem__(0,p['validation_recovery_episodes'][0]),
                 lambda p:p['records'][0]['offline_window'].__setitem__(0,99999),
                 lambda p:p['records'][0].update(map='dataset_test2'),
                 lambda p:p['records'][0].update(role='diagnostic_only'),
                 lambda p:p.update(parent_frozen_split_sha256='wrong')]
        for change in changes:
            with tempfile.TemporaryDirectory() as tmp:
                p=copy.deepcopy(original);change(p);path=Path(tmp)/'p.json';path.write_text(json.dumps(p))
                with self.assertRaises(ValueError):rt.load_protocol(path)

    def test_exposure_does_not_mislabel_missing_metadata_clean(self):
        p={'train_recovery_episodes':['t'],'validation_recovery_episodes':['v'],'diagnostic_only_episodes':['d']}
        self.assertIsNone(rt.exposure(p,{})['recovery_validation_exposed'])
        cfg={'training':{'training_episodes':['t','v']}}
        self.assertTrue(rt.exposure(p,cfg)['recovery_validation_exposed'])
        with self.assertRaisesRegex(ValueError,'reserved'):rt.validate_training_membership(p,cfg)

    def test_training_loader_excludes_reserved_and_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'source.pt';source.write_bytes(b'fixture')
            cfg={'architecture':'test','training':{'training_episodes':['base'],'validation_episodes':['nav']}}
            source.with_suffix('.json').write_text(json.dumps({'config':cfg}))
            p={'source':str(source),'source_sha256':rt.file_hash(source),
               'source_sidecar_sha256':rt.file_hash(source.with_suffix('.json')),'source_architecture':'test',
               'source_training_episode_count':1,'train_recovery_episodes':['t'],
               'validation_recovery_episodes':['v'],'diagnostic_only_episodes':['d']}
            context={k:[SimpleNamespace(name=n)] for k,n in [('base_train','base'),('navigation_validation','nav'),('recovery_train','t'),('recovery_validation','v')]}
            train,held=rt.training_partitions(p,context)
            self.assertEqual([m.name for m in train],['base','t']);self.assertEqual([m.name for m in held],['v'])
            context['recovery_train']=[SimpleNamespace(name='v')]
            with self.assertRaisesRegex(ValueError,'crossed'):rt.training_partitions(p,context)

    def test_streaks_reset_at_episode_boundaries_and_support_is_null(self):
        e=np.zeros((3,10));p=np.zeros_like(e)
        r=rt.episode_metrics(e,p)
        self.assertEqual(r['longest_idle_seconds'],3/24)
        self.assertIsNone(r['dx']['sign_accuracy'])
        a=rt.aggregate([r,r]);self.assertEqual(a['episode_mean']['longest_idle_seconds'],3/24)
        self.assertIsNone(a['backward_recall']);self.assertEqual(a['dx']['significant_frames'],0)

    def test_episode_and_frame_weighting_are_distinct(self):
        e=np.zeros((1,10));p=np.zeros_like(e);a=rt.episode_metrics(e,p)
        e=np.zeros((9,10));p=np.ones_like(e);b=rt.episode_metrics(e,p)
        r=rt.aggregate([a,b]);self.assertEqual(r['episode_mean']['idle_fraction'],.5)
        self.assertEqual(r['frame_weighted']['idle_fraction'],.1)

    def test_sign_metrics_use_threshold_and_track_false_backward(self):
        e=np.zeros((4,10));p=e.copy();e[:,8]=[1,3,-4,0];p[:,8]=[-1,2,0,0]
        e[0,2]=1;p[1,2]=1
        r=rt.episode_metrics(e,p)
        self.assertEqual(r['dx']['significant_frames'],2)
        self.assertEqual(r['dx']['sign_correct'],1)
        self.assertEqual(r['dx']['nonzero_predicted_zero'],1)
        self.assertEqual(r['backward_tp'],0);self.assertEqual(r['false_backward'],1)

    def test_incomplete_or_nonfinite_predictions_rejected(self):
        for e,p in [(np.zeros((2,10)),np.zeros((1,10))),(np.zeros((1,10)),np.full((1,10),np.nan))]:
            with self.assertRaises(ValueError):rt.episode_metrics(e,p)


if __name__=='__main__':unittest.main()
