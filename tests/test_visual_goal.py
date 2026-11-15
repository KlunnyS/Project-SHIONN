import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from models.imitation.checkpoint import save_checkpoint
from models.imitation.inference import PolicyInference
from models.imitation.mouse_bins import MouseBins
from models.imitation.preprocess import PREPROCESSING_CONFIG
from models.imitation.visual_goal import (
    VISUAL_GOAL_ARCHITECTURE_VERSION,
    VISUAL_GOAL_CONFIG,
    VisualGoalRecurrentPolicy,
)
from models.imitation.visual_goal_dataset import VisualGoalSequenceDataset
from models.imitation.visual_goal_evaluation import compare_visual_goals
from models.imitation.visual_goal_training import visual_goal_batch_loss
from models.imitation.visual_targets import extract_target
from tests.test_visual_targets import FakeCapture, VisualTargetTests

BINS = {
    k: {"magnitude_edges": [], "representatives": [-3.0, 0.0, 3.0]}
    for k in ("dx", "dy")
}


def tiny(gate=False):
    model = VisualGoalRecurrentPolicy(3, 3, 7, gate, 8, 9)
    model.base.trunk = nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(12, 512)
    )
    return model


class VisualGoalModelTests(unittest.TestCase):
    def test_zero_adapter_baseline_equivalence_and_recurrent_consistency(self):
        for gate in [False, True]:
            model = tiny(gate).eval()
            x = torch.rand(2, 3, 12, 8, 8)
            crop = torch.rand(2, 3, 96, 96)
            with torch.no_grad():
                base, h = model.base(x)
                out, state = model(x, crop)
                for k in base:
                    torch.testing.assert_close(base[k], out[k], rtol=0, atol=0)
                torch.testing.assert_close(h, state, rtol=0, atol=0)
                goal = model.encode_goal(crop)
                hidden = None
                steps = []
                for i in range(3):
                    step, hidden = model.step(x[:, i], goal, hidden)
                    steps.append(step)
                for k in out:
                    torch.testing.assert_close(
                        out[k], torch.stack([s[k] for s in steps], 1)
                    )
                torch.testing.assert_close(state, hidden)
                valid = torch.tensor([[True, False, False], [True, True, True]])
                _, masked = model(x, crop, valid=valid)
                _, first = model(x[:1, :1], crop[:1])
                torch.testing.assert_close(masked[:, :1], first)

    def test_crop_contract_and_goal_change_affects_logits_after_head_perturbation(self):
        model = tiny()
        x = torch.rand(1, 2, 12, 8, 8)
        with self.assertRaises(ValueError):
            model(x, torch.zeros(1, 3, 8, 8))
        with self.assertRaises(ValueError):
            model(x, torch.full((1, 3, 96, 96), float("nan")))
        with self.assertRaises(ValueError):
            model(x)
        with torch.no_grad():
            model.goal_heads["mouse_dx_logits"].weight.normal_()
        a = torch.zeros(1, 3, 96, 96)
        b = torch.ones_like(a)
        initial = torch.zeros(1, 1, 7)
        before = initial.clone()
        report = compare_visual_goals(
            model, x, a, b, hidden=initial, dx_representatives=[-3, 0, 3]
        )
        self.assertGreater(
            report["metrics"]["mouse_dx_logits"]["mean_total_variation"], 0
        )
        torch.testing.assert_close(initial, before)
        self.assertTrue(model.training)

    def test_loss_backward_burn_mask_and_frozen_base(self):
        torch.manual_seed(4)
        model = tiny()
        model.base.requires_grad_(False)
        batch = {
            "frames": torch.randint(0, 255, (1, 4, 12, 8, 8), dtype=torch.uint8),
            "targets": torch.zeros(1, 4, 10),
            "valid": torch.ones(1, 4, dtype=torch.bool),
            "loss_mask": torch.tensor([[False, False, True, True]]),
            "goal_crop": torch.randint(0, 255, (1, 3, 96, 96), dtype=torch.uint8),
        }
        # Nonzero test heads allow gradient testing beyond the zero-residual start.
        with torch.no_grad():
            model.goal_heads["mouse_dx_logits"].weight.normal_()
        loss, parts, n = visual_goal_batch_loss(
            model,
            batch,
            burn_in=2,
            binary_class_weights=torch.ones(8, 2),
            mouse_bins=MouseBins(BINS),
        )
        loss.backward()
        self.assertEqual(n, 2)
        self.assertTrue(all(p.grad is None for p in model.base.parameters()))
        self.assertGreater(
            sum(
                float(p.grad.abs().sum())
                for p in model.goal_projection.parameters()
                if p.grad is not None
            ),
            0,
        )
        batch["loss_mask"][:] = False
        with self.assertRaises(ValueError):
            visual_goal_batch_loss(
                model,
                batch,
                burn_in=2,
                binary_class_weights=torch.ones(8, 2),
                mouse_bins=MouseBins(BINS),
            )

    def test_constant_control_is_independent_of_selected_crop(self):
        model = tiny().eval()
        with torch.no_grad():
            model.goal_heads["mouse_dx_logits"].weight.normal_()
        batch = {
            "frames": torch.ones(1, 1, 12, 8, 8, dtype=torch.uint8),
            "targets": torch.zeros(1, 1, 10),
            "valid": torch.ones(1, 1, dtype=torch.bool),
            "loss_mask": torch.ones(1, 1, dtype=torch.bool),
            "goal_crop": torch.zeros(1, 3, 96, 96, dtype=torch.uint8),
        }
        options = {
            "burn_in": 0,
            "binary_class_weights": torch.ones(8, 2),
            "mouse_bins": MouseBins(BINS),
            "goal_mode": "constant",
        }
        first = visual_goal_batch_loss(model, batch, **options)[0]
        batch["goal_crop"].fill_(255)
        second = visual_goal_batch_loss(model, batch, **options)[0]
        torch.testing.assert_close(first, second, rtol=0, atol=0)

    def test_detached_burn_state_blocks_previous_frames(self):
        model = tiny()
        x = torch.rand(1, 4, 12, 8, 8, requires_grad=True)
        crop = torch.rand(1, 3, 96, 96)
        _, h = model(x[:, :2], crop)
        out, _ = model(x[:, 2:], crop, hidden=h.detach())
        sum(v.square().mean() for v in out.values()).backward()
        self.assertEqual(float(x.grad[:, :2].abs().sum()), 0)
        self.assertGreater(float(x.grad[:, 2:].abs().sum()), 0)

    def test_checkpoint_inference_persistence_reset_and_rgb_contract(self):
        prior_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, prior_threads)
        with tempfile.TemporaryDirectory() as tmp:
            model = VisualGoalRecurrentPolicy(3, 3, 7, False, 8, 9).eval()
            path = Path(tmp) / "goal.pt"
            config = {
                "architecture": VISUAL_GOAL_ARCHITECTURE_VERSION,
                "preprocessing": PREPROCESSING_CONFIG,
                "target_processing": {"mouse_bins": BINS},
                "memory": {"type": "gru", "hidden_size": 7, "feature_gate": False},
                "visual_goal": {
                    "preprocessing": VISUAL_GOAL_CONFIG,
                    "embedding_size": 8,
                    "adapter_size": 9,
                },
            }
            save_checkpoint(
                path,
                model=model,
                optimizer=torch.optim.Adam(model.parameters()),
                epoch=0,
                global_step=0,
                best_val_loss=1.0,
                config=config,
            )
            policy = PolicyInference(path, "cpu")
            frame = np.zeros((180, 320, 3), np.uint8)
            with self.assertRaises(RuntimeError):
                policy.predict(frame)
            policy.set_visual_goal(np.zeros((10, 20, 3), np.uint8), goal_id="A")
            embed = policy.visual_goal_embedding.clone()
            action, info = policy.predict_with_diagnostics(frame)
            self.assertEqual(info["visual_goal"]["goal_id"], "A")
            policy.predict(frame)
            torch.testing.assert_close(embed, policy.visual_goal_embedding)
            policy.reset_memory()
            self.assertEqual(policy.visual_goal_id, "A")
            self.assertIsNone(policy.hidden)
            policy.reset()
            self.assertIsNone(policy.visual_goal_embedding)
            other = PolicyInference(path, "cpu")
            self.assertIsNone(other.visual_goal_embedding)
            config["visual_goal"]["preprocessing"] = {}
            with patch(
                "models.imitation.inference.load_checkpoint",
                return_value={"config": config},
            ):
                with self.assertRaisesRegex(ValueError, "preprocessing"):
                    PolicyInference(path, "cpu")


class VisualGoalDataTests(unittest.TestCase):
    setUp = VisualTargetTests.setUp

    def make_dataset(self):
        source_meta = json.loads((self.path / "metadata.json").read_text())
        source_meta.update(map="fixture", action_owner="human")
        (self.path / "metadata.json").write_text(json.dumps(source_meta))
        from models.imitation.visual_targets import sha256

        self.ep["provenance"]["metadata_sha256"] = sha256(self.path / "metadata.json")
        ep = copy.deepcopy(self.ep)
        ep.update(
            {
                "map_name": "fixture",
                "source_session_id": "session",
                "start_group_id": "start",
                "annotations": [
                    {
                        "frame_idx": i,
                        "goal_id": "fixture_A",
                        "target_kind": "waypoint",
                        "visible": True,
                        "x_norm": 0.0,
                        "y_norm": 0.0,
                        "action_owner": "human",
                    }
                    for i in range(3)
                ],
            }
        )
        self.manifest = {
            "schema": "shionn_goal_annotations_v1",
            "coordinate_system": "normalized_frame_xy_origin_top_left_y_down",
            "episodes": [ep],
        }

        def capture(_):
            return FakeCapture(self.frames)

        with patch("cv2.VideoCapture", side_effect=capture):
            self.descriptor = extract_target(
                ep, [10, 20, 30, 40], self.path / "crop.png"
            )
            return VisualGoalSequenceDataset(
                self.manifest, [self.descriptor], sequence_length=2, burn_in=2
            )

    def test_sequence_wait_context_and_tail_and_persistent_crop(self):
        data = self.make_dataset()
        self.assertEqual(len(data), 1)
        batch = data[0]
        self.assertEqual(batch["valid"].tolist(), [False, True, True, True])
        self.assertEqual(batch["loss_mask"].tolist(), [False, False, True, True])
        self.assertEqual(batch["frames"][1:, 9, 0, 0].tolist(), [0, 1, 2])
        self.assertEqual(batch["goal_crop"].shape, (3, 96, 96))
        batch["goal_crop"][:] = 99
        self.assertFalse(np.all(data[0]["goal_crop"] == 99))
        with patch("cv2.VideoCapture", side_effect=lambda _: FakeCapture(self.frames)):
            tail = VisualGoalSequenceDataset(
                self.manifest, [self.descriptor], sequence_length=4, burn_in=0
            )[0]
        self.assertEqual(tail["valid"].tolist(), [True, True, False, False])

    def test_epoch_validation_and_optimizer_membership_guard(self):
        from models.imitation.visual_goal_training import run_visual_goal_epoch

        data = self.make_dataset()
        loader = DataLoader(data, batch_size=1)
        model = tiny()
        options = {
            "binary_class_weights": torch.ones(8, 2),
            "mouse_bins": MouseBins(BINS),
            "max_batches": 1,
        }
        metrics = run_visual_goal_epoch(model, loader, **options)
        self.assertEqual(metrics["optimizer_steps"], 0)
        self.assertEqual(metrics["frames"], 2)
        self.assertTrue(model.training)
        with self.assertRaisesRegex(ValueError, "non-TRAIN"):
            run_visual_goal_epoch(
                model, loader, optimizer=torch.optim.Adam(model.parameters()), **options
            )

    def test_descriptor_and_split_provenance_fail_closed(self):
        self.make_dataset()
        bad = copy.deepcopy(self.descriptor)
        bad["goal_id"] = "wrong"
        with patch("cv2.VideoCapture", side_effect=lambda _: FakeCapture(self.frames)):
            with self.assertRaisesRegex(ValueError, "goal mismatch"):
                VisualGoalSequenceDataset(self.manifest, [bad])
            with self.assertRaises(ValueError):
                VisualGoalSequenceDataset(
                    self.manifest, [self.descriptor], role="train"
                )
        ep = copy.deepcopy(self.manifest["episodes"][0])
        ep["episode_id"] = "other"
        ep["split_role"] = "train"
        ep["training_approved"] = True
        self.manifest["episodes"].append(ep)
        with self.assertRaisesRegex(ValueError, "leakage"):
            VisualGoalSequenceDataset(self.manifest, [self.descriptor])

    def test_valid_goal_transition_rejected_for_single_crop_pilot(self):
        self.make_dataset()
        row = self.manifest["episodes"][0]["annotations"][-1]
        row.update(
            goal_id="goal_B", goal_transition=True, transition_reason="explicit change"
        )
        with self.assertRaisesRegex(ValueError, "transitions"):
            VisualGoalSequenceDataset(self.manifest, [self.descriptor])

    def test_renaming_duplicate_source_cannot_bypass_split_exclusion(self):
        self.make_dataset()
        other = copy.deepcopy(self.manifest["episodes"][0])
        other.update(
            episode_id="renamed",
            source_session_id="new_session",
            start_group_id="new_start",
            split_role="train",
            training_approved=True,
        )
        self.manifest["episodes"].append(other)
        with self.assertRaisesRegex(ValueError, "Source video leakage"):
            VisualGoalSequenceDataset(self.manifest, [self.descriptor])

    def test_collection_reservation_and_goal_metadata_cannot_be_relabelled(self):
        self.make_dataset()
        metadata = json.loads((self.path / "metadata.json").read_text())
        metadata.update(
            requested_waypoint="A",
            collection={
                "start_id": "start",
                "session_id": "session",
                "intended_role": "heldout_reserved",
            },
        )
        (self.path / "metadata.json").write_text(json.dumps(metadata))
        self.manifest["waypoint_goal_ids"] = {"A": "fixture_A", "B": "fixture_B"}
        ep = self.manifest["episodes"][0]
        ep["split_role"] = "train"
        ep["training_approved"] = True
        self.descriptor["membership"] = "train"
        self.descriptor["training_approved"] = True
        with self.assertRaisesRegex(ValueError, "Reserved collection"):
            VisualGoalSequenceDataset(self.manifest, [self.descriptor], role="train")
        ep["split_role"] = "annotation_only"
        self.manifest["waypoint_goal_ids"]["A"] = "wrong_goal"
        with self.assertRaisesRegex(ValueError, "requested waypoint"):
            VisualGoalSequenceDataset(self.manifest, [self.descriptor])

    def test_source_map_and_owner_must_match_actual_metadata(self):
        self.make_dataset()
        self.manifest["episodes"][0]["map_name"] = "different_map"
        with self.assertRaisesRegex(ValueError, "metadata map"):
            VisualGoalSequenceDataset(self.manifest, [self.descriptor])
        self.manifest["episodes"][0]["map_name"] = "fixture"
        metadata = json.loads((self.path / "metadata.json").read_text())
        metadata["action_owner"] = "policy"
        (self.path / "metadata.json").write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, "ownership"):
            VisualGoalSequenceDataset(self.manifest, [self.descriptor])


class VisualGoalPreparationTests(unittest.TestCase):
    def test_preparation_rejects_development_before_loading_source(self):
        from prepare_visual_goal_pilot import prepare

        manifest = {
            "schema": "shionn_goal_annotations_v1",
            "coordinate_system": "normalized_frame_xy_origin_top_left_y_down",
            "episodes": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / "a.json"
            targets = root / "t.json"
            annotations.write_text(json.dumps(manifest))
            targets.write_text("[]")
            with patch("prepare_visual_goal_pilot.warm_start_visual_policy") as load:
                with self.assertRaises(ValueError):
                    prepare(annotations, targets, root / "missing.pt")
                load.assert_not_called()

    def test_warm_start_preserves_source_and_records_identity(self):
        from models.imitation.recurrent import RECURRENT_ARCHITECTURE_VERSION
        from models.imitation.visual_goal_training import warm_start_visual_policy

        original = tiny().base
        config = {
            "architecture": RECURRENT_ARCHITECTURE_VERSION,
            "preprocessing": PREPROCESSING_CONFIG,
            "target_processing": {"mouse_bins": BINS},
            "memory": {"type": "gru", "hidden_size": 7, "feature_gate": False},
        }

        # Replace the constructed base trunk only, keeping strict state loading.
        def factory(*args):
            return tiny()

        with (
            patch(
                "models.imitation.visual_goal_training.load_checkpoint",
                return_value={
                    "config": config,
                    "model_state_dict": original.state_dict(),
                },
            ),
            patch(
                "models.imitation.visual_goal_training.VisualGoalRecurrentPolicy",
                side_effect=factory,
            ),
            patch(
                "models.imitation.visual_goal_training.sha256", return_value="a" * 64
            ),
        ):
            model, new = warm_start_visual_policy(Path("source.pt"))
        self.assertEqual(new["architecture"], VISUAL_GOAL_ARCHITECTURE_VERSION)
        self.assertEqual(config["architecture"], RECURRENT_ARCHITECTURE_VERSION)
        self.assertTrue(all(not p.requires_grad for p in model.base.parameters()))
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, model.base.state_dict()[name])
        self.assertEqual(new["visual_goal"]["source_checkpoint_sha256"], "a" * 64)


class ExplicitApprovalTests(unittest.TestCase):
    setUp = VisualTargetTests.setUp
    make_dataset = VisualGoalDataTests.make_dataset

    def test_versioned_review_allows_training_without_editing_capture(self):
        self.make_dataset()
        ep = self.manifest["episodes"][0]
        ep["split_role"] = "train"
        ep["training_approved"] = True
        self.descriptor["membership"] = "train"
        self.descriptor["training_approved"] = True
        metadata_before = (self.path / "metadata.json").read_bytes()
        approval = {
            "schema": "shionn_visual_goal_training_approvals_v1",
            "episodes": [
                {
                    "episode_id": ep["episode_id"],
                    "training_approved": True,
                    "video_sha256": ep["provenance"]["video_sha256"],
                    "actions_sha256": ep["provenance"]["actions_sha256"],
                    "metadata_sha256": ep["provenance"]["metadata_sha256"],
                    "authorization_ref": "synthetic fixture only; not real dataset approval",
                }
            ],
        }
        with patch("cv2.VideoCapture", side_effect=lambda _: FakeCapture(self.frames)):
            with self.assertRaisesRegex(ValueError, "approval"):
                VisualGoalSequenceDataset(
                    self.manifest, [self.descriptor], role="train"
                )
            data = VisualGoalSequenceDataset(
                self.manifest,
                [self.descriptor],
                role="train",
                training_approvals=approval,
            )
            self.assertEqual(data.role, "train")
            approval["episodes"][0]["actions_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "approval"):
                VisualGoalSequenceDataset(
                    self.manifest,
                    [self.descriptor],
                    role="train",
                    training_approvals=approval,
                )
            approval["episodes"][0]["actions_sha256"] = ep["provenance"][
                "actions_sha256"
            ]
            approval["episodes"][0]["video_sha256"] = "f" * 64
            with self.assertRaisesRegex(ValueError, "approval"):
                VisualGoalSequenceDataset(
                    self.manifest,
                    [self.descriptor],
                    role="train",
                    training_approvals=approval,
                )
        self.assertEqual(metadata_before, (self.path / "metadata.json").read_bytes())
