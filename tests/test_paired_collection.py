import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from paired_goal_collection import commands, group_context, load_plan, register_pose
from registered_starts import console_query, reject_setup_events


class PairedCollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.plan = self.root / "plan.json"
        plan = json.loads(
            Path("configs/goal_choice_collection_plan_v1.json").read_text()
        )
        plan["excluded_annotation_ledgers"] = []
        self.plan.write_text(json.dumps(plan))
        self.registry = self.root / "starts1.json"
        register_pose(
            self.plan,
            None,
            self.registry,
            "train_01",
            "setpos 128 96 64.03125;setang 0 85 0",
        )

    def test_counts_are_six_individual_trajectories_in_three_pairs(self):
        plan = load_plan(self.plan)
        self.assertEqual(sum(len(g["goals"]) for g in plan["groups"]), 6)
        lines = commands(self.plan, self.registry, "train_01")
        self.assertEqual(len(lines), 2)
        self.assertIn("--requested-waypoint A", lines[0])
        self.assertIn("--collection-session goal_choice_train01_session_01", lines[1])

    def test_registration_is_append_only_and_rejects_exposed_pose(self):
        with self.assertRaises(FileExistsError):
            register_pose(
                self.plan,
                None,
                self.registry,
                "train_01",
                "setpos 128 96 64;setang 0 85 0",
            )
        with self.assertRaisesRegex(ValueError, "close"):
            register_pose(
                self.plan,
                None,
                self.root / "bad.json",
                "train_01",
                "setpos 320 26.44379 64.03125;setang 0 85 0",
            )
        with self.assertRaises(ValueError):
            register_pose(
                self.plan,
                None,
                self.root / "bad.json",
                "train_01",
                "setpos 1 2 3;setang 0 90 0;quit",
            )
        self.assertFalse((self.root / "bad.json").exists())

    def test_whole_group_session_and_first_pair_gate(self):
        with self.assertRaises(ValueError):
            group_context(
                self.plan, self.registry, "train_01", "different_session", "A"
            )
        with self.assertRaisesRegex(ValueError, "first pair"):
            commands(self.plan, self.registry, "train_02")
        with self.assertRaisesRegex(ValueError, "review"):
            commands(self.plan, self.registry, "train_02", stage="full")

    def test_future_positions_and_headings_cannot_be_renamed_duplicates(self):
        with self.assertRaises(ValueError):
            register_pose(
                self.plan,
                self.registry,
                self.root / "bad.json",
                "train_02",
                "setpos 129 96 64;setang 0 95 0",
            )
        with self.assertRaisesRegex(ValueError, "heading"):
            register_pose(
                self.plan,
                self.registry,
                self.root / "bad.json",
                "train_02",
                "setpos 512 96 64;setang 0 85 0",
            )
        second = self.root / "starts2.json"
        register_pose(
            self.plan,
            self.registry,
            second,
            "train_02",
            "setpos 512 96 64.03125;setang 0 95 0",
        )
        self.assertEqual(len(json.loads(second.read_text())["starts"]), 2)
        self.assertEqual(len(json.loads(self.registry.read_text())["starts"]), 1)

    def test_full_matrix_release_requires_intact_review_and_all_starts(self):
        from models.imitation.visual_targets import sha256

        second = self.root / "starts2.json"
        third = self.root / "starts3.json"
        register_pose(
            self.plan,
            self.registry,
            second,
            "train_02",
            "setpos 512 96 64.03125;setang 0 95 0",
        )
        register_pose(
            self.plan, second, third, "eval_01", "setpos 320 224 64.03125;setang 0 90 0"
        )
        evidence = self.root / "review_evidence"
        evidence.write_text("synthetic integrity fixture")
        report = {
            "status": "passed",
            "plan_sha256": sha256(self.plan),
            "group_id": "train_01",
            "start_position": [128, 96, 64.03125],
            "start_angles": [0, 85, 0],
            "map_sha256": "a" * 64,
            "capture_config": {"fps": 24},
            "artifacts": [{"path": str(evidence), "sha256": sha256(evidence)}],
        }
        review = self.root / "review.json"
        review.write_text(json.dumps(report))
        lines = commands(
            self.plan, third, "train_02", stage="full", first_pair_review=review
        )
        self.assertIn("--requested-waypoint B", lines[0])
        self.assertIn("--first-pair-review", lines[0])
        evidence.write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            commands(
                self.plan, third, "eval_01", stage="full", first_pair_review=review
            )

    def test_split_role_cannot_split_A_B(self):
        plan = load_plan(self.plan)
        plan["groups"][0]["goals"] = ["A"]
        self.plan.write_text(json.dumps(plan))
        with self.assertRaises(ValueError):
            load_plan(self.plan)

    def test_setup_waypoint_or_terminal_event_rejected(self):
        for event in ["waypoint_reached|A", "goal_reached|1", "episode_failed|timeout"]:
            controller = Mock(event_buffer=["EVT|chamber_ready|2.1", "EVT|" + event])
            with self.assertRaisesRegex(RuntimeError, "setup consumed"):
                reject_setup_events(controller)
        reject_setup_events(Mock(event_buffer=["EVT|chamber_ready|2.1"]))

    def test_event_during_verification_query_cannot_be_discarded(self):
        controller = Mock()
        controller._reject_setup_terminals = True
        controller.event_buffer = []

        def read(**kwargs):
            controller.event_buffer.append("EVT|waypoint_reached|B")
            return "EVT|waypoint_reached|B\n"

        controller.read_console.side_effect = read
        with self.assertRaisesRegex(RuntimeError, "waypoint_reached"):
            console_query(controller, "getpos")


class PairIntegrityTests(unittest.TestCase):
    def setUp(self):
        PairedCollectionTests.setUp(self)
        import csv

        from models.imitation.dataset import ACTION_COLUMNS
        from models.imitation.visual_targets import sha256

        self.capture_root = self.root / "captures"
        self.metadata_paths = []
        for goal in ["A", "B"]:
            p = self.capture_root / goal / ("episode_" + goal)
            p.mkdir(parents=True)
            self.metadata_paths.append(p / "metadata.json")
            result = p.parent / "result.json"
            result.write_text(
                json.dumps({"outcome": "waypoint_reached", "episode": str(p)})
            )
            (p / "video.mp4").write_bytes(("fake_video_" + goal).encode())
            context = group_context(
                self.plan,
                self.registry,
                "train_01",
                "goal_choice_train01_session_01",
                goal,
            )
            context.update(map_sha256="a" * 64)
            pose = {"position": [128, 96, 64.03125], "angles": [0, 85, 0]}
            verification = {
                "verified": True,
                "start_id": "goal_choice_train_01",
                "config_sha256": sha256(self.registry),
                "pre_recording_verification": {
                    "verified": True,
                    "samples": [pose, pose],
                },
            }
            meta = {
                "capture_valid": True,
                "map": "goal_choice_v1",
                "requested_waypoint": goal,
                "action_owner": "human",
                "training_approved": False,
                "collection": context,
                "registered_start": verification,
                "setup_events_checked": True,
                "waypoint_events": [
                    {
                        "name": "waypoint_reached",
                        "value": goal,
                        "received_monotonic": 1.11,
                    }
                ],
                "capture_config": load_plan(self.plan)["capture_config"],
            }
            (p / "metadata.json").write_text(json.dumps(meta))
            with (p / "actions.csv").open("w") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_idx",
                        "frame_id",
                        "frame_age_seconds",
                        "frame_sampled_monotonic",
                        *ACTION_COLUMNS,
                    ],
                )
                writer.writeheader()
                for i in range(3):
                    writer.writerow(
                        {
                            "frame_idx": i,
                            "frame_id": i + 1,
                            "frame_age_seconds": 0.01,
                            "frame_sampled_monotonic": 1 + i * 0.04,
                            **{k: 0 for k in ACTION_COLUMNS},
                        }
                    )

    def run_review(self, **kwargs):
        import numpy as np

        from review_paired_goal_capture import review_pair
        from tests.test_visual_targets import FakeCapture

        frames = [np.zeros((180, 320, 3), np.uint8) for _ in range(3)]
        with patch("cv2.VideoCapture", side_effect=lambda _: FakeCapture(frames)):
            return review_pair(self.plan, self.registry, self.capture_root, **kwargs)

    def test_integrity_does_not_approve_training_or_release_annotation_gate(self):
        report = self.run_review()
        self.assertEqual(
            report["status"], "integrity_passed_visual_annotation_review_pending"
        )
        self.assertFalse(report["training_approved"])
        path = self.root / "review.json"
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, "not passed"):
            commands(
                self.plan,
                self.registry,
                "train_02",
                stage="full",
                first_pair_review=path,
            )

    def test_wrong_event_or_pose_cannot_pass(self):
        path = self.metadata_paths[0]
        meta = json.loads(path.read_text())
        meta["waypoint_events"][0]["value"] = "B"
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "waypoint"):
            self.run_review()
        meta["waypoint_events"][0]["value"] = "A"
        meta["registered_start"]["pre_recording_verification"]["samples"][0][
            "position"
        ][0] = float("nan")
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "pose"):
            self.run_review()

    def test_capture_config_mismatch_rejected(self):
        path = self.metadata_paths[1]
        meta = json.loads(path.read_text())
        meta["capture_config"]["fps"] = 12
        path.write_text(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "Capture conditions"):
            self.run_review()

    def test_annotation_source_substitution_rejected(self):
        import types

        report = self.run_review()
        episodes = []
        for row in report["episodes"]:
            ep = {
                "episode_id": row["episode_id"],
                "provenance": row["provenance"],
                "source_path": row["source_path"],
                "start_group_id": "goal_choice_train_01",
                "source_session_id": "goal_choice_train01_session_01",
                "requested_goal_id": "goal_" + row["goal"],
            }
            episodes.append((ep, None, None))
        episodes[0][0]["source_path"] = str(self.root / "other_source")
        annotations = self.root / "labels.json"
        annotations.write_text(
            json.dumps({"waypoint_goal_ids": {"A": "goal_A", "B": "goal_B"}})
        )
        targets = self.root / "targets.json"
        targets.write_text("[]")
        note = self.root / "visual.md"
        note.write_text("synthetic fixture review, not real evidence")
        with patch(
            "review_paired_goal_capture.VisualGoalSequenceDataset",
            return_value=types.SimpleNamespace(episodes=episodes),
        ):
            with self.assertRaisesRegex(ValueError, "source differs"):
                self.run_review(
                    annotations=annotations, targets=targets, visual_review=note
                )


class SetupChunkTests(unittest.TestCase):
    def test_raw_chunk_observer_retains_split_waypoint_after_ready(self):
        from registered_starts import SetupEventGuard

        guard = SetupEventGuard()
        guard.feed("EVT|episode_failed|timeout\n")  # old epoch
        guard.check()
        guard.feed("EVT|chamber_ready|2.1\nEVT|waypo")
        guard.feed("int_reached|A\n")
        with self.assertRaisesRegex(RuntimeError, "waypoint_reached"):
            guard.check()

    def test_wrapper_notifies_optional_observer_without_changing_output(self):
        from wrapper import Portal2Controller

        controller = Portal2Controller(8020, log_file=None)
        controller.sock = Mock()
        controller.sock.recv.side_effect = [b"EVT|waypo", BlockingIOError()]
        observer = Mock()
        controller.console_observer = observer
        text = controller.read_console()
        observer.assert_called_once_with("EVT|waypo")
        self.assertEqual(text, "EVT|waypo")
