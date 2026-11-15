"""Episode-local crop-conditioned sequences with explicit source/split checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .goals import validate_goal_annotation_manifest
from .visual_targets import load_reviewed_episode, sha256


def preprocess_goal_rgb(rgb):
    if (
        not isinstance(rgb, np.ndarray)
        or rgb.dtype != np.uint8
        or rgb.ndim != 3
        or rgb.shape[2] != 3
        or min(rgb.shape[:2]) < 1
    ):
        raise ValueError("Expected nonempty uint8 RGB crop")
    return (
        cv2.resize(rgb, (96, 96), interpolation=cv2.INTER_AREA)
        .transpose(2, 0, 1)
        .copy()
    )


def verify_descriptor(episode, descriptor):
    if descriptor.get("schema") != "shionn_persistent_visual_target_v1":
        raise ValueError("Unsupported visual-target descriptor")
    if (
        descriptor.get("episode_id") != episode["episode_id"]
        or descriptor.get("goal_id") != episode["requested_goal_id"]
    ):
        raise ValueError("Descriptor episode/goal mismatch")
    if (
        descriptor.get("source_video_sha256") != episode["provenance"]["video_sha256"]
        or descriptor.get("source_frame_idx") != 0
    ):
        raise ValueError("Descriptor source mismatch")
    path = Path(descriptor["crop_path"])
    if sha256(path) != descriptor.get("crop_sha256"):
        raise ValueError("Target crop hash mismatch")
    crop_bgr = cv2.imread(str(path))
    if crop_bgr is None:
        raise ValueError("Target crop cannot be decoded")
    capture = cv2.VideoCapture(str(Path(episode["source_path"]) / "video.mp4"))
    try:
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError("Missing source goal frame")
    frame_hash = hashlib.sha256(
        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()
    ).hexdigest()
    if frame_hash != descriptor.get("source_frame_rgb_sha256") or list(
        frame.shape
    ) != descriptor.get("source_frame_shape"):
        raise ValueError("Target frame fingerprint mismatch")
    bounds = descriptor.get("native_bounds_xyxy")
    if (
        not isinstance(bounds, list)
        or len(bounds) != 4
        or any(type(v) is not int for v in bounds)
    ):
        raise ValueError("Invalid native crop bounds")
    left, top, right, bottom = bounds
    if (
        not 0 <= left < right <= frame.shape[1]
        or not 0 <= top < bottom <= frame.shape[0]
    ):
        raise ValueError("Crop outside source frame")
    if not np.array_equal(crop_bgr, frame[top:bottom, left:right]):
        raise ValueError("Crop pixels do not match selected source region")
    return preprocess_goal_rgb(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


class VisualGoalSequenceDataset:
    """Full histories retained, wait loss masked, per-chunk hidden state reset.

    No stochastic augmentation: flips must transform both scene, goal and actions.
    Returns dicts with uint8 frames/crop, float32 targets, bool valid/loss_mask.
    Train mode requires explicit approved TRAIN records; development is read-only.
    """

    def __init__(
        self,
        manifest,
        descriptors,
        *,
        role="annotation_only",
        sequence_length=16,
        burn_in=16,
        max_total_frames=4000,
        training_approvals=None,
    ):
        if (
            type(sequence_length) is not int
            or sequence_length < 1
            or type(burn_in) is not int
            or burn_in < 0
        ):
            raise ValueError("Invalid sequence length/burn-in")
        validation = validate_goal_annotation_manifest(manifest)
        if not validation["valid"]:
            raise ValueError(validation["errors"])
        if role not in {
            "annotation_only",
            "train",
            "development",
            "validation",
            "regression",
        }:
            raise ValueError("Unsupported visual dataset role")
        # Group labels are claims; identical source video cannot become a fresh
        # holdout just by renaming episode/session/start IDs.
        source_roles = {}
        for ep in manifest["episodes"]:
            source_roles.setdefault(ep["provenance"]["video_sha256"], set()).add(
                ep["split_role"]
            )
        if any(
            "train" in roles and roles != {"train"} for roles in source_roles.values()
        ):
            raise ValueError("Source video leakage across training and held-out roles")
        selected = [e for e in manifest["episodes"] if e["split_role"] == role]
        if not selected or sum(e["frame_count"] for e in selected) > max_total_frames:
            raise ValueError("Empty selection or total frame budget exceeded")
        if len({d["episode_id"] for d in descriptors}) != len(descriptors):
            raise ValueError("Duplicate target descriptors")
        by_episode = {d["episode_id"]: d for d in descriptors}
        approvals = {}
        if training_approvals is not None:
            if (
                training_approvals.get("schema")
                != "shionn_visual_goal_training_approvals_v1"
            ):
                raise ValueError("Unsupported explicit training approval schema")
            entries = training_approvals.get("episodes", [])
            if len({a["episode_id"] for a in entries}) != len(entries):
                raise ValueError("Duplicate training approvals")
            approvals = {a["episode_id"]: a for a in entries}
        self.role = role
        self.sequence_length, self.burn_in = sequence_length, burn_in
        self.episodes = []
        self.chunks = []
        for episode in selected:
            if episode["episode_id"] not in by_episode:
                raise ValueError("Missing visual target descriptor")
            if any(
                row["goal_id"] != episode["requested_goal_id"]
                for row in episode["annotations"]
            ):
                raise ValueError(
                    "Persistent visual pilot does not support within-episode goal transitions"
                )
            descriptor = by_episode[episode["episode_id"]]
            if role == "train" and (
                descriptor.get("training_approved") is not True
                or descriptor.get("membership") != "train"
            ):
                raise ValueError("Visual target is not training-approved")
            metadata = json.loads(
                (Path(episode["source_path"]) / "metadata.json").read_text()
            )
            if metadata.get("map") != episode.get("map_name"):
                raise ValueError("Source metadata map differs from annotation map")
            if metadata.get("action_owner") != "human" or any(
                row["action_owner"] != "human" for row in episode["annotations"]
            ):
                raise ValueError(
                    "Visual imitation requires verified human action ownership"
                )
            collection = metadata.get("collection")
            if collection:
                if episode["start_group_id"] != collection.get("start_id") or episode[
                    "source_session_id"
                ] != collection.get("session_id"):
                    raise ValueError(
                        "Collection start/session differs from annotations"
                    )
                goal_ids = manifest.get("waypoint_goal_ids", {})
                requested = metadata.get("requested_waypoint")
                if (
                    requested not in goal_ids
                    or episode["requested_goal_id"] != goal_ids[requested]
                ):
                    raise ValueError(
                        "Collection requested waypoint differs from visual goal"
                    )
                if (
                    role == "train"
                    and collection.get("intended_role") != "train_candidate"
                ):
                    raise ValueError("Reserved collection cohort cannot enter training")
                if (
                    role in {"validation", "regression"}
                    and collection.get("intended_role") != "heldout_reserved"
                ):
                    raise ValueError(
                        "Training-candidate collection cannot become held-out evaluation"
                    )
            if role == "train" and metadata.get("map") in {
                "dataset_test11",
                "dataset_test12",
            }:
                raise ValueError(
                    "Frozen validation/regression map cannot enter training"
                )
            data = load_reviewed_episode(
                episode,
                for_training=role == "train",
                training_approval=approvals.get(episode["episode_id"]),
            )
            crop = verify_descriptor(episode, descriptor)
            active = np.flatnonzero(data["loss_mask"])
            if not len(active):
                raise ValueError("No supervised frames after waiting mask")
            index = len(self.episodes)
            self.episodes.append((episode, data, crop))
            for start in range(int(active[0]), len(data["frames"]), sequence_length):
                if data["loss_mask"][start : start + sequence_length].any():
                    self.chunks.append((index, start))

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        ep_index, start = self.chunks[index]
        episode, data, crop = self.episodes[ep_index]
        total = self.burn_in + self.sequence_length
        frames = np.zeros((total, 12, 180, 320), np.uint8)
        targets = np.zeros((total, 10), np.float32)
        valid = np.zeros(total, bool)
        loss = np.zeros(total, bool)
        for i in range(total):
            frame = start - self.burn_in + i
            if 0 <= frame < len(data["frames"]):
                ids = [max(0, frame - 3 + j) for j in range(4)]
                frames[i] = (
                    data["frames"][ids].transpose(0, 3, 1, 2).reshape(12, 180, 320)
                )
                targets[i] = data["actions"][frame]
                valid[i] = True
                loss[i] = i >= self.burn_in and data["loss_mask"][frame]
        return {
            "frames": frames,
            "targets": targets,
            "valid": valid,
            "loss_mask": loss,
            "goal_crop": crop.copy(),
            "episode_id": episode["episode_id"],
            "goal_id": episode["requested_goal_id"],
        }
