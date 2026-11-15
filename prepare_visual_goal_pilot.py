"""Validate a visual-goal data contract and prepare a manifest; never train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from models.imitation.visual_goal_dataset import VisualGoalSequenceDataset
from models.imitation.visual_goal_training import warm_start_visual_policy
from models.imitation.visual_targets import sha256


def prepare(
    annotations,
    targets,
    source,
    *,
    training_role="train",
    evaluation_role="validation",
    training_approvals=None,
):
    manifest = json.loads(annotations.read_text())
    descriptors = json.loads(targets.read_text())
    approval_doc = (
        None
        if training_approvals is None
        else json.loads(training_approvals.read_text())
    )
    train = VisualGoalSequenceDataset(
        manifest, descriptors, role=training_role, training_approvals=approval_doc
    )
    evaluation = VisualGoalSequenceDataset(manifest, descriptors, role=evaluation_role)
    if training_role != "train":
        raise ValueError("Training preparation cannot use development-only data")
    train_groups = {ep["start_group_id"] for ep, _, _ in train.episodes}
    eval_groups = {ep["start_group_id"] for ep, _, _ in evaluation.episodes}
    if train_groups & eval_groups:
        raise ValueError("Start-group leakage")
    for data in (train, evaluation):
        groups = {}
        for ep, _, _ in data.episodes:
            groups.setdefault(ep["start_group_id"], set()).add(ep["requested_goal_id"])
        if any(len(goals) < 2 for goals in groups.values()):
            raise ValueError("Every start needs paired requested goals")
    model, config = warm_start_visual_policy(source, device="cpu")
    return {
        "schema": "shionn_visual_goal_pilot_preparation_v1",
        "status": "prepared_not_trained",
        "annotations_sha256": sha256(annotations),
        "target_manifest_sha256": sha256(targets),
        "training_approval_sha256": None
        if training_approvals is None
        else sha256(training_approvals),
        "source_checkpoint_sha256": sha256(source),
        "model_config": config,
        "train_episodes": [ep["episode_id"] for ep, _, _ in train.episodes],
        "evaluation_episodes": [ep["episode_id"] for ep, _, _ in evaluation.episodes],
        "seed": 20261115,
        "update_cap": 200,
        "batch_size": 2,
        "sequence_length": 16,
        "burn_in": 16,
        "optimizer_proposal": {"type": "Adam", "lr": 0.001},
        "arms": ["selected_visual_crop", "constant_black_crop_matched_adapter"],
        "trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "resource_note": "GPU runtime/VRAM unmeasured; no training or model promotion authorized",
        "acceptance_reference": "docs/GOAL_CHOICE_PAIR_REVIEW_2026_11_15.md",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--training-approvals", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = prepare(
        args.annotations,
        args.targets,
        args.source,
        training_approvals=args.training_approvals,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
