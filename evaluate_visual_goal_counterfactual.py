"""Bounded CPU-only, read-only visual-goal sensitivity screen (no gameplay)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from models.imitation.inference import PolicyInference
from models.imitation.visual_goal import VisualGoalRecurrentPolicy
from models.imitation.visual_goal_dataset import VisualGoalSequenceDataset
from models.imitation.visual_goal_evaluation import compare_visual_goals
from models.imitation.visual_targets import sha256


def evaluate(
    checkpoint, annotations, targets, *, role="annotation_only", max_chunks=16
):
    if type(max_chunks) is not int or not 1 <= max_chunks <= 64:
        raise ValueError("Counterfactual screen requires 1..64 chunks")
    manifest = json.loads(annotations.read_text())
    descriptors = json.loads(targets.read_text())
    data = VisualGoalSequenceDataset(
        manifest, descriptors, role=role, sequence_length=16, burn_in=16
    )
    inference = PolicyInference(checkpoint, device="cpu")
    if not isinstance(inference.policy, VisualGoalRecurrentPolicy):
        raise ValueError("Counterfactual crop screen needs a visual-goal checkpoint")
    goals = {}
    goal_sources = {}
    for ep, _, crop in data.episodes:
        if ep["requested_goal_id"] not in goals:
            goals[ep["requested_goal_id"]] = crop
            goal_sources[ep["requested_goal_id"]] = ep["episode_id"]
    if len(goals) != 2:
        raise ValueError("Select exactly two distinct visual goals")
    names = sorted(goals)
    crops = [
        torch.from_numpy(goals[name]).unsqueeze(0).float() / 255.0 for name in names
    ]
    rows = []
    for index, batch in enumerate(
        DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
    ):
        if index >= max_chunks:
            break
        report = compare_visual_goals(
            inference.policy,
            batch["frames"].float() / 255.0,
            *crops,
            valid=batch["valid"],
            decision_mask=batch["loss_mask"],
            dx_representatives=inference.mouse_bins.representatives(0),
        )
        rows.append(
            {
                "episode_id": batch["episode_id"][0],
                "recorded_goal": batch["goal_id"][0],
                "supervised_frames": int(batch["loss_mask"].sum()),
                **report,
            }
        )
    return {
        "schema": "shionn_visual_goal_counterfactual_v1",
        "device": "cpu",
        "checkpoint_sha256": sha256(checkpoint),
        "annotations_sha256": sha256(annotations),
        "target_manifest_sha256": sha256(targets),
        "goals": names,
        "goal_crop_source_episodes": goal_sources,
        "chunks": rows,
        "all_chunks_evaluated": len(data) <= max_chunks,
        "dataset_role": role,
        "limits": "observed branches, not expert counterfactual action labels or gameplay success",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--annotations", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument(
        "--role",
        choices=["annotation_only", "development", "validation", "regression"],
        default="annotation_only",
    )
    p.add_argument("--max-chunks", type=int, default=16)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = evaluate(
        args.checkpoint,
        args.annotations,
        args.targets,
        role=args.role,
        max_chunks=args.max_chunks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "chunks": len(report["chunks"]),
                "all_chunks_evaluated": report["all_chunks_evaluated"],
                "report": str(args.output),
                "interpretation": "sensitivity only",
            }
        )
    )


if __name__ == "__main__":
    main()
