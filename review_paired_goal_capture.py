"""Offline first-pair integrity/annotation gate; never approves training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import numpy as np

from models.imitation.dataset import ACTION_COLUMNS
from models.imitation.visual_goal_dataset import VisualGoalSequenceDataset
from models.imitation.visual_targets import sha256
from paired_goal_collection import group_context, load_plan
from registered_starts import load_start


def review_pair(
    plan_path, registry, root, *, annotations=None, targets=None, visual_review=None
):
    plan = load_plan(plan_path)
    group = plan["groups"][0]
    start = load_start(registry, group["start_id"], plan["map"])
    excluded = set()
    for path in plan["excluded_annotation_ledgers"]:
        excluded.update(
            ep["provenance"]["video_sha256"]
            for ep in json.loads(Path(path).read_text())["episodes"]
        )
    rows = []
    artifacts = [
        {"path": str(Path(path).resolve()), "sha256": sha256(Path(path))}
        for path in (plan_path, registry)
    ]
    capture_config = None
    map_hash = None
    seen = set()
    for goal in ["A", "B"]:
        result_path = Path(root) / goal / "result.json"
        result = json.loads(result_path.read_text())
        if result.get("outcome") != "waypoint_reached" or not result.get("episode"):
            raise ValueError("Missing correct waypoint outcome")
        episode = Path(result["episode"])
        metadata = json.loads((episode / "metadata.json").read_text())
        collection = metadata.get("collection", {})
        context = group_context(
            plan_path, registry, group["id"], group["session"], goal
        )
        for key in (
            "plan_sha256",
            "group_id",
            "intended_role",
            "session_id",
            "start_id",
            "registry_sha256",
            "requested_waypoint",
        ):
            if collection.get(key) != context[key]:
                raise ValueError("Collection provenance mismatch: " + key)
        if (
            metadata.get("capture_valid") is not True
            or metadata.get("map") != plan["map"]
            or metadata.get("requested_waypoint") != goal
            or metadata.get("action_owner") != "human"
        ):
            raise ValueError("Invalid map/capture/goal/ownership")
        if metadata.get("training_approved") is not False:
            raise ValueError("First pair must remain pending review")
        events = metadata.get("waypoint_events", [])
        if (
            len(events) != 1
            or events[0].get("name") != "waypoint_reached"
            or events[0].get("value") != goal
        ):
            raise ValueError("Wrong or ambiguous first waypoint event")
        config = metadata.get("capture_config")
        if not config or (capture_config is not None and config != capture_config):
            raise ValueError("Capture conditions differ")
        if config != plan.get("capture_config"):
            raise ValueError("Capture conditions differ from frozen plan")
        capture_config = config
        current_map_hash = collection.get("map_sha256")
        if (
            not isinstance(current_map_hash, str)
            or len(current_map_hash) != 64
            or (map_hash and current_map_hash != map_hash)
        ):
            raise ValueError("Missing/different map fingerprint")
        map_hash = current_map_hash
        verification = metadata.get("registered_start", {})
        final = verification.get("pre_recording_verification", {})
        if (
            verification.get("verified") is not True
            or final.get("verified") is not True
            or verification.get("start_id") != start["id"]
            or verification.get("config_sha256") != start["config_sha256"]
        ):
            raise ValueError("Start verification identity failed")
        samples = final.get("samples", [])
        if len(samples) != 2:
            raise ValueError("Missing stable final pose samples")
        for sample in samples:
            if any(
                len(sample.get(key, [])) != 3
                or not all(
                    type(v) in (int, float) and math.isfinite(v) for v in sample[key]
                )
                for key in ("position", "angles")
            ):
                raise ValueError("Invalid measured pose")
            if (
                math.dist(sample["position"], start["position"]) > 1
                or max(
                    abs((a - b + 180) % 360 - 180)
                    for a, b in zip(sample["angles"], start["angles"])
                )
                > 0.5
            ):
                raise ValueError("Start out of tolerance")
        if math.dist(samples[0]["position"], samples[1]["position"]) > 0.25:
            raise ValueError("Start not stable")
        if metadata.get("setup_events_checked") is not True:
            raise ValueError("Setup events not verified")
        with (episode / "actions.csv").open() as handle:
            actions = list(csv.DictReader(handle))
        if not actions or [int(r["frame_idx"]) for r in actions] != list(
            range(len(actions))
        ):
            raise ValueError("Invalid action alignment")
        for row in actions:
            if not all(np.isfinite(float(row[k])) for k in ACTION_COLUMNS):
                raise ValueError("Nonfinite actions")
            if any(float(row[k]) not in (0, 1) for k in ACTION_COLUMNS[:8]):
                raise ValueError("Invalid binary action")
            if any(
                float(row[k]) != 0 for k in ("use", "fire_left", "fire_right", "jump")
            ):
                raise ValueError("Unexpected interaction/jump in navigation pair")
            age = float(row["frame_age_seconds"])
            if not math.isfinite(age) or not 0 <= age <= max(0.1, 2.5 / config["fps"]):
                raise ValueError("Invalid/stale observation timing")
        event_time = events[0].get("received_monotonic")
        if type(event_time) not in (int, float) or not math.isfinite(event_time):
            raise ValueError("Missing waypoint receive timestamp")
        last_sample = float(actions[-1]["frame_sampled_monotonic"])
        if not math.isfinite(last_sample) or not 0 <= event_time - last_sample <= 0.5:
            raise ValueError("Waypoint event is not aligned to captured trajectory end")
        frame_ids = [int(row["frame_id"]) for row in actions]
        if any(b - a != 1 for a, b in zip(frame_ids, frame_ids[1:])):
            raise ValueError("Skipped/reused source frame IDs require separate review")
        capture = cv2.VideoCapture(str(episode / "video.mp4"))
        n = 0
        try:
            while n <= len(actions):
                ok, frame = capture.read()
                if not ok:
                    break
                n += 1
        finally:
            capture.release()
        if n != len(actions):
            raise ValueError("Frame/action count mismatch")
        video_hash = sha256(episode / "video.mp4")
        if video_hash in excluded or video_hash in seen:
            raise ValueError("Duplicate or exposed source video")
        seen.add(video_hash)
        for source in [
            result_path,
            episode / "metadata.json",
            episode / "actions.csv",
            episode / "video.mp4",
        ]:
            artifacts.append({"path": str(source.resolve()), "sha256": sha256(source)})
        rows.append(
            {
                "goal": goal,
                "episode_id": episode.name,
                "frames": n,
                "source_path": str(episode.resolve()),
                "provenance": {
                    key + "_sha256": sha256(episode / name)
                    for key, name in [
                        ("video", "video.mp4"),
                        ("actions", "actions.csv"),
                        ("metadata", "metadata.json"),
                    ]
                },
                "video_sha256": video_hash,
            }
        )
    status = "integrity_passed_visual_annotation_review_pending"
    if any(x is not None for x in [annotations, targets, visual_review]):
        if any(x is None for x in [annotations, targets, visual_review]):
            raise ValueError(
                "Provide annotations, targets and visual-review note together"
            )
        if not Path(visual_review).read_text().strip():
            raise ValueError("Empty visual review note")
        ledger = json.loads(Path(annotations).read_text())
        mapping = ledger.get("waypoint_goal_ids", {})
        if set(mapping) != {"A", "B"} or mapping["A"] == mapping["B"]:
            raise ValueError(
                "Annotation ledger needs distinct waypoint_goal_ids for A/B"
            )
        dataset = VisualGoalSequenceDataset(
            ledger, json.loads(Path(targets).read_text())
        )
        if {ep["episode_id"] for ep, _, _ in dataset.episodes} != {
            r["episode_id"] for r in rows
        }:
            raise ValueError("Annotation cohort differs from captured pair")
        sources_by_episode = {row["episode_id"]: row for row in rows}
        if any(
            ep["provenance"] != sources_by_episode[ep["episode_id"]]["provenance"]
            or str(Path(ep["source_path"]).resolve())
            != sources_by_episode[ep["episode_id"]]["source_path"]
            for ep, _, _ in dataset.episodes
        ):
            raise ValueError("Annotation source differs from captured pair")
        goals_by_episode = {row["episode_id"]: row["goal"] for row in rows}
        if any(
            ep["start_group_id"] != group["start_id"]
            or ep["source_session_id"] != group["session"]
            for ep, _, _ in dataset.episodes
        ):
            raise ValueError("Annotation start/session group mismatch")
        if any(
            ep["requested_goal_id"] != mapping[goals_by_episode[ep["episode_id"]]]
            for ep, _, _ in dataset.episodes
        ):
            raise ValueError("Annotation requested waypoint mismatch")
        target_records = json.loads(Path(targets).read_text())
        for descriptor in target_records:
            artifacts.append(
                {
                    "path": str(Path(descriptor["crop_path"]).resolve()),
                    "sha256": descriptor["crop_sha256"],
                }
            )
        for path in [annotations, targets, visual_review]:
            artifacts.append(
                {"path": str(Path(path).resolve()), "sha256": sha256(Path(path))}
            )
        status = "passed"
    return {
        "schema": "shionn_paired_goal_review_v1",
        "status": status,
        "group_id": group["id"],
        "plan_sha256": sha256(Path(plan_path)),
        "registry_sha256": start["config_sha256"],
        "start_position": start["position"],
        "start_angles": start["angles"],
        "map_sha256": map_hash,
        "capture_config": capture_config,
        "episodes": rows,
        "artifacts": artifacts,
        "training_approved": False,
        "limits": "Integrity and documented visual review, not navigation learning or generalization",
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--plan", type=Path, default=Path("configs/goal_choice_collection_plan_v1.json")
    )
    p.add_argument("--start-config", type=Path, required=True)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--targets", type=Path)
    p.add_argument("--visual-review", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = review_pair(
        args.plan,
        args.start_config,
        args.root,
        annotations=args.annotations,
        targets=args.targets,
        visual_review=args.visual_review,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "training_approved": False}))


if __name__ == "__main__":
    main()
