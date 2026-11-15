"""Offline start registration, paired collection commands and integrity checks."""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path

from models.imitation.visual_targets import sha256
from registered_starts import POSE, load_start


def load_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("schema") != "shionn_paired_goal_collection_v1":
        raise ValueError("Invalid collection schema")
    groups = plan.get("groups", [])
    for field in ("id", "start_id", "session"):
        values = [g[field] for g in groups]
        if len(set(values)) != len(values):
            raise ValueError(f"Duplicate collection {field}")
    if (
        len(groups) != 3
        or sum(g["intent"] == "train_candidate" for g in groups) != 2
        or sum(g["intent"] == "heldout_reserved" for g in groups) != 1
    ):
        raise ValueError(
            "Pilot requires two training-candidate pairs and one reserved pair"
        )
    if any(sorted(g["goals"]) != ["A", "B"] for g in groups):
        raise ValueError("Each group requires exactly A and B")
    for key in (
        "minimum_position_separation_units",
        "minimum_group_angle_difference_degrees",
    ):
        if (
            type(plan.get(key)) not in (int, float)
            or not math.isfinite(plan[key])
            or plan[key] <= 0
        ):
            raise ValueError("Invalid diversity gate")
    return plan


def group_context(
    plan_path,
    registry_path,
    group_id,
    session,
    goal,
    *,
    stage="first_pair",
    first_pair_review=None,
):
    plan = load_plan(plan_path)
    group = next((g for g in plan["groups"] if g["id"] == group_id), None)
    if group is None or session != group["session"] or goal not in group["goals"]:
        raise ValueError("Collection group/session/goal mismatch")
    registry = json.loads(Path(registry_path).read_text())
    available = {s["id"] for s in registry.get("starts", [])}
    if stage == "first_pair" and group_id != plan["groups"][0]["id"]:
        raise ValueError("Validate the first pair before releasing later groups")
    if stage not in {"first_pair", "full"}:
        raise ValueError("Unknown collection stage")
    if stage == "full":
        if first_pair_review is None:
            raise ValueError("Later groups require the passed first-pair review")
        review = json.loads(Path(first_pair_review).read_text())
        if review.get("status") != "passed" or review.get("plan_sha256") != sha256(
            Path(plan_path)
        ):
            raise ValueError("First-pair review is not passed or plan identity differs")
        if review.get("group_id") != plan["groups"][0]["id"]:
            raise ValueError("Review is for the wrong first group")
        first_start = load_start(
            Path(registry_path), plan["groups"][0]["start_id"], plan["map"]
        )
        if (
            review.get("start_position") != first_start["position"]
            or review.get("start_angles") != first_start["angles"]
        ):
            raise ValueError("Reviewed first start changed")
        for artifact in review.get("artifacts", []):
            if sha256(Path(artifact["path"])) != artifact["sha256"]:
                raise ValueError("First-pair review evidence changed")
        if not review.get("artifacts"):
            raise ValueError("First-pair review has no pinned evidence")
    if stage == "full" and not all(g["start_id"] in available for g in plan["groups"]):
        raise ValueError("Full matrix needs all three registered starts")
    requested = load_start(Path(registry_path), group["start_id"], plan["map"])
    exposed = set()
    for path in plan["excluded_annotation_ledgers"]:
        ledger = json.loads(Path(path).read_text())
        exposed.update(ep["start_group_id"] for ep in ledger["episodes"])
    starts = []
    for slot in plan["groups"]:
        if slot["start_id"] not in available:
            continue
        start = load_start(Path(registry_path), slot["start_id"], plan["map"])
        if start["id"] in exposed:
            raise ValueError("Exposed development start cannot be reused")
        for excluded in plan["excluded_poses"]:
            if (
                math.dist(start["position"], excluded["position"])
                < plan["minimum_position_separation_units"]
            ):
                raise ValueError(
                    "New start is too close to an exposed development pose"
                )
        for old in starts:
            if (
                math.dist(start["position"], old["position"])
                < plan["minimum_position_separation_units"]
            ):
                raise ValueError("Prospective start groups are too close")
            angle = max(
                abs((a - b + 180) % 360 - 180)
                for a, b in zip(start["angles"], old["angles"])
            )
            if angle < plan["minimum_group_angle_difference_degrees"]:
                raise ValueError(
                    "Prospective start orientations are insufficiently different"
                )
        starts.append(start)
    return {
        "plan_sha256": sha256(Path(plan_path)),
        "plan_path": str(Path(plan_path).resolve()),
        "group_id": group_id,
        "intended_role": group["intent"],
        "session_id": session,
        "start_id": requested["id"],
        "registry_sha256": sha256(Path(registry_path)),
        "requested_waypoint": goal,
        "stage": stage,
        "membership": "development_pending",
        "first_pair_review_sha256": None
        if first_pair_review is None
        else sha256(Path(first_pair_review)),
        "expected_map_sha256": None if stage == "first_pair" else review["map_sha256"],
        "expected_capture_config": plan.get("capture_config")
        if stage == "first_pair"
        else review["capture_config"],
    }


def register_pose(plan_path, base, output, group_id, pose_text):
    plan = load_plan(plan_path)
    group = next(g for g in plan["groups"] if g["id"] == group_id)
    match = POSE.fullmatch(pose_text.strip())
    if not match:
        raise ValueError(
            "Paste one complete setpos ...;setang ... line, without other console commands"
        )
    values = list(map(float, match.groups()))
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Nonfinite registered pose")
    config = (
        {"schema": "shionn_registered_starts_v1", "starts": []}
        if base is None
        else json.loads(Path(base).read_text())
    )
    if config.get("schema") != "shionn_registered_starts_v1":
        raise ValueError("Invalid source registry")
    if any(s["id"] == group["start_id"] for s in config["starts"]):
        raise ValueError("Start already registered; use a new reviewed plan/version")
    entry = {
        "id": group["start_id"],
        "map": plan["map"],
        "position": values[:3],
        "angles": values[3:],
        "provenance": "user-provided getpos; no automatic live validation",
        "eligibility": "development_pending",
        "collection_intent": group["intent"],
    }
    config["starts"].append(entry)
    # Validate planned geometry against exposed and already registered poses
    # before publishing; caller still must hand-check visibility and collision.
    for old in [*plan["excluded_poses"], *config["starts"][:-1]]:
        if (
            math.dist(entry["position"], old["position"])
            < plan["minimum_position_separation_units"]
        ):
            raise ValueError("Pose too close to another/exposed start")
    for old in config["starts"][:-1]:
        if (
            max(
                abs((a - b + 180) % 360 - 180)
                for a, b in zip(entry["angles"], old["angles"])
            )
            < plan["minimum_group_angle_difference_degrees"]
        ):
            raise ValueError("Use a genuinely different camera heading")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        handle.write(json.dumps(config, indent=2) + "\n")
    return entry


def commands(
    plan_path, registry, group_id, *, stage="first_pair", first_pair_review=None
):
    plan = load_plan(plan_path)
    g = next(g for g in plan["groups"] if g["id"] == group_id)
    result = []
    for goal in g["goals"]:
        group_context(
            plan_path,
            registry,
            group_id,
            g["session"],
            goal,
            stage=stage,
            first_pair_review=first_pair_review,
        )
        root = f"model_attempts/goal_choice_pilot_pending/{g['session']}/{goal}"
        cmd = [
            ".venv/bin/python",
            "record_recovery.py",
            "--map",
            plan["map"],
            "--registered-start",
            g["start_id"],
            "--start-config",
            str(registry),
            "--collection-plan",
            str(plan_path),
            "--collection-group",
            group_id,
            "--collection-session",
            g["session"],
            "--collection-stage",
            stage,
            "--map-file",
            "/mnt/game-main/SteamLibrary/steamapps/common/Portal 2/portal2/maps/goal_choice_v1.bsp",
            "--requested-waypoint",
            goal,
            "--manual-focus",
            "--session-controls",
            "--max-seconds",
            "20",
            "--fps",
            "24",
            "--width",
            "1920",
            "--height",
            "1080",
            "--output",
            "DP-1",
            "--episodes-root",
            root,
            "--result-file",
            root + "/result.json",
        ]
        if first_pair_review is not None:
            cmd.extend(["--first-pair-review", str(first_pair_review)])
        result.append(shlex.join(cmd))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--pose", required=True)
    reg.add_argument("--base", type=Path)
    reg.add_argument("--output", type=Path, required=True)
    cmd = sub.add_parser("commands")
    cmd.add_argument("--first-pair-review", type=Path)
    cmd.add_argument("--start-config", type=Path, required=True)
    cmd.add_argument("--stage", choices=["first_pair", "full"], default="first_pair")
    for parser in (reg, cmd):
        parser.add_argument(
            "--plan",
            type=Path,
            default=Path("configs/goal_choice_collection_plan_v1.json"),
        )
        parser.add_argument(
            "--group", choices=["train_01", "train_02", "eval_01"], required=True
        )
    args = p.parse_args()
    if args.action == "register":
        print(
            json.dumps(
                register_pose(args.plan, args.base, args.output, args.group, args.pose),
                indent=2,
            )
        )
    else:
        for line in commands(
            args.plan,
            args.start_config,
            args.group,
            stage=args.stage,
            first_pair_review=args.first_pair_review,
        ):
            print(line)


if __name__ == "__main__":
    main()
