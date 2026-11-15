"""Goal descriptors and offline validation for frame-aligned waypoint labels."""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

GOAL_ANNOTATION_SCHEMA = "shionn_goal_annotations_v1"
GOAL_COORDINATE_SYSTEM = "normalized_frame_xy_origin_top_left_y_down"
GOAL_KIND_IDS = {"exit": 0, "waypoint": 1}
GOAL_KINDS = {value: key for key, value in GOAL_KIND_IDS.items()}
ACTION_OWNERS = {"human", "policy", "unknown"}
SPLIT_ROLES = {
    "annotation_only",
    "train",
    "development",
    "validation",
    "regression",
    "quarantined",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GoalCondition:
    """Caller-owned persistent goal plus its current image-relative projection.

    Coordinates are updated as the view changes while ``goal_id`` remains the
    selected objective. An invisible target has no pixel projection and maps
    to canonical zero coordinates at the model boundary.
    """

    goal_id: str
    x_norm: float | None
    y_norm: float | None
    visible: bool
    target_kind: str

    def __post_init__(self):
        if not isinstance(self.goal_id, str) or not self.goal_id.strip():
            raise ValueError("goal_id must be a nonempty stable identifier")
        if self.target_kind not in GOAL_KIND_IDS:
            raise ValueError(f"Unsupported target kind: {self.target_kind!r}")
        if not isinstance(self.visible, bool):
            raise ValueError("visible must be boolean")
        if self.visible:
            if self.x_norm is None or self.y_norm is None:
                raise ValueError("Visible goals require x_norm and y_norm")
            if not all(
                type(value) in (int, float)
                and math.isfinite(value)
                and -1 <= value <= 1
                for value in (self.x_norm, self.y_norm)
            ):
                raise ValueError(
                    "Visible goal coordinates must be finite and in [-1,1]"
                )
        elif self.x_norm is not None or self.y_norm is not None:
            raise ValueError("Invisible goals must not retain stale image coordinates")

    def as_tensor(self, *, device=None, batch_size: int = 1) -> torch.Tensor:
        """Return [B,4] model input; target kind remains an integer ID."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        x = self.x_norm if self.visible else 0.0
        y = self.y_norm if self.visible else 0.0
        row = [x, y, float(self.visible), float(GOAL_KIND_IDS[self.target_kind])]
        return torch.tensor(row, dtype=torch.float32, device=device).repeat(
            batch_size, 1
        )


def validate_goal_annotation_manifest(
    manifest: dict,
    *,
    require_counterfactual_pairs: bool = False,
) -> dict:
    """Validate frame completeness, coordinate semantics, transitions and split leakage."""
    errors: list[str] = []
    if manifest.get("schema") != GOAL_ANNOTATION_SCHEMA:
        errors.append(f"schema must be {GOAL_ANNOTATION_SCHEMA!r}")
    if manifest.get("coordinate_system") != GOAL_COORDINATE_SYSTEM:
        errors.append(f"coordinate_system must be {GOAL_COORDINATE_SYSTEM!r}")
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        errors.append("episodes must be a nonempty list")
        episodes = []

    seen_ids: set[str] = set()
    group_roles: dict[tuple[str, str], set[str]] = {}
    goals_by_start: dict[tuple[str, str], set[str]] = {}
    for ep_index, episode in enumerate(episodes):
        prefix = f"episodes[{ep_index}]"
        if not isinstance(episode, dict):
            errors.append(f"{prefix} must be an object")
            continue
        episode_id = episode.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id:
            errors.append(f"{prefix}.episode_id must be nonempty")
            episode_id = f"<missing:{ep_index}>"
        if episode_id in seen_ids:
            errors.append(f"duplicate episode_id {episode_id!r}")
        seen_ids.add(episode_id)

        role = episode.get("split_role")
        if role not in SPLIT_ROLES:
            errors.append(f"{prefix}.split_role must be one of {sorted(SPLIT_ROLES)}")
        if role == "train" and not episode.get("training_approved", False):
            errors.append(
                f"{prefix} is assigned to train without training_approved=true"
            )
        if role == "train" and episode.get("map_name") in {
            "dataset_test11",
            "dataset_test12",
        }:
            errors.append(
                f"{prefix} places frozen holdout map {episode.get('map_name')} in train"
            )
        if episode.get("quarantined") and role == "train":
            errors.append(f"{prefix} is quarantined but assigned to train")

        for group_field in ("source_session_id", "start_group_id"):
            group = episode.get(group_field)
            if not isinstance(group, str) or not group:
                errors.append(
                    f"{prefix}.{group_field} must be nonempty for leakage checks"
                )
            elif role in SPLIT_ROLES:
                group_roles.setdefault((group_field, group), set()).add(role)
        if role in SPLIT_ROLES - {"annotation_only"} and isinstance(
            episode.get("start_group_id"), str
        ):
            goal_group = (role, episode["start_group_id"])
            requested = episode.get("requested_goal_id")
            if isinstance(requested, str) and requested:
                goals_by_start.setdefault(goal_group, set()).add(requested)

        frame_count = episode.get("frame_count")
        if type(frame_count) is not int or frame_count < 1:
            errors.append(f"{prefix}.frame_count must be a positive integer")
            frame_count = 0
        provenance = episode.get("provenance")
        if not isinstance(provenance, dict):
            errors.append(f"{prefix}.provenance is required for every annotated source")
        else:
            for hash_name in ("video_sha256", "actions_sha256", "metadata_sha256"):
                if not isinstance(
                    provenance.get(hash_name), str
                ) or not SHA256_RE.fullmatch(provenance[hash_name]):
                    errors.append(
                        f"{prefix}.provenance.{hash_name} must be a SHA-256 hex digest"
                    )

        annotations = episode.get("annotations")
        if not isinstance(annotations, list):
            errors.append(f"{prefix}.annotations must be a list")
            continue
        indices = [
            row.get("frame_idx") if isinstance(row, dict) else None
            for row in annotations
        ]
        if any(type(index) is not int for index in indices) or indices != list(
            range(frame_count)
        ):
            errors.append(
                f"{prefix}.annotations must cover every frame_idx 0..{frame_count - 1} exactly once"
            )

        last_goal_id = episode.get("requested_goal_id")
        for row_index, row in enumerate(annotations):
            row_prefix = f"{prefix}.annotations[{row_index}]"
            if not isinstance(row, dict):
                errors.append(f"{row_prefix} must be an object")
                continue
            goal_id = row.get("goal_id")
            is_transition = row.get("goal_transition", False)
            transition_reason = row.get("transition_reason")
            if not isinstance(is_transition, bool):
                errors.append(f"{row_prefix}.goal_transition must be boolean")
                is_transition = False
            if is_transition and (
                not isinstance(transition_reason, str) or not transition_reason.strip()
            ):
                errors.append(f"{row_prefix} goal transition requires a reason")
            if not is_transition and transition_reason is not None:
                errors.append(
                    f"{row_prefix} has a transition reason without a transition"
                )
            if not isinstance(goal_id, str) or not goal_id:
                errors.append(f"{row_prefix}.goal_id must be nonempty")
            elif last_goal_id is not None and goal_id != last_goal_id:
                if not is_transition or not transition_reason:
                    errors.append(
                        f"{row_prefix} changes goal_id without an explicit transition and reason"
                    )
            elif is_transition and goal_id == last_goal_id:
                errors.append(
                    f"{row_prefix} declares a goal transition without changing goal_id"
                )
            if goal_id is not None:
                last_goal_id = goal_id
            if row.get("target_kind") not in GOAL_KIND_IDS:
                errors.append(f"{row_prefix}.target_kind must be exit or waypoint")
            if not isinstance(row.get("visible"), bool):
                errors.append(f"{row_prefix}.visible must be boolean")
            elif row["visible"]:
                coords = (row.get("x_norm"), row.get("y_norm"))
                if not all(
                    type(value) in (int, float)
                    and math.isfinite(value)
                    and -1 <= value <= 1
                    for value in coords
                ):
                    errors.append(
                        f"{row_prefix} visible coordinates must be finite and in [-1,1]"
                    )
            elif row.get("x_norm") is not None or row.get("y_norm") is not None:
                errors.append(
                    f"{row_prefix} invisible target must not carry stale coordinates"
                )
            if row.get("action_owner") not in ACTION_OWNERS:
                errors.append(
                    f"{row_prefix}.action_owner must be human, policy, or unknown"
                )
            elif role == "train" and row["action_owner"] != "human":
                errors.append(
                    f"{row_prefix} is not human action and cannot be an expert target"
                )

    for (field, group), roles in group_roles.items():
        training_roles = "train" in roles
        held_roles = bool(
            roles
            & {
                "annotation_only",
                "development",
                "validation",
                "regression",
                "quarantined",
            }
        )
        if training_roles and held_roles:
            errors.append(
                f"dataset leakage: {field} {group!r} occurs in train and held-out roles"
            )
    if require_counterfactual_pairs:
        if not any(len(goal_ids) >= 2 for goal_ids in goals_by_start.values()):
            errors.append(
                "counterfactual coverage: no start group has paired requested goals"
            )
        for (role, start_group), goal_ids in goals_by_start.items():
            if len(goal_ids) < 2:
                errors.append(
                    f"counterfactual coverage: {role} start group {start_group!r} "
                    "must contain at least two requested goals"
                )
    return {"valid": not errors, "episode_count": len(episodes), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--require-counterfactual-pairs",
        action="store_true",
        help="require at least two requested goal IDs for every start group",
    )
    args = parser.parse_args()
    result = validate_goal_annotation_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        require_counterfactual_pairs=args.require_counterfactual_pairs,
    )
    print(json.dumps(result, indent=2))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
