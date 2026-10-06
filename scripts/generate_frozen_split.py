"""Generate a reviewable split snapshot from existing cache metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from models.imitation.dataset import discover_cached_episodes


MAP_ROLES = {
    **{f"dataset_test{number}": "train" for number in range(1, 11)},
    "dataset_test11": "validation",
    "dataset_test12": "validation",
    "evaluation1": "evaluation",
    "evaluation2": "evaluation",
}


def build_manifest(cache_groups: dict[str, Path], split_id: str) -> dict:
    episodes = []
    seen = set()
    for group, cache_dir in cache_groups.items():
        for item in discover_cached_episodes(cache_dir):
            match = re.fullmatch(r"episode_(\d{4})(\d{2})(\d{2})_\d{6}_\d+", item.name)
            if not match:
                raise ValueError(f"Cannot determine recording date for {item.name}")
            if item.name in seen:
                raise ValueError(f"Duplicate episode ID: {item.name}")
            seen.add(item.name)
            role = MAP_ROLES.get(item.map_name)
            if role is None:
                raise ValueError(f"No frozen role for chamber {item.map_name}")
            if group != "base" and role != "train":
                raise ValueError(f"Extra cache cannot contain {role} episode {item.name}")
            episodes.append({
                "id": item.name,
                "map": item.map_name,
                "role": role,
                "recorded_at": "-".join(match.groups()),
                "cache_group": group,
            })
    return {
        "version": 1,
        "split_id": split_id,
        "map_roles": MAP_ROLES,
        "episodes": sorted(episodes, key=lambda item: item["id"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-cache", type=Path, default=Path("data/datasets/cached_frames"))
    parser.add_argument("--recovery-cache", type=Path, default=Path("data/datasets/recovery_cached_frames"))
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new versioned path")
    groups = {"base": args.base_cache, "recovery": args.recovery_cache}
    for group, path in groups.items():
        if not path.is_dir():
            parser.error(f"{group} cache missing: {path}")
    manifest = build_manifest(groups, args.split_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    counts = {role: sum(item["role"] == role for item in manifest["episodes"])
              for role in ("train", "validation", "evaluation")}
    print(f"Wrote {args.output}: {counts}")


if __name__ == "__main__":
    main()
