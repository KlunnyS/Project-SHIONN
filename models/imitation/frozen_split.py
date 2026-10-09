"""Versioned, episode-level split enforcement for cached demonstrations."""

from __future__ import annotations

# Python standard library: read, validate, and fingerprint the split manifest.
import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

# Sibling module: cached episode description from models.imitation.dataset.
from .dataset import EpisodeManifest

DEFAULT_SPLIT_PATH = Path("data/splits/frozen_v1.json")
ROLES = {"train", "validation", "evaluation"}


@dataclass(frozen=True)
class FrozenSplit:
    split_id: str
    digest: str
    map_roles: dict[str, str]
    episodes: dict[tuple[str, str], dict]

    def verify_cache(self, manifests: list[EpisodeManifest], group: str) -> None:
        expected = {name: item for (cache_group, name), item in self.episodes.items()
                    if cache_group == group}
        actual = {item.name: item for item in manifests}
        if len(actual) != len(manifests):
            raise ValueError(f"Duplicate episode IDs in {group} cache")
        missing = set(expected) - set(actual)
        added = set(actual) - set(expected)
        if missing or added:
            raise ValueError(
                f"{group} cache differs from frozen split {self.split_id}: "
                f"missing {sorted(missing)[:3]}, unregistered {sorted(added)[:3]}"
            )
        for name, item in actual.items():
            if item.map_name != expected[name]["map"]:
                raise ValueError(f"Frozen split map mismatch for {name}: {item.map_name}")

    def role(self, manifest: EpisodeManifest, group: str) -> str:
        return self.episodes[(group, manifest.name)]["role"]


def load_frozen_split(path: Path) -> FrozenSplit:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if payload.get("version") != 1 or not isinstance(payload.get("split_id"), str):
        raise ValueError(f"Unsupported frozen split format in {path}")
    map_roles = payload.get("map_roles")
    if not isinstance(map_roles, dict) or not map_roles or set(map_roles.values()) - ROLES:
        raise ValueError("Frozen split needs valid map roles")
    episodes: dict[tuple[str, str], dict] = {}
    ids: set[str] = set()
    for item in payload.get("episodes", []):
        name = item["id"]
        group = item["cache_group"]
        map_name = item["map"]
        role = item["role"]
        date.fromisoformat(item["recorded_at"])
        if not name or not group or (group, name) in episodes or name in ids:
            raise ValueError(f"Duplicate or empty frozen episode ID: {name}")
        if role not in ROLES or map_roles.get(map_name) != role:
            raise ValueError(f"Invalid frozen split role for {name} on {map_name}")
        if group != "base" and role != "train":
            raise ValueError(f"Extra cache episode {name} cannot be {role}")
        ids.add(name)
        episodes[(group, name)] = item
    if not episodes:
        raise ValueError("Frozen split has no episodes")
    return FrozenSplit(payload["split_id"], hashlib.sha256(raw).hexdigest(), map_roles, episodes)
