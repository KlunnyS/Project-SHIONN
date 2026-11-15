"""Verified offline visual targets; no engine state or training membership changes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .goals import validate_goal_annotation_manifest
from .preprocess import action_array, preprocess_bgr_frame, read_actions


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reviewed_episode(
    episode: dict,
    *,
    for_training: bool = False,
    max_frames: int = 2000,
    training_approval: dict | None = None,
) -> dict:
    """Return complete RGB history/actions and a wait-loss mask, never trim source.

    Offline arrays: frames [T,180,320,3] uint8, actions [T,A] float32,
    loss_mask [T] bool. No recurrent segments are shared across episodes.
    """
    if for_training and (
        episode.get("split_role") != "train"
        or episode.get("training_approved") is not True
    ):
        raise ValueError("Episode is not explicitly training-approved")
    count = episode["frame_count"]
    if not 0 < count <= max_frames:
        raise ValueError("Episode exceeds bounded frame budget")
    path = Path(episode["source_path"])
    for name in ("video", "actions", "metadata"):
        filename = {
            "video": "video.mp4",
            "actions": "actions.csv",
            "metadata": "metadata.json",
        }[name]
        if sha256(path / filename) != episode["provenance"][name + "_sha256"]:
            raise ValueError(f"Source hash mismatch: {filename}")
    metadata = json.loads((path / "metadata.json").read_text())
    if metadata.get("capture_valid") is not True:
        raise ValueError("Capture validity is not verified")
    if for_training and metadata.get("training_approved") is False:
        # Never rewrite immutable capture metadata to grant later membership.
        approval = training_approval or {}
        if (
            approval.get("episode_id") != episode.get("episode_id")
            or approval.get("training_approved") is not True
            or approval.get("video_sha256") != episode["provenance"]["video_sha256"]
            or approval.get("actions_sha256") != episode["provenance"]["actions_sha256"]
            or approval.get("metadata_sha256")
            != episode["provenance"]["metadata_sha256"]
            or not isinstance(approval.get("authorization_ref"), str)
            or not approval["authorization_ref"].strip()
        ):
            raise ValueError(
                "Source metadata forbids training without a matching versioned review approval"
            )
    rows = read_actions(path / "actions.csv")
    if len(rows) != count:
        raise ValueError("Action/annotation count mismatch")
    capture = cv2.VideoCapture(str(path / "video.mp4"))
    frames = []
    try:
        for _ in range(count + 1):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(preprocess_bgr_frame(frame))
    finally:
        capture.release()
    if len(frames) != count:
        raise ValueError("Decoded video/annotation count mismatch")
    mask = np.ones(count, dtype=bool)
    for phase in episode.get("phases_half_open", []):
        start, end = phase["start"], phase["end"]
        if (
            type(start) is not int
            or type(end) is not int
            or not 0 <= start < end <= count
        ):
            raise ValueError("Invalid phase boundaries")
        if phase["name"] == "initial_wait":
            mask[start:end] = False
    return {
        "frames": np.stack(frames),
        "actions": action_array(rows),
        "loss_mask": mask,
    }


def extract_target(episode: dict, bounds: list[int], output: Path) -> dict:
    """Extract frame-zero landmark crop in native pixels, retaining lossless RGB.

    Bounds refer to the reviewed 320x180 frame: [left,top,right,bottom), not
    world coordinates. Crops never update as the camera changes.
    """
    if len(bounds) != 4 or any(type(v) is not int for v in bounds):
        raise ValueError("Crop requires four integer bounds")
    left, top, right, bottom = bounds
    if not 0 <= left < right <= 320 or not 0 <= top < bottom <= 180:
        raise ValueError("Invalid visual crop bounds")
    data = load_reviewed_episode(episode)
    source = Path(episode["source_path"]) / "video.mp4"
    capture = cv2.VideoCapture(str(source))
    try:
        ok, native = capture.read()
    finally:
        capture.release()
    if not ok:
        raise ValueError("Missing target source frame")
    h, w = native.shape[:2]
    x0, y0 = left * w // 320, top * h // 180
    x1, y1 = (right * w + 319) // 320, (bottom * h + 179) // 180
    crop = native[y0:y1, x0:x1]
    if output.exists():
        raise FileExistsError(output)
    ok, encoded = cv2.imencode(".png", crop)
    if not ok:
        raise ValueError("Could not encode target crop")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(encoded.tobytes())
    return {
        "schema": "shionn_persistent_visual_target_v1",
        "goal_id": episode["requested_goal_id"],
        "episode_id": episode["episode_id"],
        "source_video_sha256": episode["provenance"]["video_sha256"],
        "source_frame_idx": 0,
        "source_frame_rgb_sha256": hashlib.sha256(
            cv2.cvtColor(native, cv2.COLOR_BGR2RGB).tobytes()
        ).hexdigest(),
        "source_frame_shape": list(native.shape),
        "review_bounds_xyxy": bounds,
        "native_bounds_xyxy": [x0, y0, x1, y1],
        "crop_path": str(output),
        "crop_sha256": sha256(output),
        "crop_color_order": "RGB when decoded",
        "selection": "human-reviewed visual landmark; approach region entry is success",
        "membership": "development_only",
        "training_approved": False,
        "persistent": True,
        "goal_embedding": "not implemented",
        "loss_enabled_frames": int(data["loss_mask"].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    result = validate_goal_annotation_manifest(manifest)
    if not result["valid"]:
        raise ValueError(result["errors"])
    if args.output_dir.exists():
        raise FileExistsError(
            "Use a new output directory; target evidence is never overwritten"
        )
    selections = {
        "goal_choice_v1_A": [176, 91, 205, 110],
        "goal_choice_v1_B": [121, 90, 142, 112],
    }
    for episode in manifest["episodes"]:
        load_reviewed_episode(episode)
        if episode["requested_goal_id"] not in selections:
            raise ValueError("No reviewed crop selection for this goal")
    descriptors = [
        extract_target(
            ep,
            selections[ep["requested_goal_id"]],
            args.output_dir / (ep["requested_goal_id"] + ".png"),
        )
        for ep in manifest["episodes"]
    ]
    (args.output_dir / "targets.json").write_text(
        json.dumps(descriptors, indent=2) + "\n"
    )
    print(json.dumps({"targets": len(descriptors), "training_approved": False}))


if __name__ == "__main__":
    main()
