"""Bounded setup-only pose restoration; never neural policy observations."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from pathlib import Path

from episode_events import EpisodeEventStream

NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
POSE = re.compile(
    r"\bsetpos\s+("
    + NUMBER
    + r")\s+("
    + NUMBER
    + r")\s+("
    + NUMBER
    + r");\s*setang\s+("
    + NUMBER
    + r")\s+("
    + NUMBER
    + r")\s+("
    + NUMBER
    + r")"
)


def load_start(path: Path, start_id: str, map_name: str) -> dict:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema") != "shionn_registered_starts_v1":
        raise ValueError("Unsupported registered-start schema")
    starts = config.get("starts", [])
    ids = [s.get("id") for s in starts]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate start IDs")
    selected = next((s for s in starts if s.get("id") == start_id), None)
    if selected is None or selected.get("map") != map_name:
        raise ValueError("Unknown start or map mismatch")
    for field in ("position", "angles"):
        values = selected.get(field)
        if (
            not isinstance(values, list)
            or len(values) != 3
            or not all(type(v) in (int, float) and math.isfinite(v) for v in values)
        ):
            raise ValueError(f"Invalid {field}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", map_name):
        raise ValueError("Unsafe map name")
    return {
        **selected,
        "config_sha256": hashlib.sha256(raw).hexdigest(),
        "config_path": str(path.resolve()),
    }


class SetupEventGuard:
    """Observe raw console chunks through setup, including split event lines."""

    def __init__(self, *, wait_for_ready=True):
        self.stream = EpisodeEventStream()
        self.ready = not wait_for_ready
        self.error = None

    def feed(self, text):
        self.stream.feed(text)
        while (event := self.stream.pop()) is not None:
            if event.name == "chamber_ready":
                if self.ready:
                    self.error = "Invalid start: chamber reloaded during setup"
                self.ready = True
            elif self.ready and event.name in {
                "waypoint_reached",
                "goal_reached",
                "episode_failed",
            }:
                self.error = f"Invalid start: setup consumed {event.name}|{event.value}"

    def check(self):
        if self.error:
            raise RuntimeError(self.error)


def reject_setup_events(controller):
    """Reject spent waypoint triggers/terminal events from this map-load epoch."""
    guard = getattr(controller, "_setup_event_guard", None)
    if isinstance(guard, SetupEventGuard):
        guard.check()
        return
    lines = getattr(controller, "event_buffer", [])
    if not isinstance(lines, list):
        return
    stream = EpisodeEventStream()
    stream.feed("\n".join(lines) + "\n")
    while (event := stream.pop()) is not None:
        if event.name in {"waypoint_reached", "goal_reached", "episode_failed"}:
            raise RuntimeError(
                f"Invalid start: setup consumed {event.name}|{event.value}"
            )


def console_query(controller, command: str, timeout: float = 2.0) -> str:
    """Require complete unique marker lines; echoed commands are not replies."""
    marker = "SHIONN_START_" + uuid.uuid4().hex
    if (
        controller.send_command(f"echo {marker}_BEGIN; {command}; echo {marker}_END")
        is False
    ):
        raise RuntimeError("Start verification command failed")
    deadline = time.monotonic() + timeout
    buffer = ""
    while time.monotonic() < deadline:
        buffer = (buffer + controller.read_console(print_to_terminal=False))[-65536:]
        if getattr(controller, "_reject_setup_terminals", False) is True:
            reject_setup_events(controller)
        if time.monotonic() >= deadline:
            raise RuntimeError("Late start telemetry reply")
        lines = buffer.splitlines(keepends=True)
        complete = [line.strip() for line in lines if line.endswith(("\n", "\r"))]
        if marker + "_BEGIN" in complete and marker + "_END" in complete:
            begin, end = (
                complete.index(marker + "_BEGIN"),
                complete.index(marker + "_END"),
            )
            if end > begin:
                return "\n".join(complete[begin + 1 : end])
        time.sleep(0.02)
    raise RuntimeError("Timed out waiting for fresh start telemetry")


def read_pose(controller, timeout=2.0) -> dict:
    body = console_query(controller, "getpos", timeout)
    matches = list(POSE.finditer(body))
    if len(matches) != 1:
        raise RuntimeError("Missing or ambiguous player pose telemetry")
    values = list(map(float, matches[0].groups()))
    if not all(math.isfinite(v) for v in values):
        raise RuntimeError("Nonfinite player pose")
    return {
        "position": values[:3],
        "angles": values[3:],
        "measured_monotonic": time.monotonic(),
    }


def verify_start(
    controller, start: dict, *, position_tolerance=1.0, angle_tolerance=0.5, timeout=2.0
) -> dict:
    """Require two fresh stable pose readings; getpos is setup-only telemetry."""
    if not all(
        math.isfinite(v) and v > 0
        for v in (position_tolerance, angle_tolerance, timeout)
    ):
        raise ValueError("Verification limits must be finite and positive")
    samples = []
    deadline = time.monotonic() + timeout
    last_error = "No settled pose received"
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        sample = read_pose(controller, remaining)
        distance = math.dist(sample["position"], start["position"])
        error = max(
            abs((a - b + 180) % 360 - 180)
            for a, b in zip(sample["angles"], start["angles"])
        )
        if distance > position_tolerance or error > angle_tolerance:
            last_error = (
                f"Start pose outside tolerance: position={distance:.3f}, angles={error:.3f}; "
                f"measured position={sample['position']}, angles={sample['angles']}"
            )
            samples = []
        else:
            if (
                samples
                and math.dist(samples[-1]["position"], sample["position"]) > 0.25
            ):
                last_error = "Player position not stable during start verification"
                samples = []
            samples.append(sample)
            if len(samples) == 2:
                break
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    if len(samples) != 2:
        raise RuntimeError(last_error)
    return {
        "start_id": start["id"],
        "map": start["map"],
        "config_sha256": start["config_sha256"],
        "config_path": start["config_path"],
        "requested_position": start["position"],
        "requested_angles": start["angles"],
        "position_tolerance_units": position_tolerance,
        "angle_tolerance_degrees": angle_tolerance,
        "verified": True,
        "samples": samples,
        "privileged_setup_only": True,
    }


def restore_start(controller, start: dict, *, attempts=2, timeout=2.0) -> dict:
    """Use setpos/setang, temporarily enabling cheats and restoring their setting."""
    if type(attempts) is not int or not 1 <= attempts <= 3:
        raise ValueError("Restore attempts must be between one and three")
    body = console_query(controller, "sv_cheats", timeout)
    match = re.search(r'"sv_cheats"\s*=\s*"([01])"', body)
    if not match:
        raise RuntimeError("Cannot determine initial sv_cheats; refusing positioning")
    initial = match[1]
    try:
        for attempt in range(attempts):
            command = "sv_cheats 1; setpos " + " ".join(map(str, start["position"]))
            command += "; setang " + " ".join(map(str, start["angles"]))
            console_query(controller, command, timeout)
            time.sleep(0.1)
            try:
                result = verify_start(controller, start, timeout=timeout)
                result["restore_attempts"] = attempt + 1
                return result
            except RuntimeError:
                if attempt + 1 == attempts:
                    raise
    finally:
        guard = getattr(controller, "_reject_setup_terminals", False) is True
        controller._reject_setup_terminals = False
        try:
            restored = console_query(
                controller, f"sv_cheats {initial}; sv_cheats", timeout
            )
            if not re.search(r'"sv_cheats"\s*=\s*"' + initial + r'"', restored):
                raise RuntimeError(
                    "Could not verify restoration of sv_cheats; recording refused"
                )
        finally:
            controller._reject_setup_terminals = guard
