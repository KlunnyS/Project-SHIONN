"""Desktop sampling and non-blocking camera observations for model attempts.

Only the runner reads netconsole. Camera measurements never enter the policy.
"""

from __future__ import annotations

import re
import threading
import time
import uuid


ANGLES = re.compile(r"setang\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def angle_delta(current, previous):
    return (current - previous + 180.0) % 360.0 - 180.0


class DesktopTelemetry:
    """Sample independently of inference, logging actual collection intervals."""

    def __init__(self, sample, log, hz=15.0):
        self.sample = sample
        self.log = log
        self.interval = 1.0 / hz
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None
        self._thread = threading.Thread(target=self._run, name="desktop-telemetry", daemon=True)
        self.samples = 0
        self.focus_losses = 0
        self.cursor_exits = 0
        self.unknown_focus = 0
        self.unknown_cursor = 0
        self.max_sample_gap_seconds = None
        self.max_collection_seconds = 0.0
        self._last_sample_time = None
        self._ended_monotonic = None
        self._previous_focus = None
        self._previous_inside = None

    def start(self):
        self._thread.start()

    def collect(self):
        began = time.monotonic()
        error = None
        try:
            state = self.sample()
        except Exception as exc:
            state = {"window": None, "cursor": None, "active_workspace": None,
                     "cursor_inside_game_window": None}
            error = f"{type(exc).__name__}: {exc}"
        ended = time.monotonic()
        focused = (state.get("window") or {}).get("portal_focused")
        inside = state.get("cursor_inside_game_window")
        record = dict(state, sample_started_monotonic=began,
                      sample_finished_monotonic=ended, portal_focused=focused,
                      collection_seconds=ended - began, error=error)
        with self._lock:
            self.samples += 1
            self.max_collection_seconds = max(self.max_collection_seconds, ended - began)
            if self._last_sample_time is not None:
                gap = ended - self._last_sample_time
                self.max_sample_gap_seconds = max(self.max_sample_gap_seconds or 0, gap)
            self._last_sample_time = ended
            self.focus_losses += focused is False and self._previous_focus is not False
            self.cursor_exits += inside is False and self._previous_inside is not False
            self.unknown_focus += focused is None
            self.unknown_cursor += inside is None
            self._previous_focus = focused
            self._previous_inside = inside
            self._latest = record
        self.log.write("desktop_state", phase="telemetry", **record)
        return record

    def _run(self):
        while not self._stop.is_set():
            began = time.monotonic()
            self.collect()
            self._stop.wait(max(0, self.interval - (time.monotonic() - began)))

    def latest(self, max_age=0.25):
        with self._lock:
            record = dict(self._latest) if self._latest else None
        if record is None:
            return {"portal_focused": None, "cursor_inside_game_window": None,
                    "sample_age_seconds": None, "stale": True, "window": None}
        # Age includes collection time: these IPC queries are not an atomic snapshot.
        age = time.monotonic() - record["sample_started_monotonic"]
        record.update(sample_age_seconds=age, stale=age > max_age)
        if record["stale"]:
            record["portal_focused"] = None
            record["cursor_inside_game_window"] = None
        return record

    def stop(self):
        self._ended_monotonic = time.monotonic()
        self._stop.set()
        # Each compositor query has a bounded timeout. Join before closing log.
        if self._thread.ident is not None:
            self._thread.join()

    def summary(self):
        gap = self.max_sample_gap_seconds
        if self._last_sample_time is not None:
            ended = self._ended_monotonic if self._ended_monotonic is not None else time.monotonic()
            gap = max(gap or 0, ended - self._last_sample_time)
        return {"samples": self.samples, "focus_losses": self.focus_losses,
                "cursor_exits": self.cursor_exits, "unknown_focus_samples": self.unknown_focus,
                "unknown_cursor_samples": self.unknown_cursor,
                "max_collection_seconds": self.max_collection_seconds,
                "max_sample_gap_seconds": gap}


class CameraTelemetry:
    """Bracket getpos replies so stale/partial console data cannot look fresh."""

    def __init__(self, controller, log, hz=15.0, timeout=0.5):
        self.controller = controller
        self.log = log
        self.interval = 1.0 / hz
        self.timeout = timeout
        self.next_request = 0
        self.pending = None
        self.buffer = ""
        self.previous = None
        self.prefix = "SHIONN_CAMERA_" + uuid.uuid4().hex
        self.serial = 0
        self.samples = 0
        self.unknown_samples = 0
        self.max_sample_gap_seconds = None
        self._last_received = None
        self._ended_monotonic = None

    def request(self, tick):
        now = time.monotonic()
        if self.pending is not None or now < self.next_request:
            return
        self.serial += 1
        marker = f"{self.prefix}_{self.serial}"
        self.pending = {"marker": marker, "requested_monotonic": now, "after_tick": tick}
        self.buffer = ""
        sent = self.controller.send_command(f"echo {marker}_BEGIN; getpos; echo {marker}_END")
        self.next_request = now + self.interval
        if sent is False:
            self._finish(None, "command_failed", tick)

    def consume(self, output, tick):
        if self.pending is None:
            return
        self.buffer = (self.buffer + output)[-16384:]
        begin = self.pending["marker"] + "_BEGIN"
        end = self.pending["marker"] + "_END"
        # Match whole lines, excluding any echoed command text.
        # Source's echo appends a space before CRLF on the live netconsole.
        # Normalize line whitespace while still requiring an exact marker line.
        lines = [line.strip() for line in self.buffer.splitlines()]
        if begin in lines and end in lines and lines.index(end) > lines.index(begin):
            body = "\n".join(lines[lines.index(begin) + 1:lines.index(end)])
            match = ANGLES.search(body)
            self._finish(tuple(map(float, match.groups())) if match else None,
                         None if match else "missing_angles", tick)
        elif time.monotonic() - self.pending["requested_monotonic"] >= self.timeout:
            self._finish(None, "timeout", tick)

    def _finish(self, angles, error, tick):
        now = time.monotonic()
        pending = self.pending
        if angles is not None and now - pending["requested_monotonic"] > self.timeout:
            angles, error = None, "late_reply"
        delta = None
        if angles is not None and self.previous is not None:
            delta = [angle_delta(value, old) for value, old in zip(angles, self.previous["angles"])]
        record = dict(pending, received_monotonic=now, received_after_tick=tick,
                      angles=angles, delta_degrees=delta, error=error,
                      previous_request_monotonic=(self.previous or {}).get("requested_monotonic"))
        self.log.write("camera_response", **record)
        if error != "run_ended":
            self.samples += 1
            self.unknown_samples += angles is None
            if self._last_received is not None:
                self.max_sample_gap_seconds = max(self.max_sample_gap_seconds or 0, now - self._last_received)
            self._last_received = now
        self.previous = dict(pending, angles=angles) if angles is not None else None
        self.pending = None

    def close(self, tick):
        self._ended_monotonic = time.monotonic()
        if self.pending is not None:
            self._finish(None, "run_ended", tick)

    def summary(self):
        gap = self.max_sample_gap_seconds
        if self._last_received is not None:
            ended = self._ended_monotonic if self._ended_monotonic is not None else time.monotonic()
            gap = max(gap or 0, ended - self._last_received)
        return {"samples": self.samples, "unknown_samples": self.unknown_samples,
                "max_sample_gap_seconds": gap}
