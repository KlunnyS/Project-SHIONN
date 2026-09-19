"""Continuously record human demonstrations from a Portal 2 chamber.

The recorder is intentionally video-only. ``WaylandCamera`` captures raw video
frames and ``FFmpegVideoWriter`` passes ``-an`` to FFmpeg, so episode MP4 files
never contain an audio stream.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import deque
from dataclasses import dataclass

from recorder import EpisodeRecorder
from wrapper import Portal2Controller, is_game_running, launch_game


EVENT_PATTERN = re.compile(
    r"EVT\|(chamber_ready|goal_reached|episode_failed)\|([^\r\n]*)"
)


@dataclass(frozen=True)
class EpisodeEvent:
    name: str
    value: str


class EpisodeEventStream:
    """Turn arbitrarily chunked netconsole text into SHIONN events."""

    def __init__(self) -> None:
        self._partial_line = ""
        self._events: deque[EpisodeEvent] = deque()

    def clear(self) -> None:
        self._partial_line = ""
        self._events.clear()

    def feed(self, text: str) -> None:
        if not text:
            return
        lines = (self._partial_line + text).splitlines(keepends=True)
        self._partial_line = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._partial_line = lines.pop()
        for line in lines:
            match = EVENT_PATTERN.search(line)
            if match:
                self._events.append(EpisodeEvent(match.group(1), match.group(2).strip()))

    def pop(self) -> EpisodeEvent | None:
        return self._events.popleft() if self._events else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously record 24 FPS/FHD Portal 2 demonstration episodes."
    )
    parser.add_argument("--map", dest="map_name", default="dataset_test1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--focus-delay", type=float, default=5.0)
    parser.add_argument("--restart-delay", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument(
        "--episodes", "--episode",
        type=int,
        default=0,
        help="Number of successful episodes to record; failures are retried. 0 records until Ctrl+C.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument(
        "--output",
        help="Wayland output name from `wf-recorder -L`; defaults to the first output.",
    )
    parser.add_argument(
        "--mouse-device",
        help="Pointer event path or case-insensitive name fragment, such as /dev/input/event9 or Basilisk.",
    )
    parser.add_argument(
        "--no-launch",
        action="store_true",
        help="Fail instead of launching Portal 2 when it is not running.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.duration <= 0:
        raise ValueError("--duration must be greater than zero")
    if args.focus_delay < 0 or args.restart_delay < 0:
        raise ValueError("delays must be zero or greater")
    if args.ready_timeout <= 0:
        raise ValueError("--ready-timeout must be greater than zero")
    if args.episodes < 0:
        raise ValueError("--episodes must be zero or greater")
    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise ValueError("capture dimensions and FPS must be greater than zero")


def connect_controller(controller: Portal2Controller, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if controller.connect():
            time.sleep(0.5)
            controller.read_console(print_to_terminal=False)
            controller.event_buffer.clear()
            return
        print("Waiting for Portal 2's netconsole...")
        time.sleep(2.0)
    raise RuntimeError(
        "Could not connect to Portal 2. Restart it with '-netconport "
        f"{controller.port}'."
    )


def wait_for_camera(recorder: EpisodeRecorder, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while recorder.camera.get_latest_frame() is None:
        if recorder.camera.proc and recorder.camera.proc.poll() is not None:
            detail = recorder.camera.error_summary()
            message = "wf-recorder exited before producing a frame"
            if detail:
                message += f":\n{detail}"
            raise RuntimeError(message)
        if time.monotonic() >= deadline:
            raise RuntimeError(
                "No frames received from wf-recorder; check Wayland permissions and the selected output"
            )
        time.sleep(0.05)


def read_events(
    controller: Portal2Controller, event_stream: EpisodeEventStream
) -> list[EpisodeEvent]:
    event_stream.feed(controller.read_console(print_to_terminal=False))
    events = []
    while True:
        event = event_stream.pop()
        if event is None:
            return events
        events.append(event)


def load_map_and_wait_for_ready(
    controller: Portal2Controller,
    event_stream: EpisodeEventStream,
    map_name: str,
    timeout: float,
) -> None:
    # Discard any terminal output queued by the previous map before waiting for
    # this map's chamber-ready event.
    controller.read_console(print_to_terminal=False)
    controller.event_buffer.clear()
    event_stream.clear()
    controller.send_command(f"map {map_name}")
    print(f"Loading map '{map_name}' and waiting for EVT|chamber_ready...")

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for event in read_events(controller, event_stream):
            if event.name == "chamber_ready":
                print("Chamber ready; recording starts now.")
                return
        time.sleep(0.05)
    raise RuntimeError(
        f"Map '{map_name}' did not emit EVT|chamber_ready within {timeout:g} seconds"
    )


def event_outcome(event: EpisodeEvent) -> str | None:
    if event.name == "goal_reached":
        return "goal_reached"
    if event.name == "episode_failed":
        reason = re.sub(r"[^a-zA-Z0-9_-]+", "_", event.value).strip("_")
        return reason or "episode_failed"
    return None


def wait_for_terminal_event(
    controller: Portal2Controller,
    event_stream: EpisodeEventStream,
    duration: float,
) -> str:
    deadline = time.monotonic() + duration
    while True:
        for event in read_events(controller, event_stream):
            outcome = event_outcome(event)
            if outcome:
                print(f"Terminal event received: {event.name}|{event.value}")
                return outcome

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            print(f"Local {duration:g}-second recording limit reached.")
            return "timeout"
        time.sleep(min(0.05, remaining))


@dataclass
class RecordingProgress:
    successes: int = 0
    attempts: int = 0


def record_episodes(
    recorder: EpisodeRecorder,
    controller: Portal2Controller,
    event_stream: EpisodeEventStream,
    args: argparse.Namespace,
    progress: RecordingProgress,
) -> None:
    while not args.episodes or progress.successes < args.episodes:
        load_map_and_wait_for_ready(
            controller,
            event_stream,
            args.map_name,
            args.ready_timeout,
        )
        # Discard Alt-Tab/menu/reset input accumulated while no episode was
        # active. Held movement keys remain held and are captured on tick 0.
        recorder.tracker.get_snapshot_and_reset()
        recorder.start_recording(map_name=args.map_name)
        outcome = wait_for_terminal_event(
            controller,
            event_stream,
            args.duration,
        )
        recorder.stop_recording(outcome)
        progress.attempts += 1
        if outcome == "goal_reached":
            progress.successes += 1

        target = f"/{args.episodes}" if args.episodes else ""
        attempt_word = "attempt" if progress.attempts == 1 else "attempts"
        print(
            f"Successful episodes: {progress.successes}{target} "
            f"({progress.attempts} {attempt_word}; last outcome: {outcome})."
        )
        if args.episodes and progress.successes >= args.episodes:
            print("Success target reached. Returning Portal 2 to the main menu.")
            controller.send_command("disconnect")
            time.sleep(0.5)
            return

        delay_unit = "second" if args.restart_delay == 1 else "seconds"
        print(f"Restarting the chamber in {args.restart_delay:g} {delay_unit}...")
        time.sleep(args.restart_delay)


def main() -> None:
    args = parse_args()
    validate_args(args)

    if os.geteuid() == 0 and os.environ.get("SUDO_USER"):
        raise RuntimeError(
            "Do not run this recorder with sudo. Root cannot normally access "
            "your Wayland capture session. Run ./install_dependencies.sh "
            "--permissions, log out and back in, then run it as your desktop user."
        )

    # Constructing the recorder validates input-device access before launching
    # a game that the process would be unable to record.
    recorder = EpisodeRecorder(
        fps=args.fps,
        width=args.width,
        height=args.height,
        video_crf=args.crf,
        controller=Portal2Controller(args.port),
        output_name=args.output,
        mouse_device=args.mouse_device,
    )
    controller = recorder.controller
    event_stream = EpisodeEventStream()

    if not is_game_running():
        if args.no_launch:
            raise RuntimeError("Portal 2 is not running")
        if not launch_game(args.port):
            raise RuntimeError("Portal 2 did not start")
    else:
        print("Portal 2 is already running.")

    tracker_started = False
    camera_started = False
    progress = RecordingProgress()
    try:
        connect_controller(controller)
        recorder.tracker.start()
        tracker_started = True
        recorder.camera.start()
        camera_started = True
        wait_for_camera(recorder)

        print(
            f"Capture ready: {args.width}x{args.height} at {args.fps} FPS, audio disabled."
        )
        print(f"Capture output: {recorder.camera.output_name or 'wf-recorder default'}")
        print(
            f"Alt-tab to Portal 2 now. The first map load starts in {args.focus_delay:g} seconds."
        )
        time.sleep(args.focus_delay)

        record_episodes(recorder, controller, event_stream, args, progress)
    except KeyboardInterrupt:
        print("\nStopping after Ctrl+C.")
        if recorder.recording:
            recorder.stop_recording("interrupted")
    finally:
        if recorder.recording:
            recorder.stop_recording("interrupted")
        if tracker_started:
            recorder.tracker.stop()
        if camera_started:
            recorder.camera.stop()
        controller.disconnect()
        print(
            f"Recorder stopped after {progress.successes} successful episode(s) "
            f"in {progress.attempts} completed attempt(s)."
        )


if __name__ == "__main__":
    main()
