"""Continuously record human demonstrations from a Portal 2 chamber.

The recorder is intentionally video-only. ``WaylandCamera`` captures raw video
frames and ``FFmpegVideoWriter`` passes ``-an`` to FFmpeg, so episode MP4 files
never contain an audio stream.
"""

from __future__ import annotations

# Python standard library: CLI, event timing, and recording metadata.
import argparse
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

# Project modules at the repository root: capture and Portal 2 control.
from episode_events import EpisodeEvent, EpisodeEventStream
from recorder import EpisodeRecorder
from wrapper import Portal2Controller, is_game_running, launch_game


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuously record 24 FPS/FHD Portal 2 demonstration episodes."
    )
    parser.add_argument(
        "--map", dest="map_names", action="append", metavar="NAME",
        help="Map to record. Repeat for round-robin recording; defaults to dataset_test1.",
    )
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--focus-delay", type=float, default=5.0)
    parser.add_argument("--restart-delay", type=float, default=1.0)
    parser.add_argument("--ready-timeout", type=float, default=60.0)
    parser.add_argument(
        "--episodes", "--episode",
        type=int,
        default=0,
        help="Successful episodes per map; failures retry the same map. 0 cycles until K or Ctrl+C.",
    )
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument(
        "--episodes-root", type=Path, default=None,
        help="Recording root; proactive-turn mode defaults to its dedicated folder.",
    )
    parser.add_argument(
        "--proactive-turn", action="store_true",
        help=(
            "Record a training-only turn correction from a manually selected "
            "pre-wall position; press physical P to begin each clip."
        ),
    )
    parser.add_argument(
        "--turn-direction", choices=("left", "right"),
        help="Required with --proactive-turn; saved in episode metadata.",
    )
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
    parser.add_argument('--oracle-source',action='store_true',help='One fresh diagnostic demonstration with pose/map fingerprint; excluded from training')
    parser.add_argument('--oracle-map-file',type=Path)
    parser.add_argument('--oracle-checkpoint',type=Path,default=Path('models/imitation/checkpoints/runs/v5_recovery_proactive_v3/best.pt'))
    args = parser.parse_args(argv)
    args.map_names = args.map_names or ["dataset_test1"]
    # Keep proactive examples separate by default while retaining the existing
    # normal-recording path and honoring an explicitly supplied root.
    if args.episodes_root is None:
        args.episodes_root = Path('model_attempts/expert_oracle_20261112/sources_pending') if args.oracle_source else Path(
            "episodes/corrections/proactive_turns"
            if args.proactive_turn else "episodes"
        )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if getattr(args,'oracle_source',False):
        if args.episodes_root.resolve().is_relative_to(Path('episodes').resolve()) or args.episodes_root.resolve().is_relative_to(Path('data/datasets').resolve()):raise ValueError('Oracle sources must stay outside ordinary training roots')
        if args.proactive_turn or args.episodes!=1 or len(args.map_names)!=1 or args.map_names[0]!='dataset_test1' or args.fps!=24 or not args.no_launch or not args.output or not 0<args.duration<=55 or not 0<=args.focus_delay<=10 or not 0<args.ready_timeout<=60:
            raise ValueError('Oracle source requires one test1 episode,24FPS, no-launch, duration at most55, without proactive mode')
        if args.oracle_map_file is None or not args.oracle_map_file.is_file() or args.oracle_map_file.stem!=args.map_names[0] or args.oracle_map_file.suffix.lower()!='.bsp':
            raise ValueError('Provide the actual installed dataset_test1.bsp with --oracle-map-file')
        if not args.oracle_checkpoint.is_file() or not args.oracle_checkpoint.with_suffix('.json').is_file():raise ValueError('Oracle bin reference missing')
    if len(set(args.map_names)) != len(args.map_names):
        raise ValueError("--map names must be unique")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in args.map_names):
        raise ValueError("--map names must contain only letters, digits, underscores, or hyphens")
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
    if args.proactive_turn:
        if args.turn_direction not in ("left", "right"):
            raise ValueError("--proactive-turn requires --turn-direction left or right")
        training_maps = {f"dataset_test{number}" for number in range(1, 11)}
        outside_training = sorted(set(args.map_names) - training_maps)
        if outside_training:
            raise ValueError(
                "proactive-turn training recordings must use dataset_test1 through "
                f"dataset_test10; got {', '.join(outside_training)}"
            )
    elif args.turn_direction is not None:
        raise ValueError("--turn-direction can only be used with --proactive-turn")


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
        camera_error = getattr(recorder.camera, "capture_error", None)
        if isinstance(camera_error, str) and camera_error:
            raise RuntimeError(
                f"Camera capture failed before the first frame: {camera_error}"
            )
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
                print("Chamber ready.")
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
    recorder: EpisodeRecorder | None = None,
    reject_new_chamber: bool = False,
    require_focus: bool = False,
    requested_waypoint: str | None = None,
    waypoint_events: list | None = None,
) -> str:
    if requested_waypoint is not None and requested_waypoint not in ("A", "B"):
        raise ValueError("requested_waypoint must be A or B")
    remaining_duration = duration if duration > 0 else None
    previous_tick = time.monotonic()
    while True:
        tracker = getattr(recorder, "tracker", None) if recorder is not None else None
        tracker_errors = tracker.capture_errors() if tracker is not None and hasattr(tracker, "capture_errors") else ()
        if isinstance(tracker_errors, (tuple, list)) and tracker_errors:
            return "capture_error"
        capture_failure = getattr(recorder, "capture_failure_reason", None) if recorder is not None else None
        if isinstance(capture_failure, str) and capture_failure:
            return "capture_error"
        if require_focus:
            from run_imitation import query_hyprland_active_window
            if not (query_hyprland_active_window('auto') or {}).get('portal_focused'):
                return 'focus_lost'
        now = time.monotonic()
        if remaining_duration is not None and (recorder is None or not recorder.paused):
            remaining_duration -= now - previous_tick
        previous_tick = now

        if recorder is not None:
            for control in recorder.tracker.pop_control_events():
                if control == "stop":
                    print("K pressed; ending recording session.")
                    return "interrupted"
                if control == "restart":
                    print("R pressed; restarting this chamber's model attempt.")
                    return "restart_requested"
                if control == "pause":
                    if require_focus:return "oracle_paused"
                    recorder.toggle_pause()

        for event in read_events(controller, event_stream):
            if reject_new_chamber and event.name == "chamber_ready":
                print("A new chamber-ready event arrived during recovery; rejecting this clip.")
                return "map_reloaded"
            if requested_waypoint is not None:
                if waypoint_events is not None:
                    waypoint_events.append({"name": event.name, "value": event.value,
                                            "received_monotonic": time.monotonic()})
                if event.name == "waypoint_reached":
                    if event.value not in ("A", "B"):
                        return "invalid_waypoint_event"
                    return "waypoint_reached" if event.value == requested_waypoint else "wrong_waypoint"
                if event.name == "goal_reached":
                    return "wrong_destination"
            outcome = event_outcome(event)
            if outcome:
                print(f"Terminal event received: {event.name}|{event.value}")
                return outcome

        if recorder is not None and recorder.paused:
            time.sleep(0.05)
            continue
        if remaining_duration is not None and remaining_duration <= 0:
            print(f"Local {duration:g}-second recording limit reached.")
            return "timeout"
        time.sleep(min(0.05, remaining_duration) if remaining_duration is not None else 0.05)


def wait_for_proactive_start(
    recorder: EpisodeRecorder,
    controller: Portal2Controller,
    event_stream: EpisodeEventStream,
) -> str:
    """Wait for the operator to position before the turn and press physical P."""
    print(
        "Position Portal 2 at the chosen pre-wall turn approach. "
        "No data is being recorded yet; press P to start the proactive-turn clip."
    )
    while True:
        capture_errors = recorder.tracker.capture_errors()
        if isinstance(capture_errors, (tuple, list)) and capture_errors:
            print("Input capture failed before recording: " + "; ".join(capture_errors))
            return "capture_error"
        for control in recorder.tracker.pop_control_events():
            if control == "start":
                return "start"
            if control == "stop":
                print("K pressed before recording; ending session without a clip.")
                return "interrupted"
            if control == "pause":
                print("Esc has no effect before a proactive-turn clip starts.")

        for event in read_events(controller, event_stream):
            if event.name == "chamber_ready":
                print("Chamber reloaded before recording; retrying from a fresh chamber.")
                return "restart_requested"
            outcome = event_outcome(event)
            if outcome:
                print(
                    f"Chamber ended before recording ({outcome}); "
                    "no clip was written."
                )
                return "restart_requested"
        time.sleep(0.05)


@dataclass
class RecordingProgress:
    successes: int = 0
    attempts: int = 0
    successes_by_map: dict[str, int] = field(default_factory=dict)


def record_episodes(
    recorder: EpisodeRecorder,
    controller: Portal2Controller,
    event_stream: EpisodeEventStream,
    args: argparse.Namespace,
    progress: RecordingProgress,
) -> None:
    progress.successes_by_map = {name: 0 for name in args.map_names}
    map_index = 0
    while True:
        map_name = args.map_names[map_index]
        load_map_and_wait_for_ready(
            controller,
            event_stream,
            map_name,
            args.ready_timeout,
        )
        if getattr(args, "proactive_turn", False):
            start_result = wait_for_proactive_start(
                recorder, controller, event_stream
            )
            if start_result in ("interrupted", "capture_error"):
                return
            if start_result != "start":
                time.sleep(args.restart_delay)
                continue

        # Discard menu, positioning, and start-key input accumulated while no
        # episode was active. Held movement keys remain captured on tick 0.
        recorder.tracker.get_snapshot_and_reset()
        metadata_extra = None
        if getattr(args, "proactive_turn", False):
            metadata_extra = {
                "recording_mode": "proactive_turn",
                "turn_direction": args.turn_direction,
            }
        if getattr(args,'oracle_source',False):
            from expert_action_replay import source_metadata,read_pose,verify_pose,wait_idle_frames
            metadata_extra=source_metadata(controller,recorder.tracker,map_name,args.oracle_map_file,args.oracle_checkpoint,recorder.camera)
        recorder.start_recording(map_name=map_name, metadata_extra=metadata_extra)
        if getattr(args,'oracle_source',False):
            print('ORACLE WAIT: keep all controls released for 3 seconds.',flush=True)
            time.sleep(3.)
            wait_idle_frames(recorder)
            verify_pose(metadata_extra['oracle']['initial_pose'],read_pose(controller))
            print('ORACLE GO: demonstrate the route through the exit. Do not pause.',flush=True)
        oracle_options={'reject_new_chamber':True,'require_focus':True} if getattr(args,'oracle_source',False) else {}
        outcome = wait_for_terminal_event(
            controller,
            event_stream,
            args.duration,
            recorder=recorder,
            **oracle_options,
        )
        terminal_outcome = outcome
        finalized_outcome = recorder.stop_recording(outcome)
        if finalized_outcome == "invalid":
            outcome = finalized_outcome
        if getattr(args,'oracle_source',False):
            if finalized_outcome == "invalid":
                progress.attempts += 1
                print("Oracle recording invalidated; it was not finalized as a source episode.")
                return
            from expert_action_replay import finalize_source
            finalize_source(recorder.ep_dir,outcome)
            progress.attempts+=1
            progress.successes+=int(outcome=='goal_reached')
            print(f'Oracle source saved for review: {recorder.ep_dir}',flush=True)
            return
        progress.attempts += 1
        if terminal_outcome == "interrupted" or finalized_outcome == "invalid":
            return
        if outcome == "goal_reached":
            progress.successes += 1
            progress.successes_by_map[map_name] += 1
            map_index = (map_index + 1) % len(args.map_names)

        map_target = f"/{args.episodes}" if args.episodes else ""
        total_target = f"/{args.episodes * len(args.map_names)}" if args.episodes else ""
        attempt_word = "attempt" if progress.attempts == 1 else "attempts"
        print(
            f"{map_name}: {progress.successes_by_map[map_name]}{map_target} successful; "
            f"overall: {progress.successes}{total_target} "
            f"({progress.attempts} {attempt_word}; last outcome: {outcome})."
        )
        if args.episodes and all(
            count >= args.episodes for count in progress.successes_by_map.values()
        ):
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
        episodes_root=args.episodes_root,
        start_enabled=args.proactive_turn,
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
        print("During recording: Esc pauses/resumes; K ends and saves the partial episode. Ctrl+C also stops.")
        if args.proactive_turn:
            print("Proactive-turn mode: position before the turn, then press physical P to begin capture.")
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
