"""Record a human correction from the current Portal 2 state without reloading."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from record_dataset import EpisodeEventStream, connect_controller, wait_for_camera, wait_for_terminal_event
from recorder import EpisodeRecorder
from wrapper import Portal2Controller, is_game_running


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="Chamber currently loaded in Portal 2")
    parser.add_argument("--source-attempt", type=Path,
                        help="Model attempt JSONL that produced the stuck state")
    parser.add_argument("--episodes-root", type=Path, default=Path("episodes/recovery"))
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--focus-delay", type=float, default=5.0)
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--mouse-device")
    args = parser.parse_args(argv)
    if args.max_seconds <= 0 or args.focus_delay < 0 or args.fps <= 0:
        parser.error("max-seconds and fps must be positive; focus-delay cannot be negative")
    if args.source_attempt and not args.source_attempt.is_file():
        parser.error(f"source attempt does not exist: {args.source_attempt}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not is_game_running():
        raise RuntimeError("Portal 2 must remain open at the model's stuck position")
    recorder = EpisodeRecorder(
        fps=args.fps, width=args.width, height=args.height, video_crf=args.crf,
        controller=Portal2Controller(args.port, log_file=None),
        output_name=args.output, mouse_device=args.mouse_device,
        episodes_root=args.episodes_root,
    )
    controller = recorder.controller
    tracker_started = camera_started = False
    try:
        connect_controller(controller)
        recorder.tracker.start()
        tracker_started = True
        recorder.camera.start()
        camera_started = True
        wait_for_camera(recorder)
        print(
            f"Portal 2 will stay on {args.map}; recording starts in {args.focus_delay:g}s. "
            "Focus the game, then demonstrate recovery. Ctrl+C stops recording."
        )
        time.sleep(args.focus_delay)
        controller.read_console(print_to_terminal=False)
        controller.event_buffer.clear()
        recorder.tracker.get_snapshot_and_reset()
        recorder.start_recording(
            map_name=args.map,
            metadata_extra={
                "source": "model_recovery",
                "source_attempt": str(args.source_attempt.resolve()) if args.source_attempt else None,
            },
        )
        outcome = wait_for_terminal_event(
            controller, EpisodeEventStream(), args.max_seconds
        )
        recorder.stop_recording(outcome)
    except KeyboardInterrupt:
        print("\nStopping recovery recording.")
    finally:
        if recorder.recording:
            recorder.stop_recording("interrupted")
        if tracker_started:
            recorder.tracker.stop()
        if camera_started:
            recorder.camera.stop()
        controller.disconnect()


if __name__ == "__main__":
    main()
