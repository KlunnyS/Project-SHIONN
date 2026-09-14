"""Run a trained SHIONN imitation policy against a live Portal 2 window."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from recorder import WaylandCamera, get_default_output
from wrapper import Portal2Controller, is_game_running, launch_game


TERMINAL_EVENTS = ("EVT|goal_reached", "EVT|episode_failed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a trained imitation checkpoint in Portal 2.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/imitation/checkpoints/best.pt"))
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--output", help="Wayland output name from `wf-recorder -L`; defaults to the first output")
    parser.add_argument("--map", dest="map_name", help="Optional map to load after connecting, for example puzzlemaker/preview")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Stop automatically; use 0 to run until Ctrl+C")
    parser.add_argument("--dry-run", action="store_true", help="Print predictions without sending actions to Portal 2")
    parser.add_argument("--no-launch", action="store_true", help="Fail instead of launching Portal 2 when it is not running")
    parser.add_argument("--continue-after-event", action="store_true", help="Keep running after goal/failure events")
    return parser.parse_args()


def connect_controller(port: int, attempts: int = 10, delay: float = 2.0) -> Portal2Controller:
    controller = Portal2Controller(port)
    for attempt in range(1, attempts + 1):
        if controller.connect():
            time.sleep(0.5)
            controller.read_console(print_to_terminal=False)
            return controller
        if attempt < attempts:
            print(f"Retrying console connection ({attempt}/{attempts})...")
            time.sleep(delay)
    raise RuntimeError(f"Could not connect to Portal 2 on netcon port {port}")


def main() -> None:
    args = parse_args()
    from models.imitation.inference import PolicyInference

    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.max_seconds < 0:
        raise ValueError("--max-seconds must be zero or greater")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    policy = PolicyInference(args.checkpoint, device=args.device)

    if not is_game_running():
        if args.no_launch:
            raise RuntimeError("Portal 2 is not running")
        if not launch_game(args.port):
            raise RuntimeError("Portal 2 did not start")

    controller = connect_controller(args.port)
    camera = WaylandCamera(
        width=args.width,
        height=args.height,
        fps=max(1, round(args.fps)),
        output_name=args.output or get_default_output(),
    )

    try:
        if args.map_name:
            controller.load_map(args.map_name, wait_for_load=True)
        camera.start()
        print(f"Policy device: {policy.device}")
        print(f"Capture output: {camera.output_name or 'wf-recorder default'}")
        print(f"Starting in {args.countdown:g} seconds. Press Ctrl+C for the emergency stop.")
        time.sleep(max(0.0, args.countdown))

        frame_wait_deadline = time.monotonic() + 10.0
        while camera.get_latest_frame() is None:
            if time.monotonic() >= frame_wait_deadline:
                raise RuntimeError("No frames received from wf-recorder; check the output name and Wayland permissions")
            time.sleep(0.05)

        started = time.monotonic()
        next_tick = started
        tick_duration = 1.0 / args.fps
        while not args.max_seconds or time.monotonic() - started < args.max_seconds:
            frame = camera.get_latest_frame()
            if frame is not None:
                action = policy.predict(frame)
                if args.dry_run:
                    print(action)
                else:
                    controller.apply_action(action)

            console_output = controller.read_console(print_to_terminal=False)
            if "EVT|chamber_ready" in console_output:
                policy.reset()
                controller.release_policy_actions()
            if any(event in console_output for event in TERMINAL_EVENTS):
                policy.reset()
                controller.release_policy_actions()
                print("Episode terminal event received.")
                if not args.continue_after_event:
                    break

            next_tick += tick_duration
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\nEmergency stop requested.")
    finally:
        controller.release_policy_actions()
        camera.stop()
        controller.disconnect()
        print("Policy stopped and held actions released.")


if __name__ == "__main__":
    main()
