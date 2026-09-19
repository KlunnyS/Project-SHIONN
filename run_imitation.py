"""Run a trained SHIONN imitation policy against a live Portal 2 window."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import evdev
import numpy as np

from recorder import FFmpegVideoWriter, WaylandCamera, get_default_output
from wrapper import Portal2Controller, is_game_running, launch_game


TERMINAL_EVENTS = ("EVT|goal_reached", "EVT|episode_failed")
PORTAL_WINDOW_CLASS = "steam_app_620"
EPISODE_FAILURE_PATTERN = re.compile(r"EVT\|episode_failed\|([^\r\n]*)")


def classify_terminal_events(
    console_output: str, max_seconds: float
) -> tuple[bool, bool]:
    """Return (terminal event received, chamber timeout ignored)."""
    if "EVT|goal_reached" in console_output:
        return True, False

    has_failure_marker = "EVT|episode_failed" in console_output
    if not has_failure_marker:
        return False, False

    failure_reasons = [
        match.group(1).strip() for match in EPISODE_FAILURE_PATTERN.finditer(console_output)
    ]
    only_timeout = bool(failure_reasons) and all(
        reason == "timeout" for reason in failure_reasons
    )
    if max_seconds > 0 and only_timeout:
        return False, True
    return True, False


class EscapeKeyMonitor:
    """Watch readable physical keyboards for a global Escape key press."""

    def __init__(self) -> None:
        self._stop_requested = threading.Event()
        self._stopping = threading.Event()
        self._devices: list[evdev.InputDevice] = []
        self._threads: list[threading.Thread] = []

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    def wait(self, timeout: float) -> bool:
        return self._stop_requested.wait(timeout)

    def _watch_device(self, device: evdev.InputDevice) -> None:
        try:
            for event in device.read_loop():
                if self._stopping.is_set():
                    return
                if (
                    event.type == evdev.ecodes.EV_KEY
                    and event.code == evdev.ecodes.KEY_ESC
                    and event.value == 1
                ):
                    self._stop_requested.set()
                    return
        except OSError:
            if not self._stopping.is_set():
                print(f"Warning: Escape-key monitor lost {device.path} ({device.name})")

    def start(self) -> None:
        for path in evdev.list_devices():
            try:
                device = evdev.InputDevice(path)
                keys = device.capabilities().get(evdev.ecodes.EV_KEY, [])
            except OSError:
                continue
            name = device.name.casefold()
            if (
                evdev.ecodes.KEY_ESC not in keys
                or evdev.ecodes.KEY_W not in keys
                or evdev.ecodes.KEY_A not in keys
                or "virtual" in name
                or "uinput" in name
                or "portal2-wrapper" in name
            ):
                device.close()
                continue
            self._devices.append(device)

        if not self._devices:
            raise RuntimeError(
                "No readable physical keyboard with an Escape key was found; "
                "check /dev/input/event* permissions"
            )

        for device in self._devices:
            thread = threading.Thread(
                target=self._watch_device,
                args=(device,),
                name=f"escape-monitor-{Path(device.path).name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)
        device_names = ", ".join(device.name for device in self._devices)
        print(f"Escape-key stop enabled: {device_names}")

    def stop(self) -> None:
        self._stopping.set()
        for device in self._devices:
            device.close()
        for thread in self._threads:
            thread.join(timeout=0.5)
        self._devices.clear()
        self._threads.clear()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    parser.add_argument("--record-video", action="store_true", help="Save the frames seen by the policy as an MP4")
    parser.add_argument("--recording-dir", type=Path, default=Path("model_attempts"), help="Directory for --record-video output")
    parser.add_argument("--log-actions", action="store_true", help="Save structured per-tick diagnostics without requiring video recording")
    parser.add_argument("--log-file", type=Path, help="Custom JSONL diagnostics path; implies --log-actions")
    parser.add_argument("--verbose", action="store_true", help="Print a concise runtime status line once per second")
    parser.add_argument("--status-every", type=float, default=1.0, help="Seconds between --verbose status lines and focus checks")
    parser.add_argument("--keep-focused", action="store_true", help="On Hyprland, focus and activate Portal input, restoring it if focus is lost")
    parser.add_argument("--hyprland-instance", default="auto", help="Hyprland instance signature; auto discovers it for SSH sessions")
    return parser.parse_args(argv)


def make_attempt_recording_path(
    recording_dir: Path, now: datetime | None = None
) -> Path:
    """Create the separate model-attempt directory and return a unique MP4 path."""
    recording_dir.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return recording_dir / f"attempt_{timestamp}.mp4"


class AttemptLog:
    """Line-buffered JSONL diagnostics that remain useful after interruption."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8", buffering=1)

    def write(self, record_type: str, **values) -> None:
        record = {
            "type": record_type,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            **values,
        }
        self._handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._handle.close()


def hyprland_instance_candidates(instance: str = "auto") -> list[str]:
    """Find usable Hyprland signatures even when SSH omitted the session env."""
    if instance and instance != "auto":
        return [instance]
    candidates = []
    environment_instance = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    if environment_instance:
        candidates.append(environment_instance)
    runtime_dir = Path(os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"))
    socket_root = runtime_dir / "hypr"
    try:
        discovered = sorted(
            (path for path in socket_root.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        discovered = []
    candidates.extend(path.name for path in discovered if path.name not in candidates)
    return candidates


def query_hyprland_active_window(instance: str = "auto") -> dict | None:
    """Return a small active-window snapshot, or None outside Hyprland."""
    if shutil.which("hyprctl") is None:
        return None
    for candidate in hyprland_instance_candidates(instance):
        try:
            result = subprocess.run(
                ["hyprctl", "-i", candidate, "activewindow", "-j"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            window = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            continue
        return {
            "class": window.get("class"),
            "title": window.get("title"),
            "monitor": window.get("monitor"),
            "fullscreen": window.get("fullscreen"),
            "at": window.get("at"),
            "size": window.get("size"),
            "portal_focused": window.get("class") == PORTAL_WINDOW_CLASS,
            "hyprland_instance": candidate,
        }
    return None


def focus_portal_window(instance: str = "auto") -> dict | None:
    """Ask Hyprland to focus Portal and return the resulting active window."""
    if shutil.which("hyprctl") is None:
        return None
    for candidate in hyprland_instance_candidates(instance):
        try:
            subprocess.run(
                [
                    "hyprctl", "-i", candidate, "dispatch", "focuswindow",
                    f"class:^({PORTAL_WINDOW_CLASS})$",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            time.sleep(0.1)
        except (OSError, subprocess.SubprocessError):
            continue
        status = query_hyprland_active_window(candidate)
        if status is not None:
            return status
    return None


def activate_portal_input(
    controller: Portal2Controller, instance: str = "auto"
) -> dict | None:
    """Focus Portal, click inside it to acquire XWayland input, and unpause."""
    status = focus_portal_window(instance)
    if status is None:
        return None
    position = status.get("at")
    size = status.get("size")
    candidate = status.get("hyprland_instance")
    if not (
        isinstance(position, list) and len(position) == 2
        and isinstance(size, list) and len(size) == 2
        and candidate
    ):
        return None
    center_x = int(position[0] + size[0] / 2)
    center_y = int(position[1] + size[1] / 2)
    try:
        subprocess.run(
            ["hyprctl", "-i", candidate, "dispatch", "movecursor", str(center_x), str(center_y)],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if not controller.click_virtual_mouse("left"):
        return None
    controller.send_command("unpause")
    time.sleep(0.1)
    return query_hyprland_active_window(candidate)


def frame_change(previous_sample: np.ndarray | None, frame: np.ndarray) -> tuple[float | None, np.ndarray]:
    """Cheap visual-motion metric over a sparse BGR sample."""
    sample = frame[::16, ::16].astype(np.int16)
    change = None if previous_sample is None else float(np.abs(sample - previous_sample).mean())
    return change, sample


def action_label(action: dict[str, int]) -> str:
    active = [name for name in (
        "move_w", "move_a", "move_s", "move_d", "jump", "use",
        "fire_left", "fire_right",
    ) if action.get(name)]
    return "+".join(active) if active else "idle"


def apply_predicted_action(
    controller: Portal2Controller,
    action: dict[str, int],
    *,
    dry_run: bool,
    verbose: bool,
) -> bool:
    """Apply an action unless dry-run is active; return whether it was sent."""
    if dry_run:
        if not verbose:
            print(action)
        return False
    controller.apply_action(action)
    return True


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
    if args.status_every <= 0:
        raise ValueError("--status-every must be greater than zero")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    policy = PolicyInference(args.checkpoint, device=args.device)

    if not is_game_running():
        if args.no_launch:
            raise RuntimeError("Portal 2 is not running")
        if not launch_game(args.port):
            raise RuntimeError("Portal 2 did not start")

    controller = connect_controller(args.port)
    # Per-frame mouse prints obscure the runner's status output. Every predicted
    # movement is still available in the optional JSONL diagnostics.
    controller.trace_mouse_moves = False
    camera = WaylandCamera(
        width=args.width,
        height=args.height,
        fps=max(1, round(args.fps)),
        output_name=args.output or get_default_output(),
    )
    escape_monitor = EscapeKeyMonitor()
    video_writer = None
    video_path = None
    attempt_log = None
    log_path = None
    tick_count = 0
    focus_loss_count = 0
    visual_change_total = 0.0
    visual_change_samples = 0
    previous_frame_sample = None
    input_ready = None
    stop_reason = "time_limit"

    try:
        escape_monitor.start()
        if args.map_name:
            controller.load_map(args.map_name, wait_for_load=True)
        camera.start()
        print(f"Policy device: {policy.device}")
        print(f"Capture output: {camera.output_name or 'wf-recorder default'}")
        print(
            f"Starting in {args.countdown:g} seconds. "
            "Press Esc or Ctrl+C for the emergency stop."
        )
        if escape_monitor.wait(max(0.0, args.countdown)):
            print("\nEscape pressed. Cancelling policy run.")
            return

        frame_wait_deadline = time.monotonic() + 10.0
        while camera.get_latest_frame() is None:
            if escape_monitor.stop_requested:
                print("\nEscape pressed. Cancelling policy run.")
                return
            if time.monotonic() >= frame_wait_deadline:
                detail = camera.error_summary()
                message = "No frames received from wf-recorder; check the output name and Wayland permissions"
                if detail:
                    message += f"\nwf-recorder output:\n{detail}"
                raise RuntimeError(message)
            time.sleep(0.05)

        attempt_time = datetime.now()
        if args.record_video:
            video_path = make_attempt_recording_path(args.recording_dir, attempt_time)
            video_writer = FFmpegVideoWriter(
                str(video_path), args.width, args.height, max(1, round(args.fps))
            )
            print(f"Recording model attempt to: {video_path}")

        if args.log_file:
            log_path = args.log_file
        elif args.record_video:
            log_path = video_path.with_suffix(".jsonl")
        elif args.log_actions:
            log_path = make_attempt_recording_path(
                args.recording_dir, attempt_time
            ).with_suffix(".jsonl")
        if log_path is not None:
            attempt_log = AttemptLog(log_path)
            attempt_log.write(
                "metadata",
                checkpoint=str(args.checkpoint),
                device=str(policy.device),
                map=args.map_name,
                capture={
                    "output": camera.output_name,
                    "width": args.width,
                    "height": args.height,
                    "fps": args.fps,
                },
                video=str(video_path) if video_path else None,
                dry_run=args.dry_run,
                keep_focused=args.keep_focused,
            )
            print(f"Writing attempt diagnostics to: {log_path}")

        focus_status = query_hyprland_active_window(args.hyprland_instance)
        if args.keep_focused and not args.dry_run:
            focus_status = activate_portal_input(controller, args.hyprland_instance)
            input_ready = bool((focus_status or {}).get("portal_focused"))
        elif args.keep_focused and not (focus_status or {}).get("portal_focused"):
            focus_status = focus_portal_window(args.hyprland_instance)
        if args.keep_focused and not (focus_status or {}).get("portal_focused"):
            raise RuntimeError(
                "Could not focus Portal 2 through Hyprland; refusing to inject global mouse input"
            )
        if attempt_log is not None:
            attempt_log.write(
                "focus", event="initial", window=focus_status,
                input_activated=input_ready,
            )

        started = time.monotonic()
        next_tick = started
        next_status = started
        tick_duration = 1.0 / args.fps
        while not args.max_seconds or time.monotonic() - started < args.max_seconds:
            if escape_monitor.stop_requested:
                print("\nEscape pressed. Stopping policy run.")
                stop_reason = "escape_key"
                if attempt_log is not None:
                    attempt_log.write("stop", reason="escape_key")
                break
            frame = camera.get_latest_frame()
            action = None
            action_applied = False
            diagnostics = None
            change = None
            if frame is not None:
                change, previous_frame_sample = frame_change(previous_frame_sample, frame)
                if change is not None:
                    visual_change_total += change
                    visual_change_samples += 1
                if video_writer is not None:
                    video_writer.write(frame)
                action, diagnostics = policy.predict_with_diagnostics(frame)
                action_applied = apply_predicted_action(
                    controller,
                    action,
                    dry_run=args.dry_run,
                    verbose=args.verbose,
                )
                tick_count += 1

            console_output = controller.read_console(print_to_terminal=False)
            console_events = [
                event for event in ("EVT|chamber_ready", *TERMINAL_EVENTS)
                if event in console_output
            ]
            terminal_received, ignored_chamber_timeout = classify_terminal_events(
                console_output, args.max_seconds
            )
            if "EVT|chamber_ready" in console_output:
                policy.reset()
                controller.release_policy_actions()
            if ignored_chamber_timeout:
                print(
                    "Chamber timeout event ignored; "
                    f"the runner will stop at {args.max_seconds:g} seconds."
                )
            if terminal_received:
                policy.reset()
                controller.release_policy_actions()
                if "EVT|goal_reached" in console_output:
                    terminal_outcome = "goal_reached"
                    failure_reason = None
                else:
                    terminal_outcome = "episode_failed"
                    match = EPISODE_FAILURE_PATTERN.search(console_output)
                    failure_reason = match.group(1).strip() if match else None
                if attempt_log is not None:
                    attempt_log.write(
                        "terminal", outcome=terminal_outcome,
                        reason=failure_reason,
                    )
                print("Episode terminal event received.")
                if not args.continue_after_event:
                    stop_reason = terminal_outcome
                    break

            now = time.monotonic()
            if now >= next_status:
                focus_status = query_hyprland_active_window(args.hyprland_instance)
                if args.keep_focused and not (focus_status or {}).get("portal_focused"):
                    focus_loss_count += 1
                    controller.release_policy_actions()
                    policy.reset()
                    focus_status = (
                        focus_portal_window(args.hyprland_instance)
                        if args.dry_run
                        else activate_portal_input(controller, args.hyprland_instance)
                    )
                    input_ready = not args.dry_run and bool(
                        (focus_status or {}).get("portal_focused")
                    )
                    if not (focus_status or {}).get("portal_focused"):
                        raise RuntimeError("Portal 2 lost focus and could not be refocused")
                if args.verbose and action is not None:
                    focus_text = (
                        "unknown" if focus_status is None
                        else "portal" if focus_status["portal_focused"]
                        else f"{focus_status['class']}"
                    )
                    print(
                        f"status t={now - started:6.1f}s ticks={tick_count} "
                        f"focus={focus_text} input={'ready' if input_ready else 'n/a'} "
                        f"visual_delta={change or 0.0:.3f} "
                        f"action={action_label(action)} "
                        f"mouse=({action['mouse_dx']},{action['mouse_dy']})"
                    )
                next_status = now + args.status_every

            if attempt_log is not None and action is not None:
                attempt_log.write(
                    "tick",
                    tick=tick_count,
                    elapsed_seconds=now - started,
                    action=action,
                    policy=diagnostics,
                    action_applied=action_applied,
                    visual_delta=change,
                    focus=focus_status,
                    input_ready=input_ready,
                    console_events=console_events,
                )

            next_tick += tick_duration
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\nEmergency stop requested.")
        stop_reason = "keyboard_interrupt"
        if attempt_log is not None:
            attempt_log.write("stop", reason="keyboard_interrupt")
    except Exception as error:
        stop_reason = "error"
        if attempt_log is not None:
            attempt_log.write("error", error_type=type(error).__name__, message=str(error))
        raise
    finally:
        controller.release_policy_actions()
        escape_monitor.stop()
        if video_writer is not None:
            try:
                video_writer.release()
                print(f"Model attempt saved to: {video_path}")
            except Exception as error:
                print(f"Warning: could not finalize model attempt video: {error}")
        if attempt_log is not None:
            attempt_log.write(
                "summary",
                ticks=tick_count,
                focus_losses=focus_loss_count,
                input_ready=input_ready,
                mean_visual_delta=(
                    visual_change_total / visual_change_samples
                    if visual_change_samples else None
                ),
                stop_reason=stop_reason,
            )
            attempt_log.close()
            print(f"Attempt diagnostics saved to: {log_path}")
        camera.stop()
        controller.disconnect()
        print("Policy stopped and held actions released.")


if __name__ == "__main__":
    main()
