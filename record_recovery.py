"""Record a human correction from the current Portal 2 state without reloading."""

from __future__ import annotations

# Python standard library: CLI paths and recording timing.
import argparse
import json
import time
from pathlib import Path

# Project modules at the repository root: reuse recorder events and game control.
from record_dataset import EpisodeEventStream, connect_controller, wait_for_camera, wait_for_terminal_event
from recorder import EpisodeRecorder
from run_imitation import (
    STARTUP_PITCH_TOLERANCE, activate_portal_input, disable_chamber_timeout,
    query_hyprland_active_window, query_hyprland_cursor, read_player_angles,
    read_portal_binding, restore_portal_binding,
)
from wrapper import Portal2Controller, is_game_running


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="Chamber currently loaded in Portal 2")
    parser.add_argument("--source-attempt", type=Path,
                        help="Model attempt JSONL that produced the stuck state")
    parser.add_argument("--result-file", type=Path,
                        help="Write the completed outcome and episode path for a session runner")
    parser.add_argument("--episodes-root", type=Path, default=Path("episodes/recovery"))
    parser.add_argument("--max-seconds", type=float, default=30.0)
    parser.add_argument("--keep-focused", action="store_true",
                        help="Reacquire Portal 2 focus and mouse capture before recording")
    parser.add_argument("--disable-chamber-timeout", action="store_true",
                        help="Require timeout suppression in the current chamber")
    parser.add_argument("--session-controls", action="store_true",
                        help="Enable R to discard this correction and restart its model attempt")
    parser.add_argument("--manual-focus", action="store_true",
                        help="Skip automatic mouse checks and wait for P before recording")
    parser.add_argument("--test-mode", action="store_true",
                        help="Exercise takeover and completion without saving a recovery episode")
    parser.add_argument("--hyprland-instance", default="auto")
    parser.add_argument("--focus-delay", type=float, default=5.0)
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--output")
    parser.add_argument("--mouse-device")
    args = parser.parse_args(argv)
    if args.max_seconds < 0 or args.focus_delay < 0 or args.fps <= 0:
        parser.error("max-seconds cannot be negative (0 disables it); fps must be positive; focus-delay cannot be negative")
    if args.keep_focused and args.focus_delay < 2:
        parser.error("--keep-focused requires at least 2 seconds of focus delay to verify the physical mouse")
    if args.source_attempt and not args.source_attempt.is_file():
        parser.error(f"source attempt does not exist: {args.source_attempt}")
    if args.manual_focus and not args.session_controls:
        parser.error("--manual-focus requires --session-controls")
    return args


def angle_delta(current: float, previous: float) -> float:
    """Shortest signed rotation between two Source-engine view angles."""
    return (current - previous + 180.0) % 360.0 - 180.0


class HandoffDiagnostics:
    """Compare physical mouse motion with game yaw before accepting a clip."""

    def __init__(self, tracker, instance: str, path: Path | None = None) -> None:
        self.tracker = tracker
        self.instance = instance
        self.path = path
        self.previous_mouse = tracker.mouse_counters_snapshot()
        self.previous_angles = None
        self.raw_by_direction = {"left": 0, "right": 0}
        self.yaw_by_direction = {"left": 0.0, "right": 0.0}

    def write(self, data: dict) -> None:
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(data) + "\n")

    def sample(self, angles: tuple[float, float, float]) -> None:
        current_mouse = self.tracker.mouse_counters_snapshot()
        left = max(0, current_mouse["left"] - self.previous_mouse["left"])
        right = max(0, current_mouse["right"] - self.previous_mouse["right"])
        yaw = (abs(angle_delta(angles[1], self.previous_angles[1]))
               if self.previous_angles is not None else 0.0)
        horizontal = left + right
        if horizontal:
            self.raw_by_direction["left"] += left
            self.raw_by_direction["right"] += right
            self.yaw_by_direction["left"] += yaw * left / horizontal
            self.yaw_by_direction["right"] += yaw * right / horizontal
        self.previous_mouse = current_mouse
        self.previous_angles = angles
        status = query_hyprland_active_window(self.instance)
        resolved_instance = (status or {}).get("hyprland_instance", self.instance)
        cursor = query_hyprland_cursor(resolved_instance) if resolved_instance != "auto" else None
        self.write({
            "type": "sample", "time": time.monotonic(),
            "physical_mouse": current_mouse, "angles": angles,
            "cursor": cursor, "portal_focused": (status or {}).get("portal_focused"),
        })

    def weak_directions(self) -> list[str]:
        """Flag clear failures, allowing small exploratory mouse movements."""
        return [
            direction for direction in ("left", "right")
            if self.raw_by_direction[direction] >= 500
            and self.yaw_by_direction[direction] / self.raw_by_direction[direction] < 0.01
        ]

    def reset_response(self) -> None:
        self.raw_by_direction = {"left": 0, "right": 0}
        self.yaw_by_direction = {"left": 0.0, "right": 0.0}
        self.previous_angles = None
        self.previous_mouse = self.tracker.mouse_counters_snapshot()

    def finish(self, outcome: str) -> None:
        self.write({
            "type": "result", "outcome": outcome,
            "physical_counts": self.raw_by_direction,
            "camera_yaw_degrees": self.yaw_by_direction,
            "weak_directions": self.weak_directions(),
        })


def pointer_inside_portal(window: dict) -> bool:
    """Check both focus and global pointer position during the edge sweep."""
    instance = window.get("hyprland_instance")
    position = window.get("at")
    size = window.get("size")
    if not (instance and isinstance(position, list) and len(position) == 2
            and isinstance(size, list) and len(size) == 2):
        raise RuntimeError("Portal 2 window geometry is unavailable; cannot verify mouse confinement")
    active = query_hyprland_active_window(instance)
    cursor = query_hyprland_cursor(instance)
    if active is None or cursor is None:
        raise RuntimeError("Hyprland focus or cursor position is unavailable; cannot verify mouse confinement")
    return bool(active.get("portal_focused")) and (
        position[0] <= cursor[0] < position[0] + size[0]
        and position[1] <= cursor[1] < position[1] + size[1]
    )


def wait_for_physical_mouse(
    controller: Portal2Controller, tracker, focus_delay: float,
    instance: str, starting_angles: tuple[float, float, float],
    focus_status: dict | None = None,
    diagnostics: HandoffDiagnostics | None = None,
) -> str | None:
    """Verify camera response and pointer confinement before saving any clip."""
    for attempt in range(2):
        previous_angles = starting_angles
        camera_rotation = 0.0
        deadline = time.monotonic() + focus_delay
        while time.monotonic() < deadline:
            if focus_status is not None and not pointer_inside_portal(focus_status):
                print("Mouse left Portal 2 or the window lost focus; restarting this chamber without a correction.")
                if diagnostics is not None:
                    diagnostics.finish("pointer_or_focus_lost")
                return "restart_requested"
            for control in tracker.pop_control_events():
                if control == "restart":
                    if diagnostics is not None:
                        diagnostics.finish("restart_requested")
                    return "restart_requested"
                if control == "stop":
                    if diagnostics is not None:
                        diagnostics.finish("interrupted")
                    return "interrupted"
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            try:
                angles = read_player_angles(controller, timeout=0.5)
            except RuntimeError:
                continue
            if diagnostics is not None:
                diagnostics.sample(angles)
            camera_rotation += max(
                abs(angle_delta(angles[0], previous_angles[0])),
                abs(angle_delta(angles[1], previous_angles[1])),
            )
            previous_angles = angles
        weak = diagnostics.weak_directions() if diagnostics is not None else []
        if camera_rotation >= 1.5 and not weak:
            if diagnostics is not None:
                diagnostics.finish("ready")
            return None
        if attempt == 0:
            if diagnostics is not None:
                diagnostics.finish("retry_capture")
            print(
                f"Mouse check: camera turned {camera_rotation:.1f} degrees"
                + (f" but {', '.join(weak)} physical turns barely registered;" if weak else ";")
                + " "
                "recentering the pointer and retrying Portal 2 capture once."
            )
            before_retry = read_player_angles(controller)
            status = activate_portal_input(controller, instance, force_recenter=True)
            if not (status or {}).get("portal_focused"):
                raise RuntimeError("Could not reacquire Portal 2 input for recovery")
            focus_status = status if focus_status is not None else None
            starting_angles = read_player_angles(controller)
            if max(
                abs(angle_delta(starting_angles[0], before_retry[0])),
                abs(angle_delta(starting_angles[1], before_retry[1])),
            ) > STARTUP_PITCH_TOLERANCE:
                print("Mouse capture moved the camera during retry; restarting this chamber without a correction.")
                if diagnostics is not None:
                    diagnostics.finish("capture_moved_camera")
                return "restart_requested"
            if diagnostics is not None:
                diagnostics.reset_response()
    if diagnostics is not None:
        diagnostics.finish("weak_mouse_response")
    print("Physical mouse did not reliably turn the camera; restarting this chamber without recording.")
    return "restart_requested"


def write_result(path: Path | None, outcome: str, episode: str | None) -> None:
    """Report a finished correction or pre-recording control to the session."""
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"outcome": outcome, "episode": episode}) + "\n", encoding="utf-8")


def chamber_still_active(controller: Portal2Controller) -> bool:
    """Only a finished chamber may skip recovery; setup failures remain errors."""
    try:
        disable_chamber_timeout(controller, require_active=True)
    except RuntimeError as error:
        if str(error) != "Chamber is already finished or not active for recovery":
            raise
        return False
    return True


def wait_for_manual_start(
    tracker, focus_delay: float, controller: Portal2Controller | None = None,
    diagnostics: HandoffDiagnostics | None = None,
) -> str:
    """Let the operator verify confinement and reframe before beginning a clip."""
    earliest_start = time.monotonic() + focus_delay
    next_sample = time.monotonic()
    while True:
        if diagnostics is not None and controller is not None and time.monotonic() >= next_sample:
            try:
                diagnostics.sample(read_player_angles(controller, timeout=0.5))
            except RuntimeError:
                pass
            next_sample = time.monotonic() + 0.25
        for control in tracker.pop_control_events():
            if control == "restart":
                if diagnostics is not None:
                    diagnostics.finish("restart_requested")
                return "restart_requested"
            if control == "stop":
                if diagnostics is not None:
                    diagnostics.finish("interrupted")
                return "interrupted"
            if control == "start" and time.monotonic() >= earliest_start:
                if diagnostics is not None and diagnostics.weak_directions():
                    diagnostics.finish("weak_mouse_response")
                    print("Physical mouse barely turned the camera; restarting without a correction.")
                    return "restart_requested"
                if diagnostics is not None:
                    diagnostics.finish("ready")
                return "start"
        time.sleep(0.1)


class TestRecoveryMonitor:
    """Expose pre-recording controls to the event loop without writing media."""

    def __init__(self, tracker) -> None:
        self.tracker = tracker
        self.paused = False

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        print("Test monitoring paused." if self.paused else "Test monitoring resumed.")
        return self.paused


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not is_game_running():
        raise RuntimeError("Portal 2 must remain open at the model's stuck position")
    recorder = EpisodeRecorder(
        fps=args.fps, width=args.width, height=args.height, video_crf=args.crf,
        controller=Portal2Controller(args.port, log_file=None),
        output_name=args.output, mouse_device=args.mouse_device,
        episodes_root=args.episodes_root, restart_enabled=args.session_controls,
        start_enabled=args.manual_focus,
        prepare_output=not args.test_mode,
        resume_delay_seconds=3.0,
    )
    controller = recorder.controller
    tracker_started = camera_started = False
    saved_restart_binding = None
    diagnostics = None
    try:
        connect_controller(controller)
        if args.session_controls:
            saved_restart_binding = read_portal_binding(controller, "r")
            controller.send_command("unbind r")
        if args.disable_chamber_timeout and not chamber_still_active(controller):
            print("Chamber finished before recovery began; no correction will be recorded.")
            write_result(args.result_file, "model_chamber_finished", None)
            return
        recorder.tracker.start()
        tracker_started = True
        recorder.camera.start()
        camera_started = True
        wait_for_camera(recorder)
        initial_angles = None
        if args.keep_focused:
            initial_angles = read_player_angles(controller)
            # The policy's virtual mouse may leave Hyprland's confined pointer
            # away from the center even while Portal remains focused. Reset it
            # before handing control to the physical mouse.
            focus_status = activate_portal_input(
                controller, args.hyprland_instance, force_recenter=True,
            )
            if not (focus_status or {}).get("portal_focused"):
                raise RuntimeError("Could not reacquire Portal 2 input for human recovery")
            captured_angles = read_player_angles(controller)
            diagnostic_path = (
                args.result_file.with_name("handoff_diagnostics.jsonl")
                if args.result_file else
                args.source_attempt.with_name("handoff_diagnostics.jsonl")
                if args.source_attempt else None
            )
            diagnostics = HandoffDiagnostics(
                recorder.tracker, args.hyprland_instance, diagnostic_path,
            )
            if diagnostic_path is not None:
                print(f"Handoff diagnostics: {diagnostic_path}")
            pitch_jump = angle_delta(captured_angles[0], initial_angles[0])
            yaw_jump = angle_delta(captured_angles[1], initial_angles[1])
            if not args.manual_focus and max(abs(pitch_jump), abs(yaw_jump)) > STARTUP_PITCH_TOLERANCE:
                print(
                    f"Camera moved before the focus prompt (pitch {pitch_jump:+.1f}, "
                    f"yaw {yaw_jump:+.1f} degrees); "
                    "discarding this handoff and retrying the chamber."
                )
                write_result(args.result_file, "restart_requested", None)
                return
        if args.manual_focus:
            print(
                f"Portal 2 will stay on {args.map}. Check mouse focus, re-align the view, "
                f"then press P to start recovery recording (after at least {args.focus_delay:g}s). "
                "R retries this chamber; K stops the session."
            )
        else:
            print(
                f"Portal 2 will stay on {args.map}; recording starts in {args.focus_delay:g}s. "
                "Move the physical mouse toward the other monitor and back, then return the view near its starting angle. "
                + ("Esc pauses/resumes; K stops the session; R restarts this chamber."
                   if args.session_controls else "Esc pauses/resumes; K ends this recording.")
            )
        early_outcome = None
        if args.manual_focus:
            early_outcome = wait_for_manual_start(
                recorder.tracker, args.focus_delay, controller, diagnostics,
            )
            if early_outcome == "start":
                early_outcome = None
        elif args.keep_focused:
            early_outcome = wait_for_physical_mouse(
                controller, recorder.tracker, args.focus_delay,
                args.hyprland_instance, captured_angles, focus_status,
                diagnostics,
            )
        else:
            time.sleep(args.focus_delay)
        if early_outcome is not None:
            write_result(args.result_file, early_outcome, None)
            return
        if args.keep_focused and not args.manual_focus and not pointer_inside_portal(focus_status):
            print("Mouse left Portal 2 before recording; restarting this chamber without a correction.")
            write_result(args.result_file, "restart_requested", None)
            return
        if args.keep_focused and not args.manual_focus:
            ready_angles = read_player_angles(controller)
            pitch_shift = angle_delta(ready_angles[0], captured_angles[0])
            yaw_shift = angle_delta(ready_angles[1], captured_angles[1])
            if max(abs(pitch_shift), abs(yaw_shift)) > STARTUP_PITCH_TOLERANCE:
                print(
                    f"Camera moved too far during the focus wait (pitch {pitch_shift:+.1f}, "
                    f"yaw {yaw_shift:+.1f} degrees); restarting the chamber without a correction."
                )
                write_result(args.result_file, "restart_requested", None)
                return
        if args.disable_chamber_timeout and not chamber_still_active(controller):
            print("Chamber finished during handoff; no correction will be recorded.")
            write_result(args.result_file, "model_chamber_finished", None)
            return
        controller.read_console(print_to_terminal=False)
        controller.event_buffer.clear()
        recorder.tracker.get_snapshot_and_reset()
        if args.test_mode:
            print("TEST MODE: takeover ready; observing controls and chamber events without recording a clip.")
            outcome = wait_for_terminal_event(
                controller, EpisodeEventStream(), args.max_seconds,
                recorder=TestRecoveryMonitor(recorder.tracker), reject_new_chamber=True,
            )
            write_result(args.result_file, outcome, None)
            return
        recorder.start_recording(
            map_name=args.map,
            metadata_extra={
                "source": "model_recovery",
                "source_attempt": str(args.source_attempt.resolve()) if args.source_attempt else None,
            },
        )
        outcome = wait_for_terminal_event(
            controller, EpisodeEventStream(), args.max_seconds,
            recorder=recorder, reject_new_chamber=True,
        )
        recorder.stop_recording("discarded" if outcome == "restart_requested" else outcome)
        write_result(args.result_file, outcome, str(Path(recorder.ep_dir).resolve()))
    except KeyboardInterrupt:
        print("\nStopping recovery recording.")
    finally:
        if recorder.recording:
            recorder.stop_recording("interrupted")
        if tracker_started:
            recorder.tracker.stop()
        if camera_started:
            recorder.camera.stop()
        if saved_restart_binding is not None:
            restore_portal_binding(controller, "r", saved_restart_binding)
        controller.disconnect()


if __name__ == "__main__":
    main()
