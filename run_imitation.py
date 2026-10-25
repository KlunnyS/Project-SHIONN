"""Run a trained SHIONN imitation policy against a live Portal 2 window."""

from __future__ import annotations

# Python standard library: CLI, processes, threads, timing, and attempt logs.
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

# Installed dependencies: input devices and numerical frame arrays.
import evdev
import numpy as np

# Project modules at the repository root: capture and Portal 2 control.
from recorder import FFmpegVideoWriter, WaylandCamera, get_default_output
from wrapper import Portal2Controller, is_game_running, launch_game


TERMINAL_EVENTS = ("EVT|goal_reached", "EVT|episode_failed")
PORTAL_WINDOW_CLASS = "steam_app_620"
GAMESCOPE_WINDOW_CLASS = "gamescope"
EPISODE_FAILURE_PATTERN = re.compile(r"EVT\|episode_failed\|([^\r\n]*)")
DIAGNOSTIC_MIN_SECONDS = 1.0
WALL_STUCK_VISUAL_DELTA = 0.05
TURN_LOOP_MOUSE_MIN_ABS = 45
PLAYER_ANGLES_PATTERN = re.compile(
    r"setang\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)"
)
STARTUP_PITCH_TOLERANCE = 8.0


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
    """Watch physical keyboards for emergency stop and session controls."""

    def __init__(self, handoff_key: int | None = None, restart_key: int | None = None,
                 session_stop_key: int | None = None) -> None:
        self._stop_requested = threading.Event()
        self._handoff_requested = threading.Event()
        self._restart_requested = threading.Event()
        self._session_stop_requested = threading.Event()
        self._stopping = threading.Event()
        self.handoff_key = handoff_key
        self.restart_key = restart_key
        self.session_stop_key = session_stop_key
        self._devices: list[evdev.InputDevice] = []
        self._threads: list[threading.Thread] = []

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested.is_set()

    @property
    def handoff_requested(self) -> bool:
        return self._handoff_requested.is_set()

    @property
    def restart_requested(self) -> bool:
        return self._restart_requested.is_set()

    @property
    def session_stop_requested(self) -> bool:
        return self._session_stop_requested.is_set()

    def wait(self, timeout: float) -> bool:
        return self._stop_requested.wait(timeout)

    def _watch_device(self, device: evdev.InputDevice) -> None:
        try:
            for event in device.read_loop():
                if self._stopping.is_set():
                    return
                if event.type == evdev.ecodes.EV_KEY and event.value == 1:
                    if event.code == evdev.ecodes.KEY_ESC:
                        self._stop_requested.set()
                        return
                    if event.code == self.session_stop_key:
                        self._session_stop_requested.set()
                        return
                    if event.code == self.restart_key:
                        self._restart_requested.set()
                        return
                    if event.code == self.handoff_key:
                        self._handoff_requested.set()
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
    parser.add_argument("--jump-threshold", type=float,
                        help="Override the binary jump decision threshold for live runs")
    parser.add_argument("--move-w-threshold", type=float,
                        help="Override the forward decision threshold for a live calibration test")
    parser.add_argument(
        "--mouse-max-abs", type=int,
        help="Clamp applied mouse dx/dy to +/- this value while preserving raw predictions in JSONL",
    )
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--output", help="Wayland output name from `wf-recorder -L`; defaults to the first output")
    parser.add_argument("--map", dest="map_name", help="Optional map to load after connecting, for example puzzlemaker/preview")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--max-seconds", type=float, default=60.0, help="Stop automatically; use 0 to run until Ctrl+C")
    parser.add_argument("--disable-chamber-timeout", action="store_true",
                        help="Require the installed SHIONN VScript to acknowledge timeout suppression")
    parser.add_argument("--dry-run", action="store_true", help="Print predictions without sending actions to Portal 2")
    parser.add_argument("--no-launch", action="store_true", help="Fail instead of launching Portal 2 when it is not running")
    parser.add_argument("--continue-after-event", action="store_true", help="Keep running after goal/failure events")
    parser.add_argument("--recovery-handoff", action="store_true",
                        help="Enable P to stop the policy at its current position for a recovery recording")
    parser.add_argument("--record-video", action="store_true", help="Save the frames seen by the policy as an MP4")
    parser.add_argument("--recording-dir", type=Path, default=Path("model_attempts"), help="Directory for --record-video output")
    parser.add_argument("--log-actions", action="store_true", help="Save structured per-tick diagnostics without requiring video recording")
    parser.add_argument("--log-file", type=Path, help="Custom JSONL diagnostics path; implies --log-actions")
    parser.add_argument("--verbose", action="store_true", help="Print a concise runtime status line once per second")
    parser.add_argument("--status-every", type=float, default=1.0, help="Seconds between --verbose status lines and focus checks")
    parser.add_argument("--keep-focused", action="store_true", help="Focus Portal 2 and restore focus if it is lost")
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


def is_portal_window(window: dict) -> bool:
    """Accept either Portal's XWayland surface or its Gamescope wrapper."""
    if window.get("class") == PORTAL_WINDOW_CLASS:
        return True
    return (
        window.get("class") == GAMESCOPE_WINDOW_CLASS
        and isinstance(window.get("title"), str)
        and bool(re.fullmatch(r"PORTAL 2(?: - .*)?", window["title"], re.IGNORECASE))
    )


def niri_is_active() -> bool:
    """Detect a Niri session from its IPC socket environment."""
    return bool(os.environ.get("NIRI_SOCKET")) and shutil.which("niri") is not None


def query_niri_active_window() -> dict | None:
    """Return a normalized focused-window record from Niri's documented IPC."""
    if not niri_is_active():
        return None
    try:
        result = subprocess.run(
            ["niri", "msg", "--json", "focused-window"], check=True,
            capture_output=True, text=True, timeout=2,
        )
        window = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if not isinstance(window, dict):
        return None
    app_id = window.get("app_id") or ""
    title = window.get("title") or ""
    normalized = {
        "address": window.get("id"), "class": app_id, "title": title,
        "workspace": {"id": window.get("workspace_id")},
        # Niri's stable Window IPC type does not expose a fullscreen flag.
        "fullscreen": None,
        "niri_window_id": window.get("id"), "window_manager": "niri",
    }
    # Gamescope's title identifies the wrapped Portal window in current Steam
    # setups; direct XWayland Portal windows use Steam's app ID/class.
    normalized["portal_focused"] = (
        app_id == PORTAL_WINDOW_CLASS
        or (app_id == GAMESCOPE_WINDOW_CLASS and bool(
            re.fullmatch(r"PORTAL 2(?: - .*)?", title, re.IGNORECASE)
        ))
    )
    return normalized


def query_niri_workspace() -> dict | None:
    """Return focused Niri workspace metadata; cursor coordinates are not exposed."""
    if not niri_is_active():
        return None
    try:
        result = subprocess.run(
            ["niri", "msg", "--json", "workspaces"], check=True,
            capture_output=True, text=True, timeout=2,
        )
        workspaces = json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None
    if not isinstance(workspaces, list):
        return None
    active = next((item for item in workspaces if item.get("is_focused")), None)
    if active is None:
        return None
    return {"id": active.get("id"), "name": active.get("name"),
            "monitor": active.get("output"), "window_manager": "niri"}


def query_hyprland_active_window(instance: str = "auto") -> dict | None:
    """Return a small active-window snapshot, or None outside Hyprland."""
    if niri_is_active():
        return query_niri_active_window()
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
            "address": window.get("address"),
            "class": window.get("class"),
            "title": window.get("title"),
            "monitor": window.get("monitor"),
            "workspace": window.get("workspace"),
            "fullscreen": window.get("fullscreen"),
            "at": window.get("at"),
            "size": window.get("size"),
            "portal_focused": is_portal_window(window),
            "hyprland_instance": candidate,
        }
    return None


def focus_portal_window(instance: str = "auto") -> dict | None:
    """Focus the exact visible Portal or Gamescope window, not another game."""
    if niri_is_active():
        try:
            result = subprocess.run(
                ["niri", "msg", "--json", "windows"], check=True,
                capture_output=True, text=True, timeout=2,
            )
            windows = json.loads(result.stdout)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            return None
        if not isinstance(windows, list):
            return None
        targets = [window for window in windows if isinstance(window, dict) and (
            window.get("app_id") == PORTAL_WINDOW_CLASS
            or (window.get("app_id") == GAMESCOPE_WINDOW_CLASS and re.fullmatch(
                r"PORTAL 2(?: - .*)?", str(window.get("title", "")), re.IGNORECASE
            ))
        )]
        if not targets:
            return None
        target = next((item for item in targets if item.get("app_id") == GAMESCOPE_WINDOW_CLASS), targets[0])
        try:
            subprocess.run(
                ["niri", "msg", "action", "focus-window", "--id", str(target["id"])],
                check=True, capture_output=True, text=True, timeout=2,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        time.sleep(0.1)
        status = query_niri_active_window()
        return status if status and status.get("niri_window_id") == target.get("id") else None
    if shutil.which("hyprctl") is None:
        return None
    for candidate in hyprland_instance_candidates(instance):
        try:
            clients_result = subprocess.run(
                ["hyprctl", "-i", candidate, "clients", "-j"],
                check=True, capture_output=True, text=True, timeout=2,
            )
            clients = json.loads(clients_result.stdout)
            targets = [
                window for window in clients
                if is_portal_window(window) and not window.get("hidden")
                and window.get("visible", True)
                and re.fullmatch(r"0x[0-9a-fA-F]+", str(window.get("address", "")))
            ]
            if not targets:
                continue
            # Gamescope owns the outer window when present; the nested game is
            # not a separate focus target in the host compositor.
            target = next(
                (window for window in targets if window.get("class") == GAMESCOPE_WINDOW_CLASS),
                targets[0],
            )
            subprocess.run(
                [
                    "hyprctl", "-i", candidate, "dispatch", "focuswindow",
                    f"address:{target['address']}",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            time.sleep(0.1)
        except (OSError, ValueError, TypeError, subprocess.SubprocessError):
            continue
        status = query_hyprland_active_window(candidate)
        if status is not None and status["portal_focused"] and status.get("address") == target["address"]:
            return status
    return None


def query_hyprland_cursor(instance: str) -> tuple[int, int] | None:
    """Return compositor cursor coordinates, if available."""
    try:
        result = subprocess.run(
            ["hyprctl", "-i", instance, "-j", "cursorpos"], check=True,
            capture_output=True, text=True, timeout=2,
        )
        position = json.loads(result.stdout)
        return int(position["x"]), int(position["y"])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None


def query_hyprland_active_workspace(instance: str) -> dict | None:
    """Return the currently active Hyprland workspace, if available."""
    if niri_is_active():
        return query_niri_workspace()
    try:
        result = subprocess.run(
            ["hyprctl", "-i", instance, "-j", "activeworkspace"], check=True,
            capture_output=True, text=True, timeout=2,
        )
        workspace = json.loads(result.stdout)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return None
    return {
        "id": workspace.get("id"),
        "name": workspace.get("name"),
        "monitor": workspace.get("monitor"),
    }


def desktop_input_snapshot(window: dict | None, instance: str) -> dict:
    """Describe game-window geometry, compositor cursor, and workspace state."""
    if instance == "auto":
        instance = (window or {}).get("hyprland_instance") or next(
            iter(hyprland_instance_candidates("auto")), "auto"
        )
    cursor = None if niri_is_active() else query_hyprland_cursor(instance)
    workspace = query_hyprland_active_workspace(instance)
    inside_window = None
    if window is not None and cursor is not None:
        position = window.get("at")
        size = window.get("size")
        if (
            isinstance(position, list) and len(position) >= 2
            and isinstance(size, list) and len(size) >= 2
        ):
            left, top = position[:2]
            width, height = size[:2]
            inside_window = (
                left <= cursor[0] < left + width
                and top <= cursor[1] < top + height
            )
    return {
        "window": window,
        "cursor": {"x": cursor[0], "y": cursor[1]} if cursor is not None else None,
        "cursor_inside_game_window": inside_window,
        "active_workspace": workspace,
        "game_window_workspace": (window or {}).get("workspace"),
        "game_window_monitor": (window or {}).get("monitor"),
    }


def fullscreen_portal_window(instance: str = "auto") -> dict | None:
    """Focus Portal and put it in compositor fullscreen before capture."""
    status = focus_portal_window(instance)
    if not (status or {}).get("portal_focused"):
        return status
    if (status or {}).get("window_manager") == "niri":
        # FullscreenWindow is a toggle and Niri's Window response has no
        # fullscreen state. Do not toggle blindly; Gamescope owns game mode.
        return query_niri_active_window()
    if status.get("fullscreen"):
        return status
    candidate = status.get("hyprland_instance")
    if not candidate:
        return None
    try:
        subprocess.run(
            ["hyprctl", "-i", candidate, "dispatch", "fullscreen", "0"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        time.sleep(0.1)
    except (OSError, subprocess.SubprocessError):
        return None
    return query_hyprland_active_window(candidate)


def prepare_portal_visual_state(
    controller: Portal2Controller, instance: str = "auto"
) -> dict | None:
    """Match the clean fullscreen, equipped-viewmodel training input."""
    status = fullscreen_portal_window(instance)
    niri_focus_only = (status or {}).get("window_manager") == "niri"
    if not (status or {}).get("portal_focused") or (
        not niri_focus_only and not status.get("fullscreen")
    ):
        return None
    controller.send_command("r_drawviewmodel 1")
    controller.send_command("use weapon_portalgun")
    time.sleep(0.1)
    return status


def read_portal_binding(controller: Portal2Controller, key: str) -> str:
    """Read a simple Portal binding so session controls can restore it later."""
    if not re.fullmatch(r"[A-Za-z0-9]+", key):
        raise ValueError(f"Invalid Portal key name: {key!r}")
    controller.read_console(print_to_terminal=False)
    controller.send_command(f"bind {key}")
    output = ""
    deadline = time.monotonic() + 2.0
    binding = None
    while time.monotonic() < deadline:
        output += controller.read_console(print_to_terminal=False)
        match = re.search(rf'"{key}"\s*=\s*"([^"\r\n]*)"', output)
        if match:
            binding = match.group(1)
            break
        if f'"{key}" is not bound' in output:
            binding = ""
            break
        time.sleep(0.05)
    if binding is None:
        raise RuntimeError(f"Could not read Portal 2's {key} binding")
    if not re.fullmatch(r"[+\-A-Za-z0-9_ ]*", binding):
        raise RuntimeError(f"{key} has a complex binding; cannot safely restore it")
    return binding


def restore_portal_binding(controller: Portal2Controller, key: str, binding: str) -> None:
    """Restore a previously read binding, including an unbound key."""
    controller.send_command(f'bind {key} "{binding}"' if binding else f"unbind {key}")


def click_to_capture_portal_input(controller: Portal2Controller) -> bool:
    """Capture the XWayland pointer without creating another uinput mouse.

    A left-click is the game's normal pointer-capture path. Temporarily unbind
    its attack action, deliver a core X11 click, then restore that binding.
    """
    if shutil.which("xte") is None:
        raise RuntimeError("xte is required for Portal 2 mouse capture")
    binding = read_portal_binding(controller, "mouse1")

    controller.send_command("unbind mouse1")
    try:
        time.sleep(0.05)
        subprocess.run(
            ["xte", "mouseclick 1"], check=True, capture_output=True,
            text=True, timeout=2,
        )
        return True
    except (OSError, subprocess.SubprocessError):
        return False
    finally:
        restore_portal_binding(controller, "mouse1", binding)


def refresh_gamescope_pointer_grab(status: dict) -> dict | None:
    """Briefly leave and restore the game's workspace to renew its pointer grab.

    Hyprland can report Gamescope focused even when relative mouse capture has
    been lost. A workspace focus transition has restored capture in live tests.
    This runs only during the human-recovery handoff, before recording starts.
    """
    instance = status.get("hyprland_instance")
    workspace = status.get("workspace")
    workspace_id = workspace.get("id") if isinstance(workspace, dict) else None
    address = status.get("address")
    if not instance or not isinstance(workspace_id, int) or workspace_id <= 0 or not address:
        return None
    switched = False
    try:
        subprocess.run(
            ["hyprctl", "-i", instance, "dispatch", "workspace", "emptym"],
            check=True, capture_output=True, text=True, timeout=2,
        )
        switched = True
        time.sleep(0.5)
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        if switched:
            # Restore by ID, not "previous": another monitor may have its own
            # previous-workspace history.
            try:
                subprocess.run(
                    ["hyprctl", "-i", instance, "dispatch", "workspace", str(workspace_id)],
                    check=True, capture_output=True, text=True, timeout=2,
                )
                time.sleep(0.5)
            except (OSError, subprocess.SubprocessError) as error:
                raise RuntimeError(
                    f"Could not restore Portal 2 workspace {workspace_id} after pointer refresh"
                ) from error
    restored = query_hyprland_active_window(instance)
    if not restored or restored.get("address") != address or not restored.get("portal_focused"):
        restored = focus_portal_window(instance)
    restored_workspace = restored.get("workspace") if restored else None
    if (
        restored and restored.get("address") == address
        and restored.get("portal_focused")
        and isinstance(restored_workspace, dict)
        and restored_workspace.get("id") == workspace_id
    ):
        return restored
    return None


def activate_portal_input(
    controller: Portal2Controller, instance: str = "auto", *,
    force_recenter: bool = False,
) -> dict | None:
    """Focus Portal and acquire native XWayland or Gamescope-relative input."""
    status = query_hyprland_active_window(instance)
    already_focused = bool((status or {}).get("portal_focused"))
    if not already_focused:
        status = focus_portal_window(instance)
    if not (status or {}).get("portal_focused"):
        return None
    if status.get("window_manager") == "niri":
        if status.get("class") == GAMESCOPE_WINDOW_CLASS:
            if force_recenter:
                raise RuntimeError(
                    "Automated recovery mouse reacquisition is not supported on Niri yet; "
                    "Niri does not expose cursor coordinates and its Gamescope grab refresh "
                    "has not been validated."
                )
            # Niri can focus Gamescope, but focusing alone does not prove that
            # its nested game has captured relative mouse input.
            controller.send_command("unpause")
            time.sleep(0.1)
            return query_niri_active_window()
        if not click_to_capture_portal_input(controller):
            return None
        controller.send_command("unpause")
        time.sleep(0.1)
        return query_niri_active_window()
    candidate = status.get("hyprland_instance")
    if status.get("class") == GAMESCOPE_WINDOW_CLASS:
        # With --force-grab-cursor Gamescope owns the Wayland relative pointer.
        # An XTEST click targets the host XWayland server, not its nested game,
        # and warping the compositor cursor can disrupt that relative grab.
        if force_recenter:
            print("Refreshing Gamescope mouse capture via a brief workspace switch.")
            status = refresh_gamescope_pointer_grab(status)
            if status is None:
                return None
        controller.send_command("unpause")
        time.sleep(0.1)
        return query_hyprland_active_window(candidate)
    position = status.get("at")
    size = status.get("size")
    if not (
        isinstance(position, list) and len(position) == 2
        and isinstance(size, list) and len(size) == 2
        and candidate
    ):
        return None
    cursor = query_hyprland_cursor(candidate)
    cursor_outside = cursor is not None and not (
        position[0] <= cursor[0] < position[0] + size[0]
        and position[1] <= cursor[1] < position[1] + size[1]
    )
    # A confined but uncaptured pointer can still be pinned inside the window.
    # Recovery handoffs explicitly request a recenter before the capture click.
    if not already_focused or cursor_outside or force_recenter:
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
    if not click_to_capture_portal_input(controller):
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


def limit_mouse_action(
    action: dict[str, int], max_abs: int | None
) -> dict[str, int]:
    """Return an action whose applied mouse deltas respect an optional cap."""
    limited = dict(action)
    if max_abs is None:
        return limited
    for name in ("mouse_dx", "mouse_dy"):
        value = int(limited.get(name, 0))
        limited[name] = max(-max_abs, min(max_abs, value))
    return limited


def movement_commanded(action: dict[str, int]) -> bool:
    return any(action.get(name) for name in ("move_w", "move_a", "move_s", "move_d"))


def action_is_idle(action: dict[str, int]) -> bool:
    return not any(action.get(name) for name in (
        "move_w", "move_a", "move_s", "move_d", "jump", "use",
        "fire_left", "fire_right", "mouse_dx", "mouse_dy",
    ))


class RuntimeDiagnosticTracker:
    """Emit start/end records for sustained rollout failure signatures."""

    def __init__(
        self,
        *,
        minimum_seconds: float = DIAGNOSTIC_MIN_SECONDS,
        wall_visual_delta: float = WALL_STUCK_VISUAL_DELTA,
        turn_mouse_min_abs: int = TURN_LOOP_MOUSE_MIN_ABS,
    ) -> None:
        self.minimum_seconds = minimum_seconds
        self.wall_visual_delta = wall_visual_delta
        self.turn_mouse_min_abs = turn_mouse_min_abs
        self.counts = {
            "policy_freeze": 0,
            "wall_stuck": 0,
            "turn_loop": 0,
        }
        self._states = {
            name: {"started_at": None, "active": False, "details": {}}
            for name in self.counts
        }
        self._turn_direction = 0

    def reset(self) -> None:
        """Discard pending streaks after an episode, policy, or focus reset."""
        for state in self._states.values():
            state.update(started_at=None, active=False, details={})
        self._turn_direction = 0

    def _advance(
        self,
        name: str,
        condition: bool,
        elapsed_seconds: float,
        **details,
    ) -> list[dict]:
        state = self._states[name]
        events = []
        if condition:
            if state["started_at"] is None:
                state["started_at"] = elapsed_seconds
                state["details"] = details
            duration = elapsed_seconds - state["started_at"]
            if not state["active"] and duration >= self.minimum_seconds:
                state["active"] = True
                self.counts[name] += 1
                events.append({
                    "event": name,
                    "phase": "start",
                    "started_at_seconds": state["started_at"],
                    "duration_seconds": duration,
                    **state["details"],
                })
        else:
            if state["active"]:
                events.append({
                    "event": name,
                    "phase": "end",
                    "started_at_seconds": state["started_at"],
                    "duration_seconds": elapsed_seconds - state["started_at"],
                    **state["details"],
                })
            state.update(started_at=None, active=False, details={})
        return events

    def update(
        self,
        elapsed_seconds: float,
        action: dict[str, int],
        visual_delta: float | None,
    ) -> list[dict]:
        """Update all signatures from the applied action and current frame delta."""
        events = self._advance(
            "policy_freeze", action_is_idle(action), elapsed_seconds
        )
        moving = movement_commanded(action)
        events.extend(self._advance(
            "wall_stuck",
            moving
            and visual_delta is not None
            and visual_delta <= self.wall_visual_delta,
            elapsed_seconds,
            visual_delta_max=self.wall_visual_delta,
        ))

        mouse_dx = int(action.get("mouse_dx", 0))
        turn_direction = (
            1 if moving and mouse_dx >= self.turn_mouse_min_abs
            else -1 if moving and mouse_dx <= -self.turn_mouse_min_abs
            else 0
        )
        if turn_direction != self._turn_direction:
            events.extend(self._advance(
                "turn_loop", False, elapsed_seconds
            ))
            self._turn_direction = turn_direction
        events.extend(self._advance(
            "turn_loop",
            turn_direction != 0,
            elapsed_seconds,
            direction="right" if turn_direction > 0 else "left",
            mouse_min_abs=self.turn_mouse_min_abs,
        ))
        return events


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


def disable_chamber_timeout(
    controller: Portal2Controller, wait_seconds: float = 3.0,
    require_active: bool = False,
) -> None:
    """Fail closed unless the loaded chamber acknowledges timeout suppression."""
    function = "DisableTimeoutForRecovery()" if require_active else "DisableTimeout()"
    # ent_fire is cheat-gated in Portal 2. The server-side script command is
    # available with sv_cheats 0 and can call the logic_script's own scope.
    controller.send_command(
        'script Entities.FindByName(null,"shionn_event_script").GetScriptScope().'
        + function
    )
    deadline = time.monotonic() + wait_seconds
    output = ""
    while time.monotonic() < deadline:
        output = (output + controller.read_console(print_to_terminal=False))[-4096:]
        if "EVT|timeout_disabled|1" in output:
            return
        if "EVT|timeout_disable_failed|1" in output:
            raise RuntimeError("Chamber is already finished or not active for recovery")
        time.sleep(0.05)
    raise RuntimeError(
        "Chamber timeout suppression was not acknowledged; check the installed "
        "shionn_events.nut and Portal 2 console output: " + repr(output[-500:])
    )


def load_recovery_map(controller: Portal2Controller, map_name: str, timeout: float = 30.0) -> None:
    """Wait for this chamber's ready event, not a getpos reply from the old map."""
    controller.read_console(print_to_terminal=False)
    controller.event_buffer.clear()
    controller.send_command(f"map {map_name}")
    output = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output = (output + controller.read_console(print_to_terminal=False))[-4096:]
        if "EVT|chamber_ready|" in output:
            print(f"Chamber '{map_name}' is ready.")
            return
        time.sleep(0.05)
    raise RuntimeError(f"Chamber '{map_name}' did not become ready within {timeout:g}s")


def read_player_angles(controller: Portal2Controller, timeout: float = 2.0) -> tuple[float, float, float]:
    """Read the current view angle from the game's non-cheat getpos command."""
    controller.read_console(print_to_terminal=False)
    controller.send_command("getpos")
    output = ""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        output += controller.read_console(print_to_terminal=False)
        match = PLAYER_ANGLES_PATTERN.search(output)
        if match:
            return tuple(float(value) for value in match.groups())
        time.sleep(0.05)
    raise RuntimeError("Could not read Portal 2 camera angle with getpos")


def prepare_recovery_model_start(
    controller: Portal2Controller, map_name: str, instance: str,
    disable_timeout: bool,
) -> dict:
    """Reject a mouse-capture pitch snap before any policy frame is recorded."""
    for attempt in range(2):
        load_recovery_map(controller, map_name)
        if disable_timeout:
            disable_chamber_timeout(controller)
        expected_pitch = read_player_angles(controller)[0]
        status = prepare_portal_visual_state(controller, instance)
        if status is None:
            raise RuntimeError("Could not prepare Portal 2's fullscreen visual state")
        status = activate_portal_input(controller, instance)
        if not (status or {}).get("portal_focused"):
            raise RuntimeError("Could not reacquire Portal 2 input before recovery model attempt")
        actual_pitch = read_player_angles(controller)[0]
        pitch_change = (actual_pitch - expected_pitch + 180.0) % 360.0 - 180.0
        if abs(pitch_change) <= STARTUP_PITCH_TOLERANCE:
            print(f"Camera startup check passed (pitch change {pitch_change:+.1f} degrees).")
            return status
        print(
            f"Camera jumped {pitch_change:+.1f} degrees during focus/input capture; "
            + ("reloading chamber once." if attempt == 0 else "stopping before model input.")
        )
    raise RuntimeError("Portal 2 camera snapped during startup twice; no model actions were sent")


def main() -> None:
    args = parse_args()
    # Project inference module is loaded here, after CLI parsing, for runner startup.
    from models.imitation.inference import PolicyInference

    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.max_seconds < 0:
        raise ValueError("--max-seconds must be zero or greater")
    if args.disable_chamber_timeout and not args.map_name:
        raise ValueError("--disable-chamber-timeout requires --map")
    if args.recovery_handoff and (not args.map_name or not args.keep_focused):
        raise ValueError("--recovery-handoff requires --map and --keep-focused")
    if args.recovery_handoff and niri_is_active():
        raise RuntimeError(
            "Automated recovery handoff is currently Hyprland-only. Niri focus works, "
            "but cursor containment and Gamescope mouse-grab reacquisition need laptop validation."
        )
    if args.status_every <= 0:
        raise ValueError("--status-every must be greater than zero")
    if args.mouse_max_abs is not None and args.mouse_max_abs <= 0:
        raise ValueError("--mouse-max-abs must be greater than zero")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    policy = PolicyInference(
        args.checkpoint, device=args.device, jump_threshold=args.jump_threshold,
        move_w_threshold=args.move_w_threshold,
    )

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
    escape_monitor = EscapeKeyMonitor(
        evdev.ecodes.KEY_P if args.recovery_handoff else None,
        evdev.ecodes.KEY_R if args.recovery_handoff else None,
        evdev.ecodes.KEY_K if args.recovery_handoff else None,
    )
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
    visual_state = None
    saved_restart_binding = None
    stop_reason = "time_limit"
    diagnostic_tracker = RuntimeDiagnosticTracker()

    try:
        if args.recovery_handoff:
            saved_restart_binding = read_portal_binding(controller, "r")
            controller.send_command("unbind r")
        escape_monitor.start()
        if args.recovery_handoff:
            visual_state = prepare_recovery_model_start(
                controller, args.map_name, args.hyprland_instance,
                args.disable_chamber_timeout,
            )
            input_ready = True
        elif args.map_name:
            controller.load_map(args.map_name, wait_for_load=True)
            if args.disable_chamber_timeout:
                disable_chamber_timeout(controller)
        if args.keep_focused and not args.recovery_handoff:
            visual_state = prepare_portal_visual_state(
                controller, args.hyprland_instance
            )
            if visual_state is None:
                raise RuntimeError(
                    "Could not prepare Portal 2's fullscreen visual state"
                )
        camera.start()
        print(f"Policy device: {policy.device}")
        print(f"Capture output: {camera.output_name or 'wf-recorder default'}")
        print(
            f"Starting in {args.countdown:g} seconds. "
            "P hands off, R restarts this chamber, K stops the session; Esc/Ctrl+C is emergency stop."
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
                visual_state=visual_state,
                jump_threshold=args.jump_threshold,
                move_w_threshold=args.move_w_threshold,
                mouse_max_abs=args.mouse_max_abs,
                chamber_timeout_disabled=args.disable_chamber_timeout,
                mouse_bins=policy.mouse_bins.config if policy.mouse_bins is not None else None,
                runtime_diagnostics={
                    "minimum_seconds": diagnostic_tracker.minimum_seconds,
                    "wall_stuck_visual_delta_max": diagnostic_tracker.wall_visual_delta,
                    "turn_loop_mouse_min_abs": diagnostic_tracker.turn_mouse_min_abs,
                },
            )
            print(f"Writing attempt diagnostics to: {log_path}")

        focus_status = query_hyprland_active_window(args.hyprland_instance)
        if args.recovery_handoff:
            # A previous human takeover or workspace refresh may have left
            # another window active. Re-focus the game before rejecting the
            # next model attempt; this does not inject any mouse input.
            if not (focus_status or {}).get("portal_focused"):
                focus_status = focus_portal_window(args.hyprland_instance)
        elif args.keep_focused and not args.dry_run:
            focus_status = activate_portal_input(controller, args.hyprland_instance)
            input_ready = bool((focus_status or {}).get("portal_focused"))
        elif args.keep_focused and not (focus_status or {}).get("portal_focused"):
            focus_status = focus_portal_window(args.hyprland_instance)
        if args.keep_focused and not (focus_status or {}).get("portal_focused"):
            raise RuntimeError(
                "Could not focus Portal 2 through the active compositor; refusing to inject input"
            )
        if attempt_log is not None:
            attempt_log.write(
                "focus", event="initial", window=focus_status,
                input_activated=input_ready,
            )
            attempt_log.write(
                "desktop_state", phase="initial",
                **desktop_input_snapshot(focus_status, args.hyprland_instance),
            )

        started = time.monotonic()
        next_tick = started
        next_status = started
        tick_duration = 1.0 / args.fps
        while not args.max_seconds or time.monotonic() - started < args.max_seconds:
            if escape_monitor.session_stop_requested:
                print("\nK pressed. Stopping recovery session.")
                stop_reason = "session_stop"
                if attempt_log is not None:
                    attempt_log.write("stop", reason=stop_reason)
                break
            if escape_monitor.restart_requested:
                print("\nR pressed. Restarting this chamber from the beginning.")
                stop_reason = "restart_requested"
                if attempt_log is not None:
                    attempt_log.write("stop", reason=stop_reason)
                break
            if escape_monitor.stop_requested:
                print("\nEscape pressed. Stopping policy run.")
                stop_reason = "escape_key"
                if attempt_log is not None:
                    attempt_log.write("stop", reason="escape_key")
                break
            if args.recovery_handoff and escape_monitor.handoff_requested:
                pending_output = controller.read_console(print_to_terminal=False)
                terminal_received, _ = classify_terminal_events(pending_output, args.max_seconds)
                if terminal_received:
                    stop_reason = (
                        "goal_reached" if "EVT|goal_reached" in pending_output
                        else "episode_failed"
                    )
                    print(f"\nChamber ended before takeover ({stop_reason}); skipping recovery.")
                else:
                    stop_reason = "recovery_handoff"
                    print("\nP pressed. Releasing policy controls for human recovery. Keep the mouse still until the focus prompt.")
                if attempt_log is not None:
                    attempt_log.write("stop", reason=stop_reason)
                break
            frame = camera.get_latest_frame()
            action = None
            raw_action = None
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
                raw_action, diagnostics = policy.predict_with_diagnostics(frame)
                action = limit_mouse_action(raw_action, args.mouse_max_abs)
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
                diagnostic_tracker.reset()
                controller.release_policy_actions()
            if ignored_chamber_timeout:
                print(
                    "Chamber timeout event ignored; "
                    f"the runner will stop at {args.max_seconds:g} seconds."
                )
            if terminal_received:
                policy.reset()
                diagnostic_tracker.reset()
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
                    diagnostic_tracker.reset()
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
                desktop_state = desktop_input_snapshot(
                    focus_status, args.hyprland_instance
                )
                if attempt_log is not None:
                    attempt_log.write(
                        "desktop_state", phase="periodic",
                        elapsed_seconds=now - started,
                        tick=tick_count,
                        **desktop_state,
                    )
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
                        f" cursor={desktop_state['cursor']}"
                        f" cursor_in_window={desktop_state['cursor_inside_game_window']}"
                        f" workspace={desktop_state['active_workspace']}"
                    )
                next_status = now + args.status_every

            if action is not None:
                for event in diagnostic_tracker.update(
                    now - started, action, change
                ):
                    if attempt_log is not None:
                        attempt_log.write(
                            "diagnostic", tick=tick_count,
                            elapsed_seconds=now - started, **event,
                        )
                    if args.verbose and event["phase"] == "start":
                        print(
                            f"diagnostic t={now - started:6.1f}s "
                            f"event={event['event']}"
                        )

            if attempt_log is not None and action is not None:
                attempt_log.write(
                    "tick",
                    tick=tick_count,
                    elapsed_seconds=now - started,
                    action=action,
                    raw_action=raw_action,
                    mouse_limited=raw_action != action,
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
        if saved_restart_binding is not None:
            restore_portal_binding(controller, "r", saved_restart_binding)
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
                diagnostic_counts=diagnostic_tracker.counts,
            )
            attempt_log.close()
            print(f"Attempt diagnostics saved to: {log_path}")
        camera.stop()
        controller.disconnect()
        print("Policy stopped and held actions released.")


if __name__ == "__main__":
    main()
